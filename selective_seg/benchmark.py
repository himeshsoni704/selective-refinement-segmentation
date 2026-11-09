"""Reproducible wall-clock inference benchmarks; profiled passes never supply latency."""
import json
import platform
import resource
import sys
from pathlib import Path
import numpy as np
import torch
from .data import read_sample, read_image
from .objectives import confusion_matrix, miou
from .evaluation import correction_diagnostics


def environment(device):
    return {'python':sys.version.split()[0], 'pytorch':str(torch.__version__),
            'cuda_runtime':torch.version.cuda,'cudnn':torch.backends.cudnn.version(),
            'platform':platform.platform(),'cpu':platform.processor(),
            'gpu':torch.cuda.get_device_name(device) if device.type=='cuda' else None,
            'precision':'float16 autocast segmenters; float32 routing' if device.type=='cuda' else 'float32',
            'torch_threads':torch.get_num_threads(),'batch_size':1,
            'deterministic_algorithms':torch.are_deterministic_algorithms_enabled(),
            'cudnn_benchmark':torch.backends.cudnn.benchmark,
            'tf32_matmul':torch.backends.cuda.matmul.allow_tf32}


def summarize_latency(samples):
    values = np.asarray(samples,dtype=float)
    medians = np.median(values,axis=1)
    return {'mean_image_latency_ms':float(medians.mean()),
            'median_image_latency_ms':float(np.median(medians)),
            'p95_image_latency_ms':float(np.percentile(medians,95)),
            'mean_pass_latency_ms':values.mean(0).tolist(),
            'mean_within_image_cv':float((values.std(1)/values.mean(1).clip(1e-9)).mean()),
            'aggregation':'mean/median/p95 across per-image medians of repeated unprofiled passes'}


def profile_prediction(pipeline,image,mode,operations):
    if not operations:
        return pipeline.predict(image,mode,diagnostics=False,profile=True)[1],None
    activities = [torch.profiler.ProfilerActivity.CPU]
    if pipeline.device.type == 'cuda':
        activities.append(torch.profiler.ProfilerActivity.CUDA)
    with torch.profiler.profile(activities=activities,with_flops=True) as prof:
        _,detail = pipeline.predict(image,mode,diagnostics=False,profile=True)
    events = prof.key_averages()
    return detail, {'estimated_flops':sum(e.flops for e in events),
                    'operators_without_flop_estimates':sorted({e.key for e in events if e.key.startswith('aten::') and not e.flops})}


def evaluate(pipeline,records,destination,mode='selective',warmup=20,passes=3,
             profile_stages=False,profile_operations=False,save_masks=False):
    if not records or warmup < 0 or passes < 1:
        raise ValueError('Need records, nonnegative warmup and positive passes')
    destination = Path(destination)
    destination.mkdir(parents=True,exist_ok=True)
    cfg = pipeline.cfg
    for i in range(warmup):
        pipeline.predict(read_image(records[i%len(records)]['image'],cfg),mode,diagnostics=False)
    cm = torch.zeros(cfg['classes'],cfg['classes'],dtype=torch.long)
    times,rows,operation_counts = [],[],[]
    totals = {'corrected_pixels':0,'damaged_pixels':0,'valid_pixels':0,
              'confident_errors':0,'recovered_confident_errors':0}
    pair_c = torch.zeros_like(cm)
    pair_d = torch.zeros_like(cm)
    actions = {'skip':0,'low':0,'high':0}
    calls = {'low':0,'high':0}
    peak_allocated = peak_reserved = 0
    for record in records:
        image,truth = read_sample(record,cfg)
        measurements = []
        for _ in range(passes):
            if pipeline.device.type == 'cuda':
                torch.cuda.reset_peak_memory_stats(pipeline.device)
            mask,detail = pipeline.predict(image,mode,diagnostics=False)
            measurements.append(detail['end_to_end_ms'])
            if pipeline.device.type == 'cuda':
                peak_allocated = max(peak_allocated,torch.cuda.max_memory_allocated(pipeline.device))
                peak_reserved = max(peak_reserved,torch.cuda.max_memory_reserved(pipeline.device))
        times.append(measurements)
        cm += confusion_matrix(mask,truth,cfg['classes'],cfg['ignore_index'])
        # Explicitly outside measured inference. No masks enter pipeline.predict().
        pc,_ = pipeline.overview(image)
        diag = correction_diagnostics(pc.argmax(1)[0],mask,truth,cfg,pc.max(1).values[0])
        for key in totals:
            totals[key] += diag[key]
        pair_c += torch.tensor(diag['correction_pairs_predicted_to_true'])
        pair_d += torch.tensor(diag['damage_pairs_true_to_predicted'])
        for action in detail['final_actions']:
            actions[['skip','low','high'][action]] += 1
        for key in calls:
            calls[key] += detail['refinement_calls'][key]
        row = {'id':record['id'],'image_hw':list(image.shape[-2:]),'latency_ms':measurements,
               'estimated_refine_ms':detail['estimated_refine_ms'], 'escalations':detail['escalations'],**diag}
        budget = cfg['routing'].get('max_refine_ms')
        row['estimated_refine_budget_overrun'] = budget is not None and detail['estimated_refine_ms'] > budget+1e-6
        if save_masks:
            np.save(destination/f'image_{len(rows)+1:04d}.npy',mask.numpy())
        rows.append(row)
    # All profiling happens after ALL unprofiled images, avoiding profiler/allocator
    # effects on subsequent headline timing and reserved-memory measurements.
    if profile_stages or profile_operations:
        for record,row in zip(records,rows):
            image = read_image(record['image'],cfg)
            budget = cfg['routing'].get('max_refine_ms')
            stage,ops = profile_prediction(pipeline,image,mode,profile_operations)
            row['stage_ms'] = stage['stage_ms']
            measured = sum(stage['stage_ms'].get(k,0.) for k in ('low','high','acceptance'))
            row['profiled_refine_ms'] = measured
            row['profiled_refine_budget_overrun'] = budget is not None and measured > budget
            if ops is not None:
                row.update(ops)
                operation_counts.append(ops['estimated_flops'])
    tp = cm.diagonal().double()
    union = cm.sum(0)+cm.sum(1)-tp
    report = {'mIoU':miou(cm),'per_class_IoU':[float(t/u) if u else None for t,u in zip(tp,union)],
              'confusion':cm.tolist(),**summarize_latency(times),**totals,
              'correction_pairs_predicted_to_true':pair_c.tolist(),'damage_pairs_true_to_predicted':pair_d.tolist(),
              'confident_error_recovery_rate':totals['recovered_confident_errors']/totals['confident_errors'] if totals['confident_errors'] else None,
              'actions':actions,'refinement_calls':calls,
              'peak_gpu_bytes':peak_allocated if pipeline.device.type=='cuda' else None,
              'peak_gpu_reserved_bytes':peak_reserved if pipeline.device.type=='cuda' else None,
              'cpu_process_peak_rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if sys.platform=='darwin' else 1024),
              'cpu_memory_scope':'process lifetime high-water mark, including loading/diagnostics/profiling; not isolated inference',
              'mean_estimated_flops':float(np.mean(operation_counts)) if operation_counts else None,
              'operation_count_scope':'PyTorch profiler supported operators only; multiply-add counted as two FLOPs; excludes unestimated operators',
              'mode':mode,'warmup':warmup,'passes':passes,'environment':environment(pipeline.device),
              'image_ids':[r['id'] for r in records],'config':cfg,'provenance':pipeline.provenance,
              'timing_excludes':'decoding/file I/O, checkpoint loading, output writing, diagnostic construction and extra profiling passes',
              'timing_includes':'decoded CPU input to completed CPU mask, CPU work, transfers, all enabled inference modules',
              'budget_scope':'estimated refinement only; profiled comparisons include profiling overhead; not total-latency guarantees',
              'images':rows}
    (destination/'metrics.json').write_text(json.dumps(report,indent=2))
    print(json.dumps({k:v for k,v in report.items() if k not in {'images','config','confusion','provenance'}},indent=2),flush=True)
    return report
