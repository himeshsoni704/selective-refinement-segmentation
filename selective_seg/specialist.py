"""Independent agriculture/rangeland feasibility experiment; never edits six-class output."""
import hashlib
import json
import random
import time
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from .data import read_sample
from .engine import amp, config_signature, fingerprint, load_segmenter, Pipeline, sync
from .geometry import tiles, crop_context, center_prediction


def settings(cfg):
    s = dict(backbones='combined', resnet='resnet18', levels='all', projection_dim=32,
             input_size=320, pretrained=True, freeze_backbones=False, agriculture=1,
             rangeland=2, lr=1e-4, steps=10000, validate_every=500,
             class_weighting='sqrt_inverse', max_class_weight=4.0)
    s.update(cfg.get('specialist', {}))
    if cfg['bands'] != 3 or cfg['spectral']['enabled']:
        raise ValueError('Specialist experiment requires unmodified RGB inputs')
    if s['backbones'] not in {'resnet', 'efficientnet', 'combined'}:
        raise ValueError('Unknown specialist backbones')
    if s['resnet'] not in {'resnet18', 'resnet50'} or s['levels'] not in {'all', 'shallow', 'deep'}:
        raise ValueError('Invalid specialist backbone or feature levels')
    if s['agriculture'] == s['rangeland'] or any(s[k] not in range(cfg['classes']) for k in ('agriculture','rangeland')):
        raise ValueError('Specialist class IDs must be distinct valid classes')
    if s['projection_dim'] < 1 or s['input_size'] < 32 or s['steps'] < 1 or s['validate_every'] < 1 or s['lr'] <= 0:
        raise ValueError('Invalid specialist size or training settings')
    if s['class_weighting'] not in {'none','sqrt_inverse'} or s['max_class_weight'] < 1:
        raise ValueError('Invalid specialist class weighting')
    if s['freeze_backbones'] and not s['pretrained']:
        raise ValueError('Do not freeze randomly initialized backbones')
    return s


def signature(cfg):
    return hashlib.sha256(json.dumps({'version':1, 'base':config_signature(cfg),
                                     'specialist':settings(cfg)}, sort_keys=True).encode()).hexdigest()


def binary_target(truth, cfg):
    s = settings(cfg)
    target = torch.full_like(truth, 255)
    target[truth == s['agriculture']] = 0
    target[truth == s['rangeland']] = 1
    return target


class MultiScaleSpecialist(nn.Module):
    """Actual torchvision stage outputs, projected to a shared spatial grid.

    Conditional binary probabilities are undefined evidence for other land classes.
    Frozen BN statistics make batch-one crop training safe; affine weights still learn.
    """
    def __init__(self, cfg, initialize_pretrained=False):
        super().__init__()
        from torchvision import models
        s = settings(cfg)
        self.settings = s
        self.branches = nn.ModuleDict()
        self.nodes = {}
        use_weights = initialize_pretrained and s['pretrained']
        if s['backbones'] in {'resnet','combined'}:
            weights = (models.ResNet18_Weights.DEFAULT if s['resnet'] == 'resnet18'
                       else models.ResNet50_Weights.DEFAULT) if use_weights else None
            net = getattr(models, s['resnet'])(weights=weights)
            self.branches['resnet'] = nn.Sequential(net.conv1, net.bn1, net.relu, net.maxpool,
                                                    net.layer1, net.layer2, net.layer3, net.layer4)
            self.nodes['resnet'] = [4,5,6,7]
        if s['backbones'] in {'efficientnet','combined'}:
            net = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT if use_weights else None)
            self.branches['efficientnet'] = net.features[:8]
            self.nodes['efficientnet'] = [2,3,5,7]
        indices = {'all':[0,1,2,3], 'shallow':[0,1], 'deep':[2,3]}[s['levels']]
        self.nodes = {k:[v[i] for i in indices] for k,v in self.nodes.items()}
        for name in self.branches:
            self.branches[name] = self.branches[name][:max(self.nodes[name])+1]
        # Probe the installed implementation rather than assuming channel widths.
        self.branches.eval()
        with torch.no_grad():
            features = self._features(torch.zeros(1,3,64,64))
        self.feature_spec = [{'name':name, 'channels':f.shape[1],
                              'stride':64//f.shape[-1]} for name,f in features]
        d = s['projection_dim']
        self.projections = nn.ModuleList([nn.Conv2d(f.shape[1],d,1) for _,f in features])
        self.fusion = nn.Sequential(nn.Conv2d(d*len(features),d,3,padding=1),
                                    nn.GroupNorm(1,d), nn.GELU(), nn.Conv2d(d,2,1))
        self.register_buffer('mean',torch.tensor([.485,.456,.406])[None,:,None,None])
        self.register_buffer('std',torch.tensor([.229,.224,.225])[None,:,None,None])
        if s['freeze_backbones']:
            self.branches.requires_grad_(False)
        self.train()

    def _features(self, x):
        result = []
        for name,branch in self.branches.items():
            value = x
            for i,layer in enumerate(branch):
                value = layer(value)
                if i in self.nodes[name]:
                    result.append((f'{name}.{i}',value))
                if i == max(self.nodes[name]):
                    break
        return result

    def train(self, mode=True):
        super().train(mode)
        for module in self.branches.modules():
            if isinstance(module, nn.BatchNorm2d):
                module.eval()
        if self.settings['freeze_backbones']:
            self.branches.eval()
        return self

    def forward(self, x):
        features = self._features((x-self.mean)/self.std)
        size = features[0][1].shape[-2:]
        projected = [F.interpolate(layer(f),size,mode='bilinear',align_corners=False)
                     for layer,(_,f) in zip(self.projections,features)]
        return self.fusion(torch.cat(projected,1))


def masked_loss(logits, truth, weights=None):
    if not (truth != 255).any():
        return logits.sum()*0
    return F.cross_entropy(logits,truth,weight=weights,ignore_index=255)


def crop_logits(model, image, tile, cfg, device):
    crop = crop_context(image[None],tile,cfg['tile_size'],cfg['border'])
    crop = F.interpolate(crop,(settings(cfg)['input_size'],)*2,mode='bilinear',
                         align_corners=False,antialias=True).to(device)
    with amp(device):
        logits = model(crop)
    return center_prediction(logits.float(),tile,cfg)


def counts_metrics(cm):
    cm = cm.double()
    union = cm.sum(0)+cm.sum(1)-cm.diag()
    iou = [float(cm[i,i]/union[i]) if union[i] else None for i in range(len(cm))]
    present = [v for v in iou if v is not None]
    return {'confusion':cm.long().tolist(), 'per_class_IoU':iou,
            'mIoU':sum(present)/len(present) if present else None,
            'accuracy':float(cm.diag().sum()/cm.sum()) if cm.sum() else None}


def confusion(pred, target):
    valid = target != 255
    return torch.bincount(2*target[valid]+pred[valid],minlength=4).reshape(2,2)


@torch.no_grad()
def evaluate_model(model, cfg, records, device, coarse=None):
    """Every tile, unique central pixels; no ground-truth candidate gating."""
    model.eval()
    cm, coarse_cm, six_cm = torch.zeros(2,2,dtype=torch.long), torch.zeros(2,2,dtype=torch.long), torch.zeros(cfg['classes'],cfg['classes'],dtype=torch.long)
    total_nll = total_brier = 0.
    n = corrected = damaged = confident_wrong = confident_fixed = 0
    bins = torch.zeros(15,3,dtype=torch.float64)
    image_times, crop_times, image_ids = [], [], []
    s = settings(cfg)
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)
    # One untimed warmup, including transfer and resize.
    if records:
        image,_ = read_sample(records[0],cfg)
        crop_logits(model,image,tiles(*image.shape[-2:],cfg['tile_size'])[0],cfg,device)
        sync(device)
    for record in records:
        image,truth = read_sample(record,cfg)
        pc = coarse.overview(image)[0] if coarse else None
        elapsed_image = 0.
        for tile in tiles(*image.shape[-2:],cfg['tile_size']):
            y,x,h,w = tile
            sync(device)
            start = time.perf_counter()
            logits = crop_logits(model,image,tile,cfg,device).cpu()[0]
            sync(device)
            elapsed = (time.perf_counter()-start)*1000
            elapsed_image += elapsed
            crop_times.append(elapsed)
            target = binary_target(truth[y:y+h,x:x+w],cfg)
            valid = target != 255
            prob = logits.softmax(0)
            pred = prob.argmax(0)
            cm += confusion(pred,target)
            if not valid.any():
                continue
            t, p = target[valid], prob[:,valid].T
            confidence, label = p.max(1)
            correct = label == t
            indices = (confidence*15).long().clamp(max=14)
            bins[:,0] += torch.bincount(indices,minlength=15)
            bins[:,1] += torch.bincount(indices,weights=confidence.double(),minlength=15)
            bins[:,2] += torch.bincount(indices,weights=correct.double(),minlength=15)
            total_nll += float(F.cross_entropy(logits[None],target[None],ignore_index=255,reduction='sum'))
            total_brier += float((p-F.one_hot(t,2)).square().sum())
            n += len(t)
            if pc is not None:
                tile_pc = pc[0,:,y:y+h,x:x+w]
                cp = tile_pc[[s['agriculture'],s['rangeland']]]
                cp = cp/cp.sum(0,keepdim=True).clamp_min(1e-8)
                coarse_cm += confusion(cp.argmax(0),target)
                six_pred = tile_pc.argmax(0)
                six_truth = truth[y:y+h,x:x+w]
                six_cm += torch.bincount(cfg['classes']*six_truth[valid]+six_pred[valid],minlength=cfg['classes']**2).reshape(cfg['classes'],cfg['classes'])
                binary_as_six = torch.where(pred == 0,s['agriculture'],s['rangeland'])
                wrong = six_pred != six_truth
                corrected += int((valid & wrong & (binary_as_six == six_truth)).sum())
                damaged += int((valid & ~wrong & (binary_as_six != six_truth)).sum())
                confident = valid & wrong & (tile_pc.max(0).values >= .9)
                confident_wrong += int(confident.sum())
                confident_fixed += int((confident & (binary_as_six == six_truth)).sum())
        image_times.append(elapsed_image)
        image_ids.append(record['id'])
    if not n:
        raise ValueError('Evaluation has no agriculture/rangeland pixels')
    report = {'binary_class_order':['agriculture','rangeland'],
              'source_class_ids':[s['agriculture'],s['rangeland']],
              'device':str(device), 'torch_version':str(torch.__version__),
              'specialist':counts_metrics(cm), 'valid_pixels':n, 'NLL':total_nll/n,
              'Brier':total_brier/n, 'ECE':float((bins[:,1]-bins[:,2]).abs().sum()/n),
              'calibrated':False, 'image_ids':image_ids, 'feature_spec':model.feature_spec,
              'parameters':sum(p.numel() for p in model.parameters()),
              'mean_specialist_image_ms':sum(image_times)/len(image_times),
              'median_crop_ms':float(torch.tensor(crop_times).median()),
              'p95_crop_ms':float(torch.tensor(crop_times).quantile(.95)),
              'peak_gpu_bytes':torch.cuda.max_memory_allocated(device) if device.type == 'cuda' else None,
              'timing_note':'One warmup; single pass. Includes crop/resize/transfer/forward/central logits download; excludes coarse, metrics, I/O. Not full-pipeline benchmark.'}
    if coarse:
        report.update(coarse_conditional_binary=counts_metrics(coarse_cm),
                      coarse_six_class_on_AR_pixels=counts_metrics(six_cm),
                      diagnostic_correctable_pixels=corrected, diagnostic_damage_pixels=damaged,
                      coarse_confident_errors=confident_wrong, specialist_correct_on_confident_errors=confident_fixed)
    return report


def train_specialist(cfg, records, run, device, steps=None):
    s = settings(cfg)
    device,run = torch.device(device),Path(run)
    run.mkdir(parents=True,exist_ok=True)
    if (run/'specialist.pt').exists():
        raise ValueError('Use a new run directory; specialist resume is not implemented')
    random.seed(cfg['seed'])
    torch.manual_seed(cfg['seed'])
    train = [r for r in records if r['split'] == 'train']
    val = [r for r in records if r['split'] == 'val']
    if not train or not val:
        raise ValueError('Specialist requires separate train and val records')
    eligible, counts = [], torch.zeros(2,dtype=torch.long)
    for record in train:
        _,truth = read_sample(record,cfg)
        target = binary_target(truth,cfg)
        counts += torch.bincount(target[target != 255],minlength=2)
        for tile in tiles(*truth.shape,cfg['tile_size']):
            y,x,h,w = tile
            if (target[y:y+h,x:x+w] != 255).any():
                eligible.append((record,tile))
    if (counts == 0).any():
        raise ValueError('Both agriculture and rangeland must occur in training data')
    weights = (counts.sum()/counts.float()).sqrt() if s['class_weighting'] == 'sqrt_inverse' else torch.ones(2)
    weights = (weights/weights.min()).clamp(max=s['max_class_weight']).to(device)
    model = MultiScaleSpecialist(cfg,initialize_pretrained=True).to(device)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=s['lr'],weight_decay=.01)
    scaler = torch.amp.GradScaler('cuda',enabled=device.type == 'cuda')
    total = s['steps'] if steps is None else steps
    if total < 1:
        raise ValueError('steps must be positive')
    best = -1.
    for step in range(1,total+1):
        model.train()
        record,tile = random.choice(eligible)
        image,truth = read_sample(record,cfg)
        y,x,h,w = tile
        target = binary_target(truth[y:y+h,x:x+w],cfg)[None].to(device)
        optimizer.zero_grad(set_to_none=True)
        loss = masked_loss(crop_logits(model,image,tile,cfg,device),target,weights)
        if not torch.isfinite(loss):
            raise RuntimeError('Nonfinite specialist loss')
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(),1.)
        scaler.step(optimizer)
        scaler.update()
        if step == 1 or step % 100 == 0:
            print(json.dumps({'stage':'specialist','step':step,'loss':float(loss.detach())}),flush=True)
        if step % s['validate_every'] == 0 or step == total:
            report = evaluate_model(model,cfg,val,device)
            score = report['specialist']['mIoU']
            if score > best:
                best = score
                torch.save({'stage':'specialist','signature':signature(cfg),'model':model.state_dict(),
                            'config':cfg,'step':step,'best_val_miou':best,'class_counts':counts,
                            'class_weights':weights.cpu(),'feature_spec':model.feature_spec,
                            'train_paths':[r['image'] for r in train],
                            'val_paths':[r['image'] for r in val]},run/'specialist.pt')
                (run/'specialist_validation.json').write_text(json.dumps(report,indent=2))
            print(json.dumps({'step':step,'val_binary_miou':score,'best':best}),flush=True)


def evaluate_specialist(cfg, records, run, split, device, output, coarse_run=None):
    if split not in {'val','test'}:
        raise ValueError('Specialist evaluation requires val or test')
    device = torch.device(device)
    path = Path(run)/'specialist.pt'
    ck = torch.load(path,map_location='cpu',weights_only=True)
    if ck['stage'] != 'specialist' or ck['signature'] != signature(cfg):
        raise ValueError('Incompatible specialist checkpoint/configuration')
    chosen = [r for r in records if r['split'] == split]
    if not chosen:
        raise ValueError('No evaluation records')
    paths = {r['image'] for r in chosen}
    forbidden = set(ck['train_paths']) | (set(ck['val_paths']) if split == 'test' else set())
    if paths & forbidden:
        raise ValueError('Evaluation overlaps specialist training/tuning images')
    model = MultiScaleSpecialist(cfg).to(device)
    model.load_state_dict(ck['model'])
    coarse = None
    coarse_hash = None
    if coarse_run:
        coarse_path = Path(coarse_run)/'coarse.pt'
        network,base = load_segmenter(coarse_path,cfg,'coarse',device)
        if paths & set(base.get('train_paths',[])):
            raise ValueError('Evaluation overlaps coarse training images')
        coarse = Pipeline(cfg,network,None,None,device=device)
        coarse_hash = fingerprint(coarse_path)
    report = evaluate_model(model,cfg,chosen,device,coarse)
    report.update(split=split,specialist_sha256=fingerprint(path),coarse_sha256=coarse_hash,
                  signature=signature(cfg))
    output = Path(output)
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))
    return report
