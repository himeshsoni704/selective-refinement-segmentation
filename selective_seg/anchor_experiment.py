"""Validation study of anchor-centred context. See readme.txt."""
import argparse
import json
import random
import shutil
import statistics
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .anchor_context import ACTIONS, VIEWS, ContextReader, allocate, high_logits
from .benchmark import environment
from .data import load_config, read_manifest, read_sample
from .diagnostics import write
from .engine import Pipeline, amp, config_signature, fingerprint, sync
from .evaluation import correction_diagnostics
from .geometry import center_prediction, resize_long, tiles, tile_descriptor
from .hybrid_evidence import area, crop_input, pair_update
from .hybrid_metrics import Metrics, paired_bootstrap
from .objectives import confusion_matrix
from .specialist import counts_metrics
from .training import seed_all

CONTROLS = ("keep", "legacy_low", "legacy_high", "anchor_high", "wide_native", "wide_ring_control", "wide_resized")


def prediction(base, reader, image, pc, tile, action, cache):
    if action == "keep":
        return area(pc, tile)[0]
    if action.startswith("legacy_"):
        level = 1 if action == "legacy_low" else 2
        return base.refine(image, pc, tile, level, return_details=True)["probabilities"][0]
    if action == "anchor_high":
        logits = high_logits(base, image, pc, tile)
    elif action.startswith("wide_"):
        logits = high_logits(base, image, pc, tile, 896, 640 if action == "wide_resized" else 896,
                             action == "wide_ring_control")
    else:
        local = high_logits(base, image, pc, tile)
        logits = reader(local, image, tile, action, cache)
    return logits[0].softmax(0).cpu()


def coarse_overview(base, image, mode):
    if mode not in ("coarse768", "coarse1024"):
        return base.overview(image)
    side = int(mode[6:])
    with amp(base.device):
        out = base.coarse(resize_long(image[None], side).to(base.device))
    logits = F.interpolate(out["logits"].float().cpu(), image.shape[-2:], mode="bilinear",
                           align_corners=False)
    return logits.softmax(1), out["features"].float().cpu()


def load_pair_model(path, base, groups):
    from .specialist import MultiScaleSpecialist, settings, signature
    ck = torch.load(path, map_location="cpu", weights_only=True)
    cfg = ck["config"]
    if config_signature(cfg) != config_signature(base.cfg):
        raise ValueError("Specialist preprocessing or geometry differs")
    if {r["image"] for s in ("selector","val") for r in groups[s]} & set(ck["train_paths"]):
        raise ValueError("Specialist training overlaps development outcomes")
    pair = settings(cfg)
    if (pair["agriculture"], pair["rangeland"]) != (1,2):
        raise ValueError("Expected source IDs agriculture=1, rangeland=2")
    if ck["stage"] == "hybrid_specialist":
        from .hybrid_specialist import load
        model, _ = load(cfg, path, base.device)
    elif ck["stage"] == "specialist" and ck["signature"] == signature(cfg):
        model = MultiScaleSpecialist(cfg).to(base.device)
        model.load_state_dict(ck["model"])
    else:
        raise ValueError("Incompatible specialist")
    return model.eval().requires_grad_(False), cfg


def grid_for(image, cfg):
    return tiles(*image.shape[-2:], cfg["tile_size"])


def descriptors(image, pc, features, grid, cfg):
    return torch.cat([tile_descriptor(image[None], pc, features, t, cfg) for t in grid])


def train_reader(base, records, reader, steps, rng, out, detail_only=False):
    reader.train()
    opt = torch.optim.AdamW(reader.parameters(), lr=3e-4)
    ignore = base.cfg["ignore_index"]
    actions = list(ACTIONS[1:])
    for step in range(steps):
        if step % len(actions) == 0:
            rng.shuffle(actions)
        rec = rng.choice(records)
        image, truth = read_sample(rec, base.cfg)
        grid = grid_for(image, base.cfg)
        eligible = [t for t in grid if (area(truth, t) != ignore).any()]
        if not eligible:
            raise ValueError(f"All-ignore training image: {rec['id']}")
        tile = rng.choice(eligible)
        pc, _ = base.overview(image)
        target = area(truth, tile)[None].to(base.device)
        local = high_logits(base, image, pc, tile)
        logits = reader(local, image, tile, "detail" if detail_only else actions[step % len(actions)], {})
        loss = F.cross_entropy(logits, target, ignore_index=ignore)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Reader loss at step {step}, image {rec['id']}")
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % 100 == 0:
            print(f"reader step {step+1}/{steps} loss {loss.item():.5f}", flush=True)
    torch.save(reader.state_dict(), out)
    reader.eval()


def brier_sum(p, truth, valid):
    y = truth[valid]
    q = p[:, valid]
    return (1 + q.square().sum(0) - 2*q.gather(0, y[None])[0]).sum(dtype=torch.float64)


@torch.no_grad()
def outcomes(base, reader, rec):
    cfg = base.cfg
    image, truth = read_sample(rec, cfg)
    pc, f = base.overview(image)
    grid = grid_for(image, cfg)
    z = descriptors(image, pc, f, grid, cfg)
    gains, times, counts = [], [], []
    valid_total = max(1, int((truth != cfg["ignore_index"]).sum()))
    for tile in grid:
        target = area(truth, tile)
        valid = target != cfg["ignore_index"]
        losses, elapsed, cm = [], [], []
        for action in ACTIONS:
            # Independent alternatives, including cold scene setup. Live runs share it.
            sync(base.device)
            start = time.perf_counter()
            p = prediction(base, reader, image, pc, tile, action, {})
            sync(base.device)
            elapsed.append((time.perf_counter()-start)*1000)
            losses.append(float(brier_sum(p, target, valid)))
            cm.append(confusion_matrix(p.argmax(0), target, cfg["classes"], cfg["ignore_index"]))
        gains.append([(losses[0]-loss)/valid_total for loss in losses])
        times.append(elapsed)
        counts.append(torch.stack(cm))
    return dict(id=rec["id"], z=z, gain=torch.tensor(gains), ms=torch.tensor(times),
                cm=torch.stack(counts))


def policy_fit(rows, epochs):
    z = torch.cat([r["z"] for r in rows])
    y = torch.cat([r["gain"][:, 1:] for r in rows])
    valid = torch.cat([r["cm"][:, 0].sum((1,2)) > 0 for r in rows])
    z, y = z[valid], y[valid]
    if len(z) < 2 or not torch.isfinite(y).all():
        raise ValueError("Insufficient valid selector targets")
    mean, std = z.mean(0), z.std(0, unbiased=False).clamp_min(1e-5)
    scale = y.std(unbiased=False).clamp_min(1e-6)
    model = nn.Sequential(nn.Linear(z.shape[1],64), nn.ReLU(), nn.Linear(64,len(ACTIONS)-1))
    opt = torch.optim.AdamW(model.parameters(), lr=.001)
    for _ in range(epochs):
        for idx in torch.randperm(len(z)).split(128):
            loss = F.mse_loss(model((z[idx]-mean)/std), y[idx]/scale)
            opt.zero_grad()
            loss.backward()
            opt.step()
    prices = torch.cat([r["ms"] for r in rows]).median(0).values
    prices[0] = 0
    return model.eval(), mean, std, scale, prices


def eligible(z, fraction):
    # Descriptor ends in six class means, five coarse stats, four RGB stats.
    entropy = z[:, -9]
    allowed = np.zeros(len(z), dtype=bool)
    allowed[entropy.argsort(descending=True, stable=True)[:int(np.ceil(len(z)*fraction))]] = True
    return allowed


def decisions(mode, z, bundle, allowance, fraction, seed):
    model, mean, std, scale, prices = bundle
    allowed = eligible(z, fraction)
    if mode == "learned":
        scores = F.pad(model((z-mean)/std)*scale, (1,0)).detach().numpy()
    else:
        scores = np.zeros((len(z),len(ACTIONS)))
        score = z[:, -9].numpy() if mode != "random" else np.random.default_rng(seed).random(len(z))
        fixed = mode if mode in ACTIONS else "large"
        scores[:, ACTIONS.index(fixed)] = score + 1e-8
    return allocate(scores, prices.numpy(), allowance, allowed)


@torch.no_grad()
def benchmark(base, reader, records, mode, bundle, budget, fraction, repetitions, seed, specialist=None):
    cfg = base.cfg
    metric = Metrics(cfg["classes"], ignore=cfg["ignore_index"])
    rows, latencies = [], []
    calls = {name:0 for name in (*ACTIONS, *CONTROLS)}
    peak = 0
    for i, rec in enumerate(records):
        image, truth = read_sample(rec, cfg)
        samples, acquisition_times = [], []
        for repeat in range(repetitions + int(i == 0)):
            if base.device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(base.device)
            sync(base.device)
            start = time.perf_counter()
            pc, f = coarse_overview(base, image, mode)
            grid = grid_for(image, cfg)
            if budget is None:
                selected = [("keep" if mode.startswith("coarse") else mode)]*len(grid)
            else:
                z = descriptors(image, pc, f, grid, cfg)
                selected = [ACTIONS[a] for a in decisions(mode,z,bundle,budget,fraction,seed+i)]
            result = pc.clone()
            cache = {}
            sync(base.device)
            acquisition_start = time.perf_counter()
            for tile, action in zip(grid, selected):
                if action != "keep":
                    p = prediction(base,reader,image,pc,tile,action,cache)
                    if specialist is not None:
                        pair_model, pair_cfg = specialist
                        pair_logits = pair_model(crop_input(image,tile,pair_cfg,base.device))
                        pair_p = center_prediction(pair_logits.float(),tile,pair_cfg).softmax(1).cpu()
                        p = pair_update(p[None],pair_p,(1,2),p[[1,2]].sum(0)[None]>=.35)[0]
                    area(result,tile)[0].copy_(p)
            sync(base.device)
            acquisition_times.append(1000*(time.perf_counter()-acquisition_start))
            mask = result[0].argmax(0)
            sync(base.device)
            samples.append(1000*(time.perf_counter()-start))
            if base.device.type == "cuda":
                peak = max(peak, torch.cuda.max_memory_allocated(base.device))
        elapsed = statistics.median(samples[-repetitions:])
        latencies.append(elapsed)
        for action in selected:
            calls[action] += 1
        cm = metric.add(result[0],truth)
        diag = correction_diagnostics(pc[0].argmax(0),mask,truth,cfg)
        rows.append(dict(id=rec["id"], confusion=cm, ms=elapsed,
                         acquisition_ms=statistics.median(acquisition_times[-repetitions:]), actions=selected, **diag))
    return dict(**metric.report(), images=rows, calls=calls, peak_gpu_bytes=peak,
                mean_ms=float(np.mean(latencies)), median_ms=float(np.median(latencies)),
                p95_ms=float(np.percentile(latencies,95)), incremental_budget_ms=budget,
                observed_acquisition_overrun_rate=(sum(r["acquisition_ms"]>budget for r in rows)/len(rows)
                                                   if budget is not None else None),
                specialist_calls=sum(calls.values())-calls["keep"] if specialist else 0)


def experiment(args, root):
    cfg = load_config(args.config)
    if (cfg["classes"], cfg["bands"],cfg["tile_size"],cfg["border"]) != (6,3,512,64):
        raise ValueError("This protocol requires six classes, RGB, 512 tiles and a 64 halo")
    if cfg.get("research"):
        raise ValueError("Use the baseline config without optional descriptor extensions")
    records = read_manifest(args.manifest)
    groups = {s:[r for r in records if r["split"] == s] for s in ("train","selector","val")}
    if not all(groups.values()):
        raise ValueError("Train, selector and val splits are required")
    base = Pipeline.from_run(cfg,args.base_run,args.device)
    for stage in ("coarse","low","high"):
        ck = torch.load(Path(args.base_run)/(stage+".pt"),map_location="cpu",weights_only=True)
        if set(ck["train_paths"]) & {r["image"] for r in groups["selector"]+groups["val"]}:
            raise ValueError(f"{stage} checkpoint overlaps selector/val images")
    seed_all(args.seed)
    reader = ContextReader(6).to(base.device)
    write(root/"protocol.json",dict(args=vars(args), config=cfg, views=VIEWS, actions=ACTIONS,
         specialist_sha256=fingerprint(args.specialist) if args.specialist else None,
         manifest_sha256=fingerprint(args.manifest), checkpoints=base.provenance,
         code=subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip(),
         environment=environment(base.device), free_disk_bytes=shutil.disk_usage(root).free,
         splits={s:[r["id"] for r in rs] for s,rs in groups.items()}))
    rng = random.Random(args.seed)
    if args.stage == "pilot":
        t = time.perf_counter()
        train_reader(base,groups["train"],reader,5,rng,root/"pilot_reader.pt")
        sync(base.device)
        step_s = (time.perf_counter()-t)/5
        t = time.perf_counter()
        row = outcomes(base,reader,groups["val"][0])
        image_s = time.perf_counter()-t
        report = dict(step_seconds=step_s, exhaustive_image_seconds=image_s,
            projected_both_reader_minutes=2*args.steps*step_s/60,
            projected_selector_plus_val_cache_minutes=(len(groups["selector"])+len(groups["val"]))*image_s/60,
            approx_cache_bytes=sum(v.numel()*v.element_size() for v in row.values() if torch.is_tensor(v))
            *(len(groups["selector"])+len(groups["val"])), free_disk_bytes=shutil.disk_usage(root).free,
            peak_gpu_bytes=torch.cuda.max_memory_allocated(base.device) if base.device.type=="cuda" else None)
        report["approx_benchmark_minutes"] = len(groups["val"])*args.passes*image_s * (
            len(CONTROLS)+8+len(args.budgets)*(7+int(bool(args.specialist)))) / len(ACTIONS) / 60
        write(root/"pilot.json",report)
        print(json.dumps(report,indent=2),flush=True)
        return
    def run_report(name, mode, bundle=None, budget=None, specialist=None):
        report = benchmark(base,reader,groups["val"],mode,bundle,budget,args.candidate_fraction,
                           args.passes,args.seed,specialist)
        if name != "keep":
            baseline = json.loads((root/"keep.json").read_text())
            report["paired_vs_coarse"] = paired_bootstrap(report["images"],baseline["images"],1000,args.seed)
        write(root/(name+".json"),report)
        print(name,report["mIoU"],report["mean_ms"],flush=True)

    if args.stage == "locked-test":
        study = Path(args.trained_run)
        original = json.loads((study/"protocol.json").read_text())
        test = [r for r in records if r["split"]=="test"]
        if not test or original["manifest_sha256"] != fingerprint(args.manifest) or original["checkpoints"] != base.provenance or original["config"] != cfg:
            raise ValueError("Locked study inputs differ or test split is empty")
        if original["args"]["stage"] != "study" or json.loads((study/"status.json").read_text())["status"] != "complete":
            raise ValueError("Expected a completed development study")
        for stage in ("coarse","low","high"):
            ck = torch.load(Path(args.base_run)/(stage+".pt"),map_location="cpu",weights_only=True)
            if {r["image"] for r in test} & set(ck["train_paths"]):
                raise ValueError("Test overlaps frozen expert training")
        ck = torch.load(study/"policy.pt",map_location="cpu",weights_only=True)
        if ck["reader_sha256"] != fingerprint(study/"reader.pt"):
            raise ValueError("Reader and policy differ")
        reader.load_state_dict(torch.load(study/"reader.pt",map_location="cpu",weights_only=True))
        reader.eval()
        model = nn.Sequential(nn.Linear(len(ck["mean"]),64), nn.ReLU(), nn.Linear(64,len(ACTIONS)-1))
        model.load_state_dict(ck["model"])
        bundle = (model.eval(), ck["mean"], ck["std"], ck["scale"], ck["prices"])
        for mode in ("keep","entropy","learned"):
            cap = None if mode == "keep" else args.budgets[0]
            report = benchmark(base,reader,test,mode,bundle,cap,args.candidate_fraction,args.passes,args.seed)
            if mode != "keep":
                report["paired_vs_coarse"] = paired_bootstrap(report["images"],reference,1000,args.seed)
            else:
                reference = report["images"]
            write(root/(mode+".json"),report)
        return
    for mode in (*CONTROLS, "coarse768", "coarse1024"):
        run_report(mode,mode)
    if args.stage == "controls":
        return
    initial = {k:v.clone() for k,v in reader.state_dict().items()}
    train_reader(base,groups["train"],reader,args.steps,rng,root/"detail_only_reader.pt",detail_only=True)
    run_report("trained_detail_only","detail")
    reader.load_state_dict(initial)
    train_reader(base,groups["train"],reader,args.steps,random.Random(args.seed),root/"reader.pt")
    for mode in ACTIONS[1:]:
        run_report("dense_"+mode, mode)
    rows = {}
    for split in ("selector","val"):
        rows[split] = []
        for rec in groups[split]:
            rows[split].append(outcomes(base,reader,rec))
        torch.save(rows[split],root/(split+"_outcomes.pt"))
    bundle = policy_fit(rows["selector"],args.epochs)
    model, mean, std, scale, prices = bundle
    torch.save(dict(model=model.state_dict(),mean=mean,std=std,scale=scale,prices=prices,
                    reader_sha256=fingerprint(root/"reader.pt")),root/"policy.pt")
    specialist = load_pair_model(args.specialist, base, groups) if args.specialist else None
    for budget in args.budgets:
        for fraction in sorted({1.,args.candidate_fraction}):
            cm = torch.zeros(6,6,dtype=torch.long)
            for row in rows["val"]:
                chosen = allocate(row["gain"].numpy(),prices.numpy(),budget,eligible(row["z"],fraction))
                cm += row["cm"][torch.arange(len(chosen)),chosen].sum(0)
            write(root/f"label_assisted_{budget:g}ms_cap{fraction:g}.json",counts_metrics(cm))
        for mode in ("random","entropy","detail","medium","scene","all","learned"):
            run_report(f"{mode}_{budget:g}ms",mode,bundle,budget)
        if specialist:
            run_report(f"learned_plus_pair_{budget:g}ms","learned",bundle,budget,specialist)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage",choices=("pilot","controls","study","locked-test"))
    p.add_argument("--config",default="configs/deepglobe_rgb.json")
    p.add_argument("--manifest",default="runs/deepglobe/manifest.json")
    p.add_argument("--base-run",default="runs/deepglobe")
    p.add_argument("--output",required=True,help="Fresh directory")
    p.add_argument("--trained-run",help="Completed development study for locked-test")
    p.add_argument("--specialist",help="Optional frozen binary checkpoint for post-fusion paired ablation")
    p.add_argument("--steps",type=int,default=2000)
    p.add_argument("--epochs",type=int,default=60)
    p.add_argument("--passes",type=int,default=3)
    p.add_argument("--seed",type=int,default=42)
    p.add_argument("--candidate-fraction",type=float,default=1.)
    p.add_argument("--budgets",type=float,nargs="+",default=[100.,250.])
    p.add_argument("--device",default="cuda")
    args = p.parse_args()
    if min(args.steps,args.epochs,args.passes)<1 or not 0<args.candidate_fraction<=1:
        p.error("Require positive counts and candidate fraction in (0,1]")
    if any(not np.isfinite(b) or b<0 for b in args.budgets) or len(set(args.budgets))!=len(args.budgets):
        p.error("Budgets must be finite, distinct, nonnegative milliseconds")
    if args.stage == "locked-test" and (not args.trained_run or len(args.budgets)!=1 or args.specialist):
        p.error("locked-test needs --trained-run and one prespecified --budgets value (core policy only)")
    root = Path(args.output)
    root.mkdir(parents=True,exist_ok=False)
    try:
        experiment(args,root)
    except Exception as exc:
        write(root/"status.json",dict(status="failed",error=repr(exc)))
        raise
    write(root/"status.json",dict(status="complete"))


if __name__ == "__main__":
    main()
