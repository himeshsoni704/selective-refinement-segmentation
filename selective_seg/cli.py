import argparse
import json
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from .data import load_config, read_manifest, read_image, read_sample
from .engine import Pipeline, config_signature, fingerprint
from .objectives import confusion_matrix, miou
from .research import signature, settings
from .training import train_segmenter, cache_targets, train_selector, calibrate


def apply_calibration(cfg, path, run):
    if not path:
        return cfg
    c = json.loads(Path(path).read_text())
    if c["signature"] != config_signature(cfg) or c["utility"] != cfg["utility"]:
        raise ValueError("Calibration/config mismatch")
    if c["selector_sha256"] != fingerprint(Path(run)/"selector.pt"):
        raise ValueError("Calibration is stale: selector changed")
    for stage in ("coarse","low","high"):
        if c["provenance"][stage] != fingerprint(Path(run)/f"{stage}.pt"):
            raise ValueError("Calibration is stale: segmenter changed")
    if c.get('research_signature',signature({})) != signature(cfg):
        raise ValueError('Calibration research configuration changed')
    for name in ('scout','acceptance'):
        if settings(cfg)[name] and c['provenance'].get(name) != fingerprint(Path(run)/(name+'.pt')):
            raise ValueError('Calibration auxiliary changed: '+name)
    if settings(cfg)['escalation'] and c.get('escalation_sha256') != fingerprint(Path(run)/'escalation.pt'):
        raise ValueError('Calibration escalation changed')
    cfg["routing"] = c["routing"]
    return cfg


def save_prediction(mask, details, prefix):
    prefix = Path(prefix)
    prefix.parent.mkdir(parents=True,exist_ok=True)
    array = mask.numpy()
    np.save(str(prefix)+".npy",array)
    if array.max() < 256:
        Image.fromarray(array.astype(np.uint8)).save(str(prefix)+".png")
    Path(str(prefix)+".json").write_text(json.dumps(details,indent=2))


def main():
    parser = argparse.ArgumentParser(description="Train and evaluate selective refinement")
    parser.add_argument("command",choices=["validate-data","train-coarse","train-low","train-high",
                                          "cache","train-selector","train-scout","train-acceptance","train-escalation","calibrate","predict","evaluate","benchmark","analyze-cache"])
    parser.add_argument("--config",required=True)
    parser.add_argument("--manifest")
    parser.add_argument("--run",default="runs/experiment")
    parser.add_argument("--device",default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--steps",type=int)
    parser.add_argument("--epochs",type=int)
    parser.add_argument("--resume",action="store_true")
    parser.add_argument("--split",choices=["selector","val","test"],default="selector")
    parser.add_argument("--output")
    parser.add_argument("--train-cache")
    parser.add_argument("--val-cache")
    parser.add_argument("--calibration")
    parser.add_argument("--image")
    parser.add_argument("--warmup",type=int,default=20)
    parser.add_argument("--passes",type=int,default=3)
    parser.add_argument('--mode',choices=['selective','coarse','low','high','random','entropy'],default='selective')
    parser.add_argument('--profile-stages',action='store_true')
    parser.add_argument('--profile-operations',action='store_true')
    parser.add_argument('--save-masks',action='store_true')
    args = parser.parse_args()
    cfg = load_config(args.config)
    run = Path(args.run)
    run.mkdir(parents=True,exist_ok=True)
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        parser.error("CUDA is unavailable; use --device cpu only for development/smoke tests")
    if args.command in {"validate-data","train-coarse","train-low","train-high","train-scout","train-acceptance","cache","evaluate","benchmark"}:
        if not args.manifest:
            parser.error("This command requires --manifest")
        records = read_manifest(args.manifest)
    if args.command == "validate-data":
        for r in records:
            image, mask = read_sample(r,cfg)
            print(r["id"],r["split"],list(image.shape),torch.unique(mask).tolist())
    elif args.command in {"train-coarse","train-low","train-high"}:
        train_segmenter(cfg,records,run,args.command.split('-')[1],args.device,args.steps,args.resume)
    elif args.command in {'train-scout','train-acceptance'}:
        from .auxiliary import train_auxiliary
        train_auxiliary(cfg,records,run,args.command.split('-')[1],args.device,args.epochs)
    elif args.command == 'train-escalation':
        from .auxiliary import train_escalation
        train_escalation(cfg,args.train_cache or run/'selector_targets.pt',args.val_cache or run/'val_targets.pt',run,args.device,args.epochs)
    elif args.command == "cache":
        cache_targets(cfg,records,run,args.split,args.output or run/f"{args.split}_targets.pt",args.device)
    elif args.command == "train-selector":
        train_selector(cfg,args.train_cache or run/"selector_targets.pt",args.val_cache or run/"val_targets.pt",
                       run,args.device,args.epochs)
    elif args.command == 'analyze-cache':
        from .training import load_cache
        from .evaluation import analyze_cache
        report = analyze_cache(cfg,load_cache(args.val_cache or run/'val_targets.pt',cfg,'val'))
        destination = Path(args.output or run/'validation_diagnostics.json')
        destination.write_text(json.dumps(report,indent=2))
        print(json.dumps(report,indent=2))
    elif args.command == "calibrate":
        calibrate(cfg,args.val_cache or run/"val_targets.pt",run,args.output or run/"calibration.json")
    elif args.command in {"predict","evaluate","benchmark"}:
        if not args.calibration:
            print("WARNING: using provisional, uncalibrated T1/T2 and probability scores",flush=True)
        cfg = apply_calibration(cfg,args.calibration,run)
        pipeline = Pipeline.from_run(cfg,run,args.device,need_selector=args.mode == 'selective')
        if args.command == "predict":
            if not args.image:
                parser.error("predict requires --image")
            mask, details = pipeline.predict(read_image(args.image,cfg),args.mode)
            save_prediction(mask,details,args.output or run/"prediction")
        else:
            selected = [r for r in records if r["split"] == args.split]
            if args.command == 'evaluate' and args.split != 'test':
                parser.error('evaluate uses --split test; use benchmark --split val for development')
            if args.command == 'benchmark' and args.split not in {'val','test'}:
                parser.error('benchmark requires --split val or test')
            if not selected or args.passes < 1 or args.warmup < 0:
                parser.error("Need test records, passes >=1 and warmup >=0")
            from .benchmark import evaluate
            evaluate(pipeline,selected,args.output or run/('test_results_'+args.mode),args.mode,
                     args.warmup,args.passes,args.profile_stages,args.profile_operations,args.save_masks)


if __name__ == '__main__':
    main()
