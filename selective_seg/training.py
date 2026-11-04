import copy
import json
import random
import time
from pathlib import Path
import torch
from torch.nn import functional as F
from .data import read_sample
from .models import Segmenter, OutcomeSelector, freeze
from .objectives import segmentation_loss, selector_loss, outcome_targets, utility, miou
from .geometry import resize_long, tiles, crop_context, center_prediction, tile_descriptor
from .engine import Pipeline, load_segmenter, config_signature, fingerprint, amp, sync
from .routing import choose_actions


def seed_all(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def segment_batch(record, cfg, stage, model, coarse_pipeline, device, deterministic=False):
    image, mask = read_sample(record, cfg)
    image, mask = image[None], mask[None]
    if stage == "coarse":
        small = resize_long(image, cfg["coarse_size"]).to(device)
        truth = F.interpolate(mask[:,None].float(), small.shape[-2:], mode="nearest")[:,0].long().to(device)
        return model(small), truth
    pc, _ = coarse_pipeline.overview(image)
    grid = tiles(*image.shape[-2:], cfg["tile_size"])
    grid = [t for t in grid if (mask[...,t[0]:t[0]+t[2],t[1]:t[1]+t[3]] != cfg["ignore_index"]).any()]
    if not grid:
        raise ValueError("Training/validation image has no labeled tile: " + record["id"])
    if deterministic:
        tile = grid[len(grid)//2]
    else:
        tile = random.choice(grid)
    crop = crop_context(image, tile, cfg["tile_size"], cfg["border"])
    context = crop_context(pc, tile, cfg["tile_size"], cfg["border"])
    if stage == "low":
        crop = F.interpolate(crop, (cfg["low_size"],)*2, mode="bilinear", align_corners=False, antialias=True)
        context = F.interpolate(context, crop.shape[-2:], mode="bilinear", align_corners=False)
    output = model(crop.to(device), context.to(device))
    # Both actions learn on the SAME original-resolution central pixels.
    # Upsample low logits before CE; do not discard tiny labels by shrinking masks.
    output["logits"] = center_prediction(output["logits"], tile, cfg)
    y,x,h,w = tile
    return output, mask[..., y:y+h, x:x+w].to(device)


def train_segmenter(cfg, records, run, stage, device, steps=None, resume=False):
    device, run = torch.device(device), Path(run)
    run.mkdir(parents=True, exist_ok=True)
    seed_all(cfg["seed"])
    train = [r for r in records if r["split"] == "train"]
    val = [r for r in records if r["split"] == "val"]
    if not train or not val:
        raise ValueError("Segmenter training requires train and val records")
    coarse_pipeline = None
    model = Segmenter(cfg, context=stage != "coarse", initialize_pretrained=stage == "coarse" and not resume).to(device)
    if stage != "coarse":
        coarse, _ = load_segmenter(run / "coarse.pt", cfg, "coarse", device)
        coarse_pipeline = Pipeline(cfg, coarse, None, None, device=device)
        # Initialize separate refiner weights from the trained coarse representation.
        model.encoder.load_state_dict(coarse.encoder.state_dict())
        model.decoder.load_state_dict(coarse.decoder.state_dict())
        model.color.load_state_dict(coarse.color.state_dict())
    settings = cfg["training"]
    total = steps or settings["coarse_steps" if stage == "coarse" else "refiner_steps"]
    optimizer = torch.optim.AdamW(model.parameters(), lr=settings["lr"], weight_decay=.01)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    best, start = float("inf"), 0
    latest = run / f"{stage}_latest.pt"
    if resume:
        ckpt = torch.load(latest, map_location=device, weights_only=True)
        if ckpt["signature"] != config_signature(cfg) or ckpt["stage"] != stage:
            raise ValueError("Incompatible resume checkpoint")
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scaler.load_state_dict(ckpt["scaler"])
        start, best = ckpt["step"], ckpt["best_val"]
    for step in range(start, total):
        model.train()
        warmup = min(settings["warmup_steps"], max(1,total//10))
        factor = min(1.,(step+1)/warmup) * max(0.,1-step/max(total,1))**.9
        for group in optimizer.param_groups:
            group["lr"] = settings["lr"] * factor
        optimizer.zero_grad(set_to_none=True)
        with amp(device):
            output, truth = segment_batch(random.choice(train), cfg, stage, model, coarse_pipeline, device)
            loss = segmentation_loss(output, truth, cfg)
        if not torch.isfinite(loss):
            raise RuntimeError("Nonfinite training loss")
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
        scaler.step(optimizer)
        scaler.update()
        if (step+1) % settings["log_every"] == 0 or step == start:
            print(json.dumps({"stage":stage,"step":step+1,"loss":float(loss.detach())}), flush=True)
        if (step+1) % settings["validate_every"] == 0 or step+1 == total:
            model.eval()
            values = []
            with torch.no_grad(), amp(device):
                for r in val[:settings["validation_images"]]:
                    out, target = segment_batch(r,cfg,stage,model,coarse_pipeline,device,True)
                    values.append(float(segmentation_loss(out,target,cfg)))
            score = sum(values)/len(values)
            improved = score < best
            best = min(best, score)
            checkpoint = {"model":model.state_dict(), "optimizer":optimizer.state_dict(),
                          "scaler":scaler.state_dict(), "stage":stage,"step":step+1,
                          "best_val":best, "signature":config_signature(cfg),"config":cfg,
                          "train_paths":[r["image"] for r in train]}
            torch.save(checkpoint, latest)
            if improved:
                torch.save(checkpoint, run/f"{stage}.pt")
            print(json.dumps({"stage":stage,"validation_loss":score,"best":best}), flush=True)


@torch.no_grad()
def cache_targets(cfg, records, run, split, destination, device):
    if split not in {"selector","val"}:
        raise ValueError("Only selector/val caches are allowed; do not train or tune on test")
    pipeline = Pipeline.from_run(cfg, run, device)
    chosen = [r for r in records if r["split"] == split]
    if not chosen:
        raise ValueError("No records for " + split)
    train_paths = set()
    for stage in ("coarse","low","high"):
        ckpt = torch.load(Path(run)/f"{stage}.pt", map_location="cpu", weights_only=True)
        train_paths.update(ckpt["train_paths"])
    if any(r["image"] in train_paths for r in chosen):
        raise ValueError("Target image overlaps segmenter training images")
    images = []
    for record in chosen:
        image, truth = read_sample(record,cfg)
        pc, features = pipeline.overview(image)
        coarse = pc.argmax(1)[0]
        rows = []
        for tile in tiles(*image.shape[-2:],cfg["tile_size"]):
            y,x,h,w = tile
            preds, costs = [], []
            for action in (1,2):
                sync(pipeline.device)
                start = time.perf_counter()
                preds.append(pipeline.refine(image,pc,tile,action))
                sync(pipeline.device)
                costs.append((time.perf_counter()-start)*1000)
            target = outcome_targets(coarse[y:y+h,x:x+w],*preds,truth[y:y+h,x:x+w],cfg)
            if target is None:
                # Preserve geometry/cost at calibration, but never fit on missing labels.
                target = {"transitions":torch.zeros(2,4),
                          "confusion":torch.zeros(3,cfg["classes"],cfg["classes"]),
                          "counts":torch.zeros(3,cfg["classes"],cfg["classes"],dtype=torch.long),
                          "valid_pixels":torch.tensor(0)}
            target["z"] = tile_descriptor(image[None],pc,features,tile,cfg)[0]
            target["cost_ms"] = torch.tensor(costs)
            target["tile"] = torch.tensor(tile)
            rows.append(target)
        if rows:
            images.append({"id":record["id"], **{k:torch.stack([r[k] for r in rows]) for k in rows[0]}})
        print(json.dumps({"cached_image":record["id"],"tiles":len(rows)}),flush=True)
    if not images:
        raise ValueError("No valid target pixels")
    cache = {"split":split,"signature":config_signature(cfg),"provenance":pipeline.provenance,
             "images":images, "image_paths":[r["image"] for r in chosen]}
    Path(destination).parent.mkdir(parents=True,exist_ok=True)
    torch.save(cache,destination)


def load_cache(path, cfg, split):
    cache = torch.load(path,map_location="cpu",weights_only=True)
    if cache["split"] != split or cache["signature"] != config_signature(cfg):
        raise ValueError("Cache split/config mismatch")
    return cache


def flatten_cache(cache):
    valid = torch.cat([im["valid_pixels"] for im in cache["images"]]) > 0
    if not valid.any():
        raise ValueError("Cache has no labeled tiles")
    return {k:torch.cat([im[k] for im in cache["images"]])[valid]
            for k in ("z","transitions","confusion")}


def train_selector(cfg, train_path, val_path, run, device, epochs=None):
    train_cache, val_cache = load_cache(train_path,cfg,"selector"), load_cache(val_path,cfg,"val")
    if train_cache["provenance"] != val_cache["provenance"]:
        raise ValueError("Training and validation targets came from different models")
    if set(train_cache["image_paths"]) & set(val_cache["image_paths"]):
        raise ValueError("Selector train/validation overlap")
    seed_all(cfg["seed"])
    train, val = flatten_cache(train_cache), flatten_cache(val_cache)
    device = torch.device(device)
    model = OutcomeSelector(train["z"].shape[-1],cfg["classes"]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(),lr=cfg["training"]["selector_lr"],weight_decay=.01)
    best, stale = float("inf"), 0
    for epoch in range(epochs or cfg["training"]["selector_epochs"]):
        model.train()
        order = torch.randperm(len(train["z"]))
        losses = []
        for idx in order.split(cfg["training"]["selector_batch"]):
            target = {k:v[idx].to(device) for k,v in train.items()}
            optimizer.zero_grad(set_to_none=True)
            loss = selector_loss(model(target["z"]),target,cfg)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
            optimizer.step()
            losses.append(float(loss.detach()))
        model.eval()
        with torch.no_grad():
            target = {k:v.to(device) for k,v in val.items()}
            score = float(selector_loss(model(target["z"]),target,cfg))
        if score < best:
            best, stale = score, 0
            torch.save({"model":model.state_dict(),"signature":config_signature(cfg),
                        "provenance":train_cache["provenance"],"utility":cfg["utility"],
                        "best_val":best,"epoch":epoch+1},Path(run)/"selector.pt")
        else:
            stale += 1
        print(json.dumps({"selector_epoch":epoch+1,"train_loss":sum(losses)/len(losses),
                          "val_loss":score}),flush=True)
        if stale >= cfg["training"]["patience"]:
            break


def calibrate(cfg, val_path, run, destination):
    cache = load_cache(val_path,cfg,"val")
    ckpt = torch.load(Path(run)/"selector.pt",map_location="cpu",weights_only=True)
    if (ckpt["provenance"] != cache["provenance"] or ckpt["utility"] != cfg["utility"]
            or ckpt["signature"] != config_signature(cfg)):
        raise ValueError("Selector/cache/utility mismatch")
    flat = flatten_cache(cache)
    model = freeze(OutcomeSelector(flat["z"].shape[-1],cfg["classes"]))
    model.load_state_dict(ckpt["model"])
    with torch.no_grad():
        pred = model(flat["z"])
        logits = pred["benefit_logits"][:,2]
        true_u = utility(flat["transitions"],flat["confusion"],cfg)
        labels = (true_u.max(-1).values > 0).float()
    # Scalar temperature only, no training/test-label leakage. Degenerate val labels
    # cannot support meaningful probability calibration; retain T=1 and report it.
    temperature = 1.
    calibrated = bool(labels.min() != labels.max())
    if calibrated:
        log_temp = torch.zeros((),requires_grad=True)
        optim = torch.optim.LBFGS([log_temp],lr=.1,max_iter=50)
        def closure():
            optim.zero_grad()
            loss = F.binary_cross_entropy_with_logits(logits/log_temp.exp().clamp(.05,20),labels)
            loss.backward()
            return loss
        optim.step(closure)
        temperature = float(log_temp.detach().exp().clamp(.05,20))
    costs = torch.cat([im["cost_ms"] for im in cache["images"]]).median(0).values.tolist()
    candidate = copy.deepcopy(cfg)
    candidate["routing"]["temperature"] = temperature
    candidate["routing"]["action_ms"] = costs
    with torch.no_grad():
        predictions = [model(im["z"]) for im in cache["images"]]
    best = None
    for low in (.15,.3,.45,.6,.75):
        for high in (.3,.45,.6,.75,.9):
            if low >= high:
                continue
            candidate["routing"].update(T1=high,T2=low)
            total = torch.zeros(cfg["classes"],cfg["classes"],dtype=torch.long)
            selected = 0
            for im,prediction in zip(cache["images"],predictions):
                action,_,_ = choose_actions(prediction,candidate)
                total += im["counts"][torch.arange(len(action)),action].sum(0)
                selected += int((action>0).sum())
            score = miou(total)
            if best is None or (score,-selected) > (best[0],-best[1]):
                best = (score,selected,copy.deepcopy(candidate["routing"]))
    result = {"routing":best[2],"validation_miou":best[0],"temperature_fitted":calibrated,
              "signature":config_signature(cfg),"provenance":cache["provenance"],
              "utility":cfg["utility"],"selector_sha256":fingerprint(Path(run)/"selector.pt"),
              "note":"Validation tuning only; costs are estimates, not a hard real-time guarantee."}
    Path(destination).write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2),flush=True)
