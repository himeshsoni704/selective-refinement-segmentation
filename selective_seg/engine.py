import hashlib
import json
import time
from pathlib import Path
import torch
from torch.nn import functional as F
from .models import Segmenter, OutcomeSelector, DetailScout, AcceptanceGate, EscalationSelector, freeze
from .geometry import resize_long, tiles, crop_context, center_prediction, tile_descriptor, escalation_descriptor
from .routing import choose_actions
from .research import settings, signature, descriptor_dim
from .objectives import utility


def amp(device):
    return torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda")


def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def fingerprint(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024*1024), b""):
            h.update(block)
    return h.hexdigest()


def config_signature(cfg):
    # Routing can be tuned without invalidating frozen-model outcomes.
    keys = ["classes", "bands", "backbone", "spectral", "tile_size", "border", "low_size",
            "coarse_size", "ignore_index", "image_scale", "array_layout", "mask_colors", "mask_values"]
    return hashlib.sha256(json.dumps({k:cfg.get(k) for k in keys}, sort_keys=True).encode()).hexdigest()


def load_segmenter(path, cfg, stage, device):
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    if ckpt["stage"] != stage or ckpt["signature"] != config_signature(cfg):
        raise ValueError(f"Wrong stage or incompatible configuration: {path}")
    model = Segmenter(cfg, context=stage != "coarse").to(device)
    model.load_state_dict(ckpt["model"])
    return freeze(model), ckpt


class Pipeline:
    def __init__(self, cfg, coarse, low, high, selector=None, device="cpu"):
        self.cfg, self.device = cfg, torch.device(device)
        self.coarse, self.low, self.high, self.selector = coarse, low, high, selector
        self.provenance = {}
        self.scout = self.acceptance = self.escalation = None
        self.aux_train_paths = set()

    @classmethod
    def from_run(cls, cfg, run, device, need_selector=False):
        run, device = Path(run), torch.device(device)
        models, provenance = [], {}
        for stage in ("coarse", "low", "high"):
            p = run / f"{stage}.pt"
            model, _ = load_segmenter(p, cfg, stage, device)
            models.append(model)
            provenance[stage] = fingerprint(p)
        auxiliaries, aux_paths = {}, set()
        for name, constructor in [('scout', lambda: DetailScout(cfg['bands'])),
                                  ('acceptance', lambda: AcceptanceGate(cfg['classes']))]:
            if not settings(cfg)[name]:
                continue
            path = run / (name+'.pt')
            ck = torch.load(path, map_location='cpu', weights_only=True)
            base = {k:provenance[k] for k in ('coarse','low','high')}
            if ck['provenance'] != base or ck['signature'] != config_signature(cfg):
                raise ValueError('Stale auxiliary checkpoint: '+name)
            model = constructor().to(device)
            model.load_state_dict(ck['model'])
            auxiliaries[name] = freeze(model)
            aux_paths.update(ck['train_paths'])
            provenance[name] = fingerprint(path)
        selector = None
        if need_selector:
            ckpt = torch.load(run / "selector.pt", map_location="cpu", weights_only=True)
            if ckpt["provenance"] != provenance or ckpt["signature"] != config_signature(cfg):
                raise ValueError("Selector was trained for different segmenters/configuration")
            if ckpt["utility"] != cfg["utility"]:
                raise ValueError("Utility changed: retrain selector and recalibrate routing")
            if ckpt.get('research_signature', signature({})) != signature(cfg):
                raise ValueError('Selector descriptor/research configuration changed')
            selector = OutcomeSelector(descriptor_dim(models[0].feature_dim, cfg), cfg['classes'],
                                       settings(cfg)['shared_marginals']).to(device)
            selector.load_state_dict(ckpt["model"])
            freeze(selector)
        out = cls(cfg, *models, selector=selector, device=device)
        out.provenance = provenance
        out.aux_train_paths = aux_paths
        for name, model in auxiliaries.items():
            setattr(out, name, model)
        if need_selector and settings(cfg)['escalation']:
            ck = torch.load(run/'escalation.pt', map_location='cpu', weights_only=True)
            if ck['provenance'] != provenance or ck['research_signature'] != signature(cfg) or ck['utility'] != cfg['utility']:
                raise ValueError('Stale escalation checkpoint')
            out.escalation = EscalationSelector(cfg['classes']).to(device)
            out.escalation.load_state_dict(ck['model'])
            freeze(out.escalation)
        return out

    @torch.no_grad()
    def overview(self, image):
        # UHR image remains on CPU. Only thumbnails and selected crops enter GPU memory.
        if image.ndim == 3:
            image = image[None]
        small = resize_long(image, self.cfg["coarse_size"]).to(self.device)
        with amp(self.device):
            result = self.coarse(small)
        # Match segmentation convention: interpolate logits, then softmax/argmax.
        logits = F.interpolate(result["logits"].float().cpu(), image.shape[-2:],
                               mode="bilinear", align_corners=False)
        probabilities = logits.softmax(1)
        return probabilities, result["features"].float().cpu()

    @torch.no_grad()
    def refine(self, image, pc, tile, action, return_details=False):
        cfg = self.cfg
        if image.ndim == 3:
            image = image[None]
        crop = crop_context(image, tile, cfg["tile_size"], cfg["border"])
        context = crop_context(pc, tile, cfg["tile_size"], cfg["border"])
        if action == 1:
            crop = F.interpolate(crop, (cfg["low_size"],)*2, mode="bilinear", align_corners=False, antialias=True)
            context = F.interpolate(context, crop.shape[-2:], mode="bilinear", align_corners=False)
            model = self.low
        elif action == 2:
            model = self.high
        else:
            raise ValueError("refine action must be 1 or 2")
        with amp(self.device):
            out = model(crop.to(self.device), context.to(self.device))
        logits = center_prediction(out['logits'].float(), tile, cfg).cpu()
        probability = logits.softmax(1)
        result = {'logits': logits, 'probabilities': probability,
                  'features': out['features'].float().cpu(), 'mask': probability.argmax(1)[0]}
        return result if return_details else result['mask']

    @torch.no_grad()
    def scout_tile(self, image, tile):
        y,x,h,w = tile
        logits = self.scout(image[None,:,y:y+h,x:x+w].to(self.device))
        return logits.float().softmax(1).cpu()

    @torch.no_grad()
    def accepted(self, coarse, refined):
        if self.acceptance is None:
            return refined
        gate = self.acceptance(coarse.to(self.device), refined.to(self.device)).sigmoid().cpu()
        return torch.where((gate >= .5)[:,None], refined, coarse)

    @torch.no_grad()
    def predict(self, image, mode='selective', diagnostics=True, profile=False):
        """Decoded CPU tensor -> CPU mask. Diagnostics are built AFTER headline timing.

        profile=True introduces stage synchronizations and must not supply headline timing.
        Baseline modes bypass selectors/scout/escalation; gate remains an explicit ablation.
        """
        if mode not in {'selective','coarse','low','high','random','entropy'}:
            raise ValueError('Unknown inference mode')
        if mode == 'selective' and self.selector is None:
            raise ValueError('Load a trained selector first')
        sync(self.device)
        started = time.perf_counter()
        stages = {}
        def begin():
            if profile:
                sync(self.device)
            return time.perf_counter()
        def end(name, t):
            if profile:
                sync(self.device)
                stages[name] = stages.get(name, 0.) + (time.perf_counter()-t)*1000
        t = begin()
        pc, features = self.overview(image)
        end('coarse', t)
        grid = tiles(*image.shape[-2:], self.cfg['tile_size'])
        t = begin()
        output = pc.argmax(1)[0].clone()
        end('assembly', t)
        n = len(grid)
        support = torch.tensor([h*w for _,_,h,w in grid], device=self.device)
        pred = q = u = None
        if mode == 'selective':
            scouts = []
            t = begin()
            if self.scout is not None:
                scouts = [self.scout_tile(image, tile) for tile in grid]
            end('scout', t)
            t = begin()
            z = torch.cat([tile_descriptor(image[None], pc, features, tile, self.cfg,
                                           scouts[i] if scouts else None) for i,tile in enumerate(grid)])
            end('descriptors', t)
            t = begin()
            pred = self.selector(z.to(self.device))
            actions, q, u = choose_actions(pred, self.cfg, support=support)
            actions = actions.cpu()
            end('routing', t)
        elif mode in {'coarse','low','high'}:
            actions = torch.full((n,), {'coarse':0,'low':1,'high':2}[mode], dtype=torch.long)
        else:
            from .evaluation import baseline_actions
            t = begin()
            actions = baseline_actions(pc, grid, self.cfg, mode)
            end('routing', t)
        costs = self.cfg['routing']['action_ms']
        spent = sum(costs[int(a)-1] for a in actions if a)
        initial = actions.clone()
        escalated = torch.zeros(n, dtype=torch.bool)
        budget = self.cfg['routing'].get('max_refine_ms')
        for i,tile in enumerate(grid):
            y,x,h,w = tile
            a = int(actions[i])
            if not a:
                continue
            t = begin()
            raw = self.refine(image, pc, tile, a, return_details=True)['probabilities']
            end('low' if a == 1 else 'high', t)
            coarse = pc[...,y:y+h,x:x+w]
            t = begin()
            current = self.accepted(coarse, raw)
            end('acceptance', t)
            if mode == 'selective' and a == 1 and self.escalation is not None:
                t = begin()
                ep = self.escalation(escalation_descriptor(coarse, current).to(self.device))
                # Stage-two metric utility uses the current tile context, explicitly a local surrogate.
                gain = utility(ep['transitions'], ep['confusion'], self.cfg,
                               support[i:i+1])[:,1].item()
                allowed = budget is None or spent + costs[1] <= budget + 1e-8
                go = allowed and gain > self.cfg['routing'].get('escalation_min_gain',0.) + self.cfg['routing'].get('cost_lambda',0.)*costs[1]
                end('escalation', t)
                if go:
                    t = begin()
                    high = self.refine(image, pc, tile, 2, return_details=True)['probabilities']
                    end('high', t)
                    t = begin()
                    # Gate trained against coarse, so do not substitute low as gate input.
                    current = self.accepted(coarse, high)
                    end('acceptance', t)
                    spent += costs[1]
                    actions[i], escalated[i] = 2, True
            t = begin()
            output[y:y+h,x:x+w] = current.argmax(1)[0]
            end('assembly', t)
        sync(self.device)
        elapsed = (time.perf_counter()-started)*1000
        records = []
        if diagnostics:
            for i,tile in enumerate(grid):
                rec = {'tile': list(tile), 'action': ['skip','low','high'][int(actions[i])],
                       'initial_action': ['skip','low','high'][int(initial[i])], 'escalated':bool(escalated[i])}
                if pred is not None:
                    rec.update(p_any_benefit=float(q[i]), utility_low_high=u[i].cpu().tolist(),
                               corrected_fraction_low_high=pred['corrected'][i].cpu().tolist(),
                               damaged_fraction_low_high=pred['damaged'][i].cpu().tolist(),
                               expected_tp_counts_coarse_low_high=(pred['tp'][i]*tile[2]*tile[3]).cpu().tolist(),
                               expected_fp_counts_coarse_low_high=(pred['fp'][i]*tile[2]*tile[3]).cpu().tolist(),
                               expected_fn_counts_coarse_low_high=(pred['fn'][i]*tile[2]*tile[3]).cpu().tolist())
                records.append(rec)
        return output, {'end_to_end_ms':elapsed, 'tiles':records, 'stage_ms':stages,
                        'profiled':profile, 'estimated_refine_ms':spent,
                        'initial_actions':initial.tolist(), 'final_actions':actions.tolist(),
                        'escalations':int(escalated.sum()),
                        'refinement_calls':{'low':int((initial==1).sum()),
                                            'high':int((initial==2).sum()+escalated.sum())}}
