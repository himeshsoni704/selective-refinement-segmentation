"""Auxiliary training and calibration. Ground truth never enters inference inputs."""
import copy
import json
from pathlib import Path
import torch
from torch.nn import functional as F
from .data import read_sample
from .models import DetailScout, AcceptanceGate, EscalationSelector, OutcomeSelector, freeze
from .objectives import acceptance_loss, selector_loss, utility, miou
from .geometry import tiles
from .engine import Pipeline, config_signature, fingerprint
from .research import settings, signature
from .routing import choose_actions


def train_auxiliary(cfg, records, run, name, device, epochs=None):
    """Fit scout/gate ONLY on a disjoint `auxiliary` split; validate on val."""
    from .training import seed_all
    seed_all(cfg['seed'])
    if name not in {'scout','acceptance'}:
        raise ValueError('Unknown auxiliary')
    train = [r for r in records if r['split'] == 'auxiliary']
    val = [r for r in records if r['split'] == 'val']
    if not train or not val:
        raise ValueError('Need disjoint auxiliary and val manifest splits')
    raw_cfg = copy.deepcopy(cfg)
    raw_cfg['research'] = {}
    pipeline = Pipeline.from_run(raw_cfg, run, device)
    for stage in ('coarse','low','high'):
        ck = torch.load(Path(run)/(stage+'.pt'), map_location='cpu', weights_only=True)
        if set(ck['train_paths']) & {r['image'] for r in train+val}:
            raise ValueError('Auxiliary/validation images overlap segmenter training')
    model = (DetailScout(cfg['bands']) if name == 'scout' else AcceptanceGate(cfg['classes'])).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg['training']['selector_lr'])
    best, stale = float('inf'), 0
    def loss_for(record):
        image, truth = read_sample(record,cfg)
        pc,_ = pipeline.overview(image)
        for tile in tiles(*image.shape[-2:],cfg['tile_size']):
            y,x,h,w = tile
            target = truth[y:y+h,x:x+w][None].to(device)
            valid = target != cfg['ignore_index']
            if not valid.any():
                continue
            coarse = pc[...,y:y+h,x:x+w].to(device)
            refs = [pipeline.refine(image,pc,tile,a,True)['probabilities'].to(device) for a in (1,2)]
            if name == 'scout':
                wrong = coarse.argmax(1) != target
                recoverable = wrong & ((refs[0].argmax(1)==target)|(refs[1].argmax(1)==target))
                labels = torch.where(wrong,2,0).long()
                labels[recoverable] = 1
                labels[~valid] = cfg['ignore_index']
                logits = model(image[None,:,y:y+h,x:x+w].to(device))
                logits = F.interpolate(logits,(h,w),mode='bilinear',align_corners=False)
                yield F.cross_entropy(logits,labels,ignore_index=cfg['ignore_index'])
            else:
                yield sum(acceptance_loss(model(coarse,r),coarse,r,target,cfg['ignore_index'],
                                          cfg['training'].get('gate_class_weights')) for r in refs)/2
    for epoch in range(epochs or cfg['training'].get('auxiliary_epochs',10)):
        model.train()
        losses = []
        for record in train:
            for loss in loss_for(record):
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
                opt.step()
                losses.append(float(loss.detach()))
        model.eval()
        with torch.no_grad():
            scores = [float(loss) for record in val for loss in loss_for(record)]
        if not losses or not scores:
            raise ValueError('Auxiliary train/val contains no valid targets')
        score = sum(scores)/len(scores)
        if score < best:
            best, stale = score, 0
            torch.save({'model':model.state_dict(),'signature':config_signature(cfg),
                        'provenance':pipeline.provenance,'train_paths':[r['image'] for r in train],
                        'config':cfg,'validation_loss':score},Path(run)/(name+'.pt'))
        else:
            stale += 1
        print(json.dumps({'module':name,'epoch':epoch+1,'train_loss':sum(losses)/len(losses),
                          'validation_loss':score}),flush=True)
        if stale >= cfg['training']['patience']:
            break


def train_escalation(cfg, train_path, val_path, run, device, epochs=None):
    from .training import load_cache, seed_all
    seed_all(cfg['seed'])
    train, val = load_cache(train_path,cfg,'selector'), load_cache(val_path,cfg,'val')
    if train['provenance'] != val['provenance'] or set(train['image_paths']) & set(val['image_paths']):
        raise ValueError('Escalation cache mismatch or overlap')
    model = EscalationSelector(cfg['classes']).to(device)
    opt = torch.optim.AdamW(model.parameters(),lr=cfg['training']['selector_lr'])
    best, stale = float('inf'),0
    # Stage-two prediction uses current-tile utility, matching inference exactly.
    def losses(cache):
        for im in cache['images']:
            for i in range(len(im['z'])):
                if im['valid_pixels'][i] <= 0:
                    continue
                target = {'transitions':im['escalation_transitions'][i:i+1].to(device),
                          'confusion':im['escalation_confusion'][i:i+1].to(device),
                          'valid_pixels':im['valid_pixels'][i:i+1].to(device)}
                yield selector_loss(model(im['escalation_z'][i:i+1].to(device)),target,cfg)
    for epoch in range(epochs or cfg['training']['selector_epochs']):
        model.train()
        for loss in losses(train):
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
            opt.step()
        model.eval()
        with torch.no_grad():
            values = [float(l) for l in losses(val)]
        if not values:
            raise ValueError('No labeled escalation validation tiles')
        score = sum(values)/len(values)
        if score < best:
            best, stale = score, 0
            torch.save({'model':model.state_dict(),'provenance':train['provenance'],
                        'research_signature':signature(cfg),'utility':cfg['utility'],
                        'validation_loss':score},Path(run)/'escalation.pt')
        else:
            stale += 1
        print(json.dumps({'escalation_epoch':epoch+1,'validation_loss':score}),flush=True)
        if stale >= cfg['training']['patience']:
            break


def calibrate_direct(cfg, val_path, run, destination):
    """Tune lambda on validation at the configured budget, replaying exact deployed policy."""
    from .training import load_cache
    cache = load_cache(val_path,cfg,'val')
    ck = torch.load(Path(run)/'selector.pt',map_location='cpu',weights_only=True)
    if ck['provenance'] != cache['provenance'] or ck['research_signature'] != signature(cfg) or ck['utility'] != cfg['utility']:
        raise ValueError('Stale selector/cache')
    model = freeze(OutcomeSelector(cache['images'][0]['z'].shape[-1],cfg['classes'],settings(cfg)['shared_marginals']))
    model.load_state_dict(ck['model'])
    escalation = None
    extra = {}
    if settings(cfg)['escalation']:
        path = Path(run)/'escalation.pt'
        ec = torch.load(path,map_location='cpu',weights_only=True)
        if ec['provenance'] != cache['provenance'] or ec['research_signature'] != signature(cfg) or ec['utility'] != cfg['utility']:
            raise ValueError('Stale escalation/cache')
        escalation = freeze(EscalationSelector(cfg['classes']))
        escalation.load_state_dict(ec['model'])
        extra['escalation_sha256'] = fingerprint(path)
    candidate = copy.deepcopy(cfg)
    costs = torch.cat([im['cost_ms'] for im in cache['images']]).median(0).values.clamp_min(1e-6).tolist()
    candidate['routing']['action_ms'] = costs
    with torch.no_grad():
        preds = [model(im['z']) for im in cache['images']]
        epreds = [[escalation(z[None]) for z in im['escalation_z']] for im in cache['images']] if escalation else None
        best = None
        for lam in cfg['routing'].get('lambda_grid',[0.,1e-6,1e-5,1e-4,1e-3]):
            candidate['routing']['cost_lambda'] = lam
            total = torch.zeros(cfg['classes'],cfg['classes'],dtype=torch.long)
            spent_total = 0.
            for k,(im,pred) in enumerate(zip(cache['images'],preds)):
                # Use geometry, not labeled support, to match mask-free inference.
                support = im['tile'][:,2]*im['tile'][:,3]
                actions,_,_ = choose_actions(pred,candidate,support=support)
                spent = sum(costs[int(a)-1] for a in actions if a)
                if escalation:
                    for i in range(len(actions)):
                        if actions[i] != 1:
                            continue
                        gain = float(utility(epreds[k][i]['transitions'],epreds[k][i]['confusion'],cfg,support[i:i+1])[:,1])
                        budget = candidate['routing'].get('max_refine_ms')
                        if (budget is None or spent+costs[1] <= budget+1e-8) and gain > candidate['routing'].get('escalation_min_gain',0.)+lam*costs[1]:
                            actions[i] = 2
                            spent += costs[1]
                total += im['counts'][torch.arange(len(actions)),actions].sum(0)
                spent_total += spent
            score = miou(total)
            key = (score,-spent_total)
            if best is None or key > best[0]:
                best = (key,copy.deepcopy(candidate['routing']))
    result = {'routing':best[1],'validation_miou':best[0][0], 'temperature_fitted':False,
              'signature':config_signature(cfg),'research_signature':signature(cfg),
              'provenance':cache['provenance'],'utility':cfg['utility'],
              'selector_sha256':fingerprint(Path(run)/'selector.pt'),**extra,
              'note':'Validation lambda search; refinement costs are estimates, not total-latency guarantees.'}
    Path(destination).write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2),flush=True)
