"""Versioned optional research settings, separate from frozen segmenter identity."""
import hashlib
import json

CACHE_VERSION = 2


def settings(cfg):
    supplied = cfg.get('research', {})
    return {**{'descriptor_grid': 0, 'scout': False, 'acceptance': False,
               'escalation': False, 'shared_marginals': False}, **supplied}


def signature(cfg):
    payload = {'schema': CACHE_VERSION, 'research': settings(cfg)}
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def descriptor_dim(feature_dim, cfg):
    r = settings(cfg)
    g = r['descriptor_grid']
    return feature_dim + cfg['classes'] + 9 + (cfg['classes'] + 2) * g*g + (3 if r['scout'] else 0)


def auxiliary_names(cfg):
    return [name for name in ('scout', 'acceptance', 'escalation') if settings(cfg)[name]]
