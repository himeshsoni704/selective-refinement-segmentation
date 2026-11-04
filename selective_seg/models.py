import torch
from torch import nn
from torch.nn import functional as F


class SpectralColor(nn.Module):
    """Learned pseudo-color, conditioned on spectral statistics and predicted classes.

    Coarse pass: class proportions are predicted from band statistics.
    Refinement: class proportions come from frozen coarse probabilities.
    Ground-truth classes are NEVER inputs. RGB is identity unless enabled explicitly.
    """
    def __init__(self, bands, classes, enabled=False):
        super().__init__()
        self.enabled = enabled
        if not enabled:
            if bands != 3:
                raise ValueError("Non-RGB inputs require spectral.enabled=true")
            return
        self.class_predictor = nn.Sequential(nn.Linear(2 * bands, 64), nn.ReLU(),
                                             nn.Linear(64, classes))
        self.patch_weights = nn.Sequential(nn.Linear(2 * bands, 64), nn.ReLU(),
                                           nn.Linear(64, 3 * bands))
        self.class_weights = nn.Parameter(torch.randn(classes, 3, bands) * 0.02)
        self.bands = bands

    def forward(self, x, class_prob=None):
        if not self.enabled:
            return x, x.new_zeros(()), None
        stats = torch.cat((x.mean((-2, -1)), x.std((-2, -1), unbiased=False)), 1)
        inferred_classes = None
        if class_prob is None:
            class_prob = self.class_predictor(stats).softmax(-1)
            inferred_classes = class_prob
        else:
            class_prob = class_prob.mean((-2, -1)) if class_prob.ndim == 4 else class_prob
            class_prob = class_prob.detach()
        logits = self.patch_weights(stats).reshape(-1, 3, self.bands)
        logits = logits + torch.einsum("bc,crs->brs", class_prob, self.class_weights)
        weights = logits.softmax(-1)
        rgb = torch.einsum("brs,bshw->brhw", weights, x)
        # Discourage the three output channels from selecting identical mixtures.
        norm = F.normalize(weights, dim=-1)
        overlap = norm @ norm.transpose(-1, -2)
        off_diagonal = ~torch.eye(3, dtype=torch.bool, device=x.device)
        diversity = overlap[:, off_diagonal].square().mean()
        return rgb, diversity, inferred_classes


class Segmenter(nn.Module):
    """MiT-B0 encoder + official SegFormer fusion decoder + optional Pc conditioning."""
    def __init__(self, cfg, context=False, initialize_pretrained=False):
        super().__init__()
        self.context = context
        self.classes = cfg["classes"]
        self.color = SpectralColor(cfg["bands"], self.classes, cfg["spectral"]["enabled"])
        self.tiny = cfg.get("backbone") == "tiny"
        if self.tiny:
            # Only for offline smoke tests; never label its results as SegFormer-B0.
            self.encoder = nn.Sequential(nn.Conv2d(3, 32, 3, 2, 1), nn.GroupNorm(4, 32),
                                         nn.GELU(), nn.Conv2d(32, 64, 3, 2, 1),
                                         nn.GroupNorm(8, 64), nn.GELU())
            self.decoder = nn.Conv2d(64, 64, 1)
            hidden = 64
            self.feature_dim = 64
        else:
            from transformers import SegformerConfig, SegformerModel
            from transformers.models.segformer.modeling_segformer import SegformerDecodeHead
            if initialize_pretrained and cfg.get("pretrained"):
                self.encoder = SegformerModel.from_pretrained(cfg["pretrained"])
                config = self.encoder.config
                if list(config.hidden_sizes) != [32, 64, 160, 256]:
                    raise ValueError("This configuration requires a MiT-B0 encoder")
            else:
                config = SegformerConfig()
                self.encoder = SegformerModel(config)
            config.num_labels = self.classes
            config.decoder_hidden_size = 256
            self.decoder = SegformerDecodeHead(config)
            self.decoder.classifier = nn.Identity()
            hidden = config.decoder_hidden_size
            self.feature_dim = config.hidden_sizes[-1]
        self.classifier = nn.Conv2d(hidden + (self.classes if context else 0), self.classes, 1)
        self.register_buffer("mean", torch.tensor([.485, .456, .406])[None, :, None, None])
        self.register_buffer("std", torch.tensor([.229, .224, .225])[None, :, None, None])

    def forward(self, x, coarse_prob=None):
        rgb, spectral_reg, spectral_classes = self.color(x, coarse_prob)
        normalized = (rgb - self.mean) / self.std
        if self.tiny:
            features = self.encoder(normalized)
            decoded = self.decoder(features)
        else:
            states = self.encoder(normalized, output_hidden_states=True, return_dict=True).hidden_states
            features = states[-1]
            decoded = self.decoder(states)
        if self.context:
            if coarse_prob is None:
                raise ValueError("Refiner requires coarse probabilities")
            pc = F.interpolate(coarse_prob, decoded.shape[-2:], mode="bilinear", align_corners=False)
            decoded = torch.cat((decoded, pc), 1)
        logits = self.classifier(decoded)
        logits = F.interpolate(logits, x.shape[-2:], mode="bilinear", align_corners=False)
        return {"logits": logits, "features": features, "spectral_reg": spectral_reg,
                "spectral_class_prob": spectral_classes}


class OutcomeSelector(nn.Module):
    """Predict normalized confusion tables and correction/damage for both actions."""
    def __init__(self, descriptor_dim, classes):
        super().__init__()
        self.classes = classes
        self.trunk = nn.Sequential(nn.LayerNorm(descriptor_dim), nn.Linear(descriptor_dim, 128),
                                   nn.ReLU(), nn.Dropout(.1))
        # [correct->correct, correct->wrong, wrong->correct, wrong->wrong]
        self.transitions = nn.Linear(128, 2 * 4)
        # States [coarse, low, high]; rows=true, columns=predicted class.
        self.confusion = nn.Linear(128, 3 * classes * classes)
        # Probability positive utility for [low, high, either].
        self.benefit = nn.Linear(128, 3)

    def forward(self, z):
        h = self.trunk(z)
        trans = self.transitions(h).reshape(-1, 2, 4).softmax(-1)
        c = self.classes
        cm = self.confusion(h).reshape(-1, 3, c * c).softmax(-1).reshape(-1, 3, c, c)
        tp = cm.diagonal(dim1=-2, dim2=-1)
        return {"transitions": trans, "corrected": trans[..., 2], "damaged": trans[..., 1],
                "confusion": cm, "tp": tp, "fp": cm.sum(-2) - tp,
                "fn": cm.sum(-1) - tp, "benefit_logits": self.benefit(h)}


def freeze(module):
    module.eval()
    module.requires_grad_(False)
    return module
