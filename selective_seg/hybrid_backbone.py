"""Stride-aligned binary specialist variants and incremental backbone exits."""
import torch
from torch import nn
from torch.nn import functional as F

from .specialist import MultiScaleSpecialist


def align(x, size):
    return x if x.shape[-2:] == size else F.interpolate(
        x, size, mode="bilinear", align_corners=False)


def conv(inputs, outputs, kernel=3, dilation=1):
    return nn.Sequential(nn.Conv2d(inputs, outputs, kernel, padding=dilation * (kernel // 2),
                                   dilation=dilation, bias=False),
                         nn.GroupNorm(1, outputs), nn.GELU())


class InceptionFusion(nn.Module):
    def __init__(self, width):
        super().__init__()
        inner = max(2, width // 4)
        self.paths = nn.ModuleList([
            conv(width, inner, 1), conv(width, inner),
            nn.Sequential(conv(width, inner), conv(inner, inner)),
            conv(width, inner, dilation=2),
        ])
        self.compress = conv(4 * inner, width, 1)

    def forward(self, x):
        return self.compress(torch.cat([path(x) for path in self.paths], 1))


class HybridSpecialist(MultiScaleSpecialist):
    """Use probed torchvision stages and bounded-width fusion at each stride."""

    def __init__(self, cfg, initialize_pretrained=False):
        super().__init__(cfg, initialize_pretrained)
        width = self.settings["projection_dim"]
        self.variant = cfg["hybrid"]["fusion"]
        self.has_exits = cfg["hybrid"]["train_exits"]
        self.strides = sorted({entry["stride"] for entry in self.feature_spec})
        if self.strides != [4, 8, 16, 32] or self.settings["levels"] != "all":
            raise ValueError("The hybrid workflow requires all four feature strides")
        self.groups = {stride: [i for i, entry in enumerate(self.feature_spec)
                                if entry["stride"] == stride] for stride in self.strides}
        self.stage_index = {item["name"]: i for i, item in enumerate(self.feature_spec)}
        self.levels = nn.ModuleList(
            conv(width * len(self.groups[stride]), width, 1) for stride in self.strides)
        del self.fusion
        if self.variant == "concat":
            self.concat = conv(4 * width, width)
        else:
            self.smooth = nn.ModuleList(
                InceptionFusion(width) if self.variant == "inception" else conv(width, width)
                for _ in self.strides)
        self.classify = nn.Conv2d(width, 2, 1)
        if self.has_exits:
            self.exit_heads = nn.ModuleList([
                nn.Conv2d(width, 2, 1),
                nn.Sequential(conv(2 * width, width), nn.Conv2d(width, 2, 1)),
            ])
            self.gain_heads = nn.ModuleList([
                nn.Linear(width, 1), nn.Linear(2 * width, 1),
            ])

    def stages(self, image):
        """Yield one aligned level at a time, running each backbone block once."""
        image = (image - self.mean) / self.std
        state = {name: image for name in self.branches}
        offset = dict.fromkeys(self.branches, 0)
        for level, stride in enumerate(self.strides):
            tensors = []
            for name, branch in self.branches.items():
                stop = self.nodes[name][level]
                for i in range(offset[name], stop + 1):
                    state[name] = branch[i](state[name])
                offset[name] = stop + 1
                proj = self.projections[self.stage_index[f"{name}.{stop}"]]
                tensors.append(proj(state[name]))
            size = tensors[0].shape[-2:]
            yield self.levels[level](torch.cat([align(tensor, size) for tensor in tensors], 1))

    def deep(self, levels):
        if self.variant == "concat":
            fused = self.concat(torch.cat([align(x, levels[0].shape[-2:]) for x in levels], 1))
        else:
            fused = self.smooth[-1](levels[-1])
            for index in range(2, -1, -1):
                fused = self.smooth[index](levels[index] + align(fused, levels[index].shape[-2:]))
        return self.classify(fused)

    def packets(self, image, use_exits=True):
        """Yield (logits, learned deep accuracy gain, cross-level feature disagreement)."""
        levels = []
        for index, level in enumerate(self.stages(image)):
            levels.append(level)
            if self.has_exits and use_exits and index < 2:
                first = levels[0]
                combined = first if index == 0 else torch.cat([first, align(level, first.shape[-2:])], 1)
                pooled = combined.mean((-2, -1))
                yield self.exit_heads[index](combined), self.gain_heads[index](pooled).squeeze(1), None
        disagreement = 1 - F.cosine_similarity(
            levels[0], align(levels[-1], levels[0].shape[-2:]), dim=1).unsqueeze(1)
        yield self.deep(levels), None, disagreement

    def forward(self, image):
        return next(self.packets(image, use_exits=False))[0]
