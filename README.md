# Selective segmentation with low/high refinement

A trainable research implementation of your five requested changes. It contains
code and tests, **not trained benchmark weights or demonstrated accuracy gains**.
Target: RGB DeepGlobe/Inria on one NVIDIA GPU with roughly 24 GB VRAM. Optional
spectral inputs use a fixed, configured band count and order. Start with RGB.

## Architecture

```text
RGB image / normalized spectral cube (kept on CPU)
   |
   +--> Resize whole image to 640 on long side
   |       --> [spectral only: learned class-conditioned pseudo-color]
   |       --> coarse SegFormer-B0 --> probabilities + native 256-d features
   |
   +--> Tile-level detail scan on original-resolution bands
           --> residual mean/max and horizontal/vertical edge strength
                    |
                    v
   Per-tile descriptor: 256 pooled features + C class proportions
                        + 5 uncertainty/boundary statistics + 4 detail statistics
                    |
           LayerNorm --> Linear(128) --> ReLU --> Dropout(0.1)
                    |
       +------------+----------------------+------------------+
       |            |                      |                  |
   corrections   damage              class confusion      benefit logits
   low / high    low / high          coarse / low / high  low / high / either
       +------------+----------------------+------------------+
                    |
           calibrated routing score q
                    |
       q >= T1: high     T2 <= q < T1: low     q < T2: skip
             |                  |                  |
       640x640 crop       same field of view       keep coarse
       full resolution    resized to 320x320       prediction
             |                  |
       optional class/patch-aware spectral-to-3-channel projection
             |                  |
       high SegFormer-B0   low SegFormer-B0
       + coarse context    + coarse context
             |                  |
       keep centre       upsample logits to 640, keep centre
             +------------------+
                      |
           replace selected 512x512 centres
```

There are now **three independent segmentation networks** (coarse, low, high),
plus one MLP selector. Low/high refiners start from trained coarse representation
weights, then train separately. They use the official Hugging Face SegFormer
MiT-B0 encoder and fusion decoder, with a custom classifier that concatenates
coarse probabilities before classification. The coarse model uses the same decoder
without the additional context channels. Pretraining initializes the encoder;
the decoder and task classifiers are trained for your dataset.

Low refinement sees a downsampled **local crop**, not another copy of the entire
small image. This preserves a meaningful extra local view. Upsampling its output
does not restore detail that its input lost. Train both refiners against original
centre labels; only low input/context are downsampled.

## Exact meanings of outputs and routing

The selector cannot measure actual TP/FP/FN at inference: those require labels.
It predicts their expected values. Actual labels are used only to construct
training/validation targets and evaluate test predictions.

For each action, a 4-state softmax predicts fractions of valid tile pixels:
`correct->correct`, `correct->wrong` (damage), `wrong->correct` (correction),
`wrong->wrong`. This makes correction and damage nonnegative and their sum <= 1.

For each state (coarse, low, high), a C-by-C joint distribution predicts the
normalized confusion matrix. Rows are true classes, columns predicted classes.
TP is its diagonal, FN is row sum minus TP, FP is column sum minus TP. This
ensures internally valid TP/FP/FN statistics. Multiplying by tile pixel count
produces expected counts, not integer observations. On unlabeled inputs, the
number of unknown/ignored ground-truth pixels is unavailable; reported counts
scale by the real image area within the tile.

For action a, utility is:

```
U_a = corrected_a - damage_penalty * damaged_a
      + confusion_weight * sum_c normalized_class_weight_c * (
            tp_reward * (TP_a,c - TP_coarse,c)
          + fp_penalty * (FP_coarse,c - FP_a,c)
          + fn_penalty * (FN_coarse,c - FN_a,c))
```

All terms are fractions, so large tiles do not automatically dominate. Default
damage penalty is 2, an experimental starting point. Inria gives building more
weight than background. TP/FP/FN terms intentionally add class emphasis; they
overlap with correction/damage and are not independent evidence. Set
`confusion_weight=0` to ablate them. This utility is **not an mIoU estimator**.

The shared routing score is explicitly:

`q = estimated P(max(U_low, U_high) > 0 | coarse information + detail statistics)`.

Its training label is 1 if either measured action has positive utility. It is
not segmentation confidence or a probability that high refinement is optimal.
The requested ordered thresholds are a heuristic to test: a larger q does not
prove high refinement is better than low. For comparison, a future allocator
can directly choose between predicted action utilities/costs.

Default T1=0.75, T2=0.45 are **provisional**. The calibration command fits a scalar
temperature for q and searches thresholds on validation images. If validation
labels have only one benefit class, temperature fitting is skipped and flagged.
No calibration method guarantees reliability on a geographically shifted city.

The default `utility_veto=true` makes TP/FP/FN and damage affect routing directly:
if the threshold-selected action has U<=0, skip it; a high action can downgrade
to low if low is beneficial. It never upgrades a low-band tile. Set this flag
false for the exact threshold-only ablation. `max_fraction` optionally caps the
number of refined tiles. `max_refine_ms` optionally caps an **estimated refinement
budget**, using measured action costs and a greedy utility/cost order. It is not
a hard end-to-end deadline or an optimal knapsack solver. With equal tile costs,
division by cost changes no ranking. Calibration stores action timing estimates.

## Spectral support and false colors

RGB mode is identity preprocessing before ImageNet normalization. It creates no
new bands. For real multispectral/hyperspectral data:

1. Export each image as a CHW `.npy` cube; set `bands` and normalization scale.
2. Use a fixed sensor band order and common class taxonomy within a run.
3. Enable `spectral.enabled` and start a new experiment; RGB checkpoints are not
   interchangeable with a differently configured spectral adapter.

The projection produces three learned band mixtures, with nonnegative weights
summing to one. It is **patch-aware** through per-band mean/std and **class-aware**
through a class-indexed bank of mixtures. In the coarse pass, an auxiliary head
predicts class proportions from spectral statistics, supervised using training
mask proportions. In refiners, the conditioning comes from frozen coarse
probabilities. Ground-truth class labels never enter an inference forward pass.
A weak diversity penalty discourages all three pseudo-color channels collapsing
to the same mixture.

These are task-dependent pseudo-colors, not physically calibrated natural RGB.
Three-channel projection is lossy. Compare with fixed band selection and, for a
strong spectral study, a model consuming all bands. The provided generic adapter
does not implement wavelength alignment across sensors, atmospheric correction,
ENVI/GeoTIFF ingestion, georeferencing, or physical reflectance calibration.
Perform those dataset-specific steps upstream. `spectral_template.json` has a
placeholder count of 16 bands, not a claim about your sensor.

## Installation

Use Python 3.10+ in a fresh environment. Install a CUDA-compatible PyTorch build
appropriate to your GPU/driver using https://pytorch.org/get-started/locally/.
Then, from this directory:

```bash
python -m pip install -e '.[test]'
python -m pytest -q
```

The first real coarse training run downloads `nvidia/mit-b0` encoder weights.
Set `pretrained` to null for an offline randomly initialized experiment, or to
a compatible local Hugging Face encoder directory. A `tiny` backbone exists
only for smoke tests and must never be reported as SegFormer-B0.

## Data contract

Create a JSON manifest. Image/mask paths are relative to the manifest directory:

```json
[
  {"id":"train_001", "image":"images/train_001.png", "mask":"masks/train_001.png", "split":"train"},
  {"id":"selector_001", "image":"images/selector_001.png", "mask":"masks/selector_001.png", "split":"selector"},
  {"id":"val_001", "image":"images/val_001.png", "mask":"masks/val_001.png", "split":"val"},
  {"id":"test_001", "image":"images/test_001.png", "mask":"masks/test_001.png", "split":"test"}
]
```

Split ORIGINAL images/geographic regions before tiling. `train` is only for
segmenters; `selector` images must not train any segmenter. `val` is used for
checkpoint selection and routing calibration. `test` is evaluated only after
decisions are fixed. Path/ID overlap is rejected; renamed duplicate content or
overlapping geographical crops remain your responsibility to exclude.

Inria config expects masks with 0=background and 255=building and converts these
to IDs 0/1. If your masks already contain IDs 0/1, set `mask_values` to null.
Do not accidentally treat building value 255 as ignore before remapping.

DeepGlobe config explicitly maps RGB mask colors to six class IDs, with black
unknown=ignore 255. It errors on unmapped colors (e.g. lossy JPEG masks). Verify
the supplied mapping against your downloaded release before training.

RGB inputs divide by 255. Spectral values must map to [0,1] using a configured
scalar or per-band scale established from sensor specifications/training data.
Do not compute normalization from validation/test labels or mix band orders.

## Training commands

Example for Inria; substitute your manifest path. These are separate stages and
must run in order. Use a new run directory for another seed/dataset/architecture.

```bash
python -m selective_seg.cli validate-data --config configs/inria_rgb.json --manifest data/manifest.json
python -m selective_seg.cli train-coarse --config configs/inria_rgb.json --manifest data/manifest.json --run runs/inria --device cuda
python -m selective_seg.cli train-low --config configs/inria_rgb.json --manifest data/manifest.json --run runs/inria --device cuda
python -m selective_seg.cli train-high --config configs/inria_rgb.json --manifest data/manifest.json --run runs/inria --device cuda
python -m selective_seg.cli cache --config configs/inria_rgb.json --manifest data/manifest.json --run runs/inria --split selector --device cuda
python -m selective_seg.cli cache --config configs/inria_rgb.json --manifest data/manifest.json --run runs/inria --split val --device cuda
python -m selective_seg.cli train-selector --config configs/inria_rgb.json --run runs/inria --device cuda
python -m selective_seg.cli calibrate --config configs/inria_rgb.json --run runs/inria
python -m selective_seg.cli evaluate --config configs/inria_rgb.json --manifest data/manifest.json --run runs/inria --split test --calibration runs/inria/calibration.json --device cuda
```

Predict an unlabeled image:

```bash
python -m selective_seg.cli predict --config configs/inria_rgb.json --run runs/inria --calibration runs/inria/calibration.json --image data/new_image.png --output predictions/new_image --device cuda
```

Output includes a class-index PNG/NPY mask and JSON with per-tile actions,
probabilities, correction/damage fractions, expected TP/FP/FN counts, and latency.
Evaluation also produces aggregate confusion, per-class IoU, mIoU, action counts,
and peak GPU allocation. Index-mask PNGs are not display-colored overlays.

Use `--steps 100` for an initial training diagnostic, or `--epochs 5` for selector
diagnostics. `--resume` restores the latest segmenter model, optimizer, scaler,
and step; it does not provide bit-identical RNG replay. Best-validation weights
are stored as `coarse.pt`, `low.pt`, `high.pt`; optimizer checkpoints have
`_latest.pt`. Selector training uses validation multitask loss for early stopping,
an intentional change from the original proposal's Spearman criterion.

## Practical training notes

- Batch size is 1 with CUDA autocast/GradScaler and gradient clipping. The
  original proposal's batch 8 is not assumed to fit this extended model.
- Refiners are trained sequentially; only coarse plus the active refiner must
  occupy GPU memory during training. Full images/large coarse maps remain on CPU;
  only thumbnails/crops move to GPU. Features are pooled on their native grid.
- Refiners sample tile centres uniformly. Borders supply context but only valid
  centre pixels contribute segmentation loss. No spectral augmentation is assumed.
- Segmenter validation uses whole thumbnails for coarse, and a deterministic
  centre tile per validation image for refiners. It is a checkpoint-loss proxy,
  not full-image validation mIoU. Final evaluation processes whole test images.
- Target caches contain both action outcomes and model SHA-256 fingerprints.
  Regenerate caches, retrain the selector and recalibrate if a segmenter changes.
  Change utility weights before selector training; changing them later invalidates
  benefit labels/calibration. Routing thresholds alone can be tuned afterward.
- Selector loss combines soft-label transition/CM cross-entropies, positive-utility
  BCE, utility Huber loss, and a consistency penalty between error marginals.
- Timing includes resize, coarse pass, CPU descriptor scan, transfers, scoring,
  cropping, refinement, merge and diagnostic construction. It excludes file I/O,
  model loading and writing outputs. Default evaluation warms up 20 images and
  takes the median of 3 passes per image with CUDA synchronization.
- The code is a correctness-first implementation; CPU descriptors/transfer can
  bottleneck performance. Profile before claiming speedups. 24 GB is the target,
  not a measured guarantee, especially with very large spectral cubes in host RAM.

## Verification and experiments

Run a complete synthetic workflow without downloading weights:

```bash
python smoke_test.py --work smoke_work
```

The tests cover known correction/damage counts, class confusion, ignored pixels,
threshold boundaries, utility veto, budget limits, partial border tiles, spectral
gradient flow, selector backward pass, preservation of skipped predictions, and
an actual randomly initialized SegFormer-B0 forward/backward pass.

Before scientific claims, compare: coarse-only; all-low; all-high; original
gain-only selector; thresholds-only; correction/damage heads; +confusion utility;
+detail branch; and, on spectral data, fixed versus learned color projection.
The current CLI supplies the extended model, not implementations of every paper
baseline. Additional research metrics such as boundary accuracy, object recall,
bootstrap intervals and three-seed summaries are not yet automated here.
Threshold tuning uses validation only; report final claims on untouched test data.

## Sources and scope

- Starting point: your Team 11 Term-I proposal (frozen coarse/refiner, held-out
  selector targets, context crops and equal-runtime evaluation).
- SegFormer implementation/API: https://huggingface.co/docs/transformers/model_doc/segformer
- AMP API: https://docs.pytorch.org/docs/stable/notes/amp_examples.html
- Spectral projection is an experimental implementation of your requested idea;
  task-driven spectral reduction has prior work, e.g. https://arxiv.org/abs/2401.11420.

This combination is a research hypothesis. Neither novelty nor performance
improvement is established merely by implementing the components.
