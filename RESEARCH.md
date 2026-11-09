# Selective refinement research workflow

This implementation adds testable mechanisms, not a claim of novelty or a measured
accuracy improvement. The historical reported 65.98% mIoU and 1.32 s/image are not
results of this revision. Rebenchmark the original checkpoints under the new timing
boundary before comparing. No real-data training or GPU benchmarking was performed
in the implementation environment.

## Compatibility and experiment tracks

`configs/deepglobe_rgb.json` retains the original threshold router and architecture.
Existing coarse/low/high checkpoints remain loadable: segmenter signatures deliberately
exclude research settings. Existing baseline selector/calibration checkpoints can be
used with the original configuration. Newly generated caches use schema 2; old caches
must be regenerated. Do not overwrite a baseline run: copy its three frozen segmenter
checkpoints into a new run directory.

Presets:

| Configuration | Purpose | Requires |
|---|---|---|
| `deepglobe_rgb.json` | Original router | Existing baseline run |
| `deepglobe_routing.json` | Direct aggregate-mIoU allocation | New caches, selector, calibration |
| `deepglobe_research.json` | All core modules | Auxiliary split; scout/gate training, new caches, both selectors, calibration |
| `deepglobe_improved_training.json` | Stronger segmenter controls | Retrain segmenters; then rebuild downstream artifacts |

`utility.mode` accepts `legacy`, `class_weighted`, or `miou`. `legacy` preserves the
historical formula including its configured weights. `class_weighted` uses the same
formula with an explicitly chosen nonuniform `utility.class_weights` vector; it is
an ablation of weighting, not a different loss. mIoU mode requires `routing.policy=direct`.
The original threshold policy remains `legacy`; direct policy compares feasible
low/high actions by positive marginal benefit per estimated cost, with optional
`cost_lambda`. It is greedy, not an optimal knapsack algorithm. It considers cheap
low alternatives when high is unaffordable, but does not revisit already allocated
initial actions. All cost budgets are per image and apply to estimated refinement
work, not total wall-clock latency.

## Core modules and their scope

* C1: `routing.py`: action comparison with low fallback and explicit budgets.
* C2: `objectives.py`, `training.py`: image-grouped metric loss and confusion-count
  marginal gains. Training uses valid-pixel counts; inference and calibration routing
  use geometric tile area, never ground-truth masks. This assumes ignored support is
  representative. Image mIoU is a surrogate for dataset-aggregate mIoU; IoU of expected
  counts is not expected IoU. `shared_marginals` enforces shared true-class row sums.
* C3: `geometry.py`: optional pooled class, margin, and boundary grids; original
  descriptor remains available with `descriptor_grid=0`.
* C4: `models.py`, `auxiliary.py`: shallow native-resolution scout predicting correct,
  recoverable, and unrecoverable regions, supervised using frozen raw refiner outcomes.
  Scout evidence is pooled into the selector; the scout supplements the coarse model.
* C5: `models.py`, `auxiliary.py`, `engine.py`: soft-class-conditioned acceptance gate.
  Only one-correct disagreements supervise its binary decision. Both-correct and
  both-wrong cases are masked from gate loss. It cannot fix pixels wrong in both
  candidates. Optional `training.gate_class_weights` weights supervision. This is
  class-conditioned correction/damage acceptance, not an additional pairwise loss.
* C6: `auxiliary.py`, `engine.py`: one optional low-to-high escalation. Its observation
  contains coarse and accepted-low soft distributions, changes, entropy and margin.
  It predicts current-versus-high outcomes. High still consumes coarse context; its
  acceptance gate also compares against coarse, matching training. Escalation uses
  **tile-local** mIoU utility, not an image-context second-stage optimizer. Low + high
  are both charged. Initial planned work is reserved before row-order escalation.

`Pipeline.refine(..., return_details=True)` exposes logits, probabilities, features
and hard predictions. Gate/scout/escalation checkpoints and descriptor settings are
fingerprinted; changing a gate requires rebuilding effective selector outcomes.
Do not toggle modules on an existing selector checkpoint: retrain the affected
selector and recalibrate. Research configuration changes deliberately invalidate
caches even when a particular toggle would not change every tensor.

## Data split discipline

Use distinct image paths for `train`, `auxiliary`, `selector`, `val`, and `test`.
The manifest reader rejects duplicate image paths. It cannot identify distinct files
that depict overlapping geographic regions: split scenes/geography before tiling.

1. Segmenters fit on `train`.
2. Scout and acceptance fit on `auxiliary`, validate on `val`.
3. Freeze auxiliaries, cache effective outcomes on `selector` and `val`.
4. Initial selector and escalation selector fit on `selector`, validate on `val`.
5. Calibrate deployment routing on `val` only.
6. Evaluate frozen choices on `test`.

Caching rejects segmenter/auxiliary training-image overlap. The auxiliary split must
not be carved from images already used to fit the frozen segmenters. If no unused
images remain, use a fresh split and retrain segmenters, or implement a properly
cross-fitted study. Merely renaming/copying training images is not a valid split.
Cross-fitting automation is not implemented.

## Commands: reuse frozen segmenters

Create `runs/research` containing copies of `coarse.pt`, `low.pt`, and `high.pt`.
The manifest must refer to the same original training identities and disjoint new
auxiliary images. Commands below use a shell variable only to shorten the example.

```bash
CFG=configs/deepglobe_research.json
RUN=runs/research
MANIFEST=runs/deepglobe/manifest.json
python -m selective_seg.cli train-scout --config "$CFG" --run "$RUN" --manifest "$MANIFEST" --device cuda
python -m selective_seg.cli train-acceptance --config "$CFG" --run "$RUN" --manifest "$MANIFEST" --device cuda
python -m selective_seg.cli cache --config "$CFG" --run "$RUN" --manifest "$MANIFEST" --split selector --device cuda
python -m selective_seg.cli cache --config "$CFG" --run "$RUN" --manifest "$MANIFEST" --split val --device cuda
python -m selective_seg.cli train-selector --config "$CFG" --run "$RUN" --device cuda
python -m selective_seg.cli train-escalation --config "$CFG" --run "$RUN" --device cuda
python -m selective_seg.cli calibrate --config "$CFG" --run "$RUN"
python -m selective_seg.cli benchmark --config "$CFG" --run "$RUN" --manifest "$MANIFEST" --split val --calibration "$RUN/calibration.json" --warmup 20 --passes 3 --device cuda
python -m selective_seg.cli evaluate --config "$CFG" --run "$RUN" --manifest "$MANIFEST" --split test --calibration "$RUN/calibration.json" --warmup 20 --passes 3 --profile-stages --profile-operations --device cuda
```

For the routing-only preset, omit both auxiliary training commands and train-escalation.
This is the first experiment to run when no disjoint auxiliary data is available.
Cache timing defaults to two warm-up calls per action and one measured call per tile;
research presets use three calls and their median. Warm-up/measurement include gate
acceptance. Costs are hardware/configuration-specific and must be recalibrated after
moving hardware or changing batching/precision.

## Reproducible computation protocol

`benchmark` accepts val/test; `evaluate` accepts only test. `--mode` selects
`selective`, `coarse`, `low`, `high`, `random`, or `entropy`. Coarse/low/high are full
reference policies and ignore refinement caps. Random/entropy obey configured caps
and estimated budget, using `routing.baseline_action` (default high). Random uses a
seeded permutation; repeat seeds for a distribution rather than a single reference.
Acceptance remains active in low/high controls if enabled; use the original config
for ungated references. Models are all resident after loading, including inactive
branches: reported GPU memory reflects this deployment layout, not a minimal
standalone coarse model. Timing only runs required branches.

Headline timing uses synchronized wall clock from decoded CPU input to completed CPU
mask, including CPU work/transfers and enabled modules. File/checkpoint I/O, verbose
diagnostics and extra profiling passes are excluded. Stage profiling introduces extra
synchronizations and must never replace the unprofiled timing. Profiling runs after
timing; use a separate benchmark process without profiling for final hardware studies.
Always warm up the entire inference path on representative shapes and repeat until
pass variability stabilizes. Default 20 warm-ups and 3 passes are a starting point,
not a guarantee. Hold hardware load, CPU threads, batch size and precision constant.

`metrics.json` contains raw per-image repetitions, mean/median/p95 of per-image
medians, mean pass latency and within-image CV, operation estimates per profiled
image, stage timing, metadata, GPU allocated/reserved peaks, action counts and actual
refiner call counts. CPU peak RSS is the process-lifetime high-water mark including
loading/diagnostics/profiling, not isolated inference memory. GPU peaks are read from
measured passes before extra diagnostics; reserved memory can reflect allocator
history. Use separate fresh processes for comparable memory numbers.

Operation counts are PyTorch profiler estimates for supported ops (multiply-add =
2 FLOPs). The report enumerates operators without estimates; these are not complete
hardware operation counts and must not be labeled complete FLOPs. Stage budget
comparisons include profiling overhead. `max_refine_ms` is not a hard real-time or
whole-image budget. The report compares estimated and profiled refinement work;
whole-image latency includes all overhead.

For each budget, make a separate config/run calibration, tune on val, then freeze it.
Plot measured latency vs mIoU; compute speedup as reference latency / candidate
latency. Do not derive savings from skipped-tile percentages. Bootstrap paired images
for uncertainty of *aggregate confusion-derived mIoU*, and use multiple selector seeds.
No significance or novelty is established by the synthetic tests.

## Ablations and stronger training

Compare the original pipeline, coarse-only, always-low/high, random/entropy controls,
direct allocation with legacy utility, class reweighting, metric utility, descriptors,
scout, acceptance, escalation, full core, and leave-one-module-out variants. Keep
segmenters fixed in the first track. Report per-class IoU and directional correction
and damage matrices, plus confident-error recovery (default confidence >= .9).

The improved-training preset enables within-image class-balanced tile sampling,
label-preserving flips/90-degree rotations, CE plus soft-IoU loss, full-tile validation
on all validation images, and mIoU checkpoint selection. `training.class_weights`
optionally supplies positive CE weights. This is a stronger baseline, not novelty.
All downstream caches/checkpoints/calibrations must be rebuilt after segmenter changes.
Full-tile validation rereads samples/coarse context and is intentionally correctness
first; it can be expensive on large validation sets.

## Deferred extensions

Wider-context actions, hard patch-wise band selection, arbitrary-channel spectral
encoders, and spatial-versus-spectral allocation are **not implemented**. They remain
a separate study requiring new training data/actions and diagnostic evidence, as in
the implementation plan. Existing optional three-channel soft spectral projection is
preserved. Do not claim hard band-selection savings or multispectral validation from
RGB experiments. Class-based thresholds, a separate pairwise contrastive objective,
and an exact oracle optimizer are also not added by this revision.

## Verification

```bash
python -m pytest -q
python smoke_test.py
python smoke_research.py
```

The second smoke test exercises all core modules, effective caches, calibration,
escalation, profiling and baselines on tiny synthetic data, and verifies rejection of
auxiliary/selector leakage. The tests include a real SegFormer-B0 forward/backward pass
with random initialization; they do not download pretrained weights or measure real accuracy.

Validation-only headroom diagnostic (greedy, not an exact oracle upper bound):

```bash
python -m selective_seg.cli analyze-cache --config configs/deepglobe_routing.json --run runs/routing
```

This reports coarse/always-low/always-high confusion matrices and label-assisted
allocation at the configured budget. It cannot be used as a deployable inference mode.
