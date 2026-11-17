# Hybrid specialist experiments

The new CLI uses the existing DeepGlobe manifest and frozen segmenter checkpoints.
It writes each operation to a new directory. The original CLI still runs the
original specialist and selective pipeline.

## Audit of the original paths

| Component | Original implementation | Redesigned experiment |
|---|---|---|
| Coarse | SegFormer-B0 using MiT-B0 and the multistage SegFormer decoder. The RGB config initializes the encoder from `nvidia/mit-b0`, resizes the long side to 640, and upsamples logits before softmax. Defaults are 40,000 coarse updates, batch one, AdamW at 6e-5 with warmup and decay, and the best validation-loss checkpoint from up to eight validation images. | Reuse the exact frozen checkpoint and geometry. Coarse confidence, normalized entropy, pair probability, boundary and texture cues rank candidates. |
| Selector | 271 inputs from 256 pooled features, six class probabilities and nine uncertainty, boundary and texture statistics. The MLP predicts transition, confusion and benefit distributions. The legacy rule thresholds the probability that either action helps. | Keep the coarse descriptor and compare prediction-only vs. expanded specialist evidence. Regress the signed image-level mIoU gain of each action on each candidate. Rank positive gains with an optional incremental time penalty. |
| Crop and reconstruction | Non-overlapping 512-pixel centers, 64-pixel halo, replicate-padded edges. Low sees 320 and high 640. Selected central predictions replace coarse labels. | Reuse crop utilities and edge alignment. Publish only A/R changes at prediction-supported pixels. Keep the A/R probability mass and unrelated probabilities fixed. Protect the original winner when a third class would overtake the pair. |
| Standalone specialist | ResNet-18 by default (ResNet-50 optional), EfficientNet-B0 or both. Four stages per branch, learned 1×1 projection, all projected levels upsampled and concatenated at the shallowest level. Weighted binary CE on A/R labels, weights fitted on train. Runs every tile with uncalibrated binary metrics. | Compare stride-aligned concatenation, FPN and an Inception-inspired FPN. Train on the existing train split, choose checkpoints by aggregate validation binary mIoU, and optionally calibrate exit temperatures. |
| Integration | `engine.Pipeline.from_run` loads coarse/low/high, selector and enabled research auxiliaries. It never loads `specialist.pt`. The `evaluate-specialist` entry point can load a coarse baseline for diagnostics only. Reports from these two paths measure different tasks. | Load frozen specialist on candidate tiles only. Pass evidence to a separately trained selector. Evaluate existing refiners or the cached specialist prediction as competing actions. |
| Evaluation and provenance | Original benchmark synchronizes wall-clock time from decoded CPU image to CPU mask, repeating unprofiled passes. Configuration signatures gate loading. Segmenters are hashed in reports; calibration checks selector and dependent checkpoint hashes. Standalone specialist writes its own hashes. | Preserve those controls. Hash all consumed checkpoints and the specialist calibration. Require matching cache/selector provenance. Report confusion per image, confidence metrics, paired bootstrap, budget use and whole-pipeline latency. |

The audited base is `b3e9abfce241b90c635793450283d7d4a1792cf5`.
The repository contained one historical report, `metrics.json` (SHA-256
`1958ccdfff145778ce791e349a73552892f090863e6addcb8dc10653b8f04b16`).
It lists 81 images, mIoU 0.6598055, mean 1536.34 ms/image, and 1498/97/430
skip/low/high tiles. It records coarse SHA-256
`4ab988f228bbd6f133c6b705c177a12187f5b73e7d9bae22d67bfdc37656f508`,
low `cb4c43beffa884eee4eafd50a105b5d56ec98a907237acf22a637ddedd21c6cd`,
and high `c7d22638b7119ed672da1300a5cf331150060a0591bdae4670c813c443431619`.
No specialist hash appears, consistent with the executable graph. The supplied
1322.13 ms report, 581.45 ms coarse report, baseline recheck and standalone
specialist report are external to this checkout. Treat the reported identical
selective/recheck outputs as one observation. On the data machine, compare them
with `sha256sum report.json recheck.json` and `cmp report.json recheck.json`,
then inspect each command, mode, split, checkpoint provenance, and timing scope
before assigning any result to a treatment. The reported binary 0.700369 mIoU
does not measure six-class integration.

## Essential architecture

1. Freeze coarse, low and high. Extract candidates from the original grid
   using six-class probabilities, A/R mass and ambiguity, low confidence,
   coarse label edges and the RGB local-detail residual. Rank those satisfying
   the average A/R-mass threshold and apply the candidate cap. This heuristic
   can miss confident errors. Reports include candidate recall of true A/R
   errors and confident errors using labels only after inference.
2. Run the specialist once per candidate. Both encoders see the same context
   crop. Binary 0 maps to agriculture/source 1 and binary 1 to rangeland/source 2.
   Compute entropy, hard disagreement in source ID space, a soft distance to
   coarse conditional A/R probabilities, confident evidence, boundary uncertainty,
   directional disagreement statistics and a heuristic cosine gap between the
   projected shallow and deep features. The cosine gap is an optional statistic,
   not a calibrated inter-backbone disagreement. Outside prediction-derived A/R
   support, mark the maps unsupported. For missing or early-exit evidence, explicit
   flags distinguish missing features from zeros. Per-tile outputs are optional.
3. Freeze these networks. Cache selector and validation features and run each
   allowed action on each candidate offline. A target is the difference in
   **image-level mIoU** after replacing that tile in the coarse confusion matrix.
   A single-tile target is a marginal effect around the coarse baseline, so
   simultaneous effects can interact. Fit a small action regression head with
   signed targets. Compare none/predictions/full evidence under the same candidate
   rule, action set and budget.
4. For neighborhood ablations, add four directional seam consistency scores and
   presence bits. Compare no context, an average of adjacent latent descriptors,
   a learned weighted mix of the 3×3 positions, and a 3×3 convolution. The grid
   aligns truncated edge tiles by index. There is still one decision per
   non-overlapping central tile.
5. Keep or apply the selected action on each tile. `--actions low high` is the
   clean evidence-only comparison. `--actions specialist low high` adds a merge
   action using probabilities already computed by the specialist, with no second
   specialist forward. `--actions specialist` isolates that alternative. In all
   these configurations, a true A/R pixel predicted as a third class by coarse
   is outside the scope of the A↔R update.

### Fusion choices

These channels and strides come from the installed torchvision feature probes.
At 320-pixel input the maps are 80², 40², 20² and 10².

| Stride | ResNet-18 | EfficientNet-B0 |
|---:|---|---|
| 4 | layer1, 64 | features[2], 24 |
| 8 | layer2, 128 | features[3], 40 |
| 16 | layer3, 256 | features[5], 112 |
| 32 | layer4, 512 | features[7], 320 |

Each branch projects to 32 channels before fusion. The corresponding stride
features are compressed back to 32 channels. The concatenation control aligns
levels on the stride-4 grid. FPN uses top-down addition and local smoothing.
Inception replaces FPN smoothing with bounded-width parallel 1×1, 3×3,
stacked 3×3, and dilated 3×3 branches followed by compression. The binary decoder
emits two logits at stride 4. It restores the physical crop size and discards the
halo for evaluation or merging. Start with single-backbone controls and FPN.

### Optional early exits

Train a separate checkpoint with `--train-exits`. Two auxiliary heads operate
after stride 4 and stride 8, supervised by masked CE. Additional scalar heads
estimate the deeper head's binary pixel-accuracy gain. `calibrate` fits
per-exit temperatures and confidence/benefit thresholds on the existing
validation split. The threshold search minimizes mean exit depth within the
configured validation binary-mIoU tolerance of full-depth performance. This
uses depth as a cost proxy; measured latency is the result. Confidence uses
the 10th percentile over prediction-supported pixels. Patches without support
continue deeper. Pooling, exiting and backbone execution run incrementally,
so a shallow exit skips the later blocks of both encoders.

Compare full, confidence and benefit policies on the **same exit-trained**
checkpoint. Retrain the selector for each policy because the evidence changes.
A separate full-depth checkpoint without auxiliary heads isolates auxiliary
training from inference-time exits. Use standalone `--exit-policy shallow` and `--exit-policy intermediate`
for forced-depth feature quality comparisons on the same exit-trained checkpoint. Keep all adaptive policies off for the
primary hybrid ablation.

## Reproduce the experiments

From the repository root using your existing CUDA PyTorch/torchvision environment:

```bash
python -m pip install -e '.[specialist,test]'
python -m pytest -q
MANIFEST=/absolute/path/to/your/manifest.json
BASE=/absolute/path/to/your/existing/deepglobe/run
ROOT=runs/hybrid-fpn-seed42
CFG=configs/deepglobe_specialist.json
CPU_THREADS=8
export OMP_NUM_THREADS="$CPU_THREADS"
export MKL_NUM_THREADS="$CPU_THREADS"
H=(python -u -m selective_seg.hybrid_cli)
SPEC=(--config "$CFG" --device cuda --fusion fpn --backbones combined)
DATA=(--manifest "$MANIFEST" --base-run "$BASE")
```

Run the short resource screen first and record the command, free disk, free
VRAM and pilot estimates. It measures four batch-one training iterations and
a short specialist forward sample. The dataset index and validation scan can
increase elapsed time. Verify that the chosen GPU has room without displacing
existing jobs. Change ROOT to start a new independent experiment.

```bash
"${H[@]}" preflight "${SPEC[@]}" "${DATA[@]}" --pilot --output "$ROOT"
"${H[@]}" train-specialist "${SPEC[@]}" "${DATA[@]}" --output "$ROOT/specialist"
"${H[@]}" calibrate "${SPEC[@]}" "${DATA[@]}" \
  --specialist-run "$ROOT/specialist" --output "$ROOT/calibration"
FROZEN=(--specialist-run "$ROOT/specialist" --calibration "$ROOT/calibration/calibration.json")
"${H[@]}" specialist "${SPEC[@]}" "${DATA[@]}" "${FROZEN[@]}" --output "$ROOT/binary-val"
```

Use the following for one integration treatment. Set `TAG` separately for each
treatment. For the no-specialist control omit `FROZEN` and use `--evidence none`;
otherwise include it throughout. Changing the action set, evidence or
neighborhood requires new caches and a new selector. Changing only the
refinement budget does not.

```bash
TAG=full-conv
ROUTER=(--evidence full --neighbourhood conv --actions low high --candidate-fraction .30)
for SPLIT in selector val; do
  "${H[@]}" cache "${SPEC[@]}" "${DATA[@]}" "${FROZEN[@]}" "${ROUTER[@]}" \
    --split "$SPLIT" --output "$ROOT/$TAG/cache/$SPLIT"
done
"${H[@]}" train-selector "${SPEC[@]}" "${ROUTER[@]}" \
  --selector-run "$ROOT/$TAG/cache" --output "$ROOT/$TAG/model"
"${H[@]}" preflight "${SPEC[@]}" "${DATA[@]}" "${FROZEN[@]}" "${ROUTER[@]}" \
  --selector-run "$ROOT/$TAG/model" --pilot
"${H[@]}" evaluate "${SPEC[@]}" "${DATA[@]}" "${FROZEN[@]}" "${ROUTER[@]}" \
  --selector-run "$ROOT/$TAG/model" --split val --all-budgets --output "$ROOT/$TAG/val"
```

Reproduce A and B with the **original** RGB config. Use the selector calibration
from the existing run for B. Use new destinations. After freezing selections
on validation, repeat these commands with `--split test` and new outputs.
Reference CIs require the regenerated comparator report with per-image confusion.

```bash
for MODE in coarse selective; do
  CALIBRATION=()
  if [[ "$MODE" == selective ]]; then
    CALIBRATION=(--calibration "$BASE/calibration.json")
  fi
  python -m selective_seg.cli benchmark --config configs/deepglobe_rgb.json \
    --manifest "$MANIFEST" --run "$BASE" --split val --mode "$MODE" \
    "${CALIBRATION[@]}" --device cuda --passes 3 --warmup 10 \
    --probability-metrics --output "$ROOT/legacy-val-$MODE"
done
```

Final paired example after that baseline recheck on test and after fixing all
variants and budgets on validation:

```bash
"${H[@]}" evaluate "${SPEC[@]}" "${DATA[@]}" "${FROZEN[@]}" "${ROUTER[@]}" \
  --selector-run "$ROOT/$TAG/model" --split test --all-budgets \
  --paired-baseline "$ROOT/legacy-test-selective/metrics.json" --output "$ROOT/$TAG/test"
```

### Ablation matrix

| ID | Treatment | CLI flags relative to the fixed RGB setup |
|---|---|---|
| A | Original coarse | Original CLI, `--mode coarse` |
| B | Original selective | Original CLI, `--mode selective` with matching calibration |
| C | ResNet-only | `--backbones resnet --fusion concat` |
| D | EfficientNet-only | `--backbones efficientnet --fusion concat` |
| E | Hybrid concat | `--backbones combined --fusion concat` |
| F | Hybrid FPN | `--backbones combined --fusion fpn` |
| G | Hybrid Inception | `--backbones combined --fusion inception` |
| H0 | Candidate-matched coarse evidence | `--evidence none --neighbourhood none --actions low high` |
| H | Specialist probability summary | `--evidence predictions --neighbourhood none --actions low high` |
| I | Add uncertainty and disagreement | `--evidence full --neighbourhood none --actions low high` |
| J | Add surrounding context | I with `--neighbourhood mean`, then `learned`, then `conv` |
| K | Add a direct specialist refinement option | Best J variant with `--actions specialist low high`. Compare with `--actions specialist` at the same evidence settings. |
| L | Adaptive specialist | Add `--train-exits` at fitting, calibration and evaluation. Compare separately cached/fitted `--exit-policy full`, `confidence` and `benefit`. |

Repeat with `--seed 42`, `--seed 43`, `--seed 44` and separate roots when
feasible. Keep steps, split, crop and preprocessing fixed. Label-derived patch
selection is used solely during specialist fitting. Out-of-run checkpoint and
selector provenance guards assume images kept their original paths. Audit
geographic overlap separately. Brier is multiclass summed squared error and
risk–coverage uses 15 confidence bins. Report calibration changes together with
accuracy and abstention coverage, and use paired bootstraps on common images.

## Budget and resource interpretation

Both caps are proportions of **all unique central tiles per image**. One tile
is the minimum when the chosen fraction is positive. At 25 tiles the six
refinement caps are 1/2/5/7/12/25. Total candidates are capped independently
(7 at 30%), so the refinement curve may saturate before 100%. For full
candidate screening use a separate `--candidate-fraction 1 --min-pair-mass 0`
treatment. It keeps the all-image denominator and can run 25 specialist forwards
per image. A budget is a maximum; nonpositive gains can leave it unspent. Specialist
calls precede selection and their time is charged even for rejected tiles.
The inference report includes actual tile and area fractions, calls, probability
metrics and mean/median/p95 elapsed wall time. The standalone binary report has its
own timing scope and cannot substitute for integrated latency.

At 32-channel fusion, plan roughly 60 MB of FP32 specialist parameters per
combined checkpoint, about 240 MB for parameters + gradients + Adam state
before activations/workspace, and up to double checkpoint footprint briefly during
a best-model replacement. The cache stores tile descriptors and outcome scalars,
typically low tens of megabytes across hundreds of images. It stores no dense
maps. Optional `--save-maps` export can dominate disk: one 2448² image may use
tens of MB before compression. Start without maps for sweeps. Full GPU use
includes activations, old base checkpoints, cached tile probabilities and the CUDA
allocator. Peak GPU numbers are reported from synchronized measured passes.
Use the pilot on the real GPU before a long run.

The main paper comparison is specialist-guided routing vs. the matched no-evidence
control and original selective method at the same measured latency. Check whether
it reduces both directional A/R confusions and whether training a stronger
whole-image coarse baseline offers a better cost point. Projection, FPN and
early exit are design alternatives; their benefit remains an empirical question.

## Verification performed

Environment: Python 3.12, PyTorch 2.15.0+cpu and torchvision 0.29.2+cpu.
The full CPU suite passed (35 tests). The synthetic integration case used four
distinct train/selector/val/test images, one specialist training step and two
selector epochs. It exercised calibration, a six-budget sweep, map export,
provenance, protected-class reconstruction, paired bootstrap and new-directory
enforcement. At its 100% synthetic budget it invoked the specialist four times
and selected two specialist merges, one low and one high action. A forward hook
verified that a forced confidence exit skipped the final backbone block.
A standard 320×320 probe produced 1×2×80×80 logits for all five variants:

| Variant | Trainable architecture parameters before freezing |
|---|---:|
| ResNet-18 concat | 11,248,706 |
| EfficientNet-B0 concat | 3,652,734 |
| Combined concat | 14,864,190 |
| Combined FPN | 14,864,382 |
| Combined Inception | 14,862,910 |

These are execution and architecture checks. This workspace had neither the
DeepGlobe manifest/images nor the trained segmenter checkpoints or a CUDA GPU.
No real-data quality, speedup, GPU peak, convergence or significance claims
are made. The 0.700369 binary and 0.659806 selective figures are prior reports,
with only the checked-in legacy report independently inspected here.
