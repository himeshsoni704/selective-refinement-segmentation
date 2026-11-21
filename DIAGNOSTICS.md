# Diagnostic experiments and novelty assessment — 21 November 2026

The current evidence supports investigating routing and damaging refinements before
training another backbone. It does not establish a new segmentation architecture
or a statistically certified class-damage controller.

## Completed validation measurements

The existing `runs/deepglobe/val_targets.pt` cache matches SHA-256 hashes of the
current coarse/low/high checkpoints. These are pooled confusion metrics on 80
validation images, not the previously reported 81-image test results.

| Original six-class path | mIoU | Rangeland IoU |
|---|---:|---:|
| Coarse | 60.05% | 34.58% |
| Always low | 58.35% | 28.86% |
| Always high | 65.91% | 39.45% |
| Label-assisted greedy, 20% tile cap | 68.12% | See cached report |
| Label-assisted greedy, full tile allowance | 74.16% | See cached report |

The label-assisted policy uses validation labels. It recomputes image-level
marginal mIoU after each chosen tile and assembles final confusion counts for the
reported pooled dataset score. It is a heuristic, not an exact oracle upper bound.
Equal tile caps are not equal compute: report action costs and full timings too.

Fresh evaluation with the same FP32 crop/resize/mask/metric implementation on all
80 validation images (2,000 tiles per model):

| Binary specialist | Agriculture IoU | Rangeland IoU | mIoU | NLL |
|---|---:|---:|---:|---:|
| Original combined, saved step 9,000 | 92.58% | 47.45% | 70.02% | 0.23488 |
| FPN retry-fixed, saved step 8,000 | 90.44% | 37.99% | 64.22% | 0.34619 |

Both checkpoints have finite stored states. Both FP32 evaluations found zero
nonfinite raw/central logits and probabilities, checking all tiles including
ignored regions. The lower FPN score persists without mixed-precision inference.
These models differ in architecture and training history; this is a checkpoint
comparison, not an isolated causal test of FPN fusion. A saved best step does not
prove the configured training run completed. No new backbone training was started.

## Code review findings

1. **Precision mismatch (fixed).** Hybrid training validation had a local BF16
   edit, but `specialist_maps` and calibration still used the global FP16 context.
   This could make deployment numerically different from validation. A shared
   specialist precision policy now defaults to BF16 on supported GPUs and FP32
   otherwise. Explicit FP32/FP16/BF16 switches allow diagnostic reproduction.
   FP16 training uses gradient scaling; BF16/FP32 do not. Nonfinite gradients fail.
2. **Unsafe confidence accumulation (fixed).** NaNs can reach confidence-bin
   conversion and fail with an unrelated bincount error. Metrics now validate the
   full probability tensor before mutating confusion counts, including ignored
   pixels, normalization, ranges, shapes, and empty supervision. Specialist
   inference, validation and exit calibration check logits as well.
3. **Different merge rules (made explicit).** The original path replaces all
   six-class center predictions. The hybrid path conditions refiner outputs on
   A/R, preserves pair mass, and protects other original winning classes. Comparing
   those paths as if the only difference were specialist evidence is confounded.
   The new evidence off/on study uses the same hybrid merge rule on both sides.
4. **Outcome representation.** The original selector predicts aggregate
   correction/damage and per-action confusion, while the hybrid selector predicts
   a scalar single-tile marginal mIoU. Neither predicts calibrated class-specific
   negative-flip risk for constrained deployment. Fresh hybrid caches now retain
   class corrections, class damage and action confusion ledgers for the first
   observed-outcome experiment; this is not a new learned risk controller.
5. **Non-additive IoU.** Summing independently predicted single-tile IoU gains
   need not maximize final IoU. The original direct metric policy recomputes
   marginals; hybrid `choose` ranks fixed scalar gains. Existing tile centers are
   nonoverlapping, so confusion changes are additive even though IoU is not.
6. **Budget interpretation.** Hybrid routing enforces tile caps and an optional
   time penalty; it does not guarantee a total-latency budget. Specialist inference
   is paid on all candidates, even skipped tiles. The observed class-cap study
   adds an estimated action-time cap, but its report explicitly excludes common
   overhead and makes no end-to-end guarantee.
7. **Split limitations.** File-path overlap guards do not establish geographical
   independence. Validation was already used for checkpoint and selector selection.
   The new results are development evidence, not an independent final test or
   certification set. Paired bootstrap uses images, not independent pixels;
   geographic clusters would be preferable when metadata is available.
8. **Layer claim.** Both specialists compute four backbone strides. The FPN
   configuration has `train_exits=false` and `exit_policy=full`. No shallow-only,
   deep-only or early-exit empirical result follows from those weights. Train such
   ablations only after these diagnostics justify that expense.

Pre-existing edits in `hybrid_specialist.py` and `hybrid_metrics.py` were preserved
in intent and extended. The untracked audit text and `.before_bf16` backup were
left untouched and are excluded from commits.

## Related work and the novelty boundary

- [SegBlocks](https://arxiv.org/abs/2011.12025): learned block-resolution allocation
  for segmentation already covers adaptive spatial compute.
- [PointRend, CVPR 2020](https://openaccess.thecvf.com/content_CVPR_2020/html/Kirillov_PointRend_Image_Segmentation_As_Rendering_CVPR_2020_paper.html):
  adaptive point processing for segmentation refinement.
- [GeoAgent, ICCV 2023](https://arxiv.org/abs/2309.15372): scale-adaptive context
  for remote-sensing patches. A wider-context explanation needs direct evidence.
- [Positive-Congruent Training, CVPR 2021](https://openaccess.thecvf.com/content/CVPR2021/html/Yan_Positive-Congruent_Training_Towards_Regression-Free_Model_Updates_CVPR_2021_paper.html):
  reducing newly introduced errors (negative flips) already has precedent in
  model updates; transfer to refinement is a distinction to investigate.
- [Learn then Test](https://arxiv.org/abs/2110.01052): held-out statistical risk
  control with multiple constraints is established.
- [Conditional Conformal Risk Adaptation](https://arxiv.org/abs/2504.07611):
  conditional risk adaptation for segmentation prediction sets is established.

**Assessment:** ResNet/EfficientNet fusion, FPN, CNN layers, neighborhood features,
early exits, or simply adding a constrained optimizer do not individually support
a strong novelty claim. A potential contribution is an efficient action-conditional
class-damage predictor and allocation method that improves correction–damage–cost
tradeoffs against strong penalty baselines. A limited literature search cannot
establish that this combination is unprecedented.

Separate constraints can select useful operating points, but scalarization can
recover many supported Pareto points. Compare swept penalties, rather than a
single convenient lambda. Report infeasibility/fallback-to-skip, damage
normalization, class support, calibration uncertainty, and solver overhead.
Prediction intervals on action damage do not automatically certify the final
selected and merged policy. Fit the predictor, choose the policy, calibrate the
complete policy, and evaluate on disjoint scene groups. Arbitrary refinement
policies need not have the monotone risk structure required by some conformal
methods; a proof must match the actual policy and sampling assumptions.

## Executable experiment order

1. `selective_seg.diagnostics specialist`: compare original and FPN on validation
   in explicit precision, with per-class IoU, confusion, NLL, state/output checks,
   image progress and checkpoint/manifest hashes.
2. `selective_seg.diagnostics controls`: replay verified six-class cache outcomes
   for coarse/always-low/always-high, existing calibrated selector, random high,
   label-assisted scalar utility and dynamically recomputed mIoU assistance.
3. `selective_seg.diagnostic_suite`: generate fresh fixed-candidate low/high
   outcome caches for selector and val, fit evidence-off and evidence-on selectors,
   then evaluate six budgets with three synchronized timing passes. All frozen
   models, candidate selection, merge rules, images, seeds and budgets are fixed.
   The comparison asserts exact equality of candidates, coarse descriptors,
   confusion outcomes and class correction/damage targets. End-to-end timing
   includes specialist calls. A 30% candidate cap limits every budget above 30%.
4. `selective_seg.routing_tradeoffs`: from the same observed outcomes compare
   gain-only, swept scalar damage penalties, and per-class caps. All share an
   estimated compute allowance and tile cap. Damage cap denominator is the count
   of coarse-correct pixels of that class in each image. The objective here is
   corrected pixels, not exact mIoU maximization. This uses validation labels and
   establishes only an empirical feasibility/tradeoff screen.
5. Only if step 4 shows a useful frontier: implement a learned class-damage
   predictor, reserve scene-level policy-selection/calibration/evaluation groups,
   and test whether the advantage survives prediction error and distribution
   shift. If evidence on/off fails, avoid adding specialist complexity. If it
   succeeds only before charging specialist cost, investigate cheaper evidence.

All steps use the existing segmenter/specialist weights. Only the small routing
selectors are trained. The final test split is not touched by this suite.

## Run and inspect

```bash
# Uses the known local Python environment; CUDA is checked inside the service.
./scripts/start_diagnostic_service.sh
# Full raw artifacts are ignored by Git under runs/diagnostics-20261121-gpu/.
cat runs/diagnostics-20261121-gpu/status.json
tail -n 30 runs/diagnostics-20261121-gpu/service.log
systemctl --user status selective-refinement-diagnostics-20261121.service
journalctl --user -u selective-refinement-diagnostics-20261121.service -n 30
```

The transient service has `Restart=no`; lingering was already enabled. It survives
terminal closure/logout but not reboot. The example persistent unit is provided
for deliberate installation; it was not enabled to rerun experiments on boot.

A second launcher cannot use an active unit name. To retry after failure, inspect
logs and clear the failed unit or set `UNIT_NAME` to a new name. Reuse the same
output only with identical source/input/options: completed stages are verified by
hash, and unfinished strict-directory stages are preserved with `.interrupted-*`
names before being recomputed. A file lock rejects simultaneous suites. Changed
inputs or code require a new output directory. The suite does not resume optimizer
states or partial cache generation within a stage.

For an explicit CPU run use `DEVICE=cpu` and a different output directory. CPU
accuracy checks do not establish GPU latency. A small `--limit 1 --epochs 2
--passes 1` suite is an execution pilot, never accuracy evidence.
