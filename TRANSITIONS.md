# Consistent class-transition selection

This experiment asks whether predicting a consistent joint class-transition table
improves the correction–damage trade-off over direct gain regression. It is an
experimental addition, not evidence that the method is novel or better. It does
not retrain the segmenters or change the running diagnostic pipeline.

## Model and decisions

For each candidate tile and action, predict `T[a,y,c,r]`: expected pixel counts
for truth class `y`, coarse winner `c`, and refined winner `r`. There are six
classes and two actions (low/high). The model factors this table as:

```
N(c) * P(valid | c,z) * P(y | c,z,valid) * P(r | y,c,z,a,valid)
```

`N(c)` is the observed coarse-winner histogram over **all** pixels. The validity
head estimates ignored-label mass without seeing the ground-truth ignore mask.
`P(y,c)` is shared across actions. Thus every action has the same coarse
confusion marginal and true-class support. The conditional head is masked to
the actual agriculture/rangeland merge: other coarse winners cannot change.
The predicted table retains wrong-to-wrong class transitions as well as
corrections and damage. Counts are float expectations, not rounded pixel labels.

The allocator sums predicted tile confusion matrices over the whole image.
After accepting a tile/action, it recomputes all remaining marginal mIoU gains.
It allows at most one action per tile, a tile-fraction cap, and an estimated
refinement-time cap. Optional constraints limit expected damage separately for
each class to a fraction of that class's predicted initially correct pixels.
Policies with damage penalties instead subtract normalized expected damaged
pixels from marginal mIoU gain. Selection stops when no feasible positive gain
remains. This is a greedy heuristic; no optimality is asserted.

The time cap is `budget_fraction * total_tiles * median_high_action_ms`, with
median action costs estimated only on the fitting subset. It constrains estimated
action cost, not total runtime. Actual end-to-end time includes overview,
descriptors, CPU head/allocator, transfers, and selected refinements. Image
decoding and metric calculation are outside measured inference.

**Predicted damage caps are not guarantees about actual damage.** Reports include
actual damage per class, predicted-damage error, and image-level violations of the
corresponding fraction of actual initially correct pixels. No uncertainty-bound
or conformal calibration claim is made.

## Fixed protocol

- Frozen existing coarse/low/high checkpoints, 30% candidate cap, same A/R merge,
  no specialist evidence. This isolates the transition model from the separate
  specialist experiment. GPU segmenters retain the existing FP16 autocast path;
  heads and count predictions use FP32, allocator arithmetic uses FP64.
- Fresh pixel-aligned joint targets, including ignore handling and edge tiles.
  Existing marginal caches cannot recover all joint transitions.
- Deterministic 60/20 image split of the 80-image selector partition for fitting
  and early stopping. The partition is fixed independently of initialization.
  Heads use seeds 42, 43, 44, at most 200 epochs, patience 20. Loss is tile/image
  count-weighted likelihood for transitions and valid/coarse marginals; scalar
  control retains smooth-L1 gain regression.
- Evaluate on 80 validation images. No test samples enter this run. Existing
  backbone development has used validation, so these remain development results.
- Eight policies: existing `PatchSelector` architecture retrained on the same
  fit/tune partition; independent action-specific coarse marginals; shared
  marginals with static scoring; shared with recomputed scoring; damage penalties
  1 and 2; predicted per-class damage caps of 1% and 5%.
- Six budget fractions: 5%, 10%, 20%, 30%, 50%, 100%. The unchanged 30% candidate
  cap means larger budgets cannot refine every tile. All policies have identical
  candidate, tile and estimated-time constraints. Parameter counts are saved:
  the heads are not parameter-matched, which is a limitation of this first study.
- Paired image bootstrap intervals (1,000 draws) against both coarse and the
  matched scalar control; report every seed/policy, not only the best outcome.
  Intervals are exploratory and not adjusted for multiple comparisons.
- Real live inference for scalar, independent, shared, and shared 1%-cap policies
  at 20% budget, seed 42, three passes, one excluded warmup. Selected actions and
  confusion matrices must exactly match replay before timings are accepted.

The independent-head ablation predicts an action-specific `P(y|c,z,a)` in
addition to the coarse head; this intentionally removes shared marginal
consistency. The static shared ablation isolates the effect of recomputing gains.
The scalar control uses the existing model and loss but is retrained without
validation early stopping. Its results are not directly interchangeable with
the older six-class replacement-merge baseline.

## Run and monitor

Run from the repository with its existing Python environment:

```bash
bash scripts/start_transition_service.sh
tail -F runs/transitions-20261121-gpu/service.log
```

The launcher starts `selective-refinement-transitions-20261121.service`. It waits
for `runs/diagnostics-20261121-gpu/status.json` to report `complete` before using
the GPU, protecting the older run's measurements. If that job fails, this job
fails with an explanation; a stale running status times out after 48 hours.
`WAIT_FOR_STATUS=none` disables the wait when the GPU is free. Do not use that
override while another benchmark is running. This is a transient user service,
independent of the chat/terminal, but it is not scheduled to restart after reboot.

```bash
watch -n 5 'cat runs/transitions-20261121-gpu/status.json'
systemctl --user status selective-refinement-transitions-20261121.service
```

Direct invocation, or a smaller explicitly labeled pilot:

```bash
/home/himesh/Downloads/.venv/bin/python -u -m selective_seg.transition_experiment \
  --output runs/transitions-pilot --device cpu --limit 2 --epochs 2 \
  --seeds 42 --passes 1
```

Outputs: `identity.json` binds source, config, manifest and checkpoint hashes;
`protocol.json` lists image partitions; `cache/{selector,val}` stores atomic
per-image targets; `seed*/{scalar,independent,shared}.pt` stores heads and training
histories; policy directories contain per-budget paired reports and live
20%-budget timings; `summary.json` compares all seeds and policies. `status.json`
and `service.log` report progress and failures. A lock prevents two processes
writing one run. Repeating the same command resumes completed rows/heads/live
reports; changed source, options or inputs require a new output directory.

`TransitionPipeline.predict(image)` is the deployable interface. It accepts
no target mask and does not execute refiners before choosing actions.

## Research claim to test

The candidate contribution is the combination of class-transition consistency,
marginal mIoU reallocation, and explicit class-damage constraints under a compute
budget. Adaptive segmentation itself already has precedents, including
[SegBlocks](https://arxiv.org/abs/2011.12025) and
[PointRend](https://openaccess.thecvf.com/content_CVPR_2020/html/Kirillov_PointRend_Image_Segmentation_As_Rendering_CVPR_2020_paper.html).
This first run must establish a reproducible improvement over scalar penalties
and independent heads. A stronger paper also needs capacity-matched controls,
calibrated uncertainty, another class pair/dataset, and a held-out final test
after the method is frozen. A successful implementation alone establishes none
of those claims.
