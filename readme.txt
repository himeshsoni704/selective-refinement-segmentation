ANCHOR-CENTRED MULTI-SCALE CONTEXT EXPERIMENTS
============================================

Research question
At the same observed end-to-end latency, does selecting a lower-resolution
context footprint for one detailed anchor recover rangeland while avoiding
damage to correctly classified agriculture? Each action writes only the valid
anchor. All original checkpoints and dataset partitions are read-only.

Quick start on the training workstation
---------------------------------------
From inside your existing checkout, with its working GPU Python environment:

    git fetch origin research/hybrid-specialist
    git switch research/hybrid-specialist
    git pull --ff-only origin research/hybrid-specialist
    export PYTHON_BIN="${VIRTUAL_ENV:+$VIRTUAL_ENV/bin/}python"
    if [ -x /home/himesh/Downloads/.venv/bin/python ]; then
        export PYTHON_BIN=/home/himesh/Downloads/.venv/bin/python
    fi
    bash scripts/start_anchor_service.sh --steps 2000 --epochs 60 --passes 3 --budgets 100 250

The existing defaults are configs/deepglobe_rgb.json,
runs/deepglobe/manifest.json and runs/deepglobe/{coarse,low,high}.pt.
Override them with CONFIG, MANIFEST, BASE_RUN and PYTHON_BIN when required.
Use the baseline config consistent with your original checkpoints; the loader
checks the signature. The full study reads images on train, selector and val.
The manifest validator also checks that all listed paths exist.

The launcher checks the user systemd manager, prints available disk, executes a
five-step foreground pilot, prints measured memory/time/storage estimates and
the exact worker command, then submits the study under a unique user unit.
Keep this shell open until the unit name appears. The pilot trains throwaway
weights; the background worker starts the real training from its seeded init.
Batch size is one. The default settings are research choices, not calibrated
DeepGlobe optima. Budget numbers are additional acquisition milliseconds.
Choose them based on the pilot and validation measurements on your hardware.
Avoid comparing timings while another job uses the same GPU. The launcher
does not stop other processes. For service persistence across complete
logout check “loginctl show-user "$USER" -p Linger” on the workstation.

    systemctl --user list-units --all 'anchor-context-*'
    # Substitute the output directory printed by the launcher:
    tail -f runs/anchor-XXXXXXXX/service.log
    cat runs/anchor-XXXXXXXX/study/status.json

Outputs are created under a fresh unique directory. Existing study outputs
cannot be overwritten. A completed worker writes status.json with
status=complete. A failed worker records its exception. Start a fresh run
after fixing a failure; optimizer resume is not implemented.

Run the decisive controls alone, or run the pilot separately
------------------------------------------------------------
These are separate foreground commands. Use a new output name on every run.
The full study already includes all controls.

    python -m selective_seg.anchor_experiment pilot --output runs/anchor-pilot-01
    python -m selective_seg.anchor_experiment controls --output runs/anchor-controls-01

Experiment order and decision rules
-----------------------------------
1. Frozen baselines. Compare keep, legacy_low, legacy_high, coarse768,
   coarse1024 and anchor_high. The latter correctly centres the high view on
   the valid tile at truncated edges; legacy_high retains the old nominal tile
   geometry. For the original trained skip/low/high selector use its existing
   calibration and the same manifest, threads and machine:

    OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 python -m selective_seg.cli benchmark \
      --config configs/deepglobe_rgb.json --manifest runs/deepglobe/manifest.json \
      --run runs/deepglobe --calibration runs/deepglobe/calibration.json \
      --mode selective --split val --device cuda --passes 3 --warmup 20 \
      --probability-metrics --output runs/legacy-selective-val-anchor01

   The original calibrated utility must match the supplied config. Use the
   config/calibration from your actual run if their paths differ. The original
   benchmark has a different warmup scheme; recheck timing sensitivity.

2. Frozen window intervention. Compare wide_native (896 at native density)
   and wide_ring_control (same forward shape, real inner 640 and replicated
   outer RGB). The coarse probability ring remains real in both. Compare
   wide_resized (896 -> 640) for the fixed input-size alternative. This last
   comparison changes the centre's sampling density. Check A/R corrections
   and damage on interiors and padded images before interpreting effects.

3. Train a target-only control and a jointly trained scale-aware reader. Both
   use the same frozen high-refiner logits, head shape, initial weights, tile
   sequence and step count. In the multi-scale run one action is sampled per
   step; balanced five-step cycles exercise each action. The shared CNN
   encodes medium 768 -> 384 and large 1024 -> 256 RGB views. A full-scene
   160x160 encoder input supplies a shared, lazily populated scene feature
   cache. The scene action samples a target-centred ROI of side 2*max(H,W)
   on that lattice (zero outside the image), so even edge anchors have the
   whole landscape inside their largest box. No full-resolution canvas of
   that extent is allocated. Local RGB uses replicated boundary padding.
   Frozen high uses a native 640 view around a central 512 tile. Scale IDs,
   spatial 4x4 bins and learned source weights condition a residual correction
   to the anchor logits. Medium, large and scene share one encoder; weights
   of the existing local SegFormer stay frozen. This is a tile-conditioning
   vector, not pixel-aligned cross-attention.

   Compare trained_detail_only, dense_detail, dense_medium, dense_large,
   dense_scene, dense_all against anchor_high, wide_native and wide_resized.
   Dense models run on all tiles. Only the anchor receives a six-class mask.
   If a single fixed action dominates, report that before claiming adaptivity.

4. Generate counterfactuals on selector and validation images. Save one
   signed Brier improvement and six-class confusion per tile/action, the
   271 pre-acquisition features, image IDs and incremental time estimates.
   Keep, detail, medium, large, scene and all are six mutually exclusive
   actions. The “all” outcome is measured directly; component gains are never
   summed. Negative gains remain in training. Only selector rows fit the
   gain MLP and feature scaling. Validation hindsight results optimize the
   additive image-normalized Brier gain under measured cost estimates; they
   are neither deployable policies nor exact mIoU upper bounds.

5. Live budget comparison. For each allowance compare random-wide, entropy-
   wide, entropy-ranked fixed detail/medium/scene/all and learned tile/action
   selection. The entropy-wide arm is also the fixed large-scale comparator.
   All arms use the same candidate eligibility and incremental allowance.
   Multiple-choice knapsack charges each selected action conservatively to
   integer milliseconds. Prices are selector-split medians and include local
   refinement and cold scene setup. Actual inference reuses the scene cache.
   --candidate-fraction 1 is the default: measure headroom without prematurely
   excluding confident errors. For a capped replication use, for example:

    bash scripts/start_anchor_service.sh --candidate-fraction 0.3 --seed 42 --budgets 100 250

   The cap ranks coarse mean entropy; each positive fraction rounds up. A new
   seed is a complete independent fit. Repeat the chosen protocol at 43 and
   44. Match methods using measured mean/P95 image latency, interpolating
   only between observed operating points. An estimated allowance is not a
   hard wall-clock deadline. Compare the observed acquisition overrun rate.

6. Optional paid specialist ablation. Include the preferred frozen original
   binary specialist or a compatible hybrid checkpoint via:

    bash scripts/start_anchor_service.sh --specialist /absolute/path/to/specialist.pt --budgets 100 250

   On exactly the learned policy's selected anchors this arm applies the
   specialist after fusion, with the existing pair-mass update, cutoff .35
   and original source IDs 1/2. Other class winners are protected. Its extra
   computations are counted and timed, not included in the selector's learned
   prices. Compare at equal observed latency across budgets, as well as at
   identical selected anchors. Validate checkpoint provenance and results;
   this arm is an ablation, not evidence that the specialist adds value.

7. Locked core-policy test. Only after choosing one model, candidate cap,
   budget and seed on validation, run one test evaluation. Example:

    python -m selective_seg.anchor_experiment locked-test \
      --trained-run runs/anchor-XXXXXXXX/study --output runs/anchor-locked-01 \
      --budgets 100 --candidate-fraction 1 --passes 3 --seed 42

   Pass the identical --config/--manifest/--base-run if you overrode defaults.
   This loads the completed reader and policy and reports coarse, entropy-wide
   and learned outputs on the original test split. No selector fitting or cost
   calibration occurs. Test results have previously motivated the A/R question;
   disclose this and seek independent geographic replication before generalizing.

Reading outputs
---------------
protocol.json records config, code revision, hardware, hashes and split IDs.
reader.pt and detail_only_reader.pt hold the new weights. policy.pt contains
normalizers, output scale, cost estimates, weights and the reader's SHA-256.
Every dense/live report includes six-class mIoU, class IoUs, pixel accuracy,
NLL/Brier/ECE, risk-coverage bins, action counts, peak allocated GPU bytes,
end-to-end mean/median/P95 and per-image confusion and corrected/damaged pairs.
Rangeland is source 2, agriculture is 1. For agri->range changes compare
correction_pairs_predicted_to_true[1][2] and damage_pairs_true_to_predicted[1][2].
Report each relative to the appropriate class support. Bootstrap the difference
between the two policies you compare (hybrid_metrics.paired_bootstrap); the
saved CIs compare to coarse. Latencies cover decoded CPU input to final CPU
mask, including routing, transfers and output assembly; metrics are untimed.

Novelty boundary
----------------
Foveation for Segmentation of Ultra-High Resolution Images already learns
a location-specific field-of-view/resolution trade-off:
https://rt416.github.io/pdf/foveation_2020.pdf
GeoAgent already learns patch-specific context scale in remote sensing:
https://openaccess.thecvf.com/content/ICCV2023/papers/Liu_Seeing_Beyond_the_Patch_Scale-Adaptive_Semantic_Segmentation_of_High-resolution_Remote_ICCV_2023_paper.pdf
The experiment tests whether selective paid views add value beyond a fixed
wider window and an amortized scene view under measured budgets. Concentric
geometry alone is not a novelty claim. GPU performance and segmentation gains
must be measured on the actual workstation.
