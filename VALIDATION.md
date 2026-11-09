# Research extension verification

- `python -m pytest -q`: 21 tests passed on CPU (including random-initialized SegFormer-B0 forward/backward).
- `python smoke_test.py`: original CLI training/cache/calibration/evaluation workflow passed.
- `python smoke_research.py`: scout and acceptance training, versioned effective caches, image-grouped metric selector, escalation training, calibration, benchmark profiling and all baseline modes passed on synthetic images.
- Auxiliary training/selector overlap rejection is exercised end to end.
- `git diff --check` and Python compilation passed.
- Runtime: Python 3.12, PyTorch 2.15.0+cpu, Transformers 4.57.1.

No real DeepGlobe data/checkpoints or CUDA device were available. These checks establish execution and selected invariants, **not** mIoU improvements, statistical significance, GPU speedup, complete FLOP accounting, or research novelty. See RESEARCH.md for limitations and the real-data evaluation protocol.

---

## Original project validation record

# Validation performed

Validated locally on 28 October 2026 using Python 3.12, PyTorch 2.14.1 and
Transformers 4.57.1 on macOS CPU. No NVIDIA GPU was available in this environment.

- `python -m pytest -q`: **10 passed**.
- Real, randomly initialized SegFormer-B0 encoder/decoder: forward and backward
  pass with coarse-context conditioning passed.
- Synthetic full CLI workflow: data validation, coarse training, low training,
  high training, selector/validation target generation, selector training,
  threshold calibration and test-image inference passed.
- Both low and high routing branches were exercised independently, including
  partial tiles at image edges. Skipped predictions preserve coarse labels.
- Optional 8-band learned pseudo-color projection: gradient checks passed for
  the class predictor, class mixture bank and patch-conditioned mixing network.
- Inria 0/255 label remapping and image-path split-overlap checks passed.
- All-ignore target handling and exclusion from selector fitting passed.
- Synthetic validation contained only one positive-benefit label category;
  the calibration command correctly reported that temperature fitting was skipped.

These checks establish executable code paths and key numerical invariants.
They do not establish DeepGlobe/Inria accuracy, convergence, novelty, calibration
under city shift, CUDA behavior, real 24 GB GPU memory usage, or runtime advantage.
Pretrained checkpoint downloading was not exercised; the real backbone test
used local random initialization. Synthetic weights and scores are not shipped
as trained research results.

Run the included tests and a short CUDA diagnostic on your training machine
before starting the full iteration schedule.
