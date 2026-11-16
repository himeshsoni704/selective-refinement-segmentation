# Run the agriculture–rangeland specialist

These commands train the independent combined ResNet-18 + EfficientNet-B0
specialist added in `70081bc`. They do not train an enhanced selector yet.
Use a Linux machine with your dataset, a working CUDA-enabled PyTorch/torchvision
pair, and your existing Python environment activated. Replace `data/manifest.json`
and `runs/deepglobe` with your actual manifest and coarse-checkpoint directory.
Run commands from the repository root.

## 1. Pull, install and check

```bash
cd /path/to/selective-refinement-segmentation
git pull --ff-only origin main
python -m pip install -e '.[test,specialist]'
python -c 'import torch, torchvision; print("torch:", torch.__version__, "torchvision:", torchvision.__version__); assert torch.cuda.is_available(), "CUDA unavailable in this Python environment"; print(torch.cuda.get_device_name(0))'
python -m selective_seg.cli validate-data \
  --config configs/deepglobe_specialist.json \
  --manifest data/manifest.json
```

The manifest needs separate `train` and `val` images; use geographically separate
`test` images for final evaluation. Use the same class mapping and image splits as
your existing coarse experiment. Initial training downloads ImageNet backbone
weights. This is a new specialist run: existing coarse/low/high weights need not
be retrained. Installing the extra does not by itself guarantee a CUDA build.

## 2. Train in the terminal

Choose this **or** the background service below, not both for the same run.

```bash
python -u -m selective_seg.cli train-specialist \
  --config configs/deepglobe_specialist.json \
  --manifest data/manifest.json \
  --run runs/specialist_combined \
  --device cuda
```

Default: 10,000 steps, full-validation evaluation every 500 steps, input size 320,
shallow+deep features from both backbones. First step and every 100th step log loss;
full validation can take time without printing intermediate progress.

Optional quick workflow check **in a separate directory**, before the full run:

```bash
python -u -m selective_seg.cli train-specialist \
  --config configs/deepglobe_specialist.json \
  --manifest data/manifest.json \
  --run runs/specialist_smoke \
  --device cuda --steps 2
```

Even two steps finish with full validation; this checks execution, not accuracy.
Use a fresh run directory if `specialist.pt` already exists. This trainer saves
best-validation weights but does not support optimizer/RNG resume.

## 3. Standalone systemd background command

Run this from the repository root with the desired Python environment active.
It uses the current interpreter's absolute path, so activating Conda/venv inside
the service is unnecessary. Set the manifest/run paths below if yours differ.
This requires a Linux host with systemd and a functioning user session; ordinary
Colab/Kaggle notebooks and containers without systemd cannot use it.

```bash
sudo loginctl enable-linger "$(id -un)" && \
systemd-run --user \
  --unit=specialist-train \
  --description="Agriculture-rangeland specialist training" \
  --property=Type=exec \
  --property=Restart=no \
  --working-directory="$PWD" \
  --setenv="PATH=$PATH" \
  --setenv="LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-}" \
  --setenv=PYTHONUNBUFFERED=1 \
  "$(python -c 'import sys; print(sys.executable)')" \
  -u -m selective_seg.cli train-specialist \
  --config configs/deepglobe_specialist.json \
  --manifest data/manifest.json \
  --run runs/specialist_combined \
  --device cuda
```

`enable-linger` keeps your user manager running after logout. Only that setup
command uses sudo; training runs as your own user. If lingering is already enabled,
the `systemd-run` command can be run on its own. This is a transient service:
it survives terminal closure/logout, **not machine shutdown/reboot**. Restart is
disabled because automatic restarts cannot resume this trainer's optimizer state.
Only PATH/LD_LIBRARY_PATH are forwarded here; if your setup relies on custom
CUDA_VISIBLE_DEVICES, HF_HOME or proxy variables, add matching `--setenv` options.

Watch training (Ctrl+C exits the log viewer; training continues):

```bash
journalctl --user -u specialist-train.service -f -o cat
```

Check status and recent logs:

```bash
systemctl --user status specialist-train.service --no-pager
journalctl --user -u specialist-train.service -n 100 --no-pager
```

Stop training deliberately:

```bash
systemctl --user stop specialist-train.service
```

After successful completion, a transient unit may unload; check journal logs and
`runs/specialist_combined/specialist.pt` for the result. If a failed unit name
remains registered, clear it before submitting a fresh run:

```bash
systemctl --user reset-failed specialist-train.service
```

Use a new run path (for example `runs/specialist_combined_v2`) for a new training
attempt if a checkpoint exists. Do not delete an existing trained checkpoint just
to bypass the guard. A stopped/interrupted run can contain an earlier best model;
its checkpoint `step` indicates when that model was saved, not completion of all
10,000 steps. The journal records training progress and process exit.

## 4. Evaluate after training completes

Compare with your existing coarse model. `runs/deepglobe/coarse.pt` must exist and
match the base data/geometry configuration:

```bash
python -m selective_seg.cli evaluate-specialist \
  --config configs/deepglobe_specialist.json \
  --manifest data/manifest.json \
  --run runs/specialist_combined \
  --coarse-run runs/deepglobe \
  --split val --device cuda
```

If you only want specialist metrics, omit `--coarse-run runs/deepglobe`.
After fixing all experiment choices, use `--split test` for the untouched test set.
Do not repeatedly tune from test results.

## 5. Read the output

```bash
python -m json.tool runs/specialist_combined/specialist_val.json
```

- `specialist`: binary confusion matrix, agriculture/rangeland IoUs, mIoU, accuracy.
- `coarse_conditional_binary`: coarse A/R choice on exactly the same true A/R pixels.
- `coarse_six_class_on_AR_pixels`: original coarse six-class errors on those pixels.
- `diagnostic_correctable_pixels`, `diagnostic_damage_pixels`: possible corrections
  and new mistakes, **not actual changes to the final segmentation**.
- `coarse_confident_errors`, `specialist_correct_on_confident_errors`: evidence for
  the future error-risk candidate pathway.
- `NLL`, `Brier`, `ECE`: uncalibrated confidence diagnostics.
- `mean_specialist_image_ms`, `median_crop_ms`, `p95_crop_ms`, `peak_gpu_bytes`:
  initial cost screen; not a repeated full-pipeline speed benchmark.

See [SPECIALIST.md](SPECIALIST.md) for architecture, ablations, limitations and
planned selector integration. Send the evaluation JSON when reviewing whether
specialist evidence is useful enough to proceed.

Systemd references: [systemd-run](https://freedesktop.org/software/systemd/man/latest/systemd-run.html),
[loginctl lingering](https://freedesktop.org/software/systemd/man/latest/loginctl.html).

Verification: fenced Bash command blocks were syntax-checked. No service was
started in the development container; execution requires your Linux host,
dataset, active Python environment and GPU.
