# RUNBOOK.md — SeisMambaKAN

Operational guide. Read this when you sit down at the keyboard and need to
get something done. For *why* things are the way they are, see `CLAUDE.md`.

---

## Table of contents

1. [First-time setup](#1-first-time-setup)
2. [Daily workflow](#2-daily-workflow)
3. [Common operations](#3-common-operations)
4. [Drive layout](#4-drive-layout)
5. [Troubleshooting](#5-troubleshooting)
6. [Known issues / TODOs](#6-known-issues--todos)

---

## 1. First-time setup

### 1.1 Local IDE (one-time)

You already have Python + Git installed and the repo cloned.

```bash
git clone https://github.com/huseyinokanozturk/SeisMambaKAN.git
cd SeisMambaKAN
# Optional: local virtualenv if you want to run unit tests offline.
# python -m venv .venv && .venv/Scripts/activate    # Windows
# pip install -r requirements.txt
```

You will **not** train locally; the IDE is for editing code only.

### 1.2 Google Colab extension (one-time)

The **official Google Colab VS Code extension** is the recommended bridge.

1. VS Code → Extensions panel → search `Google Colab` → Install.
2. Command Palette (`Ctrl+Shift+P`) → `Google Colab: Sign in`.
3. Open `notebooks/Colab.ipynb` in VS Code → top-right kernel picker →
   pick the Colab runtime (A100 if available).

Once connected, the notebook runs on Colab's hardware but you edit it
in VS Code. Cells output stream back to VS Code.

### 1.3 Drive structure (one-time)

Create this folder structure on Drive, populated by you once:

```
MyDrive/Proje_SeisMamba/SeisMambaKAN/
├── data/processed/all/{train,val,test}/*.tar
├── data/processed/sample/{train,val,test}/*.tar
└── wheels/
    ├── mamba_ssm-2.2.6.post3-cp312-cp312-linux_x86_64.whl
    └── causal_conv1d-1.5.3.post1-cp312-cp312-linux_x86_64.whl
```

`experiments/` and `results/` will be created automatically by the
training/eval flows.

Wheels can be (re-)downloaded with:
```bash
pip download mamba-ssm==2.2.6.post3 causal-conv1d==1.5.3.post1 \
  --no-deps --dest /content/drive/MyDrive/Proje_SeisMamba/SeisMambaKAN/wheels \
  --python-version 3.12 --platform manylinux2014_x86_64 --only-binary=:all:
```
Verify the cp312-cp312 wheel matches the Colab Python version. If Colab
moves to Python 3.13, rebuild/redownload.

---

## 2. Daily workflow

A standard "edit -> train -> inspect" loop looks like this:

```
┌───── local IDE ─────┐                ┌──────── Colab ─────────┐
│ 1. edit code        │                │ 4. open Colab.ipynb    │
│ 2. git add / commit │ ─── push ───>  │ 5. Cell 1: bootstrap   │
│ 3. git push         │                │ 6. Cell 2: setup       │
└─────────────────────┘                │ 7. Cell 3: train       │
                                       │ 8. ... eval / infer    │
                                       └────────────────────────┘
```

1. Edit code in VS Code (local). Save.
2. `git add -p && git commit -m "..." && git push`.
3. Switch to Colab.ipynb (already open in VS Code via Google Colab extension).
4. **Run Cell 1** (bootstrap). If Colab was already running this session,
   this is a `git pull --rebase`; otherwise it's a fresh clone.
5. **Run Cell 2** (setup) — only the first time per Colab runtime,
   unless you reset the runtime.
6. **Run Cell 3** (train / eval / infer).

### Re-running with new code

If you only edited code and want to re-train:

```python
# In a notebook cell:
!cd /content/SeisMambaKAN && git pull --rebase
!python /content/SeisMambaKAN/run.py train --epochs 10
```

Setup does not need to run again unless dependencies changed.

---

## 3. Common operations

### 3.1 Train

Most common: train with config defaults.
```bash
python run.py train
```

Override per-run:
```bash
python run.py train --epochs 30 --batch-size 256 --lr 3e-4 --seed 42
python run.py train --amp --data-mode sample            # smoke test
python run.py train --data-mode all                     # production run
```

Output: `experiments/exp_XXX/{best_model.pth, checkpoints/*, logs.txt, events.out.tfevents.*}`
plus a mirror at `drive.experiments_dir/exp_XXX/`.

### 3.2 Evaluate

```bash
python run.py eval                       # latest exp, split=val
python run.py eval --exp 7 --split test  # specific exp, specific split
python run.py eval --prefer last         # use last.pth instead of best_model.pth
```

Output: `results/exp_XXX/{val,test}/metrics/*.json` (+ plots) and a Drive
mirror at `drive.results_dir/exp_XXX/`.

### 3.3 Single-trace inference + plot

```bash
python run.py infer                              # latest exp, random index
python run.py infer --exp 7 --index 42 --split test
python run.py infer --no-show --no-save          # CI-style, no GUI/file
```

Output: `results/exp_XXX/inference/{split}_idx{N}.png`.

### 3.4 TensorBoard

```bash
# Local / non-Colab:
python run.py tb                  # all experiments
python run.py tb --exp 7          # only exp_007

# Colab: paste this into a notebook cell instead of using run.py:
%load_ext tensorboard
%tensorboard --logdir experiments --port 6006
```

### 3.5 Drive sync

```bash
# Bring processed data from Drive (only needed if not done at setup):
python run.py data --mode all
python run.py data --mode sample --refresh    # wipe + re-copy

# Push an experiment + its results to Drive:
python run.py push --exp 7
python run.py push --exp 7 --no-results       # checkpoints only

# Pull an experiment back to local Colab:
python run.py pull-exp --exp 7
```

### 3.6 Project state

```bash
python run.py status
```
Shows: env, training config, present data shards, recent experiments,
latest checkpoint path.

### 3.7 Resetting things

```bash
make clean-data       # remove data/processed
make clean-exp        # remove experiments/, results/
```

Then re-pull from Drive: `python run.py data --mode all`.

---

## 4. Drive layout

This is the canonical layout. The code expects it to be exactly this
(paths configurable in `configs/paths.yaml`).

```
/content/drive/MyDrive/Proje_SeisMamba/SeisMambaKAN/
│
├── data/processed/
│   ├── all/
│   │   ├── train/train_000000.tar ... train_NNNNNN.tar
│   │   ├── val/val_000000.tar ...
│   │   └── test/test_000000.tar ...
│   └── sample/    (same structure, smaller)
│
├── experiments/
│   └── exp_XXX/
│       ├── best_model.pth          # state_dict only
│       ├── config_used.yaml        # snapshot of all configs at training time
│       ├── logs.txt                # human-readable training log
│       ├── events.out.tfevents.*   # TensorBoard
│       └── checkpoints/
│           ├── last.pth            # full state (model + optim + epoch)
│           └── checkpoint_epoch_NNN.pth
│
├── results/
│   └── exp_XXX/
│       ├── val/metrics/val_metrics.json
│       ├── val/metrics/val_per_trace.csv    (if save_per_trace_csv)
│       ├── test/metrics/test_metrics.json
│       └── inference/test_idx42.png ...
│
└── wheels/
    ├── mamba_ssm-2.2.6.post3-cp312-cp312-linux_x86_64.whl
    └── causal_conv1d-1.5.3.post1-cp312-cp312-linux_x86_64.whl
```

---

## 5. Troubleshooting

### "No experiments found" when running `eval` or `infer`
You have not trained yet, or the experiments dir is empty. Either run
`python run.py train` first, or `python run.py pull-exp --exp 7` to
fetch one from Drive.

### `mamba_ssm` import fails on Colab
1. Confirm the Colab Python is 3.12 (`!python -V`). Wheels are
   cp312-cp312; a 3.13 runtime will not find them.
2. Confirm `drive.wheels_dir` (Drive) contains the two wheels named in
   `configs/paths.yaml -> colab`.
3. Re-run `python run.py setup --skip-data --skip-requirements` to
   reinstall just torch + wheels.

### CUDA OOM during training
- Lower batch size: `python run.py train --batch-size 128`.
- Switch to `--data-mode sample` to confirm code, then go back to `all`.
- Check `dataloader.num_workers` in `config.yaml`; on Colab 4-8 is usually
  the sweet spot.

### Loss is NaN / inf
- This is a known stability risk (no gradient clipping yet — see
  TODO #1 below). Reduce LR with `--lr 1e-4` as a first probe.
- If it persists, switch `--amp` off (`--no-amp`).

### "Drive not mounted" warning
The script ran outside Colab. `setup` / `data` / `push` are Colab-only.
Use `python run.py status` to confirm whether Drive is mounted.

### `git pull` conflicts in Colab
The bootstrap cell stashes before pulling. If a conflict occurs, the
stashed changes remain in `git stash list` — `git stash pop` after
resolving. For permanent edits, push from local instead of editing inside
Colab.

### Eval / inference reads the wrong checkpoint
`run.py eval` auto-picks the **highest-numbered** `exp_XXX`. To force a
different one, pass `--exp N` or `--ckpt PATH`. Verify with
`python run.py status`.

---

## 6. Known issues / TODOs

(Mirrors the diagnosis report; intentionally tracked here so it survives
across chats.)

1. **No gradient clipping** in `src/trainer.py`. Mamba can produce large
   early gradients; this is likely the root cause of "loss explodes around
   epoch 3-4". Fix: insert `clip_grad_norm_` between `scaler.unscale_`
   and `scaler.step`.
2. **No LR scheduler / warmup**. Constant 5e-4 LR is too hot for
   Mamba+KAN. Add `OneCycleLR(max_lr=3e-4, pct_start=0.1)` or warmup +
   cosine.
3. **Detection head outputs probability** (`sigmoid`), and the loss uses
   `F.binary_cross_entropy` (not BCE-with-logits). Refactor: drop the
   final sigmoid, use `BCEWithLogitsLoss`.
4. **Phase loss collapses on pure-noise traces** (mask sum ~= 0).
   Currently falls back to `sq_error.mean()`. Should drop those traces
   per-sample instead of per-timestep.
5. **`use_focal` in the detection loss** lacks the `(1-alpha)` term for
   the negative class; only the positive side is scaled. Minor, but if
   focal gets re-enabled, verify.
6. **Architecture is under-parameterized.** Stem channels = 16, peak
   bottleneck channels = 64. Likely capacity-limited. Raise to
   `32 -> 48 -> 96 -> 128 -> 192 -> 256` and re-benchmark.
7. **KAN in every decoder stage** is expensive. Try KAN only at
   `dec_stage_0` + a head-side KAN; ConvNormAct in the rest.

These will be addressed in a separate "recovery patch" branch once the
pipeline (this RUNBOOK) is verified to work end-to-end.
