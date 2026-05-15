# SeisMambaKAN

A novel hybrid deep-learning model combining **Mamba** (state-space models) and
**KAN** (Kolmogorov-Arnold Networks) for earthquake **detection** and seismic
**P / S phase picking** on the STEAD dataset.

> Bitirme projesi — Hüseyin Okan Öztürk

---

## Quickstart (Colab)

```
1. Open notebooks/Colab.ipynb (button below).
2. Run cells [1] -> [2]  (Drive mount, clone, install, sync data)
3. Run cell  [3]         (train / eval / infer)
```

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/huseyinokanozturk/SeisMambaKAN/blob/main/notebooks/Colab.ipynb)

## Single-command interface (`run.py`)

```bash
# One-time Colab bootstrap (mount Drive, clone, install, copy data)
python run.py setup --data-mode all

# Sync processed data from Drive (idempotent)
python run.py data --mode all [--refresh]

# Train with CLI overrides
python run.py train --epochs 30 --batch-size 256 --lr 3e-4

# Evaluate latest experiment on validation set
python run.py eval --split val

# Single-trace inference + plot
python run.py infer --split test --index 42

# TensorBoard
python run.py tb

# Mirror an experiment + its results to Drive
python run.py push --exp 7

# Pull an experiment from Drive
python run.py pull-exp --exp 7

# Project state snapshot (env, data shards, experiments)
python run.py status
```

`make` shortcuts are also available — see `Makefile`.

## Where things live

| Where | What |
|---|---|
| GitHub (this repo) | All source code + configs + notebook |
| Drive: `MyDrive/Proje_SeisMamba/SeisMambaKAN/data/processed/{all,sample}/` | Processed WebDataset shards |
| Drive: `.../experiments/exp_XXX/` | Training checkpoints, logs, TB events (mirrored from Colab) |
| Drive: `.../results/exp_XXX/` | Eval metrics, inference plots (mirrored from Colab) |
| Drive: `.../wheels/` | Pre-built `mamba_ssm` + `causal_conv1d` wheels |

Local repo never contains data, checkpoints, or results (see `.gitignore`).

## Configuration

Three YAMLs in `configs/`:

- `config.yaml` — data / loss / augmentation / training hyperparams
- `model_config.yaml` — architecture (encoder, decoder, heads)
- `paths.yaml` — file-system paths (Drive, Colab, local). **No code hardcodes paths.**

CLI flags to `run.py train` override values in `config.yaml` for that one run.

## Architecture (brief)

```
input (B, 3, 6000)
        │
        ▼
   Stem Conv1d
        │
        ▼  4 encoder stages, each: DownSample -> Mamba blocks
   Bottleneck (Mamba)
        │
        ▼  4 decoder stages, each: UpSample -> Skip -> KAN block
   Heads: detection (sigmoid), p_gaussian, s_gaussian
```

See `src/models/network.py` and `src/models/blocks.py`.

## Repository layout

```
SeisMambaKAN/
├── run.py                  # top-level CLI (typer)
├── Makefile                # short-hand wrappers around run.py
├── colab_setup.py          # Colab bootstrap functions
├── train.py / evaluate.py / inference.py
├── notebooks/
│   └── Colab.ipynb         # 3-cell production notebook
├── configs/
│   ├── config.yaml
│   ├── model_config.yaml
│   └── paths.yaml          # ← single source of truth for all paths
├── scripts/
│   ├── preprocess.py       # STEAD HDF5 → WebDataset
│   └── sync_drive.py       # Drive <-> Colab sync
├── src/
│   ├── dataset.py
│   ├── losses.py
│   ├── metrics.py
│   ├── trainer.py
│   ├── utils.py            # env + paths + experiment lookup
│   └── models/
│       ├── network.py
│       └── blocks.py
├── data/                   # gitignored (populated by `run.py data`)
├── experiments/            # gitignored (populated by training)
└── results/                # gitignored (populated by eval/infer)
```

## Further reading

- `CLAUDE.md` — project conventions and rules (for AI assistants and humans)
- `RUNBOOK.md` — operational runbook (first-time setup, daily workflows, troubleshooting)
