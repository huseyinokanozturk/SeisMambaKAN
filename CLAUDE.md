# CLAUDE.md — SeisMambaKAN project rules

This file is the **single source of truth** for how the project is structured
and how work in it should be done. It is meant for AI coding assistants
(Claude / Cursor / Copilot) *and* humans returning to the codebase after a
break.

When in doubt about a path, naming, or convention, read this file first.

---

## 1. What this project is

- A research codebase for a thesis: hybrid Mamba (SSM) + KAN (Kolmogorov-
  Arnold Networks) deep model for earthquake detection + seismic P/S
  phase picking on the STEAD dataset.
- Three model outputs per trace:
  - `detection`  : (B, T) sigmoid probability over time
  - `p_gaussian` : (B, T) Gaussian-shaped peak around P arrival
  - `s_gaussian` : (B, T) Gaussian-shaped peak around S arrival
- Baseline being compared against: a separate **SeisConformer** project
  (TensorFlow, 1D-CNN + BiLSTM + Conformer + attention U-Net decoders),
  built by the same author. SeisConformer reaches SOTA on STEAD
  (F1 0.9999, P MAE 19 ms, S MAE 85 ms) — *better* than published
  EQTransformer numbers on the same test set.
- **SeisMambaKAN target is SeisConformer parity, not "thesis-defending
  acceptable".** Phase 7 (see `PHASE_7.md`) is the active recipe
  rewrite to close the gap. The contribution is the Mamba+KAN hybrid
  *architecture*; the metrics must at minimum match the SC baseline,
  otherwise the architecture cannot be defended.

## 2. The three environments

| Name      | Used for           | Where code runs   |
|-----------|--------------------|-------------------|
| `local`   | Editing code       | Windows / IDE     |
| `colab`   | Training & eval    | Google Colab A100 |
| `drive`   | Storage backend    | Google Drive (shared between local + colab) |

Workflow:
1. Edit code in `local`, push to GitHub.
2. In `colab`: pull GitHub repo, sync processed data from `drive`, train.
3. Training mirrors checkpoints/logs to `drive` automatically.
4. Inspect results: pull experiment from `drive` to `local` if needed.

**Data and experiments never live in Git.** The repo only contains source,
configs, and small assets. `.gitignore` enforces this; do not weaken it.

## 3. Hard rules

### 3.1 Paths
- Code **never** hardcodes a filesystem path. All paths come from
  `configs/paths.yaml` and are resolved via `src/utils.py:project_root()`.
- Drive paths live under the `drive:` section of `paths.yaml`.
- Colab runtime paths live under `colab:` section.
- If you need a new path, add it to `paths.yaml`; do not inline it.

### 3.2 Entry points
- **Users do not call `train.py` / `evaluate.py` / `inference.py` directly.**
  They call `python run.py <command>`. Those three files remain as thin
  CLI-compatible modules (importable via `from evaluate import main`) but
  the canonical UX is `run.py`.
- Adding a new operation = adding a new Typer command in `run.py` and the
  underlying helper in `src/` or `scripts/`.

### 3.3 Configs
- Three YAML files live in `configs/`:
  - `config.yaml`       : data, loss, augmentation, training hyperparams
  - `model_config.yaml` : architecture
  - `paths.yaml`        : filesystem paths (only)
- Override values from the command line, not by editing YAML for a single
  run. `run.py train` accepts `--epochs`, `--batch-size`, `--lr`, `--seed`,
  `--data-mode`, `--amp/--no-amp`. Add more via
  `src/utils.py:apply_dotted_overrides`.

### 3.4 Experiment naming
- Experiments are `experiments/exp_XXX/` where XXX is a zero-padded
  3-digit incrementing index, created by `src/trainer.py:prepare_experiment_dirs`.
- `best_model.pth` lives at `experiments/exp_XXX/best_model.pth` and
  contains *only* the model `state_dict`.
- Full state (model + optimizer + epoch) lives at
  `experiments/exp_XXX/checkpoints/{last.pth, checkpoint_epoch_NNN.pth}`.

### 3.5 Results
- Eval and inference outputs go to `results/exp_XXX/{val,test,inference}/`.
- They are **also** mirrored to `drive.results_dir/exp_XXX/` if Drive is
  mounted (controlled per command via `--no-drive-mirror`).
- Never put eval/inference output inside `experiments/` — that directory
  is for training artifacts only.

### 3.6 Drive mirroring
- Trainer (`src/trainer.py`) mirrors checkpoints + logs to
  `paths.yaml -> experiments.drive_root_dir` *during* training.
- Evaluator and inference scripts mirror their outputs to
  `paths.yaml -> results.drive_root_dir` at the end of their run.
- The mirror is one-way (Colab -> Drive). To pull back, use
  `run.py pull-exp --exp N`.

### 3.7 Coding conventions
- Python ≥ 3.10 (Colab default).
- Use `pathlib.Path` over `os.path`.
- Type hints on public functions; `from __future__ import annotations`
  on new modules.
- New tests go under `tests/` and mirror the source layout. Tests should
  not depend on Drive being mounted or real STEAD data.
- Logging: `print(...)` is fine for CLI/Colab output. For console output
  inside `run.py`, prefer the `_ok`, `_warn`, `_err`, `_bullet` helpers.

### 3.8 Dependencies
- Mamba + causal_conv1d are installed from **upstream GitHub release
  wheels**, auto-matched to the currently installed torch + python +
  CUDA. The URL is constructed at setup time by
  `colab_setup.py:install_mamba_official_wheels` from
  `paths.yaml -> colab.{mamba_version,causal_version,wheel_cxx11_abi}`.
  Do NOT replace this with plain `pip install mamba-ssm`; that builds
  from source on Colab and takes 15-30 min, often failing.
- `paths.yaml -> colab.{mamba_wheel_name,causal_wheel_name}` (legacy
  Drive-wheel fallback) is optional. Leave it as `null` unless you have
  to operate offline.
- If you bump `target_torch_version`, also bump `mamba_version` /
  `causal_version` to a release that publishes a wheel for the new
  torch major.minor. Check the available wheels with:
  `curl -s https://api.github.com/repos/state-spaces/mamba/releases/tags/v<VER>`
- Adding a new dependency: append to `requirements.txt`, and verify on
  Colab end-to-end before merging.

### 3.9 Commits and pushes
- Project commits do not include an AI co-author trailer. (`huseyinokanozturk` is the sole author.)
- Prefer small, focused commits with messages like
  `evaluate: auto-pick latest experiment when --exp omitted`.

## 4. Architectural decisions worth remembering

- **Normalization is applied once**, at preprocess time
  (`scripts/preprocess.py:normalize_waveform`). `dataset.py` does *not*
  re-normalize. Do not introduce a second normalization step.
- **Loss masking**: phase (P/S) loss is multiplied by `det_target` and
  normalized by `mask.sum()` (`src/losses.py`). For pure-noise traces
  this currently falls back to `sq_error.mean()`; see TODO #1 in
  `RUNBOOK.md`.
- **AMP**: off by default. Mamba is unstable under fp16; if AMP is
  enabled, prefer bf16 on A100. See `model_config.yaml -> model.use_amp`.
- **Gradient clipping**: currently *not* implemented in `trainer.py`.
  This is a known stability risk and is a target of the upcoming
  recovery patch.

## 5. Out-of-scope / forbidden

- Do not commit:
  - `data/` (raw or processed)
  - `experiments/`, `results/`
  - `*.tar`, `*.npy`, `*.h5`, `*.hdf5`, `*.csv`, `*.pth`
- Do not edit STEAD data files. If a preprocessing parameter needs to
  change, re-run `scripts/preprocess.py` and overwrite the WebDataset
  shards on Drive.
- Do not introduce a new orchestration framework (Hydra, Lightning,
  Accelerate, …) for this thesis. Plain PyTorch + Typer is the chosen
  stack. The user has limited remaining time and wants minimal moving
  parts.

## 6. When something is unclear

- Read `PHASE_7.md` for the current recipe / target answer.
- Read `RUNBOOK.md` for the operational answer.
- Read `configs/paths.yaml` for the where-things-go answer.
- Read `src/trainer.py` for the training-flow answer.
- If the answer is still not there, **ask** rather than guess. The user
  values correctness over speed.
