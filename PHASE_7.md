# Phase 7 — Full SeisConformer Recipe Alignment

**Status:** in progress (implementation underway 2026-05-17).
**Goal:** Reach SeisConformer-parity picking metrics on STEAD test set:
P MAE ≤ 25 ms, S MAE ≤ 90 ms, trace F1 ≥ 0.999, P/S hit-rate F1 ≥ 0.98.
This is the SOTA bar — relaxed "thesis-defending" framings are
explicitly rejected (see `memory/feedback_target_is_sota.md`).

## Why Phase 7

`exp_002` (the latest 5-epoch run from Phases 0–5) reached:

| metric | exp_002 | SeisConformer (user's own) | EQT paper |
|--------|---------|-----------------------------|-----------|
| trace F1 | 0.994 | 0.9999 | 0.9998 |
| P MAE | 107 ms | 19 ms | 10 ms |
| S MAE | 145 ms | 85 ms | 10 ms |

The picker has been iterated 3 times (v1 → v3 in Phase 5). v3 matches
the original union picker; sweep tuning only nibbles at outliers. The
binding constraint is no longer the picker — it is the *training-side
recipe* (labels, loss weighting, augmentation tightness, training
length). Phase 7 rewrites that recipe to match the SeisConformer one
the user already proved is SOTA on this dataset, with the surgical
upgrades verified against EQTransformer's published Table 2/3.

## Change list

All 14 changes go in **one cohesive commit** (`phase-7: full
SeisConformer alignment`). Training is from scratch — no resume from
`exp_002` since labels and loss change.

| # | File | Field | From (Phase 5) | To (Phase 7) |
|---|------|-------|----------------|---------------|
| 1 | `src/dataset.py` (`make_detection_label`) | Detection label | Binary [P, S] (Gaussian-margin) | **Binary box [P-50, S+50]** — same as SC `_make_labels.make_detection_trace` |
| 2 | `configs/config.yaml` (`loss.phase`) | Peak weight | shared `peak_weight_scale: 5.0` | **`p_peak_weight: 100.0`, `s_peak_weight: 150.0`** (SC exact ratio) |
| 3 | `configs/model_config.yaml` (`heads`) | P/S head activation | `"none"` (linear) | **`"sigmoid"`** (clamp becomes redundant) |
| 4 | `configs/config.yaml` (`augmentation.random_shift`) | Max shift | 3000 samples (30 s) | **30 samples (0.3 s)** — SC level; reverses Phase 2.1 |
| 5 | `configs/config.yaml` (`augmentation.additive_noise`) | Noise std | `std_range: [0.05, 0.30]` | **`std: 0.05`** (fixed, no range) |
| 6 | `configs/config.yaml` (`augmentation.amplitude_scale`) | Scale range | `[0.5, 2.0]` | **`[0.9, 1.1]`** |
| 7 | `configs/config.yaml` (`augmentation.channel_dropout`) | Drop prob | 0.2 | **0.1** |
| 8 | `configs/config.yaml` (`loss.weights`) | Weights | det 0.1 / p 0.3 / s 0.4 | **det 0.4 / p 1.0 / s 1.2** (SC exact) |
| 9 | `src/metrics.py:pick_phases` | Picker | Union ∩ peak-bound + parabolic | **Multi-criteria + Gaussian refinement** (ported from SC `picker.py` V7) |
| 10 | `src/metrics.py:compute_phase_metrics` | Reported metrics | num_gt, num_pred, pick_rate, mae, medae, std, hit_rates | **+ mean_sec (μ), MAPE, tp_pick/fp_pick/fn_pick, precision_pick/recall_pick/f1_pick** |
| 11 | `configs/config.yaml` (`metrics.phase_tolerance`) | Pick tolerance | (missing) | **`pick_tolerance_sec: 0.5`** — EQT standard |
| 12 | `SeisMambaKAN/PHASE_7.md` | Tracking | — | This file |
| 13 | `SeisMambaKAN/CLAUDE.md` | Baseline target | "SC parity allowed lower" | **"SC parity is the target, not the ceiling"** |
| 14 | Training run | Epochs | 5 | **25+** with patience=12 (full convergence) |

Phase 2.5 smart-shift clamp can stay as a no-op safety net under
shift=30 (it never triggers because lo<hi always holds in the
6000-sample window with a 30-sample shift). Architecture (Phase 3),
training loop (Phase 4), trainer mirroring, EMA, OneCycleLR all carry
over unchanged.

## Acceptance gates

### Smoke gate (5-epoch sanity check before committing to a 25-epoch run)
- `val_total < 0.005` AND `val_p < 0.002`
- Validation loss strictly decreasing epoch-over-epoch
- Trace F1 ≥ 0.99 already at epoch 5

If these fail, halt and diagnose — do not burn 25 epochs on a broken
recipe.

### Full-run gate (test set, after 25-epoch convergence)
- **Trace F1 ≥ 0.999**
- **P:** MAE ≤ 30 ms mean, hit_rate_small ≥ 0.40, f1_pick ≥ 0.98
- **S:** MAE ≤ 100 ms mean, hit_rate_small ≥ 0.20, f1_pick ≥ 0.95

Hitting full-run gate ≈ SC parity. Falling short by < 20% → extend to
35–40 epochs (within the 1000 TL monthly budget; see `Cost projection`).

### Stretch gate (only if full-run lands within 20% of SC)
- P MAE ≤ 19 ms, S MAE ≤ 85 ms (SC exact)

## Cost projection

- Smoke 5-epoch ≈ 75 TL
- Full 25-epoch ≈ 375 TL
- Stretch +10 epochs ≈ 150 TL
- **Total Phase 7 budget: ~600 TL.** Inside the 1000 TL monthly margin.

## Implementation order (single commit at the end)

1. `PHASE_7.md` (this file) — tracking
2. `CLAUDE.md` — target update
3. `configs/config.yaml` — all six fields (shift, noise, amp,
   channel_dropout, loss.weights, loss.phase split, pick_tolerance_sec)
4. `configs/model_config.yaml` — P/S sigmoid
5. `src/dataset.py` — Box detection label
6. `src/losses.py` — split phase peak weights into P/S; carry phase_name
7. `src/metrics.py` — picker rewrite (SC-port) + metric additions
8. Manual review pass — diff each file, run `python run.py status`
9. Single commit, push to GitHub
10. Colab: pull repo, run smoke (5 epochs), then full 25-epoch run.

## Quick-resume protocol

If session is interrupted before training:
1. `cd SeisMambaKAN && python run.py status` to confirm working tree
2. Check the change list table above — each row is one diff
3. The single commit must contain all 14 changes; do not push partial.

If session is interrupted during training:
1. `python run.py status` to find current `exp_NNN/`
2. `tail experiments/exp_NNN/training_log.csv` for last completed epoch
3. Resume via `python run.py train --resume experiments/exp_NNN/checkpoints/last.pth`
   (the trainer's resume path is unaffected by Phase 7 since model arch
   is unchanged — only label/loss/aug differ, and those are
   data-loader-side).
