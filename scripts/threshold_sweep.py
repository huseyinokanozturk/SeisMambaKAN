"""
Threshold sweep for picker hyperparameters (Phase 5.1).

Workflow:
1. Run the model once on the chosen split (val by default), cache raw head
   outputs in RAM (~5 GB for full STEAD val, fits in Colab A100 RAM).
2. Grid-search over (det_window_threshold, p_amp_threshold, s_amp_threshold,
   trace_threshold). For each combination, recompute picks + summary metrics
   without re-running the model.
3. Print the top-N cells by a composite score (we minimize P/S median errors
   while keeping trace F1 high) and save the full grid as JSON next to
   the eval results.

Typical Colab use:
    python scripts/threshold_sweep.py --split val
    python scripts/threshold_sweep.py --split val --exp 2
"""

from __future__ import annotations

import argparse
import copy
import itertools
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch

# Allow `python scripts/threshold_sweep.py` from project root.
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluate import (
    load_all_configs,
    resolve_experiment_dir,
    resolve_checkpoint,
    _infer_exp_id_from_path,
    _safe_torch_load,
)
from src.dataset import build_dataloader
from src.metrics import (
    pick_phases,
    _extract_heads_from_outputs,
    _extract_label_curves,
)
from src.models.network import SeisMambaKAN
from src.utils import project_root


# -----------------------------------------------------------------------------
# Cache stage
# -----------------------------------------------------------------------------


def _cache_predictions(
    model: torch.nn.Module,
    loader,
    device: torch.device,
    max_batches: Optional[int] = None,
) -> Dict[str, np.ndarray]:
    """
    Run the model on the full loader and store everything the picker + metric
    layer needs into numpy arrays. Float16 keeps memory manageable for ~125K
    val samples (~3 GB instead of ~6 GB at fp32).
    """
    det_preds: List[np.ndarray] = []
    p_preds: List[np.ndarray] = []
    s_preds: List[np.ndarray] = []

    gt_event: List[int] = []
    p_idx_true: List[int] = []
    s_idx_true: List[int] = []

    model.eval()
    device_type = device.type

    n_samples = 0
    t0 = time.time()

    with torch.no_grad():
        for batch_idx, (x, labels) in enumerate(loader):
            if max_batches is not None and batch_idx >= int(max_batches):
                break
            x = x.to(device, non_blocking=True)
            if x.dim() == 3:
                x_model = x.permute(0, 2, 1).contiguous()
            else:
                x_model = x

            with torch.amp.autocast(device_type=device_type, enabled=False):
                outputs = model(x_model)
            det_out, p_out, s_out = _extract_heads_from_outputs(outputs)

            det_preds.append(det_out.detach().cpu().numpy().astype(np.float16))
            p_preds.append(p_out.detach().cpu().numpy().astype(np.float16))
            s_preds.append(s_out.detach().cpu().numpy().astype(np.float16))

            det_true, p_gauss_true, s_gauss_true, p_idx_t, s_idx_t = (
                _extract_label_curves(labels)
            )
            B = det_true.shape[0]
            for i in range(B):
                gt_flag = int(
                    (det_true[i, 0].max() >= 0.5)
                    or (p_gauss_true[i, 0].max() > 0.0)
                    or (s_gauss_true[i, 0].max() > 0.0)
                )
                gt_event.append(gt_flag)

                if p_idx_t is not None:
                    p_idx_true.append(int(p_idx_t[i].item()))
                else:
                    p_idx_true.append(int(np.argmax(p_gauss_true[i, 0])) if gt_flag else -1)
                if s_idx_t is not None:
                    s_idx_true.append(int(s_idx_t[i].item()))
                else:
                    s_idx_true.append(int(np.argmax(s_gauss_true[i, 0])) if gt_flag else -1)
            n_samples += B
            if (batch_idx + 1) % 50 == 0:
                elapsed = time.time() - t0
                print(f"  cached {n_samples} samples in {elapsed:.1f}s")

    # Model heads return (B, T) — already 2D, no channel dim to squeeze.
    cache = {
        "det_pred": np.concatenate(det_preds, axis=0),  # (N, T)
        "p_pred": np.concatenate(p_preds, axis=0),
        "s_pred": np.concatenate(s_preds, axis=0),
        "gt_event": np.asarray(gt_event, dtype=np.int8),
        "p_idx_true": np.asarray(p_idx_true, dtype=np.int32),
        "s_idx_true": np.asarray(s_idx_true, dtype=np.int32),
    }
    print(
        f"[sweep] cached {n_samples} samples "
        f"(det.shape={cache['det_pred'].shape}, dtype={cache['det_pred'].dtype})"
    )
    return cache


# -----------------------------------------------------------------------------
# Per-combo evaluation
# -----------------------------------------------------------------------------


def _evaluate_combo(
    cache: Dict[str, np.ndarray],
    base_picker_cfg: Dict[str, Any],
    overrides: Dict[str, float],
    trace_threshold: float,
    sample_rate: float,
    small_tol: float = 0.01,
    medium_tol: float = 0.02,
    large_tol: float = 0.05,
) -> Dict[str, Any]:
    """Score a single threshold combination using the cached predictions."""
    picker_cfg = dict(base_picker_cfg)
    picker_cfg.update(overrides)

    det_pred_all = cache["det_pred"]
    p_pred_all = cache["p_pred"]
    s_pred_all = cache["s_pred"]
    gt_event = cache["gt_event"]
    p_idx_true = cache["p_idx_true"]
    s_idx_true = cache["s_idx_true"]

    N = det_pred_all.shape[0]

    tp = fp = tn = fn = 0
    p_errors: List[float] = []
    s_errors: List[float] = []
    n_p_picked = n_s_picked = 0
    n_p_gt = n_s_gt = 0

    for i in range(N):
        det_i = det_pred_all[i].astype(np.float32)
        gt_flag = bool(gt_event[i])
        pred_flag = bool(float(det_i.max()) >= trace_threshold)

        if gt_flag and pred_flag:
            tp += 1
        elif gt_flag and not pred_flag:
            fn += 1
        elif (not gt_flag) and pred_flag:
            fp += 1
        else:
            tn += 1

        if not (gt_flag and pred_flag):
            continue

        pick = pick_phases(
            det_curve=det_i,
            p_curve=p_pred_all[i].astype(np.float32),
            s_curve=s_pred_all[i].astype(np.float32),
            sample_rate=sample_rate,
            picker_cfg=picker_cfg,
        )

        p_gt = int(p_idx_true[i])
        s_gt = int(s_idx_true[i])

        if p_gt >= 0:
            n_p_gt += 1
            if pick["p_time"] is not None:
                n_p_picked += 1
                p_errors.append(abs(float(pick["p_time"]) - p_gt / sample_rate))
        if s_gt >= 0:
            n_s_gt += 1
            if pick["s_time"] is not None:
                n_s_picked += 1
                s_errors.append(abs(float(pick["s_time"]) - s_gt / sample_rate))

    def _safe(num, den):
        return float(num) / float(den) if den > 0 else 0.0

    trace_prec = _safe(tp, tp + fp)
    trace_rec = _safe(tp, tp + fn)
    trace_f1 = _safe(2 * trace_prec * trace_rec, trace_prec + trace_rec)
    trace_spec = _safe(tn, tn + fp)

    def _phase_stats(errors: List[float], n_picked: int, n_gt: int) -> Dict[str, float]:
        if not errors:
            return {
                "pick_rate": _safe(n_picked, n_gt),
                "mae": float("nan"),
                "medae": float("nan"),
                "std": float("nan"),
                "hit_small": 0.0,
                "hit_medium": 0.0,
                "hit_large": 0.0,
            }
        arr = np.asarray(errors)
        return {
            "pick_rate": _safe(n_picked, n_gt),
            "mae": float(arr.mean()),
            "medae": float(np.median(arr)),
            "std": float(arr.std()),
            "hit_small": float(np.mean(arr <= small_tol)),
            "hit_medium": float(np.mean(arr <= medium_tol)),
            "hit_large": float(np.mean(arr <= large_tol)),
        }

    return {
        "overrides": overrides,
        "trace_threshold": trace_threshold,
        "trace": {
            "tp": tp,
            "fp": fp,
            "tn": tn,
            "fn": fn,
            "precision": trace_prec,
            "recall": trace_rec,
            "f1": trace_f1,
            "specificity": trace_spec,
        },
        "p": _phase_stats(p_errors, n_p_picked, n_p_gt),
        "s": _phase_stats(s_errors, n_s_picked, n_s_gt),
    }


def _composite_score(row: Dict[str, Any]) -> float:
    """
    Smaller = better. We want low P/S median error, high pick rates, high
    trace F1. Mean is sensitive to outliers; we keep it secondary.
    """
    p = row["p"]
    s = row["s"]
    trace = row["trace"]

    # NaN-safe defaults punish empty-pick cells.
    p_med = p["medae"] if np.isfinite(p["medae"]) else 1.0
    s_med = s["medae"] if np.isfinite(s["medae"]) else 1.0
    p_mae = p["mae"] if np.isfinite(p["mae"]) else 1.0
    s_mae = s["mae"] if np.isfinite(s["mae"]) else 1.0

    return (
        2.0 * p_med
        + 2.0 * s_med
        + 0.2 * p_mae
        + 0.2 * s_mae
        - 0.5 * trace["f1"]
        - 0.3 * p["pick_rate"]
        - 0.3 * s["pick_rate"]
        - 0.5 * p["hit_small"]
        - 0.5 * s["hit_small"]
    )


# -----------------------------------------------------------------------------
# Pretty print
# -----------------------------------------------------------------------------


def _format_row(row: Dict[str, Any], rank: int) -> str:
    o = row["overrides"]
    p = row["p"]
    s = row["s"]
    t = row["trace"]
    return (
        f"#{rank:02d}  det_thr={o.get('det_window_threshold', '?'):.2f} "
        f"p_amp={o.get('p_amp_threshold', '?'):.2f} "
        f"s_amp={o.get('s_amp_threshold', '?'):.2f} "
        f"trace_thr={row['trace_threshold']:.2f}  "
        f"|  F1={t['f1']:.4f} spec={t['specificity']:.4f}  "
        f"|  P: pick={p['pick_rate']:.3f} medae={p['medae']*1000:6.1f}ms "
        f"hit_small={p['hit_small']:.3f}  "
        f"|  S: pick={s['pick_rate']:.3f} medae={s['medae']*1000:6.1f}ms "
        f"hit_small={s['hit_small']:.3f}"
    )


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------


def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--exp", type=int, default=None, help="Experiment id (3-digit).")
    p.add_argument("--ckpt", type=str, default=None, help="Explicit checkpoint path.")
    p.add_argument("--split", type=str, default="val", choices=["val", "test"])
    p.add_argument("--data-mode", choices=["all", "sample"], default=None,
                   help="Override data.mode from config.yaml.")
    p.add_argument("--max-batches", type=int, default=None,
                   help="Cap on inference batches (smoke test).")
    p.add_argument("--top-n", type=int, default=10,
                   help="Number of best cells to print.")
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> None:
    args = _parse_args(argv)

    main_cfg, model_cfg, paths_cfg = load_all_configs()
    if args.data_mode is not None:
        main_cfg.setdefault("data", {})["mode"] = args.data_mode
        print(f"[sweep] data.mode override: {args.data_mode}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    exp_root = project_root() / paths_cfg.get("experiments", {}).get("root_dir", "experiments")

    if args.ckpt:
        ckpt_path = Path(args.ckpt)
        if not ckpt_path.exists():
            raise FileNotFoundError(ckpt_path)
        exp_id = _infer_exp_id_from_path(ckpt_path)
    else:
        exp_dir = resolve_experiment_dir(exp_root, exp_id=args.exp)
        if exp_dir is None:
            raise FileNotFoundError(f"No experiments under {exp_root}.")
        ckpt_path = resolve_checkpoint(exp_dir, prefer="best")
        exp_id = _infer_exp_id_from_path(exp_dir)

    print(f"[sweep] checkpoint: {ckpt_path}")
    print(f"[sweep] split:      {args.split}")
    print(f"[sweep] device:     {device}")

    model = SeisMambaKAN(model_cfg).to(device)
    state = _safe_torch_load(ckpt_path, device)
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    model.load_state_dict(state)

    loader = build_dataloader(
        split=args.split,
        cfg=main_cfg,
        paths_cfg=paths_cfg,
        is_train=False,
    )

    print("[sweep] caching predictions on", args.split, "split...")
    cache = _cache_predictions(model, loader, device, max_batches=args.max_batches)

    # Free GPU memory before the CPU grid search.
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    metrics_cfg = main_cfg.get("metrics", {})
    sample_rate = float(metrics_cfg.get("sample_rate", 100.0))
    base_picker_cfg = copy.deepcopy(metrics_cfg.get("picker", {}))

    # Grid expanded (v2) to explore lower det_window thresholds and the
    # original p_amp=0.25 setting, since the v1 grid bottomed out at its
    # lowest values (sweep kept choosing det_window=0.5) and the original
    # p_amp=0.25 outperformed everything inside the v1 grid.
    det_window_grid = [0.3, 0.4, 0.5, 0.6]
    p_amp_grid = [0.10, 0.15, 0.20, 0.25]
    s_amp_grid = [0.10, 0.15, 0.20]
    trace_thr_grid = [0.6, 0.7, 0.8]

    combos = list(itertools.product(det_window_grid, p_amp_grid, s_amp_grid, trace_thr_grid))
    print(f"[sweep] grid: {len(combos)} combinations on {cache['det_pred'].shape[0]} samples")

    results: List[Dict[str, Any]] = []
    t0 = time.time()
    for i, (dw, pa, sa, tt) in enumerate(combos):
        overrides = {
            "det_window_threshold": dw,
            "p_amp_threshold": pa,
            "s_amp_threshold": sa,
        }
        row = _evaluate_combo(
            cache,
            base_picker_cfg=base_picker_cfg,
            overrides=overrides,
            trace_threshold=tt,
            sample_rate=sample_rate,
        )
        row["score"] = _composite_score(row)
        results.append(row)
        if (i + 1) % 5 == 0:
            elapsed = time.time() - t0
            print(f"  combo {i+1}/{len(combos)}  ({elapsed:.1f}s elapsed)")

    results.sort(key=lambda r: r["score"])

    print(f"\n[sweep] best {min(args.top_n, len(results))} combos by composite score:\n")
    for rank, row in enumerate(results[: args.top_n], start=1):
        print(_format_row(row, rank))

    # Persist full grid + best.
    out_dir = project_root() / paths_cfg.get("results", {}).get("root_dir", "results")
    if exp_id is not None:
        out_dir = out_dir / f"exp_{exp_id:03d}" / args.split
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "threshold_sweep.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "split": args.split,
                "n_samples": int(cache["det_pred"].shape[0]),
                "grid": {
                    "det_window_threshold": det_window_grid,
                    "p_amp_threshold": p_amp_grid,
                    "s_amp_threshold": s_amp_grid,
                    "trace_threshold": trace_thr_grid,
                },
                "best": results[0],
                "all": results,
            },
            f,
            indent=2,
        )
    print(f"\n[sweep] full grid written to {out_path}")

    best = results[0]
    print("\n[sweep] suggested config.yaml metrics.detection / metrics.picker:")
    print(f"  detection.trace_threshold:    {best['trace_threshold']}")
    print(f"  picker.det_window_threshold:  {best['overrides']['det_window_threshold']}")
    print(f"  picker.p_amp_threshold:       {best['overrides']['p_amp_threshold']}")
    print(f"  picker.s_amp_threshold:       {best['overrides']['s_amp_threshold']}")


if __name__ == "__main__":
    main()
