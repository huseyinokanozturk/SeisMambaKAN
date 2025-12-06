from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import csv
import json

import numpy as np
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt


# =============================================================================
# Utility functions
# =============================================================================


def _to_numpy_1d(x: torch.Tensor | np.ndarray) -> np.ndarray:
    """Convert input tensor/array to a contiguous 1D NumPy array."""
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    x = np.asarray(x)
    if x.ndim != 1:
        x = np.reshape(x, (-1,))
    return np.ascontiguousarray(x)


def _safe_div(numerator: float, denominator: float, eps: float = 1e-8) -> float:
    """Numerically safe division."""
    return float(numerator / (denominator + eps))


def _format_float(x: float | None) -> Optional[float]:
    """Ensure floats are JSON-serializable (convert NumPy floats)."""
    if x is None:
        return None
    return float(x)


# =============================================================================
# Output / label head extraction
# =============================================================================


def _extract_heads_from_outputs(
    outputs: Any,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Extract detection, P, and S heads from model outputs.

    This helper makes metrics robust to different naming conventions
    used in the network (for example, "detection"/"det", "p"/"p_gauss").
    """
    if not isinstance(outputs, dict):
        raise TypeError(
            f"Expected model outputs to be a dict, got {type(outputs)} instead. "
            "metrics.evaluate_model_on_loader currently supports only dict outputs."
        )

    keys = list(outputs.keys())
    lower_map = {k: k.lower() for k in keys}

    def find_exact(candidates: List[str]) -> Optional[str]:
        for cand in candidates:
            if cand in outputs:
                return cand
        return None

    # Detection head
    det_key = find_exact(["detection", "det", "detect", "det_out", "y_det"])
    if det_key is None and keys:
        det_key = keys[0]

    # P head
    p_key = find_exact(["p", "p_gauss", "p_gaussian", "p_out", "phase_p"])
    if p_key is None:
        for k, lk in lower_map.items():
            if "p" in lk and "s" not in lk and "det" not in lk:
                p_key = k
                break

    # S head
    s_key = find_exact(["s", "s_gauss", "s_gaussian", "s_out", "phase_s"])
    if s_key is None:
        for k, lk in lower_map.items():
            if "s" in lk and "p" not in lk and "det" not in lk:
                s_key = k
                break

    missing = []
    if det_key is None:
        missing.append("detection")
    if p_key is None:
        missing.append("p")
    if s_key is None:
        missing.append("s")

    if missing:
        raise KeyError(
            f"Could not infer head(s) {missing} from model outputs. "
            f"Available keys: {list(outputs.keys())}"
        )

    return outputs[det_key], outputs[p_key], outputs[s_key]


def _extract_label_curves(
    labels: Dict[str, Any]
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Optional[torch.Tensor], Optional[torch.Tensor]]:
    """
    Extract detection, P and S ground-truth curves (and optional indices)
    from a labels dict with flexible key names.

    Returns
    -------
    det_true : np.ndarray of shape (B, 1, T)
    p_true   : np.ndarray of shape (B, 1, T)
    s_true   : np.ndarray of shape (B, 1, T)
    p_idx_true_tensor : Optional[torch.Tensor] of shape (B,)
    s_idx_true_tensor : Optional[torch.Tensor] of shape (B,)
    """
    if not isinstance(labels, dict):
        raise TypeError(
            f"Expected labels to be a dict, got {type(labels)} instead. "
            "The dataset must return (x, labels_dict)."
        )

    keys = list(labels.keys())
    lower_map = {k: k.lower() for k in keys}

    def find_exact_label(candidates: List[str]) -> Optional[str]:
        for cand in candidates:
            if cand in labels:
                return cand
        return None

    # Detection label key
    det_label_key = find_exact_label(["det", "detection", "y_det", "y_detection", "label_det"])
    if det_label_key is None:
        for k, lk in lower_map.items():
            if "det" in lk:
                det_label_key = k
                break

    # P Gaussian label key
    p_label_key = find_exact_label(["p_gauss", "p_target", "p_label", "gauss_p", "p"])
    if p_label_key is None:
        for k, lk in lower_map.items():
            if "p" in lk and "s" not in lk and "det" not in lk:
                p_label_key = k
                break

    # S Gaussian label key
    s_label_key = find_exact_label(["s_gauss", "s_target", "s_label", "gauss_s", "s"])
    if s_label_key is None:
        for k, lk in lower_map.items():
            if "s" in lk and "p" not in lk and "det" not in lk:
                s_label_key = k
                break

    missing = []
    if det_label_key is None:
        missing.append("det")
    if p_label_key is None:
        missing.append("p_gauss")
    if s_label_key is None:
        missing.append("s_gauss")

    if missing:
        raise KeyError(
            f"Could not infer label(s) {missing} from labels dict. "
            f"Available keys: {list(labels.keys())}"
        )

    det_true_t = labels[det_label_key].detach().cpu()
    p_true_t = labels[p_label_key].detach().cpu()
    s_true_t = labels[s_label_key].detach().cpu()

    det_true = det_true_t.numpy()
    p_true = p_true_t.numpy()
    s_true = s_true_t.numpy()

    if det_true.ndim == 2:
        det_true = det_true[:, None, :]
    if p_true.ndim == 2:
        p_true = p_true[:, None, :]
    if s_true.ndim == 2:
        s_true = s_true[:, None, :]

    p_idx_true_tensor = labels.get("p_idx", None)
    s_idx_true_tensor = labels.get("s_idx", None)

    return det_true, p_true, s_true, p_idx_true_tensor, s_idx_true_tensor


# =============================================================================
# Phase picker
# =============================================================================


def pick_phases(
    det_curve: np.ndarray | torch.Tensor,
    p_curve: np.ndarray | torch.Tensor,
    s_curve: np.ndarray | torch.Tensor,
    sample_rate: float,
    picker_cfg: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Phase picker for a single trace.

    This function is agnostic to ground-truth labels. It simply
    finds the most likely P and S picks inside a detection window.
    """
    det = _to_numpy_1d(det_curve)
    p = _to_numpy_1d(p_curve)
    s = _to_numpy_1d(s_curve)

    T = det.shape[0]
    assert p.shape[0] == T and s.shape[0] == T, "All curves must have the same length."

    use_detection_window = bool(picker_cfg.get("use_detection_window", True))
    det_window_threshold = float(picker_cfg.get("det_window_threshold", 0.3))
    p_amp_threshold = float(picker_cfg.get("p_amp_threshold", 0.1))
    s_amp_threshold = float(picker_cfg.get("s_amp_threshold", 0.1))
    min_ps_gap_sec = float(picker_cfg.get("min_ps_gap_sec", 0.0))
    max_search_pad_sec = float(picker_cfg.get("max_search_pad_sec", 0.0))

    pad_samples = int(round(max_search_pad_sec * sample_rate))

    det_max = float(det.max()) if T > 0 else 0.0
    has_event_pred = True

    # Detection window for phase search
    if use_detection_window:
        mask = det >= det_window_threshold
        if np.any(mask):
            idx = np.where(mask)[0]
            start = max(int(idx[0]) - pad_samples, 0)
            end = min(int(idx[-1]) + pad_samples, T - 1)
        else:
            start, end = 0, T - 1
    else:
        start, end = 0, T - 1

    # P-phase pick
    p_idx: Optional[int] = None
    p_time: Optional[float] = None
    p_amp: Optional[float] = None

    if end >= start:
        p_window = p[start : end + 1]
        p_rel_idx = int(np.argmax(p_window))
        p_idx_candidate = start + p_rel_idx
        p_amp_candidate = float(p[p_idx_candidate])

        if p_amp_candidate >= p_amp_threshold:
            p_idx = p_idx_candidate
            p_amp = p_amp_candidate
            p_time = p_idx / sample_rate

    # S-phase pick
    s_idx: Optional[int] = None
    s_time: Optional[float] = None
    s_amp: Optional[float] = None

    if end >= start:
        s_window = s[start : end + 1]
        s_rel_idx = int(np.argmax(s_window))
        s_idx_candidate = start + s_rel_idx
        s_amp_candidate = float(s[s_idx_candidate])

        if s_amp_candidate >= s_amp_threshold:
            s_idx = s_idx_candidate
            s_amp = s_amp_candidate
            s_time = s_idx / sample_rate

    ps_gap_ok: Optional[bool] = None
    if p_time is not None and s_time is not None and min_ps_gap_sec > 0.0:
        ps_gap_ok = (s_time - p_time) >= min_ps_gap_sec

    return {
        "has_event_pred": has_event_pred,
        "det_max": det_max,
        "p_idx": p_idx,
        "s_idx": s_idx,
        "p_time": p_time,
        "s_time": s_time,
        "p_amp": p_amp,
        "s_amp": s_amp,
        "ps_gap_ok": ps_gap_ok,
    }


# =============================================================================
# Detection metrics
# =============================================================================


def compute_detection_metrics(
    y_event_true: List[int],
    y_event_pred: List[int],
    y_ts_true: Optional[np.ndarray] = None,
    y_ts_pred: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """
    Compute detection metrics at both trace-level and (optionally) time-step level.
    """
    assert len(y_event_true) == len(y_event_pred), "Mismatched trace-level lengths."

    tp = fp = tn = fn = 0

    for y_t, y_p in zip(y_event_true, y_event_pred):
        if y_t == 1 and y_p == 1:
            tp += 1
        elif y_t == 0 and y_p == 0:
            tn += 1
        elif y_t == 0 and y_p == 1:
            fp += 1
        elif y_t == 1 and y_p == 0:
            fn += 1

    total = tp + tn + fp + fn

    trace_acc = _safe_div(tp + tn, total)
    trace_prec = _safe_div(tp, tp + fp)
    trace_rec = _safe_div(tp, tp + fn)
    trace_f1 = _safe_div(2.0 * trace_prec * trace_rec, trace_prec + trace_rec)
    trace_spec = _safe_div(tn, tn + fp)

    metrics: Dict[str, Any] = {
        "trace": {
            "tp": int(tp),
            "fp": int(fp),
            "tn": int(tn),
            "fn": int(fn),
            "accuracy": _format_float(trace_acc),
            "precision": _format_float(trace_prec),
            "recall": _format_float(trace_rec),
            "f1": _format_float(trace_f1),
            "specificity": _format_float(trace_spec),
        }
    }

    # Time-step level metrics
    if y_ts_true is not None and y_ts_pred is not None:
        y_ts_true = np.asarray(y_ts_true).astype(int).reshape(-1)
        y_ts_pred = np.asarray(y_ts_pred).astype(int).reshape(-1)
        assert y_ts_true.shape == y_ts_pred.shape, "Mismatched timestep shapes."

        tp_ts = int(np.sum((y_ts_true == 1) & (y_ts_pred == 1)))
        tn_ts = int(np.sum((y_ts_true == 0) & (y_ts_pred == 0)))
        fp_ts = int(np.sum((y_ts_true == 0) & (y_ts_pred == 1)))
        fn_ts = int(np.sum((y_ts_true == 1) & (y_ts_pred == 0)))
        total_ts = tp_ts + tn_ts + fp_ts + fn_ts

        acc_ts = _safe_div(tp_ts + tn_ts, total_ts)
        prec_ts = _safe_div(tp_ts, tp_ts + fp_ts)
        rec_ts = _safe_div(tp_ts, tp_ts + fn_ts)
        f1_ts = _safe_div(2.0 * prec_ts * rec_ts, prec_ts + rec_ts)
        spec_ts = _safe_div(tn_ts, tn_ts + fp_ts)

        metrics["timestep"] = {
            "tp": int(tp_ts),
            "fp": int(fp_ts),
            "tn": int(tn_ts),
            "fn": int(fn_ts),
            "accuracy": _format_float(acc_ts),
            "precision": _format_float(prec_ts),
            "recall": _format_float(rec_ts),
            "f1": _format_float(f1_ts),
            "specificity": _format_float(spec_ts),
        }

    return metrics


# =============================================================================
# Phase metrics
# =============================================================================


def compute_phase_metrics(
    phase_name: str,
    gt_indices: List[int],
    pred_indices: List[Optional[int]],
    sample_rate: float,
    tolerance_cfg: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Compute phase picking metrics (MAE, median AE, STD, hit-rates, etc.)
    for a single phase (P or S).
    """
    assert len(gt_indices) == len(pred_indices), "Mismatched phase index lists."

    phase_tol = tolerance_cfg.get(phase_name, {})
    tol_small = float(phase_tol.get("small", 0.01))
    tol_medium = float(phase_tol.get("medium", 0.02))
    tol_large = float(phase_tol.get("large", 0.05))

    errors_sec: List[float] = []
    picked_count = 0

    for gt_idx, pred_idx in zip(gt_indices, pred_indices):
        if pred_idx is None:
            continue
        picked_count += 1
        err_samples = int(pred_idx) - int(gt_idx)
        err_sec = err_samples / sample_rate
        errors_sec.append(err_sec)

    num_gt = len(gt_indices)
    num_pred = picked_count

    if len(errors_sec) == 0:
        return {
            "num_gt": int(num_gt),
            "num_pred": int(num_pred),
            "pick_rate": _format_float(_safe_div(num_pred, num_gt)),
            "mae_sec": None,
            "medae_sec": None,
            "std_sec": None,
            "hit_rate_small": None,
            "hit_rate_medium": None,
            "hit_rate_large": None,
            "tolerance_sec": {
                "small": tol_small,
                "medium": tol_medium,
                "large": tol_large,
            },
        }

    errors_sec_arr = np.asarray(errors_sec, dtype=float)
    abs_err = np.abs(errors_sec_arr)

    mae = float(np.mean(abs_err))
    medae = float(np.median(abs_err))
    std = float(np.std(errors_sec_arr))

    hit_small = _safe_div(np.sum(abs_err < tol_small), len(abs_err))
    hit_medium = _safe_div(np.sum(abs_err < tol_medium), len(abs_err))
    hit_large = _safe_div(np.sum(abs_err < tol_large), len(abs_err))

    return {
        "num_gt": int(num_gt),
        "num_pred": int(num_pred),
        "pick_rate": _format_float(_safe_div(num_pred, num_gt)),
        "mae_sec": _format_float(mae),
        "medae_sec": _format_float(medae),
        "std_sec": _format_float(std),
        "hit_rate_small": _format_float(hit_small),
        "hit_rate_medium": _format_float(hit_medium),
        "hit_rate_large": _format_float(hit_large),
        "tolerance_sec": {
            "small": tol_small,
            "medium": tol_medium,
            "large": tol_large,
        },
    }


# =============================================================================
# Visualization helpers
# =============================================================================


def _plot_single_example(
    trace: Dict[str, Any],
    sample_rate: float,
    out_path: Path,
    title: str,
) -> None:
    """
    Plot detection, P and S curves for a single example and save as PNG.

    The figure title also contains P/S sample indices and differences.
    """
    det_true = trace["det_true"]
    det_pred = trace["det_pred"]
    p_true = trace["p_true"]
    p_pred = trace["p_pred"]
    s_true = trace["s_true"]
    s_pred = trace["s_pred"]

    p_idx_true = trace["p_idx_true"]
    s_idx_true = trace["s_idx_true"]
    p_idx_pred = trace["p_idx_pred"]
    s_idx_pred = trace["s_idx_pred"]

    # Build info strings for title
    if p_idx_pred is not None:
        p_d_samples = int(p_idx_pred) - int(p_idx_true)
        p_d_sec = p_d_samples / sample_rate
        p_info = f"P: gt={p_idx_true}, pred={p_idx_pred}, d={p_d_samples} ({p_d_sec:.3f}s)"
    else:
        p_info = f"P: gt={p_idx_true}, pred=None"

    if s_idx_pred is not None:
        s_d_samples = int(s_idx_pred) - int(s_idx_true)
        s_d_sec = s_d_samples / sample_rate
        s_info = f"S: gt={s_idx_true}, pred={s_idx_pred}, d={s_d_samples} ({s_d_sec:.3f}s)"
    else:
        s_info = f"S: gt={s_idx_true}, pred=None"

    full_title = f"{title}\n{p_info} | {s_info}"

    T = det_true.shape[0]
    t = np.arange(T) / sample_rate

    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

    # Detection
    ax = axes[0]
    ax.plot(t, det_true, label="Detection (GT)", linewidth=1.0)
    ax.plot(t, det_pred, label="Detection (Pred)", linewidth=1.0, linestyle="--")
    ax.set_ylabel("Det prob")
    ax.set_title(full_title)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right", fontsize=8)

    # P-phase
    ax = axes[1]
    ax.plot(t, p_true, label="P Gaussian (GT)", linewidth=1.0)
    ax.plot(t, p_pred, label="P Gaussian (Pred)", linewidth=1.0, linestyle="--")

    if p_idx_true is not None:
        ax.axvline(p_idx_true / sample_rate, color="g", linestyle="-", linewidth=1.0, label="P GT")
    if p_idx_pred is not None:
        ax.axvline(p_idx_pred / sample_rate, color="r", linestyle="--", linewidth=1.0, label="P Pred")

    ax.set_ylabel("P amplitude")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right", fontsize=8)

    # S-phase
    ax = axes[2]
    ax.plot(t, s_true, label="S Gaussian (GT)", linewidth=1.0)
    ax.plot(t, s_pred, label="S Gaussian (Pred)", linewidth=1.0, linestyle="--")

    if s_idx_true is not None:
        ax.axvline(s_idx_true / sample_rate, color="g", linestyle="-", linewidth=1.0, label="S GT")
    if s_idx_pred is not None:
        ax.axvline(s_idx_pred / sample_rate, color="r", linestyle="--", linewidth=1.0, label="S Pred")

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("S amplitude")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right", fontsize=8)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _save_example_plots(
    event_traces: List[Dict[str, Any]],
    sample_rate: float,
    out_dir: Path,
    split_name: str,
) -> None:
    """
    Save:
      - one random event example
      - three worst event examples (by max(P_abs_err, S_abs_err))
    as PNG files into the metrics directory.
    """
    if not event_traces:
        return

    rng = np.random.RandomState(42)

    # Random example
    rand_idx = int(rng.randint(0, len(event_traces)))
    rand_trace = event_traces[rand_idx]
    rand_path = out_dir / f"{split_name}_random_example.png"

    _plot_single_example(
        trace=rand_trace,
        sample_rate=sample_rate,
        out_path=rand_path,
        title=f"{split_name.upper()} - Random Example (idx={rand_idx})",
    )

    # Worst-3 by error (combined P/S absolute error)
    scores = []
    for tr in event_traces:
        p_err = tr.get("p_abs_err_sec", None)
        s_err = tr.get("s_abs_err_sec", None)
        p_val = p_err if p_err is not None else 0.0
        s_val = s_err if s_err is not None else 0.0
        scores.append(max(p_val, s_val))

    scores = np.asarray(scores, dtype=float)
    order = np.argsort(scores)
    worst_indices = order[-3:]

    for rank, idx in enumerate(reversed(worst_indices), start=1):
        tr = event_traces[int(idx)]
        out_path = out_dir / f"{split_name}_worst_{rank}.png"
        _plot_single_example(
            trace=tr,
            sample_rate=sample_rate,
            out_path=out_path,
            title=f"{split_name.upper()} - Worst #{rank} (err={scores[idx]:.3f} s)",
        )


# =============================================================================
# High-level evaluation
# =============================================================================


@torch.no_grad()
def evaluate_model_on_loader(
    model: torch.nn.Module,
    data_loader: DataLoader,
    device: torch.device,
    main_cfg: Dict[str, Any],
    split_name: str = "val",
    exp_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """
    Evaluate a trained model on a given DataLoader and compute:
      - trace-level detection metrics
      - time-step-level detection metrics
      - P and S phase picking metrics
      - diagnostic plots for one random and three worst event examples

    IMPORTANT:
      - Phase metrics are only computed on traces that contain a ground-truth
        event (gt_event_flag == 1).
      - Phase picks are only searched when the model also predicts an event
        on that trace (pred_event_flag == 1). Otherwise P/S prediction is None.
    """
    metrics_cfg = main_cfg.get("metrics", {})
    sample_rate = float(metrics_cfg.get("sample_rate", 100.0))
    detection_cfg = metrics_cfg.get("detection", {})
    picker_cfg = metrics_cfg.get("picker", {})
    phase_tol_cfg = metrics_cfg.get("phase_tolerance", {})
    eval_cfg = metrics_cfg.get("eval", {})

    trace_threshold = float(detection_cfg.get("trace_threshold", 0.5))
    timestep_threshold = float(detection_cfg.get("timestep_threshold", 0.5))

    max_batches = eval_cfg.get("max_batches", None)
    save_json = bool(eval_cfg.get("save_json", True))
    save_per_trace_csv = bool(eval_cfg.get("save_per_trace_csv", False))
    output_subdir = str(eval_cfg.get("output_dir", "metrics"))

    metrics_out_dir: Optional[Path] = None
    if exp_dir is not None:
        metrics_out_dir = Path(exp_dir) / output_subdir
        metrics_out_dir.mkdir(parents=True, exist_ok=True)

    model.eval()
    device_type = device.type

    y_event_true: List[int] = []
    y_event_pred: List[int] = []

    y_ts_true_list: List[np.ndarray] = []
    y_ts_pred_list: List[np.ndarray] = []

    p_gt_indices: List[int] = []
    p_pred_indices: List[Optional[int]] = []

    s_gt_indices: List[int] = []
    s_pred_indices: List[Optional[int]] = []

    per_trace_rows: List[Dict[str, Any]] = []
    event_traces: List[Dict[str, Any]] = []

    for batch_idx, (x, labels) in enumerate(data_loader):
        if (max_batches is not None) and (batch_idx >= int(max_batches)):
            break

        x = x.to(device, non_blocking=True)

        if x.dim() == 3:
            x_model = x.permute(0, 2, 1).contiguous()
        else:
            x_model = x

        with torch.amp.autocast(device_type=device_type, enabled=False):
            outputs = model(x_model)

        det_out, p_out, s_out = _extract_heads_from_outputs(outputs)

        det_pred = det_out.detach().cpu().numpy()
        p_pred = p_out.detach().cpu().numpy()
        s_pred = s_out.detach().cpu().numpy()

        if det_pred.ndim == 2:
            det_pred = det_pred[:, None, :]
        if p_pred.ndim == 2:
            p_pred = p_pred[:, None, :]
        if s_pred.ndim == 2:
            s_pred = s_pred[:, None, :]

        det_true, p_gauss_true, s_gauss_true, p_idx_true_tensor, s_idx_true_tensor = (
            _extract_label_curves(labels)
        )

        B = det_true.shape[0]

        for i in range(B):
            det_true_i = det_true[i, 0]
            det_pred_i = det_pred[i, 0]
            p_pred_i = p_pred[i, 0]
            s_pred_i = s_pred[i, 0]
            p_gauss_true_i = p_gauss_true[i, 0]
            s_gauss_true_i = s_gauss_true[i, 0]

            # ------------------------------------------------------------------
            # 1) Trace-level event flags
            # ------------------------------------------------------------------
            gt_event_flag = int(
                (det_true_i.max() >= 0.5)
                or (p_gauss_true_i.max() > 0.0)
                or (s_gauss_true_i.max() > 0.0)
            )

            pred_event_flag = int(det_pred_i.max() >= trace_threshold)

            y_event_true.append(gt_event_flag)
            y_event_pred.append(pred_event_flag)

            # ------------------------------------------------------------------
            # 2) Time-step level detection labels
            # ------------------------------------------------------------------
            y_ts_true = (det_true_i >= 0.5).astype(int)
            y_ts_pred = (det_pred_i >= timestep_threshold).astype(int)
            y_ts_true_list.append(y_ts_true)
            y_ts_pred_list.append(y_ts_pred)

            # ------------------------------------------------------------------
            # 3) Ground-truth phase indices (meaningful only for event traces)
            # ------------------------------------------------------------------
            if p_idx_true_tensor is not None:
                p_idx_true = int(p_idx_true_tensor[i].item())
            else:
                p_idx_true = int(np.argmax(p_gauss_true_i))

            if s_idx_true_tensor is not None:
                s_idx_true = int(s_idx_true_tensor[i].item())
            else:
                s_idx_true = int(np.argmax(s_gauss_true_i))

            # ------------------------------------------------------------------
            # 4) Phase metrics and visualization only for GT event traces
            #    P/S picks are searched only when the model predicts an event.
            # ------------------------------------------------------------------
            if gt_event_flag == 1:
                # Ground-truth indices for all event traces
                p_gt_indices.append(p_idx_true)
                s_gt_indices.append(s_idx_true)

                if pred_event_flag == 1:
                    pick_result = pick_phases(
                        det_curve=det_pred_i,
                        p_curve=p_pred_i,
                        s_curve=s_pred_i,
                        sample_rate=sample_rate,
                        picker_cfg=picker_cfg,
                    )
                    p_idx_pred = pick_result["p_idx"]
                    s_idx_pred = pick_result["s_idx"]
                else:
                    p_idx_pred = None
                    s_idx_pred = None

                p_pred_indices.append(p_idx_pred)
                s_pred_indices.append(s_idx_pred)

                # Errors for visualization ranking
                p_err_sec = None
                p_abs_err_sec = None
                if p_idx_pred is not None:
                    p_err_samples = int(p_idx_pred) - int(p_idx_true)
                    p_err_sec = p_err_samples / sample_rate
                    p_abs_err_sec = abs(p_err_sec)

                s_err_sec = None
                s_abs_err_sec = None
                if s_idx_pred is not None:
                    s_err_samples = int(s_idx_pred) - int(s_idx_true)
                    s_err_sec = s_err_samples / sample_rate
                    s_abs_err_sec = abs(s_err_sec)

                event_traces.append(
                    {
                        "det_true": det_true_i.copy(),
                        "det_pred": det_pred_i.copy(),
                        "p_true": p_gauss_true_i.copy(),
                        "p_pred": p_pred_i.copy(),
                        "s_true": s_gauss_true_i.copy(),
                        "s_pred": s_pred_i.copy(),
                        "p_idx_true": p_idx_true,
                        "s_idx_true": s_idx_true,
                        "p_idx_pred": p_idx_pred,
                        "s_idx_pred": s_idx_pred,
                        "p_err_sec": p_err_sec,
                        "p_abs_err_sec": p_abs_err_sec,
                        "s_err_sec": s_err_sec,
                        "s_abs_err_sec": s_abs_err_sec,
                    }
                )

            # ------------------------------------------------------------------
            # 5) Optional per-trace CSV row
            # ------------------------------------------------------------------
            if save_per_trace_csv:
                row = {
                    "split": split_name,
                    "batch_idx": int(batch_idx),
                    "sample_idx": int(i),
                    "gt_event": int(gt_event_flag),
                    "pred_event": int(pred_event_flag),
                    "det_max": _format_float(float(det_pred_i.max())),
                    "p_idx_true": int(p_idx_true),
                    "s_idx_true": int(s_idx_true),
                    "p_idx_pred": int(p_idx_pred) if (gt_event_flag == 1 and p_idx_pred is not None) else None,
                    "s_idx_pred": int(s_idx_pred) if (gt_event_flag == 1 and s_idx_pred is not None) else None,
                }
                per_trace_rows.append(row)

    # -------------------------------------------------------------------------
    # Aggregate detection metrics
    # -------------------------------------------------------------------------
    y_ts_true_flat = np.concatenate(y_ts_true_list, axis=0) if y_ts_true_list else None
    y_ts_pred_flat = np.concatenate(y_ts_pred_list, axis=0) if y_ts_pred_list else None

    det_metrics = compute_detection_metrics(
        y_event_true=y_event_true,
        y_event_pred=y_event_pred,
        y_ts_true=y_ts_true_flat,
        y_ts_pred=y_ts_pred_flat,
    )

    # -------------------------------------------------------------------------
    # Aggregate phase metrics
    # -------------------------------------------------------------------------
    p_metrics = compute_phase_metrics(
        phase_name="p",
        gt_indices=p_gt_indices,
        pred_indices=p_pred_indices,
        sample_rate=sample_rate,
        tolerance_cfg=phase_tol_cfg,
    )

    s_metrics = compute_phase_metrics(
        phase_name="s",
        gt_indices=s_gt_indices,
        pred_indices=s_pred_indices,
        sample_rate=sample_rate,
        tolerance_cfg=phase_tol_cfg,
    )

    metrics: Dict[str, Any] = {
        "split": split_name,
        "detection": det_metrics,
        "p_phase": p_metrics,
        "s_phase": s_metrics,
        "config": {
            "sample_rate": sample_rate,
            "trace_threshold": trace_threshold,
            "timestep_threshold": timestep_threshold,
        },
    }

    # -------------------------------------------------------------------------
    # Save outputs (JSON + CSV + PNG figures) if requested
    # -------------------------------------------------------------------------
    if metrics_out_dir is not None:
        if save_json:
            json_path = metrics_out_dir / f"{split_name}_metrics.json"
            with json_path.open("w", encoding="utf-8") as f:
                json.dump(metrics, f, indent=2)

        if save_per_trace_csv and per_trace_rows:
            csv_path = metrics_out_dir / f"{split_name}_per_trace.csv"
            fieldnames = list(per_trace_rows[0].keys())
            with csv_path.open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                for row in per_trace_rows:
                    writer.writerow(row)

        _save_example_plots(
            event_traces=event_traces,
            sample_rate=sample_rate,
            out_dir=metrics_out_dir,
            split_name=split_name,
        )

    return metrics
