from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import csv
import json

import numpy as np
import torch
from torch.utils.data import DataLoader

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


def _format_float(x: float) -> float:
    """Ensure floats are JSON-serializable (convert NumPy floats)."""
    return float(x)


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

    Parameters
    ----------
    det_curve : array-like of shape (T,)
        Detection probability curve in [0, 1].
    p_curve : array-like of shape (T,)
        Predicted P-phase Gaussian curve (continuous values).
    s_curve : array-like of shape (T,)
        Predicted S-phase Gaussian curve (continuous values).
    sample_rate : float
        Sampling frequency in Hz.
    picker_cfg : dict
        Configuration dictionary taken from config["metrics"]["picker"].

    Returns
    -------
    result : dict
        Dictionary with the following fields:
            has_event_pred : bool
            det_max        : float
            p_idx          : Optional[int]
            s_idx          : Optional[int]
            p_time         : Optional[float]  # seconds
            s_time         : Optional[float]  # seconds
            p_amp          : Optional[float]
            s_amp          : Optional[float]
            ps_gap_ok      : Optional[bool]
    """
    det = _to_numpy_1d(det_curve)
    p = _to_numpy_1d(p_curve)
    s = _to_numpy_1d(s_curve)

    T = det.shape[0]
    assert p.shape[0] == T and s.shape[0] == T, "All curves must have the same length."

    # -------------------------------------------------------------------------
    # Configuration
    # -------------------------------------------------------------------------
    use_detection_window = bool(picker_cfg.get("use_detection_window", True))
    det_window_threshold = float(picker_cfg.get("det_window_threshold", 0.3))
    p_amp_threshold = float(picker_cfg.get("p_amp_threshold", 0.1))
    s_amp_threshold = float(picker_cfg.get("s_amp_threshold", 0.1))
    min_ps_gap_sec = float(picker_cfg.get("min_ps_gap_sec", 0.0))
    max_search_pad_sec = float(picker_cfg.get("max_search_pad_sec", 0.0))

    pad_samples = int(round(max_search_pad_sec * sample_rate))

    # -------------------------------------------------------------------------
    # Trace-level event decision
    # (Whether this window contains an earthquake or just noise)
    # -------------------------------------------------------------------------
    det_max = float(det.max()) if T > 0 else 0.0

    # Note: event vs noise decision itself is handled in detection metrics.
    # Here we simply report det_max and always attempt picking if curves are valid.
    has_event_pred = True  # picker is agnostic; external code interprets det_max

    # -------------------------------------------------------------------------
    # Detection-based search window
    # -------------------------------------------------------------------------
    if use_detection_window:
        mask = det >= det_window_threshold
        if np.any(mask):
            idx = np.where(mask)[0]
            start = max(int(idx[0]) - pad_samples, 0)
            end = min(int(idx[-1]) + pad_samples, T - 1)
        else:
            # Fallback: use the entire trace
            start, end = 0, T - 1
    else:
        start, end = 0, T - 1

    # -------------------------------------------------------------------------
    # P-phase pick
    # -------------------------------------------------------------------------
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

    # -------------------------------------------------------------------------
    # S-phase pick
    # -------------------------------------------------------------------------
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

    # -------------------------------------------------------------------------
    # Optional P–S gap sanity check
    # -------------------------------------------------------------------------
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

    Parameters
    ----------
    y_event_true : list of int
        Ground-truth event labels per trace (0 = noise, 1 = event).
    y_event_pred : list of int
        Predicted event labels per trace.
    y_ts_true : np.ndarray, optional
        Flattened ground-truth detection labels per time-step (0/1).
    y_ts_pred : np.ndarray, optional
        Flattened predicted detection labels per time-step (0/1).

    Returns
    -------
    metrics : dict
        Dictionary with keys:
            "trace"    : dict of accuracy, precision, recall, f1, specificity
            "timestep" : dict (only if y_ts_true and y_ts_pred are provided)
    """
    assert len(y_event_true) == len(y_event_pred), "Mismatched trace-level lengths."

    # -------------------------------------------------------------------------
    # Trace-level confusion matrix
    # -------------------------------------------------------------------------
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

    # -------------------------------------------------------------------------
    # Time-step level metrics (optional)
    # -------------------------------------------------------------------------
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
# Phase metrics (P/S MAE, hit-rates, etc.)
# =============================================================================


def compute_phase_metrics(
    phase_name: str,
    gt_indices: List[int],
    pred_indices: List[Optional[int]],
    sample_rate: float,
    tolerance_cfg: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Compute phase picking metrics (MAE, median AE, STD, hit-rates, etc.).

    Parameters
    ----------
    phase_name : {"p", "s"}
        Phase identifier used for selecting tolerance configuration.
    gt_indices : list of int
        Ground-truth phase indices (samples) for traces that contain an event.
    pred_indices : list of Optional[int]
        Predicted phase indices (samples). None indicates a missing pick.
    sample_rate : float
        Sampling frequency in Hz.
    tolerance_cfg : dict
        metrics["phase_tolerance"] configuration dict.

    Returns
    -------
    metrics : dict
        Dictionary containing error statistics and hit-rates.
    """
    assert len(gt_indices) == len(pred_indices), "Mismatched phase index lists."

    # Select phase-specific tolerance thresholds
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
        # No valid picks; return metrics with NaNs
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
# High-level evaluation on a DataLoader
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
      - P and S phase picking metrics (MAE, MedAE, STD, hit-rates)

    This function is designed to be called from scripts/ (e.g., eval_val.py),
    after loading the model weights and constructing the corresponding loader.

    Parameters
    ----------
    model : torch.nn.Module
        Trained SeisMambaKAN model in evaluation mode.
    data_loader : DataLoader
        Validation or test DataLoader. It must yield (x, labels) where:
            x      : (B, T, C) float tensor
            labels : dict containing at least:
                     - "det"      : (B, 1, T) detection labels in [0, 1]
                     - "p_gauss"  : (B, 1, T) P Gaussian targets
                     - "s_gauss"  : (B, 1, T) S Gaussian targets
                     If explicit indices are available, you may add:
                     - "p_idx"    : (B,) int32 tensor
                     - "s_idx"    : (B,) int32 tensor
    device : torch.device
        Device on which to run inference.
    main_cfg : dict
        Parsed main configuration (config.yaml). Must contain a "metrics" block.
    split_name : str, optional
        Name of the evaluated split ("val", "test", etc.), used for filenames.
    exp_dir : Path, optional
        Path to the experiment directory. If provided, metrics and per-trace
        results will be saved under exp_dir / metrics_cfg["eval"]["output_dir"].

    Returns
    -------
    metrics : dict
        Aggregated metrics dictionary suitable for logging and JSON export.
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

    # Output directory inside the experiment folder (if exp_dir is given)
    metrics_out_dir: Optional[Path] = None
    if exp_dir is not None:
        metrics_out_dir = Path(exp_dir) / output_subdir
        metrics_out_dir.mkdir(parents=True, exist_ok=True)

    model.eval()
    device_type = device.type

    # -------------------------------------------------------------------------
    # Storage for detection metrics
    # -------------------------------------------------------------------------
    y_event_true: List[int] = []
    y_event_pred: List[int] = []

    y_ts_true_list: List[np.ndarray] = []
    y_ts_pred_list: List[np.ndarray] = []

    # -------------------------------------------------------------------------
    # Storage for phase metrics
    # -------------------------------------------------------------------------
    p_gt_indices: List[int] = []
    p_pred_indices: List[Optional[int]] = []

    s_gt_indices: List[int] = []
    s_pred_indices: List[Optional[int]] = []

    # Optional per-trace rows for CSV
    per_trace_rows: List[Dict[str, Any]] = []

    # -------------------------------------------------------------------------
    # Main evaluation loop
    # -------------------------------------------------------------------------
    for batch_idx, (x, labels) in enumerate(data_loader):
        if (max_batches is not None) and (batch_idx >= int(max_batches)):
            break

        x = x.to(device, non_blocking=True)

        # Model expects (B, C, T)
        if x.dim() == 3:  # (B, T, C)
            x = x.permute(0, 2, 1).contiguous()

        with torch.amp.autocast(device_type=device_type, enabled=False):
            outputs = model(x)

        # Extract detection and phase outputs
        # Expected keys in outputs: "detection", "p", "s"
        det_pred = outputs["detection"].detach().cpu().numpy()  # (B, 1, T)
        p_pred = outputs["p"].detach().cpu().numpy()            # (B, 1, T)
        s_pred = outputs["s"].detach().cpu().numpy()            # (B, 1, T)

        det_true = labels["det"].detach().cpu().numpy()         # (B, 1, T)
        p_gauss_true = labels["p_gauss"].detach().cpu().numpy() # (B, 1, T)
        s_gauss_true = labels["s_gauss"].detach().cpu().numpy() # (B, 1, T)

        # Optional ground-truth indices
        p_idx_true_tensor = labels.get("p_idx", None)
        s_idx_true_tensor = labels.get("s_idx", None)

        B = det_true.shape[0]

        for i in range(B):
            det_true_i = det_true[i, 0]   # (T,)
            det_pred_i = det_pred[i, 0]   # (T,)
            p_pred_i = p_pred[i, 0]
            s_pred_i = s_pred[i, 0]
            p_gauss_true_i = p_gauss_true[i, 0]
            s_gauss_true_i = s_gauss_true[i, 0]

            # -----------------------------------------------------------------
            # Ground-truth event / noise label per trace
            # -----------------------------------------------------------------
            # If an explicit event flag is not provided, we infer it from labels:
            # any detection > 0.5 or non-zero P/S Gaussian.
            gt_event_flag = int(
                (det_true_i.max() >= 0.5)
                or (p_gauss_true_i.max() > 0.0)
                or (s_gauss_true_i.max() > 0.0)
            )

            # Predicted event / noise per trace (trace-level decision)
            pred_event_flag = int(det_pred_i.max() >= trace_threshold)

            y_event_true.append(gt_event_flag)
            y_event_pred.append(pred_event_flag)

            # -----------------------------------------------------------------
            # Time-step level detection labels
            # -----------------------------------------------------------------
            y_ts_true = (det_true_i >= 0.5).astype(int)
            y_ts_pred = (det_pred_i >= timestep_threshold).astype(int)
            y_ts_true_list.append(y_ts_true)
            y_ts_pred_list.append(y_ts_pred)

            # -----------------------------------------------------------------
            # Ground-truth phase indices
            # -----------------------------------------------------------------
            if p_idx_true_tensor is not None:
                p_idx_true = int(p_idx_true_tensor[i].item())
            else:
                p_idx_true = int(np.argmax(p_gauss_true_i))

            if s_idx_true_tensor is not None:
                s_idx_true = int(s_idx_true_tensor[i].item())
            else:
                s_idx_true = int(np.argmax(s_gauss_true_i))

            # Only consider traces with an event for phase metrics
            if gt_event_flag == 1:
                p_gt_indices.append(p_idx_true)
                s_gt_indices.append(s_idx_true)

            # -----------------------------------------------------------------
            # Phase picking on predicted curves
            # -----------------------------------------------------------------
            pick_result = pick_phases(
                det_curve=det_pred_i,
                p_curve=p_pred_i,
                s_curve=s_pred_i,
                sample_rate=sample_rate,
                picker_cfg=picker_cfg,
            )

            p_idx_pred = pick_result["p_idx"]
            s_idx_pred = pick_result["s_idx"]

            if gt_event_flag == 1:
                p_pred_indices.append(p_idx_pred)
                s_pred_indices.append(s_idx_pred)

            # -----------------------------------------------------------------
            # Optional per-trace row
            # -----------------------------------------------------------------
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
                    "p_idx_pred": int(p_idx_pred) if p_idx_pred is not None else None,
                    "s_idx_pred": int(s_idx_pred) if s_idx_pred is not None else None,
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
    # Aggregate phase metrics (P and S)
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

    # -------------------------------------------------------------------------
    # Final metrics dictionary
    # -------------------------------------------------------------------------
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
    # Save outputs (JSON + CSV) if requested
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

    return metrics
