from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import csv
import json

import numpy as np
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks


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

    # Detection head outputs LOGITS (Phase 0+ convention) — apply sigmoid
    # here so all downstream thresholding code (trace_threshold,
    # timestep_threshold, picker.det_threshold) works on probabilities.
    # Phase 7: P/S heads now output sigmoid in [0, 1] from the network
    # itself (model_config.yaml), so they are passed through unchanged.
    det_out = torch.sigmoid(outputs[det_key])

    return det_out, outputs[p_key], outputs[s_key]


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
# Phase picker — Phase 7: SeisConformer V7 port
# =============================================================================
#
# Direct algorithmic port of `Project File/projem/evaluation/picker.py` (the
# author's SeisConformer picker that achieves P MAE 19 ms / S MAE 85 ms on
# the same STEAD test set). Key differences from the Phase 5 union-band +
# parabolic picker:
#
#   * Smoothed detection trace + adaptive threshold (mean + k*std).
#   * Per-segment search instead of one global window — naturally handles
#     mid-event detection dips that broke the Phase 5 v2 "dominant
#     component" attempt.
#   * Multi-criteria P selection: peak height + temporal bonus for P near
#     the start of the detection (physically motivated).
#   * Hybrid S selection: peak height x (1 + gradient) x P-decay x
#     detection-window bonus x P-S-distance prior — discriminates the
#     S onset from the post-S coda.
#   * Gaussian-weighted center-of-mass refinement instead of parabolic;
#     more stable at low SNR per SC's notes.
#
# Single-event API preserved: the function returns one best (P, S) pair so
# the rest of `evaluate_model_on_loader` is unchanged.


def _smooth_1d(x: np.ndarray, sigma: float) -> np.ndarray:
    """Edge-safe Gaussian smoothing (sigma in samples)."""
    if sigma <= 0.0:
        return x.astype(np.float32, copy=False)
    return gaussian_filter1d(x.astype(np.float32), sigma=float(sigma), mode="nearest")


def _adaptive_threshold(x: np.ndarray, base_th: float, k: float) -> float:
    """SC rule: max(base_th, mean(x) + k * std(x)). Guards against
    pathologically high-variance noise traces."""
    mean_val = float(np.mean(x))
    std_val = float(np.std(x))
    return max(float(base_th), mean_val + float(k) * std_val)


def _refine_gaussian(x: np.ndarray, idx: int, win: int) -> float:
    """Probabilistic Gaussian-weighted center-of-mass refinement (SC V7).

    Returns a sub-sample position around `idx`. More stable than parabolic
    interpolation at low SNR — verified by SC on STEAD against the noisy
    far-tail of phase predictions.
    """
    n = int(len(x))
    if idx < win or idx >= n - win:
        return float(idx)
    lo = max(0, idx - win)
    hi = min(n, idx + win + 1)
    segment = x[lo:hi].astype(np.float64)
    if segment.size < 2:
        return float(idx)
    center = min(win, idx)
    positions = np.arange(segment.size, dtype=np.float64)
    weights = np.exp(-0.5 * ((positions - center) ** 2) / 1.0)
    weights /= float(np.sum(weights)) + 1e-10
    weighted = float(np.sum(positions * weights * segment))
    denom = float(np.sum(weights * segment)) + 1e-10
    refined = idx - center + (weighted / denom)
    return float(np.clip(refined, 0, n - 1))


def _find_event_segments(
    det: np.ndarray,
    threshold: float,
    adaptive_k: float,
    smooth_sigma: float,
    min_gap_samples: int,
    max_segment_samples: int,
) -> List[Tuple[int, int]]:
    """Find event segments from the smoothed detection trace using
    adaptive thresholding. Long segments get clipped around their center
    so a single huge above-threshold blob doesn't grab the whole window.
    """
    smooth_det = _smooth_1d(det, sigma=smooth_sigma)
    th = _adaptive_threshold(smooth_det, base_th=threshold, k=adaptive_k)

    above = (smooth_det > th).astype(np.int32)
    diff = np.diff(above, prepend=0, append=0)
    starts = np.where(diff == 1)[0]
    ends = np.where(diff == -1)[0]
    if len(starts) == 0 or len(ends) == 0:
        return []

    segments: List[Tuple[int, int]] = []
    for s, e in zip(starts, ends):
        s, e = int(s), int(e)
        if (e - s) > max_segment_samples:
            center = (s + e) // 2
            half = max_segment_samples // 2
            s = max(0, center - half)
            e = min(int(len(det)), center + half)
        if not segments or (s - segments[-1][1]) > min_gap_samples:
            segments.append((s, e))
        else:
            new_end = min(e, segments[-1][0] + max_segment_samples)
            segments[-1] = (segments[-1][0], new_end)
    return segments


def _multi_criteria_p_selection(
    p_smooth: np.ndarray,
    segment: Tuple[int, int],
    threshold: float,
    sample_rate: float,
    cfg: Dict[str, Any],
    edge_guard: int,
) -> Optional[int]:
    """SC V7 multi-criteria P selection: peak height + temporal bonus for
    P near detection-segment start (physically the correct prior).
    Returns global index or None.
    """
    seg_start, seg_end = segment
    if seg_start >= seg_end:
        return None

    back_sec = float(cfg.get("p_search_back_sec", 1.0))
    forward_sec = float(cfg.get("p_search_forward_sec", 10.0))
    min_dist_sec = float(cfg.get("p_min_distance_sec", 0.2))
    bonus_3 = float(cfg.get("p_temporal_bonus_3s", 0.3))
    bonus_5 = float(cfg.get("p_temporal_bonus_5s", 0.2))
    bonus_far = float(cfg.get("p_temporal_bonus_far", 0.1))

    search_start = max(0, seg_start - int(back_sec * sample_rate))
    search_end = min(len(p_smooth), seg_start + int(forward_sec * sample_rate))
    if search_start >= search_end:
        return None

    p_window = p_smooth[search_start:search_end]
    if p_window.size < 10:
        return None

    peaks, _props = find_peaks(
        p_window,
        height=float(threshold),
        distance=max(1, int(min_dist_sec * sample_rate)),
    )
    if len(peaks) == 0:
        return None

    # Apply edge guard: drop peaks too close to either trace boundary.
    valid = []
    n_total = int(len(p_smooth))
    for peak in peaks:
        gidx = int(search_start + peak)
        if gidx >= edge_guard and gidx <= n_total - edge_guard:
            valid.append(int(peak))
    if not valid:
        valid = [int(p) for p in peaks]
    peaks_arr = np.asarray(valid, dtype=int)

    scores: List[float] = []
    for peak_idx in peaks_arr:
        height_score = float(p_window[peak_idx])
        gpeak = int(search_start + peak_idx)
        dist_sec = abs(gpeak - seg_start) / sample_rate
        if dist_sec <= 3.0:
            temporal = bonus_3
        elif dist_sec <= 5.0:
            temporal = bonus_5
        else:
            temporal = bonus_far
        scores.append(height_score + temporal)

    best_local = int(peaks_arr[int(np.argmax(scores))])
    return int(search_start + best_local)


def _hybrid_s_selection(
    p_smooth: np.ndarray,
    s_smooth: np.ndarray,
    p_idx: int,
    segment: Tuple[int, int],
    threshold: float,
    sample_rate: float,
    cfg: Dict[str, Any],
) -> Optional[int]:
    """SC V7 hybrid S selection: peak height x (1 + gradient) x P-decay
    x detection-window bonus x P-S-distance prior. Helps discriminate
    the true S onset from late-coda spikes.
    Returns global index or None.
    """
    det_start, det_end = segment
    min_dist_sec = float(cfg.get("s_min_distance_sec", 0.3))
    pad_sec = float(cfg.get("s_search_pad_sec", 5.0))
    min_ps_sec = float(cfg.get("min_ps_diff_sec", 0.1))
    max_ps_sec = float(cfg.get("max_ps_diff_sec", 40.0))

    min_gap_samples = max(1, int(min_ps_sec * sample_rate))
    max_gap_samples = max(min_gap_samples + 1, int(max_ps_sec * sample_rate))

    search_start = int(p_idx + min_gap_samples)
    search_end = int(min(len(s_smooth), min(det_end + int(pad_sec * sample_rate),
                                            p_idx + max_gap_samples)))
    if search_start >= search_end:
        return None

    s_window = s_smooth[search_start:search_end]
    p_window = p_smooth[search_start:search_end]

    s_peaks, _props = find_peaks(
        s_window,
        height=float(threshold),
        distance=max(1, int(min_dist_sec * sample_rate)),
    )
    if len(s_peaks) == 0:
        return None

    s_grad = np.gradient(s_window)

    scores: List[float] = []
    for peak_idx in s_peaks:
        gidx = int(search_start + peak_idx)
        s_height = float(s_window[peak_idx])
        p_val = float(p_window[peak_idx])
        p_decay = 1.0 / (1.0 + p_val)
        dominance = max(0.0, s_height - p_val)
        grad_score = 1.0 if (peak_idx > 0 and float(s_grad[peak_idx]) > 0.0) else 0.9

        if gidx <= det_end:
            window_bonus = 1.3
        elif gidx <= det_end + int(2.0 * sample_rate):
            window_bonus = 1.1
        else:
            window_bonus = 0.95

        ps_dist_sec = (gidx - p_idx) / sample_rate
        if 0.5 <= ps_dist_sec <= 8.0:
            dist_bonus = 1.2
        elif 8.0 < ps_dist_sec <= 20.0:
            dist_bonus = 1.1
        elif 20.0 < ps_dist_sec <= 40.0:
            dist_bonus = 1.0
        elif ps_dist_sec < 0.5:
            dist_bonus = 0.4
        else:
            dist_bonus = 0.9

        score = (
            s_height
            * (p_decay ** 1.2)
            * (1.0 + 0.3 * grad_score)
            * (1.0 + 0.4 * dominance)
            * window_bonus
            * dist_bonus
        )
        scores.append(score)

    best_local = int(s_peaks[int(np.argmax(scores))])
    return int(search_start + best_local)


def pick_phases(
    det_curve: np.ndarray | torch.Tensor,
    p_curve: np.ndarray | torch.Tensor,
    s_curve: np.ndarray | torch.Tensor,
    sample_rate: float,
    picker_cfg: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Phase 7 picker: SeisConformer V7 port.

    Returns the single best (P, S) pair found across all detected event
    segments. If multiple segments yield valid picks, the one with the
    highest joint S-score wins (SC behaviour: first-event return is used
    as the canonical single-event answer).
    """
    det = _to_numpy_1d(det_curve)
    p = _to_numpy_1d(p_curve)
    s = _to_numpy_1d(s_curve)

    T = det.shape[0]
    assert p.shape[0] == T and s.shape[0] == T, "All curves must have the same length."

    det_threshold = float(picker_cfg.get("det_threshold", 0.5))
    p_threshold = float(picker_cfg.get("p_threshold", 0.25))
    s_threshold = float(picker_cfg.get("s_threshold", 0.25))

    smooth_sigma_det = float(picker_cfg.get("smooth_sigma_det", 2.0))
    smooth_sigma_p = float(picker_cfg.get("smooth_sigma_p", 1.0))
    smooth_sigma_s = float(picker_cfg.get("smooth_sigma_s", 1.2))

    adaptive_k = float(picker_cfg.get("adaptive_k", 0.3))
    min_gap_sec = float(picker_cfg.get("min_segment_gap_sec", 1.0))
    max_seg_sec = float(picker_cfg.get("max_segment_sec", 30.0))

    refine_win = int(picker_cfg.get("refine_window_samples", 3))
    edge_guard = int(picker_cfg.get("edge_guard_samples", 100))

    det_max = float(det.max()) if T > 0 else 0.0

    # Smoothed phase traces (used for both peak-finding and refinement).
    p_smooth = _smooth_1d(p, sigma=smooth_sigma_p)
    s_smooth = _smooth_1d(s, sigma=smooth_sigma_s)

    segments = _find_event_segments(
        det,
        threshold=det_threshold,
        adaptive_k=adaptive_k,
        smooth_sigma=smooth_sigma_det,
        min_gap_samples=int(min_gap_sec * sample_rate),
        max_segment_samples=int(max_seg_sec * sample_rate),
    )

    # No segments → no pick. has_event_pred=False here so downstream code
    # can distinguish "model didn't detect" from "picker rejected".
    if not segments:
        return {
            "has_event_pred": False,
            "det_max": det_max,
            "p_idx": None,
            "s_idx": None,
            "p_time": None,
            "s_time": None,
            "p_amp": None,
            "s_amp": None,
            "ps_gap_ok": None,
        }

    # Try each segment; keep the best (P, S) by S-side joint score.
    best: Optional[Dict[str, Any]] = None
    for seg in segments:
        seg_start, seg_end = seg
        if seg_start >= seg_end:
            continue

        p_seg = p_smooth[seg_start:seg_end]
        s_seg = s_smooth[seg_start:seg_end]
        if p_seg.size == 0 or s_seg.size == 0:
            continue

        p_th = _adaptive_threshold(p_seg, base_th=p_threshold, k=adaptive_k)
        s_base_th = _adaptive_threshold(s_seg, base_th=s_threshold, k=adaptive_k)
        # SC's "S a touch stricter than P" rule: keeps the S head from
        # picking up a late P-coda spike when S is genuinely weak.
        s_th = max(s_base_th, p_th + 0.05)

        p_idx_cand = _multi_criteria_p_selection(
            p_smooth, seg, p_th, sample_rate, picker_cfg, edge_guard
        )
        if p_idx_cand is None:
            continue
        if p_idx_cand < edge_guard or p_idx_cand > T - edge_guard:
            continue

        s_idx_cand = _hybrid_s_selection(
            p_smooth, s_smooth, p_idx_cand, seg, s_th, sample_rate, picker_cfg
        )
        if s_idx_cand is None or s_idx_cand <= p_idx_cand:
            continue

        ps_diff_sec = (s_idx_cand - p_idx_cand) / sample_rate
        min_ps_sec = float(picker_cfg.get("min_ps_diff_sec", 0.1))
        max_ps_sec = float(picker_cfg.get("max_ps_diff_sec", 40.0))
        if ps_diff_sec < min_ps_sec or ps_diff_sec > max_ps_sec:
            continue

        p_refined = _refine_gaussian(p_smooth, p_idx_cand, win=refine_win)
        s_refined = _refine_gaussian(s_smooth, s_idx_cand, win=refine_win)

        score = float(s_smooth[s_idx_cand]) * float(p_smooth[p_idx_cand])

        candidate = {
            "p_idx": int(round(p_refined)),
            "s_idx": int(round(s_refined)),
            "p_time": float(p_refined / sample_rate),
            "s_time": float(s_refined / sample_rate),
            "p_amp": float(p_smooth[p_idx_cand]),
            "s_amp": float(s_smooth[s_idx_cand]),
            "score": score,
        }
        if best is None or score > best["score"]:
            best = candidate

    if best is None:
        # Segments existed but every one failed P/S validation.
        return {
            "has_event_pred": True,
            "det_max": det_max,
            "p_idx": None,
            "s_idx": None,
            "p_time": None,
            "s_time": None,
            "p_amp": None,
            "s_amp": None,
            "ps_gap_ok": None,
        }

    return {
        "has_event_pred": True,
        "det_max": det_max,
        "p_idx": best["p_idx"],
        "s_idx": best["s_idx"],
        "p_time": best["p_time"],
        "s_time": best["s_time"],
        "p_amp": best["p_amp"],
        "s_amp": best["s_amp"],
        "ps_gap_ok": True,
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
    Compute phase picking metrics for a single phase (P or S).

    Phase 7 (EQTransformer Table 2/3 parity) reports:

      - num_gt, num_pred, pick_rate
      - mean_sec    : signed mean error (μ — EQT Table 2 column)
      - mae_sec     : mean absolute error
      - medae_sec   : median absolute error
      - std_sec     : standard deviation of signed error (σ — EQT Table)
      - mape        : mean absolute percentage error
                      mean(|err| / |gt_time_sec|) over predictions with
                      gt_time_sec > 1e-6 (EQT reports 0.00, so this is the
                      column we're explicitly chasing toward zero).
      - hit_rate_{small,medium,large} : |err| < tolerance (sub-sample bins)
      - tp_pick / fp_pick / fn_pick   : pick-level confusion matrix at
                                        |err| ≤ pick_tolerance_sec
                                        (TP: |err|≤tol; FP: |err|>tol;
                                        FN: gt exists but no prediction)
      - precision_pick / recall_pick / f1_pick : derived from above
      - tolerance_sec: dict including the hit-rate bins AND
                       `pick_tolerance_sec` (the F1 tolerance, 0.5 s
                       matches EQT's reporting).
    """
    assert len(gt_indices) == len(pred_indices), "Mismatched phase index lists."

    phase_tol = tolerance_cfg.get(phase_name, {})
    tol_small = float(phase_tol.get("small", 0.01))
    tol_medium = float(phase_tol.get("medium", 0.02))
    tol_large = float(phase_tol.get("large", 0.05))
    # Pick-level F1 tolerance is shared across phases (EQT-style).
    pick_tol_sec = float(tolerance_cfg.get("pick_tolerance_sec", 0.5))

    errors_sec: List[float] = []
    abs_pct_errors: List[float] = []
    picked_count = 0
    tp_pick = 0
    fp_pick = 0
    fn_pick = 0

    for gt_idx, pred_idx in zip(gt_indices, pred_indices):
        if pred_idx is None:
            # GT event with no model pick → false negative at any tolerance.
            fn_pick += 1
            continue
        picked_count += 1
        err_samples = int(pred_idx) - int(gt_idx)
        err_sec = err_samples / sample_rate
        errors_sec.append(err_sec)

        # MAPE: use absolute gt time as denominator; guard against
        # gt_time ~= 0 (first-sample picks) by skipping those.
        gt_time_sec = abs(int(gt_idx) / sample_rate)
        if gt_time_sec > 1e-6:
            abs_pct_errors.append(abs(err_sec) / gt_time_sec)

        # Pick-level F1 at the EQT tolerance.
        if abs(err_sec) <= pick_tol_sec:
            tp_pick += 1
        else:
            fp_pick += 1

    num_gt = len(gt_indices)
    num_pred = picked_count

    tolerance_block = {
        "small": tol_small,
        "medium": tol_medium,
        "large": tol_large,
        "pick_tolerance_sec": pick_tol_sec,
    }

    if len(errors_sec) == 0:
        return {
            "num_gt": int(num_gt),
            "num_pred": int(num_pred),
            "pick_rate": _format_float(_safe_div(num_pred, num_gt)),
            "mean_sec": None,
            "mae_sec": None,
            "medae_sec": None,
            "std_sec": None,
            "mape": None,
            "hit_rate_small": None,
            "hit_rate_medium": None,
            "hit_rate_large": None,
            "tp_pick": int(tp_pick),
            "fp_pick": int(fp_pick),
            "fn_pick": int(fn_pick),
            "precision_pick": _format_float(_safe_div(tp_pick, tp_pick + fp_pick)),
            "recall_pick": _format_float(_safe_div(tp_pick, tp_pick + fn_pick)),
            "f1_pick": _format_float(
                _safe_div(2.0 * tp_pick, 2.0 * tp_pick + fp_pick + fn_pick)
            ),
            "tolerance_sec": tolerance_block,
        }

    errors_sec_arr = np.asarray(errors_sec, dtype=float)
    abs_err = np.abs(errors_sec_arr)

    mean_signed = float(np.mean(errors_sec_arr))
    mae = float(np.mean(abs_err))
    medae = float(np.median(abs_err))
    std = float(np.std(errors_sec_arr))
    mape = float(np.mean(abs_pct_errors)) if abs_pct_errors else None

    hit_small = _safe_div(np.sum(abs_err < tol_small), len(abs_err))
    hit_medium = _safe_div(np.sum(abs_err < tol_medium), len(abs_err))
    hit_large = _safe_div(np.sum(abs_err < tol_large), len(abs_err))

    precision_pick = _safe_div(tp_pick, tp_pick + fp_pick)
    recall_pick = _safe_div(tp_pick, tp_pick + fn_pick)
    f1_pick = _safe_div(2.0 * tp_pick, 2.0 * tp_pick + fp_pick + fn_pick)

    return {
        "num_gt": int(num_gt),
        "num_pred": int(num_pred),
        "pick_rate": _format_float(_safe_div(num_pred, num_gt)),
        "mean_sec": _format_float(mean_signed),
        "mae_sec": _format_float(mae),
        "medae_sec": _format_float(medae),
        "std_sec": _format_float(std),
        "mape": _format_float(mape) if mape is not None else None,
        "hit_rate_small": _format_float(hit_small),
        "hit_rate_medium": _format_float(hit_medium),
        "hit_rate_large": _format_float(hit_large),
        "tp_pick": int(tp_pick),
        "fp_pick": int(fp_pick),
        "fn_pick": int(fn_pick),
        "precision_pick": _format_float(precision_pick),
        "recall_pick": _format_float(recall_pick),
        "f1_pick": _format_float(f1_pick),
        "tolerance_sec": tolerance_block,
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
