"""
Inference / visualization for a single trace.

Resolution rules mirror evaluate.py:
  --ckpt PATH    explicit
  --exp ID       experiments/exp_{ID:03d}/best_model.pth
  (default)      latest experiments/exp_*

Output:
  Always plots inline. If --save (default) is on, also writes the figure to
  results/exp_{ID:03d}/inference/{split}_idx{N}.png and mirrors to Drive
  results dir when mounted.
"""

from __future__ import annotations

import argparse
import re
import shutil
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import matplotlib.pyplot as plt

from src.dataset import build_dataloader
from src.models.network import SeisMambaKAN
from src.metrics import (
    pick_phases,
    _extract_heads_from_outputs,
    _extract_label_curves,
)
from src.utils import (
    is_drive_mounted,
    load_all_configs,
    project_root,
    resolve_checkpoint,
    resolve_experiment_dir,
)


# ======================================================================
# Plotting helper
# ======================================================================

def plot_single_trace_in_notebook(
    waveform: np.ndarray,
    det_true: np.ndarray,
    det_pred: np.ndarray,
    p_idx_true: Optional[int],
    s_idx_true: Optional[int],
    p_idx_pred: Optional[int],
    s_idx_pred: Optional[int],
    sample_rate: float,
    title: str,
    trace_threshold: float,
    save_path: Optional[Path] = None,
    show: bool = True,
) -> None:
    """
    Plot a single seismic trace with all information overlaid:

    - waveform (normalized, channel 0)
    - detection GT window (shaded)
    - detection Pred window (shaded, threshold-based)
    - P / S GT and Pred picks as vertical lines
    """
    T = waveform.shape[0]
    t = np.arange(T) / sample_rate

    # Normalize waveform for visualization
    w = waveform.astype(float)
    w = w / (np.max(np.abs(w)) + 1e-8)

    fig, ax = plt.subplots(1, 1, figsize=(12, 5))

    # 1) Raw waveform
    ax.plot(t, w, linewidth=0.8, label="Waveform (ch0)")

    # 2) Detection GT window
    if det_true.max() > 0.0:
        ax.fill_between(
            t,
            -1.2,
            1.2,
            where=(det_true > 0.5),
            alpha=0.15,
            label="Det GT window",
        )

    # 3) Detection Pred window
    det_pred_mask = det_pred >= trace_threshold
    if det_pred_mask.any():
        ax.fill_between(
            t,
            -1.2,
            1.2,
            where=det_pred_mask,
            alpha=0.15,
            color="orange",
            label="Det Pred window",
        )

    # 4) P / S picks (GT vs Pred)
    # P GT
    if p_idx_true is not None:
        ax.axvline(
            p_idx_true / sample_rate,
            color="green",
            linestyle="-",
            linewidth=1.2,
            label="P GT",
        )
    # P Pred
    if p_idx_pred is not None:
        ax.axvline(
            p_idx_pred / sample_rate,
            color="green",
            linestyle="--",
            linewidth=1.2,
            label="P Pred",
        )

    # S GT
    if s_idx_true is not None:
        ax.axvline(
            s_idx_true / sample_rate,
            color="red",
            linestyle="-",
            linewidth=1.2,
            label="S GT",
        )
    # S Pred
    if s_idx_pred is not None:
        ax.axvline(
            s_idx_pred / sample_rate,
            color="red",
            linestyle="--",
            linewidth=1.2,
            label="S Pred",
        )

    ax.set_title(title)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Normalized amplitude")
    ax.set_ylim(-1.2, 1.2)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right", fontsize=8)

    fig.tight_layout()
    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150)
        print(f"[INFER] Saved figure: {save_path}")
    if show:
        plt.show()
    plt.close(fig)


# ======================================================================
# CLI
# ======================================================================

def _parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Single-trace inference + plot")
    g = p.add_mutually_exclusive_group()
    g.add_argument("--exp", type=int, default=None, help="Experiment id (3-digit), else latest.")
    g.add_argument("--ckpt", type=str, default=None, help="Explicit checkpoint path.")
    p.add_argument("--split", choices=["val", "test"], default="test")
    p.add_argument("--index", type=int, default=None,
                   help="Sample index within first batch. Default: deterministic random.")
    p.add_argument("--prefer", choices=["best", "last", "auto"], default="best")
    p.add_argument("--no-save", action="store_true",
                   help="Do not save the figure to results/.")
    p.add_argument("--no-show", action="store_true",
                   help="Do not call plt.show() (useful in headless mode).")
    p.add_argument("--no-drive-mirror", action="store_true")
    return p.parse_args(argv)


def _infer_exp_id_from_path(path: Path) -> Optional[int]:
    for part in path.parts:
        m = re.match(r"^exp_(\d+)$", part)
        if m:
            return int(m.group(1))
    return None


def _mirror_to_drive(local_dir: Path, drive_root: str, exp_id: int) -> None:
    if not is_drive_mounted() or not drive_root:
        return
    drive_target = Path(drive_root) / f"exp_{exp_id:03d}"
    drive_target.mkdir(parents=True, exist_ok=True)
    files = [p for p in local_dir.rglob("*") if p.is_file()]
    for f in files:
        rel = f.relative_to(local_dir)
        out = drive_target / rel
        out.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(f, out)


# ======================================================================
# Main
# ======================================================================

def main(argv: Optional[list[str]] = None) -> None:
    args = _parse_args(argv)

    # ------------------------------------------------------------------
    # Load configs
    # ------------------------------------------------------------------
    main_cfg, model_cfg, paths_cfg = load_all_configs()

    metrics_cfg = main_cfg.get("metrics", {})
    sample_rate = float(metrics_cfg.get("sample_rate", 100.0))
    detection_cfg = metrics_cfg.get("detection", {})
    picker_cfg = metrics_cfg.get("picker", {})
    trace_threshold = float(detection_cfg.get("trace_threshold", 0.5))

    # ------------------------------------------------------------------
    # Resolve checkpoint + exp id
    # ------------------------------------------------------------------
    exp_root = project_root() / paths_cfg.get("experiments", {}).get("root_dir", "experiments")

    if args.ckpt:
        ckpt_path = Path(args.ckpt)
        if not ckpt_path.exists():
            raise FileNotFoundError(ckpt_path)
        exp_id = _infer_exp_id_from_path(ckpt_path)
    else:
        exp_dir = resolve_experiment_dir(exp_root, exp_id=args.exp)
        if exp_dir is None:
            raise FileNotFoundError(
                f"No experiments under {exp_root}. Train first, or pass --ckpt."
            )
        ckpt_path = resolve_checkpoint(exp_dir, prefer=args.prefer)
        exp_id = _infer_exp_id_from_path(exp_dir)

    # ------------------------------------------------------------------
    # Device
    # ------------------------------------------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFER] Using device: {device}")
    print(f"[INFER] Checkpoint: {ckpt_path}")
    print(f"[INFER] Exp id: {exp_id}, split: {args.split}")

    # ------------------------------------------------------------------
    # Build model and load weights
    # ------------------------------------------------------------------
    model = SeisMambaKAN(model_cfg).to(device)

    try:
        state_dict = torch.load(ckpt_path, map_location=device, weights_only=True)
    except TypeError:
        state_dict = torch.load(ckpt_path, map_location=device)

    # Accept both raw state_dict and full-state checkpoint
    if isinstance(state_dict, dict) and "model_state_dict" in state_dict:
        state_dict = state_dict["model_state_dict"]

    model.load_state_dict(state_dict)
    model.eval()

    # ------------------------------------------------------------------
    # DataLoader
    # ------------------------------------------------------------------
    loader = build_dataloader(
        split=args.split,
        cfg=main_cfg,
        paths_cfg=paths_cfg,
        is_train=False,
    )

    batch = next(iter(loader))
    x, labels = batch       # x: (B, T, C)

    B, T, C = x.shape
    print(f"[INFER] Batch shape: x = {x.shape}, split = {args.split}")

    # Choose example index
    if args.index is not None:
        idx = int(args.index) % B
    else:
        rng = np.random.RandomState(123)
        idx = int(rng.randint(0, B))

    print(f"[INFER] Using sample index {idx} within the first batch.")

    # ------------------------------------------------------------------
    # Prepare tensors and forward pass
    # ------------------------------------------------------------------
    # Keep a CPU copy for waveform plotting
    x_cpu = x.detach().cpu().numpy()
    waveform_i = x_cpu[idx, :, 0]   # use channel 0 as representative waveform

    x = x.to(device, non_blocking=True)

    # Model expects (B, C, T)
    if x.dim() == 3:  # (B, T, C)
        x_model = x.permute(0, 2, 1).contiguous()
    else:
        x_model = x

    device_type = device.type
    with torch.no_grad(), torch.amp.autocast(device_type=device_type, enabled=False):
        outputs = model(x_model)

    # Extract heads
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

    # Extract labels
    det_true, p_gauss_true, s_gauss_true, p_idx_true_tensor, s_idx_true_tensor = (
        _extract_label_curves(labels)
    )

    # ------------------------------------------------------------------
    # Select chosen sample
    # ------------------------------------------------------------------
    det_true_i = det_true[idx, 0]    # (T,)
    det_pred_i = det_pred[idx, 0]    # (T,)
    p_true_i = p_gauss_true[idx, 0]
    s_true_i = s_gauss_true[idx, 0]

    # Ground-truth event flag
    has_gt_event = bool(
        (det_true_i.max() >= 0.5)
        or (p_true_i.max() > 0.0)
        or (s_true_i.max() > 0.0)
    )

    # Ground-truth P/S indices – only if there is an event & gauss > 0
    if has_gt_event and p_true_i.max() > 0.0:
        if p_idx_true_tensor is not None:
            p_idx_true: Optional[int] = int(p_idx_true_tensor[idx].item())
        else:
            p_idx_true = int(np.argmax(p_true_i))
    else:
        p_idx_true = None

    if has_gt_event and s_true_i.max() > 0.0:
        if s_idx_true_tensor is not None:
            s_idx_true: Optional[int] = int(s_idx_true_tensor[idx].item())
        else:
            s_idx_true = int(np.argmax(s_true_i))
    else:
        s_idx_true = None

    # ------------------------------------------------------------------
    # Predicted event flag (trace-level)
    # ------------------------------------------------------------------
    det_max_pred = float(det_pred_i.max())
    has_event_pred = det_max_pred >= trace_threshold

    # If model says "no event", do NOT search for P/S
    if has_event_pred:
        p_pred_i = p_pred[idx, 0]
        s_pred_i = s_pred[idx, 0]

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

    # ------------------------------------------------------------------
    # Numeric summary
    # ------------------------------------------------------------------
    print("\n[INFER] === Numeric summary for selected trace ===")
    print(f"GT event flag       : {has_gt_event}")
    print(f"Pred event flag     : {has_event_pred} (det_max={det_max_pred:.4f}, thr={trace_threshold})")
    print(f"det_true: min={det_true_i.min():.4f}, max={det_true_i.max():.4f}")
    print(f"det_pred: min={det_pred_i.min():.4f}, max={det_pred_i.max():.4f}")
    print(f"p_true max: {p_true_i.max():.4f}")
    print(f"s_true max: {s_true_i.max():.4f}")

    print("\n[INFER] Ground-truth picks:")
    if p_idx_true is not None:
        print(f"  P_true index = {p_idx_true}, time = {p_idx_true / sample_rate:.3f} s")
    else:
        print("  P_true = None (no P label)")

    if s_idx_true is not None:
        print(f"  S_true index = {s_idx_true}, time = {s_idx_true / sample_rate:.3f} s")
    else:
        print("  S_true = None (no S label)")

    print("\n[INFER] Predicted picks:")
    if has_event_pred:
        if p_idx_pred is not None:
            print(f"  P_pred index = {p_idx_pred}, time = {p_idx_pred / sample_rate:.3f} s")
        else:
            print("  P_pred = None")

        if s_idx_pred is not None:
            print(f"  S_pred index = {s_idx_pred}, time = {s_idx_pred / sample_rate:.3f} s")
        else:
            print("  S_pred = None")
    else:
        print("  Model did NOT detect an event → P/S not computed.")

    # ------------------------------------------------------------------
    # Plot
    # ------------------------------------------------------------------
    title = f"{args.split.upper()} sample idx={idx}"

    save_path: Optional[Path] = None
    if not args.no_save:
        results_cfg = paths_cfg.get("results", {})
        results_root = project_root() / results_cfg.get("root_dir", "results")
        exp_bucket = f"exp_{exp_id:03d}" if exp_id is not None else "exp_unknown"
        out_dir = results_root / exp_bucket / "inference"
        out_dir.mkdir(parents=True, exist_ok=True)
        save_path = out_dir / f"{args.split}_idx{idx}.png"

    plot_single_trace_in_notebook(
        waveform=waveform_i,
        det_true=det_true_i,
        det_pred=det_pred_i,
        p_idx_true=p_idx_true,
        s_idx_true=s_idx_true,
        p_idx_pred=p_idx_pred,
        s_idx_pred=s_idx_pred,
        sample_rate=sample_rate,
        title=title,
        trace_threshold=trace_threshold,
        save_path=save_path,
        show=not args.no_show,
    )

    # Mirror to Drive
    if save_path is not None and exp_id is not None and not args.no_drive_mirror:
        drive_root = paths_cfg.get("results", {}).get("drive_root_dir")
        if drive_root:
            _mirror_to_drive(save_path.parent.parent, drive_root, exp_id)
            print(f"[INFER] Mirrored to {Path(drive_root) / f'exp_{exp_id:03d}'}")


if __name__ == "__main__":
    main()
