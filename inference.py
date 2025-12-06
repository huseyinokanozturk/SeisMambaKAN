from pathlib import Path
from typing import Optional

import numpy as np
import torch
import matplotlib.pyplot as plt

from src.dataset import build_dataloader, load_yaml
from src.models.network import SeisMambaKAN
from src.metrics import (
    pick_phases,
    _extract_heads_from_outputs,
    _extract_label_curves,
)

# ======================================================================
# User configuration
# ======================================================================

# Path to trained model weights (state_dict)
CKPT_PATH = Path(
    "/content/drive/MyDrive/Proje_SeisMamba/SeisMambaKAN/experiments/exp_005/best_model.pth"
)

# From which split to draw a sample
SPLIT = "test"   # "val" or "test"

# Fixed index inside first batch; set to None for random
FIXED_INDEX: Optional[int] = None


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
    plt.show()


# ======================================================================
# Main
# ======================================================================

def main() -> None:
    # ------------------------------------------------------------------
    # Load configs
    # ------------------------------------------------------------------
    cfg_path = Path("configs/config.yaml")
    model_cfg_path = Path("configs/model_config.yaml")
    paths_cfg_path = Path("configs/paths.yaml")

    main_cfg = load_yaml(cfg_path)
    model_cfg = load_yaml(model_cfg_path)
    paths_cfg = load_yaml(paths_cfg_path)

    metrics_cfg = main_cfg.get("metrics", {})
    sample_rate = float(metrics_cfg.get("sample_rate", 100.0))
    detection_cfg = metrics_cfg.get("detection", {})
    picker_cfg = metrics_cfg.get("picker", {})

    trace_threshold = float(detection_cfg.get("trace_threshold", 0.5))

    # ------------------------------------------------------------------
    # Device
    # ------------------------------------------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFER] Using device: {device}")

    # ------------------------------------------------------------------
    # Build model and load weights
    # ------------------------------------------------------------------
    model = SeisMambaKAN(model_cfg).to(device)

    # Safer torch.load (PyTorch >=2.1); fallback for older versions
    try:
        state_dict = torch.load(CKPT_PATH, map_location=device, weights_only=True)
    except TypeError:
        state_dict = torch.load(CKPT_PATH, map_location=device)

    model.load_state_dict(state_dict)
    model.eval()
    print(f"[INFER] Loaded weights from: {CKPT_PATH}")

    # ------------------------------------------------------------------
    # Build DataLoader (test or val)
    # ------------------------------------------------------------------
    loader = build_dataloader(
        split=SPLIT,
        cfg=main_cfg,
        paths_cfg=paths_cfg,
        is_train=False,
    )

    batch = next(iter(loader))
    x, labels = batch       # x: (B, T, C)

    B, T, C = x.shape
    print(f"[INFER] Batch shape: x = {x.shape}, split = {SPLIT}")

    # Choose example index
    if FIXED_INDEX is not None:
        idx = int(FIXED_INDEX) % B
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
    # Plot in notebook (no saving)
    # ------------------------------------------------------------------
    title = f"{SPLIT.upper()} sample idx={idx}"
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
    )


if __name__ == "__main__":
    main()
