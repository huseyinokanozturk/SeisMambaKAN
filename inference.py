from pathlib import Path
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
# Configuration (EDIT HERE IF NEEDED)
# ======================================================================

# Path to trained model weights (state_dict)
CKPT_PATH = Path(
    "/content/drive/MyDrive/Proje_SeisMamba/SeisMambaKAN/experiments/exp_005/best_model.pth"
)

# Which split to draw a random trace from
SPLIT = "test"   # or "val"

# Index of a fixed example (set to None to use random index)
FIXED_INDEX = None


def plot_single_trace_in_notebook(
    det_true: np.ndarray,
    det_pred: np.ndarray,
    p_true: np.ndarray,
    p_pred: np.ndarray,
    s_true: np.ndarray,
    s_pred: np.ndarray,
    p_idx_true: int,
    s_idx_true: int,
    p_idx_pred: int | None,
    s_idx_pred: int | None,
    sample_rate: float,
    title: str = "Inference Example",
) -> None:
    """
    Plot detection, P and S curves for a single trace and show in notebook.
    No file is saved; this is purely for interactive inspection.
    """
    T = det_true.shape[0]
    t = np.arange(T) / sample_rate

    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

    # Detection
    ax = axes[0]
    ax.plot(t, det_true, label="Detection (GT)", linewidth=1.0)
    ax.plot(t, det_pred, label="Detection (Pred)", linewidth=1.0, linestyle="--")
    ax.set_ylabel("Det prob")
    ax.set_title(title)
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
    plt.show()


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

    # ------------------------------------------------------------------
    # Device
    # ------------------------------------------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFER] Using device: {device}")

    # ------------------------------------------------------------------
    # Build model and load weights
    # ------------------------------------------------------------------
    model = SeisMambaKAN(model_cfg).to(device)

    # Load state_dict (weights_only=True for safety; requires newish PyTorch)
    state_dict = torch.load(CKPT_PATH, map_location=device, weights_only=True)
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

    # Get a single batch
    batch = next(iter(loader))
    x, labels = batch

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
    # Move to device and run model
    # ------------------------------------------------------------------
    x = x.to(device, non_blocking=True)

    # Model expects (B, C, T)
    if x.dim() == 3:  # (B, T, C)
        x_model = x.permute(0, 2, 1).contiguous()
    else:
        x_model = x

    device_type = device.type
    with torch.no_grad(), torch.amp.autocast(device_type=device_type, enabled=False):
        outputs = model(x_model)

    # Extract heads (detection, P, S) robustly using metrics helpers
    det_out, p_out, s_out = _extract_heads_from_outputs(outputs)

    det_pred = det_out.detach().cpu().numpy()  # (B, 1, T) or (B, T)
    p_pred = p_out.detach().cpu().numpy()
    s_pred = s_out.detach().cpu().numpy()

    # Ensure shape (B, 1, T)
    if det_pred.ndim == 2:
        det_pred = det_pred[:, None, :]
    if p_pred.ndim == 2:
        p_pred = p_pred[:, None, :]
    if s_pred.ndim == 2:
        s_pred = s_pred[:, None, :]

    # Extract labels (det_true, p_gauss_true, s_gauss_true, p_idx_true, s_idx_true)
    det_true, p_gauss_true, s_gauss_true, p_idx_true_tensor, s_idx_true_tensor = (
        _extract_label_curves(labels)
    )

    # ------------------------------------------------------------------
    # Select chosen sample
    # ------------------------------------------------------------------
    det_true_i = det_true[idx, 0]   # (T,)
    det_pred_i = det_pred[idx, 0]   # (T,)
    p_true_i = p_gauss_true[idx, 0]
    p_pred_i = p_pred[idx, 0]
    s_true_i = s_gauss_true[idx, 0]
    s_pred_i = s_pred[idx, 0]

    # Ground-truth P/S indices
    if p_idx_true_tensor is not None:
        p_idx_true = int(p_idx_true_tensor[idx].item())
    else:
        p_idx_true = int(np.argmax(p_true_i))

    if s_idx_true_tensor is not None:
        s_idx_true = int(s_idx_true_tensor[idx].item())
    else:
        s_idx_true = int(np.argmax(s_true_i))

    # Phase picking on predicted curves
    picker_cfg = metrics_cfg.get("picker", {})
    pick_result = pick_phases(
        det_curve=det_pred_i,
        p_curve=p_pred_i,
        s_curve=s_pred_i,
        sample_rate=sample_rate,
        picker_cfg=picker_cfg,
    )

    p_idx_pred = pick_result["p_idx"]
    s_idx_pred = pick_result["s_idx"]

    # ------------------------------------------------------------------
    # Print numeric summary
    # ------------------------------------------------------------------
    print("\n[INFER] === Numeric summary for selected trace ===")
    print(f"det_true: shape={det_true_i.shape}, min={det_true_i.min():.4f}, max={det_true_i.max():.4f}")
    print(f"det_pred: shape={det_pred_i.shape}, min={det_pred_i.min():.4f}, max={det_pred_i.max():.4f}")
    print(f"p_true  : shape={p_true_i.shape},   min={p_true_i.min():.4f}, max={p_true_i.max():.4f}")
    print(f"p_pred  : shape={p_pred_i.shape},   min={p_pred_i.min():.4f}, max={p_pred_i.max():.4f}")
    print(f"s_true  : shape={s_true_i.shape},   min={s_true_i.min():.4f}, max={s_true_i.max():.4f}")
    print(f"s_pred  : shape={s_pred_i.shape},   min={s_pred_i.min():.4f}, max={s_pred_i.max():.4f}")

    print("\n[INFER] Ground-truth picks:")
    print(f"  P_true index = {p_idx_true}, time = {p_idx_true / sample_rate:.3f} s")
    print(f"  S_true index = {s_idx_true}, time = {s_idx_true / sample_rate:.3f} s")

    print("\n[INFER] Predicted picks:")
    if p_idx_pred is not None:
        print(f"  P_pred index = {p_idx_pred}, time = {p_idx_pred / sample_rate:.3f} s")
    else:
        print("  P_pred = None")

    if s_idx_pred is not None:
        print(f"  S_pred index = {s_idx_pred}, time = {s_idx_pred / sample_rate:.3f} s")
    else:
        print("  S_pred = None")

    # ------------------------------------------------------------------
    # Plot in notebook (no saving)
    # ------------------------------------------------------------------
    title = f"{SPLIT.upper()} sample idx={idx}"
    plot_single_trace_in_notebook(
        det_true=det_true_i,
        det_pred=det_pred_i,
        p_true=p_true_i,
        p_pred=p_pred_i,
        s_true=s_true_i,
        s_pred=s_pred_i,
        p_idx_true=p_idx_true,
        s_idx_true=s_idx_true,
        p_idx_pred=p_idx_pred,
        s_idx_pred=s_idx_pred,
        sample_rate=sample_rate,
        title=title,
    )


if __name__ == "__main__":
    main()
