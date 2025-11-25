import sys
from pathlib import Path
import copy

# Make "src" importable
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import matplotlib.pyplot as plt

from src.dataset import load_yaml, build_dataloader


def configure_dataloader_cfg(cfg: dict) -> None:
    """Ensure dataloader settings are safe for Windows single-process tests."""
    dl = cfg.setdefault("dataloader", {})
    dl.setdefault("batch_size", 1)
    dl["num_workers"] = 0
    dl["pin_memory"] = False
    dl["persistent_workers"] = False
    dl.setdefault("shuffle_train", False)


def disable_augmentation(cfg: dict) -> None:
    """Disable all augmentation operations in-place."""
    if "augmentation" not in cfg:
        return
    aug = cfg["augmentation"]

    # Global switch (if present)
    aug["enable"] = False

    # For each sub-augmentation set probability/enable to zero/False if present
    for key, sub in aug.items():
        if not isinstance(sub, dict):
            continue
        if "prob" in sub:
            sub["prob"] = 0.0
        if "probability" in sub:
            sub["probability"] = 0.0
        if "enable" in sub:
            sub["enable"] = False


def detection_interval_from_mask(mask: torch.Tensor) -> tuple[int | None, int | None]:
    """Return first and last index where detection mask is > 0."""
    idx = torch.nonzero(mask > 0.0, as_tuple=False).flatten()
    if idx.numel() == 0:
        return None, None
    return int(idx[0].item()), int(idx[-1].item())


def plot_sample(x, y_p, y_s, y_det, p_idx, s_idx, sr, title: str) -> None:
    """Plot waveform (ch0) and labels for visual inspection."""
    x = np.asarray(x)
    y_p = np.asarray(y_p)
    y_s = np.asarray(y_s)
    y_det = np.asarray(y_det)

    # Ensure (T, C)
    if x.ndim != 2:
        raise ValueError(f"Expected 2D waveform, got {x.shape}")
    if x.shape[0] <= 16 and x.shape[1] > x.shape[0]:
        x = x.T

    T = x.shape[0]
    t = np.arange(T) / sr

    fig, axes = plt.subplots(3, 1, figsize=(14, 8), sharex=True)
    fig.suptitle(title)

    # Waveform ch0
    axes[0].plot(t, x[:, 0], color="black", linewidth=1.0)
    if p_idx is not None:
        axes[0].axvline(p_idx / sr, color="green", linestyle="--", label="P")
    if s_idx is not None:
        axes[0].axvline(s_idx / sr, color="red", linestyle="--", label="S")
    axes[0].legend()
    axes[0].set_ylabel("Amplitude")
    axes[0].grid(alpha=0.3)

    # Phase Gaussians
    axes[1].plot(t, y_p, label="P gaussian")
    axes[1].plot(t, y_s, label="S gaussian")
    axes[1].legend()
    axes[1].set_ylabel("Phase label")
    axes[1].grid(alpha=0.3)

    # Detection mask
    axes[2].plot(t, y_det, label="detection mask")
    axes[2].legend()
    axes[2].set_xlabel("Time (s)")
    axes[2].set_ylabel("Det")
    axes[2].grid(alpha=0.3)

    plt.tight_layout()
    plt.show()


def main():
    cfg = load_yaml("configs/config.yaml")
    paths = load_yaml("configs/paths.yaml")

    # Common dataloader settings (Windows-safe)
    configure_dataloader_cfg(cfg)

    # Copy configs: one with augmentation OFF, one as-is (ON)
    cfg_noaug = copy.deepcopy(cfg)
    disable_augmentation(cfg_noaug)

    cfg_aug = cfg  # use original (augmentation enabled according to config)

    # Build dataloaders
    train_loader_noaug = build_dataloader(
        split="train", cfg=cfg_noaug, paths_cfg=paths, is_train=False
    )
    train_loader_aug = build_dataloader(
        split="train", cfg=cfg_aug, paths_cfg=paths, is_train=True
    )

    total_len = int(cfg["labels"]["total_length"])
    sr = float(cfg["data"]["sampling_rate_hz"])

    max_checked = 20
    eq_checked = 0
    noise_checked = 0

    print("\n[FINAL SANITY] Starting full pipeline checks...\n")

    for i, ((x0, lab0), (x1, lab1)) in enumerate(
        zip(train_loader_noaug, train_loader_aug)
    ):
        if i >= max_checked:
            break

        # Batch size = 1 by configuration
        x0 = x0[0]  # (T, C) or (C, T)
        x1 = x1[0]

        # Labels from no-aug pipeline are used as reference
        y_det = lab0["y_det"][0]       # (T,)
        y_p = lab0["y_p"][0]
        y_s = lab0["y_s"][0]
        category = lab0["category"][0]
        p_index_raw = int(lab0["p_index"][0].item())
        s_index_raw = int(lab0["s_index"][0].item())
        p_idx = p_index_raw if p_index_raw >= 0 else None
        s_idx = s_index_raw if s_index_raw >= 0 else None

        # ---------- Shape checks ----------
        assert y_det.shape[0] == total_len, "y_det length mismatch"
        assert y_p.shape[0] == total_len, "y_p length mismatch"
        assert y_s.shape[0] == total_len, "y_s length mismatch"

        # Determine time dimension of waveform
        if x0.ndim != 2:
            raise AssertionError(f"Waveform must be 2D, got {x0.shape}")
        if x0.shape[0] == total_len:
            time_dim = 0
        elif x0.shape[1] == total_len:
            time_dim = 1
        else:
            raise AssertionError(
                f"Waveform shape {tuple(x0.shape)} not compatible with total_length={total_len}"
            )

        # ---------- Label consistency checks ----------
        if category == "earthquake_local":
            eq_checked += 1

            assert p_idx is not None and s_idx is not None, "Earthquake without P/S index"
            assert 0 <= p_idx < s_idx < total_len, "Invalid P/S ordering"

            # Argmax of gaussians must match P/S indices
            p_hat = int(torch.argmax(y_p).item())
            s_hat = int(torch.argmax(y_s).item())
            assert (
                p_hat == p_idx
            ), f"P gaussian peak {p_hat} != p_index {p_idx}"
            assert (
                s_hat == s_idx
            ), f"S gaussian peak {s_hat} != s_index {s_idx}"

            # Detection mask interval must cover [P, S]
            det_start, det_end = detection_interval_from_mask(y_det)
            assert det_start is not None and det_end is not None, "Detection mask empty for earthquake"
            assert det_start <= p_idx <= det_end, "Detection mask does not cover P index"
            assert det_start <= s_idx <= det_end, "Detection mask does not cover S index"

        else:
            noise_checked += 1
            # For pure noise, detection mask should be zero everywhere
            max_det = float(torch.max(torch.abs(y_det)).item())
            assert max_det == 0.0, f"Noise sample has non-zero detection mask ({max_det})"

        # ---------- Augmentation effect check ----------
        # Pipeline with augmentation should produce *some* difference
        diff = torch.mean(torch.abs(x1 - x0)).item()
        if cfg_aug.get("augmentation", {}).get("enable", True):
            # For earthquakes, we expect average difference to be > 0
            if category == "earthquake_local":
                assert diff > 0.0, "Augmentation enabled but no difference between x0 and x1"

        # Stop when we have checked enough examples of each type
        if eq_checked >= 5 and noise_checked >= 5:
            break

    print(f"[FINAL SANITY] Checked {eq_checked} earthquake and {noise_checked} noise samples successfully.")

    # -------- Optional visual check: plot a couple of earthquakes --------
    print("[FINAL SANITY] Plotting 2 earthquake examples for visual inspection...")

    plotted = 0
    for (x0, lab0) in train_loader_noaug:
        x0 = x0[0]
        y_det = lab0["y_det"][0]
        y_p = lab0["y_p"][0]
        y_s = lab0["y_s"][0]
        category = lab0["category"][0]
        p_index_raw = int(lab0["p_index"][0].item())
        s_index_raw = int(lab0["s_index"][0].item())
        p_idx = p_index_raw if p_index_raw >= 0 else None
        s_idx = s_index_raw if s_index_raw >= 0 else None

        if category != "earthquake_local":
            continue

        x_np = x0.numpy()
        plot_sample(
            x_np,
            y_p.numpy(),
            y_s.numpy(),
            y_det.numpy(),
            p_idx,
            s_idx,
            sr,
            title=f"Earthquake example #{plotted+1}",
        )
        plotted += 1
        if plotted >= 2:
            break

    print("[FINAL SANITY] All tests and plots completed.\n")


if __name__ == "__main__":
    main()
