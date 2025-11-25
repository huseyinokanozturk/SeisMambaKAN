# scripts/sanity_dataset_secondary_event.py

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt
import numpy as np

from src.dataset import load_yaml, build_dataloader


def plot_pair(original, augmented, p_idx, s_idx, det_mask, sr, i):
    """Plot original vs augmented waveform + detection mask."""
    original = np.asarray(original)
    augmented = np.asarray(augmented)
    det_mask = np.asarray(det_mask).reshape(-1)

    # Ensure (T, C)
    if original.ndim != 2:
        raise ValueError(f"Expected 2D waveform, got {original.shape}")
    if original.shape[0] <= 16 and original.shape[1] > original.shape[0]:
        original = original.T
    if augmented.shape[0] <= 16 and augmented.shape[1] > augmented.shape[0]:
        augmented = augmented.T

    T, C = original.shape
    t = np.arange(T) / sr

    if det_mask.shape[0] != T:
        raise ValueError(
            f"det_mask length {det_mask.shape[0]} != waveform length {T}"
        )

    fig, axes = plt.subplots(2, 1, figsize=(16, 8), sharex=True)
    fig.suptitle(f"Secondary EQ Test #{i+1}")

    # ----- ORIGINAL -----
    axes[0].plot(t, original[:, 0], color="black", linewidth=1.2, label="original")
    if p_idx is not None:
        axes[0].axvline(p_idx / sr, color="green", linestyle="--", label="P")
    if s_idx is not None:
        axes[0].axvline(s_idx / sr, color="red", linestyle="--", label="S")
    axes[0].set_title("Original waveform")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    # ----- AUGMENTED -----
    axes[1].plot(t, augmented[:, 0], color="blue", linewidth=1.0, label="augmented")
    if p_idx is not None:
        axes[1].axvline(p_idx / sr, color="green", linestyle="--", label="P")
    if s_idx is not None:
        axes[1].axvline(s_idx / sr, color="red", linestyle="--", label="S")

    scale = np.max(np.abs(original[:, 0])) or 1.0
    axes[1].plot(t, det_mask * scale, color="orange", label="det mask (scaled)")

    axes[1].set_title("Augmented waveform (secondary EQ injected)")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.show()


def main():
    cfg = load_yaml("configs/config.yaml")
    paths = load_yaml("configs/paths.yaml")

    # --------- DATALOADER SETTINGS (WINDOWS SAFE) ----------
    if "dataloader" not in cfg:
        cfg["dataloader"] = {}
    cfg["dataloader"]["num_workers"] = 0      # no multiprocessing → no pickling
    cfg["dataloader"]["pin_memory"] = False
    cfg["dataloader"]["persistent_workers"] = False
    cfg["dataloader"]["shuffle_train"] = False
    # -------------------------------------------------------

    # --------- AUGMENTATION: ONLY SECONDARY EVENT ON -------
    aug = cfg["augmentation"]
    aug["secondary_event"]["enable"] = True
    aug["secondary_event"]["probability"] = 1.0
    aug["secondary_event"]["scale_min"] = 0.9
    aug["secondary_event"]["scale_max"] = 1.0

    aug["random_shift"]["prob"] = 0.0
    aug["additive_noise"]["prob"] = 0.0
    aug["amplitude_scale"]["prob"] = 0.0
    aug["channel_dropout"]["prob"] = 0.0
    # -------------------------------------------------------

    # 1) No-augmentation config (secondary OFF)
    cfg_noaug = yaml_deepcopy(cfg)
    cfg_noaug["augmentation"]["enable"] = False
    # Secondary explicitly kapansın:
    cfg_noaug["augmentation"]["secondary_event"]["enable"] = False

    # 2) With-secondary config (secondary ON)
    cfg_aug = cfg  # zaten yukarıda secondary açık

    # Build loaders
    train_loader_noaug = build_dataloader(
        "train", cfg_noaug, paths, is_train=False
    )
    train_loader_aug = build_dataloader(
        "train", cfg_aug, paths, is_train=True
    )

    sr = float(cfg["data"]["sampling_rate_hz"])

    for i, ((x_orig_b, labels_orig), (x_aug_b, labels_aug)) in enumerate(
        zip(train_loader_noaug, train_loader_aug)
    ):
        if i >= 3:  # sadece ilk 3 örnek
            break

        # Batch'ten ilk örneği al
        x_orig = x_orig_b[0].numpy()  # (T, C)
        x_aug = x_aug_b[0].numpy()    # (T, C)

        # Labels dict içerisindeki anahtarlar:
        # 'y_det', 'y_p', 'y_s', 'category', 'trace_name', 'p_index', 's_index'
        det_mask = labels_orig["y_det"][0].numpy()

        p_raw = int(labels_orig["p_index"][0].item())
        s_raw = int(labels_orig["s_index"][0].item())
        p_idx = p_raw if p_raw >= 0 else None
        s_idx = s_raw if s_raw >= 0 else None

        plot_pair(x_orig, x_aug, p_idx, s_idx, det_mask, sr, i)


def yaml_deepcopy(cfg: dict) -> dict:
    """Cheap deep copy via yaml dump/load to avoid importing copy."""
    import yaml as _yaml

    return _yaml.safe_load(_yaml.safe_dump(cfg))


if __name__ == "__main__":
    main()
