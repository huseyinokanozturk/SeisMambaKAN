
import sys
from pathlib import Path

# Ensure project root on sys.path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import io
import json
import tarfile

import numpy as np
import matplotlib.pyplot as plt

from src.dataset import (
    load_yaml,
    make_phase_labels,
    make_detection_label,
)


def load_one_sample(split_dir: Path, category_target: str):
    """Load first sample with given trace_category from WebDataset tar shards."""
    tar_paths = sorted(split_dir.glob("*.tar"))
    if not tar_paths:
        raise FileNotFoundError(f"No .tar files found in {split_dir}")

    for tar_path in tar_paths:
        with tarfile.open(tar_path, "r") as tar:
            members = tar.getmembers()
            # we look for *.npy and matching *.json
            for m in members:
                if not m.name.endswith(".npy"):
                    continue
                stem = m.name[:-4]  # remove ".npy"
                json_member = tar.getmember(stem + ".json")
                npy_bytes = tar.extractfile(m).read()
                json_bytes = tar.extractfile(json_member).read()

                waveform_np = np.load(io.BytesIO(npy_bytes))
                meta = json.loads(json_bytes.decode("utf-8"))

                category = str(meta.get("trace_category", "unknown"))
                if category == category_target:
                    return waveform_np, meta

    raise RuntimeError(f"No sample with category={category_target} found.")


def to_int_or_none(v):
    if v is None:
        return None
    try:
        iv = int(v)
    except (TypeError, ValueError):
        return None
    if iv < 0:
        return None
    return iv


def plot_labels(waveform_np, meta, cfg, title_prefix: str):
    labels_cfg = cfg["labels"]
    sr = float(cfg["data"]["sampling_rate_hz"])
    total_length = int(labels_cfg["total_length"])

    # Ensure shape (T, C)
    if waveform_np.ndim == 1:
        waveform_np = waveform_np[:, None]
    elif waveform_np.ndim == 2 and waveform_np.shape[0] != total_length:
        if waveform_np.shape[1] == total_length:
            waveform_np = waveform_np.T

    T, C = waveform_np.shape
    assert T == total_length, f"Waveform length {T} != {total_length}"

    t = np.arange(T) / sr

    category = str(meta.get("trace_category", "unknown"))
    trace_name = str(meta.get("trace_name", ""))

    p_raw = meta.get("p_arrival_sample", None)
    s_raw = meta.get("s_arrival_sample", None)

    p_idx = to_int_or_none(p_raw)
    s_idx = to_int_or_none(s_raw)

    # Phase and detection labels
    y_p, y_s = make_phase_labels(
        length=T,
        p_index=p_idx,
        s_index=s_idx,
        cfg_labels=labels_cfg,
        device=None,
    )
    y_det = make_detection_label(
        length=T,
        p_index=p_idx,
        s_index=s_idx,
        cfg_labels=labels_cfg,
        device=None,
    )

    y_p = y_p.numpy()
    y_s = y_s.numpy()
    y_det = y_det.numpy()

    fig, axes = plt.subplots(3, 1, figsize=(14, 8), sharex=True)

    # 1) Waveform + P/S
    axes[0].set_title(
        f"{title_prefix} | trace={trace_name} | category={category} | P={p_idx}, S={s_idx}"
    )
    axes[0].plot(t, waveform_np[:, 0], label="Ch0")
    if p_idx is not None:
        axes[0].axvline(p_idx / sr, color="g", linestyle="--", label="P")
    if s_idx is not None:
        axes[0].axvline(s_idx / sr, color="r", linestyle="--", label="S")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # 2) Gaussian P and S labels
    axes[1].set_title("Gaussian labels (P and S)")
    axes[1].plot(t, y_p, label="P label", alpha=0.8)
    axes[1].plot(t, y_s, label="S label", alpha=0.8)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # 3) Detection mask
    axes[2].set_title("Detection mask (binary P–S interval)")
    axes[2].plot(t, y_det, color="orange")
    axes[2].set_ylim(-0.2, 1.2)
    axes[2].set_xlabel("Time (s)")
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


def main():
    cfg = load_yaml("configs/config.yaml")
    paths = load_yaml("configs/paths.yaml")

    mode = cfg["data"]["mode"]  # "sample" or "all"
    processed_cfg = paths["processed"][mode]
    train_dir = Path(processed_cfg["train_dir"])

    # 1) One earthquake sample
    wf_eq, meta_eq = load_one_sample(train_dir, "earthquake_local")
    plot_labels(wf_eq, meta_eq, cfg, title_prefix="Earthquake")

    # 2) One noise sample
    wf_noise, meta_noise = load_one_sample(train_dir, "noise")
    plot_labels(wf_noise, meta_noise, cfg, title_prefix="Noise")


if __name__ == "__main__":
    main()
