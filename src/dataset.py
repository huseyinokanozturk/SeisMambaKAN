from __future__ import annotations

import io
import json
from glob import glob
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader
import webdataset as wds
import yaml


# =============================================================================
# YAML LOADING
# =============================================================================


def load_yaml(path: str | Path) -> dict:
    """
    Load a YAML configuration file and return its content as a dictionary.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"YAML config not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# =============================================================================
# LABEL GENERATION UTILITIES
# =============================================================================


def _make_gaussian_label(
    length: int,
    center: int,
    sigma: float,
    peak: float = 1.0,
    device: torch.device | None = None,
) -> torch.Tensor:
    """
    Generate a 1D Gaussian label of given length and center index.
    If center is negative or outside [0, length), returns all zeros.
    """
    if center is None or center < 0 or center >= length:
        return torch.zeros(length, dtype=torch.float32, device=device)

    t = torch.arange(length, dtype=torch.float32, device=device)
    # Avoid division by zero if sigma is set incorrectly
    sigma = float(sigma) if sigma > 1e-6 else 1e-6
    return peak * torch.exp(-0.5 * ((t - center) / sigma) ** 2)


def make_phase_labels(
    length: int,
    p_index: int | None,
    s_index: int | None,
    cfg_labels: Dict[str, Any],
    device: torch.device | None = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generate Gaussian labels for P and S phase picking heads.

    Returns:
        y_p: shape (length,)
        y_s: shape (length,)
    """
    phase_cfg = cfg_labels["phase"]
    sigma_p = float(phase_cfg["p_sigma_samples"])
    sigma_s = float(phase_cfg["s_sigma_samples"])
    peak = float(phase_cfg["peak_value"])

    y_p = _make_gaussian_label(length, p_index, sigma_p, peak, device=device)
    y_s = _make_gaussian_label(length, s_index, sigma_s, peak, device=device)

    return y_p, y_s


def make_detection_label(
    length: int,
    p_index: int | None,
    s_index: int | None,
    cfg_labels: Dict[str, Any],
    device: torch.device | None = None,
) -> torch.Tensor:
    """
    Generate binary detection target:

        - 1 between P and S (optionally expanded by a small margin)
        - 0 elsewhere

    This target is used by the detection head to mark the "event window".
    The downstream picker can then restrict P/S search to this region.

    For noise traces (no valid P/S picks), the detection label is all zeros.
    """
    # Start with all zeros (noise case by default)
    y = torch.zeros(length, dtype=torch.float32, device=device)

    det_cfg = cfg_labels["detection"]
    margin = int(det_cfg.get("margin_samples", 0))

    # Normalize invalid indices: negative or None → treat as missing
    if p_index is None or p_index < 0:
        return y
    if s_index is None or s_index < 0:
        return y

    p = int(p_index)
    s = int(s_index)

    # Degenerate or inverted interval → keep all zeros
    if s <= p:
        return y

    # Apply margin on both sides, clamped to [0, length)
    start = max(0, p - margin)
    end = min(length, s + margin)

    if start < end:
        y[start:end] = 1.0

    return y


# =============================================================================
# AUGMENTATION PIPELINE
# =============================================================================


class SeismicAugmenter:
    """
    Data augmentation pipeline for seismic waveform data.

    All operations are controlled by the 'augmentation' section in config.yaml.
    Augmenter is designed to be pure-functional:

        x_out, p_out, s_out = augmenter(x_in, p_in, s_in, category)

    where P/S sample indices are consistently updated (e.g. after time shift),
    and secondary events are mixed in without touching the labels.
    """

    def __init__(self, cfg_aug: Dict[str, Any], total_length: int) -> None:
        self.cfg = cfg_aug
        self.total_length = int(total_length)

        self.enable = bool(cfg_aug.get("enable", True))

        # Sub-configs
        self.cfg_shift = cfg_aug.get("random_shift", {})
        self.cfg_noise = cfg_aug.get("additive_noise", {})
        self.cfg_scale = cfg_aug.get("amplitude_scale", {})
        self.cfg_channel = cfg_aug.get("channel_dropout", {})
        self.cfg_secondary = cfg_aug.get("secondary_event", {})

        # Secondary event parameters
        self.secondary_enable = bool(self.cfg_secondary.get("enable", False))
        self.secondary_prob = float(self.cfg_secondary.get("probability", 0.0))
        self.secondary_scale_min = float(self.cfg_secondary.get("scale_min", 0.0))
        self.secondary_scale_max = float(self.cfg_secondary.get("scale_max", 0.0))
        self.secondary_min_offset = int(
            self.cfg_secondary.get("min_offset_samples", 0)
        )
        self.eq_pool_max = int(self.cfg_secondary.get("pool_max", 300))

        # In-memory buffer of earthquake waveforms (T, C) as numpy arrays
        self.eq_pool: list[np.ndarray] = []

    def __call__(
        self,
        x: torch.Tensor,
        p_index: int | None,
        s_index: int | None,
        category: str | None = None,
    ) -> Tuple[torch.Tensor, int | None, int | None]:
        """
        Apply the configured augmentations to a single sample.

        Args:
            x: Tensor of shape (T, C)
            p_index: P-phase arrival sample index, or None/-1 if not available
            s_index: S-phase arrival sample index, or None/-1 if not available
            category: trace_category from metadata (e.g. 'earthquake_local', 'noise')

        Returns:
            (x_aug, p_index_aug, s_index_aug)
        """
        # Normalize indices: treat negative as None
        if p_index is not None and p_index < 0:
            p_index = None
        if s_index is not None and s_index < 0:
            s_index = None

        cat = (category or "").lower()

        # Store primary earthquake in pool BEFORE any augmentation
        if cat == "earthquake_local":
            self._maybe_add_to_pool(x)

        if not self.enable:
            return x, p_index, s_index

        # 1) Time shift
        x, p_index, s_index = self._random_shift(x, p_index, s_index)

        # 2) Amplitude scaling
        x = self._amplitude_scale(x)

        # 3) Additive noise
        x = self._additive_noise(x)

        # 4) Channel dropout
        x = self._channel_dropout(x)

        # 5) Secondary event injection (waveform only, labels untouched)
        x = self._inject_secondary_event(x, cat)

        return x, p_index, s_index

    # ------------------------------------------------------------------
    # Pool management
    # ------------------------------------------------------------------

    def _maybe_add_to_pool(self, x: torch.Tensor) -> None:
        """
        Add the current earthquake waveform to the secondary-event pool
        as a numpy array of shape (T, C). Pool size is capped.
        """
        if self.eq_pool_max <= 0:
            return

        # Detach from graph and move to CPU numpy
        x_np = x.detach().cpu().numpy().astype(np.float32)

        if len(self.eq_pool) < self.eq_pool_max:
            self.eq_pool.append(x_np)
        else:
            # Replace a random existing entry to keep pool fresh
            idx = int(np.random.randint(0, self.eq_pool_max))
            self.eq_pool[idx] = x_np

    # ------------------------------------------------------------------
    # Individual augmentation operations
    # ------------------------------------------------------------------

    def _random_shift(
        self,
        x: torch.Tensor,
        p_index: int | None,
        s_index: int | None,
    ) -> Tuple[torch.Tensor, int | None, int | None]:
        """
        Randomly shift the waveform in time by up to +/- max_shift_samples.
        Vacated samples are zero-filled (no wrap-around).
        P/S indices are shifted consistently. If shifted outside the window,
        they are set to None.
        """
        T, C = x.shape
        cfg = self.cfg_shift
        prob = float(cfg.get("prob", 0.0))
        max_shift = int(cfg.get("max_shift_samples", 0))

        if max_shift <= 0 or torch.rand(1).item() >= prob:
            return x, p_index, s_index

        shift = int(torch.randint(-max_shift, max_shift + 1, (1,)).item())
        if shift == 0:
            return x, p_index, s_index

        # Zero-padded shift (no circular roll)
        x_shifted = torch.zeros_like(x)

        if shift > 0:
            # shift to the right: [0:T-shift] -> [shift:T]
            x_shifted[shift:, :] = x[: T - shift, :]
        else:
            # shift < 0, shift to the left: [|shift|:T] -> [0:T-|shift|]
            k = -shift
            x_shifted[: T - k, :] = x[k:, :]

        def _shift_index(idx: int | None) -> int | None:
            if idx is None:
                return None
            new_idx = idx + shift
            if 0 <= new_idx < T:
                return int(new_idx)
            return None

        p_new = _shift_index(p_index)
        s_new = _shift_index(s_index)

        return x_shifted, p_new, s_new

    def _amplitude_scale(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply random global scaling to the waveform amplitude with some probability.
        """
        cfg = self.cfg_scale
        prob = float(cfg.get("prob", 0.0))
        scale_min = float(cfg.get("min", 1.0))
        scale_max = float(cfg.get("max", 1.0))

        if torch.rand(1).item() >= prob:
            return x

        if scale_max <= scale_min:
            scale = scale_min
        else:
            scale = torch.empty(1).uniform_(scale_min, scale_max).item()

        return x * scale

    def _additive_noise(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add Gaussian white noise to the waveform with a given probability.
        Noise is scaled relative to the normalized signal.
        """
        cfg = self.cfg_noise
        prob = float(cfg.get("prob", 0.0))
        std = float(cfg.get("std", 0.0))

        if std <= 0.0 or torch.rand(1).item() >= prob:
            return x

        noise = torch.randn_like(x) * std
        return x + noise

    def _channel_dropout(self, x: torch.Tensor) -> torch.Tensor:
        """
        Randomly set one or more channels to zero with some probability.
        """
        cfg = self.cfg_channel
        prob = float(cfg.get("prob", 0.0))
        choices = cfg.get("num_channels_choices", [])
        if not choices or torch.rand(1).item() >= prob:
            return x

        T, C = x.shape
        # Choose how many channels to drop
        k = int(choices[int(torch.randint(0, len(choices), (1,)).item())])
        k = max(1, min(k, C))

        # Randomly choose channel indices
        perm = torch.randperm(C)
        drop_idx = perm[:k]
        x = x.clone()
        x[:, drop_idx] = 0.0
        return x

    def _inject_secondary_event(
        self,
        x: torch.Tensor,
        category: str,
    ) -> torch.Tensor:
        """
        Mix a secondary earthquake event into the waveform.

        - Uses waveforms stored in self.eq_pool (earthquake_local only).
        - Scales the secondary event by a random factor in [scale_min, scale_max].
        - Does NOT change P/S indices or labels; secondary behaves as
          'structured noise' over the primary event.
        """
        if not self.secondary_enable:
            return x
        if category != "earthquake_local":
            return x
        if not self.eq_pool:
            return x
        if torch.rand(1).item() >= self.secondary_prob:
            return x

        # Draw scale factor
        if self.secondary_scale_max <= 0.0:
            return x

        scale_min = max(0.0, self.secondary_scale_min)
        scale_max = max(scale_min, self.secondary_scale_max)

        if scale_max <= scale_min:
            alpha = scale_min
        else:
            alpha = torch.empty(1).uniform_(scale_min, scale_max).item()

        # Sample a random earthquake from the pool
        idx = int(torch.randint(0, len(self.eq_pool), (1,)).item())
        sec_np = self.eq_pool[idx]  # shape (T, C)
        sec = torch.from_numpy(sec_np).to(x.device, dtype=x.dtype)

        # Defensive shape check (should always match)
        if sec.shape != x.shape:
            T = min(sec.shape[0], x.shape[0])
            C = min(sec.shape[1], x.shape[1])
            sec = sec[:T, :C]
            x_new = x.clone()
            x_new[:T, :C] = x_new[:T, :C] + alpha * sec
            return x_new

        return x + alpha * sec


# =============================================================================
# SAMPLE TRANSFORM FROM (npy, json) → (X, labels)
# =============================================================================


def _parse_meta_and_indices(
    meta: Dict[str, Any],
    cfg_labels: Dict[str, Any],
) -> Tuple[str, str, int | None, int | None]:
    """
    Extract trace_category and P/S indices from metadata, handling
    noise and missing picks according to labels.noise_policy.
    """
    category = str(meta.get("trace_category", "unknown"))
    trace_name = str(meta.get("trace_name", ""))

    noise_cfg = cfg_labels["noise_policy"]
    noise_p_idx = int(noise_cfg.get("p_index", -1))
    noise_s_idx = int(noise_cfg.get("s_index", -1))

    # Raw values from metadata (may be None or missing)
    raw_p = meta.get("p_arrival_sample", None)
    raw_s = meta.get("s_arrival_sample", None)

    def _to_int_or_none(v: Any) -> int | None:
        if v is None:
            return None
        try:
            iv = int(v)
        except (TypeError, ValueError):
            return None
        if iv < 0:
            return None
        return iv

    if category == "earthquake_local":
        p_idx = _to_int_or_none(raw_p)
        s_idx = _to_int_or_none(raw_s)
    else:
        # For noise (or any non-earthquake category), use configured policy
        p_idx = None if noise_p_idx < 0 else noise_p_idx
        s_idx = None if noise_s_idx < 0 else noise_s_idx

    return trace_name, category, p_idx, s_idx


def make_sample_transform(
    cfg: Dict[str, Any],
    paths_cfg: Dict[str, Any],
    is_train: bool,
):
    """
    Create a WebDataset map() transform that:
      1) converts (npy, json) into torch waveform,
      2) applies augmentations (train only),
      3) generates detection + P + S labels.

    Returns:
        transform: function(sample) -> (X, label_dict)
    """
    labels_cfg = cfg["labels"]
    aug_cfg = cfg["augmentation"]

    total_length = int(labels_cfg["total_length"])

    # Build augmentation pipeline (only used for training; disabled otherwise)
    augmenter = SeismicAugmenter(aug_cfg, total_length=total_length)
    if not is_train:
        # Force disable for val/test to avoid accidental changes
        augmenter.enable = False

    def _transform(sample: Tuple[Any, Any]):
        """
        WebDataset transform callback.

        Input:
            sample = (waveform_raw, meta_raw)
                waveform_raw: np.ndarray or bytes (.npy content)
                meta_raw: dict or bytes (JSON)

        Output:
            X: Tensor of shape (T, C)
            labels: dict with:
                - "y_det": (T,)
                - "y_p":   (T,)
                - "y_s":   (T,)
                - "category": str
                - "trace_name": str
                - "p_index": int
                - "s_index": int
        """
        waveform_raw, meta_raw = sample

        # --------------------------------------------------------------
        # Decode waveform (npy) if it's bytes
        # --------------------------------------------------------------
        if isinstance(waveform_raw, (bytes, bytearray)):
            waveform_np = np.load(io.BytesIO(waveform_raw))
        else:
            waveform_np = waveform_raw

        # --------------------------------------------------------------
        # Decode metadata (json) if it's bytes
        # --------------------------------------------------------------
        if isinstance(meta_raw, (bytes, bytearray)):
            meta = json.loads(meta_raw.decode("utf-8"))
        else:
            meta = meta_raw

        # --------------------------------------------------------------
        # Ensure correct shape: (T, C)
        # --------------------------------------------------------------
        if waveform_np.ndim == 1:
            waveform_np = waveform_np[:, None]
        elif waveform_np.ndim == 2 and waveform_np.shape[0] != total_length:
            # In case (C, T) is stored instead of (T, C), transpose
            if waveform_np.shape[1] == total_length:
                waveform_np = waveform_np.T

        T, C = waveform_np.shape
        if T != total_length:
            raise ValueError(
                f"Unexpected waveform length: {T}, expected {total_length}"
            )

        # Convert to torch tensor (CPU, float32)
        x = torch.from_numpy(waveform_np.astype(np.float32))

        # Extract category and P/S indices
        trace_name, category, p_idx, s_idx = _parse_meta_and_indices(
            meta, labels_cfg
        )

        # Apply augmentations only in training mode
        x_aug, p_idx_aug, s_idx_aug = augmenter(
            x, p_idx, s_idx, category=category
        )

        # Generate labels (still on CPU)
        y_p, y_s = make_phase_labels(
            length=total_length,
            p_index=p_idx_aug,
            s_index=s_idx_aug,
            cfg_labels=labels_cfg,
            device=None,
        )
        y_det = make_detection_label(
            length=total_length,
            p_index=p_idx_aug,
            s_index=s_idx_aug,
            cfg_labels=labels_cfg,
            device=None,
        )

        labels = {
            "y_det": y_det,
            "y_p": y_p,
            "y_s": y_s,
            "category": category,
            "trace_name": trace_name,
            "p_index": -1 if p_idx_aug is None else int(p_idx_aug),
            "s_index": -1 if s_idx_aug is None else int(s_idx_aug),
        }

        return x_aug, labels

    return _transform


# =============================================================================
# DATALOADER BUILDERS
# =============================================================================


def _get_split_dir(
    split: str,
    cfg: Dict[str, Any],
    paths_cfg: Dict[str, Any],
) -> str:
    """
    Resolve the directory containing WebDataset shards (.tar files)
    for a given split ('train', 'val', 'test') and data.mode ('all' or 'sample').
    """
    mode = cfg["data"]["mode"]
    processed_cfg = paths_cfg["processed"]

    if mode == "all":
        root = processed_cfg["all"]
    elif mode == "sample":
        root = processed_cfg["sample"]
    else:
        raise ValueError(f"Unsupported data.mode: {mode}")

    if split == "train":
        return root["train_dir"]
    if split == "val":
        return root["val_dir"]
    if split == "test":
        return root["test_dir"]

    raise ValueError(f"Unknown split: {split}")


def build_dataset(
    split: str,
    cfg: Dict[str, Any],
    paths_cfg: Dict[str, Any],
    is_train: bool,
) -> wds.WebDataset:
    """
    Build a WebDataset pipeline for a given split.

    Args:
        split: "train", "val" or "test"
        cfg: main configuration dictionary (from config.yaml)
        paths_cfg: paths configuration dictionary (from paths.yaml)
        is_train: whether this dataset will be used for training

    Returns:
        A WebDataset object producing (waveform_tensor, label_dict) pairs.
    """
    split_dir = _get_split_dir(split, cfg, paths_cfg)

    # Collect shard file paths explicitly instead of using a pattern inside WebDataset,
    # to avoid platform-specific path parsing issues.
    shard_paths = sorted(
        glob(str(Path(split_dir) / "*.tar"))
    )
    if not shard_paths:
        raise FileNotFoundError(f"No .tar shards found in split_dir={split_dir}")

    # Base WebDataset from explicit shard list
    # Disable internal shard shuffling; we control shuffling explicitly.
    dataset = wds.WebDataset(
        urls=shard_paths,
        shardshuffle=False,
    ).to_tuple("npy", "json")

    # Optional sample-level shuffle
    dl_cfg = cfg.get("dataloader", {})
    if is_train and dl_cfg.get("shuffle_train", True):
        # 10_000 is a reasonable default for shuffle buffer size
        dataset = dataset.shuffle(10_000)

    # Map to (X, labels) using our transform
    transform = make_sample_transform(cfg, paths_cfg, is_train=is_train)
    dataset = dataset.map(transform)

    return dataset


def build_dataloader(
    split: str,
    cfg: Dict[str, Any],
    paths_cfg: Dict[str, Any],
    is_train: bool,
) -> DataLoader:
    """
    Build a PyTorch DataLoader over the WebDataset-based dataset.

    Args:
        split: "train", "val" or "test".
        cfg: main configuration dictionary (from config.yaml).
        paths_cfg: paths configuration dictionary (from paths.yaml).
        is_train: whether this loader will be used for training.

    Returns:
        A PyTorch DataLoader providing:
            X: Tensor of shape (B, T, C)
            labels: dict with batched label tensors.
    """
    dataset = build_dataset(split, cfg, paths_cfg, is_train=is_train)

    train_cfg = cfg["training"]
    dl_cfg = cfg.get("dataloader", {})

    batch_size = int(train_cfg.get("batch_size", 32))
    num_workers = int(dl_cfg.get("num_workers", 4))
    pin_memory = bool(dl_cfg.get("pin_memory", True))
    persistent_workers = bool(dl_cfg.get("persistent_workers", True))
    prefetch_factor = int(dl_cfg.get("prefetch_factor", 2))
    drop_last = bool(dl_cfg.get("drop_last", is_train))

    # DataLoader kwargs (handle num_workers=0 case for prefetch_factor/persistent_workers)
    loader_kwargs = {
        "batch_size": batch_size,
        "shuffle": False,  # shuffling is already handled by WebDataset.shuffle(...)
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "drop_last": drop_last,
    }

    if num_workers > 0:
        loader_kwargs["persistent_workers"] = persistent_workers
        loader_kwargs["prefetch_factor"] = prefetch_factor

    loader = DataLoader(
        dataset,
        **loader_kwargs,
    )

    return loader
