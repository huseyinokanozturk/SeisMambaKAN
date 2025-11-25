from __future__ import annotations

import argparse
import io
import json
import logging
from pathlib import Path
import tarfile
from typing import Dict, Iterator, Optional

import h5py
import numpy as np
import pandas as pd
import yaml
from sklearn.model_selection import train_test_split
from tqdm.auto import tqdm


# -----------------------------------------------------------
# LOGGING CONFIGURATION
# -----------------------------------------------------------

logger = logging.getLogger("seismambakan.preprocess")
if not logger.handlers:
    _h = logging.StreamHandler()
    _f = logging.Formatter("[%(levelname)s] %(message)s")
    _h.setFormatter(_f)
    logger.addHandler(_h)
logger.setLevel(logging.INFO)


# -----------------------------------------------------------
# YAML CONFIGURATION HELPERS
# -----------------------------------------------------------

def load_yaml(path: str | Path) -> dict:
    """Load and parse YAML configuration file."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"YAML file not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# -----------------------------------------------------------
# CSV TO METADATA PROCESSING
# -----------------------------------------------------------

def read_chunked_csvs(
    csv_root: str,
    pattern: str = "chunk*.csv",
    nrows: Optional[int] = None,
) -> pd.DataFrame:
    """
    Read STEAD-style chunked CSV files and merge them into a single DataFrame.
    Adds a 'chunk_stem' column to each row (e.g., 'chunk1').
    
    Args:
        csv_root: Root directory containing CSV chunks
        pattern: Glob pattern to match CSV files
        nrows: Optional limit on number of rows to read per file
        
    Returns:
        Combined DataFrame with all chunks
    """
    csv_dir = Path(csv_root).resolve()
    paths = sorted(csv_dir.glob(pattern))
    if not paths:
        raise FileNotFoundError(f"No CSV files found: {csv_dir}/{pattern}")

    frames = []
    for p in paths:
        df = pd.read_csv(p, nrows=nrows)
        df["chunk_stem"] = p.stem  # e.g., "chunk1"
        frames.append(df)

    meta = pd.concat(frames, ignore_index=True)
    logger.info(f"[CSV] Total rows: {len(meta):,} | Number of files: {len(paths)}")
    return meta


def sample_by_category(
    df: pd.DataFrame,
    quotas: Dict[str, int],
    category_col: str = "trace_category",
    shuffle: bool = True,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Sample a specified number of examples from each category.
    
    Args:
        df: Input DataFrame
        quotas: Dictionary mapping category names to sample counts
                e.g., {"earthquake_local": 1000, "noise": 300}
        category_col: Column name containing categories
        shuffle: Whether to shuffle the results
        random_state: Random seed for reproducibility
        
    Returns:
        DataFrame with sampled examples from each category
    """
    parts = []
    for cat, k in quotas.items():
        sub = df[df[category_col] == cat]
        if shuffle:
            sub = sub.sample(frac=1.0, random_state=random_state)
        parts.append(sub.head(k))
    out = pd.concat(parts, ignore_index=True)
    if shuffle:
        out = out.sample(frac=1.0, random_state=random_state).reset_index(drop=True)
    logger.info(
        "[Sample] " +
        ", ".join(f"{c}={int(sum(out[category_col]==c))}" for c in quotas)
    )
    return out


# -----------------------------------------------------------
# METADATA CLEANING (P-S WAVE VALIDATION)
# -----------------------------------------------------------

def clean_anomalies(df: pd.DataFrame, params: Optional[dict] = None) -> pd.DataFrame:
    """
    Clean earthquake data by validating P-S wave timing constraints.
    Only processes earthquake_local traces:
    - Checks P-S time difference (in seconds)
    - Validates that P and S samples are within the time window
    
    Args:
        df: Input DataFrame containing seismic traces
        params: Dictionary with cleaning parameters:
                - min_ps: Minimum P-S time difference in seconds
                - max_ps: Maximum P-S time difference in seconds  
                - window_min: Minimum valid sample index
                - window_max: Maximum valid sample index
                
    Returns:
        Cleaned DataFrame with valid earthquake traces
    """
    params = params or {}
    min_ps = params.get("min_ps", 1.0)
    max_ps = params.get("max_ps", 31.0)
    window_min = params.get("window_min", 0)
    window_max = params.get("window_max", 6000)

    df_eq = df[df["trace_category"] == "earthquake_local"].copy()
    df_eq = df_eq.dropna(subset=["p_arrival_sample", "s_arrival_sample"])

    df_eq["p_s_diff_samples"] = df_eq["s_arrival_sample"] - df_eq["p_arrival_sample"]
    df_eq["p_s_diff_seconds"] = df_eq["p_s_diff_samples"] / 100.0  # STEAD sampling rate = 100 Hz

    mask = (
        (df_eq["p_s_diff_seconds"] > min_ps)
        & (df_eq["p_s_diff_seconds"] < max_ps)
        & (df_eq["p_arrival_sample"].between(window_min, window_max))
        & (df_eq["s_arrival_sample"].between(window_min, window_max))
    )
    df_clean = df_eq[mask].copy()
    logger.info(
        f"[Clean] EQ initial={len(df_eq):,} | remaining clean={len(df_clean):,} "
        f"({100 * len(df_clean) / max(len(df_eq),1):.1f}%)"
    )
    return df_clean


def split_dataset(
    meta_df: pd.DataFrame,
    split_cfg: dict,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Perform stratified train/validation/test split on the dataset.
    
    Args:
        meta_df: Input metadata DataFrame
        split_cfg: Dictionary containing split ratios
        random_state: Random seed for reproducibility
        
    Returns:
        Tuple of (train_df, val_df, test_df)
    """
    test_size = float(split_cfg["test"])
    val_size = float(split_cfg["val"])

    train_val_df, test_df = train_test_split(
        meta_df,
        test_size=test_size,
        stratify=meta_df["trace_category"],
        random_state=random_state,
    )

    relative_val = val_size / (1.0 - test_size)
    train_df, val_df = train_test_split(
        train_val_df,
        test_size=relative_val,
        stratify=train_val_df["trace_category"],
        random_state=random_state,
    )

    logger.info(
        f"[Split] Train={len(train_df):,}, Val={len(val_df):,}, Test={len(test_df):,}"
    )
    return train_df, val_df, test_df


# -----------------------------------------------------------
# WAVEFORM READING FROM HDF5 FILES
# -----------------------------------------------------------

def read_waveform(handle: h5py.File, trace_name: str, dataset_key: str = "waveform") -> np.ndarray:
    """
    Flexible waveform reader compatible with various STEAD HDF5 schemas.
    Attempts different schema patterns:
    - f[trace_name][dataset_key]
    - f['data'][trace_name][dataset_key]
    - f['data'][trace_name]
    - f[trace_name]
    
    Args:
        handle: Open HDF5 file handle
        trace_name: Name of the trace to read
        dataset_key: Key for the waveform dataset (default: "waveform")
        
    Returns:
        Waveform array with shape (time_samples, channels)
    """
    # Schema A: Top-level group
    if trace_name in handle:
        grp = handle[trace_name]
        if isinstance(grp, h5py.Dataset):
            arr = grp[()]
        elif dataset_key in grp:
            arr = grp[dataset_key][()]
        else:
            raise KeyError(f"Dataset '{dataset_key}' not found in '{trace_name}'.")
        if arr.ndim == 1:
            arr = arr[:, None]
        return arr

    # Schema B: Under "data" group
    if "data" in handle and isinstance(handle["data"], h5py.Group):
        data_grp = handle["data"]
        if trace_name in data_grp:
            sub = data_grp[trace_name]
            if isinstance(sub, h5py.Dataset):
                arr = sub[()]
            elif dataset_key in sub:
                arr = sub[dataset_key][()]
            else:
                raise KeyError(
                    f"Dataset '{dataset_key}' not found in data/'{trace_name}'."
                )
            if arr.ndim == 1:
                arr = arr[:, None]
            return arr

        # Sometimes traces have suffixes like 'trace_name_0'
        matches = [k for k in data_grp.keys() if k.startswith(str(trace_name))]
        if matches:
            sub = data_grp[matches[0]]
            if isinstance(sub, h5py.Dataset):
                arr = sub[()]
            elif dataset_key in sub:
                arr = sub[dataset_key][()]
            else:
                raise KeyError(
                    f"Dataset '{dataset_key}' not found in data/'{matches[0]}'."
                )
            if arr.ndim == 1:
                arr = arr[:, None]
            return arr

    # Schema C: Direct dataset
    if trace_name in handle and isinstance(handle[trace_name], h5py.Dataset):
        arr = handle[trace_name][()]
        if arr.ndim == 1:
            arr = arr[:, None]
        return arr

    raise KeyError(f"Trace '{trace_name}' not found in HDF5 file.")


def iter_waveforms_by_chunk(
    meta: pd.DataFrame,
    hdf5_root: str,
    dataset_key: str = "waveform",
    trace_col: str = "trace_name",
    chunk_col: str = "chunk_stem",
) -> Iterator[dict]:
    """
    Iterator that reads waveforms for each row in the metadata DataFrame.
    Determines which HDF5 file to read from based on the chunk_stem.
    
    Args:
        meta: Metadata DataFrame with trace information
        hdf5_root: Root directory containing HDF5 files
        dataset_key: Key for waveform datasets in HDF5 files
        trace_col: Column name containing trace identifiers
        chunk_col: Column name containing chunk identifiers
        
    Yields:
        Dictionary with keys: 'trace_name', 'waveform', 'row'
    """
    h5_dir = Path(hdf5_root).resolve()
    all_files = {}
    # First try .hdf5 extension, then .h5
    for p in sorted(h5_dir.glob("*.hdf5")):
        all_files[p.stem] = p
    if not all_files:
        for p in sorted(h5_dir.glob("*.h5")):
            all_files[p.stem] = p
    if not all_files:
        raise FileNotFoundError(f"No HDF5 files found in: {h5_dir}")

    handles: Dict[str, h5py.File] = {}

    def get_handle(stem: str) -> Optional[h5py.File]:
        """Get or create HDF5 file handle for the given chunk stem."""
        if stem in handles:
            return handles[stem]
        path = all_files.get(stem)
        if path is None:
            logger.warning(f"[HDF5] No file found for chunk_stem='{stem}', skipping.")
            return None
        try:
            f = h5py.File(path, "r")
            handles[stem] = f
            return f
        except Exception as e:
            logger.warning(f"[HDF5] Could not open {path}: {e}")
            return None

    try:
        for _, row in meta.iterrows():
            trace = row[trace_col]
            stem = row[chunk_col]

            f = get_handle(stem)
            found = False
            if f is not None:
                try:
                    x = read_waveform(f, trace, dataset_key=dataset_key)
                    yield {"trace_name": trace, "waveform": x, "row": row}
                    continue
                except KeyError:
                    pass

            # Fallback: search in other files
            for other_stem, p in all_files.items():
                if other_stem == stem:
                    continue
                f2 = get_handle(other_stem)
                if f2 is None:
                    continue
                try:
                    x = read_waveform(f2, trace, dataset_key=dataset_key)
                    yield {"trace_name": trace, "waveform": x, "row": row}
                    found = True
                    break
                except KeyError:
                    continue
            if not found:
                logger.warning(f"[HDF5] Trace '{trace}' not found in any file, skipping.")
    finally:
        for f in handles.values():
            try:
                f.close()
            except Exception:
                pass


# -----------------------------------------------------------
# WAVEFORM NORMALIZATION
# -----------------------------------------------------------

def zscore_normalize_signal(x: np.ndarray) -> np.ndarray:
    """Apply z-score normalization to signal data."""
    mean = np.mean(x, axis=0)
    std = np.std(x, axis=0) + 1e-6
    return (x - mean) / std


def normalize_waveform(
    x: np.ndarray,
    method: str = "zscore",
) -> np.ndarray:
    """
    Normalize waveform data using the specified method.
    
    Args:
        x: Input waveform array
        method: Normalization method ("zscore", "maxabs", or "none")
        
    Returns:
        Normalized waveform array
    """
    if method is None or str(method).lower() in ("none", "null"):
        return x
    m = str(method).lower()
    if m == "zscore":
        return zscore_normalize_signal(x)
    elif m == "maxabs":
        max_abs = np.max(np.abs(x), axis=0) + 1e-6
        return x / max_abs
    else:
        raise ValueError(f"Unknown normalization method: {method}")


# -----------------------------------------------------------
# WEBDATASET SHARD WRITER
# -----------------------------------------------------------

class WebDatasetShardWriter:
    """
    WebDataset shard writer for creating training-ready data archives.
    Each sample is stored as: {key}.npy + {key}.json
    Shard filenames follow the pattern: pattern.format(shard_index) (e.g., train_{:06d}.tar)
    """
    def __init__(
        self,
        out_dir: Path,
        pattern: str,
        max_samples_per_shard: int = 2048,
    ) -> None:
        """
        Initialize the shard writer.
        
        Args:
            out_dir: Output directory for shard files
            pattern: Filename pattern with format placeholder for shard index
            max_samples_per_shard: Maximum number of samples per shard file
        """
        self.out_dir = Path(out_dir)
        self.pattern = pattern
        self.max_samples_per_shard = max_samples_per_shard

        self.shard_idx = 0
        self.sample_idx = 0
        self.shard_sample_count = 0
        self.tar: Optional[tarfile.TarFile] = None

        self._open_new_shard()

    def _open_new_shard(self) -> None:
        """Open a new shard file for writing."""
        if self.tar is not None:
            self.tar.close()

        filename = self.pattern.format(self.shard_idx)
        shard_path = self.out_dir / filename
        shard_path.parent.mkdir(parents=True, exist_ok=True)

        logger.info(f"[WebDataset] Opening shard: {shard_path}")
        self.tar = tarfile.open(shard_path, mode="w")
        self.shard_sample_count = 0
        self.shard_idx += 1

    def add_sample(self, waveform: np.ndarray, meta: dict) -> None:
        """
        Add a sample to the current shard.
        
        Args:
            waveform: Waveform data array
            meta: Metadata dictionary for the sample
        """
        if self.tar is None:
            raise RuntimeError("Tar file is not open.")

        key = f"{self.sample_idx:08d}"

        # Save waveform as .npy
        buf = io.BytesIO()
        np.save(buf, waveform.astype(np.float32))
        buf.seek(0)
        npy_bytes = buf.getbuffer()

        npy_info = tarfile.TarInfo(name=f"{key}.npy")
        npy_info.size = len(npy_bytes)
        self.tar.addfile(npy_info, io.BytesIO(npy_bytes))

        # Save metadata as .json
        meta_bytes = json.dumps(meta, ensure_ascii=False).encode("utf-8")
        json_info = tarfile.TarInfo(name=f"{key}.json")
        json_info.size = len(meta_bytes)
        self.tar.addfile(json_info, io.BytesIO(meta_bytes))

        self.sample_idx += 1
        self.shard_sample_count += 1

        if self.shard_sample_count >= self.max_samples_per_shard:
            self._open_new_shard()

    def close(self) -> None:
        """Close the current shard and finalize writing."""
        if self.tar is not None:
            self.tar.close()
            logger.info(f"[WebDataset] Shard writing completed. Total samples={self.sample_idx}")
            self.tar = None


# -----------------------------------------------------------
# DATASET SPLIT WRITER
# -----------------------------------------------------------

def write_split_to_webdataset(
    df: pd.DataFrame,
    split_name: str,
    paths_cfg: dict,
    preprocessing_cfg: dict,
    webdataset_cfg: dict,
) -> None:
    """
    Write a dataset split to WebDataset shards.
    
    Args:
        df: DataFrame containing the split data
        split_name: Name of the split ("train", "val", or "test")
        paths_cfg: Path configuration dictionary
        preprocessing_cfg: Preprocessing configuration dictionary
        webdataset_cfg: WebDataset configuration dictionary
    """
    if len(df) == 0:
        logger.warning(f"[Write] Split '{split_name}' is empty, skipping.")
        return

    raw_hdf5_root = paths_cfg["data"]["raw_hdf5_dir"]
    processed_cfg = paths_cfg["processed"]
    wd_cfg = webdataset_cfg

    mode = preprocessing_cfg["_mode"]

    if split_name == "train":
        out_dir_str = (
            processed_cfg["all"]["train_dir"]
            if mode == "all"
            else processed_cfg["sample"]["train_dir"]
        )
        pattern = wd_cfg["train_pattern"]
    elif split_name == "val":
        out_dir_str = (
            processed_cfg["all"]["val_dir"]
            if mode == "all"
            else processed_cfg["sample"]["val_dir"]
        )
        pattern = wd_cfg["val_pattern"]
    elif split_name == "test":
        out_dir_str = (
            processed_cfg["all"]["test_dir"]
            if mode == "all"
            else processed_cfg["sample"]["test_dir"]
        )
        pattern = wd_cfg["test_pattern"]
    else:
        raise ValueError(f"Unknown split: {split_name}")

    out_dir = Path(out_dir_str)
    shard_size = int(wd_cfg.get("shard_size", 2048))
    norm_method = preprocessing_cfg.get("normalization", "zscore")

    writer = WebDatasetShardWriter(
        out_dir=out_dir,
        pattern=pattern,
        max_samples_per_shard=shard_size,
    )

    gen = iter_waveforms_by_chunk(
        df,
        hdf5_root=raw_hdf5_root,
        dataset_key="waveform",
        trace_col="trace_name",
        chunk_col="chunk_stem",
    )

    logger.info(
        f"[Write] Split='{split_name}' | samples={len(df):,} | "
        f"shard_size={shard_size} | normalization='{norm_method}'"
    )

    for item in tqdm(gen, total=len(df), desc=f"{split_name}"):
        x = item["waveform"]  # Shape: (time_samples, channels)
        row = item["row"]

        x = normalize_waveform(x, method=norm_method)

        # Convert P/S indices to JSON format (NaN becomes None)
        p_arr = row.get("p_arrival_sample", np.nan)
        s_arr = row.get("s_arrival_sample", np.nan)

        p_val = None if pd.isna(p_arr) else int(p_arr)
        s_val = None if pd.isna(s_arr) else int(s_arr)

        meta = {
            "trace_name": row.get("trace_name"),
            "trace_category": row.get("trace_category", "unknown"),
            "p_arrival_sample": p_val,
            "s_arrival_sample": s_val,
            "chunk_stem": row.get("chunk_stem", None),
        }

        # Optional: add extra columns if present
        for extra_col in ("source_id", "station", "network", "magnitude"):
            if isinstance(row, pd.Series) and extra_col in row.index:
                val = row[extra_col]
                if isinstance(val, (np.floating, float)) and pd.isna(val):
                    val = None
                meta[extra_col] = val

        writer.add_sample(x, meta)

    writer.close()


# -----------------------------------------------------------
# MAIN PREPROCESSING PIPELINE
# -----------------------------------------------------------

def main() -> None:
    """Main preprocessing pipeline for converting STEAD data to WebDataset format."""
    parser = argparse.ArgumentParser(
        description="STEAD → WebDataset preprocessing script (Mamba+KAN compatible)"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/config.yaml",
        help="Path to config.yaml file",
    )
    parser.add_argument(
        "--paths",
        type=str,
        default="configs/paths.yaml",
        help="Path to paths.yaml file",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["all", "sample"],
        default=None,
        help="Override the data.mode value from config",
    )

    args = parser.parse_args()

    cfg = load_yaml(args.config)
    paths_cfg = load_yaml(args.paths)

    data_cfg = cfg["data"]
    cleaning_cfg = cfg["cleaning"]
    split_cfg = cfg["split"]
    preprocessing_cfg = cfg.get("preprocessing", {})
    webdataset_cfg = paths_cfg["webdataset"]

    mode = args.mode or data_cfg.get("mode", "all")
    if mode not in ("all", "sample"):
        raise ValueError(f"Invalid mode: {mode}")
    preprocessing_cfg = dict(preprocessing_cfg)  # Create a copy
    preprocessing_cfg["_mode"] = mode

    logger.info(f"[Config] mode={mode}")
    logger.info(f"[Config] data={data_cfg}")
    logger.info(f"[Config] cleaning={cleaning_cfg}")
    logger.info(f"[Config] split={split_cfg}")
    logger.info(f"[Config] paths={paths_cfg}")

    # 1) Read CSV metadata
    raw_csv_dir = paths_cfg["data"]["raw_csv_dir"]
    meta_df = read_chunked_csvs(raw_csv_dir, pattern="chunk*.csv")

    use_noise = bool(data_cfg.get("use_noise", True))
    if use_noise:
        meta_df = meta_df[meta_df["trace_category"].isin(["earthquake_local", "noise"])]
    else:
        meta_df = meta_df[meta_df["trace_category"] == "earthquake_local"]
    meta_df = meta_df.reset_index(drop=True)

    counts = meta_df["trace_category"].value_counts().to_dict()
    eq_total = int(counts.get("earthquake_local", 0))
    noise_total = int(counts.get("noise", 0))
    logger.info(f"[Meta] Initial counts: EQ={eq_total:,}, NOISE={noise_total:,}")

    random_state = int(data_cfg.get("random_state", 42))

    # 2) Handle "all" vs "sample" mode
    if mode == "sample":
        sample_size = int(data_cfg["sample_size"])
        eq_ratio = float(data_cfg.get("eq_ratio", 0.7))

        if not use_noise:
            desired_eq = sample_size
            desired_noise = 0
        else:
            desired_eq = int(sample_size * eq_ratio)
            desired_noise = sample_size - desired_eq

        eq_count = min(desired_eq, eq_total)
        noise_count = min(desired_noise, noise_total) if use_noise else 0

        quotas = {"earthquake_local": eq_count}
        if use_noise and noise_count > 0:
            quotas["noise"] = noise_count

        logger.info(
            f"[Mode=sample] Target total={sample_size} "
            f"(EQ={desired_eq}, NOISE={desired_noise}) → "
            f"Clamped: EQ={eq_count}, NOISE={noise_count}"
        )

        meta_sampled = sample_by_category(
            meta_df,
            quotas=quotas,
            category_col="trace_category",
            shuffle=True,
            random_state=random_state,
        )
    else:
        logger.info("[Mode=all] Using all available valid rows.")
        meta_sampled = meta_df

    # 3) Clean anomalies for earthquake data
    clean_params = {
        "min_ps": float(cleaning_cfg["min_ps_seconds"]),
        "max_ps": float(cleaning_cfg["max_ps_seconds"]),
        "window_min": int(cleaning_cfg["window_samples"]["min"]),
        "window_max": int(cleaning_cfg["window_samples"]["max"]),
    }
    logger.info(f"[Clean] Parameters: {clean_params}")

    meta_eq_clean = clean_anomalies(meta_sampled, params=clean_params)
    if use_noise:
        meta_noise = meta_sampled[meta_sampled["trace_category"] != "earthquake_local"]
        meta_clean = pd.concat([meta_eq_clean, meta_noise], ignore_index=True)
    else:
        meta_clean = meta_eq_clean

    meta_clean = meta_clean.sample(
        frac=1.0, random_state=random_state
    ).reset_index(drop=True)

    logger.info(
        "[Meta] After cleaning: total={:,} (EQ={}, NOISE={})".format(
            len(meta_clean),
            int(sum(meta_clean["trace_category"] == "earthquake_local")),
            int(sum(meta_clean["trace_category"] == "noise")),
        )
    )

    # 4) Split into train/validation/test sets
    train_df, val_df, test_df = split_dataset(
        meta_clean,
        split_cfg=split_cfg,
        random_state=random_state,
    )

    # 5) Write each split to WebDataset shards
    for split_name, df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        write_split_to_webdataset(
            df=df,
            split_name=split_name,
            paths_cfg=paths_cfg,
            preprocessing_cfg=preprocessing_cfg,
            webdataset_cfg=webdataset_cfg,
        )

    logger.info("[DONE] Preprocessing + WebDataset export completed successfully.")


if __name__ == "__main__":
    main()
