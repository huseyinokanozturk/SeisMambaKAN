"""
Shared utilities for SeisMambaKAN.

Centralizes:
  - YAML loading
  - Environment detection (local vs Colab)
  - Path resolution (project root, configs, drive)
  - Experiment lookup (latest exp_XXX, latest checkpoint)
  - Config override application (deep merge)

Code anywhere in the project should import from here rather than
hardcoding paths or rolling its own YAML logic.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml


# =============================================================================
# YAML
# =============================================================================

def load_yaml(path: str | Path) -> dict:
    """Load a YAML file as a dict. Raises FileNotFoundError if missing."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"YAML config not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def dump_yaml(data: dict, path: str | Path) -> None:
    """Write a dict as YAML to path, creating parents as needed."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False)


# =============================================================================
# Environment detection
# =============================================================================

def is_colab() -> bool:
    """True if running inside Google Colab (detected via importlib, no import noise)."""
    import importlib.util
    return importlib.util.find_spec("google.colab") is not None


def is_drive_mounted() -> bool:
    """True if /content/drive/MyDrive exists (Colab Drive mount)."""
    return Path("/content/drive/MyDrive").exists()


# =============================================================================
# Project root + config resolution
# =============================================================================

def project_root() -> Path:
    """
    Return the SeisMambaKAN project root.

    Order:
      1. $SEISMAMBAKAN_ROOT env var if set
      2. Walk upward from this file until we find configs/paths.yaml
      3. Fallback: parent of this file's grandparent
    """
    env = os.environ.get("SEISMAMBAKAN_ROOT")
    if env:
        return Path(env).resolve()

    here = Path(__file__).resolve()
    for parent in [here.parent.parent, *here.parents]:
        if (parent / "configs" / "paths.yaml").exists():
            return parent

    return here.parent.parent


def configs_dir() -> Path:
    return project_root() / "configs"


def load_all_configs() -> Tuple[dict, dict, dict]:
    """
    Load (main_cfg, model_cfg, paths_cfg) from the project's configs/ dir.
    """
    cfg_dir = configs_dir()
    main_cfg = load_yaml(cfg_dir / "config.yaml")
    model_cfg = load_yaml(cfg_dir / "model_config.yaml")
    paths_cfg = load_yaml(cfg_dir / "paths.yaml")
    return main_cfg, model_cfg, paths_cfg


# =============================================================================
# Experiment lookup
# =============================================================================

_EXP_RE = re.compile(r"^exp_(\d+)$")


def list_experiments(exp_root: str | Path) -> List[Path]:
    """Return all exp_XXX directories under exp_root, sorted by index ascending."""
    exp_root = Path(exp_root)
    if not exp_root.exists():
        return []
    matches: List[Tuple[int, Path]] = []
    for d in exp_root.iterdir():
        if not d.is_dir():
            continue
        m = _EXP_RE.match(d.name)
        if m:
            matches.append((int(m.group(1)), d))
    matches.sort(key=lambda x: x[0])
    return [p for _, p in matches]


def resolve_experiment_dir(
    exp_root: str | Path,
    exp_id: Optional[int] = None,
) -> Optional[Path]:
    """
    Resolve an experiment directory.

      - exp_id given  -> exp_root/exp_{id:03d} (must exist).
      - exp_id None   -> latest exp_XXX under exp_root.
      - Returns None if exp_root is empty.
    """
    exp_root = Path(exp_root)
    if exp_id is not None:
        cand = exp_root / f"exp_{exp_id:03d}"
        if not cand.exists():
            raise FileNotFoundError(f"Experiment not found: {cand}")
        return cand
    experiments = list_experiments(exp_root)
    if not experiments:
        return None
    return experiments[-1]


def resolve_checkpoint(
    exp_dir: Path,
    prefer: str = "best",
) -> Path:
    """
    Find a checkpoint inside an experiment directory.

      prefer = "best": exp_dir/best_model.pth  (only model state_dict)
      prefer = "last": exp_dir/checkpoints/last.pth  (full state with optimizer)
      prefer = "auto": best_model.pth if present, else last.pth
    """
    best = exp_dir / "best_model.pth"
    last = exp_dir / "checkpoints" / "last.pth"

    if prefer == "best":
        if not best.exists():
            raise FileNotFoundError(f"best_model.pth not found in {exp_dir}")
        return best
    if prefer == "last":
        if not last.exists():
            raise FileNotFoundError(f"last.pth not found in {exp_dir}/checkpoints/")
        return last
    if prefer == "auto":
        if best.exists():
            return best
        if last.exists():
            return last
        raise FileNotFoundError(
            f"Neither best_model.pth nor checkpoints/last.pth found in {exp_dir}"
        )
    raise ValueError(f"Unknown prefer={prefer}; choose from best|last|auto")


# =============================================================================
# Override application (deep merge for config tweaks via CLI)
# =============================================================================

def deep_merge(base: dict, updates: dict) -> dict:
    """
    Recursively merge updates into base (in-place on base, also returned).
    For overlapping keys, dict values are merged; other values are replaced.
    """
    for k, v in updates.items():
        if (
            k in base
            and isinstance(base[k], dict)
            and isinstance(v, dict)
        ):
            deep_merge(base[k], v)
        else:
            base[k] = v
    return base


def apply_dotted_overrides(
    cfg: dict,
    overrides: Dict[str, Any],
) -> dict:
    """
    Apply dotted-key overrides to a nested dict.

    Example:
        apply_dotted_overrides(cfg, {"training.epochs": 5})
        sets cfg["training"]["epochs"] = 5.

    Values that are dicts/lists are assigned as-is. Missing intermediate
    dicts are created.
    """
    for key, value in overrides.items():
        parts = key.split(".")
        cur = cfg
        for p in parts[:-1]:
            if p not in cur or not isinstance(cur[p], dict):
                cur[p] = {}
            cur = cur[p]
        cur[parts[-1]] = value
    return cfg
