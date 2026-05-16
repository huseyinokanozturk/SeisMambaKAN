"""
Drive <-> Colab sync utilities.

Used by run.py for:
  - `run.py data`  -> pull processed data from Drive to Colab
  - `run.py push`  -> push an experiment (or results) from Colab to Drive

Standalone usage:
  python -m scripts.sync_drive pull-data --mode all
  python -m scripts.sync_drive push-experiment --exp 7
  python -m scripts.sync_drive push-results --exp 7
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

# Allow running both as a module (-m scripts.sync_drive) and as a script.
_THIS = Path(__file__).resolve()
_ROOT = _THIS.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.utils import (  # noqa: E402
    is_drive_mounted,
    load_yaml,
    project_root,
)


# =============================================================================
# Path helpers
# =============================================================================

def _paths_cfg() -> dict:
    return load_yaml(project_root() / "configs" / "paths.yaml")


def _drive_paths() -> dict:
    return _paths_cfg().get("drive", {})


def _ensure_drive() -> bool:
    if not is_drive_mounted():
        print("[ERR] Drive not mounted. Run `from google.colab import drive; drive.mount('/content/drive')`.")
        return False
    return True


# =============================================================================
# Copy with progress (rsync-like but pure Python)
# =============================================================================

def _copy_tree(
    src: Path,
    dst: Path,
    desc: str,
    refresh: bool = False,
) -> int:
    """Copy src -> dst recursively. Returns number of files copied."""
    if not src.exists():
        print(f"[WARN] Source missing: {src}")
        return 0

    if refresh and dst.exists():
        print(f"[INFO] Removing existing {dst}")
        shutil.rmtree(dst, ignore_errors=True)

    dst.mkdir(parents=True, exist_ok=True)

    files = [p for p in src.rglob("*") if p.is_file()]
    total = len(files)
    if total == 0:
        print(f"[WARN] No files under {src}")
        return 0

    # Pre-create subdirs to avoid mkdir races between parallel workers.
    for f in files:
        (dst / f.relative_to(src)).parent.mkdir(parents=True, exist_ok=True)

    from concurrent.futures import ThreadPoolExecutor, as_completed
    n_workers = 8

    def _copy_one(f: Path) -> None:
        shutil.copy2(f, dst / f.relative_to(src))

    try:
        from tqdm.auto import tqdm
        pbar = tqdm(total=total, desc=desc, unit="file")
    except ImportError:
        pbar = None

    with ThreadPoolExecutor(max_workers=n_workers) as pool:
        futures = [pool.submit(_copy_one, f) for f in files]
        for fut in as_completed(futures):
            fut.result()
            if pbar is not None:
                pbar.update(1)

    if pbar is not None:
        pbar.close()

    return total


# =============================================================================
# Public ops
# =============================================================================

def pull_data(mode: str = "all", refresh: bool = False) -> int:
    """Drive -> Colab: processed data for the given mode ('all' | 'sample')."""
    if mode == "none":
        print("[INFO] mode='none' -> nothing to pull.")
        return 0
    if not _ensure_drive():
        return 0
    drive = _drive_paths()
    src = Path(drive["data_dir"]) / mode
    dst = project_root() / "data" / "processed" / mode
    print(f"[INFO] pull-data mode={mode}: {src} -> {dst}")
    return _copy_tree(src, dst, desc=f"data:{mode}", refresh=refresh)


def push_experiment(exp_id: int) -> int:
    """Colab -> Drive: mirror experiments/exp_{id:03d} (full)."""
    if not _ensure_drive():
        return 0
    drive = _drive_paths()
    name = f"exp_{exp_id:03d}"
    src = project_root() / "experiments" / name
    dst = Path(drive["experiments_dir"]) / name
    if not src.exists():
        print(f"[ERR] No such local experiment: {src}")
        return 0
    print(f"[INFO] push-experiment: {src} -> {dst}")
    # We do not refresh (would wipe Drive history); we overlay.
    return _copy_tree(src, dst, desc=f"exp_{exp_id:03d}", refresh=False)


def push_results(exp_id: int) -> int:
    """Colab -> Drive: mirror results/exp_{id:03d} (eval / inference outputs)."""
    if not _ensure_drive():
        return 0
    drive = _drive_paths()
    name = f"exp_{exp_id:03d}"
    src = project_root() / "results" / name
    dst = Path(drive["results_dir"]) / name
    if not src.exists():
        print(f"[ERR] No such local results dir: {src}")
        return 0
    print(f"[INFO] push-results: {src} -> {dst}")
    return _copy_tree(src, dst, desc=f"results_{exp_id:03d}", refresh=False)


def pull_experiment(exp_id: int, refresh: bool = False) -> int:
    """Drive -> Colab: bring a finished experiment back to local."""
    if not _ensure_drive():
        return 0
    drive = _drive_paths()
    name = f"exp_{exp_id:03d}"
    src = Path(drive["experiments_dir"]) / name
    dst = project_root() / "experiments" / name
    if not src.exists():
        print(f"[ERR] No such Drive experiment: {src}")
        return 0
    print(f"[INFO] pull-experiment: {src} -> {dst}")
    return _copy_tree(src, dst, desc=f"pull exp_{exp_id:03d}", refresh=refresh)


# =============================================================================
# CLI (for `python -m scripts.sync_drive ...`)
# =============================================================================

def _cli() -> int:
    p = argparse.ArgumentParser(description="Drive <-> Colab sync")
    sub = p.add_subparsers(dest="cmd", required=True)

    p_pd = sub.add_parser("pull-data", help="Drive -> Colab: processed data")
    p_pd.add_argument("--mode", choices=["all", "sample"], default="all")
    p_pd.add_argument("--refresh", action="store_true")

    p_pe = sub.add_parser("push-experiment", help="Colab -> Drive: an experiment")
    p_pe.add_argument("--exp", type=int, required=True)

    p_pr = sub.add_parser("push-results", help="Colab -> Drive: eval/inference results")
    p_pr.add_argument("--exp", type=int, required=True)

    p_pe2 = sub.add_parser("pull-experiment", help="Drive -> Colab: an experiment")
    p_pe2.add_argument("--exp", type=int, required=True)
    p_pe2.add_argument("--refresh", action="store_true")

    args = p.parse_args()

    if args.cmd == "pull-data":
        n = pull_data(mode=args.mode, refresh=args.refresh)
    elif args.cmd == "push-experiment":
        n = push_experiment(exp_id=args.exp)
    elif args.cmd == "push-results":
        n = push_results(exp_id=args.exp)
    elif args.cmd == "pull-experiment":
        n = pull_experiment(exp_id=args.exp, refresh=args.refresh)
    else:
        p.print_help()
        return 1

    print(f"[OK] copied {n} files.")
    return 0


if __name__ == "__main__":
    sys.exit(_cli())
