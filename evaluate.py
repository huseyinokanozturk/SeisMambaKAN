"""
Evaluation entry point.

Resolves checkpoint and writes metrics deterministically:

  - Checkpoint:
      --ckpt PATH    (explicit; overrides everything)
      --exp ID       (integer; uses experiments/exp_{ID:03d}/best_model.pth)
      otherwise      auto-pick the latest experiments/exp_*

  - Output:
      Always writes to results/exp_{ID:03d}/{split}/{metrics,plots,...}.
      If Drive is mounted, additionally mirrors that folder to
      paths.yaml -> results.drive_root_dir/exp_{ID:03d}/.

Typical use via run.py:

    python run.py eval                      # latest exp, split=val
    python run.py eval --exp 7 --split test
    python run.py eval --ckpt path/to.pth   # custom checkpoint
"""

from __future__ import annotations

import argparse
import re
import shutil
from pathlib import Path
from typing import Optional

import torch

from src.dataset import build_dataloader
from src.metrics import evaluate_model_on_loader
from src.models.network import SeisMambaKAN
from src.utils import (
    is_drive_mounted,
    load_all_configs,
    project_root,
    resolve_checkpoint,
    resolve_experiment_dir,
)


def _parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate SeisMambaKAN checkpoint")
    g = p.add_mutually_exclusive_group()
    g.add_argument("--exp", type=int, default=None, help="Experiment id (3-digit), else latest.")
    g.add_argument("--ckpt", type=str, default=None, help="Explicit checkpoint path.")
    p.add_argument("--split", choices=["val", "test"], default="val")
    p.add_argument("--prefer", choices=["best", "last", "auto"], default="best")
    p.add_argument("--no-drive-mirror", action="store_true",
                   help="Skip mirroring results to Drive even if mounted.")
    return p.parse_args(argv)


def _infer_exp_id_from_path(path: Path) -> Optional[int]:
    """Try to recover an exp_XXX id from a checkpoint path."""
    for part in path.parts:
        m = re.match(r"^exp_(\d+)$", part)
        if m:
            return int(m.group(1))
    return None


def _mirror_to_drive(local_dir: Path, drive_root: str, exp_id: int) -> None:
    """Copy local_dir into drive_root/exp_{id:03d}/ if Drive is mounted."""
    if not is_drive_mounted():
        return
    if not drive_root:
        return
    drive_target = Path(drive_root) / f"exp_{exp_id:03d}"
    drive_target.mkdir(parents=True, exist_ok=True)
    files = [p for p in local_dir.rglob("*") if p.is_file()]
    if not files:
        return
    print(f"[Eval] mirroring {len(files)} files -> {drive_target}")
    for f in files:
        rel = f.relative_to(local_dir)
        out = drive_target / rel
        out.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(f, out)


def main(argv: Optional[list[str]] = None) -> None:
    args = _parse_args(argv)

    main_cfg, model_cfg, paths_cfg = load_all_configs()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ----- resolve checkpoint + exp id -------------------------------------
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

    print(f"[Eval] checkpoint: {ckpt_path}")
    print(f"[Eval] exp_id:     {exp_id}")
    print(f"[Eval] split:      {args.split}")
    print(f"[Eval] device:     {device}")

    # ----- build model -----------------------------------------------------
    model = SeisMambaKAN(model_cfg).to(device)
    state = torch.load(ckpt_path, map_location=device)
    # accept both raw state_dict and full-state checkpoint
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    model.load_state_dict(state)
    model.eval()

    # ----- build loader ----------------------------------------------------
    loader = build_dataloader(
        split=args.split,
        cfg=main_cfg,
        paths_cfg=paths_cfg,
        is_train=False,
    )

    # ----- output dir (local) ---------------------------------------------
    results_cfg = paths_cfg.get("results", {})
    results_root = project_root() / results_cfg.get("root_dir", "results")
    if exp_id is None:
        # unknown -> bucket under exp_unknown so we never silently overwrite
        out_dir = results_root / "exp_unknown" / args.split
    else:
        out_dir = results_root / f"exp_{exp_id:03d}" / args.split
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[Eval] results -> {out_dir}")

    # ----- evaluate --------------------------------------------------------
    metrics = evaluate_model_on_loader(
        model=model,
        data_loader=loader,
        device=device,
        main_cfg=main_cfg,
        split_name=args.split,
        exp_dir=out_dir,  # metrics.py creates out_dir/<output_subdir>/ inside
    )

    # ----- mirror to Drive -------------------------------------------------
    drive_root = results_cfg.get("drive_root_dir")
    if exp_id is not None and drive_root and not args.no_drive_mirror:
        _mirror_to_drive(
            local_dir=out_dir.parent,  # results/exp_XXX (both splits)
            drive_root=drive_root,
            exp_id=exp_id,
        )

    # ----- short summary ---------------------------------------------------
    print("\n[Eval] summary")
    for k, v in metrics.items():
        if isinstance(v, (int, float)):
            print(f"  {k:30s} = {v}")


if __name__ == "__main__":
    main()
