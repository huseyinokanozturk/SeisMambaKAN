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
from typing import Any, Optional

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
    p.add_argument("--data-mode", choices=["all", "sample"], default=None,
                   help="Override data.mode from config.yaml.")
    p.add_argument("--from-sweep", action="store_true",
                   help="Load best picker/detection thresholds from "
                        "results/exp_NNN/<split>/threshold_sweep.json "
                        "(produced by `run.py sweep`).")
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


def _safe_torch_load(path: Path, device: torch.device) -> Any:
    """
    torch.load wrapper that prefers weights_only=True (PyTorch >=2.1).
    Falls back to weights_only=False for older versions or for full-state
    checkpoints that contain non-tensor objects (e.g. optimizer state).
    """
    try:
        return torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=device)
    except Exception:
        return torch.load(path, map_location=device, weights_only=False)


def _print_metrics_summary(metrics: dict) -> None:
    """Pretty-print key scalar metrics, recursing into nested dicts."""
    def _walk(obj: Any, prefix: str = "") -> None:
        if isinstance(obj, dict):
            for k, v in obj.items():
                _walk(v, f"{prefix}{k}." if prefix else f"{k}.")
        elif isinstance(obj, (int, float)) and not isinstance(obj, bool):
            label = prefix.rstrip(".")
            print(f"  {label:40s} = {obj:.6f}" if isinstance(obj, float)
                  else f"  {label:40s} = {obj}")
    if not metrics:
        print("  (empty metrics dict)")
        return
    _walk(metrics)


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
    if args.data_mode is not None:
        main_cfg.setdefault("data", {})["mode"] = args.data_mode
        print(f"[Eval] data.mode override: {args.data_mode}")
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

    # ----- optional: load best thresholds from a prior sweep run ----------
    if args.from_sweep:
        results_cfg_pre = paths_cfg.get("results", {})
        results_root_pre = project_root() / results_cfg_pre.get("root_dir", "results")
        if exp_id is None:
            raise SystemExit(
                "[Eval] --from-sweep needs an exp id; pass --exp or rerun without "
                "--ckpt so we can infer it from the experiment dir."
            )
        sweep_json = results_root_pre / f"exp_{exp_id:03d}" / args.split / "threshold_sweep.json"
        if not sweep_json.exists():
            raise FileNotFoundError(
                f"--from-sweep: no sweep file at {sweep_json}. Run "
                f"`run.py sweep --split {args.split} --data-mode all` first."
            )
        import json as _json
        with sweep_json.open("r", encoding="utf-8") as f:
            sweep_data = _json.load(f)
        best = sweep_data.get("best", {})
        best_overrides = best.get("overrides", {})
        best_trace_thr = best.get("trace_threshold")
        metrics_cfg = main_cfg.setdefault("metrics", {})
        picker_cfg_ref = metrics_cfg.setdefault("picker", {})
        detection_cfg_ref = metrics_cfg.setdefault("detection", {})
        for k, v in best_overrides.items():
            picker_cfg_ref[k] = v
        if best_trace_thr is not None:
            detection_cfg_ref["trace_threshold"] = float(best_trace_thr)
        print(f"[Eval] --from-sweep applied from {sweep_json}:")
        for k, v in best_overrides.items():
            print(f"         picker.{k} = {v}")
        if best_trace_thr is not None:
            print(f"         detection.trace_threshold = {best_trace_thr}")

    # ----- build model -----------------------------------------------------
    model = SeisMambaKAN(model_cfg).to(device)
    state = _safe_torch_load(ckpt_path, device)
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
    _print_metrics_summary(metrics)
    print(f"\n[Eval] full metrics file: {out_dir}/metrics/{args.split}_metrics.json")


if __name__ == "__main__":
    main()
