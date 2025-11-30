"""
Grid search hyperparameter tuning for SeisMambaKAN loss weights.

- Trains a fresh model for every combination of loss weights.
- Uses a small number of epochs per combination (configurable).
- Does NOT create experiments/exp_xxx directories.
- At the end, writes the best combination to best_params.txt
  in the Drive project root:

  /content/drive/MyDrive/Proje_SeisMamba/SeisMambaKAN/best_params.txt
"""

from __future__ import annotations

import itertools
import math
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Tuple

import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

# -------------------------------------------------------------------------
# Paths and imports
# -------------------------------------------------------------------------

THIS_DIR = Path(__file__).resolve().parent
ROOT_DIR = THIS_DIR.parent  # /content/SeisMambaKAN

if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.dataset import load_yaml, build_dataloader
from src.models.network import SeisMambaKAN
from src.losses import build_loss_fn


# -------------------------------------------------------------------------
# Utils
# -------------------------------------------------------------------------

def set_global_seed(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def clone_cfg(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Deep copy config so we can safely mutate it per trial."""
    return deepcopy(cfg)


# -------------------------------------------------------------------------
# Single trial: train + validate for one (w_det, w_p, w_s)
# -------------------------------------------------------------------------

def run_single_trial(
    cfg_base: Dict[str, Any],
    paths_cfg: Dict[str, Any],
    model_cfg: Dict[str, Any],
    w_det: float,
    w_p: float,
    w_s: float,
    epochs: int,
    device: torch.device,
) -> Tuple[float, float]:
    """
    Train a fresh model with given loss weights and return:

        (best_val_total_loss, best_val_det_loss)

    over the validation set.

    This function does NOT create experiment directories or save checkpoints.
    """

    # ------------------------------------------------------------------
    # Prepare config for this trial
    # ------------------------------------------------------------------
    cfg_trial = clone_cfg(cfg_base)

    loss_cfg = cfg_trial.setdefault("loss", {})
    weights_cfg = loss_cfg.setdefault("weights", {})
    weights_cfg["detection"] = float(w_det)
    weights_cfg["p"] = float(w_p)
    weights_cfg["s"] = float(w_s)

    # Ensure dataloader config exists; keep num_workers from base or default 1
    base_dl_cfg = cfg_base.get("dataloader", {})
    dl_cfg = cfg_trial.setdefault("dataloader", {})
    dl_cfg.setdefault("num_workers", int(base_dl_cfg.get("num_workers", 1)))
    dl_cfg.setdefault("pin_memory", bool(base_dl_cfg.get("pin_memory", True)))
    dl_cfg.setdefault(
        "persistent_workers",
        bool(base_dl_cfg.get("persistent_workers", True)),
    )
    dl_cfg.setdefault("prefetch_factor", int(base_dl_cfg.get("prefetch_factor", 2)))
    dl_cfg.setdefault("drop_last", bool(base_dl_cfg.get("drop_last", True)))

    # ------------------------------------------------------------------
    # Dataloaders
    # ------------------------------------------------------------------
    train_loader: DataLoader = build_dataloader(
        split="train",
        cfg=cfg_trial,
        paths_cfg=paths_cfg,
        is_train=True,
    )

    val_loader: DataLoader = build_dataloader(
        split="val",
        cfg=cfg_trial,
        paths_cfg=paths_cfg,
        is_train=False,
    )

    # ------------------------------------------------------------------
    # Model, loss, optimizer
    # ------------------------------------------------------------------
    model = SeisMambaKAN(model_cfg)
    model.to(device)

    loss_fn = build_loss_fn(cfg_trial)

    train_cfg = cfg_trial.get("training", {})
    lr = float(train_cfg.get("learning_rate", 3.0e-4))

    reg_cfg = model_cfg.get("regularization", {})
    weight_decay = float(reg_cfg.get("weight_decay", 0.0))

    optimizer = optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
    )

    best_val_total = math.inf
    best_val_det = math.inf

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    for epoch in range(1, epochs + 1):
        model.train()
        running_train_total = 0.0
        n_train_batches = 0

        pbar = tqdm(
            train_loader,
            desc=(
                f"[det={w_det:.2f}, p={w_p:.2f}, s={w_s:.2f}] "
                f"Epoch {epoch}/{epochs} (train)"
            ),
            leave=False,
        )

        for batch in pbar:
            x, labels = batch  # x: (B, T, C), labels: dict (labels + metadata)

            # Inputs -> device
            x = x.to(device, non_blocking=True, dtype=torch.float32)
            if x.ndim == 3:
                # (B, T, C) -> (B, C, T) for Conv1d
                x = x.permute(0, 2, 1).contiguous()

            # Labels -> device (only tensor fields)
            labels_device: Dict[str, Any] = {}
            for k, v in labels.items():
                if isinstance(v, torch.Tensor):
                    labels_device[k] = v.to(device, non_blocking=True)
                else:
                    labels_device[k] = v  # keep metadata as-is

            optimizer.zero_grad(set_to_none=True)

            outputs = model(x)
            loss_dict = loss_fn(outputs, labels_device)
            loss_total = loss_dict["total"]

            loss_total.backward()
            optimizer.step()

            running_train_total += float(loss_total.detach().cpu().item())
            n_train_batches += 1

        avg_train_total = running_train_total / max(1, n_train_batches)

        # ------------------------------------------------------------------
        # Validation
        # ------------------------------------------------------------------
        model.eval()
        val_total_sum = 0.0
        val_det_sum = 0.0
        n_val_batches = 0

        with torch.no_grad():
            for batch in val_loader:
                x, labels = batch

                x = x.to(device, non_blocking=True, dtype=torch.float32)
                if x.ndim == 3:
                    x = x.permute(0, 2, 1).contiguous()

                labels_device: Dict[str, Any] = {}
                for k, v in labels.items():
                    if isinstance(v, torch.Tensor):
                        labels_device[k] = v.to(device, non_blocking=True)
                    else:
                        labels_device[k] = v

                outputs = model(x)
                loss_dict = loss_fn(outputs, labels_device)

                val_total_sum += float(loss_dict["total"].detach().cpu().item())
                val_det_sum += float(loss_dict["detection"].detach().cpu().item())
                n_val_batches += 1

        avg_val_total = val_total_sum / max(1, n_val_batches)
        avg_val_det = val_det_sum / max(1, n_val_batches)

        if avg_val_total < best_val_total:
            best_val_total = avg_val_total
            best_val_det = avg_val_det

        print(
            f"[det={w_det:.2f}, p={w_p:.2f}, s={w_s:.2f}] "
            f"Epoch {epoch:03d} | "
            f"train_total={avg_train_total:.4f} | "
            f"val_total={avg_val_total:.4f} | "
            f"val_det={avg_val_det:.4f}"
        )

    # Clean up a bit
    del model, optimizer, loss_fn, train_loader, val_loader
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return best_val_total, best_val_det


# -------------------------------------------------------------------------
# Main grid search
# -------------------------------------------------------------------------

def main() -> None:
    # -----------------------------
    # Paths and configs
    # -----------------------------
    root = ROOT_DIR
    configs_dir = root / "configs"

    config_path = configs_dir / "config.yaml"
    paths_path = configs_dir / "paths.yaml"
    model_cfg_path = configs_dir / "model_config.yaml"

    cfg = load_yaml(config_path)
    paths_cfg = load_yaml(paths_path)
    model_cfg = load_yaml(model_cfg_path)

    # Optional: force sample mode if you want faster search
    # data_cfg = cfg.setdefault("data", {})
    # data_cfg.setdefault("mode", "sample")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    seed = int(cfg.get("training", {}).get("seed", 42))
    set_global_seed(seed)

    # -----------------------------
    # Search space
    # -----------------------------
    # You set these in your last run:
    det_values = [0.1, 1.0, 1.5]
    p_values = [0.3, 1.0, 2.0]
    s_values = [0.4, 1.0, 2.0]

    epochs_per_trial = 20  # can be increased if you want more reliable ranking

    print("Search space:")
    print("  detection weights:", det_values)
    print("  P weights        :", p_values)
    print("  S weights        :", s_values)
    print("  epochs per trial :", epochs_per_trial)

    # -----------------------------
    # Grid search
    # -----------------------------
    best_overall_score = math.inf
    best_overall_det = math.inf
    best_params: Dict[str, float] = {}

    combos = list(itertools.product(det_values, p_values, s_values))
    print(f"Total combinations: {len(combos)}")

    for (w_det, w_p, w_s) in combos:
        print("\n" + "=" * 80)
        print(f"Running trial with weights: det={w_det}, p={w_p}, s={w_s}")
        print("=" * 80)

        # keep seed constant for fair comparison
        set_global_seed(seed)

        best_val_total, best_val_det = run_single_trial(
            cfg_base=cfg,
            paths_cfg=paths_cfg,
            model_cfg=model_cfg,
            w_det=w_det,
            w_p=w_p,
            w_s=w_s,
            epochs=epochs_per_trial,
            device=device,
        )

        print(
            f"[RESULT] weights det={w_det}, p={w_p}, s={w_s} "
            f"=> best_val_total={best_val_total:.4f}, "
            f"best_val_det={best_val_det:.4f}"
        )

        # Objective: minimize total validation loss
        if best_val_total < best_overall_score:
            best_overall_score = best_val_total
            best_overall_det = best_val_det
            best_params = {
                "loss.weights.detection": float(w_det),
                "loss.weights.p": float(w_p),
                "loss.weights.s": float(w_s),
            }

    print("\n" + "#" * 80)
    print("Grid search finished.")
    print("Best configuration:")
    print(best_params)
    print(f"Best val_total: {best_overall_score:.4f}")
    print(f"Best val_det  : {best_overall_det:.4f}")
    print("#" * 80)

    # -----------------------------
    # Save best params to Drive
    # -----------------------------
    drive_root = Path("/content/drive/MyDrive/Proje_SeisMamba/SeisMambaKAN")
    drive_root.mkdir(parents=True, exist_ok=True)
    best_params_path = drive_root / "best_params.txt"

    lines = [
        "Best hyperparameters found by grid search:\n",
        f"Device: {device}\n",
        f"Search space:\n",
        f"  detection: {det_values}\n",
        f"  p:         {p_values}\n",
        f"  s:         {s_values}\n",
        f"  epochs_per_trial: {epochs_per_trial}\n",
        "\n",
        "Best configuration:\n",
    ]
    for k, v in best_params.items():
        lines.append(f"{k}: {v}\n")
    lines.append("\n")
    lines.append(f"Best val_total: {best_overall_score:.6f}\n")
    lines.append(f"Best val_det  : {best_overall_det:.6f}\n")

    with open(best_params_path, "w", encoding="utf-8") as f:
        f.writelines(lines)

    print(f"\nBest parameters saved to: {best_params_path}")


if __name__ == "__main__":
    main()
