# trainer.py

import argparse
import os
import random
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import torch
from torch import nn, optim
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from dataset import build_dataloader, load_yaml
from losses import build_loss_fn
from models.network import SeisMambaKAN


# ======================================================================
# Utility helpers
# ======================================================================


def set_global_seed(seed: int) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


def prepare_output_dirs(train_cfg: Dict[str, Any]) -> Tuple[Path, Path]:
    """
    Create experiment and checkpoint directories.

    Returns:
        exp_dir: root directory for this experiment
        ckpt_dir: directory where checkpoints are stored
    """
    output_root = Path(train_cfg.get("output_dir", "experiments"))
    experiment_name = train_cfg.get("experiment_name", "default_experiment")

    exp_dir = output_root / experiment_name
    ckpt_dir = exp_dir / "checkpoints"

    exp_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    return exp_dir, ckpt_dir


def build_model_and_loss(
    model_cfg: Dict[str, Any],
    main_cfg: Dict[str, Any],
    device: torch.device,
) -> Tuple[nn.Module, nn.Module, bool]:
    """
    Build SeisMambaKAN model and the composite loss function.

    Returns:
        model: initialized model on the given device
        loss_fn: configured loss module
        use_amp: whether to use mixed precision
    """
    model = SeisMambaKAN(model_cfg)  # type: ignore[arg-type]
    model = model.to(device)

    model_cfg_model = model_cfg.get("model", {})
    use_channels_last = bool(model_cfg_model.get("use_channels_last", False))
    if use_channels_last:
        model = model.to(memory_format=torch.channels_last)

    use_amp = bool(model_cfg_model.get("use_amp", False))

    loss_fn = build_loss_fn(main_cfg)

    return model, loss_fn, use_amp


def build_optimizer(
    model: nn.Module,
    main_cfg: Dict[str, Any],
    model_cfg: Dict[str, Any],
) -> optim.Optimizer:
    """Build Adam optimizer using training and regularization configs."""
    train_cfg = main_cfg["training"]
    lr = float(train_cfg.get("learning_rate", 3e-4))

    reg_cfg = model_cfg.get("regularization", {})
    weight_decay = float(reg_cfg.get("weight_decay", 0.0))

    optimizer = optim.Adam(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
    )
    return optimizer


# ======================================================================
# Trainer class
# ======================================================================


class Trainer:
    """Main training loop holder for SeisMambaKAN."""

    def __init__(
        self,
        model: nn.Module,
        loss_fn: nn.Module,
        optimizer: optim.Optimizer,
        device: torch.device,
        train_loader: DataLoader,
        val_loader: DataLoader,
        main_cfg: Dict[str, Any],
        model_cfg: Dict[str, Any],
        exp_dir: Path,
        ckpt_dir: Path,
        use_amp: bool,
    ) -> None:
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.device = device
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.main_cfg = main_cfg
        self.model_cfg = model_cfg
        self.exp_dir = exp_dir
        self.ckpt_dir = ckpt_dir
        self.use_amp = use_amp

        self.train_cfg = main_cfg["training"]
        self.loss_cfg = main_cfg.get("loss", {})
        self.label_keys = self.loss_cfg.get("keys", {})

        self.channels_last = bool(
            model_cfg.get("model", {}).get("use_channels_last", False)
        )

        self.scaler = GradScaler(enabled=self.use_amp)

        self.best_val_loss: float = float("inf")

    # ------------------------------------------------------------------
    # Core training / validation loops
    # ------------------------------------------------------------------

    def _prepare_inputs(self, x: torch.Tensor) -> torch.Tensor:
        """
        Convert (B, T, C) to (B, C, T) and move to device,
        optionally channels_last.
        """
        x = x.to(self.device, non_blocking=True)  # (B, T, C)
        x = x.permute(0, 2, 1).contiguous()  # (B, C, T)
        if self.channels_last:
            x = x.to(memory_format=torch.channels_last)
        return x

    def _move_labels_to_device(self, labels: Dict[str, Any]) -> Dict[str, Any]:
        """
        Move tensor labels to device while keeping non-tensor metadata on CPU.

        The loss module itself can call .to(pred.device) again, but this
        avoids repeated host-to-device copies across batches.
        """
        out: Dict[str, Any] = {}
        for k, v in labels.items():
            if torch.is_tensor(v):
                out[k] = v.to(self.device, non_blocking=True)
            else:
                out[k] = v
        return out

    def train_one_epoch(self, epoch: int) -> Dict[str, float]:
        self.model.train()

        running_total = 0.0
        running_det = 0.0
        running_p = 0.0
        running_s = 0.0
        running_center = 0.0
        num_batches = 0

        pbar = tqdm(
            self.train_loader,
            desc=f"[Epoch {epoch}] Train",
            leave=False,
        )

        for batch in pbar:
            x, labels = batch  # x: (B, T, C), labels: dict
            x = self._prepare_inputs(x)
            labels = self._move_labels_to_device(labels)

            self.optimizer.zero_grad(set_to_none=True)

            with autocast(enabled=self.use_amp):
                outputs = self.model(x)
                loss_dict = self.loss_fn(outputs, labels)

                total_loss = loss_dict["total_loss"]

            self.scaler.scale(total_loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            # Aggregate losses
            total_val = float(total_loss.detach().cpu().item())
            det_val = float(loss_dict["detection_loss"].detach().cpu().item())
            p_val = float(loss_dict["p_loss"].detach().cpu().item())
            s_val = float(loss_dict["s_loss"].detach().cpu().item())
            center_val = float(
                loss_dict.get("center_loss", torch.tensor(0.0)).detach().cpu().item()
            )

            running_total += total_val
            running_det += det_val
            running_p += p_val
            running_s += s_val
            running_center += center_val
            num_batches += 1

            pbar.set_postfix(
                {
                    "total": f"{total_val:.4f}",
                    "det": f"{det_val:.4f}",
                    "p": f"{p_val:.4f}",
                    "s": f"{s_val:.4f}",
                }
            )

        if num_batches == 0:
            return {
                "total": float("nan"),
                "det": float("nan"),
                "p": float("nan"),
                "s": float("nan"),
                "center": float("nan"),
            }

        return {
            "total": running_total / num_batches,
            "det": running_det / num_batches,
            "p": running_p / num_batches,
            "s": running_s / num_batches,
            "center": running_center / num_batches,
        }

    @torch.no_grad()
    def validate_one_epoch(self, epoch: int) -> Dict[str, float]:
        self.model.eval()

        running_total = 0.0
        running_det = 0.0
        running_p = 0.0
        running_s = 0.0
        running_center = 0.0
        num_batches = 0

        pbar = tqdm(
            self.val_loader,
            desc=f"[Epoch {epoch}] Val",
            leave=False,
        )

        for batch in pbar:
            x, labels = batch
            x = self._prepare_inputs(x)
            labels = self._move_labels_to_device(labels)

            with autocast(enabled=self.use_amp):
                outputs = self.model(x)
                loss_dict = self.loss_fn(outputs, labels)

                total_loss = loss_dict["total_loss"]

            total_val = float(total_loss.detach().cpu().item())
            det_val = float(loss_dict["detection_loss"].detach().cpu().item())
            p_val = float(loss_dict["p_loss"].detach().cpu().item())
            s_val = float(loss_dict["s_loss"].detach().cpu().item())
            center_val = float(
                loss_dict.get("center_loss", torch.tensor(0.0)).detach().cpu().item()
            )

            running_total += total_val
            running_det += det_val
            running_p += p_val
            running_s += s_val
            running_center += center_val
            num_batches += 1

            pbar.set_postfix(
                {
                    "total": f"{total_val:.4f}",
                    "det": f"{det_val:.4f}",
                    "p": f"{p_val:.4f}",
                    "s": f"{s_val:.4f}",
                }
            )

        if num_batches == 0:
            return {
                "total": float("nan"),
                "det": float("nan"),
                "p": float("nan"),
                "s": float("nan"),
                "center": float("nan"),
            }

        return {
            "total": running_total / num_batches,
            "det": running_det / num_batches,
            "p": running_p / num_batches,
            "s": running_s / num_batches,
            "center": running_center / num_batches,
        }

    # ------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------

    def _save_checkpoint(
        self,
        epoch: int,
        val_metrics: Dict[str, float],
        is_best: bool,
    ) -> None:
        """Save model + optimizer state dict."""
        state = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "val_loss": val_metrics["total"],
            "config": self.main_cfg,
            "model_config": self.model_cfg,
        }

        last_path = self.ckpt_dir / "last.ckpt"
        torch.save(state, last_path)

        if is_best:
            best_path = self.ckpt_dir / "best.ckpt"
            torch.save(state, best_path)

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def fit(self) -> None:
        num_epochs = int(self.train_cfg.get("epochs", 1))

        print(f"Starting training for {num_epochs} epochs.")
        print(f"Experiment dir: {self.exp_dir}")

        for epoch in range(1, num_epochs + 1):
            train_metrics = self.train_one_epoch(epoch)
            val_metrics = self.validate_one_epoch(epoch)

            msg = (
                f"[Epoch {epoch:03d}] "
                f"Train total={train_metrics['total']:.4f}, "
                f"Val total={val_metrics['total']:.4f}, "
                f"Val det={val_metrics['det']:.4f}, "
                f"Val p={val_metrics['p']:.4f}, "
                f"Val s={val_metrics['s']:.4f}"
            )
            print(msg)

            current_val = val_metrics["total"]
            is_best = current_val < self.best_val_loss
            if is_best:
                self.best_val_loss = current_val

            self._save_checkpoint(epoch, val_metrics, is_best=is_best)


# ======================================================================
# CLI
# ======================================================================


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train SeisMambaKAN on STEAD WebDataset shards."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/config.yaml",
        help="Path to main config.yaml.",
    )
    parser.add_argument(
        "--model-config",
        type=str,
        default="configs/model_config.yaml",
        help="Path to model_config.yaml.",
    )
    parser.add_argument(
        "--paths",
        type=str,
        default="configs/paths.yaml",
        help="Path to paths.yaml.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help='Device to use, e.g. "cuda", "cuda:0", or "cpu".',
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Load configs
    main_cfg = load_yaml(args.config)
    model_cfg = load_yaml(args.model_config)
    paths_cfg = load_yaml(args.paths)

    # Device
    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU.")
        device = torch.device("cpu")
    else:
        device = torch.device(args.device)

    # Seed
    train_cfg = main_cfg["training"]
    seed = int(train_cfg.get("seed", 42))
    set_global_seed(seed)

    # Output dirs
    exp_dir, ckpt_dir = prepare_output_dirs(train_cfg)

    # Data loaders
    train_loader = build_dataloader(
        split="train",
        cfg=main_cfg,
        paths_cfg=paths_cfg,
        is_train=True,
    )
    val_loader = build_dataloader(
        split="val",
        cfg=main_cfg,
        paths_cfg=paths_cfg,
        is_train=False,
    )

    # Model, loss, optimizer
    model, loss_fn, use_amp = build_model_and_loss(model_cfg, main_cfg, device)
    optimizer = build_optimizer(model, main_cfg, model_cfg)

    # Trainer
    trainer = Trainer(
        model=model,
        loss_fn=loss_fn,
        optimizer=optimizer,
        device=device,
        train_loader=train_loader,
        val_loader=val_loader,
        main_cfg=main_cfg,
        model_cfg=model_cfg,
        exp_dir=exp_dir,
        ckpt_dir=ckpt_dir,
        use_amp=use_amp,
    )

    trainer.fit()


if __name__ == "__main__":
    main()
