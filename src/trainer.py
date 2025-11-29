from __future__ import annotations

import random
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import torch
from torch import amp
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm

from .dataset import build_dataloader, load_yaml
from .losses import build_loss_fn
from .models.network import SeisMambaKAN


# =============================================================================
# Utility helpers
# =============================================================================


def set_global_seed(seed: int) -> None:
    """Set global random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # cuDNN settings: allow benchmark for speed, but keep deterministic off
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


def get_device() -> torch.device:
    """Return CUDA device if available, otherwise CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def prepare_experiment_dirs(root: Path | str = "experiments") -> Tuple[Path, Path]:
    """
    Create a new experiment directory with incremental naming:

        experiments/exp_001
        experiments/exp_002
        ...

    Inside each experiment directory:
        - best_model.pth
        - config_used.yaml
        - events.out.tfevents*  (TensorBoard)
        - logs.txt
        - checkpoints/          (epoch checkpoints)

    Returns:
        exp_dir:  path to the new experiment directory
        ckpt_dir: path to the checkpoints subdirectory
    """
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)

    # Collect existing experiment indices
    existing_indices: list[int] = []
    for p in root.iterdir():
        if p.is_dir() and p.name.startswith("exp_"):
            try:
                idx = int(p.name.split("_")[1])
                existing_indices.append(idx)
            except (IndexError, ValueError):
                continue

    next_idx = (max(existing_indices) if existing_indices else 0) + 1
    exp_name = f"exp_{next_idx:03d}"
    exp_dir = root / exp_name
    ckpt_dir = exp_dir / "checkpoints"

    exp_dir.mkdir(parents=True, exist_ok=False)
    ckpt_dir.mkdir(parents=True, exist_ok=False)

    return exp_dir, ckpt_dir


def build_model_and_loss(
    main_cfg: Dict[str, Any],
    model_cfg: Dict[str, Any],
    device: torch.device,
) -> Tuple[torch.nn.Module, torch.nn.Module, bool, bool]:
    """
    Build SeisMambaKAN model and loss function based on the provided configs.

    Returns:
        model:                 the initialized model on the target device
        loss_fn:               configured loss module
        use_amp:               whether to use mixed precision
        use_channels_last:     whether channels_last is requested in config

    Note:
        channels_last is only meaningful for 4D tensors (N, C, H, W) used in
        Conv2d. Since the current architecture is Conv1d with shapes (B, C, T),
        the trainer will only apply channels_last when a 4D input is detected.
    """
    model = SeisMambaKAN(model_cfg)
    model = model.to(device)

    model_core_cfg = model_cfg.get("model", {})
    use_amp = bool(model_core_cfg.get("use_amp", False))
    use_channels_last = bool(model_core_cfg.get("use_channels_last", False))

    loss_fn = build_loss_fn(main_cfg)
    return model, loss_fn, use_amp, use_channels_last


def build_optimizer(
    model: torch.nn.Module,
    main_cfg: Dict[str, Any],
    model_cfg: Dict[str, Any],
) -> torch.optim.Optimizer:
    """Create Adam optimizer using training and regularization configs."""
    train_cfg = main_cfg.get("training", {})
    lr = float(train_cfg.get("learning_rate", 3.0e-4))

    reg_cfg = model_cfg.get("regularization", {})
    weight_decay = float(reg_cfg.get("weight_decay", 0.0))

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
    )
    return optimizer


# =============================================================================
# Trainer
# =============================================================================


class Trainer:
    """Main training loop for SeisMambaKAN."""

    def __init__(
        self,
        model: torch.nn.Module,
        loss_fn: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        train_loader: DataLoader,
        val_loader: DataLoader,
        main_cfg: Dict[str, Any],
        model_cfg: Dict[str, Any],
        paths_cfg: Dict[str, Any],
        exp_dir: Path,
        ckpt_dir: Path,
        use_amp: bool,
        use_channels_last: bool,
    ) -> None:
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.device = device
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.main_cfg = main_cfg
        self.model_cfg = model_cfg
        self.paths_cfg = paths_cfg
        self.exp_dir = exp_dir
        self.ckpt_dir = ckpt_dir
        self.use_amp = use_amp

        self.train_cfg = main_cfg.get("training", {})
        self.dataloader_cfg = main_cfg.get("dataloader", {})
        self.loss_cfg = main_cfg.get("loss", {})

        # Whether to attempt channels_last layout for 4D tensors
        self.channels_last = bool(use_channels_last)

        # AMP scaler (new torch.amp API)
        self.scaler = amp.GradScaler(enabled=self.use_amp)

        # Summary writer for TensorBoard (events.out.tfevents)
        self.writer = SummaryWriter(log_dir=str(self.exp_dir))

        # Log file
        self.log_file_path = self.exp_dir / "logs.txt"
        self._log_file = self.log_file_path.open("a", encoding="utf-8")

        self.best_val_loss: float = float("inf")

    # ------------------------------------------------------------------
    # Core utilities
    # ------------------------------------------------------------------

    def _prepare_inputs(self, x: torch.Tensor) -> torch.Tensor:
        """
        Move inputs to device and adjust layout.

        Current dataset provides waveforms as (B, T, C).
        For Conv1d, we convert to (B, C, T).

        If the model is ever changed to use 2D convolutions with 4D inputs
        (B, C, H, W), and use_channels_last=True in model_config.yaml, this
        function will apply channels_last memory format to the 4D tensor.

        For 3D tensors (Conv1d), channels_last is not supported by PyTorch,
        so the flag is ignored.
        """
        x = x.to(self.device, non_blocking=True)

        if x.dim() == 3:
            # (B, T, C) -> (B, C, T) for Conv1d
            x = x.permute(0, 2, 1).contiguous()
            # channels_last is not defined for 3D tensors; ignore flag here.
            return x

        if x.dim() == 4:
            # If in the future dataset/network uses Conv2d with 4D inputs,
            # channels_last can be applied here.
            if self.channels_last:
                x = x.to(memory_format=torch.channels_last)
            return x

        # Fallback: just return moved tensor
        return x

    def _move_labels_to_device(self, labels: Dict[str, Any]) -> Dict[str, Any]:
        """
        Move tensor labels to the target device. Non-tensor metadata stays on CPU.
        """
        out: Dict[str, Any] = {}
        for k, v in labels.items():
            if torch.is_tensor(v):
                out[k] = v.to(self.device, non_blocking=True)
            else:
                out[k] = v
        return out

    def _log(self, message: str) -> None:
        """Append a message to logs.txt and print it."""
        print(message)
        self._log_file.write(message + "\n")
        self._log_file.flush()

    # ------------------------------------------------------------------
    # Training / validation loop
    # ------------------------------------------------------------------

    def train_one_epoch(self, epoch: int) -> Dict[str, float]:
        self.model.train()

        running = {
            "total": 0.0,
            "detection": 0.0,
            "p": 0.0,
            "s": 0.0,
            "center_p": 0.0,
            "center_s": 0.0,
        }
        num_batches = 0

        pbar = tqdm(self.train_loader, desc=f"[Epoch {epoch}] Train", leave=False)

        for batch in pbar:
            x, labels = batch  # x: (B, T, C), labels: dict
            x = self._prepare_inputs(x)
            labels = self._move_labels_to_device(labels)

            self.optimizer.zero_grad(set_to_none=True)

            with amp.autocast(device_type=self.device.type, enabled=self.use_amp):
                outputs = self.model(x)
                loss_dict = self.loss_fn(outputs, labels)
                total_loss = loss_dict["total"]

            self.scaler.scale(total_loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            # Detach scalars for logging
            loss_cpu = {k: float(v.detach().cpu().item()) for k, v in loss_dict.items()}

            running["total"] += loss_cpu.get("total", 0.0)
            running["detection"] += loss_cpu.get("detection", 0.0)
            running["p"] += loss_cpu.get("p", 0.0)
            running["s"] += loss_cpu.get("s", 0.0)
            running["center_p"] += loss_cpu.get("center_p", 0.0)
            running["center_s"] += loss_cpu.get("center_s", 0.0)
            num_batches += 1

            pbar.set_postfix(
                total=f"{loss_cpu.get('total', 0.0):.4f}",
                det=f"{loss_cpu.get('detection', 0.0):.4f}",
                p=f"{loss_cpu.get('p', 0.0):.4f}",
                s=f"{loss_cpu.get('s', 0.0):.4f}",
            )

        if num_batches == 0:
            return {k: float("nan") for k in running}

        for k in running:
            running[k] /= num_batches

        # Log to TensorBoard
        self.writer.add_scalar("train/total_loss", running["total"], epoch)
        self.writer.add_scalar("train/det_loss", running["detection"], epoch)
        self.writer.add_scalar("train/p_loss", running["p"], epoch)
        self.writer.add_scalar("train/s_loss", running["s"], epoch)

        return running

    @torch.no_grad()
    def validate_one_epoch(self, epoch: int) -> Dict[str, float]:
        self.model.eval()

        running = {
            "total": 0.0,
            "detection": 0.0,
            "p": 0.0,
            "s": 0.0,
            "center_p": 0.0,
            "center_s": 0.0,
        }
        num_batches = 0

        pbar = tqdm(self.val_loader, desc=f"[Epoch {epoch}] Val", leave=False)

        for batch in pbar:
            x, labels = batch
            x = self._prepare_inputs(x)
            labels = self._move_labels_to_device(labels)

            with amp.autocast(device_type=self.device.type, enabled=self.use_amp):
                outputs = self.model(x)
                loss_dict = self.loss_fn(outputs, labels)

            loss_cpu = {k: float(v.detach().cpu().item()) for k, v in loss_dict.items()}

            running["total"] += loss_cpu.get("total", 0.0)
            running["detection"] += loss_cpu.get("detection", 0.0)
            running["p"] += loss_cpu.get("p", 0.0)
            running["s"] += loss_cpu.get("s", 0.0)
            running["center_p"] += loss_cpu.get("center_p", 0.0)
            running["center_s"] += loss_cpu.get("center_s", 0.0)
            num_batches += 1

            pbar.set_postfix(
                total=f"{loss_cpu.get('total', 0.0):.4f}",
                det=f"{loss_cpu.get('detection', 0.0):.4f}",
                p=f"{loss_cpu.get('p', 0.0):.4f}",
                s=f"{loss_cpu.get('s', 0.0):.4f}",
            )

        if num_batches == 0:
            return {k: float("nan") for k in running}

        for k in running:
            running[k] /= num_batches

        # Log to TensorBoard
        self.writer.add_scalar("val/total_loss", running["total"], epoch)
        self.writer.add_scalar("val/det_loss", running["detection"], epoch)
        self.writer.add_scalar("val/p_loss", running["p"], epoch)
        self.writer.add_scalar("val/s_loss", running["s"], epoch)

        return running

    # ------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------

    def _save_checkpoint(
        self,
        epoch: int,
        val_metrics: Dict[str, float],
        is_best: bool,
    ) -> None:
        """
        Save model checkpoints.

        - Always save epoch checkpoint into checkpoints/ as:
              checkpoint_epoch_{epoch:03d}.pth
        - Also save "last.pth" with the latest state.
        - If is_best, update best_model.pth in experiment root.
        """
        state = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "val_loss": val_metrics["total"],
            "config": self.main_cfg,
            "model_config": self.model_cfg,
            "paths_config": self.paths_cfg,
        }

        # Per-epoch checkpoint
        epoch_ckpt_path = self.ckpt_dir / f"checkpoint_epoch_{epoch:03d}.pth"
        torch.save(state, epoch_ckpt_path)

        # Last checkpoint (overwritten each epoch)
        last_ckpt_path = self.ckpt_dir / "last.pth"
        torch.save(state, last_ckpt_path)

        if is_best:
            # Save only model weights as best_model.pth in experiment root
            best_model_path = self.exp_dir / "best_model.pth"
            torch.save(self.model.state_dict(), best_model_path)

    # ------------------------------------------------------------------
    # Main training loop
    # ------------------------------------------------------------------

    def fit(self) -> None:
        num_epochs = int(self.train_cfg.get("epochs", 1))

        self._log(f"Starting training for {num_epochs} epochs.")
        self._log(f"Experiment directory: {self.exp_dir}")
        self._log(f"Device: {self.device.type}, AMP: {self.use_amp}")
        self._log(f"use_channels_last (4D only): {self.channels_last}")

        for epoch in range(1, num_epochs + 1):
            train_metrics = self.train_one_epoch(epoch)
            val_metrics = self.validate_one_epoch(epoch)

            log_msg = (
                f"[Epoch {epoch:03d}] "
                f"Train total={train_metrics['total']:.4f}, "
                f"Val total={val_metrics['total']:.4f}, "
                f"Val det={val_metrics['detection']:.4f}, "
                f"Val p={val_metrics['p']:.4f}, "
                f"Val s={val_metrics['s']:.4f}"
            )
            self._log(log_msg)

            current_val = val_metrics["total"]
            is_best = current_val < self.best_val_loss
            if is_best:
                self.best_val_loss = current_val

            self._save_checkpoint(epoch, val_metrics, is_best=is_best)

        # Close resources
        self.writer.close()
        self._log_file.close()


# =============================================================================
# Main entry point
# =============================================================================


def main() -> None:
    # ------------------------------------------------------------------
    # Load configs (paths are fixed; edit here if needed)
    # ------------------------------------------------------------------
    main_cfg_path = Path("config.yaml")
    model_cfg_path = Path("model_config.yaml")
    paths_cfg_path = Path("paths.yaml")

    # Fallback to configs/ subdirectory if root-level files do not exist
    if not main_cfg_path.exists():
        main_cfg_path = Path("configs/config.yaml")
    if not model_cfg_path.exists():
        model_cfg_path = Path("configs/model_config.yaml")
    if not paths_cfg_path.exists():
        paths_cfg_path = Path("configs/paths.yaml")

    main_cfg = load_yaml(main_cfg_path)
    model_cfg = load_yaml(model_cfg_path)
    paths_cfg = load_yaml(paths_cfg_path)

    # ------------------------------------------------------------------
    # Device and seed
    # ------------------------------------------------------------------
    device = get_device()

    train_cfg = main_cfg.get("training", {})
    seed = int(train_cfg.get("seed", 42))
    set_global_seed(seed)

    # ------------------------------------------------------------------
    # Experiment directories
    # ------------------------------------------------------------------
    exp_root = Path(train_cfg.get("output_dir", "experiments"))
    exp_dir, ckpt_dir = prepare_experiment_dirs(exp_root)

    # Save merged config into config_used.yaml inside the experiment folder
    import yaml

    merged_config = {
        "main": main_cfg,
        "model": model_cfg,
        "paths": paths_cfg,
    }
    config_used_path = exp_dir / "config_used.yaml"
    with config_used_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(merged_config, f, sort_keys=False)

    # ------------------------------------------------------------------
    # Dataloaders
    # ------------------------------------------------------------------
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

    # ------------------------------------------------------------------
    # Model, loss, optimizer
    # ------------------------------------------------------------------
    model, loss_fn, use_amp, use_channels_last = build_model_and_loss(
        main_cfg,
        model_cfg,
        device,
    )
    optimizer = build_optimizer(model, main_cfg, model_cfg)

    # ------------------------------------------------------------------
    # Trainer and run
    # ------------------------------------------------------------------
    trainer = Trainer(
        model=model,
        loss_fn=loss_fn,
        optimizer=optimizer,
        device=device,
        train_loader=train_loader,
        val_loader=val_loader,
        main_cfg=main_cfg,
        model_cfg=model_cfg,
        paths_cfg=paths_cfg,
        exp_dir=exp_dir,
        ckpt_dir=ckpt_dir,
        use_amp=use_amp,
        use_channels_last=use_channels_last,
    )

    trainer.fit()


if __name__ == "__main__":
    main()
