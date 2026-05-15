from __future__ import annotations

import random
import time
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
from .utils import apply_dotted_overrides

# Rich console for colored / structured terminal output (Phase 2.6).
# File logs (logs.txt + Drive mirror) remain plain text; rich is only for stdout.
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text


# =============================================================================
# Utilities
# =============================================================================


def set_global_seed(seed: int) -> None:
    """Set all major RNG seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
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

        root / exp_001
        root / exp_002
        ...

    Returns (exp_dir, ckpt_dir).
    """
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)

    existing = [d for d in root.iterdir() if d.is_dir() and d.name.startswith("exp_")]
    existing_indices = []
    for d in existing:
        try:
            idx = int(d.name.split("_")[-1])
            existing_indices.append(idx)
        except ValueError:
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
    """Build model and loss function from configs, move model to device."""
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
    """Create AdamW optimizer using training and regularization configs."""
    train_cfg = main_cfg.get("training", {})
    lr = float(train_cfg.get("learning_rate", 3.0e-4))

    reg_cfg = model_cfg.get("regularization", {})
    weight_decay = float(reg_cfg.get("weight_decay", 0.0))

    # AdamW (not Adam): decoupled weight decay, matches modern training recipes.
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
    )
    return optimizer


def estimate_steps_per_epoch(
    loader: DataLoader,
    main_cfg: Dict[str, Any],
    paths_cfg: Dict[str, Any],
    split: str = "train",
) -> int | None:
    """
    Return an int estimate of (steps/epoch) for any DataLoader.

    For map-style datasets: len(loader).
    For WebDataset IterableDataset: estimate via n_shards * shard_size / batch.
    Returns None if no estimate is possible.
    """
    try:
        return len(loader)  # works for map-style
    except TypeError:
        pass

    from glob import glob
    mode = main_cfg.get("data", {}).get("mode", "all")
    processed_cfg = paths_cfg.get("processed", {})
    split_dir_cfg = processed_cfg.get(mode, {}).get(f"{split}_dir")
    if not split_dir_cfg:
        return None

    n_shards = len(sorted(glob(str(Path(split_dir_cfg) / "*.tar"))))
    if n_shards == 0:
        return None

    shard_size = int(paths_cfg.get("webdataset", {}).get("shard_size", 2048))
    batch_size = int(main_cfg.get("training", {}).get("batch_size", 32))
    return max(1, (n_shards * shard_size) // batch_size)


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    main_cfg: Dict[str, Any],
    steps_per_epoch: int | None,
) -> torch.optim.lr_scheduler.LRScheduler | None:
    """
    Build per-step LR scheduler from training config.

    Supported:
      - "onecycle" -> OneCycleLR (warmup + cosine decay).
      - "none"     -> no scheduler (returns None).

    Returns None (with a warning) if total_steps cannot be determined.
    """
    train_cfg = main_cfg.get("training", {})
    sched_type = str(train_cfg.get("scheduler", "none")).lower()

    if sched_type == "none":
        return None

    if sched_type != "onecycle":
        raise ValueError(f"Unsupported scheduler: {sched_type!r}")

    epochs = int(train_cfg.get("epochs", 1))
    if steps_per_epoch is None or steps_per_epoch <= 0:
        print(
            "[WARN] steps_per_epoch unknown; cannot build OneCycleLR. "
            "Continuing without an LR scheduler."
        )
        return None

    total_steps = max(1, steps_per_epoch * epochs)
    max_lr = float(train_cfg.get("learning_rate", 3.0e-4))
    pct_start = float(train_cfg.get("scheduler_pct_start", 0.05))
    div_factor = float(train_cfg.get("scheduler_div_factor", 25.0))
    final_div_factor = float(train_cfg.get("scheduler_final_div_factor", 1.0e4))

    return torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=max_lr,
        total_steps=total_steps,
        pct_start=pct_start,
        anneal_strategy="cos",
        div_factor=div_factor,
        final_div_factor=final_div_factor,
    )


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
        scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
        mirror_exp_dir: Path | None = None,
        mirror_ckpt_dir: Path | None = None,
    ) -> None:
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.main_cfg = main_cfg
        self.model_cfg = model_cfg
        self.paths_cfg = paths_cfg
        self.exp_dir = exp_dir
        self.ckpt_dir = ckpt_dir
        self.use_amp = use_amp

        # Optional mirror directories (e.g., Google Drive) for experiments/checkpoints
        self.mirror_exp_dir = mirror_exp_dir
        self.mirror_ckpt_dir = mirror_ckpt_dir

        self.train_cfg = main_cfg.get("training", {})
        self.dataloader_cfg = main_cfg.get("dataloader", {})
        self.loss_cfg = main_cfg.get("loss", {})

        # Whether to attempt channels_last layout for 4D tensors
        self.channels_last = bool(use_channels_last)

        # AMP scaler (new torch.amp API)
        self.scaler = amp.GradScaler(enabled=self.use_amp)

        # --- Stability knobs (Phase 0) ---
        self.grad_clip_max_norm: float = float(
            self.train_cfg.get("grad_clip_max_norm", 0.0) or 0.0
        )
        self.nan_threshold: int = int(self.train_cfg.get("nan_threshold", 10))
        self.consecutive_nan_count: int = 0
        self.global_step: int = 0

        # --- Early stopping (Phase 2.5) ---
        es_cfg = self.train_cfg.get("early_stopping", {}) or {}
        self.es_enabled: bool = bool(es_cfg.get("enabled", False))
        self.es_patience: int = int(es_cfg.get("patience", 12))
        self.es_min_delta: float = float(es_cfg.get("min_delta", 1.0e-4))
        self.es_monitor: str = str(es_cfg.get("monitor", "total"))
        self.es_best_value: float = float("inf")
        self.es_epochs_since_improvement: int = 0

        # --- Rich console for pretty terminal output (Phase 2.6) ---
        # File logs stay plain text; console renders with ANSI styling. Colab
        # notebook cells render ANSI escapes correctly.
        self.console = Console()

        # Summary writer for TensorBoard (events.out.tfevents)
        self.writer = SummaryWriter(log_dir=str(self.exp_dir))

        # Log file (local experiment dir)
        self.log_file_path = self.exp_dir / "logs.txt"
        self._log_file = self.log_file_path.open("a", encoding="utf-8")

        # Optional mirrored log file (e.g., on Drive)
        self._mirror_log_file = None
        if self.mirror_exp_dir is not None:
            self.mirror_exp_dir.mkdir(parents=True, exist_ok=True)
            mirror_log_path = self.mirror_exp_dir / "logs.txt"
            self._mirror_log_file = mirror_log_path.open("a", encoding="utf-8")

        # Best validation loss seen so far
        self.best_val_loss: float = float("inf")

        # Step/epoch counters for ETA logging
        self.total_epochs: int = int(self.train_cfg.get("epochs", 1))

        # WebDataset (IterableDataset) için len() olmayabileceği için try/except
        try:
            self.steps_per_epoch: int | None = len(self.train_loader)
        except TypeError:
            self.steps_per_epoch = None

        if self.steps_per_epoch is not None:
            self.total_steps: int | None = self.steps_per_epoch * self.total_epochs
        else:
            self.total_steps = None

    # ------------------------------------------------------------------
    # Core utilities
    # ------------------------------------------------------------------

    @staticmethod
    def _format_seconds(seconds: float) -> str:
        """Convert seconds (float) into a HH:MM:SS string."""
        seconds = int(seconds)
        m, s = divmod(seconds, 60)
        h, m = divmod(m, 60)
        return f"{h:02d}:{m:02d}:{s:02d}"

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
            # (B, T, C) -> (B, C, T)
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
        """Append a message to logs.txt (and mirror, if any) and print it plain."""
        print(message)
        # Local log
        self._log_file.write(message + "\n")
        self._log_file.flush()
        # Optional mirrored log (e.g., on Drive)
        if self._mirror_log_file is not None:
            self._mirror_log_file.write(message + "\n")
            self._mirror_log_file.flush()

    def _log_styled(self, rich_msg: str) -> None:
        """
        Print a rich-styled message to the console and write its plain-text
        version to the log files. This is the preferred path for end-user
        epoch summaries / status lines.
        """
        self.console.print(rich_msg)
        plain = Text.from_markup(rich_msg).plain
        self._log_file.write(plain + "\n")
        self._log_file.flush()
        if self._mirror_log_file is not None:
            self._mirror_log_file.write(plain + "\n")
            self._mirror_log_file.flush()

    def _print_setup_panel(self) -> None:
        """Print a one-shot startup banner summarizing the training setup."""
        t = Table.grid(padding=(0, 1))
        t.add_column(style="dim", justify="right")
        t.add_column()
        t.add_row("Experiment", f"[bold]{self.exp_dir.name}[/]  [dim]({self.exp_dir})[/]")
        if self.mirror_exp_dir is not None:
            t.add_row("Drive mirror", str(self.mirror_exp_dir))
        t.add_row("Device", f"{self.device.type}")
        t.add_row("Epochs", str(self.total_epochs))
        if self.steps_per_epoch is not None:
            t.add_row("Steps/epoch", str(self.steps_per_epoch))
            t.add_row("Total steps", str(self.total_steps))
        else:
            t.add_row("Steps/epoch", "[yellow]unknown (IterableDataset)[/]")
        t.add_row("Batch size", str(self.train_cfg.get("batch_size", "?")))
        t.add_row("LR (peak)", f"{self.train_cfg.get('learning_rate', '?')}")
        t.add_row("Scheduler", type(self.scheduler).__name__ if self.scheduler else "[dim]none[/]")
        t.add_row("Grad clip", f"{self.grad_clip_max_norm:.2f}" if self.grad_clip_max_norm > 0 else "[dim]none[/]")
        t.add_row("AMP", "[green]on[/]" if self.use_amp else "[dim]off[/]")
        if self.es_enabled:
            t.add_row(
                "Early stop",
                f"patience={self.es_patience} monitor=[cyan]{self.es_monitor}[/] "
                f"min_delta={self.es_min_delta}",
            )
        else:
            t.add_row("Early stop", "[dim]disabled[/]")

        self.console.print(
            Panel(
                t,
                title="[bold cyan]SeisMambaKAN — training[/]",
                border_style="cyan",
                expand=False,
            )
        )

    def _print_epoch_summary(
        self,
        epoch: int,
        train_m: Dict[str, float],
        val_m: Dict[str, float],
        lr: float,
        epoch_time: float,
        eta: float,
        is_best: bool,
    ) -> None:
        """Pretty epoch summary line (logged both to console and file)."""
        marker = "[bold green]✓ best[/]" if is_best else "       "
        line = (
            f"[bold]Epoch {epoch:>3}/{self.total_epochs}[/]  {marker}  "
            f"[cyan]train[/] "
            f"total=[bold]{train_m['total']:.4f}[/] "
            f"det={train_m['detection']:.4f} "
            f"p={train_m['p']:.4f} "
            f"s={train_m['s']:.4f}   "
            f"[magenta]val[/] "
            f"total=[bold]{val_m['total']:.4f}[/] "
            f"det={val_m['detection']:.4f} "
            f"p={val_m['p']:.4f} "
            f"s={val_m['s']:.4f}   "
            f"[dim]lr={lr:.2e} • {self._format_seconds(epoch_time)} • "
            f"eta {self._format_seconds(eta)}[/]"
        )
        self._log_styled(line)

    def _print_early_stop_status(
        self,
        improved: bool,
        monitor_value: float,
        prev_best: float,
    ) -> None:
        """Compact, colored status line for early-stopping bookkeeping."""
        if improved:
            delta = prev_best - monitor_value if prev_best != float("inf") else float("inf")
            delta_str = "—" if delta == float("inf") else f"-{delta:.4f}"
            line = (
                f"   [green]↘ early-stop:[/] '{self.es_monitor}' improved "
                f"[dim]{prev_best:.4f} →[/] [bold]{monitor_value:.4f}[/] "
                f"[dim]({delta_str}, patience reset 0/{self.es_patience})[/]"
            )
        else:
            patience_n = self.es_epochs_since_improvement
            bar = "█" * patience_n + "·" * max(0, self.es_patience - patience_n)
            warn = "[yellow]" if patience_n < self.es_patience else "[red]"
            line = (
                f"   {warn}↗ early-stop:[/] no improvement on "
                f"'{self.es_monitor}' "
                f"[dim](best={self.es_best_value:.4f})[/]  "
                f"{warn}{bar}[/] [dim]{patience_n}/{self.es_patience}[/]"
            )
        self._log_styled(line)

    def _print_early_stop_triggered(self, epoch: int) -> None:
        """Loud banner when training stops because of early-stopping."""
        msg = (
            f"[bold red]🛑 Early stop triggered[/] at epoch [bold]{epoch}[/]/{self.total_epochs}. "
            f"'{self.es_monitor}' did not improve by ≥{self.es_min_delta} "
            f"for {self.es_patience} consecutive epochs "
            f"(best={self.es_best_value:.4f})."
        )
        self._log_styled(msg)

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

        # Keras-style progress bar: epoch label + bar + n/total + elapsed/remaining + postfix metrics.
        bar_format = (
            "  [Train {desc}] "
            "{bar:28} "
            "{n_fmt}/{total_fmt} "
            "[{elapsed}<{remaining} • {rate_fmt}]"
            "{postfix}"
        )
        pbar = tqdm(
            self.train_loader,
            desc=f"{epoch:>3}/{self.total_epochs}",
            bar_format=bar_format,
            ascii=" ▏▎▍▌▋▊▉█",
            leave=False,
            dynamic_ncols=True,
        )

        for batch in pbar:
            x, labels = batch  # x: (B, T, C), labels: dict
            x = self._prepare_inputs(x)
            labels = self._move_labels_to_device(labels)

            self.optimizer.zero_grad(set_to_none=True)

            with amp.autocast(device_type=self.device.type, enabled=self.use_amp):
                outputs = self.model(x)
                loss_dict = self.loss_fn(outputs, labels)
                total_loss = loss_dict["total"]

            # ----- NaN / Inf guard -----
            if not torch.isfinite(total_loss):
                self.consecutive_nan_count += 1
                self._log(
                    f"[WARN] Non-finite loss at epoch {epoch} step "
                    f"{self.global_step} (consecutive_nan={self.consecutive_nan_count}); "
                    "skipping this step."
                )
                if self.consecutive_nan_count >= self.nan_threshold:
                    raise RuntimeError(
                        f"Aborting: {self.consecutive_nan_count} consecutive "
                        f"non-finite training losses (threshold={self.nan_threshold})."
                    )
                continue
            else:
                self.consecutive_nan_count = 0

            # ----- Backward + grad clip + step -----
            self.scaler.scale(total_loss).backward()
            if self.grad_clip_max_norm > 0:
                # Unscale before clipping so the norm is in real-loss units.
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    max_norm=self.grad_clip_max_norm,
                )
            self.scaler.step(self.optimizer)
            self.scaler.update()

            # Per-step LR scheduler (OneCycleLR is per-step)
            if self.scheduler is not None:
                try:
                    self.scheduler.step()
                except Exception as e:
                    # OneCycleLR raises after total_steps is exhausted; absorb it
                    # so the rest of training/eval can still complete.
                    self._log(f"[WARN] scheduler.step() raised {type(e).__name__}: {e}")
                    self.scheduler = None
            self.global_step += 1

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

        bar_format = (
            "    [Val {desc}] "
            "{bar:28} "
            "{n_fmt}/{total_fmt} "
            "[{elapsed}<{remaining}]"
            "{postfix}"
        )
        pbar = tqdm(
            self.val_loader,
            desc=f"{epoch:>3}/{self.total_epochs}",
            bar_format=bar_format,
            ascii=" ▏▎▍▌▋▊▉█",
            leave=False,
            dynamic_ncols=True,
        )

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

        # Per-epoch checkpoint (local)
        epoch_ckpt_path = self.ckpt_dir / f"checkpoint_epoch_{epoch:03d}.pth"
        torch.save(state, epoch_ckpt_path)

        # Last checkpoint (overwritten each epoch, local)
        last_ckpt_path = self.ckpt_dir / "last.pth"
        torch.save(state, last_ckpt_path)

        # Optional mirrored checkpoints (e.g., on Drive)
        if self.mirror_ckpt_dir is not None:
            self.mirror_ckpt_dir.mkdir(parents=True, exist_ok=True)
            mirror_epoch_ckpt_path = self.mirror_ckpt_dir / f"checkpoint_epoch_{epoch:03d}.pth"
            torch.save(state, mirror_epoch_ckpt_path)
            mirror_last_ckpt_path = self.mirror_ckpt_dir / "last.pth"
            torch.save(state, mirror_last_ckpt_path)

        if is_best:
            # Save only model weights as best_model.pth in experiment root
            best_model_path = self.exp_dir / "best_model.pth"
            torch.save(self.model.state_dict(), best_model_path)

            # Also mirror best model weights if a mirror experiment dir exists
            if self.mirror_exp_dir is not None:
                self.mirror_exp_dir.mkdir(parents=True, exist_ok=True)
                mirror_best_model_path = self.mirror_exp_dir / "best_model.pth"
                torch.save(self.model.state_dict(), mirror_best_model_path)

    # ------------------------------------------------------------------
    # Main training loop
    # ------------------------------------------------------------------

    def fit(self) -> None:
        num_epochs = self.total_epochs

        # Pretty setup panel (also written to logs.txt as plain text via the
        # individual _log calls below for backward compat).
        self._print_setup_panel()
        # Plain mirror to file so logs.txt still tells the full story without ANSI.
        self._log(f"Starting training for {num_epochs} epochs.")
        self._log(f"Experiment directory: {self.exp_dir}")
        if self.mirror_exp_dir is not None:
            self._log(f"Mirror experiment directory: {self.mirror_exp_dir}")
        self._log(f"Device: {self.device.type}, AMP: {self.use_amp}")
        self._log(
            f"Steps per epoch: {self.steps_per_epoch}, total steps: {self.total_steps}"
            if self.steps_per_epoch is not None
            else "Steps per epoch: unknown (IterableDataset)."
        )

        # Global timer for ETA estimation
        start_time = time.time()

        for epoch in range(1, num_epochs + 1):
            epoch_start = time.time()

            train_metrics = self.train_one_epoch(epoch)
            val_metrics = self.validate_one_epoch(epoch)

            # Timing information
            epoch_time = time.time() - epoch_start
            elapsed = time.time() - start_time
            avg_epoch_time = elapsed / epoch
            remaining_epochs = num_epochs - epoch
            eta = avg_epoch_time * remaining_epochs

            current_lr = self.optimizer.param_groups[0]["lr"]
            self.writer.add_scalar("train/lr", current_lr, epoch)

            # Determine "is_best" up-front so the summary line can show ✓.
            current_val = val_metrics["total"]
            is_best = current_val < self.best_val_loss

            # Pretty summary
            self._print_epoch_summary(
                epoch=epoch,
                train_m=train_metrics,
                val_m=val_metrics,
                lr=current_lr,
                epoch_time=epoch_time,
                eta=eta,
                is_best=is_best,
            )

            current_val = val_metrics["total"]
            # Update best_val_loss now; the summary line already reflects is_best.
            if is_best:
                self.best_val_loss = current_val
                # Plain-text trail for logs.txt (useful when grepping the file).
                ck = self.ckpt_dir / f"checkpoint_epoch_{epoch:03d}.pth"
                bm = self.exp_dir / "best_model.pth"
                self._log(
                    f"[Epoch {epoch:03d}] New best val_total={current_val:.4f}. "
                    f"ckpt={ck} best_model={bm}"
                )

            self._save_checkpoint(epoch, val_metrics, is_best=is_best)

            # ----- Early stopping (Phase 2.5) -----
            if self.es_enabled:
                monitor_value = float(
                    val_metrics.get(self.es_monitor, val_metrics["total"])
                )
                improved = monitor_value < (self.es_best_value - self.es_min_delta)
                prev_best = self.es_best_value
                if improved:
                    self.es_best_value = monitor_value
                    self.es_epochs_since_improvement = 0
                else:
                    self.es_epochs_since_improvement += 1

                self._print_early_stop_status(improved, monitor_value, prev_best)

                if self.es_epochs_since_improvement >= self.es_patience:
                    self._print_early_stop_triggered(epoch)
                    break

        # Close resources
        self.writer.close()
        self._log_file.close()
        if self._mirror_log_file is not None:
            self._mirror_log_file.close()


# =============================================================================
# Main entry point
# =============================================================================


def main(
    overrides: Dict[str, Any] | None = None,
    model_overrides: Dict[str, Any] | None = None,
) -> None:
    """
    Training entry point.

    overrides:        dotted-key overrides applied to main config (config.yaml).
                      e.g. {"training.epochs": 5, "training.batch_size": 128}
    model_overrides:  dotted-key overrides applied to model_config.yaml.
                      e.g. {"model.dropout": 0.2}
    """
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
    # Apply CLI overrides (from run.py)
    # ------------------------------------------------------------------
    if overrides:
        apply_dotted_overrides(main_cfg, overrides)
        print(f"[Trainer] applied main-config overrides: {overrides}")
    if model_overrides:
        apply_dotted_overrides(model_cfg, model_overrides)
        print(f"[Trainer] applied model-config overrides: {model_overrides}")

    # ------------------------------------------------------------------
    # Device and seed
    # ------------------------------------------------------------------
    device = get_device()

    train_cfg = main_cfg.get("training", {})
    seed = int(train_cfg.get("seed", 42))
    set_global_seed(seed)

    # ------------------------------------------------------------------
    # Experiment directories (local + optional mirror, e.g., Google Drive)
    # ------------------------------------------------------------------
    experiments_cfg = paths_cfg.get("experiments", {})

    # training.output_dir has highest priority; otherwise fall back to paths.yaml
    exp_root = Path(
        train_cfg.get(
            "output_dir",
            experiments_cfg.get("root_dir", "experiments"),
        )
    )
    exp_dir, ckpt_dir = prepare_experiment_dirs(exp_root)

    # Optional mirror root on Drive, configured in paths.yaml as:
    # experiments:
    #   root_dir: "experiments"
    #   drive_root_dir: "/content/drive/MyDrive/Proje_SeisMamba/SeisMambaKAN/experiments"
    mirror_exp_dir = None
    mirror_ckpt_dir = None
    drive_root_dir = experiments_cfg.get("drive_root_dir")
    if drive_root_dir:
        drive_root_path = Path(drive_root_dir)
        mirror_exp_dir = drive_root_path / exp_dir.name
        mirror_ckpt_dir = mirror_exp_dir / "checkpoints"
        mirror_ckpt_dir.mkdir(parents=True, exist_ok=True)

    # Save merged config into config_used.yaml inside experiment folder(s)
    import yaml

    merged_config = {
        "main": main_cfg,
        "model": model_cfg,
        "paths": paths_cfg,
    }
    for out_dir in (exp_dir, mirror_exp_dir):
        if out_dir is None:
            continue
        out_dir.mkdir(parents=True, exist_ok=True)
        config_used_path = out_dir / "config_used.yaml"
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
    # Model, loss, optimizer, scheduler
    # ------------------------------------------------------------------
    model, loss_fn, use_amp, use_channels_last = build_model_and_loss(
        main_cfg,
        model_cfg,
        device,
    )
    optimizer = build_optimizer(model, main_cfg, model_cfg)

    # Estimate steps/epoch for OneCycleLR. WebDataset is IterableDataset so
    # len(loader) often fails — estimate from shard count instead.
    steps_per_epoch = estimate_steps_per_epoch(train_loader, main_cfg, paths_cfg, split="train")
    scheduler = build_scheduler(optimizer, main_cfg, steps_per_epoch)
    if scheduler is not None:
        print(
            f"[Trainer] scheduler={type(scheduler).__name__} "
            f"steps_per_epoch={steps_per_epoch} "
            f"total_steps={(steps_per_epoch or 0) * int(main_cfg.get('training', {}).get('epochs', 1))}"
        )

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
        scheduler=scheduler,
        mirror_exp_dir=mirror_exp_dir,
        mirror_ckpt_dir=mirror_ckpt_dir,
    )

    trainer.fit()


if __name__ == "__main__":
    main()
