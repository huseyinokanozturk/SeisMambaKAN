"""
SeisMambaKAN — top-level CLI.

One command to rule them all. Examples:

    python run.py setup --data-mode all          # Colab bootstrap
    python run.py data --mode all --refresh      # re-pull Drive data
    python run.py train --epochs 30              # train w/ override
    python run.py eval --exp 7 --split test      # evaluate exp_007
    python run.py infer --index 42               # plot one trace
    python run.py push --exp 7                   # mirror exp -> Drive
    python run.py pull-exp --exp 7               # bring exp from Drive
    python run.py tb                             # tensorboard
    python run.py status                         # quick project overview

All commands obey the paths defined in configs/paths.yaml — nothing is
hardcoded here.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

# Make sure project root is importable, regardless of CWD.
_THIS = Path(__file__).resolve()
if str(_THIS.parent) not in sys.path:
    sys.path.insert(0, str(_THIS.parent))

from src.utils import (  # noqa: E402
    is_colab,
    is_drive_mounted,
    list_experiments,
    load_all_configs,
    project_root,
    resolve_checkpoint,
    resolve_experiment_dir,
)

app = typer.Typer(
    add_completion=False,
    help="SeisMambaKAN: hybrid Mamba + KAN seismic phase picker.",
    no_args_is_help=True,
)
console = Console()


# =============================================================================
# Helpers
# =============================================================================

def _bullet(msg: str) -> None:
    console.print(f"[cyan]›[/cyan] {msg}")


def _ok(msg: str) -> None:
    console.print(f"[green]✓[/green] {msg}")


def _warn(msg: str) -> None:
    console.print(f"[yellow]![/yellow] {msg}")


def _err(msg: str) -> None:
    console.print(f"[red]✗[/red] {msg}")


def _run_subprocess(cmd: list[str], cwd: Optional[Path] = None) -> int:
    """Run a subprocess inheriting stdio. Returns the exit code."""
    console.print(f"[dim]$ {' '.join(cmd)}[/dim]")
    proc = subprocess.run(cmd, cwd=str(cwd) if cwd else None)
    return proc.returncode


# =============================================================================
# setup
# =============================================================================

@app.command(help="Bootstrap a Colab runtime: mount Drive, clone repo, copy data, install packages.")
def setup(
    data_mode: str = typer.Option("sample", "--data-mode", "-m",
                                   help="all | sample | none"),
    refresh_data: bool = typer.Option(False, "--refresh/--no-refresh",
                                       help="Wipe existing data dir before copying. "
                                            "Default False so failed mamba/torch installs "
                                            "don't trigger a 90GB re-pull on the next run."),
    skip_data: bool = typer.Option(False, "--skip-data"),
    skip_torch: bool = typer.Option(False, "--skip-torch"),
    skip_wheels: bool = typer.Option(False, "--skip-wheels"),
    skip_requirements: bool = typer.Option(False, "--skip-requirements"),
) -> None:
    if data_mode not in ("all", "sample", "none"):
        _err(f"--data-mode must be all|sample|none, got {data_mode!r}")
        raise typer.Exit(2)

    if not is_colab():
        _warn("This does not look like a Colab runtime. setup is a no-op outside Colab.")
        _warn("On Colab, run the Colab notebook from notebooks/Colab.ipynb instead.")

    from colab_setup import run_full_setup
    ok = run_full_setup(
        data_mode=data_mode,
        refresh_data=refresh_data,
        skip_data=skip_data,
        skip_torch=skip_torch,
        skip_wheels=skip_wheels,
        skip_requirements=skip_requirements,
    )
    if ok:
        _ok("Setup complete.")
    else:
        _err("Setup finished with errors.")
        raise typer.Exit(1)


# =============================================================================
# data
# =============================================================================

@app.command(help="Sync processed data from Drive into the local project tree.")
def data(
    mode: str = typer.Option("all", "--mode", "-m", help="all | sample"),
    refresh: bool = typer.Option(False, "--refresh", help="Wipe destination first."),
) -> None:
    if mode not in ("all", "sample"):
        _err(f"--mode must be all|sample, got {mode!r}")
        raise typer.Exit(2)
    from scripts.sync_drive import pull_data
    n = pull_data(mode=mode, refresh=refresh)
    _ok(f"Pulled {n} files for mode={mode}.")


# =============================================================================
# train
# =============================================================================

@app.command(help="Train SeisMambaKAN. Per-flag values override values in config.yaml.")
def train(
    epochs: Optional[int] = typer.Option(None, "--epochs", "-e"),
    batch_size: Optional[int] = typer.Option(None, "--batch-size", "-b"),
    lr: Optional[float] = typer.Option(None, "--lr"),
    seed: Optional[int] = typer.Option(None, "--seed"),
    data_mode: Optional[str] = typer.Option(None, "--data-mode",
                                             help="Override data.mode (all|sample)."),
    amp: Optional[bool] = typer.Option(None, "--amp/--no-amp",
                                        help="Override model.use_amp."),
    resume: Optional[int] = typer.Option(None, "--resume",
                                          help="Resume training in experiments/exp_NNN/ from its last.pth."),
) -> None:
    main_overrides: dict = {}
    if epochs is not None:        main_overrides["training.epochs"] = int(epochs)
    if batch_size is not None:    main_overrides["training.batch_size"] = int(batch_size)
    if lr is not None:            main_overrides["training.learning_rate"] = float(lr)
    if seed is not None:          main_overrides["training.seed"] = int(seed)
    if data_mode is not None:
        if data_mode not in ("all", "sample"):
            _err("--data-mode must be all|sample")
            raise typer.Exit(2)
        main_overrides["data.mode"] = data_mode

    model_overrides: dict = {}
    if amp is not None:
        model_overrides["model.use_amp"] = bool(amp)

    # Ensure cwd = project root so trainer's relative paths work.
    os.chdir(project_root())

    from src.trainer import main as trainer_main
    trainer_main(
        overrides=main_overrides or None,
        model_overrides=model_overrides or None,
        resume_exp=resume,
    )
    _ok("Training completed.")


# =============================================================================
# eval
# =============================================================================

@app.command(help="Evaluate a trained checkpoint. Auto-picks the latest experiment by default.")
def eval(
    exp: Optional[int] = typer.Option(None, "--exp", help="Experiment id (3-digit)."),
    ckpt: Optional[str] = typer.Option(None, "--ckpt", help="Explicit checkpoint path."),
    split: str = typer.Option("val", "--split", "-s", help="val | test"),
    prefer: str = typer.Option("best", "--prefer", help="best | last | auto"),
    data_mode: Optional[str] = typer.Option(None, "--data-mode",
                                             help="Override data.mode (all|sample)."),
    from_sweep: bool = typer.Option(False, "--from-sweep",
                                     help="Load best picker/detection thresholds "
                                          "from results/exp_NNN/<split>/threshold_sweep.json."),
    no_drive_mirror: bool = typer.Option(False, "--no-drive-mirror"),
) -> None:
    if split not in ("val", "test"):
        _err("--split must be val|test")
        raise typer.Exit(2)
    if data_mode is not None and data_mode not in ("all", "sample"):
        _err("--data-mode must be all|sample")
        raise typer.Exit(2)

    os.chdir(project_root())
    from evaluate import main as eval_main
    argv = ["--split", split, "--prefer", prefer]
    if exp is not None:
        argv += ["--exp", str(exp)]
    elif ckpt is not None:
        argv += ["--ckpt", ckpt]
    if data_mode is not None:
        argv += ["--data-mode", data_mode]
    if from_sweep:
        argv += ["--from-sweep"]
    if no_drive_mirror:
        argv += ["--no-drive-mirror"]
    eval_main(argv)


# =============================================================================
# infer
# =============================================================================

@app.command(help="Run inference on a single trace and plot the result.")
def infer(
    exp: Optional[int] = typer.Option(None, "--exp"),
    ckpt: Optional[str] = typer.Option(None, "--ckpt"),
    split: str = typer.Option("test", "--split", "-s"),
    index: Optional[int] = typer.Option(None, "--index", "-i"),
    prefer: str = typer.Option("best", "--prefer"),
    data_mode: Optional[str] = typer.Option(None, "--data-mode",
                                             help="Override data.mode (all|sample)."),
    no_save: bool = typer.Option(False, "--no-save"),
    no_show: bool = typer.Option(False, "--no-show"),
    no_drive_mirror: bool = typer.Option(False, "--no-drive-mirror"),
) -> None:
    if split not in ("val", "test"):
        _err("--split must be val|test")
        raise typer.Exit(2)
    if data_mode is not None and data_mode not in ("all", "sample"):
        _err("--data-mode must be all|sample")
        raise typer.Exit(2)

    os.chdir(project_root())
    from inference import main as infer_main
    argv = ["--split", split, "--prefer", prefer]
    if exp is not None:
        argv += ["--exp", str(exp)]
    elif ckpt is not None:
        argv += ["--ckpt", ckpt]
    if index is not None:
        argv += ["--index", str(index)]
    if data_mode is not None:
        argv += ["--data-mode", data_mode]
    if no_save:        argv.append("--no-save")
    if no_show:        argv.append("--no-show")
    if no_drive_mirror: argv.append("--no-drive-mirror")
    infer_main(argv)


# =============================================================================
# sweep
# =============================================================================

@app.command(help="Grid-search picker thresholds against val to maximise F1 + minimise P/S MAE.")
def sweep(
    exp: Optional[int] = typer.Option(None, "--exp"),
    ckpt: Optional[str] = typer.Option(None, "--ckpt"),
    split: str = typer.Option("val", "--split", "-s"),
    data_mode: Optional[str] = typer.Option(None, "--data-mode",
                                             help="Override data.mode (all|sample)."),
    top_n: int = typer.Option(10, "--top-n"),
    max_batches: Optional[int] = typer.Option(None, "--max-batches",
                                                help="Smoke-cap on inference batches."),
) -> None:
    if split not in ("val", "test"):
        _err("--split must be val|test")
        raise typer.Exit(2)
    if data_mode is not None and data_mode not in ("all", "sample"):
        _err("--data-mode must be all|sample")
        raise typer.Exit(2)

    os.chdir(project_root())
    from scripts.threshold_sweep import main as sweep_main
    argv = ["--split", split, "--top-n", str(top_n)]
    if exp is not None:
        argv += ["--exp", str(exp)]
    elif ckpt is not None:
        argv += ["--ckpt", ckpt]
    if data_mode is not None:
        argv += ["--data-mode", data_mode]
    if max_batches is not None:
        argv += ["--max-batches", str(max_batches)]
    sweep_main(argv)


# =============================================================================
# tb (tensorboard)
# =============================================================================

@app.command(help="Launch TensorBoard pointing at experiments/ (or one specific exp).")
def tb(
    exp: Optional[int] = typer.Option(None, "--exp"),
    port: int = typer.Option(6006, "--port"),
) -> None:
    _, _, paths_cfg = load_all_configs()
    exp_root = project_root() / paths_cfg.get("experiments", {}).get("root_dir", "experiments")

    if exp is not None:
        target = exp_root / f"exp_{exp:03d}"
        if not target.exists():
            _err(f"Experiment not found: {target}")
            raise typer.Exit(1)
    else:
        target = exp_root

    # On Colab use the magic; otherwise plain subprocess.
    if is_colab():
        console.print(
            "[dim]On Colab, run this in a notebook cell instead:[/dim]\n"
            f"  %load_ext tensorboard\n  %tensorboard --logdir {target} --port {port}"
        )
        return

    rc = _run_subprocess(
        ["tensorboard", "--logdir", str(target), "--port", str(port)]
    )
    if rc != 0:
        _err("tensorboard exited non-zero (is it installed?).")
        raise typer.Exit(rc)


# =============================================================================
# push / pull (Drive sync for results & experiments)
# =============================================================================

@app.command(help="Mirror an experiment (and optionally results) to Drive.")
def push(
    exp: int = typer.Option(..., "--exp", help="Experiment id to push."),
    results: bool = typer.Option(True, "--results/--no-results",
                                  help="Also mirror results/exp_XXX/."),
) -> None:
    from scripts.sync_drive import push_experiment, push_results
    n1 = push_experiment(exp_id=exp)
    _ok(f"Pushed {n1} experiment files.")
    if results:
        n2 = push_results(exp_id=exp)
        _ok(f"Pushed {n2} result files.")


@app.command("pull-exp", help="Pull an experiment from Drive to local.")
def pull_exp(
    exp: int = typer.Option(..., "--exp"),
    refresh: bool = typer.Option(False, "--refresh"),
) -> None:
    from scripts.sync_drive import pull_experiment
    n = pull_experiment(exp_id=exp, refresh=refresh)
    _ok(f"Pulled {n} files for exp_{exp:03d}.")


# =============================================================================
# status
# =============================================================================

@app.command(help="Print a snapshot of the project state.")
def status() -> None:
    main_cfg, model_cfg, paths_cfg = load_all_configs()
    root = project_root()

    env_table = Table(title="Environment", show_header=False, expand=False)
    env_table.add_row("Project root", str(root))
    env_table.add_row("Colab", "yes" if is_colab() else "no")
    env_table.add_row("Drive mounted", "yes" if is_drive_mounted() else "no")
    env_table.add_row("Data mode (config)", str(main_cfg.get("data", {}).get("mode")))
    env_table.add_row("Batch size", str(main_cfg.get("training", {}).get("batch_size")))
    env_table.add_row("Epochs", str(main_cfg.get("training", {}).get("epochs")))
    env_table.add_row("LR", str(main_cfg.get("training", {}).get("learning_rate")))
    env_table.add_row("AMP", str(model_cfg.get("model", {}).get("use_amp")))
    console.print(env_table)

    # Data shards present?
    configured_mode = main_cfg.get("data", {}).get("mode")
    shard_counts: dict[str, dict[str, int]] = {"all": {}, "sample": {}}
    data_table = Table(title="Local data shards", expand=False)
    data_table.add_column("mode/split")
    data_table.add_column("count", justify="right")
    data_table.add_column("", justify="left")
    for mode in ("all", "sample"):
        for split in ("train", "val", "test"):
            d = root / "data" / "processed" / mode / split
            n = len(list(d.glob("*.tar"))) if d.exists() else 0
            shard_counts[mode][split] = n
            tag = " ← config" if mode == configured_mode else ""
            data_table.add_row(f"{mode}/{split}", str(n), tag)
    console.print(data_table)

    # Consistency check: does configured mode have data?
    if configured_mode in ("all", "sample"):
        cfg_counts = shard_counts[configured_mode]
        if cfg_counts.get("train", 0) == 0:
            other_mode = "sample" if configured_mode == "all" else "all"
            other_has_data = shard_counts[other_mode].get("train", 0) > 0
            hint = (
                f"`run.py setup --data-mode {configured_mode}` (sync data), "
                f"OR `run.py train --data-mode {other_mode}`"
                if other_has_data
                else f"`run.py setup --data-mode {configured_mode}` (sync data)"
            )
            _warn(
                f"config.data.mode = '{configured_mode}' but "
                f"data/processed/{configured_mode}/train is empty. Fix: " + hint
            )
        else:
            _ok(f"config.data.mode = '{configured_mode}' has data ({cfg_counts['train']} train shards).")

    # Experiments
    exp_root = root / paths_cfg.get("experiments", {}).get("root_dir", "experiments")
    exps = list_experiments(exp_root)
    if not exps:
        console.print(Panel.fit("[yellow]No experiments yet.[/yellow]", title="Experiments"))
    else:
        exp_table = Table(title="Experiments (latest 10)", expand=False)
        exp_table.add_column("exp")
        exp_table.add_column("path")
        exp_table.add_column("best_model.pth", justify="center")
        exp_table.add_column("last.pth", justify="center")
        for p in exps[-10:]:
            has_best = (p / "best_model.pth").exists()
            has_last = (p / "checkpoints" / "last.pth").exists()
            exp_table.add_row(
                p.name,
                str(p.relative_to(root)),
                "✓" if has_best else "—",
                "✓" if has_last else "—",
            )
        console.print(exp_table)

    # Latest checkpoint quick-resolve
    if exps:
        try:
            latest = resolve_experiment_dir(exp_root)
            if latest:
                ck = resolve_checkpoint(latest, prefer="auto")
                console.print(f"[green]Latest checkpoint:[/green] {ck}")
        except FileNotFoundError as e:
            _warn(str(e))


# =============================================================================
# git shortcuts
# =============================================================================

@app.command("git-pull", help="git pull --rebase in the project root.")
def git_pull() -> None:
    rc = _run_subprocess(["git", "pull", "--rebase"], cwd=project_root())
    if rc != 0:
        raise typer.Exit(rc)


if __name__ == "__main__":
    app()
