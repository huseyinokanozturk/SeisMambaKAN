"""
SeisMambaKAN Colab environment setup.

This module exposes small, parameterized functions; it does NOT do anything
on import. `run.py setup` (or directly `python colab_setup.py`) drives them.

All paths come from configs/paths.yaml — nothing is hardcoded here.

Steps performed by `run_full_setup()`:
  1) Ensure Drive is mounted (warn if not).
  2) Pull / clone the GitHub repo into /content/SeisMambaKAN.
  3) Copy processed data from Drive -> Colab (mode = "all" | "sample" | "none").
  4) Install torch (cu121), mamba wheels (from Drive), then requirements.txt.
  5) Sanity check imports + CUDA.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

# We intentionally avoid importing src.utils here, because this file may run
# *before* the repo is cloned. Read paths.yaml ourselves.
import yaml


# =============================================================================
# Small shell helper
# =============================================================================

def _run(cmd: str, desc: str = "", check: bool = False) -> bool:
    if desc:
        print(f"[INFO] {desc}")
    result = subprocess.run(cmd, shell=True, text=True)
    ok = result.returncode == 0
    if not ok:
        msg = f"[WARN] Command failed (rc={result.returncode}): {cmd}"
        if check:
            raise RuntimeError(msg)
        print(msg)
    return ok


def _load_paths_yaml(paths_yaml: Path) -> dict:
    if not paths_yaml.exists():
        raise FileNotFoundError(f"paths.yaml not found at {paths_yaml}")
    with paths_yaml.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


# =============================================================================
# Drive
# =============================================================================

def mount_drive(force: bool = False) -> bool:
    """
    Mount Google Drive at /content/drive. Returns True if mount looks healthy.
    Outside Colab this is a no-op that returns False.
    """
    if Path("/content/drive/MyDrive").exists() and not force:
        print("[OK] Drive already mounted.")
        return True
    try:
        from google.colab import drive  # type: ignore
        drive.mount("/content/drive", force_remount=force)
        ok = Path("/content/drive/MyDrive").exists()
        print("[OK] Drive mounted." if ok else "[WARN] Drive mount did not produce /content/drive/MyDrive.")
        return ok
    except ImportError:
        print("[WARN] google.colab not available; not on Colab? Skipping Drive mount.")
        return False


# =============================================================================
# Repo
# =============================================================================

def prepare_repo(colab_project: str, git_repo_url: str) -> None:
    """Clone fresh, or pull latest if a git repo exists at colab_project."""
    colab_path = Path(colab_project)
    if colab_path.exists() and (colab_path / ".git").exists():
        print(f"[INFO] Pulling latest in {colab_project}")
        os.chdir(colab_project)

        # Only stash if there are local modifications; otherwise skip both
        # stash and pop entirely (avoids the cosmetic "No stash entries" warning).
        dirty = subprocess.run(
            "git diff --quiet && git diff --cached --quiet",
            shell=True,
        ).returncode != 0

        if dirty:
            _run("git stash --include-untracked --quiet", "git stash (local changes)")
            _run("git pull --rebase", "git pull --rebase")
            _run("git stash pop --quiet", "git stash pop")
        else:
            _run("git pull --rebase", "git pull --rebase")
    else:
        if colab_path.exists():
            print(f"[INFO] Removing non-git folder at {colab_project}")
            shutil.rmtree(colab_path, ignore_errors=True)
        print(f"[INFO] Cloning {git_repo_url}")
        os.chdir("/content")
        _run(f"git clone {git_repo_url} {colab_project}", "git clone", check=True)


# =============================================================================
# Data sync (Drive -> Colab)
# =============================================================================

def copy_data(
    drive_data_dir: str,
    colab_data_dir: str,
    data_mode: str,
    refresh: bool = True,
) -> None:
    """
    Copy processed data from Drive to Colab.

    data_mode:
        "all"    -> copy data/processed/all
        "sample" -> copy data/processed/sample
        "none"   -> skip
    refresh:
        True  -> remove existing destination first.
        False -> skip copy if destination already non-empty.
    """
    if data_mode == "none":
        print("[INFO] data_mode='none', skipping data copy.")
        return

    if not Path("/content/drive/MyDrive").exists():
        print("[WARN] Drive not mounted; cannot copy data.")
        return

    src_root = Path(drive_data_dir) / data_mode
    dst_root = Path(colab_data_dir) / data_mode

    if not src_root.exists():
        print(f"[WARN] Source data not found: {src_root}")
        return

    if dst_root.exists() and not refresh:
        any_files = any(dst_root.rglob("*.tar"))
        if any_files:
            print(f"[OK] Data already present at {dst_root}; skipping (refresh=False).")
            return

    if dst_root.exists():
        print(f"[INFO] Removing existing {dst_root}")
        shutil.rmtree(dst_root, ignore_errors=True)
    dst_root.mkdir(parents=True, exist_ok=True)

    files = [p for p in src_root.rglob("*") if p.is_file()]
    total = len(files)
    if total == 0:
        print(f"[WARN] No files under {src_root}")
        return
    print(f"[INFO] Copying {total} files: {src_root} -> {dst_root}")

    try:
        from tqdm.auto import tqdm
        iterator = tqdm(files, desc="Copying data", unit="file")
    except ImportError:
        iterator = files

    for src in iterator:
        rel = src.relative_to(src_root)
        dst = dst_root / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)

    print(f"[OK] Data copy completed ({total} files).")


# =============================================================================
# Python env (working dir, sys.path)
# =============================================================================

def configure_env(colab_project: str) -> None:
    os.chdir(colab_project)
    if colab_project not in sys.path:
        sys.path.insert(0, colab_project)
    os.environ["SEISMAMBAKAN_ROOT"] = colab_project
    print(f"[OK] cwd={colab_project}, SEISMAMBAKAN_ROOT set.")


# =============================================================================
# Packages
# =============================================================================

def ensure_torch(target_version: str) -> None:
    """Install torch / torchvision / torchaudio at target cu121 version if needed."""
    try:
        import torch  # type: ignore
        if torch.__version__ == target_version:
            print(f"[OK] torch=={target_version} already installed.")
            return
        print(f"[INFO] torch is {torch.__version__}, reinstalling {target_version}.")
    except ImportError:
        print(f"[INFO] torch not installed, installing {target_version}.")

    # cu121 channel; lock companion versions to match torch.
    torch_ver = target_version.split("+")[0]
    cmd = (
        "pip install -q --index-url https://download.pytorch.org/whl/cu121 "
        f"torch=={torch_ver} torchvision==0.20.1 torchaudio=={torch_ver}"
    )
    _run(cmd, "Installing torch/vision/audio (cu121)")


def _detect_torch_combo() -> tuple[str, str, str]:
    """
    Inspect the *currently installed* torch and return (py_tag, torch_mm, cuda_major).

    Examples:
        py_tag    -> "cp312"
        torch_mm  -> "2.5"       (major.minor only)
        cuda_major-> "12"        (CUDA major)

    Raises if torch is missing or is a CPU-only build (no CUDA).
    """
    import torch  # type: ignore
    py_tag = f"cp{sys.version_info.major}{sys.version_info.minor}"

    base_ver = torch.__version__.split("+")[0]
    parts = base_ver.split(".")
    torch_mm = f"{parts[0]}.{parts[1]}"

    cuda_full = (torch.version.cuda or "").strip()
    if not cuda_full:
        raise RuntimeError(
            f"torch={torch.__version__} is CPU-only; cannot install CUDA-backed "
            "mamba/causal_conv1d wheels. Switch runtime to GPU and re-run setup."
        )
    cuda_major = cuda_full.split(".")[0]
    return py_tag, torch_mm, cuda_major


def install_mamba_official_wheels(
    mamba_version: str,
    causal_version: str,
    cxx11_abi: str = "FALSE",
) -> bool:
    """
    Install mamba_ssm + causal_conv1d from upstream GitHub release wheels,
    auto-matched to the installed torch (major.minor) + python tag + CUDA major.

    Returns True on success.
    """
    py_tag, torch_mm, cu_major = _detect_torch_combo()

    causal_url = (
        f"https://github.com/Dao-AILab/causal-conv1d/releases/download/"
        f"v{causal_version}/causal_conv1d-{causal_version}"
        f"+cu{cu_major}torch{torch_mm}cxx11abi{cxx11_abi}"
        f"-{py_tag}-{py_tag}-linux_x86_64.whl"
    )
    mamba_url = (
        f"https://github.com/state-spaces/mamba/releases/download/"
        f"v{mamba_version}/mamba_ssm-{mamba_version}"
        f"+cu{cu_major}torch{torch_mm}cxx11abi{cxx11_abi}"
        f"-{py_tag}-{py_tag}-linux_x86_64.whl"
    )

    print(f"[INFO] causal_conv1d: {causal_url}")
    print(f"[INFO] mamba_ssm:    {mamba_url}")

    ok_c = _run(
        f'pip install -q --no-build-isolation "{causal_url}"',
        f"Installing causal_conv1d {causal_version} (torch{torch_mm} cu{cu_major} {py_tag})",
    )
    if not ok_c:
        print("[FAIL] causal_conv1d wheel install failed (URL likely unavailable for this combo).")
        return False

    ok_m = _run(
        f'pip install -q --no-build-isolation "{mamba_url}"',
        f"Installing mamba_ssm {mamba_version} (torch{torch_mm} cu{cu_major} {py_tag})",
    )
    if not ok_m:
        print("[FAIL] mamba_ssm wheel install failed.")
        return False

    print("[OK] Mamba + causal_conv1d installed from upstream release wheels.")
    return True


def install_mamba_from_wheels(
    wheels_dir: str,
    mamba_wheel: str,
    causal_wheel: str,
) -> bool:
    """
    LEGACY: install mamba_ssm + causal_conv1d from wheels stored on Drive.
    Kept as an offline fallback. Returns True on success.

    Prefer install_mamba_official_wheels() — it auto-matches the current torch
    instead of relying on manually-curated Drive wheels.
    """
    wheels = Path(wheels_dir)
    m = wheels / mamba_wheel
    c = wheels / causal_wheel
    if not (m.exists() and c.exists()):
        print(f"[WARN] Drive wheels not found in {wheels}; skipping.")
        print(f"       expected: {mamba_wheel} and {causal_wheel}")
        return False
    ok = _run(f'pip install -q "{c}" "{m}"',
              "Installing mamba_ssm + causal_conv1d from Drive wheels")
    if ok:
        print("[OK] Mamba + causal_conv1d installed from Drive wheels.")
    return ok


def install_requirements(colab_project: str) -> None:
    req = Path(colab_project) / "requirements.txt"
    if not req.exists():
        print(f"[WARN] requirements.txt missing at {req}")
        return
    _run(f'pip install -q -r "{req}"', f"pip install -r {req.name}")
    print("[OK] requirements.txt installed.")


# =============================================================================
# Sanity check
# =============================================================================

def final_checks() -> bool:
    """Import core packages and check CUDA. Returns True if no import errors."""
    errors = []
    for pkg in ("torch", "numpy", "mamba_ssm", "efficient_kan", "webdataset", "typer", "rich"):
        try:
            __import__(pkg)
            print(f"[OK] import {pkg}")
        except ImportError as e:
            print(f"[FAIL] import {pkg}: {e}")
            errors.append(pkg)

    try:
        import torch  # type: ignore
        if torch.cuda.is_available():
            print(f"[OK] GPU: {torch.cuda.get_device_name(0)}")
        else:
            print("[WARN] CUDA not available.")
    except ImportError:
        print("[WARN] could not check CUDA (torch missing).")

    if errors:
        print(f"[FAIL] missing: {', '.join(errors)}")
        return False
    print("[OK] All sanity checks passed.")
    return True


# =============================================================================
# Orchestration
# =============================================================================

def run_full_setup(
    data_mode: str = "sample",
    refresh_data: bool = True,
    skip_data: bool = False,
    skip_torch: bool = False,
    skip_wheels: bool = False,
    skip_requirements: bool = False,
) -> bool:
    """
    Full Colab setup flow. Returns True on clean exit.

    Reads everything else from the freshly-cloned configs/paths.yaml.
    """
    # ----- 0) Drive ---------------------------------------------------------
    print("\n[1/6] Drive mount")
    mount_drive()

    # ----- 1) Repo ----------------------------------------------------------
    # We need the GitHub repo to get the canonical paths.yaml. To avoid a
    # chicken-and-egg, hardcode just two strings here: the Colab project
    # path and the GitHub URL. Everything else comes from paths.yaml.
    colab_project = "/content/SeisMambaKAN"
    git_repo_url = "https://github.com/huseyinokanozturk/SeisMambaKAN.git"

    print("\n[2/6] Repo")
    prepare_repo(colab_project, git_repo_url)

    paths_yaml = Path(colab_project) / "configs" / "paths.yaml"
    paths_cfg = _load_paths_yaml(paths_yaml)
    drive_cfg = paths_cfg.get("drive", {})
    colab_cfg = paths_cfg.get("colab", {})

    drive_data_dir = drive_cfg.get(
        "data_dir",
        "/content/drive/MyDrive/Proje_SeisMamba/SeisMambaKAN/data/processed",
    )
    colab_data_dir = str(Path(colab_project) / "data" / "processed")
    wheels_dir = drive_cfg.get(
        "wheels_dir",
        "/content/drive/MyDrive/Proje_SeisMamba/SeisMambaKAN/wheels",
    )
    target_torch = colab_cfg.get("target_torch_version", "2.5.1+cu121")

    # Upstream wheel versions (auto-matched to torch + python at install time).
    mamba_version = str(colab_cfg.get("mamba_version", "2.2.6.post3"))
    causal_version = str(colab_cfg.get("causal_version", "1.5.3.post1"))
    cxx11_abi = str(colab_cfg.get("wheel_cxx11_abi", "FALSE")).upper()

    # Legacy: Drive wheel filenames, kept as offline fallback.
    legacy_mamba_wheel = colab_cfg.get("mamba_wheel_name")
    legacy_causal_wheel = colab_cfg.get("causal_wheel_name")

    # ----- 2) Data ---------------------------------------------------------
    print("\n[3/6] Data sync")
    if skip_data:
        print("[INFO] --skip-data; not copying.")
    else:
        copy_data(drive_data_dir, colab_data_dir, data_mode, refresh=refresh_data)

    # ----- 3) Env ----------------------------------------------------------
    print("\n[4/6] Python env")
    configure_env(colab_project)

    # ----- 4) Packages -----------------------------------------------------
    print("\n[5/6] Packages")
    if not skip_torch:
        ensure_torch(target_torch)

    if not skip_wheels:
        # Strategy:
        #   1) Try official GitHub release wheels matched to the installed
        #      torch + python + CUDA. This is the supported path.
        #   2) If that fails (offline, etc.) and legacy Drive wheels are
        #      configured, fall back to those.
        wheels_ok = False
        try:
            wheels_ok = install_mamba_official_wheels(
                mamba_version=mamba_version,
                causal_version=causal_version,
                cxx11_abi=cxx11_abi,
            )
        except RuntimeError as e:
            print(f"[WARN] official-wheel install pre-check failed: {e}")

        if not wheels_ok and legacy_mamba_wheel and legacy_causal_wheel:
            print("[INFO] Falling back to Drive wheels (legacy mode).")
            install_mamba_from_wheels(wheels_dir, legacy_mamba_wheel, legacy_causal_wheel)

    if not skip_requirements:
        install_requirements(colab_project)

    # ----- 5) Checks -------------------------------------------------------
    print("\n[6/6] Sanity checks")
    return final_checks()


if __name__ == "__main__":
    # When invoked directly: sensible defaults; use run.py for flags.
    ok = run_full_setup(data_mode="sample")
    sys.exit(0 if ok else 1)
