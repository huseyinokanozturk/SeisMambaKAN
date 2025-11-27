#!/usr/bin/env python3
"""
SeisMambaKAN Colab Setup

1. Clone or update the project from GitHub into /content.
2. Optionally update the copy on Google Drive (if it exists as a git repo).
3. Copy processed data from Drive to Colab with a progress bar.
4. Configure Python environment and install required packages.
"""

import os
import sys
import shutil
import subprocess
from pathlib import Path


# ====================== CONFIG ======================

GIT_REPO_URL = "https://github.com/huseyinokanozturk/SeisMambaKAN.git"

COLAB_PROJECT = "/content/SeisMambaKAN"

# Data copy mode: "sample", "all", or "none"
DATA_MODE = "sample"

DRIVE_PROJECT = "/content/drive/MyDrive/Proje_SeisMamba/SeisMambaKAN"
DRIVE_DATA = f"{DRIVE_PROJECT}/data/processed"
COLAB_DATA = f"{COLAB_PROJECT}/data/processed"


# ====================== HELPERS ======================

def run(cmd: str, cwd: str | None = None) -> bool:
    """Run a shell command, print warnings on failure, return True if it succeeds."""
    result = subprocess.run(
        cmd,
        shell=True,
        cwd=cwd,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        print(f"[WARN] Command failed: {cmd}")
        if result.stderr:
            print(result.stderr.strip())
    return result.returncode == 0


def has_module(name: str) -> bool:
    """Return True if a Python module can be imported."""
    try:
        __import__(name)
        return True
    except Exception:
        return False


def ensure_tqdm():
    """Ensure tqdm is available for the progress bar."""
    try:
        from tqdm import tqdm  # noqa: F401
        return
    except Exception:
        print("[INFO] Installing tqdm for progress bar...")
        run("pip install -q tqdm")
        try:
            from tqdm import tqdm  # noqa: F401
        except Exception:
            print("[WARN] Failed to import tqdm even after installation.")


def copy_data_with_progress(src: Path, dst: Path):
    """Copy all files from src to dst with a progress bar."""
    from tqdm import tqdm

    if not src.exists():
        print(f"[WARN] Source data directory does not exist: {src}")
        return

    files = [p for p in src.rglob("*") if p.is_file()]
    total = len(files)

    if total == 0:
        print(f"[INFO] No files found in: {src}")
        return

    print(f"[INFO] Copying {total} files from {src} to {dst} ...")

    for file_path in tqdm(files, desc="Copying data", unit="file"):
        rel = file_path.relative_to(src)
        target_path = dst / rel
        target_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(file_path, target_path)

    print("[INFO] Data copy completed.")


# ====================== STEPS ======================

def update_drive_repo():
    """If the repo exists on Drive and is a git repo, run git pull."""
    print("\n[0/5] Updating project on Google Drive (optional)...")

    if not Path("/content/drive").exists():
        print("[INFO] Drive is not mounted. Skipping Drive repo update.")
        return

    drive_path = Path(DRIVE_PROJECT)

    if not drive_path.exists():
        print(f"[INFO] Drive project directory does not exist: {DRIVE_PROJECT}")
        return

    if not (drive_path / ".git").exists():
        print("[INFO] Drive project is not a git repository. Skipping.")
        return

    print(f"[INFO] Running git pull in {DRIVE_PROJECT} ...")
    ok_stash = run("git stash", cwd=DRIVE_PROJECT)
    ok_pull = run("git pull --rebase", cwd=DRIVE_PROJECT)
    ok_pop = run("git stash pop", cwd=DRIVE_PROJECT)

    if ok_pull:
        print("[OK] Drive repository updated.")
    else:
        print("[WARN] Drive repository could not be updated cleanly. Please fix it manually if needed.")
        if not ok_stash or not ok_pop:
            print("[WARN] Stash operations also reported issues.")


def setup_colab_repo():
    """Clone or update the repo in /content."""
    print("\n[1/5] Preparing project in Colab...")

    colab_path = Path(COLAB_PROJECT)

    if colab_path.exists() and (colab_path / ".git").exists():
        print(f"[INFO] Existing git repo found at {COLAB_PROJECT}, running git pull...")
        run("git stash", cwd=COLAB_PROJECT)
        run("git pull --rebase", cwd=COLAB_PROJECT)
        run("git stash pop", cwd=COLAB_PROJECT)
        print("[OK] Colab repository updated.")
        return

    if colab_path.exists() and not (colab_path / ".git").exists():
        print(f"[ERROR] {COLAB_PROJECT} exists but is not a git repo.")
        print("        Please remove or rename this directory and rerun the script.")
        sys.exit(1)

    print(f"[INFO] Cloning repository from {GIT_REPO_URL} into /content...")
    run(f"git clone {GIT_REPO_URL} {COLAB_PROJECT}", cwd="/content")
    print("[OK] Repository cloned into Colab.")


def copy_data():
    """Copy processed data from Drive to Colab with a progress bar."""
    print("\n[2/5] Copying processed data from Drive to Colab...")

    if DATA_MODE == "none":
        print("[INFO] DATA_MODE='none'. Skipping data copy.")
        return

    if not Path("/content/drive").exists():
        print("[WARN] Drive is not mounted. Cannot copy data.")
        return

    src_data = Path(DRIVE_DATA) / DATA_MODE
    dst_data = Path(COLAB_DATA) / DATA_MODE

    if not src_data.exists():
        print(f"[WARN] Source data directory does not exist: {src_data}")
        return

    if dst_data.exists():
        print(f"[INFO] Removing existing target directory: {dst_data}")
        shutil.rmtree(dst_data, ignore_errors=True)

    dst_data.parent.mkdir(parents=True, exist_ok=True)

    ensure_tqdm()
    try:
        from tqdm import tqdm  # noqa: F401
    except Exception:
        print("[WARN] tqdm not available. Copying without progress bar.")
        run(f"cp -r {src_data} {dst_data}")
        return

    copy_data_with_progress(src_data, dst_data)


def configure_python_env():
    """Set working directory, sys.path and environment variables."""
    print("\n[3/5] Configuring Python environment...")

    os.chdir(COLAB_PROJECT)

    if COLAB_PROJECT not in sys.path:
        sys.path.insert(0, COLAB_PROJECT)

    os.environ["SEISMAMBAKAN_ROOT"] = COLAB_PROJECT

    print(f"[OK] Working directory: {COLAB_PROJECT}")
    print(f"[OK] SEISMAMBAKAN_ROOT set to: {COLAB_PROJECT}")


def install_packages():
    """Install mamba-ssm, causal-conv1d, efficient-kan and requirements.txt."""
    print("\n[4/5] Installing Python packages...")

    # mamba-ssm + causal-conv1d (official way)
    if not has_module("mamba_ssm"):
        print("[INFO] Installing mamba-ssm (with causal-conv1d) from PyPI...")
        ok = run("pip install -q 'mamba-ssm[causal-conv1d]'")
        if not ok:
            print("[WARN] mamba-ssm[causal-conv1d] failed, trying separate install...")
            run("pip install -q mamba-ssm causal-conv1d")

    # efficient-kan (from GitHub, not on PyPI)
    if not has_module("efficient_kan"):
        print("[INFO] Installing efficient-kan from GitHub...")
        run("pip install -q 'efficient-kan @ git+https://github.com/Blealtan/efficient-kan.git'")

    # Other dependencies from requirements.txt
    req_path = Path(COLAB_PROJECT) / "requirements.txt"
    if req_path.exists():
        print(f"[INFO] Installing requirements from {req_path} ...")
        run(f"pip install -q -r {req_path}")
        print("[OK] requirements.txt installation finished.")
    else:
        print("[INFO] requirements.txt not found, skipping.")


def final_check():
    """Sanity checks: imports and GPU availability."""
    print("\n[5/5] Running final checks...")

    errors: list[str] = []
    for pkg in ["torch", "numpy", "mamba_ssm", "efficient_kan"]:
        try:
            __import__(pkg)
            print(f"[OK] {pkg} import succeeded.")
        except Exception:
            print(f"[FAIL] Could not import package: {pkg}")
            errors.append(pkg)

    try:
        import torch  # noqa: F401

        if torch.cuda.is_available():
            name = torch.cuda.get_device_name(0)
            print(f"[OK] GPU detected: {name}")
        else:
            print("[WARN] No GPU detected. Running in CPU mode.")
    except Exception:
        print("[WARN] torch is not available or failed to import.")

    print("\n" + "=" * 50)
    if not errors:
        print("[READY] Environment is ready.")
        print(f"Project directory: {COLAB_PROJECT}")
        print("\nIn your notebook, you can run:")
        print("  import os, sys")
        print(f"  sys.path.insert(0, '{COLAB_PROJECT}')")
        print(f"  os.chdir('{COLAB_PROJECT}')")
        print("\nThen start your training script, for example:")
        print("  !python train.py")
    else:
        print("[ATTENTION] Some packages are missing:")
        print("  " + ", ".join(errors))
        print("Please install them manually with:")
        print("  pip install <package-name>")
    print("=" * 50)


# ====================== MAIN ======================

def main():
    print("=" * 50)
    print("SeisMambaKAN Colab Setup")
    print("=" * 50)

    update_drive_repo()
    setup_colab_repo()
    copy_data()
    configure_python_env()
    install_packages()
    final_check()


if __name__ == "__main__":
    main()
