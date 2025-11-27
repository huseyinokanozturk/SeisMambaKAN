#!/usr/bin/env python3
"""
SeisMambaKAN Colab Setup (Python 3.10, with Mamba + KAN)

1. Clone or update the project from GitHub into /content.
2. Optionally update the copy on Google Drive (if it's a git repo).
3. Copy processed data from Drive to Colab with a progress bar.
4. Configure Python environment and install required packages (Mamba, KAN, etc.).
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
        print("[INFO] Installing tqdm...")
        run("pip install -q tqdm")


def copy_data_with_progress(src: Path, dst: Path):
    """Copy all files from src to dst with a progress bar."""
    from tqdm import tqdm

    if not src.exists():
        print(f"[WARN] Source directory does not exist: {src}")
        return

    files = [p for p in src.rglob("*") if p.is_file()]
    total = len(files)

    if total == 0:
        print(f"[INFO] No files found in: {src}")
        return

    print(f"[INFO] Copying {total} files from {src} to {dst}")

    for file_path in tqdm(files, desc="Copying data", unit="file"):
        rel = file_path.relative_to(src)
        target_path = dst / rel
        target_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(file_path, target_path)

    print("[INFO] Data copy completed.")


# ====================== STEPS ======================

def update_drive_repo():
    print("\n[0/5] Updating project on Google Drive (optional)...")

    if not Path("/content/drive").exists():
        print("[INFO] Drive is not mounted — skipping.")
        return

    drive_path = Path(DRIVE_PROJECT)

    if not drive_path.exists() or not (drive_path / ".git").exists():
        print("[INFO] Drive project not found or not a git repo — skipping.")
        return

    print(f"[INFO] Running git pull in {DRIVE_PROJECT}")
    run("git stash", cwd=DRIVE_PROJECT)
    run("git pull --rebase", cwd=DRIVE_PROJECT)
    run("git stash pop", cwd=DRIVE_PROJECT)
    print("[OK] Drive repo update finished (with possible warnings above).")


def setup_colab_repo():
    print("\n[1/5] Preparing project in Colab...")

    colab_path = Path(COLAB_PROJECT)

    if colab_path.exists() and (colab_path / ".git").exists():
        print("[INFO] Existing Colab repo found — updating...")
        run("git stash", cwd=COLAB_PROJECT)
        run("git pull --rebase", cwd=COLAB_PROJECT)
        run("git stash pop", cwd=COLAB_PROJECT)
        print("[OK] Colab repo updated.")
        return

    if colab_path.exists() and not (colab_path / ".git").exists():
        print(f"[ERROR] {COLAB_PROJECT} exists but is NOT a git repo.")
        sys.exit(1)

    print(f"[INFO] Cloning repo: {GIT_REPO_URL}")
    run(f"git clone {GIT_REPO_URL} {COLAB_PROJECT}", cwd="/content")
    print("[OK] Repository cloned.")


def copy_data():
    print("\n[2/5] Copying processed data from Drive to Colab...")

    if DATA_MODE == "none":
        print("[INFO] DATA_MODE='none' — skipping data copy.")
        return

    if not Path("/content/drive").exists():
        print("[WARN] Drive is not mounted — cannot copy data.")
        return

    src = Path(DRIVE_DATA) / DATA_MODE
    dst = Path(COLAB_DATA) / DATA_MODE

    if not src.exists():
        print(f"[WARN] Data not found: {src}")
        return

    if dst.exists():
        shutil.rmtree(dst, ignore_errors=True)

    dst.parent.mkdir(parents=True, exist_ok=True)

    ensure_tqdm()
    copy_data_with_progress(src, dst)


def configure_python_env():
    print("\n[3/5] Configuring Python environment...")

    os.chdir(COLAB_PROJECT)

    if COLAB_PROJECT not in sys.path:
        sys.path.insert(0, COLAB_PROJECT)

    os.environ["SEISMAMBAKAN_ROOT"] = COLAB_PROJECT

    print(f"[OK] Working directory: {COLAB_PROJECT}")
    print(f"[OK] SEISMAMBAKAN_ROOT set.")


def install_packages():
    print("\n[4/5] Installing Python packages...")

    # Mamba + causal-conv1d (binary wheels only, no source build)
    if not has_module("mamba_ssm"):
        print("[INFO] Installing mamba-ssm (binary wheel only, with causal-conv1d)...")
        ok = run("pip install -q --only-binary=:all: 'mamba-ssm[causal-conv1d]'")
        if not ok:
            print("[WARN] Could not install mamba-ssm from binary wheel. Skipping Mamba.")
    else:
        print("[OK] mamba_ssm already installed.")

    # efficient-kan (from GitHub)
    if not has_module("efficient_kan"):
        print("[INFO] Installing efficient-kan from GitHub...")
        run("pip install -q 'efficient-kan @ git+https://github.com/Blealtan/efficient-kan.git'")
    else:
        print("[OK] efficient_kan already installed.")

    # requirements.txt (should NOT include mamba-ssm / efficient-kan)
    req = Path(COLAB_PROJECT) / "requirements.txt"
    if req.exists():
        print(f"[INFO] Installing requirements from {req}")
        run(f"pip install -q -r {req}")
    else:
        print("[INFO] No requirements.txt found — skipping.")

    print("[OK] Package installation finished.")


def final_check():
    print("\n[5/5] Running final checks...")

    errors = []

    for pkg in ["torch", "numpy", "mamba_ssm", "efficient_kan"]:
        try:
            __import__(pkg)
            print(f"[OK] {pkg} imported.")
        except Exception:
            print(f"[FAIL] {pkg} missing.")
            errors.append(pkg)

    try:
        import torch
        if torch.cuda.is_available():
            print(f"[OK] GPU: {torch.cuda.get_device_name(0)}")
        else:
            print("[WARN] GPU not available.")
    except Exception:
        print("[WARN] torch not available for GPU check.")

    print("\n" + "=" * 50)
    if errors:
        print("Missing packages:", ", ".join(errors))
    else:
        print("Environment ready.")
    print("=" * 50)


# ====================== MAIN ======================

def main():
    print("=" * 50)
    print("SeisMambaKAN Colab Setup (Python 3.10, with Mamba + KAN)")
    print("=" * 50)

    update_drive_repo()
    setup_colab_repo()
    copy_data()
    configure_python_env()
    install_packages()
    final_check()


if __name__ == "__main__":
    main()
