#!/usr/bin/env python3
"""
SeisMambaKAN Colab Setup

1. Update project on Google Drive (optional)
2. Sync project from GitHub to /content
3. Copy processed data from Drive to Colab with a progress bar
4. Install PyTorch (2.5.1 + cu121), Mamba wheels, and other requirements
5. Set /content/SeisMambaKAN as working directory and run basic checks
"""

import os
import sys
import shutil
import subprocess
from pathlib import Path

# ----------------- SETTINGS -----------------
GIT_REPO_URL   = "https://github.com/huseyinokanozturk/SeisMambaKAN.git"
COLAB_PROJECT  = "/content/SeisMambaKAN"

DRIVE_PROJECT  = "/content/drive/MyDrive/Proje_SeisMamba/SeisMambaKAN"
DRIVE_DATA_DIR = f"{DRIVE_PROJECT}/data/processed"
COLAB_DATA_DIR = f"{COLAB_PROJECT}/data/processed"
WHEELS_DIR     = f"{DRIVE_PROJECT}/wheels"

# "sample", "all", or "none"
DATA_MODE      = "sample"

TARGET_TORCH_VERSION = "2.5.1+cu121"


def run(cmd: str, desc: str = "") -> bool:
    """Run shell command and print a short status."""
    if desc:
        print(f"[INFO] {desc}")
    result = subprocess.run(cmd, shell=True, text=True)
    if result.returncode != 0:
        print(f"[WARN] Command failed: {cmd}")
        return False
    return True


def update_drive_repo():
    """Optional: git pull inside Drive project (if it is a git repo)."""
    print("\n[0/5] Updating Drive repo (optional)...")
    drive_path = Path(DRIVE_PROJECT)

    if not Path("/content/drive").exists():
        print("[WARN] Drive is not mounted, skipping Drive repo update.")
        return

    if not drive_path.exists():
        print(f"[WARN] Drive project folder does not exist: {DRIVE_PROJECT}")
        return

    if not (drive_path / ".git").exists():
        print("[WARN] Drive project is not a git repository, skipping.")
        return

    os.chdir(DRIVE_PROJECT)
    run("git status -sb", "git status (for your info)")
    run("git stash", "git stash")
    run("git pull --rebase", "git pull --rebase")
    run("git stash pop", "git stash pop")
    print("[OK] Drive repo update attempted (check warnings above if any).")


def prepare_colab_repo():
    """Clone or update the GitHub repo into /content."""
    print("\n[1/5] Preparing Colab repo...")
    colab_path = Path(COLAB_PROJECT)

    if colab_path.exists() and (colab_path / ".git").exists():
        print("[INFO] Existing git repo found in Colab. Pulling latest changes...")
        os.chdir(COLAB_PROJECT)
        run("git stash", "git stash")
        run("git pull --rebase", "git pull --rebase")
        run("git stash pop", "git stash pop")
        print("[OK] Colab repo updated.")
    else:
        print("[INFO] Cloning fresh repo...")
        if colab_path.exists():
            run(f"rm -rf {COLAB_PROJECT}", "Removing non-git folder at COLAB_PROJECT")
        os.chdir("/content")
        run(f"git clone {GIT_REPO_URL} {COLAB_PROJECT}", "Cloning repository")
        print("[OK] Repo cloned.")


def copy_data_with_progress():
    """Copy processed data from Drive to Colab with a progress bar."""
    print("\n[2/5] Copying data from Drive → Colab...")

    if DATA_MODE == "none":
        print("[INFO] DATA_MODE='none', skipping data copy.")
        return

    src_root = Path(DRIVE_DATA_DIR) / DATA_MODE
    dst_root = Path(COLAB_DATA_DIR) / DATA_MODE

    if not Path("/content/drive").exists():
        print("[WARN] Drive is not mounted, cannot copy data.")
        return

    if not src_root.exists():
        print(f"[WARN] Source data not found: {src_root}")
        return

    # Collect files
    files = [p for p in src_root.rglob("*") if p.is_file()]
    total = len(files)
    if total == 0:
        print(f"[WARN] No files found under: {src_root}")
        return

    # Clean destination
    if dst_root.exists():
        run(f"rm -rf {dst_root}", "Removing old data folder in Colab")
    dst_root.mkdir(parents=True, exist_ok=True)

    # Try to use tqdm if available
    try:
        from tqdm import tqdm
        iterator = tqdm(files, desc="Copying data", unit="file")
        use_tqdm = True
    except Exception:
        iterator = files
        use_tqdm = False

    print(f"[INFO] Copying {total} files from {src_root} to {dst_root} ...")
    for idx, src in enumerate(iterator, 1):
        rel = src.relative_to(src_root)
        dst = dst_root / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        if not use_tqdm and idx % 10 == 0:
            print(f"Copied {idx}/{total} files")

    print("[INFO] Data copy completed.")


def configure_python_env():
    """Set working directory and basic environment variables."""
    print("\n[3/5] Configuring Python environment...")
    os.chdir(COLAB_PROJECT)
    if COLAB_PROJECT not in sys.path:
        sys.path.insert(0, COLAB_PROJECT)
    os.environ["SEISMAMBAKAN_ROOT"] = COLAB_PROJECT
    print(f"[OK] Working dir: {COLAB_PROJECT}")
    print(f"[OK] SEISMAMBAKAN_ROOT set.")


def ensure_torch_stack():
    """Ensure torch/vision/audio are installed with the target cu121 version."""
    print("\n[4/5] Installing Python packages (PyTorch + wheels + requirements)...")

    need_install = False
    try:
        import torch  # type: ignore
        current = torch.__version__
        if current != TARGET_TORCH_VERSION:
            print(f"[INFO] Torch version is {current}, expected {TARGET_TORCH_VERSION}. Reinstalling.")
            need_install = True
        else:
            print(f"[OK] Torch version already {TARGET_TORCH_VERSION}.")
    except Exception:
        print("[INFO] Torch is not installed, installing.")
        need_install = True

    if need_install:
        cmd = (
            "pip install -q --index-url https://download.pytorch.org/whl/cu121 "
            "torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1"
        )
        run(cmd, "Installing torch/vision/audio (cu121)")


def install_mamba_from_wheels():
    """Install mamba_ssm and causal_conv1d from wheels stored on Drive."""
    mamba_whl = Path(WHEELS_DIR) / "mamba_ssm-2.2.6.post3-cp312-cp312-linux_x86_64.whl"
    causal_whl = Path(WHEELS_DIR) / "causal_conv1d-1.5.3.post1-cp312-cp312-linux_x86_64.whl"

    if not mamba_whl.exists() or not causal_whl.exists():
        print(f"[WARN] Mamba wheels not found in {WHEELS_DIR}. Skipping Mamba install.")
        return

    cmd = f"pip install -q \"{causal_whl}\" \"{mamba_whl}\""
    run(cmd, "Installing mamba_ssm + causal_conv1d from Drive wheels")
    print("[OK] Mamba + causal_conv1d installed from wheels.")


def install_requirements():
    """Install remaining Python dependencies from requirements.txt."""
    req = Path(COLAB_PROJECT) / "requirements.txt"
    if not req.exists():
        print(f"[WARN] requirements.txt not found at {req}, skipping.")
        return
    cmd = f"pip install -q -r \"{req}\""
    run(cmd, f"Installing requirements from {req}")
    print("[OK] requirements.txt installation finished.")


def final_checks():
    """Import a few core packages and check GPU."""
    print("\n[5/5] Running final checks...")
    errors = []

    def check_pkg(name):
        nonlocal errors
        try:
            __import__(name)
            print(f"[OK] import {name}")
        except Exception:
            print(f"[FAIL] could not import: {name}")
            errors.append(name)

    for pkg in ["torch", "numpy", "mamba_ssm", "efficient_kan"]:
        check_pkg(pkg)

    try:
        import torch  # type: ignore
        if torch.cuda.is_available():
            print(f"[OK] GPU detected: {torch.cuda.get_device_name(0)}")
        else:
            print("[WARN] CUDA GPU not available.")
    except Exception:
        print("[WARN] torch CUDA check failed.")

    print("\n" + "=" * 50)
    if not errors:
        print("✅ READY!")
        print(f"📂 Working directory: {COLAB_PROJECT}")
        print("\nYou can now run, for example:")
        print("   import os, sys")
        print(f"   sys.path.insert(0, '{COLAB_PROJECT}')")
        print(f"   os.chdir('{COLAB_PROJECT}')")
        print("\n   # Then run your training script, e.g.:")
        print("   !python train.py")
    else:
        print("⚠️ Some packages failed to import:", ", ".join(errors))
        print("   You can try: pip install <package-name>")
    print("=" * 50)


if __name__ == "__main__":
    print("=" * 50)
    print("SeisMambaKAN Colab Setup")
    print("=" * 50)

    update_drive_repo()
    prepare_colab_repo()
    copy_data_with_progress()
    configure_python_env()
    ensure_torch_stack()
    install_mamba_from_wheels()
    install_requirements()
    final_checks()
