#!/usr/bin/env python3
"""
SeisMambaKAN Colab Setup (Python 3.10)

Steps:
1. Clone/update project from GitHub into /content.
2. Optionally update Drive copy (if it's a git repo).
3. Copy processed data from Drive → Colab with progress bar.
4. Configure Python environment.
5. Install all required packages via requirements.txt (Mamba + KAN included).
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
    """Run a shell command, print warning on failure."""
    result = subprocess.run(
        cmd, shell=True, cwd=cwd, capture_output=True, text=True
    )
    if result.returncode != 0:
        print(f"[WARN] Command failed: {cmd}")
        if result.stderr:
            print(result.stderr.strip())
    return result.returncode == 0


def ensure_tqdm():
    """Ensure tqdm exists for progress bar."""
    try:
        from tqdm import tqdm  # noqa
        return
    except Exception:
        print("[INFO] Installing tqdm...")
        run("pip install -q tqdm")


def copy_data_with_progress(src: Path, dst: Path):
    """Copy files with a tqdm progress bar."""
    from tqdm import tqdm

    if not src.exists():
        print(f"[WARN] Missing data directory: {src}")
        return

    files = [p for p in src.rglob("*") if p.is_file()]
    if not files:
        print(f"[INFO] No files found in: {src}")
        return

    print(f"[INFO] Copying {len(files)} files from {src} to {dst}")

    for file_path in tqdm(files, desc="Copying data", unit="file"):
        rel = file_path.relative_to(src)
        target = dst / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(file_path, target)

    print("[INFO] Data copy OK.")


def has_module(name: str) -> bool:
    try:
        __import__(name)
        return True
    except:
        return False


# ====================== STEPS ======================

def update_drive_repo():
    print("\n[0/5] Updating Drive repo (optional)...")

    if not Path("/content/drive").exists():
        print("[INFO] Drive not mounted — skipping.")
        return

    drive_path = Path(DRIVE_PROJECT)

    if not drive_path.exists() or not (drive_path / ".git").exists():
        print("[INFO] Drive repo not found or not a git repo — skipping.")
        return

    print(f"[INFO] Running git pull in {DRIVE_PROJECT}")
    run("git stash", cwd=DRIVE_PROJECT)
    run("git pull --rebase", cwd=DRIVE_PROJECT)
    run("git stash pop", cwd=DRIVE_PROJECT)
    print("[OK] Drive repo update attempted (see warnings above if any).")


def setup_colab_repo():
    print("\n[1/5] Preparing Colab repo...")

    colab_path = Path(COLAB_PROJECT)

    if colab_path.exists() and (colab_path / ".git").exists():
        print("[INFO] Existing repo found — updating...")
        run("git stash", cwd=COLAB_PROJECT)
        run("git pull --rebase", cwd=COLAB_PROJECT)
        run("git stash pop", cwd=COLAB_PROJECT)
        print("[OK] Repo updated.")
        return

    if colab_path.exists() and not (colab_path / ".git").exists():
        print(f"[ERROR] {COLAB_PROJECT} exists but is NOT a git repo.")
        sys.exit(1)

    print("[INFO] Cloning fresh repo...")
    run(f"git clone {GIT_REPO_URL} {COLAB_PROJECT}", cwd="/content")
    print("[OK] Repo cloned.")


def copy_data():
    print("\n[2/5] Copying data from Drive → Colab...")

    if DATA_MODE == "none":
        print("[INFO] DATA_MODE='none' — skipping.")
        return

    if not Path("/content/drive").exists():
        print("[WARN] Drive not mounted — skipping.")
        return

    src = Path(DRIVE_DATA) / DATA_MODE
    dst = Path(COLAB_DATA) / DATA_MODE

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

    print(f"[OK] Working dir: {COLAB_PROJECT}")
    print(f"[OK] SEISMAMBAKAN_ROOT set.")


def install_packages():
    print("\n[4/5] Installing Python packages (via requirements.txt)...")

    req = Path(COLAB_PROJECT) / "requirements.txt"
    if req.exists():
        print(f"[INFO] Installing: {req}")
        run(f"pip install -q -r {req}")
        print("[OK] requirements.txt installed.")
    else:
        print("[INFO] No requirements.txt found — skipping.")


def final_check():
    print("\n[5/5] Final checks...")

    errors = []
    for pkg in ["torch", "numpy", "mamba_ssm", "efficient_kan"]:
        if has_module(pkg):
            print(f"[OK] {pkg} imported.")
        else:
            print(f"[FAIL] {pkg} missing.")
            errors.append(pkg)

    try:
        import torch
        if torch.cuda.is_available():
            print(f"[OK] GPU: {torch.cuda.get_device_name(0)}")
        else:
            print("[WARN] GPU unavailable.")
    except:
        print("[WARN] torch import failed.")

    print("\n" + "=" * 50)
    if errors:
        print("Missing:", ", ".join(errors))
    else:
        print("Environment ready.")
    print("=" * 50)


# ====================== MAIN ======================

def main():
    print("=" * 50)
    print("SeisMambaKAN Colab Setup (Python 3.10, Mamba + KAN)")
    print("=" * 50)

    update_drive_repo()
    setup_colab_repo()
    copy_data()
    configure_python_env()
    install_packages()
    final_check()


if __name__ == "__main__":
    main()
