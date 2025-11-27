#!/usr/bin/env python3
"""
SeisMambaKAN Colab Setup (Mamba from wheels inside project folder)

- Wheels directory is now:
      /content/SeisMambaKAN/wheels
  and Google Drive mirror:
      /content/drive/MyDrive/Proje_SeisMamba/SeisMambaKAN/wheels
"""

import os
import sys
import shutil
import subprocess
from pathlib import Path


# ====================== CONFIG ======================

GIT_REPO_URL = "https://github.com/huseyinokanozturk/SeisMambaKAN.git"

COLAB_PROJECT = "/content/SeisMambaKAN"

DATA_MODE = "sample"  # "sample", "all", "none"

DRIVE_PROJECT = "/content/drive/MyDrive/Proje_SeisMamba/SeisMambaKAN"
DRIVE_DATA = f"{DRIVE_PROJECT}/data/processed"
COLAB_DATA = f"{COLAB_PROJECT}/data/processed"

# 🔥 Wheels folder moved inside SeisMambaKAN repo
DRIVE_WHEELS_DIR = f"{DRIVE_PROJECT}/wheels"
COLAB_WHEELS_DIR = f"{COLAB_PROJECT}/wheels"


# ====================== HELPERS ======================

def run(cmd: str, cwd: str | None = None) -> bool:
    result = subprocess.run(
        cmd, shell=True, cwd=cwd,
        capture_output=True, text=True
    )
    if result.returncode != 0:
        print(f"[WARN] Command failed: {cmd}")
        if result.stderr:
            print(result.stderr.strip())
    return result.returncode == 0


def ensure_tqdm():
    try:
        from tqdm import tqdm  # noqa
        return
    except:
        print("[INFO] Installing tqdm...")
        run("pip install -q tqdm")


def copy_data_with_progress(src: Path, dst: Path):
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
    print("\n[0/6] Updating Drive repo (optional)...")

    if not Path("/content/drive").exists():
        print("[INFO] Drive not mounted — skipping.")
        return

    if not (Path(DRIVE_PROJECT) / ".git").exists():
        print("[INFO] Drive repo missing or not git — skipping.")
        return

    run("git stash", cwd=DRIVE_PROJECT)
    run("git pull --rebase", cwd=DRIVE_PROJECT)
    run("git stash pop", cwd=DRIVE_PROJECT)
    print("[OK] Drive repo update attempted.")


def setup_colab_repo():
    print("\n[1/6] Preparing Colab repo...")

    colab_path = Path(COLAB_PROJECT)

    if colab_path.exists() and (colab_path / ".git").exists():
        print("[INFO] Updating existing repo...")
        run("git stash", cwd=COLAB_PROJECT)
        run("git pull --rebase", cwd=COLAB_PROJECT)
        run("git stash pop", cwd=COLAB_PROJECT)
        print("[OK] Repo updated.")
        return

    if colab_path.exists():
        print(f"[ERROR] {COLAB_PROJECT} exists but is NOT a git repo.")
        sys.exit(1)

    print("[INFO] Cloning repo fresh...")
    run(f"git clone {GIT_REPO_URL} {COLAB_PROJECT}", cwd="/content")
    print("[OK] Repo cloned.")


def copy_data():
    print("\n[2/6] Copying data from Drive → Colab...")

    if DATA_MODE == "none":
        print("[INFO] Skipping data copy.")
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
    print("\n[3/6] Configuring Python environment...")

    os.chdir(COLAB_PROJECT)
    if COLAB_PROJECT not in sys.path:
        sys.path.insert(0, COLAB_PROJECT)
    os.environ["SEISMAMBAKAN_ROOT"] = COLAB_PROJECT

    print(f"[OK] Working directory: {COLAB_PROJECT}")
    print("[OK] SEISMAMBAKAN_ROOT set.")


def install_mamba_from_wheels():
    print("\n[4/6] Installing Mamba from wheels...")

    # Already installed?
    if has_module("mamba_ssm"):
        print("[OK] mamba_ssm already installed.")
        return

    # Copy wheels from Drive → Colab project
    wheels_src = Path(DRIVE_WHEELS_DIR)
    wheels_dst = Path(COLAB_WHEELS_DIR)

    if not wheels_src.exists():
        print(f"[WARN] Wheels not found on Drive: {wheels_src}")
        print("       You must build wheels first.")
        return

    wheels_dst.mkdir(parents=True, exist_ok=True)
    run(f"cp {wheels_src}/*.whl {wheels_dst}/")

    # Install from local project wheels folder
    cmd = f"pip install -q {wheels_dst}/causal_conv1d-*.whl {wheels_dst}/mamba_ssm-*.whl"
    ok = run(cmd)

    if ok and has_module("mamba_ssm"):
        print("[OK] mamba_ssm installed from project wheels.")
    else:
        print("[WARN] Failed to install mamba_ssm.")


def install_requirements():
    print("\n[5/6] Installing packages from requirements.txt...")

    req = Path(COLAB_PROJECT) / "requirements.txt"
    if req.exists():
        run(f"pip install -q -r {req}")
        print("[OK] requirements installed.")
    else:
        print("[INFO] No requirements.txt — skipping.")


def final_check():
    print("\n[6/6] Final checks...")

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
    except:
        print("[WARN] torch check failed.")

    print("\n" + "=" * 50)
    if errors:
        print("Missing:", ", ".join(errors))
    else:
        print("Environment ready.")
    print("=" * 50)


def main():
    print("=" * 50)
    print("SeisMambaKAN Colab Setup (Wheels inside project)")
    print("=" * 50)

    update_drive_repo()
    setup_colab_repo()
    copy_data()
    configure_python_env()
    install_mamba_from_wheels()
    install_requirements()
    final_check()


if __name__ == "__main__":
    main()
