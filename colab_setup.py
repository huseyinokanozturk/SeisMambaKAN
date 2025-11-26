import os
import sys
import subprocess


# ==========================
# CONFIGURATION
# ==========================
GIT_REPO_URL   = "https://github.com/huseyinokanozturk/SeisMambaKAN"
REPO_DIR_NAME  = "SeisMambaKAN"

# DATA_MODE: "sample", "all", "none"
DATA_MODE        = "sample"
DRIVE_DATA_ROOT  = "/content/drive/MyDrive/Proje_SeisMamba/SeisMambaKAN/data/processed"
LOCAL_DATA_ROOT  = f"/content/{REPO_DIR_NAME}/data/processed"


def run(cmd: str):
    print(f"\n[RUN] {cmd}")
    result = subprocess.run(cmd, shell=True)
    if result.returncode != 0:
        raise RuntimeError(f"Command failed: {cmd}")


# ==========================
# 1) Clone or Pull Repo
# ==========================
os.chdir("/content")

if not os.path.exists(REPO_DIR_NAME):
    run(f"git clone {GIT_REPO_URL} {REPO_DIR_NAME}")
else:
    run(f"git -C {REPO_DIR_NAME} pull")

PROJECT_ROOT = f"/content/{REPO_DIR_NAME}"
os.chdir(PROJECT_ROOT)
print(f"[OK] Project root: {PROJECT_ROOT}")


# ==========================
# 2) Python Path & Env
# ==========================
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

os.environ["SEISMAMBAKAN_ROOT"] = PROJECT_ROOT
print("[OK] sys.path and SEISMAMBAKAN_ROOT set.")


# ==========================
# 3) Mamba + causal-conv1d Install
# ==========================
print("\n[INFO] Installing Mamba stack (causal-conv1d + mamba-ssm)...")
try:
    # causal-conv1d is a hard dependency for mamba-ssm performance
    run("pip install --no-cache-dir 'causal-conv1d>=1.4.0'")
    run("pip install --no-cache-dir 'mamba-ssm>=2.2.0'")
    print("[OK] Mamba stack installed.")
except Exception as e:
    print(f"[WARN] Mamba stack install failed: {e}")
    print("[WARN] You may need to install 'causal-conv1d' and 'mamba-ssm' manually later.")


# ==========================
# 4) Requirements Install
# ==========================
if os.path.exists("requirements.txt"):
    print("\n[INFO] Installing dependencies from requirements.txt ...")
    run("pip install -U pip")
    run("pip install -r requirements.txt")
else:
    print("[WARN] requirements.txt not found.")


# ==========================
# 5) Data Sync (Drive → Local)
# ==========================
mode = DATA_MODE.lower().strip()

if mode in ("sample", "all"):
    src_dir = os.path.join(DRIVE_DATA_ROOT, mode)
    dst_dir = os.path.join(LOCAL_DATA_ROOT, mode)

    if not os.path.exists(src_dir):
        print(f"[ERROR] Source directory does not exist: {src_dir}")
    else:
        # Ensure a clean target directory before copy
        if os.path.exists(dst_dir):
            print(f"[INFO] Clearing existing directory: {dst_dir}")
            run(f'rm -rf "{dst_dir}"')

        os.makedirs(dst_dir, exist_ok=True)

        print(f"[INFO] Copying data '{mode}' from:")
        print(f"      {src_dir}")
        print(f"  →   {dst_dir}")

        # rsync gives a nice global progress bar
        run(f'rsync -ah --info=progress2 "{src_dir}/" "{dst_dir}/"')

        print("[OK] Data copy completed.")

elif mode == "none":
    print("[INFO] DATA_MODE='none', data copy skipped.")

else:
    print(f"[WARN] Unknown DATA_MODE='{DATA_MODE}', skipping data copy.")


# ==========================
# 6) Torch / GPU Check
# ==========================
try:
    import torch
    print(f"\n[OK] torch version: {torch.__version__}")
    print(f"[OK] CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"[OK] GPU: {torch.cuda.get_device_name(0)}")
except Exception as e:
    print("[WARN] torch could not be imported:", e)


# ==========================
# 7) Sanity Imports
# ==========================
def try_import(pkg: str):
    try:
        __import__(pkg)
        print(f"[OK] import {pkg}")
    except Exception as e:
        print(f"[WARN] import {pkg} failed: {e}")

print("\n[INFO] Import sanity check:")
for pkg in ["mamba_ssm", "efficient_kan", "webdataset", "yaml", "tqdm"]:
    try_import(pkg)

print("\n✅ SeisMambaKAN environment ready.")
