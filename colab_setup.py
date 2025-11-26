import os
import sys
import subprocess
from pathlib import Path


# ==========================
# CONFIGURATION
# ==========================
GIT_REPO_URL = "https://github.com/huseyinokanozturk/SeisMambaKAN.git"
REPO_DIR_NAME = "SeisMambaKAN"

# DATA_MODE: "sample", "all", "none"
DATA_MODE = "sample"
DRIVE_DATA_ROOT = "/content/drive/MyDrive/Proje_SeisMamba/SeisMambaKAN/data/processed"
LOCAL_DATA_ROOT = f"/content/{REPO_DIR_NAME}/data/processed"


def run(cmd: str, ignore_error: bool = False, capture_output: bool = False):
    """Execute shell command with proper error handling."""
    print(f"\n[RUN] {cmd}")
    try:
        if capture_output:
            result = subprocess.run(
                cmd, 
                shell=True, 
                capture_output=True, 
                text=True, 
                check=not ignore_error
            )
            return result.stdout
        else:
            result = subprocess.run(cmd, shell=True, check=not ignore_error)
            return result.returncode == 0
    except subprocess.CalledProcessError as e:
        if not ignore_error:
            print(f"[ERROR] Command failed with exit code {e.returncode}: {cmd}")
            raise RuntimeError(f"Command failed: {cmd}")
        else:
            print(f"[WARN] Command failed but continuing: {cmd}")
            return False


def check_drive_mounted():
    """Check if Google Drive is mounted."""
    drive_path = Path("/content/drive")
    if not drive_path.exists():
        print("[ERROR] Google Drive not mounted!")
        print("[INFO] Please mount Drive using:")
        print("       from google.colab import drive")
        print("       drive.mount('/content/drive')")
        return False
    print("[OK] Google Drive is mounted.")
    return True


def sync_github_to_drive(drive_repo_path: str, git_url: str):
    """Sync GitHub repository to Google Drive (optional feature)."""
    print(f"\n[SYNC] GitHub -> Drive updating: {drive_repo_path}")
    drive_repo_path = Path(drive_repo_path)
    
    if not drive_repo_path.exists():
        drive_repo_path.parent.mkdir(parents=True, exist_ok=True)
        run(f'git clone "{git_url}" "{drive_repo_path}"')
    else:
        run(f'git -C "{drive_repo_path}" stash', ignore_error=True)
        run(f'git -C "{drive_repo_path}" pull')
        run(f'git -C "{drive_repo_path}" stash pop', ignore_error=True)
    
    print("[OK] GitHub -> Drive sync completed.")


# ==========================
# 1) Clone or Pull Repo
# ==========================
print("=" * 60)
print("SeisMambaKAN Environment Setup")
print("=" * 60)

os.chdir("/content")

repo_path = Path(f"/content/{REPO_DIR_NAME}")
if not repo_path.exists():
    print(f"\n[INFO] Cloning repository: {GIT_REPO_URL}")
    run(f'git clone "{GIT_REPO_URL}" "{REPO_DIR_NAME}"')
else:
    print(f"\n[INFO] Repository exists, pulling latest changes...")
    run(f'git -C "{REPO_DIR_NAME}" pull')

PROJECT_ROOT = str(repo_path)
os.chdir(PROJECT_ROOT)
print(f"[OK] Project root: {PROJECT_ROOT}")


# ==========================
# 2) Python Path & Env
# ==========================
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

os.environ["SEISMAMBAKAN_ROOT"] = PROJECT_ROOT
print("[OK] sys.path and SEISMAMBAKAN_ROOT configured.")


# ==========================
# 3) Mamba + causal-conv1d Install
# ==========================
print("\n" + "=" * 60)
print("Installing Mamba Stack")
print("=" * 60)

try:
    # Check if already installed
    try:
        import mamba_ssm
        import causal_conv1d
        print("[OK] Mamba stack already installed.")
    except ImportError:
        print("[INFO] Installing causal-conv1d (required for Mamba)...")
        run("pip install --no-cache-dir 'causal-conv1d>=1.4.0'")
        
        print("[INFO] Installing mamba-ssm...")
        run("pip install --no-cache-dir 'mamba-ssm>=2.2.0'")
        
        print("[OK] Mamba stack installed successfully.")
        
except Exception as e:
    print(f"[ERROR] Mamba stack installation failed: {e}")
    print("[WARN] Continuing setup, but Mamba functionality may be limited.")
    print("[INFO] You can install manually later with:")
    print("       pip install causal-conv1d mamba-ssm")


# ==========================
# 4) Requirements Install
# ==========================
print("\n" + "=" * 60)
print("Installing Project Dependencies")
print("=" * 60)

requirements_file = Path("requirements.txt")
if requirements_file.exists():
    print("[INFO] Upgrading pip...")
    run("pip install --upgrade pip", ignore_error=True)
    
    print("[INFO] Installing dependencies from requirements.txt...")
    run("pip install -r requirements.txt")
    print("[OK] Dependencies installed.")
else:
    print("[WARN] requirements.txt not found, skipping dependency installation.")


# ==========================
# 5) Data Sync (Drive → Local)
# ==========================
print("\n" + "=" * 60)
print("Data Synchronization")
print("=" * 60)

mode = DATA_MODE.lower().strip()

if mode in ("sample", "all"):
    # Check if Drive is mounted
    if not check_drive_mounted():
        print("[ERROR] Cannot sync data without Drive access.")
        print("[INFO] Continuing setup without data...")
    else:
        src_dir = Path(DRIVE_DATA_ROOT) / mode
        dst_dir = Path(LOCAL_DATA_ROOT) / mode

        if not src_dir.exists():
            print(f"[ERROR] Source directory not found: {src_dir}")
            print("[WARN] Please ensure data exists in Google Drive at:")
            print(f"       {src_dir}")
        else:
            # Clean and prepare destination
            if dst_dir.exists():
                print(f"[INFO] Removing existing data directory: {dst_dir}")
                run(f'rm -rf "{dst_dir}"')

            dst_dir.mkdir(parents=True, exist_ok=True)

            print(f"[INFO] Syncing data mode: '{mode}'")
            print(f"      Source: {src_dir}")
            print(f"      Target: {dst_dir}")

            # Use rsync for efficient copying with progress
            rsync_available = run("which rsync", ignore_error=True, capture_output=True)
            if rsync_available and rsync_available.strip():
                run(f'rsync -ah --info=progress2 "{src_dir}/" "{dst_dir}/"')
            else:
                print("[WARN] rsync not found, using cp instead...")
                run(f'cp -r "{src_dir}/"* "{dst_dir}/"')

            print("[OK] Data synchronization completed.")

elif mode == "none":
    print("[INFO] DATA_MODE='none', skipping data synchronization.")
else:
    print(f"[ERROR] Invalid DATA_MODE='{DATA_MODE}'")
    print("[INFO] Valid options: 'sample', 'all', 'none'")


# ==========================
# 6) Torch / GPU Check
# ==========================
print("\n" + "=" * 60)
print("PyTorch & GPU Configuration")
print("=" * 60)

try:
    import torch
    print(f"[OK] PyTorch version: {torch.__version__}")
    print(f"[OK] CUDA available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"[OK] GPU Device: {torch.cuda.get_device_name(0)}")
        print(f"[OK] CUDA version: {torch.version.cuda}")
        print(f"[OK] cuDNN version: {torch.backends.cudnn.version()}")
        
        # Memory info
        total_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"[OK] GPU Memory: {total_mem:.2f} GB")
    else:
        print("[WARN] CUDA not available, will use CPU (slower performance)")
        
except ImportError as e:
    print(f"[ERROR] PyTorch not installed: {e}")
    print("[INFO] Installing PyTorch...")
    run("pip install torch torchvision torchaudio")


# ==========================
# 7) Sanity Imports
# ==========================
print("\n" + "=" * 60)
print("Import Verification")
print("=" * 60)

def try_import(pkg: str, friendly_name: str = None):
    """Try importing a package and report status."""
    display_name = friendly_name or pkg
    try:
        __import__(pkg)
        print(f"✓ {display_name}")
        return True
    except ImportError as e:
        print(f"✗ {display_name} - {str(e)}")
        return False

critical_packages = {
    "mamba_ssm": "Mamba SSM",
    "efficient_kan": "Efficient KAN",
    "torch": "PyTorch",
    "numpy": "NumPy",
}

optional_packages = {
    "webdataset": "WebDataset",
    "yaml": "PyYAML",
    "tqdm": "tqdm",
    "matplotlib": "Matplotlib",
    "seaborn": "Seaborn",
}

print("\nCritical packages:")
critical_ok = all(try_import(pkg, name) for pkg, name in critical_packages.items())

print("\nOptional packages:")
optional_ok = all(try_import(pkg, name) for pkg, name in optional_packages.items())


# ==========================
# 8) Final Status Report
# ==========================
print("\n" + "=" * 60)
print("Setup Summary")
print("=" * 60)

print(f"✓ Repository: {PROJECT_ROOT}")
print(f"✓ Python: {sys.version.split()[0]}")

if critical_ok:
    print("✓ All critical packages installed")
else:
    print("✗ Some critical packages missing - please review errors above")

if optional_ok:
    print("✓ All optional packages installed")
else:
    print("⚠ Some optional packages missing")

print("\n" + "=" * 60)
if critical_ok:
    print("✅ SeisMambaKAN environment ready!")
else:
    print("⚠️  Setup completed with warnings - check messages above")
print("=" * 60)