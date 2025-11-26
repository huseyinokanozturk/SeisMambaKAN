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


def is_git_repo(path: str) -> bool:
    """Check if directory is a git repository."""
    git_dir = Path(path) / ".git"
    return git_dir.exists()


# ==========================
# 1) Clone or Pull Repo
# ==========================
print("=" * 60)
print("SeisMambaKAN Environment Setup")
print("=" * 60)

# Determine if we're already inside the repo or need to clone
current_dir = Path.cwd()
repo_path = Path(f"/content/{REPO_DIR_NAME}")

# Check if current directory IS the repo
if current_dir.name == REPO_DIR_NAME and is_git_repo(current_dir):
    print(f"\n[INFO] Already inside repository: {current_dir}")
    print("[INFO] Pulling latest changes...")
    run("git pull", ignore_error=True)
    PROJECT_ROOT = str(current_dir)
    
elif repo_path.exists() and repo_path != current_dir:
    # Repo exists elsewhere, navigate to it
    print(f"\n[INFO] Repository found at: {repo_path}")
    os.chdir(repo_path)
    
    if is_git_repo(repo_path):
        print("[INFO] Pulling latest changes...")
        run("git pull", ignore_error=True)
    else:
        print("[WARN] Directory exists but is not a git repo, re-cloning...")
        os.chdir("/content")
        run(f'rm -rf "{REPO_DIR_NAME}"')
        run(f'git clone "{GIT_REPO_URL}" "{REPO_DIR_NAME}"')
        os.chdir(repo_path)
    
    PROJECT_ROOT = str(repo_path)
    
else:
    # Repo doesn't exist, clone it
    print(f"\n[INFO] Cloning repository: {GIT_REPO_URL}")
    os.chdir("/content")
    run(f'git clone "{GIT_REPO_URL}" "{REPO_DIR_NAME}"')
    os.chdir(repo_path)
    PROJECT_ROOT = str(repo_path)

print(f"[OK] Project root: {PROJECT_ROOT}")
print(f"[OK] Current directory: {Path.cwd()}")


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
        print(f"    - mamba_ssm version: {mamba_ssm.__version__}")
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

            # Verify sync
            if dst_dir.exists() and any(dst_dir.iterdir()):
                print("[OK] Data synchronization completed.")
            else:
                print("[WARN] Data directory appears empty after sync.")

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
        module = __import__(pkg)
        version = getattr(module, '__version__', 'unknown')
        print(f"✓ {display_name} ({version})")
        return True
    except ImportError as e:
        print(f"✗ {display_name} - Not found")
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
critical_results = [try_import(pkg, name) for pkg, name in critical_packages.items()]
critical_ok = all(critical_results)

print("\nOptional packages:")
optional_results = [try_import(pkg, name) for pkg, name in optional_packages.items()]
optional_ok = all(optional_results)


# ==========================
# 8) Final Status Report
# ==========================
print("\n" + "=" * 60)
print("Setup Summary")
print("=" * 60)

print(f"✓ Repository: {PROJECT_ROOT}")
print(f"✓ Python: {sys.version.split()[0]}")
print(f"✓ Working directory: {Path.cwd()}")

if critical_ok:
    print("✓ All critical packages installed")
else:
    print("✗ Some critical packages missing - please review errors above")
    missing = [name for (pkg, name), ok in zip(critical_packages.items(), critical_results) if not ok]
    print(f"  Missing: {', '.join(missing)}")

if optional_ok:
    print("✓ All optional packages installed")
else:
    print("⚠ Some optional packages missing")
    missing = [name for (pkg, name), ok in zip(optional_packages.items(), optional_results) if not ok]
    print(f"  Missing: {', '.join(missing)}")

print("\n" + "=" * 60)
if critical_ok:
    print("✅ SeisMambaKAN environment ready!")
    print("\nYou can now run your training/inference scripts.")
else:
    print("⚠️  Setup completed with warnings - check messages above")
    print("\nSome packages are missing. Install them with:")
    print("  pip install <package-name>")
print("=" * 60)