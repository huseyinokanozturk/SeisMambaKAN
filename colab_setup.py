#!/usr/bin/env python3
"""
SeisMambaKAN Colab Setup
========================
1. Drive'daki projeyi GitHub'dan günceller
2. Veriyi Drive'dan Colab'a kopyalar  
3. Gerekli paketleri yükler
4. /content/SeisMambaKAN'da çalışmaya hazır hale getirir
"""

import os
import sys
import subprocess
from pathlib import Path

# ============== AYARLAR ==============
GIT_REPO_URL = "https://github.com/huseyinokanozturk/SeisMambaKAN.git"
DRIVE_PROJECT = "/content/drive/MyDrive/Proje_SeisMamba/SeisMambaKAN"
COLAB_PROJECT = "/content/SeisMambaKAN"
DATA_MODE = "sample"  # "sample", "all", veya "none"
DRIVE_DATA = f"{DRIVE_PROJECT}/data/processed"
COLAB_DATA = f"{COLAB_PROJECT}/data/processed"


def run(cmd):
    """Komutu çalıştır, hata varsa devam et."""
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    return result.returncode == 0


print("=" * 50)
print("SeisMambaKAN Setup Başlıyor...")
print("=" * 50)

# 1️⃣ Drive'daki Projeyi GitHub'dan Güncelle
print("\n[1/5] Drive'daki proje güncelleniyor...")
drive_path = Path(DRIVE_PROJECT)

if not drive_path.exists():
    print(f"❌ Drive klasörü yok: {DRIVE_PROJECT}")
    print("💡 Drive'da klasörü oluşturun veya yolu değiştirin")
    sys.exit(1)

os.chdir(DRIVE_PROJECT)

if (drive_path / ".git").exists():
    print("📥 Git pull yapılıyor...")
    run("git stash")
    run("git pull")
    run("git stash pop")
else:
    print("⚠️  Git repo değil, atlanıyor")

# 2️⃣ Colab'a Projeyi Kopyala
print("\n[2/5] Proje Colab'a kopyalanıyor...")
colab_path = Path(COLAB_PROJECT)

if colab_path.exists():
    run(f"rm -rf {COLAB_PROJECT}")

run(f"cp -r {DRIVE_PROJECT} {COLAB_PROJECT}")
print(f"✅ {COLAB_PROJECT}")

# 3️⃣ Veriyi Kopyala
print("\n[3/5] Veri kopyalanıyor...")

if DATA_MODE != "none":
    src_data = Path(DRIVE_DATA) / DATA_MODE
    dst_data = Path(COLAB_DATA) / DATA_MODE
    
    if src_data.exists():
        dst_data.parent.mkdir(parents=True, exist_ok=True)
        run(f"rm -rf {dst_data}")
        run(f"cp -r {src_data} {dst_data}")
        file_count = sum(1 for _ in dst_data.rglob('*') if _.is_file())
        print(f"✅ {file_count} dosya kopyalandı ({DATA_MODE})")
    else:
        print(f"⚠️  Veri bulunamadı: {src_data}")
else:
    print("⏭️  Veri kopyalama atlandı (DATA_MODE='none')")

# 4️⃣ Paketleri Yükle
print("\n[4/5] Paketler yükleniyor...")
os.chdir(COLAB_PROJECT)

# Python path'e ekle
if COLAB_PROJECT not in sys.path:
    sys.path.insert(0, COLAB_PROJECT)
os.environ["SEISMAMBAKAN_ROOT"] = COLAB_PROJECT

# Mamba (hızlı kurulum)
try:
    import mamba_ssm
    print("✅ Mamba zaten yüklü")
except:
    print("📦 Mamba yükleniyor (wheel ile hızlı)...")
    run("pip install -q causal-conv1d>=1.4.0 --no-build-isolation")
    run("pip install -q mamba-ssm>=2.2.0 --no-build-isolation")

# Requirements
if Path("requirements.txt").exists():
    print("📦 Requirements yükleniyor...")
    run("pip install -q -r requirements.txt")

# 5️⃣ Kontrol
print("\n[5/5] Kontrol yapılıyor...")

# Import test
errors = []
for pkg in ["torch", "numpy", "mamba_ssm", "efficient_kan"]:
    try:
        __import__(pkg)
        print(f"✅ {pkg}")
    except:
        print(f"❌ {pkg}")
        errors.append(pkg)

# GPU
try:
    import torch
    if torch.cuda.is_available():
        print(f"✅ GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("⚠️  GPU yok (CPU modunda)")
except:
    pass

# Sonuç
print("\n" + "=" * 50)
if not errors:
    print("✅ HAZIR!")
    print(f"📂 Çalışma dizini: {COLAB_PROJECT}")
    print("\n💡 Notebook'ta şunu çalıştırın:")
    print("   import os, sys")
    print(f"   sys.path.insert(0, '{COLAB_PROJECT}')")
    print(f"   os.chdir('{COLAB_PROJECT}')")
    print("\n   # Sonra istediğiniz script'i çalıştırın")
    print("   !python train.py")
else:
    print("⚠️  Bazı paketler eksik:", ", ".join(errors))
    print("   pip install <paket-adı> ile yükleyin")
print("=" * 50)