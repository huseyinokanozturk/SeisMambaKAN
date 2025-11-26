#!/usr/bin/env python3
"""
SeisMambaKAN Colab Setup
========================
1. GitHub'dan projeyi Colab'a klonlar/günceller
2. Drive'dan veriyi Colab'a kopyalar  
3. Gerekli paketleri yükler
4. /content/SeisMambaKAN'da çalışmaya hazır hale getirir
"""

import os
import sys
import subprocess
from pathlib import Path

# ============== AYARLAR ==============
GIT_REPO_URL = "https://github.com/huseyinokanozturk/SeisMambaKAN.git"
COLAB_PROJECT = "/content/SeisMambaKAN"
DATA_MODE = "sample"  # "sample", "all", veya "none"
DRIVE_DATA = "/content/drive/MyDrive/Proje_SeisMamba/SeisMambaKAN/data/processed"
COLAB_DATA = f"{COLAB_PROJECT}/data/processed"


def run(cmd):
    """Komutu çalıştır, hata varsa devam et."""
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    return result.returncode == 0


print("=" * 50)
print("SeisMambaKAN Setup Başlıyor...")
print("=" * 50)

# 0️⃣ Drive'daki Projeyi GitHub'dan Güncelle (İsteğe Bağlı)
print("\n[0/5] Drive'daki proje güncelleniyor...")
drive_project = "/content/drive/MyDrive/Proje_SeisMamba/SeisMambaKAN"
drive_path = Path(drive_project)

if Path("/content/drive").exists():
    if drive_path.exists():
        if (drive_path / ".git").exists():
            print("📥 Drive → Git pull yapılıyor...")
            os.chdir(drive_project)
            run("git stash")
            run("git pull")
            run("git stash pop")
            print(f"✅ Drive güncellendi: {drive_project}")
        else:
            print("⚠️  Drive klasörü git repo değil, atlanıyor")
    else:
        print(f"⚠️  Drive klasörü yok: {drive_project}")
else:
    print("⚠️  Drive mount edilmemiş, Drive güncellemesi atlanıyor")

# 1️⃣ GitHub'dan Colab'a Projeyi Kopyala/Güncelle
print("\n[1/5] Proje GitHub'dan Colab'a çekiliyor...")
colab_path = Path(COLAB_PROJECT)

if colab_path.exists() and (colab_path / ".git").exists():
    print("📥 Git pull yapılıyor...")
    os.chdir(COLAB_PROJECT)
    run("git stash")
    run("git pull")
    run("git stash pop")
    print(f"✅ Güncellendi: {COLAB_PROJECT}")
else:
    print("📥 Git clone yapılıyor...")
    os.chdir("/content")
    if colab_path.exists():
        run(f"rm -rf {COLAB_PROJECT}")
    run(f"git clone {GIT_REPO_URL} {COLAB_PROJECT}")
    print(f"✅ Klonlandı: {COLAB_PROJECT}")

# 2️⃣ Veriyi Drive'dan Colab'a Kopyala
print("\n[2/5] Veri Drive'dan kopyalanıyor...")

if DATA_MODE != "none":
    src_data = Path(DRIVE_DATA) / DATA_MODE
    dst_data = Path(COLAB_DATA) / DATA_MODE
    
    if not Path("/content/drive").exists():
        print("⚠️  Drive mount edilmemiş, veri kopyalanamıyor")
    elif src_data.exists():
        dst_data.parent.mkdir(parents=True, exist_ok=True)
        run(f"rm -rf {dst_data}")
        
        # rsync varsa kullan (daha hızlı)
        if run("which rsync"):
            run(f"rsync -a {src_data}/ {dst_data}/")
        else:
            run(f"cp -r {src_data} {dst_data}")
        
        file_count = sum(1 for _ in dst_data.rglob('*') if _.is_file())
        print(f"✅ {file_count} dosya kopyalandı ({DATA_MODE})")
    else:
        print(f"⚠️  Veri bulunamadı: {src_data}")
else:
    print("⏭️  Veri kopyalama atlandı (DATA_MODE='none')")

# 3️⃣ Python Ortamını Ayarla
print("\n[3/5] Python ortamı ayarlanıyor...")
os.chdir(COLAB_PROJECT)

if COLAB_PROJECT not in sys.path:
    sys.path.insert(0, COLAB_PROJECT)
os.environ["SEISMAMBAKAN_ROOT"] = COLAB_PROJECT
print(f"✅ Çalışma dizini: {COLAB_PROJECT}")

# 4️⃣ Paketleri Yükle
print("\n[4/5] Paketler yükleniyor...")

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