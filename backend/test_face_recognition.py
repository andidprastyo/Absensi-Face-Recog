#!/usr/bin/env python3
"""
Script untuk testing face recognition dengan mengirim gambar dari dataset.
Harus dijalankan dari direktori root proyek.
Contoh: python backend/test_face_recognition.py
"""

import requests
import time
from pathlib import Path
import random

# Tentukan direktori root proyek dan path penting lainnya
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = PROJECT_ROOT / "data" / "dataset"
# PERBAIKAN: Mengubah port dari 5000 menjadi 8000 agar sesuai dengan backend/run_system.py
BACKEND_URL = "http://localhost:8000/recognize" 

def check_backend():
    """Memeriksa apakah server backend sudah berjalan."""
    try:
        # PERBAIKAN: Mengubah port dari 5000 menjadi 8000
        response = requests.get("http://localhost:8000/api/status", timeout=3)
        if response.status_code == 200:
            print("✅ Backend berjalan.")
            return True
        else:
            print(f"⚠️ Backend merespons dengan status {response.status_code}.")
            return False
    except requests.ConnectionError:
        # PERBAIKAN: Memperbarui pesan panduan untuk mencerminkan port yang benar (8000)
        print("❌ Backend tidak berjalan atau tidak dapat dijangkau di http://localhost:8000")
        print("   Silakan jalankan backend terlebih dahulu dengan perintah:")
        print(f"   python \"{PROJECT_ROOT / 'backend' / 'run_system.py'}\"")
        return False

def find_test_images():
    """Mencari gambar secara acak dari dataset untuk dijadikan bahan tes."""
    print(f"📁 Mencari gambar di dataset: {DATASET_DIR}")
    if not DATASET_DIR.is_dir():
        print("❌ Direktori dataset tidak ditemukan.")
        return []
    
    all_images = list(DATASET_DIR.glob("**/*.jpg")) + list(DATASET_DIR.glob("**/*.png"))
    if not all_images:
        print("❌ Tidak ada gambar yang ditemukan di dalam dataset.")
        return []
        
    # Pilih 5 gambar acak atau semua jika kurang dari 5
    num_to_test = min(5, len(all_images))
    test_images = random.sample(all_images, num_to_test)
    print(f"📸 Ditemukan {len(test_images)} gambar acak untuk testing.")
    return test_images

def main():
    """Fungsi utama untuk menjalankan testing."""
    print("🧪 Testing Face Recognition System")
    print("=" * 50)

    if not check_backend():
        return

    test_images = find_test_images()
    if not test_images:
        return
    
    # Test setiap gambar
    for i, img_path in enumerate(test_images, 1):
        person_name = img_path.parent.name
        division_name = img_path.parent.parent.name
        
        print(f"\n🧪 Tes {i}/{len(test_images)}: {person_name} ({division_name})")
        print(f"   File: {img_path}")
        
        try:
            with open(img_path, 'rb') as f:
                f.seek(0) # Kembali ke awal file setelah membaca
                image_data = f.read()
                response = requests.post(
                BACKEND_URL, 
                    data=image_data,  # Mengirimkan data biner secara langsung
                    headers={'Content-Type': 'image/jpeg'},
                    timeout=10
                )

                if response.status_code == 200:
                    result = response.json()
                    if result.get('status') == 'success':
                        recognized_label = result.get('label')
                        print(f"   ✅ Berhasil dikenali sebagai: {recognized_label}")
                        if recognized_label.lower() == person_name.lower():
                            print("   🎯 Hasil SESUAI dengan nama folder.")
                        else:
                            print(f"   ⚠️  Hasil TIDAK SESUAI dengan nama folder (expected: {person_name}).")
                    else:
                        print(f"   ❌ Gagal dikenali: {result.get('message', 'Pesan error tidak diketahui')}")
                else:
                    print(f"   ❌ Gagal: Server merespons dengan kode status {response.status_code}")
                    
        except requests.exceptions.RequestException as e:
            print(f"   ❌ Error saat mengirim request: {e}")
        except Exception as e:
            print(f"   ❌ Terjadi error tak terduga: {e}")
        
        # Beri jeda antar tes
        time.sleep(2)
    
    print("\n🎉 Testing selesai!")
    print("📊 Cek hasil detail absensi dan capture wajah di dashboard:")
    # PERBAIKAN: Memperbarui pesan panduan untuk mencerminkan port yang benar (8000)
    print("   http://localhost:8000/data.html")

if __name__ == "__main__":
    main()
