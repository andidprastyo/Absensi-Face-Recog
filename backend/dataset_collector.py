import cv2
import os
import shutil
from pathlib import Path

# --- KONFIGURASI DASAR ---
DATASET_DIR = Path("data/dataset")  # Lokasi penyimpanan final
TEMP_DIR = Path("temp_dataset")     # Folder sementara sebelum diberi nama
NUM_FOTOS = 15

# Pastikan folder utama ada
os.makedirs(DATASET_DIR, exist_ok=True)
os.makedirs(TEMP_DIR, exist_ok=True)

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("❌ Gagal mengakses kamera (VideoCapture(0)).")
    exit()

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

print("=========================================")
print(f"📸 Pengambilan {NUM_FOTOS} foto dimulai...")
print("   TEKAN 'SPACE' untuk ambil foto.")
print("   TEKAN 'Q' untuk keluar.")
print("=========================================")

foto_counter = 1

while foto_counter <= NUM_FOTOS:
    ret, frame = cap.read()
    if not ret:
        print("⚠️ Gagal membaca frame dari kamera.")
        break

    display_frame = frame.copy()
    cv2.putText(display_frame, f"Foto ke: {foto_counter}/{NUM_FOTOS}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(display_frame, "Tekan SPASI untuk Ambil Foto!", (10, 460),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    cv2.imshow("Collector - Tekan SPACE untuk ambil foto, Q untuk keluar", display_frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord(' '):
        filename = TEMP_DIR / f"{foto_counter}.jpg"
        cv2.imwrite(str(filename), frame)
        print(f"  [OK] Foto tersimpan sementara: {filename}")
        foto_counter += 1

    elif key == ord('q'):
        print("❌ Proses dibatalkan oleh pengguna.")
        cap.release()
        cv2.destroyAllWindows()
        exit()

cap.release()
cv2.destroyAllWindows()

print("\n✅ Pengambilan foto selesai.")

# --- Input nama karyawan setelah selesai ---
while True:
    nama = input("\n📝 Masukkan nama karyawan untuk disimpan: ").strip()
    if nama:
        break
    print("⚠️ Nama tidak boleh kosong!")

# Folder tujuan
save_path = DATASET_DIR / nama
os.makedirs(save_path, exist_ok=True)

# Pindahkan foto dari TEMP_DIR ke folder karyawan
for foto in TEMP_DIR.glob("*.jpg"):
    target = save_path / foto.name
    shutil.move(str(foto), str(target))

print(f"\n✅ Semua foto dipindahkan ke: {save_path}")
print("📂 Cek folder data/dataset untuk hasilnya.")

# Kosongkan TEMP_DIR (tapi tidak dihapus)
for f in TEMP_DIR.glob("*"):
    try:
        f.unlink()
    except Exception:
        pass
