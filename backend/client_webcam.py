import cv2
import requests
import time
import os
import pygame

SERVER_URL = "http://127.0.0.1:8000/recognize"
AUDIO_DIR = os.path.join(os.path.dirname(__file__), 'generated_audio')

def play_audio(track_id: str):
    if not track_id: return
    try:
        file_path = os.path.join(AUDIO_DIR, f"{track_id}.mp3")
        if os.path.exists(file_path):
            pygame.mixer.music.load(file_path)
            pygame.mixer.music.play()
            print(f"🔊 Memutar audio: {track_id}.mp3")
        else:
            print(f"⚠️ File audio tidak ditemukan: {track_id}.mp3")
    except Exception as e:
        print(f"❌ Error saat memutar audio: {e}")

def run_webcam_attendance():
    pygame.init()
    pygame.mixer.init()
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Error: Tidak bisa membuka kamera.")
        return
    print("✅ Kamera siap. Tekan 'SPASI' untuk absen, 'Q' untuk keluar.")

    while True:
        ret, frame = cap.read()
        if not ret: break
        cv2.imshow('Webcam Absensi', frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'): break
        elif key == ord(' '):
            print("\n📸 Mengambil gambar...")
            success, image_bytes = cv2.imencode('.jpg', frame)
            if not success: continue
            
            print("✈️ Mengirim gambar ke server...")
            try:
                response = requests.post(SERVER_URL, data=image_bytes.tobytes(), headers={'Content-Type': 'image/jpeg'}, timeout=15)
                if response.status_code == 200:
                    result = response.json()
                    print(f"✅ [{result.get('status', '').upper()}] Server: {result.get('message', 'N/A')}")
                    play_audio(result.get('audio_track'))
                else:
                    print(f"❌ Error dari server: {response.status_code}, {response.text}")
            except requests.exceptions.RequestException as e:
                print(f"❌ Gagal terhubung ke server: {e}")
            
            while pygame.mixer.music.get_busy():
                time.sleep(0.1)
            print("\n✅ Kamera siap kembali...")

    cap.release()
    cv2.destroyAllWindows()
    pygame.quit()

if __name__ == "__main__":
    run_webcam_attendance()