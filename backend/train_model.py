#!/usr/bin/env python3
import os
import sys
import numpy as np
import joblib
import json
import csv
import sqlite3
from pathlib import Path
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from gtts import gTTS

# --- Konfigurasi ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from backend.utils import extract_face_features

DATASET_DIR = PROJECT_ROOT / "data" / "dataset"
CACHE_DIR = PROJECT_ROOT / "data" / "cache"
MODEL_DIR = PROJECT_ROOT / "backend" / "model"
AUDIO_FILES_DIR = PROJECT_ROOT / "backend" / "generated_audio"
DB_PATH = PROJECT_ROOT / "backend" / "attendance.db"
INTERNS_CSV_PATH = PROJECT_ROOT / "interns.csv"
AUDIO_TRACKING_FILE = PROJECT_ROOT / "backend" / "audio_tracking.json"

# --- FUNGSI DATABASE ---
def create_or_update_local_db():
    print("\n💾 Memeriksa dan menyinkronkan database lokal...")
    if not INTERNS_CSV_PATH.exists():
        print(f"❌ Error: File 'interns.csv' tidak ditemukan.")
        return False
    try:
        with open(INTERNS_CSV_PATH, mode='r', encoding='utf-8') as f:
            local_data = {row['name']: row for row in csv.DictReader(f)}
        
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS interns (
                id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT NOT NULL UNIQUE, universitas TEXT, kategori TEXT
            )
        ''')
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS attendance_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT, intern_id INTEGER, intern_name TEXT, universitas TEXT,
                kategori TEXT, image_url TEXT, absent_at TEXT, FOREIGN KEY (intern_id) REFERENCES interns (id)
            )
        ''')
        
        cursor.execute("SELECT name FROM interns")
        db_names = {row[0] for row in cursor.fetchall()}
        new_names = set(local_data.keys()) - db_names
        
        if not new_names:
            print("   ✅ Database 'interns' sudah sinkron.")
        else:
            print(f"   ✨ Menambahkan {len(new_names)} data baru: {new_names}")
            for name in new_names:
                data = local_data[name]
                cursor.execute("INSERT INTO interns (name, universitas, kategori) VALUES (?, ?, ?)",
                               (data['name'], data['universitas'], data['kategori']))
        conn.commit()
        conn.close()
        print("   ✅ Struktur database (interns & attendance_logs) SIAP.")
        return True
    except Exception as e:
        print(f"   ❌ Gagal menyinkronkan database: {e}")
        return False

# --- FUNGSI AUDIO ---
def load_audio_tracking():
    if AUDIO_TRACKING_FILE.exists():
        with open(AUDIO_TRACKING_FILE, 'r') as f: return json.load(f)
    return {}

def save_audio_tracking(audio_tracking):
    with open(AUDIO_TRACKING_FILE, 'w') as f: json.dump(audio_tracking, f, indent=2)
    print(f"💾 Audio tracking disimpan: {len(audio_tracking)} rekaman.")

def generate_generic_audio_files():
    print("\n🎵 Memeriksa file audio umum...")
    generic_messages = {
        "0001": "Anda sudah melakukan absensi hari ini.",
        "0002": "Wajah tidak terdeteksi, silakan coba lagi.",
        "0003": "Data Anda tidak ditemukan di sistem."
    }
    try:
        AUDIO_FILES_DIR.mkdir(exist_ok=True)
        for track_id, message in generic_messages.items():
            file_path = AUDIO_FILES_DIR / f"{track_id}.mp3"
            if not file_path.exists():
                print(f"   Membuat {track_id}.mp3...")
                gTTS(text=message, lang='id', slow=False).save(file_path)
    except Exception as e:
        print(f"   ❌ Gagal membuat audio umum: {e}")

def generate_audio_files_for_labels(labels):
    print("\n🎵 Memeriksa file audio per nama...")
    audio_tracking = load_audio_tracking()
    track_number = max(audio_tracking.values()) + 1 if audio_tracking else 4
    new_audio_generated = False
    for label in sorted(list(labels)):
        if label not in audio_tracking:
            try:
                audio_path = AUDIO_FILES_DIR / f"{track_number:04d}.mp3"
                gTTS(text=f"Absensi Berhasil, Selamat datang {label}", lang='id', slow=False).save(audio_path)
                print(f"   ✅ Berhasil membuat audio untuk {label}")
                audio_tracking[label] = track_number
                track_number += 1
                new_audio_generated = True
            except Exception as e:
                print(f"   ❌ Gagal membuat audio untuk {label}: {e}")
    if new_audio_generated: save_audio_tracking(audio_tracking)
    else: print("   ✅ Semua file audio nama sudah lengkap.")

# --- FUNGSI TRAINING ---
def train_model_full():
    print("\n🧠 Memulai proses training penuh...")
    embeddings, labels = [], []
    for person_dir in sorted(DATASET_DIR.iterdir()):
        if not person_dir.is_dir(): continue
        person_name = person_dir.name
        img_count = 0
        for img_path in person_dir.glob("*.jpg"):
            emb_list = extract_face_features(str(img_path))
            if emb_list:
                embeddings.extend(emb_list)
                labels.extend([person_name] * len(emb_list))
                img_count += len(emb_list)
        print(f"   👤 {person_name}: {img_count} wajah diekstrak.")
    if not embeddings: print("\n❌ Tidak ada wajah yang berhasil diekstrak!"); return False, []
    
    embeddings, labels = np.array(embeddings), np.array(labels)
    label_encoder = LabelEncoder().fit(labels)
    knn = KNeighborsClassifier(n_neighbors=3).fit(embeddings, label_encoder.transform(labels))
    
    MODEL_DIR.mkdir(exist_ok=True)
    joblib.dump(knn, MODEL_DIR / "knn_model.pkl")
    joblib.dump(label_encoder, MODEL_DIR / "label_encoder.pkl")
    print(f"\n✅ Training selesai! Model disimpan.")
    return True, set(labels)

# --- FUNGSI UTAMA ---
def main():
    print("="*50); print("🤖 SCRIPT TRAINING MODEL, AUDIO, & DATABASE"); print("="*50)
    if not DATASET_DIR.is_dir(): print(f"❌ Dataset tidak ditemukan."); return
    try:
        success, unique_labels = train_model_full()
        if success:
            # PANGGIL SEMUA FUNGSI SETELAH TRAINING BERHASIL
            generate_generic_audio_files()
            generate_audio_files_for_labels(unique_labels)
            if create_or_update_local_db():
                print("\n🎉 Semua proses (Training, Audio, & DB) selesai!")
            else:
                print("\n⚠️ Training & Audio berhasil, tapi sinkronisasi DB gagal.")
        else:
            print("\n❌ Proses training gagal.")
    except Exception as e:
        print(f"\n❌ Terjadi error: {e}")

if __name__ == "__main__":
    main()