import face_recognition
import cv2
import numpy as np
from typing import Union, List, Tuple

# =========================================================
# MODUL CORE MACHINE LEARNING (DLib/Face Recognition)
# Model ini bekerja dengan embedding 128-dimensi
# =========================================================

def detect_face(frame: np.ndarray) -> Union[np.ndarray, None]:
    """
    Mendeteksi wajah dalam frame dan mengembalikan citra wajah yang dipotong.
    
    Output: np.ndarray (gambar wajah yang dipotong) atau None.
    """
    try:
        # Konversi BGR (OpenCV) ke RGB (Face Recognition)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Temukan semua lokasi wajah dalam gambar
        # Menggunakan model 'cnn' lebih akurat, tetapi 'hog' (default) lebih cepat.
        face_locations = face_recognition.face_locations(rgb_frame, model="cnn")
        
        if not face_locations:
            return None # Tidak ditemukan wajah
        
        # Ambil wajah yang pertama ditemukan (asumsi hanya ada satu orang per foto dataset)
        # face_locations memberikan (top, right, bottom, left)
        top, right, bottom, left = face_locations[0]
        
        # Potong (crop) wajah dari frame
        face_image = frame[top:bottom, left:right]
        
        # Pastikan wajah yang dipotong valid
        if face_image.size == 0:
            return None
            
        return face_image
        
    except Exception as e:
        print(f"Error saat mendeteksi wajah: {e}")
        return None

def calculate_embedding(face_image: np.ndarray) -> Union[np.ndarray, None]:
    """
    Menghitung embedding wajah (128-dimensi) dari citra wajah yang dipotong.
    
    Output: np.ndarray (embedding 1D) atau None.
    """
    try:
        # Pastikan gambar wajah valid
        if face_image.size == 0:
             return None
             
        # Konversi BGR ke RGB
        rgb_face_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
        
        # --- PERBAIKAN KRITIS UNTUK VERSI LAMA face_recognition ---
        # Kami menghapus argumen 'face_locations' karena versi library Anda tidak mendukungnya
        # face_encodings akan mendeteksi ulang wajah di gambar kecil ini.
        face_encodings = face_recognition.face_encodings(rgb_face_image)
        # --------------------------------------------------------
        
        if len(face_encodings) > 0:
            # Mengembalikan embedding pertama (array 128-dimensi)
            return face_encodings[0]
        
        return None
        
    except Exception as e:
        print(f"Error saat menghitung embedding: {e}")
        return None
