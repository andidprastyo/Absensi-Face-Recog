import cv2
from backend import face_recognition_core # Perbaikan: Impor absolut
from typing import Union, List
import numpy as np

# =========================================================
# FILE JEMBATAN UNTUK TRAIN_MODEL.PY
# File ini diperlukan karena train_model.py mengimpornya.
# =========================================================

def extract_face_features(image_path: str) -> Union[List[np.ndarray], None]:
    """
    Fungsi ini dipanggil oleh train_model.py untuk mengekstrak fitur dari gambar dataset.
    
    Output: List of embeddings (atau None jika gagal). Train_model.py mengharapkan
    list karena satu gambar bisa memiliki lebih dari satu wajah (walaupun jarang).
    """
    try:
        # 1. Baca gambar
        frame = cv2.imread(image_path)
        if frame is None:
            # Tidak bisa membaca file gambar
            return None
        
        # 2. Deteksi Wajah dan Ekstraksi Embedding
        # Panggil logika inti dari face_recognition_core
        # Kita menggunakan logic yang kita set di core: detect_face mengembalikan face_image
        face_image = face_recognition_core.detect_face(frame)
        
        if face_image is None:
            return None
        
        # calculate_embedding mengembalikan numpy array (128,)
        embedding = face_recognition_core.calculate_embedding(face_image)
        
        if embedding is not None:
            # Mengembalikan embedding dalam format list of numpy array
            return [embedding] 
        
        return None

    except Exception as e:
        print(f"Error saat mengekstrak fitur untuk {image_path}: {e}")
        return None