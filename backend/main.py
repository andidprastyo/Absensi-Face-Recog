import os
import time
import json
import datetime
import uvicorn
import webbrowser
import sqlite3
from threading import Timer
from pathlib import Path
import cv2
import numpy as np
import joblib
import pytz
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from contextlib import asynccontextmanager
from . import face_recognition_core

# --- Konfigurasi ---
class AppConfig:
    BASE_DIR = Path(__file__).resolve().parent
    DB_PATH = BASE_DIR / "attendance.db"
    MODEL_DIR = BASE_DIR / "model"
    STATIC_DIR = BASE_DIR / "static"
    TEMPLATES_DIR = BASE_DIR / "templates"
    KNN_MODEL_PATH = MODEL_DIR / "knn_model.pkl"
    LABEL_ENCODER_PATH = MODEL_DIR / "label_encoder.pkl"
    AUDIO_TRACKING_FILE = BASE_DIR / "audio_tracking.json"
    IMAGE_STORAGE_DIR = BASE_DIR / "captured_images"
    TIMEZONE = 'Asia/Jakarta'
    WIB = pytz.timezone(TIMEZONE)

AppConfig.IMAGE_STORAGE_DIR.mkdir(exist_ok=True)

# --- Variabel Global ---
knn_model, label_encoder, audio_tracking = None, None, {}
INTERN_CACHE, absen_tercatat = {}, set()

# --- Fungsi Database & Startup ---
def db_connect():
    conn = sqlite3.connect(AppConfig.DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn

def load_all_data():
    global knn_model, label_encoder, audio_tracking, INTERN_CACHE, absen_tercatat
    try:
        knn_model = joblib.load(AppConfig.KNN_MODEL_PATH)
        label_encoder = joblib.load(AppConfig.LABEL_ENCODER_PATH)
        print("✅ Model ML dimuat.")
        
        with open(AppConfig.AUDIO_TRACKING_FILE, 'r') as f: audio_tracking = json.load(f)
        print(f"✅ Pemetaan audio dimuat: {len(audio_tracking)} rekaman.")

        conn = db_connect()
        cursor = conn.cursor()
        cursor.execute("SELECT id, name, universitas, kategori FROM interns")
        for row in cursor.fetchall(): INTERN_CACHE[row['name']] = dict(row)
        print(f"✅ Cache Intern dimuat: {len(INTERN_CACHE)} data.")
        
        cursor.execute("SELECT intern_name FROM attendance_logs WHERE date(absent_at) = date('now', 'localtime')")
        absen_tercatat = {row['intern_name'] for row in cursor.fetchall()}
        conn.close()
        print(f"✅ Cache absensi hari ini dimuat: {len(absen_tercatat)} orang.")
        
        AppConfig.IMAGE_STORAGE_DIR.mkdir(exist_ok=True) # Buat folder jika belum ada
    except Exception as e:
        print(f"🔥 KRITIS: Gagal memuat data saat startup: {e}")

@asynccontextmanager
async def lifespan(app: FastAPI):
    load_all_data()
    yield

app = FastAPI(title="Face Attendance API (Local DB)", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
app.mount("/static", StaticFiles(directory=AppConfig.STATIC_DIR), name="static")
app.mount("/images", StaticFiles(directory=AppConfig.IMAGE_STORAGE_DIR), name="images") # <-- PERBAIKAN PENTING
templates = Jinja2Templates(directory=AppConfig.TEMPLATES_DIR)

# --- Endpoint Utama (Absensi & Frontend) ---
@app.get("/", response_class=HTMLResponse)
async def serve_main_dashboard(request: Request):
    return templates.TemplateResponse("main.html", {"request": request})

@app.get("/{page_name}.html", response_class=HTMLResponse)
async def serve_pages(request: Request, page_name: str):
    if not (AppConfig.TEMPLATES_DIR / f"{page_name}.html").exists():
        raise HTTPException(status_code=404, detail="Halaman tidak ditemukan")
    return templates.TemplateResponse(f"{page_name}.html", {"request": request})

@app.post("/recognize")
async def recognize_face(request: Request):
    try:
        frame = cv2.imdecode(np.frombuffer(await request.body(), np.uint8), cv2.IMREAD_COLOR)
        face_image = face_recognition_core.detect_face(frame)
        if face_image is None: return JSONResponse(status_code=400, content={"audio_track": "0002"})
        emb = face_recognition_core.calculate_embedding(face_image)
        if emb is None: return JSONResponse(status_code=400, content={"audio_track": "0002"})

        label = label_encoder.inverse_transform(knn_model.predict(emb.reshape(1, -1)))[0]
        if label in absen_tercatat: return JSONResponse(status_code=200, content={"status": "fail", "message": f"{label} sudah absen hari ini.", "audio_track": "0001"})
        intern_data = INTERN_CACHE.get(label)
        if not intern_data: return JSONResponse(status_code=404, content={"status": "fail", "message": f"Data intern '{label}' tidak ditemukan.", "audio_track": "0003"})

        timestamp = datetime.datetime.now(AppConfig.WIB).strftime("%Y%m%d_%H%M%S")
        image_filename = f"{label}_{timestamp}.jpg"
        image_path = AppConfig.IMAGE_STORAGE_DIR / image_filename
        cv2.imwrite(str(image_path), frame)
        
        image_url_for_frontend = f"/images/{image_filename}"
        
        conn = db_connect()
        cursor = conn.cursor()
        cursor.execute("INSERT INTO attendance_logs (intern_id, intern_name, universitas, kategori, image_url, absent_at) VALUES (?, ?, ?, ?, ?, ?)",
            (intern_data['id'], label, intern_data['universitas'], intern_data['kategori'], image_url_for_frontend, datetime.datetime.now(AppConfig.WIB).isoformat()))
        conn.commit()
        conn.close()
        absen_tercatat.add(label)
        
        return JSONResponse(status_code=200, content={"status": "success", "message": f"Absensi {label} berhasil.", "audio_track": f"{audio_tracking.get(label):04d}"})
    except Exception as e:
        print(f"❌ Error di /recognize: {e}")
        return JSONResponse(status_code=500, content={"message": "Internal Server Error"})

# --- 7 ENDPOINT API BARU UNTUK FRONTEND ---
@app.get("/api/system-start-date")
async def get_system_start_date():
    conn = db_connect()
    cursor = conn.cursor()
    cursor.execute("SELECT MIN(absent_at) as start_date FROM attendance_logs")
    row = cursor.fetchone()
    conn.close()
    start_date = row['start_date'] if row and row['start_date'] else datetime.datetime.now().isoformat()
    return {"system_start_date": start_date, "current_date": datetime.datetime.now().isoformat()}

@app.get("/api/today-active-interns")
async def get_today_active_interns():
    conn = db_connect()
    cursor = conn.cursor()
    cursor.execute("SELECT intern_name, universitas, kategori, absent_at, image_url, intern_id FROM attendance_logs WHERE date(absent_at) = date('now', 'localtime') ORDER BY absent_at DESC")
    active_interns = [
        {
            "name": row['intern_name'],
            "jobdesk": f"{row['kategori']} - {row['universitas']}", # <-- PERBAIKAN KONSISTENSI
            "recognition_time": row['absent_at'],
            "capture_image": row['image_url'],
            "intern_id": row['intern_id']
        } for row in cursor.fetchall()
    ]
    conn.close()
    return {"total_active": len(active_interns), "active_interns": active_interns}

@app.get("/api/attendance-dates")
async def get_attendance_dates():
    conn = db_connect()
    cursor = conn.cursor()
    cursor.execute("SELECT DISTINCT date(absent_at) as attendance_date FROM attendance_logs ORDER BY attendance_date DESC")
    dates = [row['attendance_date'] for row in cursor.fetchall()]
    conn.close()
    return {"total_dates": len(dates), "dates": dates}

@app.get("/api/attendance-dates-with-range")
async def get_attendance_dates_with_range():
    conn = db_connect()
    cursor = conn.cursor()
    cursor.execute("SELECT MIN(date(absent_at)) as start_date FROM attendance_logs")
    start_row = cursor.fetchone()
    if not start_row or not start_row['start_date']: return {"date_range": [], "total_dates": 0}
    start_date = datetime.datetime.strptime(start_row['start_date'], '%Y-%m-%d').date()
    end_date = datetime.date.today()
    delta = end_date - start_date
    cursor.execute("SELECT DISTINCT date(absent_at) as attendance_date FROM attendance_logs")
    attended_dates = {row['attendance_date'] for row in cursor.fetchall()}
    conn.close()
    date_range = [{"date": (start_date + datetime.timedelta(days=i)).isoformat(), "has_attendance": (start_date + datetime.timedelta(days=i)).isoformat() in attended_dates} for i in range(delta.days + 1)]
    return {"date_range": date_range, "total_dates": len(attended_dates)}

@app.get("/api/attendance-by-date/{date_str}")
async def get_attendance_by_date(date_str: str):
    conn = db_connect()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM attendance_logs WHERE date(absent_at) = ? ORDER BY absent_at", (date_str,))
    attendees = [
        {
            "name": row['intern_name'],
            "jobdesk": f"{row['kategori']} - {row['universitas']}", # <-- PERBAIKAN KONSISTENSI
            "photo": row['image_url'], # <-- Frontend butuh 'photo'
            "recognition_time": row['absent_at'],
            "status": "Hadir"
        } for row in cursor.fetchall()
    ]
    conn.close()
    return {"date": date_str, "attendees": attendees, "total_attendees": len(attendees)}

@app.get("/api/attendance-summary")
async def get_attendance_summary():
    conn = db_connect()
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(id) as total FROM attendance_logs WHERE date(absent_at) = date('now', 'localtime')")
    total = cursor.fetchone()['total']
    conn.close()
    return {"total_attendees": total}

@app.get("/api/monthly-attendance/{year}/{month}")
async def get_monthly_attendance(year: int, month: int):
    month_str = f"{year:04d}-{month:02d}"
    conn = db_connect()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM attendance_logs WHERE strftime('%Y-%m', absent_at) = ?", (month_str,))
    
    daily_stats, total_attendance = {}, 0
    for row in cursor.fetchall():
        total_attendance += 1
        date_str = row['absent_at'][:10]
        if date_str not in daily_stats: daily_stats[date_str] = []
        daily_stats[date_str].append({"name": row['intern_name'], "jobdesk": f"{row['kategori']} - {row['universitas']}"}) # <-- PERBAIKAN KONSISTENSI
    conn.close()
    
    unique_days = len(daily_stats)
    return {
        "total_attendance": total_attendance, "unique_days": unique_days,
        "avg_daily_attendance": round(total_attendance / unique_days, 2) if unique_days > 0 else 0,
        "daily_stats": [{"date": k, "attendees": v} for k, v in daily_stats.items()]
    }

# --- Main Runner ---
def open_browser():
    time.sleep(2)
    webbrowser.open_new_tab("http://172.26.96.1:8000")

if __name__ == "__main__":
    Timer(1, open_browser).start()
    uvicorn.run("backend.main:app", host="0.0.0.0", port=8000, reload=True)