"""
EduFrame AI (AI Video Lesson Analyzer) - Merkezi Yapılandırma Dosyası
Tüm modüller için dosya yolları, model ayarları ve sabitler burada tanımlanır.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# .env dosyası varsa otomatik yükle
load_dotenv()

# --- DİZİN YAPILANDIRMASI ---
ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"

# Uygulamanın aktif ChromaDB dizini (v1).
CHROMA_DB_V1_DIR = DATA_DIR / "chroma_db"
CHROMA_DB_DIR = CHROMA_DB_V1_DIR

REFERENCE_BOOKS_DIR = DATA_DIR / "reference_books"

# Geçici medya klasörleri (Video indirme, kare yakalama, PDF çıktıları)
TEMP_DIR = ROOT / "temp_medya"
VIDEOS_DIR = TEMP_DIR / "videos"
FRAMES_DIR = TEMP_DIR / "frames"
OUTPUTS_DIR = TEMP_DIR / "outputs"

# Geriye dönük uyumluluk için eski değişken isimleri
DB_DIR = CHROMA_DB_DIR
COLLECTION = "eduframe_meb_768"
RAG_MAX_MESAFE = 1.1

# --- EĞİTİM SEVİYELERİ ---
EGITIM_SEVIYELERI = ["İlkokul", "Ortaokul", "Lise", "Üniversite"]
DEFAULT_SEVIYE = "İlkokul"

# --- GOOGLE GEMINI API AYARLARI ---
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3-flash-preview")
GEMINI_TIMEOUT = int(os.getenv("GEMINI_TIMEOUT", "20"))
GEMINI_MAX_RETRIES = int(os.getenv("GEMINI_MAX_RETRIES", "2"))
GEMINI_FALLBACK_MODELS = [m.strip() for m in os.getenv("GEMINI_FALLBACK_MODELS", "gemini-3-flash-preview,gemini-3.8-flash,gemini-flash-latest").split(",") if m.strip()]

# --- OLLAMA VE LOKAL MODEL AYARLARI (Birincil Model) ---
OLLAMA_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen3.5:4b")
EMBED_MODEL = os.getenv("EMBED_MODEL", "embeddinggemma")
CHAT_MODEL = OLLAMA_MODEL

# CPU üzerinde 4B+ model çıkarımı için gerçekçi zaman aşımı süreleri (saniye)
OLLAMA_TIMEOUT = int(os.getenv("OLLAMA_TIMEOUT", "180"))
OLLAMA_ANALYSIS_TIMEOUT = int(os.getenv("OLLAMA_ANALYSIS_TIMEOUT", "180"))
OLLAMA_QUESTION_TIMEOUT = int(os.getenv("OLLAMA_QUESTION_TIMEOUT", "180"))
OLLAMA_SECTION_TIMEOUT = int(os.getenv("OLLAMA_SECTION_TIMEOUT", "45"))

# Görev bazlı token üretim sınırları (num_predict)
OLLAMA_ANALYSIS_NUM_PREDICT = int(os.getenv("OLLAMA_ANALYSIS_NUM_PREDICT", "3500"))
OLLAMA_QUESTION_NUM_PREDICT = int(os.getenv("OLLAMA_QUESTION_NUM_PREDICT", "3500"))
OLLAMA_SECTION_NUM_PREDICT = int(os.getenv("OLLAMA_SECTION_NUM_PREDICT", "800"))

# --- VIDEO İNDİRME AYARLARI ---
DEFAULT_VIDEO_RESOLUTION = "720"  # 360 veya 720
ALLOWED_VIDEO_RESOLUTIONS = ["360", "720"]

# Klasörlerin varlığından emin ol
for path in [DATA_DIR, CHROMA_DB_DIR, REFERENCE_BOOKS_DIR, TEMP_DIR, VIDEOS_DIR, FRAMES_DIR, OUTPUTS_DIR]:
    path.mkdir(parents=True, exist_ok=True)
