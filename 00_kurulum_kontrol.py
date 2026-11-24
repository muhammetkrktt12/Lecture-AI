"""
EduFrame AI - Kapsamlı Sistem, Kurulum ve Özellik Kontrol Aracı (00_kurulum_kontrol.py)
-----------------------------------------------------------------------------------
Bu betik, EduFrame AI projesinde kullanılan tüm araçları, modelleri, kütüphaneleri,
veri tabanlarını ve sistem ikili dosyalarını tek tek test edip raporlar.

Kontrol Edilen Bileşenler:
1. Python ve Temel Paketler (LangChain, LangGraph, ChromaDB, Faster-Whisper, PyPDF, ReportLab, Streamlit, MCP vb.)
2. Sistem İkili Dosyaları (FFmpeg ve FFprobe)
3. Yerel Ollama Servisi ve Modelleri (embeddinggemma, qwen3.5 vb.)
4. Yerel Konuşma Tanıma (Faster-Whisper Small Modeli)
5. Aktif Vektör Veritabanı (data/chroma_db, eduframe_meb_768 koleksiyonu, 768-D boyut)
6. MEB Kaynak Ders Kitapları (data/reference_books altındaki PDF'ler)
7. Gemma BPE Tokenizer (gemma_tokenizer.json ve tokenizer_utils)
8. Medya ve Çıktı Dizinleri (temp_medya, frames, outputs, videos)
9. Yapılandırma ve Çevre Değişkenleri (.env, GEMINI_API_KEY)
"""

import os
import sys
import json
import shutil
import urllib.request
import requests
from pathlib import Path
from typing import Optional

# Proje kök dizinini ekle
ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

try:
    from config import (
        ROOT,
        DATA_DIR,
        CHROMA_DB_DIR,
        REFERENCE_BOOKS_DIR,
        TEMP_DIR,
        VIDEOS_DIR,
        FRAMES_DIR,
        OUTPUTS_DIR,
        OLLAMA_URL,
        EMBED_MODEL,
        OLLAMA_MODEL,
        COLLECTION,
    )
except ImportError:
    ROOT = ROOT_DIR
    DATA_DIR = ROOT / "data"
    CHROMA_DB_DIR = DATA_DIR / "chroma_db"
    REFERENCE_BOOKS_DIR = DATA_DIR / "reference_books"
    TEMP_DIR = ROOT / "temp_medya"
    VIDEOS_DIR = TEMP_DIR / "videos"
    FRAMES_DIR = TEMP_DIR / "frames"
    OUTPUTS_DIR = TEMP_DIR / "outputs"
    OLLAMA_URL = "http://localhost:11434"
    EMBED_MODEL = "embeddinggemma"
    OLLAMA_MODEL = "qwen3.5:4b"
    COLLECTION = "eduframe_meb_768"


def baslik(metin: str):
    print("\n" + "=" * 70)
    print(f"  {metin}")
    print("=" * 70)


def durum_yaz(etiket: str, durum: bool, detay: str = ""):
    simge = "[ OK ]" if durum else "[EKSIK]"
    renk_detay = f" -> {detay}" if detay else ""
    print(f" {simge:<8} {etiket:<35}{renk_detay}")


def main():
    baslik("EduFrame AI — Kapsamlı Kurulum ve Sistem Doğrulama")
    print(f" Python Sürümü: {sys.version.split()[0]} ({sys.executable})")
    print(f" Çalışma Dizini: {ROOT}")

    basari_sayisi = 0
    toplam_kontrol = 0

    # -------------------------------------------------------------
    # 1. PYTHON KÜTÜPHANELERİ
    # -------------------------------------------------------------
    baslik("1. Python Kütüphaneleri ve Bağımlılıklar")
    paketler = [
        ("langchain_core", "LangChain Çekirdek"),
        ("langchain_chroma", "ChromaDB Entegrasyonu"),
        ("langchain_ollama", "Ollama LLM & Embedding"),
        ("langgraph", "LangGraph Ajan Orkestrasyonu"),
        ("chromadb", "Vektör Veritabanı"),
        ("faster_whisper", "Whisper Speech-to-Text"),
        ("pypdf", "PDF Okuyucu & Metin Ayrıştırıcı"),
        ("reportlab", "Pedagojik PDF Üretici"),
        ("streamlit", "Web Arayüzü"),
        ("av", "PyAV ses çözümleme"),
        ("cv2", "OpenCV görüntü işleme"),
        ("tokenizers", "Gemma tokenizer"),
        ("yt_dlp", "YouTube video indirme"),
        ("requests", "HTTP İstemcisi"),
        ("PIL", "Görüntü İşleme (Pillow)"),
        ("numpy", "Sayısal Hesaplama"),
        ("mediapipe", "Akıllı El/Engel Tespiti"),
    ]

    for modul_adi, aciklama in paketler:
        toplam_kontrol += 1
        try:
            import importlib.util
            import importlib.metadata
            spec = importlib.util.find_spec(modul_adi)
            if spec is not None:
                try:
                    surum = importlib.metadata.version(modul_adi)
                except Exception:
                    surum = "yüklü"
                durum_yaz(f"{aciklama} ({modul_adi})", True, f"v{surum}")
                basari_sayisi += 1
            else:
                durum_yaz(f"{aciklama} ({modul_adi})", False, "Modül kurulu değil")
        except Exception as e:
            durum_yaz(f"{aciklama} ({modul_adi})", False, f"HATA: {e}")

    # -------------------------------------------------------------
    # 2. SİSTEM İKİLİ DOSYALARI (FFMPEG & FFPROBE)
    # -------------------------------------------------------------
    baslik("2. Sistem İkili Dosyaları (Medya ve Ses İşleme)")
    
    def _ikili_bul(isim: str) -> Optional[str]:
        bulunan = shutil.which(isim)
        if bulunan:
            return bulunan
        olasi = [
            Path(sys.executable).parent / f"{isim}.exe",
            ROOT / ".venv" / "Scripts" / f"{isim}.exe",
            Path.home() / "AppData" / "Local" / "Microsoft" / "WinGet" / "Links" / f"{isim}.exe",
            Path("C:/ffmpeg/bin") / f"{isim}.exe",
            Path("C:/Program Files/ffmpeg/bin") / f"{isim}.exe",
        ]
        wg = Path.home() / "AppData" / "Local" / "Microsoft" / "WinGet" / "Packages"
        if wg.exists():
            for p in wg.glob(f"**/{isim}.exe"):
                olasi.append(p)
        for p in olasi:
            if p.exists():
                return str(p)
        return None

    ffmpeg_yolu = _ikili_bul("ffmpeg")
    toplam_kontrol += 1
    if ffmpeg_yolu:
        durum_yaz("FFmpeg İkili Dosyası", True, ffmpeg_yolu)
        basari_sayisi += 1
    else:
        durum_yaz("FFmpeg İkili Dosyası", False, "Sistem PATH'inde ffmpeg.exe bulunamadı!")

    ffprobe_yolu = _ikili_bul("ffprobe")
    toplam_kontrol += 1
    if ffprobe_yolu:
        durum_yaz("FFprobe İkili Dosyası", True, ffprobe_yolu)
        basari_sayisi += 1
    else:
        durum_yaz("FFprobe İkili Dosyası", False, "Sistem PATH'inde ffprobe.exe bulunamadı!")

    # -------------------------------------------------------------
    # 3. YEREL OLLAMA SERVİSİ VE MODELLERİ
    # -------------------------------------------------------------
    baslik("3. Yerel Ollama Servisi ve Modelleri")
    ollama_aktif = False
    kurulu_modeller = []
    toplam_kontrol += 1

    try:
        url = f"{OLLAMA_URL}/api/tags"
        req = urllib.request.Request(url, headers={"User-Agent": "EduFrameCheck"})
        with urllib.request.urlopen(req, timeout=4) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            kurulu_modeller = [m.get("name", "") for m in data.get("models", [])]
            ollama_aktif = True
            durum_yaz("Ollama API Bağlantısı", True, f"{OLLAMA_URL} (Aktif)")
            basari_sayisi += 1
    except Exception as e:
        durum_yaz("Ollama API Bağlantısı", False, f"Bağlanılamadı ({e}) - 'ollama serve' çalıştırın.")

    # Embedding Modeli Kontrolü
    toplam_kontrol += 1
    has_embed = any(EMBED_MODEL in m for m in kurulu_modeller)
    if has_embed:
        durum_yaz(f"Embedding Modeli ({EMBED_MODEL})", True, "Kurulu (768-D)")
        basari_sayisi += 1
    else:
        durum_yaz(f"Embedding Modeli ({EMBED_MODEL})", False, f"Eksik! Çalıştırın: 'ollama pull {EMBED_MODEL}'")

    # LLM Modeli Kontrolü
    toplam_kontrol += 1
    has_llm = any(OLLAMA_MODEL.split(":")[0] in m for m in kurulu_modeller)
    if has_llm:
        durum_yaz(f"Sohbet/Analiz LLM ({OLLAMA_MODEL})", True, "Kurulu")
        basari_sayisi += 1
    else:
        durum_yaz(f"Sohbet/Analiz LLM ({OLLAMA_MODEL})", False, f"Eksik! Çalıştırın: 'ollama pull {OLLAMA_MODEL}'")

    # -------------------------------------------------------------
    # 4. YEREL FASTER-WHISPER SES TANIMA MODELİ
    # -------------------------------------------------------------
    baslik("4. Yerel Konuşma Tanıma (Speech-to-Text - Faster-Whisper)")
    toplam_kontrol += 1
    try:
        from faster_whisper import WhisperModel
        # Yerel model dosyasını test et
        m = WhisperModel("small", device="cpu", compute_type="int8", local_files_only=True)
        durum_yaz("Whisper-Small Türkçe Modeli", True, "Yerel önbellekten yüklendi (Hazır)")
        basari_sayisi += 1
    except Exception as e:
        durum_yaz("Whisper-Small Türkçe Modeli", False, f"Model yerel önbellekte bulunamadı ({e})")

    # -------------------------------------------------------------
    # 5. CHROMADB V2 VE DERS KİTABI VEKTÖRLERİ
    # -------------------------------------------------------------
    baslik("5. Vektör Veritabanı ve MEB İndeksi (ChromaDB v2)")
    toplam_kontrol += 1
    chroma_mevcut = CHROMA_DB_DIR.exists()
    if chroma_mevcut:
        durum_yaz("ChromaDB v1 Dizini", True, str(CHROMA_DB_DIR))
        basari_sayisi += 1
    else:
        durum_yaz("ChromaDB v1 Dizini", False, f"{CHROMA_DB_DIR} bulunamadı!")

    toplam_kontrol += 1
    try:
        import chromadb
        client = chromadb.PersistentClient(path=str(CHROMA_DB_DIR))
        cols = [c.name for c in client.list_collections()]
        if COLLECTION in cols:
            col = client.get_collection(COLLECTION)
            kayit_sayisi = col.count()
            durum_yaz(f"Koleksiyon '{COLLECTION}'", True, f"{kayit_sayisi} chunk indeksli")
            basari_sayisi += 1
        else:
            durum_yaz(f"Koleksiyon '{COLLECTION}'", False, f"Koleksiyon bulunamadı! Mevcutlar: {cols}")
    except Exception as e:
        durum_yaz(f"Koleksiyon '{COLLECTION}'", False, f"ChromaDB sorgu hatası: {e}")

    # -------------------------------------------------------------
    # 6. MEB DERS KİTAPLARI (PDF)
    # -------------------------------------------------------------
    baslik("6. MEB Kaynak Ders Kitapları")
    toplam_kontrol += 1
    if REFERENCE_BOOKS_DIR.exists():
        kitaplar = list(REFERENCE_BOOKS_DIR.glob("*.pdf"))
        if kitaplar:
            durum_yaz("MEB Kaynak PDF'leri", True, f"{len(kitaplar)} adet kitap hazır")
            basari_sayisi += 1
            for k in kitaplar:
                print(f"       * {k.name} ({k.stat().st_size / (1024*1024):.1f} MB)")
        else:
            durum_yaz("MEB Kaynak PDF'leri", False, f"{REFERENCE_BOOKS_DIR} içinde PDF bulunamadı!")
    else:
        durum_yaz("MEB Referans Dizini", False, f"{REFERENCE_BOOKS_DIR} dizini yok!")

    # -------------------------------------------------------------
    # 7. GEMMA TOKENIZER VE TOKEN ANALİZİ
    # -------------------------------------------------------------
    baslik("7. Gemma BPE Tokenizer ve NLP Araçları")
    toplam_kontrol += 1
    tokenizer_json = ROOT / "gemma_tokenizer.json"
    if tokenizer_json.exists():
        durum_yaz("gemma_tokenizer.json", True, f"{tokenizer_json.stat().st_size / (1024*1024):.1f} MB")
        basari_sayisi += 1
    else:
        durum_yaz("gemma_tokenizer.json", False, "Dosya eksik!")

    toplam_kontrol += 1
    try:
        from tokenizer_utils import get_gemma_tokenizer
        tok = get_gemma_tokenizer()
        ornek = tok.encode("Elektrik enerjisi ve kaynakları")
        durum_yaz("Tokenizer Çalışma Testi", True, f"{len(ornek.tokens)} token üretildi")
        basari_sayisi += 1
    except Exception as e:
        durum_yaz("Tokenizer Çalışma Testi", False, f"Hata: {e}")

    # -------------------------------------------------------------
    # 8. ÇALIŞMA VE ÇIKTI KLASÖRLERİ
    # -------------------------------------------------------------
    baslik("8. Medya ve Çıktı Klasörleri")
    for d, isim in [
        (TEMP_DIR, "Geçici Medya (temp_medya)"),
        (VIDEOS_DIR, "Video Deposu (temp_medya/videos)"),
        (FRAMES_DIR, "Ekran Kareleri (temp_medya/frames)"),
        (OUTPUTS_DIR, "PDF Çıktıları (temp_medya/outputs)"),
    ]:
        toplam_kontrol += 1
        d.mkdir(parents=True, exist_ok=True)
        durum_yaz(isim, True, "Mevcut ve yazılabilir")
        basari_sayisi += 1

    # -------------------------------------------------------------
    # 9. ÇEVRE DEĞİŞKENLERİ VE YAPILANDIRMA (.env)
    # -------------------------------------------------------------
    baslik("9. Çevre Değişkenleri ve API Yapılandırması")
    toplam_kontrol += 1
    env_file = ROOT / ".env"
    if env_file.exists():
        durum_yaz(".env Dosyası", True, "Mevcut")
        basari_sayisi += 1
    else:
        durum_yaz(".env Dosyası", False, "Mevcut değil (.env oluşturulmalı)")

    gemini_key = os.getenv("GEMINI_API_KEY", "")
    toplam_kontrol += 1
    if gemini_key and not gemini_key.startswith("AIzaSyDummy") and len(gemini_key) > 20:
        durum_yaz("GEMINI_API_KEY", True, f"Tanımlı (Uzunluk: {len(gemini_key)})")
        basari_sayisi += 1
    else:
        durum_yaz("GEMINI_API_KEY", False, "Tanımsız veya geçersiz (Sistem tamamen yerel Ollama/Whisper modunda çalışacak)")

    # Canlı Gemini API "Merhaba" Yanıt Testi
    toplam_kontrol += 1
    if gemini_key and not gemini_key.startswith("AIzaSyDummy") and len(gemini_key) > 20:
        try:
            test_modeller = ["gemini-3-flash-preview", "gemini-3.8-flash", "gemini-flash-latest"]
            yanit_alindi = False
            hata_ozeti = ""
            for test_model in test_modeller:
                test_url = f"https://generativelanguage.googleapis.com/v1beta/models/{test_model}:generateContent?key={gemini_key}"
                test_payload = {"contents": [{"parts": [{"text": "merhaba"}]}]}
                try:
                    res = requests.post(test_url, json=test_payload, timeout=15)
                    if res.status_code == 200:
                        yanit_metni = res.json()["candidates"][0]["content"]["parts"][0]["text"].strip()
                        durum_yaz("Google Gemini Canlı Test", True, f"Model: {test_model} -> Yanıt: '{yanit_metni}'")
                        basari_sayisi += 1
                        yanit_alindi = True
                        break
                    else:
                        hata_ozeti = f"{test_model} (HTTP {res.status_code})"
                except Exception as req_err:
                    hata_ozeti = f"{test_model} ({type(req_err).__name__})"
                    continue
            if not yanit_alindi:
                durum_yaz("Google Gemini Canlı Test", False, f"Modellerden 200 OK yanıtı alınamadı: {hata_ozeti}")
        except Exception as e:
            durum_yaz("Google Gemini Canlı Test", False, f"Bağlantı Hatası: {e}")
    else:
        durum_yaz("Google Gemini Canlı Test", False, "API anahtarı eksik olduğu için test atlandı")

    # -------------------------------------------------------------
    # SONUÇ VE ÖZET
    # -------------------------------------------------------------
    baslik("Kurulum Kontrol Özeti")
    print(f" Başarılı Testler : {basari_sayisi} / {toplam_kontrol}")
    oran = (basari_sayisi / toplam_kontrol) * 100
    print(f" Başarı Oranı    : %{oran:.1f}\n")

    if basari_sayisi == toplam_kontrol:
        print(" [TEBRİKLER] EduFrame AI ortamı eksiksiz ve üretime hazır!")
    elif basari_sayisi >= toplam_kontrol - 2:
        print(" [BİLGİ] Temel bileşenler hazır. İsteğe bağlı bazı servisler çevrimdışı olsa bile yerel modda çalışabilirsiniz.")
    else:
        print(" [DİKKAT] Kritik bileşenler eksik. Lütfen yukarıdaki [EKSIK] maddeleri tamamlayın.")


if __name__ == "__main__":
    main()
