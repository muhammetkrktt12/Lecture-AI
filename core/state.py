"""
EduFrame AI - LangGraph Ortak State Şeması (core/state.py)
---------------------------------------------------------
Bu dosya, LangGraph orkestratörünün düğümleri (nodes) arasında taşınan
durum (state) veri yapısını tanımlar.
"""

from typing import TypedDict, Optional, List, Dict, Any


class EduFrameState(TypedDict, total=False):
    """
    EduFrame AI ajanı için merkezi durum nesnesi.
    """
    # --- Kullanıcı Girdileri ---
    kullanici_girisi: str                   # Kullanıcının metinsel mesajı veya komutu
    video_kaynagi: Optional[str]            # YouTube URL'si veya yerel .mp4 dosya yolu
    egitim_seviyesi: str                    # 'İlkokul' | 'Ortaokul' | 'Lise' | 'Üniversite'
    sinif_duzeyi: Optional[str]             # Arayüzden seçilen sınıf
    soru_eklensin_mi: bool                  # Değerlendirme soruları oluşturulsun mu?

    # --- Orkestrasyon Kararı ---
    niyet: str                              # 'bilgi_sorgusu' | 'pdf_isleme'

    # --- RAG Katmanı ---
    rag_baglami: Optional[str]              # Seviye filtreli arama sonucu elde edilen ders kitabı metni
    rag_soru_ornekleri: Optional[List[Dict[str, Any]]]  # Kitaptan klonlanacak few-shot soru kalıpları
    rag_kaynaklari: Optional[List[Dict[str, Any]]]      # Alıntı yapılan sayfa ve kitap bilgileri

    # --- Video ve Medya İşleme ---
    video_bilgisi: Optional[Dict[str, Any]] # İndirilen videonun başlığı, süresi ve dosya yolu
    video_analizi: Optional[Dict[str, Any]] # Gemini 3 katmanlı pedagojik analiz sonucu

    # --- Soru ve Raporlama ---
    sorular: Optional[List[Dict[str, Any]]] # Üretilen değerlendirme soruları
    cikti_pdf_yolu: Optional[str]           # WeasyPrint ile üretilen PDF dosya yolu
    cikti_html_yolu: Optional[str]          # İnteraktif HTML dosya yolu

    # --- Nihai Çıktılar ve Hata Yönetimi ---
    yanit_metni: Optional[str]              # Bilgi sorguları için kullanıcıya dönen metin
    hata: Optional[str]                     # Varsa oluşan hata mesajı
