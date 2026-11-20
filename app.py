"""
EduFrame AI - AI Video Lesson Analyzer (app.py)
-----------------------------------------------
Ekran Değişimlerine ve Görsel Materyallere Dayalı Otonom Ders Notu ve PDF Fasikül Üreticisi
"""

import os
import sys
from hashlib import sha256
from html import escape
from pathlib import Path
from tempfile import NamedTemporaryFile

ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

try:
    import streamlit as st
except ImportError:
    print("Streamlit modülü henüz kurulu değil. 'pip install streamlit' komutunu çalıştırın.")
    sys.exit(1)

from config import (
    EGITIM_SEVIYELERI,
    DEFAULT_SEVIYE,
    GEMINI_API_KEY,
    REFERENCE_BOOKS_DIR,
    OUTPUTS_DIR,
    VIDEOS_DIR,
)
from core.eduframe_agentic_core import EduFrameAgenticCore
from services.pdf_generator import EksikDersKaresiHatasi
from services.rag_motoru import RAGMotoru
from services.video_processor import VideoProcessor

# --- SAYFA YAPILANDIRMASI ---
st.set_page_config(
    page_title="EduFrame AI - Görsel Materyal Odaklı Ders Fasikülü",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .main-title {
        font-size: 2.3rem;
        font-weight: 800;
        color: #1e3a8a;
        margin-bottom: 0.1rem;
    }
    .sub-title {
        font-size: 1.05rem;
        color: #475569;
        margin-bottom: 1.2rem;
    }
    .time-badge {
        background-color: #0284c7;
        color: #ffffff;
        padding: 3px 10px;
        border-radius: 12px;
        font-weight: 700;
        font-size: 0.85rem;
    }
    .box-transcript {
        background-color: #f8fafc;
        border-left: 4px solid #94a3b8;
        padding: 10px;
        border-radius: 6px;
        font-style: italic;
        color: #334155;
        margin-bottom: 8px;
    }
    .box-pedagogical {
        background-color: #eff6ff;
        border-left: 4px solid #2563eb;
        padding: 10px;
        border-radius: 6px;
        color: #1e3a8a;
        font-weight: 500;
        margin-bottom: 8px;
    }
    .box-concept {
        background-color: #f0fdf4;
        border-left: 4px solid #16a34a;
        padding: 10px;
        border-radius: 6px;
        color: #14532d;
    }
</style>
""", unsafe_allow_html=True)


# --- YAN MENÜ ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3429/3429440.png", width=65)
    st.markdown("### ⚙️ Müfredat ve Kademe")

    secilen_seviye = st.selectbox(
        "🎓 Hedef Eğitim Seviyesi:",
        options=EGITIM_SEVIYELERI,
        index=0,
    )

    sinif_secenekleri = {
        "İlkokul": ["3. Sınıf", "4. Sınıf"],
        "Ortaokul": ["5. Sınıf", "6. Sınıf", "7. Sınıf", "8. Sınıf"],
        "Lise": ["9. Sınıf", "10. Sınıf", "11. Sınıf", "12. Sınıf"],
        "Üniversite": ["Genel Lisans"]
    }

    secilen_sinif = st.selectbox(
        "🏫 Sınıf / Kademe:",
        options=sinif_secenekleri.get(secilen_seviye, ["Genel"])
    )

    soru_tercihi = st.checkbox(
        "📝 Değerlendirme ve Test Soruları Ekle",
        value=True,
        help="Kademenin ders kitabı soru formatına uygun sorular ve cevap anahtarı üretir."
    )

    st.markdown("---")
    st.markdown("### 📚 Küratörlü Kitaplık")
    try:
        rag = RAGMotoru()
        durum = rag.mevcut_veritabani_durumu()
        st.write(f"**Vektör İndeks:** `{durum.get('toplam_parca', 0)} parça`")
        st.caption("3-gram Türkçe Rerank ve Gemma Tokenizer aktif.")
    except Exception:
        pass

    vp = VideoProcessor()
    if vp.ffmpeg_yolu:
        st.success("✅ FFmpeg Algılandı (Gyan.dev)")
    else:
        st.warning("⚠️ FFmpeg Bulunamadı")


# --- ANA EKRAN ---
st.markdown('<div class="main-title">EduFrame AI – Görsel Materyal Odaklı Ders Fasikülü</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-title">'
    'Videoyu ekrandaki görsel materyal değişimlerine (slayt, şema, tahta) göre bölümler; '
    'her materyalin ekranda kaldığı süre boyunca yapılan konuşmaları ders kitabı (RAG) ile sentezleyerek kurumsal PDF ders notu oluşturur.'
    '</div>',
    unsafe_allow_html=True
)

tab_video, tab_asistan = st.tabs(["🎥 Görsel Materyal Analizi & PDF Fasikül", "💬 Müfredat Bilgi Asistanı"])


# ---------------------------------------------------------------------------
# TAB 1: GÖRSEL MATERYAL VE PDF FASİKÜL BORU HATTI
# ---------------------------------------------------------------------------
with tab_video:
    st.markdown("#### 1. Ders Videosu Kaynağı")
    kaynak_turu = st.radio("Kaynak Türü:", ["YouTube Linki", "Yerel MP4 Dosyası"], horizontal=True)

    video_kaynagi = None
    if kaynak_turu == "YouTube Linki":
        yt_input = st.text_input("YouTube URL:", placeholder="Örn: https://www.youtube.com/watch?v=1EPox1m2MPQ")
        if yt_input:
            video_kaynagi = yt_input.strip()
    else:
        dosya = st.file_uploader("Yerel MP4 Dosyası Seçin:", type=["mp4", "mkv", "mov"])
        if dosya:
            icerik = dosya.getbuffer()
            hedef = VIDEOS_DIR / sha256(icerik).hexdigest() / Path(dosya.name).name
            hedef.parent.mkdir(parents=True, exist_ok=True)
            if not hedef.is_file():
                with NamedTemporaryFile(dir=hedef.parent, delete=False) as gecici:
                    try:
                        gecici.write(icerik)
                        gecici.close()
                        Path(gecici.name).replace(hedef)
                    finally:
                        gecici.close()
                        Path(gecici.name).unlink(missing_ok=True)
            video_kaynagi = str(hedef.resolve())

    # Eğer önceden inmiş örnek video varsa pratik kullanım için öner
    ornek_mevcut = ROOT_DIR / "temp_medya" / "video_1EPox1m2MPQ.mp4"
    if not video_kaynagi and ornek_mevcut.exists():
        if st.checkbox("💡 Test için sistemdeki mevcut Fen Bilimleri videosunu kullan", value=True):
            video_kaynagi = str(ornek_mevcut.resolve())

    st.markdown("---")
    calistir_btn = st.button("🚀 Ekran Değişimlerini İncele ve PDF Ders Fasikülünü Bas", type="primary", use_container_width=True)

    if calistir_btn:
        if not video_kaynagi:
            st.error("Lütfen bir video linki girin veya MP4 dosyası yükleyin!")
        else:
            with st.status("Görsel Materyal ve Ses Akışı İnceleniyor...", expanded=True) as status:
                st.write("1️⃣ LangGraph: Niyet analizi yapılıyor ve 'pdf_isleme' boru hattı kuruluyor...")
                st.write("2️⃣ Whisper ve video işleme: Konuşma, ders bölümleri ve ekran kareleri eşleştiriliyor...")
                st.write("3️⃣ RAG: Her bölüm için seçilen sınıfın ders kitapları aranıyor...")
                st.write("4️⃣ Bölüm analizi: Ders anlatımı ve önemli kavramlar hazırlanıyor...")
                if soru_tercihi:
                    st.write(f"5️⃣ Soru Motoru: {secilen_seviye} kademesine uygun sorular ve cevap anahtarı üretiliyor...")
                st.write("📄 Ders fasikülü PDF olarak derleniyor...")

                core = EduFrameAgenticCore()
                try:
                    sonuc_state = core.calistir(
                        kullanici_girisi=f"{secilen_sinif} ders notunu çıkar",
                        video_kaynagi=video_kaynagi,
                        egitim_seviyesi=secilen_seviye,
                        sinif_duzeyi=secilen_sinif,
                        soru_eklensin_mi=soru_tercihi
                    )
                except EksikDersKaresiHatasi as hata:
                    status.update(label="Ders fasikülü üretilemedi", state="error", expanded=True)
                    st.error(str(hata))
                    st.stop()
                status.update(label="Ders Fasikülü Başarıyla Üretildi!", state="complete", expanded=False)

            # --- ÇIKTILARI GÖSTER ---
            video_analizi = sonuc_state.get("video_analizi") or {}
            bolumler = video_analizi.get("bolumler", [])
            sorular = sonuc_state.get("sorular") or []
            pdf_yolu = sonuc_state.get("cikti_pdf_yolu")

            # İNDİRME ALANI
            st.success(f"🎉 **{video_analizi.get('ders_adi', 'FEN BİLİMLERİ')} – {video_analizi.get('unite_basligi', 'Ders Notu')}**")
            
            if pdf_yolu and os.path.exists(pdf_yolu):
                with open(pdf_yolu, "rb") as f:
                    st.download_button(
                        "📥 GERÇEK KURUMSAL PDF DERS FASİKÜLÜNÜ İNDİR (.PDF)",
                        data=f.read(),
                        file_name=Path(pdf_yolu).name,
                        mime="application/pdf",
                        type="primary",
                        use_container_width=True
                    )

            # GENEL ÖZET
            if video_analizi.get("genel_ozet"):
                st.info(f"**📌 Videonun Genel Konu Özeti:**\n\n{video_analizi.get('genel_ozet')}")

            # GÖRSEL MATERYAL BÖLÜMLERİ
            st.markdown("### 🖼️ Ekran Değişimlerine Göre Bölümlenmiş Materyaller")

            for idx, bolum in enumerate(bolumler, start=1):
                b_baslik = bolum.get("bolum_basligi", f"Bölüm {idx}")
                z_aralik = bolum.get("zaman_araligi_metni", "00:00 – 00:00")

                with st.expander(f"📁 {b_baslik}  |  ⏱️ {z_aralik}", expanded=True):
                    # Ekran Görüntüsü
                    kare_yolu = bolum.get("ekran_goruntusu_yolu")
                    if kare_yolu and os.path.exists(kare_yolu) and os.path.getsize(kare_yolu) > 200:
                        st.image(kare_yolu, caption=f"Ekrandaki Materyal (Zaman Aralığı: {z_aralik})", use_container_width=True)

                    st.markdown("**1. Öğretmenin Anlatımı (Orijinal Transkript):**")
                    transkript = escape(bolum.get("transkript") or "")
                    st.markdown(f'<div class="box-transcript">"{transkript}"</div>', unsafe_allow_html=True)

                    st.markdown(f"**2. Düzenlenmiş Konu Anlatımı ({secilen_sinif} Ders Kitabı Referanslı):**")
                    anlatim = escape(bolum.get("duzenlenmis_anlatim") or "")
                    st.markdown(f'<div class="box-pedagogical">{anlatim}</div>', unsafe_allow_html=True)

                    st.markdown("**3. Bölümün Önemli Kavramları ve Açıklamaları:**")
                    kavramlar = bolum.get("onemli_kavramlar", [])
                    if kavramlar:
                        kavram_metin = "<br/>".join([f"• <b>{k.get('kavram')}:</b> {k.get('aciklama')}" for k in kavramlar])
                        st.markdown(f'<div class="box-concept">{kavram_metin}</div>', unsafe_allow_html=True)

                    if bolum.get("kitap_referansi"):
                        st.caption(f"📚 {bolum.get('kitap_referansi')}")

            # DEĞERLENDİRME VE TEST SORULARI
            if sorular:
                st.markdown("---")
                st.markdown(f"### ❓ Değerlendirme ve Test Soruları ({secilen_seviye})")
                for s_no, s in enumerate(sorular, start=1):
                    with st.container():
                        st.markdown(f"**Soru {s_no} [{s.get('tip')}]:** {s.get('soru')}")
                        if s.get("secenekler"):
                            for opt in s.get("secenekler"):
                                st.write(f"&nbsp;&nbsp;&nbsp;&nbsp;{opt}")
                        with st.expander(f"👁️ Soru {s_no} Çözüm ve Cevap Anahtarı"):
                            st.write(f"**Doğru Cevap:** {s.get('dogru_cevap')}")
                            st.write(f"**Açıklama:** {s.get('aciklama')}")
                        st.markdown("---")


# ---------------------------------------------------------------------------
# TAB 2: MÜFREDAT BİLGİ ASİSTANI
# ---------------------------------------------------------------------------
with tab_asistan:
    st.markdown("#### 💬 Resmi Müfredat Ders Kitaplarında Arama")
    st.caption("3-gram Türkçe Rerank, Gemma Tokenizer ve [K1],[K2] atıf doğrulama ile çalışır.")
    soru_girdisi = st.text_input("Ders kitabına soru sorun:", placeholder="Örn: Elektrikli araçları kullanırken nelere dikkat edilmelidir?")
    if st.button("🔍 Kitapta Ara"):
        if soru_girdisi:
            with st.spinner("Taranıyor..."):
                core = EduFrameAgenticCore()
                res = core.calistir(kullanici_girisi=soru_girdisi, egitim_seviyesi=secilen_seviye)
                st.markdown(res.get("yanit_metni", "Bilgi bulunamadı."))
