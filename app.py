"""LectraAI ders fasikülü çalışma alanı."""

import re
from hashlib import sha256
from pathlib import Path, PureWindowsPath
from tempfile import NamedTemporaryFile

import requests
import streamlit as st

from config import (
    EGITIM_SEVIYELERI, EMBED_MODEL, GEMINI_API_KEY, GEMINI_MODEL, OLLAMA_MODEL, OLLAMA_URL, VIDEOS_DIR,
)
from core.eduframe_agentic_core import EduFrameAgenticCore, OllamaFasikulHatasi
from services.gemini_analyzer import GeminiFasikulHatasi
from services.pdf_generator import EksikDersKaresiHatasi, PDFGenerator


ROOT = Path(__file__).resolve().parent
SINIFLAR = {
    "İlkokul": range(1, 5),
    "Ortaokul": range(5, 9),
    "Lise": range(9, 13),
}
GEMINI_SECENEKLERI = {
    "gemini-3-flash-preview": "Gemini 3 Flash",
    "gemini-3.8-flash": "Gemini 3.8 Flash",
    "gemini-3.5-flash-lite": "Gemini 3.5 Flash Lite",
}
VARSAYILAN_GEMINI = GEMINI_MODEL if GEMINI_MODEL in GEMINI_SECENEKLERI else next(iter(GEMINI_SECENEKLERI))
ASAMALAR = {
    "planla": (0.12, "Video kaynağı hazırlanıyor"),
    "video_indir": (0.30, "Konuşma ve ekran görüntüleri çıkarılıyor"),
    "ffmpeg_kare": (0.53, "MEB kitaplığı taranıyor"),
    "rag_baglam": (0.64, "Ders anlatımı hazırlanıyor"),
    "soru_uret": (0.92, "PDF sayfaları hazırlanıyor"),
}

st.set_page_config(page_title="LectraAI", page_icon="📘", layout="wide", initial_sidebar_state="collapsed")
st.markdown(
    """
    <style>
    .block-container { max-width: 1120px; padding-top: 4.25rem; padding-bottom: 3rem; }
    .lectra-hero {
        position: relative; isolation: isolate; overflow: hidden;
        padding: clamp(1.45rem, 4vw, 2.8rem); border: 1px solid #d2e1d8;
        border-radius: 24px; background: linear-gradient(112deg, #fff 34%, #e7f2e8 100%);
    }
    .lectra-hero::after {
        content: ""; position: absolute; z-index: -1; inset: 0 0 0 48%;
        background-image: radial-gradient(#87ad98 1px, transparent 1px);
        background-size: 15px 15px; opacity: .37;
        mask-image: linear-gradient(90deg, transparent, #000);
    }
    .lectra-brand {
        display: inline-flex; align-items: center; gap: .55rem;
        color: #225c4b; font-size: .92rem; font-weight: 750; letter-spacing: .025em;
    }
    .lectra-brand::before {
        content: ""; width: 1.2rem; height: 1.2rem; border-radius: 6px;
        background: linear-gradient(135deg, #126158 48%, #b4d7b3 49%);
    }
    .lectra-hero h1 {
        color: #14382f; font-size: clamp(2rem, 5vw, 3.4rem); line-height: 1.09;
        font-weight: 750; letter-spacing: -.055em; padding: .72rem 0 .62rem;
    }
    .lectra-hero p { color: #456458; margin: 0; font-size: 1.04rem; }
    [data-testid="stTabs"] [role="tablist"] { gap: .95rem; }
    [data-testid="stTabs"] [role="tab"] { padding-inline: .4rem; font-weight: 650; }
    .st-key-video_kaynagi, .st-key-fasikul_ayarlari, .st-key-fasikul_sonucu, .st-key-yedek_karari_kutusu {
        background: #fff; border-radius: 18px;
    }
    [data-testid="stFileUploaderDropzone"] {
        background: #f3f8f3; border: 1px dashed #80a596; border-radius: 12px;
    }
    [data-testid="stVerticalBlockBorderWrapper"] { border-radius: 18px; }
    [data-testid="stButton"] button[kind="primary"],
    [data-testid="stDownloadButton"] button[kind="primary"] { min-height: 3rem; font-weight: 700; }
    @media (max-width: 600px) {
        .block-container { padding-top: 3.4rem; }
        .lectra-hero { border-radius: 18px; }
    }
    </style>
    <header class="lectra-hero">
      <div class="lectra-brand">LectraAI</div>
      <h1>Ders Fasikülü Oluştur</h1>
      <p>Ders videolarından görsel destekli hızlıca ders notu çıkar.</p>
    </header>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(ttl=60, show_spinner=False)
def ollama_modelleri() -> list[str]:
    yanit = requests.get(f"{OLLAMA_URL.rstrip('/')}/api/tags", timeout=4)
    yanit.raise_for_status()
    return sorted({
        model["name"] for model in yanit.json().get("models", [])
        if model.get("name") and model["name"].split(":")[0] != EMBED_MODEL.split(":")[0]
        and "embed" not in model["name"].casefold()
        and "bert" not in (model.get("details") or {}).get("family", "").casefold()
    })


@st.cache_data(ttl=300, show_spinner=False)
def gemini_modelleri(api_key: str) -> list[str]:
    modeller = set()
    sayfa = None
    while True:
        parametreler = {"pageSize": 100}
        if sayfa:
            parametreler["pageToken"] = sayfa
        yanit = requests.get(
            "https://generativelanguage.googleapis.com/v1beta/models",
            headers={"x-goog-api-key": api_key}, params=parametreler, timeout=6,
        )
        yanit.raise_for_status()
        veri = yanit.json()
        for model in veri.get("models", []):
            ad = model["name"].removeprefix("models/")
            if ad in GEMINI_SECENEKLERI and "generateContent" in model.get("supportedGenerationMethods", []):
                modeller.add(ad)
        sayfa = veri.get("nextPageToken")
        if not sayfa:
            return [ad for ad in GEMINI_SECENEKLERI if ad in modeller]


def video_yukle(dosya) -> str:
    icerik = dosya.getbuffer()
    hedef = VIDEOS_DIR / sha256(icerik).hexdigest() / PureWindowsPath(dosya.name).name
    hedef.parent.mkdir(parents=True, exist_ok=True)
    if not hedef.is_file():
        with NamedTemporaryFile(dir=hedef.parent, delete=False) as gecici:
            yol = Path(gecici.name)
            try:
                gecici.write(icerik)
                gecici.close()
                yol.replace(hedef)
            finally:
                gecici.close()
                yol.unlink(missing_ok=True)
    return str(hedef.resolve())


def fasikul_uret(ayarlar: dict, *, tekrar: bool = False) -> None:
    st.session_state.pop("yedek_karari", None)
    st.session_state.pop("fasikul_hatasi", None)
    model = ayarlar["gemini_modeli"] if ayarlar["anlatim_modeli"] == "gemini" else ayarlar["ollama_modeli"]
    baslangic = "Ollama ile yeniden deneniyor" if tekrar else f"Üretim başlatılıyor · {model}"
    with st.status(baslangic, expanded=True) as durum:
        ilerleme_cubugu = st.progress(0.0)

        def ilerleme(dugum: str) -> None:
            if dugum == "gemini_analiz":
                deger, etiket = (0.80, "Değerlendirme soruları hazırlanıyor") if ayarlar["soru_eklensin_mi"] else (0.92, "PDF sayfaları hazırlanıyor")
            else:
                deger, etiket = ASAMALAR.get(dugum, (1.0, "PDF tamamlanıyor"))
            ilerleme_cubugu.progress(deger)
            durum.update(label=etiket)

        try:
            if ayarlar["anlatim_modeli"] == "ollama":
                ollama_modelleri.clear()
                try:
                    yerel = ollama_modelleri()
                except (requests.RequestException, ValueError) as hata:
                    raise OllamaFasikulHatasi("Ollama sunucusuna bağlanılamadı.") from hata
                if model not in yerel:
                    raise OllamaFasikulHatasi(f"{model} modeli Ollama’da bulunamadı.")
            sonuc = EduFrameAgenticCore().calistir(
                kullanici_girisi=f"{ayarlar['sinif_duzeyi']} ders notunu çıkar",
                ollama_yedek_onayi_gerekli=True, ilerleme=ilerleme, **ayarlar,
            )
            pdf_yolu = sonuc.get("cikti_pdf_yolu")
            if not pdf_yolu or not Path(pdf_yolu).is_file():
                raise OSError(sonuc.get("hata") or "PDF dosyası oluşturulamadı.")
        except OllamaFasikulHatasi as hata:
            st.session_state["yedek_karari"] = {"ayarlar": ayarlar.copy(), "hata": str(hata)}
            durum.update(label="Ollama çalıştırması durdu", state="error", expanded=False)
            return
        except (EksikDersKaresiHatasi, GeminiFasikulHatasi, OSError, RuntimeError) as hata:
            durum.update(label="PDF üretilemedi", state="error", expanded=True)
            st.session_state["fasikul_hatasi"] = str(hata)
            return
        st.session_state["fasikul_sonucu"] = sonuc
        ilerleme_cubugu.progress(1.0)
        durum.update(label="PDF hazır", state="complete", expanded=False)


st.write("")
tab_fasikul, tab_kitaplik = st.tabs(["Ders fasikülü", "MEB kitaplığında ara"])
with tab_fasikul:
    kaynak, ayar = st.columns([1.3, 1], gap="large")
    with kaynak, st.container(border=True, key="video_kaynagi"):
        st.subheader("Video kaynağı")
        tur = st.radio("Video kaynağı", ["YouTube", "Bilgisayardan"], horizontal=True, label_visibility="collapsed")
        baglanti = ""
        dosya = None
        if tur == "YouTube":
            baglanti = st.text_input("YouTube bağlantısı", placeholder="https://www.youtube.com/watch?v=…").strip()
        else:
            st.caption("Videoyu aşağıdaki alana sürükleyip bırak veya bilgisayarından seç.")
            dosya = st.file_uploader("Video dosyası", type=["mp4", "mkv", "mov", "webm", "avi"])

    with ayar, st.container(border=True, key="fasikul_ayarlari"):
        st.subheader("Fasikül ayarları")
        sol, sag = st.columns(2)
        seviye = sol.selectbox("Eğitim düzeyi", EGITIM_SEVIYELERI, filter_mode=None)
        siniflar = SINIFLAR.get(seviye)
        sinif = sag.selectbox("Sınıf", [f"{n}. Sınıf" for n in siniflar] if siniflar else ["Genel Lisans"],
                               index=2 if seviye == "İlkokul" else 0, filter_mode=None, disabled=siniflar is None)
        saglayici = st.radio("Model sağlayıcısı", ["Ollama", "Gemini"], horizontal=True)
        secilen_ollama, secilen_gemini = OLLAMA_MODEL, VARSAYILAN_GEMINI
        if saglayici == "Ollama":
            try:
                modeller = ollama_modelleri()
            except (requests.RequestException, ValueError):
                modeller = []
                st.caption("Ollama şu anda erişilemiyor.")
            if not modeller:
                modeller = [OLLAMA_MODEL]
            secilen_ollama = st.selectbox("Ollama modeli", modeller,
                                          index=modeller.index(OLLAMA_MODEL) if OLLAMA_MODEL in modeller else 0,
                                          filter_mode=None)
        elif GEMINI_API_KEY:
            try:
                modeller = gemini_modelleri(GEMINI_API_KEY)
            except (requests.RequestException, ValueError):
                modeller = []
                st.caption("Gemini model listesi alınamadı.")
            secilen_gemini = st.selectbox(
                "Gemini modeli", modeller, format_func=GEMINI_SECENEKLERI.__getitem__,
                index=modeller.index(VARSAYILAN_GEMINI) if VARSAYILAN_GEMINI in modeller else 0 if modeller else None,
                filter_mode=None, disabled=not modeller, placeholder="Kullanılabilir model bulunamadı" if not modeller else None,
            )
        else:
            secilen_gemini = None
            st.info("Gemini için API anahtarı ayarlanmalı.")
        sorulari_ekle = st.checkbox("Değerlendirme sorularını ve cevap anahtarını ekle", value=True)

    if st.button("PDF oluştur", type="primary", use_container_width=True,
                 disabled=not (dosya or baglanti) or (saglayici == "Gemini" and not secilen_gemini)):
        try:
            video = video_yukle(dosya) if dosya else baglanti
        except OSError as hata:
            st.error(f"Video kaydedilemedi. {hata}")
        else:
            fasikul_uret({
                "video_kaynagi": video, "egitim_seviyesi": seviye, "sinif_duzeyi": sinif,
                "anlatim_modeli": saglayici.lower(), "soru_eklensin_mi": sorulari_ekle,
                "ollama_modeli": secilen_ollama, "gemini_modeli": secilen_gemini,
            })

    if bekleyen := st.session_state.get("yedek_karari"):
        with st.container(border=True, key="yedek_karari_kutusu"):
            st.subheader("Gemini devreye girsin mi?")
            st.warning(bekleyen["hata"])
            st.caption(f"Evet seçilirse video {bekleyen['ayarlar']['gemini_modeli']} ile işlenir ve API kullanımı kotaya yansıyabilir.")
            if not GEMINI_API_KEY:
                st.caption("Gemini seçeneği için API anahtarı gerekli.")
            evet, hayir = st.columns(2)
            if evet.button("Evet, Gemini ile devam et", disabled=not GEMINI_API_KEY, use_container_width=True):
                fasikul_uret({**bekleyen["ayarlar"], "anlatim_modeli": "gemini"})
                st.rerun()
            if hayir.button("Hayır, Ollama ile yeniden dene", use_container_width=True):
                fasikul_uret(bekleyen["ayarlar"], tekrar=True)
                st.rerun()

    if hata := st.session_state.get("fasikul_hatasi"):
        st.error(hata)

    if sonuc := st.session_state.get("fasikul_sonucu"):
        analiz = sonuc.get("video_analizi") or {}
        bolumler = analiz.get("bolumler") or []
        sorular = sonuc.get("sorular") or []
        with st.container(border=True, key="fasikul_sonucu"):
            st.subheader("Fasikül hazır")
            st.caption(f"{analiz.get('sinif_duzeyi', '')} · {len(bolumler)} bölüm · {len(sorular)} değerlendirme sorusu")
            indir, bas = st.columns(2)
            pdf = Path(sonuc["cikti_pdf_yolu"])
            if pdf.is_file():
                indir.download_button("PDF’yi indir", data=pdf.read_bytes(), file_name=pdf.name,
                                      mime="application/pdf", type="primary", use_container_width=True)
            kayit = sonuc.get("icerik_kaydi_yolu")
            cerceve = None
            if kayit and re.fullmatch(r"[1-5]\.?\s*Sınıf", analiz.get("sinif_duzeyi") or ""):
                dizin = ROOT / "templates" / "cerceveler"
                if dizin.is_dir():
                    adlar = sorted(p.name for p in dizin.iterdir()
                                   if p.is_file() and p.suffix.lower() in {".jpeg", ".jpg", ".png"})
                    if adlar:
                        cerceve = bas.selectbox("Yeniden basım çerçevesi", [None, *adlar],
                                                format_func=lambda ad: ad or "Kayıtlı çerçeve", filter_mode=None)
            if kayit and bas.button("Kayıtlı içerikten yeniden bas", use_container_width=True):
                try:
                    yeniden = PDFGenerator().icerikten_yeniden_bas(Path(kayit), cerceve_adi=cerceve)
                except (OSError, ValueError, EksikDersKaresiHatasi) as hata:
                    st.error(f"PDF yeniden basılamadı. {hata}")
                else:
                    st.session_state["fasikul_sonucu"] = {**sonuc, "cikti_pdf_yolu": yeniden["pdf_path"]}
                    st.rerun()
            for i, bolum in enumerate(bolumler, 1):
                with st.expander(f"{bolum.get('bolum_basligi') or f'Bölüm {i}'} · {bolum.get('zaman_araligi_metni', '')}"):
                    if (kare := bolum.get("ekran_goruntusu_yolu")) and Path(kare).is_file():
                        st.image(kare, use_container_width=True)
                    st.markdown("**Orijinal transkript**")
                    st.write(bolum.get("transkript") or "")
                    st.markdown("**Ders anlatımı**")
                    st.write(bolum.get("duzenlenmis_anlatim") or "")
                    for kavram in bolum.get("onemli_kavramlar") or []:
                        st.write(f"{kavram.get('kavram', '')} — {kavram.get('aciklama', '')}")
                    if ek := bolum.get("kitap_ek_bilgisi"):
                        st.caption(f"{ek} · {bolum.get('kitap_referansi', '')}")
            if sorular:
                with st.expander("Değerlendirme ve cevap anahtarı"):
                    for i, soru in enumerate(sorular, 1):
                        st.markdown(f"**{i}. {soru.get('soru', '')}**")
                        for secenek in soru.get("secenekler") or []:
                            st.write(secenek)
                        st.caption(f"Cevap {soru.get('dogru_cevap', '')} · {soru.get('aciklama', '')}")

with tab_kitaplik:
    st.subheader("MEB kitaplığında ara")
    with st.form("meb_arama"):
        sorgu = st.text_input("Müfredat sorusu", placeholder="Elektrikli araçları kullanırken nelere dikkat etmeliyiz?")
        ara = st.form_submit_button("Kitapta ara")
    if ara and sorgu.strip():
        with st.spinner("Kitaplık taranıyor"):
            sonuc = EduFrameAgenticCore().calistir(
                kullanici_girisi=sorgu.strip(), egitim_seviyesi=seviye, sinif_duzeyi=sinif,
            )
        st.markdown(sonuc.get("yanit_metni", "Bilgi bulunamadı."))
