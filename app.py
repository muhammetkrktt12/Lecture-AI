"""LectraAI ders fasikülü çalışma alanı."""

import re
from base64 import b64encode
from hashlib import sha256
from io import BytesIO
from pathlib import Path, PureWindowsPath
from tempfile import NamedTemporaryFile

import requests
import streamlit as st
from PIL import Image

from config import (
    EGITIM_SEVIYELERI, EMBED_MODEL, GEMINI_API_KEY, GEMINI_MODEL, OLLAMA_MODEL, OLLAMA_URL, VIDEOS_DIR,
)
from core.eduframe_agentic_core import EduFrameAgenticCore
from services.pdf_generator import EksikDersKaresiHatasi, PDFGenerator
from services.pdf_job import PDFIsi


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
ONERILEN_OLLAMA_MODELLERI = ("qwen3.5:4b", "qwen3.5:9b")
ASAMALAR = {
    "planla": (0.12, "Video kaynağı hazırlanıyor"),
    "video_indir": (0.30, "Konuşma ve ekran görüntüleri çıkarılıyor"),
    "ffmpeg_kare": (0.53, "MEB kitaplığı taranıyor"),
    "rag_baglam": (0.64, "Ders anlatımı hazırlanıyor"),
    "soru_uret": (0.92, "PDF sayfaları hazırlanıyor"),
}

LOGO_YOLU = ROOT / "templates" / "lectra_ai.jpeg"
st.set_page_config(page_title="LectraAI", page_icon=str(LOGO_YOLU) if LOGO_YOLU.is_file() else "📘",
                   layout="wide", initial_sidebar_state="collapsed")


@st.cache_data(show_spinner=False)
def logo_gorseli(yol: Path, surum: int) -> str:
    with Image.open(yol) as logo:
        genislik, yukseklik = logo.size
        # Kare görselin etrafındaki boşluğu al, logonun çizimini ve renklerini koru.
        amblem = logo.crop((int(genislik * .025), int(yukseklik * .29),
                            int(genislik * .975), int(yukseklik * .71)))
        amblem.thumbnail((820, 420))
        tampon = BytesIO()
        amblem.convert("RGB").save(tampon, format="JPEG", quality=92)
    return b64encode(tampon.getvalue()).decode("ascii")


logo_html = (
    '<img alt="LectraAI" src="data:image/jpeg;base64,'
    + logo_gorseli(LOGO_YOLU, LOGO_YOLU.stat().st_mtime_ns) + '">'
    if LOGO_YOLU.is_file() else '<span>LectraAI</span>'
)
st.markdown(
    """
    <style>
    [data-testid="stAppViewContainer"] {
        background: radial-gradient(ellipse at 78% 0%, #142747 0%, transparent 44%), #080f20;
    }
    [data-testid="stHeader"] { background: rgba(8, 15, 32, .88); }
    .block-container { max-width: 1180px; padding-top: 4.6rem; padding-bottom: 3rem; }
    .lectra-hero {
        position: relative; isolation: isolate; overflow: hidden;
        padding: clamp(1.25rem, 2.1vw, 2rem) clamp(1.7rem, 4.4vw, 3.8rem);
        border: 1px solid #384e70; border-radius: 30px;
        background:
            radial-gradient(ellipse at 81% 82%, #0c789044, transparent 52%),
            radial-gradient(ellipse at 95% 5%, #aa744b29, transparent 44%),
            linear-gradient(120deg, #0b1428 12%, #111d36 100%);
        box-shadow: 0 26px 66px #03071185, inset 0 1px 0 #f1f8ff15;
    }
    .lectra-hero::before, .lectra-hero::after {
        content: ""; position: absolute; z-index: -1; pointer-events: none;
    }
    .lectra-hero::before {
        inset: 0 0 0 33%;
        background: radial-gradient(circle, #a5d7f03a 1px, transparent 1.5px) 0 0 / 17px 17px;
        mask-image: linear-gradient(90deg, transparent, #000 78%, #000);
    }
    .lectra-hero::after {
        top: 0; left: 7%; right: 7%; height: 1px;
        background: linear-gradient(90deg, transparent, #75e9fa 38%, #e9bb6e 76%, transparent);
    }
    .lectra-hero h1 {
        color: #f5f8ff; font-size: clamp(2.1rem, 4.7vw, 3.95rem); line-height: 1.03;
        font-weight: 850; letter-spacing: -.05em; padding: 0 0 .7rem;
    }
    .lectra-hero h1 span {
        color: #85e9fa;
        background: linear-gradient(100deg, #74e9f8 0%, #a7dbf2 42%, #f3cc88 94%);
        background-clip: text; -webkit-text-fill-color: transparent;
    }
    .lectra-hero p {
        max-width: 58ch; color: #c3d4ed; margin: 0; line-height: 1.55;
        font-size: clamp(1rem, 1.8vw, 1.17rem);
    }
    .lectra-logo {
        width: clamp(8.2rem, 15vw, 11rem); margin: 0 0 .85rem;
    }
    .lectra-logo img {
        display: block; width: 100%; height: auto;
        mix-blend-mode: screen;
    }
    .lectra-logo span { color: #edf4ff; font-size: 2rem; font-weight: 800; }
    [data-testid="stTabs"] [role="tablist"] { gap: .9rem; }
    [data-testid="stTabs"] [role="tab"] { padding-inline: .75rem; font-weight: 700; }
    .st-key-video_kaynagi, .st-key-fasikul_ayarlari, .st-key-fasikul_sonucu, .st-key-yedek_karari_kutusu {
        background: linear-gradient(145deg, #15223c, #0f192e); border-radius: 20px;
        box-shadow: 0 12px 32px #04081342;
    }
    [data-testid="stFileUploaderDropzone"] {
        background: #0c1930; border: 1px dashed #5688a8; border-radius: 14px;
    }
    [data-testid="stVerticalBlockBorderWrapper"] { border-radius: 20px; }
    [data-testid="stButton"] button[kind="primary"]:not(:disabled),
    [data-testid="stDownloadButton"] button[kind="primary"]:not(:disabled) {
        min-height: 3.15rem; font-weight: 750; color: #091b31;
        background: linear-gradient(110deg, #8deafb, #54c7ee);
        border-color: #95edfb; box-shadow: 0 6px 25px #47bdd93b;
    }
    @media (min-width: 900px) {
        .lectra-hero {
            box-sizing: border-box;
            width: calc(100vw - 2 * clamp(2rem, 2.5vw, 3rem));
            max-width: none;
            margin-left: 50%;
            transform: translateX(-50%);
        }
    }
    @media (max-width: 720px) {
        .block-container { padding-top: 3.4rem; }
        .lectra-hero { border-radius: 21px; }
        .lectra-logo { margin-bottom: .6rem; }
    }
    </style>
    <header class="lectra-hero notranslate" lang="tr" translate="no">
      <figure class="lectra-logo">{LOGO}</figure>
      <div class="lectra-hero-copy">
        <h1>Ders Fasikülü <span>Oluştur</span></h1>
        <p>Ders videolarından hızlıca görsel destekli ders notu çıkar</p>
      </div>
    </header>
    """.replace("{LOGO}", logo_html),
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
    if "fasikul_isi" in st.session_state:
        return
    st.session_state.pop("yedek_karari", None)
    st.session_state.pop("fasikul_hatasi", None)
    st.session_state.pop("fasikul_iptal_edildi", None)
    try:
        st.session_state["fasikul_isi"] = PDFIsi(ayarlar, tekrar=tekrar)
    except OSError as hata:
        st.session_state["fasikul_hatasi"] = f"PDF üretimi başlatılamadı. {hata}"
    else:
        st.rerun()


@st.fragment(run_every=0.75)
def fasikul_ilerlemesi() -> None:
    islem = st.session_state.get("fasikul_isi")
    if islem is None:
        return
    if st.button("İşlemi iptal et", key="fasikul_iptal", use_container_width=True):
        try:
            islem.iptal_et()
        except OSError as hata:
            st.session_state["fasikul_iptal_hatasi"] = str(hata)
        else:
            st.session_state.pop("fasikul_isi")
            st.session_state.pop("fasikul_iptal_hatasi", None)
            st.session_state["fasikul_iptal_edildi"] = True
            st.rerun()
    cevap = islem.sonucu_al()
    if cevap is not None:
        st.session_state.pop("fasikul_isi")
        st.session_state.pop("fasikul_iptal_hatasi", None)
        if cevap["durum"] == "hazir":
            st.session_state["fasikul_sonucu"] = cevap["sonuc"]
        elif cevap["durum"] == "ollama_hatasi":
            st.session_state["yedek_karari"] = {"ayarlar": islem.ayarlar, "hata": cevap["hata"]}
        else:
            st.session_state["fasikul_hatasi"] = cevap["hata"]
        st.rerun()
    if hata := st.session_state.get("fasikul_iptal_hatasi"):
        st.error(hata)
    asama = islem.son_asama()
    deger, etiket = ASAMALAR.get(asama, (0.0, islem.etiket))
    if asama == "gemini_analiz":
        deger, etiket = (0.80, "Değerlendirme soruları hazırlanıyor") if islem.ayarlar["soru_eklensin_mi"] else (0.92, "PDF sayfaları hazırlanıyor")
    elif asama == "pdf_rapor":
        deger, etiket = 0.98, "PDF tamamlanıyor"
    st.progress(deger, text=etiket)


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
                kurulu_modeller = set(ollama_modelleri())
            except (requests.RequestException, ValueError):
                kurulu_modeller = None
                st.caption("Ollama’ya bağlanılamadı. Model kurulum durumu doğrulanamadı.")
            modeller = list(dict.fromkeys((*ONERILEN_OLLAMA_MODELLERI, OLLAMA_MODEL, *sorted(kurulu_modeller or ()))))
            varsayilan = OLLAMA_MODEL
            if kurulu_modeller and varsayilan not in kurulu_modeller:
                varsayilan = next(ad for ad in modeller if ad in kurulu_modeller)

            def ollama_model_etiketi(ad: str) -> str:
                if kurulu_modeller is None:
                    return f"{ad} · kurulum durumu bilinmiyor"
                return ad if ad in kurulu_modeller else f"{ad} · yüklü değil"

            secilen_ollama = st.selectbox(
                "Ollama modeli", modeller, index=modeller.index(varsayilan),
                format_func=ollama_model_etiketi, filter_mode=None,
            )
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

    calisiyor = "fasikul_isi" in st.session_state
    if st.button("PDF oluştur", type="primary", use_container_width=True,
                 disabled=calisiyor or not (dosya or baglanti) or (saglayici == "Gemini" and not secilen_gemini)):
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

    if calisiyor:
        fasikul_ilerlemesi()
    if st.session_state.get("fasikul_iptal_edildi"):
        st.info("İşlem iptal edildi.")

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
            if hayir.button("Hayır, Ollama ile yeniden dene", use_container_width=True):
                fasikul_uret(bekleyen["ayarlar"], tekrar=True)

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
