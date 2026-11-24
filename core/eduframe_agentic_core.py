"""
EduFrame AI - LangGraph Orkestratörü (core/eduframe_agentic_core.py)
-------------------------------------------------------------------
Bu modül, LangGraph (Agentic AI / Conditional Routing) kullanarak sistemin
bütün bileşenlerini tek bir otonom graf üzerinde koordine eder.

Mimari Akış ve Karar Düğümleri:
1. 'planla' Düğümü: Kullanıcının niyetini (intent) ölçer.
   - Niyet 'bilgi_sorgusu' ise: Video indirme ve FFmpeg ASLA çalışmaz. Sadece
     seçilen eğitim seviyesindeki RAG'dan bilgi çekilerek yanıt verilir.
   - Niyet 'pdf_isleme' ise: yt-dlp -> Whisper ve kare bölümleme -> Bölüm bazlı RAG
     ve pedagojik sentez -> İsteğe bağlı sorular -> ReportLab PDF
     boru hattı sırasıyla işletilir.
"""

import sys
from pathlib import Path
from tempfile import mkdtemp

# Kök dizini sys.path'e ekle
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from typing import Callable, Dict, Any, List, Optional
from langgraph.graph import StateGraph, END

from core.state import EduFrameState
from services.video_downloader import VideoDownloader
from services.rag_motoru import RAGMotoru
from services.video_processor import VideoProcessor
from services.gemini_analyzer import GeminiAnalyzer, GeminiFasikulHatasi
from services.gemini_usage import gemini_kullanim_oturumu
from services.question_generator import QuestionGenerator
from services.pdf_generator import EksikDersKaresiHatasi, PDFGenerator
from config import DEFAULT_SEVIYE, FRAMES_DIR, GEMINI_MODEL, OLLAMA_MODEL, TEMP_DIR
from utils.lexical_tools import kavram_anahtari, kelimelere_ayir, lexical_similarity

# Kalite ve Doğrulama Araçları (utils.lexical_tools öncelikli)
try:
    from utils.lexical_tools import validate_citations, rerank_candidates
except ImportError:
    try:
        from quality_utils import validate_citations, rerank_candidates
    except ImportError:
        validate_citations = None
        rerank_candidates = None

# ---------------------------------------------------------------------------
# DÜĞÜM FONKSİYONLARI (GRAPH NODES)
# ---------------------------------------------------------------------------

class OllamaFasikulHatasi(RuntimeError):
    """Ollama çalıştırması kullanıcı tarafından yeniden denenebilir."""

def planla_dugumu(state: EduFrameState) -> Dict[str, Any]:
    """
    Kullanıcının niyetini analiz eder:
    - Video kaynağı varsa veya "not çıkar", "özetle", "pdf" denilmişse -> 'pdf_isleme'
    - Sadece soru veya bilgi talebi varsa -> 'bilgi_sorgusu'
    """
    kullanici_girisi = (state.get("kullanici_girisi") or "").lower().strip()
    video_kaynagi = state.get("video_kaynagi")

    print(f"\n[LangGraph: planla] Niyet Analizi Yapılıyor...")
    print(f" -> Girdi: {kullanici_girisi}")
    print(f" -> Video Kaynağı: {video_kaynagi}")

    # Video kaynağı verilmişse veya not çıkarma anahtar sözcükleri varsa
    video_istegi_kelimeleri = ["not çıkar", "özetle", "pdf", "ders notu", "analiz et", "fasikül", "yansıt"]
    
    if video_kaynagi or any(w in kullanici_girisi for w in video_istegi_kelimeleri):
        niyet = "pdf_isleme"
        print(" -> Belirlenen Rota: 'pdf_isleme' (Tam Boru Hattı)")
    else:
        niyet = "bilgi_sorgusu"
        print(" -> Belirlenen Rota: 'bilgi_sorgusu' (Yalnızca RAG)")

    return {"niyet": niyet}


def rota_karari(state: EduFrameState) -> str:
    """LangGraph koşullu geçişini belirleyen yönlendirici fonksiyon."""
    return state.get("niyet", "bilgi_sorgusu")


def bilgi_sorgusu_dugumu(state: EduFrameState) -> Dict[str, Any]:
    """
    Yalnızca RAG aramasını çalıştırır; video indirme ve görüntü işlemeyi tetiklemez.
    d6_02_answer ve d6_03_validate mantığıyla kaynak etiketleme ([K1], [K2]) ve
    atıf doğrulama (validate_citations) uygular.
    """
    print("\n[LangGraph: bilgi_sorgusu] RAG tabanlı bilgi çekiliyor (3-gram Rerank aktif)...")
    soru = state.get("kullanici_girisi", "")
    seviye = state.get("egitim_seviyesi", DEFAULT_SEVIYE)

    rag = RAGMotoru()
    # 3-gram Türkçe Lexical Reranking destekli seviye filtreli arama
    dokumanlar = rag.seviye_filtreli_arama(sorgu=soru, egitim_seviyesi=seviye, k=3, rerank=True)

    if not dokumanlar:
        yanit = f"'{seviye}' seviyesindeki referans ders kitaplarında sorunuzla ilgili doğrudan bir içerik bulunamadı."
        return {"yanit_metni": yanit, "rag_baglami": ""}

    # [K1], [K2] biçiminde kaynak etiketleri oluştur (d6_02_answer mantığı)
    baglam_parcalari = []
    known_labels = []
    kaynaklar = []

    for i, doc in enumerate(dokumanlar, 1):
        label = f"K{i}"
        known_labels.append(label)
        source = doc.metadata.get("source", "Ders Kitabı")
        page = doc.metadata.get("page", "-")
        ders = doc.metadata.get("ders_adi", "Genel")
        
        baglam_parcalari.append(f"[{label}] (Ders: {ders}, Kitap: {source}, Sayfa: {page})\n{doc.page_content}")
        kaynaklar.append({"etiket": label, "kaynak": source, "sayfa": page, "ders": ders})

    baglam_metni = "\n\n".join(baglam_parcalari)

    # Güvenlik Duvarı: Doküman talimat değil veridir (d6_05_guvenlik mantığı)
    yanit = (
        f"**EduFrame AI ({seviye} Seviyesi Müfredat Bilgisi) [K1]:**\n\n"
        f"{dokumanlar[0].page_content[:350]}...\n\n"
        f"📚 **Doğrulanmış Kaynak:** {dokumanlar[0].metadata.get('source')} (Sayfa: {dokumanlar[0].metadata.get('page')}, Ders: {dokumanlar[0].metadata.get('ders_adi')})"
    )

    # Atıf Etiketi Doğrulama (d6_03_validate)
    if validate_citations:
        dogrulama = validate_citations(yanit, known_labels)
        if not dogrulama["valid"]:
            print(f"[UYARI: Atıf Denetimi] Geçersiz atıf tespit edildi: {dogrulama['invalid']}")

    return {
        "yanit_metni": yanit,
        "rag_baglami": baglam_metni,
        "rag_kaynaklari": kaynaklar,
    }


def video_indir_dugumu(state: EduFrameState) -> Dict[str, Any]:
    """Videoyu yt-dlp ile tekil olarak indirir veya yerel dosyayı hazırlar."""
    print("\n[LangGraph: video_indir] Video hazırlanıyor...")
    kaynak = state.get("video_kaynagi") or state.get("kullanici_girisi")

    downloader = VideoDownloader()
    sonuc = downloader.video_hazirla(kaynak=kaynak, cozunurluk="720")

    if not sonuc.get("success"):
        return {"hata": sonuc.get("error", "Video hazırlanamadı.")}

    if sonuc.get("source_type") == "local":
        sonuc["file_path"] = str(Path(kaynak).resolve())
    return {"video_bilgisi": sonuc}


def rag_baglam_dugumu(state: EduFrameState) -> Dict[str, Any]:
    """Yalnızca istenen değerlendirme soruları için kitap örneklerini getirir."""
    if not state.get("soru_eklensin_mi"):
        return {}
    return {
        "rag_soru_ornekleri": RAGMotoru().soru_formatlarini_getir(
            egitim_seviyesi=state.get("egitim_seviyesi", DEFAULT_SEVIYE),
            sinif=state.get("sinif_duzeyi") or "3. Sınıf",
            k=3,
        )
    }


def _kitap_eklerini_yerlestir(
    kitap_ekleri: List[tuple[Dict[str, Any], Dict[str, Any]]],
    rag: RAGMotoru,
) -> List[Dict[str, Any]]:
    """Tekrarlanan kitap ekini en yakın bölümde gösterir."""
    gruplar: Dict[str, List[tuple[Dict[str, Any], Dict[str, Any]]]] = {}
    for bolum, destek in kitap_ekleri:
        anahtar = " ".join(kelimelere_ayir(destek["metin"]))
        gruplar.setdefault(anahtar, []).append((bolum, destek))

    kaynaklar = []
    for adaylar in gruplar.values():
        kazanan = 0
        if len(adaylar) > 1:
            alinti = adaylar[0][1]["metin"]
            anlatimlar = [bolum["duzenlenmis_anlatim"] for bolum, _ in adaylar]
            try:
                skorlar = rag.metin_benzerlikleri(alinti, anlatimlar)
            except RuntimeError as hata:
                print(f"[RAG] Ek bilgi benzerliği hesaplanamadı, sözcüksel karşılaştırma kullanılıyor: {hata}")
                skorlar = [lexical_similarity(alinti, anlatim) for anlatim in anlatimlar]
            kazanan = max(range(len(adaylar)), key=skorlar.__getitem__)

        bolum, destek = adaylar[kazanan]
        kaynak = {k: v for k, v in destek["kaynak"].items() if k != "icerik"}
        bolum["kitap_ek_bilgisi"] = destek["metin"]
        bolum["kitap_referansi"] = f"{kaynak['kitap_adi']}, basılı sayfa {kaynak['printed_page']}"
        bolum["rag_kaynaklari"] = [kaynak]
        kaynaklar.append(kaynak)
    return kaynaklar


def gemini_analiz_dugumu(state: EduFrameState) -> Dict[str, Any]:
    """Transkript ve MEB kitap kaynaklarından bölüm anlatımı üretir."""
    print("\n[LangGraph: gemini_analiz] Transkript ve ayrı kitap desteği hazırlanıyor...")
    analiz_sonucu = state["video_analizi"]
    bolumler = analiz_sonucu["bolumler"]
    seviye = analiz_sonucu["egitim_seviyesi"]
    sinif = analiz_sonucu["sinif_duzeyi"]
    ders = analiz_sonucu["ders_adi"]
    gemini_secili = state.get("anlatim_modeli") == "gemini"
    onay_gerekli = not gemini_secili and state.get("ollama_yedek_onayi_gerekli", False)
    analyzer = GeminiAnalyzer(
        model=state.get("gemini_modeli") or GEMINI_MODEL,
        ollama_model=state.get("ollama_modeli") or OLLAMA_MODEL,
        api_key="" if onay_gerekli else None,
    )
    if gemini_secili and not analyzer._api_anahtari_var_mi():
        raise GeminiFasikulHatasi("Gemini ile PDF üretmek için GEMINI_API_KEY ayarlanmalıdır.")
    rag = RAGMotoru()
    print(f"\n[Pedagojik Sentez] {len(bolumler)} bölüm için ders anlatımı hazırlanıyor...")
    gorulen_kavramlar = {}
    for bolum in bolumler:
        ham_transkript = analyzer.transkript_filtrele(bolum.get("transkript", ""))
        bolum["temel_konu"] = ham_transkript[:180] or bolum.get("bolum_basligi", "")
        if gemini_secili:
            if not analyzer.gemini_fasikul_bolumu_uret(
                bolum, sinif, [], sorted(gorulen_kavramlar.values()),
                onbellek_dizini=TEMP_DIR / "gemini_bolum_onbellek",
                egitim_seviyesi=seviye,
                bolum_turu_belirle=True,
            ):
                raise GeminiFasikulHatasi(
                    f"Gemini bölümü tamamlayamadı: {bolum['bolum_basligi']} "
                    f"({bolum['zaman_araligi_metni']}). API erişimini, kotayı ve model yanıtını kontrol edin."
                )
        else:
            analyzer.bolum_pedagojik_sentezle(
                bolum=bolum, rag_sonuc={}, sinif=sinif, ders=ders,
                egitim_seviyesi=seviye, kitap_eki_ayri=True, transkript_icerigini_koru=True,
            )
            if onay_gerekli and analyzer.ollama_devre_disi:
                raise OllamaFasikulHatasi("Ollama bölüm anlatımı sırasında yanıt vermeyi durdurdu.")

        yeni_kavramlar = []
        for kavram in bolum.get("onemli_kavramlar") or []:
            if not isinstance(kavram, dict):
                continue
            ad = str(kavram.get("kavram") or "").strip()
            anahtar = kavram_anahtari(ad)
            if anahtar and anahtar not in gorulen_kavramlar:
                gorulen_kavramlar[anahtar] = ad
                yeni_kavramlar.append(kavram)
        bolum["onemli_kavramlar"] = yeni_kavramlar

        # Standart ortak veri yapısı alanlarını güvenceye al
        b_no = bolum.get("bolum_no", 1)
        bolum["bolum_id"] = bolum.get("bolum_id") or f"bolum_{b_no:02d}"
        bolum["pedagojik_anlatim"] = bolum.get("duzenlenmis_anlatim", "")
        z_aralik = bolum.get("zaman_araligi")
        if isinstance(z_aralik, dict):
            bolum["kare_zamani"] = z_aralik.get("kare_zamani", 0.0)
        elif "kare_zamani" not in bolum:
            bolum["kare_zamani"] = float(30 * b_no)

    if onay_gerekli and not any(b.get("kaynak_tipi") == "ollama" for b in bolumler):
        raise OllamaFasikulHatasi("Ollama kullanılabilir bir ders anlatımı üretemedi.")
    kitap_eki_sec = analyzer.gemini_kitap_ek_bilgisi_sec if gemini_secili else analyzer.ollama_kitap_ek_bilgisi_sec
    kitap_ekleri = []
    for bolum in bolumler:
        bolum.pop("kitap_ek_bilgisi", None)
        bolum["kitap_referansi"] = ""
        bolum["rag_kaynaklari"] = []
        bolum["rag_sorgulari"] = []
        anlatim = bolum.get("duzenlenmis_anlatim", "")
        if not anlatim or bolum.get("bolum_turu") != "konu_anlatimi":
            continue
        belgeler = rag.seviye_filtreli_arama(
            sorgu=anlatim, egitim_seviyesi=seviye, sinif=sinif, ders_adi=ders, k=3,
        )
        bolum["rag_sorgulari"] = [anlatim]
        adaylar = [
            {**belge.metadata, "icerik": belge.page_content}
            for belge in belgeler
            if belge.page_content.strip() and belge.metadata.get("kitap_adi")
            and belge.metadata.get("has_printed_page")
            and belge.metadata.get("printed_page") is not None
        ]
        destek = kitap_eki_sec(anlatim, adaylar, sinif)
        if onay_gerekli and analyzer.ollama_devre_disi:
            raise OllamaFasikulHatasi("Ollama kitap desteği hazırlanırken yanıt vermeyi durdurdu.")
        if destek:
            kitap_ekleri.append((bolum, destek))

    analiz_sonucu["tum_kaynaklar"] = _kitap_eklerini_yerlestir(kitap_ekleri, rag)
    if gemini_secili:
        konu_basligi = next((b.get("bolum_basligi") for b in bolumler
                             if b.get("kaynak_tipi") == "gemini" and b.get("bolum_basligi")), None)
        if konu_basligi:
            analiz_sonucu["unite_basligi"] = konu_basligi
    model_kaynaklari = {b.get("kaynak_tipi") for b in bolumler}
    if len(model_kaynaklari) == 1:
        analiz_sonucu["uretim_kaynagi"] = next(iter(model_kaynaklari))
    else:
        analiz_sonucu["uretim_kaynagi"] = "karma"
    return {"video_analizi": analiz_sonucu}


def ffmpeg_kare_dugumu(state: EduFrameState) -> Dict[str, Any]:
    """Gerçek medya üzerinden bölüm, kare ve konuşma zamanlarını oluşturur."""
    video_bilgisi = state.get("video_bilgisi") or {}
    video_yolu = video_bilgisi.get("file_path")
    if not video_yolu or not Path(video_yolu).is_file():
        raise EksikDersKaresiHatasi(state.get("hata") or "Video dosyası açılamadı.")

    processor = VideoProcessor(Path(mkdtemp(prefix="fasikul_", dir=FRAMES_DIR)))
    transkript_yolu = processor.transkript_olustur(video_yolu)
    bolumler = processor.otomatik_ekran_degisimlerini_bul(
        video_yolu, transkript_yolu=transkript_yolu
    )
    if state.get("anlatim_modeli") == "gemini":
        bolumler = processor.transkript_ile_eslestir(
            bolumler, transkript_yolu, zaman_damgalarini_koru=True,
        )
    else:
        bolumler = processor.transkript_ile_eslestir(bolumler, transkript_yolu)
    if not any(b.get("transkript") for b in bolumler):
        raise EksikDersKaresiHatasi("PDF üretilemedi. Videoda eşleşen ders konuşması bulunamadı.")
    PDFGenerator.ders_karelerini_dogrula(bolumler)

    return {
        "video_analizi": {
            "bolumler": bolumler,
            "unite_basligi": (video_bilgisi.get("title") or "Video Dersi").replace("_", " "),
            "egitim_seviyesi": state.get("egitim_seviyesi", DEFAULT_SEVIYE),
            "sinif_duzeyi": state.get("sinif_duzeyi") or "3. Sınıf",
            "ders_adi": "Fen Bilimleri",
        }
    }


def soru_uret_dugumu(state: EduFrameState) -> Dict[str, Any]:
    """Few-shot mantığıyla kitaptan klonlanan formatta sorular üretir."""
    print("\n[LangGraph: soru_uret] Değerlendirme soruları oluşturuluyor...")
    video_analizi = state.get("video_analizi") or {}
    seviye = state.get("egitim_seviyesi", DEFAULT_SEVIYE)
    rag_soru_ornekleri = state.get("rag_soru_ornekleri") or []

    gemini_secili = state.get("anlatim_modeli") == "gemini"
    qgen = QuestionGenerator(
        model=state.get("gemini_modeli") or GEMINI_MODEL,
        ollama_model=state.get("ollama_modeli") or OLLAMA_MODEL,
    )
    if gemini_secili:
        rag = RAGMotoru()
        sorular = qgen.gemini_sorularini_uret(video_analizi, seviye, rag_soru_ornekleri, rag)
        if len(sorular) != 10:
            raise GeminiFasikulHatasi("Gemini ve Ollama geçerli bir 10 soruluk değerlendirme seti üretemedi.")
        return {"sorular": sorular}

    sorular = qgen.ollama_fasikul_sorularini_uret(
        video_analizi, seviye, rag_soru_ornekleri, RAGMotoru(),
    )
    if state.get("ollama_yedek_onayi_gerekli") and qgen.ollama_devre_disi and not sorular:
        raise OllamaFasikulHatasi("Ollama değerlendirme soruları sırasında yanıt vermeyi durdurdu.")
    if len(sorular) < 10:
        print(f"[Ollama] Denetimden geçen {len(sorular)}/10 soru ile PDF üretimine devam ediliyor.")

    return {"sorular": sorular}


def pdf_rapor_dugumu(state: EduFrameState) -> Dict[str, Any]:
    """ReportLab motoru ile gerçek kurumsal PDF ders fasikülünü basar."""
    print("\n[LangGraph: pdf_rapor] ReportLab ile gerçek kurumsal PDF fasikülü basılıyor...")
    video_analizi = state.get("video_analizi") or {}
    sorular = state.get("sorular") or []
    cikti_dizini = state.get("pdf_cikti_dizini")
    generator = PDFGenerator(cikti_dizini=Path(cikti_dizini)) if cikti_dizini else PDFGenerator()
    sonuc = generator.pdf_olustur(
        video_analizi=video_analizi,
        sorular=sorular,
        video_kaynak_bilgisi="YouTube",
        icerigi_kaydet=True,
    )

    return {
        "cikti_pdf_yolu": sonuc.get("pdf_path"),
        "icerik_kaydi_yolu": sonuc.get("icerik_yolu"),
        "cikti_html_yolu": sonuc.get("html_path"),
        "yanit_metni": f"Ders notu PDF'iniz başarıyla oluşturuldu: {sonuc.get('pdf_path')}"
    }


# ---------------------------------------------------------------------------
# GRAF OLUŞTURUCU (STATE GRAPH BUILDER)
# ---------------------------------------------------------------------------

def create_eduframe_graph():
    """
    EduFrame AI için koşullu yönlendirmeli LangGraph akışını derler.
    """
    workflow = StateGraph(EduFrameState)

    # 1. Düğümleri Ekle
    workflow.add_node("planla", planla_dugumu)
    workflow.add_node("bilgi_sorgusu", bilgi_sorgusu_dugumu)
    workflow.add_node("video_indir", video_indir_dugumu)
    workflow.add_node("rag_baglam", rag_baglam_dugumu)
    workflow.add_node("gemini_analiz", gemini_analiz_dugumu)
    workflow.add_node("ffmpeg_kare", ffmpeg_kare_dugumu)
    workflow.add_node("soru_uret", soru_uret_dugumu)
    workflow.add_node("pdf_rapor", pdf_rapor_dugumu)

    # 2. Başlangıç Noktası
    workflow.set_entry_point("planla")

    # 3. Koşullu Yönlendirme (planla -> bilgi_sorgusu VEYA video_indir)
    workflow.add_conditional_edges(
        "planla",
        rota_karari,
        {
            "bilgi_sorgusu": "bilgi_sorgusu",
            "pdf_isleme": "video_indir",
        }
    )

    # Bilgi sorgusu direkt sonlanır
    workflow.add_edge("bilgi_sorgusu", END)

    # PDF işleme boru hattı akışı
    workflow.add_edge("video_indir", "ffmpeg_kare")
    workflow.add_edge("ffmpeg_kare", "rag_baglam")
    workflow.add_edge("rag_baglam", "gemini_analiz")

    # Soru seçeneğine göre koşullu geçiş
    def soru_eklensin_mi_kontrol(state: EduFrameState) -> str:
        return "soru_uret" if state.get("soru_eklensin_mi", False) else "pdf_rapor"

    workflow.add_conditional_edges(
        "gemini_analiz",
        soru_eklensin_mi_kontrol,
        {
            "soru_uret": "soru_uret",
            "pdf_rapor": "pdf_rapor",
        }
    )

    workflow.add_edge("soru_uret", "pdf_rapor")
    workflow.add_edge("pdf_rapor", END)

    return workflow.compile()


class EduFrameAgenticCore:
    """Streamlit ve dış çağrılar için kullanımı kolay ajan arayüzü."""

    def __init__(self):
        self.graph = create_eduframe_graph()

    def calistir(
        self,
        kullanici_girisi: str,
        video_kaynagi: Optional[str] = None,
        egitim_seviyesi: str = DEFAULT_SEVIYE,
        soru_eklensin_mi: bool = False,
        sinif_duzeyi: Optional[str] = None,
        anlatim_modeli: str = "ollama",
        *,
        ollama_modeli: Optional[str] = None,
        gemini_modeli: Optional[str] = None,
        ollama_yedek_onayi_gerekli: bool = False,
        ilerleme: Optional[Callable[[str], None]] = None,
        pdf_cikti_dizini: Optional[str] = None,
    ) -> EduFrameState:
        """
        Grafı verilen girdilerle baştan sona işletir ve nihai durumu döner.
        """
        baslangic_state: EduFrameState = {
            "kullanici_girisi": kullanici_girisi,
            "video_kaynagi": video_kaynagi,
            "egitim_seviyesi": egitim_seviyesi,
            "soru_eklensin_mi": soru_eklensin_mi,
            "sinif_duzeyi": sinif_duzeyi,
            "anlatim_modeli": anlatim_modeli,
            "ollama_modeli": ollama_modeli,
            "gemini_modeli": gemini_modeli,
            "ollama_yedek_onayi_gerekli": ollama_yedek_onayi_gerekli,
            "pdf_cikti_dizini": pdf_cikti_dizini,
        }
        with gemini_kullanim_oturumu():
            if ilerleme is None:
                return self.graph.invoke(baslangic_state)
            sonuc = baslangic_state
            for kip, veri in self.graph.stream(baslangic_state, stream_mode=["updates", "values"]):
                if kip == "values":
                    sonuc = veri
                elif kip == "updates":
                    for dugum in veri:
                        ilerleme(dugum)
            return sonuc


# --- KÖK DİZİN DOĞRUDAN TESTİ ---
if __name__ == "__main__":
    print("=" * 60)
    print("EduFrame AI - LangGraph Orkestratörü Testi")
    print("=" * 60)

    core = EduFrameAgenticCore()

    # TEST 1: Bilgi Sorgusu Rotası
    print("\n--- TEST 1: Bilgi Sorgusu Rotası Test Ediliyor ---")
    res1 = core.calistir(
        kullanici_girisi="Kuvvet cisimlerin hareketini nasıl etkiler?",
        video_kaynagi=None,
        egitim_seviyesi="İlkokul"
    )
    print("Niyet:", res1.get("niyet"))
    print("Yanıt:", res1.get("yanit_metni")[:120] if res1.get("yanit_metni") else "Boş")
    assert res1.get("niyet") == "bilgi_sorgusu", "Rota 1 hatalı!"

    # TEST 2: PDF İşleme Rotası
    print("\n--- TEST 2: PDF İşleme Rotası Test Ediliyor ---")
    res2 = core.calistir(
        kullanici_girisi="Bu ders videosundan not çıkar",
        video_kaynagi="https://www.youtube.com/watch?v=1EPox1m2MPQ",
        egitim_seviyesi="İlkokul",
        soru_eklensin_mi=True
    )
    print("Niyet:", res2.get("niyet"))
    print("HTML Çıktı:", res2.get("cikti_html_yolu"))
    print("PDF Çıktı:", res2.get("cikti_pdf_yolu"))
    print("Soru Sayısı:", len(res2.get("sorular") or []))
    assert res2.get("niyet") == "pdf_isleme", "Rota 2 hatalı!"

    print("\n[BAŞARILI] LangGraph orkestrasyonu tüm rotalarıyla doğrulandı.")
