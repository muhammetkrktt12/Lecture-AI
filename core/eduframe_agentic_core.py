"""
EduFrame AI - LangGraph Orkestratörü (core/eduframe_agentic_core.py)
-------------------------------------------------------------------
Bu modül, LangGraph (Agentic AI / Conditional Routing) kullanarak sistemin
bütün bileşenlerini tek bir otonom graf üzerinde koordine eder.

Mimari Akış ve Karar Düğümleri:
1. 'planla' Düğümü: Kullanıcının niyetini (intent) ölçer.
   - Niyet 'bilgi_sorgusu' ise: Video indirme ve FFmpeg ASLA çalışmaz. Sadece
     seçilen eğitim seviyesindeki RAG'dan bilgi çekilerek yanıt verilir.
   - Niyet 'pdf_isleme' ise: yt-dlp -> RAG Seviye Filtresi -> Gemini Multimodal
     3 Katmanlı Sentez -> FFmpeg Kare Yakalama -> Soru Klonlama -> WeasyPrint PDF
     boru hattı sırasıyla işletilir.
"""

import os
import sys
from pathlib import Path

# Kök dizini sys.path'e ekle
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from typing import Dict, Any, List, Optional
from langgraph.graph import StateGraph, END

from core.state import EduFrameState
from services.video_downloader import VideoDownloader
from services.rag_motoru import RAGMotoru
from services.video_processor import VideoProcessor
from services.gemini_analyzer import GeminiAnalyzer
from services.question_generator import QuestionGenerator
from services.pdf_generator import PDFGenerator
from config import DEFAULT_SEVIYE

# Kalite ve Doğrulama Araçları (utils.lexical_tools öncelikli)
try:
    from utils.lexical_tools import validate_citations, rerank_candidates
except ImportError:
    try:
        from quality_utils import validate_citations, rerank_candidates
    except ImportError:
        validate_citations = None
        rerank_candidates = None

# Gemma Tokenizer (tokenizer_utils & gemma_tokenizer.json)
try:
    from tokenizer_utils import get_gemma_tokenizer
    _CORE_TOKENIZER = get_gemma_tokenizer()
except Exception:
    _CORE_TOKENIZER = None


# ---------------------------------------------------------------------------
# DÜĞÜM FONKSİYONLARI (GRAPH NODES)
# ---------------------------------------------------------------------------

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

    return {"video_bilgisi": sonuc}


def rag_baglam_dugumu(state: EduFrameState) -> Dict[str, Any]:
    """Seçilen eğitim seviyesine ait ders kitabından anlamsal bağlamı çeker."""
    print("\n[LangGraph: rag_baglam] Müfredat bağlamı ve soru formatları getiriliyor...")
    seviye = state.get("egitim_seviyesi", DEFAULT_SEVIYE)
    video_bilgisi = state.get("video_bilgisi") or {}
    video_basligi = video_bilgisi.get("title", "Fen Bilimleri Dersi")

    rag = RAGMotoru()
    sinif = state.get("sinif_duzeyi") or "3. Sınıf"
    # Başlığa ve konuya göre seviye filtreli içerik çek (Sınıf İzolasyonu korumalı)
    docs = rag.seviye_filtreli_arama(
        sorgu=video_basligi,
        egitim_seviyesi=seviye,
        sinif=sinif,
        allow_cross_grade=False,
        k=4
    )
    baglam = "\n\n".join([f"[Sayfa {d.metadata.get('page')}]: {d.page_content}" for d in docs])

    # Kitaptaki soru formatlarını klonlamak için örnekleri al
    soru_ornekleri = rag.soru_formatlarini_getir(egitim_seviyesi=seviye, k=3)

    # Gemma Tokenizer ile bağlam token analizi (tokenizer_utils)
    if _CORE_TOKENIZER:
        token_adedi = len(_CORE_TOKENIZER.encode(baglam).tokens)
        print(f"[Tokenizer Analizi] RAG Bağlamı Gemma Token Sayısı: {token_adedi} token (Maks: 2048)")

    return {
        "rag_baglami": baglam,
        "rag_soru_ornekleri": soru_ornekleri,
        "rag_kaynaklari": [{"kaynak": d.metadata.get("source"), "sayfa": d.metadata.get("page")} for d in docs]
    }


def gemini_analiz_dugumu(state: EduFrameState) -> Dict[str, Any]:
    """
    Multimodal görsel materyal analizi ve 3 Boyutlu RAG entegrasyonu:
    1. Videoyu zaman damgalı bölümlere ayırır ve kareleri çıkarır.
    2. Her kare için gerçek multimodal görsel analiz yapar (unsurlar, açıklama, konuşma uyumu, konu ve kavramlar).
    3. Çok yönlü RAG sorguları ile ders kitabından parça çeker, alakasız parçaları eler (elenen_chunklar),
       yalnızca doğrulanmış parçaları (destekleyen_chunklar) kabul eder.
    4. Video ve ders kitabı somut örneklerini harmanlayarak nesnel, 3. tekil şahıs pedagojik sentez üretir.
    """
    print("\n[LangGraph: gemini_analiz] Multimodal materyal analizi, görsel kavrama ve 3D RAG entegrasyonu...")
    video_bilgisi = state.get("video_bilgisi") or {}
    video_yolu = video_bilgisi.get("file_path", "")
    video_basligi = video_bilgisi.get("title", "Ders Anlatımı")
    seviye = state.get("egitim_seviyesi", DEFAULT_SEVIYE)
    rag_baglami = state.get("rag_baglami", "")

    analyzer = GeminiAnalyzer()
    sinif_hedef = state.get("sinif_duzeyi") or "3. Sınıf"
    analiz_sonucu = analyzer.analiz_et(
        video_yolu=video_yolu,
        egitim_seviyesi=seviye,
        sinif_duzeyi=sinif_hedef,
        rag_baglami=rag_baglami,
        video_basligi=video_basligi,
    )

    bolumler = analiz_sonucu.get("bolumler", [])
    sinif = analiz_sonucu.get("sinif_duzeyi", "3. Sınıf")
    ders = analiz_sonucu.get("ders_adi", "Fen Bilimleri")

    # Kareleri analiz öncesi fiziksel olarak diske çıkart (görsel modelin okuyabilmesi için)
    if video_yolu and os.path.exists(video_yolu):
        try:
            processor = VideoProcessor()
            bolumler = processor.bolum_karelerini_isle(video_yolu, bolumler)
            analiz_sonucu["bolumler"] = bolumler
        except Exception as e:
            print(f"[UYARI] Kare çıkarma ön adımı sırasında hata: {e}")

    # -----------------------------------------------------------------------
    # GÖRSEL VE İŞİTSEL İÇERİKLERİN BİRLİKTE ANALİZİ & ÇOK YÖNLÜ 3D RAG
    # -----------------------------------------------------------------------
    rag = RAGMotoru()
    print(f"\n[Multimodal & 3D RAG] {len(bolumler)} bölüm için görsel analiz ve konu uygunluk denetimi başlatılıyor...")

    tum_dogrulanmis_kaynaklar = []
    for bolum in bolumler:
        b_no = bolum.get("bolum_no", 1)
        b_baslik = bolum.get("bolum_basligi", "")
        b_transkript = bolum.get("transkript", "")
        kare_yolu = bolum.get("ekran_goruntusu_yolu") or ""
        zaman_aralik = bolum.get("zaman_araligi")
        zaman_metni = bolum.get("zaman_araligi_metni") or (
            f"{zaman_aralik.get('baslangic', '00:00')} - {zaman_aralik.get('bitis', '00:00')}"
            if isinstance(zaman_aralik, dict) else "00:00 - 00:00"
        )

        # 1. Aşama: Multimodal Görsel Analiz (Görsel unsurlar, açıklama, konuşma uyumu)
        gorsel_sonuc = analyzer.gorsel_materyali_analiz_et(
            resim_yolu=kare_yolu,
            zaman_damgasi=zaman_metni,
            transkript=b_transkript,
            bolum_basligi=b_baslik,
        )
        bolum["gorsel_unsurlar"] = gorsel_sonuc.get("gorsel_unsurlar", [])
        bolum["gorsel_aciklamasi"] = gorsel_sonuc.get("gorsel_aciklamasi", "")
        bolum["gorsel_konusma_uyumu"] = gorsel_sonuc.get("gorsel_konusma_uyumu", {})
        bolum["temel_konu"] = gorsel_sonuc.get("temel_konu") or b_baslik
        bolum["alt_kavramlar"] = gorsel_sonuc.get("alt_kavramlar", [])
        bolum["gorsel_analiz_kaynagi"] = gorsel_sonuc.get("gorsel_analiz_kaynagi", "bilinmiyor")

        # 2. Aşama: Çok Yönlü 3 Boyutlu RAG ve Uygunluk Puanlaması (Alakasız chunkları ele)
        rag_sonuc = rag.bolum_bazli_cok_yonlu_arama(
            bolum_basligi=b_baslik,
            transkript=b_transkript,
            egitim_seviyesi=seviye,
            sinif=sinif,
            ders_adi=ders,
            k=2,
            gorsel_unsurlar=bolum.get("gorsel_unsurlar"),
            gorsel_aciklamasi=bolum.get("gorsel_aciklamasi", ""),
            temel_konu=bolum.get("temel_konu", ""),
            alt_kavramlar=bolum.get("alt_kavramlar"),
        )

        bolum["rag_sorgulari"] = rag_sonuc.get("rag_sorgulari", [])
        bolum["destekleyen_chunklar"] = rag_sonuc.get("destekleyen_chunklar", [])
        bolum["ana_kaynaklar"] = rag_sonuc.get("ana_kaynaklar", [])
        bolum["yardimci_kaynaklar"] = rag_sonuc.get("yardimci_kaynaklar", [])
        bolum["elenen_chunklar"] = rag_sonuc.get("elenen_chunklar", [])
        bolum["kaynak_destek_durumu"] = rag_sonuc.get("kaynak_destek_durumu", "destek_yok")

        bolum["rag_3d"] = {
            "bilgi": rag_sonuc["bilgi"]["baglam"],
            "pedagoji": rag_sonuc["pedagoji"]["baglam"],
            "olcme_degerlendirme": rag_sonuc["olcme_degerlendirme"]["baglam"]
        }

        if rag_sonuc.get("en_iyi_referans") and rag_sonuc.get("kaynak_destek_durumu") != "destek_yok":
            bolum["kitap_referansi"] = rag_sonuc["en_iyi_referans"]
        elif not bolum.get("kitap_referansi") or bolum.get("kitap_referansi") == "MEB Ders Kitabı Referansı":
            bolum["kitap_referansi"] = rag_sonuc.get("en_iyi_referans") or "Doğrulanmış RAG eşleşmesi bulunamadı"
        bolum["rag_kaynaklari"] = rag_sonuc.get("tum_kaynaklar", [])

        # Sadece doğrulanmış parçaları genel kaynakça havuzuna ekle
        for k_item in rag_sonuc.get("tum_kaynaklar", []):
            tum_dogrulanmis_kaynaklar.append(k_item)

        print(f" -> [Bölüm {b_no}] Multimodal görsel & 3D RAG tamamlandı: {b_baslik[:30]}... (Uyum: {bolum.get('gorsel_konusma_uyumu', {}).get('derece')}, Destek: {bolum.get('kaynak_destek_durumu')})")

    # -----------------------------------------------------------------------
    # AŞAMA 2: PEDAGOJİK HARMANLAMA VE DERİN KONU ANLATIMI SENTEZİ
    # -----------------------------------------------------------------------
    print(f"\n[Pedagojik Sentez] 10 bölüm için öğretmen anlatımı, görsel açıklaması ve MEB ders kitabı harmanlanıyor...")
    for bolum in bolumler:
        analyzer.bolum_pedagojik_sentezle(
            bolum=bolum,
            rag_sonuc={"bilgi": {"baglam": bolum.get("rag_3d", {}).get("bilgi", "")}, "pedagoji": {"baglam": bolum.get("rag_3d", {}).get("pedagoji", "")}},
            sinif=sinif,
            ders=ders,
            egitim_seviyesi=seviye
        )

        # Standart ortak veri yapısı alanlarını güvenceye al
        b_no = bolum.get("bolum_no", 1)
        bolum["bolum_id"] = bolum.get("bolum_id") or f"bolum_{b_no:02d}"
        bolum["pedagojik_anlatim"] = bolum.get("duzenlenmis_anlatim", "")
        z_aralik = bolum.get("zaman_araligi")
        if isinstance(z_aralik, dict):
            bolum["kare_zamani"] = z_aralik.get("kare_zamani", 0.0)
        elif "kare_zamani" not in bolum:
            bolum["kare_zamani"] = float(30 * b_no)

    analiz_sonucu["tum_kaynaklar"] = tum_dogrulanmis_kaynaklar
    return {"video_analizi": analiz_sonucu}


def ffmpeg_kare_dugumu(state: EduFrameState) -> Dict[str, Any]:
    """Ekrandaki her eğitim materyali için zaman damgalı kareleri yakalar."""
    print("\n[LangGraph: ffmpeg_kare] Materyal değişim kareleri FFmpeg ile alınıyor...")
    video_bilgisi = state.get("video_bilgisi") or {}
    video_yolu = video_bilgisi.get("file_path", "")
    video_analizi = state.get("video_analizi") or {}
    bolumler = video_analizi.get("bolumler", [])

    processor = VideoProcessor()
    guncel_bolumler = processor.bolum_karelerini_isle(video_yolu, bolumler)

    video_analizi_guncel = dict(video_analizi)
    video_analizi_guncel["bolumler"] = guncel_bolumler

    return {"video_analizi": video_analizi_guncel}


def soru_uret_dugumu(state: EduFrameState) -> Dict[str, Any]:
    """Few-shot mantığıyla kitaptan klonlanan formatta sorular üretir."""
    print("\n[LangGraph: soru_uret] Değerlendirme soruları oluşturuluyor...")
    video_analizi = state.get("video_analizi") or {}
    seviye = state.get("egitim_seviyesi", DEFAULT_SEVIYE)
    rag_soru_ornekleri = state.get("rag_soru_ornekleri") or []

    qgen = QuestionGenerator()
    sorular = qgen.soru_uret(
        video_analizi=video_analizi,
        egitim_seviyesi=seviye,
        rag_soru_ornekleri=rag_soru_ornekleri,
        soru_sayisi=10
    )

    return {"sorular": sorular}


def pdf_rapor_dugumu(state: EduFrameState) -> Dict[str, Any]:
    """ReportLab motoru ile gerçek kurumsal PDF ders fasikülünü basar."""
    print("\n[LangGraph: pdf_rapor] ReportLab ile gerçek kurumsal PDF fasikülü basılıyor...")
    video_analizi = state.get("video_analizi") or {}
    sorular = state.get("sorular") or []
    video_bilgisi = state.get("video_bilgisi") or {}
    video_kaynagi = state.get("video_kaynagi") or video_bilgisi.get("title", "Video Dersi")

    generator = PDFGenerator()
    sonuc = generator.pdf_olustur(
        video_analizi=video_analizi,
        sorular=sorular,
        video_kaynak_bilgisi=str(video_kaynagi),
    )

    return {
        "cikti_pdf_yolu": sonuc.get("pdf_path"),
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
    workflow.add_edge("video_indir", "rag_baglam")
    workflow.add_edge("rag_baglam", "gemini_analiz")
    workflow.add_edge("gemini_analiz", "ffmpeg_kare")

    # Soru seçeneğine göre koşullu geçiş
    def soru_eklensin_mi_kontrol(state: EduFrameState) -> str:
        return "soru_uret" if state.get("soru_eklensin_mi", False) else "pdf_rapor"

    workflow.add_conditional_edges(
        "ffmpeg_kare",
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
    ) -> EduFrameState:
        """
        Grafı verilen girdilerle baştan sona işletir ve nihai durumu döner.
        """
        baslangic_state: EduFrameState = {
            "kullanici_girisi": kullanici_girisi,
            "video_kaynagi": video_kaynagi,
            "egitim_seviyesi": egitim_seviyesi,
            "soru_eklensin_mi": soru_eklensin_mi,
        }
        return self.graph.invoke(baslangic_state)


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
