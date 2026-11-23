"""
EduFrame AI - Doğrudan PDF Üretim Servisi (services/pdf_generator.py)
---------------------------------------------------------------------
Bu modül, ReportLab motorunu kullanarak kullanıcının belirlediği
pedagojik ders fasikülü mizanpajına tam uygun, gerçek bir .PDF dosyası basar.

Mizanpaj Mimarisi:
1. Kapak ve Başlık Bilgisi (Sayfa 1 Üst Bilgi)
2. Materyal Değişim Bölümleri (Her bölüm yeni sayfada başlar)
   - Materyal Başlığı ve Zaman Aralığı
   - Yüksek Çözünürlüklü Ekran Görüntüsü
   - Orijinal Transkript Kutusu
   - Düzenlenmiş Konu Anlatımı Kutusu
   - Bölümün Önemli Kavramları Kutusu
   - İlgili Ders Kitabı Sayfa Referansı
3. Varsa genel özet, değerlendirme soruları, cevap anahtarı ve kaynakça
"""

import os
import sys
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional
from xml.sax.saxutils import escape

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Image,
    Table,
    TableStyle,
    KeepTogether,
    HRFlowable,
    PageBreak,
)
from reportlab.pdfgen import canvas
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from PIL import Image as PILImage

try:
    from config import ROOT, OUTPUTS_DIR
except ImportError:
    ROOT = Path(__file__).resolve().parent.parent
    OUTPUTS_DIR = ROOT / "temp_medya" / "outputs"

OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# TÜRKÇE FONT YÖNETİMİ
# ---------------------------------------------------------------------------
def _turkce_fontlari_yukle():
    """Windows Arial fontlarını ReportLab'e kaydeder."""
    font_path = "C:/Windows/Fonts/arial.ttf"
    bold_path = "C:/Windows/Fonts/arialbd.ttf"
    italic_path = "C:/Windows/Fonts/ariali.ttf"
    try:
        if os.path.exists(font_path):
            pdfmetrics.registerFont(TTFont("EduArial", font_path))
            pdfmetrics.registerFont(TTFont("EduArial-Bold", bold_path if os.path.exists(bold_path) else font_path))
            pdfmetrics.registerFont(TTFont("EduArial-Italic", italic_path if os.path.exists(italic_path) else font_path))
            return "EduArial", "EduArial-Bold", "EduArial-Italic"
    except Exception as e:
        print(f"[UYARI] Arial fontu yüklenemedi: {e}")
    return "Helvetica", "Helvetica-Bold", "Helvetica-Oblique"


FONT_REG, FONT_BOLD, FONT_ITALIC = _turkce_fontlari_yukle()


# ---------------------------------------------------------------------------
# SAYFA NUMARALANDIRMA (SAYFA X / Y)
# ---------------------------------------------------------------------------
class NumberedCanvas(canvas.Canvas):
    """Her sayfanın altına toplam sayfa sayısıyla birlikte numaralandırma ekler."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_page_decorations(num_pages)
            super().showPage()
        super().save()

    def draw_page_decorations(self, total_pages):
        self.saveState()
        self.setFont(FONT_REG, 8)
        self.setFillColor(colors.HexColor("#64748b"))

        # Üst bilgi
        self.drawString(42, 805, "EduFrame AI – Otonom Ders Notu ve Pedagojik Fasikül")
        self.setStrokeColor(colors.HexColor("#e2e8f0"))
        self.setLineWidth(0.5)
        self.line(42, 800, 553, 800)

        # Alt bilgi
        page_str = f"Sayfa {self._pageNumber} / {total_pages}"
        self.drawRightString(553, 30, page_str)
        self.drawString(42, 30, "Bu ders notu video görsel materyalleri ve MEB müfredat kaynakları referans alınarak üretilmiştir.")
        self.line(42, 40, 553, 40)

        self.restoreState()


# ---------------------------------------------------------------------------
# ANA PDF ÜRETİCİ SINIF
# ---------------------------------------------------------------------------
class EksikDersKaresiHatasi(ValueError):
    """Bir ders bölümünün PDF'ye eklenebilecek karesi yok."""


class PDFGenerator:
    """
    RAG, Transkript ve Ekran Karelerini kurumsal ders fasikülü PDF'ine dönüştürür.
    """

    def __init__(self, cikti_dizini: Optional[Path] = None):
        self.cikti_dizini = Path(cikti_dizini) if cikti_dizini else OUTPUTS_DIR
        self.cikti_dizini.mkdir(parents=True, exist_ok=True)
        self._stilleri_hazirla()

    def _stilleri_hazirla(self):
        """ReportLab tipografi ve paragraf stillerini yapılandırır."""
        self.styles = getSampleStyleSheet()

        self.style_super_title = ParagraphStyle(
            "EduSuperTitle",
            fontName=FONT_BOLD,
            fontSize=17,
            leading=21,
            textColor=colors.HexColor("#1e3a8a"),
            spaceAfter=3,
        )

        self.style_sub_title = ParagraphStyle(
            "EduSubTitle",
            fontName=FONT_BOLD,
            fontSize=13,
            leading=16,
            textColor=colors.HexColor("#0f172a"),
            spaceAfter=6,
        )

        self.style_meta = ParagraphStyle(
            "EduMeta",
            fontName=FONT_BOLD,
            fontSize=8.5,
            leading=12,
            textColor=colors.HexColor("#2563eb"),
        )

        self.style_section_title = ParagraphStyle(
            "EduSectionTitle",
            fontName=FONT_BOLD,
            fontSize=11.5,
            leading=14,
            textColor=colors.HexColor("#1e293b"),
        )

        self.style_time_badge = ParagraphStyle(
            "EduTimeBadge",
            fontName=FONT_BOLD,
            fontSize=8.5,
            leading=11,
            textColor=colors.HexColor("#0369a1"),
            alignment=2,  # Sağa yasla
        )

        self.style_box_label = ParagraphStyle(
            "EduBoxLabel",
            fontName=FONT_BOLD,
            fontSize=8,
            leading=10,
            textColor=colors.HexColor("#334155"),
            spaceAfter=2,
        )

        self.style_transcript = ParagraphStyle(
            "EduTranscript",
            fontName=FONT_ITALIC,
            fontSize=8,
            leading=11.5,
            textColor=colors.HexColor("#475569"),
        )

        self.style_pedagogical = ParagraphStyle(
            "EduPedagogical",
            fontName=FONT_REG,
            fontSize=8.5,
            leading=12.5,
            textColor=colors.HexColor("#1e3a8a"),
        )

        self.style_concept_body = ParagraphStyle(
            "EduConceptBody",
            fontName=FONT_REG,
            fontSize=8,
            leading=11,
            textColor=colors.HexColor("#166534"),
        )

        self.style_ref_tag = ParagraphStyle(
            "EduRefTag",
            fontName=FONT_ITALIC,
            fontSize=6.8,
            leading=8.8,
            textColor=colors.HexColor("#475569"),
        )

        self.style_general_summary = ParagraphStyle(
            "EduGenSummary",
            fontName=FONT_REG,
            fontSize=9.5,
            leading=14.5,
            textColor=colors.HexColor("#1e293b"),
        )

        # Soru Kart Stilleri (Modern & Ferah)
        self.style_card_header = ParagraphStyle(
            "EduCardHeader",
            fontName=FONT_BOLD,
            fontSize=9,
            leading=12,
            textColor=colors.HexColor("#0f172a"),
        )

        self.style_card_qtext = ParagraphStyle(
            "EduCardQText",
            fontName=FONT_REG,
            fontSize=8.5,
            leading=12.5,
            textColor=colors.HexColor("#1e293b"),
        )

        self.style_card_options = ParagraphStyle(
            "EduCardOptions",
            fontName=FONT_REG,
            fontSize=8,
            leading=11.5,
            textColor=colors.HexColor("#334155"),
        )

        self.style_table_header = ParagraphStyle(
            "EduTblHdr",
            fontName=FONT_BOLD,
            fontSize=7.5,
            leading=9.5,
            textColor=colors.white,
            alignment=1,
        )

        self.style_table_cell = ParagraphStyle(
            "EduTblCell",
            fontName=FONT_REG,
            fontSize=6.8,
            leading=8.8,
            textColor=colors.HexColor("#1e293b"),
        )

        self.style_table_cell_bold = ParagraphStyle(
            "EduTblCellBold",
            fontName=FONT_BOLD,
            fontSize=6.8,
            leading=8.8,
            textColor=colors.HexColor("#1e293b"),
            alignment=1,
        )

    def _kutu_olustur(self, baslik: str, paragraf: Paragraph, bg_color: str, border_color: str) -> Table:
        """İçerik kutusu oluşturur (Transkript, Konu Anlatımı vb. için)."""
        icerik = [
            [Paragraph(f"<b>{baslik}</b>", self.style_box_label)],
            [paragraf]
        ]
        t = Table(icerik, colWidths=[511], splitInRow=1)
        t.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor(bg_color)),
            ("BOX", (0, 0), (-1, -1), 0.75, colors.HexColor(border_color)),
            ("TOPPADDING", (0, 0), (-1, -1), 4),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ("LEFTPADDING", (0, 0), (-1, -1), 7),
            ("RIGHTPADDING", (0, 0), (-1, -1), 7),
        ]))
        return t

    def _modern_soru_karti(self, soru_dict: Dict[str, Any]) -> Table:
        """
        Her değerlendirme sorusunu modern, ferah ve sol aksanlı kart kutusu olarak üretir.
        """
        no = soru_dict.get("no", 1)
        kategori = soru_dict.get("kategori", "💡 Soru")
        kavram = soru_dict.get("ilgili_kavram", "Genel Kavram")
        tip = soru_dict.get("tip", "Çoktan Seçmeli")
        soru_metni = soru_dict.get("soru", "")
        secenekler = soru_dict.get("secenekler", [])

        # Üst Başlık Şeridi
        ust_metin = (
            f"<b>SORU {no}</b> &nbsp;&nbsp;|&nbsp;&nbsp; "
            f"<font color='#0284c7'><b>{kategori}</b></font> &nbsp;&nbsp;|&nbsp;&nbsp; "
            f"<i>{tip}</i> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; "
            f"<font color='#059669'><b>Hedef Kavram:</b> {kavram}</font>"
        )

        hucreler = [
            [Paragraph(ust_metin, self.style_card_header)],
            [Spacer(1, 2)],
            [Paragraph(soru_metni, self.style_card_qtext)],
        ]

        if secenekler and len(secenekler) > 2:
            sec_metin = "<br/>".join([f"&nbsp;&nbsp;&nbsp;&nbsp;<b>{opt[:2]}</b> {opt[2:].strip()}" for opt in secenekler])
            hucreler.append([Spacer(1, 2)])
            hucreler.append([Paragraph(sec_metin, self.style_card_options)])
        elif secenekler and len(secenekler) == 2:
            sec_metin = f"&nbsp;&nbsp;&nbsp;&nbsp;[ &nbsp;&nbsp; ] <b>Doğru</b> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; [ &nbsp;&nbsp; ] <b>Yanlış</b>"
            hucreler.append([Spacer(1, 2)])
            hucreler.append([Paragraph(sec_metin, self.style_card_options)])
        else:
            hucreler.append([Spacer(1, 2)])
            hucreler.append([Paragraph("<b>Cevap Alanı:</b> .....................................................................................................................", self.style_card_options)])

        t = Table(hucreler, colWidths=[511])
        t.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#f8fafc")),
            ("BOX", (0, 0), (-1, -1), 0.75, colors.HexColor("#e2e8f0")),
            ("LINEBEFORE", (0, 0), (0, -1), 3.0, colors.HexColor("#0284c7")),
            ("TOPPADDING", (0, 0), (-1, -1), 4),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ("LEFTPADDING", (0, 0), (-1, -1), 8),
            ("RIGHTPADDING", (0, 0), (-1, -1), 8),
        ]))
        return t

    @staticmethod
    def _temizle_dosya_yolu(metin: str) -> str:
        """
        Windows veya Unix dosya yollarını (örn: C:\\Users\\...\\video.mp4) temizleyerek
        PDF başlıklarında yalnızca temiz, kullanıcı dostu ders adını gösterir.
        """
        if not metin:
            return ""
        metin_str = str(metin).strip()
        # Eğer Windows yolu (C:\... veya C:/...) veya slash içeriyorsa
        if re.search(r'^[a-zA-Z]:[\\/]', metin_str) or ("\\" in metin_str or "/" in metin_str):
            p = Path(metin_str)
            temiz = p.stem if p.suffix else p.name
            return temiz.replace("_", " ").strip()
        return metin_str

    def _dinamik_kaynakca_derle(
        self,
        video_analizi: Dict[str, Any],
        sorular: Optional[List[Dict[str, Any]]] = None,
    ) -> List[str]:
        """
        PDF için yalnızca kullanılan ve doğrulanmış RAG kayıtlarından oluşan dinamik kaynakça listesi derler.
        Aynı kaynağa yapılan atıfları birleştirir (deduplicate).
        Her kaynak için:
          - Kitap adı ve dosya adı
          - Sınıf düzeyi
          - Ünite ve konu
          - Fiziksel sayfa
          - Basılı sayfa (mevcutsa)
          - Kullanıldığı bölüm veya soru
        Kesinlikle uydurma kazanım kodu veya sayfa numarası eklemez.
        Doğrulanmış kaynak bulunamadığında açık fallback mesajı gösterir.
        """
        bolumler = video_analizi.get("bolumler", []) if isinstance(video_analizi, dict) else []
        sorular_list = sorular or []

        # Key: benzersiz kaynak kimliği -> {bilgiler, kullanildigi_yerler: set()}
        kaynak_havuzu: Dict[str, Dict[str, Any]] = {}

        # 1. Bölümlerden doğrulanmış kaynakları topla
        for idx, b in enumerate(bolumler, start=1):
            b_no = b.get("bolum_no", idx)
            yer = f"Bölüm {b_no}"
            b_ref = b.get("kitap_referansi")
            rag_list = b.get("rag_kaynaklari") or []

            if rag_list:
                for r in rag_list:
                    k_adi = r.get("kitap_adi") or r.get("kaynak") or "MEB Fen Bilimleri Ders Kitabı"
                    dosya = r.get("dosya_adi") or r.get("kaynak") or ""
                    sinif = r.get("sinif") or video_analizi.get("sinif_duzeyi", "")
                    unite = r.get("unite", "")
                    konu = r.get("konu", "")
                    phys_page = r.get("physical_page") if r.get("physical_page") is not None else r.get("sayfa")
                    prn_page = r.get("printed_page") if r.get("has_printed_page") else None

                    key = f"{k_adi}_{dosya}_{phys_page}_{unite}_{konu}".strip().lower()
                    kaynak_rolu = r.get("kaynak_rolu") or "ana_kaynak"
                    if key not in kaynak_havuzu:
                        kaynak_havuzu[key] = {
                            "kitap_adi": k_adi,
                            "dosya_adi": dosya,
                            "sinif": sinif,
                            "unite": unite,
                            "konu": konu,
                            "physical_page": phys_page,
                            "printed_page": prn_page,
                            "kaynak_rolu": kaynak_rolu,
                            "kullanildigi_yerler": set(),
                        }
                    kaynak_havuzu[key]["kullanildigi_yerler"].add(yer)
            elif b_ref and b_ref.strip().lower() not in [
                "meb ders kitabı referansı",
                "doğrulanmış rag eşleşmesi bulunamadı",
                "doğrulanmış ders kitabı kaynağı bulunamadı",
                "meb 3. sınıf fen bilimleri"
            ]:
                key = b_ref.strip().lower()
                if key not in kaynak_havuzu:
                    kaynak_havuzu[key] = {
                        "ham_referans": b_ref,
                        "kaynak_rolu": "ana_kaynak",
                        "kullanildigi_yerler": set(),
                    }
                kaynak_havuzu[key]["kullanildigi_yerler"].add(yer)

        # 2. Sorulardan doğrulanmış kaynakları topla
        for s in sorular_list:
            s_no = s.get("no", 1)
            yer = f"Soru {s_no}"
            s_ref = s.get("kazanim_referansi")
            s_rag = s.get("rag_kaynagi")

            if s_rag:
                k_adi = s_rag.get("kitap_adi") or s_rag.get("kaynak") or "MEB Fen Bilimleri Ders Kitabı"
                dosya = s_rag.get("dosya_adi") or s_rag.get("kaynak") or ""
                sinif = s_rag.get("sinif") or video_analizi.get("sinif_duzeyi", "")
                unite = s_rag.get("unite", "")
                konu = s_rag.get("konu", "")
                phys_page = s_rag.get("physical_page") if s_rag.get("physical_page") is not None else s_rag.get("sayfa")
                prn_page = s_rag.get("printed_page") if s_rag.get("has_printed_page") else None
                kaynak_rolu = s_rag.get("kaynak_rolu") or "ana_kaynak"

                key = f"{k_adi}_{dosya}_{phys_page}_{unite}_{konu}".strip().lower()
                if key not in kaynak_havuzu:
                    kaynak_havuzu[key] = {
                        "kitap_adi": k_adi,
                        "dosya_adi": dosya,
                        "sinif": sinif,
                        "unite": unite,
                        "konu": konu,
                        "physical_page": phys_page,
                        "printed_page": prn_page,
                        "kaynak_rolu": kaynak_rolu,
                        "kullanildigi_yerler": set(),
                    }
                kaynak_havuzu[key]["kullanildigi_yerler"].add(yer)
            elif s_ref and s_ref.strip().lower() not in [
                "meb ders kitabı referansı",
                "doğrulanmış rag eşleşmesi bulunamadı",
                "doğrulanmış ders kitabı kaynağı bulunamadı",
                "meb 3. sınıf fen bilimleri"
            ]:
                key = s_ref.strip().lower()
                if key not in kaynak_havuzu:
                    kaynak_havuzu[key] = {
                        "ham_referans": s_ref,
                        "kaynak_rolu": "ana_kaynak",
                        "kullanildigi_yerler": set(),
                    }
                kaynak_havuzu[key]["kullanildigi_yerler"].add(yer)

        # 3. Eğer hiç doğrulanmış kaynak bulunamadıysa:
        if not kaynak_havuzu:
            return ["<i>Bu bölüm için doğrulanmış ders kitabı kaynağı bulunamadı.</i>"]

        # 4. Kaynakları formatla ve sırala (Ana kaynaklar önce)
        havuz_listesi = list(kaynak_havuzu.values())
        havuz_listesi.sort(key=lambda x: (0 if x.get("kaynak_rolu") == "ana_kaynak" else 1, x.get("printed_page") or 999))

        cikti_kaynaklar = []
        for idx, info in enumerate(havuz_listesi, 1):
            yerler_sirali = sorted(
                list(info["kullanildigi_yerler"]),
                key=lambda x: (x.split()[0], int(x.split()[1]) if len(x.split()) > 1 and x.split()[1].isdigit() else 0)
            )
            yerler_str = ", ".join(yerler_sirali)
            rol_badge = " [Ana Kaynak]" if info.get("kaynak_rolu") == "ana_kaynak" else (" [Yardımcı Kaynak]" if info.get("kaynak_rolu") == "yardimci_kaynak" else "")

            if "ham_referans" in info:
                cikti_kaynaklar.append(
                    f"<b>{idx}.</b> {info['ham_referans']}{rol_badge} — <i>(Kullanıldığı Yerler: {yerler_str})</i>"
                )
            else:
                k_ad = info.get("kitap_adi") or "MEB Ders Kitabı"
                d_ad = f" (Dosya: {info['dosya_adi']})" if info.get("dosya_adi") else ""
                sinif_str = f" | {info['sinif']}" if info.get("sinif") else ""

                unite_parcalari = []
                if info.get("unite"):
                    unite_parcalari.append(f"Ünite: {info['unite']}")
                if info.get("konu"):
                    unite_parcalari.append(f"Konu: {info['konu']}")
                unite_str = f" — {', '.join(unite_parcalari)}" if unite_parcalari else ""

                sayfa_parcalari = []
                if info.get("physical_page") is not None:
                    sayfa_parcalari.append(f"Fiziksel Sayfa: {info['physical_page']}")
                if info.get("printed_page") is not None and info["printed_page"] > 0:
                    sayfa_parcalari.append(f"Basılı Sayfa: {info['printed_page']}")
                sayfa_str = f" [{', '.join(sayfa_parcalari)}]" if sayfa_parcalari else ""

                cikti_kaynaklar.append(
                    f"<b>{idx}. {k_ad}</b>{d_ad}{sinif_str}{unite_str}{sayfa_str}{rol_badge} — <i>(Kullanıldığı Yerler: {yerler_str})</i>"
                )

        return cikti_kaynaklar

    @staticmethod
    def ders_karelerini_dogrula(bolumler: List[Dict[str, Any]]) -> None:
        if not bolumler:
            raise EksikDersKaresiHatasi("PDF üretilemedi. Görseli olan ders bölümü bulunamadı.")

        eksik_kareler = []
        for idx, bolum in enumerate(bolumler, 1):
            kare_yolu = bolum.get("ekran_goruntusu_yolu")
            try:
                if not kare_yolu:
                    raise FileNotFoundError()
                with PILImage.open(kare_yolu) as kare:
                    kare.load()
                    if min(kare.size) <= 1:
                        raise ValueError("Yer tutucu görsel")
            except (OSError, ValueError):
                baslik = bolum.get("bolum_basligi") or f"Bölüm {idx}"
                zaman = bolum.get("zaman_araligi_metni") or "zaman aralığı bilinmiyor"
                eksik_kareler.append(f"{baslik} ({zaman})")
        if eksik_kareler:
            raise EksikDersKaresiHatasi(
                "PDF üretilemedi. Okunabilir ekran görüntüsü eksik olan bölümler\n"
                + "\n".join(f"- {bolum}" for bolum in eksik_kareler)
            )

    def pdf_olustur(
        self,
        video_analizi: Dict[str, Any],
        sorular: Optional[List[Dict[str, Any]]] = None,
        video_kaynak_bilgisi: str = "Video Anlatımı",
        cikis_dosya_adi: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Zaman damgalı görsel materyal bölümlerinden PDF ders fasikülü üretir.
        """
        bolumler = video_analizi.get("bolumler") or []
        self.ders_karelerini_dogrula(bolumler)

        ders_adi = (video_analizi.get("ders_adi") or "FEN BİLİMLERİ").upper()
        ham_unite = video_analizi.get("unite_basligi") or "Elektrikli Araçlar ve Elektrik Kaynakları"
        unite_basligi = self._temizle_dosya_yolu(ham_unite)
        temiz_video_kaynak = self._temizle_dosya_yolu(video_kaynak_bilgisi)
        egitim_seviyesi = video_analizi.get("egitim_seviyesi", "İlkokul")
        sinif_duzeyi = video_analizi.get("sinif_duzeyi", "3. Sınıf")
        tarih_str = datetime.now().strftime("%d.%m.%Y")

        # Provenance / Üretim Kaynağı Rozeti
        uretim_kaynagi = video_analizi.get("uretim_kaynagi", "rag_fallback")
        if uretim_kaynagi == "ollama":
            rozet = "<font color='#0284c7'><b>[Yerel Yapay Zeka (Ollama qwen3.5:4b)]</b></font>"
        elif uretim_kaynagi == "gemini":
            rozet = "<font color='#16a34a'><b>[Google Gemini API]</b></font>"
        elif uretim_kaynagi == "transkript":
            rozet = "Whisper transkripti"
        elif uretim_kaynagi == "karma":
            rozet = "Video ve bölüm analizi"
        elif uretim_kaynagi == "rag_fallback":
            rozet = "<font color='#d97706'><b>[Pedagojik RAG Bağlamı]</b></font>"
        else:
            rozet = "<font color='#64748b'><b>[MEB Müfredat Şablonu]</b></font>"

        guvenli_ad = re.sub(r'[\\/*?:"<>| ]', '_', f"{ders_adi}_{unite_basligi}")[:35]
        if not cikis_dosya_adi:
            cikis_dosya_adi = f"EduFrame_{guvenli_ad}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"

        pdf_hedef = self.cikti_dizini / cikis_dosya_adi
        print(f"\n[PDFGenerator] Gerçek PDF Dosyası Basılıyor: {pdf_hedef.name}...")

        doc = SimpleDocTemplate(
            str(pdf_hedef.resolve()),
            pagesize=A4,
            leftMargin=42,
            rightMargin=42,
            topMargin=45,
            bottomMargin=45,
        )

        flowables = []

        # ===================================================================
        # SAYFA 1 - 10: HER GÖRSEL MATERYAL İÇİN BÖLÜM SAYFASI
        # ===================================================================
        for idx, bolum in enumerate(bolumler, start=1):
            if idx > 1:
                flowables.append(PageBreak())

            # Kapak Başlığı sadece 1. Sayfada görünür
            if idx == 1:
                flowables.append(Paragraph(f"{ders_adi} – DERS NOTU FASİKÜLÜ", self.style_super_title))
                flowables.append(Paragraph(escape(unite_basligi), self.style_sub_title))
                meta_metni = (
                    f"<b>Kademe:</b> {egitim_seviyesi} ({sinif_duzeyi}) &nbsp;&nbsp;|&nbsp;&nbsp; "
                    f"<b>Tarih:</b> {tarih_str} &nbsp;&nbsp;|&nbsp;&nbsp; "
                    f"<b>Kaynak:</b> {escape(temiz_video_kaynak[:45])} &nbsp;&nbsp;|&nbsp;&nbsp; {rozet}"
                )
                flowables.append(Paragraph(meta_metni, self.style_meta))
                flowables.append(Spacer(1, 4))
                flowables.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor("#2563eb"), spaceAfter=8))

            b_baslik = bolum.get("bolum_basligi", f"Bölüm {idx} – Eğitim Materyali")
            z_aralik = bolum.get("zaman_araligi_metni") or "00:00 – 00:00"

            # Bölüm Başlığı & Zaman Aralığı Tablosu
            baslik_tablosu = Table(
                [[Paragraph(f"<b>{escape(b_baslik)}</b>", self.style_section_title),
                  Paragraph(f"<b>Zaman Aralığı: {z_aralik}</b>", self.style_time_badge)]],
                colWidths=[360, 151]
            )
            baslik_tablosu.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#f1f5f9")),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
                ("TOPPADDING", (0, 0), (-1, -1), 3),
                ("LEFTPADDING", (0, 0), (-1, -1), 6),
                ("RIGHTPADDING", (0, 0), (-1, -1), 6),
            ]))
            flowables.append(baslik_tablosu)
            flowables.append(Spacer(1, 5))

            img = Image(bolum["ekran_goruntusu_yolu"], width=470, height=260, kind="proportional")
            img.hAlign = "CENTER"
            flowables.append(img)
            flowables.append(Spacer(1, 5))

            ham_transkript = bolum.get("transkript") or "Bu aralıkta konuşma bulunamadı."
            konusma_zamani = bolum.get("konusma_zaman_araligi")
            zaman_etiketi = f" {escape(konusma_zamani)}" if konusma_zamani else ""
            t_para = Paragraph(escape(ham_transkript), self.style_transcript)
            t_box = self._kutu_olustur(f"Orijinal Transkript{zaman_etiketi}", t_para, "#f8fafc", "#cbd5e1")
            flowables.append(t_box)
            flowables.append(Spacer(1, 5))

            duzenli_anlatim = bolum.get("pedagojik_anlatim") or bolum.get("duzenlenmis_anlatim", "")
            if duzenli_anlatim:
                anlatim_html = escape(duzenli_anlatim).replace("\n", "<br/>")
                if bolum.get("kitap_ek_bilgisi"):
                    ek = escape(bolum["kitap_ek_bilgisi"])
                    anlatim_html += f"<br/><br/><b>Kitaptan ek bilgi:</b> {ek}"
                d_para = Paragraph(anlatim_html, self.style_pedagogical)
                baslik = "Düzenlenmiş Ders Anlatımı"
                if bolum.get("kaynak_tipi") == "transkript":
                    baslik = "Ders Anlatımı (Transkript)"
                d_box = self._kutu_olustur(baslik, d_para, "#eff6ff", "#93c5fd")
                flowables.append(d_box)
                flowables.append(Spacer(1, 5))

            # Önemli Kavramlar Kutusu
            kavramlar = bolum.get("onemli_kavramlar", [])
            if kavramlar:
                kavram_metinleri = []
                for k in kavramlar:
                    k_ad = k.get("kavram", "")
                    k_aciklama = k.get("aciklama", "")
                    kavram_metinleri.append(f"• <b>{escape(str(k_ad))}:</b> {escape(str(k_aciklama))}")
                k_para = Paragraph("<br/>".join(kavram_metinleri), self.style_concept_body)
                k_box = self._kutu_olustur("3. Bölümün Önemli Kavramları:", k_para, "#f0fdf4", "#86efac")
                flowables.append(k_box)
                flowables.append(Spacer(1, 4))

            # Kitap Sayfa Referansı
            kitap_ref = bolum.get("kitap_referansi", "")
            if kitap_ref and kitap_ref != "MEB Ders Kitabı Referansı":
                flowables.append(Paragraph(f"<b>MEB Ders Kitabı Doğrulaması:</b> <i>{escape(kitap_ref)}</i>", self.style_ref_tag))

        # ===================================================================
        # Bölüm verisi bir genel özet içeriyorsa ayrı sayfada göster.
        genel_ozet = video_analizi.get("genel_ozet")
        if genel_ozet:
            flowables.append(PageBreak())
            flowables.append(Paragraph("Dersin Genel Özeti", self.style_super_title))
            ozet_para = Paragraph(escape(genel_ozet), self.style_general_summary)
            flowables.append(self._kutu_olustur("Genel Ünite Özeti", ozet_para, "#f8fafc", "#94a3b8"))

        # ===================================================================
        # SAYFA 12 & 13: ÖĞRENCİ ETKİNLİK VE DEĞERLENDİRME FASİKÜLÜ (10 SORU)
        # ===================================================================
        if sorular and len(sorular) > 0:
            # 1. BÖLÜM: Soru 1 – 5 (Sayfa 12)
            flowables.append(PageBreak())
            flowables.append(Paragraph("<b>ÖĞRENCİ ETKİNLİK VE DEĞERLENDİRME FASİKÜLÜ (1 / 2)</b>", self.style_super_title))
            flowables.append(Paragraph(f"{ders_adi} &nbsp;|&nbsp; Konu: {unite_basligi} &nbsp;|&nbsp; Soru 1 – 5 &nbsp;|&nbsp; {rozet}", self.style_meta))
            flowables.append(Spacer(1, 3))
            flowables.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor("#059669"), spaceAfter=8))

            sorular_part1 = sorular[:5]
            for s in sorular_part1:
                kart = self._modern_soru_karti(s)
                flowables.append(kart)
                flowables.append(Spacer(1, 5))

            # 2. BÖLÜM: Soru 6 – 10 (Sayfa 13)
            if len(sorular) > 5:
                flowables.append(PageBreak())
                flowables.append(Paragraph("<b>ÖĞRENCİ ETKİNLİK VE DEĞERLENDİRME FASİKÜLÜ (2 / 2)</b>", self.style_super_title))
                flowables.append(Paragraph(f"{ders_adi} &nbsp;|&nbsp; Konu: {unite_basligi} &nbsp;|&nbsp; Soru 6 – 10 &nbsp;|&nbsp; {rozet}", self.style_meta))
                flowables.append(Spacer(1, 3))
                flowables.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor("#059669"), spaceAfter=8))

                sorular_part2 = sorular[5:10]
                for s in sorular_part2:
                    kart = self._modern_soru_karti(s)
                    flowables.append(kart)
                    flowables.append(Spacer(1, 5))

            # ===============================================================
            # SAYFA 14: DETAYLI ÇÖZÜMLÜ CEVAP ANAHTARI & MEB KAYNAKÇA TABLOSU
            # ===============================================================
            flowables.append(PageBreak())
            flowables.append(Paragraph("<b>DETAYLI ÇÖZÜMLÜ CEVAP ANAHTARI & KAZANIM EŞLEŞTİRME TABLOSU</b>", self.style_super_title))
            flowables.append(Paragraph(f"{ders_adi} &nbsp;|&nbsp; {unite_basligi} &nbsp;|&nbsp; Doğrulanmış Kaynakça &nbsp;|&nbsp; {rozet}", self.style_meta))
            flowables.append(Spacer(1, 3))
            flowables.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor("#2563eb"), spaceAfter=8))

            cevap_tablosu_verisi = [
                [
                    Paragraph("<b>Soru No</b>", self.style_table_header),
                    Paragraph("<b>Öğrenilen Kavram</b>", self.style_table_header),
                    Paragraph("<b>Doğru Yanıt</b>", self.style_table_header),
                    Paragraph("<b>Ders Kitabı Referansı ve Çözüm Açıklaması</b>", self.style_table_header),
                ]
            ]

            for s in sorular:
                s_no = s.get("no", 1)
                s_kavram = s.get("ilgili_kavram", "-")
                s_cevap = s.get("dogru_cevap", "-")
                s_aciklama = s.get("aciklama", "")
                s_ref = s.get("kazanim_referansi", "")

                if s_ref and s_ref not in ["Doğrulanmış ders kitabı kaynağı bulunamadı", "MEB Ders Kitabı"]:
                    hucre_metni = f"<b>Kaynak:</b> {s_ref}<br/><b>Çözüm Gerekçesi:</b> {s_aciklama}"
                else:
                    hucre_metni = f"<font color='#dc2626'><i>Kaynak: Doğrulanmış ders kitabı kaynağı bulunamadı</i></font><br/><b>Çözüm Gerekçesi:</b> {s_aciklama}"

                cevap_tablosu_verisi.append([
                    Paragraph(f"<b>Soru {s_no}</b>", self.style_table_cell_bold),
                    Paragraph(s_kavram, self.style_table_cell),
                    Paragraph(f"<b>{s_cevap}</b>", self.style_table_cell_bold),
                    Paragraph(hucre_metni, self.style_table_cell),
                ])

            ans_table = Table(cevap_tablosu_verisi, colWidths=[42, 105, 48, 316])
            ans_table.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1e3a8a")),
                ("BOX", (0, 0), (-1, -1), 0.75, colors.HexColor("#cbd5e1")),
                ("INNERGRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#e2e8f0")),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.HexColor("#ffffff"), colors.HexColor("#f8fafc")]),
                ("TOPPADDING", (0, 0), (-1, -1), 1.5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 1.5),
                ("LEFTPADDING", (0, 0), (-1, -1), 4),
                ("RIGHTPADDING", (0, 0), (-1, -1), 4),
            ]))
            flowables.append(ans_table)
            flowables.append(Spacer(1, 4))

            # Dinamik ve Doğrulanmış Kaynaklar Bölümü (Statik Kaynakça Tamamen Kaldırıldı)
            dinamik_kaynaklar = self._dinamik_kaynakca_derle(video_analizi, sorular)
            kaynak_metni = "<br/>".join(dinamik_kaynaklar)

            kaynak_kutusu = [
                [Paragraph("<b>Yararlanılan Resmi Müfredat Kaynakları (Doğrulanmış RAG Kaynakçası):</b>", self.style_box_label)],
                [Paragraph(kaynak_metni, self.style_ref_tag)]
            ]
            k_table = Table(kaynak_kutusu, colWidths=[511])
            k_table.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#f1f5f9")),
                ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
                ("TOPPADDING", (0, 0), (-1, -1), 2),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
                ("LEFTPADDING", (0, 0), (-1, -1), 6),
                ("RIGHTPADDING", (0, 0), (-1, -1), 6),
            ]))
            flowables.append(k_table)

        # PDF'i derle
        doc.build(flowables, canvasmaker=NumberedCanvas)
        print(f"[BAŞARILI] Kurumsal PDF Fasikülü üretildi: {pdf_hedef.resolve()}")

        return {
            "success": True,
            "pdf_path": str(pdf_hedef.resolve()),
            "html_path": None,
            "mesaj": "Gerçek kurumsal PDF ders notu başarıyla üretildi."
        }


if __name__ == "__main__":
    from services.gemini_analyzer import GeminiAnalyzer
    from services.question_generator import QuestionGenerator

    print("PDF Generator Test Başlatılıyor...")
    gen = GeminiAnalyzer()
    analiz = gen._orijinal_materyal_simulasyonu("İlkokul", "3. Sınıf", "Fen Bilimleri", "Elektrik Güvenliği")
    
    qgen = QuestionGenerator()
    sorular = qgen.soru_uret(analiz, egitim_seviyesi="İlkokul", soru_sayisi=10)

    pdf_servisi = PDFGenerator()
    sonuc = pdf_servisi.pdf_olustur(analiz, sorular, video_kaynak_bilgisi="YouTube: Fen Bilimleri Dersi")
    print("Test Çıktısı:", sonuc)
