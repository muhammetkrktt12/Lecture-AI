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
import json
import sys
import re
from datetime import datetime
from functools import partial
from pathlib import Path
from random import choice
from tempfile import NamedTemporaryFile
from typing import Dict, Any, List, Optional
from xml.sax.saxutils import escape

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from utils.lexical_tools import kavram_anahtari
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.styles import ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Image,
    Table,
    TableStyle,
    KeepTogether,
    Flowable,
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
_CERCEVE_DIZINI = ROOT / "templates" / "cerceveler"
_SABLON_YAN_BOSLUK = 65
_SABLON_UST_BOSLUK = 80
_SABLON_ALT_BOSLUK = 62


# ---------------------------------------------------------------------------
# TÜRKÇE FONT YÖNETİMİ
# ---------------------------------------------------------------------------
def _turkce_fontlari_yukle():
    """Örnekteki Calibri ve Aptos yüzlerini yerel fontlardan yükler."""
    try:
        font_dir = Path(os.environ.get("WINDIR", "C:/Windows")) / "Fonts"
        normal, kalin, italik = "EduCalibri", "EduCalibri-Bold", "EduCalibri-Italic"
        for ad, dosya in zip((normal, kalin, italik), ("calibri.ttf", "calibrib.ttf", "calibrii.ttf")):
            pdfmetrics.registerFont(TTFont(ad, str(font_dir / dosya)))
        pdfmetrics.registerFontFamily(normal, normal=normal, bold=kalin, italic=italik, boldItalic=kalin)
    except Exception as e:
        print(f"[UYARI] Calibri fontu yüklenemedi: {e}")
        normal, kalin, italik = "Helvetica", "Helvetica-Bold", "Helvetica-Oblique"

    aptos = {b"Aptos": normal, b"Aptos-Bold": kalin, b"Aptos-Italic": italik}
    ofis_fontlari = Path(os.environ.get("LOCALAPPDATA", str(Path.home() / "AppData/Local")))
    ofis_fontlari = ofis_fontlari / "Microsoft/FontCache/4/CloudFonts/Aptos"
    for yol in (*font_dir.glob("aptos*.ttf"), *ofis_fontlari.glob("*.ttf")):
        try:
            font = TTFont(f"EduFont-{yol.stem}", str(yol))
            if font.face.name in aptos:
                pdfmetrics.registerFont(font)
                aptos[font.face.name] = font.fontName
        except (OSError, ValueError):
            continue
    return normal, kalin, aptos[b"Aptos"], aptos[b"Aptos-Bold"], aptos[b"Aptos-Italic"]


FONT_REG, FONT_BOLD, FONT_APTOS, FONT_APTOS_BOLD, FONT_APTOS_ITALIC = _turkce_fontlari_yukle()


class _SayfaKaynagi(Flowable):
    def __init__(self, metin=""):
        super().__init__()
        self.metin = metin

    def draw(self):
        self.canv._ders_kitabi_referansi = self.metin


class _DersKutusu(Flowable):
    """Başlık ve metni ayrı konumlandırır, yalnızca metni sayfalara böler."""

    def __init__(self, baslik, metin, dolgu, kenar, alt_bosluk):
        super().__init__()
        self.baslik, self.metin = baslik, metin
        self.dolgu, self.kenar, self.alt_bosluk = dolgu, kenar, alt_bosluk

    def wrap(self, availWidth, availHeight):
        self.width = availWidth
        icerik = availWidth - 16
        self._baslik_yuksekligi = self.baslik.wrap(icerik, availHeight)[1] if self.baslik else 0
        self._metin_ustu = 6 + self._baslik_yuksekligi + (8 if self.baslik else 0)
        self.height = self._metin_ustu + self.metin.wrap(icerik, availHeight)[1] + self.alt_bosluk
        return self.width, self.height

    def split(self, availWidth, availHeight):
        self.wrap(availWidth, availHeight)
        pay = availHeight - self._metin_ustu - self.alt_bosluk
        if pay < 2 * self.metin.style.leading:
            return []
        parcalar = self.metin.split(availWidth - 16, pay)
        if len(parcalar) < 2:
            return []
        return [_DersKutusu(self.baslik if i == 0 else None, p, self.dolgu, self.kenar, self.alt_bosluk)
                for i, p in enumerate(parcalar)]

    def draw(self):
        self.canv.setFillColor(self.dolgu)
        if self.kenar:
            self.canv.setStrokeColor(self.kenar)
            self.canv.setLineWidth(0.65)
        self.canv.rect(0, 0, self.width, self.height, fill=1, stroke=int(bool(self.kenar)))
        if self.baslik:
            self.baslik.drawOn(self.canv, 8, self.height - 6 - self._baslik_yuksekligi)
        self.metin.drawOn(self.canv, 8, self.alt_bosluk)


# ---------------------------------------------------------------------------
# SAYFA NUMARALANDIRMA (SAYFA X / Y)
# ---------------------------------------------------------------------------
class _KavramKutusu(Flowable):
    """Yer kalmazsa kavramları sonraki ders bölümünün kutusuna bırakır."""

    _ZEROSIZE = True

    def __init__(self, bolum_no, kavramlar, bekleyenler, baslik_stili, metin_stili, son_bolum):
        super().__init__()
        self.bolum_no, self.kavramlar, self.bekleyenler = bolum_no, kavramlar, bekleyenler
        self.baslik_stili, self.metin_stili = baslik_stili, metin_stili
        self.son_bolum = son_bolum

    def wrap(self, availWidth, availHeight):
        self._ogeler = self.bekleyenler + self.kavramlar
        satirlar = [metin if no == self.bolum_no else f'{metin} <font size="10">({no}. bölüm)</font>'
                    for no, metin in self._ogeler]
        metin = "<br/>".join(satirlar) or "<b>Bu bölümde önemli bir kavram bulunmamaktadır.</b>"
        kutu = _DersKutusu(Paragraph("Önemli Kavramlar", self.baslik_stili), Paragraph(metin, self.metin_stili),
                          colors.HexColor("#D9F2D0"), None, 6 if self.bolum_no == 1 else 15)
        self._aralik = 7 if self.bolum_no == 1 else 20
        genislik, yukseklik = kutu.wrap(availWidth, availHeight - self._aralik)
        if yukseklik + self._aralik <= availHeight or (self.son_bolum and self._ogeler):
            self._kutu = kutu
            return genislik, yukseklik + self._aralik
        self._kutu = None
        return availWidth, 0

    def split(self, availWidth, availHeight):
        self.wrap(availWidth, availHeight)
        return self._kutu.split(availWidth, availHeight) if self._kutu else []

    def draw(self):
        if self._kutu:
            self._kutu.drawOn(self.canv, 0, 0)
            self.bekleyenler.clear()
        else:
            self.bekleyenler[:] = self._ogeler


class NumberedCanvas(canvas.Canvas):
    """Her sayfanın altına toplam sayfa sayısıyla birlikte numaralandırma ekler."""

    def __init__(self, *args, cerceve=None, ust_kayma=0, alt_kayma=0, **kwargs):
        super().__init__(*args, **kwargs)
        self._saved_page_states = []
        self._cerceve = cerceve
        self._ust_kayma = ust_kayma
        self._alt_kayma = alt_kayma
        self._ders_kitabi_referansi = ""

    def cerceveyi_ciz(self):
        if self._cerceve and self._cerceve.stem == "1":
            self.drawImage(str(self._cerceve), 0, -19.7, 607.45, 873.66)
        elif self._cerceve:
            self.drawImage(str(self._cerceve), 0, 0, *self._pagesize, preserveAspectRatio=True, anchor="c")

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
        sol = _SABLON_YAN_BOSLUK + 6
        sag = self._pagesize[0] - sol
        self.setFont(FONT_APTOS, 11)
        self.setFillColor(colors.HexColor("#242424"))
        self.drawString(sol, self._pagesize[1] - 67.2 - self._ust_kayma, "Lectra AI – Otonom Ders Notu Oluşturucu")
        self.setLineWidth(0.45)
        self.setStrokeColor(colors.HexColor("#D5DDE2"))
        ust_cizgi = self._pagesize[1] - 77 - self._ust_kayma
        self.line(sol, ust_cizgi, sag, ust_cizgi)
        self.line(sol, 65.5 + self._alt_kayma, sag, 65.5 + self._alt_kayma)

        page_str = f"Sayfa {self._pageNumber} / {total_pages}"
        self.setFont(FONT_APTOS, 9)
        self.setFillColor(colors.HexColor("#52646B"))
        self.drawRightString(sag, 43 + self._alt_kayma, page_str)
        if self._ders_kitabi_referansi:
            kaynak = Paragraph(escape(self._ders_kitabi_referansi), ParagraphStyle(
                "EduFooterSource", fontName=FONT_APTOS_ITALIC, fontSize=9, leading=11.8,
                textColor=colors.HexColor("#242424"),
            ))
            _, yukseklik = kaynak.wrap(sag - sol, 28)
            kaynak.drawOn(self, sol, 63.5 + self._alt_kayma - yukseklik)

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
        self._icerik_genisligi = 511
        self._stilleri_hazirla()

    def _stilleri_hazirla(self):
        """Başlıklar, ana metin ve yardımcı bilgiler için ortak tipografi."""
        def stil(ad, punto, satir, renk="#26333D", font=FONT_REG, **ayarlar):
            return ParagraphStyle(ad, fontName=font, fontSize=punto, leading=satir,
                                  textColor=colors.HexColor(renk), **ayarlar)

        self.style_super_title = stil("EduTitle", 16, 20, "#000000", FONT_BOLD, keepWithNext=True)
        self.style_doc_title = ParagraphStyle("EduDocTitle", parent=self.style_super_title, alignment=1)
        self.style_section_title = stil("EduSection", 14, 18, "#000000", FONT_BOLD, keepWithNext=True)
        self.style_meta = stil("EduMeta", 10, 13, "#156082", FONT_APTOS_BOLD, keepWithNext=True)
        self.style_time = stil("EduTime", 11, 14, "#156082", FONT_BOLD, alignment=2)
        self.style_first_time = ParagraphStyle("EduTimeFirst", parent=self.style_time, fontSize=10)
        self.style_box_label = stil("EduLabel", 13, 17.2, "#000000", FONT_BOLD, keepWithNext=True)
        self.style_pedagogical = stil("EduBody", 13, 17.2, "#000000", allowWidows=0, allowOrphans=0)
        self.style_concept_body = stil("EduConcept", 13, 17.2, "#275317")
        self.style_ref_tag = stil("EduSource", 10.5, 14, "#52646B", spaceAfter=6)
        self.style_card_header = stil("EduQuestionHeader", 11, 15, "#356864", FONT_BOLD)
        self.style_card_qtext = stil("EduQuestion", 13, 17)
        self.style_card_options = stil("EduOptions", 13, 17)

    def _ders_kutusu(self, baslik, metin, dolgu, kenar=None, *, stil=None, alt_bosluk=14):
        return _DersKutusu(Paragraph(baslik, self.style_box_label),
                          Paragraph(metin, stil or self.style_pedagogical),
                          colors.HexColor(dolgu), colors.HexColor(kenar) if kenar else None, alt_bosluk)

    def _modern_soru_karti(self, soru: Dict[str, Any]) -> Table:
        no = escape(str(soru.get("no", 1)))
        kategori = escape(str(soru.get("kategori") or "Değerlendirme"))
        tip = escape(str(soru.get("tip") or "Çoktan Seçmeli"))
        kavram = escape(str(soru.get("ilgili_kavram") or ""))
        secenekler = soru.get("secenekler") or []
        icerik = [
            Paragraph(f"Soru {no} · {kategori} · {tip}", self.style_card_header),
        ]
        if kavram:
            icerik.append(Paragraph(f"Hedef kavram · {kavram}", self.style_ref_tag))
        icerik.append(Paragraph(escape(str(soru.get("soru") or "")), self.style_card_qtext))
        icerik.append(Spacer(1, 5))
        if len(secenekler) > 2:
            metin = "<br/>".join(escape(str(secenek)) for secenek in secenekler)
        elif len(secenekler) == 2:
            metin = "[ &nbsp; ] Doğru &nbsp;&nbsp;&nbsp;&nbsp; [ &nbsp; ] Yanlış"
        else:
            metin = "Cevap &nbsp; __________________________________"
        icerik.append(Paragraph(metin, self.style_card_options))
        kart = Table([[icerik]], colWidths=[self._icerik_genisligi], splitInRow=1)
        kart.setStyle(TableStyle([
            ("LINEABOVE", (0, 0), (-1, 0), 0.7, colors.HexColor("#B9D0CA")),
            ("TOPPADDING", (0, 0), (-1, -1), 9),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
            ("LEFTPADDING", (0, 0), (-1, -1), 0),
            ("RIGHTPADDING", (0, 0), (-1, -1), 0),
        ]))
        return kart

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
        *,
        icerigi_kaydet: bool = False,
        cerceve_adi: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Zaman damgalı görsel materyal bölümlerinden PDF ders fasikülü üretir.
        """
        bolumler = video_analizi.get("bolumler") or []
        self.ders_karelerini_dogrula(bolumler)

        ders_adi = (video_analizi.get("ders_adi") or "FEN BİLİMLERİ").upper()
        ham_unite = video_analizi.get("unite_basligi") or "Elektrikli Araçlar ve Elektrik Kaynakları"
        unite_basligi = self._temizle_dosya_yolu(ham_unite)
        sinif_duzeyi = video_analizi.get("sinif_duzeyi", "3. Sınıf")
        sinif_eslesmesi = re.fullmatch(r"\s*(\d+)\s*\.?\s*(?:s[ıi]n[ıi]f)?\s*", str(sinif_duzeyi).lower())
        sinif = int(sinif_eslesmesi.group(1)) if sinif_eslesmesi else None
        cerceveler = (
            sorted(p for p in _CERCEVE_DIZINI.iterdir() if p.is_file() and p.suffix.lower() in {".jpeg", ".jpg", ".png"})
            if sinif in range(1, 6) and _CERCEVE_DIZINI.is_dir()
            else []
        )
        cerceve = next((p for p in cerceveler if p.name == cerceve_adi), None)
        if cerceve is None and cerceveler:
            cerceve = choice(cerceveler)
        if cerceve and cerceve.stem == "4":
            ust_kayma, alt_kayma = 0, 9
        elif cerceve and cerceve.stem == "3":
            ust_kayma, alt_kayma = 28, 9
        elif cerceve and cerceve.stem == "7":
            ust_kayma, alt_kayma = 28, 35
        elif cerceve and cerceve.stem != "1":
            ust_kayma, alt_kayma = 28, 52
        else:
            ust_kayma, alt_kayma = 0, 0
        self._icerik_genisligi = A4[0] - 2 * _SABLON_YAN_BOSLUK - 12
        guvenli_ad = re.sub(r'[\\/*?:"<>| ]', '_', f"{ders_adi}_{unite_basligi}")[:35]
        if not cikis_dosya_adi:
            cikis_dosya_adi = f"LectraAI_{guvenli_ad}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"

        pdf_hedef = self.cikti_dizini / cikis_dosya_adi
        icerik_yolu = None
        if icerigi_kaydet:
            icerik_yolu = pdf_hedef.with_suffix(".icerik.json")
            veri = {"surum": 1, "video_analizi": video_analizi, "sorular": sorular or [],
                    "video_kaynak_bilgisi": video_kaynak_bilgisi,
                    "cerceve_adi": cerceve.name if cerceve else None}
            gecici_yol = None
            try:
                with NamedTemporaryFile(mode="w", encoding="utf-8", dir=icerik_yolu.parent, delete=False) as dosya:
                    gecici_yol = Path(dosya.name)
                    json.dump(veri, dosya, ensure_ascii=False, indent=2)
                gecici_yol.replace(icerik_yolu)
                print(f"[PDFGenerator] Yeniden basım kaydı: {icerik_yolu.resolve()}")
            finally:
                if gecici_yol is not None:
                    gecici_yol.unlink(missing_ok=True)
        print(f"\n[PDFGenerator] Gerçek PDF Dosyası Basılıyor: {pdf_hedef.name}...")

        doc = SimpleDocTemplate(
            str(pdf_hedef.resolve()),
            pagesize=A4,
            title=unite_basligi,
            subject=self._temizle_dosya_yolu(video_kaynak_bilgisi),
            leftMargin=_SABLON_YAN_BOSLUK,
            rightMargin=_SABLON_YAN_BOSLUK,
            topMargin=_SABLON_UST_BOSLUK + ust_kayma,
            bottomMargin=_SABLON_ALT_BOSLUK + alt_kayma,
        )

        flowables = []
        bekleyen_kavramlar = []
        gorulen_kavramlar = set()
        for idx, bolum in enumerate(bolumler, 1):
            if idx > 1:
                flowables.append(PageBreak())
            kitap_ref = bolum.get("kitap_referansi")
            flowables.append(_SayfaKaynagi(
                f"MEB Ders Kitabı Doğrulaması: {kitap_ref}"
                if kitap_ref and kitap_ref != "MEB Ders Kitabı Referansı" else ""
            ))
            if idx == 1:
                kademe = escape(str(video_analizi.get("egitim_seviyesi") or "İlkokul"))
                kaynak = escape(self._temizle_dosya_yolu(video_kaynak_bilgisi))
                tarih = datetime.now().strftime("%d.%m.%Y")
                flowables.extend([
                    Spacer(1, 0),
                    Paragraph(escape(unite_basligi), self.style_doc_title),
                    Spacer(1, 8),
                    Paragraph(f"Kademe: {kademe} ({escape(str(sinif_duzeyi))}) | Tarih: {tarih} | Kaynak: {kaynak}",
                              self.style_meta),
                    Spacer(1, 5),
                    HRFlowable(width="100%", thickness=0.85, color=colors.HexColor("#156082")),
                    Spacer(1, 22),
                ])

            baslik = escape(str(bolum.get("bolum_basligi") or "Eğitim Materyali"))
            zaman = escape(str(bolum.get("zaman_araligi_metni") or ""))
            zaman_stili = self.style_first_time if idx == 1 else self.style_time
            baslik_genisligi = self._icerik_genisligi + (12 if idx > 1 and cerceve else 0)
            zaman_genisligi = 146 if idx == 1 else 135
            baslik_kutusu = Table([[
                Paragraph(f"{idx}. Bölüm - {baslik}", self.style_section_title),
                Paragraph(f"Zaman Aralığı: {zaman}" if zaman else "", zaman_stili),
            ]], colWidths=[baslik_genisligi - zaman_genisligi, zaman_genisligi], cornerRadii=[5] * 4)
            baslik_kutusu.setStyle(TableStyle([
                ("BOX", (0, 0), (-1, -1), 1, colors.HexColor("#196B24")),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("LEFTPADDING", (0, 0), (-1, -1), 7),
                ("RIGHTPADDING", (0, 0), (-1, -1), 8),
                ("LEFTPADDING", (1, 0), (1, -1), 1),
                ("RIGHTPADDING", (1, 0), (1, -1), 4),
                ("TOPPADDING", (0, 0), (-1, -1), 5 if idx == 1 else 10),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 5 if idx == 1 else 10),
            ]))
            kare = Image(bolum["ekran_goruntusu_yolu"], width=self._icerik_genisligi - 2, height=219)
            flowables.extend([
                KeepTogether([baslik_kutusu, Spacer(1, 10 if idx == 1 else 14), kare]),
                Spacer(1, 16 if idx == 1 else 19),
            ])

            ham = escape(str(bolum.get("transkript") or "Bu aralıkta konuşma bulunamadı."))
            konusma_zamani = bolum.get("konusma_zaman_araligi")
            transkript_basligi = "Orijinal Transkript"
            if konusma_zamani:
                transkript_basligi += f" · {escape(str(konusma_zamani))}"
            flowables.extend([
                self._ders_kutusu(transkript_basligi, ham, "#F2F2F2", "#9A9A9A",
                                  alt_bosluk=15 if idx == 1 else 22),
                Spacer(1, 13 if idx == 1 else 19),
            ])

            anlatim = bolum.get("pedagojik_anlatim") or bolum.get("duzenlenmis_anlatim")
            if anlatim:
                anlatim_basligi = ("Ders Anlatımı (Transkript)" if bolum.get("kaynak_tipi") == "transkript"
                                   else "Düzenlenmiş Ders Anlatımı")
                metin = escape(str(anlatim)).replace("\n", "<br/>")
                if bolum.get("kitap_ek_bilgisi"):
                    metin += "<br/><br/><b>Kitaptan ek bilgi:</b> " + escape(str(bolum["kitap_ek_bilgisi"]))
                flowables.extend([
                    self._ders_kutusu(anlatim_basligi, metin, "#D2E4F5", "#147CE5", alt_bosluk=20),
                ])

            kavram_metni = []
            for kavram in bolum.get("onemli_kavramlar") or []:
                if not isinstance(kavram, dict):
                    continue
                ad = str(kavram.get("kavram") or "").strip()
                anahtar = kavram_anahtari(ad)
                if not anahtar or anahtar in gorulen_kavramlar:
                    continue
                gorulen_kavramlar.add(anahtar)
                aciklama = escape(str(kavram.get("aciklama") or ""))
                kavram_metni.append((idx, f"- <b>{escape(ad)}:</b> {aciklama}"))
            flowables.append(_KavramKutusu(
                idx, kavram_metni, bekleyen_kavramlar, self.style_box_label, self.style_concept_body,
                son_bolum=idx == len(bolumler),
            ))

        if video_analizi.get("genel_ozet"):
            flowables.extend([
                PageBreak(),
                _SayfaKaynagi(),
                Paragraph("Dersin Genel Özeti", self.style_super_title),
                Spacer(1, 9),
                Paragraph(escape(str(video_analizi["genel_ozet"])), self.style_pedagogical),
            ])

        if sorular:
            soru_sayfalari = []
            sayfa = []
            kalan = doc.height - 72
            for s in sorular:
                kart = self._modern_soru_karti(s)
                yukseklik = kart.wrap(self._icerik_genisligi, doc.height)[1] + 6
                if sayfa and (len(sayfa) == 5 or yukseklik > kalan):
                    soru_sayfalari.append(sayfa)
                    sayfa = []
                    kalan = doc.height - 72
                sayfa.append((escape(str(s.get("no", 1))), kart))
                kalan -= yukseklik
            soru_sayfalari.append(sayfa)
            for sayfa in soru_sayfalari:
                flowables.extend([
                    PageBreak(),
                    _SayfaKaynagi(),
                    Paragraph("Değerlendirme", self.style_super_title),
                    Paragraph(f"Sorular {sayfa[0][0]}–{sayfa[-1][0]}", self.style_meta),
                    Spacer(1, 10),
                ])
                for _, kart in sayfa:
                    flowables.extend([kart, Spacer(1, 6)])

            flowables.extend([PageBreak(), Paragraph("Çözümlü Cevap Anahtarı", self.style_super_title), Spacer(1, 9)])
            for s in sorular:
                no = escape(str(s.get("no", 1)))
                kavram = escape(str(s.get("ilgili_kavram") or ""))
                cevap = escape(str(s.get("dogru_cevap") or ""))
                gerekce = escape(str(s.get("aciklama") or ""))
                referans = s.get("kazanim_referansi") or "Doğrulanmış ders kitabı kaynağı bulunamadı."
                if referans == "MEB Ders Kitabı":
                    referans = "Doğrulanmış ders kitabı kaynağı bulunamadı."
                flowables.append(KeepTogether([
                    Paragraph(f"Soru {no} · {kavram}" if kavram else f"Soru {no}", self.style_box_label),
                    Paragraph(f"<b>Doğru yanıt:</b> {cevap}<br/>{gerekce}", self.style_pedagogical),
                    Paragraph(escape(str(referans)), self.style_ref_tag),
                    Spacer(1, 7),
                ]))

            kaynaklar = self._dinamik_kaynakca_derle(video_analizi, sorular)
            flowables.extend([Spacer(1, 8), Paragraph("MEB Kaynakçası", self.style_section_title)])
            for kaynak in kaynaklar:
                flowables.append(Paragraph(kaynak, self.style_ref_tag))

        # PDF'i derle
        def sayfayi_hazirla(pdf_canvas, _doc):
            pdf_canvas.cerceveyi_ciz()

        doc.build(flowables, onFirstPage=sayfayi_hazirla, onLaterPages=sayfayi_hazirla,
                  canvasmaker=partial(NumberedCanvas, cerceve=cerceve, ust_kayma=ust_kayma, alt_kayma=alt_kayma))
        print(f"[BAŞARILI] Kurumsal PDF Fasikülü üretildi: {pdf_hedef.resolve()}")

        return {
            "success": True,
            "pdf_path": str(pdf_hedef.resolve()),
            "icerik_yolu": str(icerik_yolu.resolve()) if icerik_yolu else None,
            "html_path": None,
            "mesaj": "Gerçek kurumsal PDF ders notu başarıyla üretildi."
        }

    def icerikten_yeniden_bas(self, icerik_yolu: Path) -> Dict[str, Any]:
        """Kaydedilmiş ders içeriğini mevcut tasarımla yerelde basar."""
        veri = json.loads(icerik_yolu.read_text(encoding="utf-8"))
        if (not isinstance(veri, dict) or veri.get("surum") != 1
                or not isinstance(veri.get("video_analizi"), dict)
                or not isinstance(veri["video_analizi"].get("bolumler"), list)
                or not veri["video_analizi"]["bolumler"]
                or not isinstance(veri.get("sorular"), list)):
            raise ValueError("Geçerli bir ders içeriği kaydı bulunamadı.")
        ad = icerik_yolu.name.removesuffix(".icerik.json")
        zaman = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        return self.pdf_olustur(
            video_analizi=veri["video_analizi"], sorular=veri["sorular"],
            video_kaynak_bilgisi=veri.get("video_kaynak_bilgisi") or "YouTube",
            cikis_dosya_adi=f"{ad}_yeniden_{zaman}.pdf",
            cerceve_adi=veri.get("cerceve_adi"),
        )


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
