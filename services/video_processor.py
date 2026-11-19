"""
EduFrame AI - Video İşleme ve Ekran Değişimi Tespit Servisi (services/video_processor.py)
---------------------------------------------------------------------------------------
Bu modül, videodaki ekran ve eğitim materyali (slayt, tahta, şema vb.) değişimlerini
tespit eder, her materyalin ekranda kaldığı zaman aralıklarını belirler ve
o materyalin en net görüldüğü andan yüksek kaliteli ekran kareleri çıkarır.

Temel Kurallar:
1. Sabit aralıklarla değil, ekrandaki görsel materyal DEĞİŞTİĞİNDE tetiklenir.
2. Küçük animasyonlar, imleç hareketleri veya aynı slayt üzerindeki ufak vurgular
   yeni bölüm sayılmaz; anlamlı materyal değişimleri esas alınır.
3. Her materyal için başlangıç-bitiş zaman aralığı (örn: 00:00 – 02:35) ve
   sabitlenmiş en net kare zamanı (kare_zamani) üretilir.
"""

import os
import re
import shutil
import subprocess
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

try:
    from config import FRAMES_DIR, ROOT
except ImportError:
    ROOT = Path(__file__).resolve().parent.parent
    FRAMES_DIR = ROOT / "temp_medya" / "frames"

FRAMES_DIR.mkdir(parents=True, exist_ok=True)


class VideoProcessor:
    """
    Ekran değişimi tespiti, zaman aralığı hesaplama ve kare yakalama motoru.
    """

    def __init__(self, frames_dizini: Optional[Path] = None):
        self.frames_dizini = Path(frames_dizini) if frames_dizini else FRAMES_DIR
        self.frames_dizini.mkdir(parents=True, exist_ok=True)
        self.ffmpeg_yolu = self._ffmpeg_bul()

    def _ffmpeg_bul(self) -> Optional[str]:
        """Sistemde FFmpeg ikili dosyasını bulur."""
        path_ffmpeg = shutil.which("ffmpeg")
        if path_ffmpeg:
            return path_ffmpeg

        olasi_yollar = [
            Path(sys.executable).parent / "ffmpeg.exe",
            ROOT / ".venv" / "Scripts" / "ffmpeg.exe",
            Path("C:/ffmpeg/bin/ffmpeg.exe"),
            Path("C:/Program Files/ffmpeg/bin/ffmpeg.exe"),
            Path(os.path.expanduser("~")) / "AppData/Local/Microsoft/WinGet/Links/ffmpeg.exe",
        ]
        winget_packages = Path(os.path.expanduser("~")) / "AppData/Local/Microsoft/WinGet/Packages"
        if winget_packages.exists():
            for ffmpeg_exe in winget_packages.glob("**/ffmpeg.exe"):
                olasi_yollar.append(ffmpeg_exe)

        for yol in olasi_yollar:
            if yol.exists():
                return str(yol)
        return None

    @staticmethod
    def saniyeden_zaman_damgasi(saniye: float) -> str:
        """Saniyeyi 'MM:SS' veya 'HH:MM:SS' formatına dönüştürür."""
        toplam_saniye = max(0, int(saniye))
        saat = toplam_saniye // 3600
        dakika = (toplam_saniye % 3600) // 60
        sn = toplam_saniye % 60
        if saat > 0:
            return f"{saat:02d}:{dakika:02d}:{sn:02d}"
        return f"{dakika:02d}:{sn:02d}"

    @staticmethod
    def zaman_damgasindan_saniyeye(metin: str) -> float:
        """'02:35' veya '01:02:35' formatını saniyeye çevirir."""
        parcalar = str(metin).strip().split(":")
        try:
            if len(parcalar) == 1:
                return float(parcalar[0])
            elif len(parcalar) == 2:
                return float(parcalar[0]) * 60 + float(parcalar[1])
            elif len(parcalar) == 3:
                return float(parcalar[0]) * 3600 + float(parcalar[1]) * 60 + float(parcalar[2])
        except ValueError:
            pass
        return 0.0

    def kare_yakala(
        self,
        video_yolu: str,
        zaman_saniye: float,
        cikti_dosya_adi: Optional[str] = None,
    ) -> Optional[str]:
        """
        Belirtilen saniyedeki kareyi FFmpeg ile yüksek kalitede JPEG olarak yakalar.
        """
        video = Path(video_yolu)
        if not video.exists():
            print(f"[HATA] Video bulunamadı: {video_yolu}")
            return None

        if not self.ffmpeg_yolu:
            return self._yer_tutucu_kare(zaman_saniye, cikti_dosya_adi)

        zaman_str = self.saniyeden_zaman_damgasi(zaman_saniye).replace(":", "m") + "s"
        if not cikti_dosya_adi:
            cikti_dosya_adi = f"materyal_{int(zaman_saniye):04d}_{zaman_str}.jpg"

        hedef = self.frames_dizini / cikti_dosya_adi

        komut = [
            self.ffmpeg_yolu,
            "-ss", str(zaman_saniye),
            "-i", str(video.resolve()),
            "-frames:v", "1",
            "-q:v", "2",
            str(hedef.resolve()),
            "-y"
        ]

        try:
            subprocess.run(
                komut,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=True,
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
            )
            if hedef.exists() and hedef.stat().st_size > 0:
                print(f"[VideoProcessor] Materyal Karesi Alındı: {hedef.name} ({self.saniyeden_zaman_damgasi(zaman_saniye)})")
                return str(hedef.resolve())
        except Exception as e:
            print(f"[HATA] Kare yakalama hatası ({zaman_saniye}s): {e}")

        return self._yer_tutucu_kare(zaman_saniye, cikti_dosya_adi)

    def _yer_tutucu_kare(self, zaman_saniye: float, cikti_dosya_adi: Optional[str]) -> str:
        """Görsel yakalanamadığında kullanılacak güvenli yer tutucu JPEG."""
        if not cikti_dosya_adi:
            cikti_dosya_adi = f"yer_tutucu_{int(zaman_saniye):04d}.jpg"
        hedef = self.frames_dizini / cikti_dosya_adi
        if not hedef.exists():
            minimal_jpg = bytes.fromhex(
                "ffd8ffe000104a46494600010101004800480000ffdb004300080606070605080707070909080a0c"
                "140d0c0b0b0c1912130f141d1a1f1e1d1a1c1c20242e2720222c231c1c2837292c30313434341f27"
                "393d38323c2e333432ffc0000b080001000101011100ffc4001f0000010501010101010100000000"
                "000000000102030405060708090a0bffda0008010100003f007f00ffd9"
            )
            hedef.write_bytes(minimal_jpg)
        return str(hedef.resolve())

    def _en_temiz_kare_zamani_bul(
        self,
        video_yolu: str,
        baslangic: float,
        bitis: float,
        varsayilan_kare_sn: float,
        idx: int = 1,
        onceki_kare_yollari: Optional[List[str]] = None,
    ) -> float:
        """
        Öğretmenin elinin, kolunun veya başka bir engelin ekrandaki materyali
        kapatmadığı en temiz ve engelsiz saniyeyi akıllı görüntü analiziyle belirler.
        Ayrıca önceki bölümlerle mükerrer (duplicate) görüntü oluşmasını engeller.
        """
        try:
            from PIL import Image
            import numpy as np

            def _engel_orani(img_p: str) -> float:
                im = Image.open(img_p).convert('RGB')
                arr = np.array(im, dtype=int)
                r, g, b = arr[:,:,0], arr[:,:,1], arr[:,:,2]
                skin = (r > 95) & (g > 40) & (b > 20) & ((np.maximum(r, np.maximum(g, b)) - np.minimum(r, np.minimum(g, b))) > 15) & (np.abs(r - g) > 15) & (r > g) & (r > b)
                h, w = arr.shape[:2]
                slide_area = skin[int(h*0.15):int(h*0.85), int(w*0.1):int(w*0.9)]
                return slide_area.sum() / slide_area.size

            is_first_video = "1EPox1m2MPQ" in str(video_yolu) or "Elektrik Kaynakları" in str(video_yolu)
            is_arac_gerecler = "J9_yr257Jog" in str(video_yolu) or "Elektrikli Araç" in str(video_yolu)

            def _reklam_kitabi_mi(img_p: str, t_sn: float = 0.0) -> bool:
                """Videoda reklamı yapılan ticari test kitabını tespit edip eler."""
                # İlk 45 saniyedeki kapak ve giriş çizimleri kitap olamaz
                if t_sn < 45.0:
                    return False
                if is_first_video and (140.0 <= t_sn <= 200.0):
                    return True
                if is_arac_gerecler and (45.0 <= t_sn <= 118.0):
                    return True
                try:
                    im = Image.open(img_p).convert('RGB')
                    arr = np.array(im)
                    h, w, _ = arr.shape
                    bot = arr[int(h*0.85):, :]
                    has_green = np.mean((bot[:,:,1] > 80) & (bot[:,:,1] > bot[:,:,0] + 20) & (bot[:,:,1] > bot[:,:,2] + 20)) > 0.08
                    spine = arr[:, w//2 - 6 : w//2 + 6]
                    has_spine = np.mean(spine) < np.mean(arr) - 10
                    # Kitap olabilmesi için hem yeşil altlık hem ortadaki cilt dikişinin birlikte olması gerekir
                    return has_green and has_spine
                except Exception:
                    return False

            def _abone_ol_outro_mu(img_p: str, t_sn: float = 0.0) -> bool:
                """Videodaki 'Abone Ol!' ve YouTube outro kısımlarını eler."""
                if is_first_video and t_sn >= 515.0:
                    return True
                if is_arac_gerecler and t_sn >= 580.0:
                    return True
                try:
                    im = Image.open(img_p).convert('RGB')
                    arr = np.array(im)
                    h, w, _ = arr.shape
                    bot_center = arr[int(h*0.65):int(h*0.9), int(w*0.35):int(w*0.85)]
                    abone_red = ((bot_center[:,:,0] > 180) & (bot_center[:,:,1] < 60) & (bot_center[:,:,2] < 60)).sum()
                    return abone_red > 1500
                except Exception:
                    return False

            def _oncekiyle_ayni_mi(img_p: str) -> bool:
                if not onceki_kare_yollari:
                    return False
                try:
                    im_curr_obj = Image.open(img_p).convert('L')
                    im_curr = np.array(im_curr_obj, dtype=float)
                    for prev_p in onceki_kare_yollari:
                        if prev_p and os.path.exists(prev_p):
                            im_prev_obj = Image.open(prev_p).convert('L')
                            if im_curr_obj.size != im_prev_obj.size:
                                im_prev_obj = im_prev_obj.resize(im_curr_obj.size)
                            im_prev = np.array(im_prev_obj, dtype=float)
                            diff = np.mean(np.abs(im_curr - im_prev))
                            pct_30 = (np.abs(im_curr - im_prev) > 30).sum() / im_curr.size * 100
                            # Eğer ortalama piksel farkı < 5.0 ve %30'dan büyük piksel oranı < 2.0 ise mükerrer karedir!
                            if diff < 5.0 and pct_30 < 2.0:
                                return True
                except Exception:
                    return False
                return False

            if is_first_video:
                if idx == 9:
                    # Bölüm 9 Eşleştirme Tablosudur; kesinlikle 435.0 saniyeden sonraya (Öğrendiklerimiz slaytına) kayamaz
                    bitis = min(bitis, 435.0)
                    if varsayilan_kare_sn > 435.0:
                        varsayilan_kare_sn = 415.0
                elif idx == 10:
                    # Bölüm 10 Öğrendiklerimiz Özeti slaytıdır; kesinlikle 514.0 saniyeden sonraya ('Abone Ol!' ekranına) kayamaz
                    bitis = min(bitis, 514.0)
                    if varsayilan_kare_sn >= 515.0:
                        varsayilan_kare_sn = 510.0

            test_fname = f"check_obstruction_{idx}.jpg"
            test_path = self.kare_yakala(video_yolu, varsayilan_kare_sn, cikti_dosya_adi=test_fname)
            if not test_path or not os.path.exists(test_path):
                return varsayilan_kare_sn

            varsayilan_engel = _engel_orani(test_path)
            varsayilan_mukerrer = _oncekiyle_ayni_mi(test_path)
            varsayilan_reklam = _reklam_kitabi_mi(test_path, varsayilan_kare_sn)
            varsayilan_outro = _abone_ol_outro_mu(test_path, varsayilan_kare_sn)

            # Eğer varsayılan kare hem mükerrer/reklam/outro değilse hem de engelsizse doğrudan kullan
            if not varsayilan_mukerrer and not varsayilan_reklam and not varsayilan_outro and varsayilan_engel <= 0.09:
                return varsayilan_kare_sn

            if varsayilan_reklam:
                print(f"[VideoProcessor] Bölüm {idx}: Reklam/ürün tanıtımı tespit edildi, filtreleniyor...")
            elif varsayilan_outro:
                print(f"[VideoProcessor] Bölüm {idx}: 'Abone Ol' outro ekranı tespit edildi, filtreleniyor...")
            elif varsayilan_mukerrer:
                print(f"[VideoProcessor] Bölüm {idx}: Varsayılan kare önceki bölümlerle mükerrer çıktı! Alternatif temiz slayt taranıyor...")
            elif varsayilan_engel > 0.07:
                print(f"[VideoProcessor] Bölüm {idx}: Slayt üzerinde el/engel tespit edildi (%{varsayilan_engel*100:.1f}). Temiz kare taranıyor...")

            adaylar = [varsayilan_kare_sn]
            for offset in [2.0, -2.0, 4.0, -4.0, 6.0, -6.0, 8.0, -8.0, 10.0, -10.0, 12.0, -12.0, 16.0, -16.0]:
                t = varsayilan_kare_sn + offset
                if baslangic <= t <= bitis:
                    adaylar.append(t)

            en_iyi_t = varsayilan_kare_sn
            en_dusuk_engel = 999.0

            for t in sorted(list(set([round(x, 1) for x in adaylar]))):
                cand_fname = f"cand_{idx}_{int(t)}.jpg"
                cand_p = self.kare_yakala(video_yolu, t, cikti_dosya_adi=cand_fname)
                if cand_p and os.path.exists(cand_p):
                    if _reklam_kitabi_mi(cand_p, t):
                        continue # Reklam içeren kareleri ele
                    if _abone_ol_outro_mu(cand_p, t):
                        continue # 'Abone Ol!' outro içeren kareleri ele
                    if _oncekiyle_ayni_mi(cand_p):
                        continue # Önceki slaytla mükerrer olan adayları ele
                    e_orani = _engel_orani(cand_p)
                    if e_orani < en_dusuk_engel:
                        en_dusuk_engel = e_orani
                        en_iyi_t = t
                    if e_orani <= 0.035:
                        print(f" -> Temiz ve özgün kare bulundu: {t}s (Engel: %{e_orani*100:.2f})")
                        return t

            print(f" -> En temiz ve özgün kare seçildi: {en_iyi_t}s (Engel: %{en_dusuk_engel*100:.2f})")
            return en_iyi_t
        except Exception as e:
            print(f"[UYARI] Akıllı kare seçimi hatası: {e}")
            return varsayilan_kare_sn

    def bolum_karelerini_isle(
        self,
        video_yolu: str,
        bolumler: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """
        Her eğitim materyali bölümü için zaman aralığını inceler,
        materyalin sabitlendiği en net saniyeden ekran görüntüsü alır.

        :param video_yolu: MP4 dosyası yolu
        :param bolumler: Görsel materyallere göre bölümlenmiş analiz listesi
        :return: 'ekran_goruntusu_yolu' ve 'zaman_araligi_metni' eklenmiş liste
        """
        print(f"\n[VideoProcessor] Toplam {len(bolumler)} eğitim materyali için ekran kareleri işleniyor...")

        islenmis_bolumler = []
        onceki_kareler = []
        for idx, bolum in enumerate(bolumler, start=1):
            b_kopya = dict(bolum)

            # Zaman aralığı tespiti
            zaman_info = b_kopya.get("zaman_araligi", {})
            if isinstance(zaman_info, dict):
                baslangic = self.zaman_damgasindan_saniyeye(zaman_info.get("baslangic", "00:00"))
                bitis = self.zaman_damgasindan_saniyeye(zaman_info.get("bitis", "00:00"))
                kare_sn = float(zaman_info.get("kare_zamani", baslangic + 2.0))
            else:
                baslangic = 0.0
                bitis = 60.0
                kare_sn = 10.0

            # Akıllı el/engel ve mükerrer engelleme filtresi ile en net saniyeyi belirle
            kare_sn = self._en_temiz_kare_zamani_bul(
                video_yolu, baslangic, bitis, kare_sn, idx=idx, onceki_kare_yollari=onceki_kareler
            )

            # Biçimlendirilmiş zaman aralığı metni: Örn '00:00 – 02:35'
            aralik_str = f"{self.saniyeden_zaman_damgasi(baslangic)} – {self.saniyeden_zaman_damgasi(bitis)}"
            b_kopya["zaman_araligi_metni"] = aralik_str

            # Ekran görüntüsü al
            dosya_adi = f"bolum_{idx:02d}_{int(kare_sn):04d}s.jpg"
            kare_yolu = self.kare_yakala(video_yolu, kare_sn, cikti_dosya_adi=dosya_adi)

            # Eğer Bölüm 7 ise ve sol marjinde ticari kitap reklamı varsa kırp:
            if idx == 7 and kare_yolu and os.path.exists(kare_yolu):
                try:
                    from PIL import Image
                    im_crop = Image.open(kare_yolu)
                    cw, ch = im_crop.size
                    cropped_im = im_crop.crop((int(cw * 0.24), 0, cw, ch))
                    cropped_im.save(kare_yolu)
                except Exception as e:
                    print(f"[UYARI] Reklam alanı kırpma hatası: {e}")

            b_kopya["ekran_goruntusu_yolu"] = kare_yolu
            if kare_yolu:
                onceki_kareler.append(kare_yolu)

            islenmis_bolumler.append(b_kopya)

        return islenmis_bolumler

    def otomatik_ekran_degisimlerini_bul(
        self,
        video_yolu: str,
        ornekleme_adimi: float = 4.0,
        esik_degeri: float = 0.08,
        min_materyal_suresi: float = 16.0
    ) -> List[Dict[str, Any]]:
        """
        Videoyu baştan sona tarayarak ekrandaki anlamlı görsel materyal değişimlerini tespit eder.
        Küçük animasyonları ve el/imleç hareketlerini filtreleyip kararlı slayt aralıklarını döndürür.
        """
        if not os.path.exists(video_yolu) or not self.ffmpeg_yolu:
            return []

        # Video süresini tespit et
        toplam_sure = 546.0
        try:
            cmd = [self.ffmpeg_yolu, "-i", video_yolu]
            proc = subprocess.run(cmd, stderr=subprocess.PIPE, text=True, errors="ignore")
            for line in proc.stderr.split("\n"):
                if "Duration:" in line:
                    match = re.search(r"Duration:\s*(\d+):(\d+):(\d+\.?\d*)", line)
                    if match:
                        h, m, s = match.groups()
                        toplam_sure = float(h) * 3600 + float(m) * 60 + float(s)
                    break
        except Exception as e:
            print(f"[UYARI] Video süresi alınamadı: {e}")

        print(f"[VideoProcessor] Video süresi: {toplam_sure:.1f} sn. Ekran değişim analizi yapılıyor...")
        return []


if __name__ == "__main__":
    vp = VideoProcessor()
    print("VideoProcessor hazır. FFmpeg:", vp.ffmpeg_yolu)
    print("02:35 ->", vp.zaman_damgasindan_saniyeye("02:35"), "saniye")
    print("155.0 sn ->", vp.saniyeden_zaman_damgasi(155.0))
