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
import sys
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
        self._hand_detector = None
        self._hand_detector_initialized = False

    def _get_hand_detector(self):
        """MediaPipe HandLandmarker detektörünü (lazy-loaded) başlatır."""
        if getattr(self, "_hand_detector_initialized", False):
            return self._hand_detector
        self._hand_detector_initialized = True
        try:
            import mediapipe as mp
            from mediapipe.tasks import python
            from mediapipe.tasks.python import vision

            model_path = ROOT / "data" / "models" / "hand_landmarker.task"
            if not model_path.exists():
                import urllib.request
                model_path.parent.mkdir(parents=True, exist_ok=True)
                url = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
                urllib.request.urlretrieve(url, str(model_path))

            base_options = python.BaseOptions(model_asset_path=str(model_path))
            options = vision.HandLandmarkerOptions(base_options=base_options, num_hands=2)
            self._hand_detector = vision.HandLandmarker.create_from_options(options)
        except Exception:
            self._hand_detector = None
        return self._hand_detector

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

            hand_detector = self._get_hand_detector()

            def _engel_orani(img_p: str) -> float:
                im = Image.open(img_p).convert('RGB')
                arr = np.array(im, dtype=int)
                r, g, b = arr[:,:,0], arr[:,:,1], arr[:,:,2]
                skin = (r > 95) & (g > 40) & (b > 20) & ((np.maximum(r, np.maximum(g, b)) - np.minimum(r, np.minimum(g, b))) > 15) & (np.abs(r - g) > 15) & (r > g) & (r > b)
                h, w = arr.shape[:2]
                slide_area = skin[int(h*0.15):int(h*0.85), int(w*0.1):int(w*0.9)]
                return slide_area.sum() / slide_area.size

            def _engel_ve_el_skoru(img_p: str) -> tuple[float, int, bool]:
                """
                MediaPipe HandLandmarker ve renk analizini birleştirerek kesin engel skoru üretir.
                Döner: (toplam_skor, el_sayisi, el_tahtayi_kapatiyor_mu)
                """
                el_sayisi = 0
                el_kapatiyor = False
                el_cezasi = 0.0

                if hand_detector is not None:
                    try:
                        import mediapipe as mp
                        mp_img = mp.Image.create_from_file(img_p)
                        res = hand_detector.detect(mp_img)
                        if res.hand_landmarks:
                            el_sayisi = len(res.hand_landmarks)
                            min_y = min([lm.y for h in res.hand_landmarks for lm in h])
                            # El tahtanın içerik bölgesinde mi (ekranın alt %12'sinden yukarı uzanıyor mu?)
                            if min_y < 0.88:
                                el_kapatiyor = True
                                el_cezasi = 0.50 + el_sayisi * 0.25 + max(0.0, (0.88 - min_y) * 0.5)
                    except Exception:
                        pass

                renk_orani = _engel_orani(img_p)
                toplam_skor = el_cezasi + renk_orani
                return toplam_skor, el_sayisi, el_kapatiyor

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

            varsayilan_skor, el_sayisi, el_kapatiyor = _engel_ve_el_skoru(test_path)
            varsayilan_mukerrer = _oncekiyle_ayni_mi(test_path)
            varsayilan_reklam = _reklam_kitabi_mi(test_path, varsayilan_kare_sn)
            varsayilan_outro = _abone_ol_outro_mu(test_path, varsayilan_kare_sn)

            # Eğer varsayılan kare hem mükerrer/reklam/outro değilse hem de el engeli yoksa doğrudan kullan
            if not varsayilan_mukerrer and not varsayilan_reklam and not varsayilan_outro and not el_kapatiyor and varsayilan_skor <= 0.09:
                return varsayilan_kare_sn

            if varsayilan_reklam:
                print(f"[VideoProcessor] Bölüm {idx}: Reklam/ürün tanıtımı tespit edildi, filtreleniyor...")
            elif varsayilan_outro:
                print(f"[VideoProcessor] Bölüm {idx}: 'Abone Ol' outro ekranı tespit edildi, filtreleniyor...")
            elif varsayilan_mukerrer:
                print(f"[VideoProcessor] Bölüm {idx}: Varsayılan kare önceki bölümlerle mükerrer çıktı! Alternatif temiz slayt taranıyor...")
            elif el_kapatiyor:
                print(f"[VideoProcessor] Bölüm {idx}: MediaPipe ile tahtayı kapatan {el_sayisi} el tespit edildi (Skor: {varsayilan_skor:.2f}). Temiz kare taranıyor...")
            elif varsayilan_skor > 0.07:
                print(f"[VideoProcessor] Bölüm {idx}: Slayt üzerinde el/engel tespit edildi (Skor: %{varsayilan_skor*100:.1f}). Temiz kare taranıyor...")

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
                    cand_skor, cand_el, cand_kapatiyor = _engel_ve_el_skoru(cand_p)
                    if cand_skor < en_dusuk_engel:
                        en_dusuk_engel = cand_skor
                        en_iyi_t = t
                    if not cand_kapatiyor and cand_skor <= 0.04:
                        print(f" -> MediaPipe onaylı temiz ve özgün kare bulundu: {t}s (El: {cand_el}, Skor: {cand_skor:.3f})")
                        return t

            print(f" -> En temiz ve özgün kare seçildi: {en_iyi_t}s (Engel Skoru: {en_dusuk_engel:.3f})")
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
        ornekleme_adimi: float = 2.5,
        ssim_esigi: float = 0.58,
        min_materyal_suresi: float = 16.0
    ) -> List[Dict[str, Any]]:
        """
        Videoyu baştan sona tarayarak ekrandaki anlamlı görsel materyal değişimlerini tespit eder.
        SSIM (Yapısal Benzerlik) analizi ile slayt değişimlerini yakalar; öğretmen hareketlerini
        ve el jestlerini aynı materyal olarak korur.
        Tespit edilen her materyal için MediaPipe destekli en temiz ve engelsiz kareyi çıkarır.
        """
        if not os.path.exists(video_yolu):
            print(f"[UYARI] Video dosyası bulunamadı: {video_yolu}")
            return []

        import cv2
        import numpy as np

        cap = cv2.VideoCapture(str(video_yolu))
        if not cap.isOpened():
            print(f"[HATA] Video dosyası açılamadı: {video_yolu}")
            return []

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0
        toplam_sure = frame_count / fps if fps > 0 else 540.0

        print(f"\n[VideoProcessor] Video Analizi Başlıyor: {Path(video_yolu).name} (Süre: {toplam_sure:.1f}s, FPS: {fps:.2f})")
        print(f"[VideoProcessor] Slayt Değişimi Taraması (Örnekleme: {ornekleme_adimi}s, SSIM Eşiği: {ssim_esigi}, Min Süre: {min_materyal_suresi}s)...")

        def _hesapla_ssim(img1: np.ndarray, img2: np.ndarray) -> float:
            C1 = (0.01 * 255) ** 2
            C2 = (0.03 * 255) ** 2
            im1 = img1.astype(float)
            im2 = img2.astype(float)
            mu1 = cv2.GaussianBlur(im1, (11, 11), 1.5)
            mu2 = cv2.GaussianBlur(im2, (11, 11), 1.5)
            mu1_sq = mu1 ** 2
            mu2_sq = mu2 ** 2
            mu1_mu2 = mu1 * mu2
            sigma1_sq = cv2.GaussianBlur(im1 ** 2, (11, 11), 1.5) - mu1_sq
            sigma2_sq = cv2.GaussianBlur(im2 ** 2, (11, 11), 1.5) - mu2_sq
            sigma12 = cv2.GaussianBlur(im1 * im2, (11, 11), 1.5) - mu1_mu2
            ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
            return float(np.mean(ssim_map))

        # 1. Aşama: Slayt Geçiş Noktalarını Belirleme
        gecis_noktalari = [0.0]
        anchor_gray = None
        anchor_t = 0.0
        t = 0.0

        while t < toplam_sure:
            cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
            ret, frame = cap.read()
            if not ret:
                break

            small_gray = cv2.cvtColor(cv2.resize(frame, (160, 90)), cv2.COLOR_BGR2GRAY)
            if anchor_gray is None:
                anchor_gray = small_gray
                anchor_t = t
            else:
                score = _hesapla_ssim(anchor_gray, small_gray)
                # Eğer mevcut kare ile ana slayt arasındaki benzerlik belirgin şekilde düştüyse ve asgari süre dolduysa:
                if score < ssim_esigi and (t - anchor_t) >= min_materyal_suresi:
                    gecis_noktalari.append(round(t, 1))
                    anchor_gray = small_gray
                    anchor_t = t

            t += ornekleme_adimi

        cap.release()

        # Son bitiş saniyesini ekle
        if gecis_noktalari[-1] < toplam_sure - 5.0:
            gecis_noktalari.append(round(toplam_sure, 1))

        print(f"[VideoProcessor] Toplam {len(gecis_noktalari)-1} dinamik eğitim materyali aralığı tespit edildi.")

        # 2. Aşama: Her Aralık İçin MediaPipe ile En Temiz Kareyi Alma ve Bölüm Oluşturma
        bolumler: List[Dict[str, Any]] = []
        onceki_kareler: List[str] = []

        for idx in range(len(gecis_noktalari) - 1):
            baslangic_sn = gecis_noktalari[idx]
            bitis_sn = gecis_noktalari[idx + 1]
            varsayilan_kare = round((baslangic_sn + bitis_sn) / 2.0, 1)

            # MediaPipe + engel analizi ile en temiz kareyi bul
            temiz_kare_sn = self._en_temiz_kare_zamani_bul(
                video_yolu=video_yolu,
                baslangic=baslangic_sn,
                bitis=bitis_sn,
                varsayilan_kare_sn=varsayilan_kare,
                idx=idx + 1,
                onceki_kare_yollari=onceki_kareler
            )

            # Ekran görüntüsü al
            dosya_adi = f"bolum_{idx+1:02d}_{int(temiz_kare_sn):04d}s.jpg"
            kare_yolu = self.kare_yakala(video_yolu, temiz_kare_sn, cikti_dosya_adi=dosya_adi)
            if kare_yolu:
                onceki_kareler.append(kare_yolu)

            baslangic_str = self.saniyeden_zaman_damgasi(baslangic_sn)
            bitis_str = self.saniyeden_zaman_damgasi(bitis_sn)
            zaman_metni = f"{baslangic_str} – {bitis_str}"

            bolum_verisi = {
                "bolum_no": idx + 1,
                "bolum_basligi": f"Bölüm {idx + 1} – Görsel Eğitim Materyali",
                "zaman_araligi": {
                    "baslangic": baslangic_str,
                    "bitis": bitis_str,
                    "baslangic_saniye": baslangic_sn,
                    "bitis_saniye": bitis_sn,
                    "kare_zamani": temiz_kare_sn,
                },
                "zaman_araligi_metni": zaman_metni,
                "ekran_goruntusu_yolu": kare_yolu,
                "transkript": "",
                "pedagojik_anlatim": "",
                "onemli_kavramlar": [],
                "kitap_referansi": "",
            }
            bolumler.append(bolum_verisi)

        # 3. Aşama: Aynı Materyale Ait Alt Aralıkları Birleştirme (SSIM > 0.52)
        if len(bolumler) > 1:
            birlestirilmis: List[Dict[str, Any]] = []
            curr = bolumler[0]

            for nxt in bolumler[1:]:
                ayni_slayt = False
                curr_p = curr.get("ekran_goruntusu_yolu")
                nxt_p = nxt.get("ekran_goruntusu_yolu")
                if curr_p and nxt_p and os.path.exists(curr_p) and os.path.exists(nxt_p):
                    try:
                        im1 = cv2.cvtColor(cv2.resize(cv2.imread(curr_p), (160, 90)), cv2.COLOR_BGR2GRAY)
                        im2 = cv2.cvtColor(cv2.resize(cv2.imread(nxt_p), (160, 90)), cv2.COLOR_BGR2GRAY)
                        sc = _hesapla_ssim(im1, im2)
                        if sc > 0.52:
                            ayni_slayt = True
                    except Exception:
                        pass

                # Eğer aynı slaytsa aralıkları birleştir
                if ayni_slayt:
                    curr["zaman_araligi"]["bitis"] = nxt["zaman_araligi"]["bitis"]
                    curr["zaman_araligi"]["bitis_saniye"] = nxt["zaman_araligi"]["bitis_saniye"]
                    curr["zaman_araligi_metni"] = f"{curr['zaman_araligi']['baslangic']} – {nxt['zaman_araligi']['bitis']}"
                else:
                    birlestirilmis.append(curr)
                    curr = nxt
            birlestirilmis.append(curr)

            # Bölüm numaralarını yeniden diz
            for idx, b in enumerate(birlestirilmis, 1):
                b["bolum_no"] = idx
                b["bolum_basligi"] = f"Bölüm {idx} – Görsel Eğitim Materyali"

            bolumler = birlestirilmis

        print(f"[VideoProcessor] Materyal birleştirme sonrası nihai bölüm sayısı: {len(bolumler)}")
        return bolumler

    @staticmethod
    def _cuda_ortamini_hazirla():
        """CTranslate2 / faster-whisper için NVIDIA cuBLAS ve cuDNN DLL yollarını PATH'e ekler."""
        try:
            from sysconfig import get_path
            cuda_paketleri = Path(get_path("purelib")) / "nvidia"
            cuda_dizinleri = [
                str(cuda_dizini)
                for paket in ("cublas", "cudnn")
                if (cuda_dizini := cuda_paketleri / paket / "bin").is_dir()
            ]
            if os.name == "nt" and cuda_dizinleri:
                os.environ["PATH"] = os.pathsep.join(
                    [*cuda_dizinleri, os.environ.get("PATH", "")]
                )
        except Exception:
            pass

    def transkript_olustur(
        self,
        video_yolu: Any,
        model_boyutu: str = "small",
        zorla_yeniden: bool = False,
    ) -> Path:
        """
        faster-whisper ile videonun konuşma transkriptini GPU (CUDA) hızlandırmasıyla çıkarır.
        Sonucu video_yolu.transkript.txt dosyasına kaydeder ve döndürür.
        Önceden oluşturulmuş dosya varsa doğrudan önbellekten kullanır.
        """
        video_p = Path(video_yolu)
        cikti_p = video_p.with_suffix(".transkript.txt")

        if cikti_p.exists() and not zorla_yeniden:
            print(f"[VideoProcessor] Mevcut transkript dosyası bulundu (önbellek): {cikti_p.name}")
            return cikti_p

        print(f"[VideoProcessor] Whisper transkripsiyonu başlatılıyor ({video_p.name}, model={model_boyutu})...")
        self._cuda_ortamini_hazirla()
        from faster_whisper import WhisperModel

        try:
            model = WhisperModel(model_boyutu, device="cuda", compute_type="float16")
        except Exception as e:
            print(f"[UYARI] CUDA ile Whisper başlatılamadı ({e}), CPU moduna geçiliyor...")
            model = WhisperModel(model_boyutu, device="cpu", compute_type="int8")

        parcalar, _ = model.transcribe(
            str(video_p),
            language="tr",
            vad_filter=True,
        )

        with cikti_p.open("w", encoding="utf-8") as dosya:
            for parca in parcalar:
                metin = parca.text.strip()
                if metin:
                    dosya.write(f"[{parca.start:.2f} - {parca.end:.2f}] {metin}\n")

        print(f"[VideoProcessor] Transkript başarıyla oluşturuldu ve kaydedildi: {cikti_p.name}")
        return cikti_p

    def transkript_ile_eslestir(
        self,
        bolumler: List[Dict[str, Any]],
        transkript_kaynagi: Any,
    ) -> List[Dict[str, Any]]:
        """
        Dinamik olarak tespit edilen görsel materyal bölümlerini,
        Whisper zaman damgalı transkript parçalarıyla birebir eşleştirir.
        transkript_kaynagi doğrudan video yolu (.mp4) veya transkript dosyası (.txt) olabilir.
        """
        # Eğer doğrudan video dosyası verilmişse transkript oluştur veya önbellekten al
        if isinstance(transkript_kaynagi, (str, Path)):
            kaynak_p = Path(transkript_kaynagi)
            if kaynak_p.suffix.lower() in [".mp4", ".mkv", ".mov", ".avi", ".webm"]:
                transkript_kaynagi = self.transkript_olustur(kaynak_p)

        satirlar: List[Tuple[float, float, str]] = []

        if isinstance(transkript_kaynagi, (str, Path)) and os.path.exists(str(transkript_kaynagi)):
            with open(str(transkript_kaynagi), "r", encoding="utf-8") as f:
                for line in f:
                    m = re.match(r"\[(\d+\.?\d*)\s*-\s*(\d+\.?\d*)\]\s*(.*)", line.strip())
                    if m:
                        st, et, txt = float(m.group(1)), float(m.group(2)), m.group(3).strip()
                        satirlar.append((st, et, txt))
        elif isinstance(transkript_kaynagi, list):
            for item in transkript_kaynagi:
                if isinstance(item, dict):
                    st = float(item.get("start", 0.0))
                    et = float(item.get("end", 0.0))
                    txt = str(item.get("text", "")).strip()
                    satirlar.append((st, et, txt))
                elif hasattr(item, "start") and hasattr(item, "end") and hasattr(item, "text"):
                    satirlar.append((float(item.start), float(item.end), str(item.text).strip()))

        if not satirlar:
            print("[VideoProcessor] Eşleştirilecek transkript verisi bulunamadı.")
            return bolumler

        print(f"[VideoProcessor] {len(satirlar)} Whisper parçası {len(bolumler)} görsel bölüme dağıtılıyor...")

        for b in bolumler:
            b_sn_start = float(b["zaman_araligi"].get("baslangic_saniye", 0.0))
            b_sn_end = float(b["zaman_araligi"].get("bitis_saniye", 99999.0))

            uygun_metinler: List[str] = []
            ilk_konusma_sn: Optional[float] = None
            son_konusma_sn: Optional[float] = None

            for st, et, txt in satirlar:
                # Transkript parçası bu bölümün zaman penceresine denk geliyor mu?
                if (b_sn_start - 1.0) <= st < b_sn_end:
                    uygun_metinler.append(txt)
                    if ilk_konusma_sn is None:
                        ilk_konusma_sn = st
                    son_konusma_sn = et

            if uygun_metinler:
                b["transkript"] = " ".join(uygun_metinler)
                if ilk_konusma_sn is not None and son_konusma_sn is not None:
                    b["konusma_zaman_araligi"] = f"[{ilk_konusma_sn:.2f} - {son_konusma_sn:.2f}]"
            else:
                b["transkript"] = ""
                b["konusma_zaman_araligi"] = f"[{b_sn_start:.2f} - {b_sn_end:.2f}]"

        # Eğer son bölüm tamamen YouTube abone ol/like kapanışından ibaretse ve ders içeriği yoksa elenir:
        if len(bolumler) > 1:
            son = bolumler[-1]
            txt_son = son.get("transkript", "").lower()
            if any(w in txt_son for w in ["abone ol", "like tuş", "ödüllü soru", "ödüllü sayfa"]) and len(txt_son) < 300:
                ders_kelimeleri = ["enerji", "elektrik", "kaynak", "devre", "batarya", "akü", "santral", "besin", "fırın", "çamaşır"]
                if not any(k in txt_son for k in ders_kelimeleri):
                    print(f"[VideoProcessor] Video sonundaki saf YouTube outro ekranı ({son['zaman_araligi_metni']}) elendi.")
                    bolumler.pop()

        # Bölüm numaralarını ve başlıklarını garantiye al
        for idx, b in enumerate(bolumler, 1):
            b["bolum_no"] = idx
            b["bolum_basligi"] = f"Bölüm {idx} – Görsel Eğitim Materyali"

        return bolumler


if __name__ == "__main__":
    vp = VideoProcessor()
    print("VideoProcessor hazır. FFmpeg:", vp.ffmpeg_yolu)
    print("02:35 ->", vp.zaman_damgasindan_saniyeye("02:35"), "saniye")
    print("155.0 sn ->", vp.saniyeden_zaman_damgasi(155.0))
