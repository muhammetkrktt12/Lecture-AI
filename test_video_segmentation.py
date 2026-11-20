"""
EduFrame AI - Video Slayt Değişimi ve Transkript Eşleştirme Test Scripti
------------------------------------------------------------------------
Kullanım:
    # 1. Varsayılan test videosu ile:
    python test_video_segmentation.py

    # 2. Farklı bir yerel video ile:
    python test_video_segmentation.py "temp_medya/yeni_video.mp4"
    python test_video_segmentation.py "C:/Users/Nur/Videos/ders.mp4"

    # 3. YouTube linki ile:
    python test_video_segmentation.py "https://www.youtube.com/watch?v=..."
"""

import sys
import os
from pathlib import Path
from services.video_processor import VideoProcessor
from services.video_downloader import VideoDownloader

def main():
    print("\n" + "=" * 70)
    print("  EduFrame AI — Otomatik Ekran Değişimi & Transkript Eşleştirme Testi")
    print("=" * 70)

    # 1. Girdi kaynağını belirle (Argüman veya Varsayılan)
    if len(sys.argv) > 1:
        girdi = sys.argv[1].strip()
    else:
        girdi = "temp_medya/video_1EPox1m2MPQ.mp4"

    # 2. YouTube linki mi yoksa yerel dosya mı?
    if girdi.startswith(("http://", "https://", "www.youtube.com", "youtu.be")):
        print(f"[*] YouTube linki algılandı: {girdi}")
        print("[*] Video indiriliyor / hazırlanıyor...")
        downloader = VideoDownloader()
        sonuc = downloader.video_hazirla(kaynak=girdi, cozunurluk="720")
        if not sonuc.get("success"):
            print(f"[HATA] Video indirilemedi: {sonuc.get('error')}")
            return 1
        video_yolu = Path(sonuc["file_path"])
        print(f"[+] Video hazır: {video_yolu.name} ({sonuc.get('title')})")
    else:
        video_yolu = Path(girdi)
        if not video_yolu.exists():
            print(f"[HATA] Video dosyası bulunamadı: {video_yolu}")
            print("Lütfen dosya yolunu doğru yazdığınızdan veya videoyu temp_medya klasörüne koyduğunuzdan emin olun.")
            return 1

    print(f"\n[+] İşlenecek Video: {video_yolu.resolve()}")

    vp = VideoProcessor()

    # 3. Ekran değişimlerini tespit et ve MediaPipe ile temiz kareleri seç
    print("\n[Adım 1/2] Ekrandaki slayt/materyal değişimleri taranıyor...")
    transkript_yolu = vp.transkript_olustur(video_yolu)
    bolumler = vp.otomatik_ekran_degisimlerini_bul(
        str(video_yolu), transkript_yolu=transkript_yolu
    )

    # 4. Whisper transkripti ile eşleştir (txt varsa önbellekten okur, yoksa otomatik GPU Whisper çalıştırır)
    print("\n[Adım 2/2] Transkript eşleştiriliyor (gerekirse Whisper otomatik çalıştırılır)...")
    bolumler = vp.transkript_ile_eslestir(bolumler, transkript_yolu)

    print("\n" + "=" * 70)
    print(f"  TESPİT EDİLEN TOPLAM BÖLÜM SAYISI: {len(bolumler)}")
    print("=" * 70)

    for b in bolumler:
        no = b["bolum_no"]
        z_metni = b["zaman_araligi_metni"]
        kare_sn = b["zaman_araligi"]["kare_zamani"]
        kare_yolu = Path(b["ekran_goruntusu_yolu"]).name if b.get("ekran_goruntusu_yolu") else "-"
        konusma_aralik = b.get("konusma_zaman_araligi", "-")
        transkript = b.get("transkript", "").strip()
        transkript_kisa = (transkript[:100] + "...") if len(transkript) > 100 else transkript

        print(f"\n[Bölüm {no:02d}] --------------------------------------------------------")
        kare_bilgisi = f"{kare_sn:.1f}s -> {kare_yolu}" if kare_yolu != "-" else "temiz kare bulunamadı"
        print(f" ⏱️ Görsel Aralık   : {z_metni} (Kare: {kare_bilgisi})")
        print(f" 🎙️ Konuşma Aralığı : {konusma_aralik}")
        print(f" 💬 Transkript Özeti: \"{transkript_kisa}\"")

    print("\n" + "=" * 70)
    eksik_kare = sum(not b.get("ekran_goruntusu_yolu") for b in bolumler)
    eksik_metin = sum(not b.get("transkript", "").strip() for b in bolumler)
    gecersiz_kare = sum(
        not Path(b["ekran_goruntusu_yolu"]).is_file()
        or not (
            b["zaman_araligi"]["baslangic_saniye"]
            <= b["zaman_araligi"]["kare_zamani"]
            < b["zaman_araligi"]["bitis_saniye"]
        )
        for b in bolumler if b.get("ekran_goruntusu_yolu")
    )
    print(f"  Temiz karesi eksik bölüm: {eksik_kare} | Konuşması olmayan bölüm: {eksik_metin}")
    print(f"  Dosyası veya zamanı geçersiz kare: {gecersiz_kare}")
    print("  Kare ve konuşmanın içerik uyumunu bölüm bazında inceleyin.")
    print("=" * 70 + "\n")
    return int(not bolumler or bool(eksik_kare or eksik_metin or gecersiz_kare))

if __name__ == "__main__":
    sys.exit(main())
