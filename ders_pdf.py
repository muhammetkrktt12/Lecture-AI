"""Seçilen anlatım modeliyle bir yerel videodan PDF ders notu üretir."""

import argparse
import os
from pathlib import Path


def main() -> int:
    proje = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", nargs="?", help="Örn. ollama, qwen3.5:4b, qwen3.5:9b veya gemini")
    parser.add_argument(
        "video", nargs="?", type=Path,
        default=proje / "temp_medya" / "video_1EPox1m2MPQ.mp4",
        help="Yerel video dosyası. Varsayılan, Elektrik Kaynakları videosudur.",
    )
    parser.add_argument("sinif", nargs="?", default="3. Sınıf", help="Dersin sınıf düzeyi.")
    parser.add_argument("--soru-ekle", action="store_true", help="PDF'ye değerlendirme soruları ekle.")
    parser.add_argument("--yeniden-bas", type=Path, metavar="ICERIK_JSON",
                        help="Kaydedilmiş .icerik.json dosyasından modele bağlanmadan PDF oluştur.")
    args = parser.parse_args()

    if args.yeniden_bas:
        if args.model or args.soru_ekle:
            parser.error("--yeniden-bas ile model veya --soru-ekle kullanılamaz.")
        from services.pdf_generator import PDFGenerator

        try:
            sonuc = PDFGenerator().icerikten_yeniden_bas(args.yeniden_bas)
        except (OSError, ValueError) as hata:
            parser.exit(1, f"PDF yeniden basılamadı: {hata}\n")
        print(sonuc["pdf_path"])
        return 0

    if not args.model:
        parser.error("Model adı veya --yeniden-bas seçeneği gerekli.")

    video = args.video.resolve()
    if not video.is_file():
        parser.error(f"Video dosyası bulunamadı: {video}")

    # config, Ollama modelini ilk içe aktarmada ortamdan okur.
    saglayici = "gemini" if args.model.lower() == "gemini" else "ollama"
    if saglayici == "ollama" and args.model.lower() != "ollama":
        os.environ["OLLAMA_MODEL"] = args.model

    from core.eduframe_agentic_core import EduFrameAgenticCore
    from services.gemini_analyzer import GeminiFasikulHatasi
    from services.pdf_generator import EksikDersKaresiHatasi

    try:
        sonuc = EduFrameAgenticCore().calistir(
            "Ders notu",
            video_kaynagi=str(video),
            sinif_duzeyi=args.sinif,
            anlatim_modeli=saglayici,
            soru_eklensin_mi=args.soru_ekle,
        )
    except (GeminiFasikulHatasi, EksikDersKaresiHatasi) as hata:
        parser.exit(1, f"PDF üretilemedi: {hata}\n")

    pdf_yolu = sonuc.get("cikti_pdf_yolu")
    if not pdf_yolu or not Path(pdf_yolu).is_file():
        parser.exit(1, f"PDF üretilemedi: {sonuc.get('hata') or 'Çıktı dosyası bulunamadı.'}\n")
    print(Path(pdf_yolu).resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
