"""
EduFrame AI - RAG ve Vektör Veritabanı Kalite Kontrol Aracı (check_rag_quality.py)
---------------------------------------------------------------------------------
Kullanım:
    python check_rag_quality.py
    python check_rag_quality.py --sorgu "Elektrikli araçlar ve pil" --sinif "3. Sınıf"
    python check_rag_quality.py --sorgu "Besinler ve dengeli beslenme" --sinif "4. Sınıf"
"""

import sys
import argparse
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from services.rag_motoru import RAGMotoru
from config import CHROMA_DB_V1_DIR, RAG_MAX_MESAFE


def baslik(metin: str):
    print("\n" + "=" * 75)
    print(f" 🎯 {metin}")
    print("=" * 75)


def sonuc_yaz(etiket: str, durum: bool, detay: str = ""):
    simge = "✓ [GEÇTİ]" if durum else "✗ [KALDI]"
    detay_str = f" -> {detay}" if detay else ""
    print(f" {simge:<10} {etiket:<40}{detay_str}")


def koleksiyon_istatistikleri(rag: RAGMotoru):
    baslik("1. Koleksiyon Durumu, Kitap ve Token İstatistikleri")
    col = rag.get_vector_store()._collection
    toplam = col.count()
    print(f" -> Veritabanı Dizini : {rag.db_dizini}")
    print(f" -> Koleksiyon        : {rag.koleksiyon_adi}")
    print(f" -> Toplam Parça (Chunk): {toplam}")

    res = col.get(include=["metadatas"])
    kitap_stats = {}
    sinif_stats = {}
    genel_toplam_token = 0

    for m in res.get("metadatas", []):
        src = m.get("source", "Bilinmeyen")
        snf = m.get("sinif", "Bilinmeyen")
        tc = int(m.get("token_count", 0))
        genel_toplam_token += tc

        if src not in kitap_stats:
            kitap_stats[src] = {"chunks": 0, "tokens": 0, "min": 999999, "max": 0}
        kitap_stats[src]["chunks"] += 1
        kitap_stats[src]["tokens"] += tc
        kitap_stats[src]["min"] = min(kitap_stats[src]["min"], tc)
        kitap_stats[src]["max"] = max(kitap_stats[src]["max"], tc)

        if snf not in sinif_stats:
            sinif_stats[snf] = {"chunks": 0, "tokens": 0}
        sinif_stats[snf]["chunks"] += 1
        sinif_stats[snf]["tokens"] += tc

    print(f" -> Genel Toplam Token  : {genel_toplam_token:,} (Gemma BPE)")

    print("\n 📚 Kitap Bazında Chunk ve Token Dağılımı:")
    print(f" {'Kitap Dosyası':<44} | {'Chunk':<6} | {'Toplam Token':<13} | {'Ort. Token':<10} | {'Aralık (Min-Max)'}")
    print(" " + "-" * 95)
    for k, v in sorted(kitap_stats.items()):
        avg = v["tokens"] / v["chunks"] if v["chunks"] else 0
        min_max = f"{v['min']} - {v['max']}"
        print(f" {k:<44} | {v['chunks']:>5}  | {v['tokens']:>10,} t  | {avg:>8.1f} t | {min_max}")

    print("\n 🎓 Sınıf Seviyesi Dağılımı:")
    for s, v in sorted(sinif_stats.items()):
        print(f"   * {s:<18} : {v['chunks']:>4} chunk  ({v['tokens']:>8,} token)")

    return toplam > 0


def sinif_izolasyon_testleri(rag: RAGMotoru):
    baslik("2. Otomatik Sınıf İzolasyonu ve Semantik Doğrulama Testleri")
    tum_testler_basarili = True

    # TEST A: 3. Sınıf İzolasyon Testi
    sorgu_3 = "Pil, elektrikli araç gerecin çalışabilmesi için gerekli enerjiyi sağlar."
    sonuclar_3 = rag.seviye_filtreli_arama(
        sorgu=sorgu_3,
        egitim_seviyesi="İlkokul",
        sinif="3. Sınıf",
        k=4,
        allow_cross_grade=False
    )

    hata_3 = []
    if not sonuclar_3:
        hata_3.append("Hiçbir sonuç dönmedi!")
    else:
        for idx, doc in enumerate(sonuclar_3, 1):
            doc_sinif = doc.metadata.get("sinif")
            if doc_sinif != "3. Sınıf":
                hata_3.append(f"Parça #{idx}: Beklenen 3. Sınıf yerine '{doc_sinif}' geldi! ({doc.metadata.get('source')})")

    durum_3 = (len(hata_3) == 0 and len(sonuclar_3) > 0)
    sonuc_yaz("Test A: 3. Sınıf İzolasyonu (Elektrik ve Pil)", durum_3,
              f"{len(sonuclar_3)} parça döndü, 0 sızıntı" if durum_3 else "; ".join(hata_3))
    if not durum_3:
        tum_testler_basarili = False

    # TEST B: 4. Sınıf İzolasyon Testi
    sorgu_4 = "Besinler vücudumuzun enerji ihtiyacını karşılar, karbonhidratlar ve proteinler."
    sonuclar_4 = rag.seviye_filtreli_arama(
        sorgu=sorgu_4,
        egitim_seviyesi="İlkokul",
        sinif="4. Sınıf",
        k=4,
        allow_cross_grade=False
    )

    hata_4 = []
    if not sonuclar_4:
        hata_4.append("Hiçbir sonuç dönmedi!")
    else:
        for idx, doc in enumerate(sonuclar_4, 1):
            doc_sinif = doc.metadata.get("sinif")
            if doc_sinif != "4. Sınıf":
                hata_4.append(f"Parça #{idx}: Beklenen 4. Sınıf yerine '{doc_sinif}' geldi! ({doc.metadata.get('source')})")

    durum_4 = (len(hata_4) == 0 and len(sonuclar_4) > 0)
    sonuc_yaz("Test B: 4. Sınıf İzolasyonu (Besinlerimiz)", durum_4,
              f"{len(sonuclar_4)} parça döndü, 0 sızıntı" if durum_4 else "; ".join(hata_4))
    if not durum_4:
        tum_testler_basarili = False

    # TEST C: 3. Sınıf 1. Kitap (Kayaçlar ve Fosiller)
    sorgu_kaya = "Kayaçlar, madenler ve fosillerin oluşumu, yer kabuğu."
    sonuclar_kaya = rag.seviye_filtreli_arama(
        sorgu=sorgu_kaya,
        egitim_seviyesi="İlkokul",
        sinif="3. Sınıf",
        k=3,
        allow_cross_grade=False
    )
    durum_c = bool(sonuclar_kaya) and all(d.metadata.get("sinif") == "3. Sınıf" for d in sonuclar_kaya)
    sonuc_yaz("Test C: 3. Sınıf 1. Kitap (Kayaçlar ve Madenler)", durum_c,
              f"{len(sonuclar_kaya)} parça döndü" if durum_c else "Eşleşme veya sınıf uyuşmazlığı")
    if not durum_c:
        tum_testler_basarili = False

    # TEST D: 4. Sınıf (Basit Elektrik Devreleri)
    sorgu_devre = "Basit elektrik devresinde pil, ampul, anahtar ve bağlantı kablosu."
    sonuclar_devre = rag.seviye_filtreli_arama(
        sorgu=sorgu_devre,
        egitim_seviyesi="İlkokul",
        sinif="4. Sınıf",
        k=3,
        allow_cross_grade=False
    )
    durum_d = bool(sonuclar_devre) and all(d.metadata.get("sinif") == "4. Sınıf" for d in sonuclar_devre)
    sonuc_yaz("Test D: 4. Sınıf (Basit Elektrik Devreleri)", durum_d,
              f"{len(sonuclar_devre)} parça döndü" if durum_d else "Eşleşme veya sınıf uyuşmazlığı")
    if not durum_d:
        tum_testler_basarili = False

    return tum_testler_basarili


def serbest_sorgu_calistir(
    rag: RAGMotoru, sorgu: str, sinif: str, max_mesafe: float = RAG_MAX_MESAFE
):
    baslik(f"Serbest Arama Testi: '{sorgu}' (Hedef: {sinif}, max_mesafe={max_mesafe})")
    sonuclar = rag.seviye_filtreli_arama(
        sorgu=sorgu,
        egitim_seviyesi="İlkokul",
        sinif=sinif,
        k=4,
        allow_cross_grade=False,
        max_mesafe=max_mesafe,
    )

    if not sonuclar:
        print(f"\n ℹ️ [EŞLEŞME BULUNAMADI] Arama {sinif} için sonuç döndürmedi.")
        return

    for idx, doc in enumerate(sonuclar, 1):
        m = doc.metadata
        print(f"\n--- [Sonuç #{idx}] ----------------------------------------------------")
        print(f" 📖 Kitap      : {m.get('kitap_adi')} ({m.get('source')})")
        print(f" 🎓 Sınıf      : {m.get('sinif')} | Ders: {m.get('ders_adi')}")
        print(f" 📑 Ünite/Konu : {m.get('unite')} -> {m.get('konu')}")
        print(f" 📄 Sayfa      : Basılı Sayfa {m.get('printed_page', m.get('page'))} (Fiziksel: {m.get('physical_page')})")
        print(f" 📊 Skor/Yöntem: Benzerlik={m.get('vektor_skoru', 'N/A')} (Mesafe: {m.get('vektor_mesafesi', 'N/A')}) | Yöntem: {m.get('arama_yontemi')}")
        print(f" 📝 Referans   : {m.get('referans')}")
        print(f" 💬 Metin Özeti:\n    {doc.page_content[:250].strip()}...")


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="EduFrame AI - RAG Kalite ve Sınıf İzolasyon Test Aracı")
    parser.add_argument("--sorgu", type=str, default="", help="Test edilecek arama metni")
    parser.add_argument("--sinif", type=str, default="3. Sınıf", choices=["3. Sınıf", "4. Sınıf"], help="Hedef sınıf")
    parser.add_argument(
        "--max-mesafe",
        type=float,
        default=RAG_MAX_MESAFE,
        help="Serbest aramada kabul edilen en büyük vektör mesafesi",
    )
    args = parser.parse_args()

    rag = RAGMotoru(db_dizini=CHROMA_DB_V1_DIR)

    if args.sorgu:
        serbest_sorgu_calistir(rag, args.sorgu, args.sinif, args.max_mesafe)
    else:
        koleksiyon_istatistikleri(rag)
        basarili = sinif_izolasyon_testleri(rag)
        print("\n" + "=" * 75)
        if basarili:
            print(" 🎉 TÜM KALİTE VE SINIF İZOLASYON TESTLERİ BAŞARIYLA GEÇTİ!")
        else:
            print(" ⚠️ BAZI TESTLERDE UYUŞMAZLIK BULUNDU. Lütfen yukarıdaki detayları inceleyin.")
        print("=" * 75)


if __name__ == "__main__":
    main()
