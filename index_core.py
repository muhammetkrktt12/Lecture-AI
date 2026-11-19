"""
EduFrame AI - Ortak Vektör İndeksleme ve Metadata Çekirdeği (index_core.py)
-------------------------------------------------------------------------
Bu modül tüm ders kitapları için ortak ve güvenli indeksleme fonksiyonlarını içerir:
1. Dosya SHA-256 hash hesaplama ve kararlı Chunk ID üretimi
2. 768-D Embedding boyutu ön doğrulaması (ChromaDB açılmadan önce)
3. Sayfa ofseti (Fiziksel PDF sayfası vs Kitap baskı sayfası) otomatik tespiti
4. Kitaba özel 3 Seviyeli Manifest (Ünite, Konu, Alt Konu) mimarisi
5. 1000 Karakter / 150 Overlap / 9 Pedagojik ayırıcı ile metin bölme
6. Gemma BPE Tokenizer ile gerçek token sayımı ve metadata etiketleme
7. Hata toleranslı (Batch -> Retry -> Tekil Chunk Fallback) ChromaDB yazma ve detaylı loglama
8. PDF güncellendiğinde eski sürüme ait parça kayıtlarını otomatik temizleme
"""

import time
import re
import hashlib
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional, Set
import pypdf

from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ---------------------------------------------------------------------------
# 1. DOSYA HASH VE KARARLI CHUNK ID
# ---------------------------------------------------------------------------
def calculate_file_hash(file_path: Path) -> str:
    """PDF dosyasının SHA-256 hash'ini hesaplar."""
    hasher = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


def generate_chunk_id(
    filename: str,
    file_hash: str,
    physical_page: int,
    chunk_idx: int,
    version: str = ""
) -> str:
    """Tekrarlanabilir, dosya sürümüne ve fiziksel sayfa numarasına duyarlı benzersiz Chunk ID üretir."""
    if version:
        return f"{filename}:{file_hash[:12]}:{version}:phys_{physical_page:04d}:c{chunk_idx:03d}"
    return f"{filename}:{file_hash[:12]}:phys_{physical_page:04d}:c{chunk_idx:03d}"


# ---------------------------------------------------------------------------
# 2. EMBEDDING BOYUTU DOĞRULAMA (768-D GÜVENLİK KONTROLÜ)
# ---------------------------------------------------------------------------
def verify_embedding_dimension(
    ollama_url: str,
    model_name: str = "embeddinggemma",
    expected_dim: int = 768
) -> OllamaEmbeddings:
    """
    Ollama modeline anlık test sorgusu göndererek dönen vektör boyutunun
    kesinlikle expected_dim (768) olduğunu doğrular. Uyuşmazlıkta durdurur.
    """
    emb = OllamaEmbeddings(model=model_name, base_url=ollama_url)
    try:
        test_vec = emb.embed_query("Fen bilimleri ders kitabı elektrik ve kuvvet analizi")
    except Exception as e:
        raise ConnectionError(
            f"Ollama bağlantı hatası ({ollama_url}). Lütfen Ollama servisinin açık "
            f"ve '{model_name}' modelinin yüklü olduğundan emin olun! Hata: {e}"
        )

    actual_dim = len(test_vec)
    if actual_dim != expected_dim:
        raise ValueError(
            f"[KRİTİK HATA] Embedding boyutu uyuşmazlığı! Beklenen: {expected_dim}, "
            f"Ollama'dan dönen: {actual_dim}. Lütfen config veya model ayarını kontrol edin."
        )

    return emb


# ---------------------------------------------------------------------------
# 3. SAYFA OFSETİ (FİZİKSEL SAYFA VS BASILI SAYFA) OTOMATİK TESPİTİ
# ---------------------------------------------------------------------------
def detect_page_offset(reader: pypdf.PdfReader, max_scan_pages: int = 25) -> int:
    """
    Fiziksel PDF sayfa indeksi ile sayfa üzerinde yazan basılı sayfa numarası
    arasındaki farkı (offset = physical_page - printed_page) otomatik tespit eder.
    Örnek: Fiziksel 8. sayfanın altında/üstünde '7' yazıyorsa offset = 1'dir.
    """
    diff_counts: Dict[int, int] = {}
    total = len(reader.pages)
    
    for phys_p in range(5, min(max_scan_pages, total) + 1):
        try:
            txt = reader.pages[phys_p - 1].extract_text() or ""
            lines = [l.strip() for l in txt.split("\n") if l.strip()]
            for l in lines[:2] + lines[-2:]:
                if l.isdigit():
                    val = int(l)
                    if 1 <= val <= 60:
                        diff = phys_p - val
                        diff_counts[diff] = diff_counts.get(diff, 0) + 1
                        break
        except Exception:
            continue

    if diff_counts:
        best_offset = max(diff_counts.items(), key=lambda x: x[1])[0]
        return best_offset
    return 1  # Çoğu MEB kitabında kapak nedeniyle varsayılan +1


# ---------------------------------------------------------------------------
# 4. 3 SEVİYELİ AYRINTILI KİTAP MANIFESTOLARI (ÜNİTE / KONU / ALT KONU)
# ---------------------------------------------------------------------------
# Sayfa aralıkları kitabın üzerindeki BASILI SAYFA (printed_page) numaralarına göredir.
PRIMARY_BOOK_MANIFESTS = {
    "ilkokul_fenbilimleri_3.sinif_1.kitap.pdf": {
        "kitap_adi": "MEB 3. Sınıf Fen Bilimleri (1. Kitap)",
        "sinif": "3. Sınıf",
        "ders": "Fen Bilimleri",
        "seviye": "İlkokul",
        "uniteler": [
            # 1. Ünite: Bilimsel Keşif Yolculuğu (Basılı Sayfa 13 - 38)
            (13, 25, "1. Ünite: Bilimsel Keşif Yolculuğu", "Bilimsel Bilgiye Ulaşma Yolları", "Gözlem, Deney ve Bilgi Kaynakları"),
            (26, 38, "1. Ünite: Bilimsel Keşif Yolculuğu", "Bilim İnsanlarının Özellikleri", "Merak, Sorgulama ve Araştırma"),
            # 2. Ünite: Canlılar Dünyasına Yolculuk (Basılı Sayfa 39 - 80)
            (39, 57, "2. Ünite: Canlılar Dünyasına Yolculuk", "Canlıların Sınıflandırılması", "Bitkiler, Hayvanlar ve Benzerlikler"),
            (58, 68, "2. Ünite: Canlılar Dünyasına Yolculuk", "Duyu Organlarının İşlevleri", "Beş Duyu Organı ve Sağlığın Korunması"),
            (69, 80, "2. Ünite: Canlılar Dünyasına Yolculuk", "Canlıların Yaşam Döngüleri", "Doğma, Büyüme, Gelişme ve Üreme"),
            # 3. Ünite: Yer Bilimciler İş Başında (Basılı Sayfa 81 - 104)
            (81, 92, "3. Ünite: Yer Bilimciler İş Başında", "Kayaçlar, Madenler ve Mineraller", "Kayaçların Yapısı ve Ekonomik Değeri"),
            (93, 104, "3. Ünite: Yer Bilimciler İş Başında", "Fosil Oluşumu", "Fosillerin Oluşumu ve Geçmişe Işık Tutması"),
            # 4. Ünite: Maddeyi Tanıyalım, Karıştırıp Ayıralım (Basılı Sayfa 105 - 150)
            (105, 117, "4. Ünite: Maddeyi Tanıyalım", "Maddenin Hâlleri", "Katı, Sıvı ve Gaz Maddeler"),
            (118, 126, "4. Ünite: Maddeyi Tanıyalım", "Karışımlar ve Ayrılması", "Eleme, Süzme ve Mıknatısla Ayırma"),
            (127, 150, "4. Ünite: Maddeyi Tanıyalım", "Atıkların Ayrıştırılması", "Geri Dönüşüm ve Çevre Bilinci"),
        ]
    },
    "ilkokul_fenbilimleri_3.sinif_2.kitap.pdf": {
        "kitap_adi": "MEB 3. Sınıf Fen Bilimleri (2. Kitap)",
        "sinif": "3. Sınıf",
        "ders": "Fen Bilimleri",
        "seviye": "İlkokul",
        "uniteler": [
            # 5. Ünite: Hareketi Keşfediyorum (Basılı Sayfa 13 - 40)
            (13, 28, "5. Ünite: Hareketi Keşfediyorum", "Varlıkların Hareketleri", "Hızlanma, Yavaşlama, Dönme ve Sallanma"),
            (29, 40, "5. Ünite: Hareketi Keşfediyorum", "Kuvvet ve Etkileri", "İtme, Çekme Kuvveti ve Cisimleri Harekete Geçirme"),
            # 6. Ünite: Yaşamımızı Kolaylaştıran Elektrik (Basılı Sayfa 41 - 72)
            (41, 51, "6. Ünite: Yaşamımızı Kolaylaştıran Elektrik", "Elektrikli Araç Gereç", "Elektrikli Araçların Kullanım Alanları"),
            (52, 60, "6. Ünite: Yaşamımızı Kolaylaştıran Elektrik", "Elektrikli Araçların Güvenli Kullanımı", "Elektrik Çarpması ve Güvenlik Önlemleri"),
            (61, 72, "6. Ünite: Yaşamımızı Kolaylaştıran Elektrik", "Elektriğin Tasarruflu Kullanımı", "Enerji Tasarrufu ve Doğal Kaynaklar"),
            # 7. Ünite: Toprağı Tanıyorum, Tarımı Keşfediyorum (Basılı Sayfa 73 - 98)
            (73, 83, "7. Ünite: Toprağı Tanıyorum", "Toprak Oluşumu ve Yapısı", "Toprağın Önemi ve Erozyon"),
            (84, 98, "7. Ünite: Toprağı Tanıyorum", "Bitki Yetiştirme", "Tohum Çimlenmesi ve Tarımsal Üretim"),
            # 8. Ünite: Canlıların Yaşam Alanlarına Yolculuk (Basılı Sayfa 99 - 150)
            (99, 109, "8. Ünite: Canlıların Yaşam Alanları", "Yaşam Alanları", "Kara, Su ve Hava Yaşam Alanları"),
            (110, 116, "8. Ünite: Canlıların Yaşam Alanları", "Canlı Çeşitliliği", "Biyoçeşitlilik ve Nesli Tükenen Canlılar"),
            (117, 150, "8. Ünite: Canlıların Yaşam Alanları", "Yaşam Alanlarının Korunması", "Doğal Çevreyi Koruma Bilinci"),
        ]
    },
    "ilkokul_fenbilimleri_4.sinif.pdf": {
        "kitap_adi": "MEB 4. Sınıf Fen Bilimleri",
        "sinif": "4. Sınıf",
        "ders": "Fen Bilimleri",
        "seviye": "İlkokul",
        "uniteler": [
            # 1. Ünite: Yer Kabuğu ve Dünya'mızın Hareketleri (Basılı Sayfa 20 - 45)
            (20, 32, "1. Ünite: Yer Kabuğu ve Hareketler", "Yer Kabuğunun Yapısı", "Kayaçlar, Madenler ve Fosiller"),
            (33, 45, "1. Ünite: Yer Kabuğu ve Hareketler", "Dünya'mızın Hareketleri", "Dönme ve Dolanma Hareketleri, Gece-Gündüz, Yıl"),
            # 2. Ünite: Besinlerimiz (Basılı Sayfa 46 - 74)
            (46, 58, "2. Ünite: Besinlerimiz", "Besinler ve Özellikleri", "Karbonhidrat, Protein, Yağ, Vitamin, Su ve Mineraller"),
            (59, 74, "2. Ünite: Besinlerimiz", "Sağlıklı Yaşam ve Zararlı Alışkanlıklar", "Dengeli Beslenme, Sigara ve Alkolün Zararları"),
            # 3. Ünite: Kuvvetin Etkileri (Basılı Sayfa 75 - 97)
            (75, 84, "3. Ünite: Kuvvetin Etkileri", "Kuvvetin Cisimler Üzerindeki Etkileri", "Hızlandırıcı, Yavaşlatıcı, Yön ve Şekil Değiştirici Kuvvet"),
            (85, 97, "3. Ünite: Kuvvetin Etkileri", "Mıknatısların Uyguladığı Kuvvet", "Mıknatıs Kutupları ve Günlük Yaşamdaki Kullanımı"),
            # 4. Ünite: Maddenin Özellikleri (Basılı Sayfa 98 - 139)
            (98, 102, "4. Ünite: Maddenin Özellikleri", "Maddeyi Niteleyen Özellikler", "Yüzen-Batan, Suyu Emen-Emmeyen Maddeler"),
            (103, 111, "4. Ünite: Maddenin Özellikleri", "Maddenin Ölçülebilir Özellikleri", "Kütle ve Hacim Ölçümü"),
            (112, 114, "4. Ünite: Maddenin Özellikleri", "Maddenin Hâlleri", "Katı, Sıvı ve Gaz Maddelerin Belirgin Özellikleri"),
            (115, 123, "4. Ünite: Maddenin Özellikleri", "Maddenin Isı Etkisiyle Değişimi", "Erime, Donma, Buharlaşma ve Hâl Değişimi"),
            (124, 139, "4. Ünite: Maddenin Özellikleri", "Saf Madde ve Karışımlar", "Karışımların Ayrılması ve Ekonomiye Katkısı"),
            # 5. Ünite: Aydınlatma ve Ses Teknolojileri (Basılı Sayfa 140 - 179)
            (140, 146, "5. Ünite: Aydınlatma ve Ses Teknolojileri", "Aydınlatma Teknolojileri", "Geçmişten Günümüze Işık Kaynakları"),
            (147, 154, "5. Ünite: Aydınlatma ve Ses Teknolojileri", "Uygun Aydınlatma ve Tasarruf", "Göz Sağlığı ve Enerji Tasarrufu"),
            (155, 159, "5. Ünite: Aydınlatma ve Ses Teknolojileri", "Işık Kirliliği", "Işık Kirliliğinin Olumsuz Etkileri ve Önlemler"),
            (160, 168, "5. Ünite: Aydınlatma ve Ses Teknolojileri", "Ses Teknolojileri", "Geçmişten Günümüze Ses Araçları"),
            (169, 179, "5. Ünite: Aydınlatma ve Ses Teknolojileri", "Ses Kirliliği", "Gürültü ve İnsan Sağlığına Etkileri"),
            # 6. Ünite: İnsan ve Çevre (Basılı Sayfa 180 - 197)
            (180, 197, "6. Ünite: İnsan ve Çevre", "Bilinçli Tüketici", "Su, Besin ve Elektrik Tasarrufu, Geri Dönüşüm"),
            # 7. Ünite: Basit Elektrik Devreleri (Basılı Sayfa 198 - 215)
            (198, 215, "7. Ünite: Basit Elektrik Devreleri", "Devre Elemanları ve Kurulumu", "Pil, Ampul, Anahtar, Bağlantı Kablosu ve Devre Şeması"),
        ]
    }
}


def get_book_metadata_and_unit(
    filename: str,
    phys_page_no: int,
    printed_page_no: int,
    raw_text: str
) -> Tuple[Dict[str, Any], str, str, str]:
    """
    Belirtilen kitap ve basılı sayfa numarası için 3 seviyeli metadata döner:
    (base_meta, unite, konu, alt_konu).
    Manifesto dışındaki bilinmeyen kitaplarda sessizce yanlış sınıf atamak yerine
    kontrollü doğrulama yapar.
    """
    manifest = PRIMARY_BOOK_MANIFESTS.get(filename)
    if manifest:
        base_meta = {
            "kitap_adi": manifest["kitap_adi"],
            "sinif": manifest["sinif"],
            "ders_adi": manifest["ders"],
            "egitim_seviyesi": manifest["seviye"],
            "source": filename
        }
        if printed_page_no <= 0:
            return base_meta, "Ön Sayfalar", "Kapak ve Tanıtım", "Kapak / Künye"

        for start_p, end_p, u_title, k_title, a_title in manifest["uniteler"]:
            if start_p <= printed_page_no <= end_p:
                return base_meta, u_title, k_title, a_title

        # Belirlenen aralıkların dışındaki sayfalar (İçindekiler, Sözlük, Ekler vb.)
        return base_meta, "Genel Ünite", "Kitap Ekleri ve Etkinlikler", "Genel Bölüm"

    # Bilinmeyen kitap kontrolü: Asla varsayılan '3. Sınıf' tahmin etme!
    raise ValueError(
        f"[HATA] '{filename}' için tanımlı bir manifesto bulunamadı! "
        f"Lütfen kitabı PRIMARY_BOOK_MANIFESTS içerisine ekleyin."
    )


# ---------------------------------------------------------------------------
# 5. CHUNK VE TOKENIZER YARDIMCILARI
# ---------------------------------------------------------------------------
def get_tokenizer():
    """Gemma BPE Tokenizer'ı yerel dosyadan (gemma_tokenizer.json) yükler."""
    try:
        from tokenizer_utils import get_gemma_tokenizer
        return get_gemma_tokenizer()
    except Exception:
        return None


def count_tokens(text: str, tokenizer=None) -> int:
    """Metnin Gemma BPE modelindeki gerçek token sayısını hesaplar."""
    if tokenizer is None:
        tokenizer = get_tokenizer()
    if tokenizer:
        try:
            return len(tokenizer.encode(text).tokens)
        except Exception:
            pass
    # Tokenizer yüklenemezse güvenli yaklaşık hesap
    return max(1, len(text) // 4)


def clean_pedagogical_text(text: str) -> str:
    """
    MEB ders kitabı sayfalarındaki öğrenci cevap çizgilerini,
    uzun nokta serilerini ve biçimlendirme gürültüsünü temizler.
    Eğitsel soru metinlerini, şıkları, Roma rakamlarını ve açıklamaları korur.
    """
    if not text:
        return ""

    lines = text.split("\n")
    cleaned_lines = []

    for line in lines:
        line_stripped = line.strip()
        if not line_stripped:
            continue

        # 1. Tamamen nokta, alt çizgi, tire veya boşluktan oluşan satırları temizle
        # (Öğrenci cevap satırları)
        non_noise = re.sub(r'[\s\.\_\-\–\—\•\*\t\r]', '', line_stripped)
        if len(non_noise) == 0:
            continue

        # 2. Satır içindeki 4 veya daha fazla ardışık nokta dizisini ' ... ' ile sadeleştir
        line_clean = re.sub(r'(?:\s*\.){4,}', ' ... ', line_stripped)

        # 3. Satır içindeki 3 veya daha fazla ardışık alt çizgi/tire dizisini sadeleştir
        line_clean = re.sub(r'_{3,}', ' ___ ', line_clean)
        line_clean = re.sub(r'-{3,}', ' --- ', line_clean)

        # 4. Fazla boşlukları tek boşluğa indir
        line_clean = re.sub(r'[ \t]+', ' ', line_clean).strip()

        if line_clean:
            cleaned_lines.append(line_clean)

    # 5. Art arda ikiden fazla boş satırları birleştir
    result = "\n".join(cleaned_lines)
    result = re.sub(r'\n{3,}', '\n\n', result)
    return result.strip()


def create_pedagogical_splitter(
    chunk_size: int = 1000,
    chunk_overlap: int = 150
) -> RecursiveCharacterTextSplitter:
    """
    Karakter tabanlı pedagojik metin parçalayıcı (1000 karakter, 150 karakter örtüşme).
    Fen deneyleri, soru kalıpları ve kavram açıklamalarını cümle bütünlüğünü koruyarak böler.
    """
    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=[
            "\n\n",
            "\n",
            ". ",
            "? ",
            "! ",
            "; ",
            ", ",
            " ",
            ""
        ]
    )


# ---------------------------------------------------------------------------
# 6. HATA TOLERANSLI VE DETAYLI LOGLAMALI BATCH YAZICI
# ---------------------------------------------------------------------------
def safe_add_documents_batch(
    store: Chroma,
    docs: List[Document],
    ids: List[str],
    max_retries: int = 2
) -> Tuple[int, List[Dict[str, Any]]]:
    """
    Belgeleri Chroma'ya güvenli ekler.
    len(docs) == len(ids) kontrolü yapar; uyumsuzlukta durur.
    Batch hata verirse bekleyip tekrar dener; yine hata verirse tekil (1'er 1'er)
    chunk ekleyerek sağlam olanları kurtarır, başarısız chunk'ların hata detayını raporlar.
    """
    if len(docs) != len(ids):
        raise ValueError(
            f"[HATA] Belge ve ID sayıları eşleşmiyor! len(docs)={len(docs)}, len(ids)={len(ids)}"
        )

    if not docs:
        return 0, []

    # 1. Aşama: Toplu Batch denemesi
    last_err: Optional[Exception] = None
    for attempt in range(1, max_retries + 1):
        try:
            store.add_documents(documents=docs, ids=ids)
            return len(docs), []
        except Exception as e:
            last_err = e
            if attempt < max_retries:
                time.sleep(1.5)

    # 2. Aşama: Batch başarısız oldu, tekil chunk fallback
    successful = 0
    failed_items: List[Dict[str, Any]] = []

    for d, cid in zip(docs, ids):
        try:
            store.add_documents(documents=[d], ids=[cid])
            successful += 1
        except Exception as single_err:
            failed_items.append({
                "chunk_id": cid,
                "error_type": type(single_err).__name__,
                "error_message": str(single_err)
            })

    return successful, failed_items


# ---------------------------------------------------------------------------
# 7. SÜRÜM TEMİZLİĞİ (ESKİ HASH'Lİ PARÇALARI SİLME)
# ---------------------------------------------------------------------------
def cleanup_stale_chunks(
    store: Chroma,
    filename: str,
    current_hash: str,
    target_version: str = ""
) -> int:
    """
    PDF dosyası değiştiğinde veya yeniden indekslendiğinde,
    aynı dosya adına ait eski hash'li, eski şemalı veya eski sürüm parçaları ChromaDB'den temizler.
    """
    try:
        col = store._collection
        res = col.get(where={"source": filename}, include=["metadatas"])
        if not res or not res.get("ids"):
            return 0

        target_hash_prefix = current_hash[:12]
        stale_ids = []
        for cid, meta in zip(res["ids"], res["metadatas"]):
            file_h = meta.get("file_hash", "")
            is_stale_hash = bool(file_h and file_h != target_hash_prefix)
            is_legacy_schema = bool(":p" in cid and ":phys_" not in cid)
            is_stale_version = bool(target_version and f":{target_version}:" not in cid)
            if is_stale_hash or is_legacy_schema or is_stale_version:
                stale_ids.append(cid)

        if stale_ids:
            col.delete(ids=stale_ids)
            return len(stale_ids)
    except Exception as e:
        print(f"[UYARI] cleanup_stale_chunks sırasında veritabanı hatası ({filename}): {e}")
        raise RuntimeError(f"ChromaDB eski sürüm temizliği ({filename}) başarısız oldu: {e}") from e
    return 0


# ---------------------------------------------------------------------------
# 8. METADATA JSON DIŞA AKTARIMI (books_meta.json)
# ---------------------------------------------------------------------------
def export_books_meta(store: Chroma, output_path: Optional[Path] = None) -> Path:
    """
    ChromaDB ve PDF dosyalarından derlenen tüm kitap manifestolarını,
    token sayılarını, chunk dağılımlarını ve dosya hash'lerini 'data/books_meta.json' dosyasına yazar.
    """
    import json
    from datetime import datetime
    try:
        from config import DATA_DIR, REFERENCE_BOOKS_DIR, COLLECTION, EMBED_MODEL
    except ImportError:
        ROOT = Path(__file__).resolve().parent
        DATA_DIR = ROOT / "data"
        REFERENCE_BOOKS_DIR = DATA_DIR / "reference_books"
        COLLECTION = "eduframe_meb_768"
        EMBED_MODEL = "embeddinggemma"

    out_file = output_path or (DATA_DIR / "books_meta.json")

    col = store._collection
    res = col.get(include=["metadatas"])

    meta_by_book: Dict[str, Dict[str, Any]] = {}
    for m in res.get("metadatas", []):
        src = m.get("source", "Bilinmeyen")
        tc = int(m.get("token_count", 0))
        if src not in meta_by_book:
            meta_by_book[src] = {"chunks": 0, "tokens": 0, "min_tokens": 999999, "max_tokens": 0}
        meta_by_book[src]["chunks"] += 1
        meta_by_book[src]["tokens"] += tc
        meta_by_book[src]["min_tokens"] = min(meta_by_book[src]["min_tokens"], tc)
        meta_by_book[src]["max_tokens"] = max(meta_by_book[src]["max_tokens"], tc)

    export_data = {
        "veritabani": {
            "koleksiyon_adi": col.name or COLLECTION,
            "embedding_modeli": EMBED_MODEL,
            "embedding_boyutu": 768,
            "toplam_chunk": len(res.get("metadatas", [])),
            "toplam_token": sum(b["tokens"] for b in meta_by_book.values()),
            "son_guncelleme": datetime.now().isoformat(),
        },
        "kitaplar": {}
    }

    for fname, manifest in PRIMARY_BOOK_MANIFESTS.items():
        fpath = REFERENCE_BOOKS_DIR / fname
        if not fpath.exists():
            continue
        try:
            reader = pypdf.PdfReader(str(fpath))
            p_count = len(reader.pages)
            offset = detect_page_offset(reader)
        except Exception:
            p_count = 0
            offset = 1

        f_hash = calculate_file_hash(fpath)
        stats = meta_by_book.get(fname, {"chunks": 0, "tokens": 0, "min_tokens": 0, "max_tokens": 0})

        export_data["kitaplar"][fname] = {
            "kitap_adi": manifest.get("kitap_adi", fname),
            "sinif": manifest.get("sinif", ""),
            "ders": manifest.get("ders", ""),
            "egitim_seviyesi": manifest.get("seviye", ""),
            "dosya_sha256": f_hash,
            "dosya_boyutu_mb": round(fpath.stat().st_size / (1024 * 1024), 2),
            "fiziksel_sayfa_sayisi": p_count,
            "sayfa_ofseti": offset,
            "chunk_sayisi": stats["chunks"],
            "toplam_token": stats["tokens"],
            "ortalama_token": round(stats["tokens"] / stats["chunks"], 1) if stats["chunks"] else 0,
            "token_araligi": {"min": stats["min_tokens"] if stats["chunks"] else 0, "max": stats["max_tokens"]},
            "uniteler": [
                {
                    "basili_sayfa_baslangic": u[0],
                    "basili_sayfa_bitis": u[1],
                    "unite_basligi": u[2],
                    "konu_basligi": u[3],
                    "alt_konu_basligi": u[4]
                } for u in manifest.get("uniteler", [])
            ]
        }

    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(export_data, f, ensure_ascii=False, indent=2)

    return out_file
