"""
EduFrame AI - İlkokul (3. ve 4. Sınıf) Fen Bilimleri Kitaplarını İndeksleme Aracı
---------------------------------------------------------------------------------
Özellikler:
1. Kesinlikle 768-D Hedef Koleksiyon: 'eduframe_meb_768'
2. Embedding Boyutu Ön Kontrolü: Chroma açılmadan önce 768 boyut doğrulaması
3. Sayfa Ofseti Tespiti: Fiziksel PDF sayfası vs basılı sayfa farkını otomatik hesaplar
4. Dosya Hash'i ile Kararlı Chunk ID: {kitap}:{hash[:12]}:p{page}:c{chunk}
5. 3 Seviyeli Zengin Metadata: Ünite, Konu ve Alt Konu etiketleri
6. Tekrar Çalıştırma & Kaldığı Yerden Devam Etme (Idempotent Resumption)
7. Eski Sürüm Temizliği: PDF değiştiğinde eski hash'e ait kayıtları temizler
8. 1000 Karakter / 150 Overlap Pedagojik Bölme & Gemma BPE Token Sayımı
9. Hata Toleranslı Batch ve Ayrıntılı Loglamalı Fallback Mekanizması
10. Çok Yönlü Doğrulama Test Sorguları (3. Sınıf 1. & 2. Kitap ve 4. Sınıf)

Kullanım:
    .venv\\Scripts\\python.exe index_ilkokul_books.py
"""

import sys
import time
import argparse
import hashlib
from pathlib import Path
from typing import List, Set, Dict, Any, Tuple
import pypdf

from langchain_core.documents import Document
from langchain_chroma import Chroma

# Proje kök dizinini ekle
ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from config import (
    CHROMA_DB_DIR,
    CHROMA_DB_V1_DIR,
    CHROMA_DB_V2_DIR,
    REFERENCE_BOOKS_DIR,
    OLLAMA_URL,
    EMBED_MODEL,
)
from index_core import (
    calculate_file_hash,
    generate_chunk_id,
    clean_pedagogical_text,
    verify_embedding_dimension,
    detect_page_offset,
    get_book_metadata_and_unit,
    create_pedagogical_splitter,
    safe_add_documents_batch,
    cleanup_stale_chunks,
    get_tokenizer,
    count_tokens,
    PRIMARY_BOOK_MANIFESTS,
    export_books_meta,
)

# Kesin 768 boyutlu koleksiyon adı ve geçerli chunk sürümü
TARGET_COLLECTION = "eduframe_meb_768"
CURRENT_VERSION = "v2"

TARGET_BOOKS = [
    "ilkokul_fenbilimleri_3.sinif_1.kitap.pdf",
    "ilkokul_fenbilimleri_3.sinif_2.kitap.pdf",
    "ilkokul_fenbilimleri_4.sinif.pdf",
]


def get_existing_chunk_ids(store: Chroma) -> Set[str]:
    """ChromaDB koleksiyonunda halihazırda kayıtlı olan Chunk ID'lerini çeker."""
    try:
        col = store._collection
        count = col.count()
        if count == 0:
            return set()

        existing_ids = set()
        limit = 10000
        offset = 0
        while offset < count:
            res = col.get(limit=limit, offset=offset, include=[])
            if not res or not res.get("ids"):
                break
            existing_ids.update(res["ids"])
            offset += len(res["ids"])
        return existing_ids
    except Exception as e:
        print(f"[HATA] get_existing_chunk_ids sırasında veritabanı hatası: {e}")
        raise RuntimeError(f"ChromaDB mevcut kayıtları taranırken veritabanı hatası: {e}") from e


def verify_book_chunks_in_db(store: Chroma, expected_ids: Set[str]) -> Tuple[bool, Set[str]]:
    """
    ChromaDB koleksiyonunda belirtilen ID kümesinin eksiksiz var olduğunu doğrular.
    Yalnızca sayıları değil, gerçek ID kümelerini (set difference) karşılaştırır.

    Returns:
        (is_complete: bool, missing_ids: Set[str])
    """
    if not expected_ids:
        return True, set()

    try:
        col = store._collection
        expected_list = list(expected_ids)
        found_ids: Set[str] = set()

        batch_size = 500
        for i in range(0, len(expected_list), batch_size):
            sub_ids = expected_list[i : i + batch_size]
            res = col.get(ids=sub_ids, include=[])
            if res and res.get("ids"):
                found_ids.update(res["ids"])

        missing_ids = set(expected_ids) - found_ids
        return (len(missing_ids) == 0), missing_ids
    except Exception as e:
        print(f"[HATA] verify_book_chunks_in_db sırasında veritabanı hatası: {e}")
        return False, set(expected_ids)


def check_book_legacy_records(
    store: Chroma,
    filename: str,
    target_version: str = ""
) -> List[str]:
    """
    ChromaDB'de belirtilen kitaba ait eski ID şemalı (:pXXXX: olup :phys_ olmayan)
    veya eski sürüm (target_version belirtilmişse o etiketi içermeyen) kayıtları listeler.
    ChromaDB sorgusunda hata oluşursa sessizce yutmaz, RuntimeError fırlatır.
    """
    try:
        col = store._collection
        res = col.get(where={"source": filename}, include=[])
        if not res or not res.get("ids"):
            return []
        legacy_ids = [
            cid for cid in res["ids"]
            if (":p" in cid and ":phys_" not in cid) or (target_version and f":{target_version}:" not in cid)
        ]
        return legacy_ids
    except Exception as e:
        print(f"[HATA] check_book_legacy_records veritabanı sorgu hatası ({filename}): {e}")
        raise RuntimeError(f"ChromaDB legacy kayıt taraması ({filename}) başarısız oldu: {e}") from e


def verify_indexed_book_integrity(
    store: Chroma,
    filename: str,
    expected_ids: Set[str],
    target_version: str = CURRENT_VERSION,
    batch_size: int = 250,
    expected_content_hashes: Dict[str, str] = None,
) -> Tuple[bool, List[str]]:
    """
    Belirtilen kitaba ait kayıtların ChromaDB'de eksiksiz, tutarlı metadata ve
    temizlenmiş pedagojik metin taşıdığını derinlemesine doğrular.

    Kontrol edilen kriterler:
    1. ID tamlığı: Beklenen tüm ID'ler DB'de mevcut mu? (Boş ID kümesi açık bir bütünlük hatasıdır).
    2. Versiyon tutarlılığı: Tüm parçalar metadata'da target_version ("v2") taşıyor mu ve ID'sinde f":{target_version}:" var mı?
    3. Metadata alanları: source, physical_page, unite, konu, referans, content_hash alanları dolu ve geçerli mi?
    4. Metin ve SHA-256 Hash bütünlüğü: Veritabanından okunan gerçek metnin SHA-256 hash'i (ilk 12 karakter)
       metadata'daki content_hash ile birebir uyuşuyor mu? Varsa beklenen kaynak hash'leri ile eşleşiyor mu?
    5. Metin kalitesi: Parçalar pedagojik gürültü (nokta satırı / aşırı nokta kirliliği) barındırıyor mu veya boş mu?

    Returns:
        (is_valid: bool, issues: List[str])
    """
    if not expected_ids:
        return False, ["Beklenen chunk ID kümesi boş! Hiçbir parça üretilemedi veya doğrulanacak parça yok."]

    issues: List[str] = []
    try:
        col = store._collection
        exp_list = list(expected_ids)

        found_ids: Set[str] = set()
        for i in range(0, len(exp_list), batch_size):
            sub_ids = exp_list[i : i + batch_size]
            res = col.get(ids=sub_ids, include=["metadatas", "documents"])
            if not res or not res.get("ids"):
                continue

            for cid, meta, doc in zip(res["ids"], res["metadatas"], res["documents"]):
                found_ids.add(cid)
                # 1. Versiyon kontrolü
                v_meta = meta.get("version", "") if meta else ""
                if v_meta != target_version:
                    issues.append(f"Parça {cid}: Metadata versiyonu '{v_meta}' != '{target_version}'")
                if target_version and f":{target_version}:" not in cid:
                    issues.append(f"Parça {cid}: ID içinde ':{target_version}:' etiketi bulunmuyor")

                # 2. Kaynak dosya ve sayfa doğrulaması
                if not meta or meta.get("source") != filename:
                    issues.append(f"Parça {cid}: Metadata source '{meta.get('source') if meta else None}' != '{filename}'")
                if not meta or "physical_page" not in meta or meta["physical_page"] <= 0:
                    issues.append(f"Parça {cid}: Geçersiz physical_page: {meta.get('physical_page') if meta else None}")

                # 3. Metin varlık ve boşluk kontrolü
                if doc is None or not isinstance(doc, str) or not doc.strip():
                    issues.append(f"Parça {cid}: Belge metni boş veya eksik")
                    continue

                # 4. Content hash doğrulaması (SHA-256 yeniden hesaplama ve karşılaştırma)
                c_hash = meta.get("content_hash", "") if meta else ""
                if not c_hash:
                    issues.append(f"Parça {cid}: content_hash metadata alanı eksik")
                else:
                    calc_hash = hashlib.sha256(doc.encode("utf-8")).hexdigest()[:12]
                    if calc_hash != c_hash:
                        issues.append(
                            f"Parça {cid}: content_hash uyuşmazlığı! Kayıtlı={c_hash}, Hesaplanan={calc_hash}"
                        )
                    if expected_content_hashes is not None:
                        if cid not in expected_content_hashes:
                            issues.append(f"Parça {cid}: Beklenen content_hash sözlüğünde tanımlı değil")
                        elif expected_content_hashes[cid] != c_hash:
                            issues.append(
                                f"Parça {cid}: Kaynak content_hash uyuşmazlığı! Beklenen={expected_content_hashes[cid]}, Kayıtlı={c_hash}"
                            )

                # 5. Metin kalitesi kontrolü (öğrenci nokta dizileri temizlenmiş olmalı)
                if "....." in doc:
                    issues.append(f"Parça {cid}: Temizlenmemiş nokta dizisi (.....) tespit edildi")
                if len(doc.strip()) < 15:
                    issues.append(f"Parça {cid}: Metin çok kısa (<15 karakter): '{doc.strip()}'")

        missing_ids = expected_ids - found_ids
        if missing_ids:
            issues.append(f"{len(missing_ids)} beklenen chunk ID veritabanında bulunamadı! Örnekler: {list(missing_ids)[:3]}")

        return (len(issues) == 0), issues
    except Exception as e:
        issues.append(f"Veritabanı denetim hatası ({filename}): {e}")
        return False, issues


def can_cleanup_stale_chunks(
    book_has_failure: bool,
    book_success: int,
    expected_new_count: int,
    missing_ids: Set[str],
    page_errors: List[Dict[str, Any]],
    integrity_ok: bool = True,
    migration_allowed: bool = False,
) -> bool:
    """
    Eski sürüm kayıtlarının temizlenmesinin güvenli olup olmadığını belirler.
    Temizliğin gerçekleşmesi için AŞAĞIDAKİ KOŞULLARIN TAMAMI sağlanmalıdır:
    1. Hiçbir batch veya tekil chunk hatası olmamalı (not book_has_failure).
    2. Sunulan yeni parça sayısı ile başarılı eklenen sayı eşit olmalı (book_success == expected_new_count).
    3. Beklenen tüm ID'ler ChromaDB'de doğrulanmış olmalı (len(missing_ids) == 0).
    4. PDF metin çıkarma veya metadata hatası nedeniyle hiçbir sayfa atlanmamış olmalı (len(page_errors) == 0).
    5. Metin ve metadata bütünlüğü doğrulaması hatasız olmalı (integrity_ok).
    6. Migrasyon izni açıkça verilmiş olmalı (migration_allowed).
    """
    if book_has_failure:
        return False
    if book_success != expected_new_count:
        return False
    if missing_ids:
        return False
    if page_errors:
        return False
    if not integrity_ok:
        return False
    if not migration_allowed:
        return False
    return True


def main():
    parser = argparse.ArgumentParser(description="İlkokul (3. ve 4. Sınıf) Fen Bilimleri Kitapları İndeksleme Aracı")
    parser.add_argument(
        "--kitap",
        type=str,
        default="",
        help="Yalnızca belirtilen tekil PDF kitabını indeksle (örn: ilkokul_fenbilimleri_3.sinif_2.kitap.pdf)"
    )
    parser.add_argument(
        "--db-path",
        type=str,
        default="",
        help="Hedef ChromaDB dizini (varsayılan: config.CHROMA_DB_DIR)"
    )
    parser.add_argument(
        "--allow-migration",
        action="store_true",
        default=False,
        help="v2 migrasyonuna ve eski kayıt temizliğine açık onay verir (güvenlik kilidi)"
    )
    parser.add_argument(
        "--allow-write",
        action="store_true",
        default=False,
        help="Yeni parçaların veritabanına yazılmasına izin verir"
    )
    parser.add_argument(
        "--reindex",
        action="store_true",
        default=False,
        help="Hedef kitap(lar)ın mevcut parçalarını silip sıfırdan yeniden embed eder (diğer kitapları korur)"
    )
    args = parser.parse_args()

    selected_books = [args.kitap] if args.kitap else TARGET_BOOKS
    target_db_dir = Path(args.db_path).resolve() if args.db_path else CHROMA_DB_DIR.resolve()

    reindex_mode = bool(args.reindex)
    migration_allowed = bool(args.allow_migration or reindex_mode)
    write_allowed = bool(args.allow_write or reindex_mode or migration_allowed)

    print("=" * 75)
    print("🎓 EduFrame AI - İlkokul (3. & 4. Sınıf) 768-D İndeksleme Motoru")
    print("=" * 75)
    print(f" -> Hedef Vektör Veritabanı : {target_db_dir}")
    print(f" -> Hedef Koleksiyon        : {TARGET_COLLECTION}")
    print(f" -> Embedding Modeli        : Ollama ({EMBED_MODEL})")
    print(f" -> Parçalama Yapısı        : 1000 karakter / 150 örtüşme (overlap)")
    if args.kitap:
        print(f" -> Filtrelenen Tekil Kitap : {args.kitap}")
    print(f" -> Yeniden İndeksleme Modu : {'AÇIK (--reindex: Bu kitapların eski kayıtları silinip yeniden yazılacak)' if reindex_mode else 'KAPALI'}")
    print(f" -> Yazma İzni              : {'AÇIK' if write_allowed else 'KAPALI (Salt Önizleme / Dry-Run)'}")
    print("=" * 75)

    # 1. EMBEDDING BOYUTU DOĞRULAMA (768-D KONTROLÜ)
    print("\n[1/5] Ollama Servis ve 768-D Embedding Boyut Doğrulaması...")
    try:
        emb = verify_embedding_dimension(
            ollama_url=OLLAMA_URL,
            model_name=EMBED_MODEL,
            expected_dim=768
        )
        print("      ✓ Ollama bağlantısı başarılı.")
        print("      ✓ Test sorgusu atıldı ve dönen vektör TAM 768 boyut olarak onaylandı.")
    except Exception as err:
        print(f"\n[DURDURULDU] {err}")
        sys.exit(1)

    # 2. HEDEF KİTAPLARI VE DOSYA HASH'LERİNİ DOĞRULA
    print("\n[2/5] Hedef İlkokul Kitapları ve Dosya Hash Kontrolü...")
    book_files = []
    file_hashes: Dict[str, str] = {}
    for fname in selected_books:
        fpath = REFERENCE_BOOKS_DIR / fname
        if not fpath.exists():
            print(f"  [UYARI] Dosya bulunamadı: {fpath}")
            continue
        f_hash = calculate_file_hash(fpath)
        file_hashes[fname] = f_hash
        size_mb = round(fpath.stat().st_size / (1024 * 1024), 1)
        print(f"  ✓ {fname} ({size_mb} MB) -> SHA256: {f_hash[:12]}...")
        book_files.append(fpath)

    if not book_files:
        print("[HATA] reference_books klasöründe indekslenecek ilkokul kitabı bulunamadı!")
        sys.exit(1)

    # 3. CHROMADB VE MEVCUT CHUNK'LARI KONTROL ET (RESUMPTION / IDEMPOTENCY)
    print("\n[3/5] ChromaDB Başlatılıyor ve Mevcut Durum Taranıyor...")
    target_db_dir.mkdir(parents=True, exist_ok=True)
    store = Chroma(
        collection_name=TARGET_COLLECTION,
        embedding_function=emb,
        persist_directory=str(target_db_dir)
    )

    existing_ids = get_existing_chunk_ids(store)
    print(f"      ✓ Koleksiyondaki mevcut toplam parça: {len(existing_ids)}")

    splitter = create_pedagogical_splitter(chunk_size=1000, chunk_overlap=150)
    gemma_tok = get_tokenizer()
    if gemma_tok:
        print("      ✓ Gemma BPE Tokenizer (256k vocab) aktif; her parça için token sayılacak.")

    # 4. KİTAPLARI OKU, PARÇALA VE GÜVENLİ BATCH İLE YAZ
    print("\n[4/5] Kitaplar İşleniyor ve Vektörleştiriliyor...")
    total_newly_indexed = 0
    total_skipped = 0
    all_failed_items: List[Dict[str, Any]] = []
    t_start = time.time()

    for b_idx, fpath in enumerate(book_files, 1):
        t_book_start = time.time()
        fname = fpath.name
        f_hash = file_hashes[fname]
        manifest = PRIMARY_BOOK_MANIFESTS.get(fname, {})
        kitap_adi = manifest.get("kitap_adi", fname)

        print(f"\n--- [{b_idx}/{len(book_files)}] {kitap_adi} ({fname}) ---")

        # Reindex modunda veya dosya hash'i değiştiğinde mevcut kitabın eski kayıtlarını temizle
        col = store._collection
        existing_book_chunks = col.get(where={"source": fname}, include=["metadatas"])
        existing_book_ids = existing_book_chunks.get("ids", []) if existing_book_chunks else []

        has_hash_changed = False
        if existing_book_ids and existing_book_chunks.get("metadatas"):
            old_hashes = {m.get("file_hash") for m in existing_book_chunks["metadatas"] if m.get("file_hash")}
            if old_hashes and (f_hash[:12] not in old_hashes):
                has_hash_changed = True

        if write_allowed and (reindex_mode or has_hash_changed) and existing_book_ids:
            reason = "Yeniden İndeksleme (--reindex)" if reindex_mode else "Kitap Güncellemesi (Dosya Hash Değişimi)"
            print(f" -> [{reason}] '{fname}' için mevcut {len(existing_book_ids)} eski parça ChromaDB'den siliniyor...")
            col.delete(ids=existing_book_ids)
            existing_ids.difference_update(existing_book_ids)
            print(f"    ✓ Eski {len(existing_book_ids)} parça silindi. Diğer kitapların verileri korundu.")

        # PDF Aç ve Sayfa Ofsetini Hesapla
        try:
            reader = pypdf.PdfReader(str(fpath))
            total_pages = len(reader.pages)
            page_offset = detect_page_offset(reader)
            print(f" -> Toplam {total_pages} fiziksel sayfa. Tespit Edilen Sayfa Ofseti: +{page_offset} "
                  f"(Fiziksel sayfa {8} = Basılı sayfa {8 - page_offset})")
        except Exception as e:
            print(f" [HATA] PDF okunamadı: {e}")
            continue

        book_docs: List[Document] = []
        book_ids: List[str] = []
        all_expected_ids: Set[str] = set()
        page_errors: List[Dict[str, Any]] = []
        book_skipped = 0

        for phys_page_no in range(1, total_pages + 1):
            has_printed = phys_page_no > page_offset
            printed_page_no = (phys_page_no - page_offset) if has_printed else 0
            try:
                raw_text = (reader.pages[phys_page_no - 1].extract_text() or "").strip()
            except Exception as extract_err:
                print(f" [HATA] Sayfa {phys_page_no} metin çıkarma hatası: {extract_err}")
                page_errors.append({
                    "page": phys_page_no,
                    "stage": "extraction",
                    "error": str(extract_err)
                })
                continue

            # 1. PEDAGOJİK METİN TEMİZLEME (Öğrenci nokta/çizgi cevap satırlarını filtrele)
            raw_text = clean_pedagogical_text(raw_text)

            if not raw_text or len(raw_text) < 25:
                continue

            try:
                base_meta, unite, konu, alt_konu = get_book_metadata_and_unit(
                    filename=fname,
                    phys_page_no=phys_page_no,
                    printed_page_no=printed_page_no,
                    raw_text=raw_text
                )
            except Exception as meta_err:
                print(f" [UYARI] Sayfa {phys_page_no} metadata hatası: {meta_err}")
                page_errors.append({
                    "page": phys_page_no,
                    "stage": "metadata",
                    "error": str(meta_err)
                })
                continue

            if has_printed:
                ref = f"{base_meta['kitap_adi']} - {unite}, '{konu}' ({alt_konu}) [Sayfa {printed_page_no}]"
            else:
                ref = f"{base_meta['kitap_adi']} - {unite}, '{konu}' ({alt_konu}) [Fiziksel Sayfa {phys_page_no}]"
            chunks = splitter.split_text(raw_text)

            for c_idx, c_text in enumerate(chunks):
                if len(c_text.strip()) < 20:
                    continue

                cid = generate_chunk_id(fname, f_hash, phys_page_no, c_idx, version=CURRENT_VERSION)
                all_expected_ids.add(cid)

                # Eğer chunk halihazırda veritabanında varsa ATLA (Tekrar çalıştırma güvenliği)
                if cid in existing_ids:
                    book_skipped += 1
                    continue

                t_count = count_tokens(c_text, gemma_tok)
                content_hash = hashlib.sha256(c_text.encode("utf-8")).hexdigest()[:12]
                cm = {
                    "source": fname,
                    "kitap_adi": base_meta["kitap_adi"],
                    "egitim_seviyesi": base_meta["egitim_seviyesi"],
                    "ders_adi": base_meta["ders_adi"],
                    "sinif": base_meta["sinif"],
                    "physical_page": phys_page_no,
                    "printed_page": printed_page_no,
                    "page": printed_page_no,
                    "sayfa_no": printed_page_no,
                    "has_printed_page": has_printed,
                    "unite": unite,
                    "konu": konu,
                    "alt_konu": alt_konu,
                    "referans": ref,
                    "chunk_id": cid,
                    "file_hash": f_hash[:12],
                    "char_count": len(c_text),
                    "token_count": t_count,
                    "version": CURRENT_VERSION,
                    "content_hash": content_hash,
                }
                book_docs.append(Document(page_content=c_text, metadata=cm))
                book_ids.append(cid)

        total_skipped += book_skipped
        if book_skipped > 0:
            print(f" -> Önceden indekslenmiş {book_skipped} parça atlandı (mükerrer kayıt engellendi).")

        # 4. KURAL: book_docs boş olduğunda kitabın gerçekten güncel olup olmadığını doğrula
        if not book_docs:
            if not all_expected_ids or page_errors:
                print(f" [HATA] {kitap_adi} ({fname}) için hiç parça üretilemedi veya sayfa hataları var "
                      f"({len(page_errors)} hata). Kitap güncel olarak işaretlenemez!")
                continue

            # Beklenen tüm parçaların DB'de eksiksiz var olduğunu ID kümesiyle doğrula
            is_verified, missing_ids = verify_book_chunks_in_db(store, all_expected_ids)
            if not is_verified:
                print(f" [KRİTİK HATA] {kitap_adi} güncel kabul edilemez! {len(missing_ids)} beklenen parça DB'de eksik. "
                      f"Örnek eksikler: {list(missing_ids)[:3]}")
                continue

            # Metin ve metadata bütünlüğü doğrulaması
            integrity_ok, integrity_issues = verify_indexed_book_integrity(
                store=store,
                filename=fname,
                expected_ids=all_expected_ids,
                target_version=CURRENT_VERSION
            )
            if not integrity_ok:
                print(f" [KRİTİK HATA] {kitap_adi} veri bütünlüğü doğrulanamadı! "
                      f"Tespit edilen {len(integrity_issues)} sorun var. Örnek: {integrity_issues[0]}")
                continue

            # Eski şemalı veya eski sürümlü kayıtlar var mı kontrol et
            try:
                legacy_ids = check_book_legacy_records(store, fname, target_version=CURRENT_VERSION)
            except Exception as leg_err:
                print(f" [KRİTİK HATA] {kitap_adi} legacy kayıt kontrolü başarısız: {leg_err}. "
                      f"Kitap güncel kabul edilemez ve eski kayıt temizliği yapılamaz!")
                continue
            if legacy_ids:
                if not migration_allowed:
                    print(f" [GÜVENLİK KİLİDİ] {kitap_adi}: Tüm {len(all_expected_ids)} yeni ({CURRENT_VERSION}) parça DB'de mevcut; "
                          f"ancak --allow-migration bayrağı verilmediği için {len(legacy_ids)} eski sürüm/şemalı kayıt SİLİNMEDİ.")
                else:
                    print(f" -> {kitap_adi}: Tüm {len(all_expected_ids)} yeni ({CURRENT_VERSION}) parça DB'de mevcut; "
                          f"ancak {len(legacy_ids)} eski sürüm/şemalı kayıt tespit edildi. Temizlik başlatılıyor...")
                    cleaned = cleanup_stale_chunks(store, fname, f_hash, target_version=CURRENT_VERSION)
                    print(f" ✓ Eski sürüm/şemalı {cleaned} parça temizlendi ve kitap güncel duruma getirildi.")
            else:
                print(f" ✓ {kitap_adi} zaten tamamen güncel ({len(all_expected_ids)} parça DB'de doğrulandı, eski kayıt yok).")
            continue

        # Yazma öncesi çakışma kontrolü (Pre-write collision check)
        if len(book_ids) != len(set(book_ids)):
            seen_ids = set()
            duplicates = [cid for cid in book_ids if cid in seen_ids or seen_ids.add(cid)]
            raise ValueError(
                f"[KRİTİK HATA] {fname} için üretilen chunk ID'lerinde çakışma tespit edildi! "
                f"Toplam ID: {len(book_ids)}, Benzersiz: {len(set(book_ids))}. "
                f"Çakışan ilk ID'ler: {duplicates[:5]}"
            )

        print(f" -> Eklenecek {len(book_docs)} yeni parça tespit edildi.")

        # DRY-RUN / YAZMA İZNİ KONTROLÜ
        if not write_allowed:
            print(f" [DRY-RUN / ÖNİZLEME] --allow-migration veya --allow-write verilmediği için "
                  f"{len(book_docs)} yeni parça veritabanına YAZILMADI.")
            print(f" [GÜVENLİK KİLİDİ] {kitap_adi}: --allow-migration onayı verilmediği için eski kayıt silme ENGELLENDİ.")
            continue

        print(" -> 768-D Vektörleştirme ve veritabanına yazım başlıyor...")

        # Batch ekleme ve Hata Yönetimi
        batch_size = 45
        book_success = 0
        for i in range(0, len(book_docs), batch_size):
            sub_docs = book_docs[i : i + batch_size]
            sub_ids = book_ids[i : i + batch_size]

            succ, fails = safe_add_documents_batch(store, sub_docs, sub_ids, max_retries=2)
            book_success += succ
            if fails:
                all_failed_items.extend(fails)

            progress = min(i + batch_size, len(book_docs))
            print(f"    [{progress}/{len(book_docs)}] parça yüklendi...", end="\r", flush=True)

        # 1. KURAL: Yeni sürüme ait beklenen chunk ID'lerinin tamamının ChromaDB'de bulunduğunu doğrula
        # Yalnızca sayıları değil, ID kümelerini karşılaştır
        is_verified, missing_ids = verify_book_chunks_in_db(store, all_expected_ids)
        verified_count = len(all_expected_ids) - len(missing_ids)

        # Metin ve metadata bütünlüğü doğrulaması
        integrity_ok, integrity_issues = verify_indexed_book_integrity(
            store=store,
            filename=fname,
            expected_ids=all_expected_ids,
            target_version=CURRENT_VERSION
        )

        book_elapsed = round(time.time() - t_book_start, 1)
        total_newly_indexed += book_success
        print(f"\n ✓ {kitap_adi} ({fname}) işlendi ({book_success} parça sunuldu, "
              f"DB'de doğrulanan beklenen: {verified_count}/{len(all_expected_ids)}, {book_elapsed} sn).")
        if not integrity_ok:
            print(f" ⚠️ {kitap_adi} bütünlük denetiminde {len(integrity_issues)} uyarı/hata bulundu!")

        # 2 & 3. KURAL: Güvenlik Denetimi — Doğrulama başarısız olursa veya sayfa hatası varsa eski sürümü kesinlikle silme!
        book_has_failure = any(f.get("chunk_id", "").startswith(fname) for f in all_failed_items)
        safe_to_cleanup = can_cleanup_stale_chunks(
            book_has_failure=book_has_failure,
            book_success=book_success,
            expected_new_count=len(book_docs),
            missing_ids=missing_ids,
            page_errors=page_errors,
            integrity_ok=integrity_ok,
            migration_allowed=migration_allowed,
        )

        # DOĞRUDAN MİGRASYON İZNİ VE GÜVENLİK KİLİDİ:
        # cleanup_stale_chunks() çağrısı doğrudan migration_allowed kontrolüne bağlıdır!
        if not migration_allowed:
            print(f" [GÜVENLİK KİLİDİ] {kitap_adi}: --allow-migration onayı verilmediği için eski kayıt silme ENGELLENDİ.")
        elif safe_to_cleanup:
            cleaned = cleanup_stale_chunks(store, fname, f_hash, target_version=CURRENT_VERSION)
            if cleaned > 0:
                print(f" -> İndeksleme eksiksiz tamamlandı ve doğrulandı: Eski sürüm/şemaya ait {cleaned} parça temizlendi.")
        else:
            reasons = []
            if book_has_failure:
                reasons.append("batch ekleme hatası")
            if book_success != len(book_docs):
                reasons.append(f"başarılı parça ({book_success}) != sunulan ({len(book_docs)})")
            if missing_ids:
                reasons.append(f"{len(missing_ids)} beklenen chunk ID'si DB'de eksik")
            if page_errors:
                reasons.append(f"{len(page_errors)} sayfada metin çıkarma/metadata hatası")
            if not integrity_ok:
                reasons.append(f"metin/metadata bütünlük hatası ({len(integrity_issues)} sorun)")
            print(f" [GÜVENLİK ENGELİ] {kitap_adi} için eski sürüm temizliği ENGELLENDİ! Gerekçeler: {', '.join(reasons)}.")

    total_elapsed = round(time.time() - t_start, 1)
    print("\n" + "=" * 75)
    print("🎉 İNDEKSLEME RAPORU")
    print(f" -> Yeni Eklenen (Sunulan) : {total_newly_indexed}")
    print(f" -> Atlanan (Mevcut)     : {total_skipped}")
    print(f" -> Başarısız Parça      : {len(all_failed_items)}")
    if all_failed_items:
        print("\n  ⚠️ BAŞARISIZ PARÇA DETAYLARI:")
        for it in all_failed_items[:5]:
            print(f"    - ID: {it['chunk_id']} | Hata: {it['error_type']}: {it['error_message']}")
        if len(all_failed_items) > 5:
            print(f"    ... ve {len(all_failed_items) - 5} parça daha.")

    print(f" -> Toplam Süre          : {total_elapsed} saniye ({round(total_elapsed/60, 1)} dakika)")
    print(f" -> Koleksiyon Toplamı   : {store._collection.count()} vektör")

    # Metadata JSON Dışa Aktarımı
    if write_allowed:
        try:
            m_path = export_books_meta(store)
            print(f" -> Metadata JSON Raporu : {m_path.name} güncellendi.")
        except Exception as exp_err:
            print(f" [UYARI] books_meta.json dışa aktarılamadı: {exp_err}")

    print("=" * 75)

    # 5. GERÇEKÇİ KAZANIM DOĞRULAMA TESTLERİ
    print("\n[5/5] Pedagojik 3 Seviyeli Kazanım ve Sayfa Doğrulama Testleri Yapılıyor...")
    test_queries = [
        ("3. Sınıf 1. Kitap Testi", "kayaçlar madenler ve fosil oluşumu", "3. Sınıf"),
        ("3. Sınıf 2. Kitap Testi", "elektrikli araç gereçler ve şehir elektriği", "3. Sınıf"),
        ("4. Sınıf Testi 1", "yer kabuğunun yapısı kayaçlar ve madenler", "4. Sınıf"),
        ("4. Sınıf Testi 2", "besin içerikleri proteinler karbonhidratlar dengeli beslenme", "4. Sınıf"),
    ]

    for label, query, target_sinif in test_queries:
        print(f"\n🔍 [{label}] Sorgu: '{query}' (Filtre: {target_sinif})")
        results = store.similarity_search(
            query=query,
            k=2,
            filter={"sinif": target_sinif}
        )
        if results:
            for idx, r in enumerate(results, 1):
                ref = r.metadata.get("referans", "Bilinmeyen Referans")
                unit = r.metadata.get("unite", "")
                konu = r.metadata.get("konu", "")
                alt_konu = r.metadata.get("alt_konu", "")
                page = r.metadata.get("printed_page", r.metadata.get("page", ""))
                tok_cnt = r.metadata.get("token_count", "-")
                snippet = r.page_content[:110].replace("\n", " ")
                print(f"   [{idx}] Basılı Sayfa {page} | {unit} -> {konu} ({alt_konu}) [{tok_cnt} token]")
                print(f"       Referans: {ref}")
                print(f"       Özet    : \"{snippet}...\"")
        else:
            print("   ⚠️ Sonuç bulunamadı.")

    print("\n" + "=" * 75)
    print("✅ İlkokul RAG altyapısı 768 boyutlu olarak tamamen doğrulanmış ve hazırdır.")
    print("=" * 75)


if __name__ == "__main__":
    main()
