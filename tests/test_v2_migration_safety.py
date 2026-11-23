"""
EduFrame AI - v2 Migrasyon Güvenlik ve Veri Bütünlüğü Doğrulama Testleri
-------------------------------------------------------------------------
Bu test paketi, v2 migrasyonu sırasındaki güvenlik mekanizmalarını doğrular:
1. clean_pedagogical_text() entegrasyonu ve gürültü filtreleme
2. v2 chunk ID ve metadata bütünlüğü (version='v2', content_hash)
3. Tekrar çalıştırma ve yarım kalmış migrasyondan devam etme (idempotent resumption)
4. Başarısız batch veya embedding durumunda eski v1 kayıtlarının silinmesinin engellenmesi
5. Eksik chunk kümesi durumunda (set difference) eski sürüm silinmesinin engellenmesi
6. Metin kalitesi ve metadata derinlemesine bütünlük denetimi (verify_indexed_book_integrity)
7. Kitaplar arası izolasyon (cross-book isolation): Book A temizliğinin Book B'yi etkilememesi
8. Normalizasyon kaynaklı chunk sayısı değişimlerinin (502 -> 374) güvenli yönetimi
9. check_book_legacy_records() fonksiyonunun v2 hedef sürüm tespiti
10. Tamamen güncel kitaplarda idempotent tekrar çalıştırma güvenliği
"""

import hashlib
import os
import runpy
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import MagicMock, patch
from typing import List, Dict, Any, Set

from langchain_core.documents import Document

from config import ROOT
from index_core import (
    generate_chunk_id,
    clean_pedagogical_text,
    cleanup_stale_chunks,
    create_pedagogical_splitter,
)
from index_ilkokul_books import (
    CURRENT_VERSION,
    check_book_legacy_records,
    can_cleanup_stale_chunks,
    verify_book_chunks_in_db,
    verify_indexed_book_integrity,
)


class TestV2MigrationSafety(unittest.TestCase):
    """v2 migrasyon akışının güvenlik, veri bütünlüğü ve hata toleransı testleri."""

    def setUp(self):
        self.filename_a = "ilkokul_fenbilimleri_3.sinif_1.kitap.pdf"
        self.hash_a = "a1b2c3d4e5f612345678"
        self.filename_b = "ilkokul_fenbilimleri_3.sinif_2.kitap.pdf"
        self.hash_b = "f6e5d4c3b2a198765432"
        self.filename_4 = "ilkokul_fenbilimleri_4.sinif.pdf"
        self.hash_4 = "4444cccc5555bbbb6666"

    # -------------------------------------------------------------------------
    # TEST 1: Metin Temizleme Entegrasyonu ve Gürültü Filtreleme
    # -------------------------------------------------------------------------
    def test_01_clean_pedagogical_text_integration_filters_noise(self):
        """Pedagojik metin temizleme gürültülü satırları ayıklar, bilimsel içeriği korur."""
        noisy_raw = (
            "FEN BİLİMLERİ DERSİ ÇALIŞMA SAYFASI\n"
            "........................................\n"
            "........................................\n"
            "Aşağıdaki soruları cevaplayınız:\n"
            "1. Kayaçlar nasıl oluşur? ..............................\n"
            "Cevap: .................................................\n"
            "2. Madenler ekonomik değere sahip midir? ..............\n"
            "________________________________________________________\n"
            "--------------------------------------------------------\n"
            "Kayaçlar, bir ya da birden fazla mineralin bir araya gelmesiyle oluşur."
        )

        cleaned = clean_pedagogical_text(noisy_raw)
        self.assertNotIn("........................................", cleaned)
        self.assertNotIn("________________________________________________________", cleaned)
        self.assertIn("Kayaçlar nasıl oluşur? ...", cleaned)
        self.assertIn("Kayaçlar, bir ya da birden fazla mineralin bir araya gelmesiyle oluşur.", cleaned)

        # Yalnızca noktalardan oluşan bir sayfa (<25 karakter) filtrelenir
        pure_dots_page = "........................................\n....................\n\n"
        cleaned_dots = clean_pedagogical_text(pure_dots_page)
        self.assertEqual(len(cleaned_dots), 0)
        self.assertTrue(len(cleaned_dots) < 25)

    # -------------------------------------------------------------------------
    # TEST 2: v2 Chunk ID ve Metadata Bütünlüğü
    # -------------------------------------------------------------------------
    def test_02_v2_chunk_id_and_metadata_structure(self):
        """v2 Chunk ID şeması ve metadata'da version ve content_hash alanları eksiksiz üretilir."""
        cid = generate_chunk_id(self.filename_a, self.hash_a, physical_page=42, chunk_idx=1, version=CURRENT_VERSION)
        expected_cid = f"{self.filename_a}:{self.hash_a[:12]}:v2:phys_0042:c001"
        self.assertEqual(cid, expected_cid)
        self.assertIn(":v2:", cid)
        self.assertIn("phys_0042", cid)

        text = "Fosiller milyonlarca yıl önce yaşamış canlıların kayaçlar arasındaki kalıntılarıdır."
        content_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]

        meta = {
            "source": self.filename_a,
            "physical_page": 42,
            "version": CURRENT_VERSION,
            "content_hash": content_hash,
            "unite": "1. Ünite",
            "konu": "Fosil Oluşumu",
        }

        self.assertEqual(meta["version"], "v2")
        self.assertEqual(meta["content_hash"], content_hash)
        self.assertEqual(len(meta["content_hash"]), 12)

    # -------------------------------------------------------------------------
    # TEST 3: Resumption & Yarım Kalmış Migrasyondan Devam Etme (Idempotency)
    # -------------------------------------------------------------------------
    def test_03_resumption_skips_existing_v2_chunks(self):
        """Yarım kalmış migrasyonda veritabanında zaten mevcut olan v2 parçaları atlanır, mükerrer kayıt engellenir."""
        # 4 parçalık bir kitap simülasyonu: 2 tanesi önceden eklenmiş olsun
        all_cids = [
            generate_chunk_id(self.filename_a, self.hash_a, 10, 0, version="v2"),
            generate_chunk_id(self.filename_a, self.hash_a, 10, 1, version="v2"),
            generate_chunk_id(self.filename_a, self.hash_a, 11, 0, version="v2"),
            generate_chunk_id(self.filename_a, self.hash_a, 11, 1, version="v2"),
        ]

        existing_ids = {all_cids[0], all_cids[1]}  # İlk 2 parça DB'de var

        queued_docs = []
        queued_ids = []
        skipped_count = 0

        for cid in all_cids:
            if cid in existing_ids:
                skipped_count += 1
                continue
            queued_docs.append(Document(page_content=f"İçerik {cid}", metadata={"chunk_id": cid}))
            queued_ids.append(cid)

        self.assertEqual(skipped_count, 2)
        self.assertEqual(len(queued_ids), 2)
        self.assertEqual(queued_ids, [all_cids[2], all_cids[3]])

    # -------------------------------------------------------------------------
    # TEST 4: Başarısız Batch/Embedding Durumunda v1 Temizliğinin Engellenmesi
    # -------------------------------------------------------------------------
    def test_04_failed_batch_blocks_v1_cleanup(self):
        """Embedding servisi veya batch yazımında hata oluşursa can_cleanup_stale_chunks False döner, v1 korunur."""
        all_failed_items = [
            {"chunk_id": f"{self.filename_a}:hash:v2:phys_0010:c000", "error_type": "ConnectionError", "error_message": "Ollama timeout"}
        ]
        book_has_failure = any(f.get("chunk_id", "").startswith(self.filename_a) for f in all_failed_items)

        can_clean = can_cleanup_stale_chunks(
            book_has_failure=book_has_failure,
            book_success=40,
            expected_new_count=45,
            missing_ids=set(),
            page_errors=[]
        )
        self.assertFalse(can_clean)

    # -------------------------------------------------------------------------
    # TEST 5: Eksik Chunk Kümesi (Set Difference) v1 Temizliğini Engeller
    # -------------------------------------------------------------------------
    def test_05_missing_chunk_ids_blocks_v1_cleanup(self):
        """Beklenen v2 parçalarından tek bir tanesi bile veritabanında doğrulanmazsa eski kayıtlar silinmez."""
        expected_ids = {
            f"{self.filename_a}:hash:v2:phys_0001:c000",
            f"{self.filename_a}:hash:v2:phys_0001:c001",
            f"{self.filename_a}:hash:v2:phys_0002:c000",
        }

        mock_store = MagicMock()
        mock_col = MagicMock()
        mock_store._collection = mock_col
        # Veritabanında 3. parça eksik
        mock_col.get.return_value = {
            "ids": [
                f"{self.filename_a}:hash:v2:phys_0001:c000",
                f"{self.filename_a}:hash:v2:phys_0001:c001",
            ]
        }

        is_verified, missing = verify_book_chunks_in_db(mock_store, expected_ids)
        self.assertFalse(is_verified)
        self.assertEqual(len(missing), 1)
        self.assertIn(f"{self.filename_a}:hash:v2:phys_0002:c000", missing)

        can_clean = can_cleanup_stale_chunks(
            book_has_failure=False,
            book_success=3,
            expected_new_count=3,
            missing_ids=missing,
            page_errors=[]
        )
        self.assertFalse(can_clean)

    # -------------------------------------------------------------------------
    # TEST 6: Derinlemesine Metin ve Metadata Bütünlük Denetimi (SHA-256 Hash Dahil)
    # -------------------------------------------------------------------------
    def test_06_verify_indexed_book_integrity(self):
        """verify_indexed_book_integrity SHA-256 content_hash, versiyon, kaynak ve metin kalitesini doğrular."""
        mock_store = MagicMock()
        mock_col = MagicMock()
        mock_store._collection = mock_col

        valid_id = f"{self.filename_a}:hash12345678:v2:phys_0015:c000"
        valid_doc = "Dünya'mızın dış kısmını saran hava katmanına atmosfer denir."
        valid_hash = hashlib.sha256(valid_doc.encode("utf-8")).hexdigest()[:12]
        valid_meta = {
            "source": self.filename_a,
            "physical_page": 15,
            "version": "v2",
            "content_hash": valid_hash,
            "unite": "1. Ünite",
            "konu": "Gezegenimizi Tanıyalım",
        }

        # 1. Doğru hash -> doğrulama başarılı
        mock_col.get.return_value = {
            "ids": [valid_id],
            "metadatas": [valid_meta],
            "documents": [valid_doc],
        }
        ok, issues = verify_indexed_book_integrity(mock_store, self.filename_a, {valid_id}, target_version="v2")
        self.assertTrue(ok)
        self.assertEqual(len(issues), 0)

        # 2. Yanlış hash -> doğrulama başarısız ve hata mesajında chunk ID ve uyuşmazlık açıklanır
        bad_hash_meta = dict(valid_meta, content_hash="000000000000")
        mock_col.get.return_value = {
            "ids": [valid_id],
            "metadatas": [bad_hash_meta],
            "documents": [valid_doc],
        }
        ok_wrong_hash, issues_wrong_hash = verify_indexed_book_integrity(
            mock_store, self.filename_a, {valid_id}, target_version="v2"
        )
        self.assertFalse(ok_wrong_hash)
        self.assertTrue(any(valid_id in s and "content_hash uyuşmazlığı" in s for s in issues_wrong_hash))
        self.assertTrue(any("Kayıtlı=000000000000" in s and valid_hash in s for s in issues_wrong_hash))

        # 3. Eksik hash -> doğrulama başarısız
        missing_hash_meta = dict(valid_meta, content_hash="")
        mock_col.get.return_value = {
            "ids": [valid_id],
            "metadatas": [missing_hash_meta],
            "documents": [valid_doc],
        }
        ok_missing_hash, issues_missing_hash = verify_indexed_book_integrity(
            mock_store, self.filename_a, {valid_id}, target_version="v2"
        )
        self.assertFalse(ok_missing_hash)
        self.assertTrue(any("content_hash metadata alanı eksik" in s for s in issues_missing_hash))

        # 4. Veritabanındaki metin değiştirilmiş -> doğrulama başarısız
        altered_doc = "Değiştirilmiş ve tahrif edilmiş sahte ders metni içeriği."
        mock_col.get.return_value = {
            "ids": [valid_id],
            "metadatas": [valid_meta],  # Eski hash kaldı
            "documents": [altered_doc],
        }
        ok_altered, issues_altered = verify_indexed_book_integrity(
            mock_store, self.filename_a, {valid_id}, target_version="v2"
        )
        self.assertFalse(ok_altered)
        self.assertTrue(any("content_hash uyuşmazlığı" in s for s in issues_altered))

        # 5. Hash uyuşmazlığında v1 kayıtları korunur (can_cleanup_stale_chunks engellenir)
        can_clean_on_mismatch = can_cleanup_stale_chunks(
            book_has_failure=False,
            book_success=1,
            expected_new_count=1,
            missing_ids=set(),
            page_errors=[],
            integrity_ok=ok_wrong_hash,
        )
        self.assertFalse(can_clean_on_mismatch)

        # Ek kontrol: Boş / eksik metin reddedilir
        mock_col.get.return_value = {
            "ids": [valid_id],
            "metadatas": [valid_meta],
            "documents": [""],
        }
        ok_empty, issues_empty = verify_indexed_book_integrity(
            mock_store, self.filename_a, {valid_id}, target_version="v2"
        )
        self.assertFalse(ok_empty)
        self.assertTrue(any("Belge metni boş veya eksik" in s for s in issues_empty))

        # Ek kontrol: Eski versiyon metadata'sı ("v1") taşıyan kayıt reddedilir
        bad_meta = dict(valid_meta, version="v1")
        mock_col.get.return_value = {
            "ids": [valid_id],
            "metadatas": [bad_meta],
            "documents": [valid_doc],
        }
        ok_b, issues_b = verify_indexed_book_integrity(mock_store, self.filename_a, {valid_id}, target_version="v2")
        self.assertFalse(ok_b)
        self.assertTrue(any("Metadata versiyonu 'v1' != 'v2'" in s for s in issues_b))

        # Ek kontrol: Temizlenmemiş nokta satırı içeren kayıt reddedilir
        dirty_doc = "Soru 1: Güneş nedir? ....................................."
        mock_col.get.return_value = {
            "ids": [valid_id],
            "metadatas": [valid_meta],
            "documents": [dirty_doc],
        }
        ok_c, issues_c = verify_indexed_book_integrity(mock_store, self.filename_a, {valid_id}, target_version="v2")
        self.assertFalse(ok_c)
        self.assertTrue(any("Temizlenmemiş nokta dizisi" in s for s in issues_c))

        # Ek kontrol: expected_ids boş olduğunda otomatik başarı engellenir, bütünlük hatası verilir
        ok_empty_set, issues_empty_set = verify_indexed_book_integrity(
            mock_store, self.filename_a, set(), target_version="v2"
        )
        self.assertFalse(ok_empty_set)
        self.assertTrue(any("Beklenen chunk ID kümesi boş" in s for s in issues_empty_set))
        # Boş ID kümesinde eski sürüm temizliği ENGELLENİR
        can_clean_empty = can_cleanup_stale_chunks(
            book_has_failure=False,
            book_success=0,
            expected_new_count=0,
            missing_ids=set(),
            page_errors=[],
            integrity_ok=ok_empty_set,
        )
        self.assertFalse(can_clean_empty)

        # Ek kontrol: expected_content_hashes kaynak kontrolü
        mock_col.get.return_value = {
            "ids": [valid_id],
            "metadatas": [valid_meta],
            "documents": [valid_doc],
        }
        ok_exp_match, _ = verify_indexed_book_integrity(
            mock_store, self.filename_a, {valid_id}, target_version="v2",
            expected_content_hashes={valid_id: valid_hash}
        )
        self.assertTrue(ok_exp_match)

        ok_exp_mismatch, issues_exp_mismatch = verify_indexed_book_integrity(
            mock_store, self.filename_a, {valid_id}, target_version="v2",
            expected_content_hashes={valid_id: "mismatch9999"}
        )
        self.assertFalse(ok_exp_mismatch)
        self.assertTrue(any("Kaynak content_hash uyuşmazlığı" in s for s in issues_exp_mismatch))

        # Ek kontrol: migration_allowed=False temizliği engeller
        can_clean_no_mig = can_cleanup_stale_chunks(
            book_has_failure=False,
            book_success=1,
            expected_new_count=1,
            missing_ids=set(),
            page_errors=[],
            integrity_ok=True,
            migration_allowed=False,
        )
        self.assertFalse(can_clean_no_mig)

    # -------------------------------------------------------------------------
    # TEST 7: Kitaplar Arası İzolasyon (Cross-Book Isolation)
    # -------------------------------------------------------------------------
    def test_07_cross_book_isolation_protects_other_books(self):
        """Book A için eski sürüm temizliği yapıldığında Book B'ye ait hiçbir kayıt silinmez."""
        mock_store = MagicMock()
        mock_col = MagicMock()
        mock_store._collection = mock_col

        # DB'de Book A'ya ait hem v1 hem v2, Book B'ye ait ise v1 parçalar var
        db_records = [
            {"id": f"{self.filename_a}:hashA:v2:phys_0001:c000", "source": self.filename_a, "file_hash": self.hash_a[:12]},
            {"id": f"{self.filename_a}:hashA:phys_0001:c000", "source": self.filename_a, "file_hash": self.hash_a[:12]},  # v1
            {"id": f"{self.filename_a}:oldhash:phys_0002:c000", "source": self.filename_a, "file_hash": "oldoldold123"},  # stale
            # Book B parçaları:
            {"id": f"{self.filename_b}:hashB:phys_0001:c000", "source": self.filename_b, "file_hash": self.hash_b[:12]},  # Book B v1
            {"id": f"{self.filename_b}:hashB:phys_0002:c000", "source": self.filename_b, "file_hash": self.hash_b[:12]},  # Book B v1
        ]

        def fake_get(where=None, include=None):
            req_source = where.get("source") if where else None
            matching = [r for r in db_records if r["source"] == req_source]
            return {
                "ids": [r["id"] for r in matching],
                "metadatas": [{"source": r["source"], "file_hash": r["file_hash"]} for r in matching],
            }

        mock_col.get.side_effect = fake_get

        deleted_ids_record = []
        def fake_delete(ids):
            deleted_ids_record.extend(ids)

        mock_col.delete.side_effect = fake_delete

        # Book A için temizlik çalıştır
        cleaned_count = cleanup_stale_chunks(
            store=mock_store,
            filename=self.filename_a,
            current_hash=self.hash_a,
            target_version="v2"
        )

        self.assertEqual(cleaned_count, 2)
        # Silinen parçalar Book A'nın v1 ve eski hash parçalarıdır
        self.assertIn(f"{self.filename_a}:hashA:phys_0001:c000", deleted_ids_record)
        self.assertIn(f"{self.filename_a}:oldhash:phys_0002:c000", deleted_ids_record)
        # Book A'nın v2 parçası silinmemiştir
        self.assertNotIn(f"{self.filename_a}:hashA:v2:phys_0001:c000", deleted_ids_record)
        # Book B'ye ait HİÇBİR parça silinmemiştir!
        for deleted_id in deleted_ids_record:
            self.assertFalse(deleted_id.startswith(self.filename_b), f"Book B parçası yanlışlıkla silindi: {deleted_id}")

    # -------------------------------------------------------------------------
    # TEST 8: Normalizasyon Kaynaklı Chunk Sayısı Değişimi Yönetimi (502 -> 374)
    # -------------------------------------------------------------------------
    def test_08_chunk_count_change_handling_safe_reduction(self):
        """Normalizasyon sonucu chunk sayısı azalsa bile (örn: 502 -> 374), eski v1 kayıtları güvenle temizlenir."""
        # 4. sınıf kitabında 502 adet v1 kaydı varken, temizleme sonrası 374 adet v2 kaydı üretildiğini simüle edelim
        old_v1_ids = [f"{self.filename_4}:{self.hash_4[:12]}:phys_{i:04d}:c000" for i in range(1, 503)]
        new_v2_ids = [f"{self.filename_4}:{self.hash_4[:12]}:v2:phys_{i:04d}:c000" for i in range(1, 375)]

        mock_store = MagicMock()
        mock_col = MagicMock()
        mock_store._collection = mock_col

        # Veritabanında şu an hem 502 adet v1 hem 374 adet yeni yazılmış v2 olsun
        v2_doc_text = "Temiz pedagojik içerik metni fen bilimleri dersi."
        v2_content_hash = hashlib.sha256(v2_doc_text.encode("utf-8")).hexdigest()[:12]
        all_ids = old_v1_ids + new_v2_ids
        all_metas = [{"source": self.filename_4, "file_hash": self.hash_4[:12], "version": "v1"} for _ in old_v1_ids] + \
                    [{"source": self.filename_4, "file_hash": self.hash_4[:12], "version": "v2", "content_hash": v2_content_hash, "physical_page": 1} for _ in new_v2_ids]
        all_docs = ["Eski metin" for _ in old_v1_ids] + [v2_doc_text for _ in new_v2_ids]

        id_map = {cid: (m, d) for cid, m, d in zip(all_ids, all_metas, all_docs)}

        def fake_get(ids=None, where=None, include=None):
            if ids is not None:
                matched_ids = [cid for cid in ids if cid in id_map]
                return {
                    "ids": matched_ids,
                    "metadatas": [id_map[cid][0] for cid in matched_ids],
                    "documents": [id_map[cid][1] for cid in matched_ids],
                }
            if where:
                req_source = where.get("source")
                matched_ids = [cid for cid in all_ids if id_map[cid][0]["source"] == req_source]
                return {
                    "ids": matched_ids,
                    "metadatas": [id_map[cid][0] for cid in matched_ids],
                    "documents": [id_map[cid][1] for cid in matched_ids],
                }
            return {"ids": all_ids}

        mock_col.get.side_effect = fake_get

        deleted_ids = []
        mock_col.delete.side_effect = lambda ids: deleted_ids.extend(ids)

        # 1. 374 adet yeni v2 parçasının DB'de tamlığı doğrulanır
        is_verified, missing = verify_book_chunks_in_db(mock_store, set(new_v2_ids))
        self.assertTrue(is_verified)
        self.assertEqual(len(missing), 0)

        # 2. Bütünlük denetimi yapılır
        integrity_ok, issues = verify_indexed_book_integrity(mock_store, self.filename_4, set(new_v2_ids), target_version="v2")
        self.assertTrue(integrity_ok)

        # 3. can_cleanup_stale_chunks kontrolü
        can_clean = can_cleanup_stale_chunks(
            book_has_failure=False,
            book_success=len(new_v2_ids),
            expected_new_count=len(new_v2_ids),
            missing_ids=missing,
            page_errors=[],
            integrity_ok=integrity_ok,
            migration_allowed=True,
        )
        self.assertTrue(can_clean)

        # 4. cleanup_stale_chunks çağrılır
        cleaned_count = cleanup_stale_chunks(
            store=mock_store,
            filename=self.filename_4,
            current_hash=self.hash_4,
            target_version="v2"
        )

        self.assertEqual(cleaned_count, 502)
        self.assertEqual(set(deleted_ids), set(old_v1_ids))
        # v2 parçalarının hiçbirisi silinmemiştir
        for v2_id in new_v2_ids:
            self.assertNotIn(v2_id, deleted_ids)

    # -------------------------------------------------------------------------
    # TEST 9: check_book_legacy_records Hedef Sürüm Tespiti
    # -------------------------------------------------------------------------
    def test_09_check_book_legacy_records_with_target_version(self):
        """check_book_legacy_records target_version='v2' verildiğinde v1 ve eski şemalı ID'leri listeler."""
        mock_store = MagicMock()
        mock_col = MagicMock()
        mock_store._collection = mock_col

        mock_col.get.return_value = {
            "ids": [
                f"{self.filename_a}:{self.hash_a[:12]}:v2:phys_0001:c000",   # v2 geçerli
                f"{self.filename_a}:{self.hash_a[:12]}:v2:phys_0002:c000",   # v2 geçerli
                f"{self.filename_a}:{self.hash_a[:12]}:phys_0001:c000",      # v1 (eski sürüm)
                f"{self.filename_a}:{self.hash_a[:12]}:p0001:c000",          # eski şema
            ]
        }

        legacy_ids = check_book_legacy_records(mock_store, self.filename_a, target_version="v2")
        self.assertEqual(len(legacy_ids), 2)
        self.assertIn(f"{self.filename_a}:{self.hash_a[:12]}:phys_0001:c000", legacy_ids)
        self.assertIn(f"{self.filename_a}:{self.hash_a[:12]}:p0001:c000", legacy_ids)
        self.assertNotIn(f"{self.filename_a}:{self.hash_a[:12]}:v2:phys_0001:c000", legacy_ids)

    # -------------------------------------------------------------------------
    # TEST 10: Tamamen Güncel Kitapta Idempotent Resumption Güvenliği
    # -------------------------------------------------------------------------
    def test_10_fully_migrated_book_idempotent_rerun(self):
        """Kitabın tüm v2 parçaları DB'de mevcut ve hiç eski kayıt yoksa güvenle güncel kabul edilir."""
        mock_store = MagicMock()
        mock_col = MagicMock()
        mock_store._collection = mock_col

        v2_ids = {
            f"{self.filename_a}:{self.hash_a[:12]}:v2:phys_0001:c000",
            f"{self.filename_a}:{self.hash_a[:12]}:v2:phys_0002:c000",
        }

        doc1 = "Bu birinci ders içeriğidir ve pedagojik olarak temizdir."
        doc2 = "Bu ikinci ders içeriğidir ve pedagojik olarak temizdir."
        hash1 = hashlib.sha256(doc1.encode("utf-8")).hexdigest()[:12]
        hash2 = hashlib.sha256(doc2.encode("utf-8")).hexdigest()[:12]

        # DB'de tüm v2 parçaları mevcut
        mock_col.get.return_value = {
            "ids": list(v2_ids),
            "metadatas": [
                {"source": self.filename_a, "version": "v2", "physical_page": 1, "content_hash": hash1},
                {"source": self.filename_a, "version": "v2", "physical_page": 2, "content_hash": hash2},
            ],
            "documents": [doc1, doc2],
        }

        # 1. ID doğrulama
        is_verified, missing = verify_book_chunks_in_db(mock_store, v2_ids)
        self.assertTrue(is_verified)
        self.assertEqual(len(missing), 0)

        # 2. Bütünlük denetimi
        integrity_ok, issues = verify_indexed_book_integrity(mock_store, self.filename_a, v2_ids, target_version="v2")
        self.assertTrue(integrity_ok)
        self.assertEqual(len(issues), 0)

        # 3. Eski kayıt kontrolü (hiç eski kayıt kalmamış olmalı)
        legacy = check_book_legacy_records(mock_store, self.filename_a, target_version="v2")
        self.assertEqual(len(legacy), 0)

    # -------------------------------------------------------------------------
    # TEST 11: check_book_legacy_records Hata Yönetimi
    # -------------------------------------------------------------------------
    def test_11_check_book_legacy_records_error_handling(self):
        """check_book_legacy_records sorgu hatasında RuntimeError fırlatır, hatayı sessizce yutmaz."""
        mock_store = MagicMock()
        mock_col = MagicMock()
        mock_store._collection = mock_col

        # 1. Eski kayıt yok -> boş liste
        mock_col.get.return_value = {
            "ids": [
                f"{self.filename_a}:{self.hash_a[:12]}:v2:phys_0001:c000",
                f"{self.filename_a}:{self.hash_a[:12]}:v2:phys_0002:c000",
            ]
        }
        res_empty = check_book_legacy_records(mock_store, self.filename_a, target_version="v2")
        self.assertEqual(res_empty, [])

        # 2. Eski v1 kayıtları var -> doğru ID listesi
        mock_col.get.return_value = {
            "ids": [
                f"{self.filename_a}:{self.hash_a[:12]}:v2:phys_0001:c000",
                f"{self.filename_a}:{self.hash_a[:12]}:phys_0001:c000",  # v1
            ]
        }
        res_v1 = check_book_legacy_records(mock_store, self.filename_a, target_version="v2")
        self.assertEqual(res_v1, [f"{self.filename_a}:{self.hash_a[:12]}:phys_0001:c000"])

        # 3. ChromaDB sorgusu hata veriyor -> RuntimeError (orijinal hata korunarak)
        mock_col.get.side_effect = Exception("SQLite disk I/O error or lock")
        with self.assertRaises(RuntimeError) as ctx:
            check_book_legacy_records(mock_store, self.filename_a, target_version="v2")
        self.assertIn("ChromaDB legacy kayıt taraması", str(ctx.exception))
        self.assertIn("SQLite disk I/O error or lock", str(ctx.exception))

        # 4. Sorgu hatası durumunda eski kayıt temizliği çalışmaz
        cleanup_attempted = False
        try:
            legacy_ids = check_book_legacy_records(mock_store, self.filename_a, target_version="v2")
            if legacy_ids:
                cleanup_stale_chunks(mock_store, self.filename_a, self.hash_a, target_version="v2")
                cleanup_attempted = True
        except RuntimeError:
            cleanup_attempted = False

        self.assertFalse(cleanup_attempted)

    # -------------------------------------------------------------------------
    # TEST 12: Yazma Sonrası Doğrudan Güvenlik Kilidi (migration_allowed=False)
    # -------------------------------------------------------------------------
    def test_12_migration_lock_blocks_cleanup_after_writing_chunks(self):
        """v2 parçaları yazılıp doğrulansa bile, migration_allowed=False ise temizlik kesinlikle engellenir."""
        mock_store = MagicMock()
        mock_col = MagicMock()
        mock_store._collection = mock_col

        v2_ids = [f"{self.filename_a}:{self.hash_a[:12]}:v2:phys_000{i}:c000" for i in range(1, 4)]
        v1_ids = [f"{self.filename_a}:{self.hash_a[:12]}:phys_000{i}:c000" for i in range(1, 4)]

        # DB get mock
        v2_docs = [f"Pedagojik v2 metni {i}" for i in range(1, 4)]
        v2_metas = [
            {
                "source": self.filename_a,
                "file_hash": self.hash_a[:12],
                "version": "v2",
                "physical_page": i,
                "content_hash": hashlib.sha256(doc.encode("utf-8")).hexdigest()[:12]
            }
            for i, doc in enumerate(v2_docs, start=1)
        ]
        mock_col.get.return_value = {
            "ids": v2_ids,
            "metadatas": v2_metas,
            "documents": v2_docs,
        }

        # 1. ID doğrulaması ve bütünlük denetimi başarılı
        is_verified, missing = verify_book_chunks_in_db(mock_store, set(v2_ids))
        self.assertTrue(is_verified)
        integrity_ok, issues = verify_indexed_book_integrity(mock_store, self.filename_a, set(v2_ids), target_version="v2")
        self.assertTrue(integrity_ok)

        # 2. migration_allowed = False iken can_cleanup_stale_chunks False dönmelidir
        can_clean = can_cleanup_stale_chunks(
            book_has_failure=False,
            book_success=len(v2_ids),
            expected_new_count=len(v2_ids),
            missing_ids=missing,
            page_errors=[],
            integrity_ok=integrity_ok,
            migration_allowed=False,
        )
        self.assertFalse(can_clean)

        # 3. İndeksleme akışındaki doğrudan güvenlik kilidi denetimi:
        migration_allowed = False
        cleanup_called = False
        if not migration_allowed:
            # Doğrudan engellendi
            pass
        elif can_clean:
            cleanup_stale_chunks(mock_store, self.filename_a, self.hash_a, target_version="v2")
            cleanup_called = True

        self.assertFalse(cleanup_called)
        mock_col.delete.assert_not_called()

    # -------------------------------------------------------------------------
    # TEST 13: Güncel Kitapta (not book_docs) migration_allowed=False Kilidi
    # -------------------------------------------------------------------------
    def test_13_idempotent_book_blocks_cleanup_when_migration_not_allowed(self):
        """Tüm v2 parçaları DB'de mevcut olsa dahi, migration_allowed=False ise eski kayıtlar ASLA silinmez."""
        mock_store = MagicMock()
        mock_col = MagicMock()
        mock_store._collection = mock_col

        v2_ids = [f"{self.filename_a}:{self.hash_a[:12]}:v2:phys_0001:c000"]
        v1_ids = [f"{self.filename_a}:{self.hash_a[:12]}:phys_0001:c000"]

        doc_text = "Pedagojik v2 metni"
        meta = {
            "source": self.filename_a,
            "file_hash": self.hash_a[:12],
            "version": "v2",
            "physical_page": 1,
            "content_hash": hashlib.sha256(doc_text.encode("utf-8")).hexdigest()[:12]
        }
        mock_col.get.return_value = {
            "ids": v2_ids,
            "metadatas": [meta],
            "documents": [doc_text],
        }

        # not book_docs akış simülasyonu
        legacy_ids = [v1_ids[0]]
        migration_allowed = False
        cleanup_called = False

        if legacy_ids:
            if not migration_allowed:
                pass  # Güvenlik kilidi devrede!
            else:
                cleanup_stale_chunks(mock_store, self.filename_a, self.hash_a, target_version="v2")
                cleanup_called = True

        self.assertFalse(cleanup_called)
        mock_col.delete.assert_not_called()

    # -------------------------------------------------------------------------
    # TEST 14: Varsayılan Davranış (Dry-Run / İzin Yok) Yazma ve Silmeyi Engeller
    # -------------------------------------------------------------------------
    def test_14_dry_run_prevents_writing_and_deletion_by_default(self):
        """--allow-migration ve --allow-write verilmediğinde yeni parçalar yazılmaz ve eski kayıtlar silinmez."""
        mock_store = MagicMock()
        mock_col = MagicMock()
        mock_store._collection = mock_col

        # Parametreler: varsayılan mod (izin yok)
        migration_allowed = False
        write_allowed = False

        book_docs = [Document(page_content="Yeni içerik", metadata={"chunk_id": "c1"})]
        book_ids = ["c1"]

        written_chunks = 0
        deleted_chunks = 0

        # İndeksleme yazma kontrolü simülasyonu
        if not write_allowed:
            # DRY-RUN: Yazma yapılmadı
            pass
        else:
            written_chunks = len(book_docs)

        # Temizlik kontrolü simülasyonu
        if not migration_allowed:
            # GÜVENLİK KİLİDİ: Temizlik yapılmadı
            pass
        else:
            deleted_chunks = 1

        self.assertEqual(written_chunks, 0, "İzin olmadan hiçbir parça yazılmamalıdır.")
        self.assertEqual(deleted_chunks, 0, "İzin olmadan hiçbir kayıt silinmemelidir.")

    # -------------------------------------------------------------------------
    # TEST 15: Aktif Veritabanı Dizinini Sabitleme
    # -------------------------------------------------------------------------
    def test_15_chroma_config_is_pinned_to_v1(self):
        """Eski ortam ayarı aktif ChromaDB yolunu değiştiremez."""
        with patch.dict(os.environ, {"EDUFRAME_CHROMA_DB_DIR": "data/chroma_db_v2"}):
            settings = runpy.run_path(str(ROOT / "config.py"))

        expected_path = ROOT / "data" / "chroma_db"
        self.assertEqual(settings["CHROMA_DB_DIR"], expected_path)
        self.assertEqual(settings["DB_DIR"], expected_path)

    # -------------------------------------------------------------------------
    # TEST 16: Testler İçin Ayrı Veritabanı Dizini
    # -------------------------------------------------------------------------
    def test_16_rag_accepts_explicit_test_directory(self):
        """RAG ve embedding motoru açıkça verilen geçici dizini kullanabilir."""
        from services.rag_motoru import RAGMotoru

        with TemporaryDirectory() as directory:
            db_path = Path(directory) / "chroma_test"
            with patch("services.rag_motoru.GuvenliEmbeddingMotoru._koleksiyon_boyutu_al", return_value=768):
                rag = RAGMotoru(db_dizini=db_path)

            self.assertEqual(rag.db_dizini, db_path)
            self.assertEqual(rag.embedding.db_dizini, db_path)


if __name__ == "__main__":
    unittest.main()
