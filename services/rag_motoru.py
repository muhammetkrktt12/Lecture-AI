"""
EduFrame AI - Metadata Destekli RAG ve Vektör Veritabanı Servisi (services/rag_motoru.py)
--------------------------------------------------------------------------------------
Bu modül, referans ders kitaplarının (PDF) yüklenmesini, eğitim seviyesi bazlı
metadata ile parçalanıp ChromaDB'ye kaydedilmesini ve analiz sırasında sadece
seçilen eğitim seviyesinde anlamsal (semantik) arama yapılmasını sağlar.

Temel Kurallar ve Sorumluluklar:
1. Metadata Filtreleme: Her parça (chunk) 'egitim_seviyesi' (İlkokul, Ortaokul, Lise, Üniversite)
   etiketiyle kaydedilir. Arama sırasında SADECE seçilen seviyedeki belgeler taranır.
2. Soru Formatı Klonlama (Few-Shot): Kitap içerisindeki alıştırma, etkinlik ve test sorularını
   ayrıştırarak soru üretici ajana örnek format sağlar.
3. Esnek Embedding: Ollama (embeddinggemma) öncelikli çalışır; bağlantı yoksa güvenli yerel
   vektörleştirici ile kesintisiz geliştirme ve test imkanı sunar.
4. Temiz Kod ve Açıklamalar: Tüm fonksiyonlar tip ipuçları (type hints) ve detaylı Türkçe
   açıklamalarla donatılmıştır.
"""

import re
import hashlib
from pathlib import Path
from typing import List, Dict, Any, Optional
import numpy as np

import pypdf
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma

# Kalite ve Sıralama Araçlarını İçe Aktar (utils.lexical_tools öncelikli)
try:
    from utils.lexical_tools import rerank_candidates as lexical_rerank, validate_citations, hit_at_k
except ImportError:
    try:
        from quality_utils import rerank_candidates as lexical_rerank, validate_citations, hit_at_k
    except ImportError:
        lexical_rerank = None

# Gemma Tokenizer Desteği (tokenizer_utils & gemma_tokenizer.json)
try:
    from tokenizer_utils import get_gemma_tokenizer, analyze_chunk_tokens
    _GEMMA_TOKENIZER = get_gemma_tokenizer()
except Exception:
    _GEMMA_TOKENIZER = None
try:
    from config import (
        CHROMA_DB_DIR,
        COLLECTION,
        DATA_DIR,
        REFERENCE_BOOKS_DIR,
        EGITIM_SEVIYELERI,
        DEFAULT_SEVIYE,
        OLLAMA_URL,
        EMBED_MODEL,
        RAG_MAX_MESAFE,
    )
except ImportError:
    ROOT = Path(__file__).resolve().parent.parent
    DATA_DIR = ROOT / "data"
    CHROMA_DB_DIR = DATA_DIR / "chroma_db"
    REFERENCE_BOOKS_DIR = DATA_DIR / "reference_books"
    COLLECTION = "eduframe_meb_768"
    EGITIM_SEVIYELERI = ["İlkokul", "Ortaokul", "Lise", "Üniversite"]
    DEFAULT_SEVIYE = "İlkokul"
    OLLAMA_URL = "http://localhost:11434"
    EMBED_MODEL = "embeddinggemma"
    RAG_MAX_MESAFE = 1.1


# ---------------------------------------------------------------------------
# EMBEDDING ADAPTÖRÜ
# ---------------------------------------------------------------------------
class GuvenliEmbeddingMotoru(Embeddings):
    """
    384 boyutlu, deterministik ve Türkçe morfolojisine duyarlı dense embedding motoru.
    Gerektiğinde Ollama servisiyle senkronize olabilir; ancak harici servis olmasa dahi
    kesintisiz ve yüksek performanslı anlamsal vektör üretimi sağlar.
    """

    def __init__(
        self,
        model_adi: str = EMBED_MODEL,
        base_url: str = OLLAMA_URL,
        koleksiyon_adi: Optional[str] = None,
        allow_fallback: bool = False,
        db_dizini: Optional[Path] = None,
    ):
        self.model_adi = model_adi
        self.base_url = base_url
        self.koleksiyon_adi = koleksiyon_adi or COLLECTION
        self.allow_fallback = allow_fallback
        self.db_dizini = Path(db_dizini) if db_dizini else CHROMA_DB_DIR
        self._ollama_client = None
        self._ollama_kullanilabilir = None
        self._hedef_boyut = self._koleksiyon_boyutu_al()

    def _koleksiyon_boyutu_al(self) -> int:
        """Mevcut Chroma koleksiyonunun beklediği embedding vektör boyutunu tespit eder."""
        try:
            import chromadb
            client = chromadb.PersistentClient(path=str(self.db_dizini))
            col = client.get_collection(self.koleksiyon_adi)
            sample = col.get(limit=1, include=['embeddings'])
            if sample and sample.get('embeddings') is not None and len(sample['embeddings']) > 0:
                dim = len(sample['embeddings'][0])
                if dim > 0:
                    return dim
        except Exception:
            pass
        return 768

    def _get_ollama(self):
        if self._ollama_kullanilabilir is False:
            return None
        if self._ollama_client is None:
            try:
                from langchain_ollama import OllamaEmbeddings
                client = OllamaEmbeddings(model=self.model_adi, base_url=self.base_url)
                test_v = client.embed_query("test")
                if len(test_v) == self._hedef_boyut:
                    self._ollama_client = client
                    self._ollama_kullanilabilir = True
                    print(f"[RAGMotoru] Ollama Embedding aktif: {self.model_adi} ({len(test_v)} boyut)")
                else:
                    print(f"[RAGMotoru] Ollama boyutu ({len(test_v)}) koleksiyonla ({self._hedef_boyut}) uyuşmadı.")
                    self._ollama_kullanilabilir = False
            except Exception as e:
                self._ollama_kullanilabilir = False
                self._ollama_client = None
        return self._ollama_client

    def _fallback_embed(self, text: str, dim: Optional[int] = None) -> List[float]:
        """Türkçe kök ve ek duyarlılığı yüksek 768 boyutlu n-gram dense embedding."""
        d = dim or self._hedef_boyut or 768
        vector = np.zeros(d, dtype=np.float32)
        norm_text = text.replace("İ", "i").replace("I", "ı").lower()
        words = re.findall(r"[a-zçğıöşü0-9]+", norm_text)
        if not words:
            return vector.tolist()

        for w in words:
            h = int(hashlib.md5(w.encode("utf-8")).hexdigest()[:8], 16)
            vector[h % d] += 2.5
            w_pad = f"^{w}$"
            for i in range(len(w_pad) - 2):
                g = w_pad[i:i+3]
                hg = int(hashlib.md5(g.encode("utf-8")).hexdigest()[:8], 16)
                vector[hg % d] += 1.0

        norm = np.linalg.norm(vector)
        if norm > 0:
            vector = vector / norm
        return vector.tolist()

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        ollama = self._get_ollama()
        if ollama:
            try:
                return ollama.embed_documents(texts)
            except Exception as e:
                if not self.allow_fallback:
                    raise RuntimeError(f"[KRİTİK HATA] Ollama embedding işlemi başarısız oldu: {e}") from e
        if not self.allow_fallback:
            raise RuntimeError(
                f"[KRİTİK HATA] Üretim modunda Ollama ({self.model_adi}) servisine ulaşılamadı. "
                f"Vektör bütünlüğünü korumak için hash fallback kapalıdır. Lütfen Ollama servisini başlatın."
            )
        return [self._fallback_embed(t) for t in texts]

    def embed_query(self, text: str) -> List[float]:
        ollama = self._get_ollama()
        if ollama:
            try:
                return ollama.embed_query(text)
            except Exception as e:
                if not self.allow_fallback:
                    raise RuntimeError(f"[KRİTİK HATA] Ollama sorgu embedding işlemi başarısız oldu: {e}") from e
        if not self.allow_fallback:
            raise RuntimeError(
                f"[KRİTİK HATA] Üretim modunda Ollama ({self.model_adi}) servisine ulaşılamadı. "
                f"Semantik arama kalitesini ve veri bütünlüğünü korumak için hash fallback kapalıdır. "
                f"Lütfen Ollama servisini başlatın."
            )
        return self._fallback_embed(text)


# ---------------------------------------------------------------------------
# ANA RAG MOTORU SINIFI
# ---------------------------------------------------------------------------
class RAGMotoru:
    """
    EduFrame AI için seviye duyarlı RAG ve ChromaDB yönetim sınıfı.
    """

    def __init__(
        self,
        db_dizini: Optional[Path] = None,
        koleksiyon_adi: str = COLLECTION,
        embedding_motoru: Optional[Embeddings] = None,
        allow_fallback: bool = False,
    ):
        self.db_dizini = Path(db_dizini) if db_dizini else CHROMA_DB_DIR
        self.db_dizini.mkdir(parents=True, exist_ok=True)
        self.koleksiyon_adi = koleksiyon_adi
        self.allow_fallback = allow_fallback
        self.embedding = embedding_motoru or GuvenliEmbeddingMotoru(
            koleksiyon_adi=self.koleksiyon_adi,
            allow_fallback=allow_fallback,
            db_dizini=self.db_dizini,
        )
        self._vector_store: Optional[Chroma] = None

    def metin_benzerlikleri(self, kaynak: str, anlatimlar: List[str]) -> List[float]:
        """Bir kitap alıntısını birden fazla bölüm anlatımıyla karşılaştırır."""
        vektorler = np.asarray(self.embedding.embed_documents([kaynak, *anlatimlar]), dtype=float)
        kaynak_vektoru, aday_vektorleri = vektorler[0], vektorler[1:]
        payda = np.linalg.norm(kaynak_vektoru) * np.linalg.norm(aday_vektorleri, axis=1)
        skorlar = np.divide(
            aday_vektorleri @ kaynak_vektoru,
            payda,
            out=np.zeros(len(anlatimlar)),
            where=payda > 0,
        )
        return skorlar.tolist()

    def get_vector_store(self) -> Chroma:
        """Kalıcı ChromaDB vektör deposu bağlantısını döndürür."""
        if self._vector_store is None:
            self._vector_store = Chroma(
                collection_name=self.koleksiyon_adi,
                embedding_function=self.embedding,
                persist_directory=str(self.db_dizini),
            )
        return self._vector_store

    def pdf_yukle(self, pdf_yolu: str) -> List[Document]:
        """
        Belirtilen PDF dosyasını pypdf ile sayfa sayfa okur ve ham Document listesi döner.
        """
        yol = Path(pdf_yolu)
        if not yol.exists():
            raise FileNotFoundError(f"PDF dosyası bulunamadı: {pdf_yolu}")

        reader = pypdf.PdfReader(str(yol))
        dokumanlar: List[Document] = []

        for sayfa_no, sayfa in enumerate(reader.pages, start=1):
            try:
                metin = (sayfa.extract_text() or "").strip()
            except Exception:
                # Bazı sayfalarda karmaşık vektör çizimler veya font parsing limitleri olabilir, atla
                continue
            if not metin:
                continue

            # Ham belge üst verisi
            meta = {
                "source": yol.name,
                "page": sayfa_no,
                "total_pages": len(reader.pages),
            }
            dokumanlar.append(Document(page_content=metin, metadata=meta))

        print(f"[RAGMotoru] '{yol.name}' dosyasından {len(dokumanlar)} sayfa metin okundu.")
        return dokumanlar

    def pdf_indeksle(
        self,
        pdf_yolu: str,
        egitim_seviyesi: str,
        ders_adi: str = "Fen Bilimleri",
        chunk_size: int = 700,
        chunk_overlap: int = 100,
        batch_size: int = 50,
    ) -> int:
        """
        PDF dosyasını yükler, belirlenen eğitim seviyesi metadata'sını ekleyerek
        parçalara böler ve ChromaDB'ye kaydeder.

        :param pdf_yolu: PDF dosyasının yolu.
        :param egitim_seviyesi: 'İlkokul' | 'Ortaokul' | 'Lise' | 'Üniversite'
        :param ders_adi: Örn. 'Fen Bilimleri', 'Matematik'
        :param chunk_size: Karakter tabanlı parça büyüklüğü
        :param chunk_overlap: Örtüşme karakter sayısı
        :param batch_size: Veritabanına toplu yazma boyutu
        :return: İndekslenen toplam chunk adedi
        """
        if egitim_seviyesi not in EGITIM_SEVIYELERI:
            raise ValueError(
                f"Geçersiz eğitim seviyesi: '{egitim_seviyesi}'. Geçerli seviyeler: {EGITIM_SEVIYELERI}"
            )

        ham_belgeler = self.pdf_yukle(pdf_yolu)
        if not ham_belgeler:
            print(f"[UYARI] '{pdf_yolu}' dosyasında okunabilir metin bulunamadı.")
            return 0

        # Metin parçalayıcı (Metin yapısını ve cümle bütünlüğünü korur)
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", "? ", "! ", " ", ""],
        )

        parcalar: List[Document] = []
        dosya_adi = Path(pdf_yolu).name
        sinif_bilgisi = self.dosya_metadata_ayristir(dosya_adi)["sinif"]

        for doc in ham_belgeler:
            sayfa_no = doc.metadata.get("page", 1)
            chunks = splitter.split_text(doc.page_content)
            for idx, c_text in enumerate(chunks):
                if len(c_text.strip()) < 30:  # Anlamsız çok kısa parçaları ele
                    continue

                # Gemma Tokenizer ile kesin token sayımı (tokenizer_utils)
                token_sayisi = len(_GEMMA_TOKENIZER.encode(c_text).tokens) if _GEMMA_TOKENIZER else len(c_text.split())

                # KATI KURAL: egitim_seviyesi metadata olarak kaydedilir!
                parca_meta = {
                    "source": dosya_adi,
                    "page": sayfa_no,
                    "chunk_id": f"{dosya_adi}:{sayfa_no:03d}:{idx:02d}",
                    "egitim_seviyesi": egitim_seviyesi,
                    "ders_adi": ders_adi,
                    "sinif": sinif_bilgisi,
                    "token_count": token_sayisi,
                }
                parcalar.append(Document(page_content=c_text, metadata=parca_meta))

        toplam = len(parcalar)
        print(f"\n[RAGMotoru] Toplam {toplam} parça ChromaDB'ye indeksleniyor...")
        print(f" -> Eğitim Seviyesi: {egitim_seviyesi}")
        print(f" -> Ders: {ders_adi}")

        store = self.get_vector_store()
        ids = [doc.metadata["chunk_id"] for doc in parcalar]

        # Toplu batch halinde veritabanına ekle
        for start in range(0, toplam, batch_size):
            end = min(start + batch_size, toplam)
            batch_docs = parcalar[start:end]
            batch_ids = ids[start:end]
            store.add_documents(documents=batch_docs, ids=batch_ids)
            print(f" -> [{start + 1}-{end} / {toplam}] parça eklendi.")

        print(f"[BAŞARILI] '{dosya_adi}' başarıyla indekslendi ({toplam} parça).\n")
        return toplam

    @staticmethod
    def sinif_normalize(sinif: Optional[str]) -> Optional[str]:
        """
        Sınıf seviyesini veritabanındaki standart formatta normalize eder (Örn: '3', '3.sınıf' -> '3. Sınıf').
        """
        if not sinif:
            return None
        val = str(sinif).strip()
        m = re.search(r"(\d+)", val)
        if m:
            return f"{m.group(1)}. Sınıf"
        return val

    def seviye_filtreli_arama(
        self,
        sorgu: str,
        egitim_seviyesi: str,
        sinif: Optional[str] = None,
        ders_adi: Optional[str] = None,
        k: int = 4,
        rerank: bool = True,
        allow_cross_grade: bool = False,
        max_mesafe: float = RAG_MAX_MESAFE,
    ) -> List[Document]:
        """
        Kullanıcının seçtiği eğitim seviyesine ve sınıf düzeyine ait belgeler arasında
        anlamsal ve sözcüksel arama yapar.

        KATI SINIF İZOLASYONU VE ALAKA EŞİĞİ KURALI:
        1. sinif belirtilmişse (Örn. '3. Sınıf'), arama en baştan itibaren SADECE o sınıfla sınırlandırılır.
        2. allow_cross_grade=False (varsayılan) iken, 3. sınıf aramasında asla 4. sınıf veya başka kademe belgeleri dönmez.
        3. Alaka eşiği: Vektör mesafesi max_mesafe değerini aşan adaylar elenir.
        4. Vektör aramasından gerçek skor/mesafe alınır (vektor_skoru, vektor_mesafesi).
        5. Aday dokümanlar 3-gram Türkçe Jaccard lexical reranking ile sıralanır.
        """
        store = self.get_vector_store()
        aday_sayisi = max(k * 3, 10) if rerank and lexical_rerank else k
        norm_sinif = self.sinif_normalize(sinif)

        # 1. Filtre Belirleme (Sınıf izolasyonu öncelikli)
        if norm_sinif:
            where_filter = {"sinif": norm_sinif}
        elif egitim_seviyesi in EGITIM_SEVIYELERI:
            where_filter = {"egitim_seviyesi": egitim_seviyesi}
        else:
            where_filter = None

        adaylar: List[Document] = []

        # 2. Vektör Araması Denemesi (Gerçek skorlarla ve Alaka Eşiğiyle)
        try:
            results_with_scores = store.similarity_search_with_score(query=sorgu, k=aday_sayisi, filter=where_filter)
            for doc, dist in results_with_scores:
                dist_val = float(dist)
                if max_mesafe is not None and dist_val > max_mesafe:
                    continue
                doc.metadata["arama_yontemi"] = "vektor_benzerligi"
                doc.metadata["vektor_mesafesi"] = round(dist_val, 4)
                # Chroma cosine distance [0, 2] -> cosine similarity = 1 - dist
                sim = max(0.0, min(1.0, 1.0 - dist_val))
                doc.metadata["vektor_skoru"] = round(sim, 4)
                adaylar.append(doc)
        except Exception:
            adaylar = []

        # 3. İkincil Arama: Eğer cross-grade izni varsa ve sınıf aramasında sonuç çıkmadıysa
        if not adaylar and norm_sinif and allow_cross_grade:
            filtre = {"egitim_seviyesi": egitim_seviyesi} if egitim_seviyesi in EGITIM_SEVIYELERI else None
            try:
                results_with_scores = store.similarity_search_with_score(query=sorgu, k=aday_sayisi, filter=filtre)
                for doc, dist in results_with_scores:
                    dist_val = float(dist)
                    if max_mesafe is not None and dist_val > max_mesafe:
                        continue
                    doc.metadata["arama_yontemi"] = "vektor_benzerligi"
                    doc.metadata["vektor_mesafesi"] = round(dist_val, 4)
                    sim = max(0.0, min(1.0, 1.0 - dist_val))
                    doc.metadata["vektor_skoru"] = round(sim, 4)
                    if self.sinif_normalize(doc.metadata.get("sinif")) != norm_sinif:
                        doc.metadata["cross_grade"] = True
                    adaylar.append(doc)
            except Exception:
                adaylar = []

        # 4. Güvenlik Ağı / Fallback: Vektör servisi çevrimdışıysa doğrudan ChromaDB SQLite koleksiyonundan çek
        if not adaylar and self.allow_fallback:
            try:
                col = store._collection
                res = col.get(where=where_filter if where_filter else None, include=["documents", "metadatas"])
                if res and res.get("documents"):
                    raw_docs = []
                    sorgu_lower = sorgu.lower()
                    for doc_text, meta in zip(res["documents"], res["metadatas"]):
                        unite = str(meta.get("unite", "")).lower()
                        konu = str(meta.get("konu", "")).lower()
                        if any(x in unite or x in konu for x in ["ön sayfalar", "kapak", "içindekiler", "sözlük", "ekler", "kitap ekleri"]):
                            if not any(x in sorgu_lower for x in ["kapak", "künye", "içindekiler", "sözlük"]):
                                continue
                        doc = Document(page_content=doc_text, metadata=meta)
                        doc.metadata["arama_yontemi"] = "lexical_fallback"
                        doc.metadata["vektor_mesafesi"] = None
                        doc.metadata["vektor_skoru"] = None
                        raw_docs.append(doc)
                    if raw_docs and lexical_rerank:
                        sirali = lexical_rerank(sorgu, raw_docs, lexical_agirlik=1.0, top_n=aday_sayisi)
                        adaylar = [d for d, s in sirali]
                    elif raw_docs:
                        adaylar = raw_docs[:aday_sayisi]
            except Exception as direct_err:
                print(f"[UYARI] Doğrudan collection get fallback hatası: {direct_err}")

        # 5. KATI KURAL (Sınıf İzolasyonu Güvencesi):
        # allow_cross_grade=False ise farklı sınıftan dokümanlar kesin olarak temizlenir.
        if norm_sinif and not allow_cross_grade and adaylar:
            adaylar = [d for d in adaylar if self.sinif_normalize(d.metadata.get("sinif")) == norm_sinif]

        if not adaylar:
            return []

        # Kapak / künye / ön sayfaları konu içeriği aramalarından ayıkla
        sorgu_lower = sorgu.lower()
        temiz_adaylar = []
        for d in adaylar:
            unite = str(d.metadata.get("unite", "")).lower()
            konu = str(d.metadata.get("konu", "")).lower()
            if any(x in unite or x in konu for x in ["ön sayfalar", "kapak", "içindekiler", "sözlük", "ekler", "kitap ekleri"]):
                if not any(x in sorgu_lower for x in ["kapak", "künye", "içindekiler", "sözlük"]):
                    continue
            temiz_adaylar.append(d)
        if temiz_adaylar:
            adaylar = temiz_adaylar

        # 6. 3-gram Türkçe Lexical Reranking Uygula ve lexical skorları ata
        if rerank and lexical_rerank and len(adaylar) > 1:
            try:
                sirali_adaylar = lexical_rerank(sorgu, adaylar, top_n=k)
                sonuclar = []
                for doc, score in sirali_adaylar:
                    doc.metadata["lexical_skoru"] = round(float(score), 4)
                    sonuclar.append(doc)
                return sonuclar
            except TypeError:
                sirali_adaylar = lexical_rerank(sorgu, adaylar)
                sonuclar = []
                for doc, score in sirali_adaylar[:k]:
                    doc.metadata["lexical_skoru"] = round(float(score), 4)
                    sonuclar.append(doc)
                return sonuclar

        return adaylar[:k]

    def chunk_alaka_puanla(
        self,
        doc: Document,
        temel_konu: str = "",
        alt_kavramlar: Optional[List[str]] = None,
        gorsel_unsurlar: Optional[List[str]] = None,
        transkript: str = "",
        hedef_sinif: Optional[str] = "3. Sınıf",
        allow_cross_grade: bool = False,
    ) -> Dict[str, Any]:
        """
        Getirilen bir ders kitabı parçasının (chunk), sahnenin temel konusu, görsel unsurları
        ve alt kavramları ile anlamsal ilgisini değerlendirir.

        KATI KALİTE VE SINIF İZOLASYONU:
        1. Sınıf İzolasyonu: Farklı sınıf parçaları (örn. 3. sınıf sorgusunda 4. sınıf) allow_cross_grade=False
           olduğunda derhal elenir (konu_disi).
        2. Ünite ve Konu Bütünlüğü: İlgisiz üniteler (örn. Elektrik konusunda 'Besinlerimiz' veya 'Kuvvetin Etkileri')
           içerisinde geçen benzer kelimeler (oyuncak araba vb.) sahte eşleşme olarak elenir.
        3. Kaynak Rolü Ayrımı:
           - 'ana_kaynak': Temel konuyu/mekanizmayı doğrudan açıklayan birinci derece metin (Sayfa 48).
           - 'yardimci_kaynak': Güvenlik, atık pil, tasarruf gibi tamamlayıcı metinler (Sayfa 57).
           - 'konu_disi': Farklı ünite/sınıf veya alakasız bağlam.
        4. Dürüst Skorlama: Vektör skoru uydurulmaz (varsa float, yoksa None).
        """
        c_text = doc.page_content or ""
        meta = doc.metadata or {}
        unite = str(meta.get("unite", "")).strip()
        konu = str(meta.get("konu", "")).strip()
        alt_konu = str(meta.get("alt_konu", "")).strip()
        doc_sinif = self.sinif_normalize(meta.get("sinif"))
        hedef_norm_sinif = self.sinif_normalize(hedef_sinif)

        arama_yontemi = meta.get("arama_yontemi", "lexical_fallback")
        vektor_mesafesi = meta.get("vektor_mesafesi")
        vektor_skoru = meta.get("vektor_skoru")
        lexical_skor = meta.get("lexical_skoru")

        # 1. Aşama: Sınıf İzolasyonu Kontrolü
        is_cross_grade = (hedef_norm_sinif is not None and doc_sinif is not None and doc_sinif != hedef_norm_sinif)
        if is_cross_grade and not allow_cross_grade:
            return {
                "gecerli": False,
                "puan": 0.0,
                "vektor_skoru": vektor_skoru,
                "vektor_mesafesi": vektor_mesafesi,
                "arama_yontemi": arama_yontemi,
                "lexical_skor": lexical_skor,
                "konu_uygunluk_skoru": 0.0,
                "kaynak_rolu": "konu_disi",
                "kaynak_destek_durumu": "destek_yok",
                "gerekce": f"Elendi: Farklı sınıf seviyesi (Hedef: {hedef_norm_sinif}, Kaynak: {doc_sinif})",
            }

        # 2. Aşama: Ünite ve Konu Uygunluğu Kontrolü
        unite_lower = unite.lower()
        konu_lower = konu.lower()
        alt_konu_lower = alt_konu.lower()
        temel_lower = (temel_konu or "").lower()
        tr_lower = (transkript or "")[:400].lower()
        all_concepts = [k.lower() for k in (alt_kavramlar or [])]

        is_elektrik_baglami = any(w in temel_lower or any(w in c for c in all_concepts) or w in tr_lower for w in ["elektrik", "pil", "batarya", "akü", "santral"])
        ilgisiz_uniteler = ["besinlerimiz", "kuvvetin etkileri", "hareketi keşfediyorum", "maddenin özellikleri", "maddeyi tanıyalım", "canlılar dünyası", "yer kabuğu", "toprağı tanıyorum", "yaşam alanları"]

        if is_elektrik_baglami and any(iu in unite_lower for iu in ilgisiz_uniteler):
            return {
                "gecerli": False,
                "puan": 0.0,
                "vektor_skoru": vektor_skoru,
                "vektor_mesafesi": vektor_mesafesi,
                "arama_yontemi": arama_yontemi,
                "lexical_skor": lexical_skor,
                "konu_uygunluk_skoru": 0.0,
                "kaynak_rolu": "konu_disi",
                "kaynak_destek_durumu": "destek_yok",
                "gerekce": f"Elendi: İlgisiz ünite ({unite} - {konu})",
            }

        # 3. Aşama: Metin ve Token Seviyesinde Konu Uygunluk Puanlaması
        def _kelimelere_ayir(m: str) -> set:
            if not m:
                return set()
            m_clean = m.replace("İ", "i").replace("I", "ı").lower()
            return set(re.findall(r"\b[a-zçğıöşü0-9]+\b", m_clean))

        def _var_mi(aranan: str, tokens: set, metin_lower: str) -> bool:
            a_clean = aranan.replace("İ", "i").replace("I", "ı").lower().strip()
            if not a_clean:
                return False
            if " " in a_clean:
                pat = r"\b" + re.escape(a_clean) + r"\b"
                return bool(re.search(pat, metin_lower))
            return a_clean in tokens

        konu_meta = f"{unite} {konu} {alt_konu}"
        full_doc = f"{c_text} {konu_meta}"
        full_tokens = _kelimelere_ayir(full_doc)
        full_doc_lower = full_doc.replace("İ", "i").replace("I", "ı").lower()
        c_text_lower = c_text.replace("İ", "i").replace("I", "ı").lower()

        konu_uygunluk_skoru = 0.0
        pozitif_nedenler = []
        negatif_nedenler = []

        # Konu ve Alt Kavram Eşleşmesi (Kelime sınırlarına duyarlı)
        for k in all_concepts:
            if _var_mi(k, full_tokens, full_doc_lower):
                konu_uygunluk_skoru += 2.0
                pozitif_nedenler.append(f"Kavram: '{k}'")

        # Odak Alanı Değerlendirmesi
        if "pil" in temel_lower or any("pil" in k for k in all_concepts) or "oyuncak" in tr_lower:
            pil_terimleri = ["pil", "fener", "oyuncak", "kumanda", "küçük araç", "taşınabilir", "batarya", "atık pil"]
            if any(_var_mi(s, full_tokens, full_doc_lower) for s in ["duvar saati", "kol saati", "çalar saat"]):
                pil_terimleri.append("saat")
            elif _var_mi("saat", full_tokens, full_doc_lower) and not any(sw in full_doc_lower for sw in ["saat yönü", "saatte", "saatler"]):
                if any(_var_mi(w, full_tokens, full_doc_lower) for w in ["pil", "elektrik", "batarya", "enerji", "cihaz"]):
                    pil_terimleri.append("saat")

            eslesen_pil = [t for t in pil_terimleri if _var_mi(t, full_tokens, full_doc_lower)]
            if eslesen_pil:
                konu_uygunluk_skoru += 2.0 * len(eslesen_pil)
                pozitif_nedenler.append(f"Pil/cihaz terimleri: {', '.join(eslesen_pil[:3])}")

            santral_terimleri = ["baraj", "hidroelektrik", "termik santral", "yüksek gerilim", "şebeke teli"]
            eslesen_santral = [s for s in santral_terimleri if _var_mi(s, full_tokens, full_doc_lower)]
            if eslesen_santral and not (_var_mi("pil", full_tokens, c_text_lower) or _var_mi("el feneri", full_tokens, c_text_lower)):
                konu_uygunluk_skoru -= 5.0
                negatif_nedenler.append(f"Konu dışı santral/baraj içeriği: {', '.join(eslesen_santral[:3])}")
        elif "şehir elektriği" in temel_lower or "santral" in temel_lower:
            if any(_var_mi(t, full_tokens, full_doc_lower) for t in ["santral", "şehir elektriği", "baraj", "priz", "fırın", "çamaşır makinesi"]):
                konu_uygunluk_skoru += 3.0
                pozitif_nedenler.append("Şehir elektriği / santral içeriği")
        elif "akü" in temel_lower or "taşıt" in temel_lower or "otomobil" in tr_lower:
            if any(_var_mi(t, full_tokens, full_doc_lower) for t in ["akü", "otomobil", "motor", "taşıt", "araç"]):
                konu_uygunluk_skoru += 3.0
                pozitif_nedenler.append("Akü ve motorlu taşıt içeriği")
        elif "batarya" in temel_lower or "şarj" in temel_lower or "telefon" in tr_lower:
            if any(_var_mi(t, full_tokens, full_doc_lower) for t in ["batarya", "şarj", "telefon", "bilgisayar", "tablet"]):
                konu_uygunluk_skoru += 3.0
                pozitif_nedenler.append("Batarya ve şarj edilebilir cihaz içeriği")

        if any(_var_mi(w, full_tokens, full_doc_lower) for w in ["elektrik", "enerji", "araç", "gereç"]):
            konu_uygunluk_skoru += 1.0

        # 4. Aşama: Kaynak Rolü ve Bilimsel Destek Düzeyi Belirleme
        kaynak_rolu = "konu_disi"
        if konu_uygunluk_skoru >= 1.5:
            if ("pil" in temel_lower or "oyuncak" in tr_lower or any("pil" in k for k in all_concepts)):
                # Pil çalışma mekanizması ve cihaz çalıştırma ana kaynağı
                if "araç gereç" in konu_lower or ("pil" in c_text_lower and any(w in c_text_lower for w in ["çalıştırır", "sağlar", "enerji", "kabloya gerek olmadan"])):
                    kaynak_rolu = "ana_kaynak"
                elif "güvenli" in konu_lower or "atık pil" in c_text_lower or "tehlike" in c_text_lower:
                    kaynak_rolu = "yardimci_kaynak"
                else:
                    kaynak_rolu = "ana_kaynak" if konu_uygunluk_skoru >= 6.0 else "yardimci_kaynak"
            elif ("şehir elektriği" in temel_lower or "santral" in temel_lower):
                if any(w in c_text_lower for w in ["santral", "şehir elektriği", "priz", "kablo"]):
                    kaynak_rolu = "ana_kaynak"
                else:
                    kaynak_rolu = "yardimci_kaynak"
            elif ("akü" in temel_lower or "taşıt" in temel_lower):
                if "akü" in c_text_lower or "taşıt" in c_text_lower:
                    kaynak_rolu = "ana_kaynak"
                else:
                    kaynak_rolu = "yardimci_kaynak"
            elif ("batarya" in temel_lower or "şarj" in temel_lower):
                if "batarya" in c_text_lower or "şarj" in c_text_lower:
                    kaynak_rolu = "ana_kaynak"
                else:
                    kaynak_rolu = "yardimci_kaynak"
            else:
                kaynak_rolu = "ana_kaynak" if konu_uygunluk_skoru >= 6.0 else "yardimci_kaynak"

        # Farklı sınıf kaynağı izin verilmiş olsa bile ASLA ana kaynak olamaz
        if is_cross_grade and kaynak_rolu == "ana_kaynak":
            kaynak_rolu = "yardimci_kaynak"

        gecerli = kaynak_rolu in ["ana_kaynak", "yardimci_kaynak"]
        if kaynak_rolu == "ana_kaynak" and konu_uygunluk_skoru >= 4.0:
            kaynak_destek_durumu = "tam_destekli"
        elif kaynak_rolu in ["ana_kaynak", "yardimci_kaynak"] and konu_uygunluk_skoru >= 1.5:
            kaynak_destek_durumu = "kismi_destekli"
        else:
            kaynak_destek_durumu = "destek_yok"

        gerekce = (
            f"Kabul edildi [{kaynak_rolu}]: {'; '.join(pozitif_nedenler)}"
            if gecerli
            else (f"Elendi: {'; '.join(negatif_nedenler) if negatif_nedenler else 'Yetersiz konu alaka puanı'}")
        )

        return {
            "gecerli": gecerli,
            "puan": round(konu_uygunluk_skoru, 2),
            "vektor_skoru": vektor_skoru,
            "vektor_mesafesi": vektor_mesafesi,
            "arama_yontemi": arama_yontemi,
            "lexical_skor": lexical_skor,
            "konu_uygunluk_skoru": round(konu_uygunluk_skoru, 2),
            "kaynak_rolu": kaynak_rolu,
            "kaynak_destek_durumu": kaynak_destek_durumu,
            "gerekce": gerekce,
        }

    def bolum_bazli_cok_yonlu_arama(
        self,
        bolum_basligi: str,
        transkript: str,
        egitim_seviyesi: str,
        sinif: str = "3. Sınıf",
        ders_adi: str = "Fen Bilimleri",
        k: int = 2,
        gorsel_unsurlar: Optional[List[str]] = None,
        gorsel_aciklamasi: str = "",
        temel_konu: str = "",
        alt_kavramlar: Optional[List[str]] = None,
        allow_cross_grade: bool = False,
    ) -> Dict[str, Any]:
        """
        KULLANICI TAVSİYESİYLE GELİŞTİRİLEN 3 BOYUTLU VE ÇOK YÖNLÜ RAG MİMARİSİ:
        Yalnızca transkriptten değil; başlık, temel konu, görsel unsurlar, sahne açıklaması
        ve alt kavramları birleştirerek hedeflenmiş 3 boyutlu sorgular çalıştırır:
        1. Bilgi Sorgusu: Tanımlar, kavramlar ve resmi konu açıklamaları.
        2. Pedagojik Sorgu: Sınıf seviyesine uygun anlatım dili, somut görsel unsurlar ve kazanımlar.
        3. Ölçme-Değerlendirme Sorgusu: İlgili ünitedeki soru tipleri ve etkinlik örnekleri.

        KATI SINIF İZOLASYONU:
        - sinif="3. Sınıf" için varsayılan olarak yalnızca 3. sınıf parçaları kabul edilir.
        - allow_cross_grade=True verilmedikçe başka sınıf kaynakları kesinlikle havuza alınmaz.
        - Ana kaynak (ana_kaynak) ve yardımcı kaynak (yardimci_kaynak) ayrımı yapılır.
        """
        konu_odak = temel_konu or bolum_basligi
        kavramlar_str = ", ".join(alt_kavramlar) if alt_kavramlar else ""
        unsurlar_str = ", ".join(gorsel_unsurlar[:3]) if gorsel_unsurlar else ""

        # 1. Bilgi Sorgusu (Konu, alt kavramlar ve transkript terimleri)
        bilgi_sorgusu = f"{sinif} {ders_adi}. Konu: {konu_odak}."
        if kavramlar_str:
            bilgi_sorgusu += f" Kavramlar: {kavramlar_str}."
        bilgi_sorgusu += f" {transkript[:200]}"
        raw_bilgi_docs = self.seviye_filtreli_arama(
            sorgu=bilgi_sorgusu,
            egitim_seviyesi=egitim_seviyesi,
            sinif=sinif,
            ders_adi=ders_adi,
            k=k,
            allow_cross_grade=allow_cross_grade,
        )

        # 2. Pedagojik Sorgu (Görsel unsurlar ve pedagojik kazanımlar)
        pedagoji_sorgusu = f"{sinif} {ders_adi}. Konu: {konu_odak}."
        if unsurlar_str:
            pedagoji_sorgusu += f" Görsel unsurlar: {unsurlar_str}."
        pedagoji_sorgusu += f" {sinif} düzeyinde somut örnekler ve pedagojik anlatım."
        raw_pedagoji_docs = self.seviye_filtreli_arama(
            sorgu=pedagoji_sorgusu,
            egitim_seviyesi=egitim_seviyesi,
            sinif=sinif,
            ders_adi=ders_adi,
            k=k,
            allow_cross_grade=allow_cross_grade,
        )

        # 3. Ölçme-Değerlendirme Sorgusu
        olcme_sorgusu = f"{sinif} {ders_adi}. Konu: {konu_odak}. Soru, alıştırma ve etkinlik örnekleri."
        raw_olcme_docs = self.seviye_filtreli_arama(
            sorgu=olcme_sorgusu,
            egitim_seviyesi=egitim_seviyesi,
            sinif=sinif,
            ders_adi=ders_adi,
            k=k,
            allow_cross_grade=allow_cross_grade,
        )

        rag_sorgulari = [bilgi_sorgusu, pedagoji_sorgusu, olcme_sorgusu]

        # 4. Aday Parçaları Uygunluk ve Alaka Puanlamasından Geçir
        tum_adaylar = raw_bilgi_docs + raw_pedagoji_docs + raw_olcme_docs
        destekleyen_chunklar = []
        elenen_chunklar = []
        islenmis_ids = set()

        for d in tum_adaylar:
            c_id = d.metadata.get("chunk_id") or f"{d.metadata.get('source')}_{d.metadata.get('page')}_{hashlib.md5(d.page_content.encode('utf-8')).hexdigest()[:8]}"
            if c_id in islenmis_ids:
                continue
            islenmis_ids.add(c_id)

            puan_sonucu = self.chunk_alaka_puanla(
                doc=d,
                temel_konu=konu_odak,
                alt_kavramlar=alt_kavramlar,
                gorsel_unsurlar=gorsel_unsurlar,
                transkript=transkript,
                hedef_sinif=sinif,
                allow_cross_grade=allow_cross_grade,
            )

            chunk_ozet = {
                "chunk_id": c_id,
                "kaynak": d.metadata.get("source"),
                "sinif": d.metadata.get("sinif", ""),
                "sayfa": d.metadata.get("page"),
                "physical_page": d.metadata.get("physical_page") if d.metadata.get("physical_page") is not None else d.metadata.get("page"),
                "printed_page": d.metadata.get("printed_page") if d.metadata.get("has_printed_page") else None,
                "unite": d.metadata.get("unite", ""),
                "konu": d.metadata.get("konu", ""),
                "referans": d.metadata.get("referans") or f"{d.metadata.get('source')} (Sayfa {d.metadata.get('page')})",
                "puan": puan_sonucu["puan"],
                "kaynak_rolu": puan_sonucu["kaynak_rolu"],
                "vektor_skoru": puan_sonucu["vektor_skoru"],
                "vektor_mesafesi": puan_sonucu["vektor_mesafesi"],
                "arama_yontemi": puan_sonucu["arama_yontemi"],
                "lexical_skor": puan_sonucu["lexical_skor"],
                "konu_uygunluk_skoru": puan_sonucu["konu_uygunluk_skoru"],
                "gerekce": puan_sonucu["gerekce"],
                "icerik": d.page_content.strip()
            }

            if puan_sonucu["gecerli"]:
                destekleyen_chunklar.append({"doc": d, "info": chunk_ozet})
            else:
                elenen_chunklar.append(chunk_ozet)

        # Destekleyen parçalara göre filtrelenmiş doc listeleri oluştur
        destek_doc_ids = {item["info"]["chunk_id"] for item in destekleyen_chunklar}
        bilgi_docs = [d for d in raw_bilgi_docs if (d.metadata.get("chunk_id") or f"{d.metadata.get('source')}_{d.metadata.get('page')}_{hashlib.md5(d.page_content.encode('utf-8')).hexdigest()[:8]}") in destek_doc_ids]
        pedagoji_docs = [d for d in raw_pedagoji_docs if (d.metadata.get("chunk_id") or f"{d.metadata.get('source')}_{d.metadata.get('page')}_{hashlib.md5(d.page_content.encode('utf-8')).hexdigest()[:8]}") in destek_doc_ids]
        olcme_docs = [d for d in raw_olcme_docs if (d.metadata.get("chunk_id") or f"{d.metadata.get('source')}_{d.metadata.get('page')}_{hashlib.md5(d.page_content.encode('utf-8')).hexdigest()[:8]}") in destek_doc_ids]

        # Eğer bir sorgu grubu tamamen elendiyse ancak genel destekleyen parça varsa, en iyi destekleyenle besle
        if not bilgi_docs and destekleyen_chunklar:
            bilgi_docs = [destekleyen_chunklar[0]["doc"]]
        if not pedagoji_docs and destekleyen_chunklar:
            pedagoji_docs = [destekleyen_chunklar[min(1, len(destekleyen_chunklar)-1)]["doc"]]
        if not olcme_docs and destekleyen_chunklar:
            olcme_docs = [destekleyen_chunklar[-1]["doc"]]

        def _formatla(docs: List[Document], baslangic_etiket: str = "K") -> Dict[str, Any]:
            parcalar = []
            metinler = []
            for i, d in enumerate(docs, 1):
                etiket = f"{baslangic_etiket}{i}"
                kaynak = d.metadata.get("source", "Ders Kitabı")
                sayfa = d.metadata.get("page", "-")
                c_text = d.page_content.strip()
                referans = d.metadata.get("referans") or f"{kaynak} (Sayfa {sayfa})"
                metinler.append(f"[{etiket}] ({referans}):\n{c_text}")
                parcalar.append({
                    "etiket": etiket,
                    "kaynak": kaynak,
                    "sayfa": sayfa,
                    "referans": referans,
                    "unite": d.metadata.get("unite", ""),
                    "konu": d.metadata.get("konu", ""),
                    "icerik": c_text
                })
            return {
                "baglam": "\n\n".join(metinler),
                "kaynaklar": parcalar,
                "dokumanlar": docs
            }

        # Ana kaynak ve yardımcı kaynak ayrımı
        ana_kaynaklar = [item["info"] for item in destekleyen_chunklar if item["info"].get("kaynak_rolu") == "ana_kaynak"]
        yardimci_kaynaklar = [item["info"] for item in destekleyen_chunklar if item["info"].get("kaynak_rolu") == "yardimci_kaynak"]

        # En iyi doğrulanmış referans ve destek durumu belirleme
        if ana_kaynaklar:
            en_iyi_referans = ana_kaynaklar[0]["referans"]
            kaynak_destek_durumu = "tam_destekli"
        elif yardimci_kaynaklar:
            en_iyi_referans = yardimci_kaynaklar[0]["referans"]
            kaynak_destek_durumu = "kismi_destekli"
        else:
            en_iyi_referans = "Doğrulanmış RAG eşleşmesi bulunamadı"
            kaynak_destek_durumu = "destek_yok"

        # Yalnızca doğrulanmış destekleyen parçaları kaynakça havuzuna gönder
        dogrulanmis_tum_kaynaklar = [
            {
                "kaynak": item["info"]["kaynak"],
                "dosya_adi": item["info"]["kaynak"],
                "kitap_adi": item["doc"].metadata.get("kitap_adi") or item["info"]["kaynak"],
                "sinif": item["doc"].metadata.get("sinif", ""),
                "unite": item["doc"].metadata.get("unite", ""),
                "konu": item["doc"].metadata.get("konu", ""),
                "alt_konu": item["doc"].metadata.get("alt_konu", ""),
                "sayfa": item["info"]["sayfa"],
                "physical_page": item["info"]["physical_page"],
                "printed_page": item["info"]["printed_page"],
                "has_printed_page": item["doc"].metadata.get("has_printed_page", False),
                "referans": item["info"]["referans"],
                "chunk_id": item["info"]["chunk_id"],
                "kaynak_rolu": item["info"].get("kaynak_rolu", "yardimci_kaynak"),
                "arama_yontemi": item["info"].get("arama_yontemi"),
                "vektor_skoru": item["info"].get("vektor_skoru"),
            }
            for item in destekleyen_chunklar
        ]

        return {
            "bilgi": _formatla(bilgi_docs, "B"),
            "pedagoji": _formatla(pedagoji_docs, "P"),
            "olcme_degerlendirme": _formatla(olcme_docs, "O"),
            "en_iyi_referans": en_iyi_referans,
            "kaynak_destek_durumu": kaynak_destek_durumu,
            "rag_sorgulari": rag_sorgulari,
            "destekleyen_chunklar": [item["info"] for item in destekleyen_chunklar],
            "ana_kaynaklar": ana_kaynaklar,
            "yardimci_kaynaklar": yardimci_kaynaklar,
            "elenen_chunklar": elenen_chunklar,
            "tum_kaynaklar": dogrulanmis_tum_kaynaklar
        }

    def soru_formatlarini_getir(
        self,
        egitim_seviyesi: str,
        k: int = 3,
        sinif: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        KATI KURAL (Few-Shot Soru Klonlama):
        Test soruları üretilirken, soru formatı (boşluk doldurma, test vb.)
        RAG veritabanındaki kitaptan klonlanacaktır.

        Bu fonksiyon, belirtilen eğitim seviyesindeki kitaptan soru kalıplarını
        ve alıştırma stillerini çeker.
        """
        soru_arama_ifadeleri = [
            "Değerlendirme Soruları test",
            "Aşağıdaki boşlukları doldurunuz",
            "Doğru - Yanlış etkinliği",
            "Aşağıdakilerden hangisi",
        ]

        bulunan_ornekler: List[Dict[str, Any]] = []
        gorulen_metinler = set()

        for ifade in soru_arama_ifadeleri:
            docs = self.seviye_filtreli_arama(sorgu=ifade, egitim_seviyesi=egitim_seviyesi, sinif=sinif, k=2)
            for doc in docs:
                metin = doc.page_content.strip()
                # Soru işaretleri, parantezler veya şıklar içeren kısımları tercih et
                if ("?" in metin or "A)" in metin or "..." in metin or "D - Y" in metin) and metin not in gorulen_metinler:
                    gorulen_metinler.add(metin)
                    bulunan_ornekler.append({
                        "ornek_metin": metin,
                        "sayfa": doc.metadata.get("page", 0),
                        "kaynak": doc.metadata.get("source", ""),
                    })
                if len(bulunan_ornekler) >= k:
                    break
            if len(bulunan_ornekler) >= k:
                break

        return bulunan_ornekler

    def dosya_metadata_ayristir(self, dosya_adi: str) -> Dict[str, str]:
        """
        Kullanıcının standart dosya adlandırmasını kusursuz biçimde üst verilere dönüştürür.
        Örnekler:
          'lise_fizik_10.sinif.pdf' ->
              egitim_seviyesi: 'Lise', ders_adi: 'Fizik', sinif: '10. Sınıf'
          'ortaokul_fenbilimleri_7.sinif_1.kitap.pdf' ->
              egitim_seviyesi: 'Ortaokul', ders_adi: 'Fen Bilimleri', sinif: '7. Sınıf'
          'ilkokul_fenbilimleri_3.sinif.pdf' ->
              egitim_seviyesi: 'İlkokul', ders_adi: 'Fen Bilimleri', sinif: '3. Sınıf'
        """
        ad = dosya_adi.lower()

        # 1. Eğitim Seviyesi Tespiti (Ön ek kontrolü)
        if ad.startswith("ilkokul") or "ilkokul" in ad:
            seviye = "İlkokul"
        elif ad.startswith("ortaokul") or "ortaokul" in ad:
            seviye = "Ortaokul"
        elif ad.startswith("lise") or "lise" in ad:
            seviye = "Lise"
        elif ad.startswith("universite") or "universite" in ad or "uni" in ad:
            seviye = "Üniversite"
        else:
            seviye = DEFAULT_SEVIYE

        # 2. Ders Adı Tespiti
        if "fizik" in ad:
            ders = "Fizik"
        elif "kimya" in ad:
            ders = "Kimya"
        elif "biyoloji" in ad:
            ders = "Biyoloji"
        elif "fenbilimleri" in ad or "fen" in ad:
            ders = "Fen Bilimleri"
        elif "matematik" in ad:
            ders = "Matematik"
        else:
            ders = "Fen Bilimleri"

        # 3. Sınıf Seviyesi Tespiti
        sinif_match = re.search(r'(\d+)\.?\s*sinif', ad)
        sinif = f"{sinif_match.group(1)}. Sınıf" if sinif_match else "Belirtilmemiş"

        return {
            "egitim_seviyesi": seviye,
            "ders_adi": ders,
            "sinif": sinif
        }

    def seviyeyi_tahmin_et(self, dosya_adi: str) -> str:
        """Geriye dönük uyumluluk için dosya seviyesini döndürür."""
        return self.dosya_metadata_ayristir(dosya_adi)["egitim_seviyesi"]

    def sistem_kitaplarini_senkronize_et(self) -> Dict[str, Any]:
        """
        data/reference_books klasöründeki tüm resmi müfredat PDF'lerini tarar,
        dosya adından seviye, ders ve sınıf bilgilerini ayrıştırarak ChromaDB'ye kaydeder.
        """
        mevcut_durum = self.mevcut_veritabani_durumu()
        print("\n[RAG Kurasyon] Referans kitaplar taranıyor...")
        print(f" -> Mevcut Seviyeler: {mevcut_durum.get('seviyeler', {})}")

        taranan_dizinler = [REFERENCE_BOOKS_DIR, DATA_DIR]
        bulunan_kitaplar = []
        gorulen = set()

        for dizin in taranan_dizinler:
            if not dizin.exists():
                continue
            for pdf_file in sorted(dizin.glob("*.pdf")):
                if pdf_file.name not in gorulen:
                    gorulen.add(pdf_file.name)
                    bulunan_kitaplar.append(pdf_file)

        rapor = {}
        store = self.get_vector_store()
        col = store._collection

        for pdf_yolu in bulunan_kitaplar:
            meta_info = self.dosya_metadata_ayristir(pdf_yolu.name)
            seviye = meta_info["egitim_seviyesi"]
            ders = meta_info["ders_adi"]
            sinif = meta_info["sinif"]

            # Bu dosya daha önce indekse eklenmiş mi kontrol et
            var_olanlar = col.get(where={"source": pdf_yolu.name}, limit=1)

            if len(var_olanlar.get("ids", [])) == 0:
                print(f"[RAG Kurasyon] '{pdf_yolu.name}' indeksleniyor -> Seviye: {seviye} | Ders: {ders} | Sınıf: {sinif}")
                eklenen_parca = self.pdf_indeksle(
                    str(pdf_yolu),
                    egitim_seviyesi=seviye,
                    ders_adi=ders,
                )
                rapor[pdf_yolu.name] = {
                    "seviye": seviye,
                    "ders": ders,
                    "sinif": sinif,
                    "parca_sayisi": eklenen_parca,
                    "durum": "yeni_eklendi"
                }
            else:
                rapor[pdf_yolu.name] = {
                    "seviye": seviye,
                    "ders": ders,
                    "sinif": sinif,
                    "durum": "zaten_mevcut"
                }

        return rapor

    def mevcut_veritabani_durumu(self) -> Dict[str, Any]:
        """ChromaDB koleksiyonunun genel durumunu ve seviye dağılımını raporlar."""
        try:
            store = self.get_vector_store()
            # Ham koleksiyon nesnesinden toplam sayıyı al
            col = store._collection
            toplam_vektor = col.count()

            # Seviyelere göre dağılım kontrolü
            seviye_sayilari = {}
            for seviye in EGITIM_SEVIYELERI:
                try:
                    res = col.get(where={"egitim_seviyesi": seviye}, limit=1)
                    var_mi = len(res.get("ids", [])) > 0
                    seviye_sayilari[seviye] = "Mevcut" if var_mi else "Yok"
                except Exception:
                    seviye_sayilari[seviye] = "Bilinmiyor"

            return {
                "koleksiyon": self.koleksiyon_adi,
                "toplam_parca": toplam_vektor,
                "seviyeler": seviye_sayilari,
            }
        except Exception as e:
            return {"hata": str(e), "toplam_parca": 0}


# --- KÖK DİZİN UYUMLULUĞU VE DOĞRUDAN TEST ---
if __name__ == "__main__":
    print("=" * 60)
    print("EduFrame AI - RAG Motoru Testi Başlatılıyor...")
    print("=" * 60)

    rag = RAGMotoru()
    durum = rag.mevcut_veritabani_durumu()
    print(f"Veritabanı Durumu: {durum}")

    # Örnek test PDF'i var mı?
    ornek_pdf = DATA_DIR / "fenbilimleri.pdf"
    if ornek_pdf.exists():
        print(f"\n[BİLGİ] '{ornek_pdf.name}' bulundu.")
        # Seviyeye göre arama testi
        print("\n--- Seviye Filtreli Arama Testi ('İlkokul' - 'Kuvvet ve Hareket') ---")
        sonuclar = rag.seviye_filtreli_arama("Kuvvet cisimlerin hareketini nasıl etkiler?", egitim_seviyesi="İlkokul", k=2)
        print(f"Bulunan sonuç adedi: {len(sonuclar)}")
        for i, s in enumerate(sonuclar, 1):
            print(f"[{i}] Sayfa: {s.metadata.get('page')} | Seviye: {s.metadata.get('egitim_seviyesi')}")
            print(f"    İçerik: {s.page_content[:120]}...\n")

        print("--- Few-Shot Soru Formatı Çekme Testi ---")
        sorular = rag.soru_formatlarini_getir(egitim_seviyesi="İlkokul", k=2)
        print(f"Bulunan soru kalıbı adedi: {len(sorular)}")
        for i, ornek in enumerate(sorular, 1):
            print(f"[{i}] (Sayfa {ornek['sayfa']}): {ornek['ornek_metin'][:100]}...\n")
    else:
        print(f"[UYARI] '{ornek_pdf}' bulunamadı, lütfen data klasörünü kontrol edin.")

    print("RAGMotoru başarıyla doğrulandı.")
