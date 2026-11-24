"""
EduFrame AI - Türkçe 3-Gram Lexical Benzerlik ve Reranking Araçları (utils/lexical_tools.py)
--------------------------------------------------------------------------------------------
Bu modül hiçbir dış laboratuvar veya eğitim script'ine bağımlı olmadan çalışan
saf Türkçe metin işleme, 3-gram trigram benzerliği, hibrit reranking ve atıf doğrulama fonksiyonlarını sunar.
"""

import re
from typing import List, Dict, Any, Tuple, Set, Union, Optional


def kelimelere_ayir(metin: str) -> List[str]:
    """
    Türkçe I/İ harf dönüşümlerini koruyarak metni küçük harfli kelimelere ayırır.
    Noktalama işaretlerini ayıklar.
    """
    if not metin:
        return []
    metin = metin.replace("İ", "i").replace("I", "ı").lower()
    return re.findall(r"[a-zçğıöşü0-9]+", metin)


def kavram_anahtari(metin: str) -> str:
    """Kavram adlarının yaygın yazım ve çoğul biçimlerini karşılaştırır."""
    kelimeler = kelimelere_ayir(metin)
    if not kelimeler:
        return ""
    son = kelimeler[-1]
    if len(son) >= 6 and son.endswith(("lar", "ler")):
        kelimeler[-1] = son[:-3]
    return " ".join(kelimeler).replace("araç ve gereç", "araç gereç")


def ucgramlar(metin: str) -> Set[str]:
    """
    Türkçe ek ve kök farklılıklarına toleranslı karakter 3-gramları (trigrams) kümesi üretir.
    Kelimelerin başına '^' ve sonuna '$' ekleyerek kelime başı/sonu sınırlarını korur.
    """
    gruplar: Set[str] = set()
    for kelime in kelimelere_ayir(metin):
        kelime_pad = f"^{kelime}$"
        for i in range(len(kelime_pad) - 2):
            gruplar.add(kelime_pad[i : i + 3])
    return gruplar


def lexical_similarity(soru: str, metin: str) -> float:
    """
    Sorgunun karakter 3-gramlarının metin içerisinde bulunma oranını [0.0 - 1.0] hesaplar.
    Formül: len(soru_gram & metin_gram) / len(soru_gram)
    """
    soru_gram = ucgramlar(soru)
    if not soru_gram:
        return 0.0
    belge_gram = ucgramlar(metin)
    return len(soru_gram & belge_gram) / len(soru_gram)


def tokens(text: str) -> List[str]:
    """Metni küçük harfli kelime token'larına ayırır."""
    if not text:
        return []
    return re.findall(r"\w+", text.lower())


def lexical_score(query: str, text: str) -> float:
    """Sorgu ile metin arasındaki 3-gram lexical benzerlik skoru."""
    return lexical_similarity(query, text)


def compute_noise_penalty(
    content: str,
    dot_threshold: float = 0.15,
    min_penalty_floor: float = 0.15
) -> float:
    """
    Metindeki gürültü (boşluk doldurma noktaları, aşırı noktalama) oranına göre
    yumuşak ve kademeli bir ceza katsayısı (min_penalty_floor - 1.0) hesaplar.
    dot_threshold (varsayılan 0.15) altındaki temiz metinlerde ceza uygulanmaz (katsayı 1.0).
    """
    if not content:
        return min_penalty_floor

    c_len = len(content)
    dots = content.count('.')
    dot_ratio = dots / c_len

    if dot_ratio <= dot_threshold:
        return 1.0

    # Eşik üzerindeki gürültü için sürekli azalan kademeli ceza
    excess = dot_ratio - dot_threshold
    penalty = 1.0 - (excess / (1.0 - dot_threshold)) * 1.5
    return max(min_penalty_floor, min(1.0, penalty))


def clean_snippet_preview(text: str, max_chars: int = 150) -> str:
    """
    Arayüz ve loglama için özet kesiti üretirken baştaki nokta dizilerini ve boşlukları budar.
    """
    if not text:
        return ""
    cleaned = re.sub(r'^[\s\.\_\-\–\—\•\*]+', '', text)
    cleaned = cleaned.replace('\n', ' ')
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()
    if len(cleaned) > max_chars:
        return cleaned[:max_chars] + "..."
    return cleaned


def rerank_candidates(
    soru: str,
    adaylar: List[Any],
    lexical_agirlik: float = 0.4,
    top_n: int = 3,
    apply_noise_penalty: bool = True,
    dot_threshold: float = 0.15
) -> List[Tuple[Any, float]]:
    """
    Aday dokümanları hem ilk anlamsal sıra pozisyonu (1/sira) hem de Türkçe 3-gram
    benzerlik puanını harmanlayan ve gürültü cezası (noise penalty) uygulayan hibrit skorla yeniden sıralar.
    
    Formül:
        anlamsal_puan = 1.0 / sira
        lexical_puan = lexical_similarity(soru, doc.page_content)
        toplam_puan = ((1.0 - lexical_agirlik) * anlamsal_puan + lexical_agirlik * lexical_puan) * noise_penalty

    :param soru: Kullanıcı veya RAG sorgusu
    :param adaylar: LangChain Document nesneleri veya metin nesneleri listesi
    :param lexical_agirlik: 0.0 ile 1.0 arasında lexical ağırlık katsayısı (Varsayılan 0.4)
    :param top_n: Döndürülecek en yüksek puanlı aday sayısı
    :param apply_noise_penalty: Gürültülü/nokta ağırlıklı parçalara ceza uygulansın mı (Varsayılan True)
    :param dot_threshold: Ceza başlangıç nokta oranı eşiği (Varsayılan 0.15)
    :return: [(belge, toplam_skor), ...] listesi
    """
    if not adaylar:
        return []

    if not 0.0 <= lexical_agirlik <= 1.0:
        raise ValueError(f"lexical_agirlik 0.0 ile 1.0 arasında olmalıdır, verilen: {lexical_agirlik}")

    puanli: List[Tuple[Any, float]] = []
    for sira, belge in enumerate(adaylar, 1):
        anlamsal_sira_puani = 1.0 / sira
        content = getattr(belge, "page_content", str(belge))
        lexical_puan = lexical_similarity(soru, content)
        toplam = ((1.0 - lexical_agirlik) * anlamsal_sira_puani + lexical_agirlik * lexical_puan)

        if apply_noise_penalty:
            penalty = compute_noise_penalty(content, dot_threshold=dot_threshold)
            toplam *= penalty

        puanli.append((belge, toplam))

    puanli.sort(key=lambda item: -item[1])
    return puanli[:top_n]


def validate_citations(answer: str, known_labels: List[str]) -> Dict[str, Any]:
    """
    Yanıt metni içerisindeki [K1], [K2] vb. kaynak etiketlerini doğrular.
    Tüm atıflar bilinen etiketler listesinde varsa 'valid': True döner.
    """
    found = re.findall(r"\[(K\d+)\]", answer)
    known_set = set(known_labels)
    invalid_citations = [f for f in found if f not in known_set]
    is_valid = len(found) > 0 and len(invalid_citations) == 0
    return {
        "valid": is_valid,
        "found": found,
        "invalid": invalid_citations
    }


def hit_at_k(docs: List[Any], expected: Union[int, str, List[Any], Tuple[Any, ...], Set[Any]]) -> int:
    """
    İlk k dokümanda beklenen kaynağın (source) veya sayfanın (page) olup olmadığını döner (1 veya 0).
    expected int veya liste ise sayfa numarası (page), string ise dosya adı (source) kontrol edilir.
    """
    if not docs:
        return 0

    if isinstance(expected, (int, list, tuple, set)):
        expected_pages = {expected} if isinstance(expected, int) else set(expected)
        for d in docs:
            meta = getattr(d, "metadata", {}) or {}
            # Hem page hem sayfa_no hem printed_page kontrol et
            for key in ["page", "sayfa_no", "printed_page"]:
                if meta.get(key) in expected_pages:
                    return 1
        return 0

    # Dosya adı (source) kontrolü
    for d in docs:
        meta = getattr(d, "metadata", {}) or {}
        if meta.get("source") == expected:
            return 1
    return 0
