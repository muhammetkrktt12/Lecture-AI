"""
EduFrame AI - Seviyeye Duyarlı Değerlendirme ve Etkinlik Soruları Servisi (services/question_generator.py)
----------------------------------------------------------------------------------------------------------
Bu modül, video dersinde işlenen konular ve MEB ders kitaplarındaki kazanımlar, etkinlikler ve soru kalıplarını
birlikte sentezleyerek öğrenci seviyesine tam uyumlu, özgün ve modern değerlendirme soruları üretir.

Temel Kurallar:
1. Sorular yalnızca video görüntüsünün kendisine dair bir sınav değil; videodaki bilimsel bilgiler ile ders kitabındaki
   pedagojik etkinliklerin sentezidir.
2. Her yeni öğrenilen kavram için mutlaka en az 1 soru oluşturulur.
3. Sorular modern, ilgi çekici tematik kategorilere (💡 Kavram Keşfi, 🔍 Günlük Yaşam Analizi, 🧩 Kavram Tamamlama,
   ⚡ Elektrik Dedektifi, 🎯 Doğru mu Yanlış mı?, 🌱 Doğa ve Çevre Bilinci, 🛡️ Güvenlik Dedektifi) ayrılır.
4. Sorular karmaşık ve sıkışık metin blokları yerine ferah kart formatında sunulur.
"""

import os
import json
import time
import re
from collections import Counter
from functools import partial
from random import choice, shuffle
from pathlib import Path
import requests
from services.gemini_rate_limit import gemini_istegini_beklet, gemini_kota_hatasini_isle
from services.gemini_usage import gemini_kullanimini_kaydet
from services.gemini_analyzer import GeminiFasikulHatasi
from utils.lexical_tools import kelimelere_ayir
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional

if TYPE_CHECKING:
    from services.rag_motoru import RAGMotoru

try:
    from utils.lexical_tools import lexical_similarity
except ImportError:
    try:
        from quality_utils import lexical_similarity
    except ImportError:
        lexical_similarity = None

try:
    from config import (
        GEMINI_API_KEY,
        GEMINI_MODEL,
        GEMINI_TIMEOUT,
        GEMINI_MAX_RETRIES,
        GEMINI_FALLBACK_MODELS,
        OLLAMA_URL,
        OLLAMA_MODEL,
        OLLAMA_QUESTION_TIMEOUT,
        OLLAMA_QUESTION_NUM_PREDICT,
    )
except ImportError:
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
    GEMINI_MODEL = "gemini-3.8-flash"
    GEMINI_TIMEOUT = int(os.getenv("GEMINI_TIMEOUT", "20"))
    GEMINI_MAX_RETRIES = int(os.getenv("GEMINI_MAX_RETRIES", "2"))
    GEMINI_FALLBACK_MODELS = [m.strip() for m in os.getenv("GEMINI_FALLBACK_MODELS", "gemini-2.5-flash,gemini-flash-latest").split(",") if m.strip()]
    OLLAMA_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen3.5:4b")
    OLLAMA_QUESTION_TIMEOUT = int(os.getenv("OLLAMA_QUESTION_TIMEOUT", "90"))
    OLLAMA_QUESTION_NUM_PREDICT = int(os.getenv("OLLAMA_QUESTION_NUM_PREDICT", "2000"))


class QuestionGenerator:
    """
    Video materyal kavramlarına ve MEB müfredat standartlarına göre özgün sorular üreten motor.

    Öncelik Sırası:
    1. Öncelik: Yerel Ollama (qwen3.5:4b)
    2. Yedek: Google Gemini API (gemini-3.8-flash)
    3. Güvenlik Ağı: MEB Pedagojik Standart Soru Bankası (Gerçek RAG Kaynak Eşleştirmeli)
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = GEMINI_MODEL,
        ollama_url: str = OLLAMA_URL,
        ollama_model: str = OLLAMA_MODEL,
        gemini_timeout: int = GEMINI_TIMEOUT,
        gemini_max_retries: int = GEMINI_MAX_RETRIES,
        fallback_models: Optional[List[str]] = None,
        ollama_question_timeout: int = OLLAMA_QUESTION_TIMEOUT,
        ollama_question_num_predict: int = OLLAMA_QUESTION_NUM_PREDICT,
    ):
        self.api_key = api_key if api_key is not None else (GEMINI_API_KEY or os.getenv("GEMINI_API_KEY", ""))
        self.model = model
        self.ollama_url = ollama_url or OLLAMA_URL
        self.ollama_model = ollama_model or OLLAMA_MODEL
        self.gemini_timeout = gemini_timeout
        self.gemini_max_retries = gemini_max_retries
        self.fallback_models = fallback_models if fallback_models is not None else list(GEMINI_FALLBACK_MODELS)
        self.ollama_question_timeout = ollama_question_timeout
        self.ollama_question_num_predict = ollama_question_num_predict
        self._ollama_devre_disi = False

    def _api_anahtari_var_mi(self) -> bool:
        return bool(self.api_key and len(self.api_key.strip()) > 10 and not self.api_key.startswith("AIzaSyDummy"))

    @property
    def ollama_devre_disi(self) -> bool:
        return self._ollama_devre_disi

    @staticmethod
    def _json_sorulari_ayristir(metin: str) -> Optional[List[Dict[str, Any]]]:
        if not metin:
            return None
        metin = metin.strip()
        try:
            if "```json" in metin:
                metin = metin.split("```json")[1].split("```")[0].strip()
            elif "```" in metin:
                metin = metin.split("```")[1].split("```")[0].strip()
            veri = json.loads(metin)
            if isinstance(veri, dict) and "sorular" in veri:
                veri = veri["sorular"]
            elif isinstance(veri, dict) and "questions" in veri:
                veri = veri["questions"]
            elif isinstance(veri, dict) and "soru" in veri:
                veri = [veri]
            if isinstance(veri, list) and len(veri) > 0:
                for idx, q in enumerate(veri, 1):
                    q["no"] = idx
                return veri
        except Exception:
            pass
        return None

    def _soru_istemi_olustur(self, video_analizi: Dict[str, Any], egitim_seviyesi: str, soru_sayisi: int = 10) -> str:
        ders = video_analizi.get("ders_adi", "Fen Bilimleri")
        unite = video_analizi.get("unite_basligi", "")
        bolumler = video_analizi.get("bolumler", [])
        kavramlar = []
        for b in bolumler:
            for k in b.get("onemli_kavramlar", []):
                kavramlar.append(k.get("kavram", ""))
        kavramlar_str = ", ".join(list(set(kavramlar))[:15])

        return f"""Sen MEB müfredatına hakim uzman bir Eğitim Teknoloğusun.
Ders: {ders}, Seviye: {egitim_seviyesi}, Ünite: {unite}.
Öğrenilen Kavramlar: {kavramlar_str}.

GÖREV: Bu ders için tam {soru_sayisi} adet pedagojik, seviyeye uygun özgün değerlendirme sorusu üret.
Soruların çoğu Çoktan Seçmeli (A, B, C, D seçenekli), en az 2 tanesi Doğru - Yanlış formatında olmalıdır.

JSON FORMATI:
{{
  "sorular": [
    {{
      "no": 1,
      "kategori": "💡 Kavram Keşfi",
      "ilgili_kavram": "Kavram Adı",
      "tip": "Çoktan Seçmeli",
      "soru": "Soru metni...",
      "secenekler": ["A) ...", "B) ...", "C) ...", "D) ..."],
      "dogru_cevap": "A",
      "aciklama": "Gerekçeli çözüm açıklaması.",
      "kazanim_referansi": "MEB Ders Kitabı"
    }}
  ]
}}
SADECE geçerli bir JSON nesnesi döndür."""

    def _gemini_api_cagir(
        self,
        payload: Dict[str, Any],
        timeout: Optional[int] = None,
        max_retries: Optional[int] = None,
        *,
        asama: str = "soru_uretimi",
        yalniz_zaman_asiminda_tekrarla: bool = False,
    ) -> Optional[List[Dict[str, Any]]]:
        """
        Google Gemini API'ye istek gönderir.
        HTTP 503 (High Demand) durumunda exponential backoff ile yeniden dener.
        Gerektiğinde tanımlı alternatif modellere (GEMINI_FALLBACK_MODELS) geçer.
        """
        if not self._api_anahtari_var_mi():
            return None

        timeout_sec = timeout or self.gemini_timeout
        retries = max_retries if max_retries is not None else self.gemini_max_retries
        models_to_try = [self.model]
        if asama != "soru_kaynak_dogrulama" and not yalniz_zaman_asiminda_tekrarla:
            models_to_try.extend(m for m in self.fallback_models if m != self.model)

        zaman_asimindan_sonra = False
        headers = {"x-goog-api-key": self.api_key, "Content-Type": "application/json"}
        for current_model in models_to_try:
            for attempt in range(retries + 1):
                url = f"https://generativelanguage.googleapis.com/v1beta/models/{current_model}:generateContent"
                try:
                    gemini_istegini_beklet(current_model)
                    res = requests.post(url, json=payload, headers=headers, timeout=timeout_sec)
                    if res.status_code not in (200, 429):
                        gemini_kullanimini_kaydet(current_model, asama, durum=str(res.status_code))
                    if res.status_code == 200:
                        yanit = res.json()
                        gemini_kullanimini_kaydet(current_model, asama, yanit)
                        cands = yanit.get("candidates", [])
                        if cands and "content" in cands[0]:
                            parts = cands[0]["content"].get("parts", [])
                            if parts and "text" in parts[0]:
                                sorular = self._json_sorulari_ayristir(parts[0]["text"].strip())
                                if sorular and len(sorular) >= 1:
                                    if zaman_asimindan_sonra and asama == "soru_uretimi":
                                        self._soru_bicimini_duzelt(sorular)
                                        if not self._gemini_soru_seti_gecerli(sorular):
                                            break
                                    return sorular
                        if yalniz_zaman_asiminda_tekrarla:
                            break
                    elif res.status_code == 503:
                        hata_mesaji = "Sunucu aşırı yoğunluk bildiriyor (503 Service Unavailable)"
                        try:
                            err_data = res.json().get("error", {})
                            if "message" in err_data:
                                hata_mesaji = err_data["message"]
                        except Exception:
                            pass
                        print(f" -> [UYARI] Google Gemini API ({current_model}) 503 hatası: {hata_mesaji} (Deneme {attempt + 1}/{retries + 1})")
                        if attempt < retries and not yalniz_zaman_asiminda_tekrarla:
                            backoff = 1.5 * (2 ** attempt)
                            time.sleep(backoff)
                            continue
                        break
                    elif res.status_code == 429:
                        kota = gemini_kota_hatasini_isle(current_model, res)
                        gemini_kullanimini_kaydet(current_model, asama, durum="429", kota=kota)
                        print(f" -> [UYARI] Google Gemini API ({current_model}) kullanım sınırı (429): {kota}")
                        break
                    else:
                        print(f" -> [UYARI] Google Gemini API ({current_model}) HTTP Hatası: {res.status_code}")
                        break
                except requests.exceptions.Timeout:
                    gemini_kullanimini_kaydet(current_model, asama, durum="zaman_asimi")
                    print(f" -> [UYARI] Google Gemini API ({current_model}) zaman aşımına uğradı ({timeout_sec}s, Deneme {attempt + 1}/{retries + 1}).")
                    if attempt < retries:
                        zaman_asimindan_sonra = yalniz_zaman_asiminda_tekrarla
                        time.sleep(1.0)
                        continue
                except Exception as e:
                    print(f" -> [UYARI] Google Gemini API soru üretim hatası ({current_model}): {e}")
                    break

        if zaman_asimindan_sonra and asama == "soru_uretimi":
            raise GeminiFasikulHatasi("Gemini soru üretimi zaman aşımına uğradı ve tek yeniden deneme başarısız oldu.")
        return None

    def _ollama_ile_soru_uret(
        self, istem: str, timeout: Optional[int] = None, *, num_ctx: Optional[int] = None,
        yanit_semasi: Optional[Dict[str, Any]] = None,
    ) -> Optional[List[Dict[str, Any]]]:
        if self._ollama_devre_disi:
            return None

        tout = timeout or self.ollama_question_timeout
        try:
            print(f"[QuestionGenerator - 1. Tercih] Yerel Ollama ({self.ollama_model}) ile soru üretiliyor (Zaman aşımı: {tout}s)...")
            payload = {
                "model": self.ollama_model,
                "prompt": istem,
                "stream": False,
                "format": yanit_semasi or "json",
                "think": False,
                "options": {
                    "num_thread": 16,
                    "num_predict": self.ollama_question_num_predict,
                    "temperature": 0.2
                }
            }
            if num_ctx is not None:
                payload["options"]["num_ctx"] = num_ctx
            res = requests.post(f"{self.ollama_url.rstrip('/')}/api/generate", json=payload, timeout=tout)
            if res.status_code == 200:
                data = res.json()
                done_reason = data.get("done_reason")
                if done_reason == "length":
                    print(f" -> [UYARI] Ollama soru üretimi token sınırına ulaştı (done_reason=length, num_predict={self.ollama_question_num_predict}).")
                metin = data.get("response", "").strip()
                if not metin:
                    print(" -> [UYARI] Ollama soru üretimi boş yanıt döndürdü (response boş).")
                    return None
                sorular = self._json_sorulari_ayristir(metin)
                if not sorular:
                    print(" -> [UYARI] Ollama soru çıktısı JSON formatında ayrıştırılamadı.")
                    return None
                if len(sorular) < 1:
                    print(" -> [UYARI] Ollama hiç soru üretemedi.")
                    return None
                print(f" -> [BAŞARILI] Yerel Ollama ({self.ollama_model}) {len(sorular)} özgün soru üretti.")
                return sorular
            else:
                print(f" -> [UYARI] Ollama soru üretimi HTTP Hatası: {res.status_code}")
                if res.status_code in (404, 500, 502, 503):
                    self._ollama_devre_disi = True
                return None
        except requests.exceptions.Timeout:
            print(f" -> [UYARI] Ollama soru üretimi zaman aşımına uğradı ({tout}s).")
            self._ollama_devre_disi = True
            print(" -> [BİLGİ] Zincirleme zaman aşımını önlemek için bu oturumda sonraki Ollama çağrıları atlanacak.")
        except requests.exceptions.ConnectionError as e:
            print(f" -> [UYARI] Ollama servisine bağlanılamadı ({self.ollama_url}): {e}")
            self._ollama_devre_disi = True
            print(" -> [BİLGİ] Ollama servisi çevrimdışı, sonraki çağrılar doğrudan yedek servise aktarılacak.")
        except Exception as e:
            print(f" -> [UYARI] Ollama soru üretimi beklenmedik hata: {e}")
        return None

    def _gemini_ile_soru_uret(self, istem: str, *, kaynak_dogrulama: bool = False) -> Optional[List[Dict[str, Any]]]:
        print(f"[QuestionGenerator - 2. Tercih (Fallback)] Google Gemini ({self.model}) ile soru üretiliyor...")
        payload = {
            "contents": [{"parts": [{"text": istem}]}],
            "generationConfig": {"temperature": 0.2, "responseMimeType": "application/json"}
        }
        if kaynak_dogrulama:
            payload["generationConfig"]["maxOutputTokens"] = 2048
            if self.model == "gemini-3-flash-preview":
                payload["generationConfig"]["thinkingConfig"] = {"thinkingLevel": "LOW"}
            return self._gemini_api_cagir(
                payload, timeout=300, max_retries=1, yalniz_zaman_asiminda_tekrarla=True,
                asama="soru_kaynak_dogrulama",
            )
        sorular = self._gemini_api_cagir(
            payload, timeout=300, max_retries=1, yalniz_zaman_asiminda_tekrarla=True,
        )
        if sorular and len(sorular) >= 1:
            print(f" -> [BAŞARILI] Google Gemini ({self.model}) {len(sorular)} özgün soru üretti.")
            return sorular
        return None

    def _rag_kaynaklarini_bagla(
        self,
        sorular: List[Dict[str, Any]],
        video_analizi: Dict[str, Any],
        rag_soru_ornekleri: Optional[List[Dict[str, Any]]] = None,
        varsayilan_kaynak_tipi: str = "rag_fallback",
    ) -> List[Dict[str, Any]]:
        """
        Her soru için ölçülen kavramı ve soru metnini analiz ederek, yalnızca gerçekten
        eşleşen doğrulanmış RAG kaynağını bağlar.

        KATI KALİTE KURALI:
        1. Eşleşme bulunamadığında ASLA rastgele bir bölümün veya ilk genel referansın referansını atamaz.
        2. Doğrulanamayan sorular için 'Doğrulanmış ders kitabı kaynağı bulunamadı' yazar ve
           q['kaynak_durumu'] = 'kaynak_dogrulanamadi' işaretler.
        3. Doğrulanan sorular için eşleşen chunk_id, dosya_adi, sayfa ve referans bilgisini korur.
        """
        import re

        bolumler = video_analizi.get("bolumler", []) if isinstance(video_analizi, dict) else []
        tum_kaynaklar = video_analizi.get("tum_kaynaklar", []) if isinstance(video_analizi, dict) else []

        # Havuzdaki tüm doğrulanmış aday parçaları derle
        aday_parcalar = []
        gorulen_ids = set()

        for b in bolumler:
            rag_list = b.get("rag_kaynaklari", [])
            if rag_list:
                for r in rag_list:
                    cid = r.get("chunk_id") or r.get("referans")
                    if cid and cid not in gorulen_ids:
                        gorulen_ids.add(cid)
                        aday_parcalar.append(r)
            else:
                b_ref = b.get("kitap_referansi")
                if b_ref and b_ref not in [
                    "MEB Ders Kitabı Referansı",
                    "Doğrulanmış RAG eşleşmesi bulunamadı",
                    "Doğrulanmış ders kitabı kaynağı bulunamadı",
                ]:
                    if b_ref not in gorulen_ids:
                        gorulen_ids.add(b_ref)
                        b_kavramlar = [k.get("kavram", "") for k in b.get("onemli_kavramlar", []) if isinstance(k, dict)]
                        aday_parcalar.append({
                            "chunk_id": b.get("chunk_id") or f"bolum_{b.get('bolum_no')}_ref",
                            "referans": b_ref,
                            "unite": b.get("bolum_basligi", ""),
                            "konu": " ".join(b_kavramlar),
                            "alt_konu": b.get("temel_konu", ""),
                            "icerik": b.get("transkript", "") or " ".join(b_kavramlar),
                            "bolum_no": b.get("bolum_no"),
                        })

        for src in tum_kaynaklar:
            cid = src.get("chunk_id") or src.get("referans")
            if cid and cid not in gorulen_ids:
                gorulen_ids.add(cid)
                aday_parcalar.append(src)

        def _kelime_seti(m: str) -> set:
            if not m:
                return set()
            m_clean = m.replace("İ", "i").replace("I", "ı").lower()
            return set(re.findall(r"\b[a-zçğıöşü0-9]+\b", m_clean))

        for q in sorular:
            if not q.get("kaynak_tipi"):
                q["kaynak_tipi"] = varsayilan_kaynak_tipi

            kavram = (q.get("ilgili_kavram") or "").lower()
            kategori = (q.get("kategori") or "").lower()
            soru_metni = (q.get("soru") or "").lower()
            aciklama = (q.get("aciklama") or "").lower()
            q_full = f"{kavram} {kategori} {soru_metni} {aciklama}"

            en_iyi_aday = None
            en_iyi_skor = 0.0

            for aday in aday_parcalar:
                ref = aday.get("referans", "")
                if not ref or any(x in ref.lower() for x in ["eşleşmesi bulunamadı", "kaynağı bulunamadı"]):
                    continue

                unite = (aday.get("unite") or "").lower()
                konu = (aday.get("konu") or "").lower()
                alt_konu = (aday.get("alt_konu") or "").lower()
                icerik = (aday.get("icerik") or "").lower()
                aday_text = f"{ref.lower()} {unite} {konu} {alt_konu} {icerik}".strip()

                skor = 0.0

                # 1. 3-gram Trigram Benzerliği (Kavram ile aday konu/içerik arasında)
                if lexical_similarity and kavram:
                    kavram_sim = lexical_similarity(kavram, f"{konu} {alt_konu} {unite}")
                    skor += kavram_sim * 6.0

                # 2. Ölçülen Temel Kavram ile Kaynak Alanı Eşleşmesi
                if any(w in kavram for w in ["atık pil", "geri dönüşüm"]):
                    if any(w in aday_text for w in ["atık pil", "geri dönüşüm", "güvenli", "çevre"]) or aday.get("printed_page") == 57:
                        skor += 6.0
                elif any(w in kavram for w in ["şehir elektriği", "santral", "priz"]):
                    if any(w in aday_text for w in ["şehir elektriği", "santral", "priz", "baraj", "kablo"]):
                        skor += 6.0
                elif any(w in kavram for w in ["akü", "akümülatör"]):
                    if any(w in aday_text for w in ["akü", "taşıt", "araç"]):
                        skor += 6.0
                elif any(w in kavram for w in ["batarya", "şarj"]):
                    if any(w in aday_text for w in ["batarya", "şarj"]):
                        skor += 6.0
                elif "pil" in kavram and "atık" not in kavram:
                    if "pil" in aday_text and not any(w in aday_text for w in ["besin", "kuvvet"]):
                        skor += 6.0
                elif any(w in kavram for w in ["elektrik", "enerji", "araç", "gereç"]):
                    if any(w in aday_text for w in ["elektrik", "enerji", "araç", "gereç"]):
                        skor += 4.5

                # 3. Soru metni anahtar kelime eşleşmesi
                if any(w in soru_metni for w in ["atık pil", "geri dönüşüm kutusu"]) and (aday.get("printed_page") == 57 or "güvenli" in aday_text):
                    skor += 3.0
                elif any(w in soru_metni for w in ["şehir elektriği", "santral", "priz"]) and ("şehir" in aday_text or "santral" in aday_text or aday.get("printed_page") in [44, 48]):
                    skor += 3.0
                elif any(w in soru_metni for w in ["akü", "ulaşım", "otomobil"]) and ("akü" in aday_text or "taşıt" in aday_text):
                    skor += 3.0

                if skor > en_iyi_skor:
                    en_iyi_skor = skor
                    en_iyi_aday = aday

            # Doğrulanmış kaynak eşleşmesi için en az 3.0 güvenilirlik eşiği aranır
            if en_iyi_aday and en_iyi_skor >= 3.0:
                q["kazanim_referansi"] = en_iyi_aday.get("referans")
                q["rag_dogrulandi"] = True
                q["kaynak_durumu"] = "dogrulandi"
                q["rag_kaynagi"] = en_iyi_aday
                q["eslesen_chunk_id"] = en_iyi_aday.get("chunk_id")
            else:
                # KATI KURAL: Rastgele ilk bölümü atama! Açıkça doğrulanamadı olarak işaretle
                q["kazanim_referansi"] = "Doğrulanmış ders kitabı kaynağı bulunamadı"
                q["rag_dogrulandi"] = False
                q["kaynak_durumu"] = "kaynak_dogrulanamadi"
                q["rag_kaynagi"] = None
                q["eslesen_chunk_id"] = None

        return sorular

    @staticmethod
    def _soru_bicimini_duzelt(sorular: List[Dict[str, Any]]) -> None:
        """Modelin alan adlarını değil, eşdeğer seçenek gösterimlerini düzenler."""
        for soru in sorular:
            tip = soru.get("tip")
            if not isinstance(tip, str):
                continue
            soru["tip"] = tip = re.sub(r"\s*[-–—]\s*", " - ", tip.strip())
            if tip == "Çoktan Seçmeli":
                secenekler = soru.get("secenekler")
                if isinstance(secenekler, list) and len(secenekler) == 4 and all(isinstance(m, str) for m in secenekler):
                    etiketler = [re.match(r"^([A-D])[.)]\s*", m.strip()) for m in secenekler]
                    if not any(etiketler) or all(e and e[1] == h for e, h in zip(etiketler, "ABCD")):
                        soru["secenekler"] = [f"{h}) " + re.sub(r"^[A-D][.)]\s*", "", m.strip())
                                             for h, m in zip("ABCD", secenekler)]
                cevap = soru.get("dogru_cevap")
                if isinstance(cevap, str):
                    harf = re.match(r"^\s*([A-D])(?:\s*$|[.)])", cevap.upper())
                    if harf:
                        soru["dogru_cevap"] = harf[1]
            elif tip == "Boşluk Doldurma" and isinstance(soru.get("soru"), str):
                soru["soru"] = re.sub(r"\.{4,}|…{2,}", "____", soru["soru"])

    @staticmethod
    def _soru_bicimi_gecerli(soru: Dict[str, Any]) -> bool:
        if not isinstance(soru, dict) or not all(
            isinstance(soru.get(alan), str) and soru[alan].strip()
            for alan in ("tip", "kategori", "ilgili_kavram", "soru", "dogru_cevap", "aciklama")
        ):
            return False
        tip = soru["tip"]
        secenekler = soru.get("secenekler")
        cevap = soru["dogru_cevap"].strip()
        if tip == "Çoktan Seçmeli":
            return (
                isinstance(secenekler, list) and len(secenekler) == 4 and cevap in ("A", "B", "C", "D")
                and all(isinstance(s, str) and s.startswith(f"{harf})") and s[2:].strip()
                        for harf, s in zip("ABCD", secenekler))
                and len({" ".join(s[2:].casefold().split()) for s in secenekler}) == 4
            )
        if tip == "Doğru - Yanlış":
            return secenekler == ["Doğru", "Yanlış"] and cevap in secenekler
        if tip == "Boşluk Doldurma":
            return secenekler == [] and len(re.findall(r"_{3,}", soru["soru"])) == 1
        return False

    @classmethod
    def _gemini_soru_seti_gecerli(cls, sorular: List[Dict[str, Any]]) -> bool:
        if not sorular or len(sorular) != 10 or not all(cls._soru_bicimi_gecerli(q) for q in sorular):
            return False
        kokler = {" ".join(q["soru"].casefold().split()) for q in sorular}
        return len(kokler) == 10 and Counter(q["tip"] for q in sorular) == {
            "Çoktan Seçmeli": 5, "Doğru - Yanlış": 3, "Boşluk Doldurma": 2,
        }

    def ollama_fasikul_sorularini_uret(
        self, video_analizi: Dict[str, Any], egitim_seviyesi: str,
        rag_soru_ornekleri: List[Dict[str, Any]], rag: "RAGMotoru",
    ) -> List[Dict[str, Any]]:
        """Yerel fasikül sorularını MEB bağlamıyla üretir ve ikinci geçişte denetler."""
        sinif = video_analizi.get("sinif_duzeyi")
        ders = video_analizi.get("ders_adi")
        konu = video_analizi.get("unite_basligi") or ""
        bolumler = [
            {"baslik": b.get("bolum_basligi"), "anlatim": b.get("duzenlenmis_anlatim") or b.get("transkript"),
             "ogretmen_konusmasi": b.get("transkript"), "kavramlar": b.get("onemli_kavramlar", [])}
            for b in video_analizi.get("bolumler", [])
        ]
        alistirmalar = []
        kitap_metinleri = {}
        if sinif and ders:
            if konu:
                alistirmalar = list(rag.konu_alistirmalarini_getir(str(konu), egitim_seviyesi, sinif, ders))
            alistirma_metinleri = {d.page_content for d in alistirmalar}
            for b in bolumler:
                sorgu = " ".join(str(b[alan] or "") for alan in ("baslik", "anlatim")).strip()
                if not sorgu:
                    continue
                for belge in rag.seviye_filtreli_arama(
                    sorgu, egitim_seviyesi, sinif=sinif, ders_adi=ders, k=8, max_mesafe=None,
                ):
                    if belge.page_content.strip() and belge.page_content not in alistirma_metinleri:
                        kitap_metinleri[belge.page_content] = None
        veriler = {
            "sinif": sinif, "egitim_seviyesi": egitim_seviyesi, "ders": ders, "bolumler": bolumler,
            "bicim_ornekleri": [o["ornek_metin"] for o in rag_soru_ornekleri if o.get("ornek_metin")],
            "meb_alistirmalari": [
                {"ornek_no": i, "metin": d.page_content} for i, d in enumerate(alistirmalar, 1)
            ],
            "meb_konu_metinleri": list(kitap_metinleri),
        }
        sablon = Path(__file__).resolve().parents[1] / "templates" / "prompts" / "ollama-soru-uretimi.txt"
        istem = sablon.read_text(encoding="utf-8-sig") + "\n\nDERS VERİLERİ\n" + json.dumps(veriler, ensure_ascii=False)
        sorular = self._ollama_ile_soru_uret(
            istem, num_ctx=32768, yanit_semasi=self._ollama_sorular_semalari(10),
        ) or []
        self._ollama_soru_bicimini_duzelt(sorular)
        sorular = self._ollama_sorularini_denetle(sorular, bolumler, sinif)
        return self._ollama_fasikul_sorularini_tamamla(
            sorular, rag, egitim_seviyesi, sinif, ders, alistirmalar,
        )

    @staticmethod
    def _ollama_sorular_semalari(
        sayi: int, *, denetim: bool = False, soru_numaralari=(), tipler=(),
    ) -> Dict[str, Any]:
        if denetim:
            alanlar = {
                "soru_no": {"type": "integer"},
                "uygun": {"type": "boolean"},
                "gerekce": {"type": "string"},
                "bulunan_cevap": {"type": "string"},
                "alinti": {"type": "string"},
            }
        else:
            alanlar = {
                ad: {"type": "string"}
                for ad in ("kategori", "ilgili_kavram", "soru", "dogru_cevap", "aciklama")
            }
            alanlar.update({
                "tip": {"enum": ["Çoktan Seçmeli", "Doğru - Yanlış", "Boşluk Doldurma"]},
                "secenekler": {"type": "array", "items": {"type": "string"}, "maxItems": 4},
                "ornek_no": {"type": "integer"},
                "soru_no": {"type": "integer"},
            })
            if tipler:
                alanlar["tip"]["enum"] = list(dict.fromkeys(tipler))
        if soru_numaralari:
            alanlar["soru_no"]["enum"] = list(soru_numaralari)
        return {
            "type": "object", "required": ["sorular"],
            "properties": {"sorular": {
                "type": "array", "minItems": sayi, "maxItems": sayi,
                "items": {"type": "object", "properties": alanlar, "required": list(alanlar)},
            }},
        }

    def _ollama_sorularini_denetle(self, sorular, bolumler, sinif) -> List[Dict[str, Any]]:
        konusma = [
            b["ogretmen_konusmasi"] for b in bolumler if b.get("ogretmen_konusmasi")
        ]
        if not konusma:
            return []
        normal_konusmalar = [" ".join(metin.split()) for metin in konusma]
        baglam = json.dumps({"sinif": sinif, "ogretmen_konusmalari": konusma}, ensure_ascii=False)
        uygunlar = {}
        kokler = set()
        tip_sayilari = Counter()
        tip_sinirlari = {"Çoktan Seçmeli": 5, "Doğru - Yanlış": 3, "Boşluk Doldurma": 2}
        for baslangic in range(0, len(sorular), 5):
            denetlenecek = {
                i + 1: sorular[i] for i in range(baslangic, min(baslangic + 5, len(sorular)))
                if self._soru_bicimi_gecerli(sorular[i])
            }
            for deneme in range(3):
                if not denetlenecek:
                    break
                veri = [
                    {"soru_no": no, **{alan: q[alan] for alan in ("tip", "soru", "secenekler")}}
                    for no, q in denetlenecek.items()
                ]
                denetim = self._ollama_ile_soru_uret(
                    "Her soruyu öğretmen konuşmasına göre bağımsız çöz. bulunan_cevap alanına çoktan seçmeli için harf, "
                    "doğru-yanlış için Doğru veya Yanlış, boşluk doldurma için boşluğa gelecek ifadeyi yaz. "
                    "alinti alanına cevabın tamamını destekleyen kesintisiz metni konuşmadan noktalamasıyla aynen kopyala. "
                    "Anlamca eşdeğer ifade destek sayılır. Alıntıda kelimeleri değiştirme. Sorunun doğru cevabı "
                    "kökünden tahmin edilebiliyorsa, birden çok cevap savunulabiliyorsa, gerekli görsel eksikse "
                    "veya alıntı cevabı desteklemiyorsa uygun=false yaz. Gerekçeyi bir kısa cümlede ver.\n"
                    + baglam + "\n" + json.dumps(veri, ensure_ascii=False),
                    num_ctx=16384, yanit_semasi=self._ollama_sorular_semalari(
                        len(veri), denetim=True, soru_numaralari=denetlenecek,
                    ),
                ) or []
                karar_sayilari = Counter(q.get("soru_no") for q in denetim if type(q.get("soru_no")) is int)
                kararlar = {q["soru_no"]: q for q in denetim if type(q.get("soru_no")) is int
                            and karar_sayilari[q["soru_no"]] == 1}
                sorunlar = {}
                for no, soru in denetlenecek.items():
                    karar = kararlar.get(no, {})
                    sorun = self._ollama_soru_denetim_sorunu(soru, karar, normal_konusmalar)
                    kok = " ".join(kelimelere_ayir(soru["soru"]))
                    tip = soru["tip"]
                    if kok in kokler:
                        sorun = "Bu soru kökü daha önce kullanıldı."
                    elif tip_sayilari[tip] >= tip_sinirlari[tip]:
                        continue
                    if sorun:
                        sorunlar[no] = sorun
                    else:
                        uygunlar[no] = soru
                        kokler.add(kok)
                        tip_sayilari[tip] += 1
                hatalilar = {no: denetlenecek[no] for no in sorunlar}
                if not hatalilar:
                    break
                print(f"[Ollama soru denetimi] Yeniden yazılması gereken sorular: {list(hatalilar)}")
                if deneme == 2:
                    break
                onarim = self._ollama_ile_soru_uret(
                    "Yalnızca aşağıdaki sorunlu soruları öğretmen konuşmasına göre yeniden yaz. "
                    "Soru numarasını (soru_no) ve türünü koru. Tek anlamlı soru, doğru cevap ve sade gerekçe ver. "
                    "Önceki soru veya kavram hatalı ya da belirsizse aynı düşünceyi başka kelimelerle tekrarlama. "
                    "Öğretmenin açıkça açıkladığı farklı ve somut bir noktadan yeni soru yaz. "
                    "Çoktan seçmeli şıkları A)–D) biçiminde yaz, doğru cevap harf olsun. "
                    "Doğru - Yanlış seçenekleri Doğru ve Yanlış olsun. Boşluk doldurmada ____ ve boş seçenek listesi kullan. "
                    "Doğru - Yanlış maddesini bir yargı cümlesi olarak yaz. Soru kökü cevabı ele vermesin ve "
                    "verilmeyen bir görsele veya tabloya gönderme yapmasın. "
                    "Kaynak ekleme, ornek_no=0 yaz.\n" + baglam + "\n" + json.dumps([
                        {"soru_no": no, "onceki_soru": soru, "sorun": sorunlar[no]}
                        for no, soru in hatalilar.items()
                    ], ensure_ascii=False),
                    num_ctx=16384, yanit_semasi=self._ollama_sorular_semalari(
                        len(hatalilar), soru_numaralari=hatalilar,
                        tipler=[q["tip"] for q in hatalilar.values()],
                    ),
                ) or []
                self._ollama_soru_bicimini_duzelt(onarim)
                yeniler = {q["soru_no"]: q for q in onarim if type(q.get("soru_no")) is int
                           and q["soru_no"] in hatalilar and self._soru_bicimi_gecerli(q)
                           and q["tip"] == hatalilar[q["soru_no"]]["tip"]}
                denetlenecek = yeniler
        return [uygunlar[no] for no in sorted(uygunlar)]

    @staticmethod
    def _ollama_soru_koku_sorunu(soru) -> str:
        metin = soru["soru"].strip()
        tip = soru["tip"]
        kok = kelimelere_ayir(metin)
        cevap = soru["dogru_cevap"].strip()
        if tip == "Doğru - Yanlış" and (
            "?" in metin or (kok and kok[-1] in {"mı", "mi", "mu", "mü"})
        ):
            return "Doğru-yanlış maddesi bir yargı olmalı."
        if tip == "Çoktan Seçmeli" and re.search(
            r"\b(?:yer almaz|değildir|söylenemez|kullanılmaz|çalışmaz|yapılmamalıdır)\b",
            metin, re.IGNORECASE,
        ):
            return "Çoktan seçmeli soru kökünü olumlu bir cümleyle kur."
        if re.search(r"\b(?:görselde\w*|resimde\w*|şekildeki|tablodaki|yukarıdaki|gösterilen)\b", metin, re.IGNORECASE):
            return "Soru, yanında bulunmayan bir görsele veya başka bir içeriğe gönderme yapıyor."
        if tip == "Çoktan Seçmeli" and cevap in {"A", "B", "C", "D"}:
            secenekler = [kelimelere_ayir(s[2:]) for s in soru["secenekler"]]
            i = "ABCD".index(cevap)
            dogru = " ".join(secenekler[i])
            if dogru and f" {dogru} " in f" {' '.join(kok)} ":
                return "Doğru seçenek soru kökünde aynen geçiyor."
            if "ortak" in kok:
                diger = {w[:7] for j, secenek in enumerate(secenekler) if j != i for w in secenek if len(w) >= 7}
                kokler = {w[:7] for w in kok if len(w) >= 7}
                ayirt_edici = {w[:7] for w in secenekler[i] if len(w) >= 7} - diger
                if ayirt_edici & kokler:
                    return "Ortak özellik sorusunun kökü doğru seçenekteki ayırt edici kelimeyi ele veriyor."
        return ""

    @classmethod
    def _ollama_soru_denetim_sorunu(cls, soru, karar, konusmalar) -> str:
        if sorun := cls._ollama_soru_koku_sorunu(soru):
            return sorun
        cevap = soru["dogru_cevap"].strip()
        if karar.get("uygun") is not True:
            return karar.get("gerekce") or "9B soruyu uygun bulmadı."
        bulunan = karar.get("bulunan_cevap", "")
        if kelimelere_ayir(bulunan) != kelimelere_ayir(cevap):
            return "Bağımsız çözümde bulunan cevap üretimdeki cevapla uyuşmuyor."
        alinti = " ".join(karar.get("alinti", "").split())
        if len(kelimelere_ayir(alinti)) < 3 or not any(alinti in metin for metin in konusmalar):
            return "Cevabı destekleyen birebir konuşma alıntısı bulunamadı."
        return ""

    @classmethod
    def _ollama_soru_bicimini_duzelt(cls, sorular: List[Dict[str, Any]]) -> None:
        turler = {
            "coktan secmeli": "Çoktan Seçmeli",
            "dogru yanlis": "Doğru - Yanlış",
            "bosluk doldurma": "Boşluk Doldurma",
            "bosluk dolurma": "Boşluk Doldurma",
        }
        harfler = str.maketrans("çğıöşü", "cgiosu")
        for soru in sorular:
            tip = soru.get("tip")
            if isinstance(tip, str):
                anahtar = " ".join(kelimelere_ayir(tip)).translate(harfler)
                soru["tip"] = turler.get(anahtar, tip)
        cls._soru_bicimini_duzelt(sorular)

    def _ollama_fasikul_sorularini_tamamla(
        self, sorular: List[Dict[str, Any]], rag: "RAGMotoru", egitim_seviyesi: str,
        sinif: Optional[str], ders: Optional[str], alistirmalar: list,
    ) -> List[Dict[str, Any]]:
        if not sorular:
            return []

        cevap_konumlari = [0, 1, 2, 3, choice(range(4))]
        shuffle(cevap_konumlari)
        for no, soru in enumerate(sorular, 1):
            soru.update(no=no, kaynak_tipi="ollama", dogru_cevap=soru["dogru_cevap"].strip())
            if soru["tip"] == "Çoktan Seçmeli":
                secenekler = [metin[2:].strip() for metin in soru["secenekler"]]
                dogru = "ABCD".index(soru["dogru_cevap"])
                hedef = cevap_konumlari.pop()
                secenekler[dogru], secenekler[hedef] = secenekler[hedef], secenekler[dogru]
                soru["secenekler"] = [f"{harf}) {metin}" for harf, metin in zip("ABCD", secenekler)]
                soru["dogru_cevap"] = "ABCD"[hedef]
        self._gemini_soru_kaynaklarini_dogrula(
            sorular, rag, self._ollama_ile_soru_uret, egitim_seviyesi, sinif, ders, alistirmalar=alistirmalar,
            grup_boyutu=5, kisa_adaylar=True, ollama_baglamli_dogrulama=True,
        )
        return sorular

    def gemini_sorularini_uret(
        self,
        video_analizi: Dict[str, Any],
        egitim_seviyesi: str,
        rag_soru_ornekleri: List[Dict[str, Any]],
        rag: "RAGMotoru",
    ) -> List[Dict[str, Any]]:
        """Dersin soru setini Gemini ile, gerektiğinde Ollama ile oluşturur."""
        sinif = video_analizi.get("sinif_duzeyi")
        ders = video_analizi.get("ders_adi")
        konu = str(video_analizi.get("unite_basligi") or "")
        alistirmalar = list(rag.konu_alistirmalarini_getir(konu, egitim_seviyesi, sinif, ders)) if konu and sinif and ders else []
        veriler = {
            "sinif": sinif, "egitim_seviyesi": egitim_seviyesi, "ders": ders,
            "bolumler": [
                {"baslik": b.get("bolum_basligi"), "anlatim": b.get("duzenlenmis_anlatim") or b.get("transkript"),
                 "kavramlar": b.get("onemli_kavramlar", [])}
                for b in video_analizi.get("bolumler", [])
            ],
            "bicim_ornekleri": [o["ornek_metin"][:750] for o in rag_soru_ornekleri if o.get("ornek_metin")],
            "meb_alistirmalari": [
                {"ornek_no": i, "metin": d.page_content[:1350]} for i, d in enumerate(alistirmalar, 1)
            ],
        }
        sablon = Path(__file__).resolve().parents[1] / "templates" / "prompts" / "gemini-soru-uretimi.txt"
        istem = sablon.read_text(encoding="utf-8-sig").format(veriler=json.dumps(veriler, ensure_ascii=False))
        for saglayici, uret in (("gemini", self._gemini_ile_soru_uret), ("ollama", self._ollama_ile_soru_uret)):
            if saglayici == "gemini" and not self._api_anahtari_var_mi():
                continue
            sorular = uret(istem)
            self._soru_bicimini_duzelt(sorular or [])
            if saglayici == "gemini" and sorular and not self._gemini_soru_seti_gecerli(sorular):
                sorular = uret(istem + "\nÖnceki set biçim denetiminden geçmedi. Tam 5 çoktan seçmeli, 3 Doğru - Yanlış ve "
                               "2 Boşluk Doldurma sorusu yaz. Seçenekleri A) biçiminde yaz, doğru yanıt yalnız harf olsun. "
                               "Boşluğu ____ olarak, boşluk doldurma seçeneklerini [] olarak yaz. Her soru kökü farklı olsun.")
                self._soru_bicimini_duzelt(sorular or [])
            if self._gemini_soru_seti_gecerli(sorular):
                for no, soru in enumerate(sorular, 1):
                    soru.update(no=no, kaynak_tipi=saglayici, dogru_cevap=soru["dogru_cevap"].strip())
                kaynak_uret = partial(uret, kaynak_dogrulama=True) if saglayici == "gemini" else uret
                self._gemini_soru_kaynaklarini_dogrula(
                    sorular, rag, kaynak_uret, egitim_seviyesi, sinif, ders, alistirmalar=alistirmalar,
                    grup_boyutu=10 if saglayici == "gemini" else 2,
                    kisa_adaylar=saglayici == "gemini",
                )
                return sorular
            print(f"[Soru üretimi] {saglayici} geçerli bir 10 soruluk set döndürmedi.")
        return []

    @staticmethod
    def _ollama_alintida_bilgi_var_mi(alinti: str) -> bool:
        """Soru kökü ve etkinlik yönergesinden oluşan alıntıları eler."""
        for cumle in re.findall(r"[^.!?]+[.!?]", alinti):
            kelimeler = kelimelere_ayir(cumle)
            if len(kelimeler) < 4 or cumle.endswith("?"):
                continue
            if re.search(r"(?:alım|elim|ınız|iniz|unuz|ünüz|inizdir)$", kelimeler[-1]) or (
                kelimeler[-1] in {"yazın", "okuyun", "inceleyin", "seçin", "işaretleyin", "cevaplayın"}
            ):
                continue
            return True
        return False

    @staticmethod
    def _gemini_soru_kaynaklarini_dogrula(
        sorular: List[Dict[str, Any]],
        rag: "RAGMotoru",
        uret: Callable[[str], Optional[List[Dict[str, Any]]]],
        egitim_seviyesi: str,
        sinif: Optional[str],
        ders: Optional[str],
        *,
        alistirmalar: list,
        grup_boyutu: int,
        kisa_adaylar: bool = False,
        ollama_baglamli_dogrulama: bool = False,
    ) -> None:
        """Geniş aday aramasından yalnızca cevabı destekleyen birebir alıntıyı bağlar."""
        adaylar = {}
        veri = []
        for soru in sorular:
            soru.update(kazanim_referansi="Doğrulanmış ders kitabı kaynağı bulunamadı", rag_dogrulandi=False,
                        kaynak_durumu="kaynak_dogrulanamadi", rag_kaynagi=None, eslesen_chunk_id=None)
            if not sinif or not ders:
                continue
            cevap = soru["dogru_cevap"]
            if soru["tip"] == "Çoktan Seçmeli":
                cevap = soru["secenekler"]["ABCD".index(cevap)][2:].strip()
            sorgu = f'{soru["ilgili_kavram"]}. {soru["soru"]} {cevap}. {soru["aciklama"]}'
            docs = rag.seviye_filtreli_arama(sorgu, egitim_seviyesi, sinif=sinif, ders_adi=ders, k=8, max_mesafe=None)
            parcalar = [d for d in docs if d.metadata.get("sinif") == sinif and d.metadata.get("ders_adi") == ders
                        and d.metadata.get("kitap_adi") and d.metadata.get("has_printed_page")
                        and d.metadata.get("printed_page") is not None and d.page_content.strip()]
            secilen_ornek = soru.get("ornek_no")
            ilgili_alistirmalar = sorted(alistirmalar, key=lambda d: lexical_similarity(sorgu, d.page_content), reverse=True)[:3] if lexical_similarity else []
            if type(secilen_ornek) is int and 1 <= secilen_ornek <= len(alistirmalar):
                ilgili_alistirmalar.insert(0, alistirmalar[secilen_ornek - 1])
            alistirma_metinleri = {d.page_content for d in ilgili_alistirmalar}
            tekil = {d.page_content: d for d in [*ilgili_alistirmalar, *parcalar]}
            soru_adaylari = [
                (d, d.page_content in alistirma_metinleri, d.page_content[:1350])
                for d in tekil.values()
            ]
            if kisa_adaylar:
                sirali = []
                for doc, alistirma, metin in soru_adaylari:
                    eslesmeler = list(re.finditer(r"[^.!?]+[.!?]", metin))
                    puanli = [
                        (lexical_similarity(sorgu, eslesme[0]) if lexical_similarity else 0, i)
                        for i, eslesme in enumerate(eslesmeler)
                        if len(eslesme[0].strip()) <= 430 and len(eslesme[0].split()) >= 4
                    ]
                    if not puanli:
                        continue
                    puan, i = max(puanli, key=lambda aday: aday[0])
                    pasaj = eslesmeler[i][0].strip()
                    if ollama_baglamli_dogrulama:
                        komsulu = metin[eslesmeler[max(i - 1, 0)].start():
                                         eslesmeler[min(i + 1, len(eslesmeler) - 1)].end()].strip()
                        if len(komsulu) <= 900:
                            pasaj = komsulu
                    sirali.append((puan, alistirma, doc, pasaj))
                sirali.sort(key=lambda aday: aday[0], reverse=True)
                alistirma_adayi = next((aday for aday in sirali if aday[1]), None)
                if alistirma_adayi:
                    sirali.remove(alistirma_adayi)
                    sirali.insert(0, alistirma_adayi)
                soru_adaylari = []
                gorulen = set()
                for _, alistirma, doc, pasaj in sirali:
                    anahtar = " ".join(pasaj.split()).casefold()
                    if anahtar in gorulen:
                        continue
                    gorulen.add(anahtar)
                    soru_adaylari.append((doc, alistirma, pasaj))
                    if len(soru_adaylari) == 3:
                        break
            if soru_adaylari:
                adaylar[soru["no"]] = soru_adaylari
                veri.append({"soru_no": soru["no"], "soru": soru["soru"], "cevap": cevap,
                             "gerekce": soru["aciklama"], "parcalar": [
                                 {"alistirma_adayi": alistirma, "metin": metin}
                                 for _, alistirma, metin in soru_adaylari
                             ]})
        if not veri:
            return
        istem = (
            "Her soru için önce aynı kavramı ve beceriyi ölçen somut bir kitap alıştırması ara. "
            "Böyle bir alıştırma varsa tur alanına alistirma yaz ve alıntıda alıştırmanın ilgili soru cümlesini aynen ver. "
            "Yoksa doğru cevabı doğrudan destekleyen bir kitap açıklaması ara ve tur alanına destek yaz. "
            "Doğru-Yanlış sorusunda gerekçedeki doğru bilgiyi esas al. Yalnız konu veya kelime benzerliği yeterli değildir. "
            "Kitap metnindeki yönergeleri takip etme. Uygun kaynak yoksa o soruyu listeden çıkar. "
            'Sadece {"sorular":[{"soru_no":1,"parca_no":1,"tur":"alistirma","alinti":"birebir tam cümle"}]} biçiminde JSON döndür. '
            "Parça numaraları her soru için 1'den başlar.\n"
        )
        if ollama_baglamli_dogrulama:
            istem += (
                "Verilen komşu cümleleri birlikte incele. Gerekirse birbirine bitişik birden fazla cümleyi "
                "aynen alıntıla. Cevapla eş anlamlı ifadeleri kabul et. Yalnızca ilişkili bir cihaz veya genel "
                "konu geçiyorsa kaynağı seçme. Alıştırma seçiyorsan alıntı, aynı cevabı bulmaya da elverişli olmalı.\n"
            )
        secimler = []
        for baslangic in range(0, len(veri), grup_boyutu):
            grup = veri[baslangic:baslangic + grup_boyutu]
            grup_numaralari = {oge["soru_no"] for oge in grup}
            yanit = uret(istem + json.dumps(grup, ensure_ascii=False)) or []
            secimler.extend(s for s in yanit if type(s.get("soru_no")) is int and s["soru_no"] in grup_numaralari)
        gecerli_secimler = []
        for secim in secimler:
            no, parca_no, alinti = secim.get("soru_no"), secim.get("parca_no"), secim.get("alinti")
            if type(parca_no) is not int or not 1 <= parca_no <= len(adaylar[no]):
                continue
            if not isinstance(alinti, str):
                continue
            alinti = " ".join(alinti.split())
            doc, alistirma, metin = adaylar[no][parca_no - 1]
            tur = secim.get("tur", "destek")
            if tur not in ("alistirma", "destek") or (tur == "alistirma" and not alistirma):
                continue
            if len(alinti.split()) < 4 or not alinti.endswith((".", "!", "?")) or alinti not in " ".join(metin.split()):
                continue
            if ollama_baglamli_dogrulama and not QuestionGenerator._ollama_alintida_bilgi_var_mi(alinti):
                continue
            gecerli_secimler.append((no, doc, tur, alinti))

        if ollama_baglamli_dogrulama and gecerli_secimler:
            soru_verileri = {oge["soru_no"]: oge for oge in veri}
            kontrol = [
                {"secim_no": i, "soru": soru_verileri[no]["soru"], "cevap": soru_verileri[no]["cevap"],
                 "gerekce": soru_verileri[no]["gerekce"], "alinti": alinti}
                for i, (no, _, _, alinti) in enumerate(gecerli_secimler, 1)
            ]
            onaylar = set()
            for baslangic in range(0, len(kontrol), grup_boyutu):
                grup = kontrol[baslangic:baslangic + grup_boyutu]
                sonuc = uret(
                    "Her alıntı, doğru cevabı veya doğru-yanlış gerekçesindeki doğru önermeyi tek başına destekliyor mu? "
                    "Eş anlam ve doğrudan çıkarım kabul edilir. Aynı konudan söz etmek yeterli değildir. "
                    "Alıntıdaki bir bilginin doğru olması, soruya verilen cevabı kanıtlamıyorsa reddet. "
                    'Yalnızca {"sorular":[{"secim_no":1,"destekliyor":true}]} biçiminde JSON döndür.\n'
                    + json.dumps(grup, ensure_ascii=False)
                ) or []
                numaralar = {oge["secim_no"] for oge in grup}
                onaylar.update(oge["secim_no"] for oge in sonuc
                               if type(oge.get("secim_no")) is int and oge["secim_no"] in numaralar
                               and oge.get("destekliyor") is True)
            gecerli_secimler = [secim for i, secim in enumerate(gecerli_secimler, 1) if i in onaylar]

        for no, doc, tur, alinti in gecerli_secimler:
            soru = sorular[no - 1]
            if soru["rag_dogrulandi"]:
                continue
            kaynak = {**doc.metadata, "icerik": alinti}
            ref = f"{kaynak['kitap_adi']}, basılı sayfa {kaynak['printed_page']}"
            if tur == "alistirma":
                ref = f"İlgili alıştırma: {ref}"
            kaynak["referans"] = ref
            soru.update(kazanim_referansi=ref, rag_dogrulandi=True, kaynak_durumu="dogrulandi",
                        rag_kaynagi=kaynak, eslesen_chunk_id=kaynak.get("chunk_id"))

    def soru_uret(
        self,
        video_analizi: Dict[str, Any],
        egitim_seviyesi: str = "İlkokul",
        rag_soru_ornekleri: Optional[List[Dict[str, Any]]] = None,
        soru_sayisi: int = 10,
        yalniz_gemini: bool = False,
    ) -> List[Dict[str, Any]]:
        """
        Kullanıcının belirlediği eğitim seviyesine ve öğrenilen kavramlara tam uyumlu
        özgün değerlendirme soruları üretir.
        Önce yerel Ollama modelini, ulaşılamazsa Gemini API'yi dener.
        Tüm soruların kaynaklarını doğrulanmış gerçek RAG verilerine bağlar.
        """
        print(f"\n[QuestionGenerator] '{egitim_seviyesi}' Kademesi İçin {soru_sayisi} Özgün Değerlendirme Sorusu Hazırlanıyor...")

        # İstem hazırla
        istem = self._soru_istemi_olustur(video_analizi, egitim_seviyesi, soru_sayisi)

        # 1. Aşama: Yerel Ollama (qwen3.5:4b)
        ai_sorulari = None if yalniz_gemini else self._ollama_ile_soru_uret(istem)
        if ai_sorulari:
            for q in ai_sorulari:
                q["kaynak_tipi"] = "ollama"
            return self._rag_kaynaklarini_bagla(ai_sorulari, video_analizi, rag_soru_ornekleri, varsayilan_kaynak_tipi="ollama")

        # 2. Aşama: Google Gemini API (Ollama yanıt vermezse)
        if self._api_anahtari_var_mi():
            print("[BİLGİ] Yerel model soru üretmedi, Google Gemini API devreye alınıyor...")
            ai_sorulari = self._gemini_ile_soru_uret(istem)
            if ai_sorulari:
                for q in ai_sorulari:
                    q["kaynak_tipi"] = "gemini"
                return self._rag_kaynaklarini_bagla(ai_sorulari, video_analizi, rag_soru_ornekleri, varsayilan_kaynak_tipi="gemini")

        if yalniz_gemini:
            return []

        # 3. Aşama: Güvenlik Ağı - Pedagojik Şablon Bankası
        print("[BİLGİ] Yapay zeka servislerine ulaşılamadı. Müfredat uyumlu standart soru bankası devreye alınıyor...")

        # -------------------------------------------------------------------
        # 1. İLKOKUL (3. Sınıf Fen Bilimleri)
        # -------------------------------------------------------------------
        fallback_sorular: List[Dict[str, Any]] = []
        if egitim_seviyesi == "İlkokul":
            unite = str(video_analizi.get("unite_basligi", "")).lower()
            bolum_basliklari = " ".join([b.get("bolum_basligi", "") for b in video_analizi.get("bolumler", [])]).lower()
            is_elektrik_kaynaklari = any(k in unite or k in bolum_basliklari for k in ["kaynak", "şehir elektriği", "sehir elektrigi", "pil", "batarya", "akü", "aku"])
            is_arac_gerecler = "kullanım amaç" in unite or "aydınlatma" in bolum_basliklari or "ısınma" in bolum_basliklari or "soğutma" in bolum_basliklari

            if is_elektrik_kaynaklari or not is_arac_gerecler:
                fallback_sorular = [
                    {
                        "no": 1,
                        "kategori": "💡 Kavram Keşfi",
                        "ilgili_kavram": "Elektrik Enerjisi ve Dönüşümü",
                        "tip": "Doğru - Yanlış",
                        "soru": "Elektrik enerjisi; evimizde ve okulumuzda kullandığımız aletler sayesinde ışık (ampul), ısı (fırın, ütü) ve hareket (vantilatör) enerjisine dönüştürülebilen temel bir enerji türüdür.",
                        "secenekler": ["Doğru", "Yanlış"],
                        "dogru_cevap": "Doğru",
                        "aciklama": "Elektrik tek başına bir alet değil enerjidir. Aletlerin içine girerek ihtiyacımıza göre ışığa, ısıya veya harekete dönüşür.",
                        "kazanim_referansi": ""
                    },
                    {
                        "no": 2,
                        "kategori": "🔍 Günlük Yaşam Analizi",
                        "ilgili_kavram": "Enerji İhtiyacı ve Taşınabilirlik",
                        "tip": "Çoktan Seçmeli",
                        "soru": "Dışarıda bahçede veya parkta oynarken yanımızda taşıdığımız oyuncak arabanın veya el fenerinin aniden durmasının en temel bilimsel sebebi aşağıdakilerden hangisidir?",
                        "secenekler": [
                            "A) Havanın kararması ve güneş ışığının kaybolması",
                            "B) İçine takılan pildeki kimyasal enerjinin tükenmiş olması ve gerilimin düşmesi",
                            "C) Şehirdeki büyük elektrik santralinin kapanması",
                            "D) Aletin duvardaki prize takılmak istemesi"
                        ],
                        "dogru_cevap": "B",
                        "aciklama": "Taşınabilir aletler pildeki sınırlı kimyasal enerjiyle çalışır; pildeki enerji bittiğinde gerilim düşer ve alet durur.",
                        "kazanim_referansi": ""
                    },
                    {
                        "no": 3,
                        "kategori": "🧩 Kavram Tamamlama",
                        "ilgili_kavram": "Şehir Elektriği ve Santraller",
                        "tip": "Boşluk Doldurma",
                        "soru": "Cümledeki noktalı yeri doğru kavramla tamamlayınız:\nBarajlardaki suyun gücü, rüzgâr türbinleri veya güneş panelleri kullanılarak santrallerde yüksek güçte üretilen ve kablolarla evlerimize ulaştırılan enerji kaynağına ........................................ denir.",
                        "secenekler": [],
                        "dogru_cevap": "Şehir Elektriği",
                        "aciklama": "Santrallerde üretilen elektrik, iletim hatları ve trafolar üzerinden evlerimizin prizlerine kadar kesintisiz ulaştırılır.",
                        "kazanim_referansi": ""
                    },
                    {
                        "no": 4,
                        "kategori": "⚡ Elektrik Dedektifi",
                        "ilgili_kavram": "Priz Bağlantısı ve Yüksek Güçlü Aletler",
                        "tip": "Çoktan Seçmeli",
                        "soru": "Aşağıdaki ev aletlerinden hangisi çok yüksek miktarda enerji tükettiği için küçük kumanda pilleriyle ÇALIŞTIRILAMAZ ve mutlaka duvardaki prize takılmalıdır?",
                        "secenekler": [
                            "A) Kol saati",
                            "B) Duvar saati",
                            "C) Çamaşır makinesi",
                            "D) Televizyon uzaktan kumandası"
                        ],
                        "dogru_cevap": "C",
                        "aciklama": "Çamaşır makinesi, buzdolabı ve fırın gibi büyük aletler çok yüksek elektrik enerjisine ihtiyaç duyar; bu enerji yalnızca prizden (şehir elektriği) sağlanabilir.",
                        "kazanim_referansi": ""
                    },
                    {
                        "no": 5,
                        "kategori": "🎯 Doğru mu, Yanlış mı?",
                        "ilgili_kavram": "Pil ve Düşük Güç Tüketimi",
                        "tip": "Doğru - Yanlış",
                        "soru": "Pil; az miktarda elektrik enerjisi harcayan, hafif ve kablosuz olarak kolayca taşınabilen küçük aletleri (duvar saati, radyo, el feneri) çalıştırmak için kullanılan güvenli bir elektrik kaynağıdır. Piller genellikle tek kullanımlık (primer) olsa da şarj edilebilir (sekonder) türleri de mevcuttur.",
                        "secenekler": ["Doğru", "Yanlış"],
                        "dogru_cevap": "Doğru",
                        "aciklama": "Piller düşük enerji depolar ve kabloya ihtiyaç duymadan cihazları yanımızda taşımamızı sağlar.",
                        "kazanim_referansi": ""
                    },
                    {
                        "no": 6,
                        "kategori": "🌱 Doğa ve Çevre Bilinci",
                        "ilgili_kavram": "Atık Pil ve Geri Dönüşüm",
                        "tip": "Çoktan Seçmeli",
                        "soru": "Tükenen pilleri normal ev çöpüne, sokağa veya toprağa atmak yerine okullardaki ve marketlerdeki 'Atık Pil Kutusu'na atmamızın en önemli nedeni nedir?",
                        "secenekler": [
                            "A) Pillerin kutu içinde kendi kendine renk değiştirmesi",
                            "B) İçlerindeki zararlı kimyasal maddelerin toprağa ve içme suyuna karışıp canlılara zarar vermesini önlemek",
                            "C) Çöp poşetinin daha hafif kalmasını sağlamak",
                            "D) Pillerin kutunun içinde sihirli şekilde yeniden dolması"
                        ],
                        "dogru_cevap": "B",
                        "aciklama": "Pillerin içinde ağır metaller ve kimyasallar bulunur. Çöpe atılırlarsa toprağı ve suyu kirletir; bu yüzden mutlaka atık pil kutusunda toplanmalıdır.",
                        "kazanim_referansi": ""
                    },
                    {
                        "no": 7,
                        "kategori": "🧩 Kavram Tamamlama",
                        "ilgili_kavram": "Batarya ve Şarj Edilebilirlik",
                        "tip": "Boşluk Doldurma",
                        "soru": "Cümledeki noktalı yeri doğru kavramla tamamlayınız:\nBirden fazla pilin bir araya gelmesiyle oluşan; dizüstü bilgisayar, tablet ve akıllı telefonlarda kullanılan, bittiğinde çöpe atılmayıp adaptörle şarj edilebilen elektrik kaynağına ........................................ denir.",
                        "secenekler": [],
                        "dogru_cevap": "Batarya",
                        "aciklama": "Bataryalar şarj edilebilir özelliktedir; prizden elektrik enerjisi depolayarak aletleri kablosuz çalıştırmaya devam eder.",
                        "kazanim_referansi": ""
                    },
                    {
                        "no": 8,
                        "kategori": "🔍 Günlük Yaşam Analizi",
                        "ilgili_kavram": "Akü (Akümülatör) ve Taşıtlar",
                        "tip": "Çoktan Seçmeli",
                        "soru": "Otomobil, otobüs, kamyon, uçak ve gemi gibi motorlu ulaşım araçlarında motoru ilk çalıştıran (marş veren) ve farları aydınlatan güçlü elektrik kaynağı hangisidir?",
                        "secenekler": [
                            "A) Kumanda pili",
                            "B) Güneş pili",
                            "C) Akü (Akümülatör)",
                            "D) Rüzgâr türbini"
                        ],
                        "dogru_cevap": "C",
                        "aciklama": "Motorlu taşıtların motorunu ilk çalıştırma anında harekete geçiren ve elektrik donanımını besleyen büyük ve güçlü kaynak aküdür. Araç çalışırken alternatör vasıtasıyla şarj edilir.",
                        "kazanim_referansi": ""
                    },
                    {
                        "no": 9,
                        "kategori": "⚡ Eşleştirme ve Sınıflandırma",
                        "ilgili_kavram": "Elektrik Kaynaklarını Ayırt Etme",
                        "tip": "Çoktan Seçmeli",
                        "soru": "Aşağıdaki araç ve kullandığı temel elektrik kaynağı eşleştirmelerinden hangisi YANLIŞTIR?",
                        "secenekler": [
                            "A) Buzdolabı  ➔  Şehir Elektriği",
                            "B) Akıllı Telefon  ➔  Batarya",
                            "C) Şehirler Arası Otobüs  ➔  Kumanda Pili",
                            "D) Duvar Saati  ➔  Pil"
                        ],
                        "dogru_cevap": "C",
                        "aciklama": "Otobüs gibi büyük motorlu taşıtlar küçük kumanda pilleriyle değil, yüksek güce sahip dayanıklı akülerle çalışır.",
                        "kazanim_referansi": ""
                    },
                    {
                        "no": 10,
                        "kategori": "🛡️ Güvenlik Dedektifi",
                        "ilgili_kavram": "Elektrik Güvenliği ve Bilinçli Kullanım",
                        "tip": "Doğru - Yanlış",
                        "soru": "Evimizdeki şehir elektriği prizlerinde bulunan elektrik enerjisi, pillerdeki enerjiden çok daha yüksek ve tehlikelidir; bu yüzden prizlere kesinlikle yabancı cisim sokulmamalı ve ıslak ellerle elektrikli aletlere dokunulmamalıdır.",
                        "secenekler": ["Doğru", "Yanlış"],
                        "dogru_cevap": "Doğru",
                        "aciklama": "Şehir elektriği elektrik çarpmasına ve hayati tehlikeye neden olabilir. Güvenlik kurallarına uymak can güvenliğimiz için zorunludur.",
                        "kazanim_referansi": ""
                    }
                ]
            else:
                fallback_sorular = [
                    {
                        "no": 1,
                        "kategori": "💡 Kavram Keşfi",
                        "ilgili_kavram": "Elektrikli Araç ve Gereçler",
                        "tip": "Doğru - Yanlış",
                        "soru": "Elektrikli araç ve gereçler; elektrik enerjisini kullanarak günlük işlerimizi daha kolay, hızlı ve yorulmadan yapmamızı sağlayan teknolojik aletlerdir.",
                        "secenekler": ["Doğru", "Yanlış"],
                        "dogru_cevap": "Doğru",
                        "aciklama": "Elektrikli araçlar insan gücünden ve zamandan tasarruf sağlar; günlük hayatı büyük ölçüde kolaylaştırır.",
                        "kazanim_referansi": ""
                    },
                    {
                        "no": 2,
                        "kategori": "🔍 Günlük Yaşam Analizi",
                        "ilgili_kavram": "Elektrik Olmasaydı Yaşam",
                        "tip": "Çoktan Seçmeli",
                        "soru": "Bir akşam mahallenizde elektrikler kesildiğinde aşağıdaki durumlardan hangisi MEYDANA GELMEZ?",
                        "secenekler": [
                            "A) Buzdolabındaki yiyeceklerin zamanla ısınarak bozulma tehlikesi ortaya çıkar.",
                            "B) Odadaki tavan lambası ve avizeler yanmaz.",
                            "C) Duvardaki prizden elektrik alan çamaşır makinesi durur.",
                            "D) Mekanik yayla kurularak çalışan mekanik çalar saat anında durur."
                        ],
                        "dogru_cevap": "D",
                        "aciklama": "Mekanik kurmalı saatler elektrik enerjisine ihtiyaç duymadan yay gücüyle çalıştığı için elektrik kesintisinden etkilenmez.",
                        "kazanim_referansi": ""
                    },
                    {
                        "no": 3,
                        "kategori": "💡 Aydınlatma Amaçlı Araçlar",
                        "ilgili_kavram": "Aydınlatma Araçları",
                        "tip": "Çoktan Seçmeli",
                        "soru": "Güneş battıktan sonra karanlıkta ders çalışabilmemiz veya gece odamızı aydınlatabilmemiz için elektrik enerjisini ışığa dönüştüren araç aşağıdakilerden hangisidir?",
                        "secenekler": [
                            "A) Elektrikli süpürge",
                            "B) Masa lambası ve ampul",
                            "C) Buzdolabı",
                            "D) Tost makinesi"
                        ],
                        "dogru_cevap": "B",
                        "aciklama": "Ampul ve masa lambası elektrik enerjisini ışık enerjisine dönüştüren temel aydınlatma araçlarıdır.",
                        "kazanim_referansi": ""
                    },
                    {
                        "no": 4,
                        "kategori": "⚡ Isınma ve Pişirme Dedektifi",
                        "ilgili_kavram": "Isınma Amaçlı Araçlar",
                        "tip": "Çoktan Seçmeli",
                        "soru": "Aşağıdaki elektrikli araçlardan hangisi kullanım amacı bakımından diğerlerinden FARKLI bir grupta yer alır?",
                        "secenekler": [
                            "A) Elektrikli soba (Isıtıcı)",
                            "B) Ekmek kızartma makinesi",
                            "C) Klima ve derin dondurucu",
                            "D) Su ısıtıcısı (Kettle)"
                        ],
                        "dogru_cevap": "C",
                        "aciklama": "Elektrikli soba, ekmek kızartma makinesi ve kettle ısıtma/ısınma amaçlıdır; klima ve derin dondurucu ise soğutma amaçlıdır.",
                        "kazanim_referansi": ""
                    },
                    {
                        "no": 5,
                        "kategori": "❄️ Soğutma Dedektifi",
                        "ilgili_kavram": "Soğutma ve Besin Saklama",
                        "tip": "Doğru - Yanlış",
                        "soru": "Buzdolabı, içine konulan yiyecek ve içeceklerin ısısını düşürerek onların bozulmasını önleyen ve besinleri taze tutan soğutma amaçlı bir elektrikli araçtır.",
                        "secenekler": ["Doğru", "Yanlış"],
                        "dogru_cevap": "Doğru",
                        "aciklama": "Buzdolabı ortamı soğutarak bakterilerin üremesini yavaşlatır ve gıdaların taze kalmasını sağlar.",
                        "kazanim_referansi": ""
                    },
                    {
                        "no": 6,
                        "kategori": "🧹 Temizlik ve Hijyen",
                        "ilgili_kavram": "Temizlik Amaçlı Araçlar",
                        "tip": "Çoktan Seçmeli",
                        "soru": "Evimizde temizlik ve hijyen sağlamak amacıyla kullanılan; yerdeki toz ve kırıntıları çekerek emen elektrikli araç hangisidir?",
                        "secenekler": [
                            "A) Elektrikli süpürge",
                            "B) Bulaşık makinesi",
                            "C) Çamaşır makinesi",
                            "D) Saç kurutma makinesi"
                        ],
                        "dogru_cevap": "A",
                        "aciklama": "Elektrikli süpürge zeminlerdeki toz ve kirleri vakumlayarak temizleyen temel temizlik aracıdır.",
                        "kazanim_referansi": ""
                    },
                    {
                        "no": 7,
                        "kategori": "📱 Haberleşme ve İletişim",
                        "ilgili_kavram": "Haberleşme Araçları",
                        "tip": "Boşluk Doldurma",
                        "soru": "Cümledeki noktalı yeri doğru kavramla tamamlayınız:\nUzaktaki insanlarla sesli, görüntülü konuşmamızı, dünyadaki gelişmeleri takip etmemizi ve bilgiye hızla ulaşmamızı sağlayan televizyon, bilgisayar ve cep telefonu gibi araçlara ........................................ amaçlı elektrikli araçlar denir.",
                        "secenekler": [],
                        "dogru_cevap": "Haberleşme (İletişim)",
                        "aciklama": "Televizyon, radyo, telefon ve tablet gibi aletler iletişim ve bilgi aktarımı sağlar.",
                        "kazanim_referansi": ""
                    },
                    {
                        "no": 8,
                        "kategori": "🎯 Eşleştirme ve Sınıflandırma",
                        "ilgili_kavram": "Kullanım Amaçlarına Göre Gruplandırma",
                        "tip": "Çoktan Seçmeli",
                        "soru": "Aşağıdaki araç ve kullanım amacı eşleştirmelerinden hangisi YANLIŞTIR?",
                        "secenekler": [
                            "A) Vantilatör  ➔  Soğutma / Serinletme",
                            "B) Ütü  ➔  Isınma / Düzleştirme",
                            "C) Çamaşır Makinesi  ➔  Haberleşme",
                            "D) El Feneri  ➔  Aydınlatma"
                        ],
                        "dogru_cevap": "C",
                        "aciklama": "Çamaşır makinesi haberleşme değil, temizlik amaçlı bir araçtır.",
                        "kazanim_referansi": ""
                    },
                    {
                        "no": 9,
                        "kategori": "🌱 Bilinçli Tüketici ve Tasarruf",
                        "ilgili_kavram": "Elektrik Tasarrufu",
                        "tip": "Çoktan Seçmeli",
                        "soru": "Elektrikli araçları kullanırken enerji tasarrufu yapmak ve aile bütçemizi korumak için aşağıdakilerden hangisini YAPMALIYIZ?",
                        "secenekler": [
                            "A) Odadan çıkarken açık kalan lambayı kapatmalıyız.",
                            "B) Buzdolabının kapağını uzun süre açık bırakmalıyız.",
                            "C) İzlemediğimiz televizyonu sürekli açık tutmalıyız.",
                            "D) Gündüz güneş ışığı varken evin bütün lambalarını yakmalıyız."
                        ],
                        "dogru_cevap": "A",
                        "aciklama": "Gereksiz yanan lambaları söndürmek hem enerji tasarrufu sağlar hem de doğal kaynakları korur.",
                        "kazanim_referansi": ""
                    },
                    {
                        "no": 10,
                        "kategori": "🛡️ Güvenlik Dedektifi",
                        "ilgili_kavram": "Elektrikli Araçları Güvenli Kullanma",
                        "tip": "Doğru - Yanlış",
                        "soru": "Elektrikli araçları ıslak elle veya banyoda suyun yanında kullanmak elektrik çarpmasına yol açabileceğinden son derece tehlikelidir.",
                        "secenekler": ["Doğru", "Yanlış"],
                        "dogru_cevap": "Doğru",
                        "aciklama": "Su elektriği iletir; ıslak ellerle elektrikli aletlere dokunmak can güvenliğini tehlikeye sokar.",
                        "kazanim_referansi": ""
                    }
                ]

        # -------------------------------------------------------------------
        # 2. ORTAOKUL (5-8. Sınıf Seviyesi)
        # -------------------------------------------------------------------
        elif egitim_seviyesi == "Ortaokul":
            fallback_sorular = [
                {
                    "no": i,
                    "kategori": "🔍 Ortaokul Fen Analizi",
                    "ilgili_kavram": f"Elektrik Kavramı {i}",
                    "tip": "Çoktan Seçmeli",
                    "soru": f"Ortaokul Fen Bilimleri elektrik ünitesi analiz sorusu {i}: Elektrik enerjisinin iletimi ve kaynakların verimliliği...",
                    "secenekler": ["A) Seçenek 1", "B) Seçenek 2", "C) Seçenek 3", "D) Seçenek 4"],
                    "dogru_cevap": "B",
                    "aciklama": "Devre elemanları ve potansiyel fark prensiplerine göre doğru yanıt B seçeneğidir.",
                    "kazanim_referansi": ""
                }
                for i in range(1, soru_sayisi + 1)
            ]

        # -------------------------------------------------------------------
        # 3. LİSE (9-12. Sınıf Seviyesi)
        # -------------------------------------------------------------------
        else:
            fallback_sorular = [
                {
                    "no": i,
                    "kategori": "⚡ Lise Fizik & Elektrik Analizi",
                    "ilgili_kavram": f"Elektrik Devreleri & Enerji {i}",
                    "tip": "Analiz ve Problem Çözme",
                    "soru": f"Lise fizik düzeyi elektrik potansiyeli ve elektrostatik alan analizi {i}: Joule yasası ve ohmik iletkenler...",
                    "secenekler": ["A) Seçenek A", "B) Seçenek B", "C) Seçenek C", "D) Seçenek D"],
                    "dogru_cevap": "A",
                    "aciklama": "Elektrik potansiyeli ve direnç katsayılarına göre akım-gerilim ilişkisi incelenmiştir.",
                    "kazanim_referansi": ""
                }
                for i in range(1, soru_sayisi + 1)
            ]

        varsayilan_tur = "rag_fallback" if (video_analizi.get("bolumler") or rag_soru_ornekleri) else "static_fallback"
        return self._rag_kaynaklarini_bagla(
            fallback_sorular[:soru_sayisi],
            video_analizi=video_analizi,
            rag_soru_ornekleri=rag_soru_ornekleri,
            varsayilan_kaynak_tipi=varsayilan_tur
        )


if __name__ == "__main__":
    qg = QuestionGenerator()
    qs = qg.soru_uret({}, egitim_seviyesi="İlkokul", soru_sayisi=10)
    print(f"Toplam {len(qs)} soru üretildi.")
    for q in qs:
        print(f"[{q['no']}] {q['kategori']} ({q['tip']}) -> {q['ilgili_kavram']}")
        print(f"     Soru: {q['soru'][:90]}...")
        print(f"     Doğru: {q['dogru_cevap']} | Ref: {q['kazanim_referansi']}\n")
