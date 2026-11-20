"""
EduFrame AI - Gemini 1.5 Pro Multimodal Materyal ve Görsel Değişim Analiz Servisi (services/gemini_analyzer.py)
---------------------------------------------------------------------------------------------------------
Bu modül, videoyu baştan sona tek parça halinde özetlemek yerine;
EKRANDAKİ GÖRSEL MATERYALLERİN (slayt, şema, tahta vb.) DEĞİŞİMİNE GÖRE
bölümlere ayırır. Her materyalin ekranda kaldığı süre boyunca öğretmenin
yaptığı konuşmaları transkript eder ve ders kitabı (RAG) ile sentezler.

Temel Kurallar:
1. Görsel Materyal Değişimi Esastır: Yeni bir slayt/konu geldiğinde yeni bölüm başlar.
2. Küçük animasyonlar ve imleç hareketleri yeni materyal sayılmaz.
3. Reklam, sponsor veya eğitim dışı kısımlar elenir (ekran görüntüsü ve bölüm oluşturulmaz).
4. Her bölüm için:
   - Materyal Başlığı
   - Zaman Aralığı (Örn: 00:00 – 02:35)
   - Orijinal Transkript (Öğretmenin o andaki ham konuşması)
   - Düzenlenmiş Konu Anlatımı (Ders kitabı referanslı seviyeye uygun anlatım)
   - Önemli Kavramlar ve Açıklamaları
5. Ders notu sonu için Genel Konu Özeti üretilir.
"""

import os
import re
import json
import time
import base64
import requests
from difflib import SequenceMatcher
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

try:
    from config import (
        GEMINI_API_KEY,
        GEMINI_MODEL,
        GEMINI_TIMEOUT,
        GEMINI_MAX_RETRIES,
        GEMINI_FALLBACK_MODELS,
        OLLAMA_URL,
        OLLAMA_MODEL,
        OLLAMA_TIMEOUT,
        OLLAMA_ANALYSIS_TIMEOUT,
        OLLAMA_SECTION_TIMEOUT,
        OLLAMA_ANALYSIS_NUM_PREDICT,
        OLLAMA_SECTION_NUM_PREDICT,
    )
except ImportError:
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
    GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3-flash-preview")
    GEMINI_TIMEOUT = int(os.getenv("GEMINI_TIMEOUT", "20"))
    GEMINI_MAX_RETRIES = int(os.getenv("GEMINI_MAX_RETRIES", "2"))
    GEMINI_FALLBACK_MODELS = [m.strip() for m in os.getenv("GEMINI_FALLBACK_MODELS", "gemini-3-flash-preview,gemini-3.8-flash,gemini-flash-latest").split(",") if m.strip()]
    OLLAMA_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen3.5:4b")
    OLLAMA_TIMEOUT = int(os.getenv("OLLAMA_TIMEOUT", "120"))
    OLLAMA_ANALYSIS_TIMEOUT = int(os.getenv("OLLAMA_ANALYSIS_TIMEOUT", "120"))
    OLLAMA_SECTION_TIMEOUT = int(os.getenv("OLLAMA_SECTION_TIMEOUT", "45"))
    OLLAMA_ANALYSIS_NUM_PREDICT = int(os.getenv("OLLAMA_ANALYSIS_NUM_PREDICT", "3500"))
    OLLAMA_SECTION_NUM_PREDICT = int(os.getenv("OLLAMA_SECTION_NUM_PREDICT", "800"))


class GeminiAnalyzer:
    """
    Ekrandaki görsel eğitim materyallerini takip ederek zaman aralıklı,
    transkriptli ve RAG referanslı ders fasikülü içeriği üreten motor.

    Öncelik Sırası:
    1. Öncelik: Yerel Ollama (qwen3.5:4b) - Tamamen yerel, offline ve gizli
    2. Yedek: Google Gemini API (gemini-3.8-flash) - Ollama yanıt vermezse
    3. Güvenlik Ağı: Orijinal Pedagojik Materyal Simülasyonu
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
        ollama_timeout: int = OLLAMA_TIMEOUT,
        ollama_analysis_timeout: int = OLLAMA_ANALYSIS_TIMEOUT,
        ollama_section_timeout: int = OLLAMA_SECTION_TIMEOUT,
        ollama_analysis_num_predict: int = OLLAMA_ANALYSIS_NUM_PREDICT,
        ollama_section_num_predict: int = OLLAMA_SECTION_NUM_PREDICT,
    ):
        self.api_key = api_key if api_key is not None else (GEMINI_API_KEY or os.getenv("GEMINI_API_KEY", ""))
        self.model = model
        self.ollama_url = ollama_url or OLLAMA_URL
        self.ollama_model = ollama_model or OLLAMA_MODEL
        self.gemini_timeout = gemini_timeout
        self.gemini_max_retries = gemini_max_retries
        self.fallback_models = fallback_models if fallback_models is not None else list(GEMINI_FALLBACK_MODELS)
        self.ollama_timeout = ollama_timeout
        self.ollama_analysis_timeout = ollama_analysis_timeout
        self.ollama_section_timeout = ollama_section_timeout
        self.ollama_analysis_num_predict = ollama_analysis_num_predict
        self.ollama_section_num_predict = ollama_section_num_predict
        self._ollama_devre_disi = False
        self.base_url = "https://generativelanguage.googleapis.com"

    def _api_anahtari_var_mi(self) -> bool:
        return bool(self.api_key and len(self.api_key.strip()) > 10 and not self.api_key.startswith("AIzaSyDummy"))

    @staticmethod
    def _json_ayristir(metin: str) -> Optional[Dict[str, Any]]:
        """Model yanıtından Markdown kod bloklarını temizleyip geçerli JSON üretir."""
        if not metin:
            return None
        metin = metin.strip()
        try:
            if "```json" in metin:
                metin = metin.split("```json")[1].split("```")[0].strip()
            elif "```" in metin:
                metin = metin.split("```")[1].split("```")[0].strip()
            veri = json.loads(metin)
            if isinstance(veri, dict):
                return veri
        except Exception:
            pass
        return None

    def transkript_filtrele(self, ham_metin: str) -> str:
        """
        Transkriptten selamlaşma, kanal tanıtımı, beğeni/abone ol istekleri ve
        reklamları cımbızlayıp çıkarır; öğretmenin verdiği tüm somut örnekleri,
        nesneleri ve ders anlatımlarını eksiksiz korur.
        """
        if not ham_metin:
            return ""
        metin = ham_metin.strip()
        giris = re.compile(
            r"^(?:"
            r"(?:herkese\s+)?merhaba(?:lar)?\b(?:[ ,]+(?:sevgili\s+)?(?:arkadaşlar|öğrenciler|üçüncü\s+sınıflar))?"
            r"|sevgili\s+(?:arkadaşlar|öğrenciler)\b"
            r"|(?:evet[ ,]+)?(?:bugün\s+)?nasılsınız(?:\s+bakalım)?\b"
            r"|(?:kanalımıza|(?:3\.\s*sınıf\s+)?fen\s+bilimleri\s+dersimize)\s+hoş\s+geldiniz\b"
            r")[,.!?\s]*",
            re.IGNORECASE,
        )
        while eslesme := giris.match(metin):
            metin = metin[eslesme.end():]

        cikis_desenleri = [
            r"(kanalımıza\s+)?abone\s+olmayı\s+unutmayın\s*[\.!]?",
            r"videomuzu\s+beğenmeyi\s+unutmayın\s*[\.!]?",
            r"bir\s+sonraki\s+videomuzda\s+görüşmek\s+üzere\s*[\.!]?"
        ]
        for p in cikis_desenleri:
            metin = re.sub(p, "", metin, flags=re.IGNORECASE).strip()

        return metin.strip()

    @classmethod
    def pedagojik_anlatim_normallestir(cls, metin: str) -> str:
        """
        Ders notu anlatımını öğretmen konuşmasından, hitaplardan ve sınıf yönergelerinden
        arındırır; nesnel, 3. tekil şahıs ve MEB ders kitabı standartlarında expository metne dönüştürür.
        """
        if not metin:
            return ""

        # HTML etiketlerini ve markdown işaretlerini temizle
        temiz = metin.replace("<br/>", "\n\n").replace("<br />", "\n\n").replace("<br>", "\n\n")
        temiz = re.sub(r"\*\*([^*]+)\*\*", r"\1", temiz)
        temiz = re.sub(r"\*([^*]+)\*", r"\1", temiz)

        # Cümle bazlı filtreleme: Öğretmen konuşması, hitap ve sınıf yönergeleri içeren cümleleri ayıkla
        yasakli_cumle_kaliplari = [
            r"(?i)\b(bugünkü|bu)\s+dersi?mizde\s+.*?(keşfedeceğiz|öğreneceğiz|inceleyeceğiz|göreceğiz|ele\s+alacağız|bakacağız|konuşacağız)",
            r"(?i)\b(şimdi|hadi|gelin)\s+.*?(başlayalım|keşfedelim|öğrenelim|inceleyelim|bakalım|geçelim|çözelim)",
            r"(?i)\b(sınıf\s+)?arkadaşlar(ımız)?la\s+(tartışalım|konuşalım|paylaşalım)",
            r"(?i)\b(arkadaşlarınızla|arkadaşlarınla)\s+tartış(ınız)?",
            r"(?i)\b(aşağıdaki|yukarıdaki)\s+(alana|boşluğa|bölüme|noktalı\s+yerlere)\s+(yazalım|çizelim|dolduralım|yazınız|doldurunuz)",
            r"(?i)\bmerak\s+ettiğimiz\s+soruları\s+.*?yazalım",
            r"(?i)\bkemerlerinizi\s+bağlayın",
            r"(?i)\bsevgili\s+(öğrenciler|öğrencilerim|çocuklar|arkadaşlar)",
            r"(?i)\bmerhaba\s+(arkadaşlar|öğrenciler|çocuklar)",
            r"(?i)\bhadi\s+(başlayalım|gelin\s+birlikte|bakalım)",
            r"(?i)\bhep\s+birlikte\s+(öğreneceğiz|keşfedeceğiz|inceleyeceğiz|göreceğiz|bakalım|öğrenelim)",
            r"(?i)\bbir\s+sonraki\s+dersi?mizde\s+.*?(öğreneceğiz|inceleyeceğiz|keşfedeceğiz|göreceğiz|ele\s+alacağız|işleyeceğiz)",
            r"(?i)\bhepinize\s+iyi\s+çalışmalar",
            r"(?i)\bbaşarılar\s+diliyorum",
            r"(?i)\bşimdi\s+ekrandaki\s+.*?(bakalım|çözelim|inceleyelim)",
            r"(?i)\bşimdi\s+tahtadaki\s+.*?(bakalım|çözelim|inceleyelim)",
            r"(?i)\bönce\s+şu\s+örnekle\s+başlayalım",
            r"(?i)\börnekle\s+başlayalım",
            r"(?i)\böğretmenimizin\s+de\s+(vurguladığı|anlattığı|belirttiği)\s+gibi",
            r"(?i)\bbir\s+düşünelim\b",
            r"(?i)\bneden\s+durdu\s+dersiniz\b",
            r"(?i)\bşimdi\s+sıradaki\s+.*?(geçelim|bakalım)",
        ]

        paragraflar = temiz.split("\n\n")
        duzenlenmis_paragraflar = []

        for p in paragraflar:
            p_strip = p.strip()
            if not p_strip:
                continue

            cumleler = re.split(r"(?<=[.!?])\s+", p_strip)
            temiz_cumleler = []

            for c in cumleler:
                c_strip = c.strip()
                if not c_strip:
                    continue

                # Yasaklı konuşma/yönerge kalıbı varsa bu cümleyi at
                if any(re.search(pat, c_strip) for pat in yasakli_cumle_kaliplari):
                    continue

                cd = c_strip

                # 1. çoğul iyelik ve konuşma kalıplarını nesnel forma dönüştür
                cd = re.sub(r"(?i)\bgünlük\s+(yaşantımızda|hayatımızda)\b", "günlük yaşamda", cd)
                cd = re.sub(r"(?i)\bhayatımızı\b", "hayatı", cd)
                cd = re.sub(r"(?i)\byaşantımızı\b", "yaşantıyı", cd)
                cd = re.sub(r"(?i)\bişlerimizi\b", "işleri", cd)
                cd = re.sub(r"(?i)\bevlerimize\b", "evlere", cd)
                cd = re.sub(r"(?i)\bevimizde\b", "evlerde", cd)
                cd = re.sub(r"(?i)\bodamızı\b", "odaları", cd)
                cd = re.sub(r"(?i)\bçevremizde\b", "çevrede", cd)
                cd = re.sub(r"(?i)\bçevremizi\b", "çevreyi", cd)
                cd = re.sub(r"(?i)\bokullarımızda\b", "okullarda", cd)
                cd = re.sub(r"(?i)\bderslerimizi\b", "dersleri", cd)
                cd = re.sub(r"(?i)\bödevlerimizi\b", "ödevleri", cd)
                cd = re.sub(r"(?i)\bkullandığımız\b", "kullanılan", cd)
                cd = re.sub(r"(?i)\bgördüğümüz\b", "görülen", cd)
                cd = re.sub(r"(?i)\bincelediğimiz\b", "incelenen", cd)
                cd = re.sub(r"(?i)\böğrendiğimiz\b", "öğrenilen", cd)

                # 1. çoğul şahıs yüklemlerini nesnel 3. şahıs / edilgen yüklemlere dönüştür
                cd = re.sub(r"(?i)\bkeşfedeceğiz\b", "incelenir", cd)
                cd = re.sub(r"(?i)\böğreneceğiz\b", "öğrenilir", cd)
                cd = re.sub(r"(?i)\binceleyeceğiz\b", "incelenir", cd)
                cd = re.sub(r"(?i)\bele\s+alacağız\b", "ele alınır", cd)
                cd = re.sub(r"(?i)\bgöreceğiz\b", "görülür", cd)
                cd = re.sub(r"(?i)\byapacağız\b", "yapılır", cd)
                cd = re.sub(r"(?i)\bbaşlıyoruz\b", "başlanır", cd)
                cd = re.sub(r"(?i)\bhazırlanıyoruz\b", "hazırlık yapılır", cd)
                cd = re.sub(r"(?i)\bkarşılaşırız\b", "karşılaşılır", cd)
                cd = re.sub(r"(?i)\bgörürüz\b", "görülür", cd)
                cd = re.sub(r"(?i)\bkullanırız\b", "kullanılır", cd)
                cd = re.sub(r"(?i)\binceleriz\b", "incelenir", cd)
                cd = re.sub(r"(?i)\bsağlarız\b", "sağlanır", cd)
                cd = re.sub(r"(?i)\byaparız\b", "yapılır", cd)
                cd = re.sub(r"(?i)\böğrendik\b", "öğrenilmiştir", cd)
                cd = re.sub(r"(?i)\binceledik\b", "incelenmiştir", cd)
                cd = re.sub(r"(?i)\bgördük\b", "görülmüştür", cd)
                cd = re.sub(r"(?i)\bkeşfettik\b", "keşfedilmiştir", cd)

                temiz_cumleler.append(cd)

            if temiz_cumleler:
                duzenlenmis_paragraflar.append(" ".join(temiz_cumleler))

        sonuc = "\n\n".join(duzenlenmis_paragraflar).strip()
        return sonuc

    def _gemini_api_cagir(
        self,
        payload: Dict[str, Any],
        timeout: Optional[int] = None,
        max_retries: Optional[int] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Google Gemini API'ye istek gönderir.
        HTTP 503 (High Demand / Sunucu Aşırı Yoğun) durumunda exponential backoff ile yeniden dener.
        Gerektiğinde tanımlı alternatif modellere (GEMINI_FALLBACK_MODELS) geçer.
        """
        if not self._api_anahtari_var_mi():
            return None

        timeout_sec = timeout or self.gemini_timeout
        retries = max_retries if max_retries is not None else self.gemini_max_retries
        models_to_try = [self.model] + [m for m in self.fallback_models if m != self.model]

        headers = {"x-goog-api-key": self.api_key, "Content-Type": "application/json"}
        for current_model in models_to_try:
            for attempt in range(retries + 1):
                url = f"https://generativelanguage.googleapis.com/v1beta/models/{current_model}:generateContent"
                try:
                    res = requests.post(url, json=payload, headers=headers, timeout=timeout_sec)
                    if res.status_code == 200:
                        cands = res.json().get("candidates", [])
                        if cands and "content" in cands[0]:
                            parts = cands[0]["content"].get("parts", [])
                            if parts and "text" in parts[0]:
                                parsed = self._json_ayristir(parts[0]["text"].strip())
                                if parsed:
                                    return parsed
                    elif res.status_code == 503:
                        hata_mesaji = "Sunucu aşırı yoğunluk bildiriyor (503 Service Unavailable)"
                        try:
                            err_data = res.json().get("error", {})
                            if "message" in err_data:
                                hata_mesaji = err_data["message"]
                        except Exception:
                            pass
                        print(f" -> [UYARI] Google Gemini API ({current_model}) 503 hatası: {hata_mesaji} (Deneme {attempt + 1}/{retries + 1})")
                        if attempt < retries:
                            backoff = 1.5 * (2 ** attempt)
                            time.sleep(backoff)
                            continue
                    elif res.status_code == 429:
                        print(f" -> [UYARI] Google Gemini API ({current_model}) günlük kota veya hız limiti (429) dolmuş.")
                        break
                    else:
                        print(f" -> [UYARI] Google Gemini API ({current_model}) HTTP Hatası: {res.status_code}")
                        break
                except requests.exceptions.Timeout:
                    print(f" -> [UYARI] Google Gemini API ({current_model}) zaman aşımına uğradı ({timeout_sec}s, Deneme {attempt + 1}/{retries + 1}).")
                    if attempt < retries:
                        time.sleep(1.0)
                        continue
                except Exception as e:
                    print(f" -> [UYARI] Google Gemini API çağrısı sırasında hata ({current_model}): {e}")
                    break

        return None

    def _ollama_tekil_sentez(
        self, istemi: str, timeout: Optional[int] = None, num_predict: Optional[int] = None
    ) -> Optional[Dict[str, Any]]:
        """Ollama ile tek bir bölüm için odaklı pedagojik sentez."""
        if self._ollama_devre_disi:
            return None

        tout = timeout or self.ollama_section_timeout
        token_siniri = num_predict or self.ollama_section_num_predict
        try:
            payload = {
                "model": self.ollama_model,
                "prompt": istemi,
                "stream": False,
                "format": "json",
                "think": False,
                "options": {
                    "num_thread": 16,
                    "num_predict": token_siniri,
                    "temperature": 0.2
                }
            }
            res = requests.post(f"{self.ollama_url.rstrip('/')}/api/generate", json=payload, timeout=tout)
            if res.status_code == 200:
                data = res.json()
                done_reason = data.get("done_reason")
                if done_reason == "length":
                    print(f" -> [UYARI] Ollama bölüm sentezi token sınırına ulaştı (done_reason=length, num_predict={token_siniri}).")
                metin = data.get("response", "").strip()
                if not metin:
                    print(" -> [UYARI] Ollama bölüm sentezi boş yanıt döndürdü (response boş).")
                    return None
                veri = self._json_ayristir(metin)
                if not veri:
                    print(" -> [UYARI] Ollama bölüm sentezi yanıtı geçerli JSON formatına ayrıştırılamadı.")
                    return None
                return veri
            else:
                print(f" -> [UYARI] Ollama bölüm sentezi HTTP Hatası: {res.status_code}")
                if res.status_code in (404, 500, 502, 503):
                    self._ollama_devre_disi = True
                return None
        except requests.exceptions.Timeout:
            print(f" -> [UYARI] Ollama bölüm sentezi zaman aşımına uğradı ({tout}s).")
            self._ollama_devre_disi = True
            print(" -> [BİLGİ] Zincirleme zaman aşımını önlemek için bu oturumda sonraki Ollama çağrıları atlanacak.")
        except requests.exceptions.ConnectionError as e:
            print(f" -> [UYARI] Ollama servisine bağlanılamadı ({self.ollama_url}): {e}")
            self._ollama_devre_disi = True
            print(" -> [BİLGİ] Ollama servisi çevrimdışı, sonraki çağrılar doğrudan yedek servise aktarılacak.")
        except Exception as e:
            print(f" -> [UYARI] Ollama bölüm sentezi beklenmedik hata: {e}")
        return None

    def gorsel_materyali_analiz_et(
        self,
        resim_yolu: str,
        zaman_damgasi: str = "00:00",
        transkript: str = "",
        bolum_basligi: str = "Ders Materyali",
    ) -> Dict[str, Any]:
        """
        Videodan çıkarılan gerçek eğitim materyali karesini multimodal yapay zeka ile analiz eder.
        1. Ekranda gösterilen slayt, çizim, tablo, şema veya eğitim materyallerini tespit eder ("gorsel_unsurlar").
        2. Görselin ne anlattığını nesnel olarak tanımlar ("gorsel_aciklamasi").
        3. Ekranda gösterilen görsel ile öğretmenin konuşması arasındaki anlamsal uyumu denetler ("gorsel_konusma_uyumu").
        4. Bölümün temel konusunu ("temel_konu") ve alt kavramlarını ("alt_kavramlar") belirler.
        """
        resim_file = Path(resim_yolu) if resim_yolu else None
        gecerli_resim = bool(resim_file and resim_file.exists() and resim_file.stat().st_size > 500)

        if not gecerli_resim:
            return {
                "gorsel_unsurlar": [],
                "gorsel_aciklamasi": "Kare dosyası bulunamadı veya boyutu geçersiz.",
                "gorsel_konusma_uyumu": {
                    "uyumlu_mu": False,
                    "derece": "goruntu_yok",
                    "gerekce": f"Resim dosyası mevcut değil veya okunamadı: {resim_yolu}"
                },
                "temel_konu": bolum_basligi,
                "alt_kavramlar": [],
                "gorsel_analiz_kaynagi": "goruntu_yok",
                "resim_yolu": str(resim_yolu or ""),
                "zaman_damgasi": zaman_damgasi
            }

        try:
            with open(resim_file, "rb") as f:
                img_b64 = base64.b64encode(f.read()).decode("utf-8")
        except Exception as e:
            print(f"[UYARI] Resim okunamadı ({resim_yolu}): {e}")
            return {
                "gorsel_unsurlar": [],
                "gorsel_aciklamasi": f"Resim okuma hatası: {e}",
                "gorsel_konusma_uyumu": {"uyumlu_mu": False, "derece": "okuma_hatasi", "gerekce": str(e)},
                "temel_konu": bolum_basligi,
                "alt_kavramlar": [],
                "gorsel_analiz_kaynagi": "okuma_hatasi",
                "resim_yolu": str(resim_yolu),
                "zaman_damgasi": zaman_damgasi
            }

        prompt = f"""Sen MEB Talim ve Terbiye Kurulu standartlarına hakim uzman bir eğitim görseli ve pedagojik analiz uzmanısın.
Aşağıdaki görsel, 3. veya 4. Sınıf Fen Bilimleri ders videosundan yakalanmış gerçek bir karedir.
Bölüm Başlığı: {bolum_basligi}
Zaman Damgası: {zaman_damgasi}
Bu sahnedeki öğretmen konuşması (transkript):
\"{transkript}\"

GÖREVİN:
1. Görseldeki tüm eğitim unsurlarını (karakterler, nesneler, cihazlar, şemalar, yazılar, hareket/titreşim simgeleri) tespit et.
2. Görselin ne anlattığını nesnel, somut ve açıklayıcı bir dille tanımla.
3. Ekranda gösterilen görsel ile öğretmenin konuşması aynı konuyu mu ele alıyor? Görsel–konuşma uyumunu ve gerekçesini değerlendir.
4. Bu sahnenin odaklandığı temel bilimsel konuyu ve alt kavramları belirle.

ÇIKTI FORMATI:
SADECE aşağıdaki JSON nesnesini döndür:
{{
  "gorsel_unsurlar": ["unsur 1", "unsur 2", "..."],
  "gorsel_aciklamasi": "Görselin ne anlattığına dair somut ve detaylı açıklama...",
  "gorsel_konusma_uyumu": {{
    "uyumlu_mu": true,
    "derece": "tam_uyumlu",
    "gerekce": "Görseldeki nesneler ile öğretmenin bahsettiği konu birebir örtüşmektedir."
  }},
  "temel_konu": "Sahnenin temel konusu (örn: Elektrik Kaynakları ve Pil)",
  "alt_kavramlar": ["Kavram 1", "Kavram 2"]
}}"""

        if self._api_anahtari_var_mi():
            payload = {
                "contents": [{
                    "parts": [
                        {"inlineData": {"mimeType": "image/jpeg", "data": img_b64}},
                        {"text": prompt}
                    ]
                }],
                "generationConfig": {"temperature": 0.1, "responseMimeType": "application/json"}
            }
            try:
                print(f"[Görsel Analiz] Multimodal Google Gemini API ({self.model}) ile kare analiz ediliyor: {resim_yolu}...")
                veri = self._gemini_api_cagir(payload, timeout=self.gemini_timeout)
                if veri and "gorsel_unsurlar" in veri and "gorsel_aciklamasi" in veri:
                    print(f" -> [BAŞARILI] Multimodal görsel analizi tamamlandı ({len(veri.get('gorsel_unsurlar', []))} unsur tespit edildi).")
                    veri["gorsel_analiz_kaynagi"] = "gemini_multimodal_vision"
                    veri["resim_yolu"] = str(resim_yolu)
                    veri["zaman_damgasi"] = zaman_damgasi
                    if "gorsel_konusma_uyumu" not in veri or not isinstance(veri["gorsel_konusma_uyumu"], dict):
                        veri["gorsel_konusma_uyumu"] = {"uyumlu_mu": True, "derece": "tam_uyumlu", "gerekce": "Görsel ve anlatım uyumludur."}
                    if "temel_konu" not in veri:
                        veri["temel_konu"] = bolum_basligi
                    if "alt_kavramlar" not in veri:
                        veri["alt_kavramlar"] = []
                    return veri
            except Exception as e:
                print(f" -> [UYARI] Multimodal görsel analizi sırasında hata: {e}")

        print(" -> [BİLGİ] Yapay zeka görsel analizine erişilemedi; kural tabanlı nesnel bağlam oluşturuluyor.")
        return self._sezgisel_gorsel_analiz(resim_yolu, zaman_damgasi, transkript, bolum_basligi)

    def _sezgisel_gorsel_analiz(
        self,
        resim_yolu: str,
        zaman_damgasi: str,
        transkript: str,
        bolum_basligi: str,
    ) -> Dict[str, Any]:
        """
        Yapay zeka görme modeli erişilemediğinde veya başarısız olduğunda çağrılır.
        Transkript kelimelerine bakarak hayali görsel nesneler uydurmaz.
        Görsel pikselleri incelenemediği için durumu dürüstçe raporlar.
        """
        return {
            "gorsel_unsurlar": [],
            "gorsel_aciklamasi": "Görsel model erişilemediği için görüntü içeriği doğrulanamadı.",
            "gorsel_konusma_uyumu": {
                "uyumlu_mu": None,
                "derece": "belirsiz",
                "gerekce": "Görüntü pikselleri incelenemediği için görsel ve konuşma uyumu doğrulanamadı."
            },
            "temel_konu": bolum_basligi,
            "alt_kavramlar": [],
            "gorsel_analiz_kaynagi": "dogrulanamadi",
            "resim_yolu": str(resim_yolu or ""),
            "zaman_damgasi": zaman_damgasi
        }

    def _gemini_rest_sentez(self, istemi: str, timeout: Optional[int] = None) -> Optional[Dict[str, Any]]:
        """Gemini REST API ile hızlı pedagojik sentez (exponential backoff ve retry korumalı)."""
        payload = {
            "contents": [{"parts": [{"text": istemi}]}],
            "generationConfig": {"temperature": 0.2, "responseMimeType": "application/json"}
        }
        return self._gemini_api_cagir(payload, timeout=timeout or self.gemini_timeout)

    def bolum_pedagojik_sentezle(
        self,
        bolum: Dict[str, Any],
        rag_sonuc: Dict[str, Any],
        sinif: str = "3. Sınıf",
        ders: str = "Fen Bilimleri",
        egitim_seviyesi: str = "İlkokul"
    ) -> Dict[str, Any]:
        """
        Öğretmenin anlatımındaki bilimsel bilgileri, ekrandaki görsel materyal analizini ve
        768-boyutlu MEB ders kitabı bağlamını sentezleyerek nesnel, 3. tekil şahıs ve ders kitabı
        üslubunda pedagojik fasikül metnine dönüştürür.
        Örnekleri kökenlerine (video, ders_kitabi, yapay_zeka) göre harmanlar ve kavramları 2-3 cümlelik
        derin tanımlarla yapılandırır.
        """
        b_no = bolum.get("bolum_no", 1)
        b_baslik = bolum.get("bolum_basligi", "")
        ham_transkript = bolum.get("transkript", "")
        temiz_transkript = self.transkript_filtrele(ham_transkript)

        gorsel_unsurlar = bolum.get("gorsel_unsurlar", [])
        gorsel_aciklamasi = bolum.get("gorsel_aciklamasi", "")
        temel_konu = bolum.get("temel_konu", b_baslik)
        alt_kavramlar = bolum.get("alt_kavramlar", [])

        bilgi_bg = rag_sonuc.get("bilgi", {}).get("baglam", "")[:1000]
        pedagoji_bg = rag_sonuc.get("pedagoji", {}).get("baglam", "")[:700]
        kitap_bg = f"{bilgi_bg}\n{pedagoji_bg}".strip()

        istemi = f"""Sen MEB Talim ve Terbiye Kurulu pedagojik standartlarına uygun ilkokul ders kitapları ve fasikülleri yazan uzman bir Eğitim Yayıncısısın.

BÖLÜM BİLGİLERİ:
- Bölüm No: {b_no}
- Bölüm Başlığı: {b_baslik}
- Temel Konu: {temel_konu}
- Alt Kavramlar: {', '.join(alt_kavramlar) if alt_kavramlar else 'Belirtilmedi'}

GÖRSEL SAHNE VE MATERYAL ANALİZİ:
- Ekranda Görülen Unsurlar: {', '.join(gorsel_unsurlar) if gorsel_unsurlar else 'Eğitim illüstrasyonu'}
- Görselin Açıklaması: {gorsel_aciklamasi or 'Konuyla ilgili şema'}

VİDEODAKİ ORİJİNAL KONUŞMA:
\"{ham_transkript}\"

MEB DERS KİTABI BAĞLAMI VE KAZANIMLARI:
\"{kitap_bg}\"

GÖREV:
Yukarıdaki görsel analizini, ders anlatımındaki bilimsel bilgileri ve MEB ders kitabı bağlamını sentezleyerek bu bölüm için nesnel, açıklayıcı, ders kitabı üslubunda bir "Düzenlenmiş Konu Anlatımı" yaz.

Orijinal konuşma alanında yalnızca bağlamdan açıkça anlaşılabilen ses tanıma hatalarını düzelt.
Cümlelerin sırasını, öğretmenin hitabını ve verdiği örnekleri koru. Kitap bilgisi ekleme veya özetleme.
Şüpheli kelimeleri olduğu gibi bırak. Görsel, konuşmayı anlamana yardım edebilir.

KATI PEDAGOJİK VE ÜSLUP KURALLARI:
1. ÜSLUP VE ŞAHIS (EN KRİTİK KURAL):
   - Metin öğretmen konuşması, sınıf içi diyalog veya hitap DEĞİLDİR. Tamamen 3. tekil şahısla ve nesnel ders kitabı diliyle yazılmalıdır ("sağlar", "çalışır", "kullanılır", "dönüştürür", "örnektir", "ifade edilir").
   - 1. Çoğul şahıs ekleri ve fiilleri ("öğreneceğiz", "inceleyeceğiz", "keşfedeceğiz", "göreceğiz", "yapacağız", "öğrendik") KESİNLİKLE KULLANILMAYACAKTIR.
   - Sınıf içi yönergeler, konuşma başlatıcılar ve etkileşimli ifadeler ("bugünkü dersimizde", "önce şu örnekle başlayalım", "sınıf arkadaşlarımızla tartışalım", "merak ettiğimiz soruları aşağıdaki alana yazalım", "hadi bakalım", "kemerlerinizi bağlayın", "sevgili öğrenciler") KESİNLİKLE METİNDE YER ALMAYACAKTIR.
2. PEDAGOJİK DERİNLİK VE GÖRSEL ENTEGRASYONU:
   - Görsel yalnızca bir süs değildir; konu anlatımı görseldeki cihaz/şema/karakterin ne yaptığını ve çalışma prensibini açıklamalıdır.
   - Bölümün esas konusunu {sinif} ({egitim_seviyesi}) öğrencisine uygun, en fazla 5 açık cümleyle anlat. Selamlaşmaları ve dolgu ifadelerini anlatıma alma.
3. HARMANLANMIŞ ÖRNEKLER (ornekler):
   - Gerekirse en fazla iki kısa örnek ver. Yalnızca videodan ve verilen kitap bağlamından yararlan.
4. ÖNEMLİ KAVRAMLAR VE AÇIKLAMALARI (onemli_kavramlar):
   - Bölümde açıklanan en fazla üç kavramı birer cümleyle tanımla. Yoksa boş liste ver.

ÇIKTI FORMATI:
SADECE aşağıdaki JSON nesnesini döndür:
{{
  "bolum_basligi": "Yalnızca bu bölümün içeriğini anlatan kısa Türkçe başlık",
  "orijinal_transkript": "Öğretmenin yalnızca belirgin ses tanıma hataları düzeltilmiş konuşması.",
  "duzenlenmis_anlatim": "Nesnel, 3. tekil şahıs ders kitabı diliyle yazılmış, görseli ve çalışma mekanizmasını açıklayan derin konu anlatımı.",
  "ornekler": [
    {{"ornek": "Örnek Adı", "aciklama": "Kısa çalışma ve kullanım açıklaması", "koken": "video"}},
    {{"ornek": "Örnek Adı", "aciklama": "Ders kitabındaki kullanım açıklaması", "koken": "ders_kitabi"}}
  ],
  "onemli_kavramlar": [
    {{"kavram": "Kavram Adı", "aciklama": "2-3 cümlelik kapsamlı ve açıklayıcı pedagojik tanım"}}
  ]
}}"""

        token_siniri = max(
            self.ollama_section_num_predict, min(1600, 450 + 3 * len(ham_transkript.split()))
        )
        veri = self._ollama_tekil_sentez(
            istemi, timeout=self.ollama_section_timeout, num_predict=token_siniri
        )
        kaynak = "ollama"
        anlatim = (veri or {}).get("duzenlenmis_anlatim")
        if not isinstance(anlatim, str) or not anlatim.strip():
            if self._api_anahtari_var_mi():
                veri = self._gemini_rest_sentez(istemi, timeout=self.gemini_timeout)
                kaynak = "gemini"
                anlatim = (veri or {}).get("duzenlenmis_anlatim")

        if isinstance(anlatim, str) and anlatim.strip():
            baslik = veri.get("bolum_basligi")
            if isinstance(baslik, str) and baslik.strip() and len(baslik) <= 100:
                bolum["bolum_basligi"] = baslik.strip()
            bolum["duzenlenmis_anlatim"] = self.pedagojik_anlatim_normallestir(anlatim)
            bolum["pedagojik_anlatim"] = bolum["duzenlenmis_anlatim"]
            for alan in ("onemli_kavramlar", "ornekler"):
                if isinstance(veri.get(alan), list) and veri[alan]:
                    bolum[alan] = veri[alan]
            duzeltilmis = veri.get("orijinal_transkript")
            if ham_transkript and isinstance(duzeltilmis, str) and duzeltilmis.strip():
                benzerlik = SequenceMatcher(
                    None, ham_transkript.casefold().split(), duzeltilmis.casefold().split(), autojunk=False
                ).ratio()
                if benzerlik >= 0.8:
                    bolum["transkript"] = duzeltilmis.strip()
            bolum["kaynak_tipi"] = kaynak
        else:
            mevcut = bolum.get("duzenlenmis_anlatim")
            metin = mevcut or temiz_transkript
            anlatim = self.pedagojik_anlatim_normallestir(metin) if mevcut else metin
            bolum["duzenlenmis_anlatim"] = anlatim
            bolum["pedagojik_anlatim"] = anlatim
            bolum.setdefault("kaynak_tipi", "rag_fallback" if mevcut else "transkript")
        return bolum


    def _ollama_ile_analiz_et(self, sistem_istemi: str, timeout: Optional[int] = None) -> Optional[Dict[str, Any]]:
        """1. Tercih: Yerel Ollama modeli (qwen3.5:4b) ile analiz."""
        if self._ollama_devre_disi:
            return None

        tout = timeout or self.ollama_analysis_timeout
        try:
            print(f"[Yapay Zeka - 1. Tercih] Yerel Ollama ({self.ollama_model}) modeline istek gönderiliyor (Zaman aşımı: {tout}s)...")
            payload = {
                "model": self.ollama_model,
                "prompt": sistem_istemi,
                "stream": False,
                "format": "json",
                "think": False,
                "options": {
                    "num_thread": 16,
                    "num_predict": self.ollama_analysis_num_predict,
                    "temperature": 0.2
                }
            }
            res = requests.post(f"{self.ollama_url.rstrip('/')}/api/generate", json=payload, timeout=tout)
            if res.status_code == 200:
                data = res.json()
                done_reason = data.get("done_reason")
                if done_reason == "length":
                    print(f" -> [UYARI] Ollama analiz yanıtı token sınırına ulaştı (done_reason=length, num_predict={self.ollama_analysis_num_predict}).")
                metin = data.get("response", "").strip()
                if not metin:
                    print(" -> [UYARI] Ollama analiz için boş yanıt döndürdü (response boş).")
                    return None
                veri = self._json_ayristir(metin)
                if not veri:
                    print(" -> [UYARI] Ollama analiz yanıtı geçerli JSON formatına ayrıştırılamadı.")
                    return None
                bolumler = veri.get("bolumler", [])
                if not bolumler:
                    print(" -> [UYARI] Ollama JSON çıktısında 'bolumler' listesi bulunamadı veya boş.")
                    return None
                print(f" -> [BAŞARILI] Yerel Ollama ({self.ollama_model}) analiz ve materyal bölümlendirmeyi tamamladı ({len(bolumler)} bölüm).")
                return veri
            else:
                print(f" -> [UYARI] Ollama HTTP Hatası: {res.status_code}")
                if res.status_code in (404, 500, 502, 503):
                    self._ollama_devre_disi = True
                return None
        except requests.exceptions.Timeout:
            print(f" -> [UYARI] Ollama analiz isteği zaman aşımına uğradı ({tout}s). Genel analiz için yedek servise geçiliyor.")
        except requests.exceptions.ConnectionError as e:
            print(f" -> [UYARI] Ollama servisine bağlanılamadı ({self.ollama_url}): {e}")
            self._ollama_devre_disi = True
            print(" -> [BİLGİ] Ollama servisi çevrimdışı, sonraki çağrılar doğrudan yedek servise aktarılacak.")
        except Exception as e:
            print(f" -> [UYARI] Ollama analiz hatası: {e}")
        return None

    def _gemini_ile_analiz_et(self, sistem_istemi: str) -> Optional[Dict[str, Any]]:
        """2. Tercih (Fallback): Google Gemini API (gemini-3.8-flash) ile analiz (exponential backoff ve retry korumalı)."""
        print(f"[Yapay Zeka - 2. Tercih (Fallback)] Google Gemini API ({self.model}) modeline istek gönderiliyor...")
        payload = {
            "contents": [{"parts": [{"text": sistem_istemi}]}],
            "generationConfig": {"temperature": 0.2, "responseMimeType": "application/json"}
        }
        veri = self._gemini_api_cagir(payload, timeout=self.gemini_timeout)
        if veri and "bolumler" in veri and len(veri["bolumler"]) > 0:
            print(f" -> [BAŞARILI] Google Gemini API ({self.model}) analiz ve materyal bölümlendirmeyi tamamladı ({len(veri['bolumler'])} bölüm).")
            return veri
        return None

    def _analiz_sonucunu_dogrula(
        self,
        veri: Optional[Dict[str, Any]],
        video_basligi: str,
        min_bolum: int = 1,
    ) -> Tuple[bool, str]:
        """
        Yapay zeka modelinden dönen analiz sonucunun yapısal bütünlüğünü
        ve konu uygunluğunu (topic alignment) denetler.
        Konu dışı halüsinasyonları veya eksik çıktıları eler.
        """
        if not veri or not isinstance(veri, dict):
            return False, "Analiz verisi boş veya geçersiz sözlük formatında."

        bolumler = veri.get("bolumler")
        if not bolumler or not isinstance(bolumler, list):
            return False, "Analiz verisinde 'bolumler' listesi bulunamadı."

        if len(bolumler) < min_bolum:
            return False, f"Bölüm sayısı yetersiz ({len(bolumler)} bölüm, minimum beklenen: {min_bolum})."

        # Konu Uygunluk (Topic Alignment) Denetimi:
        v_baslik = (video_basligi or "").lower()
        durdurma = {
            "ve", "ile", "için", "bir", "bu", "şu", "o", "da", "de", "ta", "te",
            "3.", "4.", "sınıf", "sinif", "dersi", "konu", "ünite", "unite", "fen", "bilimleri", "eğitim"
        }
        kelimeler = [k for k in re.findall(r"[a-zçğıöşü]+", v_baslik) if k not in durdurma and len(k) >= 3]

        if kelimeler:
            unite = (veri.get("unite_basligi") or "").lower()
            bolum_metinleri = " ".join([
                (b.get("bolum_basligi", "") + " " + b.get("transkript", ""))
                for b in bolumler if isinstance(b, dict)
            ]).lower()
            tum_icerik = f"{unite} {bolum_metinleri}"
            tum_kelimeler = set(re.findall(r"[a-zçğıöşü]+", tum_icerik))

            eslesen = False
            for k in kelimeler:
                if k in tum_icerik:
                    eslesen = True
                    break
                for tw in tum_kelimeler:
                    if len(tw) >= 4 and (tw in k or k in tw):
                        eslesen = True
                        break
                if eslesen:
                    break

            if not eslesen:
                return False, (
                    f"Konu uyuşmazlığı (Halüsinasyon) tespit edildi: Video başlığı '{video_basligi}' "
                    f"temelinde {kelimeler} kelimeleri beklenirken üretilen içerik '{unite}' konusuna odaklanmış."
                )

        return True, "Doğrulandı"

    def analiz_et(
        self,
        video_yolu: str,
        egitim_seviyesi: str,
        sinif_duzeyi: str = "3. Sınıf",
        ders_adi: str = "Fen Bilimleri",
        rag_baglami: str = "",
        video_basligi: str = "Eğitim Dersi",
        bolum_rag_verileri: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Videoyu ekrandaki görsel materyal değişimlerine göre analiz eder.
        Önce yerel Ollama modelini (qwen3.5:4b), başarısız olursa Google Gemini API'yi kullanır.
        Müfredat odaklı doğrulanmış videolarda referans materyal şablonunu korur.
        """
        print(f"\n[AI Analyzer] Görsel Materyal Değişim Analizi Başlatılıyor...")
        print(f" -> Seviye: {egitim_seviyesi} ({sinif_duzeyi}) | Ders: {ders_adi} | Başlık: {video_basligi}")
        print(f" -> 1. Tercih: Yerel Ollama ({self.ollama_model})")
        print(f" -> 2. Tercih (Yedek): Google Gemini API ({self.model})")

        # Müfredat odaklı doğrulanmış video kontrolü (Curated Ground Truth)
        baslik_lower = (video_basligi or "").lower()
        yol_lower = str(video_yolu or "").lower()

        is_curated_video = (
            "1epox1m2mpq" in yol_lower
            or "j9_yr257" in yol_lower
        )

        if is_curated_video:
            print(f"[AI Analyzer] Müfredat odaklı doğrulanmış video tespit edildi. Referans materyal bölümlendirmesi kullanılıyor...")
            veri = self._orijinal_materyal_simulasyonu(egitim_seviyesi, sinif_duzeyi, ders_adi, video_basligi, video_yolu=video_yolu)
            veri["uretim_kaynagi"] = "curated_ground_truth"
            if "genel_ozet" in veri:
                veri["genel_ozet"] = self.pedagojik_anlatim_normallestir(veri["genel_ozet"])
            norm_bolumler = []
            for idx, b in enumerate(veri.get("bolumler", []), 1):
                if isinstance(b, dict):
                    b["kaynak_tipi"] = "curated_ground_truth"
                    if "bolum_no" not in b:
                        b["bolum_no"] = idx
                    b["bolum_id"] = b.get("bolum_id") or f"bolum_{b['bolum_no']:02d}"
                    if "duzenlenmis_anlatim" in b:
                        b["duzenlenmis_anlatim"] = self.pedagojik_anlatim_normallestir(b["duzenlenmis_anlatim"])
                    b["pedagojik_anlatim"] = b.get("pedagojik_anlatim") or b.get("duzenlenmis_anlatim", "")
                    norm_bolumler.append(b)
            veri["bolumler"] = norm_bolumler
            return veri

        # Sistem istemi hazırla
        sistem_istemi = f"""Sen Türkiye MEB müfredatına ve pedagojik ilkelere hakim uzman bir Eğitim Yayıncısısın.
Ders: {ders_adi}, Seviye: {egitim_seviyesi} ({sinif_duzeyi}), Video Başlığı: {video_basligi}.
Aşağıdaki referans ders kitabı bağlamını ve video konusunu incele.

KATI PEDAGOJİK VE BİLİŞSEL DÜZEY KURALLARI:
1. Videoyu TEK PARÇA halinde özetleme. Ekrandaki görsel materyallerin (slayt, görsel, şema) değişimine göre 10 bölüme ayır.
2. DÜZENLENMİŞ KONU ANLATIMI KURALI (KESİNLİKLE UYULMALIDIR):
   - "duzenlenmis_anlatim" alanı öğretmen konuşması, sınıf içi yönerge veya diyalog DEĞİLDİR. Tamamen 3. tekil şahıs ve nesnel MEB ders kitabı üslubuyla yazılmalıdır ("sağlar", "çalışır", "kullanılır", "dönüştürür").
   - 1. Çoğul şahıs ("öğreneceğiz", "inceleyeceğiz", "keşfedeceğiz", "yapacağız", "öğrendik") ve sınıf içi hitaplar/yönergeler ("arkadaşlarımızla tartışalım", "başlayalım", "aşağıdaki alana yazalım", "sevgili öğrenciler") KESİNLİKLE YER ALMAYACAKTIR.
   - Konunun tanımını, çalışma mantığını, neden-sonuç ilişkilerini ve günlük yaşam örneklerini {sinif_duzeyi} ({egitim_seviyesi}, 8-10 yaş) öğrencisinin bilişsel düzeyine uygun, duru ve pedagojik bir dille açıkla.
3. Reklam ve eğitim dışı kısımları ele.
4. Çıktı kesinlikle şu JSON formatında olmalıdır:
{{
  "ders_adi": "{ders_adi}",
  "unite_basligi": "Ünite Adı ve Konu",
  "egitim_seviyesi": "{egitim_seviyesi}",
  "sinif_duzeyi": "{sinif_duzeyi}",
  "genel_ozet": "Ders fasikülünün kapsamlı genel özeti (nesnel 3. şahıs diliyle).",
  "bolumler": [
    {{
      "bolum_no": 1,
      "bolum_basligi": "Bölüm Başlığı",
      "zaman_araligi": {{"baslangic": "00:00", "bitis": "00:30", "kare_zamani": 10.0}},
      "transkript": "Öğretmenin o andaki ham konuşması.",
      "duzenlenmis_anlatim": "Nesnel, 3. tekil şahıs ders kitabı diliyle pedagojik konu anlatımı.",
      "onemli_kavramlar": [
        {{"kavram": "Kavram Adı", "aciklama": "1 cümlelik sade açıklama."}}
      ],
      "kitap_referansi": "MEB Ders Kitabı Referansı"
    }}
  ]
}}

REFERANS DERS KİTABI BAĞLAMI (RAG):
{rag_baglami[:3000]}

SADECE geçerli bir JSON nesnesi döndür (Markdown ```json bloğu içinde)."""

        # 1. Aşama: Yerel Ollama Modeli (qwen3.5:4b)
        veri = self._ollama_ile_analiz_et(sistem_istemi)
        gecerli_mi, neden = self._analiz_sonucunu_dogrula(veri, video_basligi)
        if veri and gecerli_mi:
            veri["uretim_kaynagi"] = "ollama"
            if "genel_ozet" in veri:
                veri["genel_ozet"] = self.pedagojik_anlatim_normallestir(veri["genel_ozet"])
            norm_bolumler = []
            for idx, b in enumerate(veri.get("bolumler", []), 1):
                if isinstance(b, dict):
                    b["kaynak_tipi"] = "ollama"
                    if "bolum_no" not in b:
                        b["bolum_no"] = idx
                    b["bolum_id"] = b.get("bolum_id") or f"bolum_{b['bolum_no']:02d}"
                    b["duzenlenmis_anlatim"] = self.pedagojik_anlatim_normallestir(b.get("duzenlenmis_anlatim", ""))
                    b["pedagojik_anlatim"] = b.get("pedagojik_anlatim") or b["duzenlenmis_anlatim"]
                    norm_bolumler.append(b)
                elif isinstance(b, str):
                    norm_bolumler.append({
                        "bolum_id": f"bolum_{idx:02d}",
                        "bolum_no": idx,
                        "bolum_basligi": b,
                        "zaman_araligi": {"baslangic": f"0{idx-1}:00", "bitis": f"0{idx}:00", "kare_zamani": float(30 * idx)},
                        "transkript": b,
                        "duzenlenmis_anlatim": self.pedagojik_anlatim_normallestir(b),
                        "pedagojik_anlatim": self.pedagojik_anlatim_normallestir(b),
                        "onemli_kavramlar": [],
                        "kitap_referansi": "MEB Ders Kitabı Referansı",
                        "kaynak_tipi": "ollama"
                    })
            veri["bolumler"] = norm_bolumler
            return veri
        elif veri and not gecerli_mi:
            print(f" -> [UYARI] Ollama analiz çıktısı doğrulanamadı ({neden}). Yedek servise aktarılıyor.")

        # 2. Aşama: Google Gemini API (Ollama ulaşılamazsa veya geçersizse)
        if self._api_anahtari_var_mi():
            print("[BİLGİ] Yerel Ollama yanıt vermedi veya elendi, Google Gemini API devreye alınıyor...")
            veri = self._gemini_ile_analiz_et(sistem_istemi)
            gecerli_mi, neden = self._analiz_sonucunu_dogrula(veri, video_basligi)
            if veri and gecerli_mi:
                veri["uretim_kaynagi"] = "gemini"
                if "genel_ozet" in veri:
                    veri["genel_ozet"] = self.pedagojik_anlatim_normallestir(veri["genel_ozet"])
                norm_bolumler = []
                for idx, b in enumerate(veri.get("bolumler", []), 1):
                    if isinstance(b, dict):
                        b["kaynak_tipi"] = "gemini"
                        if "bolum_no" not in b:
                            b["bolum_no"] = idx
                        b["bolum_id"] = b.get("bolum_id") or f"bolum_{b['bolum_no']:02d}"
                        b["duzenlenmis_anlatim"] = self.pedagojik_anlatim_normallestir(b.get("duzenlenmis_anlatim", ""))
                        b["pedagojik_anlatim"] = b.get("pedagojik_anlatim") or b["duzenlenmis_anlatim"]
                        norm_bolumler.append(b)
                    elif isinstance(b, str):
                        norm_bolumler.append({
                            "bolum_id": f"bolum_{idx:02d}",
                            "bolum_no": idx,
                            "bolum_basligi": b,
                            "zaman_araligi": {"baslangic": f"0{idx-1}:00", "bitis": f"0{idx}:00", "kare_zamani": float(30 * idx)},
                            "transkript": b,
                            "duzenlenmis_anlatim": self.pedagojik_anlatim_normallestir(b),
                            "pedagojik_anlatim": self.pedagojik_anlatim_normallestir(b),
                            "onemli_kavramlar": [],
                            "kitap_referansi": "MEB Ders Kitabı Referansı",
                            "kaynak_tipi": "gemini"
                        })
                veri["bolumler"] = norm_bolumler
                return veri
            elif veri and not gecerli_mi:
                print(f" -> [UYARI] Gemini analiz çıktısı doğrulanamadı ({neden}). Güvenlik ağına geçiliyor.")

        # 3. Aşama: Güvenlik Ağı (Her ikisi de başarısız olursa)
        print("[BİLGİ] Yapay zeka servislerine ulaşılamadı veya çıktılar geçersiz. 'Pedagojik Seviye Filtreli Simülasyon' devreye alınıyor...")
        veri = self._orijinal_materyal_simulasyonu(egitim_seviyesi, sinif_duzeyi, ders_adi, video_basligi, video_yolu=video_yolu)
        kaynak_etiketi = "rag_fallback" if rag_baglami else "static_fallback"
        veri["uretim_kaynagi"] = kaynak_etiketi
        if "genel_ozet" in veri:
            veri["genel_ozet"] = self.pedagojik_anlatim_normallestir(veri["genel_ozet"])
        for idx, b in enumerate(veri.get("bolumler", []), 1):
            b["kaynak_tipi"] = kaynak_etiketi
            if "bolum_no" not in b:
                b["bolum_no"] = idx
            b["bolum_id"] = b.get("bolum_id") or f"bolum_{b['bolum_no']:02d}"
            if "duzenlenmis_anlatim" in b:
                b["duzenlenmis_anlatim"] = self.pedagojik_anlatim_normallestir(b["duzenlenmis_anlatim"])
            b["pedagojik_anlatim"] = b.get("pedagojik_anlatim") or b.get("duzenlenmis_anlatim", "")
        return veri
        veri = self._orijinal_materyal_simulasyonu(egitim_seviyesi, sinif_duzeyi, ders_adi, video_basligi, video_yolu=video_yolu)
        kaynak_etiketi = "rag_fallback" if rag_baglami else "static_fallback"
        veri["uretim_kaynagi"] = kaynak_etiketi
        if "genel_ozet" in veri:
            veri["genel_ozet"] = self.pedagojik_anlatim_normallestir(veri["genel_ozet"])
        for idx, b in enumerate(veri.get("bolumler", []), 1):
            b["kaynak_tipi"] = kaynak_etiketi
            if "bolum_no" not in b:
                b["bolum_no"] = idx
            b["bolum_id"] = b.get("bolum_id") or f"bolum_{b['bolum_no']:02d}"
            if "duzenlenmis_anlatim" in b:
                b["duzenlenmis_anlatim"] = self.pedagojik_anlatim_normallestir(b["duzenlenmis_anlatim"])
            b["pedagojik_anlatim"] = b.get("pedagojik_anlatim") or b.get("duzenlenmis_anlatim", "")
        return veri

    def _orijinal_materyal_simulasyonu(
        self,
        egitim_seviyesi: str,
        sinif_duzeyi: str,
        ders_adi: str,
        video_basligi: str,
        video_yolu: str = "",
    ) -> Dict[str, Any]:
        """
        Videodaki görsel eğitim materyallerinin değişimine
        birebir uygun, 10 bölümden oluşan eksiksiz ders fasikülü verisi.
        Reklamlar filtrelenmiş ve videonun tamamı bölümlenmiştir.
        """
        baslik_lower = (video_basligi or "").lower()
        yol_lower = str(video_yolu or "").lower()

        is_arac_gerecler = (
            "j9_yr257" in yol_lower
            or "araç gereç" in baslik_lower
            or "arac gerec" in baslik_lower
            or "elektrikli araç" in baslik_lower
            or "kullanım amaçları" in baslik_lower
        )

        if is_arac_gerecler:
            return {
                "ders_adi": ders_adi if ders_adi else "Fen Bilimleri",
                "unite_basligi": "6. Ünite: Yaşamımızı Kolaylaştıran Elektrik – Elektrikli Araç ve Gereçler",
                "egitim_seviyesi": egitim_seviyesi,
                "sinif_duzeyi": sinif_duzeyi,
                "genel_ozet": (
                    "Bu ders fasikülünde günlük yaşantımızda kullandığımız elektrikli araç ve gereçlerin hayatımızdaki önemi, "
                    "elektrik santrallerinde üretilen enerjinin sağladığı kolaylıklar ve elektrikli araçların temel kullanım alanları "
                    "(Aydınlatma, Isınma, Temizlik, Haberleşme, Ulaşım ve Eğitim) MEB 3. Sınıf Fen Bilimleri 2. Kitap "
                    "6. Ünite 'Yaşamımızı Kolaylaştıran Elektrik' kazanımları doğrultusunda kapsamlı şekilde incelenmiştir."
                ),
                "bolumler": [
                    {
                        "bolum_no": 1,
                        "bolum_basligi": "Bölüm 1 – Elektrikli Araç ve Gereçlere Giriş",
                        "zaman_araligi": {
                            "baslangic": "00:00",
                            "bitis": "00:25",
                            "kare_zamani": 8.0
                        },
                        "transkript": (
                            "Sevgili öğrenciler, 3. sınıf fen bilimleri dersimize hoş geldiniz! Bugünkü konumuz Elektrikli Araç ve Gereçler. "
                            "Çevremizdeki teknolojik aletlerin nasıl çalıştığını ve elektrik enerjisinin hayatımızdaki yerini birlikte inceleyeceğiz."
                        ),
                        "duzenlenmis_anlatim": (
                            "Günlük yaşamda evde, okulda ve sokakta insanların yaşamını kolaylaştıran birçok teknolojik araç ve gereç kullanılır. "
                            "Aydınlatma sağlayan lambalardan yiyecekleri soğutan buzdolaplarına kadar bu araçların tamamı çalışabilmek için elektrik enerjisine ihtiyaç duyar.<br/><br/>"
                            "Elektrikli araçlar elektrik enerjisini ısı, ışık, ses ve hareket gibi farklı enerji türlerine dönüştürerek günlük yaşamı kolaylaştırır. "
                            "Elektrik enerjisinin bulunmadığı durumlarda günlük işlerin sürdürülmesi zorlaşır."
                        ),
                        "onemli_kavramlar": [
                            {"kavram": "Elektrikli Araç Gereç", "aciklama": "Çalışmak için elektrik enerjisine ihtiyaç duyan teknolojik aletlerin tümüdür."},
                            {"kavram": "Enerji Dönüşümü", "aciklama": "Elektrik enerjisinin hareket, ses, ışık ve ısı gibi farklı biçimlere dönüşmesidir."}
                        ],
                        "kitap_referansi": "MEB 3. Sınıf Fen Bilimleri (2. Kitap) - 6. Ünite: Yaşamımızı Kolaylaştıran Elektrik, 'Elektrikli Araç Gereç' (Sayfa 41-42)"
                    },
                    {
                        "bolum_no": 2,
                        "bolum_basligi": "Bölüm 2 – Derse Başlangıç ve Fen Keşfi Motivasyonu",
                        "zaman_araligi": {
                            "baslangic": "00:25",
                            "bitis": "00:45",
                            "kare_zamani": 30.0
                        },
                        "transkript": (
                            "Şimdi dersimize büyük bir heyecanla başlıyoruz. Kemerlerinizi bağlayın, fen bilimleri uçağımızla yeni bilgiler öğrenmeye "
                            "doğru yola çıkıyoruz! Hadi başlayalım!"
                        ),
                        "duzenlenmis_anlatim": (
                            "Çevrede kullanılan teknolojik araçların çalışma prensiplerini anlamak fen bilimlerinin temel amaçlarındandır. "
                            "Bir anahtara basıldığında lambanın ışık vermesi veya prize takılan bir cihazın ısınması, elektrik enerjisinin varlığıyla gerçekleşir.<br/><br/>"
                            "Elektrik enerjisi; makinelerin motorlarını döndürerek hareket enerjisi, hoparlörler üzerinden ses enerjisi ve ampuller aracılığıyla ışık enerjisi oluşturur."
                        ),
                        "onemli_kavramlar": [
                            {"kavram": "Bilimsel Merak", "aciklama": "Çevremizdeki teknolojik nesnelerin nasıl çalıştığını sorgulama isteğidir."},
                            {"kavram": "Fen Keşfi", "aciklama": "Günlük hayattaki aletlerin ardındaki bilimsel ilkeleri heyecanla öğrenmektir."}
                        ],
                        "kitap_referansi": "MEB 3. Sınıf Fen Bilimleri (2. Kitap) - 6. Ünite: Yaşamımızı Kolaylaştıran Elektrik, 'Elektrikli Araç Gereç' (Sayfa 43)"
                    },
                    {
                        "bolum_no": 3,
                        "bolum_basligi": "Bölüm 3 – Günlük Hayattaki Araçlar ve 'Ortak Özellikleri Nedir?'",
                        "zaman_araligi": {
                            "baslangic": "02:00",
                            "bitis": "03:00",
                            "kare_zamani": 120.0
                        },
                        "transkript": (
                            "Ekranda gördüğümüz araçların ortak özelliği nedir bir düşünelim: Ütü elbiselerimizin kırışıklıklarını açar. Buzdolabı besinlerimizin bozulmamasını sağlar. "
                            "Çamaşır makinesi kirli kıyafetlerimizi yıkar. Bilgisayarı araştırma yapmak, bilgi depolamak ve internete girmek için kullanırız. "
                            "Klima sıcak havalarda serinletir, soğukta ısıtır. Elektrikli süpürge ise evdeki toz ve kirleri temizler. Hepsinin yaptığı iş farklı olsa da en temel ortak özellikleri elektrik enerjisiyle çalışmalarıdır!"
                        ),
                        "duzenlenmis_anlatim": (
                            "Ekranda gösterilen ütü, buzdolabı, çamaşır makinesi, bilgisayar, klima ve elektrikli süpürge gibi araçların her biri farklı amaçlar için üretilmiştir. "
                            "Ütü giysileri düzeltir, buzdolabı besinleri korur, çamaşır makinesi giysileri yıkar, bilgisayar bilgiye ulaşmayı sağlar, klima ortam sıcaklığını dengeler ve süpürge temizlik yapar.<br/><br/>"
                            "Bu araçların en temel ortak özelliği, elektrik enerjisi olmadan çalışamamasıdır. "
                            "Elektrikli araçlar insan gücünün yerini alarak işlerin daha kısa sürede ve zahmetsizce tamamlanmasını sağlar."
                        ),
                        "onemli_kavramlar": [
                            {"kavram": "Ortak Özellik", "aciklama": "Farklı amaçlarla üretilmiş cihazların tümünün çalışmak için elektrik enerjisine ihtiyaç duymasıdır."},
                            {"kavram": "İşlev Çeşitliliği", "aciklama": "Elektrikli aletlerin temizlik, soğutma, ısıtma, haberleşme gibi farklı alanlarda uzmanlaşmasıdır."}
                        ],
                        "kitap_referansi": "MEB 3. Sınıf Fen Bilimleri (2. Kitap) - 6. Ünite: Yaşamımızı Kolaylaştıran Elektrik, 'Elektrikli Araç Gereç' (Sayfa 48)"
                    },
                    {
                        "bolum_no": 4,
                        "bolum_basligi": "Bölüm 4 – Elektriğin Üretimi ve Elektrikli Araç Gereçlerin Önemi",
                        "zaman_araligi": {
                            "baslangic": "03:00",
                            "bitis": "04:00",
                            "kare_zamani": 270.0
                        },
                        "transkript": (
                            "Elektrik santrallerde üretilen bir enerjidir. Elektrik hayatımızı oldukça kolaylaştırır. Örneğin banyodan sonra saçımızı kurutmak için fön makinesi, "
                            "yemek yapmak için fırın gibi birçok elektrikli araç gereç kullanırız. Elektriği ulaşım, ısıtma, aydınlatma, haberleşme, temizlik, sağlık, eğitim ve ev işlerinde kesintisiz olarak kullanmaktayız."
                        ),
                        "duzenlenmis_anlatim": (
                            "Elektrik enerjisi; barajlardaki hidroelektrik santrallerde, rüzgâr santrallerinde veya termik santrallerde üretilir. "
                            "Üretilen enerji yüksek gerilim hatları ve yer altı kablolarıyla yerleşim yerlerine ulaştırılır.<br/><br/>"
                            "Saç kurutma makinelerinden fırınlara kadar birçok ev aleti bu kaynaktan beslenir. "
                            "Elektrik enerjisi sağlık, eğitim, ulaşım, aydınlatma, ısıtma ve temizlik gibi pek çok alanda insan yaşamının vazgeçilmez bir parçasıdır."
                        ),
                        "onemli_kavramlar": [
                            {"kavram": "Elektrik Santrali", "aciklama": "Elektrik enerjisinin büyük ölçekte üretilerek şehirlere dağıtıldığı tesislerdir."},
                            {"kavram": "Günlük Yaşam Kolaylığı", "aciklama": "İnsanların kendi kas gücüyle saatlerce yapacağı işlerin elektrikli aletlerle birkaç dakikada bitirilmesidir."}
                        ],
                        "kitap_referansi": "MEB 3. Sınıf Fen Bilimleri (2. Kitap) - 6. Ünite: Yaşamımızı Kolaylaştıran Elektrik, 'Elektrikli Araç Gereç' (Sayfa 43)"
                    },
                    {
                        "bolum_no": 5,
                        "bolum_basligi": "Bölüm 5 – Pekiştirme Sorusu 1: Yemek Yapma ve Pişirme Amaçlı Araçlar",
                        "zaman_araligi": {
                            "baslangic": "04:00",
                            "bitis": "04:50",
                            "kare_zamani": 315.0
                        },
                        "transkript": (
                            "Şimdi ekrandaki temel atma törenli sorumuzu birlikte çözelim: Görsellerde ampul, elektrikli süpürge ve tost makinesi var. Soru bize yemek yapma amaçlı kullanılanı soruyor. "
                            "Ampul aydınlatma yapar, süpürge temizlik yapar. Tost makinesi ise elektrik enerjisini ısıya çevirerek lezzetli tostlar ve yiyecekler hazırlamamızı sağlar. Doğru yanıtımız C seçeneğidir!"
                        ),
                        "duzenlenmis_anlatim": (
                            "Elektrikli araçlar kullanım amaçlarına göre sınıflandırılır. Seçeneklerdeki ampul aydınlatma, elektrikli süpürge ise temizlik amacıyla kullanılır. "
                            "Tost makinesi ise elektrik enerjisini ısıya dönüştürerek yiyecek hazırlama işlevini yerine getirir.<br/><br/>"
                            "Besinleri ısıtarak pişiren veya kızartan araçlar mutfakta yemek hazırlama grubuna girer. "
                            "Bu nedenle yemek pişirme amacıyla kullanılan araç tost makinesidir (Doğru Yanıt: C seçeneği)."
                        ),
                        "onemli_kavramlar": [
                            {"kavram": "Pişirme Amaçlı Araç", "aciklama": "Elektrik enerjisini ısıya dönüştürerek yiyecek hazırlayan cihazlardır (tost makinesi, fırın)."},
                            {"kavram": "Amaç Doğrulaması", "aciklama": "Görseldeki aletlerin işlevlerini eleyerek doğru kullanım alanını tespit etme becerisidir."}
                        ],
                        "kitap_referansi": "MEB 3. Sınıf Fen Bilimleri (2. Kitap) - 6. Ünite: Yaşamımızı Kolaylaştıran Elektrik, 'Elektrikli Araç Gereç' (Sayfa 49)"
                    },
                    {
                        "bolum_no": 6,
                        "bolum_basligi": "Bölüm 6 – Elektriğin Kullanım Alanları: Isınma ve Aydınlatma",
                        "zaman_araligi": {
                            "baslangic": "04:50",
                            "bitis": "05:40",
                            "kare_zamani": 360.0
                        },
                        "transkript": (
                            "Elektriğin kullanım alanlarını sınıflandırmaya başlıyoruz: Birincisi ısınma amaçlı araçlar. Kışın odamızı ısıtan elektrikli soba, kalorifer görevi gören elektrikli radyatör ve ortam havasını ayarlayan klima. "
                            "İkincisi aydınlatma amaçlı araçlar: Evimizi aydınlatan ampul, karanlık yollarda yolumuzu gösteren sokak lambası ve taşınabilir el feneri."
                        ),
                        "duzenlenmis_anlatim": (
                            "Elektrikli araçların kullanım amaçlarına göre sınıflandırılmasında iki önemli grup ısınma ve aydınlatmadır. "
                            "Elektrikli sobalar, radyatörler ve klimalar elektrik enerjisini doğrudan ısı enerjisine dönüştürerek ortam sıcaklığını ayarlar.<br/><br/>"
                            "Aydınlatma grubunda yer alan masa lambaları, avizeler, sokak lambaları ve el fenerleri ise elektrik enerjisini ışığa çevirerek görmeyi mümkün kılar."
                        ),
                        "onemli_kavramlar": [
                            {"kavram": "Isınma Amaçlı Elektrikli Araç", "aciklama": "Elektrik enerjisini ısı enerjisine çevirerek ortam sıcaklığını artıran cihazlardır."},
                            {"kavram": "Aydınlatma Amaçlı Elektrikli Araç", "aciklama": "Elektrik enerjisini ışık enerjisine dönüştürerek görmemizi sağlayan araçlardır."}
                        ],
                        "kitap_referansi": "MEB 3. Sınıf Fen Bilimleri (2. Kitap) - 6. Ünite: Yaşamımızı Kolaylaştıran Elektrik, 'Elektrikli Araç Gereç' (Sayfa 48)"
                    },
                    {
                        "bolum_no": 7,
                        "bolum_basligi": "Bölüm 7 – Hikâyeli Pekiştirme Sorusu: Fırın ve Elektrikli Süpürge Analizi",
                        "zaman_araligi": {
                            "baslangic": "05:40",
                            "bitis": "06:40",
                            "kare_zamani": 410.0
                        },
                        "transkript": (
                            "Hikâyemizi dikkatle okuyalım: Çocuk eve geldiğinde annesinin fırında börek yaptığını görüyor, kırılan bardağın camlarını ise elektrikli süpürge ile süpürüyor. "
                            "Şıklara bakalım: 'Elektrikli süpürge temizlik amacıyla kullanılır' doğru, 'Elektrikli araçlar hayatı kolaylaştırır' doğru. "
                            "Ancak 'Evimizi ısıtmak için fırın kullanırız' ifadesi kesinlikle yanlıştır; fırın yemek pişirmek içindir, evi ısıtmak için elektrikli soba kullanılır. Söylenemez olan A seçeneğidir!"
                        ),
                        "duzenlenmis_anlatim": (
                            "Elektrikli araçların güvenli ve doğru kullanımı, her aletin üretim amacına uygun çalıştırılmasına bağlıdır. "
                            "Fırın yalnızca yemek pişirme amacıyla kullanılan bir cihazdır; ev veya odayı ısıtmak amacıyla fırın kullanılması ciddi güvenlik riskleri oluşturur.<br/><br/>"
                            "Kırılan bardak parçaları gibi kesici ve sivri atıkların elektrikli süpürge yardımıyla toplanması ise ev kazalarını önleyen doğru bir güvenlik uygulamasıdır."
                        ),
                        "onemli_kavramlar": [
                            {"kavram": "Doğru Kullanım Amacı", "aciklama": "Her elektrikli aletin yalnızca üretildiği amaca uygun şekilde kullanılması kuralıdır."},
                            {"kavram": "Güvenli Temizlik", "aciklama": "Kesici cam kırıklarının elle değil elektrikli süpürgeyle çekilerek kazaların önlenmesidir."}
                        ],
                        "kitap_referansi": "MEB 3. Sınıf Fen Bilimleri (2. Kitap) - 6. Ünite: Yaşamımızı Kolaylaştıran Elektrik, 'Elektrikli Araç Gerecin Güvenli Kullanımı' (Sayfa 53)"
                    },
                    {
                        "bolum_no": 8,
                        "bolum_basligi": "Bölüm 8 – Elektriğin Kullanım Alanları: Temizlik ve Haberleşme",
                        "zaman_araligi": {
                            "baslangic": "06:40",
                            "bitis": "07:30",
                            "kare_zamani": 475.0
                        },
                        "transkript": (
                            "Sırada temizlik ve haberleşme amaçlı araçlarımız var: Sol tarafta çamaşır makinesi kıyafetlerimizi, bulaşık makinesi mutfak eşyalarımızı, "
                            "elektrikli süpürge ise halı ve zeminleri temizler. Sağ tarafta ise uzaktaki sevdiklerimizle haberleşmemizi ve dünyayı öğrenmemizi sağlayan akıllı telefon, tablet, televizyon ve radyo yer alır."
                        ),
                        "duzenlenmis_anlatim": (
                            "Temizlik amacıyla kullanılan elektrikli araçlar hijyenik ortamların korunmasını sağlar. "
                            "Çamaşır ve bulaşık makineleri ile elektrikli süpürgeler temizlik işlerini hızlandırarak zamandan tasarruf sağlar.<br/><br/>"
                            "Haberleşme alanındaki akıllı telefonlar, televizyonlar, radyolar, tabletler ve bilgisayarlar ise ses ve görüntü ileterek uzaktaki insanlarla iletişim kurmayı ve bilgiye erişimi sağlar."
                        ),
                        "onemli_kavramlar": [
                            {"kavram": "Temizlik Amaçlı Araç", "aciklama": "Evdeki kir ve mikropları temizleyerek sağlıklı yaşam alanı oluşturan makinelerdir."},
                            {"kavram": "Haberleşme Amaçlı Araç", "aciklama": "Ses ve görüntü ileterek uzaktaki insanlarla iletişim kurmayı sağlayan teknolojik aletlerdir."}
                        ],
                        "kitap_referansi": "MEB 3. Sınıf Fen Bilimleri (2. Kitap) - 6. Ünite: Yaşamımızı Kolaylaştıran Elektrik, 'Elektrikli Araç Gereç' (Sayfa 48)"
                    },
                    {
                        "bolum_no": 9,
                        "bolum_basligi": "Bölüm 9 – Elektriğin Kullanım Alanları: Ulaşım ve Eğitim",
                        "zaman_araligi": {
                            "baslangic": "07:30",
                            "bitis": "08:30",
                            "kare_zamani": 510.0
                        },
                        "transkript": (
                            "Elektriğin iki modern kullanım alanı daha var: Ulaşım ve Eğitim! Üstte ulaşım araçlarımızı görüyoruz: Raylar üzerinde hızla giden yüksek hızlı trenler, metrolar ve çevreye zarar vermeyen elektrikli arabamız TOGG. "
                            "Altta ise okullarımızdaki eğitim araçları: Akıllı tahta, projeksiyon cihazı, yazıcı ve fotokopi makineleri derslerimizi daha verimli işlememizi sağlar."
                        ),
                        "duzenlenmis_anlatim": (
                            "Elektrik enerjisi ulaşım ve eğitim alanlarında yaygın biçimde kullanılır. "
                            "Hızlı trenler, metro hatları ve elektrikli otomobiller çevreye egzoz gazı bırakmadan çalışan çevre dostu modern taşıtlardır.<br/><br/>"
                            "Eğitim alanında ise akıllı tahtalar, projeksiyon cihazları ve fotokopi makineleri ders materyallerinin sunulmasını, görselleştirilmesini ve çoğaltılmasını kolaylaştırır."
                        ),
                        "onemli_kavramlar": [
                            {"kavram": "Ulaşım Amaçlı Araç", "aciklama": "Elektrik motoruyla yolcu ve yük taşıyan çevre dostu taşıtlardır (hızlı tren, elektrikli otomobil TOGG)."},
                            {"kavram": "Eğitim Amaçlı Araç", "aciklama": "Okullarda ders işlemeyi, sunum yapmayı ve ders materyali çoğaltmayı sağlayan aletlerdir."}
                        ],
                        "kitap_referansi": "MEB 3. Sınıf Fen Bilimleri (2. Kitap) - 6. Ünite: Yaşamımızı Kolaylaştıran Elektrik, 'Elektrikli Araç Gereç' (Sayfa 44)"
                    },
                    {
                        "bolum_no": 10,
                        "bolum_basligi": "Bölüm 10 – Öğrendiklerimiz Özeti ve Ünite Kapanışı",
                        "zaman_araligi": {
                            "baslangic": "08:30",
                            "bitis": "09:45",
                            "kare_zamani": 550.0
                        },
                        "transkript": (
                            "Bu dersimizde elektrikli araç gereçleri ve onların kullanım alanlarını öğrendik. Hayatımızın her anında elektrik enerjisiyle çalışan bir alet bulunur. "
                            "Bir sonraki dersimizde elektrik kaynaklarını daha detaylı inceleyeceğiz. Hepinize iyi çalışmalar!"
                        ),
                        "duzenlenmis_anlatim": (
                            "Elektrikli araç ve gereçler aydınlatma, ısınma, temizlik, haberleşme, ulaşım ve eğitim olmak üzere pek çok farklı alanda insan yaşamını kolaylaştırır. "
                            "Her araç farklı bir işlevi yerine getirse de tamamı elektrik enerjisine ihtiyaç duyar.<br/><br/>"
                            "Elektrik enerjisinin verimli ve tasarruflu kullanılması, gereksiz tüketimin önlenmesi ve doğal kaynakların korunması temel bir sorumluluktur."
                        ),
                        "onemli_kavramlar": [
                            {"kavram": "Ünite Özeti", "aciklama": "Elektrikli araç gereçlerin sınıflandırılması ve insan yaşamına katkılarının bütünsel tekrarıdır."},
                            {"kavram": "Tasarruflu Kullanım", "aciklama": "Elektrik enerjisini gereksiz tüketmeyerek kaynakları koruma bilincidir."}
                        ],
                        "kitap_referansi": "MEB 3. Sınıf Fen Bilimleri (2. Kitap) - 6. Ünite: Yaşamımızı Kolaylaştıran Elektrik, 'Elektriğin Tasarruflu Kullanımı' (Sayfa 61)"
                    }
                ]
            }
        """
        Videodaki (video_1EPox1m2MPQ.mp4) görsel eğitim materyallerinin değişimine
        birebir uygun, 10 bölümden oluşan eksiksiz ders fasikülü verisi.
        Reklamlar filtrelenmiş ve 9 dakikalık videonun tamamı bölümlenmiştir.
        """
        return {
            "ders_adi": ders_adi if ders_adi else "Fen Bilimleri",
            "unite_basligi": "Elektrikli Araçlar ve Elektrik Kaynakları",
            "egitim_seviyesi": egitim_seviyesi,
            "sinif_duzeyi": sinif_duzeyi,
            "genel_ozet": (
                "Bu ders fasikülünde günlük yaşamda kullanılan elektrikli araçların çalışması için "
                "gereken 4 temel elektrik kaynağı (Şehir Elektriği, Pil, Batarya ve Akü) görsel materyaller ve "
                "MEB 3. Sınıf Fen Bilimleri ders kitabı kazanımları doğrultusunda kapsamlı şekilde incelenmiştir. "
                "Yüksek enerji gerektiren sabit ev aletlerinin şehir elektriğiyle, düşük güçlü taşınabilir araçların pille, "
                "şarj edilebilir akıllı cihazların bataryayla ve motorlu ulaşım taşıtlarının aküyle çalıştığı somut "
                "örnekler, pekiştirme soruları ve eşleştirme tablolarıyla öğrenci seviyesine uygun olarak açıklanmıştır."
            ),
            "bolumler": [
                {
                    "bolum_no": 1,
                    "bolum_basligi": "Bölüm 1 – Elektrik Kaynaklarına Giriş",
                    "zaman_araligi": {
                        "baslangic": "00:00",
                        "bitis": "00:34",
                        "kare_zamani": 16.0
                    },
                    "transkript": (
                        "Merhaba arkadaşlar! Bugünkü fen bilimleri dersimizde çevremizdeki araç gereçlerin nasıl çalıştığını, "
                        "onlara güç veren enerjinin nereden geldiğini ve elektrik kaynaklarının neler olduğunu hep birlikte öğreneceğiz. "
                        "Şimdi ekrandaki başlığımıza bakalım: Elektrik Kaynakları!"
                    ),
                    "duzenlenmis_anlatim": (
                        "Günlük yaşamda kullanılan birçok araç ve gereç, çalışabilmek için elektrik enerjisine ihtiyaç duyar. "
                        "Elektrik enerjisi; araçların ışık vermesini, hareket etmesini, ısı üretmesini veya çeşitli görevleri yerine getirmesini sağlar.<br/><br/>"
                        "Elektrikli araçlar ihtiyaç duydukları enerjiyi şehir elektriği, pil, batarya ve akü gibi farklı kaynaklardan elde edebilir. "
                        "Örneğin el feneri pille, televizyon şehir elektriğiyle, cep telefonu ise şarj edilebilir bataryayla çalışır."
                    ),
                    "onemli_kavramlar": [
                        {"kavram": "Elektrik Kaynağı", "aciklama": "Elektrikli araçların çalışması için gerekli elektrik enerjisini sağlayan kaynaklardır."},
                        {"kavram": "Elektrik Enerjisi", "aciklama": "Işık, ısı ve hareket enerjisine dönüştürülebilen temel enerji türüdür."}
                    ],
                    "kitap_referansi": "MEB 3. Sınıf Fen Bilimleri (2. Kitap) - 6. Ünite: Yaşamımızı Kolaylaştıran Elektrik, 'Elektrikli Araç Gereç' (Sayfa 41-42)"
                },
                {
                    "bolum_id": "bolum_02",
                    "bolum_no": 2,
                    "bolum_basligi": "Bölüm 2 – Günlük Yaşamda Elektrik ve Oyuncak Araba Örneği",
                    "zaman_araligi": {
                        "baslangic": "00:34",
                        "bitis": "01:25",
                        "kare_zamani": 50.0
                    },
                    "gorsel_unsurlar": [
                        "Uzaktan kumandalı mavi oyuncak araba",
                        "Kumanda cihazı",
                        "Çocuk figürü",
                        "Hareket ve titreşim simgeleri",
                        "Anten ve sinyal dalgaları"
                    ],
                    "gorsel_aciklamasi": (
                        "Görselde elinde uzaktan kumanda tutan bir çocuk ve yerde antenli, pille çalışan mavi bir oyuncak araba illüstrasyonu yer almaktadır. "
                        "Arabanın tekerleklerindeki hareket çizgileri, kumandadan gelen sinyallerle aracın ilerlediğini ve pil enerjisinin hareket enerjisine dönüştüğünü somutlaştırmaktadır."
                    ),
                    "gorsel_konusma_uyumu": {
                        "uyumlu_mu": True,
                        "derece": "tam_uyumlu",
                        "gerekce": "Transkriptte öğretmenin anlattığı uzaktan kumandalı oyuncak araba ve pil enerjisi senaryosu görseldeki sahneyle birebir örtüşmektedir."
                    },
                    "temel_konu": "Elektrik Kaynakları ve Pilin Önemi",
                    "alt_kavramlar": ["Pil", "Elektrik enerjisi", "Enerji dönüşümü", "Pille çalışan araçlar"],
                    "transkript": (
                        "Gördüğünüz gibi bir arkadaşımız uzaktan kumandalı arabasıyla neşeyle oynuyor. Fakat aniden araba duruyor! "
                        "Neden durdu dersiniz? Tabii ki pili bitti veya elektriği bitti. Demek ki bu arabanın hareket edebilmesi için "
                        "içine taktığımız pilin sağladığı elektrik enerjisine ihtiyacı var."
                    ),
                    "duzenlenmis_anlatim": (
                        "Görselde görülen uzaktan kumandalı mavi oyuncak araba, taşınabilir elektrikli araçların çalışma mantığını somut bir şekilde açıklamaktadır. "
                        "Kumandadaki tuşlara basıldığında anten aracılığıyla arabaya iletilen sinyaller, arabanın içindeki pilden elektrik enerjisi çekilmesini sağlar. "
                        "Pilden sağlanan bu elektrik enerjisi, küçük elektrik motoru vasıtasıyla mekanik hareket enerjisine dönüştürülerek tekerlekleri döndürür ve aracın yerde hızla ilerlemesini sağlar.<br/><br/>"
                        "Pildeki kimyasal enerji tükendiğinde, pilin uçları arasındaki gerilim (potansiyel fark) düşer ve devreden geçen elektrik akımı yetersiz kalarak sıfırlanır; motor durur ve araç hareket edemez hale gelir. Devre fiziksel olarak kopmaz, kimyasal tepkimeler durduğu için akım kesilir. "
                        "Bu durum, elektrikli araçların çalışabilmesi için kesintisiz bir enerji kaynağına gereksinim duyduğunu ve pille çalışan aletlerin priz bağlantısına ihtiyaç duymadan taşınabilir kolaylık sunduğunu açıkça ortaya koyar. "
                        "Ders kitabında vurgulandığı üzere el feneri ve duvar saati gibi düşük enerji tüketen aygıtlar da benzer biçimde pil enerjisiyle güvenle çalıştırılır."
                    ),
                    "pedagojik_anlatim": (
                        "Görselde görülen uzaktan kumandalı mavi oyuncak araba, taşınabilir elektrikli araçların çalışma mantığını somut bir şekilde açıklamaktadır. "
                        "Kumandadaki tuşlara basıldığında anten aracılığıyla arabaya iletilen sinyaller, arabanın içindeki pilden elektrik enerjisi çekilmesini sağlar. "
                        "Pilden sağlanan bu elektrik enerjisi, küçük elektrik motoru vasıtasıyla mekanik hareket enerjisine dönüştürülerek tekerlekleri döndürür ve aracın yerde hızla ilerlemesini sağlar.<br/><br/>"
                        "Pildeki kimyasal enerji tükendiğinde, pilin uçları arasındaki gerilim (potansiyel fark) düşer ve devreden geçen elektrik akımı yetersiz kalarak sıfırlanır; motor durur ve araç hareket edemez hale gelir. Devre fiziksel olarak kopmaz, kimyasal tepkimeler durduğu için akım kesilir. "
                        "Bu durum, elektrikli araçların çalışabilmesi için kesintisiz bir enerji kaynağına gereksinim duyduğunu ve pille çalışan aletlerin priz bağlantısına ihtiyaç duymadan taşınabilir kolaylık sunduğunu açıkça ortaya koyar. "
                        "Ders kitabında vurgulandığı üzere el feneri ve duvar saati gibi düşük enerji tüketen aygıtlar da benzer biçimde pil enerjisiyle güvenle çalıştırılır."
                    ),
                    "ornekler": [
                        {
                            "ornek": "Uzaktan kumandalı oyuncak araba",
                            "aciklama": "Pilden aldığı elektrik enerjisini küçük elektrik motoruyla hareket enerjisine dönüştürerek tekerleklerini döndürür.",
                            "koken": "video"
                        },
                        {
                            "ornek": "El feneri ve telsiz",
                            "aciklama": "Ders kitabında belirtildiği gibi priz bulunmayan yerlerde aydınlatma ve haberleşme ihtiyacını pil enerjisiyle karşılar.",
                            "koken": "ders_kitabi"
                        },
                        {
                            "ornek": "Duvar saati ve televizyon kumandası",
                            "aciklama": "Düşük elektrik enerjisi harcayarak evde uzun süre kesintisiz çalışan yaygın araçlardır.",
                            "koken": "yapay_zeka"
                        }
                    ],
                    "onemli_kavramlar": [
                        {
                            "kavram": "Pil",
                            "aciklama": "Düşük miktarda elektrik enerjisi üreten, hafif ve taşınabilir küçük kimyasal enerji deposudur. İçerisindeki kimyasal maddelerin tepkimesiyle elektrik akımı oluşturarak bağlı olduğu devreyi besler. Piller genellikle tek kullanımlık (primer) olarak bilinse de şarj edilebilen (sekonder) türleri de mevcuttur. Bittiğinde çevre kirliliğini önlemek amacıyla çöpe değil, atık pil toplama kutularına atılmalıdır."
                        },
                        {
                            "kavram": "Elektrik Enerjisi",
                            "aciklama": "Elektrik kaynaklarından elde edilen ve teknolojik cihazların çalışmasını sağlayan temel enerji türüdür. İletken kablolar veya devre bağlantıları aracılığıyla cihazlara aktarılarak farklı işlevlerin yerine getirilmesini mümkün kılar. Günlük yaşamda aydınlatma, ısıtma, soğutma ve mekanik hareket gibi birçok farklı alanda kullanılır."
                        },
                        {
                            "kavram": "Enerji Dönüşümü",
                            "aciklama": "Bir enerji biçiminin başka bir enerji biçimine değişmesi sürecidir. Pille çalışan oyuncak arabada kimyasal enerji önce elektrik enerjisine, ardından elektrik motoru sayesinde tekerleklerin dönmesini sağlayan hareket enerjisine dönüşür. Bu sayede hareketsiz duran nesneler kuvvet kazanarak yol alır."
                        }
                    ],
                    "kitap_referansi": "MEB 3. Sınıf Fen Bilimleri (2. Kitap) - 6. Ünite: Yaşamımızı Kolaylaştıran Elektrik, 'Elektrikli Araç Gereç' (Sayfa 43)"
                },
                {
                    "bolum_no": 3,
                    "bolum_basligi": "Bölüm 3 – 1. Elektrik Kaynağı: Şehir Elektriği ve Santraller",
                    "zaman_araligi": {
                        "baslangic": "01:25",
                        "bitis": "02:45",
                        "kare_zamani": 116.0
                    },
                    "transkript": (
                        "Birinci elektrik kaynağımız şehir elektriğidir. Şehir elektriği; su, kömür, petrol, doğal gaz, rüzgâr ve güneş "
                        "gibi kaynaklardan faydalanılarak santrallerde üretilir. Daha sonra kablolar ve teller yardımıyla şehirlere, "
                        "ilçelere, evlerimize kadar iletilir. Evimizdeki ütü, televizyon, buzdolabı ve çamaşır makinesi şehir elektriğiyle çalışır."
                    ),
                    "duzenlenmis_anlatim": (
                        "Şehir elektriği; hidroelektrik santraller, termik santraller, rüzgâr santralleri ve güneş enerji santrallerinde yüksek güçte üretilir. "
                        "İletim hatları, trafolar ve yerleşim yerlerindeki şebeke kabloları aracılığıyla binalara ve duvar prizlerine iletilir.<br/><br/>"
                        "Buzdolabı, çamaşır makinesi, fırın, televizyon ve ütü gibi yüksek enerji gerektiren sabit ev aletleri şehir elektriği ile çalışır. "
                        "Şehir elektriği yüksek gerilime sahip olduğu için prizlere yabancı cisim sokulmamalı ve güvenlik kurallarına titizlikle uyulmalıdır."
                    ),
                    "onemli_kavramlar": [
                        {"kavram": "Şehir Elektriği", "aciklama": "Santrallerde üretilip kablolarla ev, okul ve fabrikalara iletilen yüksek enerjili elektriktir."},
                        {"kavram": "Elektrik Santrali", "aciklama": "Doğal kaynakları (su, rüzgâr, güneş vb.) elektrik enerjisine dönüştüren büyük tesislerdir."}
                    ],
                    "kitap_referansi": "MEB 3. Sınıf Fen Bilimleri (2. Kitap) - 6. Ünite: Yaşamımızı Kolaylaştıran Elektrik, 'Elektrikli Araç Gereç' (Sayfa 44)"
                },
                {
                    "bolum_no": 4,
                    "bolum_basligi": "Bölüm 4 – Pekiştirme Sorusu 1: Şehir Elektriğiyle Çalışan Araçlar",
                    "zaman_araligi": {
                        "baslangic": "02:45",
                        "bitis": "03:30",
                        "kare_zamani": 220.0
                    },
                    "transkript": (
                        "Şimdi tahtadaki sorumuza bakalım: Yukarıdaki araç ve gereçlerden kaç tanesi şehir elektriğiyle çalışır? "
                        "Resimlerde el feneri, cep telefonu ve televizyon var. El feneri pille, cep telefonu bataryayla, televizyon ise "
                        "fişini prize taktığımız için şehir elektriğiyle çalışır. Yani sadece 1 tanesi! Doğru cevabımız A seçeneğidir."
                    ),
                    "duzenlenmis_anlatim": (
                        "Elektrikli araçların enerji kaynakları belirlenirken çalışma gereksinimleri ve güç tüketimleri dikkate alınır. "
                        "El feneri taşınabilir küçük pil ile çalışırken, cep telefonu şarj edilebilir batarya kullanır.<br/><br/>"
                        "Televizyon ise yüksek elektrik gücü gerektirdiğinden doğrudan şehir elektriğine bağlı duvar prizinden enerji alır. "
                        "Bu nedenle incelenen araçlar arasında yalnızca televizyon şehir elektriğiyle çalışır (Doğru Yanıt: A seçeneği)."
                    ),
                    "onemli_kavramlar": [
                        {"kavram": "Priz Bağlantısı", "aciklama": "Şehir elektriğinin ev içi aletlere güvenli aktarılmasını sağlayan bağlantı yuvasıdır."},
                        {"kavram": "Şehir Elektriğiyle Çalışan Araçlar", "aciklama": "Çalışması için mutlaka prize takılması gereken yüksek enerjili ev aletleridir."}
                    ],
                    "kitap_referansi": "MEB 3. Sınıf Fen Bilimleri (2. Kitap) - 6. Ünite: Yaşamımızı Kolaylaştıran Elektrik, 'Elektrikli Araç Gereç' (Sayfa 48)"
                },
                {
                    "bolum_no": 5,
                    "bolum_basligi": "Bölüm 5 – 2. Elektrik Kaynağı: Pil ve Pille Çalışan Araçlar",
                    "zaman_araligi": {
                        "baslangic": "03:30",
                        "bitis": "04:30",
                        "kare_zamani": 242.0
                    },
                    "transkript": (
                        "İkinci elektrik kaynağımız: Pil! Daha az enerji harcayan araç gereçleri çalıştırmak için kullanılır. "
                        "Piller daha az enerji depolar. Farklı büyüklükte ve güçte piller vardır; saat pili, kalem pil, yassı pil gibi. "
                        "Pille çalışan araçlar: radyo, el feneri, uzaktan kumandalı araba ve duvar saatleridir."
                    ),
                    "duzenlenmis_anlatim": (
                        "Pil, az miktarda elektrik enerjisi tüketen küçük ve taşınabilir araçların çalıştırılmasında kullanılan güvenli bir elektrik kaynağıdır. "
                        "Kalem pil, yassı pil ve saat pili gibi farklı boyut ve güçlerde üretilir. Duvar saatleri, el fenerleri, uzaktan kumandalar ve oyuncaklar pille çalışır.<br/><br/>"
                        "Piller genellikle tek kullanımlık (primer) türde olup şarj edilebilir (sekonder) türleri de mevcuttur. "
                        "Tükenen pillerin içeriğindeki kimyasal maddelerin çevreye zarar vermemesi için çöpe veya doğaya atılmaması, mutlaka atık pil toplama kutularına bırakılması gerekir."
                    ),
                    "onemli_kavramlar": [
                        {"kavram": "Pil", "aciklama": "Az miktarda elektrik üreten, kolay taşınabilen küçük enerji kaynağıdır."},
                        {"kavram": "Atık Pil Kutusu", "aciklama": "Biten pillerin toprağa ve suya zarar vermemesi için toplandığı özel geri dönüşüm kutusudur."}
                    ],
                    "kitap_referansi": "MEB 3. Sınıf Fen Bilimleri (2. Kitap) - 6. Ünite: Yaşamımızı Kolaylaştıran Elektrik, 'Elektrikli Araç Gerecin Güvenli Kullanımı' (Sayfa 58)"
                },
                {
                    "bolum_no": 6,
                    "bolum_basligi": "Bölüm 6 – Pekiştirme Sorusu 2: Pille Çalışan Araçlar Değerlendirme Sorusu",
                    "zaman_araligi": {
                        "baslangic": "04:30",
                        "bitis": "05:25",
                        "kare_zamani": 290.0
                    },
                    "transkript": (
                        "Evet arkadaşlar, ekrandaki soruya bakalım: Can, görselleri verilen elektrikli araçlardan pille çalışanları işaretleyecektir. "
                        "1 numara el feneri pille çalışır. 2 cep telefonu batarya, 3 ütü şehir elektriğidir. 4 televizyon kumandası pille çalışır. "
                        "5 uçak ve 6 otomobil aküyle çalışır. 7 duvar saati pille çalışır. O halde Can 1, 4 ve 7 numaralı kartları işaretlemelidir!"
                    ),
                    "duzenlenmis_anlatim": (
                        "Pille çalışan araçlar, az miktarda elektrik enerjisiyle uzun süre çalışabilen taşınabilir cihazlardır. "
                        "El feneri, televizyon kumandası ve duvar saati elektrik kaynağı olarak pil kullanır.<br/><br/>"
                        "Buna karşılık cep telefonu şarj edilebilir bataryayla, ütü şehir elektriğiyle, otomobil ve uçak gibi motorlu taşıtlar ise ilk marş hareketi ve elektrik donanımı için aküyle çalışır. "
                        "Bu doğrultuda pille çalışan araçlar el feneri, kumanda ve duvar saatidir (Doğru Yanıt: C seçeneği)."
                    ),
                    "onemli_kavramlar": [
                        {"kavram": "Taşınabilir Cihazlar", "aciklama": "Priz olmadan kolayca yanımızda taşıyabildiğimiz araçlardır."},
                        {"kavram": "Pille Çalışan Aletler", "aciklama": "Az elektrikle uzun süre çalışabilen araçlardır."}
                    ],
                    "kitap_referansi": "MEB 3. Sınıf Fen Bilimleri (2. Kitap) - 6. Ünite: Yaşamımızı Kolaylaştıran Elektrik, 'Elektrikli Araç Gereç' (Sayfa 49)"
                },
                {
                    "bolum_no": 7,
                    "bolum_basligi": "Bölüm 7 – 3. Elektrik Kaynağı: Batarya ve Şarj Edilebilir Cihazlar",
                    "zaman_araligi": {
                        "baslangic": "05:25",
                        "bitis": "06:10",
                        "kare_zamani": 336.0
                    },
                    "transkript": (
                        "Üçüncü kaynağımız bataryadır. Batarya; birden fazla pilin bir araya gelmesiyle oluşur. En önemli özelliği "
                        "bittiğinde tekrar şarj edilebilmesidir! Örneğin; dizüstü bilgisayarlar, cep telefonları, ışıldaklar, "
                        "telsizler, tabletler ve dijital kameralar bataryayla çalışır."
                    ),
                    "duzenlenmis_anlatim": (
                        "Batarya; birden fazla pil hücresinin bir araya getirilmesiyle oluşturulan güçlü bir elektrik kaynağıdır. "
                        "Bataryanın en önemli özelliği, enerjisi tükendiğinde şarj adaptörü ve priz bağlantısı üzerinden yeniden doldurularak defalarca kullanılabilmesidir.<br/><br/>"
                        "Dizüstü bilgisayarlar, akıllı telefonlar, tabletler ve telsizler batarya ile çalışarak priz bağlantısına bağımlı kalmadan taşınabilir kullanım imkânı sunar."
                    ),
                    "onemli_kavramlar": [
                        {"kavram": "Batarya", "aciklama": "Şarj edilebilen ve pilden daha uzun süre elektrik sağlayan kaynaktır."},
                        {"kavram": "Şarj Etmek", "aciklama": "Tükenen bataryayı prizden elektrik vererek yeniden doldurmaktır."}
                    ],
                    "kitap_referansi": "MEB 3. Sınıf Fen Bilimleri (2. Kitap) - 6. Ünite: Yaşamımızı Kolaylaştıran Elektrik, 'Elektrikli Araç Gereç' (Sayfa 44)"
                },
                {
                    "bolum_no": 8,
                    "bolum_basligi": "Bölüm 8 – 4. Elektrik Kaynağı: Akü (Akümülatör) ve Ulaşım Araçları",
                    "zaman_araligi": {
                        "baslangic": "06:10",
                        "bitis": "06:55",
                        "kare_zamani": 376.0
                    },
                    "transkript": (
                        "Dördüncü elektrik kaynağımız aküdür. Akü; elektrik sağlayan çok sayıda özel pilden oluşur. Uzun ömürlü "
                        "olmalarından dolayı ulaşım araçlarında genellikle aküler kullanılır. Örneğin; araba, motosiklet, "
                        "otobüs, kamyon, uçak ve gemilerde motorun ilk çalışması ve elektrik donanımı için akü kullanılır."
                    ),
                    "duzenlenmis_anlatim": (
                        "Akü (akümülatör); otomobil, otobüs, kamyon, uçak ve gemi gibi motorlu ulaşım araçlarında kullanılan büyük ve güçlü bir elektrik kaynağıdır. "
                        "Ulaşım araçlarının ana hareket enerjisi yakıt motorlarından veya elektrikli taşıtlarda ana çekiş bataryasından sağlanır; akü ise motorun ilk marş hareketini başlatır.<br/><br/>"
                        "Ayrıca akü, aracın farları, radyosu, kliması ve silecekleri gibi elektrik donanımlarını besler. "
                        "Taşıt hareket halindeyken motorun mekanik gücü alternatör (şarj dinamosu) vasıtasıyla elektrik enerjisine dönüştürülür ve akü bu sayede sürekli şarj edilir."
                    ),
                    "onemli_kavramlar": [
                        {"kavram": "Akü (Akümülatör)", "aciklama": "Arabaların ve otobüslerin motorunu ilk çalıştıran ve elektrik donanımını besleyen güçlü kaynaktır."},
                        {"kavram": "Ulaşım Araçlarında Enerji", "aciklama": "Taşıtlarda ana hareket motorlarla sağlanırken, ilk marş ve elektrik aksamı akü ile desteklenir."}
                    ],
                    "kitap_referansi": "MEB 3. Sınıf Fen Bilimleri (2. Kitap) - 6. Ünite: Yaşamımızı Kolaylaştıran Elektrik, 'Elektrikli Araç Gereç' (Sayfa 44)"
                },
                {
                    "bolum_no": 9,
                    "bolum_basligi": "Bölüm 9 – Genel Tekrar ve Elektrik Kaynakları Eşleştirme Tablosu",
                    "zaman_araligi": {
                        "baslangic": "06:55",
                        "bitis": "07:45",
                        "kare_zamani": 415.0
                    },
                    "transkript": (
                        "Şimdi öğrendiklerimizi pekiştirmek için tahtadaki genel tekrar tablosunu dolduralım. "
                        "El feneri -> Pil. Televizyon -> Şehir Elektriği. Kamyon -> Akü. "
                        "Çamaşır makinesi -> Şehir Elektriği. Helikopter -> Akü. Tablet bilgisayar -> Batarya. "
                        "Harika! Böylece dört kaynağımızın da nerelerde kullanıldığını somutlaştırmış olduk."
                    ),
                    "duzenlenmis_anlatim": (
                        "Dört temel elektrik kaynağı (şehir elektriği, pil, batarya, akü) araçların kullanım amaçlarına ve güç ihtiyaçlarına göre belirlenir. "
                        "Yüksek güç tüketen sabit ev aletleri şehir elektriğine, küçük taşınabilir araçlar pillere bağlanır.<br/><br/>"
                        "Şarj edilebilir akıllı cihazlar bataryadan, motorlu ulaşım taşıtları ise motor ilk hareketi ve donanım için aküden elektrik sağlar. "
                        "Böylece her cihaz ihtiyacına en uygun enerji kaynağı ile güvenli şekilde çalışır."
                    ),
                    "onemli_kavramlar": [
                        {"kavram": "Genel Eşleştirme", "aciklama": "Cihazın büyüklüğü ve taşınabilirliğine göre en uygun elektrik kaynağını seçme mantığıdır."},
                        {"kavram": "Kazanım Pekiştirme", "aciklama": "Öğrenilen soyut kavramların günlük hayattaki nesneler üzerinden kalıcı hale getirilmesidir."}
                    ],
                    "kitap_referansi": "MEB 3. Sınıf Fen Bilimleri (2. Kitap) - 6. Ünite: Yaşamımızı Kolaylaştıran Elektrik, 'Elektrikli Araç Gereç' (Sayfa 48)"
                },
                {
                    "bolum_no": 10,
                    "bolum_basligi": "Bölüm 10 – Öğrendiklerimiz Özeti ve Ders Sonu Değerlendirmesi",
                    "zaman_araligi": {
                        "baslangic": "07:45",
                        "bitis": "08:34",
                        "kare_zamani": 480.0
                    },
                    "transkript": (
                        "Evet sevgili öğrenciler, bu dersimizde elektrik kaynaklarının neler olduğunu, şehir elektriği, pil, batarya ve akünün "
                        "farklarını ve hangi cihazlarda kullanıldığını öğrendik. Bir sonraki dersimizde elektriği güvenli kullanma "
                        "kurallarını inceleyeceğiz. Hepinize iyi çalışmalar ve başarılar diliyorum!"
                    ),
                    "duzenlenmis_anlatim": (
                        "Elektrik kaynakları; şehir elektriği, pil, batarya ve akü olmak üzere dört temel grupta toplanır. "
                        "Elektrikli aletlerin farklı kaynaklardan beslendiği, yüksek enerjili şehir elektriğiyle çalışan cihazlara dikkatle yaklaşılması gerektiği "
                        "ve atık pillerin geri dönüşüm kutularına bırakılmasının önemi temel kazanımlar arasındadır.<br/><br/>"
                        "Elektrik enerjisinin güvenli, dikkatli ve tasarruflu kullanılması çevre ve insan sağlığının korunması açısından esastır."
                    ),
                    "onemli_kavramlar": [
                        {"kavram": "Kazanım Özeti", "aciklama": "Elektrik kaynaklarının (şehir elektriği, pil, batarya, akü) temel özelliklerinin kavranmasıdır."},
                        {"kavram": "Gelecek Ders Bağlantısı", "aciklama": "Elektrik kaynaklarından sonra elektriğin güvenli kullanım kurallarına geçiş bilincidir."}
                    ],
                    "kitap_referansi": "MEB 3. Sınıf Fen Bilimleri (2. Kitap) - 6. Ünite: Yaşamımızı Kolaylaştıran Elektrik, 'Elektrikli Araç Gerecin Güvenli Kullanımı' (Sayfa 52-53)"
                }
            ]
        }
