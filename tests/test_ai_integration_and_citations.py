"""
EduFrame AI - Yapay Zeka Entegrasyonu, Hata Dayanıklılığı ve Dinamik RAG Kaynakça Testleri
(tests/test_ai_integration_and_citations.py)
-----------------------------------------------------------------------------------------
Bu test paketi 10 kritik senaryoyu doğrular:
1. Ollama Başarılı Yanıt Senaryosu
2. Ollama Zaman Aşımı -> Gemini Fallback
3. Gemini HTTP 503 Hatası -> Exponential Backoff & Model Fallback
4. İki Yapay Zeka Başarısız -> Pedagojik RAG Fallback
5. İki Yapay Zeka + RAG Yok -> Güvenlik Ağı Şablonu (static_fallback)
6. Dinamik Kaynakça Üretimi (Çoklu Benzersiz RAG Parçaları)
7. Dinamik Kaynakça Tekilleştirme (Deduplication - Bölüm & Soru Birleştirme)
8. Kaynak Bulunamadığında Fallback Mesajı
9. PDF Başlık ve Üst Bilgi Dosya Yolu Temizleme (Sanitization)
10. Cevap Anahtarı Açıklama/Kaynak Ayrımı ve Gerçek RAG Eşleştirmesi
"""

import os
import json
import unittest
from unittest.mock import patch, MagicMock

import requests

from services.gemini_analyzer import GeminiAnalyzer
from services.question_generator import QuestionGenerator
from services.pdf_generator import PDFGenerator


class TestAIIntegrationAndCitations(unittest.TestCase):
    """Yapay Zeka Dayanıklılığı ve Dinamik Kaynakça Testleri."""

    def setUp(self):
        self.analyzer = GeminiAnalyzer(api_key="AIzaTestKeyForMockingOnly123456789")
        self.qgen = QuestionGenerator(api_key="AIzaTestKeyForMockingOnly123456789")
        self.pdf_gen = PDFGenerator()

    # -----------------------------------------------------------------------
    # SENARYO 1: Ollama Başarılı Yanıt Senaryosu
    # -----------------------------------------------------------------------
    @patch("requests.post")
    def test_scenario_1_ollama_success(self, mock_post):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "response": '{"ders_adi": "Fen Bilimleri", "unite_basligi": "Elektrik", "bolumler": [{"bolum_no": 1, "bolum_basligi": "Giriş", "transkript": "Ders", "duzenlenmis_anlatim": "Konu", "onemli_kavramlar": []}]}'
        }
        mock_post.return_value = mock_resp

        sonuc = self.analyzer.analiz_et(
            video_yolu="test.mp4",
            egitim_seviyesi="İlkokul",
            sinif_duzeyi="3. Sınıf",
            ders_adi="Fen Bilimleri",
            rag_baglami="Referans metin",
            video_basligi="Elektrikli Araçlar"
        )

        self.assertEqual(sonuc.get("uretim_kaynagi"), "ollama")
        self.assertEqual(len(sonuc.get("bolumler", [])), 1)
        self.assertEqual(sonuc["bolumler"][0].get("kaynak_tipi"), "ollama")

    # -----------------------------------------------------------------------
    # SENARYO 2: Ollama Zaman Aşımı -> Gemini Fallback
    # -----------------------------------------------------------------------
    @patch("requests.post")
    def test_scenario_2_ollama_timeout_gemini_fallback(self, mock_post):
        def side_effect(url, **kwargs):
            if "localhost" in url or "11434" in url:
                raise requests.exceptions.Timeout("Ollama CPU inference timed out after 90s")
            # Gemini isteği
            m_resp = MagicMock()
            m_resp.status_code = 200
            m_resp.json.return_value = {
                "candidates": [{
                    "content": {
                        "parts": [{
                            "text": '{"ders_adi": "Fen Bilimleri", "unite_basligi": "Elektrik", "bolumler": [{"bolum_no": 1, "bolum_basligi": "Gemini Bölüm", "transkript": "Konuşma", "duzenlenmis_anlatim": "Gemini Anlatım", "onemli_kavramlar": []}]}'
                        }]
                    }
                }]
            }
            return m_resp

        mock_post.side_effect = side_effect

        sonuc = self.analyzer.analiz_et(
            video_yolu="test.mp4",
            egitim_seviyesi="İlkokul",
            sinif_duzeyi="3. Sınıf",
            ders_adi="Fen Bilimleri",
            rag_baglami="Referans RAG",
            video_basligi="Elektrik"
        )

        self.assertEqual(sonuc.get("uretim_kaynagi"), "gemini")
        self.assertEqual(sonuc["bolumler"][0].get("kaynak_tipi"), "gemini")
        self.assertEqual(sonuc["bolumler"][0]["bolum_basligi"], "Gemini Bölüm")

    # -----------------------------------------------------------------------
    # SENARYO 3: Gemini 503 Hatası -> Exponential Backoff & Model Fallback
    # -----------------------------------------------------------------------
    @patch("time.sleep", return_value=None)  # Testi hızlandırmak için sleep'i atla
    @patch("requests.post")
    def test_scenario_3_gemini_503_retry_and_model_fallback(self, mock_post, mock_sleep):
        attempt_counts = {"gemini-3.8-flash": 0, "gemini-2.5-flash": 0}

        def side_effect(url, **kwargs):
            if "gemini-3.8-flash" in url:
                attempt_counts["gemini-3.8-flash"] += 1
                resp = MagicMock()
                resp.status_code = 503
                resp.json.return_value = {
                    "error": {
                        "code": 503,
                        "message": "This model is currently experiencing high demand.",
                        "status": "UNAVAILABLE"
                    }
                }
                return resp
            elif "gemini-2.5-flash" in url:
                attempt_counts["gemini-2.5-flash"] += 1
                resp = MagicMock()
                resp.status_code = 200
                resp.json.return_value = {
                    "candidates": [{
                        "content": {
                            "parts": [{
                                "text": '{"bolumler": [{"bolum_no": 1, "bolum_basligi": "Yedek Modelden Başarılı", "transkript": "T", "duzenlenmis_anlatim": "D", "onemli_kavramlar": []}]}'
                            }]
                        }
                    }]
                }
                return resp
            raise ValueError(f"Beklenmeyen URL: {url}")

        mock_post.side_effect = side_effect

        analyzer = GeminiAnalyzer(
            api_key="AIzaValidKeyForTest1234567890",
            model="gemini-3.8-flash",
            gemini_max_retries=2,
            fallback_models=["gemini-2.5-flash"]
        )

        res = analyzer._gemini_ile_analiz_et("Test Prompt")

        self.assertIsNotNone(res)
        self.assertEqual(res["bolumler"][0]["bolum_basligi"], "Yedek Modelden Başarılı")
        # 3.8-flash modeli retries+1 defa (3 kez) 503 verdi, sonra 2.5-flash'a geçti
        self.assertEqual(attempt_counts["gemini-3.8-flash"], 3)
        self.assertEqual(attempt_counts["gemini-2.5-flash"], 1)

    # -----------------------------------------------------------------------
    # SENARYO 4: İki Yapay Zeka Başarısız -> Pedagojik RAG Fallback
    # -----------------------------------------------------------------------
    @patch("requests.post")
    def test_scenario_4_both_ai_fail_rag_fallback(self, mock_post):
        mock_post.side_effect = requests.exceptions.ConnectionError("Network down")

        sonuc = self.analyzer.analiz_et(
            video_yolu="test_video.mp4",
            egitim_seviyesi="İlkokul",
            sinif_duzeyi="3. Sınıf",
            ders_adi="Fen Bilimleri",
            rag_baglami="MEB 3. Sınıf Fen Bilimleri 6. Ünite Elektrikli Araçlar...",
            video_basligi="Elektrikli Araç ve Gereçler"
        )

        self.assertEqual(sonuc.get("uretim_kaynagi"), "rag_fallback")
        self.assertTrue(len(sonuc.get("bolumler", [])) >= 10)
        self.assertEqual(sonuc["bolumler"][0].get("kaynak_tipi"), "rag_fallback")

    # -----------------------------------------------------------------------
    # SENARYO 5: İki Yapay Zeka + RAG Yok -> Güvenlik Ağı Şablonu (static_fallback)
    # -----------------------------------------------------------------------
    @patch("requests.post")
    def test_scenario_5_both_ai_and_rag_fail_static_fallback(self, mock_post):
        mock_post.side_effect = requests.exceptions.ConnectionError("Network down")

        sonuc = self.analyzer.analiz_et(
            video_yolu="test_video.mp4",
            egitim_seviyesi="İlkokul",
            sinif_duzeyi="3. Sınıf",
            ders_adi="Fen Bilimleri",
            rag_baglami="",  # RAG bağlamı yok
            video_basligi="Elektrikli Araç ve Gereçler"
        )

        self.assertEqual(sonuc.get("uretim_kaynagi"), "static_fallback")
        self.assertEqual(sonuc["bolumler"][0].get("kaynak_tipi"), "static_fallback")

    # -----------------------------------------------------------------------
    # SENARYO 6: Dinamik Kaynakça Üretimi (Çoklu Benzersiz RAG Parçaları)
    # -----------------------------------------------------------------------
    def test_scenario_6_dynamic_bibliography_multiple_distinct_rag_chunks(self):
        video_analizi = {
            "sinif_duzeyi": "3. Sınıf",
            "bolumler": [
                {
                    "bolum_no": 1,
                    "bolum_basligi": "Bölüm 1 - Elektrikli Araçlar",
                    "rag_kaynaklari": [
                        {
                            "kitap_adi": "MEB 3. Sınıf Fen Bilimleri (2. Kitap)",
                            "dosya_adi": "ilkokul_fenbilimleri_3.sinif_2.kitap.pdf",
                            "sinif": "3. Sınıf",
                            "unite": "6. Ünite",
                            "konu": "Elektrikli Araç Gereç",
                            "physical_page": 41,
                            "printed_page": 41,
                            "has_printed_page": True
                        }
                    ]
                },
                {
                    "bolum_no": 6,
                    "bolum_basligi": "Bölüm 6 - Aydınlatma Araçları",
                    "rag_kaynaklari": [
                        {
                            "kitap_adi": "MEB 3. Sınıf Fen Bilimleri (2. Kitap)",
                            "dosya_adi": "ilkokul_fenbilimleri_3.sinif_2.kitap.pdf",
                            "sinif": "3. Sınıf",
                            "unite": "6. Ünite",
                            "konu": "Aydınlatma ve Isınma",
                            "physical_page": 48,
                            "printed_page": 48,
                            "has_printed_page": True
                        }
                    ]
                }
            ]
        }
        sorular = [
            {
                "no": 1,
                "rag_kaynagi": {
                    "kitap_adi": "MEB 3. Sınıf Fen Bilimleri (2. Kitap)",
                    "dosya_adi": "ilkokul_fenbilimleri_3.sinif_2.kitap.pdf",
                    "sinif": "3. Sınıf",
                    "unite": "6. Ünite",
                    "konu": "Elektrikli Araç Gereç",
                    "physical_page": 41,
                    "printed_page": 41,
                    "has_printed_page": True
                }
            }
        ]

        kaynaklar = self.pdf_gen._dinamik_kaynakca_derle(video_analizi, sorular)

        self.assertEqual(len(kaynaklar), 2)
        # 1. parça (Sayfa 41) hem Bölüm 1 hem Soru 1'de kullanıldı
        sayfa_41_entry = [k for k in kaynaklar if "41" in k][0]
        self.assertIn("MEB 3. Sınıf Fen Bilimleri (2. Kitap)", sayfa_41_entry)
        self.assertIn("ilkokul_fenbilimleri_3.sinif_2.kitap.pdf", sayfa_41_entry)
        self.assertIn("Fiziksel Sayfa: 41", sayfa_41_entry)
        self.assertIn("Basılı Sayfa: 41", sayfa_41_entry)
        self.assertIn("Bölüm 1", sayfa_41_entry)
        self.assertIn("Soru 1", sayfa_41_entry)

        # 2. parça (Sayfa 48) Bölüm 6'da kullanıldı
        sayfa_48_entry = [k for k in kaynaklar if "48" in k][0]
        self.assertIn("Fiziksel Sayfa: 48", sayfa_48_entry)
        self.assertIn("Bölüm 6", sayfa_48_entry)

    # -----------------------------------------------------------------------
    # SENARYO 7: Dinamik Kaynakça Tekilleştirme (Deduplication)
    # -----------------------------------------------------------------------
    def test_scenario_7_dynamic_bibliography_deduplication(self):
        aynı_kaynak = {
            "kitap_adi": "MEB 3. Sınıf Fen Bilimleri (2. Kitap)",
            "dosya_adi": "ilkokul_fenbilimleri_3.sinif_2.kitap.pdf",
            "sinif": "3. Sınıf",
            "unite": "6. Ünite",
            "konu": "Elektrikli Araç Gereç",
            "physical_page": 41,
            "printed_page": 41,
            "has_printed_page": True
        }
        video_analizi = {
            "bolumler": [
                {"bolum_no": 1, "rag_kaynaklari": [aynı_kaynak]},
                {"bolum_no": 2, "rag_kaynaklari": [aynı_kaynak]},
                {"bolum_no": 5, "rag_kaynaklari": [aynı_kaynak]}
            ]
        }
        sorular = [
            {"no": 1, "rag_kaynagi": aynı_kaynak},
            {"no": 4, "rag_kaynagi": aynı_kaynak}
        ]

        kaynaklar = self.pdf_gen._dinamik_kaynakca_derle(video_analizi, sorular)

        # 3 bölüm ve 2 soru aynı kaynağı referans verdi, tek bir kayıt olmalı
        self.assertEqual(len(kaynaklar), 1)
        entry = kaynaklar[0]
        self.assertIn("Bölüm 1", entry)
        self.assertIn("Bölüm 2", entry)
        self.assertIn("Bölüm 5", entry)
        self.assertIn("Soru 1", entry)
        self.assertIn("Soru 4", entry)

    # -----------------------------------------------------------------------
    # SENARYO 8: Kaynak Bulunamadığında Fallback Mesajı
    # -----------------------------------------------------------------------
    def test_scenario_8_fallback_message_when_rag_missing(self):
        video_analizi = {"bolumler": []}
        sorular = []

        kaynaklar = self.pdf_gen._dinamik_kaynakca_derle(video_analizi, sorular)

        self.assertEqual(len(kaynaklar), 1)
        self.assertIn("Bu bölüm için doğrulanmış ders kitabı kaynağı bulunamadı", kaynaklar[0])

    # -----------------------------------------------------------------------
    # SENARYO 9: PDF Başlık ve Üst Bilgi Dosya Yolu Temizleme (Sanitization)
    # -----------------------------------------------------------------------
    def test_scenario_9_pdf_header_title_path_sanitization(self):
        yol_1 = r"C:\Users\nurmu\OneDrive\Desktop\AIVideoLessonAnalyzer\temp_medya\videos\elektrikli_araclar_dersi.mp4"
        yol_2 = "C:/Users/test/workspace/videos/fen_3_sinif_video.mkv"
        sade_baslik = "3. Sınıf Fen Bilimleri Elektrik Ünitesi"

        self.assertEqual(self.pdf_gen._temizle_dosya_yolu(yol_1), "elektrikli araclar dersi")
        self.assertEqual(self.pdf_gen._temizle_dosya_yolu(yol_2), "fen 3 sinif video")
        self.assertEqual(self.pdf_gen._temizle_dosya_yolu(sade_baslik), sade_baslik)

    # -----------------------------------------------------------------------
    # SENARYO 10: Cevap Anahtarı Açıklama/Kaynak Ayrımı ve Gerçek RAG Eşleştirmesi
    # -----------------------------------------------------------------------
    def test_scenario_10_answer_key_source_separation_and_binding(self):
        # 1. Gerçek RAG referansı bulunan bölüm
        video_analizi = {
            "unite_basligi": "Elektrikli Araç ve Gereçler",
            "bolumler": [
                {
                    "bolum_no": 1,
                    "bolum_basligi": "Bölüm 1 - Elektrikli Araç ve Gereçler",
                    "onemli_kavramlar": [{"kavram": "Elektrikli Araç ve Gereçler"}],
                    "kitap_referansi": "MEB 3. Sınıf Fen Bilimleri (2. Kitap) - 6. Ünite (Sayfa 41-42)"
                }
            ]
        }

        # Soru üret
        sorular = self.qgen.soru_uret(video_analizi, egitim_seviyesi="İlkokul", soru_sayisi=1)
        q = sorular[0]

        # Açıklama salt gerekçe içermeli, uydurma 's. 178' gibi referans içermemeli
        self.assertNotIn("s. 178", q["aciklama"])
        self.assertNotIn("Sayfa 178", q["aciklama"])

        # Referans gerçek RAG kaynağından bağlanmış olmalı
        self.assertEqual(q["kazanim_referansi"], "MEB 3. Sınıf Fen Bilimleri (2. Kitap) - 6. Ünite (Sayfa 41-42)")
        self.assertTrue(q.get("rag_dogrulandi"))

        # 2. RAG eşleşmesi olmayan boş analiz senaryosu
        sorular_bos = self.qgen.soru_uret({}, egitim_seviyesi="İlkokul", soru_sayisi=1)
        q_bos = sorular_bos[0]

        # Asla sayfa uydurmamalı, doğrulanmadığını belirtmeli
        self.assertEqual(q_bos["kazanim_referansi"], "Doğrulanmış ders kitabı kaynağı bulunamadı")
        self.assertFalse(q_bos.get("rag_dogrulandi"))

    # -----------------------------------------------------------------------
    # SENARYO 11: Ollama Payloadında Üst Düzey think=False ve num_predict Doğrulaması
    # -----------------------------------------------------------------------
    @patch("requests.post")
    def test_scenario_11_ollama_payload_top_level_think_and_num_predict(self, mock_post):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "response": '{"bolumler": [{"bolum_no": 1, "bolum_basligi": "B", "duzenlenmis_anlatim": "A", "onemli_kavramlar": []}]}'
        }
        mock_post.return_value = mock_resp

        analyzer = GeminiAnalyzer(api_key="AIzaTestKeyForMockingOnly123456789")

        # 1. Analiz çağrısı payload testi
        analyzer._ollama_ile_analiz_et("Analiz İstem")
        self.assertTrue(mock_post.called)
        sent_payload = mock_post.call_args[1]["json"]

        # think=False üst düzeyde olmalı
        self.assertIn("think", sent_payload)
        self.assertFalse(sent_payload["think"])
        # options içinde think olmamalı
        self.assertNotIn("think", sent_payload.get("options", {}))
        # format=json ve stream=False olmalı
        self.assertEqual(sent_payload["format"], "json")
        self.assertFalse(sent_payload["stream"])
        # num_predict analiz için uygun değerde olmalı (3500)
        self.assertEqual(sent_payload["options"]["num_predict"], 3500)

        # 2. Bölüm pedagojik sentez payload testi
        mock_resp.json.return_value = {
            "response": '{"duzenlenmis_anlatim": "Pedagojik metin...", "onemli_kavramlar": []}'
        }
        analyzer._ollama_tekil_sentez("Bölüm İstem")
        sent_payload_section = mock_post.call_args[1]["json"]
        self.assertIn("think", sent_payload_section)
        self.assertFalse(sent_payload_section["think"])
        self.assertNotIn("think", sent_payload_section.get("options", {}))
        self.assertEqual(sent_payload_section["options"]["num_predict"], 800)

        # 3. Soru üretimi payload testi
        mock_resp.json.return_value = {
            "response": '{"sorular": [{"no": 1, "soru": "S1", "secenekler": ["A"], "dogru_cevap": "A", "aciklama": "Aciklama", "kategori": "K", "ilgili_kavram": "K", "tip": "Coktan Secmeli"}] * 5}'
        }
        # Geçerli 5 soru içeren JSON
        valid_q_json = json.dumps({
            "sorular": [
                {"no": i, "soru": f"S{i}", "secenekler": ["A", "B"], "dogru_cevap": "A", "aciklama": "Gerekce", "kategori": "💡 Kavram", "ilgili_kavram": "Elektrik", "tip": "Çoktan Seçmeli"}
                for i in range(1, 6)
            ]
        })
        mock_resp.json.return_value = {"response": valid_q_json}
        qgen = QuestionGenerator(api_key="AIzaTestKeyForMockingOnly123456789")
        qgen._ollama_ile_soru_uret("Soru İstem")
        sent_payload_q = mock_post.call_args[1]["json"]
        self.assertIn("think", sent_payload_q)
        self.assertFalse(sent_payload_q["think"])
        self.assertNotIn("think", sent_payload_q.get("options", {}))
        self.assertEqual(sent_payload_q["options"]["num_predict"], qgen.ollama_question_num_predict)

    # -----------------------------------------------------------------------
    # SENARYO 12: Ollama done_reason=length ve Boş Yanıt Yönetimi
    # -----------------------------------------------------------------------
    @patch("requests.post")
    def test_scenario_12_ollama_done_reason_length_and_empty_response(self, mock_post):
        analyzer = GeminiAnalyzer(api_key="AIzaTestKeyForMockingOnly123456789")

        # 1. done_reason="length" ama geçerli JSON parçası kurtarılamıyor
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "done_reason": "length",
            "response": '{"bolumler": [{"bolum_no": 1, ' # Kesilmiş JSON
        }
        mock_post.return_value = mock_resp
        res = analyzer._ollama_ile_analiz_et("Test Prompt")
        self.assertIsNone(res)

        # 2. Tamamen boş response alanı
        mock_resp.json.return_value = {
            "done_reason": "stop",
            "response": "   "
        }
        res_empty = analyzer._ollama_ile_analiz_et("Test Prompt")
        self.assertIsNone(res_empty)

        # 3. Bölüm sentezinde done_reason="length" ve kesilmiş metin
        res_sec = analyzer._ollama_tekil_sentez("Bölüm Prompt")
        self.assertIsNone(res_sec)

        # 4. Soru üretiminde boş response
        qgen = QuestionGenerator(api_key="AIzaTestKeyForMockingOnly123456789")
        res_q = qgen._ollama_ile_soru_uret("Soru Prompt")
        self.assertIsNone(res_q)

    # -----------------------------------------------------------------------
    # SENARYO 13: Ollama JSON Ayrıştırma ve HTTP Hataları Yönetimi
    # -----------------------------------------------------------------------
    @patch("requests.post")
    def test_scenario_13_ollama_json_parse_and_http_errors(self, mock_post):
        analyzer = GeminiAnalyzer(api_key="AIzaTestKeyForMockingOnly123456789")

        # 1. HTTP 500 İç Sunucu Hatası
        mock_resp = MagicMock()
        mock_resp.status_code = 500
        mock_resp.json.return_value = {"error": "Internal server error"}
        mock_post.return_value = mock_resp
        res_500 = analyzer._ollama_ile_analiz_et("Test Prompt")
        self.assertIsNone(res_500)
        self.assertTrue(analyzer._ollama_devre_disi)

        # 2. HTTP 200 ancak metin JSON değil
        analyzer_2 = GeminiAnalyzer(api_key="AIzaTestKeyForMockingOnly123456789")
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "done_reason": "stop",
            "response": "Bu bir model çıktısıdır ama JSON değildir."
        }
        res_invalid_json = analyzer_2._ollama_ile_analiz_et("Test Prompt")
        self.assertIsNone(res_invalid_json)

    # -----------------------------------------------------------------------
    # SENARYO 14: Tekrarlayan Zaman Aşımını Önleme (Circuit Breaker)
    # -----------------------------------------------------------------------
    @patch("requests.post")
    def test_scenario_14_circuit_breaker_prevents_cascading_delays(self, mock_post):
        mock_post.side_effect = requests.exceptions.Timeout("Ollama connection timed out")

        analyzer = GeminiAnalyzer(api_key="AIzaTestKeyForMockingOnly123456789")
        self.assertFalse(analyzer._ollama_devre_disi)

        # İlk bölüm sentezi zaman aşımına uğrar ve devreyi açar (circuit breaker)
        res1 = analyzer._ollama_tekil_sentez("Bölüm 1")
        self.assertIsNone(res1)
        self.assertTrue(analyzer._ollama_devre_disi)
        self.assertEqual(mock_post.call_count, 1)

        # İkinci bölüm sentezi istek göndermeden derhal None döner
        res2 = analyzer._ollama_tekil_sentez("Bölüm 2")
        self.assertIsNone(res2)
        # mock_post çağrı sayısı artmamalı
        self.assertEqual(mock_post.call_count, 1)

        # Analiz de devre açıkken istek göndermez
        res_analiz = analyzer._ollama_ile_analiz_et("Analiz")
        self.assertIsNone(res_analiz)
        self.assertEqual(mock_post.call_count, 1)
        res_sec = analyzer._ollama_tekil_sentez("Bölüm 1")
        self.assertIsNone(res_sec)
        self.assertEqual(mock_post.call_count, 1)

    # -----------------------------------------------------------------------
    # SENARYO 15: Bölüm Sentezinde İlk Bölüm Hata Verince Kalan Bölümlerde Ollama Atlanması
    # -----------------------------------------------------------------------
    @patch("requests.post")
    def test_scenario_15_section_circuit_breaker_in_bolum_sentezle(self, mock_post):
        # 1. bölümde timeout ver
        mock_post.side_effect = requests.exceptions.Timeout("Section 1 timeout")

        analyzer = GeminiAnalyzer(api_key="")  # Gemini yok, RAG fallback bekliyoruz

        bolum_1 = {
            "bolum_no": 1,
            "bolum_basligi": "Bölüm 1",
            "transkript": "Transkript 1",
            "duzenlenmis_anlatim": "Eski Anlatım",
            "onemli_kavramlar": []
        }
        rag_sonuc = {
            "bilgi": {"baglam": "MEB Bilgi"},
            "pedagoji": {"baglam": "MEB Pedagoji"}
        }

        # Bölüm 1 sentezle -> Ollama timeout yaşar, _ollama_devre_disi = True olur
        res_b1 = analyzer.bolum_pedagojik_sentezle(bolum_1, rag_sonuc)
        self.assertEqual(res_b1["kaynak_tipi"], "rag_fallback")
        self.assertTrue(analyzer._ollama_devre_disi)
        self.assertEqual(mock_post.call_count, 1)

        # Bölüm 2 sentezle -> Ollama istek göndermeden anında atlanır
        bolum_2 = {
            "bolum_no": 2,
            "bolum_basligi": "Bölüm 2",
            "transkript": "Transkript 2",
            "duzenlenmis_anlatim": "Eski Anlatım 2",
            "onemli_kavramlar": []
        }
        res_b2 = analyzer.bolum_pedagojik_sentezle(bolum_2, rag_sonuc)
        self.assertEqual(res_b2["kaynak_tipi"], "rag_fallback")
        # mock_post çağrı sayısı hala 1 olmalı! (Bölüm 2 için hiç istek yapılmadı)
        self.assertEqual(mock_post.call_count, 1)

    # -----------------------------------------------------------------------
    # SENARYO 16: Pedagojik Normalizasyon Konuşma Kalıplarını ve Yönergeleri Temizler
    # -----------------------------------------------------------------------
    def test_scenario_16_pedagogical_normalization_filters_conversational_phrases(self):
        yasakli_girdiler = [
            "Bugünkü dersimizde birlikte keşfedeceğiz.",
            "Önce şu örnekle başlayalım.",
            "Sınıf arkadaşlarımızla tartışalım.",
            "Merak ettiğimiz soruları aşağıdaki alana yazalım.",
            "Kemerlerinizi bağlayın, fen bilimleri uçağımızla yeni bilgiler öğrenmeye doğru yola çıkıyoruz!",
            "Bir sonraki dersimizde elektrik kaynaklarını daha detaylı inceleyeceğiz.",
            "Hepinize iyi çalışmalar ve başarılar diliyorum!"
        ]

        for y in yasakli_girdiler:
            temiz = GeminiAnalyzer.pedagojik_anlatim_normallestir(y)
            # Tek başına verilen yasaklı cümle tamamen elenmelidir
            self.assertEqual(temiz, "", f"Yasaklı cümle elenmedi: {y}")

        # Karışık metin testi (Yasaklı cümleler + ders kitabı metni)
        karisik_metin = (
            "Bugünkü dersimizde birlikte keşfedeceğiz. "
            "Günlük yaşamda kullanılan birçok araç ve gereç, çalışabilmek için elektrik enerjisine ihtiyaç duyar. "
            "Önce şu örnekle başlayalım. "
            "Elektrik enerjisi; araçların ışık vermesini, hareket etmesini, ısı üretmesini veya çeşitli görevleri yerine getirmesini sağlar.\n\n"
            "Sınıf arkadaşlarımızla tartışalım. "
            "Elektrikli araçlar ihtiyaç duydukları enerjiyi şehir elektriği, pil, batarya ve akü gibi farklı kaynaklardan elde edebilir. "
            "Merak ettiğimiz soruları aşağıdaki alana yazalım. "
            "Örneğin el feneri pille, televizyon şehir elektriğiyle, cep telefonu ise şarj edilebilir bataryayla çalışır."
        )

        beklenen = (
            "Günlük yaşamda kullanılan birçok araç ve gereç, çalışabilmek için elektrik enerjisine ihtiyaç duyar. "
            "Elektrik enerjisi; araçların ışık vermesini, hareket etmesini, ısı üretmesini veya çeşitli görevleri yerine getirmesini sağlar.\n\n"
            "Elektrikli araçlar ihtiyaç duydukları enerjiyi şehir elektriği, pil, batarya ve akü gibi farklı kaynaklardan elde edebilir. "
            "Örneğin el feneri pille, televizyon şehir elektriğiyle, cep telefonu ise şarj edilebilir bataryayla çalışır."
        )

        sonuc = GeminiAnalyzer.pedagojik_anlatim_normallestir(karisik_metin)
        self.assertEqual(sonuc, beklenen)

        # Kullanıcının beklediği saf ders kitabı metni normalizasyondan %100 aynen çıkmalıdır
        self.assertEqual(GeminiAnalyzer.pedagojik_anlatim_normallestir(beklenen), beklenen)

    # -----------------------------------------------------------------------
    # SENARYO 17: Simülasyon Şablonları Tamamen Nesnel ve Ders Kitabı Üslubundadır
    # -----------------------------------------------------------------------
    def test_scenario_17_simulation_templates_are_objective_and_pure(self):
        analyzer = GeminiAnalyzer(api_key="")

        yasakli_sozcukler = [
            "keşfedeceğiz",
            "öğreneceğiz",
            "tartışalım",
            "başlayalım",
            "öğrendik",
            "kemerlerinizi",
            "dersimize hoş geldiniz",
            "arkadaşlarımızla"
        ]

        # Şablon 1: Araç ve Gereçler
        sim1 = analyzer._orijinal_materyal_simulasyonu("İlkokul", "3. Sınıf", "Fen Bilimleri", "Elektrikli Araç ve Gereçler")
        self.assertEqual(len(sim1["bolumler"]), 10)
        for b in sim1["bolumler"]:
            anlatim = b.get("duzenlenmis_anlatim", "").lower()
            for yasak in yasakli_sozcukler:
                self.assertNotIn(yasak, anlatim, f"Şablon 1 Bölüm {b['bolum_no']} içinde '{yasak}' bulundu!")

        # Şablon 2: Elektrik Kaynakları (Analiz videosu)
        sim2 = analyzer._orijinal_materyal_simulasyonu("İlkokul", "3. Sınıf", "Fen Bilimleri", "Elektrik Kaynakları")
        self.assertEqual(len(sim2["bolumler"]), 10)
        for b in sim2["bolumler"]:
            anlatim = b.get("duzenlenmis_anlatim", "").lower()
            for yasak in yasakli_sozcukler:
                self.assertNotIn(yasak, anlatim, f"Şablon 2 Bölüm {b['bolum_no']} içinde '{yasak}' bulundu!")

        # Bölüm 1 tam metin doğrulaması (kullanıcının beklediği örnek metin)
        b1_anlatim = sim2["bolumler"][0]["duzenlenmis_anlatim"]
        self.assertIn("Günlük yaşamda kullanılan birçok araç ve gereç, çalışabilmek için elektrik enerjisine ihtiyaç duyar.", b1_anlatim)
        self.assertIn("Örneğin el feneri pille, televizyon şehir elektriğiyle, cep telefonu ise şarj edilebilir bataryayla çalışır.", b1_anlatim)

    # -----------------------------------------------------------------------
    # SENARYO 18: Yapay Zeka Sentez Çıktılarının Otomatik Normalleştirilmesi
    # -----------------------------------------------------------------------
    @patch("requests.post")
    def test_scenario_18_ollama_and_gemini_synthesis_outputs_normalized(self, mock_post):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "response": json.dumps({
                "duzenlenmis_anlatim": (
                    "Bugünkü dersimizde birlikte keşfedeceğiz. "
                    "Pil, az miktarda elektrik enerjisi harcayan küçük araçları çalıştırmak için kullanılır. "
                    "Sınıf arkadaşlarımızla tartışalım."
                ),
                "onemli_kavramlar": [{"kavram": "Pil", "aciklama": "Elektrik kaynağı."}]
            })
        }
        mock_post.return_value = mock_resp

        analyzer = GeminiAnalyzer(api_key="")
        bolum = {
            "bolum_no": 4,
            "bolum_basligi": "Bölüm 4 – Pil",
            "transkript": "Pil az enerji depolar.",
            "duzenlenmis_anlatim": "Eski metin"
        }
        rag_sonuc = {
            "bilgi": {"baglam": "MEB Pil"},
            "pedagoji": {"baglam": "MEB Pedagoji"}
        }

        sonuc = analyzer.bolum_pedagojik_sentezle(bolum, rag_sonuc)
        self.assertEqual(sonuc["kaynak_tipi"], "ollama")

        # Öğretmen konuşması cümleleri atılmış, sadece nesnel ders kitabı cümlesi kalmış olmalıdır
        self.assertNotIn("Bugünkü dersimizde birlikte keşfedeceğiz.", sonuc["duzenlenmis_anlatim"])
        self.assertNotIn("Sınıf arkadaşlarımızla tartışalım.", sonuc["duzenlenmis_anlatim"])
        self.assertIn("Pil, az miktarda elektrik enerjisi harcayan küçük araçları çalıştırmak için kullanılır.", sonuc["duzenlenmis_anlatim"])

    # -----------------------------------------------------------------------
    # SENARYO 21: Çok Yönlü RAG Arama ve Uygunluk Filtresi (Alakasız Parça Eleme)
    # -----------------------------------------------------------------------
    def test_scenario_21_multifaceted_rag_and_relevance_filtering(self):
        from langchain_core.documents import Document
        from services.rag_motoru import RAGMotoru

        rag = RAGMotoru()

        # 1. Alakalı parça (Pil ve pille çalışan aletler)
        doc_alakali = Document(
            page_content="El feneri, uzaktan kumanda ve saat gibi az elektrik tüketen araçlar pil ile çalışır. Biten piller atık pil kutusuna atılmalıdır.",
            metadata={"source": "fen_3_kitap.pdf", "page": 47, "unite": "6. Ünite", "konu": "Elektrik Kaynakları"}
        )
        puan_alakali = rag.chunk_alaka_puanla(
            doc=doc_alakali,
            temel_konu="Elektrik Kaynakları ve Pilin Önemi",
            alt_kavramlar=["Pil", "Elektrik enerjisi", "Pille çalışan araçlar"],
            gorsel_unsurlar=["Uzaktan kumandalı araba", "Kumanda"],
            transkript="Pille çalışan uzaktan kumandalı araba örneği."
        )
        self.assertTrue(puan_alakali["gecerli"])
        self.assertGreater(puan_alakali["puan"], 2.0)
        self.assertIn("Kabul edildi", puan_alakali["gerekce"])

        # 2. Alakasız parça (Baraj ve hidroelektrik santraller)
        doc_alakasiz = Document(
            page_content="Hidroelektrik santrallerde baraj göllerindeki suyun gücü türbinleri döndürür. Üretilen yüksek gerilim iletim hatları ile şehirlere taşınır.",
            metadata={"source": "fen_3_kitap.pdf", "page": 57, "unite": "6. Ünite", "konu": "Şehir Elektriği"}
        )
        puan_alakasiz = rag.chunk_alaka_puanla(
            doc=doc_alakasiz,
            temel_konu="Elektrik Kaynakları ve Pilin Önemi",
            alt_kavramlar=["Pil", "Elektrik enerjisi", "Pille çalışan araçlar"],
            gorsel_unsurlar=["Uzaktan kumandalı araba", "Kumanda"],
            transkript="Pille çalışan uzaktan kumandalı araba örneği."
        )
        self.assertFalse(puan_alakasiz["gecerli"])
        self.assertIn("Elendi", puan_alakasiz["gerekce"])

    # -----------------------------------------------------------------------
    # SENARYO 22: Video ve Ders Kitabı Örneklerinin Harmanlanması ve Kavram Derinliği
    # -----------------------------------------------------------------------
    @patch("requests.post")
    def test_scenario_22_example_harmonization_and_expanded_concept_definitions(self, mock_post):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "response": json.dumps({
                "duzenlenmis_anlatim": (
                    "Görseldeki uzaktan kumandalı oyuncak araba, pilin sağladığı elektrik enerjisini küçük bir motor yardımıyla "
                    "hareket enerjisine dönüştürerek tekerlekleri döndürür. Pil kimyasal deposunu tükettiğinde elektrik akımı kesilir ve araç durur."
                ),
                "ornekler": [
                    {"ornek": "Uzaktan kumandalı araba", "aciklama": "Pil enerjisiyle tekerleklerini döndürür.", "koken": "video"},
                    {"ornek": "El feneri", "aciklama": "Kitapta belirtilen taşınabilir aydınlatma aracıdır.", "koken": "ders_kitabi"},
                    {"ornek": "Duvar saati", "aciklama": "Düşük enerjiyle aylarca çalışan ev aletidir.", "koken": "yapay_zeka"}
                ],
                "onemli_kavramlar": [
                    {
                        "kavram": "Pil",
                        "aciklama": "Düşük elektrik enerjisi sağlayan taşınabilir kimyasal kaynaktır. Devreye elektrik akımı sağlayarak küçük araçları çalıştırır. Bittiğinde çevre kirliliğini önlemek için atık pil kutusuna atılmalıdır."
                    }
                ]
            })
        }
        mock_post.return_value = mock_resp

        analyzer = GeminiAnalyzer(api_key="")
        bolum = {
            "bolum_no": 2,
            "bolum_basligi": "Bölüm 2 – Günlük Yaşamda Elektrik ve Oyuncak Araba Örneği",
            "transkript": "Gördüğünüz gibi arkadaşımız pille çalışan uzaktan kumandalı arabayla oynuyor.",
            "gorsel_unsurlar": ["Uzaktan kumandalı araba", "Kumanda"],
            "gorsel_aciklamasi": "Pille çalışan oyuncak araba sahnesi.",
            "temel_konu": "Elektrik Kaynakları ve Pilin Önemi",
            "alt_kavramlar": ["Pil", "Elektrik enerjisi"]
        }
        rag_sonuc = {
            "bilgi": {"baglam": "MEB Pil konusu"},
            "pedagoji": {"baglam": "MEB 3. sınıf kazanımları"}
        }

        sentez = analyzer.bolum_pedagojik_sentezle(bolum, rag_sonuc)

        # 1. Harmanlanmış örnekler köken kontrolü
        self.assertIn("ornekler", sentez)
        kokeler = {o.get("koken") for o in sentez["ornekler"]}
        self.assertIn("video", kokeler)
        self.assertIn("ders_kitabi", kokeler)
        self.assertIn("yapay_zeka", kokeler)

        # 2. Önemli kavram derinlik kontrolü (2-3 cümle / zengin açıklama)
        self.assertIn("onemli_kavramlar", sentez)
        pil_kavrami = next((k for k in sentez["onemli_kavramlar"] if k.get("kavram") == "Pil"), None)
        self.assertIsNotNone(pil_kavrami)
        self.assertGreater(len(pil_kavrami["aciklama"]), 60)
        self.assertIn("atık pil", pil_kavrami["aciklama"].lower())

        # 3. Ortak veri yapısı ve üslup kontrolü
        self.assertEqual(sentez["pedagojik_anlatim"], sentez["duzenlenmis_anlatim"])
        self.assertNotIn("arkadaşlarımız", sentez["pedagojik_anlatim"])



if __name__ == "__main__":
    unittest.main()
