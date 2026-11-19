"""
EduFrame AI - Video Analiz Halüsinasyon Engelleme ve Konu Uygunluk Testleri
(tests/test_video_hallucination_prevention.py)
------------------------------------------------------------------------------
Bu test paketi, video analiz hattında yapay zeka modellerinin (Ollama / Gemini)
görsel veya işitsel verisi olmayan videolara dair konu dışı halüsinasyonlar
(örn. Elektrik Kaynakları videosu için Hareket, Toprak ve Mıknatıs uydurması)
üretmesini engelleyen doğrulama mantığını ve müfredat koruma mekanizmalarını test eder.
"""

import unittest
from unittest.mock import patch, MagicMock

from services.gemini_analyzer import GeminiAnalyzer


class TestVideoHallucinationPrevention(unittest.TestCase):
    """Konu Uyuşmazlığı ve Halüsinasyon Engelleme Testleri."""

    def setUp(self):
        self.analyzer = GeminiAnalyzer(api_key="AIzaTestKeyForMockingOnly123456789")

    def test_topic_alignment_rejects_unrelated_hallucination(self):
        """
        Video başlığı 'Elektrik Kaynakları' iken dönen 'Varlıkların Hareketleri ve Toprak Özellikleri'
        çıktısı konu uyuşmazlığı nedeniyle kesinlikle reddedilmelidir.
        """
        halusinasyon_verisi = {
            "ders_adi": "Fen Bilimleri",
            "unite_basligi": "Varlıkların Hareketleri ve Toprak Özellikleri",
            "bolumler": [
                {
                    "bolum_no": 1,
                    "bolum_basligi": "Hareket Eden Varlıkların Tanımlanması",
                    "transkript": "Karekod üzerinden videoya erişelim. Hareket eden varlıklara dikkat edelim.",
                    "duzenlenmis_anlatim": "Varlıklar hareket eder."
                },
                {
                    "bolum_no": 2,
                    "bolum_basligi": "Toprağın İçinde Neler Vardır?",
                    "transkript": "Aşağıdaki toprak görsellerini inceleyelim.",
                    "duzenlenmis_anlatim": "Toprak kayaçlardan oluşur."
                },
                {
                    "bolum_no": 3,
                    "bolum_basligi": "Mıknatıs Kutupları ve Özellikleri",
                    "transkript": "Mıknatısı ve mıknatıs kutuplarını tanıyalım.",
                    "duzenlenmis_anlatim": "Mıknatıslar çeker."
                }
            ]
        }

        gecerli_mi, neden = self.analyzer._analiz_sonucunu_dogrula(
            veri=halusinasyon_verisi,
            video_basligi="3. Sınıf Fen Bilimleri - Elektrik Kaynakları"
        )

        self.assertFalse(gecerli_mi, "Konu dışı halüsinasyon doğrulanamaz olarak işaretlenmelidir.")
        self.assertIn("konu uyuşmazlığı", neden.lower())
        self.assertIn("varlıkların hareketleri ve toprak özellikleri", neden.lower())

    def test_topic_alignment_accepts_valid_topic(self):
        """
        Video başlığı ile uyuşan geçerli bir analiz çıktısı onaylanmalıdır.
        """
        gecerli_veri = {
            "ders_adi": "Fen Bilimleri",
            "unite_basligi": "6. Ünite: Yaşamımızı Kolaylaştıran Elektrik",
            "bolumler": [
                {
                    "bolum_no": 1,
                    "bolum_basligi": "Elektrik Kaynaklarına Giriş",
                    "transkript": "Elektrik enerjisi ve elektrik kaynaklarını öğreneceğiz.",
                    "duzenlenmis_anlatim": "Elektrikli araçlar elektrik enerjisiyle çalışır."
                }
            ]
        }

        gecerli_mi, neden = self.analyzer._analiz_sonucunu_dogrula(
            veri=gecerli_veri,
            video_basligi="3. Sınıf Fen Bilimleri - Elektrik Kaynakları"
        )

        self.assertTrue(gecerli_mi)
        self.assertEqual(neden, "Doğrulandı")

    @patch.object(GeminiAnalyzer, "_ollama_ile_analiz_et")
    def test_analiz_et_rejects_hallucinated_ollama_and_falls_back(self, mock_ollama):
        """
        Ollama konu dışı halüsinasyon döndürdüğünde, analiz_et bunu reddedip güvenli şablona düşmelidir.
        """
        mock_ollama.return_value = {
            "ders_adi": "Fen Bilimleri",
            "unite_basligi": "Varlıkların Hareketleri ve Toprak Özellikleri",
            "bolumler": [
                {
                    "bolum_no": 1,
                    "bolum_basligi": "Hareket Eden Varlıklar",
                    "transkript": "Hareket",
                    "duzenlenmis_anlatim": "Hareket"
                }
            ]
        }

        # Bilinmeyen bir video yolu (curated olmayan) ile çağır
        sonuc = self.analyzer.analiz_et(
            video_yolu="bilinmeyen_video_123.mp4",
            egitim_seviyesi="İlkokul",
            sinif_duzeyi="3. Sınıf",
            ders_adi="Fen Bilimleri",
            rag_baglami="",
            video_basligi="Elektrik Enerjisi ve Kaynakları"
        )

        # Halüsinasyon reddedilip güvenlik ağına (static_fallback / simulation) düşmeli
        self.assertNotEqual(sonuc.get("uretim_kaynagi"), "ollama")
        self.assertNotEqual(sonuc.get("unite_basligi"), "Varlıkların Hareketleri ve Toprak Özellikleri")

    def test_curated_video_1epox1m2mpq_uses_authentic_10_sections(self):
        """
        1EPox1m2MPQ videosu analiz edildiğinde her zaman 10 bölümlük gerçek elektrik kaynakları yapısı korunmalıdır.
        """
        sonuc = self.analyzer.analiz_et(
            video_yolu="temp_medya/video_1EPox1m2MPQ.mp4",
            egitim_seviyesi="İlkokul",
            sinif_duzeyi="3. Sınıf",
            ders_adi="Fen Bilimleri",
            rag_baglami="Elektrikli Araçlar",
            video_basligi="3. Sınıf Fen Bilimleri - Elektrik Kaynakları"
        )

        self.assertEqual(sonuc.get("uretim_kaynagi"), "curated_ground_truth")
        self.assertEqual(len(sonuc.get("bolumler", [])), 10)
        self.assertEqual(sonuc.get("unite_basligi"), "Elektrikli Araçlar ve Elektrik Kaynakları")

        # Bölüm 1: Elektrik Kaynaklarına Giriş
        self.assertEqual(sonuc["bolumler"][0]["bolum_basligi"], "Bölüm 1 – Elektrik Kaynaklarına Giriş")
        # Bölüm 2: Günlük Yaşamda Elektrik ve Oyuncak Araba Örneği
        self.assertEqual(sonuc["bolumler"][1]["bolum_basligi"], "Bölüm 2 – Günlük Yaşamda Elektrik ve Oyuncak Araba Örneği")


if __name__ == "__main__":
    unittest.main()
