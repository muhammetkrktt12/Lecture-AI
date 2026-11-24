"""Streamlit'ten bağımsız çalıştırılan ve iptal edilebilen PDF işi."""

import json
import os
import signal
import subprocess
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Lock

from config import OUTPUTS_DIR, ROOT, TEMP_DIR


_YAYIM_KILIDI = Lock()


class PDFIsi:
    def __init__(self, ayarlar: dict, *, tekrar: bool = False):
        self.ayarlar = ayarlar.copy()
        self.etiket = "Ollama ile yeniden deneniyor" if tekrar else "PDF üretimi başlatılıyor"
        self._gecici = TemporaryDirectory(prefix="pdf_isi_", dir=TEMP_DIR, ignore_cleanup_errors=True)
        self.dizin = Path(self._gecici.name).resolve()
        self._son_asama = ""
        try:
            (self.dizin / "ayarlar.json").write_text(json.dumps(self.ayarlar), encoding="utf-8")
            secenekler = (
                {"creationflags": subprocess.CREATE_NO_WINDOW}
                if os.name == "nt" else {"start_new_session": True}
            )
            self._surec = subprocess.Popen(
                [sys.executable, "-m", "services.pdf_job", str(self.dizin)],
                cwd=ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, **secenekler,
            )
        except OSError:
            self._gecici.cleanup()
            raise

    def son_asama(self) -> str:
        try:
            satirlar = (self.dizin / "asamalar.txt").read_text(encoding="utf-8").splitlines()
        except OSError:
            return self._son_asama
        if satirlar:
            self._son_asama = satirlar[-1]
        return self._son_asama

    def iptal_et(self) -> None:
        try:
            if self._surec.poll() is None:
                if os.name == "nt":
                    sonuc = subprocess.run(
                        ["taskkill.exe", "/PID", str(self._surec.pid), "/T", "/F"], capture_output=True,
                        creationflags=subprocess.CREATE_NO_WINDOW, timeout=10,
                    )
                    if sonuc.returncode and self._surec.poll() is None:
                        raise OSError("PDF üretim işlemi durdurulamadı.")
                else:
                    try:
                        os.killpg(self._surec.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                self._surec.wait(timeout=5)
        except subprocess.TimeoutExpired as hata:
            raise OSError("İptal işlemi zaman aşımına uğradı.") from hata
        self._gecici.cleanup()

    def sonucu_al(self) -> dict | None:
        if self._surec.poll() is None:
            return None
        try:
            cevap = json.loads((self.dizin / "sonuc.json").read_text(encoding="utf-8"))
            if cevap["durum"] == "hazir":
                sonuc = cevap["sonuc"]
                pdf = Path(sonuc["cikti_pdf_yolu"])
                icerik = Path(sonuc["icerik_kaydi_yolu"]) if sonuc.get("icerik_kaydi_yolu") else None
                if pdf.resolve().parent != self.dizin or (icerik and icerik.resolve().parent != self.dizin):
                    raise OSError("PDF çıktısı geçici iş dizininde bulunamadı.")
                with _YAYIM_KILIDI:
                    hedef = OUTPUTS_DIR / pdf.name
                    if hedef.exists() or hedef.with_suffix(".icerik.json").exists():
                        hedef = hedef.with_name(f"{hedef.stem}_{self.dizin.name[-8:]}{hedef.suffix}")
                    if icerik:
                        icerik_hedefi = hedef.with_suffix(".icerik.json")
                        icerik.replace(icerik_hedefi)
                        sonuc["icerik_kaydi_yolu"] = str(icerik_hedefi)
                    try:
                        pdf.replace(hedef)
                    except OSError:
                        if icerik:
                            icerik_hedefi.unlink(missing_ok=True)
                        raise
                sonuc["cikti_pdf_yolu"] = str(hedef)
            return cevap
        except (OSError, ValueError, KeyError):
            return {"durum": "hata", "hata": "PDF üretim işlemi sonuç dosyasını tamamlayamadı."}
        finally:
            self._gecici.cleanup()


def _calistir(dizin: Path) -> None:
    import requests

    from config import OLLAMA_URL
    from core.eduframe_agentic_core import EduFrameAgenticCore, OllamaFasikulHatasi

    def ilerleme(dugum: str) -> None:
        with (dizin / "asamalar.txt").open("a", encoding="utf-8") as dosya:
            dosya.write(dugum + "\n")

    try:
        ayarlar = json.loads((dizin / "ayarlar.json").read_text(encoding="utf-8"))
        if ayarlar["anlatim_modeli"] == "ollama":
            try:
                yanit = requests.get(f"{OLLAMA_URL.rstrip('/')}/api/tags", timeout=4)
                yanit.raise_for_status()
                modeller = {m.get("name") for m in yanit.json().get("models", [])}
            except (requests.RequestException, ValueError) as hata:
                raise OllamaFasikulHatasi("Ollama sunucusuna bağlanılamadı.") from hata
            if ayarlar["ollama_modeli"] not in modeller:
                raise OllamaFasikulHatasi(f"{ayarlar['ollama_modeli']} modeli Ollama’da bulunamadı.")
        sonuc = EduFrameAgenticCore().calistir(
            kullanici_girisi=f"{ayarlar['sinif_duzeyi']} ders notunu çıkar",
            ollama_yedek_onayi_gerekli=True, ilerleme=ilerleme, pdf_cikti_dizini=str(dizin), **ayarlar,
        )
        if not sonuc.get("cikti_pdf_yolu") or not Path(sonuc["cikti_pdf_yolu"]).is_file():
            raise OSError(sonuc.get("hata") or "PDF dosyası oluşturulamadı.")
        cevap = {
            "durum": "hazir", "sonuc": {
                anahtar: sonuc.get(anahtar) for anahtar in
                ("cikti_pdf_yolu", "icerik_kaydi_yolu", "video_analizi", "sorular")
            },
        }
    except OllamaFasikulHatasi as hata:
        cevap = {"durum": "ollama_hatasi", "hata": str(hata)}
    except Exception as hata:
        cevap = {"durum": "hata", "hata": str(hata)}
    (dizin / "sonuc.json").write_text(json.dumps(cevap), encoding="utf-8")


if __name__ == "__main__":
    _calistir(Path(sys.argv[1]))
