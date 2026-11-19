import os
from pathlib import Path
from sysconfig import get_path

# CTranslate2, CUDA DLL'lerini çalışırken PATH üzerinden yükler.
cuda_paketleri = Path(get_path("purelib")) / "nvidia"
cuda_dizinleri = [
    str(cuda_dizini)
    for paket in ("cublas", "cudnn")
    if (cuda_dizini := cuda_paketleri / paket / "bin").is_dir()
]
if os.name == "nt" and cuda_dizinleri:
    os.environ["PATH"] = os.pathsep.join(
        [*cuda_dizinleri, os.environ.get("PATH", "")]
    )

from faster_whisper import WhisperModel

video_yolu = Path(r"D:\AIVideoLessonAnalyzer\temp_medya\video_1EPox1m2MPQ.mp4")
cikti_yolu = video_yolu.with_suffix(".transkript.txt")

model = WhisperModel("small", device="cuda", compute_type="float16")
parcalar, _ = model.transcribe(
    str(video_yolu),
    language="tr",
    vad_filter=True,
)

with cikti_yolu.open("w", encoding="utf-8") as dosya:
    for parca in parcalar:
        metin = parca.text.strip()
        if metin:
            dosya.write(f"[{parca.start:.2f} - {parca.end:.2f}] {metin}\n")

print(f"Transkript kaydedildi: {cikti_yolu}")
