"""Gemini yanıtlarının token kullanımını aşama bazında kaydeder."""

import json
from collections import Counter, defaultdict
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4


_calisma = ContextVar("gemini_kullanim_calismasi", default=None)
_kayit_dizini = Path(__file__).resolve().parents[1] / "temp_medya" / "gemini_kullanim"
_alanlar = {
    "giris": "promptTokenCount",
    "gorunur_cikti": "candidatesTokenCount",
    "dusunme": "thoughtsTokenCount",
    "onbellek_girisi": "cachedContentTokenCount",
}


@contextmanager
def gemini_kullanim_oturumu():
    kayitlar = []
    yol = _kayit_dizini / f"{uuid4().hex}.jsonl"
    onceki = _calisma.set((yol, kayitlar))
    try:
        yield
    finally:
        _calisma.reset(onceki)
        toplamlar = defaultdict(Counter)
        for kayit in kayitlar:
            toplam = toplamlar[kayit["model"], kayit["asama"]]
            toplam["yanit"] += 1
            if kayit["token"] is None:
                toplam["olculemeyen"] += 1
            else:
                toplam.update(kayit["token"])
        for (model, asama), toplam in toplamlar.items():
            print(
                f"[Gemini toplam] {model} | {asama} | {toplam['yanit']} deneme | "
                f"giriş={toplam['giris']} görünür çıktı={toplam['gorunur_cikti']} "
                f"düşünme={toplam['dusunme']} ölçülemeyen={toplam['olculemeyen']}"
            )
        if kayitlar and yol.is_file():
            print(f"[Gemini kullanım kaydı] {yol}")


def gemini_kullanimini_kaydet(model, asama, yanit=None, *, durum="200", kota=None):
    yanit = yanit or {}
    metadata = yanit.get("usageMetadata")
    token = (
        {alan: metadata.get(api_alani, 0) for alan, api_alani in _alanlar.items()}
        if isinstance(metadata, dict) and metadata else None
    )
    adaylar = yanit.get("candidates") or []
    kayit = {
        "zaman": datetime.now(timezone.utc).isoformat(),
        "model": model,
        "model_surumu": yanit.get("modelVersion"),
        "asama": asama,
        "durum": durum,
        "bitis_nedeni": adaylar[0].get("finishReason") if adaylar else None,
        "token": token,
    }
    if kota is not None:
        kayit["kota"] = kota
    degerler = " ".join(f"{alan}={deger}" for alan, deger in token.items()) if token else "token verisi yok"
    print(f"[Gemini kullanım] {model} | {asama} | {durum} | {degerler}")
    calisma = _calisma.get()
    if calisma is None:
        yol = _kayit_dizini / "bagimsiz_cagrilar.jsonl"
    else:
        yol, kayitlar = calisma
        kayitlar.append(kayit)
    try:
        yol.parent.mkdir(parents=True, exist_ok=True)
        with yol.open("a", encoding="utf-8") as dosya:
            dosya.write(json.dumps(kayit, ensure_ascii=False) + "\n")
    except OSError as hata:
        print(f"[Gemini kullanım] Kayıt yazılamadı: {hata}")
