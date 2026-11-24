"""Gemini isteklerini aralıklandırır ve sunucunun bekleme süresini uygular."""

import os
import time
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from threading import Lock


_kilit = Lock()
_son_istek = {}
_bekleme_sonu = {}


def gemini_istegini_beklet(model: str) -> None:
    rpm = float(os.getenv("GEMINI_REQUESTS_PER_MINUTE", "0"))
    aralik = 60 / rpm if rpm > 0 else 0
    while True:
        with _kilit:
            simdi = time.monotonic()
            sonraki = max(_son_istek.get(model, -aralik) + aralik, _bekleme_sonu.get(model, 0))
            if sonraki <= simdi:
                _son_istek[model] = simdi
                return
        time.sleep(sonraki - simdi)


def gemini_kota_hatasini_isle(model: str, yanit) -> dict:
    """Google'ın bildirdiği kota türünü ve varsa yeniden istek zamanını okur."""
    saniye = 0.0
    retry_after = yanit.headers.get("Retry-After")
    if retry_after:
        try:
            saniye = float(retry_after)
        except ValueError:
            try:
                tarih = parsedate_to_datetime(retry_after)
                saniye = (tarih - datetime.now(timezone.utc)).total_seconds()
            except (TypeError, ValueError, OverflowError):
                pass
    try:
        hata = yanit.json().get("error", {})
    except ValueError:
        hata = {}
    kotalar = []
    for ayrinti in hata.get("details") or []:
        tur = ayrinti.get("@type", "")
        if tur.endswith("google.rpc.RetryInfo"):
            gecikme = ayrinti.get("retryDelay", "")
            try:
                saniye = max(saniye, float(gecikme.removesuffix("s")))
            except ValueError:
                pass
        elif tur.endswith("google.rpc.QuotaFailure"):
            for ihlal in ayrinti.get("violations") or []:
                kota = ihlal.get("quotaId") or ihlal.get("quotaMetric")
                if kota:
                    kotalar.append(kota)
    if saniye > 0:
        with _kilit:
            _bekleme_sonu[model] = max(_bekleme_sonu.get(model, 0), time.monotonic() + saniye)
    return {"kotalar": kotalar, "bekleme_saniye": max(0, saniye)}
