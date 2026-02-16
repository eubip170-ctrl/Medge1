# infra/news_client.py
# Client NewsAPI (top-headlines + fallback everything) + utilità orari.

from __future__ import annotations

import datetime as dt
import os
from typing import List, Tuple

from zoneinfo import ZoneInfo  # Python 3.9+
import requests


# Puoi cambiare questa default key o usare l'env NEWSAPI_KEY
DEFAULT_NEWSAPI_KEY = "574232afaa764d4fa5dc47d82a7ef4e9"

DEFAULT_SOURCES_IT = [
    "ansa",
    "il-sole-24-ore",
    "la-repubblica",
    "google-news-it",
    "football-italia",
]


def _headers(api_key: str) -> dict:
    return {"X-Api-Key": (api_key or "").strip()}


def fetch_top_headlines(
    api_key: str,
    page_size: int = 12,
    sources: List[str] | None = None,
) -> dict:
    url = "https://newsapi.org/v2/top-headlines"
    params = {"pageSize": int(page_size)}
    if sources:
        params["sources"] = ",".join(sources)
    r = requests.get(url, params=params, headers=_headers(api_key), timeout=10)
    try:
        r.raise_for_status()
        return r.json()
    except Exception:
        return {"status": "error", "error": {"code": r.status_code, "message": r.text}}


def fetch_everything_recent(
    api_key: str,
    page_size: int = 12,
    sources: List[str] | None = None,
    lookback_hours: int = 24,
) -> dict:
    url = "https://newsapi.org/v2/everything"
    since = (
        dt.datetime.now(dt.timezone.utc)
        - dt.timedelta(hours=int(lookback_hours))
    ).isoformat(timespec="seconds").replace("+00:00", "Z")

    params = {
        "from": since,
        "sortBy": "publishedAt",
        "pageSize": int(page_size),
    }
    if sources:
        params["sources"] = ",".join(sources)

    r = requests.get(url, params=params, headers=_headers(api_key), timeout=10)
    try:
        r.raise_for_status()
        return r.json()
    except Exception:
        return {"status": "error", "error": {"code": r.status_code, "message": r.text}}


def dedup_by_title(articles: List[dict]) -> List[dict]:
    seen, out = set(), []
    for a in articles or []:
        t = (a.get("title") or "").strip()
        k = t.lower()
        if t and k not in seen:
            seen.add(k)
            out.append(a)
    return out


def get_headlines_it(
    api_key: str | None = None,
    page_size: int = 12,
    lookback_hours: int = 24,
    sources: List[str] | None = None,
) -> Tuple[List[dict], bool]:
    """
    Ritorna (lista_articoli, used_fallback).
    - prova prima /top-headlines
    - se vuoto, usa /everything sulle ultime lookback_hours
    """
    key = (api_key or os.getenv("NEWSAPI_KEY") or DEFAULT_NEWSAPI_KEY).strip()
    if not key:
        raise RuntimeError(
            "NEWSAPI key non configurata. Imposta env NEWSAPI_KEY "
            "oppure modifica DEFAULT_NEWSAPI_KEY in infra/news_client.py."
        )

    srcs = sources or DEFAULT_SOURCES_IT

    data = fetch_top_headlines(key, page_size=page_size, sources=srcs)
    articles = data.get("articles") if data and data.get("status") == "ok" else None

    used_fallback = False
    if not articles:
        fb = fetch_everything_recent(
            key,
            page_size=page_size,
            sources=srcs,
            lookback_hours=lookback_hours,
        )
        if fb and fb.get("status") == "ok":
            articles = fb.get("articles", [])
            used_fallback = True
        else:
            err = (fb or {}).get("error", {})
            raise RuntimeError(
                f"Errore NewsAPI: {err.get('code')} – {err.get('message')}"
            )

    articles = dedup_by_title(articles or [])
    return articles, used_fallback


def format_rome_time(ts_iso: str) -> str:
    """
    Converte un timestamp ISO NewsAPI in stringa 'YYYY-MM-DD HH:MM' fuso orario Roma.
    """
    try:
        ts = dt.datetime.fromisoformat((ts_iso or "").replace("Z", "+00:00"))
        return ts.astimezone(ZoneInfo("Europe/Rome")).strftime("%Y-%m-%d %H:%M")
    except Exception:
        return ts_iso or ""
