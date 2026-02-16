# core/research_core.py
# Backend per la sezione "Research & News":
# - generazione di market briefing multi-orizzonte con diversi focus
#   tramite OpenAI (NO Ollama).

from __future__ import annotations

import locale
import os
from datetime import date, datetime
from typing import Iterable, List


def format_report_date(d: date | None = None, locale_code: str = "it_IT.UTF-8") -> str:
    """
    Restituisce una stringa data 'bella' (es. "Lunedì 02 Dicembre 2025").
    Se la localizzazione non funziona, usa un formato ISO semplice.
    """
    d = d or date.today()
    dt = datetime(d.year, d.month, d.day)

    try:
        locale.setlocale(locale.LC_TIME, locale_code)
    except Exception:
        try:
            locale.setlocale(locale.LC_TIME, "it_IT")
        except Exception:
            pass

    try:
        return dt.strftime("%A %d %B %Y").title()
    except Exception:
        return dt.strftime("%Y-%m-%d")


def _normalize_horizon(horizon: str) -> str:
    """
    Normalizza l'orizzonte in una delle tre etichette:
    'daily', 'weekly', 'monthly'.
    """
    h = (horizon or "").strip().lower()
    if "settiman" in h or "week" in h:
        return "weekly"
    if "mensil" in h or "month" in h:
        return "monthly"
    return "daily"


_FOCUS_MAP = {
    "azioni": "global equities (US, Europe, EM, sectors and key single names)",
    "equities": "global equities (US, Europe, EM, sectors and key single names)",
    "obbligazioni": "rates & credit (sovereign curves, credit spreads, corporate bonds)",
    "bonds": "rates & credit (sovereign curves, credit spreads, corporate bonds)",
    "macro": "macro data & central banks (inflation, growth, jobs, Fed, ECB, BoE, BoJ)",
    "geopolitico": "geopolitics & policy (elections, conflicts, sanctions, regulation)",
    "geopolitics": "geopolitics & policy (elections, conflicts, sanctions, regulation)",
    "multi-asset": "cross-asset view (equities, bonds, FX, commodities, macro, geopolitics)",
}


def _focus_description(focus_list: Iterable[str]) -> str:
    """
    Trasforma la lista dei focus ("Azioni", "Macro", ...) in descrizione inglese.
    Se non viene passato nulla, usa un focus cross-asset generico.
    """
    items: List[str] = []
    for f in focus_list:
        key = (f or "").strip().lower()
        if not key:
            continue
        items.append(_FOCUS_MAP.get(key, key))

    if not items:
        return "cross-asset markets (equities, bonds, FX, commodities, macro and geopolitics)"

    # rimuove duplicati preservando l’ordine
    seen = set()
    out: List[str] = []
    for x in items:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return ", ".join(out)


def build_briefing_prompt(
    horizon: str,
    focus_list: Iterable[str],
    report_date: date | None = None,
) -> str:
    """
    Costruisce il prompt per il modello LLM in base a:
    - orizzonte (giornaliero / settimanale / mensile),
    - focus (azioni, obbligazioni, macro, geopolitico, multi-asset),
    - data del report.
    """
    horizon_norm = _normalize_horizon(horizon)
    date_str = format_report_date(report_date)

    horizon_label = {
        "daily": "previous trading day",
        "weekly": "last trading week (5–7 sessions)",
        "monthly": "last trading month",
    }.get(horizon_norm, "previous trading day")

    focus_desc = _focus_description(focus_list)

    prompt = f"""
Act as a senior sell-side financial analyst writing a {horizon_norm} market briefing
for professional investors and portfolio managers.

Report date: {date_str}
Time horizon: {horizon_label}
Main focus areas: {focus_desc}

Write the briefing in **English**, concise but comprehensive,
with a tone similar to Bloomberg or the Financial Times.

Structure the report in clear markdown sections:

1. Market snapshot (overall risk-on/off tone, main moves across assets).
2. Equities: indices, sectors and notable single names moves and drivers.
3. Fixed income & credit: government bond curves, credit spreads, key moves.
4. Macro & central banks: latest data, inflation, growth, central-bank rhetoric.
5. FX & commodities: main currency pairs and key commodities (oil, gold, etc.).
6. Geopolitics & policy risks: relevant developments and how they impact markets.
7. Takeaways & positioning ideas: 3–5 bullet points with key conclusions.

Rules:
- Give **more detail** to the areas included in the focus list above;
  non-focused areas can be shorter, but must still be covered.
- Use short paragraphs and bullet points when it helps readability.
- Avoid hallucinating precise data prints; focus on qualitative summary and drivers.
- Do NOT add disclaimers or extra commentary before or after the briefing.
"""
    return prompt.strip()


def generate_market_briefing(
    horizon: str,
    focus_list: Iterable[str],
    report_date: date | None = None,
    host: str | None = None,
    model: str | None = None,
) -> str:
    """
    Genera il briefing tramite OpenAI (NO Ollama).
    Mantiene la stessa firma della versione precedente per minimizzare modifiche.

    Config:
    - OPENAI_API_KEY (obbligatoria)
    - OPENAI_BASE_URL (opzionale) oppure parametro host=
    - OPENAI_MODEL_NEWS / OPENAI_MODEL (opzionale) oppure parametro model=
    """
    try:
        from openai import OpenAI  # type: ignore
    except ImportError as e:
        raise RuntimeError(
            "Python package 'openai' non installato. "
            "Installa nel venv con: pip install openai"
        ) from e

    api_key = (os.getenv("OPENAI_API_KEY") or "").strip()
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY non impostata nelle variabili d'ambiente.")

    base_url = (host or os.getenv("OPENAI_BASE_URL") or "https://api.openai.com/v1").strip()
    model_cfg = (
        model
        or os.getenv("OPENAI_MODEL_NEWS")
        or os.getenv("OPENAI_MODEL")
        or "gpt-4o-mini"
    ).strip()

    prompt = build_briefing_prompt(horizon=horizon, focus_list=focus_list, report_date=report_date)

    # temperatura “da analyst” (non troppo creativa)
    try:
        temperature = float(os.getenv("OPENAI_TEMPERATURE", "0.4"))
    except Exception:
        temperature = 0.4

    client = OpenAI(api_key=api_key, base_url=base_url)

    resp = client.chat.completions.create(
        model=model_cfg,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a senior sell-side analyst. "
                    "Write Bloomberg/FT-like market briefings. "
                    "No disclaimers, no preambles."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        temperature=temperature,
        max_tokens=1400,
    )

    text = (resp.choices[0].message.content or "").strip()
    return text
