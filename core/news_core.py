# core/news_core.py

import os
from datetime import datetime
from typing import Optional, Tuple

# Modello usato per il briefing
OPENAI_BRIEFING_MODEL = os.getenv("OPENAI_BRIEFING_MODEL", "gpt-5-mini").strip()


def _briefing_scope(briefing_type: str, language: str) -> Tuple[str, str]:
    bt = (briefing_type or "").strip().lower()
    is_it = (language or "it").startswith("it")

    # --- Time scope
    if bt in ("giornaliero", "daily"):
        time_scope = "nell’ultima seduta (close-to-close)" if is_it else "in the last session (close-to-close)"
    elif bt in ("settimanale", "weekly"):
        time_scope = "nelle ultime 5–7 sedute" if is_it else "over the last 5–7 sessions"
    elif bt in ("mensile", "monthly"):
        time_scope = "nelle ultime ~20–22 sedute" if is_it else "over the last ~20–22 sessions"
    else:
        time_scope = "nel contesto più recente" if is_it else "in the recent context"

    # --- Focus scope
    if bt in ("azioni", "equity", "equities", "stocks"):
        focus = (
            "prioritizzando **azionario** (indici, settori, leadership/laggards, earnings se rilevante), "
            "senza trascurare tassi/FX come driver"
            if is_it
            else "prioritising **equities** (indices, sectors, leaders/laggards, earnings if relevant), "
                 "while still referencing rates/FX as key drivers"
        )
    elif bt in ("obbligazioni", "bonds", "rates", "credit"):
        focus = (
            "prioritizzando **tassi & credito** (curva, real rates, spread), con collegamenti a equity/FX"
            if is_it
            else "prioritising **rates & credit** (curve, real rates, spreads), linking back to equity/FX"
        )
    elif bt in ("macro", "macroeconomia"):
        focus = (
            "prioritizzando **macro & banche centrali** (inflazione, crescita, lavoro, guidance), "
            "e l’impatto cross-asset"
            if is_it
            else "prioritising **macro & central banks** (inflation, growth, jobs, guidance) "
                 "and the cross-asset transmission"
        )
    elif bt in ("geopolitico", "geopolitics", "policy"):
        focus = (
            "prioritizzando **geopolitica & policy** (rischi, scenari, canali di trasmissione su energia/FX/risk)"
            if is_it
            else "prioritising **geopolitics & policy** (risks, scenarios, transmission via energy/FX/risk)"
        )
    else:
        focus = (
            "coprendo **cross-asset** (equity, tassi/credito, FX, commodities; cripto solo se materiale)"
            if is_it
            else "covering **cross-asset** (equities, rates/credit, FX, commodities; crypto only if material)"
        )

    return time_scope, focus


def _system_prompt_for_briefing(language: str) -> str:
    is_it = (language or "it").startswith("it")
    if is_it:
        return (
            "Ruolo: sei un **senior cross-asset strategist** (sell-side) e scrivi una nota per investitori "
            "professionali (PM, risk committee).\n\n"
            "Stile: conciso, tono Bloomberg/FT. Frasi corte/medie, niente riempitivi.\n\n"
            "Vincoli:\n"
            "- NIENTE consigli personalizzati o trade call.\n"
            "- NON inventare numeri o fatti.\n"
            "- Se manca evidenza su un evento specifico, resta su driver generici o condizionali.\n"
            "- Non menzionare limiti real-time o che sei un modello.\n\n"
            "Output: Markdown pulito. Le intestazioni di sezione DEVONO essere **in grassetto** e identiche al template."
        )
    return (
        "Role: you are a **senior cross-asset strategist** writing for professional investors.\n\n"
        "Style: concise, Bloomberg/FT tone. Short/medium sentences.\n\n"
        "Constraints:\n"
        "- No personalised advice or explicit trade calls.\n"
        "- Do NOT fabricate numbers or specific facts.\n"
        "- If unsupported, keep drivers generic/conditional.\n"
        "- Never mention being a model or lacking real-time data.\n\n"
        "Output: clean Markdown. Section headings MUST be **bold** and match the template exactly."
    )


def _fix_headings_md(text: str) -> str:
    """Se il modello dimentica i titoli in **grassetto**, li correggiamo in modo soft."""
    if not text:
        return text

    headings = [
        "Executive summary",
        "Azioni",
        "Equities",
        "Obbligazioni & tassi (e credito)",
        "Rates & Credit",
        "Macro & banche centrali",
        "Macro & Central banks",
        "Valute & materie prime",
        "FX & Commodities",
        "Cripto",
        "Crypto",
        "Rischi e temi chiave da monitorare",
        "Key risks & themes to watch",
    ]

    out_lines = []
    for line in text.splitlines():
        stripped = line.strip()
        if stripped in headings and not stripped.startswith("**"):
            out_lines.append(f"**{stripped}**")
        else:
            out_lines.append(line)
    return "\n".join(out_lines).strip()


def generate_market_briefing(
    briefing_type: str = "Giornaliero",
    language: str = "it",
    context: Optional[str] = None,
) -> str:
    """
    Briefing cross-asset professionale.
    Se `context` è fornito, viene trattato come “market tape” fact-based.
    """
    # Import LAZY per evitare qualsiasi rischio di circular import con altri moduli
    from .ai_client import chat_completion

    lang = (language or "it").lower()
    lang = "it" if lang.startswith("it") else "en"

    time_scope, focus_scope = _briefing_scope(briefing_type, lang)
    system_prompt = _system_prompt_for_briefing(lang)
    today_str = datetime.now().strftime("%Y-%m-%d")

    ctx_block = ""
    if context and context.strip():
        if lang == "it":
            ctx_block = f"""
Market tape (usa SOLO questi fatti quando citi movimenti; se un'asset class non è coperta, resta qualitativo):
{context.strip()}
"""
        else:
            ctx_block = f"""
Market tape (use ONLY these facts when referencing moves; if something is not covered, stay qualitative):
{context.strip()}
"""

    if lang == "it":
        user_prompt = f"""
Scrivi un **briefing di mercato ({briefing_type})** {time_scope}, {focus_scope}.
Data di riferimento: **{today_str}**.

{ctx_block}

REGOLE DI FORMATO (OBBLIGATORIE):
- Ogni sezione deve iniziare con una riga di intestazione ESATTAMENTE così (con **grassetto**):
  **Executive summary**
  **Azioni**
  **Obbligazioni & tassi (e credito)**
  **Macro & banche centrali**
  **Valute & materie prime**
  **Cripto**
  **Rischi e temi chiave da monitorare**
- NON numerare le sezioni. NON aggiungere altre intestazioni.
- Evita ripetizioni meccaniche tipo “Move → Driver → …” come unica forma: usa bullet puliti con etichette in grassetto.

TEMPLATE (compila rispettando la struttura):

**Executive summary**
- **Tone:** Risk-on / Risk-off / Neutral + 1 frase (coerente col market tape se presente)
- **Equities:** 1 riga (direzione + leaders/laggards)
- **Rates/Credit:** 1 riga (curva + credito)
- **FX/Commodities:** 1 riga (USD + oil/gold)
- **What to watch:** 1–2 trigger (senza inventare eventi)

**Azioni**
- **Move:** ... **Drivers:** ... **Implications/Risks:** ...
- (3–5 bullet totali)

**Obbligazioni & tassi (e credito)**
- **Move:** ... **Drivers:** ... **Implications/Risks:** ...
- (3–5 bullet)

**Macro & banche centrali**
- **Move:** ... **Drivers:** ... **Implications/Risks:** ...
- (3–5 bullet)

**Valute & materie prime**
- **Move:** ... **Drivers:** ... **Implications/Risks:** ...
- (3–5 bullet)

**Cripto**
- Se non materiale: 1 riga sola “Non driver principale nel periodo.”
- Se materiale: max 3 righe (driver + rischio)

**Rischi e temi chiave da monitorare**
- 4–6 bullet “trigger → impatto” (una riga ciascuno)

Lunghezza: ~280–520 parole. Zero preamboli.
"""
    else:
        user_prompt = f"""
Write a **market briefing ({briefing_type})** {time_scope}, {focus_scope}.
Reference date: **{today_str}**.

{ctx_block}

FORMAT RULES (MANDATORY):
- Each section must start with a heading EXACTLY like this (bold):
  **Executive summary**
  **Equities**
  **Rates & Credit**
  **Macro & Central banks**
  **FX & Commodities**
  **Crypto**
  **Key risks & themes to watch**
- Do NOT number sections. Do NOT add extra headings.

TEMPLATE:

**Executive summary**
- **Tone:** Risk-on / Risk-off / Neutral + 1 sentence
- **Equities:** 1 line
- **Rates/Credit:** 1 line
- **FX/Commodities:** 1 line
- **What to watch:** 1–2 triggers (no invented events)

**Equities**
- **Move:** ... **Drivers:** ... **Implications/Risks:** ...
- (3–5 bullets)

**Rates & Credit**
- **Move:** ... **Drivers:** ... **Implications/Risks:** ...
- (3–5 bullets)

**Macro & Central banks**
- **Move:** ... **Drivers:** ... **Implications/Risks:** ...
- (3–5 bullets)

**FX & Commodities**
- **Move:** ... **Drivers:** ... **Implications/Risks:** ...
- (3–5 bullets)

**Crypto**
- If not material: one line “Not a key driver in this period.”
- If material: max 3 lines

**Key risks & themes to watch**
- 4–6 one-line bullets “trigger → impact”

Length: ~280–520 words. No preamble.
"""

    text = chat_completion(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        model=OPENAI_BRIEFING_MODEL,
        max_output_tokens=900,
    )
    return _fix_headings_md((text or "").strip())
