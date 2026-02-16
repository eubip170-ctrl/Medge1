# core/ai_client.py

import os
from typing import Optional

from openai import OpenAI

# ============================================================
#  CONFIG API KEY / MODELLO
# ============================================================

# 1) Preferito: variabile di ambiente OPENAI_API_KEY
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()

# 2) In alternativa puoi incollare qui la chiave (NON consigliato in produzione)
if not OPENAI_API_KEY:
    OPENAI_API_KEY = "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"


if not OPENAI_API_KEY or OPENAI_API_KEY.startswith("INSERISCI-"):
    raise RuntimeError(
        "OPENAI_API_KEY non impostata. "
        "Imposta la variabile ambiente OPENAI_API_KEY oppure "
        "sostituisci 'INSERISCI-LA-TUA-API-KEY-QUI' con la tua chiave reale."
    )

client = OpenAI(api_key=OPENAI_API_KEY)

# Modello di default (puoi cambiarlo via env)
DEFAULT_MODEL = os.getenv("OPENAI_MODEL", "gpt-5-mini").strip()


# ============================================================
#  WRAPPER CHAT COMPLETION (Responses API)
# ============================================================

def chat_completion(
    system_prompt: str,
    user_prompt: str,
    model: Optional[str] = None,
    max_output_tokens: int = 600,
) -> str:
    """
    Wrapper unico usato da tutto il resto del codice (news, briefing, Q&A).

    Parametri:
      - system_prompt: istruzioni di ruolo
      - user_prompt: contenuto principale
      - model: id modello (es. "gpt-5-nano")
      - max_output_tokens: limite massimo di token di output

    Restituisce:
      - stringa di testo della risposta del modello

    Comportamento:
      - se la risposta è "incomplete" per max_output_tokens MA abbiamo del testo,
        lo usiamo comunque (niente errore).
      - se non arriva nessun testo, allora sì → errore.
    """
    # Se non viene passato il modello, usa il default
    if model is None or isinstance(model, int):
        model = DEFAULT_MODEL

    try:
        resp = client.responses.create(
            model=model,
            input=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            # Per gpt-5-nano: riduciamo l'effort di reasoning,
            # così lascia più spazio ai token di testo.
            reasoning={"effort": "low"},
            max_output_tokens=max_output_tokens,
        )
    except Exception as e:
        raise RuntimeError(f"Errore chiamata OpenAI: {e}") from e

    # ============== Estrazione testo in modo robusto ==============
    text_chunks: list[str] = []

    output = getattr(resp, "output", None)
    if output is None:
        # fallback: struttura dict
        try:
            output = resp.model_dump().get("output", [])
        except Exception:
            output = []

    for item in output or []:
        # item può essere oggetto pydantic o dict
        if isinstance(item, dict):
            item_type = item.get("type")
            content_list = item.get("content", []) or []
        else:
            item_type = getattr(item, "type", None)
            content_list = getattr(item, "content", []) or []

        if item_type != "message":
            continue

        for block in content_list:
            if isinstance(block, dict):
                btype = block.get("type")
                text = block.get("text")
            else:
                btype = getattr(block, "type", None)
                text = getattr(block, "text", None)

            if btype in ("output_text", "text") and text:
                text_chunks.append(text)

    text = "\n".join(text_chunks).strip()

    status = getattr(resp, "status", None)
    incomplete_reason = getattr(getattr(resp, "incomplete_details", None), "reason", None)

    # 🔴 Caso 1: nessun testo → errore
    if not text:
        if status == "incomplete" and incomplete_reason == "max_output_tokens":
            raise RuntimeError(
                f"Risposta OpenAI troncata per max_output_tokens={max_output_tokens} "
                "e nessun testo utile è stato restituito."
            )
        raise RuntimeError("Nessun testo restituito da OpenAI.")

    # 🟡 Caso 2: risposta 'incomplete' ma abbiamo testo → ok, lo usiamo lo stesso
    # (se vuoi, potresti loggare un warning, ma non disturbiamo la UI)
    return text
