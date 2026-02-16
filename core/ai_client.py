# core/ai_client.py
from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

# Cache client (lazy)
_CLIENT = None

# Default model se non lo imposti in secrets/env
DEFAULT_MODEL = "gpt-5-mini"


def _read_setting(name: str, default: str = "") -> str:
    """
    Legge impostazioni in ordine:
      1) Streamlit secrets (locale: .streamlit/secrets.toml, cloud: Settings->Secrets)
      2) Variabili ambiente
      3) default
    """
    # 1) streamlit secrets
    try:
        import streamlit as st  # type: ignore

        v = st.secrets.get(name, None)
        if v is not None:
            s = str(v).strip()
            if s:
                return s
    except Exception:
        pass

    # 2) env var
    s = os.getenv(name, "").strip()
    if s:
        return s

    return default


def _get_api_key() -> str:
    key = _read_setting("OPENAI_API_KEY", "")
    if not key:
        raise RuntimeError(
            "OPENAI_API_KEY mancante.\n"
            "- Locale: crea .streamlit/secrets.toml con OPENAI_API_KEY\n"
            "- Streamlit Cloud: Manage app → Settings → Secrets\n"
        )
    return key


def _get_model(model: Optional[str]) -> str:
    if model and isinstance(model, str) and model.strip():
        return model.strip()
    return _read_setting("OPENAI_MODEL", DEFAULT_MODEL) or DEFAULT_MODEL


def _get_client():
    """
    Lazy import del pacchetto openai per evitare crash all'import del modulo
    (se openai non è installato, l'app può comunque avviarsi e mostriamo un errore chiaro quando serve).
    """
    global _CLIENT
    if _CLIENT is not None:
        return _CLIENT

    try:
        from openai import OpenAI  # type: ignore
    except Exception as e:
        raise RuntimeError(
            "Pacchetto 'openai' non installato.\n"
            "Aggiungi in requirements.txt: openai>=1.0.0"
        ) from e

    _CLIENT = OpenAI(api_key=_get_api_key())
    return _CLIENT


def chat_completion(
    system_prompt: str,
    user_prompt: str,
    model: Optional[str] = None,
    max_output_tokens: int = 600,
    temperature: float = 0.2,
    reasoning_effort: str = "low",
) -> str:
    """
    Wrapper unico per ottenere testo dal modello tramite Responses API.
    - Se la risposta è incomplete ma contiene testo, restituisce comunque il testo.
    - Se non c'è testo, alza errore.
    """
    client = _get_client()
    model_name = _get_model(model)

    payload: Dict[str, Any] = {
        "model": model_name,
        "input": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": float(temperature),
        "max_output_tokens": int(max_output_tokens),
    }

    # reasoning è opzionale: lo mettiamo solo se richiesto
    if reasoning_effort:
        payload["reasoning"] = {"effort": str(reasoning_effort)}

    try:
        resp = client.responses.create(**payload)
    except Exception as e:
        raise RuntimeError(f"Errore chiamata OpenAI: {e}") from e

    # Via proprietà comoda dell'SDK (quando disponibile)
    text = (getattr(resp, "output_text", None) or "").strip()
    if text:
        return text

    # Fallback robusto: estrai testo dalla struttura output
    chunks: List[str] = []
    output = getattr(resp, "output", None)

    if output is None:
        try:
            output = resp.model_dump().get("output", [])
        except Exception:
            output = []

    for item in output or []:
        if isinstance(item, dict):
            itype = item.get("type")
            content = item.get("content", []) or []
        else:
            itype = getattr(item, "type", None)
            content = getattr(item, "content", []) or []

        if itype != "message":
            continue

        for block in content or []:
            if isinstance(block, dict):
                btype = block.get("type")
                btext = block.get("text")
            else:
                btype = getattr(block, "type", None)
                btext = getattr(block, "text", None)

            if btype in ("output_text", "text") and btext:
                chunks.append(str(btext))

    text = "\n".join(chunks).strip()

    if not text:
        status = getattr(resp, "status", None)
        reason = getattr(getattr(resp, "incomplete_details", None), "reason", None)
        if status == "incomplete" and reason == "max_output_tokens":
            raise RuntimeError(
                f"Risposta troncata per max_output_tokens={max_output_tokens} e nessun testo utile restituito."
            )
        raise RuntimeError("Nessun testo restituito da OpenAI.")

    return text
