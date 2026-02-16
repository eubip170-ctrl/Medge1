# core/ai_client.py
from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

_CLIENT = None
DEFAULT_MODEL = "gpt-5-mini"


def _read_setting(name: str, default: str = "") -> str:
    """
    Legge impostazioni in ordine:
      1) Streamlit secrets (Cloud: Manage app -> Settings -> Secrets)
      2) Variabili ambiente
      3) default
    """
    try:
        import streamlit as st  # type: ignore

        v = st.secrets.get(name, None)
        if v is not None:
            s = str(v).strip()
            if s:
                return s
    except Exception:
        pass

    s = os.getenv(name, "").strip()
    if s:
        return s

    return default


def _get_api_key() -> str:
    key = _read_setting("OPENAI_API_KEY", "")
    if not key:
        raise RuntimeError(
            "OPENAI_API_KEY mancante.\n"
            "Streamlit Cloud: Manage app → Settings → Secrets\n"
            'Esempio:\nOPENAI_API_KEY="..."\nOPENAI_MODEL="gpt-5-mini"\n'
        )
    return key


def _get_model(model: Optional[str]) -> str:
    if model and isinstance(model, str) and model.strip():
        return model.strip()
    return _read_setting("OPENAI_MODEL", DEFAULT_MODEL) or DEFAULT_MODEL


def _is_reasoning_model(model_name: str) -> bool:
    mn = (model_name or "").strip().lower()
    return mn.startswith("gpt-5") or mn.startswith("o1") or mn.startswith("o3") or mn.startswith("o4")


def _temperature_is_supported(model_name: str, reasoning_effort: Optional[str]) -> bool:
    """
    Regola pratica (in base alle linee guida GPT-5.2):
      - GPT-5.2: temperature supportata SOLO con reasoning.effort="none"
      - GPT-5 (es. gpt-5, gpt-5-mini, gpt-5-nano): temperature NON supportata
      - Altri modelli (es. gpt-4.1, gpt-4o, ecc.): in genere supportata
    """
    mn = (model_name or "").strip().lower()
    eff = (reasoning_effort or "").strip().lower()

    if mn.startswith("gpt-5.2"):
        return eff == "none"
    if mn.startswith("gpt-5"):
        return False
    return True


def _get_client():
    global _CLIENT
    if _CLIENT is not None:
        return _CLIENT

    try:
        from openai import OpenAI  # type: ignore
    except Exception as e:
        raise RuntimeError(
            "Pacchetto 'openai' non installato. Aggiungi in requirements.txt: openai>=1.0.0"
        ) from e

    _CLIENT = OpenAI(api_key=_get_api_key())
    return _CLIENT


def chat_completion(
    system_prompt: str,
    user_prompt: str,
    model: Optional[str] = None,
    max_output_tokens: int = 600,
    temperature: Optional[float] = 0.2,
    reasoning_effort: str = "low",
) -> str:
    """
    Wrapper unico usato dall'app.
    - Usa Responses API.
    - Se la risposta è incomplete ma contiene testo, restituisce comunque il testo.
    - Evita parametri non supportati (es. temperature su gpt-5-mini).
    """
    client = _get_client()
    model_name = _get_model(model)

    payload: Dict[str, Any] = {
        "model": model_name,
        "input": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "max_output_tokens": int(max_output_tokens),
    }

    # Passa reasoning SOLO ai modelli reasoning
    if _is_reasoning_model(model_name) and reasoning_effort:
        payload["reasoning"] = {"effort": str(reasoning_effort)}

    # Passa temperature SOLO se supportata dal modello
    if temperature is not None and _temperature_is_supported(model_name, reasoning_effort):
        payload["temperature"] = float(temperature)

    try:
        resp = client.responses.create(**payload)
    except Exception as e:
        raise RuntimeError(f"Errore chiamata OpenAI: {e}") from e

    # Metodo comodo dell'SDK (quando presente)
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
