# ui/page_optimization.py
from __future__ import annotations

from typing import Any, Dict, Optional
import streamlit as st


def render_optimization_page(*args: Any, **kwargs: Any) -> None:
    """
    Stub sicuro: evita crash di import/syntax e accetta qualsiasi firma
    (runs/pid/state/res ecc.). Poi potrai rimettere la logica di ottimizzazione.
    """
    st.subheader("Ottimizzazione")

    # Mostra cosa sta arrivando (debug utile)
    if kwargs:
        with st.expander("Debug params (kwargs)"):
            for k, v in kwargs.items():
                st.write(k, type(v).__name__)

    st.info(
        "Pagina Ottimizzazione caricata correttamente (stub). "
        "Ora possiamo reintrodurre la logica un pezzo alla volta senza bloccare il deploy."
    )
