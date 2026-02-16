# ui/page_ideas.py

import streamlit as st
import pandas as pd

from core.ideas_core import build_nodes_from_df, ideas_to_dot


def _next_node_id() -> str:
    """Genera un id incrementale: n1, n2, n3..."""
    cur = st.session_state.get("ideas_next_id", 1)
    st.session_state["ideas_next_id"] = cur + 1
    return f"n{cur}"


def render_ideas_page():
    st.title("Generatore di idee & Schemi mentali")

    st.markdown(
        """
        Qui puoi **mappare le tue idee di trading/investimento** come una
        mind map: idea principale, sotto-idee, gestione del rischio, trigger,
        note operative, ecc.

        1. Definisci l'**idea principale**.
        2. Aggiungi nodi figli (setup, rischio, macro, ecc.).
        3. La mappa mentale si aggiorna automaticamente.
        """
    )

    # ==============================
    # 1) Idea principale
    # ==============================
    st.subheader("Idea principale")

    col1, col2 = st.columns([2, 1])
    with col1:
        main_idea = st.text_input(
            "Titolo idea",
            value=st.session_state.get("ideas_main_title", "Nuova idea di trading"),
        )
        st.session_state["ideas_main_title"] = main_idea

    with col2:
        main_category = st.selectbox(
            "Categoria idea principale",
            ["Setup", "Strategia", "Macro", "Tematica", "Altro"],
            index=1,  # Strategia
        )

    # ==============================
    # 2) Tabella nodi (schema mentale)
    # ==============================
    st.subheader("Schema mentale – nodi")

    # Inizializza tabella se non esiste
    if "ideas_nodes_df" not in st.session_state:
        st.session_state["ideas_nodes_df"] = pd.DataFrame(
            [
                {
                    "id": "root",
                    "label": main_idea,
                    "parent_id": "",
                    "category": main_category,
                    "priority": 5,
                }
            ]
        )
        st.session_state["ideas_next_id"] = 1

    df_nodes = st.session_state["ideas_nodes_df"]

    # Mantieni aggiornato il nodo root con titolo/categoria
    df_nodes.loc[df_nodes["id"] == "root", "label"] = main_idea
    df_nodes.loc[df_nodes["id"] == "root", "category"] = main_category

    # ----- Pulsanti rapidi per aggiungere nodi -----
    c_add1, c_add2, c_add3 = st.columns(3)
    with c_add1:
        if st.button("➕ Nodo figlio dell'idea principale"):
            df = st.session_state["ideas_nodes_df"].copy()
            new_row = {
                "id": _next_node_id(),
                "label": "Nuovo nodo (figlio root)",
                "parent_id": "root",
                "category": "Setup",
                "priority": 5,
            }
            df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
            st.session_state["ideas_nodes_df"] = df
            df_nodes = df

    with c_add2:
        if st.button("➕ Nodo senza parent (nuova radice logica)"):
            df = st.session_state["ideas_nodes_df"].copy()
            new_row = {
                "id": _next_node_id(),
                "label": "Nuovo nodo indipendente",
                "parent_id": "",
                "category": "Altro",
                "priority": 5,
            }
            df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
            st.session_state["ideas_nodes_df"] = df
            df_nodes = df

    with c_add3:
        if st.button("🗑️ Svuota tutti i nodi (reset)"):
            st.session_state["ideas_nodes_df"] = pd.DataFrame(
                [
                    {
                        "id": "root",
                        "label": main_idea,
                        "parent_id": "",
                        "category": main_category,
                        "priority": 5,
                    }
                ]
            )
            st.session_state["ideas_next_id"] = 1
            df_nodes = st.session_state["ideas_nodes_df"]

    st.caption(
        "Suggerimento: usa i pulsanti qui sopra per creare i nodi.  \n"
        "Se vuoi, puoi comunque modificare a mano `id` e `parent_id` nella tabella."
    )

    edited_df = st.data_editor(
        df_nodes,
        num_rows="dynamic",
        use_container_width=True,
        hide_index=True,
        column_config={
            "id": st.column_config.TextColumn("ID nodo"),
            "label": st.column_config.TextColumn("Descrizione"),
            "parent_id": st.column_config.TextColumn("Parent ID (vuoto = nessun padre)"),
            "category": st.column_config.TextColumn("Categoria"),
            "priority": st.column_config.NumberColumn(
                "Priorità", min_value=1, max_value=10, step=1
            ),
        },
        key="ideas_nodes_editor",
    )

    # Salva tabella aggiornata
    st.session_state["ideas_nodes_df"] = edited_df

    # ==============================
    # 3) Visualizzazione mappa mentale
    # ==============================
    st.subheader("Mappa mentale")

    try:
        nodes = build_nodes_from_df(edited_df)
        if not nodes:
            st.info("Aggiungi almeno un nodo per vedere la mappa.")
        else:
            dot = ideas_to_dot(nodes)
            st.graphviz_chart(dot, use_container_width=True)
    except Exception as e:
        st.error(f"Errore nella generazione della mappa: {e}")

    st.divider()

    # ==============================
    # 4) Metadati operativi dell'idea
    # ==============================
    st.subheader("Metadati operativi dell'idea")

    colA, colB, colC = st.columns(3)
    with colA:
        timeframe = st.selectbox(
            "Timeframe principale", ["Intraday", "Daily", "Weekly", "Multi-year"]
        )
    with colB:
        risk_level = st.slider("Esposizione al rischio (1–10)", 1, 10, 6)
    with colC:
        conviction = st.slider("Convinzione (1–10)", 1, 10, 7)

    st.caption(
        f"Timeframe: **{timeframe}**, Rischio: **{risk_level}/10**, "
        f"Convinzione: **{conviction}/10**"
    )
