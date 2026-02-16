# app_streamlit.py – entry point / router

import streamlit as st

# ✅ Deve essere la prima chiamata Streamlit (prima di qualsiasi st.* altrove)
st.set_page_config(page_title="MEDGE – Portfolio Analytics", layout="wide", page_icon="📈")

# Theme (CSS globale)
from ui.theme import use_pro_theme
use_pro_theme()

# Pagine
from ui.page_portfolio import render_portfolio_page
from ui.page_research import render_research_page
from ui.page_pac import render_pac_page
from ui.page_ideas import render_ideas_page


def main():
    st.title("MEDGE – Portfolio Analytics")

    st.sidebar.title("Sezioni")
    section = st.sidebar.radio(
        "Vai a:",
        (
            "Portafoglio",
            "Research & News",
            "PAC & Investimenti programmati",
            "Generatore di idee",
        ),
        index=0,
    )

    if section == "Portafoglio":
        render_portfolio_page()
    elif section == "Research & News":
        render_research_page()
    elif section == "PAC & Investimenti programmati":
        render_pac_page()
    else:
        render_ideas_page()


if __name__ == "__main__":
    main()
