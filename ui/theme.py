# ui/theme.py
"""
Tema unico dark, professionale per app di portafogli:
- main scuro
- sidebar scura in continuità
- testi chiari
- accenti ciano/blu
"""

import streamlit as st

# =========================
#   PALETTE DARK
# =========================

PALETTE = {
    # Background principali
    "bg": "#020720",          # sfondo principale
    "card_bg": "#020827",     # sfondo card / contenitori
    "sidebar_bg": "#050C30",  # sidebar scura

    # Testi
    "text": "#d2d3d4",        # testo principale
    "text_muted": "#dadbdb",  # testi secondari
    "heading": "#f9fafb",     # titoli molto chiari

    # Bordi / griglie
    "border": "#1B2036",
    "border_strong": "#334155",
    "grid": "#1f2937",

    # Tabelle
    "table_header": "#050C30",
    "table_alt": "#020617",

    # Input / codice
    "input_bg": "#0A154D",
    "code_bg": "#0A154D",

    # Bottoni / link / accenti (base)
    "button_bg": "#0A154D",
    "link": "#38bdf8",
    "accent": "#38bdf8",
    "accent_soft": "rgba(56, 189, 248, 0.28)",

    # Verde / rosso per performance
    "positive": "#22c55e",
    "negative": "#ef4444",

    # Bottoni avanzati (primario/secondario)
    "button_primary_bg": "#38bdf8",
    "button_primary_text": "#020617",
    "button_secondary_bg": "#0A154D",
    "button_secondary_text": "#7D7B8B",

    # Topbar
    "topbar_bg": "#050C30",
    "topbar_text": "#e5e7eb",
    "topbar_border": "rgba(15,23,42,0.9)",

    # Scrollbar
    "scrollbar_track": "#00082B",
    "scrollbar_thumb": "#1B2036",

    # Alert (per markup custom)
    "alert_info_bg": "rgba(56, 189, 248, 0.10)",
    "alert_info_border": "#38bdf8",
    "alert_warning_bg": "rgba(234, 179, 8, 0.12)",
    "alert_warning_border": "#eab308",
    "alert_danger_bg": "rgba(239, 68, 68, 0.12)",
    "alert_danger_border": "#ef4444",

    # Tag / chip
    "tag_bg": "#050C30",
    "tag_text": "#e5e7eb",
    "chip_neutral_bg": "#0A154D",
    "chip_neutral_text": "#d2d3d4",
    "chip_positive_bg": "rgba(34, 197, 94, 0.15)",
    "chip_positive_text": "#bbf7d0",
    "chip_negative_bg": "rgba(239, 68, 68, 0.15)",
    "chip_negative_text": "#fecaca",
}


# =========================
#   MATPLOTLIB / PLOTLY
# =========================

def apply_mpl_theme() -> None:
    """Allinea Matplotlib alla palette dark dell'app."""
    try:
        import matplotlib as mpl
    except Exception:
        return

    pal = PALETTE

    mpl.rcParams.update(
        {
            "figure.facecolor": pal["card_bg"],
            "axes.facecolor": pal["card_bg"],
            "savefig.facecolor": pal["card_bg"],
            "axes.edgecolor": pal["border"],
            "axes.labelcolor": pal["text"],
            "text.color": pal["text"],
            "xtick.color": pal["text_muted"],
            "ytick.color": pal["text_muted"],
            "grid.color": pal["grid"],
            "grid.alpha": 0.4,
            "axes.grid": True,
            "figure.edgecolor": pal["border"],
        }
    )


def apply_plotly_theme(fig) -> None:
    """Applica la palette dark alla singola figura Plotly."""
    pal = PALETTE
    fig.update_layout(
        paper_bgcolor=pal["card_bg"],
        plot_bgcolor=pal["card_bg"],
        font=dict(color=pal["text"]),
        xaxis=dict(
            gridcolor=pal["grid"],
            zerolinecolor=pal["grid"],
            linecolor=pal["border"],
            tickfont=dict(color=pal["text_muted"]),
            title_font=dict(color=pal["text"]),
        ),
        yaxis=dict(
            gridcolor=pal["grid"],
            zerolinecolor=pal["grid"],
            linecolor=pal["border"],
            tickfont=dict(color=pal["text_muted"]),
            title_font=dict(color=pal["text"]),
        ),
        legend=dict(bgcolor=pal["card_bg"], bordercolor=pal["border"]),
        hoverlabel=dict(bgcolor=pal["card_bg"], font_color=pal["text"]),
    )


# =========================
#   CSS BASE
# =========================

def _inject_css(pal: dict) -> None:
    accent = pal["accent"]
    accent_soft = pal["accent_soft"]
    positive = pal["positive"]
    negative = pal["negative"]

    st.markdown(
        f"""
    <style>
      :root {{
        --bg: {pal['bg']};
        --card-bg: {pal['card_bg']};
        --sidebar-bg: {pal['sidebar_bg']};
        --text: {pal['text']};
        --text-muted: {pal['text_muted']};
        --heading: {pal['heading']};
        --border: {pal['border']};
        --border-strong: {pal['border_strong']};
        --grid: {pal['grid']};
        --table-header: {pal['table_header']};
        --table-alt: {pal['table_alt']};
        --input-bg: {pal['input_bg']};
        --code-bg: {pal['code_bg']};
        --link: {pal['link']};
        --button-bg: {pal['button_bg']};
        --accent: {accent};
        --accent-soft: {accent_soft};
        --positive: {positive};
        --negative: {negative};

        --button-primary-bg: {pal['button_primary_bg']};
        --button-primary-text: {pal['button_primary_text']};
        --button-secondary-bg: {pal['button_secondary_bg']};
        --button-secondary-text: {pal['button_secondary_text']};

        --topbar-bg: {pal['topbar_bg']};
        --topbar-text: {pal['topbar_text']};
        --topbar-border: {pal['topbar_border']};

        --scrollbar-track: {pal['scrollbar_track']};
        --scrollbar-thumb: {pal['scrollbar_thumb']};

        --alert-info-bg: {pal['alert_info_bg']};
        --alert-info-border: {pal['alert_info_border']};
        --alert-warning-bg: {pal['alert_warning_bg']};
        --alert-warning-border: {pal['alert_warning_border']};
        --alert-danger-bg: {pal['alert_danger_bg']};
        --alert-danger-border: {pal['alert_danger_border']};

        --tag-bg: {pal['tag_bg']};
        --tag-text: {pal['tag_text']};
        --chip-neutral-bg: {pal['chip_neutral_bg']};
        --chip-neutral-text: {pal['chip_neutral_text']};
        --chip-positive-bg: {pal['chip_positive_bg']};
        --chip-positive-text: {pal['chip_positive_text']};
        --chip-negative-bg: {pal['chip_negative_bg']};
        --chip-negative-text: {pal['chip_negative_text']};
      }}

      /* Layout principale */
      [data-testid="stAppViewContainer"] {{
        background: var(--bg);
        color: var(--text);
      }}

      .block-container {{
        padding-top: 1.3rem;
        padding-bottom: 2rem;
      }}

      section.main > div {{
        color: var(--text);
      }}

      /* Sidebar SCURA */
      [data-testid="stSidebar"] {{
        background: var(--sidebar-bg);
        color: #e5e7eb;
        border-right: 1px solid rgba(15,23,42,0.9);
      }}

      [data-testid="stSidebar"] * {{
        font-size: 0.86rem;
        color: #e5e7eb !important;
      }}

      [data-testid="stSidebar"] a {{
        color: #93c5fd !important;
      }}

      [data-testid="stSidebar"] svg {{
        color: #e5e7eb !important;
        fill: #e5e7eb !important;
      }}

      /* Titoli / link */
      h1, h2, h3, h4, h5, h6 {{
        color: var(--heading) !important;
        letter-spacing: 0.01em;
      }}

      h1 {{
        font-size: 1.6rem !important;
        font-weight: 600 !important;
      }}

      h2 {{
        font-size: 1.25rem !important;
        font-weight: 600 !important;
        padding-bottom: 0.25rem;
        border-bottom: 1px solid var(--border);
      }}

      h3 {{
        font-size: 1.05rem !important;
        font-weight: 600 !important;
      }}

      a, a:link, a:visited {{
        color: var(--link) !important;
        text-decoration: none;
      }}
      a:hover {{
        text-decoration: underline;
      }}

      /* Testo normale */
      .stMarkdown p, .stMarkdown li {{
        font-size: 0.94rem;
      }}

      /* Caption / testi secondari */
      [data-testid="stCaptionContainer"],
      [data-testid="stCaptionContainer"] *,
      .stCaption,
      .stMarkdown p small,
      .stMarkdown em,
      .stMarkdown span[style*="opacity"] {{
        color: var(--text-muted) !important;
        opacity: 0.95 !important;
      }}

      /* Card principali */
      .stExpander,
      [data-testid="stPlotlyChart"] > div,
      [data-testid="stMetric"],
      [data-testid="stDataFrame"],
      [data-testid="stTable"],
      .stAlert {{
        background: var(--card-bg) !important;
        border-radius: 6px !important;
        border: 1px solid var(--border) !important;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.45) !important;
      }}

      .stExpander > div {{
        background: transparent !important;
      }}

      /* Alert custom (se usi classi .alert-*) */
      .alert-info {{
        background: var(--alert-info-bg) !important;
        border-color: var(--alert-info-border) !important;
      }}
      .alert-warning {{
        background: var(--alert-warning-bg) !important;
        border-color: var(--alert-warning-border) !important;
      }}
      .alert-danger {{
        background: var(--alert-danger-bg) !important;
        border-color: var(--alert-danger-border) !important;
      }}

      /* Inputs */
      input, textarea, select {{
        color: var(--text) !important;
        background: var(--input-bg) !important;
        border: 1px solid var(--border) !important;
      }}
      input::placeholder, textarea::placeholder {{
        color: var(--text-muted) !important;
        opacity: 0.9 !important;
      }}

      .stTextInput > div > div,
      .stTextArea > div > textarea,
      .stSelectbox > div > div,
      .stMultiSelect > div > div {{
        background: var(--input-bg) !important;
        color: var(--text) !important;
        border-radius: 4px !important;
        border: 1px solid var(--border) !important;
      }}

      /* Radio / checkbox */
      div[role="radiogroup"] label,
      div[data-baseweb="radio"] label,
      label[data-baseweb="checkbox"] {{
        color: var(--text) !important;
        opacity: 0.98 !important;
        font-size: 0.9rem !important;
        font-weight: 500 !important;
      }}

      /* Slider */
      [data-baseweb="slider"] > div {{
        background: transparent !important;
      }}
      [data-baseweb="slider"] div[data-testid="stThumbValue"] {{
        background: var(--card-bg) !important;
        color: var(--text) !important;
        border: 1px solid var(--border) !important;
      }}
      [data-baseweb="slider"] [aria-valuenow] {{
        background: var(--accent) !important;
      }}
      [data-baseweb="slider"] [aria-valuenow]::before {{
        background: var(--accent) !important;
      }}

      /* Bottoni */
      .stButton > button, .stDownloadButton > button {{
        background: var(--button-secondary-bg) !important;
        color: var(--button-secondary-text) !important;
        border: 1px solid var(--border) !important;
        border-radius: 4px !important;
        padding: 0.28rem 0.95rem !important;
        font-size: 0.86rem !important;
        font-weight: 500 !important;
        box-shadow: 0 1px 3px rgba(0,0,0,0.5) !important;
      }}
      .stButton > button:hover, .stDownloadButton > button:hover {{
        border-color: var(--accent) !important;
        box-shadow: 0 0 0 1px var(--accent-soft) !important;
        background: var(--button-primary-bg) !important;
        color: var(--button-primary-text) !important;
      }}

      /* Metriche */
      [data-testid="stMetric"] {{
        padding: 0.55rem 0.8rem !important;
        margin-bottom: 0.4rem !important;
      }}
      [data-testid="stMetric"] label {{
        color: var(--text-muted) !important;
        opacity: 1 !important;
        font-size: 0.78rem !important;
        text-transform: uppercase;
        letter-spacing: 0.06em;
      }}
      [data-testid="stMetric"] [data-testid="stMetricValue"] {{
        color: var(--text) !important;
        font-size: 1.05rem !important;
        font-weight: 600 !important;
      }}
      [data-testid="stMetric"] [data-testid="stMetricDelta"] {{
        font-size: 0.8rem !important;
      }}

      /* Tabelle */
      [data-testid="stTable"] table {{
        background: var(--card-bg) !important;
        color: var(--text) !important;
        border-collapse: collapse !important;
        border: 1px solid var(--border) !important;
      }}
      [data-testid="stTable"] thead tr th {{
        background: var(--table-header) !important;
        color: var(--text) !important;
        border-bottom: 1px solid var(--border-strong) !important;
        font-size: 0.8rem !important;
        text-transform: uppercase;
        letter-spacing: 0.05em;
      }}
      [data-testid="stTable"] tbody tr:nth-child(even) td {{
        background: var(--table-alt) !important;
      }}
      [data-testid="stTable"] td, [data-testid="stTable"] th {{
        border-color: var(--border) !important;
        padding: 0.25rem 0.5rem !important;
        font-size: 0.82rem !important;
      }}

      /* Dataframe */
      [data-testid="stDataFrame"] {{
        background: var(--card-bg) !important;
        color: var(--text) !important;
      }}
      [data-testid="stDataFrame"] div[role="columnheader"] {{
        background: var(--table-header) !important;
        color: var(--text) !important;
        border-bottom: 1px solid var(--border-strong) !important;
        font-size: 0.8rem !important;
        text-transform: uppercase;
        letter-spacing: 0.05em;
      }}
      [data-testid="stDataFrame"] div[role="gridcell"] {{
        background: var(--card-bg) !important;
        color: var(--text) !important;
        border-right: 1px solid var(--border) !important;
        font-size: 0.8rem !important;
      }}
      [data-testid="stDataFrame"] div[role="row"]:nth-child(even) div[role="gridcell"] {{
        background: var(--table-alt) !important;
      }}
      [data-testid="stDataFrame"] ::-webkit-scrollbar-thumb {{
        background: var(--scrollbar-thumb) !important;
      }}

      /* Scrollbar globale */
      ::-webkit-scrollbar {{
        width: 8px;
        height: 8px;
      }}
      ::-webkit-scrollbar-track {{
        background: var(--scrollbar-track);
      }}
      ::-webkit-scrollbar-thumb {{
        background: var(--scrollbar-thumb);
        border-radius: 999px;
      }}

      /* Plotly container */
      [data-testid="stPlotlyChart"] > div {{
        padding: 0.4rem 0.4rem 0.1rem 0.4rem !important;
      }}

      /* Code blocks */
      code, pre {{
        background: var(--code-bg) !important;
        color: var(--text) !important;
        border-radius: 3px !important;
        border: 1px solid var(--border) !important;
      }}

      /* Tabs */
      .stTabs [data-baseweb="tab-list"] > div {{
        border-bottom: 1px solid var(--border) !important;
      }}
      .stTabs [data-baseweb="tab"] {{
        background: transparent !important;
        border-radius: 0 !important;
        border-bottom: 2px solid transparent !important;
        padding-bottom: 0.25rem !important;
        font-size: 0.85rem !important;
        color: var(--text-muted) !important;
      }}
      .stTabs [data-baseweb="tab"][aria-selected="true"] {{
        border-bottom-color: var(--accent) !important;
        color: var(--heading) !important;
      }}

      hr {{
        border-color: var(--border) !important;
        margin: 0.6rem 0 !important;
      }}

      /* Classi opzionali per testo colore performance */
      .delta-positive {{
        color: var(--positive) !important;
      }}
      .delta-negative {{
        color: var(--negative) !important;
      }}

      /* Tag / chip custom per portafogli */
      .portfolio-tag {{
        display: inline-flex;
        align-items: center;
        gap: 0.25rem;
        background: var(--tag-bg);
        color: var(--tag-text);
        border-radius: 999px;
        padding: 0.12rem 0.55rem;
        font-size: 0.78rem;
        border: 1px solid var(--border);
      }}

      .portfolio-chip {{
        display: inline-flex;
        align-items: center;
        gap: 0.25rem;
        border-radius: 999px;
        padding: 0.1rem 0.45rem;
        font-size: 0.76rem;
        border: 1px solid var(--border);
      }}

      .portfolio-chip--neutral {{
        background: var(--chip-neutral-bg);
        color: var(--chip-neutral-text);
      }}

      .portfolio-chip--positive {{
        background: var(--chip-positive-bg);
        color: var(--chip-positive-text);
      }}

      .portfolio-chip--negative {{
        background: var(--chip-negative-bg);
        color: var(--chip-negative-text);
      }}
    </style>
    """,
        unsafe_allow_html=True,
    )


# =========================
#   TOPBAR + LEGGIBILITÀ
# =========================

def inject_topbar_css() -> None:
    st.markdown(
        """
    <style>
      header[data-testid="stHeader"],
      div[data-testid="stHeader"] {
        background: var(--topbar-bg) !important;
        color: var(--topbar-text) !important;
        border-bottom: 1px solid var(--topbar-border) !important;
      }
      header[data-testid="stHeader"] [data-testid="stToolbar"],
      [data-testid="stToolbar"] {
        background: transparent !important;
      }
      header[data-testid="stHeader"] [data-testid="stToolbar"] *,
      [data-testid="stToolbar"] button,
      [data-testid="stToolbar"] svg,
      [data-testid="stToolbar"] span {
        color: var(--topbar-text) !important;
        fill: var(--topbar-text) !important;
      }
    </style>
    """,
        unsafe_allow_html=True,
    )


def inject_text_legibility_hardening() -> None:
    """Rende leggibili tutti i testi con opacità bassa sullo sfondo scuro."""
    st.markdown(
        """
    <style>
      [style*="opacity: 0.6"], [style*="opacity:0.6"],
      [style*="opacity: 0.5"], [style*="opacity:0.5"],
      [style*="opacity: 0.4"], [style*="opacity:0.4"],
      [style*="opacity: 0.3"], [style*="opacity:0.3"],
      small, em {
        color: var(--text-muted) !important;
        opacity: 0.95 !important;
      }
    </style>
    """,
        unsafe_allow_html=True,
    )


def inject_sidebar_text_override() -> None:
    """Forza la sidebar a testo chiaro e leggibile."""
    st.markdown(
        """
    <style>
      [data-testid="stSidebar"] *,
      [data-testid="stSidebar"] .stMarkdown,
      [data-testid="stSidebar"] .stMarkdown p,
      [data-testid="stSidebar"] .stMarkdown li,
      [data-testid="stSidebar"] label,
      [data-testid="stSidebar"] small,
      [data-testid="stSidebar"] span,
      [data-testid="stSidebar"] div {
         color: #e5e7eb !important;
         opacity: 1 !important;
      }
      [data-testid="stSidebar"] input::placeholder,
      [data-testid="stSidebar"] textarea::placeholder {
        color: #cbd5f5 !important;
        opacity: 0.9 !important;
      }
    </style>
    """,
        unsafe_allow_html=True,
    )


# =========================
#   FUNZIONE PUBBLICA
# =========================

def use_pro_theme() -> None:
    """
    Da chiamare in app_streamlit.py
    subito dopo st.set_page_config().
    """
    pal = PALETTE
    st.session_state["_pro_theme_palette"] = pal

    _inject_css(pal)
    inject_topbar_css()
    inject_text_legibility_hardening()
    inject_sidebar_text_override()
    apply_mpl_theme()
