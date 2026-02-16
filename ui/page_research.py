# ui/page_research.py

from __future__ import annotations

import os
import locale
import inspect
from datetime import datetime
from typing import List, Tuple, Dict, Optional

import datetime as dt
import numpy as np
import pandas as pd
import requests
import streamlit as st

from core.news_core import generate_market_briefing
from core.ai_client import chat_completion


# =========================================================
#   CSS (leggero) – migliora leggibilità/spazi
# =========================================================
def _inject_research_css() -> None:
    st.markdown(
        """
<style>
/* padding pagina */
section.main .block-container{
  padding-top: 0.85rem !important;
  padding-bottom: 0.95rem !important;
}
div[data-testid="stVerticalBlock"]{ gap: 0.65rem !important; }
hr{ margin: 0.55rem 0 !important; }

/* “card” soft */
.po-card{
  border: 1px solid rgba(255,255,255,0.10);
  border-radius: 12px;
  padding: 0.85rem 0.95rem;
  background: rgba(17,24,39,0.20);
}

/* titoli compatti */
.po-title{
  font-size: 0.80rem;
  text-transform: uppercase;
  letter-spacing: 0.09em;
  color: rgba(180,189,201,1.0);
  margin: 0 0 0.35rem 0;
}

/* caption più leggibile */
.po-cap{
  color: rgba(180,189,201,1.0);
  font-size: 0.78rem;
}
</style>
        """,
        unsafe_allow_html=True,
    )


def _cols(spec, *, gap: str = "small", valign: Optional[str] = None):
    """st.columns compat: vertical_alignment è disponibile solo in alcune versioni."""
    try:
        sig = inspect.signature(st.columns)
        if "vertical_alignment" in sig.parameters and valign:
            return st.columns(spec, gap=gap, vertical_alignment=valign)
    except Exception:
        pass
    return st.columns(spec, gap=gap)


# =========================================================
#   CONFIG – MARKETSTACK (robusto)
# =========================================================

DEFAULT_MARKETSTACK_API_KEY = "9c304b60c6e2eccfe98fc0dbd0f3571b"

def _resolve_marketstack_key() -> str:
    # 1) env (priorità massima)
    key = os.getenv("MARKETSTACK_API_KEY", "").strip()
    if key:
        return key

    # 2) streamlit secrets (seconda priorità)
    try:
        key = (st.secrets.get("MARKETSTACK_API_KEY") or "").strip()
        if key:
            return key
    except Exception:
        pass

    # 3) fallback hardcoded (ultima spiaggia)
    return DEFAULT_MARKETSTACK_API_KEY.strip()


MARKETSTACK_API_KEY = _resolve_marketstack_key()

# prova v2 -> fallback v1
MARKETSTACK_ENDPOINTS = [
    (os.getenv("MARKETSTACK_BASE_URL", "").strip() or "https://api.marketstack.com/v2/eod"),
    "https://api.marketstack.com/v1/eod",
]

# =========================================================
#   HELPERS – AI (Focus briefing + Market AI)
# =========================================================
def _system_prompt_strat(language: str) -> str:
    is_it = (language or "it").lower().startswith("it")
    if is_it:
        return (
            "Sei un senior cross-asset strategist (sell-side). Tono Bloomberg/FT, conciso e prudente. "
            "No trade call, no consigli personalizzati, no numeri inventati, no eventi specifici non supportati. "
            "Non parlare di limiti real-time o del fatto che sei un modello."
        )
    return (
        "You are a senior cross-asset strategist (sell-side). Bloomberg/FT tone, concise and risk-aware. "
        "No trade calls, no personalised advice, no fabricated numbers, no unsupported specifics. "
        "Do not mention being a model or lacking real-time data."
    )



# =========================================================
#   CONFIG – OPENAI (UI: Focus + Market AI)
# =========================================================
OPENAI_UI_MODEL = os.getenv("OPENAI_UI_MODEL", "").strip()
if not OPENAI_UI_MODEL:
    # fallback: usa lo stesso del briefing se non hai una variabile dedicata
    OPENAI_UI_MODEL = os.getenv("OPENAI_BRIEFING_MODEL", "gpt-5-mini").strip()



def _generate_focus_briefing_ui(
    focus_type: str,
    language: str,
    context: Optional[str] = None,
) -> str:
    sys = _system_prompt_strat(language)
    is_it = (language or "it").lower().startswith("it")

    ctx = ""
    if context and context.strip():
        ctx = f"\n\nContesto:\n{context.strip()}\n" if is_it else f"\n\nContext:\n{context.strip()}\n"

    if is_it:
        user_prompt = f"""
Scrivi un **Focus briefing** su **{focus_type}**.

{ctx}

Formato OBBLIGATORIO (Markdown), massimo ~10–14 frasi:

**Bottom line**
- 1 frase netta (regime/tema + rischio dominante)

**Driver**
- 4–6 righe: fondamentali + tecnici (posizionamento/vol/liquidità) se rilevanti

**Scenari**
- 3 bullet: **Base / Upside / Downside** → trigger → impatto qualitativo cross-asset

**Trigger da monitorare**
- 3 bullet “trigger → impatto”

Vincoli: prudente, niente numeri inventati, niente date/eventi inventati, no trade call.
"""
    else:
        user_prompt = f"""
Write a **Focus briefing** on **{focus_type}**.

{ctx}

MANDATORY format (Markdown), max ~10–14 sentences:

**Bottom line**
- 1 crisp sentence (regime/theme + dominant risk)

**Drivers**
- 4–6 lines: fundamentals + technicals (positioning/vol/liquidity) if relevant

**Scenarios**
- 3 bullets: **Base / Upside / Downside** → trigger → qualitative cross-asset impact

**Triggers to watch**
- 3 bullets “trigger → impact”

Constraints: risk-aware, no fabricated numbers, no invented dates/events, no trade calls.
"""

    out = chat_completion(
        system_prompt=sys,
        user_prompt=user_prompt,
        model=OPENAI_UI_MODEL,
        max_output_tokens=520,
    )
    return (out or "").strip()


def _generate_market_ai_ui(
    question: str,
    language: str,
    context: Optional[str] = None,
) -> str:
    sys = _system_prompt_strat(language)
    is_it = (language or "it").lower().startswith("it")

    ctx = ""
    if context and context.strip():
        ctx = f"\n\nContesto disponibile:\n{context.strip()}\n" if is_it else f"\n\nAvailable context:\n{context.strip()}\n"

    if is_it:
        user_prompt = f"""
Domanda: {question}

{ctx}

Rispondi in Markdown con questa struttura (senza consigli operativi):

**Risposta breve**
- 2–3 frasi

**Meccanismo**
- 4–6 frasi (perché succede, canali principali)

**Implicazioni (high level)**
- 3 bullet (solo concetti, niente “compra/vendi”)

**Rischi / cosa può invalidare lo scenario**
- 3 bullet
"""
    else:
        user_prompt = f"""
Question: {question}

{ctx}

Answer in Markdown with this structure (no trading instructions):

**Short answer**
- 2–3 sentences

**Mechanism**
- 4–6 sentences (why, key channels)

**High-level implications**
- 3 bullets (conceptual only)

**Risks / what could break the view**
- 3 bullets
"""

    out = chat_completion(
        system_prompt=sys,
        user_prompt=user_prompt,
        model=OPENAI_UI_MODEL,
        max_output_tokens=700,
    )
    return (out or "").strip()


# =========================================================
#   MARKETSTACK – CLIENT SEMPLICE EOD (robusto)
# =========================================================
def _load_close_from_marketstack_simple(
    ticker: str,
    start: dt.date,
    end: dt.date,
    api_key: Optional[str] = None,
) -> pd.Series:
    key = (api_key or MARKETSTACK_API_KEY or "").strip()
    if not key:
        raise RuntimeError("Marketstack API key non configurata (MARKETSTACK_API_KEY o st.secrets).")

    if start > end:
        raise RuntimeError("Intervallo date non valido (start > end).")

    params = {
        "access_key": key,
        "symbols": ticker.upper(),
        "date_from": start.strftime("%Y-%m-%d"),
        "date_to": end.strftime("%Y-%m-%d"),
        "limit": 1000,
        "sort": "ASC",
    }

    last_err: Optional[str] = None

    for base_url in MARKETSTACK_ENDPOINTS:
        frames: List[pd.DataFrame] = []
        offset = 0

        while True:
            params["offset"] = offset
            resp = requests.get(base_url, params=params, timeout=12)

            try:
                js = resp.json()
            except Exception:
                js = None

            # errori HTTP
            if resp.status_code != 200:
                last_err = f"[{base_url}] HTTP {resp.status_code}: {resp.text[:250]}"
                break

            # alcuni provider ritornano 200 ma con payload "error"
            if isinstance(js, dict) and js.get("error"):
                last_err = f"[{base_url}] Provider error: {js.get('error')}"
                break

            data = (js or {}).get("data") or []
            if not data:
                break

            frames.append(pd.DataFrame.from_records(data))

            pag = (js or {}).get("pagination") or {}
            count = pag.get("count", len(data))
            total = pag.get("total", count)

            if not isinstance(count, int) or count <= 0:
                break

            offset += count
            if isinstance(total, int) and offset >= total:
                break

        if frames:
            df = pd.concat(frames, ignore_index=True)

            if "date" not in df.columns:
                raise RuntimeError(f"Campo 'date' mancante per {ticker} nella risposta Marketstack.")

            # ✅ parsing date ROBUSTO:
            # - utc=True → evita index misti
            # - tz_convert(None) → rende tz-naive (datetime64[ns])
            # - normalize() → 00:00:00 per confronti (YTD, nearest, ecc.)
            date_raw = pd.to_datetime(df["date"], errors="coerce", utc=True)
            mask_valid = ~date_raw.isna()
            if not mask_valid.any():
                raise RuntimeError(f"Tutte le date sono invalide per {ticker} (Marketstack).")

            df = df.loc[mask_valid].copy()
            idx = pd.DatetimeIndex(date_raw[mask_valid]).tz_convert(None).normalize()
            df.index = idx
            df = df.sort_index()
            df = df[~df.index.duplicated(keep="last")]

            # --- scegli adj_close se disponibile, altrimenti close
            if "adj_close" in df.columns and df["adj_close"].notna().any():
                src = df["adj_close"]
            elif "close" in df.columns and df["close"].notna().any():
                src = df["close"]
            else:
                raise RuntimeError(f"Dati incompleti per {ticker}: manca adj_close/close valido.")

            s = pd.to_numeric(src, errors="coerce").dropna()
            if s.empty:
                raise RuntimeError(f"Serie close vuota dopo pulizia per {ticker}.")

            s.name = ticker.upper()
            return s

        # nessun frame: prova endpoint successivo
        continue

    if last_err:
        raise RuntimeError(f"Nessun dato per {ticker}. Ultimo errore: {last_err}")
    raise RuntimeError(f"Nessun dato EOD restituito da Marketstack per {ticker} (nessun record).")

@st.cache_data(ttl=900, show_spinner=False)
def _fetch_panel_close_for_research(
    tickers: List[str],
    start: dt.date,
    end: dt.date,
) -> Tuple[pd.DataFrame, Dict[str, str]]:
    if not tickers:
        return pd.DataFrame(), {}

    today = dt.date.today()
    if end > today:
        end = today

    if start > end:
        return pd.DataFrame(), {"__range__": "Intervallo date non valido (start > end)."}

    frames: List[pd.Series] = []
    errors: Dict[str, str] = {}

    uniq = sorted({t.strip().upper() for t in tickers if isinstance(t, str) and t.strip()})

    for t in uniq:
        try:
            s = _load_close_from_marketstack_simple(t, start, end)
            if s is None or s.empty:
                continue
            s.index = pd.to_datetime(s.index).normalize()
            s = s[~s.index.duplicated(keep="last")]
            frames.append(s.rename(t))
        except Exception as e:  # noqa: BLE001
            errors[t] = str(e)

    if not frames:
        return pd.DataFrame(), errors

    out = pd.concat(frames, axis=1).sort_index()

    ordered_cols = [
        t.strip().upper()
        for t in tickers
        if isinstance(t, str) and t.strip().upper() in out.columns
    ]
    out = out.loc[:, ordered_cols]

    return out.dropna(how="all"), errors


def _perf_change_window(df: pd.DataFrame, mode: str) -> pd.Series:
    if df is None or df.empty:
        return pd.Series(dtype=float)

    data = df.ffill().bfill()
    if data.empty:
        return pd.Series(dtype=float)

    last = data.iloc[-1]
    mode = (mode or "").upper()

    def _ref_for_mode() -> pd.Series:
        n = len(data)
        if mode == "1D" and n >= 2:
            return data.iloc[-2]
        if mode == "5D" and n >= 6:
            return data.iloc[-6]
        if mode == "1M" and n >= 22:
            return data.iloc[-22]
        if mode == "3M" and n >= 66:
            return data.iloc[-66]
        if mode == "6M" and n >= 126:
            return data.iloc[-126]
        if mode == "1Y" and n >= 252:
            return data.iloc[-252]
        if mode == "YTD":
            year_start = pd.Timestamp(dt.date(dt.date.today().year, 1, 1))
            idx = data.index.get_indexer([year_start], method="nearest")[0]
            return data.iloc[idx]
        return data.iloc[0]

    ref = _ref_for_mode()
    with np.errstate(divide="ignore", invalid="ignore"):
        out = (last / ref - 1.0) * 100.0
    return out.replace([np.inf, -np.inf], np.nan)


# =========================================================
#   HEATMAP (Marketstack)
# =========================================================
def plotly_express_treemap(universe: pd.DataFrame, heat_period: str, max_abs: float):
    import plotly.express as px

    fig = px.treemap(
        universe,
        path=["Gruppo", "Nome"],
        values="Size",
        color="Perf_%",
        range_color=[-max_abs, max_abs],
        color_continuous_scale="RdYlGn",
        custom_data=["Ticker", "Perf_%"],
        maxdepth=2,
    )
    fig.update_traces(
        texttemplate="<b>%{label}</b><br>%{customdata[0]}<br>%{customdata[1]:.2f}%",
        hovertemplate="<b>%{label}</b><br>Ticker: %{customdata[0]}<br>Perf: %{customdata[1]:.2f}%<extra></extra>",
        marker=dict(cornerradius=4),
    )
    fig.update_layout(
        height=420,
        margin=dict(l=0, r=0, t=30, b=0),
        coloraxis_colorbar=dict(
            title=f"{heat_period} %",
            ticksuffix="%",
            thickness=12,
        ),
    )
    return fig


def render_research_heatmap() -> None:
    st.subheader("🔎 Research – Heatmap (Marketstack)")

    try:
        import plotly.express as px  # noqa: F401
    except Exception:
        st.error("Installa dipendenze: `pip install plotly` (necessario per la heatmap)")
        return

    colA, colB, colC = st.columns([1, 1, 1])
    with colA:
        start_date = st.date_input(
            "Period: from",
            value=(dt.date.today() - dt.timedelta(days=365)),
            key="heat_start_date",
        )
    with colB:
        end_date = st.date_input(
            "Period: to",
            value=dt.date.today(),
            key="heat_end_date",
        )
    with colC:
        heat_period = st.selectbox(
            "Heatmap Color for",
            ["1D", "5D", "1M", "3M", "6M", "YTD", "1Y"],
            index=0,
            key="heat_period",
        )

    SECTORS = [
        {"Gruppo": "U.S. Sectors (ETF)", "Nome": "Consumer Discretionary", "Ticker": "XLY"},
        {"Gruppo": "U.S. Sectors (ETF)", "Nome": "Consumer Staples",       "Ticker": "XLP"},
        {"Gruppo": "U.S. Sectors (ETF)", "Nome": "Energy",                 "Ticker": "XLE"},
        {"Gruppo": "U.S. Sectors (ETF)", "Nome": "Financials",             "Ticker": "XLF"},
        {"Gruppo": "U.S. Sectors (ETF)", "Nome": "Health Care",            "Ticker": "XLV"},
        {"Gruppo": "U.S. Sectors (ETF)", "Nome": "Industrials",            "Ticker": "XLI"},
        {"Gruppo": "U.S. Sectors (ETF)", "Nome": "Materials",              "Ticker": "XLB"},
        {"Gruppo": "U.S. Sectors (ETF)", "Nome": "Real Estate",            "Ticker": "XLRE"},
        {"Gruppo": "U.S. Sectors (ETF)", "Nome": "Technology",             "Ticker": "XLK"},
        {"Gruppo": "U.S. Sectors (ETF)", "Nome": "Utilities",              "Ticker": "XLU"},
        {"Gruppo": "U.S. Sectors (ETF)", "Nome": "Communication Services", "Ticker": "XLC"},
    ]

    REGIONAL_ETF = [
        {"Gruppo": "Regional Equity ETFs", "Nome": "USA – S&P 500",   "Ticker": "SPY"},
        {"Gruppo": "Regional Equity ETFs", "Nome": "Developed ex-US", "Ticker": "EFA"},
        {"Gruppo": "Regional Equity ETFs", "Nome": "Emerging Mkts",   "Ticker": "EEM"},
        {"Gruppo": "Regional Equity ETFs", "Nome": "Japan",           "Ticker": "EWJ"},
        {"Gruppo": "Regional Equity ETFs", "Nome": "UK",              "Ticker": "EWU"},
        {"Gruppo": "Regional Equity ETFs", "Nome": "Germany",         "Ticker": "EWG"},
        {"Gruppo": "Regional Equity ETFs", "Nome": "China",           "Ticker": "MCHI"},
    ]

    MEGACAPS = [
        {"Gruppo": "Mega Caps", "Nome": "Apple",     "Ticker": "AAPL"},
        {"Gruppo": "Mega Caps", "Nome": "Microsoft", "Ticker": "MSFT"},
        {"Gruppo": "Mega Caps", "Nome": "NVIDIA",    "Ticker": "NVDA"},
        {"Gruppo": "Mega Caps", "Nome": "Amazon",    "Ticker": "AMZN"},
        {"Gruppo": "Mega Caps", "Nome": "Alphabet",  "Ticker": "GOOGL"},
        {"Gruppo": "Mega Caps", "Nome": "Meta",      "Ticker": "META"},
        {"Gruppo": "Mega Caps", "Nome": "Tesla",     "Ticker": "TSLA"},
    ]

    universe = pd.DataFrame(SECTORS + REGIONAL_ETF + MEGACAPS)
    tickers = [t for t in universe["Ticker"].unique() if isinstance(t, str)]

    prices, errors = _fetch_panel_close_for_research(tickers, start_date, end_date)

    if prices is None or prices.empty:
        st.error("Nessun dato disponibile da Marketstack per costruire la Heatmap nel periodo selezionato.")
        if errors:
            with st.expander("Dettagli errori Marketstack"):
                st.write(errors)
        return

    universe = universe[universe["Ticker"].isin(prices.columns)]
    if universe.empty:
        st.info("Nessun ticker dell’universo è stato trovato dal provider Marketstack.")
        return

    perf = _perf_change_window(prices[universe["Ticker"].unique()], heat_period)
    universe["Perf_%"] = pd.to_numeric(universe["Ticker"].map(perf), errors="coerce").round(2)
    universe = universe.dropna(subset=["Perf_%"]).copy()
    universe["Size"] = 1.0

    if universe.empty:
        st.info("Nessun dato valido per la Heatmap (tutte le performance sono NaN).")
        return

    max_abs = float(np.nanmax(np.abs(universe["Perf_%"].values)))
    if not np.isfinite(max_abs) or max_abs <= 0:
        max_abs = 1.0

    fig = plotly_express_treemap(universe, heat_period, max_abs)
    st.plotly_chart(fig, use_container_width=True, theme=None)

    if errors:
        st.caption("Alcuni simboli non sono stati caricati: " + ", ".join(sorted(errors.keys())))

    st.caption("Fonte: Marketstack (client semplificato lato Research).")


# =========================================================
#   MARKET DASHBOARD
# =========================================================
def render_market_dashboard() -> None:
    st.subheader("📊 Research – Market Dashboard (Marketstack)")

    try:
        import plotly.graph_objects as go
    except Exception:
        st.error("Installa dipendenze: `pip install plotly` (necessario per la dashboard)")
        return

    colA, colB = st.columns([1, 1])
    with colA:
        start_date = st.date_input(
            "Period: from",
            value=(dt.date.today() - dt.timedelta(days=365)),
            key="dash_start_date",
        )
    with colB:
        end_date = st.date_input(
            "Period: to",
            value=dt.date.today(),
            key="dash_end_date",
        )

    INDEX_MAP = {
        "SPX (USA)":        "SPY",
        "Nasdaq 100":       "QQQ",
        "Dow Jones":        "DIA",
        "Developed ex-US":  "EFA",
        "Emerging Mkts":    "EEM",
        "Japan":            "EWJ",
        "UK":               "EWU",
        "Germany":          "EWG",
        "China":            "MCHI",
    }

    SECTORS = [
        {"Settore": "Cons. Discretionary", "Ticker": "XLY"},
        {"Settore": "Cons. Staples",       "Ticker": "XLP"},
        {"Settore": "Energy",              "Ticker": "XLE"},
        {"Settore": "Financials",          "Ticker": "XLF"},
        {"Settore": "Health Care",         "Ticker": "XLV"},
        {"Settore": "Industrials",         "Ticker": "XLI"},
        {"Settore": "Materials",           "Ticker": "XLB"},
        {"Settore": "Real Estate",         "Ticker": "XLRE"},
        {"Settore": "Technology",          "Ticker": "XLK"},
        {"Settore": "Utilities",           "Ticker": "XLU"},
        {"Settore": "Comm. Services",      "Ticker": "XLC"},
    ]

    INDUSTRIES = ["SMH", "IGV", "XOP", "XRT", "XHB", "XBI", "ITA", "IYR"]

    BONDS = [
        {"Categoria": "Treasury 1–3 anni",     "Ticker": "SHY"},
        {"Categoria": "Treasury 7–10 anni",    "Ticker": "IEF"},
        {"Categoria": "Treasury 20+ anni",     "Ticker": "TLT"},
        {"Categoria": "Aggregate Bonds",       "Ticker": "AGG"},
        {"Categoria": "Investment Grade Corp", "Ticker": "LQD"},
        {"Categoria": "High Yield (Junk)",     "Ticker": "HYG"},
        {"Categoria": "TIPS (Inflation-link)", "Ticker": "TIP"},
    ]

    FX = [
        {"Pair": "EUR/USD", "Ticker": "EURUSD"},
        {"Pair": "GBP/USD", "Ticker": "GBPUSD"},
        {"Pair": "USD/JPY", "Ticker": "USDJPY"},
        {"Pair": "USD/CHF", "Ticker": "USDCHF"},
    ]

    COMMODS = [
        {"Nome": "Gold (GLD ETF)",     "Ticker": "GLD"},
        {"Nome": "Oil (USO ETF)",      "Ticker": "USO"},
        {"Nome": "Silver (SLV ETF)",   "Ticker": "SLV"},
        {"Nome": "Agriculture (DBA)",  "Ticker": "DBA"},
    ]

    def _norm_100(df: pd.DataFrame) -> pd.DataFrame:
        df = df.ffill().bfill()
        if df.empty:
            return df
        base = df.iloc[0].replace(0, np.nan)
        return (df.divide(base)).multiply(100.0)

    def _plot_lines(df_norm: pd.DataFrame, title: str):
        df_norm = df_norm.dropna(how="all")
        if df_norm.empty:
            st.info(f"Nessun dato per: {title}")
            return
        fig = go.Figure()
        for col in df_norm.columns:
            fig.add_trace(go.Scatter(x=df_norm.index, y=df_norm[col], mode="lines", name=col))
        fig.update_layout(
            title=title,
            height=320,
            legend=dict(orientation="h", y=-0.2),
            margin=dict(l=10, r=10, t=40, b=10),
            yaxis_title="Index (Base=100)",
        )
        st.plotly_chart(fig, use_container_width=True, theme=None)

    indices_symbols  = [s for s in INDEX_MAP.values() if s]
    sector_symbols   = [s["Ticker"] for s in SECTORS]
    industry_symbols = INDUSTRIES
    bond_symbols     = [s["Ticker"] for s in BONDS]
    fx_symbols       = [s["Ticker"] for s in FX]
    cmd_symbols      = [s["Ticker"] for s in COMMODS]

    all_symbols = list(set(indices_symbols + sector_symbols + industry_symbols + bond_symbols + fx_symbols + cmd_symbols))

    prices_all, errors_all = _fetch_panel_close_for_research(all_symbols, start_date, end_date)
    if prices_all is None or prices_all.empty:
        st.error("Nessun dato disponibile da Marketstack per costruire la dashboard.")
        if errors_all:
            with st.expander("Dettagli errori Marketstack"):
                st.write(errors_all)
        return

    if errors_all:
        st.caption("Ticker con errori/assenza dati: " + ", ".join(sorted(errors_all.keys())))

    st.markdown("### 📈 Global Indices & Regional ETFs – Comparison (Base=100)")
    idx_pairs = [(label, sym) for label, sym in INDEX_MAP.items() if sym in prices_all]
    idx_df = pd.DataFrame({label: prices_all[sym] for label, sym in idx_pairs}) if idx_pairs else pd.DataFrame()
    _plot_lines(_norm_100(idx_df), "Global Indices & Regional ETFs")

    st.markdown("### 🧭 Sectors & Industries (Base=100)")
    left, right = st.columns(2)

    with left:
        sec_pairs = [(s["Settore"], s["Ticker"]) for s in SECTORS if s["Ticker"] in prices_all]
        sec_df = pd.DataFrame({name: prices_all[tkr] for (name, tkr) in sec_pairs}) if sec_pairs else pd.DataFrame()
        _plot_lines(_norm_100(sec_df), "U.S. Sectors (SPDR)")

    with right:
        ind_cols = {t: prices_all[t] for t in INDUSTRIES if t in prices_all}
        ind_df = pd.DataFrame(ind_cols) if ind_cols else pd.DataFrame()
        _plot_lines(_norm_100(ind_df), "Industries / Thematic ETFs")

    st.markdown("### 🧾 Bond – ETF (Base=100)")
    bond_pairs = [(b["Categoria"], b["Ticker"]) for b in BONDS if b["Ticker"] in prices_all]
    bond_df = pd.DataFrame({name: prices_all[tkr] for (name, tkr) in bond_pairs}) if bond_pairs else pd.DataFrame()
    _plot_lines(_norm_100(bond_df), "Bond ETFs")

    st.markdown("### 💱 FX – Major Pairs (Base=100)")
    fx_pairs = [(f["Pair"], f["Ticker"]) for f in FX if f["Ticker"] in prices_all]
    fx_df = pd.DataFrame({name: prices_all[tkr] for (name, tkr) in fx_pairs}) if fx_pairs else pd.DataFrame()
    if fx_df.empty:
        st.info("FX non disponibile (spesso non coperto dal piano/endpoint).")
    else:
        _plot_lines(_norm_100(fx_df), "Forex majors (proxy)")

    st.markdown("### 🛢️ Commodities – Proxy ETFs (Base=100)")
    cmd_pairs = [(c["Nome"], c["Ticker"]) for c in COMMODS if c["Ticker"] in prices_all]
    cmd_df = pd.DataFrame({name: prices_all[tkr] for (name, tkr) in cmd_pairs}) if cmd_pairs else pd.DataFrame()
    _plot_lines(_norm_100(cmd_df), "Commodities (proxy via ETFs)")

    st.caption("Fonte: Marketstack (client semplificato lato Research).")


# =========================================================
#   PAGINA PRINCIPALE
# =========================================================
def render_research_page() -> None:
    """Pagina Research con Briefing (core), Focus + MarketAI (UI), Pulse + Heatmap + Dashboard."""

    _inject_research_css()

    # Locale IT per data (se disponibile)
    try:
        locale.setlocale(locale.LC_TIME, "it_IT.UTF-8")
    except Exception:
        try:
            locale.setlocale(locale.LC_TIME, "it_IT")
        except Exception:
            pass

    today_str = datetime.now().strftime("%A %d %B %Y").title()

    st.subheader("🔍 Research")
    st.caption("Briefing di mercato, focus tematici, Market AI e dashboard di mercato.")
    st.markdown(f"🗓 **{today_str}**")

    tab_ai, tab_market = st.tabs(["📑 Briefing & AI", "📊 Market Dashboard & Heatmap"])

    # =========================================
    #   TAB 1 – BRIEFING + FOCUS + AI + PULSE
    # =========================================
    with tab_ai:
        col_left, col_right = st.columns([1.55, 1.0], gap="large")

        # ---------- SINISTRA: Briefing + Focus + Market AI ----------
        with col_left:
            st.markdown("### 🧠 Market Briefing")

            if "research_briefing_cache" not in st.session_state:
                st.session_state["research_briefing_cache"] = {}
            if "research_briefing_last_ts" not in st.session_state:
                st.session_state["research_briefing_last_ts"] = {}

            pulse_text = st.session_state.get("research_pulse_text", "")

            c1, c2, c3 = st.columns([1.25, 0.85, 1.0], gap="small")

            with c1:
                briefing_type = st.radio(
                    "Tipo di briefing",
                    options=["Giornaliero", "Settimanale", "Mensile", "Azioni", "Obbligazioni", "Macro", "Geopolitico"],
                    index=0,
                    horizontal=True,
                    key="br_type",
                )

            with c2:
                lang_label = st.radio(
                    "Lingua",
                    options=["Italiano", "English"],
                    index=0,
                    horizontal=True,
                    key="br_lang",
                )
                language = "it" if lang_label == "Italiano" else "en"

            with c3:
                use_pulse = st.checkbox("Usa Pulse come contesto", value=True, key="br_use_pulse")
                gen_brief = st.button("🔄 Genera / Aggiorna", use_container_width=True, key="br_gen")

            cache_key = f"{language}::{briefing_type}"

            if gen_brief:
                try:
                    ctx = pulse_text if (use_pulse and pulse_text) else None
                    with st.spinner("Genero il briefing..."):
                        txt = generate_market_briefing(
                            briefing_type=briefing_type,
                            language=language,
                            context=ctx,
                        )
                    st.session_state["research_briefing_cache"][cache_key] = txt
                    st.session_state["research_briefing_last_ts"][cache_key] = datetime.now().strftime("%Y-%m-%d %H:%M")
                except Exception as e:
                    st.error(f"Impossibile generare il briefing: {e}")

            st.markdown("---")

            text_to_show = st.session_state["research_briefing_cache"].get(cache_key, "")
            last_ts = st.session_state["research_briefing_last_ts"].get(cache_key)

            if last_ts:
                st.caption(f"Last generated: **{last_ts}**")

            if text_to_show:
                st.download_button(
                    "⬇️ Scarica briefing (.md)",
                    data=text_to_show.encode("utf-8"),
                    file_name=f"market_briefing_{language}_{briefing_type.lower()}_{dt.date.today().isoformat()}.md",
                    mime="text/markdown",
                    use_container_width=True,
                )
                st.markdown(text_to_show)
            else:
                st.info("Premi **Genera / Aggiorna** (meglio se prima aggiorni Pulse a destra).")

            # ---------- Focus briefing (ALLINEATO) ----------
            st.markdown("---")
            st.markdown("### 🎯 Focus briefing")

            # colonne allineate in basso (se supportato)
            cf1, cf2 = _cols([1.35, 0.65], gap="small", valign="bottom")

            with cf1:
                focus_type = st.selectbox(
                    "Tema del focus",
                    options=["Macro", "Azioni", "Obbligazioni", "Geopolitico", "Cripto"],
                    index=0,
                    key="focus_type",
                )

            with cf2:
                # fallback: se vertical_alignment non esiste, aggiungiamo un piccolo spacer
                try:
                    sig = inspect.signature(st.columns)
                    has_valign = "vertical_alignment" in sig.parameters
                except Exception:
                    has_valign = False
                if not has_valign:
                    st.markdown("<div style='height: 1.55rem'></div>", unsafe_allow_html=True)

                gen_focus = st.button("Genera Focus", use_container_width=True, key="focus_gen")

            if gen_focus:
                try:
                    ctx = text_to_show if text_to_show else pulse_text
                    with st.spinner("Genero Focus briefing..."):
                        fb = _generate_focus_briefing_ui(focus_type=focus_type, language=language, context=ctx)
                    st.session_state["research_focus_briefing"] = fb
                except Exception as e:
                    st.error(f"Errore Focus briefing: {e}")

            focus_text = st.session_state.get("research_focus_briefing", "")
            if focus_text:
                st.markdown(focus_text)

            # ---------- Market AI ----------
            st.markdown("---")
            st.markdown("### 💡 Market AI")

            use_ctx = st.checkbox("Usa briefing + pulse come contesto", value=True, key="ai_use_ctx")
            question = st.text_area(
                "Domanda",
                placeholder="Es. 'Cosa implica un bull steepening per equity e credito?'",
                key="ai_question",
            )

            if st.button("Chiedi a Market AI", use_container_width=True, key="ai_ask"):
                if not question.strip():
                    st.warning("Inserisci una domanda prima di inviare.")
                else:
                    try:
                        ctx = None
                        if use_ctx:
                            parts = []
                            if pulse_text:
                                parts.append("Market Pulse:\n" + pulse_text)
                            if text_to_show:
                                parts.append("Briefing:\n" + text_to_show)
                            ctx = "\n\n".join(parts) if parts else None

                        with st.spinner("Genero la risposta..."):
                            ans = _generate_market_ai_ui(question=question, language=language, context=ctx)
                        st.session_state["research_ai_answer"] = ans
                    except Exception as e:
                        st.error(f"Errore Market AI: {e}")
                        st.session_state["research_ai_answer"] = ""

            ai_answer = st.session_state.get("research_ai_answer", "")
            if ai_answer:
                st.markdown("#### Risposta")
                st.markdown(ai_answer)

        # ---------- DESTRA: Market Pulse ----------
        with col_right:
            st.markdown("### 📌 Market Pulse (Marketstack)")
            st.caption("Snapshot sintetico per ridurre briefing generici. Premi **Aggiorna Pulse** prima del briefing.")

            pulse_safe = ["SPY", "QQQ", "EFA", "EEM", "TLT", "IEF", "LQD", "HYG", "GLD", "USO"]
            use_fx = st.checkbox("Include FX (se supportato dal piano)", value=False, key="pulse_use_fx")
            pulse_tickers = pulse_safe + (["EURUSD", "USDJPY"] if use_fx else [])

            end_date = dt.date.today()
            start_date = end_date - dt.timedelta(days=420)

            if st.button("📡 Aggiorna Pulse", use_container_width=True, key="pulse_refresh"):
                prices, errors = _fetch_panel_close_for_research(pulse_tickers, start_date, end_date)

                if prices is None or prices.empty:
                    st.session_state["research_pulse_df"] = pd.DataFrame()
                    st.session_state["research_pulse_text"] = ""

                    st.warning("Pulse non disponibile (nessun dato dal provider per questo set).")

                    if not MARKETSTACK_API_KEY:
                        st.error("MARKETSTACK_API_KEY non trovata. Impostala in env o in `.streamlit/secrets.toml`.")

                    if errors:
                        with st.expander("Dettagli errori Marketstack"):
                            st.write(errors)
                    else:
                        st.caption("Nessun errore esplicito: possibile piano/endpoint non compatibile o nessun record nel range.")

                else:
                    win_labels = ["1D", "5D", "1M", "YTD", "1Y"]
                    perf = {w: _perf_change_window(prices, w) for w in win_labels}
                    df = pd.DataFrame(perf).round(2)
                    df.index.name = "Ticker"
                    df = df.replace([np.inf, -np.inf], np.nan)

                    st.session_state["research_pulse_df"] = df

                    # testo contesto per LLM (fatti calcolati)
                    lines = []
                    for t in df.index:
                        r1 = df.loc[t, "1D"] if "1D" in df.columns else np.nan
                        r5 = df.loc[t, "5D"] if "5D" in df.columns else np.nan
                        rM = df.loc[t, "1M"] if "1M" in df.columns else np.nan

                        def f(x: float) -> str:
                            return f"{x:+.2f}%" if np.isfinite(x) else "n/a"

                        if np.isfinite(r1) or np.isfinite(r5) or np.isfinite(rM):
                            lines.append(f"- {t}: 1D {f(r1)}, 5D {f(r5)}, 1M {f(rM)}")

                    st.session_state["research_pulse_text"] = "\n".join(lines)

                    if errors:
                        st.caption("Ticker con errori/assenza dati: " + ", ".join(sorted(errors.keys())))

            df_pulse = st.session_state.get("research_pulse_df", pd.DataFrame())
            if df_pulse is None or df_pulse.empty:
                st.info("Premi **Aggiorna Pulse** per caricare lo snapshot.")
            else:
                st.dataframe(df_pulse, use_container_width=True, height=320)
                st.caption("Suggerimento: abilita “Usa Pulse come contesto” quando generi il briefing.")

    # =========================================
    #   TAB 2 – MARKET DASHBOARD & HEATMAP
    # =========================================
    with tab_market:
        render_research_heatmap()
        st.markdown("---")
        render_market_dashboard()
