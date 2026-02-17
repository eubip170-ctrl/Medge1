# ui/analisi_page.py
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import json
import os
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import requests

from core.portfolio_analisi import normalize_series_daily, normalize_df_daily
from core.analisi_metrics import (
    metric_specs,
    compute_metrics_table,
    compute_asset_correlation,
)

AN_PREFIX = "AN"


def _k(name: str) -> str:
    return f"{AN_PREFIX}_{name}"


def _inject_css() -> None:
    st.markdown(
        """
<style>
.an-card{
  border: 1px solid rgba(0,0,0,0.12);
  background: rgba(255,255,255,0.55);
  border-radius: 12px;
  padding: 0.8rem 0.9rem;
}
.an-title{ font-size: 1.15rem; font-weight: 800; margin: 0 0 0.35rem 0; }
.an-sub{ opacity: 0.70; font-size: 0.92rem; margin: 0 0 0.2rem 0; }
</style>
        """,
        unsafe_allow_html=True,
    )


def _plotly_layout(fig: go.Figure, height: int = 440, dark: bool = False) -> go.Figure:
    if dark:
        paper = "#0E1521"
        plot = "#0E1521"
        grid = "rgba(148,163,184,0.10)"
        fontc = "#E5E7EB"
    else:
        paper = "rgba(0,0,0,0)"
        plot = "rgba(0,0,0,0)"
        grid = "rgba(0,0,0,0.12)"
        fontc = None

    fig.update_layout(
        height=height,
        margin=dict(l=10, r=10, t=28, b=10),
        paper_bgcolor=paper,
        plot_bgcolor=plot,
        legend=dict(orientation="h", y=-0.22, x=0.0),
        font=dict(size=12, family="Arial, Helvetica, sans-serif", color=fontc),
    )
    fig.update_xaxes(showgrid=True, gridcolor=grid, zeroline=False)
    fig.update_yaxes(showgrid=True, gridcolor=grid, zeroline=False)
    return fig


def _base100(s: pd.Series) -> pd.Series:
    s = normalize_series_daily(s).dropna().astype(float)
    if len(s) < 2:
        return pd.Series(dtype=float)
    b = float(s.iloc[0])
    if not np.isfinite(b) or b == 0:
        return pd.Series(dtype=float)
    return (s / b) * 100.0


# ---------- color utilities (rosso -> bianco -> verde) ----------
def _hex_to_rgb(h: str) -> Tuple[int, int, int]:
    h = h.lstrip("#")
    return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)


def _rgb_to_hex(rgb: Tuple[int, int, int]) -> str:
    return "#{:02x}{:02x}{:02x}".format(*rgb)


def _lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


def _blend(c1: str, c2: str, t: float) -> str:
    r1, g1, b1 = _hex_to_rgb(c1)
    r2, g2, b2 = _hex_to_rgb(c2)
    rgb = (
        int(_lerp(r1, r2, t)),
        int(_lerp(g1, g2, t)),
        int(_lerp(b1, b2, t)),
    )
    return _rgb_to_hex(rgb)


def _score_to_bg(score_0_1: float) -> str:
    red = "#fecaca"
    white = "#ffffff"
    green = "#bbf7d0"

    s = float(score_0_1)
    s = 0.5 if not np.isfinite(s) else max(0.0, min(1.0, s))

    if s < 0.5:
        return _blend(red, white, s / 0.5)
    return _blend(white, green, (s - 0.5) / 0.5)


def _style_metrics_table(df: pd.DataFrame) -> "pd.io.formats.style.Styler":
    specs = {s.label: s for s in metric_specs()}

    def row_style(row: pd.Series) -> List[str]:
        label = str(row.name)
        spec = specs.get(label, None)

        vals = pd.to_numeric(row, errors="coerce").astype(float)
        out = ["background-color: #ffffff" for _ in vals.index]

        finite = vals[np.isfinite(vals)]
        if finite.empty or spec is None:
            return out

        vmin = float(finite.min())
        vmax = float(finite.max())
        denom = (vmax - vmin) if np.isfinite(vmax - vmin) and (vmax - vmin) > 0 else np.nan

        for i, col in enumerate(vals.index):
            v = float(vals.loc[col]) if np.isfinite(vals.loc[col]) else np.nan
            if not np.isfinite(v):
                out[i] = "background-color: #ffffff"
                continue

            if spec.neutral_band is not None:
                lo, hi = spec.neutral_band
                if np.isfinite(lo) and np.isfinite(hi) and (lo <= v <= hi):
                    out[i] = "background-color: #ffffff"
                    continue

            if not np.isfinite(denom):
                out[i] = "background-color: #ffffff"
                continue

            score = (v - vmin) / denom
            if not spec.higher_is_better:
                score = 1.0 - score

            bg = _score_to_bg(score)
            out[i] = f"background-color: {bg};"
        return out

    sty = df.style.apply(row_style, axis=1)

    idx = pd.IndexSlice
    for sp in metric_specs():
        if sp.label not in df.index:
            continue
        subset = idx[sp.label, :]
        if sp.fmt == "pct":
            sty = sty.format(lambda x: "–" if not np.isfinite(x) else f"{x*100:,.2f}%", subset=subset)
        elif sp.fmt == "ratio":
            sty = sty.format(lambda x: "–" if not np.isfinite(x) else f"{x:,.3f}", subset=subset)
        else:
            sty = sty.format(lambda x: "–" if not np.isfinite(x) else f"{x:,.3f}", subset=subset)

    return sty


def _plot_lines(df: pd.DataFrame, ytitle: str, yfmt: str, height: int = 460, dark: bool = False) -> go.Figure:
    fig = go.Figure()
    if df is None or df.empty:
        return _plotly_layout(fig, height=height, dark=dark)

    for col in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df[col],
                mode="lines",
                name=str(col),
                line=dict(width=2),
                hovertemplate="%{x|%Y-%m-%d}<br><b>%{fullData.name}</b>: %{y:" + yfmt + "}<extra></extra>",
            )
        )
    fig.update_yaxes(title=ytitle)
    return _plotly_layout(fig, height=height, dark=dark)


def _plot_heatmap_corr(corr: pd.DataFrame, height: int = 560, dark: bool = False) -> go.Figure:
    fig = go.Figure()
    if corr is None or corr.empty:
        return _plotly_layout(fig, height=height, dark=dark)

    labels = corr.columns.astype(str).tolist()
    fig.add_trace(
        go.Heatmap(
            z=corr.values,
            x=labels,
            y=labels,
            zmin=-1,
            zmax=1,
            zmid=0,
            colorscale="RdBu",
            reversescale=False,
            colorbar=dict(thickness=12, title="ρ"),
            hovertemplate="x=%{x}<br>y=%{y}<br>ρ=%{z:.2f}<extra></extra>",
        )
    )
    fig.update_xaxes(side="top", showgrid=False, tickangle=45, tickfont=dict(size=10))
    fig.update_yaxes(showgrid=False, tickfont=dict(size=10))
    return _plotly_layout(fig, height=height, dark=dark)


# ===========================
# Extra helpers: drawdown
# ===========================
def _drawdown_from_equity(eq: pd.Series) -> pd.Series:
    s = pd.Series(eq).astype(float).dropna()
    if len(s) < 2:
        return pd.Series(dtype=float)
    peak = s.cummax()
    dd = (s / peak) - 1.0
    return dd.replace([np.inf, -np.inf], np.nan).dropna()


def _plot_drawdown(dd: pd.Series, title: str = "Drawdown", height: int = 260, dark: bool = False) -> go.Figure:
    fig = go.Figure()
    if dd is None or dd.empty:
        return _plotly_layout(fig, height=height, dark=dark)

    fig.add_trace(
        go.Scatter(
            x=dd.index,
            y=dd.values,
            mode="lines",
            name=title,
            line=dict(width=2),
            fill="tozeroy",
            hovertemplate="%{x|%Y-%m-%d}<br>DD: %{y:.2%}<extra></extra>",
        )
    )
    fig.update_yaxes(title="Drawdown", tickformat=".0%")
    return _plotly_layout(fig, height=height, dark=dark)


# ===========================
# Marketstack helpers
# ===========================
def _secrets_get(key: str, default: str = "") -> str:
    try:
        v = st.secrets.get(key, default)
        return str(v).strip() if v is not None else default
    except Exception:
        return default


def _marketstack_key() -> str:
    return (_secrets_get("MARKETSTACK_API_KEY") or os.getenv("MARKETSTACK_API_KEY", "")).strip()


def _marketstack_base_url() -> str:
    # se un domani passi a v2: metti MARKETSTACK_BASE_URL nei secrets
    return (_secrets_get("MARKETSTACK_BASE_URL") or os.getenv("MARKETSTACK_BASE_URL", "https://api.marketstack.com/v1")).strip().rstrip("/")


@st.cache_data(ttl=60 * 60 * 12, show_spinner=False)
def _marketstack_get(path: str, params: dict) -> dict:
    key = _marketstack_key()
    if not key:
        return {"_error": "MARKETSTACK_API_KEY mancante."}

    base = _marketstack_base_url()
    url = f"{base}/{path.lstrip('/')}"
    qp = dict(params or {})
    # marketstack v1 usa access_key
    qp["access_key"] = key

    try:
        r = requests.get(url, params=qp, timeout=15)
        try:
            data = r.json()
        except Exception:
            data = {"_error": f"Risposta non JSON (HTTP {r.status_code})."}
        if r.status_code != 200:
            return {"_error": f"HTTP {r.status_code}", "_raw": data}
        return data if isinstance(data, dict) else {"data": data}
    except Exception as e:
        return {"_error": f"{type(e).__name__}: {e}"}


@st.cache_data(ttl=60 * 60 * 12, show_spinner=False)
def marketstack_ticker_info(symbol: str) -> dict:
    # v1: /tickers?search=... oppure /tickers/{symbol}
    # Provo prima endpoint diretto, poi fallback su search
    sym = str(symbol).strip().upper()
    d1 = _marketstack_get(f"tickers/{sym}", params={})
    if isinstance(d1, dict) and not d1.get("_error"):
        return d1
    d2 = _marketstack_get("tickers", params={"search": sym, "limit": 5})
    return d2


# ===========================
# OpenAI / AI analysis helper
# ===========================
def _openai_model_default() -> str:
    return (_secrets_get("OPENAI_MODEL") or os.getenv("OPENAI_MODEL", "gpt-5-mini")).strip()


@st.cache_data(ttl=60 * 60 * 6, show_spinner=False)
def _ai_analyze_cached(ticker: str, payload_json: str, model: str) -> str:
    # Import lazy: se openai non c'è, non rompe tutta la pagina
    from core.ai_client import chat_completion  # noqa

    system_prompt = (
        "Sei un analista finanziario prudente. "
        "Fornisci analisi sintetica, punti chiave, rischi, scenario base, e considerazioni sul portafoglio. "
        "Non inventare dati: usa solo le info fornite."
    )
    user_prompt = (
        f"TICKER: {ticker}\n"
        f"CONTESTO (JSON):\n{payload_json}\n\n"
        "Output richiesto:\n"
        "1) Snapshot (1 riga)\n"
        "2) Punti chiave (max 6 bullet)\n"
        "3) Rischi (max 5 bullet)\n"
        "4) Come incide in portafoglio (max 5 bullet)\n"
        "5) Nota finale: cosa verificheresti con dati fondamentali/news\n"
    )

    return chat_completion(system_prompt=system_prompt, user_prompt=user_prompt, model=model, max_output_tokens=650)


# ===========================
# Monte Carlo (senza yfinance)
# ===========================
def _ensure_psd_cov(cov: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    cov = (cov + cov.T) / 2.0
    w, v = np.linalg.eigh(cov)
    w = np.maximum(w, eps)
    return (v * w) @ v.T


def _simulate_increments(
    logret_hist: np.ndarray,  # (T, n)
    days: int,
    sims: int,
    seed: int,
    method: str,  # "Bootstrap" | "GBM" | "t-copula"
    drift_scale: float,
    t_df: int,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    T, n = logret_hist.shape
    R = logret_hist.astype(np.float32)

    if method == "Bootstrap":
        idx = rng.integers(0, T, size=(days, sims))
        return R[idx] * np.float32(drift_scale)

    mu = R.mean(axis=0).astype(np.float32) * np.float32(drift_scale)
    cov = np.cov(R, rowvar=False).astype(np.float64)
    cov = _ensure_psd_cov(cov)
    L = np.linalg.cholesky(cov).astype(np.float32)

    Z = rng.standard_normal((days, sims, n)).astype(np.float32)
    base = mu + (Z @ L.T)

    if method == "t-copula":
        df = max(3, int(t_df))
        chi2 = rng.chisquare(df, size=(days, sims, 1)).astype(np.float32)
        scale = np.sqrt(df / chi2)
        base = mu + (Z @ L.T) * scale

    return base


def _equity_from_increments(
    inc: np.ndarray,          # (days, sims, n_assets)
    last_prices: np.ndarray,  # (n_assets,)
    weights: np.ndarray,      # (n_assets,)
    capital: float,
    mode: str,                # Rebalance | Buy&Hold
) -> np.ndarray:
    days, sims, n = inc.shape
    w = weights.astype(np.float32)
    w = w / np.sum(w)
    capital = np.float32(capital)

    equity = np.empty((days + 1, sims), dtype=np.float32)
    equity[0, :] = capital

    if mode == "Rebalance":
        port_r = np.expm1(inc) @ w
        equity[1:, :] = capital * np.cumprod(1.0 + port_r, axis=0)
        return equity

    S0 = last_prices.astype(np.float32)
    shares = (capital * w) / S0
    prices = np.broadcast_to(S0, (sims, n)).copy()
    for t in range(days):
        prices *= np.exp(inc[t])
        equity[t + 1, :] = prices @ shares
    return equity


def _compute_bands(paths: np.ndarray, alpha: float) -> dict:
    return {
        "mean": paths.mean(axis=1),
        "med": np.quantile(paths, 0.50, axis=1),
        "lo":  np.quantile(paths, alpha, axis=1),
        "hi":  np.quantile(paths, 1.0 - alpha, axis=1),
    }


def _cvar(vals: np.ndarray, alpha: float) -> float:
    q = np.quantile(vals, alpha)
    tail = vals[vals <= q]
    return float(tail.mean()) if len(tail) else float(q)


# =========================================================
# PAGE
# =========================================================
def render_analisi_page(res: Optional[dict] = None, state: Any = None) -> None:
    _inject_css()

    state_dict = state if isinstance(state, dict) else st.session_state
    if res is None and isinstance(state_dict, dict):
        res = state_dict.get("res")

    runs = state_dict.get("portfolio_runs", {}) if isinstance(state_dict, dict) else {}

    if not isinstance(res, dict) or res.get("equity") is None:
        st.info("Nessun risultato disponibile. Esegui prima il run del portafoglio.")
        return

    # --- active (fallback) ---
    eq_active = normalize_series_daily(res.get("equity")).dropna()
    prices_active = normalize_df_daily(res.get("prices")) if isinstance(res.get("prices"), pd.DataFrame) else pd.DataFrame()

    # --- portfolio series map (multi) ---
    port_map: Dict[str, pd.Series] = {}
    if isinstance(runs, dict) and runs:
        for pid, r in runs.items():
            try:
                nm = str(r.get("name", pid))
                eq = r.get("equity")
                if isinstance(eq, pd.Series) and not eq.empty:
                    port_map[nm] = normalize_series_daily(eq).dropna()
            except Exception:
                pass

    if not port_map and not eq_active.empty:
        port_map["PORT"] = eq_active

    # --- asset series map (solo dal portafoglio attivo) ---
    asset_map: Dict[str, pd.Series] = {}
    if isinstance(prices_active, pd.DataFrame) and not prices_active.empty:
        for c in prices_active.columns:
            s = prices_active[c]
            if isinstance(s, pd.Series):
                s = normalize_series_daily(s).dropna()
                if len(s) > 2:
                    asset_map[str(c).upper()] = s

    st.markdown(
        "<div class='an-card'><div class='an-title'>Analisi</div>"
        "<div class='an-sub'>Grafico comparativo + tabella metriche unica (colorata) + correlazione titoli leggibile</div></div>",
        unsafe_allow_html=True,
    )

    # =========================================================
    # 1) GRAFICO IN ALTO
    # =========================================================
    st.markdown("### Confronto (grafico)")

    scope = st.radio(
        "Cosa vuoi confrontare?",
        ["Portafogli", "Titoli (portafoglio attivo)", "Portafogli + Titoli"],
        horizontal=True,
        key=_k("scope_compare"),
    )

    view_mode = st.radio(
        "Vista",
        ["Base=100", "$10.000"],
        horizontal=True,
        key=_k("view_mode"),
    )

    series_map: Dict[str, pd.Series] = {}
    if scope in ("Portafogli", "Portafogli + Titoli"):
        series_map.update(port_map)
    if scope in ("Titoli (portafoglio attivo)", "Portafogli + Titoli"):
        series_map.update(asset_map)

    if not series_map:
        st.info("Niente da plottare (mancano portafogli/titoli).")
    else:
        base100_df = pd.DataFrame({k: _base100(v) for k, v in series_map.items()}).dropna(how="all").sort_index()
        base100_df = base100_df.ffill().dropna(how="all")

        if base100_df.empty:
            st.info("Serie insufficienti per costruire il grafico.")
        else:
            plot_df = base100_df.copy()
            if view_mode == "$10.000":
                plot_df = (plot_df / 100.0) * 10_000.0

            fig = _plot_lines(
                plot_df,
                ytitle=("Value" if view_mode == "$10.000" else "Index (Base=100)"),
                yfmt=".2f",
                height=480,
                dark=False,
            )
            st.plotly_chart(fig, use_container_width=True, theme=None, key=_k("plot_compare_top"))

    st.markdown("---")

    # =========================================================
    # 2) METRICHE: TABELLA UNICA + COLORI
    # =========================================================
    st.markdown("### Metriche (tabella unica)")

    bench_name = "BENCH" if "BENCH" in series_map else None
    rf_annual = float(state_dict.get("portfolio_global_params", {}).get("rf_annual", 0.0)) if isinstance(state_dict, dict) else 0.0

    mt = compute_metrics_table(series_map=series_map, rf_annual=rf_annual, bench_name=bench_name)
    if mt is None or mt.empty:
        st.info("Metriche non disponibili (serie insufficienti).")
    else:
        st.dataframe(
            _style_metrics_table(mt),
            use_container_width=True,
            height=720,
        )
        st.caption("Colori: rosso = peggio, verde = meglio, bianco = neutro (dove definito).")

    st.markdown("---")

    # =========================================================
    # 3) CORRELAZIONE TITOLI
    # =========================================================
    st.markdown("### Correlazione titoli (portafoglio attivo)")

    if prices_active is None or prices_active.empty:
        st.info("Mancano i prezzi dei titoli per calcolare la correlazione.")
        return

    tickers_all = [str(c).upper() for c in prices_active.columns]
    n_all = len(tickers_all)

    max_n_default = min(25, n_all) if n_all > 0 else 0
    max_n = st.slider(
        "Numero massimo di titoli da mostrare nella matrice",
        min_value=5 if n_all >= 5 else max(1, n_all),
        max_value=max(5, n_all),
        value=max_n_default if n_all >= 5 else n_all,
        step=1,
        key=_k("corr_max_n"),
    )

    default_subset = tickers_all[:max_n]
    subset = st.multiselect(
        "Seleziona titoli (se vuoto, uso default)",
        options=tickers_all,
        default=[],
        key=_k("corr_subset"),
    )
    use_cols = (subset[:max_n] if subset else default_subset)

    prices_sub = prices_active.copy()
    prices_sub.columns = [str(c).upper() for c in prices_sub.columns]
    prices_sub = prices_sub[[c for c in use_cols if c in prices_sub.columns]]

    corr = compute_asset_correlation(prices_sub)
    if corr is None or corr.empty:
        st.info("Correlazione non calcolabile (dati insufficienti).")
        return

    figc = _plot_heatmap_corr(corr, height=600, dark=False)
    st.plotly_chart(figc, use_container_width=True, theme=None, key=_k("plot_corr_heat"))

    st.markdown("**Tabella (numerica, senza colori)**")
    st.dataframe(corr.round(2), use_container_width=True, height=520)

    st.download_button(
        "Download correlazione (CSV)",
        data=corr.to_csv().encode("utf-8"),
        file_name="correlation_matrix.csv",
        mime="text/csv",
        key=_k("dl_corr_csv"),
    )

    # =========================================================
    # 4) MONTE CARLO (AUTO) — senza yfinance
    # =========================================================
    st.markdown("---")
    st.markdown("### Monte Carlo (auto)")

    with st.expander("🎲 Simulazione Monte Carlo (si aggiorna automaticamente)", expanded=False):
        if prices_sub is None or prices_sub.empty or prices_sub.shape[1] < 1:
            st.info("Servono prezzi (almeno 1 ticker) per la simulazione.")
        else:
            # controlli (no RUN button)
            cA, cB, cC, cD, cE = st.columns([1.2, 1.0, 1.0, 1.0, 1.0])
            with cA:
                horizon_days = st.slider("Horizon (days)", 30, 3650, 365, 5, key=_k("mc_days"))
            with cB:
                sims = st.slider("Runs", 500, 100000, 25000, 500, key=_k("mc_sims"))
            with cC:
                model = st.selectbox("Model", ["Bootstrap", "GBM", "t-copula"], index=0, key=_k("mc_model"))
            with cD:
                alpha = st.select_slider("VaR alpha", options=[0.01, 0.025, 0.05, 0.10], value=0.05, key=_k("mc_alpha"))
            with cE:
                mode = st.selectbox("Portfolio mode", ["Rebalance", "Buy&Hold"], index=0, key=_k("mc_mode"))

            cF, cG, cH, cI = st.columns([1.0, 1.0, 1.0, 1.0])
            with cF:
                lookback_days = st.slider("Lookback (days)", 90, 5000, 756, 21, key=_k("mc_lookback"))
            with cG:
                drift_scale = st.slider("Drift scale", 0.0, 2.0, 1.0, 0.05, key=_k("mc_drift"))
            with cH:
                t_df = st.slider("t df", 3, 30, 7, 1, key=_k("mc_tdf"))
            with cI:
                seed = st.number_input("Seed", min_value=0, max_value=10_000_000, value=42, step=1, key=_k("mc_seed"))

            initial_capital = float(state_dict.get("portfolio_global_params", {}).get("initial_capital", 100_000.0)) if isinstance(state_dict, dict) else 100_000.0
            capital = st.number_input("Capital", min_value=100.0, value=float(initial_capital), step=100.0, key=_k("mc_capital"))

            # weights: prova da res, altrimenti equal
            cols = [str(c).upper() for c in prices_sub.columns]
            w = None
            w_raw = res.get("weights") if isinstance(res, dict) else None
            if isinstance(w_raw, pd.Series) and not w_raw.empty:
                ww = w_raw.copy()
                ww.index = ww.index.astype(str).str.upper()
                ww = ww.reindex(cols).fillna(0.0).astype(float)
                if float(ww.sum()) > 0:
                    w = (ww / float(ww.sum())).to_numpy(dtype=np.float32)
            if w is None:
                w = np.repeat(1.0 / len(cols), len(cols)).astype(np.float32)

            # subset storico
            px_win = prices_sub.copy()
            px_win = px_win.dropna(how="all").ffill().dropna()
            if px_win.empty or len(px_win) < 50:
                st.warning("Prezzi insufficienti: aumenta lo storico o controlla i dati.")
            else:
                px_win = px_win.iloc[-int(lookback_days):] if len(px_win) > int(lookback_days) else px_win

                logret = np.log(px_win / px_win.shift(1)).dropna()
                if len(logret) < 30:
                    st.warning("Log-returns insufficienti: aumenta il lookback.")
                else:
                    # guard RAM
                    if (horizon_days + 1) * sims > 120_000_000:
                        st.error("Runs*days troppo alto per la RAM. Riduci Horizon o Runs.")
                    else:
                        last_prices = px_win.iloc[-1].to_numpy(dtype=np.float32)

                        inc = _simulate_increments(
                            logret_hist=logret.values,
                            days=int(horizon_days),
                            sims=int(sims),
                            seed=int(seed),
                            method=str(model),
                            drift_scale=float(drift_scale),
                            t_df=int(t_df),
                        )
                        paths = _equity_from_increments(
                            inc=inc,
                            last_prices=last_prices,
                            weights=w,
                            capital=float(capital),
                            mode=str(mode),
                        )
                        bd = _compute_bands(paths, float(alpha))

                        # focus day
                        day_sel = st.slider("Focus day", 0, int(horizon_days), min(30, int(horizon_days)), key=_k("mc_focus_day"))

                        vals = paths[int(day_sel)].astype(np.float64)
                        mean_v = float(vals.mean())
                        med_v = float(np.quantile(vals, 0.50))
                        var_v = float(np.quantile(vals, float(alpha)))
                        cvar_v = _cvar(vals, float(alpha))
                        best_v = float(np.quantile(vals, 1 - float(alpha)))
                        p_loss = float((vals < float(capital)).mean())

                        c1, c2, c3, c4, c5, c6 = st.columns(6, gap="small")
                        with c1:
                            st.metric("CAPITAL", f"{capital:,.2f}")
                        with c2:
                            st.metric("MEAN", f"{mean_v:,.2f}", delta=f"{(mean_v - float(capital)):+,.2f}")
                        with c3:
                            st.metric("MEDIAN", f"{med_v:,.2f}", delta=f"{(med_v - float(capital)):+,.2f}")
                        with c4:
                            st.metric(f"VaR (q{alpha*100:.1f}%)", f"{var_v:,.2f}", delta=f"{(var_v - float(capital)):+,.2f}")
                        with c5:
                            st.metric("CVaR (ES)", f"{cvar_v:,.2f}", delta=f"{(cvar_v - float(capital)):+,.2f}")
                        with c6:
                            st.metric("P(LOSS)", f"{p_loss*100:.1f}%")

                        # charts
                        left, right = st.columns([2.35, 1.0], gap="small")
                        x = np.arange(int(horizon_days) + 1)

                        with left:
                            fig = go.Figure()
                            fig.add_trace(go.Scatter(x=x, y=bd["hi"], mode="lines", line=dict(width=0), hoverinfo="skip", showlegend=False))
                            fig.add_trace(
                                go.Scatter(
                                    x=x,
                                    y=bd["lo"],
                                    mode="lines",
                                    fill="tonexty",
                                    name=f"Band {alpha*100:.1f}–{(1-alpha)*100:.1f}",
                                    line=dict(width=0),
                                    hovertemplate="Day %{x}<br>Equity %{y:.2f}<extra></extra>",
                                )
                            )
                            fig.add_trace(go.Scatter(x=x, y=bd["mean"], mode="lines", name="Mean", line=dict(width=2)))
                            fig.add_trace(go.Scatter(x=x, y=bd["med"], mode="lines", name="Median", line=dict(width=2, dash="dot")))
                            fig.add_vline(x=int(day_sel), line_width=1, line_dash="dash", opacity=0.6)

                            fig.update_layout(
                                template="plotly_dark",
                                height=640,
                                margin=dict(l=10, r=10, t=10, b=10),
                                paper_bgcolor="#0E1521",
                                plot_bgcolor="#0E1521",
                                xaxis_title="Days",
                                yaxis_title="Equity",
                                hovermode="x unified",
                                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0, font=dict(size=11)),
                                font=dict(color="#E5E7EB"),
                            )
                            fig.update_xaxes(showgrid=True, gridcolor="rgba(148,163,184,0.10)", zeroline=False)
                            fig.update_yaxes(showgrid=True, gridcolor="rgba(148,163,184,0.10)", zeroline=False)

                            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False}, theme=None)

                        with right:
                            hdf = pd.DataFrame({"equity": vals})
                            hist = px.histogram(hdf, x="equity", nbins=55)
                            hist.update_layout(
                                template="plotly_dark",
                                height=640,
                                margin=dict(l=10, r=10, t=10, b=10),
                                paper_bgcolor="#0E1521",
                                plot_bgcolor="#0E1521",
                                xaxis_title="Equity",
                                yaxis_title="Count",
                                showlegend=False,
                                font=dict(color="#E5E7EB"),
                            )
                            hist.add_vline(x=var_v, line_dash="dash", line_width=2)
                            hist.add_vline(x=mean_v, line_width=2)
                            hist.add_vline(x=best_v, line_dash="dash", line_width=2)
                            hist.update_xaxes(showgrid=True, gridcolor="rgba(148,163,184,0.10)", zeroline=False)
                            hist.update_yaxes(showgrid=True, gridcolor="rgba(148,163,184,0.10)", zeroline=False)

                            st.plotly_chart(hist, use_container_width=True, config={"displayModeBar": False}, theme=None)

                        out = pd.DataFrame(
                            {
                                "day": np.arange(int(horizon_days) + 1),
                                "mean": bd["mean"].astype(np.float64),
                                "median": bd["med"].astype(np.float64),
                                f"q{float(alpha):.3f}": bd["lo"].astype(np.float64),
                                f"q{1-float(alpha):.3f}": bd["hi"].astype(np.float64),
                            }
                        )
                        st.download_button(
                            "⬇ Export Quantiles CSV",
                            data=out.to_csv(index=False).encode("utf-8"),
                            file_name="montecarlo_quantiles.csv",
                            mime="text/csv",
                            key=_k("mc_dl_csv"),
                        )

    # =========================================================
    # 5) DETTAGLIO COMPONENTI (EXPANDER PER TICKER)
    # =========================================================
    st.markdown("---")
    st.markdown("### Dettaglio componenti (per ticker)")

    with st.expander("🏷️ Sector map (opzionale)", expanded=False):
        st.caption("Formato: `AAPL:Tech, MSFT:Tech, XOM:Energy` (separatore `,` o `;`).")
        sector_map_text = st.text_area("Sector map", value=state_dict.get(_k("sector_map_text"), "") if isinstance(state_dict, dict) else "", key=_k("sector_map_text"))
        # salva nel session state (per non perderlo al rerun)
        st.session_state[_k("sector_map_text")] = sector_map_text

    def _parse_sector_map(txt: str) -> dict:
        d: dict = {}
        if not txt or not str(txt).strip():
            return d
        for pair in str(txt).strip().split(","):
            pair = pair.strip()
            if not pair:
                continue
            # supporta anche ';'
            for sub in pair.split(";"):
                sub = sub.strip()
                if ":" in sub:
                    t, sct = sub.split(":", 1)
                    d[t.strip().upper()] = sct.strip()
        return d

    sector_map = _parse_sector_map(st.session_state.get(_k("sector_map_text"), ""))

    # correlation matrix (full) su tutti i titoli attivi
    px_full = prices_active.copy()
    px_full.columns = [str(c).upper() for c in px_full.columns]
    px_full = px_full.dropna(how="all").ffill().dropna()
    corr_full = compute_asset_correlation(px_full) if (px_full is not None and not px_full.empty) else pd.DataFrame()

    # portfolio curve for comparison
    port_name = "PORT"
    port_curve = eq_active if isinstance(eq_active, pd.Series) else pd.Series(dtype=float)
    port_curve = normalize_series_daily(port_curve).dropna()

    for ticker, s_price in asset_map.items():
        with st.expander(f"📌 {ticker}", expanded=False):
            s_price = normalize_series_daily(s_price).dropna()
            if s_price.empty or len(s_price) < 10:
                st.info("Serie prezzi insufficiente.")
                continue

            tabs = st.tabs(["Grafico", "Drawdown", "Metriche", "Correlazioni", "Settore & Info", "AI (gpt-5-mini)"])

            # --- Grafico
            with tabs[0]:
                df_plot = pd.DataFrame()
                df_plot[ticker] = _base100(s_price)
                if port_curve is not None and not port_curve.empty:
                    df_plot[port_name] = _base100(port_curve.reindex(df_plot.index).dropna())
                df_plot = df_plot.dropna(how="all").ffill().dropna(how="all")
                figp = _plot_lines(df_plot, ytitle="Index (Base=100)", yfmt=".2f", height=420, dark=False)
                st.plotly_chart(figp, use_container_width=True, theme=None, key=_k(f"comp_plot_{ticker}"))

            # --- Drawdown
            with tabs[1]:
                dd_asset = _drawdown_from_equity(s_price)
                figd1 = _plot_drawdown(dd_asset, title=f"DD {ticker}", height=260, dark=False)
                st.plotly_chart(figd1, use_container_width=True, theme=None, key=_k(f"dd_{ticker}"))

                if port_curve is not None and not port_curve.empty:
                    dd_port = _drawdown_from_equity(port_curve.reindex(dd_asset.index).dropna())
                    figd2 = _plot_drawdown(dd_port, title=f"DD {port_name}", height=260, dark=False)
                    st.plotly_chart(figd2, use_container_width=True, theme=None, key=_k(f"dd_port_{ticker}"))

            # --- Metriche
            with tabs[2]:
                sm = {ticker: s_price}
                if port_curve is not None and not port_curve.empty:
                    sm[port_name] = port_curve
                mt2 = compute_metrics_table(series_map=sm, rf_annual=rf_annual, bench_name=None)
                if mt2 is None or mt2.empty:
                    st.info("Metriche non disponibili.")
                else:
                    st.dataframe(_style_metrics_table(mt2), use_container_width=True, height=560)

            # --- Correlazioni
            with tabs[3]:
                if corr_full is None or corr_full.empty or ticker not in corr_full.columns:
                    st.info("Correlazioni non disponibili.")
                else:
                    row = corr_full[ticker].drop(index=ticker, errors="ignore").sort_values(ascending=False)
                    st.markdown("**Top correlazioni (ρ)**")
                    st.dataframe(row.head(12).to_frame("ρ").round(3), use_container_width=True, height=360)

                    # mini-heatmap top N + ticker
                    topn = row.head(10).index.tolist()
                    cols_hm = [ticker] + topn
                    sub = corr_full.loc[cols_hm, cols_hm]
                    fhm = _plot_heatmap_corr(sub, height=420, dark=False)
                    st.plotly_chart(fhm, use_container_width=True, theme=None, key=_k(f"corr_hm_{ticker}"))

            # --- Settore & Info (Marketstack)
            with tabs[4]:
                sector = sector_map.get(ticker, "—")
                st.metric("Sector (map)", sector)

                ms = marketstack_ticker_info(ticker)
                if isinstance(ms, dict) and ms.get("_error"):
                    st.warning(f"Marketstack: {ms.get('_error')}")
                    if ms.get("_raw") is not None:
                        st.json(ms.get("_raw"))
                else:
                    st.markdown("**Marketstack ticker info (raw)**")
                    st.json(ms)

            # --- AI (gpt-5-mini)
            with tabs[5]:
                st.caption("Per evitare chiamate continue a ogni rerun: attiva il toggle, poi la risposta resta cached per ~6 ore.")
                use_ai = st.toggle("Genera analisi AI", value=False, key=_k(f"ai_toggle_{ticker}"))
                if not use_ai:
                    st.info("Toggle OFF.")
                else:
                    model = _openai_model_default()
                    # payload sintetico (no dati sensibili)
                    payload = {
                        "as_of": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
                        "ticker": ticker,
                        "sector_map": sector_map.get(ticker, None),
                        "corr_top": (
                            corr_full[ticker].drop(index=ticker, errors="ignore").sort_values(ascending=False).head(6).to_dict()
                            if isinstance(corr_full, pd.DataFrame) and (not corr_full.empty) and (ticker in corr_full.columns)
                            else {}
                        ),
                        "metrics_table": None,
                    }
                    mt_ai = compute_metrics_table(series_map={ticker: s_price}, rf_annual=rf_annual, bench_name=None)
                    if isinstance(mt_ai, pd.DataFrame) and not mt_ai.empty:
                        payload["metrics_table"] = mt_ai[ticker].to_dict() if ticker in mt_ai.columns else mt_ai.to_dict()

                    ms_info = marketstack_ticker_info(ticker)
                    if isinstance(ms_info, dict) and not ms_info.get("_error"):
                        payload["marketstack"] = ms_info

                    payload_json = json.dumps(payload, ensure_ascii=False, indent=2)

                    try:
                        with st.spinner("Chiamata OpenAI..."):
                            txt = _ai_analyze_cached(ticker=ticker, payload_json=payload_json, model=model)
                        st.markdown(txt)
                    except Exception as e:
                        st.error(f"AI error: {e}")
                        st.caption("Controlla OPENAI_API_KEY e OPENAI_MODEL nei Secrets, e che il pacchetto openai sia in requirements.txt.")
