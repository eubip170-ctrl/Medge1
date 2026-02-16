# ui/analisi_page.py
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

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


def _plotly_layout(fig: go.Figure, height: int = 440) -> go.Figure:
    fig.update_layout(
        height=height,
        margin=dict(l=10, r=10, t=28, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        legend=dict(orientation="h", y=-0.22, x=0.0),
        font=dict(size=12, family="Arial, Helvetica, sans-serif"),
    )
    fig.update_xaxes(showgrid=True, gridcolor="rgba(0,0,0,0.12)", zeroline=False)
    fig.update_yaxes(showgrid=True, gridcolor="rgba(0,0,0,0.12)", zeroline=False)
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
    # palette “morbida” (leggibile)
    red = "#fecaca"    # soft red
    white = "#ffffff"
    green = "#bbf7d0"  # soft green

    s = float(score_0_1)
    s = 0.5 if not np.isfinite(s) else max(0.0, min(1.0, s))

    if s < 0.5:
        return _blend(red, white, s / 0.5)     # 0..0.5 => red->white
    return _blend(white, green, (s - 0.5) / 0.5)  # 0.5..1 => white->green


def _style_metrics_table(df: pd.DataFrame) -> "pd.io.formats.style.Styler":
    """
    df: index = metric label, columns = series name, values numeric.
    Colora per riga (metrica) usando ranking relativo, ma:
      - se la metrica ha neutral_band e il valore è dentro => bianco
      - higher_is_better inverte dove serve
    """
    specs = {s.label: s for s in metric_specs()}

    def row_style(row: pd.Series) -> List[str]:
        label = str(row.name)
        spec = specs.get(label, None)

        vals = pd.to_numeric(row, errors="coerce").astype(float)
        out = ["background-color: #ffffff" for _ in vals.index]

        finite = vals[np.isfinite(vals)]
        if finite.empty or spec is None:
            return out

        # ranking -> score 0..1
        vmin = float(finite.min())
        vmax = float(finite.max())
        denom = (vmax - vmin) if np.isfinite(vmax - vmin) and (vmax - vmin) > 0 else np.nan

        for i, col in enumerate(vals.index):
            v = float(vals.loc[col]) if np.isfinite(vals.loc[col]) else np.nan
            if not np.isfinite(v):
                out[i] = "background-color: #ffffff"
                continue

            # neutral band => white
            if spec.neutral_band is not None:
                lo, hi = spec.neutral_band
                if np.isfinite(lo) and np.isfinite(hi) and (lo <= v <= hi):
                    out[i] = "background-color: #ffffff"
                    continue

            if not np.isfinite(denom):
                out[i] = "background-color: #ffffff"
                continue

            score = (v - vmin) / denom
            # invert if lower is better
            if not spec.higher_is_better:
                score = 1.0 - score

            bg = _score_to_bg(score)
            out[i] = f"background-color: {bg};"
        return out

    sty = df.style.apply(row_style, axis=1)

    # formatting per riga (subset = riga)
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


def _plot_lines(df: pd.DataFrame, ytitle: str, yfmt: str, height: int = 460) -> go.Figure:
    fig = go.Figure()
    if df is None or df.empty:
        return _plotly_layout(fig, height=height)

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
    return _plotly_layout(fig, height=height)


def _plot_heatmap_corr(corr: pd.DataFrame, height: int = 560) -> go.Figure:
    fig = go.Figure()
    if corr is None or corr.empty:
        return _plotly_layout(fig, height=height)

    labels = corr.columns.astype(str).tolist()
    fig.add_trace(
        go.Heatmap(
            z=corr.values,
            x=labels,
            y=labels,
            zmin=-1,
            zmax=1,
            zmid=0,
            # palette più leggibile (diverging classica)
            colorscale="RdBu",
            reversescale=False,
            colorbar=dict(thickness=12, title="ρ"),
            hovertemplate="x=%{x}<br>y=%{y}<br>ρ=%{z:.2f}<extra></extra>",
        )
    )
    fig.update_xaxes(side="top", showgrid=False, tickangle=45, tickfont=dict(size=10))
    fig.update_yaxes(showgrid=False, tickfont=dict(size=10))
    return _plotly_layout(fig, height=height)


def render_analisi_page(res: Optional[dict] = None, state: Any = None) -> None:
    _inject_css()

    state_dict = state if isinstance(state, dict) else st.session_state
    if res is None and isinstance(state_dict, dict):
        res = state_dict.get("res")

    # prendi runs multi se ci sono
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

    # se non c'è multi, usa almeno l'active
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
    # 1) GRAFICO IN ALTO: Portafogli / Titoli / Entrambi
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
            )
            st.plotly_chart(fig, use_container_width=True, theme=None, key=_k("plot_compare_top"))

    st.markdown("---")

    # =========================================================
    # 2) METRICHE: TABELLA UNICA + COLORI
    # =========================================================
    st.markdown("### Metriche (tabella unica)")

    # bench: se vuoi beta/alpha/IR/TE, serve una colonna bench dentro series_map.
    # qui non forzo benchmark esterno: uso solo se esiste una serie chiamata "BENCH"
    bench_name = "BENCH" if "BENCH" in series_map else None

    rf_annual = float(state_dict.get("portfolio_global_params", {}).get("rf_annual", 0.0)) if isinstance(state_dict, dict) else 0.0

    mt = compute_metrics_table(series_map=series_map, rf_annual=rf_annual, bench_name=bench_name)
    if mt is None or mt.empty:
        st.info("Metriche non disponibili (serie insufficienti).")
    else:
        # TABELLA UNICA, NON espandibile
        st.dataframe(
            _style_metrics_table(mt),
            use_container_width=True,
            height=720,
        )

        st.caption("Colori: rosso = peggio, verde = meglio, bianco = neutro (dove definito).")

    st.markdown("---")

    # =========================================================
    # 3) CORRELAZIONE TITOLI (leggibile)
    # =========================================================
    st.markdown("### Correlazione titoli (portafoglio attivo)")

    if prices_active is None or prices_active.empty:
        st.info("Mancano i prezzi dei titoli per calcolare la correlazione.")
        return

    # limita dimensione (leggibilità)
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

    # scegli subset (default = primi N in ordine)
    default_subset = tickers_all[:max_n]
    subset = st.multiselect(
        "Seleziona titoli (se vuoto, uso default)",
        options=tickers_all,
        default=[],
        key=_k("corr_subset"),
    )
    if subset:
        use_cols = subset[:max_n]
    else:
        use_cols = default_subset

    prices_sub = prices_active.copy()
    prices_sub.columns = [str(c).upper() for c in prices_sub.columns]
    prices_sub = prices_sub[[c for c in use_cols if c in prices_sub.columns]]

    corr = compute_asset_correlation(prices_sub)
    if corr is None or corr.empty:
        st.info("Correlazione non calcolabile (dati insufficienti).")
        return

    # HEATMAP (colori ok, leggibile, tickangle e size)
    figc = _plot_heatmap_corr(corr, height=600)
    st.plotly_chart(figc, use_container_width=True, theme=None, key=_k("plot_corr_heat"))

    # TABELLA NUMERICA SENZA COLORI (leggibile)
    st.markdown("**Tabella (numerica, senza colori)**")
    st.dataframe(corr.round(2), use_container_width=True, height=520)

    st.download_button(
        "Download correlazione (CSV)",
        data=corr.to_csv().encode("utf-8"),
        file_name="correlation_matrix.csv",
        mime="text/csv",
        key=_k("dl_corr_csv"),
    )
