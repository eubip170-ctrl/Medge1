# ui/portfolio_overview.py
from __future__ import annotations

import inspect
from typing import Any, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from core.portfolio_core import (
    cagr_from_returns,
    vol_ann,
    sharpe_ratio,
    sortino_ratio,
    drawdown_series,
    es_cvar,
    omega_ratio,
    rachev_ratio,
    risk_contributions,
)

# =========================
#  OVERVIEW LAYOUT KNOBS
# =========================
OV_MAIN_CHART_FIGSIZE = (10.8, 3.1)
OV_PIE_FIGSIZE        = (3.6, 2.4)   # un filo più grande (meno “schiacciato”)
OV_BAR_FIGSIZE        = (3.6, 2.4)

OV_BREAK_TABLE_H      = 150
OV_BREAK_ROW_H        = 26
OV_BREAK_TOP_N        = 6


# =========================================================
# Streamlit compat wrappers
# =========================================================
def _st_pyplot(fig: plt.Figure, *, stretch: bool = True) -> None:
    """Compat: Streamlit nuovi -> width='stretch'|'content'."""
    try:
        sig = inspect.signature(st.pyplot)
        if "width" in sig.parameters:
            st.pyplot(fig, width="stretch" if stretch else "content")
        else:
            st.pyplot(fig, use_container_width=stretch)
    except Exception:
        st.pyplot(fig)


def _st_dataframe(
    df: pd.DataFrame,
    *,
    height: Optional[int] = None,
    hide_index: bool = True,
    stretch: bool = True,
    row_height: Optional[int] = None,
) -> None:
    kwargs: dict[str, Any] = {}
    if height is not None:
        kwargs["height"] = height

    try:
        sig = inspect.signature(st.dataframe)
        if "hide_index" in sig.parameters:
            kwargs["hide_index"] = hide_index
        if row_height is not None and "row_height" in sig.parameters:
            kwargs["row_height"] = row_height
        if "width" in sig.parameters:
            kwargs["width"] = "stretch" if stretch else "content"
        else:
            kwargs["use_container_width"] = stretch
    except Exception:
        kwargs["use_container_width"] = stretch

    st.dataframe(df, **kwargs)


def _st_radio_compact(options: list[str], *, key: str, index: int = 0) -> str:
    """Radio compatto: se possibile nasconde label."""
    try:
        sig = inspect.signature(st.radio)
        if "label_visibility" in sig.parameters:
            return st.radio(
                "View",
                options=options,
                index=index,
                horizontal=True,
                key=key,
                label_visibility="collapsed",
            )
    except Exception:
        pass
    return st.radio("View", options=options, index=index, horizontal=True, key=key)


def _card_container():
    """Container con bordo se supportato."""
    try:
        sig = inspect.signature(st.container)
        if "border" in sig.parameters:
            return st.container(border=True)
    except Exception:
        pass
    return st.container()


# =========================================================
# CSS (Overview – compatto ma leggibile)
# =========================================================
def _inject_overview_css() -> None:
    st.markdown(
        """
<style>
section.main .block-container{
  padding-top: 0.85rem !important;
  padding-bottom: 0.95rem !important;
}
div[data-testid="stVerticalBlock"]{ gap: 0.65rem !important; }
hr{ margin: 0.45rem 0 !important; }

/* KPI */
.po-kpi{
  background: var(--card-bg, #111827);
  border: 1px solid var(--border, rgba(255,255,255,0.10));
  border-radius: 10px;
  padding: 0.55rem 0.80rem;
  height: 82px;
}
.po-kpi .lbl{
  font-size: 0.70rem;
  text-transform: uppercase;
  letter-spacing: 0.09em;
  color: var(--text-muted, #b4bdc9);
}
.po-kpi .val{
  margin-top: 0.18rem;
  font-size: 1.10rem;
  font-weight: 650;
  color: var(--heading, #f8fafc);
  line-height: 1.05;
}
.po-kpi .sub{
  margin-top: 0.12rem;
  font-size: 0.74rem;
  color: var(--text-muted, #b4bdc9);
}

.po-card-title{
  font-size: 0.76rem;
  text-transform: uppercase;
  letter-spacing: 0.09em;
  color: var(--text-muted, #b4bdc9);
  margin: 0 0 0.28rem 0;
}

/* liste compatte */
.po-list{
  margin: 0.10rem 0 0 0;
  padding-left: 1.00rem;
  color: var(--text, #e5e7eb);
  font-size: 0.76rem;
  line-height: 1.25;
}
.po-list li{ margin: 0.10rem 0; }
.po-muted{ color: var(--text-muted, #b4bdc9); }

/* metriche compatte (label | value) */
.po-metrics{
  display:grid;
  grid-template-columns: 1fr auto;
  gap: 0.16rem 0.70rem;
  margin-top: 0.10rem;
  font-size: 0.76rem;
}
.po-metrics .k{
  color: var(--text-muted, #b4bdc9);
  text-transform: uppercase;
  letter-spacing: 0.06em;
  font-size: 0.68rem;
}
.po-metrics .v{
  text-align:right;
  font-weight:650;
  color: var(--heading, #f8fafc);
}

/* Gauge */
.po-gauge{
  position: relative;
  margin-top: 0.24rem;
  height: 9px;
  border-radius: 999px;
  border: 1px solid var(--border, rgba(255,255,255,0.10));
  background: linear-gradient(90deg, #16a34a 0%, #f59e0b 55%, #ef4444 100%);
  overflow: hidden;
}
.po-gauge-dot{
  position: absolute;
  top: 50%;
  transform: translate(-50%, -50%);
  width: 11px;
  height: 11px;
  border-radius: 999px;
  background: var(--card-bg, #111827);
  border: 2px solid var(--heading, #f8fafc);
  box-shadow: 0 0 0 2px rgba(0,0,0,0.25);
}
.po-gauge-scale{
  display:flex;
  justify-content: space-between;
  margin-top: 0.18rem;
  font-size: 0.64rem;
  color: var(--text-muted, #b4bdc9);
}

/* dataframe padding */
[data-testid="stDataFrame"]{ padding: 0.20rem 0.20rem 0.10rem 0.20rem !important; }

/* centra immagini st.pyplot */
div[data-testid="stImage"]{ text-align:center !important; }
div[data-testid="stImage"] img{
  display:block !important;
  margin-left:auto !important;
  margin-right:auto !important;
}
</style>
        """,
        unsafe_allow_html=True,
    )


# =========================================================
# Helpers
# =========================================================
def _normalize_index_daily(s: pd.Series) -> pd.Series:
    if s is None or len(s) == 0:
        return pd.Series(dtype=float)
    out = pd.Series(s).dropna().copy()
    idx = pd.to_datetime(out.index)
    if getattr(idx, "tz", None) is not None:
        idx = idx.tz_convert(None)
    try:
        idx = idx.tz_localize(None)
    except Exception:
        pass
    out.index = idx.normalize()
    out = out[~out.index.duplicated(keep="last")]
    return out.sort_index()


def _fmt_pct(x: Any, digits: int = 2, *, fraction: bool = True) -> str:
    try:
        if x is None or not np.isfinite(float(x)):
            return "–"
        v = float(x) * (100.0 if fraction else 1.0)
        return f"{v:.{digits}f}%"
    except Exception:
        return "–"


def _fmt_float(x: Any, digits: int = 3) -> str:
    try:
        if x is None or not np.isfinite(float(x)):
            return "–"
        return f"{float(x):.{digits}f}"
    except Exception:
        return "–"


def _fmt_money(x: Any) -> str:
    try:
        if x is None or not np.isfinite(float(x)):
            return "–"
        return f"{float(x):,.0f}"
    except Exception:
        return "–"


def _compute_ytd_and_1y(eq: pd.Series) -> Tuple[Optional[float], Optional[float]]:
    eq = _normalize_index_daily(eq)
    if eq.empty:
        return None, None

    last_dt = eq.index.max()
    last_val = float(eq.iloc[-1])

    ytd = None
    try:
        y0 = pd.Timestamp(year=last_dt.year, month=1, day=1)
        idx = eq.index.get_indexer([y0], method="nearest")[0]
        ref = float(eq.iloc[idx])
        if ref != 0:
            ytd = (last_val / ref) - 1.0
    except Exception:
        pass

    one_y = None
    try:
        t = last_dt - pd.Timedelta(days=365)
        idx = eq.index.get_indexer([t], method="nearest")[0]
        ref = float(eq.iloc[idx])
        if ref != 0:
            one_y = (last_val / ref) - 1.0
    except Exception:
        pass

    return ytd, one_y


def _risk_score(vol_ann_val: float, mdd_val: float) -> Tuple[float, str]:
    v = float(vol_ann_val) if np.isfinite(vol_ann_val) else np.nan
    d = float(abs(mdd_val)) if np.isfinite(mdd_val) else np.nan

    v_n = np.clip(v / 0.30, 0.0, 1.0)
    d_n = np.clip(d / 0.60, 0.0, 1.0)
    score = float(np.clip(10.0 * (0.60 * v_n + 0.40 * d_n), 0.0, 10.0))

    if score < 3.0:
        label = "Low"
    elif score < 6.0:
        label = "Medium"
    elif score < 8.0:
        label = "High"
    else:
        label = "Very High"
    return score, label


def _weights_used(res: dict, state: Any = None) -> pd.Series:
    tickers: list[str] = []
    if isinstance(res.get("prices"), pd.DataFrame) and not res["prices"].empty:
        tickers = list(res["prices"].columns)
    elif isinstance(res.get("returns_assets"), pd.DataFrame) and not res["returns_assets"].empty:
        tickers = list(res["returns_assets"].columns)

    # 1) weights_used_for_run
    try:
        if state is not None:
            w_run = state.get("weights_used_for_run")
            if isinstance(w_run, pd.Series) and not w_run.empty:
                w = w_run.reindex(tickers).fillna(0.0).astype(float)
                s = float(w.sum())
                return (w / s) if s > 0 else w
    except Exception:
        pass

    # 2) res["weights"]
    w_res = res.get("weights")
    if isinstance(w_res, pd.Series) and not w_res.empty:
        w = w_res.reindex(tickers).fillna(0.0).astype(float)
        s = float(w.sum())
        return (w / s) if s > 0 else w

    # 3) fallback equal
    n = len(tickers)
    if n <= 0:
        return pd.Series(dtype=float)
    return pd.Series(1.0 / n, index=tickers, dtype=float)


# =========================================================
# Plots
# =========================================================
def _plot_equity(eq: pd.Series) -> plt.Figure:
    eq = _normalize_index_daily(eq)
    fig, ax = plt.subplots(figsize=OV_MAIN_CHART_FIGSIZE)
    ax.plot(eq.index, eq.values, linewidth=1.7)
    ax.set_title("Portfolio value (full period)", fontsize=10.5)
    ax.grid(True, alpha=0.22)
    ax.tick_params(axis="both", labelsize=8.6)
    fig.tight_layout(pad=0.45)
    return fig


def _plot_drawdown(eq: pd.Series) -> plt.Figure:
    eq = _normalize_index_daily(eq)
    dd = drawdown_series(eq)
    fig, ax = plt.subplots(figsize=OV_MAIN_CHART_FIGSIZE)
    ax.plot(dd.index, dd.values, linewidth=1.7)
    ax.set_title("Drawdown (full period)", fontsize=10.5)
    ax.grid(True, alpha=0.22)
    ax.tick_params(axis="both", labelsize=8.6)
    fig.tight_layout(pad=0.45)
    return fig


def _plot_weights_pie_small(weights: pd.Series) -> plt.Figure:
    w = weights.dropna().astype(float)
    w = w[w > 0].copy()
    w = (w / w.sum()) if w.sum() > 0 else w

    top_n = 6
    if len(w) > top_n:
        top = w.sort_values(ascending=False).head(top_n)
        other = 1.0 - float(top.sum())
        w = top.copy()
        if other > 0:
            w.loc["Other"] = other

    fig, ax = plt.subplots(figsize=OV_PIE_FIGSIZE)
    ax.pie(
        w.values,
        labels=list(w.index),
        autopct=lambda p: f"{p:.0f}%" if p >= 6 else "",
        startangle=90,
        pctdistance=0.72,
        labeldistance=1.06,
        textprops={"fontsize": 8},
        wedgeprops={"linewidth": 1.0, "edgecolor": "white"},
    )
    ax.set_title("Components", fontsize=10, pad=4)
    ax.axis("equal")
    fig.tight_layout(pad=0.25)
    return fig


def _plot_top_contrib_mini(contrib_pp: pd.Series) -> plt.Figure:
    s = contrib_pp.dropna().astype(float).sort_values(ascending=True).tail(5)
    fig, ax = plt.subplots(figsize=OV_BAR_FIGSIZE)

    if s.empty:
        ax.text(0.5, 0.5, "No data", ha="center", va="center")
        ax.axis("off")
        fig.tight_layout(pad=0.2)
        return fig

    ax.barh(s.index, s.values)
    ax.set_title("Top contributors", fontsize=10, pad=4)
    ax.tick_params(axis="both", labelsize=8)
    ax.grid(True, axis="x", alpha=0.20)
    ax.set_xlabel("")
    # più spazio a sinistra per i ticker
    fig.subplots_adjust(left=0.35, right=0.98, top=0.88, bottom=0.20)
    return fig


# =========================================================
# Main renderer
# =========================================================
def render_portfolio_overview(res: Optional[dict] = None, state: Any = None) -> None:
    """
    Layout (minima modifica):
    - KPI strip
    - Row A: Chart (sx) + griglia 2x2 (dx): Components | Contributors / Risk snapshot | Drawdown quality
    - Row B: Breakdown full width
    """
    _inject_overview_css()

    if res is None and state is not None:
        res = state.get("res")

    if not isinstance(res, dict):
        st.info("Nessun risultato disponibile. Premi **Run / Update** nella sezione Parametri.")
        return

    eq_raw = res.get("equity")
    if eq_raw is None or len(eq_raw) == 0:
        st.info("Risultati presenti ma **equity** vuota. Premi **Run / Update** oppure verifica i ticker.")
        st.caption(f"Debug: keys res = {list(res.keys())}")
        return

    eq = _normalize_index_daily(eq_raw)
    rets = eq.pct_change().dropna()

    rf_ann = float(state.get("rf_annual", 0.0)) if state is not None else 0.0
    r_alpha = float(state.get("rachev_alpha", 0.05)) if state is not None else 0.05

    cagr_val = cagr_from_returns(rets) if not rets.empty else np.nan
    vol_val = vol_ann(rets) if not rets.empty else np.nan
    sharpe_val = sharpe_ratio(rets, rf_annual=rf_ann) if not rets.empty else np.nan
    sortino_val = sortino_ratio(rets, rf_annual=rf_ann) if not rets.empty else np.nan
    mdd_val = drawdown_series(eq).min() if len(eq) > 1 else np.nan

    ytd_ret, ret_1y = _compute_ytd_and_1y(eq)
    risk_score, risk_label = _risk_score(vol_val, mdd_val)

    es95 = es_cvar(rets, 0.95) if not rets.empty else np.nan
    es99 = es_cvar(rets, 0.99) if not rets.empty else np.nan
    omega0 = omega_ratio(rets, 0.0) if not rets.empty else np.nan
    rachev = rachev_ratio(rets, r_alpha) if not rets.empty else np.nan

    met_series = res.get("metrics")
    ulcer = met_series.get("Ulcer Index") if isinstance(met_series, pd.Series) else np.nan
    burke = met_series.get("Burke") if isinstance(met_series, pd.Series) else np.nan
    sterling = met_series.get("Sterling") if isinstance(met_series, pd.Series) else np.nan
    pain = met_series.get("Pain Ratio") if isinstance(met_series, pd.Series) else np.nan

    last_date = eq.index.max().date()
    port_value = float(eq.iloc[-1])

    w_used = _weights_used(res, state=state)

    # Risk contributions (opzionale)
    rc_df = None
    try:
        rets_assets = res.get("returns_assets")
        rets_port = res.get("returns_portfolio")
        if (
            isinstance(rets_assets, pd.DataFrame)
            and not rets_assets.empty
            and isinstance(rets_port, pd.Series)
            and not rets_port.empty
        ):
            tickers = list(rets_assets.columns)
            w_vec = w_used.reindex(tickers).fillna(0.0).astype(float).values
            _, _, rc_df = risk_contributions(rets_assets.loc[:, tickers], w_vec, rets_port, alpha=0.95)
    except Exception:
        rc_df = None

    # =========================================================
    # KPI strip
    # =========================================================
    st.markdown("### Overview")
    k1, k2, k3, k4, k5, k6 = st.columns([1.25, 0.95, 0.90, 0.90, 0.90, 1.20], gap="small")

    with k1:
        st.markdown(
            f"""
<div class="po-kpi">
  <div class="lbl">Portfolio</div>
  <div class="val">Custom</div>
  <div class="sub">As of {last_date}</div>
</div>
            """,
            unsafe_allow_html=True,
        )
    with k2:
        st.markdown(
            f"""
<div class="po-kpi">
  <div class="lbl">Value</div>
  <div class="val">{_fmt_money(port_value)}</div>
  <div class="sub">Full period</div>
</div>
            """,
            unsafe_allow_html=True,
        )
    with k3:
        st.markdown(
            f"""
<div class="po-kpi">
  <div class="lbl">YTD</div>
  <div class="val">{_fmt_pct(ytd_ret)}</div>
  <div class="sub">Since Jan 1</div>
</div>
            """,
            unsafe_allow_html=True,
        )
    with k4:
        st.markdown(
            f"""
<div class="po-kpi">
  <div class="lbl">1Y</div>
  <div class="val">{_fmt_pct(ret_1y)}</div>
  <div class="sub">Trailing 12m</div>
</div>
            """,
            unsafe_allow_html=True,
        )
    with k5:
        st.markdown(
            f"""
<div class="po-kpi">
  <div class="lbl">Vol (ann.)</div>
  <div class="val">{_fmt_pct(vol_val)}</div>
  <div class="sub">σ annualized</div>
</div>
            """,
            unsafe_allow_html=True,
        )
    with k6:
        pos = float(np.clip(risk_score, 0.0, 10.0)) / 10.0 * 100.0
        st.markdown(
            f"""
<div class="po-kpi">
  <div class="lbl">Risk level</div>
  <div class="val">{risk_label} ({risk_score:.1f}/10)</div>
  <div class="po-gauge"><div class="po-gauge-dot" style="left:{pos}%;"></div></div>
  <div class="po-gauge-scale"><span>0</span><span>5</span><span>10</span></div>
</div>
            """,
            unsafe_allow_html=True,
        )

    # =========================================================
    # Row A: Chart (sx) + griglia 2x2 (dx)
    # =========================================================
    left, right = st.columns([2.85, 1.15], gap="large")

    with left:
        with _card_container():
            st.markdown('<div class="po-card-title">Chart</div>', unsafe_allow_html=True)
            chart_mode = _st_radio_compact(["Portfolio value", "Drawdown"], key="po_overview_chart_mode", index=0)
            fig = _plot_drawdown(eq) if chart_mode == "Drawdown" else _plot_equity(eq)
            _st_pyplot(fig, stretch=True)
            plt.close(fig)

    with right:
        # griglia 2x2 nella colonna destra (qui sta il “fix” principale)
        r1c1, r1c2 = st.columns(2, gap="small")
        r2c1, r2c2 = st.columns(2, gap="small")

        with r1c1:
            with _card_container():
                st.markdown('<div class="po-card-title">Components</div>', unsafe_allow_html=True)
                fig = _plot_weights_pie_small(w_used)
                _st_pyplot(fig, stretch=True)
                plt.close(fig)

        with r1c2:
            with _card_container():
                st.markdown('<div class="po-card-title">Contributors</div>', unsafe_allow_html=True)
                rets_assets = res.get("returns_assets")
                if isinstance(rets_assets, pd.DataFrame) and not rets_assets.empty:
                    mean_ann = (rets_assets.mean() * 252.0).reindex(w_used.index)
                    contrib_pp = (mean_ann * w_used).dropna() * 100.0
                    fig = _plot_top_contrib_mini(contrib_pp)
                    _st_pyplot(fig, stretch=True)
                    plt.close(fig)
                else:
                    st.caption("Not enough data.")

        with r2c1:
            with _card_container():
                st.markdown('<div class="po-card-title">Risk snapshot</div>', unsafe_allow_html=True)
                st.markdown(
                    f"""
<div class="po-metrics">
  <div class="k">CAGR</div><div class="v">{_fmt_pct(cagr_val)}</div>
  <div class="k">Sharpe</div><div class="v">{_fmt_float(sharpe_val, 3)} <span class="po-muted">(rf={rf_ann:.3f})</span></div>
  <div class="k">Sortino</div><div class="v">{_fmt_float(sortino_val, 3)}</div>
  <div class="k">MDD</div><div class="v">{_fmt_pct(mdd_val)}</div>
  <div class="k">ES95</div><div class="v">{_fmt_float(es95, 4)}</div>
  <div class="k">ES99</div><div class="v">{_fmt_float(es99, 4)}</div>
  <div class="k">Omega</div><div class="v">{_fmt_float(omega0, 3)}</div>
  <div class="k">Rachev</div><div class="v">{_fmt_float(rachev, 3)} <span class="po-muted">(α={r_alpha:.2f})</span></div>
</div>
                    """,
                    unsafe_allow_html=True,
                )

        with r2c2:
            with _card_container():
                st.markdown('<div class="po-card-title">Drawdown quality</div>', unsafe_allow_html=True)
                st.markdown(
                    f"""
<div class="po-metrics">
  <div class="k">Ulcer</div><div class="v">{_fmt_float(ulcer, 3)}</div>
  <div class="k">Burke</div><div class="v">{_fmt_float(burke, 3)}</div>
  <div class="k">Sterling</div><div class="v">{_fmt_float(sterling, 3)}</div>
  <div class="k">Pain</div><div class="v">{_fmt_float(pain, 3)}</div>
</div>
                    """,
                    unsafe_allow_html=True,
                )

    # =========================================================
    # Row B: Breakdown full width (così la pagina respira)
    # =========================================================
    with _card_container():
        st.markdown('<div class="po-card-title">Breakdown by asset</div>', unsafe_allow_html=True)

        df_break = pd.DataFrame({"Ticker": w_used.index, "Weight": w_used.values})
        if isinstance(rc_df, pd.DataFrame) and not rc_df.empty:
            for col in ["Vol RC %", "CVaR95 RC %"]:
                if col in rc_df.columns:
                    df_break[col] = rc_df.reindex(w_used.index)[col].values

        df_break = df_break.sort_values("Weight", ascending=False)

        df_top = df_break.head(OV_BREAK_TOP_N).copy()
        df_show = df_top.copy()
        df_show["Weight"] = df_show["Weight"].map(lambda x: f"{x*100:.2f}%")
        if "Vol RC %" in df_show.columns:
            df_show["Vol RC %"] = df_show["Vol RC %"].map(lambda x: f"{x*100:.2f}%" if np.isfinite(x) else "–")
        if "CVaR95 RC %" in df_show.columns:
            df_show["CVaR95 RC %"] = df_show["CVaR95 RC %"].map(lambda x: f"{x*100:.2f}%" if np.isfinite(x) else "–")

        _st_dataframe(df_show, height=OV_BREAK_TABLE_H, hide_index=True, stretch=True, row_height=OV_BREAK_ROW_H)

        with st.expander("Show full breakdown"):
            df_full = df_break.copy()
            df_full["Weight"] = df_full["Weight"].map(lambda x: f"{x*100:.2f}%")
            if "Vol RC %" in df_full.columns:
                df_full["Vol RC %"] = df_full["Vol RC %"].map(lambda x: f"{x*100:.2f}%" if np.isfinite(x) else "–")
            if "CVaR95 RC %" in df_full.columns:
                df_full["CVaR95 RC %"] = df_full["CVaR95 RC %"].map(lambda x: f"{x*100:.2f}%" if np.isfinite(x) else "–")
            _st_dataframe(df_full, height=320, hide_index=True, stretch=True, row_height=OV_BREAK_ROW_H)
