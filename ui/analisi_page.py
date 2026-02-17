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


# =========================
# CSS
# =========================
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

/* --- MonteCarlo panels (scuri, stile terminal) --- */
.mc-panel{
  background: #0E1521;
  border: 1px solid rgba(148,163,184,0.18);
  border-radius: 14px;
  padding: 10px 10px 8px 10px;
  margin-bottom: 6px;
}
.mc-h{
  display:flex;
  justify-content:space-between;
  align-items:center;
}
.mc-title{
  font-size: 0.92rem;
  font-weight: 900;
  color: #F9FAFB;
  letter-spacing: -0.02em;
}
.mc-note{
  font-size: 0.78rem;
  color: rgba(229,231,235,0.65);
}
</style>
        """,
        unsafe_allow_html=True,
    )


# =========================
# Plotly layout helpers
# =========================
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


def _plotly_layout_terminal(fig: go.Figure, height: int = 640) -> go.Figure:
    fig.update_layout(
        template="plotly_dark",
        height=height,
        margin=dict(l=10, r=10, t=10, b=10),
        paper_bgcolor="#0E1521",
        plot_bgcolor="#0E1521",
        font=dict(color="#E5E7EB", size=12, family="Arial, Helvetica, sans-serif"),
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0.0, font=dict(size=11)),
    )
    fig.update_xaxes(showgrid=True, gridcolor="rgba(148,163,184,0.10)", zeroline=False)
    fig.update_yaxes(showgrid=True, gridcolor="rgba(148,163,184,0.10)", zeroline=False)
    return fig


# =========================
# Base helpers
# =========================
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
            colorscale="RdBu",
            reversescale=False,
            colorbar=dict(thickness=12, title="ρ"),
            hovertemplate="x=%{x}<br>y=%{y}<br>ρ=%{z:.2f}<extra></extra>",
        )
    )
    fig.update_xaxes(side="top", showgrid=False, tickangle=45, tickfont=dict(size=10))
    fig.update_yaxes(showgrid=False, tickfont=dict(size=10))
    return _plotly_layout(fig, height=height)


# =========================
# Monte Carlo core (NO yfinance)
# =========================
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
    inc: np.ndarray,  # (days, sims, n_assets)
    last_prices: np.ndarray,  # (n_assets,)
    weights: np.ndarray,  # (n_assets,)
    capital: float,
    mode: str,  # Rebalance | Buy&Hold
) -> np.ndarray:
    days, sims, n = inc.shape
    w = weights.astype(np.float32)
    s = float(np.sum(w))
    if not np.isfinite(s) or s <= 0:
        w = np.ones(n, dtype=np.float32) / float(n)
    else:
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
        "lo": np.quantile(paths, alpha, axis=1),
        "hi": np.quantile(paths, 1.0 - alpha, axis=1),
    }


def _cvar(vals: np.ndarray, alpha: float) -> float:
    q = np.quantile(vals, alpha)
    tail = vals[vals <= q]
    return float(tail.mean()) if len(tail) else float(q)


def _build_weights_series(obj: Any, tickers: List[str]) -> pd.Series:
    if isinstance(obj, pd.Series):
        w = obj.copy()
    elif isinstance(obj, dict):
        w = pd.Series(obj)
    else:
        try:
            w = pd.Series(list(obj), index=tickers, dtype=float)
        except Exception:
            w = pd.Series(dtype=float)

    w.index = [str(x).upper() for x in w.index]
    t2 = [str(t).upper() for t in tickers]
    w = w.reindex(t2).fillna(0.0).astype(float)
    s = float(w.sum())
    if np.isfinite(s) and s > 0:
        w = w / s
    return w


def _render_monte_carlo_terminal(
    prices: pd.DataFrame,
    weights_a: pd.Series,
    label_a: str,
    weights_b: Optional[pd.Series] = None,
    label_b: Optional[str] = None,
    initial_capital: float = 100_000.0,
    currency: str = "€",
    key_prefix: str = "MC",
) -> None:
    prices = prices.copy()
    prices.columns = [str(c).upper() for c in prices.columns]
    prices = prices.dropna(how="all")

    if prices.empty or prices.shape[1] < 1:
        st.info("Monte Carlo: prezzi non disponibili.")
        return

    tickers = [str(c).upper() for c in prices.columns]

    lr = np.log(prices / prices.shift(1)).replace([np.inf, -np.inf], np.nan).dropna(how="any")
    if lr.shape[0] < 30:
        st.warning("Monte Carlo: storico insufficiente (servono ~30 osservazioni).")
        return

    with st.expander("🎲 Monte Carlo — scenario simulation (terminal)", expanded=True):
        c1, c2, c3, c4 = st.columns([1.0, 1.0, 1.0, 1.0], gap="small")
        with c1:
            days = st.slider("Horizon (days)", 30, 3650, 365, 5, key=f"{key_prefix}__days")
        with c2:
            sims = st.slider("Runs", 500, 100_000, 25_000, 500, key=f"{key_prefix}__sims")
        with c3:
            visible = st.slider("Visible paths", 0, 1500, 450, 50, key=f"{key_prefix}__visible")
        with c4:
            model = st.selectbox("Model", ["Bootstrap", "GBM", "t-copula"], index=0, key=f"{key_prefix}__model")

        d1, d2, d3, d4 = st.columns([1.0, 1.0, 1.0, 1.0], gap="small")
        with d1:
            mode = st.selectbox("Portfolio mode", ["Rebalance", "Buy&Hold"], index=0, key=f"{key_prefix}__mode")
        with d2:
            alpha = st.select_slider("VaR alpha", options=[0.01, 0.025, 0.05, 0.10], value=0.05, key=f"{key_prefix}__alpha")
        with d3:
            drift_scale = st.slider("Drift scale", 0.0, 2.0, 1.0, 0.05, key=f"{key_prefix}__drift")
        with d4:
            seed = st.number_input("Seed", min_value=0, max_value=10_000_000, value=42, step=1, key=f"{key_prefix}__seed")

        t_df = st.slider("t df (fat tails)", 3, 30, 7, 1, key=f"{key_prefix}__tdf")

    est = int(days) * int(sims) * int(len(tickers))
    if est > 160_000_000:
        st.error("Parametri troppo grandi per RAM (days*sims*assets). Riduci Runs o Horizon.")
        return

    last_prices = prices.iloc[-1].to_numpy(dtype=np.float32)

    wA = weights_a.reindex(tickers).fillna(0.0).to_numpy(dtype=np.float32)
    wB = None
    if weights_b is not None and label_b:
        wB = weights_b.reindex(tickers).fillna(0.0).to_numpy(dtype=np.float32)

    data_sig = (
        str(prices.index.min()),
        str(prices.index.max()),
        int(prices.shape[0]),
        tuple(tickers),
        float(np.nan_to_num(prices.iloc[-1].astype(float), nan=0.0).sum()),
    )

    key = (
        data_sig,
        int(days),
        int(sims),
        int(seed),
        str(model),
        float(drift_scale),
        int(t_df),
        str(mode),
        float(initial_capital),
        float(alpha),
        tuple(np.round(wA.astype(float), 8)),
        tuple(np.round(wB.astype(float), 8)) if wB is not None else None,
    )

    cache_key = f"{key_prefix}__mc_cache"
    cached = st.session_state.get(cache_key, None)

    need_compute = True
    if isinstance(cached, dict) and cached.get("key") == key:
        need_compute = False

    if need_compute:
        with st.spinner("Running Monte Carlo..."):
            inc = _simulate_increments(
                logret_hist=lr.values,
                days=int(days),
                sims=int(sims),
                seed=int(seed),
                method=str(model),
                drift_scale=float(drift_scale),
                t_df=int(t_df),
            )
            paths_a = _equity_from_increments(inc=inc, last_prices=last_prices, weights=wA, capital=float(initial_capital), mode=str(mode))
            bands_a = _compute_bands(paths_a, float(alpha))

            paths_b = None
            bands_b = None
            if wB is not None and label_b:
                paths_b = _equity_from_increments(inc=inc, last_prices=last_prices, weights=wB, capital=float(initial_capital), mode=str(mode))
                bands_b = _compute_bands(paths_b, float(alpha))

            st.session_state[cache_key] = {"key": key, "paths_a": paths_a, "bands_a": bands_a, "paths_b": paths_b, "bands_b": bands_b}

    pack = st.session_state.get(cache_key, {})
    paths_a = pack.get("paths_a")
    bands_a = pack.get("bands_a")
    paths_b = pack.get("paths_b")
    bands_b = pack.get("bands_b")

    if paths_a is None or bands_a is None:
        st.error("Errore interno Monte Carlo: risultati non disponibili.")
        return

    day_sel = st.slider("Focus day", 0, int(days), min(30, int(days)), key=f"{key_prefix}__focus")

    def kpis(paths: np.ndarray) -> dict:
        vals = paths[int(day_sel)].astype(np.float64)
        mean_v = float(vals.mean())
        var_v = float(np.quantile(vals, float(alpha)))
        cvar_v = _cvar(vals, float(alpha))
        best_v = float(np.quantile(vals, 1.0 - float(alpha)))
        p_loss = float((vals < float(initial_capital)).mean())
        med_v = float(np.quantile(vals, 0.50))
        return {"vals": vals, "mean": mean_v, "median": med_v, "var": var_v, "cvar": cvar_v, "best": best_v, "p_loss": p_loss}

    ka = kpis(paths_a)
    kb = kpis(paths_b) if paths_b is not None else None

    c1, c2, c3, c4, c5, c6 = st.columns(6, gap="small")
    with c1:
        st.metric("CAPITAL", f"{initial_capital:,.2f}{currency}")
    with c2:
        st.metric(f"MEAN ({label_a})", f"{ka['mean']:,.2f}{currency}", delta=f"{(ka['mean']-initial_capital):+,.2f}{currency}")
    with c3:
        st.metric(f"VaR (q{alpha*100:.1f}%)", f"{ka['var']:,.2f}{currency}", delta=f"{(ka['var']-initial_capital):+,.2f}{currency}")
    with c4:
        st.metric("CVaR (ES)", f"{ka['cvar']:,.2f}{currency}", delta=f"{(ka['cvar']-initial_capital):+,.2f}{currency}")
    with c5:
        st.metric(f"P(LOSS {label_a})", f"{ka['p_loss']*100:.1f}%")
    with c6:
        st.metric(f"MEAN ({label_b})" if kb and label_b else "MEAN (2nd)", f"{kb['mean']:,.2f}{currency}" if kb else "—")

    left, right = st.columns([2.35, 1.0], gap="small")
    x = np.arange(int(days) + 1)

    with left:
        st.markdown(
            f"""
<div class="mc-panel">
  <div class="mc-h">
    <div class="mc-title">PATHS • FAN CHART ({label_a}{' vs '+label_b if label_b else ''})</div>
    <div class="mc-note">Assets={len(tickers)} • FocusDay={day_sel}</div>
  </div>
</div>
            """,
            unsafe_allow_html=True,
        )

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=x, y=bands_a["hi"], mode="lines", line=dict(width=0), hoverinfo="skip", showlegend=False))
        fig.add_trace(
            go.Scatter(
                x=x,
                y=bands_a["lo"],
                mode="lines",
                fill="tonexty",
                name=f"{label_a} band {alpha}-{1-alpha}",
                fillcolor="rgba(59,130,246,0.18)",
                line=dict(color="rgba(59,130,246,0.0)"),
                hovertemplate="Day %{x}<br>Equity %{y:.2f}<extra></extra>",
            )
        )
        fig.add_trace(go.Scatter(x=x, y=bands_a["mean"], mode="lines", name=f"{label_a} mean", line=dict(color="rgba(229,231,235,0.95)", width=2)))
        fig.add_trace(go.Scatter(x=x, y=bands_a["med"], mode="lines", name=f"{label_a} median", line=dict(color="rgba(59,130,246,0.95)", width=2, dash="dot")))

        if int(visible) > 0:
            rng = np.random.default_rng(int(seed))
            ncols = paths_a.shape[1]
            k = min(int(visible), ncols)
            if k > 0:
                idx = rng.choice(ncols, size=k, replace=False)
                sample = paths_a[:, idx]
                for i in range(sample.shape[1]):
                    fig.add_trace(
                        go.Scattergl(
                            x=x,
                            y=sample[:, i],
                            mode="lines",
                            line=dict(color="rgba(148,163,184,0.10)", width=1),
                            opacity=0.9,
                            showlegend=False,
                            hoverinfo="skip",
                        )
                    )

        if bands_b is not None and label_b:
            fig.add_trace(go.Scatter(x=x, y=bands_b["mean"], mode="lines", name=f"{label_b} mean", line=dict(color="rgba(16,185,129,0.95)", width=2)))
            fig.add_trace(go.Scatter(x=x, y=bands_b["med"], mode="lines", name=f"{label_b} median", line=dict(color="rgba(16,185,129,0.85)", width=2, dash="dot")))

        fig.add_vline(x=int(day_sel), line_width=1, line_dash="dash", opacity=0.6, line_color="rgba(148,163,184,0.55)")
        fig.update_xaxes(title="Days")
        fig.update_yaxes(title=f"Equity ({currency})")
        st.plotly_chart(_plotly_layout_terminal(fig, height=640), use_container_width=True, theme=None, key=f"{key_prefix}__fan")

    with right:
        st.markdown(
            """
<div class="mc-panel">
  <div class="mc-h">
    <div class="mc-title">DISTRIBUTION (focus day)</div>
    <div class="mc-note">VaR / Mean / Best</div>
  </div>
</div>
            """,
            unsafe_allow_html=True,
        )

        vals = ka["vals"]
        hist = go.Figure()
        hist.add_trace(go.Histogram(x=vals, nbinsx=55, name="equity"))
        hist.add_vline(x=ka["var"], line_dash="dash", line_width=2, line_color="rgba(239,68,68,0.90)")
        hist.add_vline(x=ka["mean"], line_width=2, line_color="rgba(229,231,235,0.90)")
        hist.add_vline(x=ka["best"], line_dash="dash", line_width=2, line_color="rgba(16,185,129,0.90)")
        hist.update_xaxes(title=f"Equity ({currency})")
        hist.update_yaxes(title="Count")
        hist.update_layout(showlegend=False)
        st.plotly_chart(_plotly_layout_terminal(hist, height=640), use_container_width=True, theme=None, key=f"{key_prefix}__hist")

    out = pd.DataFrame(
        {
            "day": np.arange(int(days) + 1),
            f"{label_a}_mean": bands_a["mean"].astype(np.float64),
            f"{label_a}_median": bands_a["med"].astype(np.float64),
            f"{label_a}_q{float(alpha):.3f}": bands_a["lo"].astype(np.float64),
            f"{label_a}_q{1.0-float(alpha):.3f}": bands_a["hi"].astype(np.float64),
        }
    )
    if bands_b is not None and label_b:
        out[f"{label_b}_mean"] = bands_b["mean"].astype(np.float64)
        out[f"{label_b}_median"] = bands_b["med"].astype(np.float64)
        out[f"{label_b}_q{float(alpha):.3f}"] = bands_b["lo"].astype(np.float64)
        out[f"{label_b}_q{1.0-float(alpha):.3f}"] = bands_b["hi"].astype(np.float64)

    st.download_button(
        "⬇ Export quantiles CSV",
        data=out.to_csv(index=False).encode("utf-8"),
        file_name="montecarlo_quantiles.csv",
        mime="text/csv",
        key=f"{key_prefix}__dl",
        use_container_width=True,
    )


# =========================
# Drawdown helpers (componenti)
# =========================
def _drawdown_from_equity(eq: pd.Series) -> pd.Series:
    s = pd.Series(eq).astype(float).dropna()
    if len(s) < 2:
        return pd.Series(dtype=float)
    peak = s.cummax()
    dd = (s / peak) - 1.0
    return dd.replace([np.inf, -np.inf], np.nan).dropna()


def _plot_drawdown_compare(dd_asset: pd.Series, dd_port: Optional[pd.Series], asset_label: str, height: int = 300) -> go.Figure:
    fig = go.Figure()
    if dd_asset is None or dd_asset.empty:
        return _plotly_layout(fig, height=height)

    # Asset
    fig.add_trace(
        go.Scatter(
            x=dd_asset.index,
            y=dd_asset.values,
            mode="lines",
            name=f"DD {asset_label}",
            line=dict(width=2),
            fill="tozeroy",
            hovertemplate="%{x|%Y-%m-%d}<br>DD: %{y:.2%}<extra></extra>",
        )
    )

    # Portafoglio (stessa figura)
    if dd_port is not None and isinstance(dd_port, pd.Series) and (not dd_port.empty):
        fig.add_trace(
            go.Scatter(
                x=dd_port.index,
                y=dd_port.values,
                mode="lines",
                name="DD PORT",
                line=dict(width=2, dash="dot"),
                hovertemplate="%{x|%Y-%m-%d}<br>DD: %{y:.2%}<extra></extra>",
            )
        )

    fig.update_yaxes(title="Drawdown", tickformat=".0%")
    return _plotly_layout(fig, height=height)


def _parse_sector_map(txt: str) -> Dict[str, str]:
    d: Dict[str, str] = {}
    if not txt or not str(txt).strip():
        return d
    for part in str(txt).split(","):
        part = part.strip()
        if ":" in part:
            t, sct = part.split(":", 1)
            d[t.strip().upper()] = sct.strip()
    return d


def _safe_pct(s: pd.Series, n: int) -> float:
    s = pd.Series(s).astype(float).dropna()
    if len(s) <= n or n <= 0:
        return float("nan")
    a = float(s.iloc[-n - 1])
    b = float(s.iloc[-1])
    if not np.isfinite(a) or a == 0:
        return float("nan")
    return (b / a) - 1.0


@st.cache_data(ttl=3600, show_spinner=False)
def _ai_analyze_ticker_cached(ticker: str, context: str) -> str:
    # import lazy: evita crash se ai_client non disponibile
    from core.ai_client import chat_completion

    system = (
        "Sei un analista finanziario quantitativo. "
        "Scrivi in italiano, stile sintetico e operativo. "
        "Struttura: (1) snapshot (trend/volatilità/drawdown), (2) rischi principali, "
        "(3) catalizzatori possibili, (4) lettura correlazioni (diversificazione), "
        "(5) conclusione in 3 bullet. "
        "Non dare consigli personalizzati, solo analisi tecnica/quantitativa dei dati forniti."
    )
    return chat_completion(
        system_prompt=system,
        user_prompt=context,
        model="gpt-5-mini",
        max_output_tokens=750,
    )


# =========================
# MAIN
# =========================
def render_analisi_page(res: Optional[dict] = None, state: Any = None) -> None:
    _inject_css()

    state_dict = state if isinstance(state, dict) else st.session_state
    if res is None and isinstance(state_dict, dict):
        res = state_dict.get("res")

    runs = state_dict.get("portfolio_runs", {}) if isinstance(state_dict, dict) else {}

    if not isinstance(res, dict) or res.get("equity") is None:
        st.info("Nessun risultato disponibile. Esegui prima il run del portafoglio.")
        return

    # global params
    g = state_dict.get("portfolio_global_params", {}) if isinstance(state_dict, dict) else {}
    rf_annual = float(g.get("rf_annual", 0.0))
    initial_capital = float(g.get("initial_capital", 100_000.0))
    currency = str(g.get("currency", "€"))

    # active series
    eq_active = normalize_series_daily(res.get("equity")).dropna()
    prices_active = normalize_df_daily(res.get("prices")) if isinstance(res.get("prices"), pd.DataFrame) else pd.DataFrame()

    # portfolio series map (multi)
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
        port_map[str(res.get("name", "PORT"))] = eq_active

    # asset series map (only active)
    asset_map: Dict[str, pd.Series] = {}
    if isinstance(prices_active, pd.DataFrame) and not prices_active.empty:
        prices_active.columns = [str(c).upper() for c in prices_active.columns]
        for c in prices_active.columns:
            s = prices_active[c]
            if isinstance(s, pd.Series):
                s = normalize_series_daily(s).dropna()
                if len(s) > 2:
                    asset_map[str(c).upper()] = s

    st.markdown(
        "<div class='an-card'><div class='an-title'>Analisi</div>"
        "<div class='an-sub'>Grafico comparativo + tabella metriche unica (colorata) + correlazione titoli + Monte Carlo (no yfinance) + dettaglio componenti</div></div>",
        unsafe_allow_html=True,
    )

    # =========================================================
    # 1) TOP CHART
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
    # 2) METRICS TABLE
    # =========================================================
    st.markdown("### Metriche (tabella unica)")

    bench_name = "BENCH" if "BENCH" in series_map else None
    mt = compute_metrics_table(series_map=series_map, rf_annual=rf_annual, bench_name=bench_name)
    if mt is None or mt.empty:
        st.info("Metriche non disponibili (serie insufficienti).")
    else:
        st.dataframe(_style_metrics_table(mt), use_container_width=True, height=720)
        st.caption("Colori: rosso = peggio, verde = meglio, bianco = neutro (dove definito).")

    st.markdown("---")

    # =========================================================
    # 3) CORRELATION
    # =========================================================
    st.markdown("### Correlazione titoli (portafoglio attivo)")

    has_prices = isinstance(prices_active, pd.DataFrame) and not prices_active.empty
    corr_active = pd.DataFrame()

    if not has_prices:
        st.info("Mancano i prezzi dei titoli per calcolare la correlazione (e per Monte Carlo).")
    else:
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
        else:
            corr_active = corr.copy()
            figc = _plot_heatmap_corr(corr, height=600)
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

    st.markdown("---")

    # =========================================================
    # 4) MONTE CARLO
    # =========================================================
    st.markdown("### Monte Carlo Simulation")

    choices: Dict[str, dict] = {}

    # active
    try:
        nm = str(res.get("name", "ACTIVE"))
        pr = res.get("prices")
        if isinstance(pr, pd.DataFrame) and not pr.empty:
            pr2 = normalize_df_daily(pr).dropna(how="all")
            pr2.columns = [str(c).upper() for c in pr2.columns]
            choices[nm] = {"prices": pr2, "weights": _build_weights_series(res.get("weights"), list(pr2.columns))}
    except Exception:
        pass

    # runs
    if isinstance(runs, dict) and runs:
        for pid, r in runs.items():
            try:
                nm = str(r.get("name", pid))
                pr = r.get("prices")
                if isinstance(pr, pd.DataFrame) and not pr.empty:
                    pr2 = normalize_df_daily(pr).dropna(how="all")
                    if pr2 is None or pr2.empty:
                        continue
                    pr2.columns = [str(c).upper() for c in pr2.columns]
                    choices[nm] = {"prices": pr2, "weights": _build_weights_series(r.get("weights"), list(pr2.columns))}
            except Exception:
                continue

    if not choices:
        st.info("Monte Carlo: nessun portafoglio con prezzi disponibile (esegui prima Run / Update).")
    else:
        names = list(choices.keys())
        csel1, csel2 = st.columns([1.0, 1.0], gap="small")
        with csel1:
            p1 = st.selectbox("Portfolio A", options=names, index=0, key=_k("mc_p1"))
        with csel2:
            p2 = st.selectbox("Portfolio B (opzionale)", options=["—"] + names, index=0, key=_k("mc_p2"))

        A = choices[p1]
        pricesA = A["prices"].copy()
        wA = A["weights"].copy()

        wB = None
        labelB = None
        prices_use = pricesA

        if p2 != "—":
            B = choices[p2]
            pricesB = B["prices"].copy()
            wB = B["weights"].copy()
            labelB = str(p2)

            common_cols = [c for c in pricesA.columns if c in pricesB.columns]
            if len(common_cols) >= 1:
                idx = pricesA.index.intersection(pricesB.index)
                prices_use = pricesA.loc[idx, common_cols].copy().dropna(how="all")
                wA = wA.reindex(common_cols).fillna(0.0)
                wB = wB.reindex(common_cols).fillna(0.0)
            else:
                st.warning("Portfolio A e B non hanno tickers in comune: simulo solo A.")
                wB = None
                labelB = None
                prices_use = pricesA

        _render_monte_carlo_terminal(
            prices=prices_use,
            weights_a=wA,
            label_a=str(p1),
            weights_b=wB,
            label_b=labelB,
            initial_capital=initial_capital,
            currency=currency,
            key_prefix=_k("MC"),
        )

    # =========================================================
    # 5) DETTAGLIO COMPONENTI (EXPANDER PER TICKER)
    # =========================================================
    st.markdown("---")
    st.markdown("### Dettaglio componenti (per ticker)")

    if not has_prices or not asset_map:
        st.info("Nessun titolo disponibile per il dettaglio componenti.")
        return

    port_curve = normalize_series_daily(eq_active).dropna() if isinstance(eq_active, pd.Series) else pd.Series(dtype=float)

    # correlazione completa (una volta sola)
    px_full = prices_active.copy()
    px_full.columns = [str(c).upper() for c in px_full.columns]
    px_full = px_full.dropna(how="all").ffill().dropna(how="any")
    corr_full = compute_asset_correlation(px_full) if (px_full is not None and not px_full.empty) else pd.DataFrame()

    # sector map (manuale)
    with st.expander("🏷️ Sector map (opzionale)", expanded=False):
        st.caption("Formato: AAPL:Tech, MSFT:Tech, XOM:Energy (separatore , )")
        st.text_area("Sector map", key=_k("sector_map_text"))
    sector_map = _parse_sector_map(st.session_state.get(_k("sector_map_text"), ""))

    # pesi (se disponibili) per contesto AI
    w_active = None
    try:
        w_active = res.get("weights")
        if not isinstance(w_active, pd.Series):
            if isinstance(w_active, dict):
                w_active = pd.Series(w_active)
            else:
                w_active = None
    except Exception:
        w_active = None
    if isinstance(w_active, pd.Series):
        w_active.index = [str(i).upper() for i in w_active.index]

    # guard: AI su troppi titoli è pesante
    MAX_AI = 25
    tickers_list = list(asset_map.keys())

    for i, ticker in enumerate(tickers_list):
        t = str(ticker).upper()
        s_price = asset_map[t]

        with st.expander(f"📌 {t}", expanded=False):
            s_price = normalize_series_daily(s_price).dropna()
            if s_price.empty or len(s_price) < 10:
                st.info("Serie prezzi insufficiente.")
                continue

            # ---- 5.1 grafico componente vs portafoglio (base100)
            st.markdown("#### Grafico (Base=100)")
            df_plot = pd.DataFrame({t: _base100(s_price)}).dropna(how="all")

            if not port_curve.empty:
                port_aligned = port_curve.reindex(df_plot.index).ffill()
                df_plot["PORT"] = _base100(port_aligned)

            df_plot = df_plot.ffill().dropna(how="all")
            figp = _plot_lines(df_plot, ytitle="Index (Base=100)", yfmt=".2f", height=420)
            st.plotly_chart(figp, use_container_width=True, theme=None, key=_k(f"cmp_{t}"))

            # ---- 5.2 drawdown (UN SOLO GRAFICO: asset + port)
            st.markdown("#### Drawdown (asset vs portafoglio)")
            dd_asset = _drawdown_from_equity(s_price)

            dd_port = None
            if not port_curve.empty:
                dd_port = _drawdown_from_equity(port_curve.reindex(dd_asset.index).ffill().dropna())

            st.plotly_chart(
                _plot_drawdown_compare(dd_asset, dd_port, asset_label=t, height=320),
                use_container_width=True,
                theme=None,
                key=_k(f"dd_cmp_{t}"),
            )

            # ---- 5.3 metriche (ticker + port)
            st.markdown("#### Metriche")
            sm = {t: s_price}
            if not port_curve.empty:
                sm["PORT"] = port_curve
            mt_comp = compute_metrics_table(series_map=sm, rf_annual=rf_annual, bench_name=None)
            if mt_comp is None or mt_comp.empty:
                st.info("Metriche non disponibili.")
            else:
                st.dataframe(_style_metrics_table(mt_comp), use_container_width=True, height=520)

            # ---- 5.4 correlazioni (riga ticker)
            st.markdown("#### Correlazioni (vs altri titoli)")
            top_corr_txt = ""
            if isinstance(corr_full, pd.DataFrame) and (not corr_full.empty) and (t in corr_full.columns):
                row = corr_full[t].drop(index=t, errors="ignore").sort_values(ascending=False)
                top = row.head(12)
                st.dataframe(top.to_frame("ρ").round(3), use_container_width=True, height=360)
                top_corr_txt = ", ".join([f"{idx}:{val:.2f}" for idx, val in top.head(6).items()])
            else:
                st.info("Correlazioni non disponibili per questo ticker.")

            # ---- 5.5 settore + marketstack (stub)
            st.markdown("#### Settore / Marketstack info")
            sector = sector_map.get(t, "—")
            st.write("Sector:", sector)
            st.caption("Marketstack: qui puoi mostrare info ticker se hai già una funzione pronta (es. marketstack_get_ticker_info).")

            # ---- 5.6 AI (AUTO) + CACHE
            st.markdown("#### AI (gpt-5-mini) — analisi automatica")

            if i >= MAX_AI:
                st.info(f"AI disattivata oltre i primi {MAX_AI} titoli (performance/costi).")
                continue

            last_px = float(s_price.iloc[-1])
            r_1w = _safe_pct(s_price, 5)
            r_1m = _safe_pct(s_price, 21)
            r_3m = _safe_pct(s_price, 63)
            r_1y = _safe_pct(s_price, 252)

            w_txt = "—"
            if isinstance(w_active, pd.Series) and (t in w_active.index):
                try:
                    w_txt = f"{float(w_active.loc[t])*100:.2f}%"
                except Exception:
                    w_txt = "—"

            # metriche principali (se abbiamo mt_comp)
            metrics_txt = ""
            if isinstance(mt_comp, pd.DataFrame) and (not mt_comp.empty) and (t in mt_comp.columns):
                col = mt_comp[t]
                # prendi poche righe significative se presenti
                keys = ["CAGR", "Volatilità ann.", "Sharpe", "Sortino", "Max Drawdown"]
                parts = []
                for kname in keys:
                    if kname in col.index:
                        v = col.loc[kname]
                        if np.isfinite(v):
                            parts.append(f"{kname}={float(v):.4g}")
                metrics_txt = ", ".join(parts)

            context = (
                f"TICKER={t}\n"
                f"SECTOR={sector}\n"
                f"WEIGHT_IN_PORT={w_txt}\n"
                f"LAST_PRICE={last_px:.4f}\n"
                f"RET_1W={r_1w:.4f}  RET_1M={r_1m:.4f}  RET_3M={r_3m:.4f}  RET_1Y={r_1y:.4f}\n"
                f"DRAWDOWN_MIN={float(dd_asset.min()) if not dd_asset.empty else float('nan'):.4f}\n"
                f"TOP_CORR={top_corr_txt}\n"
                f"METRICS={metrics_txt}\n"
                "Richiesta: interpreta trend/volatilità/drawdown; rischi/catalizzatori; correlazioni e implicazioni di diversificazione."
            )

            try:
                with st.spinner("Generazione analisi AI..."):
                    txt = _ai_analyze_ticker_cached(t, context)
                st.markdown(txt)
            except Exception as ex:
                st.error(f"AI non disponibile: {ex}")
