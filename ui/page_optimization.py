# ui/page_optimization.py
from __future__ import annotations

import re
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from core.portfolio_core import (
    cagr_from_returns,
    es_cvar,
    omega_ratio,
    optimize_weights,  # legacy: sharpe/sortino/cvar
    rachev_ratio,
    sharpe_ratio,
    sortino_ratio,
    vol_ann,
)

# Multi-metric (se esiste)
try:
    from core.portfolio_optimization import optimize_weights_multi  # type: ignore

    _HAS_OPT_MULTI = True
except Exception:
    optimize_weights_multi = None
    _HAS_OPT_MULTI = False


SS_RUNS = "portfolio_runs"
SS_GLOBAL = "portfolio_global_params"


# -------------------------
# Helpers data & portfolio
# -------------------------
def _safe_prices(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    d.index = pd.to_datetime(d.index, errors="coerce")
    d = d.loc[d.index.notna()].sort_index()
    d.index = d.index.normalize()
    d = d[~d.index.duplicated(keep="last")]
    # prova a forzare numerico
    for c in d.columns:
        d[c] = pd.to_numeric(d[c], errors="coerce")
    d = d.dropna(how="all")
    return d


def _rets_assets(df_prices: pd.DataFrame) -> pd.DataFrame:
    d = _safe_prices(df_prices)
    rets = d.pct_change().replace([np.inf, -np.inf], np.nan).dropna(how="all")
    return rets


def _logrets_assets(df_prices: pd.DataFrame) -> pd.DataFrame:
    d = _safe_prices(df_prices)
    lr = np.log(d / d.shift(1)).replace([np.inf, -np.inf], np.nan).dropna(how="all")
    return lr


def _port_rets(rets_assets: pd.DataFrame, w: pd.Series) -> pd.Series:
    w = w.reindex(rets_assets.columns).fillna(0.0).astype(float)
    s = float(w.sum())
    if s != 0:
        w = w / s
    rp = (rets_assets.mul(w, axis=1)).sum(axis=1)
    return rp.replace([np.inf, -np.inf], np.nan).dropna()


def _equity_from_rets(rp: pd.Series, initial: float) -> pd.Series:
    if rp is None or rp.empty:
        return pd.Series(dtype=float)
    return (1.0 + rp).cumprod() * float(initial)


def _max_dd(eq: pd.Series) -> float:
    if eq is None or len(eq) < 2:
        return np.nan
    peak = eq.cummax()
    dd = eq / peak - 1.0
    return float(dd.min())


def _ulcer_index(eq: pd.Series) -> float:
    if eq is None or len(eq) < 2:
        return np.nan
    peak = eq.cummax()
    dd = (eq / peak - 1.0).clip(upper=0.0)  # <= 0
    return float(np.sqrt(np.mean(np.square(dd.values))))


def _calmar(eq: pd.Series, rp: pd.Series) -> float:
    if eq is None or rp is None or rp.empty:
        return np.nan
    c = float(cagr_from_returns(rp))
    mdd = float(_max_dd(eq))
    if not np.isfinite(c) or not np.isfinite(mdd) or mdd >= 0:
        return np.nan
    den = abs(mdd)
    return float(c / den) if den > 1e-12 else np.nan


def _metrics_pack(eq: pd.Series, rp: pd.Series, rf_annual: float, rachev_alpha: float) -> Dict[str, float]:
    if eq is None or rp is None or rp.empty or len(eq) < 2:
        return {}
    out: Dict[str, float] = {}
    out["CAGR"] = float(cagr_from_returns(rp))
    out["Vol (ann.)"] = float(vol_ann(rp))
    out["Sharpe"] = float(sharpe_ratio(rp, rf_annual=rf_annual))
    out["Sortino"] = float(sortino_ratio(rp, rf_annual=rf_annual))
    out["Max Drawdown"] = float(_max_dd(eq))
    out["Calmar"] = float(_calmar(eq, rp))
    out["Ulcer Index"] = float(_ulcer_index(eq))
    out["CVaR 95% (daily)"] = float(es_cvar(rp, 0.95))
    out["CVaR 99% (daily)"] = float(es_cvar(rp, 0.99))
    out["Omega(τ=0)"] = float(omega_ratio(rp, 0.0))
    out[f"Rachev (α={rachev_alpha:.2f})"] = float(rachev_ratio(rp, rachev_alpha))
    return out


def _turnover(w0: pd.Series, w1: pd.Series) -> float:
    a = w0.reindex(w1.index).fillna(0.0).astype(float)
    b = w1.reindex(w1.index).fillna(0.0).astype(float)
    return float(0.5 * np.sum(np.abs((b - a).values)))


def _apply_holding_constraints(w: pd.Series, max_holdings: int, min_w_threshold: float) -> pd.Series:
    ww = w.copy().astype(float)

    if min_w_threshold > 0:
        ww[ww.abs() < float(min_w_threshold)] = 0.0

    if max_holdings and max_holdings > 0 and ww.ne(0).sum() > max_holdings:
        top = ww.abs().sort_values(ascending=False).head(int(max_holdings)).index
        ww.loc[~ww.index.isin(top)] = 0.0

    s = float(ww.sum())
    if s != 0:
        ww = ww / s
    return ww


def _apply_bounds(w: pd.Series, min_w: float, max_w: float) -> pd.Series:
    ww = w.copy().astype(float).clip(lower=float(min_w), upper=float(max_w))
    s = float(ww.sum())
    if s != 0:
        ww = ww / s
    return ww


def _plot_compare_two_base100(eq_a: pd.Series, eq_b: pd.Series, name_a: str, name_b: str) -> None:
    if eq_a is None or eq_b is None or len(eq_a) < 2 or len(eq_b) < 2:
        return
    a = pd.Series(eq_a).dropna()
    b = pd.Series(eq_b).dropna()
    if a.empty or b.empty:
        return

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot((a / float(a.iloc[0])) * 100.0, label=name_a)
    ax.plot((b / float(b.iloc[0])) * 100.0, label=name_b)
    ax.set_title("Equity comparison (Base=100)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    st.pyplot(fig)
    plt.close(fig)


def _parse_sector_map(txt: str) -> dict:
    d: dict = {}
    if not txt or not txt.strip():
        return d
    for pair in re.split(r"[;,]+", txt.strip()):
        if ":" in pair:
            t, s = pair.strip().split(":", 1)
            d[t.strip().upper()] = s.strip()
    return d


def _parse_sector_caps(txt: str) -> dict:
    d: dict = {}
    if not txt or not txt.strip():
        return d
    for pair in re.split(r"[;,]+", txt.strip()):
        if "=" in pair:
            s, v = pair.strip().split("=", 1)
            try:
                d[s.strip()] = float(v.strip())
            except Exception:
                pass
    return d


# -------------------------
# Monte Carlo (no yfinance)
# -------------------------
def _ensure_psd_cov(cov: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    cov = (cov + cov.T) / 2.0
    w, v = np.linalg.eigh(cov)
    w = np.maximum(w, eps)
    return (v * w) @ v.T


def _simulate_increments(
    logret_hist: np.ndarray,   # (T, n)
    days: int,
    sims: int,
    seed: int,
    method: str,               # "Bootstrap" | "GBM" | "t-copula"
    drift_scale: float,
    t_df: int,
) -> np.ndarray:
    """
    Ritorna increments (log-returns) shape (days, sims, n_assets) float32
    """
    rng = np.random.default_rng(int(seed))
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


def _equity_paths_from_increments(
    inc: np.ndarray,          # (days, sims, n_assets)
    last_prices: np.ndarray,  # (n_assets,)
    weights: np.ndarray,      # (n_assets,)
    capital: float,
    mode: str,                # "Rebalance" | "Buy&Hold"
) -> np.ndarray:
    days, sims, n = inc.shape
    w = weights.astype(np.float32)
    s = float(np.sum(w))
    if s <= 0:
        raise RuntimeError("Pesi non validi (somma <= 0).")
    w = w / np.float32(s)

    cap = np.float32(capital)
    equity = np.empty((days + 1, sims), dtype=np.float32)
    equity[0, :] = cap

    if mode == "Rebalance":
        # port_r shape (days, sims)
        port_r = (np.expm1(inc) * w.reshape(1, 1, n)).sum(axis=2)
        equity[1:, :] = cap * np.cumprod(1.0 + port_r, axis=0)
        return equity

    # Buy&Hold: compra quote iniziali e lascia correre
    S0 = last_prices.astype(np.float32)
    shares = (cap * w) / S0  # (n,)

    # prezzi simulati: S_t = S0 * exp(cumsum(inc))
    log_prices = np.cumsum(inc, axis=0)  # (days, sims, n)
    prices = S0.reshape(1, 1, n) * np.exp(log_prices)  # (days, sims, n)
    equity[1:, :] = (prices * shares.reshape(1, 1, n)).sum(axis=2)
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


def _try_plotly():
    try:
        import plotly.graph_objects as go  # type: ignore
        import plotly.express as px  # type: ignore

        return go, px
    except Exception:
        return None, None


def _render_monte_carlo_section(
    prices_window: pd.DataFrame,
    tickers: List[str],
    w_current: pd.Series,
    w_opt: Optional[pd.Series],
    initial_capital: float,
    currency: str = "€",
    key_prefix: str = "mc",
) -> None:
    st.markdown("### 🎲 Monte Carlo (scenario simulation) — senza yfinance")
    st.caption("Simula scenari usando i prezzi già presenti nel portafoglio (window selezionata).")

    if prices_window is None or prices_window.empty or len(tickers) == 0:
        st.info("Prezzi window non disponibili per Monte Carlo.")
        return

    lr = _logrets_assets(prices_window[tickers]).dropna()
    if lr.shape[0] < 30:
        st.warning("Storico insufficiente per Monte Carlo (serve almeno ~30 osservazioni).")
        return

    # ---- controlli
    cA, cB, cC, cD = st.columns([1.1, 1.0, 1.0, 1.0], gap="small")
    with cA:
        horizon_days = st.slider("Horizon (days)", 30, 3650, 365, 5, key=f"{key_prefix}__days")
    with cB:
        sims = st.slider("Runs", 500, 100_000, 25_000, 500, key=f"{key_prefix}__sims")
    with cC:
        plot_sims = st.slider("Visible paths", 0, 1500, 450, 50, key=f"{key_prefix}__plot_sims")
    with cD:
        method = st.selectbox("Model", ["Bootstrap", "GBM", "t-copula"], index=0, key=f"{key_prefix}__model")

    cE, cF, cG, cH = st.columns([1.0, 1.0, 1.0, 1.0], gap="small")
    with cE:
        mode = st.selectbox("Portfolio mode", ["Rebalance", "Buy&Hold"], index=0, key=f"{key_prefix}__mode")
    with cF:
        alpha = st.select_slider("VaR alpha", options=[0.01, 0.025, 0.05, 0.10], value=0.05, key=f"{key_prefix}__alpha")
    with cG:
        drift_scale = st.slider("Drift scale", 0.0, 2.0, 1.0, 0.05, key=f"{key_prefix}__drift")
    with cH:
        seed = st.number_input("Seed", min_value=0, max_value=10_000_000, value=42, step=1, key=f"{key_prefix}__seed")

    choices = ["Current"]
    if w_opt is not None and isinstance(w_opt, pd.Series) and w_opt.abs().sum() > 0:
        choices += ["Optimized", "Both"]
    sim_target = st.selectbox("Simulate", choices, index=0, key=f"{key_prefix}__target")

    # guard RAM
    n_assets = lr.shape[1]
    est = int(horizon_days) * int(sims) * int(n_assets)
    if est > 160_000_000:
        st.error("Parametri troppo grandi per RAM (days*sims*assets). Riduci Runs o Horizon.")
        return

    run_mc = st.button("▶ RUN Monte Carlo", use_container_width=True, key=f"{key_prefix}__run")
    if not run_mc:
        st.info("Imposta i parametri e premi **RUN Monte Carlo**.")
        return

    go, px = _try_plotly()
    if go is None or px is None:
        st.error("Plotly non disponibile: aggiungi `plotly` in requirements.txt e redeploy.")
        return

    last_prices = prices_window[tickers].iloc[-1].to_numpy(dtype=np.float32)

    w_cur_np = w_current.reindex(tickers).fillna(0.0).to_numpy(dtype=np.float32)
    w_opt_np = None
    if w_opt is not None:
        w_opt_np = w_opt.reindex(tickers).fillna(0.0).to_numpy(dtype=np.float32)

    with st.spinner("Running Monte Carlo..."):
        inc = _simulate_increments(
            logret_hist=lr.values,
            days=int(horizon_days),
            sims=int(sims),
            seed=int(seed),
            method=str(method),
            drift_scale=float(drift_scale),
            t_df=int(7),  # puoi esporlo se vuoi
        )

        paths_cur = _equity_paths_from_increments(
            inc=inc,
            last_prices=last_prices,
            weights=w_cur_np,
            capital=float(initial_capital),
            mode=str(mode),
        )

        paths_opt = None
        if sim_target in ("Optimized", "Both") and w_opt_np is not None:
            paths_opt = _equity_paths_from_increments(
                inc=inc,
                last_prices=last_prices,
                weights=w_opt_np,
                capital=float(initial_capital),
                mode=str(mode),
            )

        bd_cur = _compute_bands(paths_cur, float(alpha))
        bd_opt = _compute_bands(paths_opt, float(alpha)) if paths_opt is not None else None

    day_sel = st.slider("Focus day", 0, int(horizon_days), min(30, int(horizon_days)), key=f"{key_prefix}__focus")

    def _kpis(paths: np.ndarray) -> dict:
        vals = paths[int(day_sel)].astype(np.float64)
        mean_v = float(vals.mean())
        med_v = float(np.quantile(vals, 0.50))
        var_v = float(np.quantile(vals, float(alpha)))
        cvar_v = _cvar(vals, float(alpha))
        best_v = float(np.quantile(vals, 1.0 - float(alpha)))
        p_loss = float((vals < float(initial_capital)).mean())
        return {"vals": vals, "mean": mean_v, "median": med_v, "var": var_v, "cvar": cvar_v, "best": best_v, "p_loss": p_loss}

    k_cur = _kpis(paths_cur)
    k_opt = _kpis(paths_opt) if paths_opt is not None else None

    # KPI tiles
    c1, c2, c3, c4, c5, c6 = st.columns(6, gap="small")
    with c1:
        st.metric("CAPITAL", f"{initial_capital:,.2f}{currency}")
    with c2:
        st.metric("MEAN", f"{k_cur['mean']:,.2f}{currency}", delta=f"{(k_cur['mean']-initial_capital):+,.2f}{currency}")
    with c3:
        st.metric(f"VaR (q{alpha*100:.1f}%)", f"{k_cur['var']:,.2f}{currency}", delta=f"{(k_cur['var']-initial_capital):+,.2f}{currency}")
    with c4:
        st.metric("CVaR (ES)", f"{k_cur['cvar']:,.2f}{currency}", delta=f"{(k_cur['cvar']-initial_capital):+,.2f}{currency}")
    with c5:
        st.metric("P(LOSS)", f"{k_cur['p_loss']*100:.1f}%")
    with c6:
        st.metric("MEAN (OPT)", f"{k_opt['mean']:,.2f}{currency}" if k_opt else "—")

    left, right = st.columns([2.35, 1.0], gap="small")
    x = np.arange(int(horizon_days) + 1)

    # ---- FAN CHART + PATHS (come la prima immagine)
    with left:
        fig = go.Figure()

        # banda quantile current
        fig.add_trace(go.Scatter(x=x, y=bd_cur["hi"], mode="lines", line=dict(width=0), hoverinfo="skip", showlegend=False))
        fig.add_trace(
            go.Scatter(
                x=x,
                y=bd_cur["lo"],
                mode="lines",
                fill="tonexty",
                name=f"Band {alpha}-{1-alpha}",
                fillcolor="rgba(59,130,246,0.18)",
                line=dict(color="rgba(59,130,246,0.0)"),
                hovertemplate="Day %{x}<br>Equity %{y:.2f}<extra></extra>",
            )
        )

        fig.add_trace(go.Scatter(x=x, y=bd_cur["mean"], mode="lines", name="Mean", line=dict(color="rgba(229,231,235,0.95)", width=2)))
        fig.add_trace(go.Scatter(x=x, y=bd_cur["med"], mode="lines", name="Median", line=dict(color="rgba(59,130,246,0.95)", width=2, dash="dot")))

        # sample paths (nuvola)
        if int(plot_sims) > 0:
            rng = np.random.default_rng(int(seed))
            ncols = paths_cur.shape[1]
            k = min(int(plot_sims), ncols)
            idx = rng.choice(ncols, size=k, replace=False)
            sample = paths_cur[:, idx]
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

        fig.add_vline(x=int(day_sel), line_width=1, line_dash="dash", opacity=0.6, line_color="rgba(148,163,184,0.55)")

        fig.update_layout(
            template="plotly_dark",
            height=640,
            margin=dict(l=10, r=10, t=10, b=10),
            paper_bgcolor="#0E1521",
            plot_bgcolor="#0E1521",
            xaxis_title="Days",
            yaxis_title=f"Equity ({currency})",
            hovermode="x unified",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0, font=dict(size=11)),
            font=dict(color="#E5E7EB"),
        )
        fig.update_xaxes(showgrid=True, gridcolor="rgba(148,163,184,0.10)", zeroline=False)
        fig.update_yaxes(showgrid=True, gridcolor="rgba(148,163,184,0.10)", zeroline=False)

        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    # ---- DISTRIBUTION (come la prima immagine)
    with right:
        hdf = pd.DataFrame({"equity": k_cur["vals"]})
        hist = px.histogram(hdf, x="equity", nbins=55)
        hist.add_vline(x=k_cur["var"], line_dash="dash", line_width=2, line_color="rgba(239,68,68,0.90)")
        hist.add_vline(x=k_cur["mean"], line_width=2, line_color="rgba(229,231,235,0.90)")
        hist.add_vline(x=k_cur["best"], line_dash="dash", line_width=2, line_color="rgba(16,185,129,0.90)")
        hist.update_layout(
            template="plotly_dark",
            height=640,
            margin=dict(l=10, r=10, t=10, b=10),
            paper_bgcolor="#0E1521",
            plot_bgcolor="#0E1521",
            xaxis_title=f"Equity ({currency})",
            yaxis_title="Count",
            showlegend=False,
            font=dict(color="#E5E7EB"),
        )
        hist.update_xaxes(showgrid=True, gridcolor="rgba(148,163,184,0.10)", zeroline=False)
        hist.update_yaxes(showgrid=True, gridcolor="rgba(148,163,184,0.10)", zeroline=False)

        st.plotly_chart(hist, use_container_width=True, config={"displayModeBar": False})

    # Export
    out = pd.DataFrame(
        {
            "day": x,
            "cur_mean": bd_cur["mean"].astype(np.float64),
            "cur_median": bd_cur["med"].astype(np.float64),
            f"cur_q{alpha:.3f}": bd_cur["lo"].astype(np.float64),
            f"cur_q{1-alpha:.3f}": bd_cur["hi"].astype(np.float64),
        }
    )
    if bd_opt is not None:
        out["opt_mean"] = bd_opt["mean"].astype(np.float64)
        out["opt_median"] = bd_opt["med"].astype(np.float64)
        out[f"opt_q{alpha:.3f}"] = bd_opt["lo"].astype(np.float64)
        out[f"opt_q{1-alpha:.3f}"] = bd_opt["hi"].astype(np.float64)

    st.download_button(
        "⬇ Export Quantiles CSV",
        data=out.to_csv(index=False).encode("utf-8"),
        file_name="montecarlo_quantiles.csv",
        mime="text/csv",
        use_container_width=True,
    )


# =========================================================
# ENTRYPOINT (compatibile con chiamate diverse)
# =========================================================
def render_optimization_page(*args, **kwargs) -> None:
    """
    Compatibilità:
      - render_optimization_page(runs, pid, state)
      - render_optimization_page(res=runs, pid=..., state=...)
      - render_optimization_page(runs=..., pid=..., state=...)
    """
    runs = kwargs.get("runs") or kwargs.get("res") or (args[0] if len(args) > 0 else None)
    pid = kwargs.get("pid") or (args[1] if len(args) > 1 else None)
    state = kwargs.get("state") or (args[2] if len(args) > 2 else st.session_state)

    st.subheader("Ottimizzazione — What-if + vincoli + confronto + Monte Carlo")

    if not isinstance(runs, dict) or not runs:
        st.info("Definisci portafogli e premi **Run / Update (ALL)**.")
        return

    if pid is None or pid not in runs:
        pid = list(runs.keys())[0]

    r = runs.get(pid) or {}
    if "prices" not in r or r["prices"] is None or getattr(r["prices"], "empty", True):
        st.info("Questo portafoglio non ha prices validi.")
        return

    g = state.get(SS_GLOBAL, {}) or {}
    rf_global = float(g.get("rf_annual", 0.01))
    rachev_alpha = float(g.get("rachev_alpha", 0.05))
    initial_capital = float(g.get("initial_capital", 100_000.0))
    currency = str(g.get("currency", "€"))

    prices_now = _safe_prices(r["prices"].copy())
    cols_px = list(prices_now.columns)
    if not cols_px:
        st.info("Prices senza colonne/tickers.")
        return

    # current weights
    w_raw = r.get("weights")
    if w_raw is None:
        w_current = pd.Series(1.0 / len(cols_px), index=cols_px, dtype=float)
    else:
        if isinstance(w_raw, pd.Series):
            w_current = w_raw.reindex(cols_px).fillna(0.0).astype(float)
        else:
            try:
                w_current = pd.Series(list(w_raw), index=cols_px, dtype=float)
            except Exception:
                w_current = pd.Series(0.0, index=cols_px, dtype=float)

    s0 = float(w_current.sum())
    if s0 != 0:
        w_current = w_current / s0

    st.markdown(f"**Portfolio attivo:** {r.get('name', pid)}")

    cL, cR = st.columns([1.15, 1.0], gap="large")

    # ---------- LEFT: controls ----------
    with cL:
        st.markdown("### 🎯 Setup (what-if)")

        opt_mode = st.radio(
            "Optimization window",
            ["Last N days", "Custom dates (within prices)"],
            index=0,
            horizontal=True,
            key=f"opt_window_mode__{pid}",
        )

        dmin = pd.to_datetime(prices_now.index.min()).date()
        dmax = pd.to_datetime(prices_now.index.max()).date()

        if opt_mode == "Last N days":
            n_days = st.number_input("N days", min_value=60, max_value=5000, value=756, step=21, key=f"opt_lastn__{pid}")
            p_end = dmax
            p_start = (pd.to_datetime(p_end) - pd.Timedelta(days=int(n_days))).date()
            if p_start < dmin:
                p_start = dmin
        else:
            cc1, cc2 = st.columns(2)
            with cc1:
                p_start = st.date_input(
                    "From",
                    value=max(dmin, (pd.to_datetime(dmax) - pd.Timedelta(days=756)).date()),
                    min_value=dmin,
                    max_value=dmax,
                    key=f"opt_from__{pid}",
                )
            with cc2:
                p_end = st.date_input("To", value=dmax, min_value=dmin, max_value=dmax, key=f"opt_to__{pid}")

        st.caption(f"Window: **{p_start} → {p_end}**")

        BASE_METRICS = ["max_sharpe", "max_sortino", "min_cvar95", "min_cvar99"]
        EXTRA_METRICS = [
            "min_vol",
            "max_return",
            "max_cagr",
            "min_drawdown",
            "max_calmar",
            "min_ulcer",
            "max_omega0",
            "max_rachev",
        ]
        ALL_METRICS = BASE_METRICS + EXTRA_METRICS

        st.markdown("### 🧠 Obiettivo")
        mode2 = st.radio(
            "Optimization mode",
            ["Single metric", "Multi-metric (composite)"],
            horizontal=True,
            key=f"opt_mode2__{pid}",
        )

        if (mode2 == "Multi-metric (composite)") and (not _HAS_OPT_MULTI):
            st.warning("Per multi-metrica/metriche extra serve `optimize_weights_multi` in core (non disponibile ora).")

        sel: List[str]
        w_map: Dict[str, float]

        if mode2 == "Multi-metric (composite)":
            sel = st.multiselect(
                "Select metrics",
                options=ALL_METRICS,
                default=["max_sharpe", "min_drawdown"],
                key=f"opt_multi_metrics__{pid}",
            )
            w_map = {}
            if sel:
                st.caption("Pesi relativi (normalizzati automaticamente).")
                cols_w = st.columns(min(3, len(sel)))
                for i, m in enumerate(sel):
                    with cols_w[i % len(cols_w)]:
                        w_map[m] = st.number_input(
                            f"w({m})",
                            min_value=0.0,
                            max_value=10.0,
                            value=1.0,
                            step=0.1,
                            key=f"opt_w__{pid}__{m}",
                        )
            else:
                w_map = {}
        else:
            obj = st.selectbox("Objective", ALL_METRICS, index=0, key=f"opt_objective__{pid}")
            sel = [obj]
            w_map = {obj: 1.0}

        st.markdown("### ⚙️ Parametri base")
        cc1, cc2, cc3 = st.columns(3)
        with cc1:
            min_w = st.number_input("Min weight", value=0.0, step=0.01, key=f"opt_min_w__{pid}")
        with cc2:
            max_w = st.number_input("Max weight", value=1.0, step=0.01, key=f"opt_max_w__{pid}")
        with cc3:
            rf_opt_in = st.number_input("RF annual (optimization)", value=rf_global, step=0.001, format="%.3f", key=f"opt_rf__{pid}")

        st.markdown("### 🧱 Vincoli extra")
        c4, c5, c6 = st.columns(3)
        with c4:
            max_holdings = st.number_input("Max holdings (0=off)", min_value=0, max_value=100, value=0, step=1, key=f"opt_max_holdings__{pid}")
        with c5:
            min_thr = st.number_input("Min weight threshold", min_value=0.0, max_value=0.20, value=0.0, step=0.005, format="%.3f", key=f"opt_min_thr__{pid}")
        with c6:
            turnover_cap = st.number_input("Turnover cap (0=off)", min_value=0.0, max_value=2.0, value=0.0, step=0.05, format="%.2f", key=f"opt_turnover__{pid}")

        tcost_bps = st.number_input("Transaction cost (bps)", min_value=0.0, max_value=200.0, value=0.0, step=1.0, key=f"opt_tcost__{pid}")

        with st.expander("🏷️ Sector constraints (optional)", expanded=False):
            sector_map_text = st.text_area("Sector map (Ticker:Sector)", value="", placeholder="AAPL:Tech, MSFT:Tech, XOM:Energy", key=f"opt_sector_map__{pid}")
            sector_caps_text = st.text_area("Sector caps (Sector=cap)", value="", placeholder="Tech=0.6, Energy=0.3", key=f"opt_sector_caps__{pid}")

        do_opt = st.button("🚀 Optimize now", type="primary", use_container_width=True, key=f"btn_optimize__{pid}")

    # ---------- RIGHT: snapshot ----------
    with cR:
        st.markdown("### 📌 Current snapshot")
        st.dataframe(
            pd.DataFrame({"Weight": w_current}).sort_values("Weight", ascending=False).style.format({"Weight": "{:.2%}"}),
            use_container_width=True,
            height=260,
        )

        rets_all = _rets_assets(prices_now)
        rets_win = rets_all.loc[(rets_all.index >= pd.to_datetime(p_start)) & (rets_all.index <= pd.to_datetime(p_end))].copy()

        if rets_win.empty:
            st.warning("Finestra returns vuota: allarga la window.")
        else:
            rp_cur = _port_rets(rets_win, w_current)
            eq_cur = _equity_from_rets(rp_cur, initial_capital)
            kpi_cur = _metrics_pack(eq_cur, rp_cur, rf_global, rachev_alpha)
            if kpi_cur:
                st.markdown("**KPI (Current, window)**")
                st.dataframe(pd.DataFrame({"Current": kpi_cur}), use_container_width=True)

    # ---------- optimization run ----------
    opt_key = f"opt_res__{pid}"

    if do_opt:
        try:
            rets_all = _rets_assets(prices_now)
            rets_win = rets_all.loc[(rets_all.index >= pd.to_datetime(p_start)) & (rets_all.index <= pd.to_datetime(p_end))].copy()

            if rets_win.empty or rets_win.shape[1] == 0:
                st.error("Not enough data: returns window vuota.")
            else:
                bounds = (float(min_w), float(max_w))
                if bounds[0] < 0 or bounds[1] > 1 or bounds[0] > bounds[1]:
                    st.error("Limiti pesi non validi (range [0,1] e min <= max).")
                else:
                    sector_map = _parse_sector_map(sector_map_text)
                    sector_caps = _parse_sector_caps(sector_caps_text)

                    need_multi = (mode2 == "Multi-metric (composite)") or (sel and sel[0] not in BASE_METRICS)

                    if need_multi:
                        if not _HAS_OPT_MULTI or optimize_weights_multi is None:
                            raise RuntimeError("Metriche extra/multi richiedono optimize_weights_multi in core (non presente).")

                        w_opt_raw = optimize_weights_multi(
                            rets_assets=rets_win,
                            rf_annual=float(rf_opt_in),
                            metrics=sel,
                            metric_weights=w_map,
                            bounds=bounds,
                            sector_map=sector_map if sector_map else None,
                            sector_caps=sector_caps if sector_caps else None,
                            tail_alpha=float(rachev_alpha),
                            omega_tau=0.0,
                            n_starts=10,
                        )
                    else:
                        w_opt_raw = optimize_weights(
                            rets_assets=rets_win,
                            rf_annual=float(rf_opt_in),
                            objective=sel[0],
                            bounds=bounds,
                            sector_map=sector_map if sector_map else None,
                            sector_caps=sector_caps if sector_caps else None,
                        )

                    cols_all = list(rets_win.columns)
                    if isinstance(w_opt_raw, pd.Series):
                        w_opt = w_opt_raw.reindex(cols_all).fillna(0.0).astype(float)
                    else:
                        w_opt = pd.Series(w_opt_raw, index=cols_all, dtype=float)

                    s = float(w_opt.sum())
                    if s != 0:
                        w_opt = w_opt / s

                    w_opt = _apply_holding_constraints(w_opt, int(max_holdings), float(min_thr))
                    w_opt = _apply_bounds(w_opt, float(min_w), float(max_w))

                    to = _turnover(w_current.reindex(w_opt.index).fillna(0.0), w_opt)
                    cost_pen = to * (float(tcost_bps) / 10000.0)

                    rp_opt = _port_rets(rets_win, w_opt)
                    if rp_opt.empty:
                        raise ValueError("Portfolio returns vuote dopo optimization.")

                    if cost_pen > 0 and len(rp_opt) > 0:
                        rp_opt.iloc[0] = float(rp_opt.iloc[0]) - float(cost_pen)

                    eq_opt = _equity_from_rets(rp_opt, initial_capital)

                    state[opt_key] = {
                        "w_opt": w_opt,
                        "turnover": to,
                        "cost_pen": cost_pen,
                        "eq_opt": eq_opt,
                        "rp_opt": rp_opt,
                        "window": (p_start, p_end),
                        "metrics": sel,
                        "metric_weights": w_map,
                        "bounds": bounds,
                        "rf_opt": float(rf_opt_in),
                    }

                    st.success("Ottimizzazione completata (quick backtest).")

        except Exception as e:
            st.error(f"Optimization error: {e}")

    # ---------- render results ----------
    opt_res = state.get(opt_key)
    w_opt: Optional[pd.Series] = None

    if opt_res is not None and isinstance(opt_res, dict):
        w_opt = opt_res.get("w_opt", None)
        eq_opt = opt_res.get("eq_opt", None)
        rp_opt = opt_res.get("rp_opt", None)
        to = float(opt_res.get("turnover", np.nan))
        cost_pen = float(opt_res.get("cost_pen", 0.0))

        st.markdown("---")
        st.markdown("## ✅ Results (Current vs Optimized)")

        rets_all = _rets_assets(prices_now)
        rets_win = rets_all.loc[(rets_all.index >= pd.to_datetime(p_start)) & (rets_all.index <= pd.to_datetime(p_end))].copy()
        rp_cur = _port_rets(rets_win, w_current)
        eq_cur = _equity_from_rets(rp_cur, initial_capital)

        m_cur = _metrics_pack(eq_cur, rp_cur, rf_global, rachev_alpha)
        m_opt = _metrics_pack(eq_opt, rp_opt, rf_global, rachev_alpha) if (eq_opt is not None and rp_opt is not None) else {}

        if m_cur and m_opt:
            dfk = pd.DataFrame({"Current": m_cur, "Optimized": m_opt})
            st.dataframe(dfk, use_container_width=True)

        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("Turnover", f"{to:.2%}" if np.isfinite(to) else "–")
        with c2:
            st.metric("Cost penalty (approx)", f"{cost_pen*100:.2f}%")
        with c3:
            st.metric("Holdings (opt)", int((w_opt.abs() > 0).sum()) if isinstance(w_opt, pd.Series) else 0)

        if isinstance(w_opt, pd.Series):
            dfw = pd.DataFrame(
                {"Current": w_current.reindex(w_opt.index).fillna(0.0), "Optimized": w_opt.reindex(w_opt.index).fillna(0.0)}
            )
            dfw["Δ"] = dfw["Optimized"] - dfw["Current"]
            dfw["|Δ|"] = dfw["Δ"].abs()
            dfw = dfw.sort_values("|Δ|", ascending=False)

            st.markdown("### 🧾 Weights (delta view)")
            st.dataframe(
                dfw.style.format({"Current": "{:.2%}", "Optimized": "{:.2%}", "Δ": "{:+.2%}", "|Δ|": "{:.2%}"}),
                use_container_width=True,
                height=420,
            )

        st.markdown("### 📈 Equity comparison (Base=100)")
        if eq_opt is not None:
            _plot_compare_two_base100(eq_cur, eq_opt, "Current", "Optimized")

        # turnover cap check
        if float(turnover_cap) > 0 and np.isfinite(to) and to > float(turnover_cap) + 1e-12:
            st.warning(f"Turnover cap violato: {to:.2%} > {float(turnover_cap):.2%}.")

    # ---------- Monte Carlo integrated ----------
    st.markdown("---")
    prices_window = prices_now.loc[(prices_now.index >= pd.to_datetime(p_start)) & (prices_now.index <= pd.to_datetime(p_end))].copy()
    with st.expander("🎲 Monte Carlo (integrato in Ottimizzazione)", expanded=False):
        _render_monte_carlo_section(
            prices_window=prices_window,
            tickers=cols_px,
            w_current=w_current,
            w_opt=w_opt if isinstance(w_opt, pd.Series) else None,
            initial_capital=float(initial_capital),
            currency=currency,
            key_prefix=f"mc__{pid}",
        )
