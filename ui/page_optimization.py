# ui/page_optimization.py — Sezione "Ottimizzazione" (What-if + multi-metric)
from __future__ import annotations

import re
from typing import Dict, List, Optional
from ui.page_optimization import render_optimization_page

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
    optimize_weights,  # legacy: sharpe/sortino/cvar
)

# Se esiste, abilita metriche extra + multi-metric
try:
    from core.portfolio_optimization import optimize_weights_multi  # type: ignore
    _HAS_OPT_MULTI = True
except Exception:
    optimize_weights_multi = None
    _HAS_OPT_MULTI = False


SS_RUNS = "portfolio_runs"
SS_ACTIVE = "active_portfolio_pid"
SS_GLOBAL = "portfolio_global_params"


# -------------------------
# Helpers data
# -------------------------
def _safe_prices(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    d.index = pd.to_datetime(d.index, errors="coerce")
    d = d.loc[d.index.notna()].sort_index()
    d.index = d.index.normalize()
    d = d[~d.index.duplicated(keep="last")]
    return d


def _rets_assets(df_prices: pd.DataFrame) -> pd.DataFrame:
    d = _safe_prices(df_prices)
    rets = d.pct_change().replace([np.inf, -np.inf], np.nan).dropna(how="all")
    return rets


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
    dd = (eq / peak - 1.0).clip(upper=0.0)  # negativo/0
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
    out["ES/CVaR 95% (daily)"] = float(es_cvar(rp, 0.95))
    out["ES/CVaR 99% (daily)"] = float(es_cvar(rp, 0.99))
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
    d = {}
    if not txt or not txt.strip():
        return d
    for pair in re.split(r"[;,]+", txt.strip()):
        if ":" in pair:
            t, s = pair.strip().split(":", 1)
            d[t.strip().upper()] = s.strip()
    return d


def _parse_sector_caps(txt: str) -> dict:
    d = {}
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


# =========================================================
# ✅ ENTRYPOINT richiesto dall'import
# =========================================================
def render_optimization_page(runs: Dict[str, dict], pid: str, state) -> None:
    """
    UI ottimizzazione modulare. `runs` è il dict dei portafogli calcolati.
    `pid` è il portfolio attivo. `state` è st.session_state.
    """

    if not runs:
        st.info("Run multi prima.")
        return

    if pid not in runs:
        pid = list(runs.keys())[0]

    r = runs.get(pid)
    if not r or "prices" not in r or r["prices"] is None or r["prices"].empty:
        st.info("Questo portafoglio non ha prices validi.")
        return

    g = state.get(SS_GLOBAL, {}) or {}
    rf_global = float(g.get("rf_annual", 0.01))
    rachev_alpha = float(g.get("rachev_alpha", 0.05))
    initial_capital = float(g.get("initial_capital", 100_000.0))

    prices_now = _safe_prices(r["prices"].copy())
    cols_px = list(prices_now.columns)

    # current weights
    w_raw = r.get("weights")
    if w_raw is None:
        w_current = pd.Series(1.0 / len(cols_px), index=cols_px, dtype=float) if cols_px else pd.Series(dtype=float)
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
                p_start = st.date_input("From", value=max(dmin, (pd.to_datetime(dmax) - pd.Timedelta(days=756)).date()),
                                        min_value=dmin, max_value=dmax, key=f"opt_from__{pid}")
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
            st.warning("Per multi-metrica/metriche extra serve `optimize_weights_multi` in core. Ora non è disponibile.")

        sel: List[str] = []
        w_map: Dict[str, float] = {}

        if mode2 == "Multi-metric (composite)":
            sel = st.multiselect(
                "Select metrics",
                options=ALL_METRICS,
                default=["max_sharpe", "min_drawdown"],
                key=f"opt_multi_metrics__{pid}",
            )
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
            sector_map_text = st.text_area("Sector map (Ticker:Sector)", value="", placeholder="AAPL:Tech, MSFT:Tech, XOM:Energy",
                                           key=f"opt_sector_map__{pid}")
            sector_caps_text = st.text_area("Sector caps (Sector=cap)", value="", placeholder="Tech=0.6, Energy=0.3",
                                            key=f"opt_sector_caps__{pid}")

        do_opt = st.button("🚀 Optimize now", type="primary", use_container_width=True, key=f"btn_optimize__{pid}")

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

                    # Decide solver
                    need_multi = (mode2 == "Multi-metric (composite)") or (sel and sel[0] not in ["max_sharpe", "max_sortino", "min_cvar95", "min_cvar99"])

                    if need_multi:
                        if not _HAS_OPT_MULTI or optimize_weights_multi is None:
                            raise RuntimeError("Metriche extra / multi-metric richiedono optimize_weights_multi (non disponibile).")

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

                    # vincoli extra
                    w_opt = _apply_holding_constraints(w_opt, int(max_holdings), float(min_thr))
                    w_opt = _apply_bounds(w_opt, float(min_w), float(max_w))

                    # turnover + tcost
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
    if opt_res is None or not isinstance(opt_res, dict):
        return

    w_opt = opt_res["w_opt"].copy()
    eq_opt = opt_res["eq_opt"].copy()
    rp_opt = opt_res["rp_opt"].copy()
    to = float(opt_res.get("turnover", np.nan))
    cost_pen = float(opt_res.get("cost_pen", 0.0))

    st.markdown("---")
    st.markdown("## ✅ Results (Current vs Optimized)")

    rets_all = _rets_assets(prices_now)
    rets_win = rets_all.loc[(rets_all.index >= pd.to_datetime(p_start)) & (rets_all.index <= pd.to_datetime(p_end))].copy()
    rp_cur = _port_rets(rets_win, w_current)
    eq_cur = _equity_from_rets(rp_cur, initial_capital)

    m_cur = _metrics_pack(eq_cur, rp_cur, rf_global, rachev_alpha)
    m_opt = _metrics_pack(eq_opt, rp_opt, rf_global, rachev_alpha)

    if m_cur and m_opt:
        dfk = pd.DataFrame({"Current": m_cur, "Optimized": m_opt})
        st.dataframe(dfk, use_container_width=True)

    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Turnover", f reopening the full task completed quickly. Yes. you can apply. f"{to:.2%}" if np.isfinite(to) else "–")
    with c2:
        st.metric("Cost penalty (approx)", f"{cost_pen*100:.2f}%")
    with c3:
        st.metric("Holdings (opt)", int((w_opt.abs() > 0).sum()))

    dfw = pd.DataFrame({
        "Current": w_current.reindex(w_opt.index).fillna(0.0),
        "Optimized": w_opt.reindex(w_opt.index).fillna(0.0),
    })
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
    _plot_compare_two_base100(eq_cur, eq_opt, "Current", "Optimized")

    # turnover cap check
    turnover_cap = float(state.get(f"opt_turnover__{pid}", 0.0))
    if turnover_cap > 0 and np.isfinite(to) and to > turnover_cap + 1e-12:
        st.warning(f"Turnover cap violato: {to:.2%} > {turnover_cap:.2%}.")

    # add to comparison
    cA, cB = st.columns([1.2, 2.0])
    with cA:
        add_now = st.button("➕ Add optimized to comparison", use_container_width=True, key=f"add_opt__{pid}")
    with cB:
        st.caption("Aggiunge un nuovo portafoglio *_OPT* usando gli stessi prices (no refetch).")

    if add_now:
        new_pid = f"{pid}_OPT"
        new_name = f"{r.get('name', pid)} — Optimized"

        res_opt = dict(r)
        res_opt["pid"] = new_pid
        res_opt["name"] = new_name
        res_opt["prices"] = prices_now.copy()
        res_opt["weights"] = w_opt.reindex(prices_now.columns).fillna(0.0).astype(float).values.tolist()
        res_opt["equity"] = eq_opt.copy()

        runs[new_pid] = res_opt
        state[SS_RUNS] = runs
        st.success("Aggiunto al confronto: portafoglio ottimizzato.")
