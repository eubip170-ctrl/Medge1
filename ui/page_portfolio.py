# ui/page_portfolio.py — Pagina "Portafoglio" (Overview, Analisi, Testing, Ottimizzazione)
from __future__ import annotations

import re
import uuid
from datetime import date
from typing import Dict, List, Optional, Tuple
from ui.page_optimization import render_optimization_page

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from ui.page_testing import render_testing_page
from ui.analisi_page import render_analisi_page

from core.portfolio_core import (
    compute_pipeline,
    cagr_from_returns,
    vol_ann,
    sharpe_ratio,
    sortino_ratio,
    drawdown_series,
    es_cvar,
    omega_ratio,
    rachev_ratio,
    risk_contributions,
    optimize_weights,
)

# =========================================================
# Session keys
# =========================================================
SS_SPECS = "portfolio_specs"      # list[dict]
SS_RUNS = "portfolio_runs"        # dict[pid -> res dict]
SS_ACTIVE = "active_portfolio_pid"
SS_GLOBAL = "portfolio_global_params"


# =========================================================
# CSS — stile come screenshot (blu scuro + stepper bianco)
# =========================================================
def _inject_ui_css() -> None:
    st.markdown(
        """
<style>
:root{
  --bg-input: #0b1633;
  --bd-input: rgba(59,130,246,.55);
  --bd-input-strong: rgba(59,130,246,.85);
  --txt: #ffffff;
  --muted: rgba(255,255,255,.78);
  --btn-step-bg: #ffffff;
  --btn-step-txt: #0b2a66;
}

div[data-testid="stTextInput"] input,
div[data-testid="stNumberInput"] input{
  background: var(--bg-input) !important;
  color: var(--txt) !important;
  border: 1px solid var(--bd-input) !important;
  border-radius: 10px !important;
  font-weight: 800 !important;
}

div[data-testid="stTextInput"] input::placeholder,
div[data-testid="stNumberInput"] input::placeholder{
  color: var(--muted) !important;
}

div[data-testid="stTextInput"] input:focus,
div[data-testid="stNumberInput"] input:focus{
  outline: none !important;
  border: 1px solid var(--bd-input-strong) !important;
  box-shadow: 0 0 0 3px rgba(59,130,246,.25) !important;
}

div[data-testid="stNumberInput"] button{
  background: var(--btn-step-bg) !important;
  color: var(--btn-step-txt) !important;
  border: 1px solid rgba(0,0,0,.08) !important;
  border-radius: 10px !important;
  font-weight: 900 !important;
}

div[data-testid="stNumberInput"]{
  margin-top: -6px;
}

div[data-testid="stCheckbox"] label{
  color: rgba(255,255,255,.85) !important;
  font-weight: 700 !important;
}
</style>
        """,
        unsafe_allow_html=True,
    )


# =========================================================
# Helpers (generic)
# =========================================================
def _grid_cols(n: int) -> int:
    return 1 if n <= 1 else (2 if n <= 4 else 3)


def _new_rid() -> str:
    return uuid.uuid4().hex[:10]


def _equal_split(total: float, n: int, decimals: int = 2) -> List[float]:
    if n <= 0:
        return []
    base = round(total / n, decimals)
    vals = [base] * n
    diff = round(total - sum(vals), decimals)
    vals[-1] = round(vals[-1] + diff, decimals)
    return [max(0.0, float(v)) for v in vals]


def _round_fix_total(v: np.ndarray, total: float = 100.0, decimals: int = 2, fix_idx: Optional[int] = None) -> List[float]:
    vv = np.clip(v.astype(float), 0.0, np.inf)
    vv = np.round(vv, decimals)
    diff = round(total - float(np.sum(vv)), decimals)
    if abs(diff) > (10 ** (-decimals)) / 2 and len(vv) > 0:
        k = int(fix_idx) if (fix_idx is not None and 0 <= int(fix_idx) < len(vv)) else (len(vv) - 1)
        vv[k] = max(0.0, float(vv[k]) + diff)
    vv = np.clip(vv, 0.0, np.inf)
    return [float(x) for x in vv]


def _manual_normalize_100(weights_pct: List[float], locks: List[bool], decimals: int = 2) -> List[float]:
    """
    NORMALIZE MANUALE:
    - locked invariati
    - unlocked riscalati per somma=100
    - se locked_sum>100: scala solo locked e azzera gli altri
    """
    n = len(weights_pct)
    if n == 0:
        return []

    v = np.array([max(0.0, float(x or 0.0)) for x in weights_pct], dtype=float)
    lk = np.array([bool(x) for x in locks], dtype=bool)

    locked_sum = float(np.sum(v[lk])) if np.any(lk) else 0.0
    if locked_sum > 100.0:
        out = np.zeros(n, dtype=float)
        if locked_sum > 0 and np.any(lk):
            out[lk] = v[lk] * (100.0 / locked_sum)
        fix = int(np.where(lk)[0][-1]) if np.any(lk) else None
        return _round_fix_total(out, total=100.0, decimals=decimals, fix_idx=fix)

    remaining = 100.0 - locked_sum
    out = v.copy()

    idx_un = np.where(~lk)[0].tolist()
    if len(idx_un) == 0:
        s = float(np.sum(out))
        if s > 0 and abs(s - 100.0) > 1e-9:
            out = out * (100.0 / s)
        return _round_fix_total(out, total=100.0, decimals=decimals, fix_idx=n - 1)

    unlocked_sum = float(np.sum(out[idx_un]))
    if unlocked_sum <= 0:
        per = remaining / len(idx_un) if remaining > 0 else 0.0
        for i in idx_un:
            out[i] = per
    else:
        scale = remaining / unlocked_sum if unlocked_sum > 0 else 0.0
        for i in idx_un:
            out[i] = out[i] * scale

    fix = idx_un[-1] if idx_un else (n - 1)
    return _round_fix_total(out, total=100.0, decimals=decimals, fix_idx=fix)


def _components_to_tickers_and_weights(comps: List[dict]) -> Tuple[List[str], pd.Series, List[str]]:
    warns: List[str] = []
    if not comps:
        return [], pd.Series(dtype=float), ["Nessuna componente."]

    raw_tickers = [str(c.get("ticker", "")).strip().upper() for c in comps]
    raw_w = [float(c.get("w", 0.0) or 0.0) for c in comps]

    pairs = [(t, w) for t, w in zip(raw_tickers, raw_w) if t]
    if len(pairs) < len(raw_tickers):
        warns.append("Alcune righe senza ticker: ignorate.")

    if not pairs:
        return [], pd.Series(dtype=float), ["Inserisci almeno 1 ticker valido."]

    order: List[str] = []
    acc: Dict[str, float] = {}
    for t, w in pairs:
        if t not in acc:
            acc[t] = 0.0
            order.append(t)
        acc[t] += max(0.0, float(w))

    if len(order) < len(pairs):
        warns.append("Tickers duplicati: ho sommato i pesi per ticker uguale.")

    w_pct = pd.Series([acc[t] for t in order], index=order, dtype=float)
    s = float(w_pct.sum())
    if s <= 0:
        ws = _equal_split(100.0, len(order), decimals=2)
        w_pct = pd.Series(ws, index=order, dtype=float)
        warns.append("Pesi a 0: fallback equal-weight.")

    # NON forzo 100 automaticamente: normalize solo con pulsante
    w_frac = (w_pct / 100.0).astype(float)
    return order, w_frac, warns


# =========================================================
# Editor componenti (NO bilanciamento automatico)
# =========================================================
def _init_components(pid: str, seed: Optional[List[str]]) -> List[dict]:
    key = f"{pid}__components"
    if key in st.session_state and isinstance(st.session_state[key], list) and len(st.session_state[key]) > 0:
        return st.session_state[key]

    if seed is None:
        tickers = ["AAPL", "MSFT", "GOOGL"]
    else:
        tickers = [t.strip().upper() for t in (seed or []) if str(t).strip()]
        if not tickers:
            tickers = [""]

    ws = _equal_split(100.0, len(tickers), decimals=2)
    comps = [{"rid": _new_rid(), "ticker": tickers[i], "w": float(ws[i]), "lock": False} for i in range(len(tickers))]

    st.session_state[key] = comps
    st.session_state.setdefault(f"{pid}__step", 0.01)  # default step fine
    return comps


def _render_components_editor(pid: str, seed: Optional[List[str]]) -> Tuple[List[str], pd.Series, List[str]]:
    comps_key = f"{pid}__components"
    comps = _init_components(pid, seed=seed)

    step = st.number_input(
        "Step peso (%)",
        min_value=0.01,
        max_value=10.0,
        value=float(st.session_state.get(f"{pid}__step", 0.01)),
        step=0.01,
        format="%.2f",
        key=f"{pid}__step",
    )

    c1, c2, c3, c4 = st.columns([1.0, 1.0, 1.1, 2.0])
    with c1:
        if st.button("➕ Aggiungi riga", type="primary", use_container_width=True, key=f"{pid}__add"):
            comps.append({"rid": _new_rid(), "ticker": "", "w": 0.0, "lock": False})
            st.session_state[comps_key] = comps
            st.rerun()

    with c2:
        if st.button("⚖️ Equal (rispetta Lock)", use_container_width=True, key=f"{pid}__equal"):
            weights = [float(x.get("w", 0.0) or 0.0) for x in comps]
            locks = [bool(x.get("lock", False)) for x in comps]

            locked_sum = float(np.sum([weights[i] for i in range(len(comps)) if locks[i]]))
            if locked_sum > 100.0:
                norm = _manual_normalize_100(weights, locks, decimals=2)
            else:
                remaining = 100.0 - locked_sum
                idx_un = [i for i in range(len(comps)) if not locks[i]]
                eq = _equal_split(remaining, len(idx_un), decimals=2) if idx_un else []
                for k, i_un in enumerate(idx_un):
                    weights[i_un] = eq[k]
                norm = _round_fix_total(
                    np.array(weights, dtype=float),
                    total=100.0,
                    decimals=2,
                    fix_idx=(idx_un[-1] if idx_un else (len(comps) - 1)),
                )

            for i in range(len(comps)):
                comps[i]["w"] = float(norm[i])
                rid = str(comps[i]["rid"])
                st.session_state[f"{pid}__w_{rid}"] = float(norm[i])

            st.session_state[comps_key] = comps

    with c3:
        if st.button("🧮 Normalize (100%)", use_container_width=True, key=f"{pid}__norm"):
            weights = [float(x.get("w", 0.0) or 0.0) for x in comps]
            locks = [bool(x.get("lock", False)) for x in comps]
            norm = _manual_normalize_100(weights, locks, decimals=2)

            for i in range(len(comps)):
                comps[i]["w"] = float(norm[i])
                rid = str(comps[i]["rid"])
                st.session_state[f"{pid}__w_{rid}"] = float(norm[i])

            st.session_state[comps_key] = comps

    with c4:
        st.caption("Nessun bilanciamento automatico: Lock agisce solo su Equal/Normalize.")

    st.markdown("---")

    h1, h2, h3, h4 = st.columns([3.2, 2.1, 0.9, 0.6], gap="small")
    with h1:
        st.caption("Ticker")
    with h2:
        st.caption("Weight %")
    with h3:
        st.caption("Lock")
    with h4:
        st.caption("")

    for c in comps:
        rid = str(c["rid"])
        k_t = f"{pid}__t_{rid}"
        k_w = f"{pid}__w_{rid}"
        k_l = f"{pid}__l_{rid}"

        st.session_state.setdefault(k_t, str(c.get("ticker", "") or ""))
        st.session_state.setdefault(k_w, float(c.get("w", 0.0) or 0.0))
        st.session_state.setdefault(k_l, bool(c.get("lock", False)))

        col_t, col_w, col_l, col_x = st.columns([3.2, 2.1, 0.9, 0.6], gap="small")

        with col_t:
            t_in = st.text_input("Ticker", key=k_t, label_visibility="collapsed", placeholder="ES. AAPL")
            c["ticker"] = str(t_in).strip().upper()

        with col_w:
            w_in = st.number_input(
                "Weight",
                min_value=0.0,
                max_value=100.0,
                value=float(st.session_state.get(k_w, float(c.get("w", 0.0)))),
                step=float(step),
                format="%.2f",
                key=k_w,
                label_visibility="collapsed",
            )
            c["w"] = float(w_in)

        with col_l:
            lk = st.checkbox("Lock", value=bool(st.session_state.get(k_l, False)), key=k_l, label_visibility="collapsed")
            c["lock"] = bool(lk)

        with col_x:
            if st.button("🗑️", use_container_width=True, key=f"{pid}__del_{rid}"):
                if len(comps) > 1:
                    comps2 = [x for x in comps if str(x["rid"]) != rid]
                    st.session_state[comps_key] = comps2
                    st.rerun()
                else:
                    comps[0]["ticker"] = ""
                    comps[0]["w"] = 0.0
                    comps[0]["lock"] = False
                    st.session_state[comps_key] = comps
                    st.rerun()

    st.session_state[comps_key] = comps

    sum_w = float(np.nansum([float(x.get("w", 0.0) or 0.0) for x in comps])) if comps else 0.0
    st.metric("WEIGHT SUM (EDITED)", f"{sum_w:.2f}%")
    if abs(sum_w - 100.0) > 1e-6:
        st.warning("La somma non è 100%. Premi **Normalize (100%)** se vuoi forzarla (rispettando i Lock).")

    tickers_out, w_frac, warns = _components_to_tickers_and_weights(comps)
    return tickers_out, w_frac, warns


# =========================================================
# Charts/KPI (overview multi)
# =========================================================
def _plot_compare_base100(equity_map: Dict[str, pd.Series], title: str) -> None:
    if not equity_map:
        st.info("Nessuna equity disponibile.")
        return
    fig, ax = plt.subplots(figsize=(10, 4))
    for name, eq in equity_map.items():
        s = pd.Series(eq).dropna()
        if len(s) < 2:
            continue
        base = float(s.iloc[0])
        if not np.isfinite(base) or base == 0:
            continue
        base100 = (s / base) * 100.0
        ax.plot(base100.index, base100.values, label=name)
    ax.set_title(title)
    ax.set_xlabel("Date")
    ax.set_ylabel("Index (Base=100)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


def _kpi_from_res(res: dict, rf_annual: float, rachev_alpha: float) -> Dict[str, float]:
    eq = res.get("equity")
    if eq is None or len(eq) < 2:
        return {}
    eq = pd.Series(eq).dropna()
    rets = eq.pct_change().dropna()
    if rets.empty:
        return {}
    return {
        "CAGR": float(cagr_from_returns(rets)),
        "Vol (ann.)": float(vol_ann(rets)),
        "Sharpe": float(sharpe_ratio(rets, rf_annual=rf_annual)),
        "Sortino": float(sortino_ratio(rets, rf_annual=rf_annual)),
        "Max Drawdown": float(drawdown_series(eq).min()),
        "CVaR 95% (daily)": float(es_cvar(rets, 0.95)),
        "Omega(τ=0)": float(omega_ratio(rets, 0.0)),
        f"Rachev (α={rachev_alpha:.2f})": float(rachev_ratio(rets, rachev_alpha)),
    }


# =========================================================
# Extra metrics (optimization)
# =========================================================
def _safe_prices(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    d.index = pd.to_datetime(d.index, errors="coerce")
    d = d.loc[d.index.notna()].sort_index()
    d.index = d.index.normalize()
    d = d[~d.index.duplicated(keep="last")]
    return d


def _rets_assets_from_prices(prices: pd.DataFrame) -> pd.DataFrame:
    d = _safe_prices(prices)
    rets = d.pct_change().replace([np.inf, -np.inf], np.nan).dropna(how="all")
    return rets


def _portfolio_returns_from_assets(rets_assets: pd.DataFrame, w: pd.Series) -> pd.Series:
    ww = w.reindex(rets_assets.columns).fillna(0.0).astype(float)
    s = float(ww.sum())
    if s != 0:
        ww = ww / s
    rp = (rets_assets.mul(ww, axis=1)).sum(axis=1)
    return rp.replace([np.inf, -np.inf], np.nan).dropna()


def _equity_from_rets(rp: pd.Series, initial: float) -> pd.Series:
    if rp is None or rp.empty:
        return pd.Series(dtype=float)
    eq = (1.0 + rp).cumprod() * float(initial)
    return eq


def _rf_daily_from_annual(rf_annual: float, periods: int = 252) -> float:
    try:
        return float((1.0 + float(rf_annual)) ** (1.0 / float(periods)) - 1.0)
    except Exception:
        return 0.0


def _ulcer_index(eq: pd.Series) -> float:
    # Ulcer index: sqrt(mean(dd^2)), dd in decimals
    if eq is None or len(eq) < 2:
        return np.nan
    e = pd.Series(eq).dropna()
    if e.empty:
        return np.nan
    peak = e.cummax()
    dd = (e / peak - 1.0).clip(upper=0.0)
    return float(np.sqrt(np.nanmean(np.square(dd.values))))


def _pain_index(eq: pd.Series) -> float:
    # Pain index: average drawdown depth (absolute), dd in decimals
    if eq is None or len(eq) < 2:
        return np.nan
    e = pd.Series(eq).dropna()
    if e.empty:
        return np.nan
    peak = e.cummax()
    dd = (e / peak - 1.0).clip(upper=0.0)
    return float(np.nanmean(np.abs(dd.values)))


def _calmar_ratio(cagr: float, max_dd: float) -> float:
    if not np.isfinite(cagr) or not np.isfinite(max_dd):
        return np.nan
    denom = abs(float(max_dd))
    return float(cagr / denom) if denom > 0 else np.nan


def _mar_ratio(cagr: float, rf_annual: float, max_dd: float) -> float:
    # MAR spesso usato come (CAGR - RF) / |MaxDD|
    if not np.isfinite(cagr) or not np.isfinite(max_dd):
        return np.nan
    denom = abs(float(max_dd))
    num = float(cagr) - float(rf_annual)
    return float(num / denom) if denom > 0 else np.nan


def _sterling_ratio(eq: pd.Series, cagr: float, rf_annual: float, threshold: float = 0.10) -> float:
    # Sterling: (CAGR-RF)/avg drawdown oltre soglia (approx)
    if eq is None or len(eq) < 2:
        return np.nan
    e = pd.Series(eq).dropna()
    if e.empty:
        return np.nan
    dd = (e / e.cummax() - 1.0).values
    depths = np.abs(dd[dd < -abs(float(threshold))])
    if depths.size == 0:
        depths = np.array([abs(float(np.nanmin(dd)))]) if np.isfinite(np.nanmin(dd)) else np.array([])
    if depths.size == 0:
        return np.nan
    avg_depth = float(np.nanmean(depths))
    num = float(cagr) - float(rf_annual)
    return float(num / avg_depth) if avg_depth > 0 else np.nan


def _kappa_3(rp: pd.Series, rf_annual: float, target: float = 0.0) -> float:
    # Kappa(3) approx su returns giornalieri
    if rp is None or rp.empty:
        return np.nan
    r = pd.Series(rp).dropna()
    if r.empty:
        return np.nan

    rf_d = _rf_daily_from_annual(rf_annual)
    ex = r - rf_d
    # LPM3 sul target (di ex returns)
    downside = np.maximum(0.0, float(target) - ex.values)
    lpm3 = float(np.nanmean(np.power(downside, 3)))
    if not np.isfinite(lpm3) or lpm3 <= 0:
        return np.nan

    # annualizza in modo semplice
    mu_ann = float(np.nanmean(ex.values) * 252.0)
    denom = float((lpm3 ** (1.0 / 3.0)) * (252.0 ** (1.0 / 3.0)))
    return float(mu_ann / denom) if denom > 0 else np.nan


def _burke_ratio(eq: pd.Series, rp: pd.Series, rf_annual: float) -> float:
    # Burke ratio approx: (CAGR-RF)/sqrt(sum(drawdown^2))
    if eq is None or len(eq) < 2 or rp is None or rp.empty:
        return np.nan
    e = pd.Series(eq).dropna()
    if e.empty:
        return np.nan
    dd = (e / e.cummax() - 1.0).clip(upper=0.0).values
    denom = float(np.sqrt(np.nansum(np.square(dd))))
    if denom <= 0 or not np.isfinite(denom):
        return np.nan
    cagr = float(cagr_from_returns(pd.Series(rp).dropna()))
    num = float(cagr) - float(rf_annual)
    return float(num / denom)


def _bench_metrics(port: pd.Series, bench: pd.Series) -> Dict[str, float]:
    # expects aligned daily returns
    p = pd.Series(port).dropna()
    b = pd.Series(bench).dropna()
    idx = p.index.intersection(b.index)
    if idx.empty:
        return {"Beta (vs bench)": np.nan, "Alpha annual (vs bench)": np.nan, "Information ratio": np.nan, "Tracking error (ann.)": np.nan}

    p = p.loc[idx].astype(float)
    b = b.loc[idx].astype(float)

    var_b = float(np.nanvar(b.values, ddof=1))
    cov_pb = float(np.nancov(p.values, b.values, ddof=1)[0, 1]) if len(idx) > 2 else np.nan
    beta = (cov_pb / var_b) if (np.isfinite(cov_pb) and np.isfinite(var_b) and var_b > 0) else np.nan

    mu_p = float(np.nanmean(p.values))
    mu_b = float(np.nanmean(b.values))
    alpha_ann = float((mu_p - (beta * mu_b if np.isfinite(beta) else 0.0)) * 252.0) if np.isfinite(mu_p) and np.isfinite(mu_b) else np.nan

    active = (p - b).values
    te_ann = float(np.nanstd(active, ddof=1) * np.sqrt(252.0)) if len(active) > 2 else np.nan
    ir = float((np.nanmean(active) * 252.0) / te_ann) if np.isfinite(te_ann) and te_ann > 0 else np.nan

    return {
        "Beta (vs bench)": float(beta) if np.isfinite(beta) else np.nan,
        "Alpha annual (vs bench)": float(alpha_ann) if np.isfinite(alpha_ann) else np.nan,
        "Information ratio": float(ir) if np.isfinite(ir) else np.nan,
        "Tracking error (ann.)": float(te_ann) if np.isfinite(te_ann) else np.nan,
    }


def _metrics_pack(eq: pd.Series, rp: pd.Series, rf_annual: float, rachev_alpha: float, bench_rp: Optional[pd.Series] = None) -> Dict[str, float]:
    if eq is None or len(eq) < 2 or rp is None or rp.empty:
        return {}

    eq = pd.Series(eq).dropna()
    rp = pd.Series(rp).replace([np.inf, -np.inf], np.nan).dropna()
    if eq.empty or rp.empty:
        return {}

    cagr = float(cagr_from_returns(rp))
    vol = float(vol_ann(rp))
    sh = float(sharpe_ratio(rp, rf_annual=rf_annual))
    so = float(sortino_ratio(rp, rf_annual=rf_annual))
    maxdd = float(drawdown_series(eq).min())
    cvar95 = float(es_cvar(rp, 0.95))
    cvar99 = float(es_cvar(rp, 0.99))
    omg = float(omega_ratio(rp, 0.0))
    rach = float(rachev_ratio(rp, rachev_alpha))

    ulcer = float(_ulcer_index(eq))
    pain = float(_pain_index(eq))
    calmar = float(_calmar_ratio(cagr, maxdd))
    mar = float(_mar_ratio(cagr, rf_annual, maxdd))
    sterling = float(_sterling_ratio(eq, cagr, rf_annual))
    kappa3 = float(_kappa_3(rp, rf_annual))
    burke = float(_burke_ratio(eq, rp, rf_annual))

    skew = float(rp.skew()) if len(rp) > 2 else np.nan
    exkurt = float(rp.kurt()) if len(rp) > 3 else np.nan  # pandas = excess kurtosis
    win = float((rp > 0).mean()) if len(rp) > 0 else np.nan

    out = {
        "CAGR": cagr,
        "Volatilità annuale": vol,
        "Sharpe ratio": sh,
        "Sortino ratio": so,
        "Max drawdown": maxdd,
        "Ulcer index": ulcer,
        "Burke ratio": burke,
        "Kappa ratio (3)": kappa3,
        "Sterling ratio": sterling,
        "Calmar ratio": calmar,
        "MAR": mar,
        "Pain ratio": ( (cagr - rf_annual) / pain if np.isfinite(pain) and pain > 0 else np.nan ),
        "Rachev": rach,
        "CVaR 95% (daily)": cvar95,
        "CVaR 99% (daily)": cvar99,
        "Omega (τ=0)": omg,
        "Skewness": skew,
        "Excess kurtosis": exkurt,
        "Win rate (daily)": win,
    }

    if bench_rp is not None and isinstance(bench_rp, pd.Series) and not bench_rp.empty:
        out.update(_bench_metrics(rp, bench_rp))

    return out


def _format_metrics_table(df: pd.DataFrame) -> pd.DataFrame:
    # Heuristic formatting for UI
    pct_like = {
        "CAGR",
        "Volatilità annuale",
        "Max drawdown",
        "Ulcer index",
        "CVaR 95% (daily)",
        "CVaR 99% (daily)",
        "Alpha annual (vs bench)",
        "Tracking error (ann.)",
        "Win rate (daily)",
    }
    out = df.copy()
    for c in out.columns:
        for idx in out.index:
            v = out.at[idx, c]
            if v is None or (isinstance(v, float) and not np.isfinite(v)):
                out.at[idx, c] = "–"
                continue

            if idx in pct_like:
                out.at[idx, c] = f"{float(v) * 100:.2f}%"
            else:
                out.at[idx, c] = f"{float(v):.4f}"
    return out


# =========================================================
# Main render
# =========================================================
def render_portfolio_page() -> None:
    st.header("Portafoglio — Multi")
    _inject_ui_css()

    g = st.session_state.get(SS_GLOBAL, {})
    g_start = g.get("start_date_val", date(2020, 1, 1))
    g_end = g.get("end_date_val", date.today())
    g_rf = float(g.get("rf_annual", 0.01))
    g_initial = float(g.get("initial_capital", 100_000.0))
    g_rachev = float(g.get("rachev_alpha", 0.05))
    g_n = int(g.get("n_ports", 2))

    with st.expander("Parametri portafoglio (Multi)", expanded=True):
        st.subheader("Core Parameters (global)")

        c0, c1, c2 = st.columns([0.8, 1, 1])
        with c0:
            n_ports = st.number_input("Numero portafogli", min_value=1, max_value=8, value=g_n, step=1, key="pf_n_ports")
        with c1:
            start_date = st.date_input("From", value=g_start, key="pf_from_global")
        with c2:
            end_date = st.date_input("To", value=g_end, key="pf_to_global")

        c3, c4, c5 = st.columns(3)
        with c3:
            rf_annual = st.number_input("Risk-free (annual)", value=g_rf, step=0.001, format="%.3f", key="pf_rf_global")
        with c4:
            initial_capital = st.number_input("Initial Capital", value=g_initial, step=1000.0, format="%.2f", key="pf_initial_global")
        with c5:
            rachev_alpha = st.slider("Rachev α", min_value=0.01, max_value=0.20, value=g_rachev, step=0.01, key="pf_rachev_alpha_global")

        st.markdown("---")
        st.subheader("Definizione portafogli (Ticker + Weight% + Lock)")

        if SS_SPECS not in st.session_state or not isinstance(st.session_state[SS_SPECS], list):
            st.session_state[SS_SPECS] = []
        specs_raw: List[dict] = st.session_state[SS_SPECS]

        while len(specs_raw) < int(n_ports):
            i = len(specs_raw) + 1
            specs_raw.append({
                "pid": f"P{i}",
                "name": f"PORT {i}",
                "tickers": [],
                "weights_frac": pd.Series(dtype=float),
            })
        if len(specs_raw) > int(n_ports):
            specs_raw = specs_raw[: int(n_ports)]
            st.session_state[SS_SPECS] = specs_raw

        normalize_on_run = st.checkbox(
            "Normalizza pesi in Run (consigliato)",
            value=bool(g.get("normalize_on_run", True)),
            key="pf_norm_run",
        )

        for i, sp in enumerate(specs_raw):
            pid = sp["pid"]
            with st.expander(f"{pid} — Config", expanded=(i == 0)):
                sp["name"] = st.text_input("Nome portafoglio", value=sp.get("name", pid), key=f"{pid}_name")
                seed = None if i == 0 else []
                tickers_out, w_frac, warns = _render_components_editor(pid, seed=seed)

                sp["tickers"] = tickers_out
                sp["weights_frac"] = w_frac

                if warns:
                    for m in warns:
                        st.warning(m)

        st.session_state[SS_SPECS] = specs_raw

        st.markdown("---")
        run_all = st.button("🚀 Run / Update (ALL)", type="primary", use_container_width=True, key="btn_run_update_all")

        st.session_state[SS_GLOBAL] = {
            "n_ports": int(n_ports),
            "start_date_val": start_date,
            "end_date_val": end_date,
            "start_date": str(start_date),
            "end_date": str(end_date),
            "rf_annual": float(rf_annual),
            "initial_capital": float(initial_capital),
            "rachev_alpha": float(rachev_alpha),
            "normalize_on_run": bool(normalize_on_run),
        }

    # ------------------------------------
    # RUN PIPELINE MULTI
    # ------------------------------------
    if run_all:
        specs_list: List[dict] = st.session_state.get(SS_SPECS, [])
        runs: Dict[str, dict] = {}
        errors: List[str] = []

        for sp in specs_list:
            pid = sp["pid"]
            name = str(sp.get("name", pid)).strip() or pid
            tickers = sp.get("tickers") or []

            if not tickers:
                errors.append(f"[{pid}] tickers vuoti.")
                continue

            w_ser = sp.get("weights_frac")
            weights = None
            if isinstance(w_ser, pd.Series) and len(w_ser) > 0:
                w = w_ser.reindex(tickers).fillna(0.0).astype(float)

                if bool(st.session_state[SS_GLOBAL].get("normalize_on_run", True)):
                    s = float(w.sum())
                    if s > 0:
                        w = w / s

                if float(w.sum()) > 0:
                    weights = w.values.tolist()

            try:
                res = compute_pipeline(
                    tickers=tickers,
                    weights=weights,
                    start=st.session_state[SS_GLOBAL]["start_date"],
                    end=st.session_state[SS_GLOBAL]["end_date"],
                    rf_annual=float(st.session_state[SS_GLOBAL]["rf_annual"]),
                    initial=float(st.session_state[SS_GLOBAL]["initial_capital"]),
                    rachev_alpha=float(st.session_state[SS_GLOBAL]["rachev_alpha"]),
                )
                res = dict(res) if isinstance(res, dict) else {"res": res}
                res["pid"] = pid
                res["name"] = name
                runs[pid] = res
            except Exception as e:
                errors.append(f"[{pid}] Run error: {e}")

        st.session_state[SS_RUNS] = runs

        if runs:
            if SS_ACTIVE not in st.session_state or st.session_state[SS_ACTIVE] not in runs:
                st.session_state[SS_ACTIVE] = list(runs.keys())[0]

        if errors:
            for m in errors:
                st.error(m)

    # ------------------------------------
    # ACTIVE RES
    # ------------------------------------
    runs = st.session_state.get(SS_RUNS, {})
    if not isinstance(runs, dict):
        runs = {}

    if runs:
        pid_list = list(runs.keys())
        active_pid = st.selectbox(
            "Portafoglio attivo (Testing/Ottimizzazione + compat res singolo)",
            options=pid_list,
            index=pid_list.index(st.session_state.get(SS_ACTIVE, pid_list[0])) if st.session_state.get(SS_ACTIVE) in pid_list else 0,
            key="pf_active_pid_picker",
            format_func=lambda pid: f"{pid} — {runs[pid].get('name', pid)}",
        )
        st.session_state[SS_ACTIVE] = active_pid
        active_res = runs[active_pid]
        st.session_state["res"] = active_res
    else:
        active_res = st.session_state.get("res")

    # ------------------------------------
    # TABS
    # ------------------------------------
    tab_overview, tab_analysis, tab_testing, tab_opt = st.tabs(["Overview", "Analisi", "Testing", "Ottimizzazione"])

    # =========================
    # OVERVIEW
    # =========================
    with tab_overview:
        st.subheader("Overview — Compare + KPI + Risk contributions")

        if not runs:
            st.info("Definisci almeno 1 portafoglio e premi **Run / Update (ALL)**.")
        else:
            equity_map: Dict[str, pd.Series] = {}
            for pid, r in runs.items():
                eq = r.get("equity")
                if eq is None:
                    continue
                eq = pd.Series(eq).dropna()
                if len(eq) < 2:
                    continue
                nm = str(r.get("name", pid))
                equity_map[nm] = eq

            _plot_compare_base100(equity_map, "Equity comparison (Base=100)")

            rows = []
            rf = float(st.session_state[SS_GLOBAL]["rf_annual"])
            ra = float(st.session_state[SS_GLOBAL]["rachev_alpha"])

            for pid, r in runs.items():
                nm = str(r.get("name", pid))
                m = _kpi_from_res(r, rf_annual=rf, rachev_alpha=ra)
                if m:
                    rows.append({"PID": pid, "Portfolio": nm, **m})

            if rows:
                df = pd.DataFrame(rows)
                df_fmt = df.copy()

                pct_cols = [c for c in df_fmt.columns if c in ("CAGR", "Vol (ann.)", "Max Drawdown", "CVaR 95% (daily)")]
                num_cols = [c for c in df_fmt.columns if c not in ("PID", "Portfolio") and c not in pct_cols]

                for c in pct_cols:
                    df_fmt[c] = df_fmt[c].map(lambda x: f"{x*100:.2f}%" if np.isfinite(x) else "–")
                for c in num_cols:
                    df_fmt[c] = df_fmt[c].map(lambda x: f"{x:.3f}" if np.isfinite(x) else "–")

                st.dataframe(df_fmt, use_container_width=True, hide_index=True)
            else:
                st.info("KPI non disponibili.")

            st.markdown("### Risk contributions (per portafoglio)")

            kcols = _grid_cols(len(runs))
            cols = st.columns(kcols, gap="medium")

            for i, (pid, r) in enumerate(runs.items()):
                with cols[i % kcols]:
                    st.markdown(f"**{r.get('name', pid)}**")

                    prices = r.get("prices")
                    w_raw = r.get("weights")
                    if prices is None or not isinstance(prices, pd.DataFrame) or prices.empty or w_raw is None:
                        st.caption("N/A (mancano prices/weights).")
                        continue

                    try:
                        if isinstance(w_raw, pd.Series):
                            w_ser = w_raw.copy()
                        else:
                            w_ser = pd.Series(list(w_raw), index=list(prices.columns), dtype=float)

                        w_ser = w_ser.groupby(level=0).sum().reindex(prices.columns).fillna(0.0).astype(float)
                        s = float(w_ser.sum())
                        if s > 0:
                            w_ser = w_ser / s

                        rets_assets = (
                            prices.sort_index()
                            .pct_change()
                            .replace([np.inf, -np.inf], np.nan)
                            .dropna(how="all")
                        )

                        eq = pd.Series(r.get("equity")).dropna()
                        rets_port = eq.pct_change().replace([np.inf, -np.inf], np.nan).dropna()

                        _, _, rc_df = risk_contributions(
                            rets_assets=rets_assets.loc[:, w_ser.index],
                            weights=w_ser.values,
                            rets_port=rets_port,
                            alpha=0.95,
                        )
                        st.dataframe(rc_df, use_container_width=True, height=260)
                    except Exception as e:
                        st.caption(f"Errore RC: {e}")

    # =========================
    # ANALISI
    # =========================
    with tab_analysis:
        st.subheader("Analisi — usa portafogli definiti in alto")
        if not runs and (active_res is None):
            st.info("Definisci portafogli e premi **Run / Update (ALL)**.")
        else:
            render_analisi_page(res=active_res, state=st.session_state)

    # =========================
    # TESTING
    # =========================
    with tab_testing:
        st.subheader("Testing — Stress / Crisi / Macro shock")

        if not runs and (active_res is None):
            st.info("Definisci portafogli e premi **Run / Update (ALL)**.")
        else:
            pid = str(st.session_state.get(SS_ACTIVE, "P1"))
            render_testing_page(res=active_res, state=st.session_state, pid=pid)

    # =========================
    # OTTIMIZZAZIONE
    # =========================
    with tab_opt:
        render_optimization_page(
            res=active_res,
            runs=runs,
            state=st.session_state,
            pid=str(st.session_state.get(SS_ACTIVE, "P1")),
            ss_global_key=SS_GLOBAL,
            ss_runs_key=SS_RUNS,
        )
