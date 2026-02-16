# ui/components/multi_portfolio_builder.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st

from core.portfolio_analisi import normalize_series_daily, normalize_df_daily

# OPTIONAL: Marketstack fetch for benchmark/compare not in res["prices"]
try:
    from infra.marketdata.marketstack_client import load_ohlcv_from_marketstack
except Exception:
    load_ohlcv_from_marketstack = None


# -----------------------------
# Models
# -----------------------------
@dataclass
class PortInstance:
    pid: str
    name: str
    equity: pd.Series
    prices: pd.DataFrame
    weights: pd.Series
    holdings_meta: Optional[pd.DataFrame] = None


@dataclass
class MultiPortfolioContext:
    page_prefix: str
    ports: List[PortInstance]
    bench: str
    compare: List[str]
    view_mode: str
    start_dt: Optional[pd.Timestamp]
    end_dt: Optional[pd.Timestamp]
    fetch_errors: Dict[str, str]


# -----------------------------
# Keys (namespaced per page)
# -----------------------------
def _k(page_prefix: str, name: str) -> str:
    return f"{page_prefix}_{name}"


# -----------------------------
# Helpers
# -----------------------------
def _discover_saved_runs(state: Any) -> Dict[str, dict]:
    if not isinstance(state, dict):
        return {}

    candidates = ["portfolio_runs", "runs", "results", "res_list", "res_multi", "portfolios", "multi_res"]
    for k in candidates:
        v = state.get(k, None)
        if isinstance(v, dict):
            out = {}
            for name, rr in v.items():
                if isinstance(rr, dict) and rr.get("equity") is not None:
                    out[str(name)] = rr
            if out:
                return out
        if isinstance(v, list):
            out = {}
            for i, rr in enumerate(v):
                if isinstance(rr, dict) and rr.get("equity") is not None:
                    nm = rr.get("name", None)
                    out[str(nm) if nm else f"Run {i+1}"] = rr
            if out:
                return out
    return {}


def _normalize_weights(weights: Optional[pd.Series]) -> pd.Series:
    if not isinstance(weights, pd.Series) or weights.empty:
        return pd.Series(dtype=float)
    w = pd.to_numeric(weights, errors="coerce").dropna().astype(float)
    w.index = w.index.astype(str).str.upper().str.strip()
    w = w[w > 0]
    if w.empty:
        return pd.Series(dtype=float)
    s = float(w.sum())
    if not np.isfinite(s) or s <= 0:
        return pd.Series(dtype=float)
    return (w / s).sort_values(ascending=False)


def _weights_from_editor(
    available_tickers: List[str],
    default_weights: Optional[pd.Series],
    editor_key: str,
) -> pd.Series:
    if default_weights is None or not isinstance(default_weights, pd.Series) or default_weights.empty:
        df0 = pd.DataFrame({"Ticker": [], "Weight%": []})
    else:
        w = pd.to_numeric(default_weights, errors="coerce").dropna()
        w.index = w.index.astype(str).str.upper().str.strip()
        w = w[w > 0]
        df0 = pd.DataFrame({"Ticker": w.index.tolist(), "Weight%": (w.values * 100.0)})

    edited = st.data_editor(
        df0,
        use_container_width=True,
        hide_index=True,
        num_rows="dynamic",
        key=editor_key,
        column_config={
            "Ticker": st.column_config.SelectboxColumn("Ticker", options=sorted(set(available_tickers)), required=True),
            "Weight%": st.column_config.NumberColumn("Weight%", min_value=0.0, max_value=100.0, step=0.25),
        },
    )

    if edited is None or edited.empty:
        return pd.Series(dtype=float)

    edited = edited.copy()
    edited["Ticker"] = edited["Ticker"].astype(str).str.upper().str.strip()
    edited["Weight%"] = pd.to_numeric(edited["Weight%"], errors="coerce").fillna(0.0)
    edited = edited[edited["Ticker"].astype(str).str.len() > 0]
    edited = edited.groupby("Ticker", as_index=False)["Weight%"].sum()
    edited = edited[edited["Weight%"] > 0]

    if edited.empty:
        return pd.Series(dtype=float)

    tot = float(edited["Weight%"].sum())
    st.caption(f"Totale pesi inseriti: {tot:.2f}% (normalizzo automaticamente a 100%)")

    w = pd.Series(edited["Weight%"].values, index=edited["Ticker"].values).astype(float)
    if tot > 0:
        w = (w / tot)
    return w.sort_values(ascending=False)


def _equity_from_prices_weights(prices: pd.DataFrame, weights: pd.Series) -> pd.Series:
    if prices is None or prices.empty or weights is None or weights.empty:
        return pd.Series(dtype=float)

    px = prices.copy()
    px.columns = [str(c).upper() for c in px.columns]
    w = weights.copy()
    w.index = w.index.astype(str).str.upper()

    tick = [t for t in w.index if t in px.columns]
    if len(tick) < 1:
        return pd.Series(dtype=float)

    px = px[tick].sort_index().ffill().dropna(how="all")
    rets = px.pct_change().fillna(0.0)
    ww = w.loc[tick].astype(float)
    s = float(ww.sum())
    if not np.isfinite(s) or s <= 0:
        return pd.Series(dtype=float)
    ww = ww / s

    port_rets = rets.dot(ww.values)
    eq = (1.0 + port_rets).cumprod()
    eq.name = "CUSTOM"
    return normalize_series_daily(eq).dropna()


@st.cache_data(ttl=1800, show_spinner=False)
def _fetch_close_marketstack(symbol: str, start: str, end: str) -> pd.Series:
    if load_ohlcv_from_marketstack is None:
        return pd.Series(dtype=float)
    try:
        df = load_ohlcv_from_marketstack(ticker=str(symbol).upper(), start=start, end=end)
        if df is None or df.empty or "close" not in df.columns:
            return pd.Series(dtype=float)
        s = df["close"].astype(float).dropna()
        s = normalize_series_daily(s)
        s.name = str(symbol).upper()
        return s
    except Exception:
        return pd.Series(dtype=float)


def build_bench_compare(
    prices_internal: pd.DataFrame,
    bench: str,
    compare: List[str],
    start: str,
    end: str,
) -> Tuple[Optional[pd.Series], Optional[pd.DataFrame], Dict[str, str]]:
    errors: Dict[str, str] = {}

    bench_u = str(bench).upper()
    compare_u = [str(x).upper() for x in (compare or []) if str(x).strip()]
    compare_u = list(dict.fromkeys(compare_u))

    bench_s: Optional[pd.Series] = None
    if isinstance(prices_internal, pd.DataFrame) and not prices_internal.empty and bench_u in prices_internal.columns:
        bench_s = normalize_series_daily(prices_internal[bench_u]).dropna()
    else:
        s = _fetch_close_marketstack(bench_u, start, end)
        if not s.empty:
            bench_s = s
        else:
            errors[bench_u] = "Benchmark non presente nei prezzi interni e fetch esterno non disponibile/failed."

    comp_cols: List[pd.Series] = []
    for sym in compare_u:
        if isinstance(prices_internal, pd.DataFrame) and not prices_internal.empty and sym in prices_internal.columns:
            s = normalize_series_daily(prices_internal[sym]).dropna()
            if not s.empty:
                comp_cols.append(s.rename(sym))
        else:
            s = _fetch_close_marketstack(sym, start, end)
            if not s.empty:
                comp_cols.append(s.rename(sym))
            else:
                errors[sym] = "Compare non presente nei prezzi interni e fetch esterno non disponibile/failed."

    comp_df: Optional[pd.DataFrame] = None
    if comp_cols:
        comp_df = pd.concat(comp_cols, axis=1).sort_index().dropna(how="all")

    return bench_s, comp_df, errors


# -----------------------------
# MAIN: render bar + build N portfolios
# -----------------------------
def render_multi_portfolio_bar_and_build(
    *,
    page_prefix: str,
    res: dict,
    state: Any,
    title: str,
    default_n: int = 2,
    max_n: int = 6,
    show_bench_compare: bool = True,
) -> MultiPortfolioContext:
    """
    - Renderizza la barra multi-portfolio (con componenti/parametri per ogni portafoglio)
    - Costruisce e ritorna N PortInstance (Current / Saved / Custom weights)
    - Chiavi sempre namespaced con page_prefix -> zero interferenza tra pagine
    """
    eq_current = normalize_series_daily(res["equity"]).dropna()
    prices_current = normalize_df_daily(res.get("prices")) if isinstance(res.get("prices"), pd.DataFrame) else pd.DataFrame()
    weights_current = _normalize_weights(res.get("weights")) if isinstance(res.get("weights"), pd.Series) else pd.Series(dtype=float)
    meta_current = res.get("holdings_meta") if isinstance(res.get("holdings_meta"), pd.DataFrame) else None

    start_dt = eq_current.index.min()
    end_dt = eq_current.index.max()

    saved_runs = _discover_saved_runs(state if isinstance(state, dict) else {})
    run_options = ["Current res"] + [f"Saved run: {k}" for k in saved_runs.keys()] + ["Custom weights (build here)"]

    internal_opts = list(prices_current.columns) if not prices_current.empty else []
    all_tickers = sorted(set([str(x).upper() for x in internal_opts] + [
        "AAPL","MSFT","NVDA","GLD","TLT","AMZN","GOOGL","META","SPY","QQQ","VTI","ACWI","AGG"
    ]))

    st.markdown(f"### {title}")

    c1, c2, c3, c4 = st.columns([0.85, 1.35, 1.35, 1.05], gap="medium")
    with c1:
        n_ports = st.number_input(
            "Numero portafogli",
            min_value=1, max_value=max_n,
            value=int(st.session_state.get(_k(page_prefix, "n_ports"), default_n)),
            step=1,
            key=_k(page_prefix, "n_ports"),
        )
        n_ports = int(n_ports)

    if show_bench_compare:
        with c2:
            bench_opts = ["SPY", "QQQ", "ACWI", "VTI", "AGG"] + internal_opts
            bench = st.selectbox("Benchmark", options=bench_opts, index=0, key=_k(page_prefix, "bench"))
        with c3:
            compare = st.multiselect(
                "Compare tickers",
                options=sorted(set(all_tickers)),
                default=st.session_state.get(_k(page_prefix, "compare_default"), []),
                key=_k(page_prefix, "compare"),
            )
            st.session_state[_k(page_prefix, "compare_default")] = compare
        with c4:
            view_mode = st.radio("View", ["Base=100", "$10,000"], horizontal=True, index=0, key=_k(page_prefix, "view_mode"))
    else:
        bench, compare, view_mode = "SPY", [], "Base=100"
        with c2:
            st.caption("Benchmark/Compare disabilitati su questa pagina.")
        with c3:
            st.caption("")
        with c4:
            view_mode = st.radio("View", ["Base=100", "$10,000"], horizontal=True, index=0, key=_k(page_prefix, "view_mode"))

    specs_key = _k(page_prefix, "port_specs")
    if specs_key not in st.session_state or not isinstance(st.session_state[specs_key], list):
        st.session_state[specs_key] = []

    specs: List[dict] = st.session_state[specs_key]

    while len(specs) < n_ports:
        i = len(specs) + 1
        specs.append({
            "pid": f"P{i}",
            "name": f"PORT {i}",
            "source": "Current res",
            "run_name": "",
        })
    if len(specs) > n_ports:
        specs = specs[:n_ports]
        st.session_state[specs_key] = specs

    st.markdown("---")
    st.caption("Configura ciascun portafoglio: sorgente e/o componenti (Ticker + pesi).")

    for i, s in enumerate(specs):
        pid = s["pid"]
        with st.expander(f"{s.get('name','PORT')} • {pid}", expanded=(i == 0)):
            a, b, c = st.columns([1.2, 1.3, 1.5], gap="medium")
            with a:
                s["name"] = st.text_input("Nome portafoglio", value=s.get("name", f"PORT {i+1}"), key=_k(page_prefix, f"name_{pid}"))
            with b:
                s["source"] = st.selectbox(
                    "Source",
                    options=run_options,
                    index=run_options.index(s.get("source","Current res")) if s.get("source") in run_options else 0,
                    key=_k(page_prefix, f"source_{pid}"),
                )
            with c:
                if isinstance(s["source"], str) and s["source"].startswith("Saved run:"):
                    s["run_name"] = str(s["source"]).replace("Saved run:", "").strip()
                    st.caption(f"Run salvato: {s['run_name']}")
                elif s["source"] == "Current res":
                    s["run_name"] = ""
                    st.caption("Uso res corrente.")
                else:
                    s["run_name"] = ""
                    st.caption("Costruisci qui i componenti (deterministico).")

            if s["source"] == "Custom weights (build here)":
                st.markdown("**Componenti (Ticker + Weight%)**")
                default_w = weights_current if pid == "P1" else pd.Series(dtype=float)
                w = _weights_from_editor(
                    available_tickers=all_tickers,
                    default_weights=default_w,
                    editor_key=_k(page_prefix, f"editor_weights_{pid}"),
                )
                st.session_state[_k(page_prefix, f"weights_{pid}")] = w
            else:
                if _k(page_prefix, f"weights_{pid}") not in st.session_state:
                    st.session_state[_k(page_prefix, f"weights_{pid}")] = pd.Series(dtype=float)

    st.session_state[specs_key] = specs

    # Build bench/compare series (optional)
    fetch_errors: Dict[str, str] = {}
    if show_bench_compare and prices_current is not None and not prices_current.empty:
        _, _, fetch_errors = build_bench_compare(
            prices_internal=prices_current,
            bench=bench,
            compare=compare,
            start=str(start_dt.date()),
            end=str(end_dt.date()),
        )

    # Build ports
    ports: List[PortInstance] = []
    for s in specs:
        pid = s["pid"]
        name = str(s.get("name", pid)).strip() or pid
        source = s.get("source", "Current res")

        if source == "Current res":
            ports.append(PortInstance(
                pid=pid, name=name,
                equity=eq_current.copy().rename(name),
                prices=prices_current,
                weights=weights_current,
                holdings_meta=meta_current,
            ))
            continue

        if isinstance(source, str) and source.startswith("Saved run:"):
            run_name = s.get("run_name", "").strip()
            rr = saved_runs.get(run_name, None)
            if isinstance(rr, dict) and rr.get("equity") is not None:
                eq = normalize_series_daily(rr["equity"]).dropna().rename(name)
                pr = normalize_df_daily(rr.get("prices")) if isinstance(rr.get("prices"), pd.DataFrame) else prices_current
                ww = _normalize_weights(rr.get("weights")) if isinstance(rr.get("weights"), pd.Series) else pd.Series(dtype=float)
                mm = rr.get("holdings_meta") if isinstance(rr.get("holdings_meta"), pd.DataFrame) else meta_current
                ports.append(PortInstance(pid=pid, name=name, equity=eq, prices=pr, weights=ww, holdings_meta=mm))
            else:
                st.error(f"[{page_prefix}:{pid}] Saved run '{run_name}' non valido: equity mancante.")
            continue

        # Custom weights
        w = st.session_state.get(_k(page_prefix, f"weights_{pid}"), pd.Series(dtype=float))
        w = _normalize_weights(w)
        eq = _equity_from_prices_weights(prices_current, w).rename(name)
        if eq.empty:
            st.error(f"[{page_prefix}:{pid}] Custom weights: impossibile costruire equity (tickers non presenti in prices).")
            continue
        ports.append(PortInstance(
            pid=pid, name=name,
            equity=eq, prices=prices_current, weights=w,
            holdings_meta=meta_current,
        ))

    return MultiPortfolioContext(
        page_prefix=page_prefix,
        ports=ports,
        bench=bench,
        compare=compare,
        view_mode=view_mode,
        start_dt=start_dt,
        end_dt=end_dt,
        fetch_errors=fetch_errors,
    )


def grid_cols(n: int) -> int:
    return 1 if n <= 1 else (2 if n <= 4 else 3)
