# ui/page_testing.py — Testing: What-if / Crisi / Macro (WB) / External shock / Weak spots
from __future__ import annotations

import re
import numbers
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from infra.marketdata.marketstack_client import load_ohlcv_from_marketstack


# =========================
# Helpers (dates / series)
# =========================
def _normalize_index_daily(s: pd.Series) -> pd.Series:
    if s is None or len(s) == 0:
        return s
    s = pd.Series(s).dropna()
    idx = pd.to_datetime(s.index, errors="coerce")
    s = s.loc[idx.notna()].copy()
    idx = pd.to_datetime(s.index, errors="coerce")

    # remove tz if present
    try:
        if getattr(idx, "tz", None) is not None:
            idx = idx.tz_convert(None)
    except Exception:
        pass
    try:
        idx = idx.tz_localize(None)
    except Exception:
        pass

    idx = pd.DatetimeIndex(idx).normalize()
    s.index = idx
    s = s[~s.index.duplicated(keep="last")]
    return s.sort_index()


def _clip_inclusive(s: pd.Series, start_dt: pd.Timestamp, end_dt: pd.Timestamp) -> pd.Series:
    if s is None or len(s) == 0:
        return s
    s = _normalize_index_daily(s)
    a = pd.to_datetime(start_dt).normalize()
    b = pd.to_datetime(end_dt).normalize() + pd.Timedelta(days=1) - pd.Timedelta(microseconds=1)
    return s[(s.index >= a) & (s.index <= b)]


def _returns_from_prices(px: pd.Series, start_dt, end_dt, reindex_business: bool = True) -> pd.Series:
    s = _clip_inclusive(px, start_dt, end_dt)
    if s is None or s.empty:
        return pd.Series(dtype=float)
    if reindex_business:
        full_idx = pd.date_range(s.index.min(), s.index.max(), freq="B")
        s = s.reindex(full_idx).ffill(limit=3)
    r = s.pct_change().replace([np.inf, -np.inf], np.nan).dropna()
    return r


def _fmt_num4(x):
    try:
        if isinstance(x, numbers.Number) and np.isfinite(float(x)):
            return f"{float(x):.4f}"
    except Exception:
        pass
    return x


def _safe_weights_from_res(res: dict, prices: pd.DataFrame) -> pd.Series:
    cols = list(prices.columns)
    w_raw = res.get("weights")

    if isinstance(w_raw, pd.Series):
        w = w_raw.reindex(cols).fillna(0.0).astype(float)
    elif isinstance(w_raw, (list, tuple, np.ndarray)) and len(w_raw) == len(cols):
        w = pd.Series(list(w_raw), index=cols, dtype=float).fillna(0.0).astype(float)
    else:
        # fallback equal
        n = len(cols)
        w = pd.Series((1.0 / n) if n else 0.0, index=cols, dtype=float)

    s = float(w.sum())
    if s > 0:
        w = w / s
    return w


def _build_equity_if_missing(prices: pd.DataFrame, w: pd.Series, initial: float) -> pd.Series:
    rets = prices.sort_index().pct_change().replace([np.inf, -np.inf], np.nan).dropna(how="all")
    rets = rets.reindex(columns=w.index).fillna(0.0)
    rp = (rets * w.values).sum(axis=1)
    eq = (1.0 + rp).cumprod() * float(initial)
    return _normalize_index_daily(eq)


# =========================
# World Bank API helpers
# =========================
WB_BASE = "https://api.worldbank.org/v2"


@st.cache_data(ttl=7 * 24 * 3600, show_spinner=False)
def _wb_get_page(url: str) -> Tuple[dict, list]:
    import requests

    r = requests.get(url, timeout=25)
    status = r.status_code
    if status >= 400:
        raise RuntimeError(f"WorldBank HTTP {status}: {r.text[:300]}")
    js = r.json()
    if not isinstance(js, list) or len(js) < 2:
        raise RuntimeError("WorldBank: unexpected JSON format")
    meta, data = js[0], js[1]
    if data is None:
        data = []
    return meta, data


def _wb_fetch_indicator_series(country: str, indicator: str, start_y: int, end_y: int) -> pd.Series:
    url = f"{WB_BASE}/country/{country}/indicator/{indicator}?date={start_y}:{end_y}&format=json&per_page=20000"
    meta, rows = _wb_get_page(url)
    out = {}
    for row in rows:
        try:
            y = int(row.get("date"))
            v = row.get("value")
            if v is None:
                continue
            out[y] = float(v)
        except Exception:
            continue
    s = pd.Series(out).sort_index().dropna()
    return s


@st.cache_data(ttl=7 * 24 * 3600, show_spinner=False)
def _wb_search_countries(term: str, max_results: int = 60) -> pd.DataFrame:
    """
    Cerca paesi WB per nome o codice.
    Ritorna colonne: id (ISO3/agg), iso2Code, name.
    """
    term = (term or "").strip().lower()
    if len(term) < 2:
        return pd.DataFrame(columns=["id", "iso2Code", "name"])

    # scan pages until enough matches
    per_page = 300
    page = 1
    matches = []
    pages_total = 999

    while page <= pages_total and len(matches) < max_results:
        url = f"{WB_BASE}/country?format=json&per_page={per_page}&page={page}"
        meta, rows = _wb_get_page(url)
        pages_total = int(meta.get("pages", 1))

        for r in rows:
            cid = str(r.get("id", "")).strip()
            iso2 = str(r.get("iso2Code", "")).strip()
            name = str(r.get("name", "")).strip()
            hay = f"{cid} {iso2} {name}".lower()
            if term in hay:
                matches.append((cid, iso2, name))
                if len(matches) >= max_results:
                    break

        page += 1

    df = pd.DataFrame(matches, columns=["id", "iso2Code", "name"]).drop_duplicates()
    return df


@st.cache_data(ttl=7 * 24 * 3600, show_spinner=False)
def _wb_search_indicators(term: str, max_results: int = 80) -> pd.DataFrame:
    """
    Cerca indicatori WB per keyword nel nome/ID.
    Fa paging incrementale e si ferma quando trova abbastanza match.
    Colonne: id, name, sourceNote.
    """
    term = (term or "").strip().lower()
    if len(term) < 2:
        return pd.DataFrame(columns=["id", "name", "sourceNote"])

    per_page = 1000
    page = 1
    matches = []
    pages_total = 999

    # NB: indicator list è grande; facciamo scan finché troviamo abbastanza match
    while page <= pages_total and len(matches) < max_results:
        url = f"{WB_BASE}/indicator?format=json&per_page={per_page}&page={page}"
        meta, rows = _wb_get_page(url)
        pages_total = int(meta.get("pages", 1))

        for r in rows:
            iid = str(r.get("id", "")).strip()
            nm = str(r.get("name", "")).strip()
            note = str(r.get("sourceNote", "") or "").strip()
            hay = f"{iid} {nm}".lower()
            if term in hay:
                matches.append((iid, nm, note))
                if len(matches) >= max_results:
                    break

        page += 1

        # hard safety stop (evita scans interminabili)
        if page > 25 and len(matches) == 0:
            break

    df = pd.DataFrame(matches, columns=["id", "name", "sourceNote"]).drop_duplicates()
    return df


def _transform_indicator(s: pd.Series, how: str) -> pd.Series:
    s = s.copy().sort_index().dropna()
    if how == "Level":
        return s
    if how == "Δ (YoY change)":
        return s.diff(1)
    if how == "Δ% (YoY % change)":
        return s.pct_change() * 100.0
    return s


def _estimate_beta_annual(ind: pd.Series, equity_daily: pd.Series, lag: int) -> Optional[dict]:
    """
    Regressione annuale: eq_ann_ret ~ beta * ind(+lag) + alpha
    ind: serie annuale (index=year int)
    equity_daily: serie daily
    lag: anni (0/1/2...)
    """
    if ind is None or ind.empty or equity_daily is None or equity_daily.empty:
        return None

    eq = _normalize_index_daily(equity_daily)
    eq_ann = eq.resample("A-DEC").last().pct_change().dropna()
    eq_ann.index = eq_ann.index.year

    x = ind.copy().sort_index().shift(lag)
    common_years = sorted(set(x.dropna().index).intersection(set(eq_ann.index)))
    if len(common_years) < 3:
        return None

    X = x.reindex(common_years).astype(float).values
    Y = eq_ann.reindex(common_years).astype(float).values
    m = np.isfinite(X) & np.isfinite(Y)
    X, Y = X[m], Y[m]
    if X.size < 3:
        return None

    Xmat = np.column_stack([X, np.ones_like(X)])
    coef, *_ = np.linalg.lstsq(Xmat, Y, rcond=None)
    beta = float(coef[0])
    alpha = float(coef[1])

    rho = float(np.corrcoef(X, Y)[0, 1]) if (np.std(X, ddof=1) > 0 and np.std(Y, ddof=1) > 0) else np.nan
    ss_tot = float(((Y - Y.mean()) ** 2).sum())
    yhat = beta * X + alpha
    ss_res = float(((Y - yhat) ** 2).sum())
    r2 = (1 - ss_res / ss_tot) if ss_tot > 0 else np.nan

    return {"n": int(X.size), "beta": beta, "alpha": alpha, "rho": rho, "r2": r2, "lag": int(lag)}


# =========================
# Crisi storiche (preset)
# =========================
@dataclass(frozen=True)
class Crisis:
    label: str
    start: date
    end: date


CRISES: List[Crisis] = [
    Crisis("COVID crash (Feb–Mar 2020)", date(2020, 2, 19), date(2020, 3, 23)),
    Crisis("Inflation / Rates bear (2022)", date(2022, 1, 3), date(2022, 10, 12)),
    Crisis("Euro debt stress (2011)", date(2011, 4, 1), date(2011, 10, 4)),
    Crisis("GFC (2008)", date(2007, 10, 9), date(2009, 3, 9)),
    Crisis("Dot-com (2000–2002)", date(2000, 3, 24), date(2002, 10, 9)),
]


def _crisis_metrics(eq: pd.Series, d0: date, d1: date) -> dict:
    s = _clip_inclusive(eq, pd.Timestamp(d0), pd.Timestamp(d1))
    if s is None or s.empty or len(s) < 2:
        return {}

    peak = s.cummax()
    dd = (s / peak) - 1.0
    mdd = float(dd.min())
    trough_dt = dd.idxmin()
    start_val = float(s.iloc[0])
    end_val = float(s.iloc[-1])
    ret = (end_val / start_val) - 1.0

    # recovery date: first date after trough that exceeds previous peak
    pre_trough_peak = float(peak.loc[:trough_dt].max())
    after = s.loc[trough_dt:]
    rec = after[after >= pre_trough_peak]
    rec_dt = rec.index.min() if len(rec) else None
    days_to_rec = int((rec_dt - trough_dt).days) if rec_dt is not None else None

    return {
        "Window return": ret,
        "Max drawdown": mdd,
        "Trough date": trough_dt.date(),
        "Recovery date": (rec_dt.date() if rec_dt is not None else None),
        "Days to recovery": days_to_rec,
    }


# =========================
# MAIN
# =========================
def render_testing_page(res: Optional[dict], state: dict, pid: str = "P1") -> None:
    st.subheader("Testing — Stress / Crisi / Macro shock")

    if not res or not isinstance(res, dict):
        st.info("Esegui prima **Run / Update (ALL)** nella sezione Portafoglio.")
        return

    prices = res.get("prices")
    if prices is None or not isinstance(prices, pd.DataFrame) or prices.empty:
        st.info("Questo portafoglio non ha `prices` validi in res. (Serve per Testing).")
        return

    g = state.get("portfolio_global_params", {}) if isinstance(state, dict) else {}
    initial_capital = float(g.get("initial_capital", 100_000.0))
    st.caption(f"Initial capital: **{initial_capital:,.2f}**")

    w = _safe_weights_from_res(res, prices)

    equity = res.get("equity")
    if equity is None or len(pd.Series(equity)) < 2:
        eq = _build_equity_if_missing(prices, w, initial_capital)
    else:
        eq = _normalize_index_daily(pd.Series(equity).dropna())

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "What-if (assets)",
        "Crisi storiche",
        "Macro (World Bank)",
        "External ticker shock",
        "Weak spots",
    ])

    # =========================
    # TAB 1 — WHAT-IF
    # =========================
    with tab1:
        st.markdown("### 🎛️ What-if (asset returns)")
        kpref = f"test__{pid}__whatif__"

        # init state
        for t in w.index:
            state.setdefault(f"{kpref}ret_{t}", 0.0)

        def _reset_all():
            for t in w.index:
                state[f"{kpref}ret_{t}"] = 0.0

        def _apply_all():
            shock = float(state.get(f"{kpref}shock_all", 0.0))
            for t in w.index:
                state[f"{kpref}ret_{t}"] = shock

        with st.expander("⚙️ Quick options", expanded=False):
            c1, c2 = st.columns(2)
            with c1:
                st.number_input("Global shock (%)", key=f"{kpref}shock_all", value=float(state.get(f"{kpref}shock_all", 0.0)),
                                step=0.5, format="%.2f")
                st.button("⚡ Apply shock to all", on_click=_apply_all, use_container_width=True, key=f"{kpref}apply_all_btn")
            with c2:
                st.button("🔄 Reset sliders", on_click=_reset_all, use_container_width=True, key=f"{kpref}reset_btn")

        ncol = max(1, min(3, len(w)))
        cols = st.columns(ncol)
        sel = {}

        for i, (t, ww) in enumerate(w.items()):
            with cols[i % ncol]:
                key = f"{kpref}ret_{t}"
                val = st.slider(
                    f"{t} – return (%)",
                    min_value=-100.0,
                    max_value=1000.0,
                    value=float(state.get(key, 0.0)),
                    step=0.1,
                    key=key,
                )
                sel[t] = float(val) / 100.0
                st.caption(f"Weight: {float(ww):.2%}")

        tot = float(sum(float(w[t]) * float(sel.get(t, 0.0)) for t in w.index))
        final_val = initial_capital * (1.0 + tot)
        st.info(f"**What-if (assets only):** {tot * 100:.2f}%  –  **Final value:** {final_val:,.2f}")

    # =========================
    # TAB 2 — CRISI STORICHE
    # =========================
    with tab2:
        st.markdown("### 🧨 Crisi storiche (su equity del portafoglio)")
        st.caption("Mostro metriche **solo se** la finestra è coperta dai dati caricati (start/end del tuo Run).")

        dmin = eq.index.min().date()
        dmax = eq.index.max().date()
        st.caption(f"Dati disponibili: **{dmin} → {dmax}**")

        labels = [c.label for c in CRISES]
        pick = st.selectbox("Seleziona crisi", labels, index=0, key=f"test__{pid}__crisis_pick")
        csel = next(c for c in CRISES if c.label == pick)

        if csel.end < dmin or csel.start > dmax:
            st.warning("Questa crisi è fuori dal range dei dati attuali. "
                       "Allarga il periodo globale (From/To) e rifai Run se vuoi analizzarla su questo portafoglio.")
        else:
            met = _crisis_metrics(eq, csel.start, csel.end)
            if not met:
                st.warning("Dati insufficienti nella finestra selezionata.")
            else:
                df = pd.DataFrame({"Value": met})
                df["Value"] = df["Value"].map(lambda x: f"{x*100:.2f}%" if isinstance(x, float) and ("drawdown" in str(df.index).lower()) else x)
                # formattiamo manualmente:
                shown = {
                    "Window return": f"{met['Window return']*100:.2f}%",
                    "Max drawdown": f"{met['Max drawdown']*100:.2f}%",
                    "Trough date": met["Trough date"],
                    "Recovery date": met["Recovery date"],
                    "Days to recovery": met["Days to recovery"],
                }
                st.dataframe(pd.DataFrame({"Value": shown}), use_container_width=True)

                s = _clip_inclusive(eq, pd.Timestamp(csel.start), pd.Timestamp(csel.end))
                fig, ax = plt.subplots(figsize=(10, 3.6))
                ax.plot(s.index, s.values)
                ax.set_title(f"Equity in crisis window: {csel.label}")
                ax.grid(True, alpha=0.3)
                fig.tight_layout()
                st.pyplot(fig)
                plt.close(fig)

    # =========================
    # TAB 3 — MACRO (WORLD BANK)
    # =========================
    with tab3:
        st.markdown("### 🌍 Macro stress test (World Bank)")
        st.caption("Qui puoi: (1) cercare **qualsiasi paese** WB, (2) cercare **qualsiasi indicatore** WB, "
                   "(3) applicare shock anche con **beta manuale** se la stima storica fallisce.")

        kpref = f"test__{pid}__wb__"

        # --- Country search
        c1, c2, c3 = st.columns([1.4, 1.8, 1.2])
        with c1:
            country_query = st.text_input("Cerca paese (nome/codice)", value=str(state.get(f"{kpref}country_q", "Italy")),
                                          key=f"{kpref}country_q")
        with c2:
            if st.button("🔎 Search countries", use_container_width=True, key=f"{kpref}country_btn"):
                try:
                    dfc = _wb_search_countries(country_query, max_results=80)
                    state[f"{kpref}country_hits"] = dfc.to_dict("records")
                except Exception as e:
                    state[f"{kpref}country_hits"] = []
                    st.error(f"Country search failed: {e}")

            hits = state.get(f"{kpref}country_hits", [])
            opts = hits if isinstance(hits, list) else []
            if opts:
                labels = [f"{o['name']} ({o['id']}/{o['iso2Code']})" for o in opts]
                pick_i = st.selectbox("Risultati", options=list(range(len(opts))), format_func=lambda i: labels[i],
                                      key=f"{kpref}country_pick")
                picked = opts[int(pick_i)]
                state[f"{kpref}country_code"] = picked.get("id") or picked.get("iso2Code")
            else:
                st.caption("Nessun risultato (o non hai cercato ancora).")
        with c3:
            manual_country = st.text_input("Country code (manual)", value=str(state.get(f"{kpref}country_code", "ITA")),
                                           help="ISO3/ISO2 o aggregati tipo EUU, WLD",
                                           key=f"{kpref}country_code_in")

        country_code = (manual_country or "ITA").strip().upper()

        # --- Indicator search
        i1, i2 = st.columns([1.6, 1.4])
        with i1:
            ind_query = st.text_input("Cerca indicatore (keyword)", value=str(state.get(f"{kpref}ind_q", "inflation")),
                                      key=f"{kpref}ind_q")
        with i2:
            if st.button("🔎 Search indicators", use_container_width=True, key=f"{kpref}ind_btn"):
                try:
                    dfi = _wb_search_indicators(ind_query, max_results=120)
                    state[f"{kpref}ind_hits"] = dfi.to_dict("records")
                except Exception as e:
                    state[f"{kpref}ind_hits"] = []
                    st.error(f"Indicator search failed: {e}")

        hits_i = state.get(f"{kpref}ind_hits", [])
        opts_i = hits_i if isinstance(hits_i, list) else []

        # picker indicator (or manual code)
        p1, p2 = st.columns([2.4, 1.0])
        with p1:
            if opts_i:
                labels_i = [f"{o['name']}  —  [{o['id']}]" for o in opts_i]
                pick_j = st.selectbox("Indicator results", options=list(range(len(opts_i))),
                                      format_func=lambda j: labels_i[j], key=f"{kpref}ind_pick")
                picked_i = opts_i[int(pick_j)]
                state[f"{kpref}ind_code"] = picked_i["id"]
            else:
                st.caption("Cerca un indicatore per vedere risultati (anche 20k+ indicatori WB).")
        with p2:
            ind_code_manual = st.text_input("Indicator code (manual)", value=str(state.get(f"{kpref}ind_code", "FP.CPI.TOTL.ZG")),
                                            key=f"{kpref}ind_code_in")

        ind_code = (ind_code_manual or "").strip()

        # --- Years / options
        cur_year = int(pd.Timestamp.today().year)
        y1, y2, y3 = st.columns([1, 1, 1.2])
        with y1:
            start_year = st.number_input("Start year", min_value=1960, max_value=cur_year, value=int(state.get(f"{kpref}y0", 2000)),
                                         step=1, key=f"{kpref}y0")
        with y2:
            end_year = st.number_input("End year", min_value=1960, max_value=cur_year, value=int(state.get(f"{kpref}y1", cur_year - 1)),
                                       step=1, key=f"{kpref}y1")
        with y3:
            transform = st.selectbox("Transformation", ["Level", "Δ (YoY change)", "Δ% (YoY % change)"],
                                     index=int(state.get(f"{kpref}tx_i", 0)), key=f"{kpref}tx")

        o1, o2, o3, o4 = st.columns([1.2, 1.2, 1.2, 1.4])
        with o1:
            lag_mode = st.selectbox("Lag", ["Auto (0 or 1)", "0", "1"], index=int(state.get(f"{kpref}lag_i", 0)), key=f"{kpref}lag")
        with o2:
            min_years = st.slider("Min years (estimation)", 3, 15, int(state.get(f"{kpref}miny", 5)), 1, key=f"{kpref}miny")
        with o3:
            fallback = st.checkbox("Fallback EUU→WLD if scarce", value=bool(state.get(f"{kpref}fb", True)), key=f"{kpref}fb")
        with o4:
            manual_beta_on = st.checkbox("Use manual beta override", value=bool(state.get(f"{kpref}mb", False)), key=f"{kpref}mb")

        # --- Fetch + estimate
        beta = np.nan
        rho = np.nan
        r2 = np.nan
        n_used = 0
        chosen_lag = 0
        src_country_used = country_code
        fetch_err = None

        ind_series = pd.Series(dtype=float)
        if ind_code:
            try:
                ind_series = _wb_fetch_indicator_series(country_code, ind_code, int(start_year), int(end_year))
                if ind_series.size < 3 and fallback:
                    for alt in ["EUU", "WLD"]:
                        tmp = _wb_fetch_indicator_series(alt, ind_code, 1960, cur_year - 1)
                        if tmp.size >= 3:
                            ind_series = tmp
                            src_country_used = alt
                            st.info(f"Using fallback series for indicator: **{alt}**")
                            break
            except Exception as e:
                fetch_err = str(e)

        if fetch_err:
            st.error(f"World Bank fetch error: {fetch_err}")

        ind_tx = _transform_indicator(ind_series, transform).dropna() if not ind_series.empty else pd.Series(dtype=float)

        # estimate beta unless manual
        if not manual_beta_on:
            if ind_tx.size >= 3:
                if lag_mode == "Auto (0 or 1)":
                    r0 = _estimate_beta_annual(ind_tx, eq, lag=0)
                    r1 = _estimate_beta_annual(ind_tx, eq, lag=1)
                    cand = [r for r in [r0, r1] if r is not None]
                    if cand:
                        best = sorted(
                            cand,
                            key=lambda d: (0 if np.isnan(d["rho"]) else abs(d["rho"]), d["n"]),
                            reverse=True
                        )[0]
                        beta = best["beta"]
                        rho = best["rho"]
                        r2 = best["r2"]
                        n_used = best["n"]
                        chosen_lag = best["lag"]
                else:
                    rr = _estimate_beta_annual(ind_tx, eq, lag=int(lag_mode))
                    if rr:
                        beta = rr["beta"]
                        rho = rr["rho"]
                        r2 = rr["r2"]
                        n_used = rr["n"]
                        chosen_lag = rr["lag"]

        # manual beta fallback
        if manual_beta_on or (not np.isfinite(beta)):
            beta = st.number_input(
                "Beta manual (portfolio return per 1 unit shock dell’indicatore scelto)",
                value=float(state.get(f"{kpref}beta_manual", 0.0)),
                step=0.1,
                format="%.4f",
                key=f"{kpref}beta_manual",
                help="Se la stima storica non riesce o vuoi forzare una view discrezionale."
            )

        if (not manual_beta_on) and np.isfinite(beta) and n_used < int(min_years):
            st.warning(f"Stima con pochi dati: years used = {n_used} < {min_years} (interpretare con cautela).")

        # --- Shock slider SEMPRE disponibile
        shock = st.slider(
            "Macro shock (unità coerente con Transformation scelta)",
            min_value=-10.0,
            max_value=10.0,
            value=float(state.get(f"{kpref}shock", 0.0)),
            step=0.25,
            key=f"{kpref}shock",
            help="Se Transformation=Level e l’indicatore è in %, lo shock è in **punti percentuali**."
        )

        impact = float(beta) * float(shock) if np.isfinite(beta) else np.nan

        met = {
            "Country/area used": src_country_used,
            "Indicator code": ind_code,
            "Transformation": transform,
            "Lag chosen": chosen_lag if (not manual_beta_on) else "manual",
            "Years used": n_used if (not manual_beta_on) else "manual",
            "Correlation ρ": rho if (not manual_beta_on) else np.nan,
            "R²": r2 if (not manual_beta_on) else np.nan,
            "Beta": beta,
            "Shock": shock,
            "Expected impact on annual portfolio return": impact,
        }
        dfm = pd.DataFrame({"Value": met})
        dfm["Value"] = dfm["Value"].map(_fmt_num4)
        st.dataframe(dfm, use_container_width=True)

        # --- quick plot (indicator)
        if not ind_tx.empty:
            fig, ax = plt.subplots(figsize=(10, 3.3))
            ax.plot(ind_tx.index, ind_tx.values)
            ax.set_title("World Bank indicator (annual, transformed)")
            ax.grid(True, alpha=0.3)
            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)
        else:
            st.info("Nessun dato indicatore disponibile (prova: cambia paese, estendi anni, o usa fallback EUU/WLD).")

    # =========================
    # TAB 4 — EXTERNAL TICKER SHOCK
    # =========================
    with tab4:
        st.markdown("### 🌐 External ticker shock")
        st.caption("Stima beta giornaliero Portfolio|Ticker su un periodo selezionato (Marketstack).")

        kpref = f"test__{pid}__ext__"
        dmin, dmax = eq.index.min(), eq.index.max()
        default_start = max(dmin, dmax - pd.Timedelta(days=365))

        c1, c2 = st.columns(2)
        with c1:
            p0 = st.date_input("Period from", value=default_start.date(), min_value=dmin.date(), max_value=dmax.date(), key=f"{kpref}p0")
        with c2:
            p1 = st.date_input("Period to", value=dmax.date(), min_value=dmin.date(), max_value=dmax.date(), key=f"{kpref}p1")

        p0, p1 = pd.to_datetime(p0), pd.to_datetime(p1)
        if p1 < p0:
            st.error("Invalid interval (end < start).")
        else:
            tick = st.text_input("Ticker (Marketstack)", value=str(state.get(f"{kpref}t", "SPY")), key=f"{kpref}t").strip().upper()

            @st.cache_data(ttl=6 * 3600, show_spinner=False)
            def _load_close(ticker: str, start: str, end: str) -> pd.Series:
                df = load_ohlcv_from_marketstack(ticker=ticker, start=start, end=end)
                s = pd.Series(df["close"]).dropna()
                return _normalize_index_daily(s)

            rho = np.nan
            beta = np.nan
            ext_impact = np.nan

            try:
                s = _load_close(tick, start=str((p0 - pd.Timedelta(days=10)).date()), end=str((p1 + pd.Timedelta(days=2)).date()))
                rt = _returns_from_prices(s, p0, p1, reindex_business=True)

                eq_clip = _clip_inclusive(eq, p0, p1)
                rp = eq_clip.pct_change().replace([np.inf, -np.inf], np.nan).dropna()

                pair = pd.concat([rp.rename("rp"), rt.rename("rt")], axis=1, join="inner").dropna()
                if len(pair) < 20:
                    st.info(f"Dati insufficienti per stimare (obs={len(pair)} < 20).")
                else:
                    rho = float(pair["rp"].corr(pair["rt"]))
                    sig_p = float(pair["rp"].std(ddof=1))
                    sig_t = float(pair["rt"].std(ddof=1))
                    beta = float(rho * (sig_p / sig_t)) if sig_t > 0 else np.nan

                    shock_pct = st.slider(f"Hypothetical return for {tick} (%)", -50.0, 50.0, 0.0, 0.5, key=f"{kpref}shock")
                    if np.isfinite(beta):
                        ext_impact = beta * (shock_pct / 100.0)

                    met = pd.DataFrame({
                        "Value": {
                            "Obs (daily)": int(len(pair)),
                            "Correlation ρ": rho,
                            "β Portfolio|Ticker": beta,
                            "Shock": shock_pct / 100.0,
                            "Expected impact on portfolio (daily)": ext_impact,
                        }
                    })
                    met["Value"] = met["Value"].map(_fmt_num4)
                    st.dataframe(met, use_container_width=True)
            except Exception as e:
                st.warning(f"Unable to compute: {e}")

    # =========================
    # TAB 5 — WEAK SPOTS
    # =========================
    with tab5:
        st.markdown("### 🧩 Weak spots (semplice ma utile)")
        st.caption("Heuristiche rapide su: volatilità, correlazione col portafoglio, contributo a varianza (approx).")

        # last 252 business days
        rets_assets = prices.sort_index().pct_change().replace([np.inf, -np.inf], np.nan).dropna(how="all")
        if rets_assets.empty:
            st.info("Rendimenti asset non disponibili.")
            return

        lookback = 252
        ra = rets_assets.tail(lookback).copy()
        rp = (ra.fillna(0.0) * w.values).sum(axis=1)

        vol_a = ra.std(ddof=1)
        corr_a = ra.corrwith(rp)
        # contribution to variance approx: w_i^2 * var_i + 2*w_i*sum_j w_j*cov_ij (qui usiamo approx w_i * cov(i,p))
        cov_ip = ra.covwith(rp) if hasattr(ra, "covwith") else pd.Series({c: float(np.cov(ra[c].dropna(), rp.loc[ra[c].dropna().index].dropna())[0, 1]) for c in ra.columns})
        contr_var_approx = w * cov_ip

        df = pd.DataFrame({
            "Weight": w,
            "Vol (daily)": vol_a,
            "Corr vs Port": corr_a,
            "Cov(i,Port)": cov_ip,
            "Var contrib approx": contr_var_approx,
        }).replace([np.inf, -np.inf], np.nan).dropna(how="all")

        df = df.sort_values("Var contrib approx", ascending=False)
        df_disp = df.copy()
        df_disp["Weight"] = df_disp["Weight"].map(lambda x: f"{x:.2%}" if np.isfinite(x) else "–")
        for c in ["Vol (daily)", "Corr vs Port", "Cov(i,Port)", "Var contrib approx"]:
            df_disp[c] = df_disp[c].map(_fmt_num4)

        st.dataframe(df_disp, use_container_width=True, height=420)
        st.caption("Suggerimento: gli asset in cima sono spesso i primi candidati per cap/hedge/lock.")
