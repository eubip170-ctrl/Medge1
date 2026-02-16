# core/portfolio_pipeline.py
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

from .portfolio_data import DataLoader, clean_prices, pct_returns
from .portfolio_metrics import (
    drawdown_series,
    cagr_from_returns,
    vol_ann,
    ulcer_index_from_equity,
    burke_ratio,
    sterling_ratio,
    kappa_ratio,
    sharpe_ratio,
    sortino_ratio,
    rachev_ratio,
    es_cvar,
    omega_ratio,
    pain_ratio,
)


# -------------------------
# Normalizzazione input
# -------------------------
def _normalize_tickers_and_weights(
    tickers: Any,
    weights: Any = None,
) -> Tuple[List[str], Optional[np.ndarray]]:
    """
    Ritorna (tickers_list, weights_array_or_None)

    Supporta:
      - tickers str: "AAPL,MSFT,GOOGL"
      - tickers list[str]
      - tickers list[dict] (es. [{"Ticker":"AAPL","Weight %":33.3,"Lock":False}, ...])
      - tickers DataFrame config (colonne: Ticker / Weight % / Lock ...)
    """
    # 1) tickers come stringa
    if isinstance(tickers, str):
        tlist = [t.strip().upper() for t in tickers.split(",") if t.strip()]
        return tlist, None if weights is None else _normalize_weights(weights, tlist)

    # 2) tickers come DataFrame (tabella config)
    if isinstance(tickers, pd.DataFrame):
        df = tickers.copy()
        cols = {str(c).strip().lower(): c for c in df.columns}
        col_t = cols.get("ticker") or cols.get("tickers")
        col_w = cols.get("weight %") or cols.get("weight") or cols.get("w")
        tlist: List[str] = []
        wlist: List[float] = []

        for _, row in df.iterrows():
            t = str(row[col_t]).strip().upper() if col_t is not None else ""
            if not t:
                continue
            tlist.append(t)
            if col_w is not None:
                try:
                    wlist.append(float(row[col_w]))
                except Exception:
                    wlist.append(np.nan)

        if not tlist:
            return [], None

        if col_w is not None and any(np.isfinite(w) for w in wlist):
            w_arr = np.array([0.0 if not np.isfinite(x) else float(x) for x in wlist], dtype=float)
            # se sembrano percentuali (>1) converto in frazioni
            if np.nanmax(w_arr) > 1.0 + 1e-9:
                w_arr = w_arr / 100.0
            return tlist, w_arr
        return tlist, None if weights is None else _normalize_weights(weights, tlist)

    # 3) tickers come list/tuple/set
    if isinstance(tickers, (list, tuple, set)):
        if len(tickers) == 0:
            return [], None

        # 3a) list di dict
        if all(isinstance(x, dict) for x in tickers):
            tlist: List[str] = []
            wlist: List[float] = []
            for x in tickers:
                t = str(x.get("ticker") or x.get("Ticker") or "").strip().upper()
                if not t:
                    continue
                tlist.append(t)
                w = x.get("weight")
                if w is None:
                    w = x.get("Weight %")
                try:
                    wlist.append(float(w) if w is not None else np.nan)
                except Exception:
                    wlist.append(np.nan)

            if not tlist:
                return [], None

            if any(np.isfinite(w) for w in wlist):
                w_arr = np.array([0.0 if not np.isfinite(x) else float(x) for x in wlist], dtype=float)
                if np.nanmax(w_arr) > 1.0 + 1e-9:
                    w_arr = w_arr / 100.0
                return tlist, w_arr

            return tlist, None if weights is None else _normalize_weights(weights, tlist)

        # 3b) list di stringhe (o convertibili)
        tlist = [str(x).strip().upper() for x in tickers if str(x).strip()]
        return tlist, None if weights is None else _normalize_weights(weights, tlist)

    # fallback
    return [], None


def _normalize_weights(weights: Any, tickers: List[str]) -> Optional[np.ndarray]:
    """
    Normalizza weights se passati separatamente:
      - list/tuple/np.array con stessa lunghezza di tickers
      - dict/Series mapping ticker->peso
    Ritorna array float oppure None se non interpretabile.
    """
    if weights is None:
        return None

    # Series
    if isinstance(weights, pd.Series):
        w = weights.reindex(tickers).astype(float)
        return w.values

    # dict mapping
    if isinstance(weights, dict):
        arr = np.array([float(weights.get(t, 0.0)) for t in tickers], dtype=float)
        # se percentuali > 1, converto
        if np.nanmax(arr) > 1.0 + 1e-9:
            arr = arr / 100.0
        return arr

    # list/tuple/array
    if isinstance(weights, (list, tuple, np.ndarray)):
        if len(weights) != len(tickers):
            return None
        arr = np.array(weights, dtype=float)
        if np.nanmax(arr) > 1.0 + 1e-9:
            arr = arr / 100.0
        return arr

    return None


def _make_weights(tickers: List[str], w_in: Optional[np.ndarray]) -> np.ndarray:
    n = len(tickers)
    if n == 0:
        return np.array([], dtype=float)

    if w_in is None or len(w_in) != n:
        return np.repeat(1.0 / n, n)

    w = np.array(w_in, dtype=float)
    w[~np.isfinite(w)] = 0.0
    s = float(w.sum())
    if s <= 0:
        return np.repeat(1.0 / n, n)
    return w / s


# -------------------------
# Fetch prezzi robusto (1 ticker per volta)
# -------------------------
def _fetch_prices_with_missing(
    tickers: List[str],
    start: str = "2020-01-01",
    end: Optional[str] = None,
    interval: str = "1d",
) -> Tuple[pd.DataFrame, List[str], List[str]]:
    """
    Scarica i prezzi Close per ogni ticker (una richiesta per ticker).
    Ritorna: (prices_df, valid_tickers, missing)

    - Se un ticker fallisce, viene messo in missing e si continua.
    - prices_df: colonne = valid_tickers
    """
    if not tickers:
        return pd.DataFrame(), [], []

    close_cols: Dict[str, pd.Series] = {}
    missing: List[str] = []

    for t in tickers:
        try:
            dl = DataLoader([t], start, end, interval)  # compat: DataLoader(tickers,start,end,interval)
            px = dl.fetch()  # dovrebbe tornare DataFrame con colonna t
            if px is None or getattr(px, "empty", True):
                missing.append(t)
                continue

            # se per qualche motivo non ha la colonna attesa, prendi la prima
            if t in px.columns:
                s = px[t]
            else:
                s = px.iloc[:, 0]
            close_cols[t] = pd.to_numeric(s, errors="coerce")
        except Exception:
            missing.append(t)

    if not close_cols:
        return pd.DataFrame(), [], tickers

    prices = pd.DataFrame(close_cols)
    prices = clean_prices(prices)
    valid = list(prices.columns)

    # se dopo clean resta vuoto:
    if prices.empty or len(valid) == 0:
        return pd.DataFrame(), [], tickers

    return prices, valid, missing


# -------------------------
# Pipeline principale
# -------------------------
def compute_pipeline(
    tickers: Any,
    weights: Any = None,
    start: str = "2020-01-01",
    end: Optional[str] = None,
    rf_annual: float = 0.00,
    initial: float = 100_000.0,
    rachev_alpha: float = 0.05,
) -> Dict[str, Any]:
    """
    Pipeline principale:
      - normalizza tickers/pesi
      - scarica i prezzi (no yfinance)
      - calcola returns asset + returns portafoglio
      - costruisce equity
      - calcola metriche
    """

    tickers_list, w_from_cfg = _normalize_tickers_and_weights(tickers, weights)

    if not tickers_list:
        raise ValueError("tickers vuoti: inserisci almeno 1 ticker valido.")

    prices, valid_tickers, missing = _fetch_prices_with_missing(
        tickers_list,
        start=start,
        end=end,
        interval="1d",
    )

    if len(valid_tickers) == 0 or prices.empty:
        raise RuntimeError("Tutti i ticker forniti sono risultati non validi (nessun dato scaricato).")

    # weights: se dalla config avevamo pesi, usali; altrimenti prova weights separati
    w_in = w_from_cfg
    if w_in is None:
        w_in = _normalize_weights(weights, valid_tickers)

    w = _make_weights(valid_tickers, w_in)

    # returns assets
    rets_assets = pct_returns(prices)
    if rets_assets.empty:
        raise RuntimeError("Rendimenti asset vuoti: dati insufficienti dopo la pulizia.")

    # returns portfolio (da returns asset + pesi)
    w_s = pd.Series(w, index=valid_tickers, dtype=float)
    rets_port = (rets_assets.mul(w_s, axis=1)).sum(axis=1).replace([np.inf, -np.inf], np.nan).dropna()
    if rets_port.empty:
        raise RuntimeError("Rendimenti portafoglio vuoti: controlla prezzi/pesi.")

    # equity
    equity = (1.0 + rets_port).cumprod() * float(initial)
    equity = equity.replace([np.inf, -np.inf], np.nan).dropna()
    if equity.empty:
        raise RuntimeError("Equity vuota: controlla dati e pesi.")

    # =============== Metriche ===============
    cagr = float(cagr_from_returns(rets_port))
    sigma_ann = float(vol_ann(rets_port))

    dd = drawdown_series(equity)
    mdd = float(dd.min()) if dd is not None and len(dd) else np.nan

    ulcer = float(ulcer_index_from_equity(equity))
    burke = float(burke_ratio(equity, cagr))
    sterling = float(sterling_ratio(equity, cagr))

    calmar = (cagr / abs(mdd)) if (np.isfinite(mdd) and mdd < 0) else np.nan
    mar = calmar

    pain = float(pain_ratio(equity, cagr))
    kappa3 = float(kappa_ratio(rets_port, mar_annual=rf_annual, order=3))
    sharpe = float(sharpe_ratio(rets_port, rf_annual=rf_annual))
    sortino = float(sortino_ratio(rets_port, rf_annual=rf_annual))
    rachev = float(rachev_ratio(rets_port, alpha=rachev_alpha))
    es95 = float(es_cvar(rets_port, 0.95))
    es99 = float(es_cvar(rets_port, 0.99))
    omega0 = float(omega_ratio(rets_port, threshold=0.0))

    metrics = pd.Series(
        {
            "CAGR": cagr,
            "Volatilità ann.": sigma_ann,
            "Sharpe": sharpe,
            "Sortino": sortino,
            "Max Drawdown": mdd,
            "Ulcer Index": ulcer,
            "Burke": burke,
            "Kappa(3)": kappa3,
            "Sterling": sterling,
            "Calmar": calmar,
            "MAR": mar,
            "Pain Ratio": pain,
            f"Rachev (α={rachev_alpha:.2f})": rachev,
            "ES/CVaR 95% (giorn.)": es95,
            "ES/CVaR 99% (giorn.)": es99,
            "Omega(τ=0)": omega0,
        }
    )

    return {
        "prices": prices,
        "equity": equity,
        "returns_portfolio": rets_port,
        "returns_assets": rets_assets,
        "weights": pd.Series(w, index=valid_tickers),
        "metrics": metrics,
        "correlation": rets_assets.corr(),
        "missing": missing,
        "valid_tickers": valid_tickers,
    }
