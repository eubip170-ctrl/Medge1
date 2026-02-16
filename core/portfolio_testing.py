# core/portfolio_testing.py
from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Dict, Iterable, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from core.portfolio_core import (
    cagr_from_returns,
    vol_ann,
    sharpe_ratio,
    sortino_ratio,
    drawdown_series,
)


# -----------------------------
# Index / data normalization
# -----------------------------
def _to_naive_daily_index(idx: pd.Index) -> pd.DatetimeIndex:
    dt = pd.to_datetime(idx, errors="coerce")
    try:
        if getattr(dt, "tz", None) is not None:
            dt = dt.tz_convert(None)
    except Exception:
        pass
    try:
        dt = dt.tz_localize(None)
    except Exception:
        pass
    return pd.DatetimeIndex(dt).normalize()


def normalize_df_daily(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    out = df.copy()
    out.index = _to_naive_daily_index(out.index)
    out = out[~out.index.duplicated(keep="last")]
    return out.sort_index()


def normalize_series_daily(s: pd.Series) -> pd.Series:
    if s is None or len(s) == 0:
        return pd.Series(dtype=float)
    out = pd.Series(s).dropna().astype(float).copy()
    out.index = _to_naive_daily_index(out.index)
    out = out[~out.index.duplicated(keep="last")]
    return out.sort_index()


# -----------------------------
# Weights alignment
# -----------------------------
WeightsLike = Union[pd.Series, Dict[str, float], List[float], Tuple[float, ...], np.ndarray]


def align_weights(prices: pd.DataFrame, weights: Optional[WeightsLike]) -> pd.Series:
    """
    Returns weights Series aligned to prices.columns.
    If weights missing/invalid -> equal-weight.
    Always clipped >=0 and normalized to sum=1 (if sum>0).
    """
    if prices is None or prices.empty:
        return pd.Series(dtype=float)

    cols = [str(c).upper() for c in prices.columns]
    n = len(cols)

    if weights is None:
        return pd.Series(1.0 / n, index=cols, dtype=float)

    # Series / dict with tickers
    if isinstance(weights, pd.Series):
        w = weights.copy()
        if w.index.dtype != object:
            w.index = w.index.astype(str)
        w.index = w.index.astype(str).str.upper()
        w = w.groupby(level=0).sum().reindex(cols).fillna(0.0).astype(float)

    elif isinstance(weights, dict):
        w = pd.Series(weights, dtype=float)
        w.index = w.index.astype(str).str.upper()
        w = w.groupby(level=0).sum().reindex(cols).fillna(0.0).astype(float)

    else:
        # list/array -> assume same order as prices.columns
        arr = np.array(weights, dtype=float).reshape(-1)
        if arr.size != n:
            return pd.Series(1.0 / n, index=cols, dtype=float)
        w = pd.Series(arr, index=cols, dtype=float)

    w = pd.to_numeric(w, errors="coerce").fillna(0.0).astype(float).clip(lower=0.0)
    s = float(w.sum())
    if s > 0:
        w = w / s
    else:
        w = pd.Series(1.0 / n, index=cols, dtype=float)
    return w


# -----------------------------
# Portfolio returns/equity
# -----------------------------
def portfolio_returns(prices: pd.DataFrame, w: pd.Series) -> pd.Series:
    """
    Buy&hold style: daily returns from prices * fixed weights.
    """
    prices = normalize_df_daily(prices)
    if prices.empty:
        return pd.Series(dtype=float)

    cols = [str(c).upper() for c in prices.columns]
    prices.columns = cols

    w = pd.Series(w, dtype=float).reindex(cols).fillna(0.0).astype(float)
    s = float(w.sum())
    if s > 0:
        w = w / s

    rets = prices.pct_change().replace([np.inf, -np.inf], np.nan).dropna(how="all")
    if rets.empty:
        return pd.Series(dtype=float)

    pr = (rets.mul(w, axis=1)).sum(axis=1)
    pr.name = "PORT_RET"
    return pr.dropna()


def portfolio_equity(port_rets: pd.Series, initial: float = 100.0) -> pd.Series:
    r = normalize_series_daily(port_rets).dropna()
    if r.empty:
        return pd.Series(dtype=float)
    eq = (1.0 + r).cumprod() * float(initial)
    eq.name = "PORT_EQUITY"
    return eq


# -----------------------------
# Stats / stress utilities
# -----------------------------
def window_slice(s: pd.Series, start: date, end: date) -> pd.Series:
    ss = normalize_series_daily(s)
    if ss.empty:
        return ss
    a = pd.Timestamp(start).normalize()
    b = pd.Timestamp(end).normalize()
    return ss[(ss.index >= a) & (ss.index <= b)]


def compute_stats_from_equity(eq: pd.Series, rf_annual: float = 0.0) -> Dict[str, float]:
    eq = normalize_series_daily(eq).dropna()
    if len(eq) < 2:
        return {}

    rets = eq.pct_change().replace([np.inf, -np.inf], np.nan).dropna()
    if rets.empty:
        return {}

    out = {
        "Total return": float(eq.iloc[-1] / eq.iloc[0] - 1.0),
        "CAGR": float(cagr_from_returns(rets)),
        "Vol (ann.)": float(vol_ann(rets)),
        "Sharpe": float(sharpe_ratio(rets, rf_annual=rf_annual)),
        "Sortino": float(sortino_ratio(rets, rf_annual=rf_annual)),
        "Max Drawdown": float(drawdown_series(eq).min()),
    }
    return out


def estimate_beta_corr(port_rets: pd.Series, proxy_rets: pd.Series) -> Dict[str, float]:
    p = normalize_series_daily(port_rets).dropna().astype(float)
    x = normalize_series_daily(proxy_rets).dropna().astype(float)
    idx = p.index.intersection(x.index)
    if len(idx) < 20:
        return {}

    p = p.loc[idx]
    x = x.loc[idx]

    vx = float(np.var(x.values, ddof=1))
    if not np.isfinite(vx) or vx <= 0:
        beta = np.nan
    else:
        cov = float(np.cov(p.values, x.values, ddof=1)[0, 1])
        beta = cov / vx

    corr = float(np.corrcoef(p.values, x.values)[0, 1])
    r2 = float(corr * corr) if np.isfinite(corr) else np.nan
    return {"beta": float(beta), "corr": float(corr), "r2": float(r2), "n": int(len(idx))}


# -----------------------------
# Crisis presets
# -----------------------------
CRISIS_PRESETS = [
    {"name": "2008 — GFC (Sep 2008 → Mar 2009)", "start": date(2008, 9, 1), "end": date(2009, 3, 31)},
    {"name": "2011 — Euro debt (Jul 2011 → Oct 2011)", "start": date(2011, 7, 1), "end": date(2011, 10, 31)},
    {"name": "2018 — Q4 selloff (Oct 2018 → Dec 2018)", "start": date(2018, 10, 1), "end": date(2018, 12, 31)},
    {"name": "2020 — Covid crash (Feb 2020 → Mar 2020)", "start": date(2020, 2, 15), "end": date(2020, 3, 31)},
    {"name": "2022 — Inflation bear (Jan 2022 → Oct 2022)", "start": date(2022, 1, 1), "end": date(2022, 10, 31)},
]


# -----------------------------
# Weak points
# -----------------------------
def weak_points_summary(prices: pd.DataFrame, w: pd.Series) -> Dict[str, object]:
    prices = normalize_df_daily(prices)
    if prices.empty:
        return {}

    cols = [str(c).upper() for c in prices.columns]
    prices.columns = cols
    w = pd.Series(w, dtype=float).reindex(cols).fillna(0.0).astype(float)
    s = float(w.sum())
    if s > 0:
        w = w / s

    out: Dict[str, object] = {}

    # concentration
    hhi = float((w**2).sum())
    out["hhi"] = hhi
    out["top3_pct"] = float(w.sort_values(ascending=False).head(3).sum() * 100.0)
    out["top5_pct"] = float(w.sort_values(ascending=False).head(5).sum() * 100.0)

    # correlations
    rets = prices.pct_change().replace([np.inf, -np.inf], np.nan).dropna(how="all")
    if rets.shape[0] >= 30 and rets.shape[1] >= 2:
        corr = rets.corr().replace([np.inf, -np.inf], np.nan)
        out["avg_corr"] = float(np.nanmean(corr.values[np.triu_indices_from(corr.values, k=1)]))

        # top correlated pairs
        pairs = []
        for i in range(len(cols)):
            for j in range(i + 1, len(cols)):
                v = corr.iloc[i, j]
                if np.isfinite(v):
                    pairs.append((cols[i], cols[j], float(v)))
        pairs.sort(key=lambda x: x[2], reverse=True)
        out["top_corr_pairs"] = pairs[:8]

        pairs.sort(key=lambda x: x[2])
        out["low_corr_pairs"] = pairs[:8]
    else:
        out["avg_corr"] = np.nan
        out["top_corr_pairs"] = []
        out["low_corr_pairs"] = []

    return out
