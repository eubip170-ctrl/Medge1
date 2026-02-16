# core/analisi_metrics.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from core.portfolio_core import (
    cagr_from_returns,
    vol_ann,
    sharpe_ratio,
    sortino_ratio,
    drawdown_series,
    es_cvar,
    omega_ratio,
    rachev_ratio,
)

TRADING_DAYS = 252


@dataclass(frozen=True)
class MetricSpec:
    key: str
    label: str
    higher_is_better: bool
    fmt: str  # 'pct', 'ratio', 'raw'
    neutral_band: Optional[Tuple[float, float]] = None


def metric_specs() -> List[MetricSpec]:
    return [
        MetricSpec("cagr", "CAGR", True, "pct", neutral_band=(0.00, 0.08)),
        MetricSpec("vol_ann", "Volatilità annuale", False, "pct", neutral_band=(0.10, 0.20)),
        MetricSpec("sharpe", "Sharpe ratio", True, "ratio", neutral_band=(0.50, 1.00)),
        MetricSpec("sortino", "Sortino ratio", True, "ratio", neutral_band=(0.75, 1.50)),
        MetricSpec("max_dd", "Max drawdown", True, "pct", neutral_band=(-0.30, -0.10)),
        MetricSpec("ulcer", "Ulcer index", False, "raw", neutral_band=(0.05, 0.15)),
        MetricSpec("burke", "Burke ratio", True, "ratio"),
        MetricSpec("kappa3", "Kappa ratio (3)", True, "ratio"),
        MetricSpec("sterling", "Sterling ratio", True, "ratio"),
        MetricSpec("calmar", "Calmar ratio", True, "ratio"),
        MetricSpec("mar", "MAR", True, "ratio"),
        MetricSpec("pain_ratio", "Pain ratio", True, "ratio"),
        MetricSpec("rachev", "Rachev", True, "ratio"),
        MetricSpec("cvar95", "CVaR 95% (daily)", True, "pct"),
        MetricSpec("cvar99", "CVaR 99% (daily)", True, "pct"),
        MetricSpec("omega0", "Omega (τ=0)", True, "ratio"),
        MetricSpec("beta", "Beta (vs bench)", False, "raw", neutral_band=(0.80, 1.20)),
        MetricSpec("alpha_ann", "Alpha annuale (vs bench)", True, "pct", neutral_band=(-0.02, 0.02)),
        MetricSpec("info_ratio", "Information ratio", True, "ratio"),
        MetricSpec("tracking_err", "Tracking error (ann.)", False, "pct"),
        MetricSpec("skew", "Skewness", True, "raw", neutral_band=(-0.5, 0.5)),
        MetricSpec("ex_kurt", "Excess kurtosis", False, "raw", neutral_band=(0.0, 3.0)),
        MetricSpec("win_rate", "Win rate (daily)", True, "pct", neutral_band=(0.48, 0.55)),
    ]


def _to_series(x) -> pd.Series:
    if x is None:
        return pd.Series(dtype=float)
    if isinstance(x, pd.Series):
        s = x.copy()
    else:
        s = pd.Series(x)
    s = pd.to_numeric(s, errors="coerce").dropna()
    return s.astype(float)


def _returns_from_series(series: pd.Series) -> pd.Series:
    s = _to_series(series)
    if len(s) < 2:
        return pd.Series(dtype=float)
    return s.pct_change().replace([np.inf, -np.inf], np.nan).dropna()


def ulcer_index(series: pd.Series) -> float:
    s = _to_series(series)
    if len(s) < 2:
        return np.nan
    dd = drawdown_series(s).dropna()
    if dd.empty:
        return np.nan
    return float(np.sqrt(np.mean(np.square(dd.values))))


def pain_index(series: pd.Series) -> float:
    s = _to_series(series)
    if len(s) < 2:
        return np.nan
    dd = drawdown_series(s).dropna()
    if dd.empty:
        return np.nan
    return float(np.mean(np.abs(dd.values)))



def calmar_ratio(series: pd.Series) -> float:
    r = _returns_from_series(series)
    if r.empty:
        return np.nan
    cagr = float(cagr_from_returns(r))
    mdd = float(drawdown_series(_to_series(series)).min())
    den = abs(mdd) if np.isfinite(mdd) else np.nan
    if not np.isfinite(den) or den <= 0:
        return np.nan
    return float(cagr / den)


def mar_ratio(series: pd.Series) -> float:
    return calmar_ratio(series)


def sterling_ratio(series: pd.Series, rf_annual: float = 0.0) -> float:
    r = _returns_from_series(series)
    if r.empty:
        return np.nan
    cagr = float(cagr_from_returns(r))
    dd = drawdown_series(_to_series(series)).dropna()
    if dd.empty:
        return np.nan
    depths = np.abs(dd.values)
    k = max(5, int(0.05 * len(depths)))
    worst = np.sort(depths)[-k:]
    avg_worst = float(np.mean(worst)) if len(worst) else np.nan
    if not np.isfinite(avg_worst) or avg_worst <= 0:
        return np.nan
    return float((cagr - float(rf_annual)) / avg_worst)


def burke_ratio(series: pd.Series, rf_annual: float = 0.0) -> float:
    r = _returns_from_series(series)
    if r.empty:
        return np.nan
    cagr = float(cagr_from_returns(r))
    dd_rms = float(ulcer_index(series))
    if not np.isfinite(dd_rms) or dd_rms <= 0:
        return np.nan
    return float((cagr - float(rf_annual)) / dd_rms)


def kappa_ratio(rets: pd.Series, order: int = 3, threshold: float = 0.0) -> float:
    r = _to_series(rets)
    if r.empty:
        return np.nan
    excess = r - float(threshold)
    downside = np.maximum(-excess.values, 0.0)
    lpm = float(np.mean(downside ** order))
    if not np.isfinite(lpm) or lpm <= 0:
        return np.nan
    denom = lpm ** (1.0 / order)
    mu = float(np.mean(excess.values))
    if denom == 0:
        return np.nan
    return float(mu / denom)


def _align(a: pd.Series, b: pd.Series) -> Tuple[pd.Series, pd.Series]:
    a = _to_series(a)
    b = _to_series(b)
    df = pd.concat([a.rename("a"), b.rename("b")], axis=1).dropna()
    if df.empty:
        return pd.Series(dtype=float), pd.Series(dtype=float)
    return df["a"], df["b"]


def beta_alpha_annual(rets_port: pd.Series, rets_bench: pd.Series, rf_annual: float = 0.0) -> Tuple[float, float]:
    rp, rb = _align(rets_port, rets_bench)
    if rp.empty or rb.empty or len(rp) < 10:
        return np.nan, np.nan

    rf_d = (1.0 + float(rf_annual)) ** (1.0 / TRADING_DAYS) - 1.0 if rf_annual is not None else 0.0
    x = rb - rf_d
    y = rp - rf_d

    var = float(np.var(x.values, ddof=1))
    if not np.isfinite(var) or var <= 0:
        return np.nan, np.nan
    cov = float(np.cov(x.values, y.values, ddof=1)[0, 1])
    beta = cov / var
    alpha_daily = float(np.mean(y.values) - beta * np.mean(x.values))
    alpha_ann = alpha_daily * TRADING_DAYS
    return float(beta), float(alpha_ann)


def tracking_error_annual(rets_port: pd.Series, rets_bench: pd.Series) -> float:
    rp, rb = _align(rets_port, rets_bench)
    if rp.empty or rb.empty or len(rp) < 10:
        return np.nan
    diff = rp - rb
    return float(diff.std(ddof=1) * np.sqrt(TRADING_DAYS))


def information_ratio(rets_port: pd.Series, rets_bench: pd.Series) -> float:
    rp, rb = _align(rets_port, rets_bench)
    if rp.empty or rb.empty or len(rp) < 10:
        return np.nan
    diff = rp - rb
    te = float(diff.std(ddof=1))
    if not np.isfinite(te) or te <= 0:
        return np.nan
    return float((diff.mean() / te) * np.sqrt(TRADING_DAYS))


def compute_metrics_table(
    series_map: Dict[str, pd.Series],
    rf_annual: float = 0.0,
    bench_name: Optional[str] = None,
) -> pd.DataFrame:
    specs = metric_specs()
    names = list(series_map.keys())
    if not names:
        return pd.DataFrame()

    rets_map: Dict[str, pd.Series] = {n: _returns_from_series(series_map[n]) for n in names}
    bench_rets = rets_map.get(bench_name, pd.Series(dtype=float)) if bench_name else pd.Series(dtype=float)

    out = pd.DataFrame(index=[s.label for s in specs], columns=names, dtype=float)

    for n in names:
        s = series_map.get(n)
        r = rets_map.get(n, pd.Series(dtype=float))
        if s is None or r.empty:
            continue

        cagr = float(cagr_from_returns(r))
        vol = float(vol_ann(r))
        shrp = float(sharpe_ratio(r, rf_annual=rf_annual))
        sort = float(sortino_ratio(r, rf_annual=rf_annual))
        mdd = float(drawdown_series(_to_series(s)).min())
        ui = float(ulcer_index(s))
        burke = float(burke_ratio(s, rf_annual=rf_annual))
        kappa3 = float(kappa_ratio(r, order=3, threshold=0.0))
        sterling = float(sterling_ratio(s, rf_annual=rf_annual))
        calmar = float(calmar_ratio(s))
        mar = float(mar_ratio(s))
        pain = float(pain_index(s))
        pain_ratio_val = float((cagr - rf_annual) / pain) if (np.isfinite(pain) and pain > 0) else np.nan
        rach = float(rachev_ratio(r, 0.05))
        cvar95 = float(es_cvar(r, 0.95))
        cvar99 = float(es_cvar(r, 0.99))
        omg = float(omega_ratio(r, 0.0))
        win = float((r > 0).mean()) if len(r) else np.nan
        skew = float(r.skew()) if len(r) > 2 else np.nan
        exk = float(r.kurt()) if len(r) > 3 else np.nan

        beta = alpha_ann = te = ir = np.nan
        if bench_name and bench_name in rets_map and n != bench_name and not bench_rets.empty:
            beta, alpha_ann = beta_alpha_annual(r, bench_rets, rf_annual=rf_annual)
            te = tracking_error_annual(r, bench_rets)
            ir = information_ratio(r, bench_rets)

        values = {
            "CAGR": cagr,
            "Volatilità annuale": vol,
            "Sharpe ratio": shrp,
            "Sortino ratio": sort,
            "Max drawdown": mdd,
            "Ulcer index": ui,
            "Burke ratio": burke,
            "Kappa ratio (3)": kappa3,
            "Sterling ratio": sterling,
            "Calmar ratio": calmar,
            "MAR": mar,
            "Pain ratio": pain_ratio_val,
            "Rachev": rach,
            "CVaR 95% (daily)": cvar95,
            "CVaR 99% (daily)": cvar99,
            "Omega (τ=0)": omg,
            "Beta (vs bench)": beta,
            "Alpha annuale (vs bench)": alpha_ann,
            "Information ratio": ir,
            "Tracking error (ann.)": te,
            "Skewness": skew,
            "Excess kurtosis": exk,
            "Win rate (daily)": win,
        }

        for spec in specs:
            out.loc[spec.label, n] = values.get(spec.label, np.nan)

    return out


def compute_asset_correlation(prices: pd.DataFrame) -> pd.DataFrame:
    if prices is None or not isinstance(prices, pd.DataFrame) or prices.empty:
        return pd.DataFrame()
    df = prices.copy().sort_index().ffill()
    rets = df.pct_change().replace([np.inf, -np.inf], np.nan).dropna(how="all")
    if rets.empty:
        return pd.DataFrame()
    return rets.corr()
