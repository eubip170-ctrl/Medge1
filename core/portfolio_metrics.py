# core/portfolio_metrics.py

import numpy as np
import pandas as pd
from math import sqrt

from .portfolio_core import TRADING_DAYS


# =============================
# 📈 Serie di drawdown
# =============================
def drawdown_series(equity_curve: pd.Series) -> pd.Series:
    """Ritorna la serie di drawdown (equity / max cum - 1)."""
    eq = pd.Series(equity_curve).astype(float)
    return eq / eq.cummax() - 1.0


# =============================
# 🧮 Metriche base
# =============================
def cagr_from_returns(
    returns: pd.Series,
    periods_per_year: int = TRADING_DAYS,
) -> float:
    r = pd.Series(returns).dropna()
    if r.empty:
        return np.nan
    growth = float((1.0 + r).prod())
    n = r.shape[0]
    if n == 0 or growth <= 0:
        return np.nan
    return growth ** (periods_per_year / n) - 1.0


def vol_ann(
    returns: pd.Series,
    periods_per_year: int = TRADING_DAYS,
) -> float:
    r = pd.Series(returns).dropna()
    if r.empty:
        return np.nan
    return r.std(ddof=0) * sqrt(periods_per_year)


def downside_deviation_ann(
    returns: pd.Series,
    mar_per_period: float = 0.0,
    periods_per_year: int = TRADING_DAYS,
) -> float:
    r = pd.Series(returns).dropna()
    if r.empty:
        return np.nan
    downside = np.minimum(r - mar_per_period, 0.0)
    return np.sqrt((downside ** 2).mean()) * sqrt(periods_per_year)


def ulcer_index_from_equity(equity_curve: pd.Series) -> float:
    dd = drawdown_series(equity_curve)
    return float(np.sqrt(np.mean(np.square(dd))))


# =============================
# 📉 Rachev & momenti inferiori
# =============================
def rachev_ratio(returns: pd.Series, alpha: float = 0.05):
    r = pd.Series(returns).dropna()
    if r.empty:
        return np.nan
    lower_q = np.percentile(r, 100 * alpha)
    upper_q = np.percentile(r, 100 * (1 - alpha))
    upper_tail = r[r >= upper_q]
    lower_tail = r[r <= lower_q]
    if len(upper_tail) == 0 or len(lower_tail) == 0:
        return np.nan
    denom = abs(float(lower_tail.mean()))
    return np.nan if denom == 0 else float(upper_tail.mean()) / denom


def lpm(
    returns: pd.Series,
    mar_per_period: float = 0.0,
    order: int = 3,
):
    r = pd.Series(returns).dropna()
    shortfall = np.clip(mar_per_period - r, 0, None)
    return (shortfall ** order).mean()


def kappa_ratio(
    returns: pd.Series,
    mar_annual: float = 0.0,
    order: int = 3,
    periods_per_year: int = TRADING_DAYS,
):
    r = pd.Series(returns).dropna()
    if r.empty:
        return np.nan
    mar_per = mar_annual / periods_per_year
    num = (r - mar_per).mean() * periods_per_year
    denom = (lpm(r, mar_per, order)) ** (1 / order)
    denom *= periods_per_year ** (1 / order)
    return np.nan if denom == 0 else num / denom


# =============================
# ⚖️ Sharpe, Sortino, ecc.
# =============================
def sharpe_ratio(
    returns: pd.Series,
    rf_annual: float = 0.0,
    periods_per_year: int = TRADING_DAYS,
):
    r = pd.Series(returns).dropna()
    if r.empty:
        return np.nan
    rf_per = rf_annual / periods_per_year
    excess = r - rf_per
    sigma = vol_ann(r, periods_per_year)
    if not np.isfinite(sigma) or sigma == 0:
        return np.nan
    return (excess.mean() * periods_per_year) / sigma


def sortino_ratio(
    returns: pd.Series,
    rf_annual: float = 0.0,
    periods_per_year: int = TRADING_DAYS,
):
    r = pd.Series(returns).dropna()
    if r.empty:
        return np.nan
    rf_per = rf_annual / periods_per_year
    num = (r.mean() - rf_per) * periods_per_year
    dd = downside_deviation_ann(
        r,
        mar_per_period=rf_per,
        periods_per_year=periods_per_year,
    )
    if not np.isfinite(dd) or dd == 0:
        return np.nan
    return num / dd


def burke_ratio(equity_curve: pd.Series, cagr: float):
    dd = drawdown_series(equity_curve).values
    rms_drawdown = np.sqrt(np.mean(np.square(dd)))
    return np.nan if rms_drawdown == 0 else cagr / rms_drawdown


def sterling_ratio(equity_curve: pd.Series, cagr: float):
    avg_dd = drawdown_series(equity_curve).abs().mean()
    return np.nan if avg_dd == 0 else cagr / avg_dd


def pain_index(equity_curve: pd.Series):
    return drawdown_series(equity_curve).abs().mean()


def pain_ratio(equity_curve: pd.Series, cagr: float):
    p = pain_index(equity_curve)
    return np.nan if p == 0 else cagr / p


# =============================
# 🧨 ES / CVaR & Omega
# =============================
def es_cvar(returns: pd.Series, alpha: float = 0.95):
    """
    ES/CVaR a livello alpha: media delle perdite (loss = -r) oltre la VaR alpha.
    """
    r = pd.Series(returns).dropna()
    if r.empty:
        return np.nan
    losses = -r.values
    var = np.quantile(losses, alpha)
    tail = losses[losses >= var]
    return float(np.mean(tail)) if len(tail) > 0 else np.nan


def omega_ratio(returns: pd.Series, threshold: float = 0.0):
    r = pd.Series(returns).dropna()
    if r.empty:
        return np.nan
    gains = np.clip(r - threshold, 0, None)
    losses = np.clip(threshold - r, 0, None)
    den = losses.mean()
    return np.inf if den == 0 else gains.mean() / den

