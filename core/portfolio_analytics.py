# core/portfolio_analytics.py

import numpy as np
import pandas as pd
from math import sqrt
from dataclasses import dataclass, field

# SciPy per cdf normale
try:
    from scipy.stats import norm

    _HAS_SCIPY = True
except Exception:
    _HAS_SCIPY = False

    def _norm_cdf(x: np.ndarray) -> np.ndarray:
        import math

        v = np.asarray(x, dtype=float)
        return 0.5 * (
            1.0
            + np.vectorize(lambda z: math.erf(z / sqrt(2.0)))(v)
        )


# ARCH per GARCH (opzionale)
try:
    from arch import arch_model

    _HAS_ARCH = True
except Exception:
    _HAS_ARCH = False


# =============================
# 📊 Volume Profile
# =============================
def volume_profile_features(
    ohlcv_win: pd.DataFrame,
    bins: int = 50,
) -> dict:
    """
    Calcolo semplice del volume profile su 'close' pesato per 'volume'.
    Restituisce POC, VAH, VAL con value area ~70%.
    """
    if (
        ohlcv_win is None
        or ohlcv_win.empty
        or "close" not in ohlcv_win
        or "volume" not in ohlcv_win
    ):
        return {}
    close = pd.Series(ohlcv_win["close"]).astype(float)
    vol = (
        pd.Series(ohlcv_win["volume"])
        .astype(float)
        .clip(lower=0)
    )
    pr_min, pr_max = float(close.min()), float(close.max())
    if pr_max <= pr_min:
        return {}

    hist, edges = np.histogram(
        close.values,
        bins=bins,
        range=(pr_min, pr_max),
        weights=vol.values,
    )
    poc_idx = int(np.argmax(hist))
    poc = 0.5 * (edges[poc_idx] + edges[poc_idx + 1])

    # value area: accumula intorno al POC fino al 70%
    total = hist.sum()
    target = 0.70 * total
    left = right = poc_idx
    acc = hist[poc_idx]

    while acc < target and (left > 0 or right < len(hist) - 1):
        lnext = hist[left - 1] if left > 0 else -1
        rnext = hist[right + 1] if right < len(hist) - 1 else -1
        if rnext >= lnext and right < len(hist) - 1:
            right += 1
            acc += hist[right]
        elif left > 0:
            left -= 1
            acc += hist[left]
        else:
            break

    val = edges[left]  # lower bound
    vah = edges[right + 1]  # upper bound

    return {"poc": float(poc), "val": float(val), "vah": float(vah)}


# =============================
# 📈 Trend / flow leggeri
# =============================
def compute_orderflow_trend_features(
    ohlcv: pd.DataFrame,
    lookback: int = 60,
) -> dict:
    """
    Feature leggere: slope log-prezzi, R^2, momentum 20, volumi avg.
    """
    if ohlcv is None or ohlcv.empty or "close" not in ohlcv:
        return {}
    s = pd.Series(ohlcv["close"]).astype(float).tail(lookback)
    if s.size < 5:
        return {}
    x = np.arange(s.size)
    X = np.column_stack([x, np.ones_like(x)])

    beta, alpha = np.linalg.lstsq(
        X,
        np.log(s.values),
        rcond=None,
    )[0]
    yhat = beta * x + alpha
    ss_tot = float(
        ((np.log(s.values) - np.log(s.values).mean()) ** 2).sum()
    )
    ss_res = float(((np.log(s.values) - yhat) ** 2).sum())
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
    mom20 = (
        float(s.iloc[-1] / s.iloc[max(0, s.size - 21)] - 1.0)
        if s.size >= 21
        else np.nan
    )
    if "volume" in ohlcv.columns:
        vol_avg = float(
            pd.Series(ohlcv["volume"]).tail(lookback).mean()
        )
    else:
        vol_avg = np.nan

    return {
        "trend_slope_log": float(beta),
        "trend_R2": r2,
        "momentum_20": mom20,
        "avg_volume": vol_avg,
    }


# =============================
# 🎯 Probabilità fine periodo
# =============================
@dataclass
class EWMAConfig:
    lam: float = 0.94


@dataclass
class GARCHConfig:
    dist: str = "t"  # ignorato se arch non disponibile


@dataclass
class EVTConfig:
    use_evt: bool = False
    threshold_q: float = 0.95
    min_exceedances: int = 50


@dataclass
class EndPeriodConfig:
    use_garch: bool = False
    ewma: EWMAConfig = field(default_factory=EWMAConfig)
    garch: GARCHConfig = field(default_factory=GARCHConfig)
    evt: EVTConfig = field(default_factory=EVTConfig)


class EndPeriodProbability:
    """
    Stima semplice: volatilità condizionata con EWMA o GARCH(1,1) se disponibile.
    Probabilità su orizzonte T assumendo normalità (evt opzionale: fattore tail).
    """

    def __init__(self, cfg: EndPeriodConfig):
        self.cfg = cfg
        self._last_sigma_path = None
        self._evt_scale = 1.0

    def _ewma_sigma_path(self, r: pd.Series) -> np.ndarray:
        lam = float(self.cfg.ewma.lam)
        s2 = []
        r = pd.Series(r).dropna()
        var = (
            float(r.iloc[:20].var(ddof=1))
            if r.size >= 20
            else float(r.var(ddof=1))
        )
        for ret in r:
            var = lam * var + (1 - lam) * (float(ret) ** 2)
            s2.append(max(var, 1e-12))
        return np.sqrt(np.asarray(s2, dtype=float))

    def fit_residuals_for_evt(self, r: pd.Series):
        r = pd.Series(r).dropna()
        if r.empty:
            self._evt_scale = 1.0
            return

        s_path = self._ewma_sigma_path(r)
        z = r.values / np.where(s_path <= 1e-12, np.nan, s_path)
        z = z[np.isfinite(z)]

        if (
            self.cfg.evt.use_evt
            and z.size
            >= max(100, self.cfg.evt.min_exceedances + 10)
        ):
            q = np.quantile(np.abs(z), self.cfg.evt.threshold_q)
            tail = np.abs(z)[np.abs(z) >= q]
            if tail.size >= self.cfg.evt.min_exceedances and q > 0:
                self._evt_scale = float(np.mean(tail) / q)
                self._evt_scale = float(
                    np.clip(self._evt_scale, 1.0, 2.5)
                )
            else:
                self._evt_scale = 1.0
        else:
            self._evt_scale = 1.0

    def _sigma_T(self, r: pd.Series, T: int) -> float:
        r = pd.Series(r).dropna()
        if r.empty or T <= 0:
            return np.nan

        if self.cfg.use_garch and _HAS_ARCH:
            try:
                am = arch_model(
                    r * 100.0,
                    vol="GARCH",
                    p=1,
                    q=1,
                    o=0,
                    dist=self.cfg.garch.dist,
                )
                res = am.fit(disp="off")
                f = (
                    res.forecast(
                        horizon=T,
                        reindex=False,
                    )
                    .variance.values[-1]
                )  # var_1..var_T (%^2)
                sigma = float(
                    np.sqrt(np.sum(f)) / 100.0
                )  # torna da % a unit
            except Exception:
                sigma = float(r.std(ddof=1) * sqrt(T))
        else:
            s_path = self._ewma_sigma_path(r)
            self._last_sigma_path = (
                s_path[-T:].copy()
                if s_path.size >= T
                else s_path.copy()
            )
            sig_daily = (
                float(np.nanmean(self._last_sigma_path))
                if self._last_sigma_path.size
                else float(r.std(ddof=1))
            )
            sigma = sig_daily * sqrt(T)

        return float(sigma * self._evt_scale)

    def _mu_T(self, r: pd.Series, T: int) -> float:
        m = float(pd.Series(r).dropna().mean())
        return m * T

    def prob_ge(self, r: pd.Series, T: int, tau_log: float) -> float:
        """
        Probabilità che log-return finale >= tau_log.
        """
        if T <= 0:
            return np.nan
        muT = self._mu_T(r, T)
        sigT = self._sigma_T(r, T)
        if not np.isfinite(sigT) or sigT <= 0:
            return np.nan
        z = (tau_log - muT) / sigT
        if _HAS_SCIPY:
            return float(1.0 - norm.cdf(z))
        return float(1.0 - _norm_cdf(z))

    def prob_le(self, r: pd.Series, T: int, tau_log: float) -> float:
        """
        Probabilità che log-return finale <= tau_log.
        """
        if T <= 0:
            return np.nan
        muT = self._mu_T(r, T)
        sigT = self._sigma_T(r, T)
        if not np.isfinite(sigT) or sigT <= 0:
            return np.nan
        z = (tau_log - muT) / sigT
        if _HAS_SCIPY:
            return float(norm.cdf(z))
        return float(_norm_cdf(z))


# =============================
# ⛳ Hitting probability (GBM)
# =============================
def hitting_probability_gbm(
    mu: float,
    sigma: float,
    T: int,
    tau_simple: float,
) -> float:
    """
    Probabilità che il log-return superi ln(1+tau) entro T (barriera superiore).
    Formula di riflessione per Brownian con drift m = mu - 0.5*sigma^2.
    """
    if (
        not np.isfinite(mu)
        or not np.isfinite(sigma)
        or sigma <= 0
        or T <= 0
        or tau_simple <= 0
    ):
        return np.nan

    a = float(np.log1p(tau_simple))
    m = float(mu - 0.5 * sigma * sigma)
    sT = float(sigma * np.sqrt(T))
    z1 = (a - m * T) / sT
    z2 = (-a - m * T) / sT

    if _HAS_SCIPY:
        return float(
            1.0
            - norm.cdf(z1)
            + np.exp(2 * m * a / (sigma * sigma)) * norm.cdf(z2)
        )
    return float(
        1.0
        - _norm_cdf(z1)
        + np.exp(2 * m * a / (sigma * sigma)) * _norm_cdf(z2)
    )
