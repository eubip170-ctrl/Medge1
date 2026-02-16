# core/portfolio_optimization.py

import numpy as np
import pandas as pd
import warnings
from math import sqrt

from .portfolio_core import TRADING_DAYS

# ---- SciPy per l'ottimizzazione
try:
    from scipy.optimize import minimize
    _HAS_SCIPY = True
except Exception:
    _HAS_SCIPY = False


# ============================================================
# 🔍 Risk Contributions (Volatilità & CVaR)  (invariato)
# ============================================================
def risk_contributions(
    rets_assets: pd.DataFrame,
    w: np.ndarray,
    rets_port: pd.Series,
    alpha: float = 0.95,
):
    w = np.asarray(w, dtype=float)
    tickers = list(rets_assets.columns)

    cov_ann = rets_assets.cov().values * TRADING_DAYS
    sigma_ann = float(np.sqrt(max(0.0, w @ cov_ann @ w)))

    if sigma_ann > 0:
        mrc_vol = (cov_ann @ w) / sigma_ann
        crc_vol = w * mrc_vol
        pct_vol = crc_vol / sigma_ann
    else:
        mrc_vol = np.full_like(w, np.nan)
        crc_vol = np.full_like(w, np.nan)
        pct_vol = np.full_like(w, np.nan)

    losses = -pd.Series(rets_port).dropna()
    if len(losses) == 0:
        ES = np.nan
        crc_es = np.full_like(w, np.nan)
        pct_es = np.full_like(w, np.nan)
    else:
        q = float(np.quantile(losses.values, alpha))
        mask = losses >= q
        tail_R = rets_assets.loc[mask]
        ES = float(losses.loc[mask].mean()) if mask.sum() > 0 else np.nan

        if mask.sum() > 0 and ES > 0:
            grad_es = -tail_R.mean().values
            crc_es = w * grad_es
            pct_es = crc_es / ES
        else:
            crc_es = np.full_like(w, np.nan)
            pct_es = np.full_like(w, np.nan)

    df = pd.DataFrame(
        {
            "Weight": w,
            "Vol RC (ann.)": crc_vol,
            "Vol RC %": pct_vol,
            f"CVaR{int(alpha * 100)} RC (daily)": crc_es,
            f"CVaR{int(alpha * 100)} RC %": pct_es,
        },
        index=tickers,
    )

    return sigma_ann, ES, df


# ============================================================
# ⚙️ Helpers base
# ============================================================
def _prep_rets(rets_assets: pd.DataFrame) -> pd.DataFrame:
    d = rets_assets.copy()
    d = d.replace([np.inf, -np.inf], np.nan)
    d = d.dropna(how="all")
    d = d.dropna(axis=1, how="all")
    # per stabilità numerica (missing => 0), così l'obiettivo non esplode
    d = d.fillna(0.0)
    return d


def _portfolio_returns(w: np.ndarray, rets_assets: pd.DataFrame) -> np.ndarray:
    X = rets_assets.values
    return X @ np.asarray(w, dtype=float)


def _annualized_mean_std(w: np.ndarray, rets_assets: pd.DataFrame, rf_annual: float):
    w = np.asarray(w, dtype=float)
    mu_ann = float((rets_assets.mean().values @ w) * TRADING_DAYS)
    cov_ann = rets_assets.cov().values * TRADING_DAYS
    sigma_ann = float(np.sqrt(max(0.0, w @ cov_ann @ w)))
    rf = float(rf_annual)
    return mu_ann, sigma_ann, rf


def _downside_dev_ann(w: np.ndarray, rets_assets: pd.DataFrame, rf_annual: float):
    rf_per = float(rf_annual) / TRADING_DAYS
    rp = _portfolio_returns(w, rets_assets)
    downside = np.minimum(rp - rf_per, 0.0)
    return float(np.sqrt(np.mean(downside ** 2)) * sqrt(TRADING_DAYS))


def _es_objective(w: np.ndarray, rets_assets: pd.DataFrame, alpha: float) -> float:
    rp = _portfolio_returns(w, rets_assets)
    losses = -rp
    q = np.quantile(losses, alpha)
    tail = losses[losses >= q]
    return float(np.mean(tail)) if len(tail) > 0 else 0.0


# ============================================================
# 📈 Obiettivi “extra” (CAGR/DD/Calmar/Ulcer/Omega/Rachev)
# ============================================================
def _cagr_from_rp(rp: np.ndarray) -> float:
    rp = np.asarray(rp, dtype=float)
    if rp.size < 2:
        return np.nan
    eq = np.cumprod(1.0 + rp)
    if not np.isfinite(eq[-1]) or eq[-1] <= 0:
        return np.nan
    years = rp.size / float(TRADING_DAYS)
    if years <= 0:
        return np.nan
    return float(eq[-1] ** (1.0 / years) - 1.0)


def _max_drawdown_from_rp(rp: np.ndarray) -> float:
    rp = np.asarray(rp, dtype=float)
    if rp.size < 2:
        return np.nan
    eq = np.cumprod(1.0 + rp)
    peak = np.maximum.accumulate(eq)
    dd = eq / peak - 1.0
    return float(np.min(dd))


def _ulcer_index_from_rp(rp: np.ndarray) -> float:
    rp = np.asarray(rp, dtype=float)
    if rp.size < 2:
        return np.nan
    eq = np.cumprod(1.0 + rp)
    peak = np.maximum.accumulate(eq)
    dd = eq / peak - 1.0  # <= 0
    # Ulcer index classico usa drawdown% RMS; qui è in "decimali"
    return float(np.sqrt(np.mean(dd ** 2)))


def _omega_ratio(rp: np.ndarray, tau: float = 0.0) -> float:
    rp = np.asarray(rp, dtype=float)
    if rp.size < 5:
        return np.nan
    x = rp - float(tau)
    gains = np.sum(np.maximum(x, 0.0))
    losses = np.sum(np.maximum(-x, 0.0))
    if losses <= 0:
        return np.inf
    return float(gains / losses)


def _rachev_ratio(rp: np.ndarray, alpha: float = 0.05) -> float:
    """
    Rachev = E[r | r >= q_hi] / |E[r | r <= q_lo]|
    con q_lo=quantile(alpha), q_hi=quantile(1-alpha)
    """
    rp = np.asarray(rp, dtype=float)
    if rp.size < 20:
        return np.nan
    a = float(alpha)
    a = min(max(a, 0.001), 0.2)

    q_lo = np.quantile(rp, a)
    q_hi = np.quantile(rp, 1.0 - a)

    tail_neg = rp[rp <= q_lo]
    tail_pos = rp[rp >= q_hi]

    if tail_neg.size == 0 or tail_pos.size == 0:
        return np.nan

    den = abs(float(np.mean(tail_neg)))
    if den <= 0:
        return np.inf
    return float(np.mean(tail_pos) / den)


# ============================================================
# 🧠 Ottimizzazione pesi di portafoglio (estesa)
# ============================================================
def optimize_weights(
    rets_assets: pd.DataFrame,
    rf_annual: float,
    objective: str = "max_sharpe",
    bounds: tuple[float, float] = (0.0, 1.0),
    sector_map: dict | None = None,
    sector_caps: dict | None = None,
    tail_alpha: float = 0.05,          # usato per max_rachev
    omega_tau: float = 0.0,            # usato per max_omega0 (tau configurabile)
    n_starts: int = 7,                 # multi-start per obiettivi non convessi
) -> pd.Series:
    """
    Obiettivi supportati:
      - max_sharpe
      - max_sortino
      - min_cvar95
      - min_cvar99
      - min_vol
      - max_return
      - max_cagr
      - min_drawdown
      - max_calmar
      - min_ulcer
      - max_omega0
      - max_rachev
    """

    if not _HAS_SCIPY:
        raise RuntimeError("SciPy non disponibile: impossibile eseguire l'ottimizzazione.")

    rets_assets = _prep_rets(rets_assets)
    if rets_assets.empty or rets_assets.shape[1] == 0:
        raise ValueError("rets_assets vuoto dopo pulizia.")

    n = rets_assets.shape[1]
    tickers = list(rets_assets.columns)

    # vincolo somma pesi = 1
    cons = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]

    # vincoli settoriali opzionali
    if sector_map and sector_caps:
        sectors = set(sector_map.values())
        for s in sectors:
            idx = [i for i, t in enumerate(tickers) if sector_map.get(t, None) == s]
            cap = float(sector_caps.get(s, 1.0))
            if idx:
                cons.append({"type": "ineq", "fun": lambda w, idx=idx, cap=cap: cap - np.sum(w[idx])})

    bnds = tuple([bounds] * n)
    obj_name = str(objective).strip().lower()

    # -------- obiettivo --------
    if obj_name == "max_sharpe":
        def obj(w):
            mu, sig, rf = _annualized_mean_std(w, rets_assets, rf_annual)
            if not np.isfinite(sig) or sig <= 0:
                return 1e6
            return -((mu - rf) / sig)

    elif obj_name == "max_sortino":
        def obj(w):
            mu, sig, rf = _annualized_mean_std(w, rets_assets, rf_annual)
            dd = _downside_dev_ann(w, rets_assets, rf_annual)
            if not np.isfinite(dd) or dd <= 0:
                return 1e6
            return -((mu - rf) / dd)

    elif obj_name == "min_cvar95":
        def obj(w):
            return _es_objective(w, rets_assets, alpha=0.95)

    elif obj_name == "min_cvar99":
        def obj(w):
            return _es_objective(w, rets_assets, alpha=0.99)

    elif obj_name == "min_vol":
        def obj(w):
            _, sig, _ = _annualized_mean_std(w, rets_assets, rf_annual)
            return float(sig) if np.isfinite(sig) else 1e6

    elif obj_name == "max_return":
        def obj(w):
            mu, _, _ = _annualized_mean_std(w, rets_assets, rf_annual)
            return -float(mu) if np.isfinite(mu) else 1e6

    elif obj_name == "max_cagr":
        def obj(w):
            rp = _portfolio_returns(w, rets_assets)
            c = _cagr_from_rp(rp)
            return -float(c) if np.isfinite(c) else 1e6

    elif obj_name == "min_drawdown":
        def obj(w):
            rp = _portfolio_returns(w, rets_assets)
            mdd = _max_drawdown_from_rp(rp)  # negativo
            if not np.isfinite(mdd):
                return 1e6
            return abs(float(mdd))  # minimizza magnitudine

    elif obj_name == "max_calmar":
        def obj(w):
            rp = _portfolio_returns(w, rets_assets)
            c = _cagr_from_rp(rp)
            mdd = _max_drawdown_from_rp(rp)
            if not np.isfinite(c) or not np.isfinite(mdd) or mdd >= 0:
                return 1e6
            den = abs(float(mdd))
            if den <= 1e-12:
                return 1e6
            calmar = float(c) / den
            return -calmar if np.isfinite(calmar) else 1e6

    elif obj_name == "min_ulcer":
        def obj(w):
            rp = _portfolio_returns(w, rets_assets)
            ui = _ulcer_index_from_rp(rp)
            return float(ui) if np.isfinite(ui) else 1e6

    elif obj_name == "max_omega0":
        def obj(w):
            rp = _portfolio_returns(w, rets_assets)
            om = _omega_ratio(rp, tau=float(omega_tau))
            # omega può essere inf -> ottimo: ritorna -inf "clippato"
            if om == np.inf:
                return -1e9
            return -float(om) if np.isfinite(om) else 1e6

    elif obj_name == "max_rachev":
        def obj(w):
            rp = _portfolio_returns(w, rets_assets)
            rr = _rachev_ratio(rp, alpha=float(tail_alpha))
            if rr == np.inf:
                return -1e9
            return -float(rr) if np.isfinite(rr) else 1e6

    else:
        raise ValueError(f"Obiettivo non riconosciuto: {objective!r}")

    # -------- multi-start --------
    n_starts = int(max(1, n_starts))
    starts: List[np.ndarray] = []

    # start 1: equal weight
    starts.append(np.full(n, 1.0 / n, dtype=float))

    # altri: dirichlet random
    rng = np.random.default_rng(42)
    for _ in range(n_starts - 1):
        w = rng.dirichlet(np.ones(n))
        starts.append(w.astype(float))

    best = None
    best_fun = np.inf

    for w0 in starts:
        res = minimize(
            obj,
            w0,
            method="SLSQP",
            bounds=bnds,
            constraints=cons,
            options={"maxiter": 1200, "ftol": 1e-9, "disp": False},
        )
        if res.success and np.isfinite(res.fun) and res.fun < best_fun:
            best_fun = float(res.fun)
            best = res.x.copy()

    if best is None:
        warnings.warn("Ottimizzazione non convergente: uso equal-weight.")
        best = np.full(n, 1.0 / n)

    w_opt = np.maximum(best, 0.0)
    s = float(w_opt.sum())
    if s <= 0:
        w_opt = np.full(n, 1.0 / n)
    else:
        w_opt = w_opt / s

    return pd.Series(w_opt, index=tickers)

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple

# -------------------------------------------------------------------
# Multi-objective optimization (weighted sum + optional constraints)
# -------------------------------------------------------------------

def _metric_direction(name: str) -> str:
    """
    'max' = più alto è meglio
    'min' = più basso è meglio
    """
    name = name.lower().strip()
    if name in ("min_vol", "min_cvar95", "min_cvar99", "min_drawdown", "min_ulcer"):
        return "min"
    return "max"


def _safe_float(x: float, bad: float) -> float:
    try:
        if x is None or not np.isfinite(x):
            return bad
        return float(x)
    except Exception:
        return bad


def _metric_value(
    name: str,
    w: np.ndarray,
    rets_assets: pd.DataFrame,
    rf_annual: float,
    tail_alpha: float = 0.05,
    omega_tau: float = 0.0,
) -> float:
    """
    Restituisce il valore della metrica per il portafoglio pesato w.
    Richiede che nel file esistano i helper usati sotto (se mancano, sostituisci con i tuoi).
    """
    name = name.lower().strip()
    rp = (rets_assets.values @ np.asarray(w, dtype=float))
    rp = np.asarray(rp, dtype=float)

    # --- basic
    if name == "max_sharpe":
        mu, sig, rf = _annualized_mean_std(w, rets_assets, rf_annual)
        return (mu - rf) / sig if sig and np.isfinite(sig) else np.nan

    if name == "max_sortino":
        mu, sig, rf = _annualized_mean_std(w, rets_assets, rf_annual)
        dd = _downside_dev_ann(w, rets_assets, rf_annual)
        return (mu - rf) / dd if dd and np.isfinite(dd) else np.nan

    if name == "min_cvar95":
        return -_es_objective(w, rets_assets, alpha=0.95)  # negato -> "max" friendly (vedi sotto)
    if name == "min_cvar99":
        return -_es_objective(w, rets_assets, alpha=0.99)

    if name == "min_vol":
        _, sig, _ = _annualized_mean_std(w, rets_assets, rf_annual)
        return -sig  # negato -> "max" friendly

    if name == "max_return":
        mu, _, _ = _annualized_mean_std(w, rets_assets, rf_annual)
        return mu

    # --- path dependent (se hai già questi helper dal mio file esteso, ok)
    if name == "max_cagr":
        return _cagr_from_rp(rp)
    if name == "min_drawdown":
        mdd = _max_drawdown_from_rp(rp)     # negativo
        return -abs(mdd)                    # "max" friendly
    if name == "max_calmar":
        c = _cagr_from_rp(rp)
        mdd = _max_drawdown_from_rp(rp)
        if not np.isfinite(c) or not np.isfinite(mdd) or mdd >= 0:
            return np.nan
        den = abs(mdd)
        return c / den if den > 1e-12 else np.nan
    if name == "min_ulcer":
        ui = _ulcer_index_from_rp(rp)
        return -ui  # "max" friendly
    if name == "max_omega0":
        om = _omega_ratio(rp, tau=float(omega_tau))
        return om
    if name == "max_rachev":
        rr = _rachev_ratio(rp, alpha=float(tail_alpha))
        return rr

    raise ValueError(f"Metrica non supportata: {name}")


def _estimate_scales(
    metrics: List[str],
    rets_assets: pd.DataFrame,
    rf_annual: float,
    tail_alpha: float,
    omega_tau: float,
    n_samples: int = 200,
    seed: int = 42,
) -> Dict[str, float]:
    """
    Stima una scala robusta per ogni metrica per normalizzare il composite score.
    """
    rng = np.random.default_rng(seed)
    n = rets_assets.shape[1]
    vals: Dict[str, List[float]] = {m: [] for m in metrics}

    for _ in range(int(max(50, n_samples))):
        w = rng.dirichlet(np.ones(n))
        for m in metrics:
            try:
                v = _metric_value(m, w, rets_assets, rf_annual, tail_alpha, omega_tau)
                if np.isfinite(v):
                    vals[m].append(float(v))
            except Exception:
                pass

    scales: Dict[str, float] = {}
    for m, arr in vals.items():
        if len(arr) < 10:
            scales[m] = 1.0
            continue
        a = np.asarray(arr, dtype=float)
        med = np.median(a)
        mad = np.median(np.abs(a - med))
        # scala robusta: MAD -> ~std
        s = float(mad * 1.4826)
        scales[m] = s if np.isfinite(s) and s > 1e-9 else float(np.std(a) if np.std(a) > 1e-9 else 1.0)

    return scales


def optimize_weights_multi(
    rets_assets: pd.DataFrame,
    rf_annual: float,
    metrics: List[str],
    metric_weights: Optional[Dict[str, float]] = None,
    bounds: Tuple[float, float] = (0.0, 1.0),
    sector_map: dict | None = None,
    sector_caps: dict | None = None,
    tail_alpha: float = 0.05,
    omega_tau: float = 0.0,
    constraints: Optional[Dict[str, Tuple[str, float]]] = None,
    n_starts: int = 10,
    scale_samples: int = 250,
) -> pd.Series:
    """
    Multi-obiettivo via weighted sum (con normalizzazione robusta),
    con vincoli opzionali sulle metriche.

    constraints esempio:
      {
        "min_drawdown": (">=", -0.25),   # attenzione: questa metrica è "max friendly" -> usa valore reale desiderato coerente
        "min_vol": ("<=", 0.18),
        "max_sharpe": (">=", 1.0),
      }
    """

    if not _HAS_SCIPY:
        raise RuntimeError("SciPy non disponibile: impossibile eseguire l'ottimizzazione.")

    rets_assets = _prep_rets(rets_assets)
    if rets_assets.empty or rets_assets.shape[1] == 0:
        raise ValueError("rets_assets vuoto dopo pulizia.")

    metrics = [m.strip().lower() for m in (metrics or []) if str(m).strip()]
    metrics = list(dict.fromkeys(metrics))  # unique preserve order
    if not metrics:
        raise ValueError("Seleziona almeno 1 metrica.")

    n = rets_assets.shape[1]
    tickers = list(rets_assets.columns)

    # weights per metriche
    if metric_weights is None:
        metric_weights = {m: 1.0 for m in metrics}
    else:
        metric_weights = {str(k).lower().strip(): float(v) for k, v in metric_weights.items() if str(k).strip()}
        for m in metrics:
            metric_weights.setdefault(m, 1.0)

    # normalizza i pesi metrica
    sw = sum(abs(metric_weights[m]) for m in metrics)
    if sw <= 0:
        metric_weights = {m: 1.0 for m in metrics}
        sw = float(len(metrics))
    metric_weights = {m: float(metric_weights[m]) / sw for m in metrics}

    # stima scale per standardizzare (Sharpe vs CAGR etc.)
    scales = _estimate_scales(metrics, rets_assets, rf_annual, tail_alpha, omega_tau, n_samples=scale_samples)

    # vincoli: somma pesi = 1
    cons = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]

    # vincoli settoriali
    if sector_map and sector_caps:
        sectors = set(sector_map.values())
        for s in sectors:
            idx = [i for i, t in enumerate(tickers) if sector_map.get(t, None) == s]
            cap = float(sector_caps.get(s, 1.0))
            if idx:
                cons.append({"type": "ineq", "fun": lambda w, idx=idx, cap=cap: cap - np.sum(w[idx])})

    # vincoli metriche (epsilon constraint)
    if constraints:
        for m, (op, thr) in constraints.items():
            m = str(m).lower().strip()
            op = str(op).strip()
            thr = float(thr)

            def _g(w, m=m, op=op, thr=thr):
                v = _metric_value(m, w, rets_assets, rf_annual, tail_alpha, omega_tau)
                if not np.isfinite(v):
                    # se invalido: vincolo fallisce
                    return -1e6
                if op == ">=":
                    return float(v - thr)
                if op == "<=":
                    return float(thr - v)
                # default >=
                return float(v - thr)

            cons.append({"type": "ineq", "fun": _g})

    bnds = tuple([bounds] * n)

    # objective: massimizza somma pesata delle metriche standardizzate -> minimizziamo negativo
    def obj(w):
        score = 0.0
        for m in metrics:
            v = _metric_value(m, w, rets_assets, rf_annual, tail_alpha, omega_tau)
            if not np.isfinite(v):
                return 1e6
            sc = float(scales.get(m, 1.0))
            sc = sc if sc > 1e-12 else 1.0
            # standardizzazione “scale-only” (no mean), sufficiente per evitare che una metrica domini per unità
            z = float(v) / sc
            score += float(metric_weights[m]) * z
        return -score

    # multi-start
    rng = np.random.default_rng(42)
    starts = [np.full(n, 1.0 / n)]
    for _ in range(max(1, int(n_starts)) - 1):
        starts.append(rng.dirichlet(np.ones(n)))

    best_x = None
    best_fun = np.inf

    for w0 in starts:
        res = minimize(
            obj,
            w0,
            method="SLSQP",
            bounds=bnds,
            constraints=cons,
            options={"maxiter": 1500, "ftol": 1e-9, "disp": False},
        )
        if res.success and np.isfinite(res.fun) and res.fun < best_fun:
            best_fun = float(res.fun)
            best_x = res.x.copy()

    if best_x is None:
        warnings.warn("Ottimizzazione multi non convergente: uso equal-weight.")
        best_x = np.full(n, 1.0 / n)

    w_opt = np.maximum(best_x, 0.0)
    s = float(w_opt.sum())
    if s <= 0:
        w_opt = np.full(n, 1.0 / n)
    else:
        w_opt = w_opt / s

    return pd.Series(w_opt, index=tickers)

