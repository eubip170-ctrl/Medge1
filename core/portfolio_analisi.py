# core/portfolio_analisi.py
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

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

# =============================================================================
# Time-index safety helpers
# =============================================================================
def _to_naive_daily_index(idx: pd.Index) -> pd.DatetimeIndex:
    dt = pd.to_datetime(idx, errors="coerce")

    # tz -> naive safely
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


def normalize_series_daily(s: pd.Series) -> pd.Series:
    if s is None or len(s) == 0:
        return pd.Series(dtype=float)
    out = pd.Series(s).dropna().astype(float).copy()
    out.index = _to_naive_daily_index(out.index)
    out = out[~out.index.duplicated(keep="last")]
    return out.sort_index()


def normalize_df_daily(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    out = df.copy()
    out.index = _to_naive_daily_index(out.index)
    out = out[~out.index.duplicated(keep="last")]
    return out.sort_index()


# =============================================================================
# Metrics helpers
# =============================================================================
def _safe_float(x: Any) -> float:
    try:
        v = float(x)
        return v if np.isfinite(v) else np.nan
    except Exception:
        return np.nan


def _max_drawdown_from_equity(eq: pd.Series) -> float:
    eq = normalize_series_daily(eq)
    if len(eq) < 2:
        return np.nan
    dd = drawdown_series(eq)
    try:
        return float(dd.min())
    except Exception:
        return np.nan


def _base100_from_prices(px: pd.Series) -> pd.Series:
    px = normalize_series_daily(px).dropna()
    if px.empty:
        return px
    base = float(px.iloc[0])
    if not np.isfinite(base) or base == 0:
        return pd.Series(dtype=float)
    return (px / base) * 100.0


def _beta_alpha_r2(port_rets: pd.Series, bench_rets: pd.Series) -> Tuple[float, float, float]:
    """
    Beta = cov(p,b)/var(b)
    Alpha_ann = (mean_p - beta*mean_b)*252
    R2 = corr^2
    """
    p = pd.Series(port_rets).dropna().astype(float)
    b = pd.Series(bench_rets).dropna().astype(float)
    if p.empty or b.empty:
        return np.nan, np.nan, np.nan

    idx = p.index.intersection(b.index)
    if len(idx) < 2:
        return np.nan, np.nan, np.nan

    p = p.loc[idx]
    b = b.loc[idx]

    vb = np.var(b.values, ddof=1)
    if not np.isfinite(vb) or vb <= 0:
        return np.nan, np.nan, np.nan

    cov = np.cov(p.values, b.values, ddof=1)[0, 1]
    beta = cov / vb

    mp = float(np.mean(p.values))
    mb = float(np.mean(b.values))
    alpha_ann = (mp - beta * mb) * 252.0

    corr = np.corrcoef(p.values, b.values)[0, 1]
    r2 = float(corr * corr) if np.isfinite(corr) else np.nan
    return float(beta), float(alpha_ann), r2


def _tracking_error_ir(port_rets: pd.Series, bench_rets: pd.Series) -> Tuple[float, float]:
    p = pd.Series(port_rets).dropna().astype(float)
    b = pd.Series(bench_rets).dropna().astype(float)
    if p.empty or b.empty:
        return np.nan, np.nan

    idx = p.index.intersection(b.index)
    if len(idx) < 2:
        return np.nan, np.nan

    diff = (p.loc[idx] - b.loc[idx]).dropna()
    if len(diff) < 2:
        return np.nan, np.nan

    te = float(np.std(diff.values, ddof=1) * np.sqrt(252.0))
    ir = float((np.mean(diff.values) * 252.0) / te) if (np.isfinite(te) and te > 0) else np.nan
    return te, ir


def _downside_capture(port_rets: pd.Series, bench_rets: pd.Series) -> float:
    """Downside capture (%) using days where bench < 0."""
    p = pd.Series(port_rets).dropna().astype(float)
    b = pd.Series(bench_rets).dropna().astype(float)
    idx = p.index.intersection(b.index)
    if len(idx) < 10:
        return np.nan

    p = p.loc[idx]
    b = b.loc[idx]

    mask = b < 0
    if int(mask.sum()) < 5:
        return np.nan

    mb = float(np.mean(b[mask].values))
    mp = float(np.mean(p[mask].values))
    if mb == 0:
        return np.nan
    return float((mp / mb) * 100.0)


def _corr_beta(port_rets: pd.Series, proxy_rets: pd.Series) -> Tuple[float, float]:
    """
    Return (corr, beta) of portfolio vs proxy.
    beta = cov(p, x) / var(x)
    """
    p = pd.Series(port_rets).dropna().astype(float)
    x = pd.Series(proxy_rets).dropna().astype(float)
    if p.empty or x.empty:
        return np.nan, np.nan
    idx = p.index.intersection(x.index)
    if len(idx) < 20:
        return np.nan, np.nan
    p = p.loc[idx]
    x = x.loc[idx]

    vx = np.var(x.values, ddof=1)
    if not np.isfinite(vx) or vx <= 0:
        beta = np.nan
    else:
        cov = np.cov(p.values, x.values, ddof=1)[0, 1]
        beta = float(cov / vx)

    corr = np.corrcoef(p.values, x.values)[0, 1]
    corr = float(corr) if np.isfinite(corr) else np.nan
    return corr, beta


def _normalize_weights(weights: Optional[pd.Series]) -> pd.Series:
    if not isinstance(weights, pd.Series) or weights.empty:
        return pd.Series(dtype=float)
    w = pd.to_numeric(weights, errors="coerce").dropna().astype(float)
    w.index = w.index.astype(str).str.upper()
    w = w[w > 0]
    if w.empty:
        return pd.Series(dtype=float)
    s = float(w.sum())
    if not np.isfinite(s) or s <= 0:
        return pd.Series(dtype=float)
    return (w / s).sort_values(ascending=False)


def _standardize_meta(meta: Optional[pd.DataFrame]) -> pd.DataFrame:
    """
    Best effort standardization.
    Expected output columns (optional): ticker, country, region, sector
    """
    if not isinstance(meta, pd.DataFrame) or meta.empty:
        return pd.DataFrame()

    df = meta.copy()

    # ticker source: column or index
    cols_low = [str(c).lower().strip() for c in df.columns]
    if "ticker" not in cols_low and "symbol" not in cols_low:
        if df.index.dtype == object:
            df = df.reset_index().rename(columns={"index": "ticker"})
        else:
            return pd.DataFrame()

    # rename known variants
    rename = {}
    for c in df.columns:
        cl = str(c).lower().strip()
        if cl in ("ticker", "symbol", "ric"):
            rename[c] = "ticker"
        elif cl in ("country", "nation"):
            rename[c] = "country"
        elif cl in ("region",):
            rename[c] = "region"
        elif cl in ("sector", "gics_sector", "industry_sector"):
            rename[c] = "sector"
    df = df.rename(columns=rename)

    if "ticker" not in df.columns:
        return pd.DataFrame()

    df["ticker"] = df["ticker"].astype(str).str.upper().str.strip()
    df = df.drop_duplicates(subset=["ticker"], keep="last")

    for c in ("country", "region", "sector"):
        if c in df.columns:
            df[c] = df[c].astype(str).str.strip()

    return df


# =============================================================================
# Defaults for geo/sector proxies (you can override via maps)
# =============================================================================
DEFAULT_GEO_INDEX_MAP: Dict[str, str] = {
    "USA": "SPY",
    "US": "SPY",
    "United States": "SPY",
    "United Kingdom": "EWU",
    "UK": "EWU",
    "Germany": "EWG",
    "France": "EWQ",
    "Italy": "EWI",
    "Japan": "EWJ",
    "China": "MCHI",
    "India": "INDA",
    "Canada": "EWC",
    "Brazil": "EWZ",
    "Australia": "EWA",
}

DEFAULT_SECTOR_PROXY_MAP: Dict[str, str] = {
    "Technology": "XLK",
    "Financials": "XLF",
    "Healthcare": "XLV",
    "Energy": "XLE",
    "Consumer Discretionary": "XLY",
    "Consumer Staples": "XLP",
    "Industrials": "XLI",
    "Materials": "XLB",
    "Utilities": "XLU",
    "Real Estate": "XLRE",
    "Communication Services": "XLC",
}


# =============================================================================
# Pack
# =============================================================================
@dataclass
class AnalysisPack:
    # existing
    base100: pd.DataFrame
    compare_summary: pd.DataFrame
    corr_matrix: pd.DataFrame
    corr_to_port: pd.Series
    diversifiers: pd.Series
    peers: pd.Series
    insights: Dict[str, Any]

    # new (dashboard style)
    geo_exposure: pd.DataFrame = field(default_factory=pd.DataFrame)
    indices_correlation: pd.DataFrame = field(default_factory=pd.DataFrame)
    sector_allocation: pd.DataFrame = field(default_factory=pd.DataFrame)
    sector_analysis: pd.DataFrame = field(default_factory=pd.DataFrame)


# =============================================================================
# Main
# =============================================================================
def compute_analysis_pack(
    *,
    equity_portfolio: pd.Series,
    bench_prices: Optional[pd.Series] = None,
    compare_prices: Optional[pd.DataFrame] = None,
    weights: Optional[pd.Series] = None,
    holdings_meta: Optional[pd.DataFrame] = None,
    proxy_prices: Optional[pd.DataFrame] = None,
    geo_index_map: Optional[Dict[str, str]] = None,
    sector_proxy_map: Optional[Dict[str, str]] = None,
    min_weight_pct: float = 0.50,
) -> AnalysisPack:
    """
    Output pack for Analysis UI.

    Required:
    - equity_portfolio

    Optional (for the “mock-like” exposure sections):
    - weights: Series(ticker->weight)
    - holdings_meta: DataFrame with columns at least ticker, and optionally country/region/sector
    - proxy_prices: DataFrame of prices for proxies (indices/ETFs) referenced by geo_index_map/sector_proxy_map

    If optional inputs missing, those sections return empty DataFrames (safe).
    """

    eq = normalize_series_daily(equity_portfolio).dropna()
    if eq.empty:
        return AnalysisPack(
            base100=pd.DataFrame(),
            compare_summary=pd.DataFrame(),
            corr_matrix=pd.DataFrame(),
            corr_to_port=pd.Series(dtype=float),
            diversifiers=pd.Series(dtype=float),
            peers=pd.Series(dtype=float),
            insights={"error": "equity empty"},
        )

    # ---------------------------------------------------------------------
    # Price panel for chart + metrics (PORT + BENCH + compare)
    # ---------------------------------------------------------------------
    panel: Dict[str, pd.Series] = {"PORT": eq}

    if bench_prices is not None and len(bench_prices) > 0:
        bp = normalize_series_daily(bench_prices).dropna()
        if not bp.empty:
            panel["BENCH"] = bp

    if isinstance(compare_prices, pd.DataFrame) and not compare_prices.empty:
        cp = normalize_df_daily(compare_prices)
        for c in cp.columns:
            s = pd.to_numeric(cp[c], errors="coerce").dropna()
            if not s.empty:
                panel[str(c).upper()] = s

    px_df = pd.DataFrame(panel).sort_index().ffill().dropna(how="all")

    base100 = pd.DataFrame({c: _base100_from_prices(px_df[c]) for c in px_df.columns}).dropna(how="all")

    rets_df = px_df.pct_change().dropna(how="all")
    port_rets = rets_df["PORT"].dropna()
    bench_rets = rets_df["BENCH"].dropna() if "BENCH" in rets_df.columns else pd.Series(dtype=float)

    # ---------------------------------------------------------------------
    # Compare summary (advanced metrics)
    # ---------------------------------------------------------------------
    rows: List[Dict[str, Any]] = []
    for col in rets_df.columns:
        r = rets_df[col].dropna()
        if r.empty:
            continue

        eq_tmp = (1.0 + r).cumprod()
        mdd = _max_drawdown_from_equity(eq_tmp)

        row: Dict[str, Any] = {
            "TICKER": str(col).upper(),
            "CAGR": _safe_float(cagr_from_returns(r)),
            "VOL": _safe_float(vol_ann(r)),
            "SHARPE": _safe_float(sharpe_ratio(r, rf_annual=0.0)),
            "SORTINO": _safe_float(sortino_ratio(r, rf_annual=0.0)),
            "MAXDD": _safe_float(mdd),
        }

        if not bench_rets.empty and col != "BENCH":
            beta, alpha_ann, r2 = _beta_alpha_r2(r, bench_rets)
            te, ir = _tracking_error_ir(r, bench_rets)
            dcap = _downside_capture(r, bench_rets)
            row.update({"BETA": beta, "ALPHA": alpha_ann, "R2": r2, "TE": te, "IR": ir, "DOWNSIDE_CAP": dcap})
        else:
            row.update({"BETA": np.nan, "ALPHA": np.nan, "R2": np.nan, "TE": np.nan, "IR": np.nan, "DOWNSIDE_CAP": np.nan})

        rows.append(row)

    compare_summary = pd.DataFrame(rows)
    if not compare_summary.empty:
        order = ["PORT", "BENCH"] + [c for c in compare_summary["TICKER"].tolist() if c not in ("PORT", "BENCH")]
        compare_summary["__ord"] = compare_summary["TICKER"].map({k: i for i, k in enumerate(order)}).fillna(9999)
        compare_summary = compare_summary.sort_values("__ord").drop(columns=["__ord"])

    # ---------------------------------------------------------------------
    # Correlation matrix (returns)
    # ---------------------------------------------------------------------
    corr_matrix = rets_df.corr().replace([np.inf, -np.inf], np.nan).fillna(0.0)
    corr_to_port = corr_matrix.get("PORT", pd.Series(dtype=float)).drop(labels=["PORT"], errors="ignore")

    diversifiers = corr_to_port.sort_values().head(3) if not corr_to_port.empty else pd.Series(dtype=float)
    peers = corr_to_port.sort_values(ascending=False).head(3) if not corr_to_port.empty else pd.Series(dtype=float)

    # ---------------------------------------------------------------------
    # Insights
    # ---------------------------------------------------------------------
    insights: Dict[str, Any] = {}
    w_norm = _normalize_weights(weights)

    if not w_norm.empty:
        insights["concentration_hhi"] = float((w_norm**2).sum())
        insights["top3_weight_pct"] = float(w_norm.head(3).sum() * 100.0)

    # strengths/weaknesses quick (PORT vs BENCH)
    if not compare_summary.empty and "BENCH" in compare_summary["TICKER"].values:
        try:
            P = compare_summary[compare_summary["TICKER"] == "PORT"].iloc[0]
            B = compare_summary[compare_summary["TICKER"] == "BENCH"].iloc[0]
            strengths: List[str] = []
            weaknesses: List[str] = []

            if np.isfinite(P["CAGR"]) and np.isfinite(B["CAGR"]) and P["CAGR"] > B["CAGR"]:
                strengths.append(f"Above-benchmark CAGR ({(P['CAGR']-B['CAGR'])*100:+.1f}%)")
            if np.isfinite(P["SHARPE"]) and np.isfinite(B["SHARPE"]) and P["SHARPE"] > B["SHARPE"]:
                strengths.append(f"Superior Sharpe ({P['SHARPE']:.2f} vs {B['SHARPE']:.2f})")
            if np.isfinite(P["MAXDD"]) and np.isfinite(B["MAXDD"]) and P["MAXDD"] > B["MAXDD"]:
                strengths.append(f"Shallower drawdown ({P['MAXDD']*100:.1f}% vs {B['MAXDD']*100:.1f}%)")

            if np.isfinite(P.get("BETA", np.nan)) and P.get("BETA", np.nan) > 1.05:
                weaknesses.append(f"High beta exposure ({P['BETA']:.2f})")
            if insights.get("concentration_hhi", 0.0) > 0.15:
                weaknesses.append(f"Elevated concentration risk (HHI {insights['concentration_hhi']:.2f}, Top3 {insights.get('top3_weight_pct', np.nan):.1f}%)")
            if np.isfinite(P.get("DOWNSIDE_CAP", np.nan)) and P.get("DOWNSIDE_CAP", np.nan) > 100:
                weaknesses.append(f"Downside capture >100% ({P['DOWNSIDE_CAP']:.0f}%)")

            insights["strengths"] = strengths[:4]
            insights["weaknesses"] = weaknesses[:4]
        except Exception:
            pass

    # ---------------------------------------------------------------------
    # NEW: Geographic exposure + indices correlation
    # ---------------------------------------------------------------------
    geo_exposure = pd.DataFrame()
    indices_correlation = pd.DataFrame()

    meta = _standardize_meta(holdings_meta)
    if not w_norm.empty and not meta.empty and ("country" in meta.columns or "region" in meta.columns):
        wdf = pd.DataFrame({"ticker": w_norm.index, "w": w_norm.values})
        dfm = wdf.merge(meta, on="ticker", how="left")

        geo_col = "country" if "country" in dfm.columns else "region"
        dfm[geo_col] = dfm[geo_col].fillna("Unknown")

        g = dfm.groupby(geo_col)["w"].sum().sort_values(ascending=False) * 100.0
        geo_exposure = pd.DataFrame({"Country": g.index.astype(str), "W%": g.values})
        geo_exposure = geo_exposure[geo_exposure["W%"] >= float(min_weight_pct)].reset_index(drop=True)

        # indices correlation (needs proxy_prices + mapping)
        if isinstance(proxy_prices, pd.DataFrame) and not proxy_prices.empty and not port_rets.empty:
            proxies = normalize_df_daily(proxy_prices).ffill().dropna(how="all")
            pret = proxies.pct_change().dropna(how="all")

            gmap = dict(DEFAULT_GEO_INDEX_MAP)
            if isinstance(geo_index_map, dict) and geo_index_map:
                gmap.update({str(k): str(v).upper() for k, v in geo_index_map.items()})

            rows_idx: List[Dict[str, Any]] = []
            for country in geo_exposure["Country"].astype(str).tolist()[:8]:
                px = str(gmap.get(country, "")).upper().strip()
                if px and px in pret.columns:
                    corr, beta = _corr_beta(port_rets, pret[px])
                    rows_idx.append({"Index": px, "Country": country, "Correlation": corr, "Beta": beta})
                else:
                    rows_idx.append({"Index": px if px else "–", "Country": country, "Correlation": np.nan, "Beta": np.nan})

            indices_correlation = pd.DataFrame(rows_idx)

    # ---------------------------------------------------------------------
    # NEW: Sector allocation + sector analysis
    # ---------------------------------------------------------------------
    sector_allocation = pd.DataFrame()
    sector_analysis = pd.DataFrame()

    if not w_norm.empty and not meta.empty and "sector" in meta.columns:
        wdf = pd.DataFrame({"ticker": w_norm.index, "w": w_norm.values})
        dfm = wdf.merge(meta, on="ticker", how="left")
        dfm["sector"] = dfm["sector"].fillna("Unknown")

        s = dfm.groupby("sector")["w"].sum().sort_values(ascending=False) * 100.0
        sector_allocation = pd.DataFrame({"Sector": s.index.astype(str), "W%": s.values})
        sector_allocation = sector_allocation[sector_allocation["W%"] >= float(min_weight_pct)].reset_index(drop=True)

        # sector analysis needs proxy_prices (sector ETFs) OR at least correlations against BENCH if present
        if isinstance(proxy_prices, pd.DataFrame) and not proxy_prices.empty and not port_rets.empty:
            proxies = normalize_df_daily(proxy_prices).ffill().dropna(how="all")
            pret = proxies.pct_change().dropna(how="all")

            smap = dict(DEFAULT_SECTOR_PROXY_MAP)
            if isinstance(sector_proxy_map, dict) and sector_proxy_map:
                smap.update({str(k): str(v).upper() for k, v in sector_proxy_map.items()})

            rows_sec: List[Dict[str, Any]] = []
            for sec in sector_allocation["Sector"].astype(str).tolist()[:12]:
                px = str(smap.get(sec, "")).upper().strip()
                if px and px in pret.columns:
                    corr, _beta = _corr_beta(port_rets, pret[px])
                    rows_sec.append({"Sector": sec, "Performance": "–", "Correlation": corr})
                else:
                    rows_sec.append({"Sector": sec, "Performance": "–", "Correlation": np.nan})

            sector_analysis = pd.DataFrame(rows_sec)

    return AnalysisPack(
        base100=base100,
        compare_summary=compare_summary,
        corr_matrix=corr_matrix,
        corr_to_port=corr_to_port,
        diversifiers=diversifiers,
        peers=peers,
        insights=insights,
        geo_exposure=geo_exposure,
        indices_correlation=indices_correlation,
        sector_allocation=sector_allocation,
        sector_analysis=sector_analysis,
    )
