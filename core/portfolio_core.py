# core/portfolio_core.py
"""
Aggregatore di comodo per mantenere compatibile l'app.

Da qui re-esportiamo le funzioni principali dai moduli:
  - portfolio_data
  - portfolio_metrics
  - portfolio_optimization
  - portfolio_plots
  - portfolio_pipeline
  - portfolio_analytics
"""

from __future__ import annotations

# ✅ Soluzione B: la costante sta qui (core), non in portfolio_data
TRADING_DAYS: int = 252

from .portfolio_data import (
    DataLoader,
    load_ohlcv_from_yf,  # stub: non usa yfinance (solleva errore se chiamata)
    ensure_dataframe,
    clean_prices,
    extract_close,
    pct_returns,
    log_returns,
    portfolio_value_from_prices,
)

from .portfolio_metrics import (
    drawdown_series,
    cagr_from_returns,
    vol_ann,
    downside_deviation_ann,
    ulcer_index_from_equity,
    rachev_ratio,
    lpm,
    kappa_ratio,
    sharpe_ratio,
    sortino_ratio,
    burke_ratio,
    sterling_ratio,
    pain_index,
    pain_ratio,
    es_cvar,
    omega_ratio,
)

from .portfolio_optimization import (
    optimize_weights,
    risk_contributions,
)

from .portfolio_plots import (
    plot_equity,
    plot_drawdown,
    plot_heatmap,
    plot_performance_with_window,
    plot_va_lines,
)

from .portfolio_pipeline import compute_pipeline

from .portfolio_analytics import (
    volume_profile_features,
    compute_orderflow_trend_features,
    EndPeriodConfig,
    EndPeriodProbability,
    EWMAConfig,
    GARCHConfig,
    EVTConfig,
    hitting_probability_gbm,
)

__all__ = [
    # data
    "TRADING_DAYS",
    "DataLoader",
    "load_ohlcv_from_yf",
    "ensure_dataframe",
    "clean_prices",
    "extract_close",
    "pct_returns",
    "log_returns",
    "portfolio_value_from_prices",
    # metrics
    "drawdown_series",
    "cagr_from_returns",
    "vol_ann",
    "downside_deviation_ann",
    "ulcer_index_from_equity",
    "rachev_ratio",
    "lpm",
    "kappa_ratio",
    "sharpe_ratio",
    "sortino_ratio",
    "burke_ratio",
    "sterling_ratio",
    "pain_index",
    "pain_ratio",
    "es_cvar",
    "omega_ratio",
    # optimization
    "optimize_weights",
    "risk_contributions",
    # plots
    "plot_equity",
    "plot_drawdown",
    "plot_heatmap",
    "plot_performance_with_window",
    "plot_va_lines",
    # pipeline
    "compute_pipeline",
    # analytics
    "volume_profile_features",
    "compute_orderflow_trend_features",
    "EndPeriodConfig",
    "EndPeriodProbability",
    "EWMAConfig",
    "GARCHConfig",
    "EVTConfig",
    "hitting_probability_gbm",
]
