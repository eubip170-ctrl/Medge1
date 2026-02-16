# core/portfolio_plots.py

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from .portfolio_metrics import drawdown_series


def plot_equity(equity: pd.Series):
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(equity.index, equity.values, label="Valore Portafoglio")
    ax.set_title("Evoluzione del Valore del Portafoglio")
    ax.set_xlabel("Data")
    ax.set_ylabel("Valore")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def plot_drawdown(equity: pd.Series):
    dd = drawdown_series(equity)
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(dd.index, dd.values)
    ax.set_title("Drawdown")
    ax.set_xlabel("Data")
    ax.set_ylabel("Drawdown (decimali)")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def plot_heatmap(pv: pd.DataFrame, title: str = ""):
    if pv is None or pv.empty:
        return None
    fig, ax = plt.subplots(figsize=(6, 4))
    im = ax.imshow(pv.values, aspect="auto", origin="lower")
    ax.set_xticks(range(len(pv.columns)))
    ax.set_xticklabels([str(c) for c in pv.columns], rotation=45, ha="right")
    ax.set_yticks(range(len(pv.index)))
    ax.set_yticklabels([str(i) for i in pv.index])
    ax.set_title(title)
    ax.set_xlabel("τ (%)")
    ax.set_ylabel("T (giorni)")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    return fig


def plot_performance_with_window(
    ohlcv_full: pd.DataFrame,
    ohlcv_sel: pd.DataFrame,
    tkr: str,
):
    try:
        fig, ax = plt.subplots(figsize=(12, 4))
        ax.plot(ohlcv_full.index, ohlcv_full["close"], label="Close (full)")
        ax.plot(ohlcv_sel.index, ohlcv_sel["close"], label="Close (window)")
        ax.set_title(f"{tkr} — Close (full vs window)")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        return fig
    except Exception:
        return None


def plot_va_lines(ohlcv_sel: pd.DataFrame, vp_feats: dict, tkr: str):
    try:
        vah = vp_feats.get("vah")
        val = vp_feats.get("val")
        poc = vp_feats.get("poc")

        fig, ax = plt.subplots(figsize=(12, 4))
        ax.plot(ohlcv_sel.index, ohlcv_sel["close"], label="Close")
        if poc is not None and np.isfinite(poc):
            ax.axhline(poc, linestyle="--", label="POC")
        if vah is not None and np.isfinite(vah):
            ax.axhline(vah, linestyle=":", label="VAH")
        if val is not None and np.isfinite(val):
            ax.axhline(val, linestyle=":", label="VAL")
        ax.set_title(f"{tkr} — VA lines")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        return fig
    except Exception:
        return None
