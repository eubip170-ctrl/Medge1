# core/portfolio_analisi_multi.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import pandas as pd

from core.portfolio_analisi import AnalysisPack, compute_analysis_pack, normalize_series_daily


@dataclass
class MultiAnalysisPack:
    """
    - base100: dataframe per grafico performance con PIÙ portafogli + BENCH/COMPARE (se presenti)
    - packs: pack per singolo portafoglio (dettagli, metriche, corr, insights ecc.)
    - errors: eventuali problemi per singolo portafoglio (equity vuota, ecc.)
    """
    base100: pd.DataFrame
    packs: Dict[str, AnalysisPack]
    errors: Dict[str, str]


def compute_multi_analysis_pack(
    *,
    equity_map: Dict[str, pd.Series],
    bench_prices: Optional[pd.Series] = None,
    compare_prices: Optional[pd.DataFrame] = None,
    weights_map: Optional[Dict[str, pd.Series]] = None,
) -> MultiAnalysisPack:
    packs: Dict[str, AnalysisPack] = {}
    errors: Dict[str, str] = {}

    port_lines = []
    other_panel: Optional[pd.DataFrame] = None

    weights_map = weights_map or {}

    for name, eq in (equity_map or {}).items():
        eqn = normalize_series_daily(eq).dropna()
        if eqn.empty:
            errors[name] = "equity empty"
            continue

        w = weights_map.get(name)
        pack = compute_analysis_pack(
            equity_portfolio=eqn,
            bench_prices=bench_prices,
            compare_prices=compare_prices,
            weights=w,
        )
        packs[name] = pack

        # linea PORT di questo portafoglio
        if isinstance(pack.base100, pd.DataFrame) and not pack.base100.empty and "PORT" in pack.base100.columns:
            port_lines.append(pack.base100["PORT"].rename(str(name)))

        # prendi BENCH/COMPARE una sola volta (dal primo pack utile)
        if other_panel is None and isinstance(pack.base100, pd.DataFrame) and not pack.base100.empty:
            other_panel = pack.base100.drop(columns=["PORT"], errors="ignore")

    base100 = pd.DataFrame()
    if port_lines:
        base100 = pd.concat(port_lines, axis=1).sort_index()

    if other_panel is not None and not other_panel.empty:
        base100 = pd.concat([base100, other_panel], axis=1) if not base100.empty else other_panel.copy()

    if not base100.empty:
        base100 = base100.sort_index().ffill().dropna(how="all")

    return MultiAnalysisPack(base100=base100, packs=packs, errors=errors)
