from dataclasses import dataclass
import pandas as pd
from typing import Dict, Optional

@dataclass
class PortfolioOverviewData:
    equity_curve: pd.Series
    drawdown_curve: Optional[pd.Series]
    allocation: pd.DataFrame
    top_contributors: pd.DataFrame
    risk_overview: Dict[str, float]
    ai_comment: str
    label_period: str = "Full period"
