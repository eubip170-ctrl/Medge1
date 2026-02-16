from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Literal, Optional, Sequence

import numpy as np
import pandas as pd

from infra.marketdata.marketstack_client import load_ohlcv_from_marketstack

# ====================================================================================
#                            RISK / RETURN HELPERS
# ====================================================================================

RISK_TO_ER: dict[int, float] = {
    1: 0.02, 2: 0.03, 3: 0.04, 4: 0.05, 5: 0.06,
    6: 0.07, 7: 0.08, 8: 0.09, 9: 0.10, 10: 0.12,
}


def risk_label(v: int) -> str:
    mapping = {
        1: "Very Low",  2: "Very Low",
        3: "Low",       4: "Low-Average",
        5: "Medium-",   6: "Medium",
        7: "Medium+",   8: "High",
        9: "Very High", 10: "Extreme",
    }
    v_int = int(max(1, min(10, v)))
    return mapping.get(v_int, f"{v_int}/10")


def expected_return_from_risk(level: int) -> float:
    level = int(max(1, min(10, level)))
    return float(RISK_TO_ER.get(level, 0.06))


# ====================================================================================
#                            DCA SCHEDULE / RULES
# ====================================================================================

@dataclass
class PostPeriodRule:
    enabled: bool = False
    trigger_mode: Literal["N", "date"] | None = None
    N: Optional[int] = None
    date: Optional[date] = None
    action: Literal["freeze", "inc", "dec"] | None = None
    pct: Optional[float] = None


def gen_pac_dates(start_d: date, end_d: date, freq_key: str,
                  max_rows: int = 100_000) -> list[pd.Timestamp]:
    """
    Genera le date PAC tra start e end (inclusi).
    Per frequenze mensili cerca di preservare il giorno del mese.
    """
    dates: list[pd.Timestamp] = []
    cur = pd.Timestamp(start_d)
    end_ts = pd.Timestamp(end_d)
    count = 0

    if freq_key == "1 week":
        step = pd.DateOffset(weeks=1)
    elif freq_key == "2 weeks":
        step = pd.DateOffset(weeks=2)
    elif freq_key in {"1 month", "2 months", "3 months", "6 months"}:
        step_months = {"1 month": 1, "2 months": 2, "3 months": 3, "6 months": 6}[freq_key]
        anchor_day = cur.day
        while cur <= end_ts and count < max_rows:
            dates.append(cur)
            cur = cur + pd.DateOffset(months=step_months)
            try:
                cur = cur.replace(day=anchor_day)
            except ValueError:
                # ultimo giorno valido del mese
                pass
            count += 1
        return dates
    else:
        # fallback mensile
        step = pd.DateOffset(months=1)

    while cur <= end_ts and count < max_rows:
        dates.append(cur)
        cur = cur + step
        count += 1
    return dates


def build_dca_schedule(
    freq: str,
    amount: float,
    amount_type: Literal["Fixed", "Variable"],
    delta_pct: float,
    start_date: date,
    end_date: date,
    rule: Optional[PostPeriodRule] = None,
) -> pd.DataFrame:
    """
    Motore centrale per il piano PAC/DCA.

    Ritorna un DataFrame con:
    - Date
    - Deposit (€)
    - Regime
    - Frequency
    - Amount type
    - (opzionale) Post-period rule
    """
    if amount <= 0:
        raise ValueError("Amount must be > 0.")
    if end_date <= start_date:
        raise ValueError("End date must be later than start date.")

    dates = gen_pac_dates(start_date, end_date, freq)
    if not dates:
        return pd.DataFrame(columns=["Date", "Deposit (€)", "Regime"])

    if amount_type == "Fixed":
        amounts = np.full(len(dates), fill_value=amount, dtype=float)
        regime_labels = ["Fixed"] * len(dates)
    else:
        g1 = 1.0 + float(delta_pct) / 100.0
        n = len(dates)
        amounts = np.zeros(n, dtype=float)
        regime_labels = ["Base"] * n
        for i in range(n):
            amounts[i] = amount * (g1 ** i)

        if rule and rule.enabled:
            if rule.trigger_mode == "N":
                idx_trigger = int(rule.N or 0)
            elif rule.trigger_mode == "date" and rule.date is not None:
                dt_trg = pd.Timestamp(rule.date)
                idxs = [
                    i for i, d in enumerate(dates)
                    if pd.Timestamp(d).normalize() >= dt_trg.normalize()
                ]
                idx_trigger = idxs[0] if idxs else n
            else:
                idx_trigger = n

            n = len(dates)
            idx_trigger = max(0, min(idx_trigger, n))
            if idx_trigger < n:
                base_amt = amounts[idx_trigger - 1] if idx_trigger > 0 else amount
                if rule.action == "freeze":
                    for i in range(idx_trigger, n):
                        amounts[i] = base_amt
                        regime_labels[i] = "Frozen"
                elif rule.action in {"inc", "dec"} and rule.pct is not None:
                    pct2 = float(rule.pct)
                    if rule.action == "inc":
                        g2, tag = 1.0 + pct2 / 100.0, f"+{pct2:.2f}%"
                    else:
                        g2, tag = 1.0 - pct2 / 100.0, f"-{pct2:.2f}%"
                    for k, i in enumerate(range(idx_trigger, n), start=1):
                        amounts[i] = base_amt * (g2 ** k)
                        regime_labels[i] = f"Post-rule ({tag}/period)"

    df = pd.DataFrame({
        "Date": pd.to_datetime(dates).date,
        "Deposit (€)": np.round(amounts, 2),
        "Regime": regime_labels,
    })
    df["Frequency"] = freq
    df["Amount type"] = (
        "Fixed"
        if amount_type == "Fixed"
        else f"Variable (base {delta_pct:+.2f}%/period)"
    )

    if amount_type == "Variable" and rule and rule.enabled:
        if rule.trigger_mode == "N":
            trg_desc = f"after {rule.N} deposits"
        else:
            trg_desc = f"from {rule.date}"
        if rule.action == "freeze":
            rule_desc = f"Rule: freeze amount {trg_desc}"
        elif rule.action == "inc":
            rule_desc = f"Rule: +{rule.pct:.2f}%/period {trg_desc}"
        else:
            rule_desc = f"Rule: -{rule.pct:.2f}%/period {trg_desc}"
        df["Post-period rule"] = rule_desc

    return df


# ====================================================================================
#                  COMPOUND INTEREST (ESTIMATED PATH)
# ====================================================================================

def compound_path(
    dates_list: Sequence[pd.Timestamp | date],
    amounts_list: Sequence[float],
    end_dt: date,
    er_annual: float,
) -> pd.DataFrame:
    """
    Simula la capitalizzazione giornaliera tra i versamenti.
    Ritorna un DataFrame con:
    - Date
    - Contribution (€)
    - Cumulative contributions (€)
    - Estimated value (€)
    - Estimated interest (€)
    """
    if len(dates_list) != len(amounts_list):
        raise ValueError("dates_list and amounts_list must have the same length.")

    idx = [pd.Timestamp(d).normalize() for d in dates_list]
    amt = np.array(list(amounts_list), dtype=float)

    if not len(idx):
        return pd.DataFrame()

    r_day = (1.0 + float(er_annual)) ** (1 / 365.0) - 1.0
    val = 0.0
    cum = 0.0
    rows = []

    prev_date = idx[0]
    for d, a in zip(idx, amt):
        ddays = max(0, (d - prev_date).days)
        if ddays > 0:
            val = val * ((1.0 + r_day) ** ddays)
        val += float(a)
        cum += float(a)
        rows.append([d.date(), a, cum, val, val - cum])
        prev_date = d

    end_ts = pd.Timestamp(end_dt).normalize()
    if prev_date < end_ts:
        ddays = (end_ts - prev_date).days
        if ddays > 0:
            val = val * ((1.0 + r_day) ** ddays)
        rows.append([end_ts.date(), 0.0, cum, val, val - cum])

    out = pd.DataFrame(rows, columns=[
        "Date",
        "Contribution (€)",
        "Cumulative contributions (€)",
        "Estimated value (€)",
        "Estimated interest (€)",
    ])
    for c in out.columns[1:]:
        out[c] = np.round(out[c].astype(float), 2)
    return out


# ====================================================================================
#                       DCA QUESTIONNAIRE TEMPLATES (DEPOSITS)
# ====================================================================================

DCA_TEMPLATES = [
    {
        "name": "Fixed Monthly DCA",
        "desc": "Fixed monthly contribution. Simple and disciplined.",
        "freq": "1 month",
        "amount": 200.0,
        "type": "Fixed",
        "delta_base": 0.0,
        "rule": {"enabled": False},
        "risk": 5,
    },
    {
        "name": "Micro-DCA (Weekly)",
        "desc": "Small contributions every week (reduces timing risk).",
        "freq": "1 week",
        "amount": 60.0,
        "type": "Fixed",
        "delta_base": 0.0,
        "rule": {"enabled": False},
        "risk": 4,
    },
    {
        "name": "Bi-weekly Fixed",
        "desc": "Fixed contribution every 2 weeks.",
        "freq": "2 weeks",
        "amount": 120.0,
        "type": "Fixed",
        "delta_base": 0.0,
        "rule": {"enabled": False},
        "risk": 5,
    },
    {
        "name": "Gradual Growth (+1%/period)",
        "desc": "Variable amount growing +1% per period (compounded).",
        "freq": "1 month",
        "amount": 180.0,
        "type": "Variable",
        "delta_base": 1.0,
        "rule": {"enabled": False},
        "risk": 6,
    },
    {
        "name": "Glidepath Decreasing (-0.5%/period)",
        "desc": "Variable amount gradually decreasing: -0.5% per period.",
        "freq": "1 month",
        "amount": 220.0,
        "type": "Variable",
        "delta_base": -0.5,
        "rule": {"enabled": False},
        "risk": 4,
    },
    {
        "name": "Annual Step-up (+5% after 12 deposits)",
        "desc": "After 12 deposits, increase by +5% per period.",
        "freq": "1 month",
        "amount": 200.0,
        "type": "Variable",
        "delta_base": 0.0,
        "rule": {
            "enabled": True,
            "trigger_mode": "N",
            "N": 12,
            "action": "inc",
            "pct": 5.0,
        },
        "risk": 6,
    },
    {
        "name": "Freeze after 24 deposits",
        "desc": "+1% per period for 24 deposits, then freeze the amount.",
        "freq": "1 month",
        "amount": 180.0,
        "type": "Variable",
        "delta_base": 1.0,
        "rule": {
            "enabled": True,
            "trigger_mode": "N",
            "N": 24,
            "action": "freeze",
        },
        "risk": 5,
    },
    {
        "name": "Contrarian Step-up (+3% after 12 deposits)",
        "desc": "Programmed increase after N deposits (simplified contrarian).",
        "freq": "1 month",
        "amount": 170.0,
        "type": "Variable",
        "delta_base": 0.5,
        "rule": {
            "enabled": True,
            "trigger_mode": "N",
            "N": 12,
            "action": "inc",
            "pct": 3.0,
        },
        "risk": 7,
    },
    {
        "name": "Aggressive Growth (+1.5%/period)",
        "desc": "Constant growth of the amount: +1.5% per period.",
        "freq": "1 month",
        "amount": 220.0,
        "type": "Variable",
        "delta_base": 1.5,
        "rule": {"enabled": False},
        "risk": 8,
    },
    {
        "name": "Defensive Income (Fixed)",
        "desc": "Fixed amount, more defensive profile.",
        "freq": "1 month",
        "amount": 250.0,
        "type": "Fixed",
        "delta_base": 0.0,
        "rule": {"enabled": False},
        "risk": 3,
    },
    {
        "name": "Quarterly Fixed",
        "desc": "Fixed contribution every 3 months (good for bonuses/quarterly income).",
        "freq": "3 months",
        "amount": 600.0,
        "type": "Fixed",
        "delta_base": 0.0,
        "rule": {"enabled": False},
        "risk": 4,
    },
    {
        "name": "Bi-monthly with Freeze",
        "desc": "+0.8% for 18 deposits (every 2 months), then freeze the amount.",
        "freq": "2 months",
        "amount": 200.0,
        "type": "Variable",
        "delta_base": 0.8,
        "rule": {
            "enabled": True,
            "trigger_mode": "N",
            "N": 18,
            "action": "freeze",
        },
        "risk": 5,
    },
]


def build_plan_from_template(tmpl: dict, start_d: date, end_d: date) -> pd.DataFrame:
    """Costruisce il piano partendo da un template DCA_TEMPLATES."""
    rule_cfg = tmpl.get("rule", {}) or {}
    rule = PostPeriodRule(
        enabled=bool(rule_cfg.get("enabled", False)),
        trigger_mode=rule_cfg.get("trigger_mode"),
        N=rule_cfg.get("N"),
        date=rule_cfg.get("date"),
        action=rule_cfg.get("action"),
        pct=rule_cfg.get("pct"),
    )
    return build_dca_schedule(
        freq=tmpl["freq"],
        amount=float(tmpl["amount"]),
        amount_type="Fixed" if tmpl["type"] == "Fixed" else "Variable",
        delta_pct=float(tmpl.get("delta_base", 0.0)),
        start_date=start_d,
        end_date=end_d,
        rule=rule,
    )


# ====================================================================================
#                      TEMPLATE COMPONENTS (ETF PORTFOLIOS)
# ====================================================================================

def _norm_weights(rows: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    if df.empty or "weight" not in df.columns:
        return df
    s = float(df["weight"].sum())
    if s > 0:
        df["weight"] = df["weight"] * (100.0 / s)
    return df


def allocation_table(components: list[dict], capital: float) -> pd.DataFrame:
    df = _norm_weights(components).copy()
    if not df.empty:
        df["Allocation (€)"] = np.round((df["weight"] / 100.0) * float(capital), 2)
        df = df[["ticker", "name", "asset_class", "region", "weight", "Allocation (€)"]]
    return df


def parse_overrides(txt: str) -> dict[str, str]:
    m: dict[str, str] = {}
    for pair in (txt or "").split(";"):
        if "=" in pair:
            a, b = pair.split("=", 1)
            a, b = a.strip(), b.strip()
            if a and b:
                m[a] = b
    return m


def map_symbol(tk: str, ov_map: dict[str, str]) -> Optional[str]:
    if tk in ov_map:
        return ov_map[tk]
    bad = any(x in tk for x in ["(", ")", " "])
    return None if bad else tk


def normalize_base_100(ser: pd.Series) -> pd.Series:
    s = pd.Series(ser).dropna()
    if s.empty:
        return s
    base = float(s.iloc[0])
    if base == 0 or not np.isfinite(base):
        return pd.Series(dtype=float, index=s.index)
    return (s / base) * 100.0


def resample_close(ser: pd.Series, how: str) -> pd.Series:
    s = pd.Series(ser).dropna()
    if s.empty:
        return s
    if how.startswith("Daily"):
        return s.asfreq("B").ffill()
    if "Weekly" in how:
        return s.resample("W-FRI").last().ffill()
    return s.resample("M").last().ffill()


def fetch_prices_marketstack(symbol: str, start_str: str, end_str: str) -> pd.Series:
    """Ritorna la serie dei close giornalieri usando il client Marketstack."""
    dfp = load_ohlcv_from_marketstack(ticker=symbol, start=start_str, end=end_str)
    if isinstance(dfp, pd.DataFrame) and "close" in dfp.columns:
        s = pd.Series(dfp["close"]).dropna()
        s.index = pd.to_datetime(s.index)
        return s.sort_index()
    return pd.Series(dtype=float)


def get_component_templates(universe_label: str) -> dict[str, dict]:
    """
    Template di portafoglio ETF (US / UCITS).
    Qui puoi copiare tutti quelli che avevi nel vecchio script; per brevità ne metto alcuni.
    """
    if "US ETFs" in universe_label:
        return {
            "100% Global Equity (US)": dict(
                desc="Global equity exposure with a single ETF.",
                items=[
                    {
                        "ticker": "VT",
                        "name": "Vanguard Total World",
                        "asset_class": "Equity",
                        "region": "Global",
                        "weight": 100.0,
                    }
                ],
            ),
            "60/40 (US)": dict(
                desc="Classic 60/40.",
                items=[
                    {
                        "ticker": "VTI",
                        "name": "US Total Market",
                        "asset_class": "Equity",
                        "region": "USA",
                        "weight": 36.0,
                    },
                    {
                        "ticker": "VXUS",
                        "name": "Ex-US Total Market",
                        "asset_class": "Equity",
                        "region": "Ex-USA",
                        "weight": 24.0,
                    },
                    {
                        "ticker": "BND",
                        "name": "US Aggregate Bond",
                        "asset_class": "Bond",
                        "region": "USA",
                        "weight": 28.0,
                    },
                    {
                        "ticker": "BNDX",
                        "name": "Intl Agg Bond (Hedged)",
                        "asset_class": "Bond",
                        "region": "Global",
                        "weight": 12.0,
                    },
                ],
            ),
            "All Weather (US, approx)": dict(
                desc="Inspired by All Weather.",
                items=[
                    {
                        "ticker": "SPY",
                        "name": "S&P 500",
                        "asset_class": "Equity",
                        "region": "USA",
                        "weight": 30.0,
                    },
                    {
                        "ticker": "TLT",
                        "name": "US Treasuries 20+Y",
                        "asset_class": "Bond (Long)",
                        "region": "USA",
                        "weight": 40.0,
                    },
                    {
                        "ticker": "IEF",
                        "name": "US Treasuries 7–10Y",
                        "asset_class": "Bond (Int)",
                        "region": "USA",
                        "weight": 15.0,
                    },
                    {
                        "ticker": "GLD",
                        "name": "Gold",
                        "asset_class": "Gold",
                        "region": "Global",
                        "weight": 7.5,
                    },
                    {
                        "ticker": "DBC",
                        "name": "Commodities Broad",
                        "asset_class": "Commodities",
                        "region": "Global",
                        "weight": 7.5,
                    },
                ],
            ),
            "Permanent Portfolio (US)": dict(
                desc="Four pillars (Equities, LT Bonds, Gold, Cash-like).",
                items=[
                    {
                        "ticker": "VTI",
                        "name": "US Total Market",
                        "asset_class": "Equity",
                        "region": "USA",
                        "weight": 25.0,
                    },
                    {
                        "ticker": "TLT",
                        "name": "US Treasuries 20+Y",
                        "asset_class": "Bond (Long)",
                        "region": "USA",
                        "weight": 25.0,
                    },
                    {
                        "ticker": "GLD",
                        "name": "Gold",
                        "asset_class": "Gold",
                        "region": "Global",
                        "weight": 25.0,
                    },
                    {
                        "ticker": "BIL",
                        "name": "US T-Bills 1–3M",
                        "asset_class": "Cash-like",
                        "region": "USA",
                        "weight": 25.0,
                    },
                ],
            ),
        }
    else:
        return {
            "100% Global Equity (UCITS)": dict(
                desc="Global equity exposure (UCITS).",
                items=[
                    {
                        "ticker": "VWCE",
                        "name": "Vanguard FTSE All-World UCITS",
                        "asset_class": "Equity",
                        "region": "Global",
                        "weight": 100.0,
                    }
                ],
            ),
            "60/40 (UCITS)": dict(
                desc="World + EM + global bonds (EUR-hedged).",
                items=[
                    {
                        "ticker": "IWDA",
                        "name": "MSCI World UCITS",
                        "asset_class": "Equity",
                        "region": "DM",
                        "weight": 48.0,
                    },
                    {
                        "ticker": "EIMI",
                        "name": "MSCI EM IMI UCITS",
                        "asset_class": "Equity",
                        "region": "EM",
                        "weight": 12.0,
                    },
                    {
                        "ticker": "AGGH",
                        "name": "Global Agg (EUR Hedged) UCITS",
                        "asset_class": "Bond",
                        "region": "Global (Hedged)",
                        "weight": 40.0,
                    },
                ],
            ),
            "All Weather (UCITS, approx)": dict(
                desc="Inspired by All Weather in UCITS.",
                items=[
                    {
                        "ticker": "CSPX",
                        "name": "S&P 500 UCITS",
                        "asset_class": "Equity",
                        "region": "USA",
                        "weight": 30.0,
                    },
                    {
                        "ticker": "IDTL",
                        "name": "$ Treasury 20+yr UCITS",
                        "asset_class": "Bond (Long)",
                        "region": "USA (USD)",
                        "weight": 40.0,
                    },
                    {
                        "ticker": "IUS7",
                        "name": "$ Treasury 7–10yr UCITS",
                        "asset_class": "Bond (Int)",
                        "region": "USA (USD)",
                        "weight": 15.0,
                    },
                    {
                        "ticker": "SGLD",
                        "name": "Physical Gold ETC",
                        "asset_class": "Gold",
                        "region": "Global",
                        "weight": 7.5,
                    },
                    {
                        "ticker": "CMOD",
                        "name": "Commodities UCITS",
                        "asset_class": "Commodities",
                        "region": "Global",
                        "weight": 7.5,
                    },
                ],
            ),
            "Permanent Portfolio (UCITS)": dict(
                desc="Four pillars (UCITS).",
                items=[
                    {
                        "ticker": "VWCE",
                        "name": "FTSE All-World UCITS",
                        "asset_class": "Equity",
                        "region": "Global",
                        "weight": 25.0,
                    },
                    {
                        "ticker": "IDTL",
                        "name": "$ Treasury 20+yr UCITS",
                        "asset_class": "Bond (Long)",
                        "region": "USA (USD)",
                        "weight": 25.0,
                    },
                    {
                        "ticker": "SGLD",
                        "name": "Physical Gold ETC",
                        "asset_class": "Gold",
                        "region": "Global",
                        "weight": 25.0,
                    },
                    {
                        "ticker": "Cash (proxy)",
                        "name": "EUR Money Market",
                        "asset_class": "Cash-like",
                        "region": "EUR",
                        "weight": 25.0,
                    },
                ],
            ),
        }


def build_performance_panel(
    comp_df: pd.DataFrame,
    bench_syms: list[str],
    overrides_map: dict[str, str],
    start_d: date,
    end_d: date,
    freq_label: str,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """
    Carica prezzi via Marketstack e ritorna:
    - panel: DataFrame con serie normalizzate Base=100
    - comp_df_sym: comp_df con colonna 'ms_symbol'
    - no_data: lista ticker senza dati
    """
    comp_df = comp_df.copy()
    comp_df = comp_df[comp_df["weight"] > 0].reset_index(drop=True)
    comp_df["ms_symbol"] = comp_df["ticker"].apply(lambda t: map_symbol(str(t), overrides_map))
    comp_syms = [s for s in comp_df["ms_symbol"] if isinstance(s, str) and s.strip()]

    all_syms = sorted(set(comp_syms + bench_syms))
    if not all_syms:
        return pd.DataFrame(), comp_df, []

    start_str = pd.Timestamp(start_d).strftime("%Y-%m-%d")
    end_str = pd.Timestamp(end_d).strftime("%Y-%m-%d")

    close_map: dict[str, pd.Series] = {}
    no_data: list[str] = []
    for s in all_syms:
        try:
            ser = fetch_prices_marketstack(s, start_str, end_str)
        except Exception:
            ser = pd.Series(dtype=float)
        if ser.empty:
            no_data.append(s)
        else:
            ser_r = resample_close(ser, freq_label)
            close_map[s] = normalize_base_100(ser_r)

    if not close_map:
        return pd.DataFrame(), comp_df, sorted(set(no_data))

    union_idx = None
    for ser in close_map.values():
        union_idx = ser.index if union_idx is None else union_idx.union(ser.index)
    union_idx = union_idx.sort_values()

    panel = pd.DataFrame(index=union_idx)
    for s, ser in close_map.items():
        panel[s] = ser.reindex(union_idx).ffill()

    return panel, comp_df, sorted(set(no_data))
