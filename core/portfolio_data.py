# core/portfolio_data.py
from __future__ import annotations

import re
from typing import Any, Dict, IO, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd


# -------------------------
# Utilities / validation
# -------------------------
def ensure_dataframe(obj: Any, *, copy: bool = True) -> pd.DataFrame:
    if obj is None:
        return pd.DataFrame()
    if isinstance(obj, pd.DataFrame):
        return obj.copy() if copy else obj
    if isinstance(obj, pd.Series):
        return obj.to_frame().copy() if copy else obj.to_frame()
    try:
        df = pd.DataFrame(obj)
        return df.copy() if copy else df
    except Exception:
        return pd.DataFrame()


def clean_prices(df: pd.DataFrame) -> pd.DataFrame:
    d = ensure_dataframe(df, copy=True)
    if d.empty:
        return d

    d.index = pd.to_datetime(d.index, errors="coerce")
    d = d.loc[d.index.notna()].sort_index()
    d.index = d.index.normalize()
    d = d[~d.index.duplicated(keep="last")]

    for c in d.columns:
        d[c] = pd.to_numeric(d[c], errors="coerce")

    d = d.replace([np.inf, -np.inf], np.nan).dropna(how="all")
    return d


def pct_returns(prices: pd.DataFrame) -> pd.DataFrame:
    p = clean_prices(prices)
    if p.empty:
        return pd.DataFrame()
    return p.pct_change().replace([np.inf, -np.inf], np.nan).dropna(how="all")


def log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    p = clean_prices(prices)
    if p.empty:
        return pd.DataFrame()
    return np.log(p / p.shift(1)).replace([np.inf, -np.inf], np.nan).dropna(how="all")


def portfolio_value_from_prices(
    prices: pd.DataFrame,
    weights: Union[pd.Series, Dict[str, float], Sequence[float]],
    initial_capital: float = 100_000.0,
) -> pd.Series:
    p = clean_prices(prices)
    if p.empty or p.shape[1] == 0:
        return pd.Series(dtype=float)

    cols = list(p.columns)

    if isinstance(weights, pd.Series):
        w = weights.reindex(cols).fillna(0.0).astype(float)
    elif isinstance(weights, dict):
        w = pd.Series({c: float(weights.get(c, 0.0)) for c in cols}, index=cols, dtype=float)
    else:
        try:
            w = pd.Series(list(weights), index=cols, dtype=float)
        except Exception:
            w = pd.Series(0.0, index=cols, dtype=float)

    s = float(w.sum())
    if s != 0:
        w = w / s

    r_assets = pct_returns(p)
    if r_assets.empty:
        return pd.Series(dtype=float)

    rp = (r_assets.mul(w, axis=1)).sum(axis=1).replace([np.inf, -np.inf], np.nan).dropna()
    if rp.empty:
        return pd.Series(dtype=float)

    return (1.0 + rp).cumprod() * float(initial_capital)


# -------------------------
# OHLCV handling (no Yahoo)
# -------------------------
def extract_close(df_ohlcv: pd.DataFrame) -> pd.DataFrame:
    d = ensure_dataframe(df_ohlcv, copy=True)
    if d.empty:
        return d

    # single-asset columns
    for cand in ["Adj Close", "AdjClose", "Close", "close"]:
        if cand in d.columns:
            out = d[[cand]].copy()
            out.columns = ["CLOSE"]
            return clean_prices(out)

    # MultiIndex (Ticker, Field)
    if isinstance(d.columns, pd.MultiIndex) and d.columns.nlevels >= 2:
        tickers = sorted({c[0] for c in d.columns})
        fields = ["Adj Close", "AdjClose", "Close", "close"]

        closes: Dict[str, pd.Series] = {}
        for t in tickers:
            for f in fields:
                if (t, f) in d.columns:
                    closes[str(t)] = pd.to_numeric(d[(t, f)], errors="coerce")
                    break

        return clean_prices(pd.DataFrame(closes, index=d.index))

    # Flat columns like "AAPL_Close" or "AAPL Adj Close"
    closes2: Dict[str, pd.Series] = {}
    for c in d.columns:
        c_str = str(c)
        m = re.match(r"^(.+?)\s*[_\-\s]\s*(Adj\s*Close|AdjClose|Close)$", c_str, flags=re.IGNORECASE)
        if m:
            t = m.group(1).strip().upper()
            closes2[t] = pd.to_numeric(d[c], errors="coerce")

    if closes2:
        return clean_prices(pd.DataFrame(closes2, index=d.index))

    # fallback: treat as prices
    return clean_prices(d)


def load_ohlcv_from_yf(*args, **kwargs) -> pd.DataFrame:
    # compat stub: NON supportato
    raise RuntimeError(
        "load_ohlcv_from_yf() non è disponibile: yfinance è stato rimosso. "
        "Carica i dati da CSV/Excel o usa un tuo provider."
    )


# -------------------------
# CSV loader (optional helper)
# -------------------------
def load_prices_from_csv(file_or_path: Union[str, IO]) -> pd.DataFrame:
    df = pd.read_csv(file_or_path)
    if df.shape[1] < 2:
        raise ValueError("CSV non valido: servono almeno Date + 1 colonna prezzo.")

    date_col = df.columns[0]
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.loc[df[date_col].notna()].copy()
    df = df.set_index(date_col).sort_index()

    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    return clean_prices(df)


# -------------------------
# DataLoader (compatibilità con vecchie firme)
# -------------------------
class DataLoader:
    """
    DataLoader compatibile con firme legacy e con .fetch().

    Supporta creazione tipo:
      - DataLoader()
      - DataLoader(prices_df)
      - DataLoader(tickers, start, end)
      - DataLoader(provider, tickers, start, end, interval)
    e poi:
      - dl.fetch()  -> scarica dati (senza yfinance) usando Stooq

    Nota: se provider è "yf"/"yfinance"/"yahoo" NON usa yfinance: viene mappato a Stooq.
    """

    def __init__(self, *args, **kwargs) -> None:
        self.provider: Optional[str] = kwargs.pop("provider", None)
        self.tickers: list[str] = []
        self.start = kwargs.pop("start", None)
        self.end = kwargs.pop("end", None)
        self.interval: str = kwargs.pop("interval", "1d")

        self.prices: Optional[pd.DataFrame] = kwargs.pop("prices", None)
        self.ohlcv: Optional[pd.DataFrame] = kwargs.pop("ohlcv", None)

        # ---- parse positional args for compatibility ----
        if len(args) == 0:
            return

        # DataLoader(prices_df)
        if len(args) == 1 and isinstance(args[0], pd.DataFrame):
            self.prices = clean_prices(args[0])
            return

        # Pattern A: DataLoader(provider, tickers, start, end, interval?)
        if isinstance(args[0], str) and len(args) >= 2:
            self.provider = args[0]
            self._set_tickers(args[1])
            if len(args) >= 3:
                self.start = args[2]
            if len(args) >= 4:
                self.end = args[3]
            if len(args) >= 5 and isinstance(args[4], str):
                self.interval = args[4]
            return

        # Pattern B: DataLoader(tickers, start, end, interval?)
        self._set_tickers(args[0])
        if len(args) >= 2:
            self.start = args[1]
        if len(args) >= 3:
            self.end = args[2]
        if len(args) >= 4 and isinstance(args[3], str):
            self.interval = args[3]

    def _set_tickers(self, tickers: Any) -> None:
        if tickers is None:
            self.tickers = []
            return
        if isinstance(tickers, str):
            self.tickers = [t.strip().upper() for t in tickers.split(",") if t.strip()]
            return
        if isinstance(tickers, (list, tuple, set)):
            self.tickers = [str(t).strip().upper() for t in tickers if str(t).strip()]
            return
        self.tickers = []

    # --- setters ---
    def set_prices(self, prices: pd.DataFrame) -> pd.DataFrame:
        self.prices = clean_prices(prices)
        return self.prices

    def set_ohlcv(self, ohlcv: pd.DataFrame) -> pd.DataFrame:
        self.ohlcv = ensure_dataframe(ohlcv, copy=True)
        return self.ohlcv

    def from_csv(self, file_or_path: Union[str, IO]) -> pd.DataFrame:
        self.prices = load_prices_from_csv(file_or_path)
        return self.prices

    # --- getters ---
    def get_prices(self) -> pd.DataFrame:
        if self.prices is not None and not getattr(self.prices, "empty", True):
            return self.prices
        raise RuntimeError("Prezzi non caricati. Usa fetch() oppure carica da CSV/Excel o set_prices().")

    def get_ohlcv(self) -> pd.DataFrame:
        if self.ohlcv is None or getattr(self.ohlcv, "empty", True):
            raise RuntimeError("OHLCV non caricato. Usa fetch() oppure set_ohlcv() o carica da file.")
        return self.ohlcv

    def get_close(self) -> pd.DataFrame:
        if self.ohlcv is not None and not getattr(self.ohlcv, "empty", True):
            return extract_close(self.ohlcv)
        return self.get_prices()

    # -------------------------
    # FETCH (no yfinance) -> Stooq
    # -------------------------
    def fetch(self, *args, **kwargs) -> pd.DataFrame:
        """
        Scarica dati OHLCV e prezzi Close.

        Compat:
          - fetch()
          - fetch(tickers, start, end, interval)
          - fetch(provider=..., tickers=..., start=..., end=..., interval=...)

        Default provider: stooq.
        Se provider è "yf"/"yfinance"/"yahoo" viene mappato a stooq (non usa yfinance).
        """
        # parse positional overrides
        if len(args) >= 1 and args[0] is not None:
            # può essere provider string oppure tickers
            if isinstance(args[0], str) and len(args) >= 2:
                self.provider = args[0]
                self._set_tickers(args[1])
                if len(args) >= 3:
                    self.start = args[2]
                if len(args) >= 4:
                    self.end = args[3]
                if len(args) >= 5 and isinstance(args[4], str):
                    self.interval = args[4]
            else:
                self._set_tickers(args[0])
                if len(args) >= 2:
                    self.start = args[1]
                if len(args) >= 3:
                    self.end = args[2]
                if len(args) >= 4 and isinstance(args[3], str):
                    self.interval = args[3]

        # keyword overrides
        if "provider" in kwargs and kwargs["provider"] is not None:
            self.provider = kwargs["provider"]
        if "tickers" in kwargs and kwargs["tickers"] is not None:
            self._set_tickers(kwargs["tickers"])
        if "start" in kwargs and kwargs["start"] is not None:
            self.start = kwargs["start"]
        if "end" in kwargs and kwargs["end"] is not None:
            self.end = kwargs["end"]
        if "interval" in kwargs and kwargs["interval"] is not None:
            self.interval = str(kwargs["interval"])

        if not self.tickers:
            raise ValueError("tickers vuoti: inserisci almeno 1 ticker nel portafoglio.")

        provider = (self.provider or "stooq").strip().lower()
        # se qualcuno passa "yf" per legacy, NON usare yfinance: mappa a stooq
        if provider in {"yf", "yfinance", "yahoo"}:
            provider = "stooq"

        if provider != "stooq":
            raise RuntimeError(f"Provider '{provider}' non supportato (solo 'stooq' senza yfinance).")

        i_code = self._stooq_interval_code(self.interval)
        start_dt = pd.to_datetime(self.start, errors="coerce") if self.start is not None else None
        end_dt = pd.to_datetime(self.end, errors="coerce") if self.end is not None else None

        ohlcv_list: list[pd.DataFrame] = []
        close_cols: Dict[str, pd.Series] = {}

        for t in self.tickers:
            df_one = self._fetch_stooq_one(t, i_code)
            if start_dt is not None:
                df_one = df_one.loc[df_one.index >= start_dt]
            if end_dt is not None:
                df_one = df_one.loc[df_one.index <= end_dt]

            if df_one.empty:
                raise RuntimeError(f"Nessun dato scaricato per {t} (controlla ticker / mercato).")

            close_cols[t] = df_one["Close"].astype(float)
            ohlcv_list.append(df_one)

        # prices (Close)
        prices_df = pd.DataFrame(close_cols)
        self.prices = clean_prices(prices_df)

        # ohlcv MultiIndex: (Ticker, Field)
        ohlcv_multi = pd.concat(ohlcv_list, axis=1, keys=self.tickers)
        # assicura index datetime clean
        ohlcv_multi.index = pd.to_datetime(ohlcv_multi.index, errors="coerce")
        ohlcv_multi = ohlcv_multi.loc[ohlcv_multi.index.notna()].sort_index()
        self.ohlcv = ohlcv_multi

        return self.prices

    @staticmethod
    def _stooq_interval_code(interval: str) -> str:
        s = (interval or "1d").strip().lower()
        if s in {"1d", "d", "day", "daily"}:
            return "d"
        if s in {"1w", "w", "week", "weekly", "1wk"}:
            return "w"
        if s in {"1m", "m", "month", "monthly", "1mo"}:
            return "m"
        # fallback
        return "d"

    @staticmethod
    def _stooq_symbol(ticker: str) -> str:
        # stooq usa tipicamente lowercase e suffissi tipo ".us"
        t = (ticker or "").strip()
        if not t:
            return ""
        # se contiene già un suffisso (es. BRK.B.US) -> lower
        if "." in t:
            return t.lower()
        # default US
        return f"{t.lower()}.us"

    @classmethod
    def _fetch_stooq_one(cls, ticker: str, i_code: str) -> pd.DataFrame:
        sym = cls._stooq_symbol(ticker)
        if not sym:
            raise ValueError("Ticker vuoto.")

        url = f"https://stooq.com/q/d/l/?s={sym}&i={i_code}"
        df = pd.read_csv(url)

        # stooq columns: Date, Open, High, Low, Close, Volume
        if "Date" not in df.columns:
            raise RuntimeError(f"Formato Stooq inatteso per {ticker}.")

        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        df = df.loc[df["Date"].notna()].copy()
        df = df.set_index("Date").sort_index()

        for c in ["Open", "High", "Low", "Close", "Volume"]:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce")
            else:
                # se manca qualche colonna, creala
                df[c] = np.nan

        df = df[["Open", "High", "Low", "Close", "Volume"]]
        df = df.replace([np.inf, -np.inf], np.nan).dropna(how="all")
        return df
