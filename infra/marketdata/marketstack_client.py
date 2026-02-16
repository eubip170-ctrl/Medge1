# infra/marketdata/marketstack_client.py

import requests
import pandas as pd
from dataclasses import dataclass
from typing import Iterable, List, Dict, Any, Optional

# ============================================================
#  CONFIGURAZIONE BASE
# ============================================================

# ⚠️ In produzione è meglio usare:
#    import os
#    MARKETSTACK_API_KEY = os.getenv("MARKETSTACK_API_KEY", "")
MARKETSTACK_API_KEY = "9c304b60c6e2eccfe98fc0dbd0f3571b"

BASE_EOD_URL = "https://api.marketstack.com/v2/eod"
BASE_EOD_LATEST_URL = "https://api.marketstack.com/v2/eod/latest"
BASE_TICKERSLIST_URL = "https://api.marketstack.com/v2/tickerslist"
BASE_TICKERINFO_URL = "https://api.marketstack.com/v2/tickerinfo"
BASE_EXCHANGES_URL = "https://api.marketstack.com/v2/exchanges"


class MarketstackError(RuntimeError):
    """Errore generico per tutte le chiamate a Marketstack."""


def _get_api_key(api_key: Optional[str] = None) -> str:
    key = (api_key or MARKETSTACK_API_KEY or "").strip()
    if not key:
        raise MarketstackError("Marketstack API key non configurata.")
    return key


def _request_json(url: str, params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Piccolo wrapper per:
      - aggiungere access_key
      - gestire HTTP error
      - gestire errori Marketstack (campo 'error' nel JSON)
    """
    resp = requests.get(url, params=params, timeout=15)
    try:
        resp.raise_for_status()
    except Exception as e:
        raise MarketstackError(f"Errore HTTP {resp.status_code} su {url}: {e}") from e

    data = resp.json()
    if isinstance(data, dict) and "error" in data:
        err = data["error"] or {}
        code = err.get("code", "api_error")
        msg = err.get("message", "Errore Marketstack")
        raise MarketstackError(f"[{code}] {msg}")

    if not isinstance(data, dict):
        raise MarketstackError(f"Risposta inattesa da Marketstack: {type(data)}")

    return data


# ============================================================
#  FUNZIONE STORICA: OHLCV GIORNALIERO
# ============================================================

def load_ohlcv_from_marketstack(
    ticker: str,
    start: str,
    end: str,
    api_key: Optional[str] = None,
) -> pd.DataFrame:
    """
    Carica OHLCV giornalieri da Marketstack e restituisce un DataFrame con:
      - index: datetime (naive, giornaliero, ordinato)
      - columns: ['open','high','low','close','volume'] (float)

    Usa i campi adj_* se disponibili, altrimenti quelli raw.
    (Questa funzione è compatibile con il resto della tua app.)
    """
    key = _get_api_key(api_key)

    if not start or not end:
        raise MarketstackError("start/end sono obbligatori (YYYY-MM-DD).")

    params: Dict[str, Any] = {
        "access_key": key,
        "symbols": ticker,
        "date_from": start,
        "date_to": end,
        "limit": 1000,
        "sort": "ASC",
    }

    frames: List[pd.DataFrame] = []
    offset = 0

    while True:
        params["offset"] = offset
        js = _request_json(BASE_EOD_URL, params=params)

        data = js.get("data", [])
        if not data:
            break

        frames.append(pd.DataFrame.from_records(data))

        pag = js.get("pagination", {}) or {}
        count = int(pag.get("count", len(data)) or 0)
        total = int(pag.get("total", count) or 0)

        if count <= 0:
            break
        offset += count
        if offset >= total:
            break

    if not frames:
        raise MarketstackError(f"Nessun dato per {ticker} da Marketstack.")

    df = pd.concat(frames, ignore_index=True)

    # ----- Gestione colonna data -----
    if "date" not in df.columns:
        raise MarketstackError(f"Campo 'date' mancante per {ticker} da Marketstack.")

    idx = pd.to_datetime(df["date"], utc=True, errors="coerce")
    mask_valid = ~idx.isna()
    if not mask_valid.any():
        raise MarketstackError(f"Tutte le date sono invalide per {ticker}.")

    df = df.loc[mask_valid].copy()

    # FIX robusto: rendi idx un DatetimeIndex vero prima di tz_convert
    idx_sel = pd.Index(idx)[mask_valid]                 # funziona sia se idx è Series che Index
    idx_dt = pd.to_datetime(idx_sel, utc=True, errors="coerce")
    mask_dt = ~pd.isna(idx_dt)

    # riallinea df dopo il drop dei NaT
    df = df.loc[mask_dt].copy()

    # ora idx è sicuramente DatetimeIndex tz-aware => tz_convert OK
    idx_clean = pd.DatetimeIndex(idx_dt[mask_dt]).tz_convert(None)
    df.index = idx_clean
    df = df.sort_index()

    # ----- Usa adj_* se presenti -----
    cols_pref = {
        "open":   ("adj_open",   "open"),
        "high":   ("adj_high",   "high"),
        "low":    ("adj_low",    "low"),
        "close":  ("adj_close",  "close"),
        "volume": ("adj_volume", "volume"),
    }

    out = pd.DataFrame(index=df.index)

    for canon, (adj_name, raw_name) in cols_pref.items():
        if adj_name in df.columns and df[adj_name].notna().any():
            src = df[adj_name]
        elif raw_name in df.columns and df[raw_name].notna().any():
            src = df[raw_name]
        else:
            raise MarketstackError(
                f"Dati incompleti per {ticker}: mancano sia '{adj_name}' "
                f"sia '{raw_name}' per il campo '{canon}'."
            )

        out[canon] = pd.to_numeric(src, errors="coerce")

    out = out.astype(float)
    out = out.dropna(how="all").sort_index()

    if out.empty:
        raise MarketstackError(f"OHLCV vuoto dopo la pulizia per {ticker}.")

    # normalizza a indice giornaliero (solo data) e rimuove duplicati
    out.index = pd.to_datetime(out.index, errors="coerce").normalize()
    out = out[~out.index.duplicated(keep="last")]

    return out


# ============================================================
#  METADATA: EXCHANGES & TICKERS
# ============================================================

def list_exchanges(
    search: Optional[str] = None,
    limit: int = 1000,
    offset: int = 0,
    api_key: Optional[str] = None,
) -> pd.DataFrame:
    """
    Restituisce la lista degli exchange disponibili (MIC, nome, paese, ecc.).
    Serve per popolare la tendina "Exchange" nello screener.
    """
    key = _get_api_key(api_key)

    params: Dict[str, Any] = {
        "access_key": key,
        "limit": max(1, min(limit, 1000)),
        "offset": max(0, offset),
    }
    if search:
        params["search"] = search

    js = _request_json(BASE_EXCHANGES_URL, params=params)
    data = js.get("data") or []
    if not data:
        return pd.DataFrame(columns=["mic", "name", "acronym", "country", "country_code"])

    rows = []
    for item in data:
        se = item or {}
        rows.append(
            {
                "mic": se.get("mic"),
                "name": se.get("name"),
                "acronym": se.get("acronym"),
                "country": se.get("country"),
                "country_code": se.get("country_code"),
            }
        )
    return pd.DataFrame(rows)


def fetch_tickers_list(
    exchange_mic: Optional[str] = None,
    search: Optional[str] = None,
    limit: int = 200,
    offset: int = 0,
    api_key: Optional[str] = None,
) -> pd.DataFrame:
    """
    Wrapper per l'endpoint /v2/tickerslist.

    Restituisce un DataFrame con:
      - ticker
      - name
      - has_eod
      - has_intraday
      - exchange_name
      - exchange_acronym
      - exchange_mic
    """
    key = _get_api_key(api_key)

    params: Dict[str, Any] = {
        "access_key": key,
        "limit": max(1, min(limit, 1000)),
        "offset": max(0, offset),
    }
    if exchange_mic:
        params["exchange"] = exchange_mic
    if search:
        params["search"] = search

    js = _request_json(BASE_TICKERSLIST_URL, params=params)
    data = js.get("data") or []
    if not data:
        return pd.DataFrame(
            columns=[
                "ticker",
                "name",
                "has_intraday",
                "has_eod",
                "exchange_name",
                "exchange_acronym",
                "exchange_mic",
            ]
        )

    rows: List[Dict[str, Any]] = []
    for item in data:
        se = (item or {}).get("stock_exchange") or {}
        rows.append(
            {
                "ticker": item.get("ticker"),
                "name": item.get("name"),
                "has_intraday": bool(item.get("has_intraday")),
                "has_eod": bool(item.get("has_eod")),
                "exchange_name": se.get("name"),
                "exchange_acronym": se.get("acronym"),
                "exchange_mic": se.get("mic"),
            }
        )

    df = pd.DataFrame(rows)
    df = df.dropna(subset=["ticker"]).reset_index(drop=True)
    return df


# ============================================================
#  EOD LATEST (per screener)
# ============================================================

def fetch_eod_latest_bulk(
    symbols: Iterable[str],
    exchange_mic: Optional[str] = None,
    api_key: Optional[str] = None,
    chunk_size: int = 50,
) -> pd.DataFrame:
    """
    Ottiene l'ultimo EOD disponibile per una lista di ticker (max 100 / chiamata).

    Restituisce DataFrame indicizzato per 'symbol' con colonne:
      ['date','open','high','low','close','volume', ...]
    """
    key = _get_api_key(api_key)

    syms = [s.strip().upper() for s in symbols if isinstance(s, str) and s.strip()]
    syms = sorted(set(syms))
    if not syms:
        return pd.DataFrame()

    all_frames: List[pd.DataFrame] = []

    # L'API accetta max 100 symbols per chiamata → spezzettiamo
    chunk_size = max(1, min(chunk_size, 100))

    for i in range(0, len(syms), chunk_size):
        batch = syms[i : i + chunk_size]
        params: Dict[str, Any] = {
            "access_key": key,
            "symbols": ",".join(batch),
            "limit": len(batch),
        }
        if exchange_mic:
            params["exchange"] = exchange_mic

        js = _request_json(BASE_EOD_LATEST_URL, params=params)
        data = js.get("data") or []
        if not data:
            continue

        df = pd.DataFrame.from_records(data)
        if "symbol" not in df.columns and "ticker" in df.columns:
            df["symbol"] = df["ticker"]

        df["symbol"] = df["symbol"].astype(str).str.upper()
        df = df.set_index("symbol", drop=True)
        all_frames.append(df)

    if not all_frames:
        return pd.DataFrame()

    out = pd.concat(all_frames, axis=0)

    # se ci sono duplicati (stesso symbol più volte), tieni il più recente
    if "date" in out.columns:
        out["date_parsed"] = pd.to_datetime(out["date"], errors="coerce")
        out = (
            out.sort_values(["symbol", "date_parsed"])
            .groupby(level=0)
            .tail(1)
        )
        out = out.drop(columns=["date_parsed"])

    return out


# ============================================================
#  FUNDAMENTALS DI BASE (SETTORE / INDUSTRY / MARKET CAP, ecc.)
# ============================================================

def fetch_tickerinfo_bulk(
    symbols: Iterable[str],
    api_key: Optional[str] = None,
    max_items: int = 50,
) -> pd.DataFrame:
    """
    Recupera informazioni 'descrittive / fondamentali' per una lista di tickers.

    ⚠️ L'endpoint /tickerinfo lavora 1 ticker alla volta → per non bruciare
       il rate limit limitiamo di default a max_items (es. 50).
    """
    key = _get_api_key(api_key)

    syms = [s.strip().upper() for s in symbols if isinstance(s, str) and s.strip()]
    syms = sorted(set(syms))[: max(1, max_items)]
    if not syms:
        return pd.DataFrame()

    rows: List[Dict[str, Any]] = []

    for tkr in syms:
        params = {"access_key": key, "ticker": tkr}
        js = _request_json(BASE_TICKERINFO_URL, params=params)
        info = (js.get("data") or {}) if isinstance(js, dict) else {}
        if not info:
            continue

        rows.append(
            {
                "ticker": tkr,
                "item_type": info.get("item_type"),
                "sector": info.get("sector"),
                "industry": info.get("industry"),
                # questi campi potrebbero non esserci per tutti i titoli:
                "market_cap": info.get("market_cap"),
                "pe_ratio": info.get("pe_ratio"),
                "dividend_yield": info.get("dividend_yield"),
                "ipo_date": info.get("ipo_date"),
                "exchange_code": info.get("exchange_code"),
            }
        )

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    df["ticker"] = df["ticker"].astype(str).str.upper()
    return df.set_index("ticker", drop=False)


# ============================================================
#  FINVIZ-STYLE SCREENER (BACKEND)
# ============================================================

@dataclass
class ScreenerFilters:
    """
    Filtri principali in stile "Overview" di Finviz / TradingView.
    Tutti i filtri sono opzionali; il backend restituisce un DataFrame
    già pronto da passare alla tabella Streamlit.
    """
    exchange_mic: Optional[str] = None   # es. "XNAS", "XNYS"
    search: Optional[str] = None         # match su ticker o nome
    min_price: Optional[float] = None
    max_price: Optional[float] = None
    min_volume: Optional[float] = None
    max_volume: Optional[float] = None
    min_market_cap: Optional[float] = None
    max_market_cap: Optional[float] = None
    only_with_eod: bool = True
    limit: int = 200                     # max righe da /tickerslist
    offset: int = 0
    attach_fundamentals: bool = True     # se True chiama /tickerinfo
    max_fundamentals: int = 50           # per non sforare il rate limit


def run_screener(filters: ScreenerFilters, api_key: Optional[str] = None) -> pd.DataFrame:
    """
    Esegue uno screener "descrittivo" in stile Finviz:

    1) usa /tickerslist per estrarre i simboli per exchange + search;
    2) usa /eod/latest per ottenere ultimo prezzo, volume, ecc.;
    3) opzionalmente arricchisce con settore/industry/market_cap via /tickerinfo;
    4) applica i filtri numerici (prezzo, volume, market cap).

    Ritorna un DataFrame con colonne principali:
      ['Ticker','Company','Exchange','Sector','Industry','Price',
       'Change %','Volume','Market Cap', ...]
    """
    key = _get_api_key(api_key)

    # ---------- 1) Lista simboli ----------
    meta = fetch_tickers_list(
        exchange_mic=filters.exchange_mic,
        search=filters.search,
        limit=filters.limit,
        offset=filters.offset,
        api_key=key,
    )
    if meta.empty:
        return pd.DataFrame()

    if filters.only_with_eod:
        meta = meta[meta["has_eod"]].copy()
    if meta.empty:
        return pd.DataFrame()

    symbols = meta["ticker"].astype(str).str.upper().tolist()

    # ---------- 2) Ultimo EOD ----------
    eod = fetch_eod_latest_bulk(symbols, exchange_mic=filters.exchange_mic, api_key=key)
    if eod.empty:
        # restituisci almeno la parte descrittiva
        meta = meta.rename(columns={"ticker": "Ticker", "name": "Company"})
        return meta

    # uniforma chiave 'symbol'
    eod.index = eod.index.astype(str).str.upper()
    meta["ticker_up"] = meta["ticker"].astype(str).str.upper()

    df = meta.merge(
        eod,
        left_on="ticker_up",
        right_index=True,
        how="left",
        suffixes=("", "_eod"),
    )

    # ---------- 3) Colonne base prezzo / volume / change ----------
    # campi possibili: close, last, price, volume, change, change_percent
    price_col = None
    for c in ["close", "last", "price"]:
        if c in df.columns:
            price_col = c
            break

    if price_col is None:
        df["price"] = pd.NA
    else:
        df["price"] = pd.to_numeric(df[price_col], errors="coerce")

    if "volume" in df.columns:
        df["volume"] = pd.to_numeric(df["volume"], errors="coerce")
    else:
        df["volume"] = pd.NA

    if "change_percent" in df.columns:
        df["change_pct"] = pd.to_numeric(df["change_percent"], errors="coerce")
    elif price_col and "open" in df.columns:
        with pd.option_context("mode.use_inf_as_na", True):
            df["change_pct"] = (df[price_col] / df["open"] - 1.0) * 100.0
    else:
        df["change_pct"] = pd.NA

    # ---------- 4) Fundamentals opzionali ----------
    if filters.attach_fundamentals:
        info = fetch_tickerinfo_bulk(
            symbols=df["ticker"].tolist(),
            api_key=key,
            max_items=filters.max_fundamentals,
        )
        if not info.empty:
            info = info.drop(columns=[c for c in ["ticker"] if c in info.columns])
            info.index = info.index.astype(str).str.upper()
            df = df.merge(
                info,
                left_on="ticker_up",
                right_index=True,
                how="left",
                suffixes=("", "_info"),
            )

    # normalizza market_cap se presente
    if "market_cap" in df.columns:
        df["market_cap"] = pd.to_numeric(df["market_cap"], errors="coerce")

    # ---------- 5) Applicazione filtri numerici ----------
    m = pd.Series(True, index=df.index)

    if filters.min_price is not None:
        m &= df["price"].ge(filters.min_price)
    if filters.max_price is not None:
        m &= df["price"].le(filters.max_price)

    if filters.min_volume is not None:
        m &= df["volume"].ge(filters.min_volume)
    if filters.max_volume is not None:
        m &= df["volume"].le(filters.max_volume)

    if "market_cap" in df.columns:
        if filters.min_market_cap is not None:
            m &= df["market_cap"].ge(filters.min_market_cap)
        if filters.max_market_cap is not None:
            m &= df["market_cap"].le(filters.max_market_cap)

    df = df.loc[m].copy()
    if df.empty:
        return pd.DataFrame()

    # ---------- 6) Output "in stile Finviz" ----------
    df_out = pd.DataFrame()

    df_out["Ticker"] = df["ticker"]
    df_out["Company"] = df["name"]
    df_out["Exchange"] = df["exchange_acronym"]
    df_out["Exchange MIC"] = df["exchange_mic"]

    if "sector" in df.columns:
        df_out["Sector"] = df["sector"]
    if "industry" in df.columns:
        df_out["Industry"] = df["industry"]

    df_out["Price"] = df["price"]
    df_out["Change %"] = df["change_pct"]
    df_out["Volume"] = df["volume"]

    if "market_cap" in df.columns:
        df_out["Market Cap"] = df["market_cap"]

    df_out = df_out.sort_values("Ticker").reset_index(drop=True)
    return df_out
