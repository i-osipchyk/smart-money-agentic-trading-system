from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime

import ccxt
import pandas as pd

_TF_SECONDS: dict[str, int] = {
    "5m": 300,
    "15m": 900,
    "1h": 3600,
    "4h": 14400,
    "1d": 86400,
}

_FETCH_LIMIT = 1000


class BinanceDataSource:
    def __init__(self) -> None:
        self._exchange = ccxt.binanceusdm()

    def get_ohlcv(
        self,
        symbol: str,
        timeframe: str,
        limit: int,
        since: datetime | None = None,
    ) -> pd.DataFrame:
        since_ms: int | None = int(since.timestamp() * 1000) if since is not None else None
        raw = self._exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit, since=since_ms)

        df = pd.DataFrame(
            raw, columns=["timestamp", "open", "high", "low", "close", "volume"]
        )
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        return df

    def fetch_range(
        self,
        symbol: str,
        timeframe: str,
        since: datetime,
        until: datetime,
        log: Callable[[str], None] | None = None,
    ) -> pd.DataFrame:
        """Paginate Binance and return a single sorted DataFrame from since to until."""
        tf_ms = _TF_SECONDS[timeframe] * 1000
        until_ms = int(until.timestamp() * 1000)
        current_since = since
        frames: list[pd.DataFrame] = []
        page = 0

        while True:
            page += 1
            since_ms = int(current_since.timestamp() * 1000)
            if log is not None:
                log(f"  page {page} (since {current_since.strftime('%Y-%m-%d %H:%M')} UTC) …")

            raw = self._exchange.fetch_ohlcv(symbol, timeframe, since=since_ms, limit=_FETCH_LIMIT)
            if not raw:
                break

            df = pd.DataFrame(raw, columns=["timestamp", "open", "high", "low", "close", "volume"])
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
            df = df[df["timestamp"].apply(lambda ts: int(ts.timestamp() * 1000)) <= until_ms]
            if not df.empty:
                frames.append(df)

            last_ms = int(raw[-1][0])
            if last_ms >= until_ms or len(raw) < _FETCH_LIMIT:
                break

            current_since = datetime.fromtimestamp((last_ms + tf_ms) / 1000, tz=UTC)

        if not frames:
            return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])

        result = pd.concat(frames, ignore_index=True)
        return result.drop_duplicates("timestamp").sort_values("timestamp").reset_index(drop=True)

    def fetch_both_ranges(
        self,
        symbol: str,
        htf_timeframe: str,
        htf_since: datetime,
        ltf_timeframe: str,
        ltf_since: datetime,
        until: datetime,
        log: Callable[[str], None] | None = None,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Fetch HTF and LTF data for a backtest range."""
        def _log(msg: str) -> None:
            if log is not None:
                log(msg)

        _log(f"Fetching HTF ({htf_timeframe}) data …")
        htf_df = self.fetch_range(symbol, htf_timeframe, htf_since, until, _log)
        _log(f"  → {len(htf_df)} candles\n")

        _log(f"Fetching LTF ({ltf_timeframe}) data …")
        ltf_df = self.fetch_range(symbol, ltf_timeframe, ltf_since, until, _log)
        _log(f"  → {len(ltf_df)} candles\n")

        return htf_df, ltf_df
