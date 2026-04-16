"""
Backtest data source.

Downloads the full HTF and LTF OHLCV history for a date range in one go
(paginating through Binance as needed), then serves sliced windows on each
iteration without touching the network again.

Usage::

    src = BacktestDataSource(
        symbol="BTC/USDT:USDT",
        htf_timeframe="1h", htf_limit=72,
        ltf_timeframe="15m", ltf_limit=16,
        bt_from=datetime(2026, 3, 1, tzinfo=UTC),
        bt_to=datetime(2026, 4, 1, tzinfo=UTC),
    )
    src.prepare(progress=print)          # one-time bulk fetch

    for candle_dt, htf_df, ltf_df in src:
        setup = strategy.detect_entry(symbol, htf_df, htf_tf, ltf_df, ltf_tf)
        ...
"""

from collections.abc import Callable, Iterator
from datetime import UTC, datetime, timedelta, timezone

import ccxt
import pandas as pd

_TF_SECONDS: dict[str, int] = {
    "5m": 300,
    "15m": 900,
    "1h": 3600,
    "4h": 14400,
    "1d": 86400,
}

# Binance USDM allows up to 1 500 candles per request; stay a bit below to be safe.
_FETCH_LIMIT = 1000


class BacktestDataSource:
    """
    Pre-fetches all candle data needed for a backtest range and iterates
    through it in-memory, yielding one (timestamp, htf_df, ltf_df) tuple per
    LTF candle step without making any further API calls.

    Args:
        symbol:         Trading pair (e.g. ``"BTC/USDT:USDT"``).
        htf_timeframe:  Higher-timeframe string (e.g. ``"1h"``).
        htf_limit:      Number of HTF candles in each window.
        ltf_timeframe:  Lower-timeframe string (e.g. ``"15m"``).
        ltf_limit:      Number of LTF candles in each window.
        bt_from:        Start of the backtest range (inclusive, tz-aware UTC).
        bt_to:          End of the backtest range (inclusive, tz-aware UTC).
    """

    def __init__(
        self,
        symbol: str,
        htf_timeframe: str,
        htf_limit: int,
        ltf_timeframe: str,
        ltf_limit: int,
        bt_from: datetime,
        bt_to: datetime,
    ) -> None:
        self._symbol = symbol
        self._htf_tf = htf_timeframe
        self._htf_limit = htf_limit
        self._ltf_tf = ltf_timeframe
        self._ltf_limit = ltf_limit

        ltf_step = _TF_SECONDS[ltf_timeframe]

        # Align both boundaries to LTF candle boundaries
        self._bt_from_ts = (int(bt_from.timestamp()) // ltf_step) * ltf_step
        self._bt_to_ts = (int(bt_to.timestamp()) // ltf_step) * ltf_step
        self._ltf_step = ltf_step

        # Prefetch start: far enough back so the first iteration has a full window
        htf_lookback = timedelta(seconds=htf_limit * _TF_SECONDS[htf_timeframe])
        ltf_lookback = timedelta(seconds=ltf_limit * ltf_step)

        self._htf_fetch_from = datetime.fromtimestamp(self._bt_from_ts, tz=UTC) - htf_lookback
        self._ltf_fetch_from = datetime.fromtimestamp(self._bt_from_ts, tz=UTC) - ltf_lookback
        self._fetch_to = datetime.fromtimestamp(self._bt_to_ts, tz=UTC)

        self._htf_df: pd.DataFrame = pd.DataFrame()
        self._ltf_df: pd.DataFrame = pd.DataFrame()
        self._exchange = ccxt.binanceusdm()

    @property
    def total_steps(self) -> int:
        """Total number of LTF candle steps in the backtest range."""
        return (self._bt_to_ts - self._bt_from_ts) // self._ltf_step + 1

    def prepare(self, progress: Callable[[str], None] | None = None) -> None:
        """
        Fetch all required candle data from Binance.  Call once before
        iterating.

        Args:
            progress: Optional callable accepting a single string — called with
                      progress messages during the fetch (e.g. ``print``).
        """
        def _log(msg: str) -> None:
            if progress is not None:
                progress(msg)

        _log(f"Fetching HTF ({self._htf_tf}) data …")
        self._htf_df = self._fetch_all(self._htf_tf, self._htf_fetch_from, self._fetch_to, _log)
        _log(f"  → {len(self._htf_df)} candles\n")

        _log(f"Fetching LTF ({self._ltf_tf}) data …")
        self._ltf_df = self._fetch_all(self._ltf_tf, self._ltf_fetch_from, self._fetch_to, _log)
        _log(f"  → {len(self._ltf_df)} candles\n")

    def __iter__(self) -> Iterator[tuple[datetime, pd.DataFrame, pd.DataFrame]]:
        """
        Yield ``(candle_dt, htf_df, ltf_df)`` for every LTF candle close in
        the backtest range.  Steps where the pre-fetched data cannot fill the
        full window are skipped silently.
        """
        if self._htf_df.empty or self._ltf_df.empty:
            raise RuntimeError("Call prepare() before iterating.")

        current_ts = self._bt_from_ts
        while current_ts <= self._bt_to_ts:
            current_dt = datetime.fromtimestamp(current_ts, tz=timezone.utc)

            htf_slice = (
                self._htf_df[self._htf_df["timestamp"] <= current_dt]
                .tail(self._htf_limit)
                .reset_index(drop=True)
            )
            ltf_slice = (
                self._ltf_df[self._ltf_df["timestamp"] <= current_dt]
                .tail(self._ltf_limit)
                .reset_index(drop=True)
            )

            if len(htf_slice) == self._htf_limit and len(ltf_slice) == self._ltf_limit:
                yield current_dt, htf_slice, ltf_slice

            current_ts += self._ltf_step

    # ---------------------------------------------------------------- internals

    def _fetch_all(
        self,
        timeframe: str,
        since: datetime,
        until: datetime,
        log: Callable[[str], None] | None,
    ) -> pd.DataFrame:
        """Paginate through Binance and return a single sorted DataFrame."""
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

            raw = self._exchange.fetch_ohlcv(
                self._symbol, timeframe, since=since_ms, limit=_FETCH_LIMIT
            )
            if not raw:
                break

            df = pd.DataFrame(raw, columns=["timestamp", "open", "high", "low", "close", "volume"])
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)

            # Keep only candles up to and including `until`
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
        result = result.drop_duplicates("timestamp").sort_values("timestamp").reset_index(drop=True)
        return result
