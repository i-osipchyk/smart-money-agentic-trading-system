from typing import Protocol

import pandas as pd


class DataSource(Protocol):
    def get_ohlcv(
        self,
        symbol: str,
        timeframe: str,
        limit: int,
    ) -> pd.DataFrame:
        """
        Fetch OHLCV candles for a symbol.

        Args:
            symbol:    trading pair, e.g. "BTC/USDT"
            timeframe: candle interval, e.g. "1d", "4h", "15m"
            limit:     number of candles to return

        Returns:
            DataFrame with columns: timestamp, open, high, low, close, volume
            timestamp is a UTC-aware datetime, all prices are floats.
        """
        ...