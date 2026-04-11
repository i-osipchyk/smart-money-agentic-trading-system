import ccxt
import pandas as pd
from datetime import UTC, datetime


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
    