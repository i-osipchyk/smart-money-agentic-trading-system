import pandas as pd
from pathlib import Path


class CSVDataSource:
    def __init__(self, data_dir: str | Path) -> None:
        self._data_dir = Path(data_dir)

    def get_ohlcv(
        self,
        symbol: str,
        timeframe: str,
        limit: int,
        filename_override: str | None = None,
    ) -> pd.DataFrame:
        filename = filename_override if filename_override is not None else self._build_filename(symbol, timeframe)
        filepath = self._data_dir / filename

        if not filepath.exists():
            raise FileNotFoundError(
                f"No data file found for {symbol} {timeframe}. "
                f"Expected: {filepath}"
            )

        df = pd.read_csv(filepath, parse_dates=["timestamp"])
        df = df.sort_values("timestamp").reset_index(drop=True)
        df = df.tail(limit).reset_index(drop=True)

        self._validate(df, filepath)
        return df

    def _build_filename(self, symbol: str, timeframe: str) -> str:
        clean_symbol = symbol.replace("/", "")
        return f"{clean_symbol}_{timeframe}_sample.csv"

    def _validate(self, df: pd.DataFrame, filepath: Path) -> None:
        required = {"timestamp", "open", "high", "low", "close", "volume"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(
                f"CSV file {filepath} is missing columns: {missing}"
            )
        if df.empty:
            raise ValueError(f"CSV file {filepath} contains no data")
        