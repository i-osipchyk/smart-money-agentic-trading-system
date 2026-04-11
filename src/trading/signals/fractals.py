import pandas as pd

from trading.core.models import Fractal, Timeframe


def detect_fractals(df: pd.DataFrame, timeframe: Timeframe, window: int = 2) -> list[Fractal]:
    """
    Detect fractal swing highs and lows.

    A fractal high is a candle with the highest high of the surrounding
    window candles on both sides. A fractal low is a candle with the
    lowest low of the surrounding window candles on both sides.

    Args:
        df:        OHLCV DataFrame, must have columns: timestamp, high, low
        timeframe: timeframe the candles belong to
        window:    number of candles required on each side to confirm a fractal

    Returns:
        List of Fractal objects ordered by timestamp.
    """
    fractals: list[Fractal] = []

    for i in range(window, len(df) - window):
        slice_highs = df["high"].iloc[i - window : i + window + 1]
        slice_lows = df["low"].iloc[i - window : i + window + 1]

        is_fractal_high = df["high"].iloc[i] == slice_highs.max() and (
            df["high"].iloc[i] > df["high"].iloc[i - window : i].max() and
            df["high"].iloc[i] > df["high"].iloc[i + 1 : i + window + 1].max()
        )

        is_fractal_low = df["low"].iloc[i] == slice_lows.min() and (
            df["low"].iloc[i] < df["low"].iloc[i - window : i].min() and
            df["low"].iloc[i] < df["low"].iloc[i + 1 : i + window + 1].min()
        )

        if is_fractal_high:
            fractals.append(
                Fractal(
                    timestamp=df["timestamp"].iloc[i],
                    price=df["high"].iloc[i],
                    is_high=True,
                    timeframe=timeframe,
                )
            )

        if is_fractal_low:
            fractals.append(
                Fractal(
                    timestamp=df["timestamp"].iloc[i],
                    price=df["low"].iloc[i],
                    is_high=False,
                    timeframe=timeframe,
                )
            )

    return sorted(fractals, key=lambda f: f.timestamp)
