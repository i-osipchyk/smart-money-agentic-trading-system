import pandas as pd

from trading.core.models import BOS, Fractal, Timeframe, Trend


def detect_bos(
    df: pd.DataFrame,
    fractals: list[Fractal],
    timeframe: Timeframe,
) -> list[BOS]:
    """
    Detect Break of Structure events.

    A bullish BOS occurs when price closes above a fractal high.
    A bearish BOS occurs when price closes below a fractal low.

    Args:
        df:        OHLCV DataFrame, must have columns: timestamp, close
        fractals:  list of fractals detected on the same DataFrame
        timeframe: timeframe the candles belong to

    Returns:
        List of BOS objects ordered by timestamp.
    """
    bos_list: list[BOS] = []

    fractal_highs = [f for f in fractals if f.is_high]
    fractal_lows = [f for f in fractals if not f.is_high]

    for i in range(1, len(df)):
        candle_time = df["timestamp"].iloc[i]
        close = df["close"].iloc[i]

        for fractal in fractal_highs:
            if fractal.timestamp >= candle_time:
                continue
            if close > fractal.price:
                bos_list.append(
                    BOS(
                        timestamp=candle_time,
                        level=fractal.price,
                        trend=Trend.BULLISH,
                        timeframe=timeframe,
                    )
                )
                fractal_highs.remove(fractal)
                break

        for fractal in fractal_lows:
            if fractal.timestamp >= candle_time:
                continue
            if close < fractal.price:
                bos_list.append(
                    BOS(
                        timestamp=candle_time,
                        level=fractal.price,
                        trend=Trend.BEARISH,
                        timeframe=timeframe,
                    )
                )
                fractal_lows.remove(fractal)
                break

    return sorted(bos_list, key=lambda b: b.timestamp)
