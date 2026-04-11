import pandas as pd

from trading.core.models import FVG, Timeframe, Trend


def detect_fvg(df: pd.DataFrame, timeframe: Timeframe) -> list[FVG]:
    """
    Detect Fair Value Gaps.

    A bullish FVG forms when candle[i+1].low > candle[i-1].high —
    a gap between the previous candle's high and the next candle's low
    that price may return to fill.

    A bearish FVG forms when candle[i+1].high < candle[i-1].low —
    a gap between the previous candle's low and the next candle's high.

    Args:
        df:        OHLCV DataFrame, must have columns: timestamp, high, low
        timeframe: timeframe the candles belong to

    Returns:
        List of FVG objects ordered by timestamp.
    """
    fvgs: list[FVG] = []

    for i in range(1, len(df) - 1):
        prev = df.iloc[i - 1]
        curr = df.iloc[i]
        nxt = df.iloc[i + 1]

        if nxt["low"] > prev["high"]:
            fvgs.append(
                FVG(
                    timestamp=curr["timestamp"],
                    top=nxt["low"],
                    bottom=prev["high"],
                    trend=Trend.BULLISH,
                    timeframe=timeframe,
                )
            )

        if nxt["high"] < prev["low"]:
            fvgs.append(
                FVG(
                    timestamp=curr["timestamp"],
                    top=prev["low"],
                    bottom=nxt["high"],
                    trend=Trend.BEARISH,
                    timeframe=timeframe,
                )
            )

    return sorted(fvgs, key=lambda f: f.timestamp)
