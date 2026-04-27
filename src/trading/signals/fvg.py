from datetime import timedelta

import pandas as pd

from trading.core.models import FVG, FvgStatus, Timeframe, Trend

_TIMEFRAME_DURATION: dict[Timeframe, timedelta] = {
    Timeframe.M5:  timedelta(minutes=5),
    Timeframe.M15: timedelta(minutes=15),
    Timeframe.H1:  timedelta(hours=1),
    Timeframe.H4:  timedelta(hours=4),
    Timeframe.D1:  timedelta(days=1),
}


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
            bottom = float(prev["high"])
            top = float(nxt["low"])
            subsequent = df.iloc[i + 2 :]
            if (subsequent["close"] < bottom).any():
                continue  # invalidated
            status = (
                FvgStatus.TESTED if (subsequent["low"] <= top).any() else FvgStatus.ACTIVE
            )
            fvgs.append(
                FVG(
                    timestamp=nxt["timestamp"] + _TIMEFRAME_DURATION[timeframe],
                    top=top,
                    bottom=bottom,
                    trend=Trend.BULLISH,
                    timeframe=timeframe,
                    status=status,
                )
            )

        if nxt["high"] < prev["low"]:
            top = float(prev["low"])
            bottom = float(nxt["high"])
            subsequent = df.iloc[i + 2 :]
            if (subsequent["close"] > top).any():
                continue  # invalidated
            status = (
                FvgStatus.TESTED if (subsequent["high"] >= bottom).any() else FvgStatus.ACTIVE
            )
            fvgs.append(
                FVG(
                    timestamp=nxt["timestamp"] + _TIMEFRAME_DURATION[timeframe],
                    top=top,
                    bottom=bottom,
                    trend=Trend.BEARISH,
                    timeframe=timeframe,
                    status=status,
                )
            )

    return sorted(fvgs, key=lambda f: f.timestamp)
