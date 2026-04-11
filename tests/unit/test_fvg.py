from datetime import UTC, datetime

import pandas as pd

from trading.core.models import Timeframe, Trend
from trading.signals.fvg import detect_fvg


def make_df(highs: list[float], lows: list[float]) -> pd.DataFrame:
    n = len(highs)
    return pd.DataFrame(
        {
            "timestamp": [datetime(2024, 1, i + 1, tzinfo=UTC) for i in range(n)],
            "open": [1.0] * n,
            "high": highs,
            "low": lows,
            "close": [1.0] * n,
            "volume": [1.0] * n,
        }
    )


def test_detects_bullish_fvg() -> None:
    # candle[i-1].high = 2.0, candle[i+1].low = 3.0 → gap between 2.0 and 3.0
    df = make_df(
        highs=[1.0, 4.0, 5.0],
        lows= [0.5, 2.5, 3.0],
    )
    fvgs = detect_fvg(df, Timeframe.D1)
    bullish = [f for f in fvgs if f.trend == Trend.BULLISH]

    assert len(bullish) == 1
    assert bullish[0].bottom == 1.0
    assert bullish[0].top == 3.0


def test_detects_bearish_fvg() -> None:
    # candle[i-1].low = 3.0, candle[i+1].high = 2.0 → gap between 2.0 and 3.0
    df = make_df(
        highs=[4.0, 2.5, 2.0],
        lows= [3.0, 1.5, 0.5],
    )
    fvgs = detect_fvg(df, Timeframe.D1)
    bearish = [f for f in fvgs if f.trend == Trend.BEARISH]

    assert len(bearish) == 1
    assert bearish[0].top == 3.0
    assert bearish[0].bottom == 2.0


def test_no_fvg_when_candles_overlap() -> None:
    df = make_df(
        highs=[3.0, 4.0, 3.5],
        lows= [1.0, 2.0, 1.5],
    )
    fvgs = detect_fvg(df, Timeframe.D1)
    assert len(fvgs) == 0


def test_returns_sorted_by_timestamp() -> None:
    df = make_df(
        highs=[1.0, 5.0, 6.0, 5.0, 6.0],
        lows= [0.5, 3.0, 4.0, 3.0, 4.0],
    )
    fvgs = detect_fvg(df, Timeframe.D1)
    timestamps = [f.timestamp for f in fvgs]
    assert timestamps == sorted(timestamps)
    