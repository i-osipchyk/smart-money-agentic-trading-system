from datetime import UTC, datetime

import pandas as pd

from trading.core.models import Timeframe, Trend
from trading.signals.bos import detect_bos
from trading.signals.fractals import detect_fractals


def make_df(highs: list[float], lows: list[float], closes: list[float]) -> pd.DataFrame:
    n = len(highs)
    return pd.DataFrame(
        {
            "timestamp": [datetime(2024, 1, i + 1, tzinfo=UTC) for i in range(n)],
            "open": [1.0] * n,
            "high": highs,
            "low": lows,
            "close": closes,
            "volume": [1.0] * n,
        }
    )


def test_detects_bullish_bos() -> None:
    # fractal high forms at candle 2 (price 5.0)
    # candle 6 closes above it at 6.0 → bullish BOS
    df = make_df(
        highs=  [2.0, 3.0, 5.0, 3.0, 2.0, 6.0, 6.0],
        lows=   [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        closes= [1.5, 2.5, 4.5, 2.5, 1.5, 6.0, 5.5],
    )
    fractals = detect_fractals(df, Timeframe.D1, window=2)
    bos_list = detect_bos(df, fractals, Timeframe.D1)
    bullish = [b for b in bos_list if b.trend == Trend.BULLISH]

    assert len(bullish) >= 1
    assert bullish[0].level == 5.0


def test_detects_bearish_bos() -> None:
    # fractal low forms at candle 2 (price 1.0)
    # candle 6 closes below it at 0.5 → bearish BOS
    df = make_df(
        highs=  [5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0],
        lows=   [2.0, 1.5, 1.0, 1.5, 2.0, 0.5, 0.5],
        closes= [3.0, 2.0, 1.2, 2.0, 3.0, 0.5, 0.8],
    )
    fractals = detect_fractals(df, Timeframe.D1, window=2)
    bos_list = detect_bos(df, fractals, Timeframe.D1)
    bearish = [b for b in bos_list if b.trend == Trend.BEARISH]

    assert len(bearish) >= 1
    assert bearish[0].level == 1.0


def test_no_bos_without_fractals() -> None:
    df = make_df(
        highs=  [1.0, 1.0, 1.0, 1.0, 1.0],
        lows=   [0.5, 0.5, 0.5, 0.5, 0.5],
        closes= [0.8, 0.8, 0.8, 0.8, 0.8],
    )
    fractals = detect_fractals(df, Timeframe.D1, window=2)
    bos_list = detect_bos(df, fractals, Timeframe.D1)
    assert len(bos_list) == 0


def test_fractal_level_broken_only_once() -> None:
    # same fractal high broken multiple times — should only register once
    df = make_df(
        highs=  [2.0, 3.0, 5.0, 3.0, 2.0, 6.0, 7.0],
        lows=   [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        closes= [1.5, 2.5, 4.5, 2.5, 1.5, 6.0, 7.0],
    )
    fractals = detect_fractals(df, Timeframe.D1, window=2)
    bos_list = detect_bos(df, fractals, Timeframe.D1)
    bullish = [b for b in bos_list if b.trend == Trend.BULLISH]

    levels_broken = [b.level for b in bullish]
    assert levels_broken.count(5.0) == 1
    