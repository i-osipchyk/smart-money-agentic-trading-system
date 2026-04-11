from datetime import UTC, datetime

import pandas as pd
import pytest

from trading.core.models import Timeframe
from trading.signals.fractals import detect_fractals


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


def test_detects_fractal_high() -> None:
    df = make_df(
        highs=[1.0, 2.0, 5.0, 2.0, 1.0],
        lows= [0.5, 1.0, 1.0, 1.0, 0.5],
    )
    fractals = detect_fractals(df, Timeframe.D1, window=2)
    highs = [f for f in fractals if f.is_high]

    assert len(highs) == 1
    assert highs[0].price == 5.0


def test_detects_fractal_low() -> None:
    df = make_df(
        highs=[5.0, 5.0, 5.0, 5.0, 5.0],
        lows= [2.0, 1.5, 0.5, 1.5, 2.0],
    )
    fractals = detect_fractals(df, Timeframe.D1, window=2)
    lows = [f for f in fractals if not f.is_high]

    assert len(lows) == 1
    assert lows[0].price == 0.5


def test_no_fractals_in_flat_market() -> None:
    df = make_df(
        highs=[1.0, 1.0, 1.0, 1.0, 1.0],
        lows= [0.5, 0.5, 0.5, 0.5, 0.5],
    )
    fractals = detect_fractals(df, Timeframe.D1, window=2)
    assert len(fractals) == 0


def test_returns_sorted_by_timestamp() -> None:
    df = make_df(
        highs=[1.0, 3.0, 1.0, 5.0, 1.0, 3.0, 1.0],
        lows= [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
    )
    fractals = detect_fractals(df, Timeframe.D1, window=2)
    timestamps = [f.timestamp for f in fractals]
    assert timestamps == sorted(timestamps)
    