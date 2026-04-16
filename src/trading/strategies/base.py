from abc import ABC, abstractmethod

import pandas as pd

from trading.core.models import StrategySetup, Timeframe


class Strategy(ABC):
    """
    Abstract base class for all trading strategies.

    Each strategy encapsulates its own detection logic and configuration.
    Strategy-specific parameters (e.g. offsets, thresholds) belong in
    ``__init__``; the market data (symbol, DataFrames, timeframes) are
    passed to ``detect_entry`` at call time.

    Subclasses must define:
      - ``name``        — short identifier used in logs and UI labels.
      - ``description`` — prose explanation of the entry rules.
      - ``detect_entry`` — core detection logic returning a ``StrategySetup``
                           when a valid setup is found, or ``None`` otherwise.
    """

    name: str
    description: str

    @abstractmethod
    def detect_entry(
        self,
        symbol: str,
        htf_df: pd.DataFrame,
        htf_timeframe: Timeframe,
        ltf_df: pd.DataFrame,
        ltf_timeframe: Timeframe,
    ) -> StrategySetup | None:
        """
        Run detection on the supplied market data.

        Args:
            symbol:        Trading pair (e.g. ``"BTC/USDT:USDT"``).
            htf_df:        Higher-timeframe OHLCV DataFrame.
            htf_timeframe: Timeframe enum value for ``htf_df``.
            ltf_df:        Lower-timeframe OHLCV DataFrame.
            ltf_timeframe: Timeframe enum value for ``ltf_df``.

        Returns:
            A ``StrategySetup`` when a valid entry is detected, ``None`` otherwise.
        """
