"""Runner package — non-GUI logic for one-time and backtest trade validation."""

from .backtest import BacktestRunner
from .config import (
    DataSourceType,
    OutputMode,
    RunConfig,
    SimulationResult,
    StrategyKey,
    TradeRecord,
    make_strategy,
)
from .onetime import OneTimeRunner
from .simulator import OrderSimulator

__all__ = [
    "BacktestRunner",
    "DataSourceType",
    "OneTimeRunner",
    "OrderSimulator",
    "OutputMode",
    "RunConfig",
    "SimulationResult",
    "StrategyKey",
    "TradeRecord",
    "make_strategy",
]
