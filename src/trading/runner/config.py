"""Shared data classes and type aliases for the runner package."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Literal

from trading.agents.llm_provider import LLMConfig
from trading.core.models import Timeframe, Trend
from trading.strategies.base import Strategy
from trading.strategies.htf_fvg_ltf_bos import HtfFvgLtfBos
from trading.strategies.htf_fvg_ltf_bos_v2 import HtfFvgLtfBosV2

OutputMode = Literal["prompt", "agent", "baseline", "strategy_inspect"]
DataSourceType = Literal["csv", "past", "live"]
StrategyKey = Literal["htf_fvg_ltf_bos", "htf_fvg_ltf_bos_v2"]

_TF_SECONDS: dict[str, int] = {
    "5m": 300,
    "15m": 900,
    "1h": 3600,
    "4h": 14400,
    "1d": 86400,
}

_FMT = "{:,.2f}"


_STRATEGY_REGISTRY: dict[str, type[Strategy]] = {
    "htf_fvg_ltf_bos": HtfFvgLtfBos,
    "htf_fvg_ltf_bos_v2": HtfFvgLtfBosV2,
}


def make_strategy(
    key: StrategyKey,
    fvg_offset_pct: float,
    block_tested_fvgs: bool = False,
) -> Strategy:
    return _STRATEGY_REGISTRY[key](
        fvg_offset_pct=fvg_offset_pct,
        block_tested_fvgs=block_tested_fvgs,
    )


def _ts(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%d %H:%M")


@dataclass
class RunConfig:
    # shared
    symbol: str
    htf_tf: Timeframe
    ltf_tf: Timeframe
    htf_limit: int
    ltf_limit: int
    fvg_offset_pct: float
    output_mode: OutputMode
    strategy: StrategyKey = "htf_fvg_ltf_bos_v2"
    block_tested_fvgs: bool = False
    # one-time data source
    data_source: DataSourceType = "live"
    htf_csv: str | None = None
    ltf_csv: str | None = None
    until: datetime | None = None
    # backtest range
    bt_from: datetime | None = None
    bt_to: datetime | None = None
    # agent
    llm_config: LLMConfig | None = None
    # simulation (agent + baseline)
    order_timeout: int = 10
    max_risk_pct: float = 1.0
    rr_ratio: float = 2.0


@dataclass(frozen=True)
class TradeRecord:
    trade_num: int
    setup_dt: datetime
    direction: Trend
    entry: float
    sl: float
    tp: float | None
    result: str  # WIN | LOSS | OPEN | CANCELED_PRICE | CANCELED_TIMEOUT
    fill_dt: datetime | None
    close_dt: datetime | None
    reasoning: str
    confidence: str


@dataclass(frozen=True)
class SimulationResult:
    trades: list[TradeRecord] = field(default_factory=list)
    skipped_no_trade: int = 0
    skipped_risk: int = 0
    skipped_active_order: int = 0
    steps_checked: int = 0
