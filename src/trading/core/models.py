from datetime import datetime
from enum import Enum

from pydantic import BaseModel


class Timeframe(str, Enum):
    M5 = "5m"
    M15 = "15m"
    H1 = "1h"
    H4 = "4h"
    D1 = "1d"


class Trend(str, Enum):
    BULLISH = "bullish"
    BEARISH = "bearish"
    RANGING = "ranging"


class FVG(BaseModel):
    timestamp: datetime
    top: float
    bottom: float
    trend: Trend
    timeframe: Timeframe


class Fractal(BaseModel):
    timestamp: datetime
    price: float
    is_high: bool
    timeframe: Timeframe


class TradeDecision(BaseModel):
    symbol: str
    should_trade: bool
    direction: Trend | None = None
    entry_price: float | None = None
    stop_loss: float | None = None
    take_profit: float | None = None
    is_market_order: bool = False  # True when entry is already through the BOS close
    reasoning: str
    confidence: str


class StrategySetup(BaseModel):
    input_data: str
    strategy_description: str
    direction: Trend
    htf_poi: str
    confirm_details: str
    target: str
    candles: str
    entry: float            # BOS level — used for RR feasibility check
    stop_loss: float
    bos_candle_close: float  # close of the LTF candle that confirmed BOS
    take_profit: float | None = None  # determined by the agent
