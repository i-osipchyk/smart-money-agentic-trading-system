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


class FvgStatus(str, Enum):
    ACTIVE = "active"
    TESTED = "tested"
    INVALIDATED = "invalidated"


class FVG(BaseModel):
    timestamp: datetime
    top: float
    bottom: float
    trend: Trend
    timeframe: Timeframe
    status: FvgStatus = FvgStatus.ACTIVE


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
    entry: float
    stop_loss: float
    take_profit: float
    detected_at: datetime | None = None
