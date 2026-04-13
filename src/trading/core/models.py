import pandas as pd
from enum import Enum
from datetime import datetime
from pydantic import BaseModel, ConfigDict


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


class SignalType(str, Enum):
    FVG = "fvg"
    BOS = "bos"
    FRACTAL = "fractal"


class Candle(BaseModel):
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float


class FVG(BaseModel):
    timestamp: datetime
    top: float
    bottom: float
    trend: Trend
    timeframe: Timeframe


class BOS(BaseModel):
    timestamp: datetime
    level: float
    trend: Trend
    timeframe: Timeframe


class Fractal(BaseModel):
    timestamp: datetime
    price: float
    is_high: bool
    timeframe: Timeframe


class PointOfInterest(BaseModel):
    timestamp: datetime
    price_top: float
    price_bottom: float
    timeframe: Timeframe
    signal_type: SignalType
    trend: Trend
    description: str


class MarketState(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    symbol: str
    htf_timeframe: Timeframe
    ltf_timeframe: Timeframe
    htf_candles: pd.DataFrame
    ltf_candles: pd.DataFrame
    trend: Trend | None = None
    points_of_interest: list[PointOfInterest] = []
    fractals: list[Fractal] = []
    fvgs: list[FVG] = []
    bos_levels: list[BOS] = []
    htf_analysis: str | None = None
    trade_decision: str | None = None


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
