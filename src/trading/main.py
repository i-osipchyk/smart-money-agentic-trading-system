import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv

from trading.agents import build_graph
from trading.core.models import MarketState, Timeframe
from trading.data import BinanceDataSource, CSVDataSource

load_dotenv()

DataSourceType = Literal["csv", "past", "live"]

_TF_SECONDS: dict[str, int] = {
    "5m": 300,
    "15m": 900,
    "1h": 3600,
    "4h": 14400,
    "1d": 86400,
}


def _print_separator(title: str) -> None:
    print(f"\n{'='*50}")
    print(f"  {title}")
    print(f"{'='*50}")


def run(
    *,
    symbol: str = "BTC/USDT:USDT",
    htf_timeframe: Timeframe = Timeframe.H1,
    ltf_timeframe: Timeframe = Timeframe.M15,
    data_source: DataSourceType = "csv",
    htf_csv: str | None = None,
    ltf_csv: str | None = None,
    until: datetime | None = None,
    htf_limit: int = 72,
    ltf_limit: int = 24,
    data_dir: Path = Path("data"),
) -> None:
    _print_separator("Smart Money Agentic Trading System")

    if data_source == "csv":
        source = CSVDataSource(data_dir=data_dir)
        htf_candles = source.get_ohlcv(
            symbol=symbol,
            timeframe=htf_timeframe.value,
            limit=htf_limit,
            filename_override=htf_csv,
        )
        ltf_candles = source.get_ohlcv(
            symbol=symbol,
            timeframe=ltf_timeframe.value,
            limit=ltf_limit,
            filename_override=ltf_csv,
        )
    else:
        binance = BinanceDataSource()

        if data_source == "past" and until is not None:
            htf_since = until - timedelta(seconds=htf_limit * _TF_SECONDS[htf_timeframe.value])
            ltf_since = until - timedelta(seconds=ltf_limit * _TF_SECONDS[ltf_timeframe.value])
        else:
            htf_since = None
            ltf_since = None

        htf_candles = binance.get_ohlcv(
            symbol=symbol,
            timeframe=htf_timeframe.value,
            limit=htf_limit,
            since=htf_since,
        )
        ltf_candles = binance.get_ohlcv(
            symbol=symbol,
            timeframe=ltf_timeframe.value,
            limit=ltf_limit,
            since=ltf_since,
        )

    print(f"\nSymbol:       {symbol}")
    print(f"HTF candles:  {len(htf_candles)} x {htf_timeframe.value}")
    print(f"LTF candles:  {len(ltf_candles)} x {ltf_timeframe.value}")
    print(f"HTF range:    {htf_candles['timestamp'].iloc[0].date()} → {htf_candles['timestamp'].iloc[-1].date()}")

    state = MarketState(
        symbol=symbol,
        htf_timeframe=htf_timeframe,
        ltf_timeframe=ltf_timeframe,
        htf_candles=htf_candles,
        ltf_candles=ltf_candles,
    )

    graph = build_graph()
    result = graph.invoke(state)

    _print_separator("HTF Agent — Market Structure")
    print(f"\nTrend:     {result['trend']}")
    print(f"POIs:      {len(result['points_of_interest'])} identified")
    print(f"\nAnalysis:  {result['htf_analysis']}")

    if result["points_of_interest"]:
        print("\nPoints of Interest:")
        for i, poi in enumerate(result["points_of_interest"], 1):
            print(
                f"  {i}. {poi.signal_type.value.upper()} | {poi.trend.value.upper()} | "
                f"{poi.price_bottom:.2f} - {poi.price_top:.2f}"
            )
            print(f"     {poi.description}")

    _print_separator("LTF Agent — Trade Decision")

    if result["trade_decision"]:
        decision = json.loads(result["trade_decision"])
        should_trade = decision["should_trade"]
        print(f"\nDecision:   {'TRADE' if should_trade else 'NO TRADE'}")

        if should_trade:
            print(f"Direction:  {decision['direction']}")
            print(f"Entry:      {decision['entry_price']}")
            print(f"Stop Loss:  {decision['stop_loss']}")
            print(f"Take Profit:{decision['take_profit']}")

        print(f"Confidence: {decision['confidence']}")
        print(f"\nReasoning:  {decision['reasoning']}")
    else:
        print("\nNo POIs found on HTF. No trade setup.")

    print()


__all__ = ["run", "DataSourceType"]
