import json
from pathlib import Path

from dotenv import load_dotenv

from trading.agents import build_graph
from trading.core.models import MarketState, Timeframe
from trading.data import BinanceDataSource, CSVDataSource

load_dotenv()


def _print_separator(title: str) -> None:
    print(f"\n{'='*50}")
    print(f"  {title}")
    print(f"{'='*50}")


def run() -> None:
    _print_separator("Smart Money Agentic Trading System")

    source = CSVDataSource(data_dir=Path("data"))
    htf_candles = source.get_ohlcv(symbol="BTC/USDT", timeframe="1h", limit=100)
    ltf_candles = source.get_ohlcv(symbol="BTC/USDT", timeframe="1h", limit=50)

    print(f"\nSymbol:       BTC/USDT")
    print(f"HTF candles:  {len(htf_candles)} x 1d")
    print(f"LTF candles:  {len(ltf_candles)} x 1d")
    print(f"HTF range:    {htf_candles['timestamp'].iloc[0].date()} → {htf_candles['timestamp'].iloc[-1].date()}")

    state = MarketState(
        symbol="BTC/USDT",
        htf_timeframe=Timeframe.D1,
        ltf_timeframe=Timeframe.D1,
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
            print(f"  {i}. {poi.signal_type.value.upper()} | {poi.trend.value.upper()} | "
                  f"{poi.price_bottom:.2f} - {poi.price_top:.2f}")
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
