from pathlib import Path

from dotenv import load_dotenv

from trading.agents import build_graph
from trading.core.models import MarketState, Timeframe
from trading.data import BinanceDataSource, CSVDataSource

load_dotenv()


def run() -> None:
    print("=== Smart Money Agentic Trading System ===\n")

    source = CSVDataSource(data_dir=Path("data"))

    htf_candles = source.get_ohlcv(symbol="BTC/USDT", timeframe="1h", limit=100)
    ltf_candles = source.get_ohlcv(symbol="BTC/USDT", timeframe="1h", limit=50)

    print(f"Loaded {len(htf_candles)} HTF candles")
    print(f"Loaded {len(ltf_candles)} LTF candles\n")

    state = MarketState(
        symbol="BTC/USDT",
        htf_timeframe=Timeframe.D1,
        ltf_timeframe=Timeframe.D1,
        htf_candles=htf_candles,
        ltf_candles=ltf_candles,
    )

    graph = build_graph()

    print("--- Running HTF Agent ---")
    result = graph.invoke(state)

    print(f"Trend:    {result['trend']}")
    print(f"POIs found: {len(result['points_of_interest'])}")
    print(f"HTF Analysis: {result['htf_analysis']}\n")

    if result["trade_decision"]:
        print("--- LTF Agent Decision ---")
        print(result["trade_decision"])
    else:
        print("--- LTF Agent ---")
        print("No POIs found on HTF. No trade setup.")
