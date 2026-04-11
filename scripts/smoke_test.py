from pathlib import Path
from trading.data import BinanceDataSource, CSVDataSource

def main() -> None:
    print("--- CSV ---")
    csv = CSVDataSource(data_dir=Path("data"))
    df_csv = csv.get_ohlcv(symbol="BTC/USDT", timeframe="1d", limit=5)
    print(df_csv.to_string())

    print("\n--- Binance ---")
    binance = BinanceDataSource()
    df_binance = binance.get_ohlcv(symbol="BTC/USDT", timeframe="1d", limit=5)
    print(df_binance.to_string())

if __name__ == "__main__":
    main()
    