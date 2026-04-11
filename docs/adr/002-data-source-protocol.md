# ADR-002: DataSource Protocol for Market Data Access

## Status
Accepted

## Context
The system needs market data in multiple contexts:

- **PoC phase** — hand-picked historical examples loaded from CSV files
- **Backtesting phase** — larger historical datasets, still from files
- **Live phase** — real-time data from Binance via API

Without an abstraction, agent code would need to know whether it is talking to a file or an exchange. This creates tight coupling: changing the data source requires changes inside agent logic, which is unrelated to data fetching.

## Decision
Define a `DataSource` protocol in `core/protocols.py`. Any class that implements this protocol can be used anywhere in the system without code changes.

The protocol specifies one primary method:

```python
def get_ohlcv(self, symbol: str, timeframe: str, limit: int) -> pd.DataFrame:
    ...
```

Two implementations will be built:
- `CSVDataSource` — reads from local CSV files, used for PoC and backtesting
- `BinanceDataSource` — wraps `ccxt`, used for backtesting, paper and live trading

Agents depend only on the protocol, never on a concrete implementation. The correct implementation is injected at startup via configuration.

## Consequences

**Positive:**
- Swapping Binance for any other exchange is a one-file change
- The entire system can run offline using CSV files — no API keys needed for development
- Each implementation can be tested in isolation with known data
- Adding a new data source (e.g. Bybit, Interactive Brokers) requires no changes to agent code

**Negative:**
- The protocol must be kept minimal — adding exchange-specific methods breaks the abstraction
- CSV data must be pre-formatted to match what the Binance implementation returns, or normalisation becomes the protocol's responsibility
