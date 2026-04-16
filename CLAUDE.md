# CLAUDE.md — Smart Money Agentic Trading System

## Project Overview

A cryptocurrency trading system that applies **Smart Money Concepts (SMC)** using a strategy + agent architecture powered by Claude AI. Strategies deterministically detect trade setups from market data; the `TradeValidationAgent` uses Claude to pressure-test each setup and produce a final `TradeDecision` with entry, stop loss, and take profit levels.

Currently targets BTC/USDT and ETH/USDT on Binance.

---

## Commands

### Package Manager: `uv`

```bash
# Install dependencies
uv sync

# Launch the validation GUI
uv run trading-validate

# Run all unit tests
uv run pytest tests/unit/

# Run all tests
uv run pytest tests/

# Run linter
uv run ruff check src/ tests/

# Run type checker
uv run mypy src/
```

### Environment

Requires an `ANTHROPIC_API_KEY` in `.env` at the project root (loaded via `python-dotenv`).

---

## Architecture

### Strategy + Validation Pipeline

```
CSVDataSource / BinanceDataSource / BacktestDataSource
        ↓
   Strategy.detect_entry()  (e.g. HtfFvgLtfBos)
   - Detects fractals + FVGs on HTF
   - Confirms BOS on LTF relative to HTF FVG
   - Computes entry, stop loss, take profit
   - Returns StrategySetup (or None — no setup)
        ↓  (only if setup found)
   TradeValidationAgent
   - Builds structured prompt from StrategySetup
   - Calls Claude to validate the setup
   - Parses response into TradeDecision
        ↓
   TradeDecision (should_trade, direction, entry, stop_loss, take_profit, reasoning, confidence)
```

### Key Directories

```
src/trading/
├── core/           # Pydantic models + DataSource protocol
├── signals/        # Pure detection functions (FVG, Fractals)
├── strategies/     # Strategy base class + concrete strategies
├── agents/         # TradeValidationAgent (prompt builder + Claude caller)
├── data/           # DataSource implementations (CSV, Binance, Backtest)
└── gui_validation.py  # Tkinter GUI: one-time validation + backtest tabs

tests/unit/         # Unit tests for signal detectors
data/               # Sample CSV files
backtests/          # Saved backtest outputs
docs/adr/           # Architecture Decision Records
```

---

## Core Concepts

### Signal Detectors (src/trading/signals/)

Pure functions: accept a pandas DataFrame + `Timeframe`, return a list of Pydantic models.

- **`detect_fractals(df, timeframe)`** — swing highs/lows using a window-based peak/valley approach
- **`detect_fvg(df, timeframe)`** — Fair Value Gaps (3-candle price inefficiencies)

### Strategies (src/trading/strategies/)

Concrete strategies extend the `Strategy` ABC. Each encapsulates its own detection logic and configuration parameters.

```python
class Strategy(ABC):
    name: str
    description: str

    def detect_entry(
        self, symbol, htf_df, htf_timeframe, ltf_df, ltf_timeframe
    ) -> StrategySetup | None: ...
```

- **`HtfFvgLtfBos`** — identifies a bullish/bearish FVG on HTF, then looks for a BOS on LTF that confirms price is reacting from that zone. Returns a `StrategySetup` with pre-computed entry, SL, and TP levels.

### Trade Validation Agent (src/trading/agents/trade_validation_agent.py)

- **`build_prompt(setup)`** — formats a `StrategySetup` into a structured Claude prompt
- **`parse_decision(symbol, response, setup)`** — extracts `TradeDecision` from the agent's response (parses a `decision` code fence; falls back to keyword search)
- **`TradeValidationAgent`** — wraps `ChatAnthropic`, exposes a single `run(prompt) -> str` method

### Data Layer (src/trading/data/)

Implementations satisfy the `DataSource` protocol (structural subtyping — no inheritance required):

```python
class DataSource(Protocol):
    def get_ohlcv(self, symbol: str, timeframe: str, limit: int) -> pd.DataFrame: ...
```

- `CSVDataSource` — reads from local CSV files; used for offline testing
- `BinanceDataSource` — live/past data via `ccxt`
- `BacktestDataSource` — fetches a bulk historical window, then streams (HTF slice, LTF slice) pairs candle-by-candle for backtesting

### Models (src/trading/core/models.py)

All data structures are Pydantic v2 `BaseModel` subclasses. Key types:
- `Timeframe` (enum), `Trend` (enum)
- `FVG`, `Fractal` — signal detector outputs
- `StrategySetup` — strategy output; input to `build_prompt()`
- `TradeDecision` — final output (should_trade, direction, entry, stop_loss, take_profit, reasoning, confidence)

---

## Code Conventions

- **Python 3.13+**, strict mypy, ruff linting (E/F/I/UP rules, 88-char line length)
- Pydantic v2 `BaseModel` for all data structures (runtime validation + serialization)
- `Protocol` for interface definitions (DataSource)
- `ABC` for strategy base class
- Pure functions for signal detection
- Private `_format_*()` helpers in strategies convert structured data to prompt text
- Enums use lowercase string values: `Trend.BULLISH = "bullish"`

---

## Testing

Tests in `tests/unit/` use pytest with helper functions that build minimal DataFrames for known patterns. There are no mocks — tests validate detector logic directly against synthetic price data.

Integration tests directory exists but is currently empty (reserved for future work).

---

## Architecture Decisions

See `docs/adr/` for rationale behind key design choices:
- **ADR-001**: Original two-agent architecture (HTF + LTF) — superseded
- **ADR-002**: `DataSource` protocol for pluggable data sources
- **ADR-003**: Strategy + TradeValidationAgent replacing the two-agent LangGraph pipeline

---

## Development Roadmap

The project follows a staged rollout:
1. PoC — CSV data, signal detection validated ✓
2. Strategy + Validation Agent — deterministic setups + Claude validation (current)
3. Backtesting — historical signal + agent validation with PnL metrics (in progress)
4. Paper trading — live data, simulated orders
5. Live trading — real order execution
