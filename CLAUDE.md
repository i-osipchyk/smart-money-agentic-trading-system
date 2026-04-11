# CLAUDE.md — Smart Money Agentic Trading System

## Project Overview

A cryptocurrency trading system that applies **Smart Money Concepts (SMC)** using a two-agent LangGraph architecture powered by Claude AI. The system analyzes market structure on higher timeframes (HTF: 4H/1D) and confirms trade setups on lower timeframes (LTF: 15m/5m) to generate trade decisions with entry, stop loss, and take profit levels.

Currently targets BTC/USDT and ETH/USDT on Binance.

---

## Commands

### Package Manager: `uv`

```bash
# Install dependencies
uv sync

# Run the main entry point
uv run trading main

# Run all unit tests
uv run pytest tests/unit/

# Run all tests
uv run pytest tests/

# Run linter
uv run ruff check src/ tests/

# Run type checker
uv run mypy src/

# Run smoke test (validates data sources)
uv run python scripts/smoke_test.py
```

### Environment

Requires an `ANTHROPIC_API_KEY` in `.env` at the project root (loaded via `python-dotenv`).

---

## Architecture

### Two-Agent LangGraph Pipeline

```
CSVDataSource / BinanceDataSource
        ↓
   HTF Agent (4H/1D)
   - Detects fractals, FVGs, BOS
   - Identifies Points of Interest (POIs)
   - Determines macro trend via Claude
        ↓  (conditional: only if POIs found)
   LTF Agent (15m/5m)
   - Detects signals relative to HTF context
   - Validates trade setup via Claude
   - Generates TradeDecision
        ↓
   TradeDecision (entry, stop_loss, take_profit, reasoning)
```

State flows through a `MarketState` Pydantic model shared across the graph.

### Key Directories

```
src/trading/
├── core/           # Pydantic models + DataSource protocol
├── signals/        # Pure detection functions (FVG, BOS, Fractals)
├── agents/         # LangGraph graph + HTF/LTF agent nodes
├── data/           # DataSource implementations (CSV, Binance)
└── main.py         # Entry point / demo runner

tests/unit/         # Unit tests for signal detectors
data/               # Sample CSV files for PoC
docs/adr/           # Architecture Decision Records
```

---

## Core Concepts

### Signal Detectors (src/trading/signals/)

Pure functions: accept a pandas DataFrame + `Timeframe`, return a list of Pydantic models.

- **`detect_fractals(df, timeframe)`** — swing highs/lows using a window-based peak/valley approach
- **`detect_fvg(df, timeframe)`** — Fair Value Gaps (3-candle price inefficiencies)
- **`detect_bos(df, fractals, timeframe)`** — Break of Structure at prior fractal levels

### Data Layer (src/trading/data/)

Implementations satisfy the `DataSource` protocol (structural subtyping — no inheritance required):

```python
class DataSource(Protocol):
    def get_candles(self, symbol: str, timeframe: Timeframe, limit: int) -> pd.DataFrame: ...
```

- `CSVDataSource` — reads from local CSV files; used for PoC and offline testing
- `BinanceDataSource` — live data via `ccxt`

### Models (src/trading/core/models.py)

All data structures are Pydantic v2 `BaseModel` subclasses. Key types:
- `Candle`, `Timeframe` (enum), `Trend` (enum)
- `FVG`, `BOS`, `Fractal`, `PointOfInterest`
- `MarketState` — shared LangGraph state (HTF/LTF candles, signals, POIs, trend, trade decision)
- `TradeDecision` — final output (entry, stop_loss, take_profit, reasoning)

### Agents (src/trading/agents/)

- `htf_agent.py` — formats signals into a prompt, calls Claude, parses trend + POIs from text response
- `ltf_agent.py` — formats HTF context + LTF signals into a prompt, calls Claude, parses `TradeDecision`
- `graph.py` — `StateGraph` wiring: HTF node → conditional edge → LTF node or END

State updates follow the LangGraph pattern:
```python
return MarketState(**{**state.model_dump(), **new_fields})
```

---

## Code Conventions

- **Python 3.13+**, strict mypy, ruff linting (E/F/I/UP rules, 88-char line length)
- Pydantic v2 `BaseModel` for all data structures (runtime validation + serialization)
- `Protocol` for interface definitions (DataSource)
- Pure functions for signal detection; agent functions take and return `MarketState`
- Private `_format_*()` helpers convert structured data to LLM prompt text
- LLM responses are parsed line-by-line with prefix matching (fragile — handle gracefully)
- Enums use lowercase string values: `Trend.BULLISH = "bullish"`

---

## Testing

Tests in `tests/unit/` use pytest with helper functions that build minimal DataFrames for known patterns. There are no mocks — tests validate detector logic directly against synthetic price data.

Integration tests directory exists but is currently empty (reserved for future work).

---

## Architecture Decisions

See `docs/adr/` for rationale behind key design choices:
- **ADR-001**: Two-agent architecture (HTF + LTF) with `MarketState` as the communication channel
- **ADR-002**: `DataSource` protocol for pluggable data sources

---

## Development Roadmap

The project follows a staged rollout:
1. PoC (current) — CSV data, demo output
2. Backtesting — historical signal validation
3. Paper trading — live data, simulated orders
4. Live trading — real order execution
