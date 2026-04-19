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
OpenAI models also supported — add `OPENAI_API_KEY` to `.env` when using agent mode with an OpenAI provider.

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
   TradeValidationAgent          ← agent mode only
   - Builds structured prompt from StrategySetup
   - Calls Claude (or OpenAI) to validate the setup
   - Parses response into TradeDecision
        ↓
   TradeDecision (should_trade, direction, entry, stop_loss, take_profit, reasoning, confidence)
```

### Live Deployment Pipeline (Lambda)

```
EventBridge cron (every LTF interval)
        ↓
   lambda_handler.handler()
   - Reads STRATEGY, SYMBOL, timeframes from env vars
   - Fetches live HTF + LTF candles via BinanceDataSource
   - Runs selected strategy from _STRATEGY_REGISTRY
        ↓  (only if setup found)
   MODE=prompt  →  TelegramNotifier.send_chunked(build_prompt(setup))
   MODE=agent   →  (not yet implemented)
```

To add a new strategy or symbol: run `setup_aws.sh` with different env vars. No code changes needed — the `STRATEGY` env var selects from `_STRATEGY_REGISTRY` in `lambda_handler.py`.

### Key Directories

```
src/trading/
├── core/              # Pydantic models + DataSource protocol
├── signals/           # Pure detection functions (FVG, Fractals)
├── strategies/        # Strategy base class + concrete strategies
├── agents/            # TradeValidationAgent + LLM provider abstraction
├── data/              # DataSource implementations (CSV, Binance, Backtest)
├── notifiers/         # TelegramNotifier (stdlib urllib, chunked sending)
├── runner/            # Non-GUI runner logic (see Runner Package below)
├── lambda_handler.py  # AWS Lambda entry point + strategy registry
└── gui_validation.py  # Tkinter GUI: one-time validation + backtest tabs

tests/unit/            # Unit tests for signal detectors
data/                  # Sample CSV files
backtests/             # Saved backtest outputs (see Backtest Output Paths)
docs/adr/              # Architecture Decision Records
Dockerfile             # Lambda container image
setup_aws.sh           # One-time AWS infrastructure setup per Lambda
deploy.sh              # Build + push image, update all Lambda functions
stack-*.env            # Per-deployment env var files (gitignored)
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
- **`TradeValidationAgent`** — wraps a LangChain chat model, exposes a single `run(prompt) -> str` method

### LLM Provider (src/trading/agents/llm_provider.py)

Multi-provider abstraction used by agent mode.

- **`LLMConfig`** — Pydantic model: `provider` + `model`
- **`create_llm_client(config)`** — returns a `BaseChatModel` (Anthropic or OpenAI)
- **`PROVIDERS`** — registry of valid provider/model combinations
- **`DEFAULT_CONFIG`** — `anthropic / claude-opus-4-5`

### Data Layer (src/trading/data/)

Implementations satisfy the `DataSource` protocol (structural subtyping — no inheritance required):

```python
class DataSource(Protocol):
    def get_ohlcv(self, symbol: str, timeframe: str, limit: int) -> pd.DataFrame: ...
```

- `CSVDataSource` — reads from local CSV files; used for offline testing
- `BinanceDataSource` — live/past data via `ccxt`
- `BacktestDataSource` — fetches a bulk historical window, then streams `(datetime, htf_df, ltf_df)` tuples candle-by-candle. Candle windows use `< current_dt` (exclusive upper bound) so they exactly match what one-time validation sees for the same timestamp and limit.

### Models (src/trading/core/models.py)

All data structures are Pydantic v2 `BaseModel` subclasses. Key types:
- `Timeframe` (enum), `Trend` (enum)
- `FVG`, `Fractal` — signal detector outputs
- `StrategySetup` — strategy output; input to `build_prompt()`
- `TradeDecision` — final output (should_trade, direction, entry, stop_loss, take_profit, reasoning, confidence)

### Runner Package (src/trading/runner/)

Non-GUI logic separated from tkinter. All classes accept callable output sinks so they are independently testable.

```
runner/
├── config.py     — RunConfig, TradeRecord, SimulationResult dataclasses; OutputMode / DataSourceType aliases
├── simulator.py  — OrderSimulator: limit-order lifecycle over BacktestDataSource
├── onetime.py    — OneTimeRunner: single snapshot, all three output modes
├── backtest.py   — BacktestRunner: full historical range, all three output modes
└── __init__.py   — re-exports all public names
```

**`RunConfig`** dataclass — all run parameters in one place:
- shared: `symbol`, `htf_tf`, `ltf_tf`, `htf_limit` (default 72), `ltf_limit` (default 24), `fvg_offset_pct`, `output_mode`
- one-time: `data_source` (`csv` / `past` / `live`), `htf_csv`, `ltf_csv`, `until`
- backtest: `bt_from`, `bt_to`
- agent: `llm_config`
- simulation: `order_timeout` (default 10), `max_risk_pct` (default 1.0%), `rr_ratio` (default 2.0)

**`OrderSimulator`** — simulates limit-order lifecycle:
- All verbose per-step lines go to a `detail_log` callback; returns `SimulationResult` only
- When a trade closes on candle X (SL, TP, or timeout), setup detection also runs on candle X for potential new trades
- Skips setups where `risk / entry > max_risk_pct` — doesn't block future setups

**`OneTimeRunner`** — fetches HTF+LTF for a single snapshot, runs strategy once:
- `prompt` mode: prints the full validation prompt to GUI
- `agent` mode: calls `TradeValidationAgent`, prints response to GUI
- `baseline` mode: computes TP via `entry ± risk × rr_ratio`, applies max-risk filter, then fetches up to 200 future LTF candles and evaluates the limit order (WIN / LOSS / CANCELED_PRICE / CANCELED_TIMEOUT / OPEN)

**`BacktestRunner`** — iterates `BacktestDataSource`:
- `prompt` mode: per-setup confirmation line to GUI, full prompt to detail file
- `agent` / `baseline` mode: `OrderSimulator` + per-trade summary rows to GUI, full detail to file; metrics printed to both

---

## GUI (src/trading/gui_validation.py)

Two tabs — **One-Time Validation** and **Backtest** — share all controls via `_build_shared_controls`:

- **Symbol / Timeframes / Limits** — same across both tabs
- **Output Mode** — three radio buttons: Prompt Validation / Agent Test / Baseline Metrics; applies to both tabs
- **Model** — provider + model dropdowns; active only in agent mode
- **Baseline Options** — Order Timeout (candles), Max Risk %, RR Ratio; shown only when output mode is Baseline; applies to both tabs

One-Time tab adds: data source selector (Live / Past / CSV), optional "Until" datetime, CSV file pickers.
Backtest tab adds: From / To datetime range.

Output channels:
- One-time: all output displayed in the GUI text area; nothing saved to file
- Backtest: summary + metrics → GUI; full detail (prompts / agent calls / trade lines) → file

### Backtest Output Paths

```
backtests/
└── {output_mode}/                    # prompt_validation | agent_test | baseline_metrics
    └── {provider}_{model}/           # agent mode only, e.g. anthropic_claude-opus-4-6
        └── {strategy}/               # e.g. htf_fvg_ltf_bos
            └── {symbol}/             # e.g. BTC-USDT-USDT  (/ and : replaced with -)
                └── {params}/         # e.g. fvg1pct_rr2_to10_risk1pct
                    └── {from}_{to}.txt
```

Params encoding per mode:
- prompt: `fvg{offset}pct`
- agent: `fvg{offset}pct_to{timeout}_risk{max_risk}pct`
- baseline: `fvg{offset}pct_rr{rr_ratio}_to{timeout}_risk{max_risk}pct`

---

## Code Conventions

- **Python 3.13+**, strict mypy, ruff linting (E/F/I/UP rules, 88-char line length)
- Pydantic v2 `BaseModel` for all data structures (runtime validation + serialization)
- `Protocol` for interface definitions (DataSource)
- `ABC` for strategy base class
- Pure functions for signal detection
- Private `_format_*()` helpers in strategies convert structured data to prompt text
- Enums use lowercase string values: `Trend.BULLISH = "bullish"`
- Runner classes accept `Callable[[str], None]` output sinks — no direct I/O or `sys.stdout` use

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
- **ADR-004**: AWS Lambda containerized deployment with strategy registry pattern

---

## Development Roadmap

The project follows a staged rollout:
1. PoC — CSV data, signal detection validated ✓
2. Strategy + Validation Agent — deterministic setups + Claude validation ✓
3. Backtesting — historical signal + agent/baseline simulation with PnL metrics ✓
4. Live deployment — containerized Lambda, EventBridge trigger, Telegram notifications ✓
5. Paper trading — live data, simulated orders
6. Live trading — real order execution
