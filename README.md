# Smart Money Agentic Trading System

An AI-powered trading system that applies Smart Money Concepts through a two-agent LangGraph architecture — one agent for macro context, one for trade execution decisions.

## Why Smart Money

Smart Money Concepts (SMC) model market structure around institutional behavior — liquidity sweeps, order blocks, and inefficiencies left by large participants. I find it the most internally consistent framework for reading price action, and it's what I use in my own manual trading.

## Why Agentic

SMC can be implemented with deterministic rules, but context matters enormously. The same Fair Value Gap means something different in an uptrend versus a ranging market. Using LLM agents allows the system to reason about context rather than pattern-match against rigid conditions — making it more adaptive and easier to extend without rewriting logic.

## What the System Does

### Timeframes

- HTF (4H / 1D): determines market structure, trend direction, and Points of Interest (POI). Can incorporate external context such as macro news and sentiment.
- LTF (15m / 5m): looks for confirmation of HTF thesis, and defines the specific entry point, Stop Loss (SL), and Take Profit (TP). A secondary lower timeframe is used if the primary does not provide enough signal.

If HTF finds a valid POI, the system hands off to LTF to confirm or discard the setup. No confirmation → no trade.

### Core Concepts (PoC)

| Concept | Used on | Purpose |
|---|---|---|
| `FVG` — Fair Value Gap | HTF + LTF | Identifies price inefficiencies as POIs and entry zones |
| `BOS` — Break of Structure | LTF | Confirms trend continuation or reversal |
| Fractals | HTF + LTF | Locates swing highs/lows to define structure |

### Data

All data access goes through a DataSource protocol — a typed interface that any implementation must satisfy. Swapping Binance for a CSV file (or any other source) requires no changes to agent logic.

Initial pairs: BTC/USDT and ETH/USDT on Binance.

## Architecture

### PoC Architecture

![image](pod_architecture.svg)

- DataSource Interface allows to connect different data sources without changing any internal structure of the application.
- HTF Data is fed into HTF Agent, that decides context and looks for a POI that aligns with this context
- The output is passed to MarketState, and consequently fed to LTF Agent, together with LTF Data
- LTF Agent decides on orders with a reasoning

### Production Architecture

TBD

## Roadmap

1. PoC — two-agent loop on hand-picked historical examples. Goal: validate the FVG + BOS logic produces sensible decisions. Output is plain text.
2. Backtesting — run the full two-agent system over a validation period of BTC/ETH data. Build a structured trade log with PnL, win rate, and risk metrics.
3. Paper trading — connect to Binance testnet. System runs live but places no real orders. Monitor latency and decision quality.
4. Live trading — real orders with hard position size limits. Risk management layer added before this phase.
5. Continuous improvement — agents review past trade logs and surface patterns. Concept library expanded (order blocks, liquidity, CHoCH).

## Tech Stack (PoC)

| Layer | Tool | Why |
|---|---|---|
| Language | Python 3.11+ | Strong typing, dataclasses, async support |
| Agent orchestration | LangGraph | Stateful multi-agent graphs with explicit handoffs |
| LLM | Claude (Anthropic) | Reasoning and contextual analysis |
| Market data | `ccxt` + Binance API | Unified exchange interface, easy to swap |
| Data validation | Pydantic v2 | Runtime type safety for market data models |
| Packaging | `pyproject.toml` + `uv` | Modern, fast dependency management |
| Linting / types | `ruff` + `mypy` | Catch errors before runtime |

## How to Run

TBD
