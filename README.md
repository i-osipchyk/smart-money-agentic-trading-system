# Smart Money Agentic Trading System

An AI-powered trading system that applies Smart Money Concepts through a strategy + agent architecture — strategies deterministically detect setups, Claude validates them.

## Why Smart Money

Smart Money Concepts (SMC) model market structure around institutional behavior — liquidity sweeps, order blocks, and inefficiencies left by large participants. I find it the most internally consistent framework for reading price action, and it's what I use in my own manual trading.

## Why Agentic

SMC can be implemented with deterministic rules, but context matters enormously. The same Fair Value Gap means something different in an uptrend versus a ranging market. Using an LLM agent to validate setups allows the system to reason about context rather than pattern-match against rigid conditions — making it more adaptive and easier to extend without rewriting logic.

## What the System Does

### Pipeline

1. A **Strategy** runs deterministic signal detection across HTF and LTF candles and returns a `StrategySetup` — a fully specified entry with pre-computed entry price, stop loss, and take profit.
2. The **TradeValidationAgent** receives the setup, builds a structured prompt, and asks Claude (or an OpenAI model) to assess whether the setup is valid given current market structure.
3. Claude returns a `TradeDecision`: trade or no trade, with direction, levels, confidence, and one-sentence reasoning.

### Current Strategy: HtfFvgLtfBos

| Step | Timeframe | Signal | Purpose |
|---|---|---|---|
| 1 | HTF (4H/1D) | Fair Value Gap (FVG) | Identifies a price inefficiency as the Point of Interest |
| 2 | LTF (15m/5m) | Break of Structure (BOS) | Confirms price is reacting from the HTF FVG |
| 3 | — | Levels | Entry at BOS level, SL beyond prior swing, TP at next FVG |

No LTF confirmation → no setup returned → agent not called.

### Data

All data access goes through a `DataSource` protocol — a typed interface any implementation must satisfy. Current implementations: CSV files (offline testing), Binance via `ccxt` (live/past), and a `BacktestDataSource` that pre-fetches a full historical window then streams `(datetime, htf_df, ltf_df)` snapshots candle-by-candle for backtesting.

Initial pairs: BTC/USDT and ETH/USDT on Binance.

## Architecture

```
CSVDataSource / BinanceDataSource / BacktestDataSource
        ↓
   Strategy.detect_entry()
   - Detects fractals + FVGs on HTF
   - Confirms BOS on LTF
   - Computes entry, SL, TP
   - Returns StrategySetup (or None)
        ↓
   Runner (OneTimeRunner / BacktestRunner)
   - prompt mode  → formats and displays the validation prompt
   - agent mode   → calls TradeValidationAgent → TradeDecision
   - baseline mode → simulates limit orders with fixed RR ratio
        ↓
   OrderSimulator  (backtest/baseline)
   - Tracks limit-order lifecycle candle by candle
   - Computes win rate, net R, and per-trade records
```

## GUI

Launch with `uv run trading-validate`. Two tabs share a common set of controls:

**Shared controls** (apply to both tabs):
- Symbol, HTF/LTF timeframe and candle count
- **Output Mode** — three options available in both tabs:
  - *Prompt Validation* — display the raw Claude prompt; no agent call
  - *Agent Test* — send the prompt to an LLM and display the decision
  - *Baseline Metrics* — simulate trades with a fixed RR ratio, no LLM
- Model selector (provider + model name) — active in Agent Test only
- Baseline Options (Order Timeout, Max Risk %, RR Ratio) — active in Baseline only

**One-Time Validation tab**: data source (Live / Past / CSV), optional until-datetime, CSV file pickers. In Baseline mode, future candles are fetched to evaluate the actual trade result (WIN / LOSS / CANCELED_PRICE / CANCELED_TIMEOUT / OPEN).

**Backtest tab**: date range From / To. Summary and metrics are shown in the GUI; full detail (prompts, agent responses, trade log) is saved to a file:

```
backtests/{mode}/{model_if_agent}/{strategy}/{symbol}/{params}/{from}_{to}.txt
```

## Live Deployment (AWS Lambda)

The system can run continuously on AWS Lambda, triggered on each LTF candle close by an EventBridge scheduled rule. Each Lambda instance runs one strategy on one symbol; adding more is a config change, not a code change.

All dependencies are packaged into a Docker image (no Lambda layers). One shared ECR repo serves all functions.

**First deployment:**
```bash
# Copy and fill in your values
cp stack-example.env stack-htf-fvg-ltf-bos-btc.env

# Create all AWS infrastructure and deploy
source stack-htf-fvg-ltf-bos-btc.env && ./setup_aws.sh

# Test the function manually
aws lambda invoke --function-name trading-signals-htf-fvg-ltf-bos-btc \
  --region us-east-1 /tmp/out.json && cat /tmp/out.json
```

**Deploy a code update** (updates all `trading-signals-*` functions at once):
```bash
AWS_ACCOUNT_ID=... AWS_REGION=... ./deploy.sh
```

**Prompt validation mode** (`MODE=prompt`): when a setup is detected, the raw Claude prompt is sent to a Telegram bot instead of being forwarded to Claude. Use this to verify the strategy is detecting correctly before enabling the full agent loop.

See [docs/adr/004-lambda-deployment-architecture.md](docs/adr/004-lambda-deployment-architecture.md) for the full design rationale.

## Roadmap

1. PoC — CSV data, signal detection validated ✓
2. Strategy + Validation Agent — deterministic setups + Claude validation ✓
3. Backtesting — run strategy over historical data, simulate orders, measure win rate and net R ✓
4. Live deployment — containerized Lambda, EventBridge trigger, Telegram notifications ✓
5. Paper trading — connect to Binance testnet, live data, simulated orders
6. Live trading — real orders with hard position size limits

## Tech Stack

| Layer | Tool | Why |
|---|---|---|
| Language | Python 3.13+ | Strong typing, modern enums, async support |
| LLM | Claude (Anthropic) or OpenAI via LangChain | Contextual reasoning for setup validation |
| Market data | `ccxt` + Binance API | Unified exchange interface, easy to swap |
| Data validation | Pydantic v2 | Runtime type safety for market data and strategy models |
| Packaging | `pyproject.toml` + `uv` | Modern, fast dependency management |
| Linting / types | `ruff` + `mypy` | Catch errors before runtime |

## How to Run

```bash
# Install dependencies
uv sync

# Add your API key(s) to .env
echo "ANTHROPIC_API_KEY=sk-ant-..." >> .env
echo "OPENAI_API_KEY=sk-..."        >> .env  # optional, for OpenAI models

# Launch the validation GUI
uv run trading-validate
```
