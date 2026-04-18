# ADR-004: AWS Lambda Containerized Deployment

## Status
Accepted

## Context

The system was previously run exclusively via a local Tkinter GUI (`trading-validate`). To operate in live regime it needs to run continuously, triggered on each LTF candle close, without requiring a local machine to be running.

Two primary concerns shaped the deployment design:

1. **Continuous execution** — the system must fire automatically at each candle interval (e.g. every 15 minutes for a 15m LTF), fetch fresh data, run the strategy, and notify when a setup is detected.
2. **Extensibility** — the live system will eventually host multiple strategies and symbols concurrently. The deployment model must accommodate this without duplication.

## Decision

Deploy as **containerized AWS Lambda functions** triggered by EventBridge scheduled rules.

### Container image, not layers

All dependencies (`ccxt`, `pandas`, `langchain-anthropic`, etc.) are packaged into a single Docker image pushed to ECR. This avoids Lambda layer size limits, makes dependency management deterministic, and keeps the deployment unit self-contained.

One shared ECR repository (`trading-signals`) serves all Lambda functions. The image is the same binary for every deployment; configuration drives behavior.

### One Lambda per strategy + symbol combination

Each deployed instance is a separate Lambda function named `trading-signals-{strategy_slug}-{symbol_slug}` (e.g. `trading-signals-htf-fvg-ltf-bos-btc`). All configuration is injected as environment variables:

| Variable | Purpose |
|---|---|
| `STRATEGY` | Registry key selecting the strategy (e.g. `htf_fvg_ltf_bos`) |
| `SYMBOL` | ccxt perpetual futures symbol (e.g. `BTC/USDT:USDT`) |
| `HTF_TIMEFRAME` / `LTF_TIMEFRAME` | Timeframe pair for the strategy |
| `HTF_LIMIT` / `LTF_LIMIT` | Rolling candle window sizes |
| `FVG_OFFSET_SPINUNITS` | FVG offset tolerance |
| `MODE` | `prompt` or `agent` |
| `TELEGRAM_BOT_TOKEN` / `TELEGRAM_CHAT_ID` | Telegram notification credentials |

Adding a new strategy or symbol means running the one-time `setup_aws.sh` script with different variable values — no code changes and no new Docker image required.

### Strategy registry pattern

The Lambda handler maintains a `_STRATEGY_REGISTRY` dict that maps string keys to factory lambdas. The `STRATEGY` env var selects which strategy to instantiate at cold start. This keeps the handler generic and makes adding a new strategy a single-line change in one file.

```python
_STRATEGY_REGISTRY: dict[str, Callable[[float], Strategy]] = {
    "htf_fvg_ltf_bos": lambda offset: HtfFvgLtfBos(fvg_offset_pct=offset),
}
```

### Deployment scripts

Two shell scripts handle the full lifecycle:

- **`setup_aws.sh`** — one-time infrastructure setup (ECR repo, IAM role, Lambda function, EventBridge rule). Idempotent; safe to re-run.
- **`deploy.sh`** — builds and pushes a new image, then updates every `trading-signals-*` Lambda function in the account to use it. A single command redeploys all live instances after a code change.

### Prompt validation mode

Before enabling the full AI agent loop in production, the system operates in `MODE=prompt`. When a setup is detected, the raw `build_prompt()` output is sent to a Telegram bot rather than being forwarded to Claude. This lets the operator inspect exactly what the agent would see and validate that the strategy is detecting correctly before incurring API costs or acting on agent decisions.

The `TelegramNotifier` uses only Python stdlib (`urllib.request`) to avoid adding a new dependency. Long prompts (which include full OHLCV tables) are split at newline boundaries into ≤4096-character chunks to respect Telegram's message limit.

### Shared infrastructure

- One ECR repository across all Lambda functions
- One IAM execution role (`trading-signals-lambda-role`) across all Lambda functions
- One EventBridge rule per Lambda (named `trading-signals-{strategy_slug}-{symbol_slug}-schedule`)

## Consequences

**Positive:**
- No persistent infrastructure; cost scales to zero when no setups are detected
- Adding a new strategy or symbol is a config change, not a deployment change
- `MODE=prompt` provides a safe validation step before enabling live agent decisions
- `deploy.sh` updates all instances atomically — no per-function deploy commands
- Cold start (~3-5s) is well within the 120s Lambda timeout; no warm-up strategy needed at this invocation frequency

**Negative:**
- Lambda has a 15-minute maximum execution timeout — unsuitable for long backtests or blocking order management loops; those remain in the local GUI
- Module-level singletons in the handler (data source, strategy, notifier) are reused across warm invocations, which is intentional for performance but means configuration changes require a re-deploy (or function update) to take effect
- `MODE=agent` is not yet implemented in the Lambda handler; attempting to use it raises `NotImplementedError`
