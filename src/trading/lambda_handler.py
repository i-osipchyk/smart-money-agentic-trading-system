"""
AWS Lambda handler for the trading signal system.

Triggered by EventBridge on each LTF candle close. Fetches live data from
Binance, runs the configured strategy, and in MODE=prompt sends the raw
validation prompt to Telegram for manual review.

Environment variables
---------------------
STRATEGY            Strategy key from the registry (default: htf_fvg_ltf_bos)
SYMBOL              ccxt perpetual futures symbol, e.g. BTC/USDT:USDT
HTF_TIMEFRAME       Higher timeframe value, e.g. 1h
LTF_TIMEFRAME       Lower timeframe value, e.g. 15m
HTF_LIMIT           Number of HTF candles to fetch (default: 72)
LTF_LIMIT           Number of LTF candles to fetch (default: 24)
FVG_OFFSET_SPINUNITS  Integer offset units (default: 10 → 0.01 = 1%)
MODE                Execution mode: prompt | agent (default: prompt)
TELEGRAM_BOT_TOKEN  Telegram bot token (required in prompt mode)
TELEGRAM_CHAT_ID    Telegram chat/user ID (required in prompt mode)
ANTHROPIC_API_KEY   Claude API key (required in agent mode only)
"""

import logging
import os
from collections.abc import Callable
from datetime import UTC, datetime

from trading.agents.trade_validation_agent import build_prompt
from trading.core.models import StrategySetup, Timeframe
from trading.data.binance_datasource import BinanceDataSource
from trading.notifiers.telegram import TelegramNotifier
from trading.strategies import HtfFvgLtfBos
from trading.strategies.base import Strategy

logging.getLogger().setLevel(logging.INFO)  # root — covers all trading.* loggers
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Strategy registry — add new strategies here as they are implemented
# ---------------------------------------------------------------------------
_STRATEGY_REGISTRY: dict[str, Callable[[float], Strategy]] = {
    "htf_fvg_ltf_bos": lambda offset: HtfFvgLtfBos(fvg_offset_pct=offset),
}

# ---------------------------------------------------------------------------
# Configuration — read once at cold start, validated immediately
# ---------------------------------------------------------------------------
_STRATEGY_NAME = os.environ.get("STRATEGY", "htf_fvg_ltf_bos")
if _STRATEGY_NAME not in _STRATEGY_REGISTRY:
    raise ValueError(
        f"Unknown STRATEGY={_STRATEGY_NAME!r}. "
        f"Valid values: {list(_STRATEGY_REGISTRY)}"
    )

_SYMBOL = os.environ["SYMBOL"]
_HTF_TF = Timeframe(os.environ["HTF_TIMEFRAME"])
_LTF_TF = Timeframe(os.environ["LTF_TIMEFRAME"])
_HTF_LIMIT = int(os.environ.get("HTF_LIMIT", "72"))
_LTF_LIMIT = int(os.environ.get("LTF_LIMIT", "24"))
_FVG_OFFSET = int(os.environ.get("FVG_OFFSET_SPINUNITS", "10")) / 1000.0
_MODE = os.environ.get("MODE", "prompt")

_TG_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "")
_TG_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID", "")

# ---------------------------------------------------------------------------
# Module-level singletons — reused across warm invocations
# ---------------------------------------------------------------------------
_datasource = BinanceDataSource()
_strategy = _STRATEGY_REGISTRY[_STRATEGY_NAME](_FVG_OFFSET)
_notifier = TelegramNotifier(token=_TG_TOKEN, chat_id=_TG_CHAT_ID) if _TG_TOKEN else None


def handler(event: dict, context: object) -> dict:
    """
    Lambda entry point.

    Returns {"status": "ok", "setup_detected": bool} on success.
    Errors in notification re-raise so Lambda marks the invocation as failed
    and CloudWatch can alarm. "No setup detected" is a normal outcome.
    """
    invocation_time = datetime.now(UTC)
    logger.info(
        "Invoked strategy=%s symbol=%s htf=%s ltf=%s mode=%s at=%s",
        _STRATEGY_NAME,
        _SYMBOL,
        _HTF_TF.value,
        _LTF_TF.value,
        _MODE,
        invocation_time.isoformat(),
    )

    htf_df = _datasource.get_ohlcv(_SYMBOL, _HTF_TF.value, _HTF_LIMIT)
    ltf_df = _datasource.get_ohlcv(_SYMBOL, _LTF_TF.value, _LTF_LIMIT)

    # Drop the last candle if it is still forming (its close time is in the future).
    # Candle duration is derived from the data to avoid hardcoding the interval.
    candle_duration = ltf_df["timestamp"].iloc[-1] - ltf_df["timestamp"].iloc[-2]
    last_candle_close = ltf_df["timestamp"].iloc[-1] + candle_duration
    if invocation_time < last_candle_close.to_pydatetime().replace(tzinfo=UTC):
        ltf_df = ltf_df.iloc[:-1]
        logger.info("Dropped incomplete LTF candle (closes at %s)", last_candle_close)

    setup = _strategy.detect_entry(_SYMBOL, htf_df, _HTF_TF, ltf_df, _LTF_TF)

    if setup is None:
        logger.info("No setup detected for %s", _SYMBOL)
        return {"status": "ok", "setup_detected": False}

    logger.info(
        "Setup detected: direction=%s entry=%.2f sl=%.2f tp=%.2f",
        setup.direction.value,
        setup.entry,
        setup.stop_loss,
        setup.take_profit,
    )

    if _MODE == "prompt":
        _handle_prompt_mode(setup, invocation_time)
    elif _MODE == "agent":
        _handle_agent_mode(setup)
    else:
        logger.warning("Unknown MODE=%r, skipping notification", _MODE)

    return {
        "status": "ok",
        "setup_detected": True,
        "direction": setup.direction.value,
    }


def _handle_prompt_mode(setup: StrategySetup, invocation_time: datetime) -> None:
    """Send the raw validation prompt to Telegram for manual review."""
    if _notifier is None:
        logger.error("TELEGRAM_BOT_TOKEN not set; cannot send prompt")
        raise RuntimeError("TELEGRAM_BOT_TOKEN is required in prompt mode")

    header = (
        f"SIGNAL — {_SYMBOL} [{_STRATEGY_NAME}]\n"
        f"Direction : {setup.direction.value.upper()}\n"
        f"Entry     : {setup.entry:,.2f}\n"
        f"Stop Loss : {setup.stop_loss:,.2f}\n"
        f"Take Prof.: {setup.take_profit:,.2f}\n"
        f"Time (UTC): {invocation_time.strftime('%Y-%m-%d %H:%M')}\n"
        f"{'─' * 40}\n\n"
    )
    full_message = header + build_prompt(setup)

    _notifier.send_chunked(full_message)
    logger.info("Prompt sent to Telegram (%d chars total)", len(full_message))


def _handle_agent_mode(setup: StrategySetup) -> None:
    """Call Claude API and send decision to Telegram. Not yet implemented."""
    raise NotImplementedError("Agent mode is not yet implemented in Lambda")
