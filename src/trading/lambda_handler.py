"""
AWS Lambda handler for the trading signal system.

Triggered by EventBridge on each LTF candle close. Fetches live data from
the configured data source, runs the strategy, and sends a notification.

Environment variables
---------------------
STRATEGY            Strategy key from the registry (default: htf_fvg_ltf_bos)
SYMBOL              Exchange symbol, e.g. BTC/USDT:USDT (Binance) or XAUUSD (cTrader)
HTF_TIMEFRAME       Higher timeframe value, e.g. 4h
LTF_TIMEFRAME       Lower timeframe value, e.g. 1h
HTF_LIMIT           Number of HTF candles to fetch (default: 72)
LTF_LIMIT           Number of LTF candles to fetch (default: 24)
FVG_OFFSET_SPINUNITS  Integer offset units (default: 10 → 0.001 = 0.1 %)
                      For v3 default 0.5 % use 5 (5/1000 = 0.005)
MODE                Execution mode: alert | prompt | agent (default: alert)
DATA_PROVIDER       Data source: binance | ctrader (default: binance)

Binance credentials (DATA_PROVIDER=binance — no auth required, public API)

cTrader credentials (DATA_PROVIDER=ctrader — all required):
CTRADER_CLIENT_ID       cTrader Open API application client ID
CTRADER_CLIENT_SECRET   cTrader Open API application client secret
CTRADER_ACCESS_TOKEN    OAuth access token for the trading account
CTRADER_ACCOUNT_ID      cTrader account ID (integer)

Telegram (required for all modes):
TELEGRAM_BOT_TOKEN  Telegram bot token
TELEGRAM_CHAT_ID    Telegram chat or user ID

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
from trading.strategies import HtfFvgLtfBos, HtfFvgLtfBosV2, HtfFvgLtfBosV3
from trading.strategies.base import Strategy

logging.getLogger().setLevel(logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Strategy registry
# ---------------------------------------------------------------------------
_STRATEGY_REGISTRY: dict[str, Callable[[float], Strategy]] = {
    "htf_fvg_ltf_bos":    lambda offset: HtfFvgLtfBos(fvg_offset_pct=offset),
    "htf_fvg_ltf_bos_v2": lambda offset: HtfFvgLtfBosV2(fvg_offset_pct=offset),
    "htf_fvg_ltf_bos_v3": lambda offset: HtfFvgLtfBosV3(fvg_offset_pct=offset),
}

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
_STRATEGY_NAME = os.environ.get("STRATEGY", "htf_fvg_ltf_bos")
if _STRATEGY_NAME not in _STRATEGY_REGISTRY:
    raise ValueError(
        f"Unknown STRATEGY={_STRATEGY_NAME!r}. "
        f"Valid values: {list(_STRATEGY_REGISTRY)}"
    )

_SYMBOL        = os.environ["SYMBOL"]
_HTF_TF        = Timeframe(os.environ["HTF_TIMEFRAME"])
_LTF_TF        = Timeframe(os.environ["LTF_TIMEFRAME"])
_HTF_LIMIT     = int(os.environ.get("HTF_LIMIT", "72"))
_LTF_LIMIT     = int(os.environ.get("LTF_LIMIT", "24"))
_FVG_OFFSET    = int(os.environ.get("FVG_OFFSET_SPINUNITS", "10")) / 1000.0
_MODE          = os.environ.get("MODE", "alert")
_DATA_PROVIDER = os.environ.get("DATA_PROVIDER", "binance")

_TG_TOKEN   = os.environ.get("TELEGRAM_BOT_TOKEN", "")
_TG_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID", "")

# ---------------------------------------------------------------------------
# Data source
# ---------------------------------------------------------------------------
if _DATA_PROVIDER == "ctrader":
    from trading.data.ctrader_datasource import CTraderDataSource
    _datasource: BinanceDataSource | CTraderDataSource = CTraderDataSource(
        client_id=os.environ["CTRADER_CLIENT_ID"],
        client_secret=os.environ["CTRADER_CLIENT_SECRET"],
        access_token=os.environ["CTRADER_ACCESS_TOKEN"],
        account_id=int(os.environ["CTRADER_ACCOUNT_ID"]),
    )
else:
    _datasource = BinanceDataSource()

# ---------------------------------------------------------------------------
# Singletons
# ---------------------------------------------------------------------------
_strategy = _STRATEGY_REGISTRY[_STRATEGY_NAME](_FVG_OFFSET)
_notifier = TelegramNotifier(token=_TG_TOKEN, chat_id=_TG_CHAT_ID) if _TG_TOKEN else None


def handler(event: dict[str, object], context: object) -> dict[str, object]:
    """
    Lambda entry point.

    Returns {"status": "ok", "setup_detected": bool} on success.
    """
    invocation_time = datetime.now(UTC)
    logger.info(
        "Invoked strategy=%s symbol=%s htf=%s ltf=%s mode=%s provider=%s at=%s",
        _STRATEGY_NAME, _SYMBOL, _HTF_TF.value, _LTF_TF.value,
        _MODE, _DATA_PROVIDER, invocation_time.isoformat(),
    )

    htf_df = _datasource.get_ohlcv(_SYMBOL, _HTF_TF.value, _HTF_LIMIT)
    ltf_df = _datasource.get_ohlcv(_SYMBOL, _LTF_TF.value, _LTF_LIMIT)

    # Drop the last candle if it is still forming.
    candle_duration = ltf_df["timestamp"].iloc[-1] - ltf_df["timestamp"].iloc[-2]
    last_candle_close = ltf_df["timestamp"].iloc[-1] + candle_duration
    if invocation_time < last_candle_close.to_pydatetime().replace(tzinfo=UTC):
        ltf_df = ltf_df.iloc[:-1]
        logger.info("Dropped incomplete LTF candle (closes at %s)", last_candle_close)

    setup = _strategy.detect_entry(_SYMBOL, htf_df, _HTF_TF, ltf_df, _LTF_TF)

    if setup is None:
        logger.info("No setup detected for %s", _SYMBOL)
        return {"status": "ok", "setup_detected": False}

    if setup.entry is not None:
        logger.info(
            "Setup detected: direction=%s entry=%.2f sl=%.2f tp=%.2f",
            setup.direction.value, setup.entry,
            setup.stop_loss or 0.0, setup.take_profit or 0.0,
        )
    else:
        logger.info("Setup detected: direction=%s (no pre-computed levels)", setup.direction.value)

    if _MODE == "alert":
        _handle_alert_mode(setup, invocation_time)
    elif _MODE == "prompt":
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


def _handle_alert_mode(setup: StrategySetup, invocation_time: datetime) -> None:
    """Send a concise signal alert to Telegram."""
    if _notifier is None:
        logger.error("TELEGRAM_BOT_TOKEN not set; cannot send alert")
        raise RuntimeError("TELEGRAM_BOT_TOKEN is required")

    sep = "─" * 32
    lines = [
        f"SIGNAL  {_SYMBOL}",
        f"Direction : {setup.direction.value.upper()}",
        f"Time (UTC): {invocation_time.strftime('%Y-%m-%d %H:%M')}",
        sep,
        setup.htf_poi,
        "",
        setup.confirm_details,
    ]
    if setup.entry is not None:
        lines += [
            sep,
            f"Entry     : {setup.entry:,.2f}",
            f"Stop Loss : {setup.stop_loss:,.2f}" if setup.stop_loss else "",
            f"Take Prof.: {setup.take_profit:,.2f}" if setup.take_profit else "",
        ]

    message = "\n".join(l for l in lines if l is not None)
    _notifier.send_chunked(message)
    logger.info("Alert sent to Telegram (%d chars)", len(message))


def _handle_prompt_mode(setup: StrategySetup, invocation_time: datetime) -> None:
    """Send the raw validation prompt to Telegram for manual review."""
    if _notifier is None:
        logger.error("TELEGRAM_BOT_TOKEN not set; cannot send prompt")
        raise RuntimeError("TELEGRAM_BOT_TOKEN is required in prompt mode")

    header_lines = [
        f"SIGNAL — {_SYMBOL} [{_STRATEGY_NAME}]",
        f"Direction : {setup.direction.value.upper()}",
    ]
    if setup.entry is not None:
        header_lines += [
            f"Entry     : {setup.entry:,.2f}",
            f"Stop Loss : {setup.stop_loss:,.2f}" if setup.stop_loss else "",
            f"Take Prof.: {setup.take_profit:,.2f}" if setup.take_profit else "",
        ]
    header_lines += [
        f"Time (UTC): {invocation_time.strftime('%Y-%m-%d %H:%M')}",
        "─" * 40,
        "",
    ]
    header = "\n".join(l for l in header_lines if l is not None) + "\n"
    full_message = header + build_prompt(setup)

    _notifier.send_chunked(full_message)
    logger.info("Prompt sent to Telegram (%d chars total)", len(full_message))


def _handle_agent_mode(setup: StrategySetup) -> None:
    """Call Claude API and send decision to Telegram. Not yet implemented."""
    raise NotImplementedError("Agent mode is not yet implemented in Lambda")
