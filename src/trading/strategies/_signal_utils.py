"""Shared utilities for the HTF FVG + LTF BOS strategy family."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Literal

import pandas as pd

from trading.core.models import FVG, Fractal, FvgStatus, Timeframe, Trend

logger = logging.getLogger(__name__)


def _fmt(p: float) -> str:
    return f"{p:,.2f}"


def _ts(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%d %H:%M")


@dataclass
class _EntrySignal:
    direction: Trend
    fvg: FVG
    swing_point: Fractal
    prior_swing: Fractal
    bos_candle_timestamp: datetime
    bos_level: float


def _log_findings(htf_fvgs: list[FVG], ltf_fractals: list[Fractal]) -> None:
    logger.info("HTF FVGs found: %d", len(htf_fvgs))
    for fvg in htf_fvgs:
        logger.info(
            "  FVG [%s][%s] top=%.2f bottom=%.2f formed=%s",
            fvg.trend.value,
            fvg.status.value,
            fvg.top,
            fvg.bottom,
            fvg.timestamp.strftime("%Y-%m-%d %H:%M"),
        )
    logger.info("LTF fractals found: %d", len(ltf_fractals))
    for f in ltf_fractals:
        logger.info(
            "  Fractal [%s] price=%.2f at=%s",
            "high" if f.is_high else "low",
            f.price,
            f.timestamp.strftime("%Y-%m-%d %H:%M"),
        )


def _find_signal(
    htf_fvgs: list[FVG],
    ltf_fractals: list[Fractal],
    ltf_df: pd.DataFrame,
    fvg_offset_pct: float = 0.0,
) -> _EntrySignal | None:
    """Find a BOS entry signal from pre-filtered HTF FVGs and LTF fractals.

    htf_fvgs should already have any unwanted statuses removed by the caller.
    """
    if not htf_fvgs or not ltf_fractals:
        return None

    last_candle_ts = ltf_df["timestamp"].iloc[-1]
    fractals_sorted = sorted(ltf_fractals, key=lambda f: f.timestamp)
    swing_lows = [f for f in fractals_sorted if not f.is_high]
    swing_highs = [f for f in fractals_sorted if f.is_high]
    bullish_fvgs = [f for f in htf_fvgs if f.trend == Trend.BULLISH]
    bearish_fvgs = [f for f in htf_fvgs if f.trend == Trend.BEARISH]

    for fvg in reversed(bullish_fvgs):
        lows_after = [lo for lo in swing_lows if lo.timestamp > fvg.timestamp]
        if not lows_after:
            continue
        swing_low = min(lows_after, key=lambda f: f.price)
        if not (fvg.bottom * (1 - fvg_offset_pct) <= swing_low.price <= fvg.top):
            continue
        prior_highs = [h for h in swing_highs if h.timestamp < swing_low.timestamp]
        if not prior_highs:
            continue
        prior_swing_high = prior_highs[-1]
        if prior_swing_high.price <= swing_low.price:
            continue
        candles_after = ltf_df[ltf_df["timestamp"] > swing_low.timestamp]
        bos_rows = candles_after[candles_after["close"] > prior_swing_high.price]
        if bos_rows.empty:
            continue
        bos_candle = bos_rows.iloc[0]
        if bos_candle["timestamp"] != last_candle_ts:
            continue
        return _EntrySignal(
            direction=Trend.BULLISH,
            fvg=fvg,
            swing_point=swing_low,
            prior_swing=prior_swing_high,
            bos_candle_timestamp=bos_candle["timestamp"],
            bos_level=prior_swing_high.price,
        )

    for fvg in reversed(bearish_fvgs):
        highs_after = [h for h in swing_highs if h.timestamp > fvg.timestamp]
        if not highs_after:
            continue
        swing_high = max(highs_after, key=lambda f: f.price)
        if not (fvg.bottom <= swing_high.price <= fvg.top * (1 + fvg_offset_pct)):
            continue
        prior_lows = [lo for lo in swing_lows if lo.timestamp < swing_high.timestamp]
        if not prior_lows:
            continue
        prior_swing_low = prior_lows[-1]
        if prior_swing_low.price >= swing_high.price:
            continue
        candles_after = ltf_df[ltf_df["timestamp"] > swing_high.timestamp]
        bos_rows = candles_after[candles_after["close"] < prior_swing_low.price]
        if bos_rows.empty:
            continue
        bos_candle = bos_rows.iloc[0]
        if bos_candle["timestamp"] != last_candle_ts:
            continue
        return _EntrySignal(
            direction=Trend.BEARISH,
            fvg=fvg,
            swing_point=swing_high,
            prior_swing=prior_swing_low,
            bos_candle_timestamp=bos_candle["timestamp"],
            bos_level=prior_swing_low.price,
        )

    return None


def _has_blocking_fvg(
    htf_fvgs: list[FVG],
    direction: Trend,
    entry: float,
    take_profit: float,
    block_fvg_mode: Literal["none", "active", "tested", "active+tested"] = "active",
) -> bool:
    if block_fvg_mode == "none":
        return False

    def _qualifies(fvg: FVG) -> bool:
        if block_fvg_mode == "active":
            return fvg.status == FvgStatus.ACTIVE
        if block_fvg_mode == "tested":
            return fvg.status == FvgStatus.TESTED
        return fvg.status in (FvgStatus.ACTIVE, FvgStatus.TESTED)

    if direction == Trend.BULLISH:
        return any(
            fvg.trend == Trend.BEARISH and _qualifies(fvg)
            and fvg.bottom < take_profit and fvg.top > entry
            for fvg in htf_fvgs
        )
    return any(
        fvg.trend == Trend.BULLISH and _qualifies(fvg)
        and fvg.top > take_profit and fvg.bottom < entry
        for fvg in htf_fvgs
    )


def _format_input_data(
    symbol: str,
    htf_df: pd.DataFrame,
    htf_timeframe: Timeframe,
    ltf_df: pd.DataFrame,
    ltf_timeframe: Timeframe,
    htf_end_idx: int = -1,
) -> str:
    return "\n".join([
        f"Symbol:     {symbol}",
        f"HTF:        {htf_timeframe.value}  ({len(htf_df)} candles)",
        f"LTF:        {ltf_timeframe.value}  ({len(ltf_df)} candles)",
        f"HTF range:  {_ts(htf_df['timestamp'].iloc[0])} → {_ts(htf_df['timestamp'].iloc[htf_end_idx])}",
        f"LTF range:  {_ts(ltf_df['timestamp'].iloc[0])} → {_ts(ltf_df['timestamp'].iloc[-1])}",
    ])


def _format_per_fvg_bos_analysis(
    bullish_fvgs: list[FVG],
    bearish_fvgs: list[FVG],
    ltf_highs: list[Fractal],
    ltf_lows: list[Fractal],
    ltf_df: pd.DataFrame,
    last_candle_ts: pd.Timestamp,
    fvg_offset_pct: float,
) -> list[str]:
    """Build per-FVG BOS analysis lines for strategy inspection output."""
    lines: list[str] = []

    lines.append("  [BULLISH] Per bullish FVG — lowest swing low after FVG, then BOS")
    if not bullish_fvgs:
        lines.append("    (no bullish FVGs)")
    for fvg in reversed(bullish_fvgs):
        lines.append(
            f"    FVG [{fvg.status.value}]  bottom {_fmt(fvg.bottom)}  top {_fmt(fvg.top)}"
            f"  formed {_ts(fvg.timestamp)}"
        )
        lows_after = [lo for lo in ltf_lows if lo.timestamp > fvg.timestamp]
        if not lows_after:
            lines.append("      Lowest swing low: (none after FVG)")
            lines.append("")
            continue
        swing_low = min(lows_after, key=lambda f: f.price)
        lines.append(f"      Lowest Swing Low: {_fmt(swing_low.price)}  at {_ts(swing_low.timestamp)}")
        zone_low = fvg.bottom * (1 - fvg_offset_pct)
        if not (zone_low <= swing_low.price <= fvg.top):
            lines.append(
                f"      Not in qualifying zone ({_fmt(zone_low)}–{_fmt(fvg.top)}) — skip"
            )
            lines.append("")
            continue
        prior_highs = [h for h in ltf_highs if h.timestamp < swing_low.timestamp]
        if not prior_highs:
            lines.append("      Prior swing high (BOS level): (none)")
        else:
            prior = prior_highs[-1]
            lines.append(
                f"      Prior Swing High (BOS level): {_fmt(prior.price)}"
                f"  at {_ts(prior.timestamp)}"
            )
            if prior.price <= swing_low.price:
                lines.append("      BOS level below swing low — invalid — skip")
                lines.append("")
                continue
            candles_after = ltf_df[ltf_df["timestamp"] > swing_low.timestamp]
            bos_rows = candles_after[candles_after["close"] > prior.price]
            if bos_rows.empty:
                lines.append("      BOS: not confirmed")
            else:
                bos_ts = bos_rows.iloc[0]["timestamp"]
                marker = "  ← ENTRY SIGNAL" if bos_ts == last_candle_ts else "  — signal expired"
                lines.append(f"      BOS: {_ts(bos_ts)}{marker}")
        lines.append("")

    lines.append("  [BEARISH] Per bearish FVG — highest swing high after FVG, then BOS")
    if not bearish_fvgs:
        lines.append("    (no bearish FVGs)")
    for fvg in reversed(bearish_fvgs):
        lines.append(
            f"    FVG [{fvg.status.value}]  bottom {_fmt(fvg.bottom)}  top {_fmt(fvg.top)}"
            f"  formed {_ts(fvg.timestamp)}"
        )
        highs_after = [h for h in ltf_highs if h.timestamp > fvg.timestamp]
        if not highs_after:
            lines.append("      Highest swing high: (none after FVG)")
            lines.append("")
            continue
        swing_high = max(highs_after, key=lambda f: f.price)
        lines.append(f"      Highest Swing High: {_fmt(swing_high.price)}  at {_ts(swing_high.timestamp)}")
        zone_high = fvg.top * (1 + fvg_offset_pct)
        if not (fvg.bottom <= swing_high.price <= zone_high):
            lines.append(
                f"      Not in qualifying zone ({_fmt(fvg.bottom)}–{_fmt(zone_high)}) — skip"
            )
            lines.append("")
            continue
        prior_lows = [lo for lo in ltf_lows if lo.timestamp < swing_high.timestamp]
        if not prior_lows:
            lines.append("      Prior swing low (BOS level): (none)")
        else:
            prior = prior_lows[-1]
            lines.append(
                f"      Prior Swing Low (BOS level): {_fmt(prior.price)}"
                f"  at {_ts(prior.timestamp)}"
            )
            if prior.price >= swing_high.price:
                lines.append("      BOS level above swing high — invalid — skip")
                lines.append("")
                continue
            candles_after = ltf_df[ltf_df["timestamp"] > swing_high.timestamp]
            bos_rows = candles_after[candles_after["close"] < prior.price]
            if bos_rows.empty:
                lines.append("      BOS: not confirmed")
            else:
                bos_ts = bos_rows.iloc[0]["timestamp"]
                marker = "  ← ENTRY SIGNAL" if bos_ts == last_candle_ts else "  — signal expired"
                lines.append(f"      BOS: {_ts(bos_ts)}{marker}")
        lines.append("")

    return lines
