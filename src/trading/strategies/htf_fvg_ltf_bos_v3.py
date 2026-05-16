import logging
from dataclasses import dataclass
from datetime import datetime

import pandas as pd

from trading.core.models import FVG, Fractal, FvgStatus, StrategySetup, Timeframe, Trend
from trading.signals.fractals import detect_fractals
from trading.signals.fvg import detect_fvg
from trading.strategies.base import Strategy

logger = logging.getLogger(__name__)

_DESCRIPTION = """\
HTF FVG + LTF BOS v3 (active-only FVGs, no pre-computed levels)

Entry logic:
1. [HTF] Identify all Fair Value Gaps (FVGs) and classify each:
   - Active:      not yet touched by price after formation.
   - Tested:      price entered the FVG zone but closed on the correct side.
   - Invalidated: a subsequent candle closed through the far side of the gap.
   Only INVALIDATED FVGs are discarded — active and tested both proceed.
   (A "tested" FVG was touched by the qualifying swing itself; that touch is
   the signal, not a reason to discard.)

2. [LTF] Bullish setup — for each active bullish HTF FVG (most recent first):
   a. Find the lowest LTF swing low that formed after the FVG.
   b. The swing low must sit inside the FVG or within fvg_offset_pct below
      the FVG bottom (bottom × (1 − offset) ≤ price ≤ top); skip if not.
   c. Find the last LTF swing high before that swing low — this is the BOS level.
   d. The setup fires when the current (last) LTF candle is the first close
      above that BOS level.

3. [LTF] Bearish setup — mirror of the bullish:
   a. Find the highest LTF swing high that formed after the FVG.
   b. Must be inside the FVG or within fvg_offset_pct above the FVG top
      (bottom ≤ price ≤ top × (1 + offset)); skip if not.
   c. Find the last LTF swing low before that swing high — BOS level.
   d. Fires when the current candle is the first close below that BOS level.

No entry price, stop loss, or take profit is pre-computed — the agent determines
these levels based on the raw setup context.\
"""


@dataclass
class _EntrySignal:
    direction: Trend
    fvg: FVG
    swing_point: Fractal
    prior_swing: Fractal
    bos_candle_timestamp: datetime
    bos_level: float


class HtfFvgLtfBosV3(Strategy):
    """
    HTF FVG + LTF BOS strategy v3.

    Detects the same FVG + swing + BOS pattern as v1/v2 but restricts to ACTIVE
    FVGs only and returns no pre-computed entry, stop-loss, or take-profit levels.
    Designed for use with an AI agent that determines trade parameters from context.

    Args:
        fvg_offset_pct: Tolerance below (bullish) or above (bearish) the FVG edge
                        for qualifying the swing point. Default 0.005 (0.5%).
    """

    name = "htf_fvg_ltf_bos_v3"
    description = _DESCRIPTION

    def __init__(self, fvg_offset_pct: float = 0.005) -> None:
        self._fvg_offset_pct = fvg_offset_pct

    def detect_entry(
        self,
        symbol: str,
        htf_df: pd.DataFrame,
        htf_timeframe: Timeframe,
        ltf_df: pd.DataFrame,
        ltf_timeframe: Timeframe,
    ) -> StrategySetup | None:
        htf_fvgs = detect_fvg(htf_df, htf_timeframe)
        ltf_fractals = detect_fractals(ltf_df, ltf_timeframe)

        _log_findings(htf_fvgs, ltf_fractals)

        signal = _find_signal(htf_fvgs, ltf_fractals, ltf_df, self._fvg_offset_pct)
        if signal is None:
            return None

        return StrategySetup(
            input_data=_format_input_data(symbol, htf_df, htf_timeframe, ltf_df, ltf_timeframe),
            strategy_description=self.description,
            direction=signal.direction,
            htf_poi=_format_htf_poi(signal),
            confirm_details=_format_confirm_details(signal),
        )


# ------------------------------------------------------------------ internals

def _log_findings(htf_fvgs: list[FVG], ltf_fractals: list[Fractal]) -> None:
    active = [f for f in htf_fvgs if f.status == FvgStatus.ACTIVE]
    logger.info("HTF FVGs found: %d total, %d active", len(htf_fvgs), len(active))
    for fvg in active:
        logger.info(
            "  FVG [%s][active] top=%.2f bottom=%.2f formed=%s",
            fvg.trend.value,
            fvg.top,
            fvg.bottom,
            fvg.timestamp.strftime("%Y-%m-%d %H:%M"),
        )
    logger.info("LTF fractals found: %d", len(ltf_fractals))


def _find_signal(
    htf_fvgs: list[FVG],
    ltf_fractals: list[Fractal],
    ltf_df: pd.DataFrame,
    fvg_offset_pct: float,
) -> _EntrySignal | None:
    if not htf_fvgs or not ltf_fractals:
        return None

    last_candle_ts = ltf_df["timestamp"].iloc[-1]

    fractals_sorted = sorted(ltf_fractals, key=lambda f: f.timestamp)
    swing_lows = [f for f in fractals_sorted if not f.is_high]
    swing_highs = [f for f in fractals_sorted if f.is_high]

    valid_bullish = [f for f in htf_fvgs if f.trend == Trend.BULLISH and f.status != FvgStatus.INVALIDATED]
    valid_bearish = [f for f in htf_fvgs if f.trend == Trend.BEARISH and f.status != FvgStatus.INVALIDATED]

    # ---------------------------------------------------------------- bullish
    for fvg in reversed(valid_bullish):
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

    # ---------------------------------------------------------------- bearish
    for fvg in reversed(valid_bearish):
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


# ------------------------------------------------------------ formatters

def _fmt(p: float) -> str:
    return f"{p:,.2f}"


def _ts(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%d %H:%M")


def _format_input_data(
    symbol: str,
    htf_df: pd.DataFrame,
    htf_timeframe: Timeframe,
    ltf_df: pd.DataFrame,
    ltf_timeframe: Timeframe,
) -> str:
    return "\n".join([
        f"Symbol:     {symbol}",
        f"HTF:        {htf_timeframe.value}  ({len(htf_df)} candles)",
        f"LTF:        {ltf_timeframe.value}  ({len(ltf_df)} candles)",
        f"HTF range:  {_ts(htf_df['timestamp'].iloc[0])} → {_ts(htf_df['timestamp'].iloc[-1])}",
        f"LTF range:  {_ts(ltf_df['timestamp'].iloc[0])} → {_ts(ltf_df['timestamp'].iloc[-1])}",
    ])


def format_strategy_components(
    symbol: str,
    htf_df: pd.DataFrame,
    htf_timeframe: Timeframe,
    ltf_df: pd.DataFrame,
    ltf_timeframe: Timeframe,
    fvg_offset_pct: float = 0.005,
) -> str:
    """Return a full human-readable breakdown of all strategy components."""
    htf_fvgs = detect_fvg(htf_df, htf_timeframe)
    ltf_fractals = detect_fractals(ltf_df, ltf_timeframe)

    sep = "─" * 56
    lines: list[str] = []

    lines += [
        "STRATEGY INSPECTION — HTF FVG + LTF BOS v3",
        sep,
        _format_input_data(symbol, htf_df, htf_timeframe, ltf_df, ltf_timeframe),
        "",
    ]

    # ---- HTF FVGs -----------------------------------------------------------
    bullish_fvgs = [f for f in htf_fvgs if f.trend == Trend.BULLISH]
    bearish_fvgs = [f for f in htf_fvgs if f.trend == Trend.BEARISH]
    valid_b = [f for f in bullish_fvgs if f.status != FvgStatus.INVALIDATED]
    valid_be = [f for f in bearish_fvgs if f.status != FvgStatus.INVALIDATED]

    lines += [
        sep,
        f"HTF FVGs  ({len(htf_fvgs)} total:"
        f" {len(bullish_fvgs)} bullish [{len(valid_b)} valid],"
        f" {len(bearish_fvgs)} bearish [{len(valid_be)} valid])",
        "  (valid = active or tested; invalidated are dropped)",
        "",
    ]
    lines.append(f"  Bullish FVGs ({len(bullish_fvgs)})")
    for i, fvg in enumerate(bullish_fvgs, 1):
        dropped = fvg.status == FvgStatus.INVALIDATED
        marker = "  ✗ dropped" if dropped else "  ✓"
        lines.append(
            f"    {i}. bottom {_fmt(fvg.bottom)}  top {_fmt(fvg.top)}"
            f"  formed {_ts(fvg.timestamp)}  [{fvg.status.value}]{marker}"
        )
    if not bullish_fvgs:
        lines.append("    (none)")
    lines.append(f"  Bearish FVGs ({len(bearish_fvgs)})")
    for i, fvg in enumerate(bearish_fvgs, 1):
        dropped = fvg.status == FvgStatus.INVALIDATED
        marker = "  ✗ dropped" if dropped else "  ✓"
        lines.append(
            f"    {i}. bottom {_fmt(fvg.bottom)}  top {_fmt(fvg.top)}"
            f"  formed {_ts(fvg.timestamp)}  [{fvg.status.value}]{marker}"
        )
    if not bearish_fvgs:
        lines.append("    (none)")
    lines.append("")

    # ---- LTF Fractals -------------------------------------------------------
    fractals_sorted = sorted(ltf_fractals, key=lambda f: f.timestamp)
    ltf_highs = [f for f in fractals_sorted if f.is_high]
    ltf_lows = [f for f in fractals_sorted if not f.is_high]

    lines += [
        sep,
        f"LTF Fractals  ({len(ltf_fractals)} total:"
        f" {len(ltf_highs)} highs, {len(ltf_lows)} lows)",
        "",
    ]
    lines.append(f"  Swing Highs ({len(ltf_highs)})")
    for i, f in enumerate(ltf_highs, 1):
        lines.append(f"    {i}. {_fmt(f.price)}  at {_ts(f.timestamp)}")
    if not ltf_highs:
        lines.append("    (none)")
    lines.append(f"  Swing Lows ({len(ltf_lows)})")
    for i, f in enumerate(ltf_lows, 1):
        lines.append(f"    {i}. {_fmt(f.price)}  at {_ts(f.timestamp)}")
    if not ltf_lows:
        lines.append("    (none)")
    lines.append("")

    last_candle_ts = ltf_df["timestamp"].iloc[-1]

    # ---- Per-FVG BOS Analysis -----------------------------------------------
    lines += [sep, "Per-FVG BOS Analysis (active FVGs only)", ""]

    lines.append("  [BULLISH] Per active bullish FVG — lowest swing low after FVG, then BOS")
    if not valid_b:
        lines.append("    (no active bullish FVGs)")
    for fvg in reversed(valid_b):
        lines.append(
            f"    FVG [active]  bottom {_fmt(fvg.bottom)}  top {_fmt(fvg.top)}"
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

    lines.append("  [BEARISH] Per active bearish FVG — highest swing high after FVG, then BOS")
    if not valid_be:
        lines.append("    (no active bearish FVGs)")
    for fvg in reversed(valid_be):
        lines.append(
            f"    FVG [active]  bottom {_fmt(fvg.bottom)}  top {_fmt(fvg.top)}"
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

    # ---- Entry detection result ---------------------------------------------
    signal = _find_signal(htf_fvgs, ltf_fractals, ltf_df, fvg_offset_pct)
    lines += [sep, "ENTRY DETECTION RESULT", ""]
    if signal is None:
        lines.append("  No entry detected.")
    else:
        lines += [
            f"  Direction:    {signal.direction.value.upper()}",
            f"  FVG:          bottom {_fmt(signal.fvg.bottom)}  top {_fmt(signal.fvg.top)}",
            f"  Swing Point:  {_fmt(signal.swing_point.price)}"
            f"  at {_ts(signal.swing_point.timestamp)}",
            f"  BOS Level:    {_fmt(signal.bos_level)}"
            f"  confirmed at {_ts(signal.bos_candle_timestamp)}",
        ]

    return "\n".join(lines)


def _format_htf_poi(signal: _EntrySignal) -> str:
    return "\n".join([
        "HTF FVG (Point of Interest — ACTIVE)",
        f"  Top:    {_fmt(signal.fvg.top)}",
        f"  Bottom: {_fmt(signal.fvg.bottom)}",
        f"  Formed: {_ts(signal.fvg.timestamp)}",
    ])


def _format_confirm_details(signal: _EntrySignal) -> str:
    swing_label = (
        "Lowest LTF Swing Low" if signal.direction == Trend.BULLISH else "Highest LTF Swing High"
    )
    prior_label = (
        "Prior LTF Swing High" if signal.direction == Trend.BULLISH else "Prior LTF Swing Low"
    )
    bos_verb = "above" if signal.direction == Trend.BULLISH else "below"

    return "\n".join([
        f"{swing_label} (after FVG)",
        f"  Price:  {_fmt(signal.swing_point.price)}",
        f"  At:     {_ts(signal.swing_point.timestamp)}",
        "",
        f"{prior_label} (BOS level)",
        f"  Price:  {_fmt(signal.prior_swing.price)}",
        f"  At:     {_ts(signal.prior_swing.timestamp)}",
        "",
        "Break of Structure (BOS)",
        f"  Level:  {_fmt(signal.bos_level)}",
        f"  Closed {bos_verb} BOS at: {_ts(signal.bos_candle_timestamp)}",
    ])
