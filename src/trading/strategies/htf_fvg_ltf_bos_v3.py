import logging

import pandas as pd

from trading.core.models import FvgStatus, StrategySetup, Timeframe, Trend
from trading.signals.fractals import detect_fractals
from trading.signals.fvg import detect_fvg
from trading.strategies._signal_utils import (
    _EntrySignal,
    _find_signal,
    _fmt,
    _format_input_data,
    _format_per_fvg_bos_analysis,
    _log_findings,
    _ts,
)
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

        valid_fvgs = [f for f in htf_fvgs if f.status != FvgStatus.INVALIDATED]
        signal = _find_signal(valid_fvgs, ltf_fractals, ltf_df, self._fvg_offset_pct)
        if signal is None:
            return None

        return StrategySetup(
            input_data=_format_input_data(symbol, htf_df, htf_timeframe, ltf_df, ltf_timeframe),
            strategy_description=self.description,
            direction=signal.direction,
            htf_poi=_format_htf_poi(signal),
            confirm_details=_format_confirm_details(signal),
        )


# ------------------------------------------------------------ formatters

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
    for label, fvg_list in [("Bullish", bullish_fvgs), ("Bearish", bearish_fvgs)]:
        lines.append(f"  {label} FVGs ({len(fvg_list)})")
        for i, fvg in enumerate(fvg_list, 1):
            marker = "  ✗ dropped" if fvg.status == FvgStatus.INVALIDATED else "  ✓"
            lines.append(
                f"    {i}. bottom {_fmt(fvg.bottom)}  top {_fmt(fvg.top)}"
                f"  formed {_ts(fvg.timestamp)}  [{fvg.status.value}]{marker}"
            )
        if not fvg_list:
            lines.append("    (none)")
    lines.append("")

    fractals_sorted = sorted(ltf_fractals, key=lambda f: f.timestamp)
    ltf_highs = [f for f in fractals_sorted if f.is_high]
    ltf_lows = [f for f in fractals_sorted if not f.is_high]

    lines += [
        sep,
        f"LTF Fractals  ({len(ltf_fractals)} total:"
        f" {len(ltf_highs)} highs, {len(ltf_lows)} lows)",
        "",
    ]
    for label, fractal_list in [("Swing Highs", ltf_highs), ("Swing Lows", ltf_lows)]:
        lines.append(f"  {label} ({len(fractal_list)})")
        for i, f in enumerate(fractal_list, 1):
            lines.append(f"    {i}. {_fmt(f.price)}  at {_ts(f.timestamp)}")
        if not fractal_list:
            lines.append("    (none)")
    lines.append("")

    last_candle_ts = ltf_df["timestamp"].iloc[-1]

    lines += [sep, "Per-FVG BOS Analysis (active FVGs only)", ""]
    lines += _format_per_fvg_bos_analysis(
        valid_b, valid_be, ltf_highs, ltf_lows,
        ltf_df, last_candle_ts, fvg_offset_pct,
    )

    valid_fvgs = [f for f in htf_fvgs if f.status != FvgStatus.INVALIDATED]
    signal = _find_signal(valid_fvgs, ltf_fractals, ltf_df, fvg_offset_pct)
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
