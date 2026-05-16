import logging
from typing import Literal

import pandas as pd

from trading.core.models import FVG, Fractal, StrategySetup, Timeframe, Trend
from trading.signals.fractals import detect_fractals
from trading.signals.fvg import detect_fvg
from trading.strategies._signal_utils import (
    _EntrySignal,
    _find_signal,
    _fmt,
    _format_input_data,
    _format_per_fvg_bos_analysis,
    _has_blocking_fvg,
    _log_findings,
    _ts,
)
from trading.strategies.base import Strategy

logger = logging.getLogger(__name__)

_DESCRIPTION = """\
HTF FVG + LTF BOS (Fair Value Gap with Break of Structure confirmation)

Entry logic:
1. [HTF] Identify all Fair Value Gaps (FVGs) and classify each:
   - Active:      not yet touched by price after formation.
   - Tested:      price entered the FVG zone but closed on the correct side.
   - Invalidated: a subsequent candle closed through the far side of the gap
                  (below bottom for bullish, above top for bearish).
   Invalidated FVGs are discarded.

2. [LTF] Bullish setup — for each active/tested bullish HTF FVG (most recent first):
   a. Collect all LTF candles formed after the FVG.
   b. Find the lowest LTF swing low in that window.
   c. The swing low must be within fvg_offset_pct below the FVG bottom
      (bottom × (1 − offset) ≤ price ≤ top); skip if not.
   d. Find the last LTF swing high before that swing low — this is the BOS level.
   e. The setup fires only when the current (last) LTF candle is the first close
      above that BOS level, confirming the bullish directional shift.
   f. If no BOS on the current candle, skip to the next FVG.

3. [LTF] Bearish setup — mirror of the bullish setup:
   a. Collect all LTF candles formed after the FVG.
   b. Find the highest LTF swing high in that window.
   c. The swing high must be within fvg_offset_pct above the FVG top
      (bottom ≤ price ≤ top × (1 + offset)); skip if not.
   d. Find the last LTF swing low before that swing high — this is the BOS level.
   e. The setup fires when the current LTF candle is the first close below the BOS level.
   f. If no BOS on the current candle, skip to the next FVG.

4. [LTF] Stop Loss:
    - For bullish entries: exactly at the lowest swing low price.
    - For bearish entries: exactly at the highest swing high price.

5. [LTF] Entry:
    Limit order placed at the BOS level (the prior swing that was broken).

6. Take Profit:
    2:1 reward/risk relative to entry and stop loss.

7. Trend filter:
    HTF fractals determine the macro trend: higher highs + higher lows = bullish,
    lower highs + lower lows = bearish, mixed = no bias.
    A setup is discarded when the signal direction conflicts with the HTF trend.
    When HTF structure is mixed (None), the setup is allowed through.

8. Path filter:
    If any active opposing-direction HTF FVG sits between entry and take profit,
    the setup is discarded — supply/demand zones on the path are likely to reject
    price before the target is reached.
    When block_tested_fvgs is enabled, tested opposing FVGs also block the setup.

The HTF FVG is the Point of Interest (POI) / demand or supply zone.
The LTF swing point marks the liquidity sweep near that zone.
The BOS confirms that smart money has absorbed liquidity and is pushing price.\
"""


class HtfFvgLtfBos(Strategy):
    """
    HTF FVG + LTF BOS strategy.

    Args:
        fvg_offset_pct:  Extends the qualifying zone beyond the FVG edge.
                         Default 0.0005 (0.05 %).
        block_fvg_mode:  Which opposing FVGs block the path filter.
        use_trend_filter: Discard setups that conflict with HTF trend.
    """

    name = "htf_fvg_ltf_bos"
    description = _DESCRIPTION

    def __init__(
        self,
        fvg_offset_pct: float = 0.0005,
        block_fvg_mode: Literal["none", "active", "tested", "active+tested"] = "active",
        use_trend_filter: bool = True,
    ) -> None:
        self._fvg_offset_pct = fvg_offset_pct
        self._block_fvg_mode = block_fvg_mode
        self._use_trend_filter = use_trend_filter

    def detect_entry(
        self,
        symbol: str,
        htf_df: pd.DataFrame,
        htf_timeframe: Timeframe,
        ltf_df: pd.DataFrame,
        ltf_timeframe: Timeframe,
    ) -> StrategySetup | None:
        htf_fvgs = detect_fvg(htf_df, htf_timeframe)
        htf_fractals = detect_fractals(htf_df, htf_timeframe)
        ltf_fractals = detect_fractals(ltf_df, ltf_timeframe)

        _log_findings(htf_fvgs, ltf_fractals)

        signal = _find_signal(htf_fvgs, ltf_fractals, ltf_df, self._fvg_offset_pct)
        if signal is None:
            return None

        if self._use_trend_filter:
            htf_trend = trend_from_fractals(htf_fractals)
            if htf_trend is not None and htf_trend != signal.direction:
                return None

        entry, stop_loss, take_profit = _compute_levels(signal)

        if _has_blocking_fvg(htf_fvgs, signal.direction, entry, take_profit, self._block_fvg_mode):
            return None

        return StrategySetup(
            input_data=_format_input_data(symbol, htf_df, htf_timeframe, ltf_df, ltf_timeframe),
            strategy_description=self.description,
            direction=signal.direction,
            htf_poi=_format_htf_poi(signal),
            confirm_details=_format_confirm_details(signal),
            target=_format_target(htf_fvgs, htf_fractals, signal.direction),
            candles=_format_candles(htf_df, htf_timeframe),
            entry=entry,
            stop_loss=stop_loss,
            take_profit=take_profit,
        )


# ------------------------------------------------------------------ internals

def trend_from_fractals(fractals: list[Fractal]) -> Trend | None:
    highs = sorted([f for f in fractals if f.is_high], key=lambda f: f.timestamp)
    lows = sorted([f for f in fractals if not f.is_high], key=lambda f: f.timestamp)
    if len(highs) < 2 or len(lows) < 2:
        return None
    if highs[-1].price > highs[-2].price and lows[-1].price > lows[-2].price:
        return Trend.BULLISH
    if highs[-1].price < highs[-2].price and lows[-1].price < lows[-2].price:
        return Trend.BEARISH
    return None


def _compute_levels(signal: _EntrySignal) -> tuple[float, float, float]:
    entry = signal.bos_level
    stop_loss = signal.swing_point.price
    if signal.direction == Trend.BULLISH:
        take_profit = entry + 2 * (entry - stop_loss)
    else:
        take_profit = entry - 2 * (stop_loss - entry)
    return entry, stop_loss, take_profit


# ------------------------------------------------------------ formatters

def _format_htf_poi(signal: _EntrySignal) -> str:
    return "\n".join([
        "HTF FVG (Point of Interest)",
        f"  Status: {signal.fvg.status.value}",
        f"  Top:    {_fmt(signal.fvg.top)}",
        f"  Bottom: {_fmt(signal.fvg.bottom)}",
        f"  Formed: {_ts(signal.fvg.timestamp)}",
    ])


def _format_confirm_details(signal: _EntrySignal) -> str:
    swing_label = "Lowest LTF Swing Low" if signal.direction == Trend.BULLISH else "Highest LTF Swing High"
    prior_label = "Prior LTF Swing High" if signal.direction == Trend.BULLISH else "Prior LTF Swing Low"
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


def _format_target(fvgs: list[FVG], fractals: list[Fractal], direction: Trend) -> str:
    lines: list[str] = []
    if direction == Trend.BULLISH:
        highs = sorted([f for f in fractals if f.is_high], key=lambda f: f.price)
        lines.append("HTF Swing Highs (TP candidates, nearest first)")
        if highs:
            lines.extend(f"  {i}. {_fmt(f.price)}  ({_ts(f.timestamp)})" for i, f in enumerate(highs, 1))
        else:
            lines.append("  (none)")
        lines.append("")
        bearish = sorted([f for f in fvgs if f.trend == Trend.BEARISH], key=lambda f: f.bottom)
        lines.append("Valid HTF Bearish FVGs — bottom as TP level (nearest first)")
        if bearish:
            lines.extend(
                f"  {i}. bottom {_fmt(fvg.bottom)} / top {_fmt(fvg.top)}  ({_ts(fvg.timestamp)})"
                for i, fvg in enumerate(bearish, 1)
            )
        else:
            lines.append("  (none)")
    else:
        lows = sorted([f for f in fractals if not f.is_high], key=lambda f: f.price, reverse=True)
        lines.append("HTF Swing Lows (TP candidates, nearest first)")
        if lows:
            lines.extend(f"  {i}. {_fmt(f.price)}  ({_ts(f.timestamp)})" for i, f in enumerate(lows, 1))
        else:
            lines.append("  (none)")
        lines.append("")
        bullish = sorted([f for f in fvgs if f.trend == Trend.BULLISH], key=lambda f: f.top, reverse=True)
        lines.append("Valid HTF Bullish FVGs — top as TP level (nearest first)")
        if bullish:
            lines.extend(
                f"  {i}. top {_fmt(fvg.top)} / bottom {_fmt(fvg.bottom)}  ({_ts(fvg.timestamp)})"
                for i, fvg in enumerate(bullish, 1)
            )
        else:
            lines.append("  (none)")
    return "\n".join(lines)


def format_strategy_components(
    symbol: str,
    htf_df: pd.DataFrame,
    htf_timeframe: Timeframe,
    ltf_df: pd.DataFrame,
    ltf_timeframe: Timeframe,
    fvg_offset_pct: float = 0.0,
    block_fvg_mode: Literal["none", "active", "tested", "active+tested"] = "active",
) -> str:
    htf_fvgs = detect_fvg(htf_df, htf_timeframe)
    htf_fractals = detect_fractals(htf_df, htf_timeframe)
    ltf_fractals = detect_fractals(ltf_df, ltf_timeframe)

    bullish_fvgs = [f for f in htf_fvgs if f.trend == Trend.BULLISH]
    bearish_fvgs = [f for f in htf_fvgs if f.trend == Trend.BEARISH]
    ltf_sorted = sorted(ltf_fractals, key=lambda f: f.timestamp)
    ltf_highs = [f for f in ltf_sorted if f.is_high]
    ltf_lows = [f for f in ltf_sorted if not f.is_high]

    sep = "─" * 56
    lines: list[str] = [
        "STRATEGY INSPECTION — HTF FVG + LTF BOS",
        sep,
        _format_input_data(symbol, htf_df, htf_timeframe, ltf_df, ltf_timeframe),
        "",
        sep,
        f"HTF FVGs ({len(htf_fvgs)} total: {len(bullish_fvgs)} bullish, {len(bearish_fvgs)} bearish)",
        "",
    ]
    for label, fvg_list in [("Bullish", bullish_fvgs), ("Bearish", bearish_fvgs)]:
        lines.append(f"  {label} FVGs ({len(fvg_list)})")
        for i, fvg in enumerate(fvg_list, 1):
            lines.append(
                f"    {i}. bottom {_fmt(fvg.bottom)}  top {_fmt(fvg.top)}"
                f"  formed {_ts(fvg.timestamp)}  [{fvg.status.value}]"
            )
        if not fvg_list:
            lines.append("    (none)")

    lines += [
        "",
        sep,
        _format_target(htf_fvgs, htf_fractals, Trend.BULLISH),
        "",
        sep,
        _format_target(htf_fvgs, htf_fractals, Trend.BEARISH),
        "",
        sep,
        "Per-FVG BOS Analysis",
        "",
    ]
    lines += _format_per_fvg_bos_analysis(
        bullish_fvgs, bearish_fvgs, ltf_highs, ltf_lows,
        ltf_df, ltf_df["timestamp"].iloc[-1], fvg_offset_pct,
    )
    lines += ["", sep, _format_candles(htf_df, htf_timeframe)]
    return "\n".join(lines) + "\n"


def _format_candles(df: pd.DataFrame, timeframe: Timeframe) -> str:
    header = (
        f"HTF Candles ({timeframe.value}, {len(df)} candles)\n"
        f"{'timestamp':<20} {'open':>12} {'high':>12} {'low':>12} {'close':>12} {'volume':>14}\n"
        + "-" * 86
    )
    rows = [
        f"{row['timestamp'].strftime('%Y-%m-%d %H:%M'):<20} "
        f"{row['open']:>12.2f} {row['high']:>12.2f} "
        f"{row['low']:>12.2f} {row['close']:>12.2f} {row['volume']:>14.4f}"
        for _, row in df.iterrows()
    ]
    return header + "\n" + "\n".join(rows)
