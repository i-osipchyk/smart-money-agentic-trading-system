from dataclasses import dataclass
from datetime import datetime

import pandas as pd

from trading.core.models import FVG, Fractal, StrategySetup, Timeframe, Trend
from trading.signals.fractals import detect_fractals
from trading.signals.fvg import detect_fvg
from trading.strategies.base import Strategy

_DESCRIPTION = """\
HTF FVG + LTF BOS (Fair Value Gap with Break of Structure confirmation)

Entry logic:
1. [HTF] Identify all valid Fair Value Gaps (FVGs).
   - Bullish FVG: candle[i+1].low > candle[i-1].high — an upward price gap.
     Invalidated when any subsequent candle closes below the gap bottom.
   - Bearish FVG: candle[i+1].high < candle[i-1].low — a downward price gap.
     Invalidated when any subsequent candle closes above the gap top.

2. [LTF] Bullish setup:
   a. The lowest LTF swing low lies inside a valid bullish HTF FVG, or just
      below it within the configured offset (% of FVG range).
   b. A prior LTF swing high (the last one before the swing low) acts as the
      Break of Structure (BOS) level.
   c. Price closes above that BOS level after the swing low — confirming the
      bullish directional shift.

3. [LTF] Bearish setup:
   a. The highest LTF swing high lies inside a valid bearish HTF FVG, or just
      above it within the configured offset.
   b. A prior LTF swing low (the last one before the swing high) acts as the
      BOS level.
   c. Price closes below that BOS level after the swing high — confirming the
      bearish directional shift.

4. [LTF] Stop Loss:
    - For bullish entries: just below the swing low (which is inside the FVG).
    - For bearish entries: just above the swing high (which is inside the FVG).

5. [HTF] Take Profit:
    - For bullish entries: swing highs or low of bearish FVG.
    - For bearish entries: swing lows or high of bullish FVG.

6. [LTF] Entry Order:
    For both bullish and bearish setups, limit order which gives risk-reward at least 2:1 based on the defined stop loss and take profit levels.

The HTF FVG is the Point of Interest (POI) / demand or supply zone.
The LTF swing point marks the liquidity sweep / wick into that zone.
The BOS confirms that smart money has absorbed liquidity and is pushing price.\
"""


@dataclass
class _EntrySignal:
    """Internal signal object — not exposed outside this module."""
    direction: Trend
    fvg: FVG
    swing_point: Fractal
    prior_swing: Fractal
    bos_candle_timestamp: datetime
    bos_level: float


class HtfFvgLtfBos(Strategy):
    """
    HTF FVG + LTF BOS strategy.

    Looks for a Fair Value Gap on the higher timeframe, then waits for a
    swing point to enter that gap on the lower timeframe and a Break of
    Structure to confirm directional intent.

    Args:
        fvg_offset_pct: Fraction of the FVG range that the LTF swing point
                        may sit outside the gap and still qualify.
                        Spinner value ÷ 1000 (e.g. 10 → 0.001 → 0.1 %).
    """

    name = "htf_fvg_ltf_bos"
    description = _DESCRIPTION

    def __init__(self, fvg_offset_pct: float = 0.001) -> None:
        self._fvg_offset_pct = fvg_offset_pct

    def detect_entry(
        self,
        symbol: str,
        htf_df: pd.DataFrame,
        htf_timeframe: Timeframe,
        ltf_df: pd.DataFrame,
        ltf_timeframe: Timeframe,
    ) -> StrategySetup | None:
        """
        Run the HTF-FVG / LTF-BOS strategy and return a StrategySetup if a
        valid entry is detected, or None otherwise.
        """
        htf_fvgs = detect_fvg(htf_df, htf_timeframe)
        htf_fractals = detect_fractals(htf_df, htf_timeframe)
        ltf_fractals = detect_fractals(ltf_df, ltf_timeframe)

        signal = _find_signal(htf_fvgs, ltf_fractals, ltf_df, self._fvg_offset_pct)
        if signal is None:
            return None

        return StrategySetup(
            input_data=_format_input_data(
                symbol, htf_df, htf_timeframe, ltf_df, ltf_timeframe, self._fvg_offset_pct
            ),
            strategy_description=self.description,
            direction=signal.direction,
            htf_poi=_format_htf_poi(signal),
            confirm_details=_format_confirm_details(signal),
            target=_format_target(htf_fvgs, htf_fractals, signal.direction),
            candles=_format_candles(htf_df, htf_timeframe),
        )


# ------------------------------------------------------------------ internals

def _find_signal(
    htf_fvgs: list[FVG],
    ltf_fractals: list[Fractal],
    ltf_df: pd.DataFrame,
    fvg_offset_pct: float,
) -> _EntrySignal | None:
    if not htf_fvgs or not ltf_fractals:
        return None

    fractals_sorted = sorted(ltf_fractals, key=lambda f: f.timestamp)
    swing_lows = [f for f in fractals_sorted if not f.is_high]
    swing_highs = [f for f in fractals_sorted if f.is_high]

    bullish_fvgs = [f for f in htf_fvgs if f.trend == Trend.BULLISH]
    bearish_fvgs = [f for f in htf_fvgs if f.trend == Trend.BEARISH]

    # ---------------------------------------------------------------- bullish
    if swing_lows and bullish_fvgs:
        lowest_swing_low = min(swing_lows, key=lambda f: f.price)

        for fvg in reversed(bullish_fvgs):
            offset = (fvg.top - fvg.bottom) * fvg_offset_pct
            if (fvg.bottom - offset) <= lowest_swing_low.price <= fvg.top:
                prior_highs = [h for h in swing_highs if h.timestamp < lowest_swing_low.timestamp]
                if not prior_highs:
                    continue

                prior_swing_high = prior_highs[-1]
                candles_after = ltf_df[ltf_df["timestamp"] > lowest_swing_low.timestamp]
                bos_rows = candles_after[candles_after["close"] > prior_swing_high.price]

                if not bos_rows.empty:
                    bos_candle = bos_rows.iloc[0]
                    return _EntrySignal(
                        direction=Trend.BULLISH,
                        fvg=fvg,
                        swing_point=lowest_swing_low,
                        prior_swing=prior_swing_high,
                        bos_candle_timestamp=bos_candle["timestamp"],
                        bos_level=prior_swing_high.price,
                    )

    # ---------------------------------------------------------------- bearish
    if swing_highs and bearish_fvgs:
        highest_swing_high = max(swing_highs, key=lambda f: f.price)

        for fvg in reversed(bearish_fvgs):
            offset = (fvg.top - fvg.bottom) * fvg_offset_pct
            if fvg.bottom <= highest_swing_high.price <= (fvg.top + offset):
                prior_lows = [lo for lo in swing_lows if lo.timestamp < highest_swing_high.timestamp]
                if not prior_lows:
                    continue

                prior_swing_low = prior_lows[-1]
                candles_after = ltf_df[ltf_df["timestamp"] > highest_swing_high.timestamp]
                bos_rows = candles_after[candles_after["close"] < prior_swing_low.price]

                if not bos_rows.empty:
                    bos_candle = bos_rows.iloc[0]
                    return _EntrySignal(
                        direction=Trend.BEARISH,
                        fvg=fvg,
                        swing_point=highest_swing_high,
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
    fvg_offset_pct: float,
) -> str:
    offset_display = fvg_offset_pct * 100
    return "\n".join([
        f"Symbol:     {symbol}",
        f"HTF:        {htf_timeframe.value}  ({len(htf_df)} candles)",
        f"LTF:        {ltf_timeframe.value}  ({len(ltf_df)} candles)",
        f"HTF range:  {_ts(htf_df['timestamp'].iloc[0])} → {_ts(htf_df['timestamp'].iloc[-1])}",
        f"LTF range:  {_ts(ltf_df['timestamp'].iloc[0])} → {_ts(ltf_df['timestamp'].iloc[-1])}",
        f"FVG offset: {offset_display:.1f} %",
    ])


def _format_htf_poi(signal: _EntrySignal) -> str:
    return "\n".join([
        "HTF FVG (Point of Interest)",
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
        f"{swing_label} (inside FVG)",
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
        # TP candidates: swing highs above (sorted ascending — nearest first)
        highs = sorted([f for f in fractals if f.is_high], key=lambda f: f.price)
        lines.append("HTF Swing Highs (TP candidates, nearest first)")
        if highs:
            for i, f in enumerate(highs, 1):
                lines.append(f"  {i}. {_fmt(f.price)}  ({_ts(f.timestamp)})")
        else:
            lines.append("  (none)")

        # TP candidates: bearish FVG bottoms (price tends to reach the low of supply zone)
        lines.append("")
        bearish = sorted(
            [f for f in fvgs if f.trend == Trend.BEARISH], key=lambda f: f.bottom
        )
        lines.append("Valid HTF Bearish FVGs — bottom as TP level (nearest first)")
        if bearish:
            for i, fvg in enumerate(bearish, 1):
                lines.append(
                    f"  {i}. bottom {_fmt(fvg.bottom)} / top {_fmt(fvg.top)}"
                    f"  ({_ts(fvg.timestamp)})"
                )
        else:
            lines.append("  (none)")

    else:  # BEARISH
        # TP candidates: swing lows below (sorted descending — nearest first)
        lows = sorted([f for f in fractals if not f.is_high], key=lambda f: f.price, reverse=True)
        lines.append("HTF Swing Lows (TP candidates, nearest first)")
        if lows:
            for i, f in enumerate(lows, 1):
                lines.append(f"  {i}. {_fmt(f.price)}  ({_ts(f.timestamp)})")
        else:
            lines.append("  (none)")

        # TP candidates: bullish FVG tops (price tends to reach the high of demand zone)
        lines.append("")
        bullish = sorted(
            [f for f in fvgs if f.trend == Trend.BULLISH], key=lambda f: f.top, reverse=True
        )
        lines.append("Valid HTF Bullish FVGs — top as TP level (nearest first)")
        if bullish:
            for i, fvg in enumerate(bullish, 1):
                lines.append(
                    f"  {i}. top {_fmt(fvg.top)} / bottom {_fmt(fvg.bottom)}"
                    f"  ({_ts(fvg.timestamp)})"
                )
        else:
            lines.append("  (none)")

    return "\n".join(lines)


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
