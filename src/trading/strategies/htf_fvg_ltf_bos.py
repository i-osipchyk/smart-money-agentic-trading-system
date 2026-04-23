import logging
from dataclasses import dataclass
from datetime import datetime

import pandas as pd

from trading.core.models import FVG, Fractal, StrategySetup, Timeframe, Trend
from trading.signals.fractals import detect_fractals
from trading.signals.fvg import detect_fvg
from trading.strategies.base import Strategy

logger = logging.getLogger(__name__)

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

        _log_findings(htf_fvgs, ltf_fractals, self._fvg_offset_pct)

        signal = _find_signal(htf_fvgs, ltf_fractals, ltf_df, self._fvg_offset_pct)
        if signal is None:
            return None

        entry, stop_loss, take_profit = _compute_levels(signal)

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
            entry=entry,
            stop_loss=stop_loss,
            take_profit=take_profit,
        )


# ------------------------------------------------------------------ internals

def _log_findings(
    htf_fvgs: list[FVG],
    ltf_fractals: list[Fractal],
    fvg_offset_pct: float,
) -> None:
    logger.info("HTF FVGs found: %d", len(htf_fvgs))
    for fvg in htf_fvgs:
        logger.info(
            "  FVG [%s] top=%.2f bottom=%.2f formed=%s",
            fvg.trend.value,
            fvg.top,
            fvg.bottom,
            fvg.timestamp.strftime("%Y-%m-%d %H:%M"),
        )

    logger.info("LTF fractals found: %d", len(ltf_fractals))
    for f in ltf_fractals:
        kind = "high" if f.is_high else "low"
        logger.info(
            "  Fractal [%s] price=%.2f at=%s",
            kind,
            f.price,
            f.timestamp.strftime("%Y-%m-%d %H:%M"),
        )

    inside: list[tuple[Fractal, FVG]] = []
    for f in ltf_fractals:
        for fvg in htf_fvgs:
            offset = (fvg.top - fvg.bottom) * fvg_offset_pct
            low = fvg.bottom - offset
            high = fvg.top + offset
            if low <= f.price <= high:
                inside.append((f, fvg))

    logger.info("LTF fractals inside an HTF FVG (±offset): %d", len(inside))
    for f, fvg in inside:
        kind = "high" if f.is_high else "low"
        logger.info(
            "  Fractal [%s] price=%.2f at=%s  →  FVG [%s] bottom=%.2f top=%.2f",
            kind,
            f.price,
            f.timestamp.strftime("%Y-%m-%d %H:%M"),
            fvg.trend.value,
            fvg.bottom,
            fvg.top,
        )


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

    bullish_fvgs = [f for f in htf_fvgs if f.trend == Trend.BULLISH]
    bearish_fvgs = [f for f in htf_fvgs if f.trend == Trend.BEARISH]

    # ---------------------------------------------------------------- bullish
    bullish_candidates: list[tuple[Fractal, FVG]] = []
    for swing_low in swing_lows:
        for fvg in reversed(bullish_fvgs):  # most-recent FVG first
            if swing_low.timestamp <= fvg.timestamp:
                continue
            offset = (fvg.top - fvg.bottom) * fvg_offset_pct
            if (fvg.bottom - offset) <= swing_low.price <= fvg.top:
                bullish_candidates.append((swing_low, fvg))
                break  # use the most-recent FVG for this swing

    if bullish_candidates:
        swing_low, fvg = min(bullish_candidates, key=lambda x: x[0].price)
        prior_highs = [h for h in swing_highs if h.timestamp < swing_low.timestamp]
        if prior_highs:
            prior_swing_high = prior_highs[-1]
            candles_after = ltf_df[ltf_df["timestamp"] > swing_low.timestamp]
            bos_rows = candles_after[candles_after["close"] > prior_swing_high.price]
            if not bos_rows.empty:
                bos_candle = bos_rows.iloc[0]
                if bos_candle["timestamp"] == last_candle_ts:
                    return _EntrySignal(
                        direction=Trend.BULLISH,
                        fvg=fvg,
                        swing_point=swing_low,
                        prior_swing=prior_swing_high,
                        bos_candle_timestamp=bos_candle["timestamp"],
                        bos_level=prior_swing_high.price,
                    )

    # ---------------------------------------------------------------- bearish
    bearish_candidates: list[tuple[Fractal, FVG]] = []
    for swing_high in swing_highs:
        for fvg in reversed(bearish_fvgs):  # most-recent FVG first
            if swing_high.timestamp <= fvg.timestamp:
                continue
            offset = (fvg.top - fvg.bottom) * fvg_offset_pct
            if fvg.bottom <= swing_high.price <= (fvg.top + offset):
                bearish_candidates.append((swing_high, fvg))
                break  # use the most-recent FVG for this swing

    if bearish_candidates:
        swing_high, fvg = max(bearish_candidates, key=lambda x: x[0].price)
        prior_lows = [lo for lo in swing_lows if lo.timestamp < swing_high.timestamp]
        if prior_lows:
            prior_swing_low = prior_lows[-1]
            candles_after = ltf_df[ltf_df["timestamp"] > swing_high.timestamp]
            bos_rows = candles_after[candles_after["close"] < prior_swing_low.price]
            if not bos_rows.empty:
                bos_candle = bos_rows.iloc[0]
                if bos_candle["timestamp"] == last_candle_ts:
                    return _EntrySignal(
                        direction=Trend.BEARISH,
                        fvg=fvg,
                        swing_point=swing_high,
                        prior_swing=prior_swing_low,
                        bos_candle_timestamp=bos_candle["timestamp"],
                        bos_level=prior_swing_low.price,
                    )

    return None


# --------------------------------------------------------- level computation

def _compute_levels(signal: _EntrySignal) -> tuple[float, float, float]:
    """
    Return (entry, stop_loss, take_profit) for a baseline limit order.

    - Entry:      BOS level (limit order placed at the prior swing that was broken).
    - Stop Loss:  1 unit beyond the swing point that entered the FVG, rounded to
                  the nearest integer (floor for bullish, ceil for bearish).
    - Take Profit: 2:1 reward/risk relative to entry.
    """
    entry = signal.bos_level

    if signal.direction == Trend.BULLISH:
        stop_loss = float(round(signal.swing_point.price) - 1)
        risk = entry - stop_loss
        take_profit = entry + 2 * risk
    else:
        stop_loss = float(round(signal.swing_point.price) + 1)
        risk = stop_loss - entry
        take_profit = entry - 2 * risk

    return entry, stop_loss, take_profit


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


def format_strategy_components(
    symbol: str,
    htf_df: pd.DataFrame,
    htf_timeframe: Timeframe,
    ltf_df: pd.DataFrame,
    ltf_timeframe: Timeframe,
    fvg_offset_pct: float,
) -> str:
    """Return a full human-readable breakdown of all strategy components."""
    htf_fvgs = detect_fvg(htf_df, htf_timeframe)
    htf_fractals = detect_fractals(htf_df, htf_timeframe)

    sep = "─" * 56
    lines: list[str] = [
        "STRATEGY INSPECTION — HTF FVG + LTF BOS",
        sep,
        _format_input_data(
            symbol, htf_df, htf_timeframe, ltf_df, ltf_timeframe, fvg_offset_pct
        ),
        "",
        sep,
        _format_target(htf_fvgs, htf_fractals, Trend.BULLISH),
        "",
        sep,
        _format_target(htf_fvgs, htf_fractals, Trend.BEARISH),
        "",
        sep,
        _format_candles(htf_df, htf_timeframe),
    ]
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
