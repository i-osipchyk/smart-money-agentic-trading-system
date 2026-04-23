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
    - Prioritize most important levels, that moved price in opposite direction, or during aggressive moves. 
      These levels should potentially have a lot of liquidity in them.

6. [LTF] Entry Order:
    Default entry is at BOS level.
    If it gives less than 2:1 reward/risk, entry order can be set lower for bullish setups, or higher for bearish to get 2:1.
    
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


class HtfFvgLtfBosV2(Strategy):
    """
    HTF FVG + LTF BOS strategy (v2 — with target selection and entry adjustment).

    Looks for a Fair Value Gap on the higher timeframe, then waits for a
    swing point to enter that gap on the lower timeframe and a Break of
    Structure to confirm directional intent.

    Extends v1 with: importance-filtered swing targets, ≥2:1 RR target filter,
    closest-target selection, and entry adjustment to achieve 2:1 RR when needed.

    Args:
        fvg_offset_pct: Fraction of the FVG range that the LTF swing point
                        may sit outside the gap and still qualify.
                        Spinner value ÷ 1000 (e.g. 10 → 0.001 → 0.1 %).
    """

    name = "htf_fvg_ltf_bos_v2"
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

        levels = _compute_levels(signal, htf_fvgs, htf_fractals, htf_df)
        if levels is None:
            return None
        entry, stop_loss, take_profit = levels

        return StrategySetup(
            input_data=_format_input_data(
                symbol, htf_df, htf_timeframe, ltf_df, ltf_timeframe, self._fvg_offset_pct
            ),
            strategy_description=self.description,
            direction=signal.direction,
            htf_poi=_format_htf_poi(signal),
            confirm_details=_format_confirm_details(signal),
            target=_format_target(htf_fvgs, htf_fractals, signal.direction, entry, stop_loss, htf_df),
            candles=_format_candles(htf_df, htf_timeframe, n_candles=20),
            entry=entry,
            stop_loss=stop_loss,
            take_profit=take_profit,
        )


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
    ltf_fractals = detect_fractals(ltf_df, ltf_timeframe)

    sep = "─" * 56
    lines: list[str] = []

    lines += [
        "STRATEGY INSPECTION — HTF FVG + LTF BOS",
        sep,
        _format_input_data(
            symbol, htf_df, htf_timeframe, ltf_df, ltf_timeframe, fvg_offset_pct
        ),
        "",
    ]

    bullish_fvgs = [f for f in htf_fvgs if f.trend == Trend.BULLISH]
    bearish_fvgs = [f for f in htf_fvgs if f.trend == Trend.BEARISH]
    lines += [
        sep,
        f"HTF FVGs  ({len(htf_fvgs)} total:"
        f" {len(bullish_fvgs)} bullish, {len(bearish_fvgs)} bearish)",
        "",
    ]
    lines.append(f"  Bullish FVGs ({len(bullish_fvgs)})")
    for i, fvg in enumerate(bullish_fvgs, 1):
        status = "tested" if _is_fvg_tested(fvg, htf_df) else "UNTESTED"
        lines.append(
            f"    {i}. bottom {_fmt(fvg.bottom)}  top {_fmt(fvg.top)}"
            f"  formed {_ts(fvg.timestamp)}  [{status}]"
        )
    if not bullish_fvgs:
        lines.append("    (none)")
    lines.append(f"  Bearish FVGs ({len(bearish_fvgs)})")
    for i, fvg in enumerate(bearish_fvgs, 1):
        status = "tested" if _is_fvg_tested(fvg, htf_df) else "UNTESTED"
        lines.append(
            f"    {i}. bottom {_fmt(fvg.bottom)}  top {_fmt(fvg.top)}"
            f"  formed {_ts(fvg.timestamp)}  [{status}]"
        )
    if not bearish_fvgs:
        lines.append("    (none)")
    lines.append("")

    htf_sorted = sorted(htf_fractals, key=lambda f: f.timestamp)
    htf_highs = [f for f in htf_sorted if f.is_high]
    htf_lows = [f for f in htf_sorted if not f.is_high]
    lines += [
        sep,
        f"HTF Fractals  ({len(htf_fractals)} total:"
        f" {len(htf_highs)} highs, {len(htf_lows)} lows)",
        "",
    ]
    lines.append(f"  Swing Highs ({len(htf_highs)})")
    for i, f in enumerate(htf_highs, 1):
        lines.append(f"    {i}. {_fmt(f.price)}  at {_ts(f.timestamp)}")
    if not htf_highs:
        lines.append("    (none)")
    lines.append(f"  Swing Lows ({len(htf_lows)})")
    for i, f in enumerate(htf_lows, 1):
        lines.append(f"    {i}. {_fmt(f.price)}  at {_ts(f.timestamp)}")
    if not htf_lows:
        lines.append("    (none)")
    lines.append("")

    ltf_sorted = sorted(ltf_fractals, key=lambda f: f.timestamp)
    ltf_highs = [f for f in ltf_sorted if f.is_high]
    ltf_lows = [f for f in ltf_sorted if not f.is_high]
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

    lines += [sep, "LTF Swing Points Inside HTF FVGs — BOS Analysis", ""]

    lines.append("  [BULLISH] Swing lows inside bullish FVGs")
    bullish_candidates: list[tuple[Fractal, FVG]] = []
    for swing_low in ltf_lows:
        for fvg in reversed(bullish_fvgs):
            if swing_low.timestamp <= fvg.timestamp:
                continue
            offset = (fvg.top - fvg.bottom) * fvg_offset_pct
            if (fvg.bottom - offset) <= swing_low.price <= fvg.top:
                bullish_candidates.append((swing_low, fvg))
                break
    if not bullish_candidates:
        lines.append("    (none)")
    for swing_low, fvg in sorted(bullish_candidates, key=lambda x: x[0].timestamp):
        lines.append(
            f"    Swing Low  {_fmt(swing_low.price)}  at {_ts(swing_low.timestamp)}"
        )
        lines.append(
            f"    FVG:       bottom {_fmt(fvg.bottom)}  top {_fmt(fvg.top)}"
            f"  formed {_ts(fvg.timestamp)}"
        )
        prior_highs = [h for h in ltf_highs if h.timestamp < swing_low.timestamp]
        if not prior_highs:
            lines.append("    Prior swing high (BOS level): (none)")
        else:
            prior = prior_highs[-1]
            lines.append(
                f"    Prior Swing High (BOS level): {_fmt(prior.price)}"
                f"  at {_ts(prior.timestamp)}"
            )
            candles_after = ltf_df[ltf_df["timestamp"] > swing_low.timestamp]
            bos_rows = candles_after[candles_after["close"] > prior.price]
            if bos_rows.empty:
                lines.append("    BOS: not confirmed (no close above BOS level)")
            else:
                bos_candle = bos_rows.iloc[0]
                bos_ts = bos_candle["timestamp"]
                if bos_ts == last_candle_ts:
                    lines.append(
                        f"    BOS: CONFIRMED on latest candle {_ts(bos_ts)}"
                        "  ← ENTRY SIGNAL"
                    )
                else:
                    lines.append(
                        f"    BOS: confirmed at {_ts(bos_ts)}"
                        "  — not on latest candle, signal expired"
                    )
        lines.append("")

    lines.append("  [BEARISH] Swing highs inside bearish FVGs")
    bearish_candidates: list[tuple[Fractal, FVG]] = []
    for swing_high in ltf_highs:
        for fvg in reversed(bearish_fvgs):
            if swing_high.timestamp <= fvg.timestamp:
                continue
            offset = (fvg.top - fvg.bottom) * fvg_offset_pct
            if fvg.bottom <= swing_high.price <= (fvg.top + offset):
                bearish_candidates.append((swing_high, fvg))
                break
    if not bearish_candidates:
        lines.append("    (none)")
    for swing_high, fvg in sorted(bearish_candidates, key=lambda x: x[0].timestamp):
        lines.append(
            f"    Swing High {_fmt(swing_high.price)}  at {_ts(swing_high.timestamp)}"
        )
        lines.append(
            f"    FVG:       bottom {_fmt(fvg.bottom)}  top {_fmt(fvg.top)}"
            f"  formed {_ts(fvg.timestamp)}"
        )
        prior_lows = [lo for lo in ltf_lows if lo.timestamp < swing_high.timestamp]
        if not prior_lows:
            lines.append("    Prior swing low (BOS level): (none)")
        else:
            prior = prior_lows[-1]
            lines.append(
                f"    Prior Swing Low (BOS level): {_fmt(prior.price)}"
                f"  at {_ts(prior.timestamp)}"
            )
            candles_after = ltf_df[ltf_df["timestamp"] > swing_high.timestamp]
            bos_rows = candles_after[candles_after["close"] < prior.price]
            if bos_rows.empty:
                lines.append("    BOS: not confirmed (no close below BOS level)")
            else:
                bos_candle = bos_rows.iloc[0]
                bos_ts = bos_candle["timestamp"]
                if bos_ts == last_candle_ts:
                    lines.append(
                        f"    BOS: CONFIRMED on latest candle {_ts(bos_ts)}"
                        "  ← ENTRY SIGNAL"
                    )
                else:
                    lines.append(
                        f"    BOS: confirmed at {_ts(bos_ts)}"
                        "  — not on latest candle, signal expired"
                    )
        lines.append("")

    signal = _find_signal(htf_fvgs, ltf_fractals, ltf_df, fvg_offset_pct)
    lines += [sep, "ENTRY DETECTION RESULT", ""]
    if signal is None:
        lines.append("  No entry detected.")
    else:
        levels = _compute_levels(signal, htf_fvgs, htf_fractals, htf_df)
        if levels is None:
            lines += [
                f"  Direction:   {signal.direction.value.upper()}",
                f"  FVG:         bottom {_fmt(signal.fvg.bottom)}"
                f"  top {_fmt(signal.fvg.top)}",
                f"  BOS Level:   {_fmt(signal.bos_level)}"
                f"  confirmed at {_ts(signal.bos_candle_timestamp)}",
                "  BLOCKED: untested FVG on path to target — no trade.",
            ]
        else:
            entry, sl, tp = levels
            lines += [
                f"  Direction:   {signal.direction.value.upper()}",
                f"  FVG:         bottom {_fmt(signal.fvg.bottom)}"
                f"  top {_fmt(signal.fvg.top)}",
                f"  Swing Point: {_fmt(signal.swing_point.price)}"
                f"  at {_ts(signal.swing_point.timestamp)}",
                f"  BOS Level:   {_fmt(signal.bos_level)}"
                f"  confirmed at {_ts(signal.bos_candle_timestamp)}",
                f"  Entry:       {_fmt(entry)}",
                f"  Stop Loss:   {_fmt(sl)}",
                f"  Take Profit: {_fmt(tp)}",
            ]

    return "\n".join(lines)


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

    for swing_low, fvg in sorted(bullish_candidates, key=lambda x: x[0].price):
        prior_highs = [h for h in swing_highs if h.timestamp < swing_low.timestamp]
        if not prior_highs:
            continue
        prior_swing_high = prior_highs[-1]
        candles_after = ltf_df[ltf_df["timestamp"] > swing_low.timestamp]
        bos_rows = candles_after[candles_after["close"] > prior_swing_high.price]
        if bos_rows.empty:
            continue
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

    for swing_high, fvg in sorted(
        bearish_candidates, key=lambda x: x[0].price, reverse=True
    ):
        prior_lows = [lo for lo in swing_lows if lo.timestamp < swing_high.timestamp]
        if not prior_lows:
            continue
        prior_swing_low = prior_lows[-1]
        candles_after = ltf_df[ltf_df["timestamp"] > swing_high.timestamp]
        bos_rows = candles_after[candles_after["close"] < prior_swing_low.price]
        if bos_rows.empty:
            continue
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

def _select_target(
    fvgs: list[FVG],
    fractals: list[Fractal],
    direction: Trend,
    entry: float,
    stop_loss: float,
) -> float | None:
    """Return the closest valid target that clears 1:1 RR, or None if none exist.

    Applies importance filtering on swing fractals before the RR check.
    For bullish setups: bearish FVG lows and swing highs above min_tp.
    For bearish setups: bullish FVG highs and swing lows below min_tp.
    """
    candidates: list[float] = []

    if direction == Trend.BULLISH:
        risk = entry - stop_loss
        min_tp = entry + risk

        all_highs = sorted([f for f in fractals if f.is_high], key=lambda f: f.timestamp)
        for f in _important_swings(all_highs):
            if f.price >= min_tp:
                candidates.append(f.price)

        for fvg in fvgs:
            if fvg.trend == Trend.BEARISH and fvg.bottom >= min_tp:
                candidates.append(fvg.bottom)

        return min(candidates) if candidates else None

    else:
        risk = stop_loss - entry
        min_tp = entry - risk

        all_lows = sorted([f for f in fractals if not f.is_high], key=lambda f: f.timestamp)
        for f in _important_swings(all_lows):
            if f.price <= min_tp:
                candidates.append(f.price)

        for fvg in fvgs:
            if fvg.trend == Trend.BULLISH and fvg.top <= min_tp:
                candidates.append(fvg.top)

        return max(candidates) if candidates else None


def _is_fvg_tested(fvg: FVG, df: pd.DataFrame) -> bool:
    """True if price has entered the FVG zone on any candle after formation."""
    subsequent = df[df["timestamp"] > fvg.timestamp]
    if subsequent.empty:
        return False
    if fvg.trend == Trend.BULLISH:
        return bool((subsequent["low"] <= fvg.top).any())
    else:
        return bool((subsequent["high"] >= fvg.bottom).any())


def _has_untested_fvg_on_path(
    fvgs: list[FVG],
    direction: Trend,
    entry: float,
    target: float,
    df: pd.DataFrame,
) -> bool:
    """True if any untested opposing-direction FVG zone overlaps the entry→target path."""
    if direction == Trend.BULLISH:
        # opposing = bearish FVGs (supply); zone must intersect (entry, target)
        blocking = [
            fvg for fvg in fvgs
            if fvg.trend == Trend.BEARISH
            and fvg.bottom < target
            and fvg.top > entry
        ]
    else:
        # opposing = bullish FVGs (demand); zone must intersect (target, entry)
        blocking = [
            fvg for fvg in fvgs
            if fvg.trend == Trend.BULLISH
            and fvg.top > target
            and fvg.bottom < entry
        ]
    return any(not _is_fvg_tested(fvg, df) for fvg in blocking)


def _compute_levels(
    signal: _EntrySignal,
    fvgs: list[FVG],
    fractals: list[Fractal],
    df: pd.DataFrame,
) -> tuple[float, float, float] | None:
    """
    Return (entry, stop_loss, take_profit) or None if an untested FVG blocks the path.

    - Entry:     BOS level by default; moved toward stop loss if needed for 2:1.
    - Stop Loss: 1 unit beyond the swing point inside the FVG.
    - Take Profit: closest valid target (importance-filtered, ≥1:1 RR).
                   Falls back to 2:1 from BOS entry when no structural target exists.
                   Entry is adjusted to achieve exactly 2:1 only when the selected
                   target gives < 2:1 at BOS; a target giving > 2:1 keeps BOS entry.
    - Returns None when an untested opposing FVG sits between entry and target.
    """
    entry = signal.bos_level

    if signal.direction == Trend.BULLISH:
        stop_loss = float(round(signal.swing_point.price) - 1)
        risk = entry - stop_loss
        fallback_tp = entry + 2 * risk
    else:
        stop_loss = float(round(signal.swing_point.price) + 1)
        risk = stop_loss - entry
        fallback_tp = entry - 2 * risk

    target = _select_target(fvgs, fractals, signal.direction, entry, stop_loss)
    if target is None:
        target = fallback_tp
    else:
        if signal.direction == Trend.BULLISH:
            rr = (target - entry) / risk
        else:
            rr = (entry - target) / risk

        if rr < 2.0:
            # Slide entry toward stop loss so the target yields exactly 2:1.
            # Solving |tp - e| / |e - sl| = 2  →  e = (tp + 2*sl) / 3
            entry = (target + 2 * stop_loss) / 3

    if _has_untested_fvg_on_path(fvgs, signal.direction, entry, target, df):
        return None

    return entry, stop_loss, target


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
        f"HTF range:  {_ts(htf_df['timestamp'].iloc[0])} → {_ts(htf_df['timestamp'].iloc[-2])}",
        f"LTF range:  {_ts(ltf_df['timestamp'].iloc[0])} → {_ts(ltf_df['timestamp'].iloc[-1])}",
        f"FVG offset: {offset_display:.1f} %",
    ])


def _format_htf_poi(signal: _EntrySignal) -> str:
    return "\n".join([
        "HTF FVG (Point of Interest)",
        f"  Top:    {_fmt(signal.fvg.top)}",
        f"  Bottom: {_fmt(signal.fvg.bottom)}",
        f"  Formed: {_ts(signal.fvg.timestamp)} (third candle close time)",
    ])


def _format_confirm_details(signal: _EntrySignal) -> str:
    swing_label = (
        "Lowest LTF Swing Low" if signal.direction == Trend.BULLISH else "Highest LTF Swing High"
    )
    prior_label = (
        "Prior LTF Swing High" if signal.direction == Trend.BULLISH else "Prior LTF Swing Low"
    )

    return "\n".join([
        f"{swing_label} (inside FVG)",
        f"  Price:  {_fmt(signal.swing_point.price)}",
        f"  At:     {_ts(signal.swing_point.timestamp)} (candle open time)",
        "",
        f"{prior_label} (BOS level)",
        f"  Price:  {_fmt(signal.prior_swing.price)}",
        f"  At:     {_ts(signal.prior_swing.timestamp)} (candle open time)",
        "",
        "Break of Structure (BOS)",
        f"  Level:  {_fmt(signal.bos_level)}",
        f"  At: {_ts(signal.bos_candle_timestamp)} (candle open time)",
    ])


def _important_swings(swings: list[Fractal]) -> list[Fractal]:
    """Return only structurally important swings (local extremes among same-type fractals).

    A swing high is important when its price is higher than both its neighbours.
    A swing low is important when its price is lower than both its neighbours.
    The first and last swing are always kept.
    Input must be sorted by timestamp.
    """
    if len(swings) <= 2:
        return list(swings)
    result = [swings[0]]
    for i in range(1, len(swings) - 1):
        prev_p = swings[i - 1].price
        cur_p = swings[i].price
        next_p = swings[i + 1].price
        is_high = swings[i].is_high
        if is_high and cur_p > prev_p and cur_p > next_p:
            result.append(swings[i])
        elif not is_high and cur_p < prev_p and cur_p < next_p:
            result.append(swings[i])
    result.append(swings[-1])
    return result


def _format_target(
    fvgs: list[FVG],
    fractals: list[Fractal],
    direction: Trend,
    entry: float,
    stop_loss: float,
    df: pd.DataFrame,
) -> str:
    lines: list[str] = []

    if direction == Trend.BULLISH:
        risk = entry - stop_loss
        min_tp = entry + risk  # 1:1 minimum

        all_highs = sorted([f for f in fractals if f.is_high], key=lambda f: f.timestamp)
        highs = sorted(
            [f for f in _important_swings(all_highs) if f.price >= min_tp],
            key=lambda f: f.price,
        )
        lines.append(f"HTF Swing Highs (≥1:1 RR from entry {_fmt(entry)}, nearest first)")
        if highs:
            for i, f in enumerate(highs, 1):
                lines.append(f"  {i}. {_fmt(f.price)}  ({_ts(f.timestamp)})")
        else:
            lines.append("  (none)")

        lines.append("")
        bearish = sorted(
            [f for f in fvgs if f.trend == Trend.BEARISH and f.bottom >= min_tp],
            key=lambda f: f.bottom,
        )
        lines.append("Valid HTF Bearish FVGs — low (≥1:1 RR, nearest first)")
        if bearish:
            for i, fvg in enumerate(bearish, 1):
                status = "tested" if _is_fvg_tested(fvg, df) else "UNTESTED"
                lines.append(f"  {i}. {_fmt(fvg.bottom)}  ({_ts(fvg.timestamp)})  [{status}]")
        else:
            lines.append("  (none)")

        lines.append("")
        lines.append("All HTF Bearish FVGs in path (entry → target) — obstruction check")
        path_blockers = [
            fvg for fvg in fvgs
            if fvg.trend == Trend.BEARISH and fvg.top > entry
        ]
        if path_blockers:
            for fvg in sorted(path_blockers, key=lambda f: f.bottom):
                status = "tested" if _is_fvg_tested(fvg, df) else "UNTESTED"
                lines.append(
                    f"  {_fmt(fvg.bottom)}–{_fmt(fvg.top)}"
                    f"  ({_ts(fvg.timestamp)})  [{status}]"
                )
        else:
            lines.append("  (none)")

    else:  # BEARISH
        risk = stop_loss - entry
        min_tp = entry - risk  # 1:1 minimum

        all_lows = sorted([f for f in fractals if not f.is_high], key=lambda f: f.timestamp)
        lows = sorted(
            [f for f in _important_swings(all_lows) if f.price <= min_tp],
            key=lambda f: f.price,
            reverse=True,
        )
        lines.append(f"HTF Swing Lows (≥1:1 RR from entry {_fmt(entry)}, nearest first)")
        if lows:
            for i, f in enumerate(lows, 1):
                lines.append(f"  {i}. {_fmt(f.price)}  ({_ts(f.timestamp)})")
        else:
            lines.append("  (none)")

        lines.append("")
        bullish = sorted(
            [f for f in fvgs if f.trend == Trend.BULLISH and f.top <= min_tp],
            key=lambda f: f.top,
            reverse=True,
        )
        lines.append("Valid HTF Bullish FVGs — high (≥1:1 RR, nearest first)")
        if bullish:
            for i, fvg in enumerate(bullish, 1):
                status = "tested" if _is_fvg_tested(fvg, df) else "UNTESTED"
                lines.append(f"  {i}. {_fmt(fvg.top)}  ({_ts(fvg.timestamp)})  [{status}]")
        else:
            lines.append("  (none)")

        lines.append("")
        lines.append("All HTF Bullish FVGs in path (target → entry) — obstruction check")
        path_blockers = [
            fvg for fvg in fvgs
            if fvg.trend == Trend.BULLISH and fvg.bottom < entry
        ]
        if path_blockers:
            for fvg in sorted(path_blockers, key=lambda f: f.top, reverse=True):
                status = "tested" if _is_fvg_tested(fvg, df) else "UNTESTED"
                lines.append(
                    f"  {_fmt(fvg.bottom)}–{_fmt(fvg.top)}"
                    f"  ({_ts(fvg.timestamp)})  [{status}]"
                )
        else:
            lines.append("  (none)")

    return "\n".join(lines)


def _format_candles(df: pd.DataFrame, timeframe: Timeframe, n_candles: int = 20) -> str:
    header = (
        f"HTF Candles ({timeframe.value}, {len(df)} candles)\n"
        f"{'timestamp':<20} {'open':>12} {'high':>12} {'low':>12} {'close':>12} {'volume':>14}\n"
        + "-" * 86
    )
    rows = [
        f"{row['timestamp'].strftime('%Y-%m-%d %H:%M'):<20} "
        f"{row['open']:>12.2f} {row['high']:>12.2f} "
        f"{row['low']:>12.2f} {row['close']:>12.2f} {row['volume']:>14.4f}"
        for _, row in df.iloc[:-1].tail(n_candles).iterrows()
    ]
    return header + "\n" + "\n".join(rows)
