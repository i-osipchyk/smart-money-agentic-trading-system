import logging
from dataclasses import dataclass
from datetime import datetime

import pandas as pd

from trading.core.models import FVG, Fractal, FvgStatus, StrategySetup, Timeframe, Trend
from trading.signals.fractals import detect_fractals
from trading.signals.fvg import detect_fvg
from trading.strategies.base import Strategy
from trading.strategies.htf_fvg_ltf_bos import trend_from_fractals

logger = logging.getLogger(__name__)

_DESCRIPTION = """\
HTF FVG + LTF BOS v2 (Fair Value Gap with Break of Structure confirmation)

Entry logic:
1. [HTF] Identify all Fair Value Gaps (FVGs) and classify each:
   - Active:      not yet touched by price after formation.
   - Tested:      price entered the FVG zone but closed on the correct side.
   - Invalidated: a subsequent candle closed through the far side of the gap.
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

3. [LTF] Bearish setup — mirror of the bullish setup (swing high within fvg_offset_pct above FVG top).

4. [LTF] Stop Loss:
    - Bullish: exactly at the lowest swing low price.
    - Bearish: exactly at the highest swing high price.

5. [HTF] Take Profit:
    - Candidates: HTF swing highs (bullish) / swing lows (bearish) and opposing
      FVG near-edges, filtered to at least 1:1 RR from the default entry.
    - Importance filter applied to swing candidates: keep first, last, and local
      extremes (higher than both neighbours for highs; lower for lows).
    - Closest qualifying candidate is selected as the target.
    - Falls back to 2:1 from BOS entry when no candidate qualifies.

6. [LTF] Entry order:
    - Default entry: BOS level (limit order at the prior swing that was broken).
    - If the selected target gives ≥ 2:1 from BOS, entry stays at BOS.
    - If the selected target gives < 2:1, entry is moved toward the stop loss so
      the target yields exactly 2:1  (entry = (target + 2 × stop_loss) / 3).

7. Trend filter:
    HTF fractals determine the macro trend: higher highs + higher lows = bullish,
    lower highs + lower lows = bearish, mixed = no bias.
    A setup is discarded when the signal direction conflicts with the HTF trend.
    When HTF structure is mixed (None), the setup is allowed through.

8. Path filter:
    If any active opposing-direction HTF FVG sits between entry and take profit,
    the setup is discarded.
    When block_tested_fvgs is enabled, tested opposing FVGs also block the setup.

The HTF FVG is the Point of Interest (POI) / demand or supply zone.
The LTF swing point marks the liquidity sweep near that zone.
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
        fvg_offset_pct:     Extends the qualifying zone beyond the FVG edge as a
                            fraction of that edge price: bullish lows qualify down to
                            bottom × (1 − pct); bearish highs qualify up to
                            top × (1 + pct). Default 0.0005 (0.05 %).
        block_tested_fvgs:  When True, tested opposing FVGs also block the path
                            filter (step 8). Default False (only active FVGs block).
    """

    name = "htf_fvg_ltf_bos_v2"
    description = _DESCRIPTION

    def __init__(
        self,
        fvg_offset_pct: float = 0.0005,
        block_tested_fvgs: bool = False,
    ) -> None:
        self._fvg_offset_pct = fvg_offset_pct
        self._block_tested_fvgs = block_tested_fvgs

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

        _log_findings(htf_fvgs, ltf_fractals)

        signal = _find_signal(htf_fvgs, ltf_fractals, ltf_df, self._fvg_offset_pct)
        if signal is None:
            return None

        htf_trend = trend_from_fractals(htf_fractals)
        if htf_trend is not None and htf_trend != signal.direction:
            return None

        levels = _compute_levels(signal, htf_fvgs, htf_fractals, self._block_tested_fvgs)
        if levels is None:
            return None
        entry, stop_loss, take_profit = levels

        return StrategySetup(
            input_data=_format_input_data(
                symbol, htf_df, htf_timeframe, ltf_df, ltf_timeframe
            ),
            strategy_description=self.description,
            direction=signal.direction,
            htf_poi=_format_htf_poi(signal),
            confirm_details=_format_confirm_details(signal),
            target=_format_target(htf_fvgs, htf_fractals, signal.direction, entry, stop_loss),
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
    fvg_offset_pct: float = 0.0,
    block_tested_fvgs: bool = False,
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
            symbol, htf_df, htf_timeframe, ltf_df, ltf_timeframe
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
        status = fvg.status.value
        lines.append(
            f"    {i}. bottom {_fmt(fvg.bottom)}  top {_fmt(fvg.top)}"
            f"  formed {_ts(fvg.timestamp)}  [{status}]"
        )
    if not bullish_fvgs:
        lines.append("    (none)")
    lines.append(f"  Bearish FVGs ({len(bearish_fvgs)})")
    for i, fvg in enumerate(bearish_fvgs, 1):
        status = fvg.status.value
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

    lines += [sep, "Per-FVG BOS Analysis", ""]

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
        prior_highs = [h for h in ltf_highs if h.timestamp < swing_low.timestamp]
        if not prior_highs:
            lines.append("      Prior swing high (BOS level): (none)")
        else:
            prior = prior_highs[-1]
            lines.append(
                f"      Prior Swing High (BOS level): {_fmt(prior.price)}"
                f"  at {_ts(prior.timestamp)}"
            )
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
        prior_lows = [lo for lo in ltf_lows if lo.timestamp < swing_high.timestamp]
        if not prior_lows:
            lines.append("      Prior swing low (BOS level): (none)")
        else:
            prior = prior_lows[-1]
            lines.append(
                f"      Prior Swing Low (BOS level): {_fmt(prior.price)}"
                f"  at {_ts(prior.timestamp)}"
            )
            candles_after = ltf_df[ltf_df["timestamp"] > swing_high.timestamp]
            bos_rows = candles_after[candles_after["close"] < prior.price]
            if bos_rows.empty:
                lines.append("      BOS: not confirmed")
            else:
                bos_ts = bos_rows.iloc[0]["timestamp"]
                marker = "  ← ENTRY SIGNAL" if bos_ts == last_candle_ts else "  — signal expired"
                lines.append(f"      BOS: {_ts(bos_ts)}{marker}")
        lines.append("")

    signal = _find_signal(htf_fvgs, ltf_fractals, ltf_df, fvg_offset_pct)
    lines += [sep, "ENTRY DETECTION RESULT", ""]
    if signal is None:
        lines.append("  No entry detected.")
    else:
        levels = _compute_levels(signal, htf_fvgs, htf_fractals, block_tested_fvgs)
        if levels is None:
            lines += [
                f"  Direction:   {signal.direction.value.upper()}",
                f"  FVG:         bottom {_fmt(signal.fvg.bottom)}"
                f"  top {_fmt(signal.fvg.top)}",
                f"  BOS Level:   {_fmt(signal.bos_level)}"
                f"  confirmed at {_ts(signal.bos_candle_timestamp)}",
                "  BLOCKED: opposing FVG on path to target — no trade.",
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
) -> None:
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
        kind = "high" if f.is_high else "low"
        logger.info(
            "  Fractal [%s] price=%.2f at=%s",
            kind,
            f.price,
            f.timestamp.strftime("%Y-%m-%d %H:%M"),
        )


def _find_signal(
    htf_fvgs: list[FVG],
    ltf_fractals: list[Fractal],
    ltf_df: pd.DataFrame,
    fvg_offset_pct: float = 0.0,
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
    for fvg in reversed(bullish_fvgs):  # most-recent FVG first
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
    for fvg in reversed(bearish_fvgs):  # most-recent FVG first
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


def _has_blocking_fvg(
    fvgs: list[FVG],
    direction: Trend,
    entry: float,
    target: float,
    block_tested: bool = False,
) -> bool:
    """True if a qualifying opposing-direction FVG overlaps the entry→target path.

    Active FVGs always qualify. Tested FVGs qualify only when block_tested is True.
    """
    def _qualifies(fvg: FVG) -> bool:
        return fvg.status == FvgStatus.ACTIVE or (block_tested and fvg.status == FvgStatus.TESTED)

    if direction == Trend.BULLISH:
        return any(
            fvg.trend == Trend.BEARISH
            and _qualifies(fvg)
            and fvg.bottom < target
            and fvg.top > entry
            for fvg in fvgs
        )
    else:
        return any(
            fvg.trend == Trend.BULLISH
            and _qualifies(fvg)
            and fvg.top > target
            and fvg.bottom < entry
            for fvg in fvgs
        )


def _compute_levels(
    signal: _EntrySignal,
    fvgs: list[FVG],
    fractals: list[Fractal],
    block_tested_fvgs: bool = False,
) -> tuple[float, float, float] | None:
    """
    Return (entry, stop_loss, take_profit) or None if a blocking FVG is on the path.

    - Entry:      BOS level by default; moved toward stop loss if needed for 2:1.
    - Stop Loss:  exactly at the swing point price.
    - Take Profit: closest valid target (importance-filtered, ≥1:1 RR).
                   Falls back to 2:1 from BOS entry when no structural target exists.
                   Entry is adjusted to achieve exactly 2:1 only when the selected
                   target gives < 2:1 at BOS; a target giving > 2:1 keeps BOS entry.
    - Returns None when any active opposing FVG sits between entry and target
      (or any tested opposing FVG too, when block_tested_fvgs is True).
    """
    entry = signal.bos_level

    if signal.direction == Trend.BULLISH:
        stop_loss = signal.swing_point.price
        risk = entry - stop_loss
        fallback_tp = entry + 2 * risk
    else:
        stop_loss = signal.swing_point.price
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

    if _has_blocking_fvg(fvgs, signal.direction, entry, target, block_tested_fvgs):
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
) -> str:
    return "\n".join([
        f"Symbol:     {symbol}",
        f"HTF:        {htf_timeframe.value}  ({len(htf_df)} candles)",
        f"LTF:        {ltf_timeframe.value}  ({len(ltf_df)} candles)",
        f"HTF range:  {_ts(htf_df['timestamp'].iloc[0])} → {_ts(htf_df['timestamp'].iloc[-2])}",
        f"LTF range:  {_ts(ltf_df['timestamp'].iloc[0])} → {_ts(ltf_df['timestamp'].iloc[-1])}",
    ])


def _format_htf_poi(signal: _EntrySignal) -> str:
    return "\n".join([
        "HTF FVG (Point of Interest)",
        f"  Status: {signal.fvg.status.value}",
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
        f"{swing_label} (after FVG)",
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
                status = fvg.status.value
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
                status = fvg.status.value
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
                status = fvg.status.value
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
                status = fvg.status.value
                lines.append(
                    f"  {_fmt(fvg.bottom)}–{_fmt(fvg.top)}"
                    f"  ({_ts(fvg.timestamp)})  [{status}]"
                )
        else:
            lines.append("  (none)")

    return "\n".join(lines)


def _format_candles(df: pd.DataFrame, timeframe: Timeframe, n_candles: int = 20) -> str:
    rows_df = df.iloc[:-1].tail(n_candles)
    header = (
        f"HTF Candles ({timeframe.value}, last {len(rows_df)} of {len(df)} candles)\n"
        f"{'timestamp':<20} {'open':>12} {'high':>12} {'low':>12} {'close':>12} {'volume':>14}\n"
        + "-" * 86
    )
    rows = [
        f"{row['timestamp'].strftime('%Y-%m-%d %H:%M'):<20} "
        f"{row['open']:>12.2f} {row['high']:>12.2f} "
        f"{row['low']:>12.2f} {row['close']:>12.2f} {row['volume']:>14.4f}"
        for _, row in rows_df.iterrows()
    ]
    return header + "\n" + "\n".join(rows)
