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
        fvg_offset_pct:     Extends the qualifying zone beyond the FVG edge as a
                            fraction of that edge price: bullish lows qualify down to
                            bottom × (1 − pct); bearish highs qualify up to
                            top × (1 + pct). Default 0.0005 (0.05 %).
        block_tested_fvgs:  When True, tested opposing FVGs also block the path
                            filter (step 8). Default False (only active FVGs block).
    """

    name = "htf_fvg_ltf_bos"
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

        entry, stop_loss, take_profit = _compute_levels(signal)

        if _has_blocking_fvg(htf_fvgs, signal.direction, entry, take_profit, self._block_tested_fvgs):
            return None

        return StrategySetup(
            input_data=_format_input_data(
                symbol, htf_df, htf_timeframe, ltf_df, ltf_timeframe
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

def trend_from_fractals(fractals: list[Fractal]) -> Trend | None:
    highs = sorted([f for f in fractals if f.is_high], key=lambda f: f.timestamp)
    lows = sorted([f for f in fractals if not f.is_high], key=lambda f: f.timestamp)

    if len(highs) < 2 or len(lows) < 2:
        return None

    hh = highs[-1].price > highs[-2].price
    hl = lows[-1].price > lows[-2].price
    lh = highs[-1].price < highs[-2].price
    ll = lows[-1].price < lows[-2].price

    if hh and hl:
        return Trend.BULLISH
    if lh and ll:
        return Trend.BEARISH
    return None


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

def _has_blocking_fvg(
    htf_fvgs: list[FVG],
    direction: Trend,
    entry: float,
    take_profit: float,
    block_tested: bool = False,
) -> bool:
    """True if a qualifying opposing-direction FVG overlaps the entry→TP path.

    Active FVGs always qualify. Tested FVGs qualify only when block_tested is True.
    """
    def _qualifies(fvg: FVG) -> bool:
        return fvg.status == FvgStatus.ACTIVE or (block_tested and fvg.status == FvgStatus.TESTED)

    if direction == Trend.BULLISH:
        return any(
            fvg.trend == Trend.BEARISH
            and _qualifies(fvg)
            and fvg.bottom < take_profit
            and fvg.top > entry
            for fvg in htf_fvgs
        )
    else:
        return any(
            fvg.trend == Trend.BULLISH
            and _qualifies(fvg)
            and fvg.top > take_profit
            and fvg.bottom < entry
            for fvg in htf_fvgs
        )


def _compute_levels(signal: _EntrySignal) -> tuple[float, float, float]:
    """
    Return (entry, stop_loss, take_profit) for a baseline limit order.

    - Entry:      BOS level (limit order placed at the prior swing that was broken).
    - Stop Loss:  exactly at the swing point price.
    - Take Profit: 2:1 reward/risk relative to entry.
    """
    entry = signal.bos_level

    if signal.direction == Trend.BULLISH:
        stop_loss = signal.swing_point.price
        risk = entry - stop_loss
        take_profit = entry + 2 * risk
    else:
        stop_loss = signal.swing_point.price
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
) -> str:
    return "\n".join([
        f"Symbol:     {symbol}",
        f"HTF:        {htf_timeframe.value}  ({len(htf_df)} candles)",
        f"LTF:        {ltf_timeframe.value}  ({len(ltf_df)} candles)",
        f"HTF range:  {_ts(htf_df['timestamp'].iloc[0])} → {_ts(htf_df['timestamp'].iloc[-1])}",
        f"LTF range:  {_ts(ltf_df['timestamp'].iloc[0])} → {_ts(ltf_df['timestamp'].iloc[-1])}",
    ])


def _format_htf_poi(signal: _EntrySignal) -> str:
    return "\n".join([
        "HTF FVG (Point of Interest)",
        f"  Status: {signal.fvg.status.value}",
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
    fvg_offset_pct: float = 0.0,
    block_tested_fvgs: bool = False,  # unused here; accepted for API symmetry with v2
) -> str:
    """Return a full human-readable breakdown of all strategy components."""
    htf_fvgs = detect_fvg(htf_df, htf_timeframe)
    htf_fractals = detect_fractals(htf_df, htf_timeframe)

    bullish_fvgs = [f for f in htf_fvgs if f.trend == Trend.BULLISH]
    bearish_fvgs = [f for f in htf_fvgs if f.trend == Trend.BEARISH]

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
    lines.append(f"  Bullish FVGs ({len(bullish_fvgs)})")
    for i, fvg in enumerate(bullish_fvgs, 1):
        lines.append(
            f"    {i}. bottom {_fmt(fvg.bottom)}  top {_fmt(fvg.top)}"
            f"  formed {_ts(fvg.timestamp)}  [{fvg.status.value}]"
        )
    if not bullish_fvgs:
        lines.append("    (none)")
    lines.append(f"  Bearish FVGs ({len(bearish_fvgs)})")
    for i, fvg in enumerate(bearish_fvgs, 1):
        lines.append(
            f"    {i}. bottom {_fmt(fvg.bottom)}  top {_fmt(fvg.top)}"
            f"  formed {_ts(fvg.timestamp)}  [{fvg.status.value}]"
        )
    if not bearish_fvgs:
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
