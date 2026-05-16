import logging
from typing import Literal

import pandas as pd

from trading.core.models import FVG, Fractal, FvgStatus, StrategySetup, Timeframe, Trend
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


class HtfFvgLtfBosV2(Strategy):
    """
    HTF FVG + LTF BOS strategy (v2 — with target selection and entry adjustment).

    Args:
        fvg_offset_pct:   Extends the qualifying zone beyond the FVG edge. Default 0.0005 (0.05 %).
        block_fvg_mode:   Which opposing FVGs block the path filter.
        use_trend_filter: Discard setups that conflict with HTF trend.
        min_rr_ratio:     Minimum RR for structural target to qualify. Default 1.0.
        htf_fvg_limit:    Tail of HTF candles used for FVG detection.
        htf_target_limit: Tail of HTF candles used for target detection.
    """

    name = "htf_fvg_ltf_bos_v2"
    description = _DESCRIPTION

    def __init__(
        self,
        fvg_offset_pct: float = 0.0005,
        block_fvg_mode: Literal["none", "active", "tested", "active+tested"] = "active",
        use_trend_filter: bool = True,
        min_rr_ratio: float = 1.0,
        htf_fvg_limit: int | None = None,
        htf_target_limit: int | None = None,
    ) -> None:
        self._fvg_offset_pct = fvg_offset_pct
        self._block_fvg_mode = block_fvg_mode
        self._use_trend_filter = use_trend_filter
        self._min_rr_ratio = min_rr_ratio
        self._htf_fvg_limit = htf_fvg_limit
        self._htf_target_limit = htf_target_limit

    def detect_entry(
        self,
        symbol: str,
        htf_df: pd.DataFrame,
        htf_timeframe: Timeframe,
        ltf_df: pd.DataFrame,
        ltf_timeframe: Timeframe,
    ) -> StrategySetup | None:
        htf_fvg_df = htf_df.tail(self._htf_fvg_limit) if self._htf_fvg_limit else htf_df
        htf_target_df = htf_df.tail(self._htf_target_limit) if self._htf_target_limit else htf_df

        htf_fvgs = detect_fvg(htf_fvg_df, htf_timeframe)
        htf_fractals = detect_fractals(htf_fvg_df, htf_timeframe)
        htf_target_fractals = detect_fractals(htf_target_df, htf_timeframe)
        ltf_fractals = detect_fractals(ltf_df, ltf_timeframe)

        _log_findings(htf_fvgs, ltf_fractals)

        signal = _find_signal(htf_fvgs, ltf_fractals, ltf_df, self._fvg_offset_pct)
        if signal is None:
            return None

        if self._use_trend_filter:
            htf_trend = trend_from_fractals(htf_fractals)
            if htf_trend is not None and htf_trend != signal.direction:
                return None

        levels = _compute_levels(signal, htf_fvgs, htf_target_fractals, self._block_fvg_mode, self._min_rr_ratio)
        if levels is None:
            return None
        entry, stop_loss, take_profit = levels

        return StrategySetup(
            input_data=_format_input_data(symbol, htf_fvg_df, htf_timeframe, ltf_df, ltf_timeframe, htf_end_idx=-2),
            strategy_description=self.description,
            direction=signal.direction,
            htf_poi=_format_htf_poi(signal),
            confirm_details=_format_confirm_details(signal),
            target=_format_target(htf_fvgs, htf_target_fractals, signal.direction, entry, stop_loss),
            candles=_format_candles(htf_fvg_df, htf_timeframe, n_candles=20),
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
    block_fvg_mode: Literal["none", "active", "tested", "active+tested"] = "active",
    use_trend_filter: bool = True,
    min_rr_ratio: float = 1.0,
    htf_fvg_limit: int | None = None,
    htf_target_limit: int | None = None,
) -> str:
    htf_fvg_df = htf_df.tail(htf_fvg_limit) if htf_fvg_limit else htf_df
    htf_target_df = htf_df.tail(htf_target_limit) if htf_target_limit else htf_df
    htf_fvgs = detect_fvg(htf_fvg_df, htf_timeframe)
    htf_fractals = detect_fractals(htf_fvg_df, htf_timeframe)
    htf_target_fractals = detect_fractals(htf_target_df, htf_timeframe)
    ltf_fractals = detect_fractals(ltf_df, ltf_timeframe)

    bullish_fvgs = [f for f in htf_fvgs if f.trend == Trend.BULLISH]
    bearish_fvgs = [f for f in htf_fvgs if f.trend == Trend.BEARISH]

    htf_sorted = sorted(htf_fractals, key=lambda f: f.timestamp)
    htf_highs = [f for f in htf_sorted if f.is_high]
    htf_lows = [f for f in htf_sorted if not f.is_high]

    ltf_sorted = sorted(ltf_fractals, key=lambda f: f.timestamp)
    ltf_highs = [f for f in ltf_sorted if f.is_high]
    ltf_lows = [f for f in ltf_sorted if not f.is_high]

    sep = "─" * 56
    lines: list[str] = [
        "STRATEGY INSPECTION — HTF FVG + LTF BOS",
        sep,
        _format_input_data(symbol, htf_df, htf_timeframe, ltf_df, ltf_timeframe),
        "",
    ]

    lines += [
        sep,
        f"HTF FVGs  ({len(htf_fvgs)} total: {len(bullish_fvgs)} bullish, {len(bearish_fvgs)} bearish)",
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
    lines.append("")

    for label, f_highs, f_lows, tf in [
        ("HTF", htf_highs, htf_lows, "htf"),
        ("LTF", ltf_highs, ltf_lows, "ltf"),
    ]:
        n_h, n_l = len(f_highs), len(f_lows)
        lines += [sep, f"{label} Fractals  ({n_h + n_l} total: {n_h} highs, {n_l} lows)", ""]
        for kind, fl in [("Swing Highs", f_highs), ("Swing Lows", f_lows)]:
            lines.append(f"  {kind} ({len(fl)})")
            for i, f in enumerate(fl, 1):
                lines.append(f"    {i}. {_fmt(f.price)}  at {_ts(f.timestamp)}")
            if not fl:
                lines.append("    (none)")
        lines.append("")

    last_candle_ts = ltf_df["timestamp"].iloc[-1]
    lines += [sep, "Per-FVG BOS Analysis", ""]
    lines += _format_per_fvg_bos_analysis(
        bullish_fvgs, bearish_fvgs, ltf_highs, ltf_lows,
        ltf_df, last_candle_ts, fvg_offset_pct,
    )

    signal = _find_signal(htf_fvgs, ltf_fractals, ltf_df, fvg_offset_pct)
    lines += [sep, "ENTRY DETECTION RESULT", ""]
    if signal is None:
        lines.append("  No entry detected.")
    elif use_trend_filter and trend_from_fractals(htf_fractals) not in (None, signal.direction):
        lines.append("  No entry detected (trend filter).")
    else:
        levels = _compute_levels(signal, htf_fvgs, htf_target_fractals, block_fvg_mode, min_rr_ratio)
        bos = signal.bos_level
        _sl = signal.swing_point.price
        _risk = bos - _sl if signal.direction == Trend.BULLISH else _sl - bos
        if levels is not None:
            _disp_entry, _, _disp_tp = levels
        else:
            _raw_target = _select_target(htf_fvgs, htf_target_fractals, signal.direction, bos, _sl, min_rr_ratio)
            _fallback = bos + 2 * _risk if signal.direction == Trend.BULLISH else bos - 2 * _risk
            _disp_tp = _raw_target if _raw_target is not None else _fallback
            _disp_entry = bos
            if _raw_target is not None and _risk:
                _rr = (_disp_tp - bos) / _risk if signal.direction == Trend.BULLISH else (bos - _disp_tp) / _risk
                if _rr < 2.0:
                    _adj = (_disp_tp + 2 * _sl) / 3
                    if not _has_blocking_fvg(htf_fvgs, signal.direction, _adj, _disp_tp, block_fvg_mode):
                        _disp_entry = _adj
        _rr_disp = abs(_disp_tp - _disp_entry) / _risk if _risk else 0.0
        lines += [
            f"  Direction:   {signal.direction.value.upper()}",
            f"  FVG:         bottom {_fmt(signal.fvg.bottom)}  top {_fmt(signal.fvg.top)}",
            f"  Swing Point: {_fmt(signal.swing_point.price)}  at {_ts(signal.swing_point.timestamp)}",
            f"  BOS Level:   {_fmt(signal.bos_level)}  confirmed at {_ts(signal.bos_candle_timestamp)}",
            f"  Entry:       {_fmt(_disp_entry)}",
            f"  Stop Loss:   {_fmt(_sl)}",
            f"  Take Profit: {_fmt(_disp_tp)}  ({_rr_disp:.1f}:1 RR)",
        ]
        if levels is None:
            _raw_t = _select_target(htf_fvgs, htf_target_fractals, signal.direction, bos, _sl, min_rr_ratio)
            if _raw_t is None and _has_structural_targets(htf_fvgs, htf_target_fractals, signal.direction, bos):
                lines.append("  FILTERED: structural target below min RR — no trade.")
            else:
                lines.append("  BLOCKED: opposing FVG on path to target — no trade.")

    return "\n".join(lines)


# ------------------------------------------------------------------ internals

def _select_target(
    fvgs: list[FVG],
    fractals: list[Fractal],
    direction: Trend,
    entry: float,
    stop_loss: float,
    min_rr_ratio: float = 1.0,
) -> float | None:
    candidates: list[float] = []
    if direction == Trend.BULLISH:
        risk = entry - stop_loss
        min_tp = entry + risk * min_rr_ratio
        all_highs = sorted([f for f in fractals if f.is_high], key=lambda f: f.timestamp)
        candidates = [f.price for f in _important_swings(all_highs) if f.price >= min_tp]
        candidates += [fvg.bottom for fvg in fvgs if fvg.trend == Trend.BEARISH and fvg.bottom >= min_tp]
        return min(candidates) if candidates else None
    else:
        risk = stop_loss - entry
        min_tp = entry - risk * min_rr_ratio
        all_lows = sorted([f for f in fractals if not f.is_high], key=lambda f: f.timestamp)
        candidates = [f.price for f in _important_swings(all_lows) if f.price <= min_tp]
        candidates += [fvg.top for fvg in fvgs if fvg.trend == Trend.BULLISH and fvg.top <= min_tp]
        return max(candidates) if candidates else None


def _has_structural_targets(
    fvgs: list[FVG], fractals: list[Fractal], direction: Trend, entry: float,
) -> bool:
    if direction == Trend.BULLISH:
        all_highs = sorted([f for f in fractals if f.is_high], key=lambda f: f.timestamp)
        if any(f.price > entry for f in _important_swings(all_highs)):
            return True
        return any(fvg.trend == Trend.BEARISH and fvg.bottom > entry for fvg in fvgs)
    else:
        all_lows = sorted([f for f in fractals if not f.is_high], key=lambda f: f.timestamp)
        if any(f.price < entry for f in _important_swings(all_lows)):
            return True
        return any(fvg.trend == Trend.BULLISH and fvg.top < entry for fvg in fvgs)


def _compute_levels(
    signal: _EntrySignal,
    fvgs: list[FVG],
    fractals: list[Fractal],
    block_fvg_mode: Literal["none", "active", "tested", "active+tested"] = "active",
    min_rr_ratio: float = 1.0,
) -> tuple[float, float, float] | None:
    entry = signal.bos_level
    stop_loss = signal.swing_point.price
    if signal.direction == Trend.BULLISH:
        risk = entry - stop_loss
        fallback_tp = entry + 2 * risk
    else:
        risk = stop_loss - entry
        fallback_tp = entry - 2 * risk

    target = _select_target(fvgs, fractals, signal.direction, entry, stop_loss, min_rr_ratio)
    if target is None:
        if _has_structural_targets(fvgs, fractals, signal.direction, entry):
            return None
        target = fallback_tp
    else:
        rr = (target - entry) / risk if signal.direction == Trend.BULLISH else (entry - target) / risk
        if rr < 2.0:
            adjusted = (target + 2 * stop_loss) / 3
            if not _has_blocking_fvg(fvgs, signal.direction, adjusted, target, block_fvg_mode):
                entry = adjusted

    if _has_blocking_fvg(fvgs, signal.direction, entry, target, block_fvg_mode):
        return None
    return entry, stop_loss, target


def _important_swings(swings: list[Fractal]) -> list[Fractal]:
    if len(swings) <= 1:
        return list(swings)
    is_high = swings[0].is_high
    result: list[Fractal] = []
    if is_high and swings[0].price > swings[1].price:
        result.append(swings[0])
    elif not is_high and swings[0].price < swings[1].price:
        result.append(swings[0])
    for i in range(1, len(swings) - 1):
        prev_p, cur_p, next_p = swings[i - 1].price, swings[i].price, swings[i + 1].price
        if (is_high and cur_p > prev_p and cur_p > next_p) or (not is_high and cur_p < prev_p and cur_p < next_p):
            result.append(swings[i])
    if is_high and swings[-1].price > swings[-2].price:
        result.append(swings[-1])
    elif not is_high and swings[-1].price < swings[-2].price:
        result.append(swings[-1])
    return result


# ------------------------------------------------------------ formatters

def _format_htf_poi(signal: _EntrySignal) -> str:
    return "\n".join([
        "HTF FVG (Point of Interest)",
        f"  Status: {signal.fvg.status.value}",
        f"  Top:    {_fmt(signal.fvg.top)}",
        f"  Bottom: {_fmt(signal.fvg.bottom)}",
        f"  Formed: {_ts(signal.fvg.timestamp)} (third candle close time)",
    ])


def _format_confirm_details(signal: _EntrySignal) -> str:
    swing_label = "Lowest LTF Swing Low" if signal.direction == Trend.BULLISH else "Highest LTF Swing High"
    prior_label = "Prior LTF Swing High" if signal.direction == Trend.BULLISH else "Prior LTF Swing Low"
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


def _format_target(
    fvgs: list[FVG], fractals: list[Fractal], direction: Trend, entry: float, stop_loss: float,
) -> str:
    lines: list[str] = []
    if direction == Trend.BULLISH:
        risk = entry - stop_loss
        min_tp = entry + risk
        all_highs = sorted([f for f in fractals if f.is_high], key=lambda f: f.timestamp)
        highs = sorted([f for f in _important_swings(all_highs) if f.price >= min_tp], key=lambda f: f.price)
        lines.append(f"HTF Swing Highs (≥1:1 RR from entry {_fmt(entry)}, nearest first)")
        if highs:
            lines.extend(f"  {i}. {_fmt(f.price)}  ({_ts(f.timestamp)})" for i, f in enumerate(highs, 1))
        else:
            lines.append("  (none)")
        lines.append("")
        bearish = sorted([f for f in fvgs if f.trend == Trend.BEARISH and f.bottom >= min_tp], key=lambda f: f.bottom)
        lines.append("Valid HTF Bearish FVGs — low (≥1:1 RR, nearest first)")
        if bearish:
            lines.extend(f"  {i}. {_fmt(fvg.bottom)}  ({_ts(fvg.timestamp)})  [{fvg.status.value}]" for i, fvg in enumerate(bearish, 1))
        else:
            lines.append("  (none)")
        lines.append("")
        lines.append("All HTF Bearish FVGs in path (entry → target) — obstruction check")
        path_blockers = sorted([f for f in fvgs if f.trend == Trend.BEARISH and f.top > entry], key=lambda f: f.bottom)
        if path_blockers:
            lines.extend(f"  {_fmt(fvg.bottom)}–{_fmt(fvg.top)}  ({_ts(fvg.timestamp)})  [{fvg.status.value}]" for fvg in path_blockers)
        else:
            lines.append("  (none)")
    else:
        risk = stop_loss - entry
        min_tp = entry - risk
        all_lows = sorted([f for f in fractals if not f.is_high], key=lambda f: f.timestamp)
        lows = sorted([f for f in _important_swings(all_lows) if f.price <= min_tp], key=lambda f: f.price, reverse=True)
        lines.append(f"HTF Swing Lows (≥1:1 RR from entry {_fmt(entry)}, nearest first)")
        if lows:
            lines.extend(f"  {i}. {_fmt(f.price)}  ({_ts(f.timestamp)})" for i, f in enumerate(lows, 1))
        else:
            lines.append("  (none)")
        lines.append("")
        bullish = sorted([f for f in fvgs if f.trend == Trend.BULLISH and f.top <= min_tp], key=lambda f: f.top, reverse=True)
        lines.append("Valid HTF Bullish FVGs — high (≥1:1 RR, nearest first)")
        if bullish:
            lines.extend(f"  {i}. {_fmt(fvg.top)}  ({_ts(fvg.timestamp)})  [{fvg.status.value}]" for i, fvg in enumerate(bullish, 1))
        else:
            lines.append("  (none)")
        lines.append("")
        lines.append("All HTF Bullish FVGs in path (target → entry) — obstruction check")
        path_blockers = sorted([f for f in fvgs if f.trend == Trend.BULLISH and f.bottom < entry], key=lambda f: f.top, reverse=True)
        if path_blockers:
            lines.extend(f"  {_fmt(fvg.bottom)}–{_fmt(fvg.top)}  ({_ts(fvg.timestamp)})  [{fvg.status.value}]" for fvg in path_blockers)
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
