"""Order simulator — simulates limit-order lifecycle over a BacktestDataSource."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from trading.core.models import StrategySetup, Timeframe, TradeDecision, Trend
from trading.data.backtest_datasource import BacktestDataSource
from trading.strategies import HtfFvgLtfBos

from .config import _FMT, SimulationResult, TradeRecord, _ts


class OrderSimulator:
    """
    Simulates limit-order lifecycle over a BacktestDataSource iteration.

    All verbose per-step lines are sent to ``detail_log``; the caller decides
    where they go (file, stdout, /dev/null).  Returns a ``SimulationResult``
    with no side effects.
    """

    def __init__(
        self,
        strategy: HtfFvgLtfBos,
        symbol: str,
        htf_tf: Timeframe,
        ltf_tf: Timeframe,
        order_timeout: int,
        max_risk_pct: float | None,
        detail_log: Callable[[str], None],
    ) -> None:
        self._strategy = strategy
        self._symbol = symbol
        self._htf_tf = htf_tf
        self._ltf_tf = ltf_tf
        self._order_timeout = order_timeout
        self._max_risk_pct = max_risk_pct
        self._detail_log = detail_log

    def _log(self, s: str) -> None:
        self._detail_log(s + "\n")

    @staticmethod
    def _finalize(
        o: dict[str, Any], result: str, close_dt: Any | None
    ) -> TradeRecord:
        return TradeRecord(
            trade_num=o["trade_num"],
            setup_dt=o["setup_dt"],
            direction=o["direction"],
            entry=o["entry"],
            sl=o["sl"],
            tp=o["tp"],
            result=result,
            fill_dt=o["fill_dt"],
            close_dt=close_dt,
            reasoning=o["reasoning"],
            confidence=o["confidence"],
        )

    def run(
        self,
        bt_source: BacktestDataSource,
        get_decision: Callable[[StrategySetup], TradeDecision | None],
    ) -> SimulationResult:
        active_order: dict[str, Any] | None = None
        trades: list[TradeRecord] = []
        skipped_no_trade = 0
        skipped_risk = 0
        step_num = 0

        for current_dt, htf_df, ltf_df in bt_source:
            step_num += 1
            candle = ltf_df.iloc[-1]

            # ---- manage active order ----------------------------------------
            if active_order is not None:
                o = active_order
                bullish = o["direction"] == Trend.BULLISH

                if not o["filled"]:
                    o["candles_waiting"] += 1

                    # Cancel: price reached TP before fill
                    if (bullish and candle["high"] >= o["tp"]) or \
                       (not bullish and candle["low"] <= o["tp"]):
                        trades.append(self._finalize(o, "CANCELED_PRICE", current_dt))
                        active_order = None
                        # fall through to detect_entry on the same candle

                    else:
                        filled_this_candle = (
                            (bullish and candle["low"] <= o["entry"]) or
                            (not bullish and candle["high"] >= o["entry"])
                        )
                        if filled_this_candle:
                            o["filled"] = True
                            o["fill_dt"] = current_dt
                            if bullish:
                                if candle["low"] <= o["sl"]:
                                    trades.append(
                                        self._finalize(o, "LOSS", current_dt)
                                    )
                                    active_order = None
                                    continue
                                if candle["high"] >= o["tp"]:
                                    trades.append(
                                        self._finalize(o, "WIN", current_dt)
                                    )
                                    active_order = None
                                    continue
                            else:
                                if candle["high"] >= o["sl"]:
                                    trades.append(
                                        self._finalize(o, "LOSS", current_dt)
                                    )
                                    active_order = None
                                    continue
                                if candle["low"] <= o["tp"]:
                                    trades.append(
                                        self._finalize(o, "WIN", current_dt)
                                    )
                                    active_order = None
                                    continue
                            continue  # filled, pending resolution

                        if o["candles_waiting"] >= self._order_timeout:
                            trades.append(
                                self._finalize(o, "CANCELED_TIMEOUT", current_dt)
                            )
                            active_order = None
                            # fall through to detect_entry on the same candle
                        else:
                            continue

                else:
                    # Filled — track TP/SL each candle
                    if bullish:
                        if candle["high"] >= o["tp"]:
                            trades.append(self._finalize(o, "WIN", current_dt))
                            active_order = None
                        elif candle["low"] <= o["sl"]:
                            trades.append(self._finalize(o, "LOSS", current_dt))
                            active_order = None
                    else:
                        if candle["low"] <= o["tp"]:
                            trades.append(self._finalize(o, "WIN", current_dt))
                            active_order = None
                        elif candle["high"] >= o["sl"]:
                            trades.append(self._finalize(o, "LOSS", current_dt))
                            active_order = None
                    if active_order is not None:
                        continue
                    # order just closed — fall through to detect_entry

            # ---- detect setup -----------------------------------------------
            try:
                setup = self._strategy.detect_entry(
                    self._symbol, htf_df, self._htf_tf, ltf_df, self._ltf_tf
                )
            except Exception as exc:
                self._log(f"[{_ts(current_dt)}] ERROR: {exc}")
                continue
            if setup is None:
                continue

            # ---- decision callback ------------------------------------------
            td: TradeDecision | None = get_decision(setup)
            if td is None or not td.should_trade:
                skipped_no_trade += 1
                self._log(
                    f"[{_ts(current_dt)}] NO TRADE  —  {td.reasoning if td else '—'}"
                )
                continue

            assert td.direction is not None
            assert td.entry_price is not None and td.stop_loss is not None
            risk = abs(td.entry_price - td.stop_loss)
            risk_pct = risk / td.entry_price * 100
            if self._max_risk_pct is not None and risk_pct > self._max_risk_pct:
                skipped_risk += 1
                self._log(
                    f"[{_ts(current_dt)}] SKIPPED (risk {risk_pct:.2f}%"
                    f" > max {self._max_risk_pct:.1f}%)"
                    f"  entry={_FMT.format(td.entry_price)}"
                    f"  sl={_FMT.format(td.stop_loss)}"
                )
                continue

            trade_num = len(trades) + 1
            self._log("─" * 60)
            self._log(f"Trade #{trade_num}  at  {_ts(current_dt)} UTC")
            self._log(f"Direction:   {td.direction.value.upper()}")
            self._log(
                f"Entry:       {_FMT.format(td.entry_price)}  (limit at BOS level)"
            )
            self._log(
                f"Stop Loss:   {_FMT.format(td.stop_loss)}"
                f"  (risk {_FMT.format(risk)})"
            )
            if td.take_profit is not None:
                rr_actual = abs(td.take_profit - td.entry_price) / risk
                self._log(
                    f"Take Profit: {_FMT.format(td.take_profit)}"
                    f"  ({rr_actual:.1f}:1 RR)"
                )
            else:
                self._log("Take Profit: —")
            self._log(f"Confidence:  {td.confidence}")
            self._log(f"Reasoning:   {td.reasoning}")

            active_order = {
                "trade_num": trade_num,
                "setup_dt": current_dt,
                "direction": td.direction,
                "entry": td.entry_price,
                "sl": td.stop_loss,
                "tp": td.take_profit,
                "risk": risk,
                "reasoning": td.reasoning,
                "confidence": td.confidence,
                "filled": False,
                "fill_dt": None,
                "candles_waiting": 0,
            }

        if active_order is not None:
            trades.append(self._finalize(active_order, "OPEN", None))

        return SimulationResult(
            trades=trades,
            skipped_no_trade=skipped_no_trade,
            skipped_risk=skipped_risk,
            steps_checked=step_num,
        )
