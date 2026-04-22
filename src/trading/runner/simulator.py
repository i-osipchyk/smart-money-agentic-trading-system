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

    Multiple concurrent orders are supported:
    - Same-direction setup while orders are active → new order added.
    - Opposite-direction setup → all active orders closed at market (candle
      close), new order placed.

    All verbose per-step lines are sent to ``detail_log``.
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

    def _close_at_market(
        self,
        o: dict[str, Any],
        exit_price: float,
        current_dt: Any,
        trades: list[TradeRecord],
    ) -> None:
        """Close a filled order at ``exit_price`` (market reversal)."""
        o_bull = o["direction"] == Trend.BULLISH
        result = (
            "WIN"
            if (o_bull and exit_price >= o["entry"]) or
               (not o_bull and exit_price <= o["entry"])
            else "LOSS"
        )
        closed = dict(o)
        closed["tp"] = exit_price  # store actual exit for RR calculation
        trades.append(self._finalize(closed, result, current_dt))
        self._log(
            f"  REVERSED #{o['trade_num']} {result}"
            f"  exit={_FMT.format(exit_price)}"
        )

    def run(
        self,
        bt_source: BacktestDataSource,
        get_decision: Callable[[StrategySetup], TradeDecision | None],
    ) -> SimulationResult:
        active_orders: list[dict[str, Any]] = []
        trades: list[TradeRecord] = []
        skipped_no_trade = 0
        skipped_risk = 0
        step_num = 0

        for current_dt, htf_df, ltf_df in bt_source:
            step_num += 1
            candle = ltf_df.iloc[-1]

            # ---- manage all active orders ------------------------------------
            still_active: list[dict[str, Any]] = []
            for o in active_orders:
                bullish = o["direction"] == Trend.BULLISH
                closed = False

                if not o["filled"]:
                    o["candles_waiting"] += 1

                    # Cancel: price reached TP before fill
                    if (bullish and candle["high"] >= o["tp"]) or \
                       (not bullish and candle["low"] <= o["tp"]):
                        trades.append(self._finalize(o, "CANCELED_PRICE", current_dt))
                        closed = True

                    else:
                        filled_now = (
                            (bullish and candle["low"] <= o["entry"]) or
                            (not bullish and candle["high"] >= o["entry"])
                        )
                        if filled_now:
                            o["filled"] = True
                            o["fill_dt"] = current_dt
                            if bullish:
                                if candle["low"] <= o["sl"]:
                                    trades.append(self._finalize(o, "LOSS", current_dt))
                                    closed = True
                                elif candle["high"] >= o["tp"]:
                                    trades.append(self._finalize(o, "WIN", current_dt))
                                    closed = True
                            else:
                                if candle["high"] >= o["sl"]:
                                    trades.append(self._finalize(o, "LOSS", current_dt))
                                    closed = True
                                elif candle["low"] <= o["tp"]:
                                    trades.append(self._finalize(o, "WIN", current_dt))
                                    closed = True

                        elif o["candles_waiting"] >= self._order_timeout:
                            trades.append(self._finalize(o, "CANCELED_TIMEOUT", current_dt))
                            closed = True

                else:
                    # Filled — track TP/SL each candle
                    if bullish:
                        if candle["high"] >= o["tp"]:
                            trades.append(self._finalize(o, "WIN", current_dt))
                            closed = True
                        elif candle["low"] <= o["sl"]:
                            trades.append(self._finalize(o, "LOSS", current_dt))
                            closed = True
                    else:
                        if candle["low"] <= o["tp"]:
                            trades.append(self._finalize(o, "WIN", current_dt))
                            closed = True
                        elif candle["high"] >= o["sl"]:
                            trades.append(self._finalize(o, "LOSS", current_dt))
                            closed = True

                if not closed:
                    still_active.append(o)

            active_orders = still_active

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

            # ---- close opposite-direction orders at market ------------------
            exit_price = float(candle["close"])
            same_dir: list[dict[str, Any]] = []
            for o in active_orders:
                if o["direction"] != td.direction:
                    if o["filled"]:
                        self._close_at_market(o, exit_price, current_dt, trades)
                    else:
                        # Unfilled pending order in opposite direction — cancel it
                        trades.append(self._finalize(o, "CANCELED_PRICE", current_dt))
                        self._log(f"  REVERSED (pending) #{o['trade_num']} CANCELED")
                else:
                    same_dir.append(o)
            active_orders = same_dir

            # ---- place new order --------------------------------------------
            trade_num = len(trades) + 1
            self._log("─" * 60)
            self._log(f"Trade #{trade_num}  at  {_ts(current_dt)} UTC")
            self._log(f"Direction:   {td.direction.value.upper()}")
            self._log(f"Entry:       {_FMT.format(td.entry_price)}  (limit at BOS level)")
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

            active_orders.append({
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
            })

        for o in active_orders:
            trades.append(self._finalize(o, "OPEN", None))

        return SimulationResult(
            trades=trades,
            skipped_no_trade=skipped_no_trade,
            skipped_risk=skipped_risk,
            skipped_active_order=0,
            steps_checked=step_num,
        )
