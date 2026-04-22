"""One-time runner — fetches a single candle snapshot and runs the strategy once."""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

from trading.agents.trade_validation_agent import (
    TradeValidationAgent,
    build_prompt,
)
from trading.core.models import Trend
from trading.data.binance_datasource import BinanceDataSource
from trading.data.csv_datasource import CSVDataSource
from trading.strategies import HtfFvgLtfBos
from trading.strategies.htf_fvg_ltf_bos import format_strategy_components

from .config import _FMT, _TF_SECONDS, RunConfig, _ts


class OneTimeRunner:
    """Fetches data for a single snapshot and runs the strategy once."""

    def __init__(self, config: RunConfig, data_dir: Path) -> None:
        self._config = config
        self._data_dir = data_dir

    def run(self, gui_output: Callable[[str], None]) -> None:
        cfg = self._config

        def out(s: str) -> None:
            gui_output(s + "\n")

        try:
            htf_df, ltf_df = self._fetch_data()
        except Exception as exc:
            out(f"[ERROR] {exc}")
            return

        strategy = HtfFvgLtfBos(fvg_offset_pct=cfg.fvg_offset_pct)

        if cfg.output_mode == "strategy_inspect":
            gui_output(
                format_strategy_components(
                    cfg.symbol, htf_df, cfg.htf_tf, ltf_df, cfg.ltf_tf,
                    cfg.fvg_offset_pct,
                )
            )
            return

        setup = strategy.detect_entry(
            cfg.symbol, htf_df, cfg.htf_tf, ltf_df, cfg.ltf_tf
        )

        if setup is None:
            out(f"Strategy:   {strategy.name}")
            out(f"Symbol:     {cfg.symbol}")
            out(f"HTF:        {cfg.htf_tf.value}  ({cfg.htf_limit} candles)")
            out(f"LTF:        {cfg.ltf_tf.value}  ({cfg.ltf_limit} candles)")
            htf_start = htf_df["timestamp"].iloc[0].strftime("%Y-%m-%d %H:%M")
            htf_end = htf_df["timestamp"].iloc[-1].strftime("%Y-%m-%d %H:%M")
            ltf_start = ltf_df["timestamp"].iloc[0].strftime("%Y-%m-%d %H:%M")
            ltf_end = ltf_df["timestamp"].iloc[-1].strftime("%Y-%m-%d %H:%M")
            out(f"HTF range:  {htf_start} → {htf_end}")
            out(f"LTF range:  {ltf_start} → {ltf_end}")
            out(f"FVG offset: {cfg.fvg_offset_pct * 100:.1f} %")
            out("─" * 44)
            out("No entry detected.")
            return

        if cfg.output_mode == "prompt":
            prompt = build_prompt(setup)
            out("─" * 44)
            out("PROMPT VALIDATION — agent not called\n")
            gui_output(prompt)

        elif cfg.output_mode == "agent":
            assert cfg.llm_config is not None
            prompt = build_prompt(setup)
            out("Sending to agent…\n")
            agent = TradeValidationAgent(cfg.llm_config)
            gui_output(agent.run(prompt))

        else:  # baseline
            entry = setup.entry
            stop_loss = setup.stop_loss
            take_profit = setup.take_profit
            risk = abs(entry - stop_loss)
            rr_actual = abs(take_profit - entry) / risk if risk else 0.0
            risk_pct = risk / entry * 100

            out("─" * 44)
            out("BASELINE EVALUATION")
            out("")
            out(f"Direction:   {setup.direction.value.upper()}")
            out(f"Entry:       {_FMT.format(entry)}")
            out(
                f"Stop Loss:   {_FMT.format(stop_loss)}"
                f"  (risk {_FMT.format(risk)}, {risk_pct:.2f}%)"
            )
            out(f"Take Profit: {_FMT.format(take_profit)}  ({rr_actual:.1f}:1 RR)")

            if risk_pct > cfg.max_risk_pct:
                out("")
                out(
                    f"SKIPPED: risk {risk_pct:.2f}%"
                    f" exceeds max {cfg.max_risk_pct:.1f}%"
                )
                return

            out("")
            last_ltf_ts = ltf_df.iloc[-1]["timestamp"]
            future = self._fetch_future_ltf(last_ltf_ts)
            if future is None or future.empty:
                out("Evaluation: no future data available")
            else:
                result, fill_dt, close_dt = self._evaluate_order(
                    entry=entry,
                    sl=stop_loss,
                    tp=take_profit,
                    direction=setup.direction,
                    future_candles=future,
                    order_timeout=cfg.order_timeout,
                )
                out(f"Result:      {result}")
                if fill_dt is not None:
                    out(f"Filled at:   {_ts(fill_dt)}")
                if close_dt is not None:
                    out(f"Closed at:   {_ts(close_dt)}")
                if result == "OPEN":
                    out(f"  (checked {len(future)} future candles)")
                elif result == "CANCELED_TIMEOUT":
                    out(
                        f"  (limit order not filled within"
                        f" {cfg.order_timeout} candles)"
                    )

    def _fetch_future_ltf(
        self, last_ltf_ts: datetime, count: int = 200
    ) -> pd.DataFrame | None:
        """Return up to ``count`` LTF candles that open after ``last_ltf_ts``."""
        cfg = self._config
        ltf_step = _TF_SECONDS[cfg.ltf_tf.value]

        if cfg.data_source == "csv":
            src = CSVDataSource(data_dir=self._data_dir)
            all_ltf = src.get_ohlcv(
                symbol=cfg.symbol,
                timeframe=cfg.ltf_tf.value,
                limit=999_999,
                filename_override=cfg.ltf_csv,
            )
            future = (
                all_ltf[all_ltf["timestamp"] > last_ltf_ts]
                .head(count)
                .reset_index(drop=True)
            )
            return future if not future.empty else None

        if cfg.data_source == "past":
            since = last_ltf_ts + timedelta(seconds=ltf_step)
            binance = BinanceDataSource()
            df = binance.get_ohlcv(
                symbol=cfg.symbol,
                timeframe=cfg.ltf_tf.value,
                limit=count,
                since=since,
            )
            return df if not df.empty else None

        return None  # live — future hasn't happened yet

    @staticmethod
    def _evaluate_order(
        entry: float,
        sl: float,
        tp: float,
        direction: Trend,
        future_candles: pd.DataFrame,
        order_timeout: int,
    ) -> tuple[str, datetime | None, datetime | None]:
        """Simulate a limit order against future candles.

        Returns (result, fill_dt, close_dt).
        Results: WIN | LOSS | CANCELED_PRICE | CANCELED_TIMEOUT | OPEN
        """
        filled = False
        fill_dt: datetime | None = None
        bullish = direction == Trend.BULLISH
        candles_waiting = 0

        for _, row in future_candles.iterrows():
            candle_dt: datetime = row["timestamp"]

            if not filled:
                candles_waiting += 1

                # Cancel if price reaches TP before fill
                if (bullish and row["high"] >= tp) or (
                    not bullish and row["low"] <= tp
                ):
                    return "CANCELED_PRICE", None, candle_dt

                # Check fill
                fill_now = (bullish and row["low"] <= entry) or (
                    not bullish and row["high"] >= entry
                )
                if fill_now:
                    filled = True
                    fill_dt = candle_dt
                    if bullish:
                        if row["low"] <= sl:
                            return "LOSS", fill_dt, candle_dt
                        if row["high"] >= tp:
                            return "WIN", fill_dt, candle_dt
                    else:
                        if row["high"] >= sl:
                            return "LOSS", fill_dt, candle_dt
                        if row["low"] <= tp:
                            return "WIN", fill_dt, candle_dt
                    continue  # filled, pending resolution

                if candles_waiting >= order_timeout:
                    return "CANCELED_TIMEOUT", None, candle_dt

            else:
                if bullish:
                    if row["high"] >= tp:
                        return "WIN", fill_dt, candle_dt
                    if row["low"] <= sl:
                        return "LOSS", fill_dt, candle_dt
                else:
                    if row["low"] <= tp:
                        return "WIN", fill_dt, candle_dt
                    if row["high"] >= sl:
                        return "LOSS", fill_dt, candle_dt

        return "OPEN", fill_dt, None

    def _fetch_data(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        cfg = self._config
        if cfg.data_source == "csv":
            src = CSVDataSource(data_dir=self._data_dir)
            htf_df = src.get_ohlcv(
                symbol=cfg.symbol,
                timeframe=cfg.htf_tf.value,
                limit=cfg.htf_limit,
                filename_override=cfg.htf_csv,
            )
            ltf_df = src.get_ohlcv(
                symbol=cfg.symbol,
                timeframe=cfg.ltf_tf.value,
                limit=cfg.ltf_limit,
                filename_override=cfg.ltf_csv,
            )
        else:
            binance = BinanceDataSource()
            htf_since: datetime | None = None
            ltf_since: datetime | None = None
            if cfg.data_source == "past" and cfg.until is not None:
                htf_since = cfg.until - timedelta(
                    seconds=cfg.htf_limit * _TF_SECONDS[cfg.htf_tf.value]
                )
                ltf_since = cfg.until - timedelta(
                    seconds=cfg.ltf_limit * _TF_SECONDS[cfg.ltf_tf.value]
                )
            htf_df = binance.get_ohlcv(
                symbol=cfg.symbol,
                timeframe=cfg.htf_tf.value,
                limit=cfg.htf_limit,
                since=htf_since,
            )
            ltf_df = binance.get_ohlcv(
                symbol=cfg.symbol,
                timeframe=cfg.ltf_tf.value,
                limit=cfg.ltf_limit,
                since=ltf_since,
            )
        return htf_df, ltf_df
