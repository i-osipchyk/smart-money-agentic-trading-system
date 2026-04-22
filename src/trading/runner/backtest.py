"""Backtest runner — iterates BacktestDataSource for all output modes."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from trading.agents.trade_validation_agent import (
    TradeValidationAgent,
    build_prompt,
    parse_decision,
)
from trading.core.models import StrategySetup, TradeDecision, Trend
from trading.data.backtest_datasource import BacktestDataSource
from trading.strategies import HtfFvgLtfBos

from .config import _FMT, RunConfig, SimulationResult, TradeRecord, _ts
from .simulator import OrderSimulator


class BacktestRunner:
    """Iterates BacktestDataSource and handles all three output modes."""

    def __init__(self, config: RunConfig) -> None:
        self._config = config

    def run(
        self,
        gui_output: Callable[[str], None],
        detail_output: Callable[[str], None],
        out_path: Path,
    ) -> None:
        cfg = self._config
        assert cfg.bt_from is not None and cfg.bt_to is not None

        strategy = HtfFvgLtfBos(fvg_offset_pct=cfg.fvg_offset_pct)
        bt_source = BacktestDataSource(
            symbol=cfg.symbol,
            htf_timeframe=cfg.htf_tf.value,
            htf_limit=cfg.htf_limit,
            ltf_timeframe=cfg.ltf_tf.value,
            ltf_limit=cfg.ltf_limit,
            bt_from=cfg.bt_from,
            bt_to=cfg.bt_to,
        )

        def both(s: str) -> None:
            gui_output(s)
            detail_output(s)

        header = (
            f"Backtest:  {strategy.name}\n"
            f"Symbol:    {cfg.symbol}\n"
            f"HTF:       {cfg.htf_tf.value}  ({cfg.htf_limit} candles)\n"
            f"LTF:       {cfg.ltf_tf.value}  ({cfg.ltf_limit} candles)\n"
            f"Range:     {cfg.bt_from.strftime('%Y-%m-%d %H:%M')} → "
            f"{cfg.bt_to.strftime('%Y-%m-%d %H:%M')} UTC\n"
            f"Mode:      {cfg.output_mode}\n"
        )
        if cfg.output_mode == "baseline":
            header += (
                f"Timeout:   {cfg.order_timeout} LTF candles\n"
                f"Max risk:  {cfg.max_risk_pct:.1f}% of entry\n"
                f"RR ratio:  {cfg.rr_ratio:.1f}:1\n"
            )
        elif cfg.output_mode == "agent":
            header += (
                f"Timeout:   {cfg.order_timeout} LTF candles\n"
                f"Max risk:  {cfg.max_risk_pct:.1f}% of entry\n"
            )
        header += f"Steps:     {bt_source.total_steps}\n" + "=" * 60 + "\n\n"
        both(header)

        gui_output("Downloading candle data…\n")
        bt_source.prepare(progress=detail_output)
        detail_output("\n")

        out_path.parent.mkdir(parents=True, exist_ok=True)
        detail_lines: list[str] = []

        def capturing_detail(s: str) -> None:
            detail_output(s)
            detail_lines.append(s)

        if cfg.output_mode == "prompt":
            self._run_prompt(bt_source, strategy, gui_output, capturing_detail)
        else:
            self._run_simulation(bt_source, strategy, gui_output, capturing_detail)

        out_path.write_text("".join(detail_lines), encoding="utf-8")
        gui_output(f"\nSaved → {out_path}\n")

    def _run_prompt(
        self,
        bt_source: BacktestDataSource,
        strategy: HtfFvgLtfBos,
        gui_output: Callable[[str], None],
        detail_output: Callable[[str], None],
    ) -> None:
        cfg = self._config
        setups_found = 0
        step_num = 0

        for current_dt, htf_df, ltf_df in bt_source:
            step_num += 1
            try:
                setup = strategy.detect_entry(
                    cfg.symbol, htf_df, cfg.htf_tf, ltf_df, cfg.ltf_tf
                )
            except Exception as exc:
                detail_output(f"[{_ts(current_dt)}] ERROR: {exc}\n")
                continue
            if setup is None:
                continue

            setups_found += 1
            gui_output(f"Setup #{setups_found}  at  {_ts(current_dt)} UTC\n")
            detail_output("─" * 60 + "\n")
            detail_output(f"Setup #{setups_found}  at  {_ts(current_dt)} UTC\n")
            detail_output("─" * 60 + "\n")
            detail_output(build_prompt(setup) + "\n\n")

        summary = (
            f"\n{'=' * 60}\n"
            f"Done.  {setups_found} setup(s) / {step_num} candles checked.\n"
        )
        gui_output(summary)
        detail_output(summary)

    def _run_simulation(
        self,
        bt_source: BacktestDataSource,
        strategy: HtfFvgLtfBos,
        gui_output: Callable[[str], None],
        detail_output: Callable[[str], None],
    ) -> None:
        cfg = self._config

        if cfg.output_mode == "agent":
            assert cfg.llm_config is not None
            agent = TradeValidationAgent(cfg.llm_config)

            def get_decision(setup: StrategySetup) -> TradeDecision:
                detail_output("Querying agent…\n")
                response = agent.run(build_prompt(setup))
                detail_output(response + "\n")
                return parse_decision(cfg.symbol, response, setup)

            metrics_title = "AGENT TEST METRICS"

        else:  # baseline
            def get_decision(setup: StrategySetup) -> TradeDecision:
                return TradeDecision(
                    symbol=cfg.symbol,
                    should_trade=True,
                    direction=setup.direction,
                    entry_price=setup.entry,
                    stop_loss=setup.stop_loss,
                    take_profit=setup.take_profit,
                    reasoning="baseline",
                    confidence="n/a",
                )

            metrics_title = "BASELINE METRICS"

        simulator = OrderSimulator(
            strategy=strategy,
            symbol=cfg.symbol,
            htf_tf=cfg.htf_tf,
            ltf_tf=cfg.ltf_tf,
            order_timeout=cfg.order_timeout,
            max_risk_pct=cfg.max_risk_pct,
            detail_log=detail_output,
        )
        result = simulator.run(bt_source, get_decision)

        # Per-trade summary rows → GUI only
        gui_output("\n")
        for t in result.trades:
            close_str = _ts(t.close_dt) if t.close_dt else "—"
            gui_output(
                f"  Trade #{t.trade_num:<3} {t.direction.value.upper():<5}"
                f"  entry={_FMT.format(t.entry)}"
                f"  sl={_FMT.format(t.sl)}"
                f"  tp={_FMT.format(t.tp) if t.tp is not None else '—'}"
                f"  open={_ts(t.setup_dt)}"
                f"  close={close_str}"
                f"  {t.result}\n"
            )

        # Metrics → both GUI and detail file
        metrics = self._format_metrics(result, metrics_title, cfg.rr_ratio)
        gui_output(metrics)
        detail_output(metrics)

    @staticmethod
    def _format_metrics(result: SimulationResult, title: str, rr_ratio: float) -> str:
        trades = result.trades
        canceled_price   = [t for t in trades if t.result == "CANCELED_PRICE"]
        canceled_timeout = [t for t in trades if t.result == "CANCELED_TIMEOUT"]
        wins   = [t for t in trades if t.result == "WIN"]
        losses = [t for t in trades if t.result == "LOSS"]
        open_  = [t for t in trades if t.result == "OPEN"]
        filled = wins + losses + open_

        win_rate = (
            len(wins) / (len(wins) + len(losses)) * 100 if (wins or losses) else 0.0
        )
        # Compute actual R from trade records (TP may vary per trade).
        def _trade_rr(t: TradeRecord) -> float:
            if t.tp is None:
                return 0.0
            return abs(t.tp - t.entry) / abs(t.entry - t.sl) if t.entry != t.sl else 0.0

        net_r = sum(_trade_rr(t) for t in wins) - len(losses) * 1.0
        avg_rr = _trade_rr(wins[0]) if len(wins) == 1 else (
            sum(_trade_rr(t) for t in wins) / len(wins) if wins else 0.0
        )

        lines = [
            "\n" + "=" * 60 + "\n",
            title + "\n",
            "─" * 60 + "\n",
            f"Candles checked:      {result.steps_checked}\n",
            f"Setups detected:      "
            f"{len(trades) + result.skipped_no_trade + result.skipped_risk}\n",
        ]
        if result.skipped_no_trade:
            lines.append(f"  No trade (filtered): {result.skipped_no_trade}\n")
        if result.skipped_risk:
            lines.append(f"  Risk too high:       {result.skipped_risk}\n")
        total_decided = len(wins) + len(losses)
        lines += [
            f"Orders placed:        {len(trades)}\n",
            f"  Canceled (price):   {len(canceled_price)}\n",
            f"  Canceled (timeout): {len(canceled_timeout)}\n",
            f"  Filled:             {len(filled)}\n",
            f"    Wins:             {len(wins)}\n",
            f"    Losses:           {len(losses)}\n",
            f"    Open (end):       {len(open_)}\n",
            f"Win rate:             {win_rate:.1f}%  ({len(wins)} / {total_decided})\n",
            f"Avg RR (wins):        {avg_rr:.2f}:1\n",
            f"Net R:                {net_r:+.2f}R"
            f"  ({len(wins)} wins, {len(losses)} losses × -1R)\n",
            "=" * 60 + "\n",
        ]
        return "".join(lines)
