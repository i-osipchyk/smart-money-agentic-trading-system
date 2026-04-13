import queue
import sys
import threading
import tkinter as tk
from datetime import UTC, datetime, timedelta
from pathlib import Path
from tkinter import ttk

from trading.agents.trade_validation_agent import (
    TradeValidationAgent,
    build_prompt,
    parse_decision,
)
from trading.core.models import Timeframe, TradeDecision, Trend
from trading.data.backtest_datasource import BacktestDataSource
from trading.data.binance_datasource import BinanceDataSource
from trading.data.csv_datasource import CSVDataSource
from trading.gui import QueueWriter
from trading.strategies import HtfFvgLtfBos

_TF_VALUES = ["5m", "15m", "1h", "4h", "1d"]
_TF_SECONDS: dict[str, int] = {
    "5m": 300,
    "15m": 900,
    "1h": 3600,
    "4h": 14400,
    "1d": 86400,
}
_DATA_DIR = Path("data")
_BACKTEST_DIR = Path("backtests")


class ValidationGUI:
    def __init__(self, root: tk.Tk) -> None:
        self._root = root
        self._root.title("Trade Validation — SMC Entry Detector")
        self._root.geometry("1100x680")
        self._root.minsize(800, 500)

        self._output_queue: queue.Queue[str | None] = queue.Queue()

        # Shared vars — kept in sync across both tabs
        self._source_var = tk.StringVar(value="csv")
        self._mode_var = tk.StringVar(value="prompt")
        self._htf_csv_var = tk.StringVar()
        self._ltf_csv_var = tk.StringVar()
        self._until_var = tk.StringVar()
        self._htf_tf_var = tk.StringVar(value="1h")
        self._ltf_tf_var = tk.StringVar(value="15m")
        self._htf_limit_var = tk.StringVar(value="72")
        self._ltf_limit_var = tk.StringVar(value="16")
        self._symbol_var = tk.StringVar(value="BTC/USDT:USDT")
        # Offset in tenths of a percent (1 = 0.1 %, 10 = 1.0 %)
        self._offset_var = tk.StringVar(value="10")

        # Backtest-specific vars
        now = datetime.now(UTC)
        self._bt_from_var = tk.StringVar(
            value=(now - timedelta(days=30)).strftime("%Y-%m-%d %H:%M")
        )
        self._bt_to_var = tk.StringVar(value=now.strftime("%Y-%m-%d %H:%M"))
        self._bt_mode_var = tk.StringVar(value="prompt")
        self._bt_timeout_var = tk.StringVar(value="10")

        # Typed widget refs — set during layout
        self._csv_frame: ttk.Frame
        self._until_frame: ttk.Frame
        self._htf_csv_combo: ttk.Combobox
        self._ltf_csv_combo: ttk.Combobox
        self._val_output_text: tk.Text
        self._bt_output_text: tk.Text
        self._submit_btn: ttk.Button
        self._bt_submit_btn: ttk.Button

        self._build_layout()
        self._populate_csv_dropdowns()
        self._refresh_until_default()

        self._ltf_tf_var.trace_add("write", lambda *_: self._refresh_until_default())
        self._source_var.trace_add("write", lambda *_: self._on_source_change())

    # ------------------------------------------------------------------ layout

    def _build_layout(self) -> None:
        notebook = ttk.Notebook(self._root)
        notebook.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)

        val_tab = ttk.Frame(notebook)
        notebook.add(val_tab, text="One-Time Validation")
        self._build_validation_tab(val_tab)

        bt_tab = ttk.Frame(notebook)
        notebook.add(bt_tab, text="Backtest")
        self._build_backtest_tab(bt_tab)

    def _make_paned(self, parent: ttk.Frame) -> tuple[ttk.Frame, ttk.Frame]:
        pane = tk.PanedWindow(
            parent, orient=tk.HORIZONTAL, sashwidth=5, sashrelief=tk.RAISED
        )
        pane.pack(fill=tk.BOTH, expand=True)

        left = ttk.Frame(pane, width=310, padding=10)
        left.pack_propagate(False)
        pane.add(left, minsize=260)

        right = ttk.Frame(pane, padding=(8, 10, 10, 10))
        pane.add(right, minsize=300)

        return left, right

    def _build_output_panel(self, parent: ttk.Frame, title: str) -> tk.Text:
        ttk.Label(parent, text=title, font=("TkDefaultFont", 10, "bold")).pack(
            anchor=tk.W, pady=(0, 4)
        )
        frame = ttk.Frame(parent)
        frame.pack(fill=tk.BOTH, expand=True)

        scrollbar = ttk.Scrollbar(frame, orient=tk.VERTICAL)
        text = tk.Text(
            frame,
            state=tk.DISABLED,
            wrap=tk.WORD,
            font=("Menlo", 11) if sys.platform == "darwin" else ("Consolas", 10),
            yscrollcommand=scrollbar.set,
        )
        scrollbar.config(command=text.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        return text

    def _build_shared_controls(self, parent: ttk.Frame) -> None:
        """HTF/LTF timeframes + candle counts, symbol, and FVG offset."""
        tf_frame = ttk.LabelFrame(parent, text="Timeframes", padding=8)
        tf_frame.pack(fill=tk.X, pady=(0, 8))

        tf_inner = ttk.Frame(tf_frame)
        tf_inner.pack(fill=tk.X)

        ttk.Label(tf_inner, text="HTF").grid(row=0, column=0, sticky=tk.W, padx=(0, 6))
        ttk.Combobox(
            tf_inner, textvariable=self._htf_tf_var, values=_TF_VALUES, state="readonly", width=7
        ).grid(row=0, column=1, sticky=tk.W, pady=(0, 4))
        ttk.Spinbox(
            tf_inner, textvariable=self._htf_limit_var, from_=10, to=500, width=5
        ).grid(row=0, column=2, sticky=tk.W, padx=(8, 4), pady=(0, 4))
        ttk.Label(tf_inner, text="candles").grid(row=0, column=3, sticky=tk.W, pady=(0, 4))

        ttk.Label(tf_inner, text="LTF").grid(row=1, column=0, sticky=tk.W, padx=(0, 6))
        ttk.Combobox(
            tf_inner, textvariable=self._ltf_tf_var, values=_TF_VALUES, state="readonly", width=7
        ).grid(row=1, column=1, sticky=tk.W)
        ttk.Spinbox(
            tf_inner, textvariable=self._ltf_limit_var, from_=10, to=500, width=5
        ).grid(row=1, column=2, sticky=tk.W, padx=(8, 4))
        ttk.Label(tf_inner, text="candles").grid(row=1, column=3, sticky=tk.W)

        sym_frame = ttk.LabelFrame(parent, text="Symbol", padding=8)
        sym_frame.pack(fill=tk.X, pady=(0, 8))
        ttk.Entry(sym_frame, textvariable=self._symbol_var, width=18).pack(anchor=tk.W)

        opt_frame = ttk.LabelFrame(parent, text="Options", padding=8)
        opt_frame.pack(fill=tk.X, pady=(0, 8))

        opt_inner = ttk.Frame(opt_frame)
        opt_inner.pack(fill=tk.X)
        ttk.Label(opt_inner, text="FVG offset (×0.1%)").grid(
            row=0, column=0, sticky=tk.W, padx=(0, 6)
        )
        ttk.Spinbox(
            opt_inner, textvariable=self._offset_var, from_=0, to=1000, width=5
        ).grid(row=0, column=1, sticky=tk.W)

    # --------------------------------------------------------- validation tab

    def _build_validation_tab(self, parent: ttk.Frame) -> None:
        left, right = self._make_paned(parent)
        self._build_validation_controls(left)
        self._val_output_text = self._build_output_panel(right, "Entry Analysis")

    def _build_validation_controls(self, parent: ttk.Frame) -> None:
        # --- Data Source ---
        src_frame = ttk.LabelFrame(parent, text="Data Source", padding=8)
        src_frame.pack(fill=tk.X, pady=(0, 8))

        for label, value in [("CSV", "csv"), ("Past Data", "past"), ("Current Data", "live")]:
            ttk.Radiobutton(
                src_frame, text=label, variable=self._source_var, value=value
            ).pack(anchor=tk.W)

        # --- CSV file pickers (shown for "csv") ---
        self._csv_frame = ttk.Frame(parent)
        self._csv_frame.pack(fill=tk.X, pady=(0, 8))

        ttk.Label(self._csv_frame, text="HTF CSV File").pack(anchor=tk.W)
        self._htf_csv_combo = ttk.Combobox(
            self._csv_frame, textvariable=self._htf_csv_var, state="readonly", width=32
        )
        self._htf_csv_combo.pack(fill=tk.X, pady=(0, 6))

        ttk.Label(self._csv_frame, text="LTF CSV File").pack(anchor=tk.W)
        self._ltf_csv_combo = ttk.Combobox(
            self._csv_frame, textvariable=self._ltf_csv_var, state="readonly", width=32
        )
        self._ltf_csv_combo.pack(fill=tk.X)

        # --- To datetime (shown for "past") ---
        self._until_frame = ttk.Frame(parent)

        ttk.Label(self._until_frame, text="To (UTC)  —  YYYY-MM-DD HH:MM").pack(anchor=tk.W)
        ttk.Entry(self._until_frame, textvariable=self._until_var, width=22).pack(
            anchor=tk.W, pady=(2, 0)
        )

        self._build_shared_controls(parent)

        # --- Mode ---
        mode_frame = ttk.LabelFrame(parent, text="Mode", padding=8)
        mode_frame.pack(fill=tk.X, pady=(0, 12))

        for label, value in [("Prompt Validation", "prompt"), ("Agent Test", "agent")]:
            ttk.Radiobutton(
                mode_frame, text=label, variable=self._mode_var, value=value
            ).pack(anchor=tk.W)

        # --- Submit ---
        self._submit_btn = ttk.Button(parent, text="Detect Entry", command=self._on_submit)
        self._submit_btn.pack(fill=tk.X)

    # ----------------------------------------------------------- backtest tab

    def _build_backtest_tab(self, parent: ttk.Frame) -> None:
        left, right = self._make_paned(parent)
        self._build_backtest_controls(left)
        self._bt_output_text = self._build_output_panel(right, "Backtest Output")

    def _build_backtest_controls(self, parent: ttk.Frame) -> None:
        self._build_shared_controls(parent)

        # --- Date Range ---
        range_frame = ttk.LabelFrame(parent, text="Date Range (UTC)", padding=8)
        range_frame.pack(fill=tk.X, pady=(0, 8))

        ttk.Label(range_frame, text="From  YYYY-MM-DD HH:MM").pack(anchor=tk.W)
        ttk.Entry(range_frame, textvariable=self._bt_from_var, width=22).pack(
            anchor=tk.W, pady=(2, 8)
        )
        ttk.Label(range_frame, text="To  YYYY-MM-DD HH:MM").pack(anchor=tk.W)
        ttk.Entry(range_frame, textvariable=self._bt_to_var, width=22).pack(
            anchor=tk.W, pady=(2, 0)
        )

        # --- Mode ---
        mode_frame = ttk.LabelFrame(parent, text="Mode", padding=8)
        mode_frame.pack(fill=tk.X, pady=(0, 8))

        for label, value in [
            ("Prompt Validation", "prompt"),
            ("Agent Test", "agent"),
            ("Baseline Metrics", "baseline"),
        ]:
            ttk.Radiobutton(
                mode_frame, text=label, variable=self._bt_mode_var, value=value
            ).pack(anchor=tk.W)

        # --- Baseline options ---
        bl_frame = ttk.LabelFrame(parent, text="Baseline Options", padding=8)
        bl_frame.pack(fill=tk.X, pady=(0, 12))

        bl_inner = ttk.Frame(bl_frame)
        bl_inner.pack(fill=tk.X)
        ttk.Label(bl_inner, text="Order timeout (LTF candles)").grid(
            row=0, column=0, sticky=tk.W, padx=(0, 6)
        )
        ttk.Spinbox(
            bl_inner, textvariable=self._bt_timeout_var, from_=1, to=500, width=5
        ).grid(row=0, column=1, sticky=tk.W)

        # --- Submit ---
        self._bt_submit_btn = ttk.Button(
            parent, text="Run Backtest", command=self._on_run_backtest
        )
        self._bt_submit_btn.pack(fill=tk.X)

    # ----------------------------------------------------------- dynamic state

    def _on_source_change(self) -> None:
        source = self._source_var.get()
        if source == "csv":
            self._until_frame.pack_forget()
            self._csv_frame.pack(fill=tk.X, pady=(0, 8), before=self._until_frame)
        elif source == "past":
            self._csv_frame.pack_forget()
            self._until_frame.pack(fill=tk.X, pady=(0, 8))
            self._refresh_until_default()
        else:
            self._csv_frame.pack_forget()
            self._until_frame.pack_forget()

    def _populate_csv_dropdowns(self) -> None:
        csv_files = sorted(f.name for f in _DATA_DIR.glob("*.csv")) if _DATA_DIR.exists() else []
        self._htf_csv_combo["values"] = csv_files
        self._ltf_csv_combo["values"] = csv_files
        if csv_files:
            self._htf_csv_var.set(csv_files[0])
            self._ltf_csv_var.set(csv_files[-1] if len(csv_files) > 1 else csv_files[0])

    def _refresh_until_default(self) -> None:
        tf = self._ltf_tf_var.get()
        period = _TF_SECONDS.get(tf, 900)
        now_ts = int(datetime.now(UTC).timestamp())
        last_closed_ts = (now_ts // period) * period
        dt = datetime.fromtimestamp(last_closed_ts, tz=UTC)
        self._until_var.set(dt.strftime("%Y-%m-%d %H:%M"))

    # ---------------------------------------------------- validation submit

    def _on_submit(self) -> None:
        self._clear_output(self._val_output_text)
        self._submit_btn.config(state=tk.DISABLED)

        thread = threading.Thread(target=self._run_analysis, daemon=True)
        thread.start()
        self._root.after(
            100,
            lambda: self._poll_output_queue(self._val_output_text, self._submit_btn),
        )

    def _run_analysis(self) -> None:
        source_raw = self._source_var.get()
        symbol = self._symbol_var.get().strip()
        htf_tf = Timeframe(self._htf_tf_var.get())
        ltf_tf = Timeframe(self._ltf_tf_var.get())

        try:
            htf_limit = int(self._htf_limit_var.get())
            ltf_limit = int(self._ltf_limit_var.get())
        except ValueError as exc:
            self._output_queue.put(f"[ERROR] Invalid candle limit: {exc}\n")
            self._output_queue.put(None)
            return

        try:
            offset_pct = float(self._offset_var.get()) / 1000.0
        except ValueError as exc:
            self._output_queue.put(f"[ERROR] Invalid FVG offset: {exc}\n")
            self._output_queue.put(None)
            return

        until: datetime | None = None
        if source_raw == "past":
            try:
                until = datetime.strptime(
                    self._until_var.get().strip(), "%Y-%m-%d %H:%M"
                ).replace(tzinfo=UTC)
            except ValueError as exc:
                self._output_queue.put(f"[ERROR] Invalid 'To' datetime: {exc}\n")
                self._output_queue.put(None)
                return

        old_stdout, old_stderr = sys.stdout, sys.stderr
        writer: QueueWriter = QueueWriter(self._output_queue)
        sys.stdout = writer  # type: ignore[assignment]
        sys.stderr = writer  # type: ignore[assignment]

        try:
            if source_raw == "csv":
                src = CSVDataSource(data_dir=_DATA_DIR)
                htf_df = src.get_ohlcv(
                    symbol=symbol,
                    timeframe=htf_tf.value,
                    limit=htf_limit,
                    filename_override=self._htf_csv_var.get() or None,
                )
                ltf_df = src.get_ohlcv(
                    symbol=symbol,
                    timeframe=ltf_tf.value,
                    limit=ltf_limit,
                    filename_override=self._ltf_csv_var.get() or None,
                )
            else:
                binance = BinanceDataSource()
                htf_since: datetime | None = None
                ltf_since: datetime | None = None
                if source_raw == "past" and until is not None:
                    htf_since = until - timedelta(
                        seconds=htf_limit * _TF_SECONDS[htf_tf.value]
                    )
                    ltf_since = until - timedelta(
                        seconds=ltf_limit * _TF_SECONDS[ltf_tf.value]
                    )
                htf_df = binance.get_ohlcv(
                    symbol=symbol, timeframe=htf_tf.value, limit=htf_limit, since=htf_since
                )
                ltf_df = binance.get_ohlcv(
                    symbol=symbol, timeframe=ltf_tf.value, limit=ltf_limit, since=ltf_since
                )

            strategy = HtfFvgLtfBos(fvg_offset_pct=offset_pct)
            setup = strategy.detect_entry(symbol, htf_df, htf_tf, ltf_df, ltf_tf)
            if setup is None:
                offset_display = offset_pct * 100
                print(f"Strategy:   {strategy.name}")
                print(f"Symbol:     {symbol}")
                print(f"HTF:        {htf_tf.value}  ({htf_limit} candles)")
                print(f"LTF:        {ltf_tf.value}  ({ltf_limit} candles)")
                print(
                    f"HTF range:  {htf_df['timestamp'].iloc[0].strftime('%Y-%m-%d %H:%M')} → "
                    f"{htf_df['timestamp'].iloc[-1].strftime('%Y-%m-%d %H:%M')}"
                )
                print(
                    f"LTF range:  {ltf_df['timestamp'].iloc[0].strftime('%Y-%m-%d %H:%M')} → "
                    f"{ltf_df['timestamp'].iloc[-1].strftime('%Y-%m-%d %H:%M')}"
                )
                print(f"FVG offset: {offset_display:.1f} %")
                print("\n" + "─" * 44)
                print("\nNo entry detected.")
            elif self._mode_var.get() == "prompt":
                prompt = build_prompt(setup)
                print("─" * 44)
                print("PROMPT VALIDATION — agent not called\n")
                print(prompt)
            else:
                prompt = build_prompt(setup)
                print("Sending to agent...\n")
                agent = TradeValidationAgent()
                print(agent.run(prompt))

        except Exception as exc:
            print(f"\n[ERROR] {exc}")
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr
            self._output_queue.put(None)

    # ----------------------------------------------------- backtest submit

    def _on_run_backtest(self) -> None:
        self._clear_output(self._bt_output_text)
        self._bt_submit_btn.config(state=tk.DISABLED)

        thread = threading.Thread(target=self._run_backtest, daemon=True)
        thread.start()
        self._root.after(
            100,
            lambda: self._poll_output_queue(self._bt_output_text, self._bt_submit_btn),
        )

    def _run_backtest(self) -> None:
        symbol = self._symbol_var.get().strip()
        htf_tf = Timeframe(self._htf_tf_var.get())
        ltf_tf = Timeframe(self._ltf_tf_var.get())
        mode = self._bt_mode_var.get()

        try:
            htf_limit = int(self._htf_limit_var.get())
            ltf_limit = int(self._ltf_limit_var.get())
        except ValueError as exc:
            self._output_queue.put(f"[ERROR] Invalid candle limit: {exc}\n")
            self._output_queue.put(None)
            return

        try:
            offset_pct = float(self._offset_var.get()) / 1000.0
        except ValueError as exc:
            self._output_queue.put(f"[ERROR] Invalid FVG offset: {exc}\n")
            self._output_queue.put(None)
            return

        try:
            bt_from = datetime.strptime(
                self._bt_from_var.get().strip(), "%Y-%m-%d %H:%M"
            ).replace(tzinfo=UTC)
            bt_to = datetime.strptime(
                self._bt_to_var.get().strip(), "%Y-%m-%d %H:%M"
            ).replace(tzinfo=UTC)
        except ValueError as exc:
            self._output_queue.put(f"[ERROR] Invalid date range: {exc}\n")
            self._output_queue.put(None)
            return

        if bt_from >= bt_to:
            self._output_queue.put("[ERROR] 'From' must be before 'To'.\n")
            self._output_queue.put(None)
            return

        order_timeout = 10
        if mode == "baseline":
            try:
                order_timeout = int(self._bt_timeout_var.get())
            except ValueError as exc:
                self._output_queue.put(f"[ERROR] Invalid order timeout: {exc}\n")
                self._output_queue.put(None)
                return

        strategy = HtfFvgLtfBos(fvg_offset_pct=offset_pct)
        bt_source = BacktestDataSource(
            symbol=symbol,
            htf_timeframe=htf_tf.value,
            htf_limit=htf_limit,
            ltf_timeframe=ltf_tf.value,
            ltf_limit=ltf_limit,
            bt_from=bt_from,
            bt_to=bt_to,
        )

        _MODE_LABELS = {
            "prompt": "prompt_validation",
            "agent": "agent_test",
            "baseline": "baseline_metrics",
        }
        mode_label = _MODE_LABELS[mode]
        from_str = bt_from.strftime("%Y%m%d-%H%M")
        to_str = bt_to.strftime("%Y%m%d-%H%M")
        out_dir = (
            _BACKTEST_DIR / mode_label / strategy.name
            if mode != "baseline"
            else _BACKTEST_DIR / "baseline_metrics" / strategy.name
        )
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{from_str}_{to_str}.txt"

        output_lines: list[str] = []

        # Tee: write to the queue (→ UI) and collect for file output
        gui_self = self

        class _Tee:
            def write(self_t, s: str) -> int:  # noqa: N805
                gui_self._output_queue.put(s)
                output_lines.append(s)
                return len(s)

            def flush(self_t) -> None:  # noqa: N805
                pass

        old_stdout, old_stderr = sys.stdout, sys.stderr
        tee = _Tee()
        sys.stdout = tee  # type: ignore[assignment]
        sys.stderr = tee  # type: ignore[assignment]

        try:
            print(f"Backtest:  {strategy.name}")
            print(f"Symbol:    {symbol}")
            print(f"HTF:       {htf_tf.value}  ({htf_limit} candles)")
            print(f"LTF:       {ltf_tf.value}  ({ltf_limit} candles)")
            print(
                f"Range:     {bt_from.strftime('%Y-%m-%d %H:%M')} → "
                f"{bt_to.strftime('%Y-%m-%d %H:%M')} UTC"
            )
            print(f"Mode:      {mode_label}")
            if mode == "baseline":
                print(f"Timeout:   {order_timeout} LTF candles")
            print(f"Steps:     {bt_source.total_steps}")
            print("=" * 60 + "\n")

            bt_source.prepare(progress=print)
            print()

            if mode == "baseline":
                self._run_baseline(
                    bt_source, strategy, symbol, htf_tf, ltf_tf,
                    order_timeout, out_path, output_lines,
                )
            elif mode == "agent":
                self._run_agent_backtest(
                    bt_source, strategy, symbol, htf_tf, ltf_tf,
                    order_timeout, out_path, output_lines,
                )
            else:  # prompt
                setups_found = 0
                step_num = 0
                for current_dt, htf_df, ltf_df in bt_source:
                    step_num += 1
                    try:
                        setup = strategy.detect_entry(symbol, htf_df, htf_tf, ltf_df, ltf_tf)
                    except Exception as exc:
                        print(f"[{current_dt.strftime('%Y-%m-%d %H:%M')}] ERROR: {exc}\n")
                        continue
                    if setup is None:
                        continue

                    setups_found += 1
                    print("─" * 60)
                    print(f"Setup #{setups_found}  at  {current_dt.strftime('%Y-%m-%d %H:%M')} UTC")
                    print("─" * 60)
                    print(build_prompt(setup))
                    print()

                print("=" * 60)
                print(f"Done. Setups found: {setups_found} / {step_num} candles checked.")
                with out_path.open("w", encoding="utf-8") as f:
                    f.writelines(output_lines)
                print(f"\nSaved → {out_path}")

        except Exception as exc:
            print(f"\n[ERROR] {exc}")
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr
            self._output_queue.put(None)

    # ----------------------------------------- shared simulation engine

    def _simulate_trades(
        self,
        bt_source: BacktestDataSource,
        strategy: HtfFvgLtfBos,
        symbol: str,
        htf_tf: Timeframe,
        ltf_tf: Timeframe,
        order_timeout: int,
        out_path: Path,
        output_lines: list[str],
        get_decision: object,   # Callable[[StrategySetup], TradeDecision | None]
        metrics_title: str,
    ) -> None:
        """
        Shared limit-order simulation engine used by both baseline and agent test.

        ``get_decision`` is called once per detected setup and must return:
        - A ``TradeDecision`` with ``should_trade=True`` to place an order, or
        - ``None`` (or ``should_trade=False``) to skip this setup.

        The decision's ``entry_price``, ``stop_loss``, and ``take_profit`` are
        used for the simulation; ``reasoning`` and ``confidence`` are stored in
        the trade record and printed in the results.

        Order lifecycle:
        - Limit order placed at ``entry_price`` (BOS level).
        - Canceled if price reaches TP before the order fills (price ran away).
        - Canceled if unfilled after ``order_timeout`` LTF candles.
        - WIN / LOSS when TP / SL is hit after fill.
        - OPEN when the backtest period ends before resolution.
        """
        _FMT = "{:,.2f}"

        def _ts(dt: datetime) -> str:
            return dt.strftime("%Y-%m-%d %H:%M")

        active_order: dict | None = None
        trades: list[dict] = []
        skipped_no_trade = 0
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

                    # Cancel: price reached TP before the order was filled
                    if (bullish and candle["high"] >= o["tp"]) or \
                       (not bullish and candle["low"] <= o["tp"]):
                        o["result"] = "CANCELED_PRICE"
                        o["close_dt"] = current_dt
                        trades.append(o)
                        active_order = None
                        continue

                    # Fill check
                    filled_this_candle = (
                        (bullish and candle["low"] <= o["entry"]) or
                        (not bullish and candle["high"] >= o["entry"])
                    )
                    if filled_this_candle:
                        o["filled"] = True
                        o["fill_dt"] = current_dt
                        if bullish:
                            if candle["low"] <= o["sl"]:
                                o["result"] = "LOSS"
                                o["close_dt"] = current_dt
                                trades.append(o)
                                active_order = None
                                continue
                            if candle["high"] >= o["tp"]:
                                o["result"] = "WIN"
                                o["close_dt"] = current_dt
                                trades.append(o)
                                active_order = None
                                continue
                        else:
                            if candle["high"] >= o["sl"]:
                                o["result"] = "LOSS"
                                o["close_dt"] = current_dt
                                trades.append(o)
                                active_order = None
                                continue
                            if candle["low"] <= o["tp"]:
                                o["result"] = "WIN"
                                o["close_dt"] = current_dt
                                trades.append(o)
                                active_order = None
                                continue
                        continue  # filled, pending resolution

                    if o["candles_waiting"] >= order_timeout:
                        o["result"] = "CANCELED_TIMEOUT"
                        o["close_dt"] = current_dt
                        trades.append(o)
                        active_order = None
                    continue

                # Filled — track TP / SL each candle
                if bullish:
                    if candle["high"] >= o["tp"]:
                        o["result"] = "WIN"
                        o["close_dt"] = current_dt
                        trades.append(o)
                        active_order = None
                    elif candle["low"] <= o["sl"]:
                        o["result"] = "LOSS"
                        o["close_dt"] = current_dt
                        trades.append(o)
                        active_order = None
                else:
                    if candle["low"] <= o["tp"]:
                        o["result"] = "WIN"
                        o["close_dt"] = current_dt
                        trades.append(o)
                        active_order = None
                    elif candle["high"] >= o["sl"]:
                        o["result"] = "LOSS"
                        o["close_dt"] = current_dt
                        trades.append(o)
                        active_order = None
                continue

            # ---- detect setup -----------------------------------------------
            try:
                setup = strategy.detect_entry(symbol, htf_df, htf_tf, ltf_df, ltf_tf)
            except Exception as exc:
                print(f"[{_ts(current_dt)}] ERROR: {exc}\n")
                continue
            if setup is None:
                continue

            # ---- decision callback ------------------------------------------
            td: TradeDecision | None = get_decision(setup)  # type: ignore[operator]
            if td is None or not td.should_trade:
                skipped_no_trade += 1
                print(f"[{_ts(current_dt)}] NO TRADE  —  {td.reasoning if td else '—'}")
                continue

            trade_num = len(trades) + 1
            risk = abs(td.entry_price - td.stop_loss)  # type: ignore[arg-type]

            print("─" * 60)
            print(f"Trade #{trade_num}  at  {_ts(current_dt)} UTC")
            print(f"Direction:   {td.direction.value.upper()}")  # type: ignore[union-attr]
            print(f"Entry:       {_FMT.format(td.entry_price)}  (limit at BOS level)")
            print(f"Stop Loss:   {_FMT.format(td.stop_loss)}  (risk {_FMT.format(risk)})")
            print(f"Take Profit: {_FMT.format(td.take_profit)}  "
                  f"(reward {_FMT.format(2 * risk)}, 2:1 RR)")
            print(f"Confidence:  {td.confidence}")
            print(f"Reasoning:   {td.reasoning}")

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
                "result": None,
                "close_dt": None,
            }

        if active_order is not None:
            active_order["result"] = "OPEN"
            active_order["close_dt"] = None
            trades.append(active_order)

        # ---- trade results summary ------------------------------------------
        print()
        for t in trades:
            result = t["result"]
            print(f"  Trade #{t['trade_num']}  {t['direction'].value.upper()}"
                  f"  →  {result}  (setup {_ts(t['setup_dt'])} UTC)")
            if t["fill_dt"]:
                print(f"    Filled:  {_ts(t['fill_dt'])} UTC")
            if t["close_dt"] and result in ("WIN", "LOSS"):
                print(f"    Closed:  {_ts(t['close_dt'])} UTC")
            if t["reasoning"] != "baseline":
                print(f"    Reasoning: {t['reasoning']}")

        # ---- metrics --------------------------------------------------------
        canceled_price   = [t for t in trades if t["result"] == "CANCELED_PRICE"]
        canceled_timeout = [t for t in trades if t["result"] == "CANCELED_TIMEOUT"]
        wins   = [t for t in trades if t["result"] == "WIN"]
        losses = [t for t in trades if t["result"] == "LOSS"]
        open_  = [t for t in trades if t["result"] == "OPEN"]
        filled = wins + losses + open_

        win_rate = (len(wins) / (len(wins) + len(losses)) * 100) if (wins or losses) else 0.0
        net_r = len(wins) * 2.0 - len(losses) * 1.0

        print()
        print("=" * 60)
        print(metrics_title)
        print("─" * 60)
        print(f"Candles checked:      {step_num}")
        print(f"Setups detected:      {len(trades) + skipped_no_trade}")
        if skipped_no_trade:
            print(f"  No trade (filtered): {skipped_no_trade}")
        print(f"Orders placed:        {len(trades)}")
        print(f"  Canceled (price):   {len(canceled_price)}")
        print(f"  Canceled (timeout): {len(canceled_timeout)}")
        print(f"  Filled:             {len(filled)}")
        print(f"    Wins:             {len(wins)}")
        print(f"    Losses:           {len(losses)}")
        print(f"    Open (end):       {len(open_)}")
        print(f"Win rate:             {win_rate:.1f}%  ({len(wins)} / {len(wins) + len(losses)})")
        print(f"Net R:                {net_r:+.1f}R  "
              f"({len(wins)} × +2R,  {len(losses)} × -1R)")
        print("=" * 60)

        with out_path.open("w", encoding="utf-8") as f:
            f.writelines(output_lines)
        print(f"\nSaved → {out_path}")

    # ----------------------------------------- mode wrappers

    def _run_baseline(
        self,
        bt_source: BacktestDataSource,
        strategy: HtfFvgLtfBos,
        symbol: str,
        htf_tf: Timeframe,
        ltf_tf: Timeframe,
        order_timeout: int,
        out_path: Path,
        output_lines: list[str],
    ) -> None:
        def get_decision(setup: object) -> TradeDecision:
            s = setup  # type: ignore[assignment]
            return TradeDecision(
                symbol=symbol,
                should_trade=True,
                direction=s.direction,
                entry_price=s.entry,
                stop_loss=s.stop_loss,
                take_profit=s.take_profit,
                reasoning="baseline",
                confidence="n/a",
            )

        self._simulate_trades(
            bt_source, strategy, symbol, htf_tf, ltf_tf,
            order_timeout, out_path, output_lines,
            get_decision=get_decision,
            metrics_title="BASELINE METRICS",
        )

    def _run_agent_backtest(
        self,
        bt_source: BacktestDataSource,
        strategy: HtfFvgLtfBos,
        symbol: str,
        htf_tf: Timeframe,
        ltf_tf: Timeframe,
        order_timeout: int,
        out_path: Path,
        output_lines: list[str],
    ) -> None:
        agent = TradeValidationAgent()

        def get_decision(setup: object) -> TradeDecision:
            s = setup  # type: ignore[assignment]
            print("Querying agent …")
            response = agent.run(build_prompt(s))
            td = parse_decision(symbol, response, s)
            return td

        self._simulate_trades(
            bt_source, strategy, symbol, htf_tf, ltf_tf,
            order_timeout, out_path, output_lines,
            get_decision=get_decision,
            metrics_title="AGENT TEST METRICS",
        )

    # -------------------------------------------------------- output helpers

    def _clear_output(self, text: tk.Text) -> None:
        text.config(state=tk.NORMAL)
        text.delete("1.0", tk.END)
        text.config(state=tk.DISABLED)

    def _poll_output_queue(self, text: tk.Text, btn: ttk.Button) -> None:
        try:
            while True:
                item = self._output_queue.get_nowait()
                if item is None:
                    btn.config(state=tk.NORMAL)
                    return
                self._append_output(text, item)
        except queue.Empty:
            pass
        self._root.after(100, lambda: self._poll_output_queue(text, btn))

    def _append_output(self, text: tk.Text, content: str) -> None:
        text.config(state=tk.NORMAL)
        text.insert(tk.END, content)
        text.see(tk.END)
        text.config(state=tk.DISABLED)


def main() -> None:
    root = tk.Tk()
    ValidationGUI(root)
    root.mainloop()
