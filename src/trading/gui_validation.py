import queue
import sys
import threading
import tkinter as tk
from datetime import UTC, datetime, timedelta
from pathlib import Path
from tkinter import ttk

from trading.agents.llm_provider import PROVIDERS, LLMConfig
from trading.core.models import Timeframe
from trading.runner import BacktestRunner, OneTimeRunner, RunConfig

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
_MODE_LABELS = {
    "prompt": "prompt_validation",
    "agent": "agent_test",
    "baseline": "baseline_metrics",
}


class ValidationGUI:
    def __init__(self, root: tk.Tk) -> None:
        self._root = root
        self._root.title("Trade Validation — SMC Entry Detector")
        self._root.geometry("1300x1000")
        self._root.minsize(900, 560)

        self._gui_queue: queue.Queue[str | None] = queue.Queue()

        # ---- shared vars ----
        self._source_var = tk.StringVar(value="csv")
        self._htf_csv_var = tk.StringVar()
        self._ltf_csv_var = tk.StringVar()
        self._until_var = tk.StringVar()
        self._htf_tf_var = tk.StringVar(value="1h")
        self._ltf_tf_var = tk.StringVar(value="15m")
        self._htf_limit_var = tk.StringVar(value="72")
        self._ltf_limit_var = tk.StringVar(value="24")
        self._symbol_var = tk.StringVar(value="BTC/USDT:USDT")
        self._offset_var = tk.StringVar(value="0.05")
        self._block_tested_var = tk.BooleanVar(value=False)
        self._strategy_var = tk.StringVar(value="htf_fvg_ltf_bos_v2")
        self._output_mode_var = tk.StringVar(value="agent")
        self._order_timeout_var = tk.StringVar(value="10")
        self._max_risk_var = tk.StringVar(value="1.0")
        self._rr_ratio_var = tk.StringVar(value="2.0")

        _default_provider = next(iter(PROVIDERS))
        self._provider_var = tk.StringVar(value=_default_provider)
        self._model_var = tk.StringVar(value=PROVIDERS[_default_provider][0])
        self._model_combos: list[ttk.Combobox] = []

        now = datetime.now(UTC)
        self._bt_from_var = tk.StringVar(
            value=(now - timedelta(days=30)).strftime("%Y-%m-%d %H:%M")
        )
        self._bt_to_var = tk.StringVar(value=now.strftime("%Y-%m-%d %H:%M"))

        # typed widget refs
        self._csv_frame: ttk.Frame
        self._until_frame: ttk.Frame
        self._htf_csv_combo: ttk.Combobox
        self._ltf_csv_combo: ttk.Combobox
        self._val_output_text: tk.Text
        self._bt_output_text: tk.Text
        self._submit_btn: ttk.Button
        self._bt_submit_btn: ttk.Button
        # model sections per tab — shown only in agent mode
        self._model_frames: list[ttk.LabelFrame] = []
        # params frames used as pack anchors for model frame insertion
        self._params_anchor_frames: list[ttk.LabelFrame] = []

        self._build_layout()
        self._populate_csv_dropdowns()
        self._refresh_until_default()

        self._ltf_tf_var.trace_add("write", lambda *_: self._refresh_until_default())
        self._source_var.trace_add("write", lambda *_: self._on_source_change())
        self._provider_var.trace_add("write", lambda *_: self._on_provider_change())
        self._output_mode_var.trace_add("write", lambda *_: self._on_mode_change())

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

        left = ttk.Frame(pane, width=360, padding=10)
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

    # --------------------------------------------------------- section builders

    def _build_output_mode_section(
        self, parent: ttk.Frame, *, one_time: bool = False
    ) -> None:
        frame = ttk.LabelFrame(parent, text="Output Mode", padding=8)
        frame.pack(fill=tk.X, pady=(0, 8))
        modes: list[tuple[str, str]] = []
        if one_time:
            modes += [
                ("Strategy Inspect", "strategy_inspect"),
                ("Prompt Validation", "prompt"),
            ]
        modes += [("Agent Test", "agent"), ("Baseline Metrics", "baseline")]
        for label, value in modes:
            ttk.Radiobutton(
                frame, text=label, variable=self._output_mode_var, value=value,
            ).pack(anchor=tk.W)

    def _build_symbol_section(self, parent: ttk.Frame) -> None:
        frame = ttk.LabelFrame(parent, text="Symbol", padding=8)
        frame.pack(fill=tk.X, pady=(0, 8))
        ttk.Entry(frame, textvariable=self._symbol_var, width=18).pack(anchor=tk.W)

    def _build_strategy_section(self, parent: ttk.Frame) -> None:
        frame = ttk.LabelFrame(parent, text="Strategy", padding=8)
        frame.pack(fill=tk.X, pady=(0, 8))
        ttk.Combobox(
            frame,
            textvariable=self._strategy_var,
            values=["htf_fvg_ltf_bos", "htf_fvg_ltf_bos_v2"],
            state="readonly",
            width=22,
        ).pack(anchor=tk.W)

    def _build_data_source_section(self, parent: ttk.Frame) -> None:
        src_frame = ttk.LabelFrame(parent, text="Data Source", padding=8)
        src_frame.pack(fill=tk.X, pady=(0, 8))
        for label, value in [
            ("CSV", "csv"), ("Past Data", "past"), ("Current Data", "live"),
        ]:
            ttk.Radiobutton(
                src_frame, text=label, variable=self._source_var, value=value,
            ).pack(anchor=tk.W)

        # Container for conditional file/date options — collapses when empty
        options_container = ttk.Frame(parent)
        options_container.pack(fill=tk.X)

        self._csv_frame = ttk.Frame(options_container)
        ttk.Label(self._csv_frame, text="HTF CSV File").pack(anchor=tk.W)
        self._htf_csv_combo = ttk.Combobox(
            self._csv_frame, textvariable=self._htf_csv_var,
            state="readonly", width=32,
        )
        self._htf_csv_combo.pack(fill=tk.X, pady=(0, 6))
        ttk.Label(self._csv_frame, text="LTF CSV File").pack(anchor=tk.W)
        self._ltf_csv_combo = ttk.Combobox(
            self._csv_frame, textvariable=self._ltf_csv_var,
            state="readonly", width=32,
        )
        self._ltf_csv_combo.pack(fill=tk.X)

        self._until_frame = ttk.Frame(options_container)
        ttk.Label(
            self._until_frame, text="To (UTC)  —  YYYY-MM-DD HH:MM"
        ).pack(anchor=tk.W)
        ttk.Entry(self._until_frame, textvariable=self._until_var, width=22).pack(
            anchor=tk.W, pady=(2, 0)
        )

        # Apply initial visibility
        self._on_source_change()

    def _build_timeframes_section(self, parent: ttk.Frame) -> None:
        frame = ttk.LabelFrame(parent, text="Timeframes", padding=8)
        frame.pack(fill=tk.X, pady=(8, 8))
        inner = ttk.Frame(frame)
        inner.pack(fill=tk.X)

        ttk.Label(inner, text="HTF").grid(row=0, column=0, sticky=tk.W, padx=(0, 6))
        ttk.Combobox(
            inner, textvariable=self._htf_tf_var, values=_TF_VALUES,
            state="readonly", width=7,
        ).grid(row=0, column=1, sticky=tk.W, pady=(0, 4))
        ttk.Spinbox(
            inner, textvariable=self._htf_limit_var, from_=10, to=500, width=5,
        ).grid(row=0, column=2, sticky=tk.W, padx=(8, 4), pady=(0, 4))
        ttk.Label(inner, text="candles").grid(
            row=0, column=3, sticky=tk.W, pady=(0, 4)
        )

        ttk.Label(inner, text="LTF").grid(row=1, column=0, sticky=tk.W, padx=(0, 6))
        ttk.Combobox(
            inner, textvariable=self._ltf_tf_var, values=_TF_VALUES,
            state="readonly", width=7,
        ).grid(row=1, column=1, sticky=tk.W)
        ttk.Spinbox(
            inner, textvariable=self._ltf_limit_var, from_=10, to=500, width=5,
        ).grid(row=1, column=2, sticky=tk.W, padx=(8, 4))
        ttk.Label(inner, text="candles").grid(row=1, column=3, sticky=tk.W)

    def _build_model_section(self, parent: ttk.Frame) -> None:
        """Create Model LabelFrame. Packed/unpacked by _on_mode_change."""
        model_frame = ttk.LabelFrame(parent, text="Model", padding=8)
        self._model_frames.append(model_frame)

        inner = ttk.Frame(model_frame)
        inner.pack(fill=tk.X)

        ttk.Label(inner, text="Provider").grid(
            row=0, column=0, sticky=tk.W, padx=(0, 6), pady=(0, 4)
        )
        ttk.Combobox(
            inner, textvariable=self._provider_var,
            values=list(PROVIDERS), state="readonly", width=12,
        ).grid(row=0, column=1, sticky=tk.W, pady=(0, 4))

        ttk.Label(inner, text="Model").grid(
            row=1, column=0, sticky=tk.W, padx=(0, 6)
        )
        model_combo = ttk.Combobox(
            inner, textvariable=self._model_var,
            values=PROVIDERS.get(self._provider_var.get(), []),
            state="readonly", width=28,
        )
        model_combo.grid(row=1, column=1, sticky=tk.W)
        self._model_combos.append(model_combo)

    def _build_strategy_params_section(self, parent: ttk.Frame) -> None:
        frame = ttk.LabelFrame(parent, text="Strategy Parameters", padding=8)
        frame.pack(fill=tk.X, pady=(0, 8))
        self._params_anchor_frames.append(frame)

        inner = ttk.Frame(frame)
        inner.pack(fill=tk.X)

        ttk.Label(inner, text="FVG offset (%)").grid(
            row=0, column=0, sticky=tk.W, padx=(0, 6)
        )
        ttk.Spinbox(
            inner, textvariable=self._offset_var,
            from_=0.0, to=100.0, increment=0.05, format="%.2f", width=7,
        ).grid(row=0, column=1, sticky=tk.W)

        ttk.Checkbutton(
            inner, text="Block tested FVGs on path",
            variable=self._block_tested_var,
        ).grid(row=1, column=0, columnspan=2, sticky=tk.W, pady=(6, 0))

        ttk.Label(inner, text="Order timeout (LTF candles)").grid(
            row=2, column=0, sticky=tk.W, padx=(0, 6), pady=(6, 0)
        )
        ttk.Spinbox(
            inner, textvariable=self._order_timeout_var, from_=1, to=500, width=5,
        ).grid(row=2, column=1, sticky=tk.W, pady=(6, 0))

        ttk.Label(inner, text="Max risk (% of entry)").grid(
            row=3, column=0, sticky=tk.W, padx=(0, 6), pady=(4, 0)
        )
        ttk.Spinbox(
            inner, textvariable=self._max_risk_var,
            from_=0.1, to=100.0, increment=0.1, format="%.1f", width=5,
        ).grid(row=3, column=1, sticky=tk.W, pady=(4, 0))

        ttk.Label(inner, text="Take profit (RR ratio)").grid(
            row=4, column=0, sticky=tk.W, padx=(0, 6), pady=(4, 0)
        )
        ttk.Spinbox(
            inner, textvariable=self._rr_ratio_var,
            from_=0.5, to=20.0, increment=0.5, format="%.1f", width=5,
        ).grid(row=4, column=1, sticky=tk.W, pady=(4, 0))

    def _build_date_range_section(self, parent: ttk.Frame) -> None:
        frame = ttk.LabelFrame(parent, text="Date Range (UTC)", padding=8)
        frame.pack(fill=tk.X, pady=(0, 8))
        ttk.Label(frame, text="From  YYYY-MM-DD HH:MM").pack(anchor=tk.W)
        ttk.Entry(frame, textvariable=self._bt_from_var, width=22).pack(
            anchor=tk.W, pady=(2, 8)
        )
        ttk.Label(frame, text="To  YYYY-MM-DD HH:MM").pack(anchor=tk.W)
        ttk.Entry(frame, textvariable=self._bt_to_var, width=22).pack(
            anchor=tk.W, pady=(2, 0)
        )

    # ----------------------------------------------------- tab builders

    def _build_validation_tab(self, parent: ttk.Frame) -> None:
        left, right = self._make_paned(parent)
        self._build_validation_controls(left)
        self._val_output_text = self._build_output_panel(right, "Entry Analysis")

    def _build_validation_controls(self, parent: ttk.Frame) -> None:
        self._build_output_mode_section(parent, one_time=True)
        self._build_symbol_section(parent)
        self._build_strategy_section(parent)
        self._build_data_source_section(parent)
        self._build_timeframes_section(parent)
        self._build_model_section(parent)
        self._build_strategy_params_section(parent)

        self._submit_btn = ttk.Button(
            parent, text="Detect Entry", command=self._on_submit
        )
        self._submit_btn.pack(fill=tk.X)

        if self._output_mode_var.get() == "agent":
            self._model_frames[-1].pack(
                fill=tk.X, pady=(0, 8), before=self._params_anchor_frames[-1]
            )

    def _build_backtest_tab(self, parent: ttk.Frame) -> None:
        left, right = self._make_paned(parent)
        self._build_backtest_controls(left)
        self._bt_output_text = self._build_output_panel(right, "Backtest Output")

    def _build_backtest_controls(self, parent: ttk.Frame) -> None:
        self._build_output_mode_section(parent, one_time=False)
        self._build_symbol_section(parent)
        self._build_strategy_section(parent)
        self._build_date_range_section(parent)
        self._build_timeframes_section(parent)
        self._build_model_section(parent)
        self._build_strategy_params_section(parent)

        self._bt_submit_btn = ttk.Button(
            parent, text="Run Backtest", command=self._on_run_backtest
        )
        self._bt_submit_btn.pack(fill=tk.X)

        if self._output_mode_var.get() == "agent":
            self._model_frames[-1].pack(
                fill=tk.X, pady=(0, 8), before=self._params_anchor_frames[-1]
            )

    # --------------------------------------------------------- dynamic state

    def _on_provider_change(self) -> None:
        provider = self._provider_var.get()
        models = PROVIDERS.get(provider, [])
        self._model_var.set(models[0] if models else "")
        for combo in self._model_combos:
            combo["values"] = models

    def _on_source_change(self) -> None:
        source = self._source_var.get()
        self._csv_frame.pack_forget()
        self._until_frame.pack_forget()
        if source == "csv":
            self._csv_frame.pack(fill=tk.X, pady=(0, 8))
        elif source == "past":
            self._until_frame.pack(fill=tk.X, pady=(0, 8))
            self._refresh_until_default()

    def _on_mode_change(self) -> None:
        mode = self._output_mode_var.get()
        for model_frame, params_frame in zip(
            self._model_frames, self._params_anchor_frames
        ):
            if mode == "agent":
                model_frame.pack(
                    fill=tk.X, pady=(0, 8), before=params_frame
                )
            else:
                model_frame.pack_forget()

    def _populate_csv_dropdowns(self) -> None:
        csv_files = (
            sorted(f.name for f in _DATA_DIR.glob("*.csv"))
            if _DATA_DIR.exists() else []
        )
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

    # ------------------------------------------------ config builders

    def _build_run_config(self) -> RunConfig | None:
        try:
            htf_limit = int(self._htf_limit_var.get())
            ltf_limit = int(self._ltf_limit_var.get())
        except ValueError as exc:
            self._gui_queue.put(f"[ERROR] Invalid candle limit: {exc}\n")
            self._gui_queue.put(None)
            return None

        try:
            offset_pct = float(self._offset_var.get()) / 100.0
        except ValueError as exc:
            self._gui_queue.put(f"[ERROR] Invalid FVG offset: {exc}\n")
            self._gui_queue.put(None)
            return None

        try:
            order_timeout = int(self._order_timeout_var.get())
            max_risk_pct = float(self._max_risk_var.get())
            rr_ratio = float(self._rr_ratio_var.get())
        except ValueError as exc:
            self._gui_queue.put(f"[ERROR] Invalid strategy parameter: {exc}\n")
            self._gui_queue.put(None)
            return None

        return RunConfig(
            symbol=self._symbol_var.get().strip(),
            htf_tf=Timeframe(self._htf_tf_var.get()),
            ltf_tf=Timeframe(self._ltf_tf_var.get()),
            htf_limit=htf_limit,
            ltf_limit=ltf_limit,
            fvg_offset_pct=offset_pct,
            output_mode=self._output_mode_var.get(),  # type: ignore[arg-type]
            strategy=self._strategy_var.get(),  # type: ignore[arg-type]
            block_tested_fvgs=self._block_tested_var.get(),
            llm_config=LLMConfig(
                provider=self._provider_var.get(),
                model=self._model_var.get(),
            ),
            order_timeout=order_timeout,
            max_risk_pct=max_risk_pct,
            rr_ratio=rr_ratio,
        )

    def _build_onetime_config(self) -> RunConfig | None:
        config = self._build_run_config()
        if config is None:
            return None

        config.data_source = self._source_var.get()  # type: ignore[assignment]
        config.htf_csv = self._htf_csv_var.get() or None
        config.ltf_csv = self._ltf_csv_var.get() or None

        if config.data_source == "past":
            try:
                config.until = datetime.strptime(
                    self._until_var.get().strip(), "%Y-%m-%d %H:%M"
                ).replace(tzinfo=UTC)
            except ValueError as exc:
                self._gui_queue.put(f"[ERROR] Invalid 'To' datetime: {exc}\n")
                self._gui_queue.put(None)
                return None

        return config

    def _build_backtest_config(self) -> RunConfig | None:
        config = self._build_run_config()
        if config is None:
            return None

        try:
            config.bt_from = datetime.strptime(
                self._bt_from_var.get().strip(), "%Y-%m-%d %H:%M"
            ).replace(tzinfo=UTC)
            config.bt_to = datetime.strptime(
                self._bt_to_var.get().strip(), "%Y-%m-%d %H:%M"
            ).replace(tzinfo=UTC)
        except ValueError as exc:
            self._gui_queue.put(f"[ERROR] Invalid date range: {exc}\n")
            self._gui_queue.put(None)
            return None

        if config.bt_from >= config.bt_to:
            self._gui_queue.put("[ERROR] 'From' must be before 'To'.\n")
            self._gui_queue.put(None)
            return None

        return config

    def _build_out_path(self, config: RunConfig) -> Path:
        strategy_name = config.strategy
        mode_label = _MODE_LABELS[config.output_mode]
        assert config.bt_from is not None and config.bt_to is not None
        from_str = config.bt_from.strftime("%Y%m%d-%H%M")
        to_str = config.bt_to.strftime("%Y%m%d-%H%M")

        model_part: tuple[str, ...] = ()
        if config.output_mode == "agent" and config.llm_config is not None:
            model_part = (
                f"{config.llm_config.provider}_{config.llm_config.model}",
            )

        symbol_folder = config.symbol.replace("/", "-").replace(":", "-")

        offset_pct = config.fvg_offset_pct * 100
        btested = "_btested" if config.block_tested_fvgs else ""
        if config.output_mode == "prompt":
            params = f"fvg{offset_pct:.4g}pct{btested}"
        elif config.output_mode == "agent":
            params = (
                f"fvg{offset_pct:.4g}pct{btested}"
                f"_to{config.order_timeout}"
                f"_risk{config.max_risk_pct:.4g}pct"
            )
        else:  # baseline
            params = (
                f"fvg{offset_pct:.4g}pct{btested}"
                f"_rr{config.rr_ratio:.4g}"
                f"_to{config.order_timeout}"
                f"_risk{config.max_risk_pct:.4g}pct"
            )

        filename = f"{from_str}_{to_str}.txt"
        return _BACKTEST_DIR.joinpath(
            mode_label, *model_part, strategy_name, symbol_folder, params, filename
        )

    # ---------------------------------------------------- validation submit

    def _on_submit(self) -> None:
        self._clear_output(self._val_output_text)
        self._submit_btn.config(state=tk.DISABLED)
        thread = threading.Thread(target=self._run_analysis, daemon=True)
        thread.start()
        self._root.after(
            100,
            lambda: self._poll_gui_queue(self._val_output_text, self._submit_btn),
        )

    def _run_analysis(self) -> None:
        config = self._build_onetime_config()
        if config is None:
            return
        try:
            OneTimeRunner(config, _DATA_DIR).run(
                gui_output=lambda s: self._gui_queue.put(s)
            )
        except Exception as exc:
            self._gui_queue.put(f"\n[ERROR] {exc}\n")
        finally:
            self._gui_queue.put(None)

    # ----------------------------------------------------- backtest submit

    def _on_run_backtest(self) -> None:
        self._clear_output(self._bt_output_text)
        self._bt_submit_btn.config(state=tk.DISABLED)
        thread = threading.Thread(target=self._run_backtest, daemon=True)
        thread.start()
        self._root.after(
            100,
            lambda: self._poll_gui_queue(self._bt_output_text, self._bt_submit_btn),
        )

    def _run_backtest(self) -> None:
        config = self._build_backtest_config()
        if config is None:
            return
        out_path = self._build_out_path(config)
        try:
            BacktestRunner(config).run(
                gui_output=lambda s: self._gui_queue.put(s),
                detail_output=lambda _: None,
                out_path=out_path,
            )
        except Exception as exc:
            self._gui_queue.put(f"\n[ERROR] {exc}\n")
        finally:
            self._gui_queue.put(None)

    # -------------------------------------------------------- output helpers

    def _clear_output(self, text: tk.Text) -> None:
        text.config(state=tk.NORMAL)
        text.delete("1.0", tk.END)
        text.config(state=tk.DISABLED)

    def _poll_gui_queue(self, text: tk.Text, btn: ttk.Button) -> None:
        try:
            while True:
                item = self._gui_queue.get_nowait()
                if item is None:
                    btn.config(state=tk.NORMAL)
                    return
                self._append_output(text, item)
        except queue.Empty:
            pass
        self._root.after(100, lambda: self._poll_gui_queue(text, btn))

    def _append_output(self, text: tk.Text, content: str) -> None:
        text.config(state=tk.NORMAL)
        text.insert(tk.END, content)
        text.see(tk.END)
        text.config(state=tk.DISABLED)


def main() -> None:
    root = tk.Tk()
    ValidationGUI(root)
    root.mainloop()
