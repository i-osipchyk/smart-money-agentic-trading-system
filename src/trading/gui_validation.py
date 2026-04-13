import queue
import sys
import threading
import tkinter as tk
from datetime import UTC, datetime, timedelta
from pathlib import Path
from tkinter import ttk

from trading.agents.trade_validation_agent import TradeValidationAgent, build_prompt
from trading.core.models import Timeframe
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


class ValidationGUI:
    def __init__(self, root: tk.Tk) -> None:
        self._root = root
        self._root.title("Trade Validation — SMC Entry Detector")
        self._root.geometry("1000x640")
        self._root.minsize(700, 500)

        self._output_queue: queue.Queue[str | None] = queue.Queue()

        self._source_var = tk.StringVar(value="csv")
        self._htf_csv_var = tk.StringVar()
        self._ltf_csv_var = tk.StringVar()
        self._until_var = tk.StringVar()
        self._htf_tf_var = tk.StringVar(value="1h")
        self._ltf_tf_var = tk.StringVar(value="15m")
        self._htf_limit_var = tk.StringVar(value="72")
        self._ltf_limit_var = tk.StringVar(value="96")
        self._symbol_var = tk.StringVar(value="BTC/USDT:USDT")
        # Offset in tenths of a percent (1 = 0.1 %, 10 = 1.0 %)
        self._offset_var = tk.StringVar(value="10")

        self._csv_frame: ttk.Frame
        self._until_frame: ttk.Frame
        self._htf_csv_combo: ttk.Combobox
        self._ltf_csv_combo: ttk.Combobox
        self._output_text: tk.Text
        self._submit_btn: ttk.Button

        self._build_layout()
        self._populate_csv_dropdowns()
        self._refresh_until_default()

        self._ltf_tf_var.trace_add("write", lambda *_: self._refresh_until_default())
        self._source_var.trace_add("write", lambda *_: self._on_source_change())

    # ------------------------------------------------------------------ layout

    def _build_layout(self) -> None:
        pane = tk.PanedWindow(
            self._root, orient=tk.HORIZONTAL, sashwidth=5, sashrelief=tk.RAISED
        )
        pane.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)

        left = ttk.Frame(pane, width=310, padding=10)
        left.pack_propagate(False)
        pane.add(left, minsize=260)

        right = ttk.Frame(pane, padding=(8, 10, 10, 10))
        pane.add(right, minsize=300)

        self._build_left_panel(left)
        self._build_right_panel(right)

    def _build_left_panel(self, parent: ttk.Frame) -> None:
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

        # --- Timeframes + Limits ---
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

        # --- Symbol ---
        sym_frame = ttk.LabelFrame(parent, text="Symbol", padding=8)
        sym_frame.pack(fill=tk.X, pady=(0, 8))
        ttk.Entry(sym_frame, textvariable=self._symbol_var, width=18).pack(anchor=tk.W)

        # --- Options ---
        opt_frame = ttk.LabelFrame(parent, text="Options", padding=8)
        opt_frame.pack(fill=tk.X, pady=(0, 12))

        opt_inner = ttk.Frame(opt_frame)
        opt_inner.pack(fill=tk.X)
        ttk.Label(opt_inner, text="FVG offset (×0.1%)").grid(
            row=0, column=0, sticky=tk.W, padx=(0, 6)
        )
        ttk.Spinbox(
            opt_inner, textvariable=self._offset_var, from_=0, to=1000, width=5
        ).grid(row=0, column=1, sticky=tk.W)

        # --- Submit ---
        self._submit_btn = ttk.Button(parent, text="Detect Entry", command=self._on_submit)
        self._submit_btn.pack(fill=tk.X)

    def _build_right_panel(self, parent: ttk.Frame) -> None:
        ttk.Label(parent, text="Entry Analysis", font=("TkDefaultFont", 10, "bold")).pack(
            anchor=tk.W, pady=(0, 4)
        )

        frame = ttk.Frame(parent)
        frame.pack(fill=tk.BOTH, expand=True)

        scrollbar = ttk.Scrollbar(frame, orient=tk.VERTICAL)
        self._output_text = tk.Text(
            frame,
            state=tk.DISABLED,
            wrap=tk.WORD,
            font=("Menlo", 11) if sys.platform == "darwin" else ("Consolas", 10),
            yscrollcommand=scrollbar.set,
        )
        scrollbar.config(command=self._output_text.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self._output_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

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

    # --------------------------------------------------------------- submit

    def _on_submit(self) -> None:
        self._output_text.config(state=tk.NORMAL)
        self._output_text.delete("1.0", tk.END)
        self._output_text.config(state=tk.DISABLED)
        self._submit_btn.config(state=tk.DISABLED)

        thread = threading.Thread(target=self._run_analysis, daemon=True)
        thread.start()
        self._root.after(100, self._poll_output_queue)

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
            # Offset spinner is in tenths of a percent → divide by 1000 for fraction
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

    def _poll_output_queue(self) -> None:
        try:
            while True:
                item = self._output_queue.get_nowait()
                if item is None:
                    self._submit_btn.config(state=tk.NORMAL)
                    return
                self._append_output(item)
        except queue.Empty:
            pass
        self._root.after(100, self._poll_output_queue)

    def _append_output(self, text: str) -> None:
        self._output_text.config(state=tk.NORMAL)
        self._output_text.insert(tk.END, text)
        self._output_text.see(tk.END)
        self._output_text.config(state=tk.DISABLED)


def main() -> None:
    root = tk.Tk()
    ValidationGUI(root)
    root.mainloop()
