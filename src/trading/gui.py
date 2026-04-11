import io
import queue
import sys
import threading
import tkinter as tk
from datetime import UTC, datetime
from pathlib import Path
from tkinter import ttk

from trading.core.models import Timeframe
from trading.main import DataSourceType, run

_TF_VALUES = ["5m", "15m", "1h", "4h", "1d"]
_TF_SECONDS: dict[str, int] = {
    "5m": 300,
    "15m": 900,
    "1h": 3600,
    "4h": 14400,
    "1d": 86400,
}
_DATA_DIR = Path("data")


class QueueWriter(io.TextIOBase):
    def __init__(self, q: "queue.Queue[str | None]") -> None:
        self._q = q

    def write(self, s: str) -> int:
        if s:
            self._q.put(s)
        return len(s)

    def flush(self) -> None:
        pass


class TradingGUI:
    def __init__(self, root: tk.Tk) -> None:
        self._root = root
        self._root.title("Smart Money Trading System")
        self._root.geometry("1000x640")
        self._root.minsize(700, 500)

        self._output_queue: queue.Queue[str | None] = queue.Queue()

        # StringVars
        self._source_var = tk.StringVar(value="csv")
        self._htf_csv_var = tk.StringVar()
        self._ltf_csv_var = tk.StringVar()
        self._until_var = tk.StringVar()
        self._htf_tf_var = tk.StringVar(value="1h")
        self._ltf_tf_var = tk.StringVar(value="15m")
        self._htf_limit_var = tk.StringVar(value="72")
        self._ltf_limit_var = tk.StringVar(value="24")
        self._symbol_var = tk.StringVar(value="BTC/USDT:USDT")

        # Widgets declared for type checker — assigned in _build_*
        self._csv_frame: ttk.Frame
        self._until_frame: ttk.Frame
        self._htf_csv_combo: ttk.Combobox
        self._ltf_csv_combo: ttk.Combobox
        self._output_text: tk.Text
        self._submit_btn: ttk.Button

        self._build_layout()
        self._populate_csv_dropdowns()
        self._refresh_until_default()

        # Trace LTF change to update until default
        self._ltf_tf_var.trace_add("write", lambda *_: self._refresh_until_default())
        self._source_var.trace_add("write", lambda *_: self._on_source_change())

    # ------------------------------------------------------------------ layout

    def _build_layout(self) -> None:
        pane = tk.PanedWindow(self._root, orient=tk.HORIZONTAL, sashwidth=5, sashrelief=tk.RAISED)
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
            ttk.Radiobutton(src_frame, text=label, variable=self._source_var, value=value).pack(anchor=tk.W)

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
        # not packed yet — shown on demand

        ttk.Label(self._until_frame, text="To (UTC)  —  YYYY-MM-DD HH:MM").pack(anchor=tk.W)
        ttk.Entry(self._until_frame, textvariable=self._until_var, width=22).pack(anchor=tk.W, pady=(2, 0))

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
            tf_inner, textvariable=self._htf_limit_var, from_=1, to=500, width=5
        ).grid(row=0, column=2, sticky=tk.W, padx=(8, 4), pady=(0, 4))
        ttk.Label(tf_inner, text="candles").grid(row=0, column=3, sticky=tk.W, pady=(0, 4))

        ttk.Label(tf_inner, text="LTF").grid(row=1, column=0, sticky=tk.W, padx=(0, 6))
        ttk.Combobox(
            tf_inner, textvariable=self._ltf_tf_var, values=_TF_VALUES, state="readonly", width=7
        ).grid(row=1, column=1, sticky=tk.W)
        ttk.Spinbox(
            tf_inner, textvariable=self._ltf_limit_var, from_=1, to=500, width=5
        ).grid(row=1, column=2, sticky=tk.W, padx=(8, 4))
        ttk.Label(tf_inner, text="candles").grid(row=1, column=3, sticky=tk.W)

        # --- Symbol ---
        sym_frame = ttk.LabelFrame(parent, text="Symbol", padding=8)
        sym_frame.pack(fill=tk.X, pady=(0, 12))
        ttk.Entry(sym_frame, textvariable=self._symbol_var, width=18).pack(anchor=tk.W)

        # --- Submit ---
        self._submit_btn = ttk.Button(parent, text="Run Analysis", command=self._on_submit)
        self._submit_btn.pack(fill=tk.X)

    def _build_right_panel(self, parent: ttk.Frame) -> None:
        ttk.Label(parent, text="Output", font=("TkDefaultFont", 10, "bold")).pack(anchor=tk.W, pady=(0, 4))

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
        else:  # live
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

        htf_csv: str | None = None
        ltf_csv: str | None = None
        if source_raw == "csv":
            htf_csv = self._htf_csv_var.get() or None
            ltf_csv = self._ltf_csv_var.get() or None

        old_stdout = sys.stdout
        old_stderr = sys.stderr
        writer: QueueWriter = QueueWriter(self._output_queue)
        sys.stdout = writer  # type: ignore[assignment]
        sys.stderr = writer  # type: ignore[assignment]
        try:
            run(
                symbol=symbol,
                htf_timeframe=htf_tf,
                ltf_timeframe=ltf_tf,
                data_source=source_raw,  # type: ignore[arg-type]
                htf_csv=htf_csv,
                ltf_csv=ltf_csv,
                until=until,
                htf_limit=htf_limit,
                ltf_limit=ltf_limit,
            )
        except Exception as exc:
            self._output_queue.put(f"\n[ERROR] {exc}\n")
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr
            self._output_queue.put(None)  # sentinel

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
    TradingGUI(root)
    root.mainloop()
