from __future__ import annotations

from collections import defaultdict, deque
from contextlib import contextmanager
from typing import Callable, Dict, Deque, List, Optional
import os
import time

import numpy as np

try:
    from rich.console import Console
    from rich.table import Table
    from rich.text import Text
    from rich import box as rich_box
except Exception:  # pragma: no cover - rich is optional
    Console = None
    Table = None
    Text = None
    rich_box = None

_FALSE_STRINGS = {"", "0", "false", "no"}


class PerfTracker:
    """Collect and report ad-hoc performance timing samples."""

    def __init__(
        self,
        enabled_env: str,
        *,
        report_interval: float = 2.0,
        sample_window: int = 240,
    ) -> None:
        flag = os.environ.get(enabled_env, "")
        self.enabled = flag.lower() not in _FALSE_STRINGS
        self._report_interval = float(max(report_interval, 0.1))
        self._samples: Dict[str, Deque[float]] = defaultdict(lambda: deque(maxlen=sample_window))
        self._last_report = 0.0
        self._active: Dict[str, List[float]] = defaultdict(list)

        if self.enabled:
            self._console: Optional[Console] = Console(highlight=False, soft_wrap=False) if Console is not None else None
            print(f"[ROI PERF] Instrumentation enabled ({enabled_env}).")
        else:
            self._console = None

        if not self.enabled:
            # Swap methods for fast no-op behaviour
            self.record_ms = self._noop_record  # type: ignore
            self.record_seconds = self._noop_record  # type: ignore
            self.maybe_report = self._noop_report  # type: ignore

    # --- Recording helpers -------------------------------------------------
    def record_ms(self, key: str, duration_ms: float) -> None:
        if not self.enabled or duration_ms < 0.0:
            return
        self._samples[key].append(float(duration_ms) / 1000.0)

    def record_seconds(self, key: str, duration_s: float) -> None:
        if not self.enabled or duration_s < 0.0:
            return
        self._samples[key].append(float(duration_s))

    # --- Reporting ---------------------------------------------------------
    def maybe_report(self) -> None:
        if not self.enabled:
            return
        now = time.perf_counter()
        if (now - self._last_report) < self._report_interval:
            return
        self._last_report = now

        stats = []
        for key, deque_vals in self._samples.items():
            if not deque_vals:
                continue
            data = np.asarray(deque_vals, dtype=np.float64)
            mean_ms = float(np.mean(data) * 1000.0)
            p95_ms = float(np.percentile(data, 95) * 1000.0)
            max_ms = float(np.max(data) * 1000.0)
            stats.append((key, mean_ms, p95_ms, max_ms, len(deque_vals)))
            deque_vals.clear()

        if not stats:
            return

        stats.sort(key=lambda item: item[0])
        console = self._console
        if console is not None and Table is not None:
            table_kwargs = {}
            if rich_box is not None:
                table_kwargs["box"] = rich_box.SIMPLE_HEAVY
            table = Table(
                title="[ROI PERF] Frame Metrics",
                header_style="bold cyan",
                title_style="bold magenta",
                **table_kwargs,
            )
            table.add_column("Metric", style="bold white")
            table.add_column("Mean (ms)", justify="right")
            table.add_column("p95 (ms)", justify="right")
            table.add_column("Max (ms)", justify="right")
            table.add_column("n", justify="right", style="dim")
            for key, mean_ms, p95_ms, max_ms, count in stats:
                table.add_row(
                    key,
                    self._make_cell(mean_ms),
                    self._make_cell(p95_ms),
                    self._make_cell(max_ms),
                    str(count),
                )
            console.print(table)
        else:
            lines = ["[ROI PERF] metrics:"]
            for key, mean_ms, p95_ms, max_ms, count in stats:
                lines.append(
                    f"  {key:<24} mean={mean_ms:7.2f}ms  p95={p95_ms:7.2f}ms  max={max_ms:7.2f}ms  n={count}"
                )
            print("\n".join(lines))

    # --- Context helper ----------------------------------------------------
    @contextmanager
    def measure(self, key: str, *, unit: str = "seconds"):
        """Measure the duration of a block and record it automatically."""
        self.start(key)
        try:
            yield
        finally:
            self.stop(key, unit=unit)

    # --- Start/Stop API ----------------------------------------------------
    def start(self, key: str) -> None:
        if not self.enabled:
            return
        self._active[key].append(time.perf_counter())

    def stop(self, key: str, *, unit: str = "seconds") -> None:
        if not self.enabled:
            return
        stack = self._active.get(key)
        if not stack:
            return
        start_time = stack.pop()
        elapsed = time.perf_counter() - start_time
        if unit == "ms":
            self.record_ms(key, elapsed * 1000.0)
        else:
            self.record_seconds(key, elapsed)

    # --- Internal helpers --------------------------------------------------
    def _make_cell(self, value_ms: float):
        style = self._perf_style(value_ms)
        if self._console is None or Text is None:
            return f"{value_ms:.2f}"
        return Text(f"{value_ms:.2f}", style=style)

    @staticmethod
    def _perf_style(value_ms: float) -> str:
        if value_ms >= 5.0:
            return "bold red"
        if value_ms >= 2.0:
            return "bold yellow"
        if value_ms >= 1.0:
            return "green"
        return "cyan"

    def _noop_record(self, *_args, **_kwargs) -> None:
        return

    def _noop_report(self) -> None:
        return
