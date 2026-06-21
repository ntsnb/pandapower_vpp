from __future__ import annotations

from datetime import datetime, timezone
import queue
from threading import Event, Thread
from time import monotonic
from typing import Any

from marl_dashboard.backend.storage.metadata_store import MetadataStore
from marl_dashboard.backend.storage.parquet_writer import ParquetWriter
from marl_dashboard.logging.event_bus import EventBus, default_event_bus
from marl_dashboard.logging.live_events import semantic_live_event

_FLUSH_SENTINEL = "__flush__"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe_ratio(value: float | int | None, total: float | int | None) -> float | None:
    if value is None or total in (None, 0):
        return None
    try:
        return float(value) / float(total)
    except (TypeError, ValueError, ZeroDivisionError):
        return None


class ExperimentLogger:
    """Low-intrusion training logger.

    The public methods only enqueue side-effect records; they never return data
    that could influence training logic.
    """

    def __init__(
        self,
        *,
        run_id: str,
        data_dir: str,
        config: dict[str, Any] | None = None,
        variable_dictionary: list[dict[str, Any]] | dict[str, Any] | None = None,
        formulas: dict[str, Any] | list[dict[str, Any]] | None = None,
        metadata: dict[str, Any] | None = None,
        async_writer: bool = True,
        flush_rows: int = 512,
        flush_interval_seconds: float | None = 5.0,
        event_bus: EventBus | None = None,
    ) -> None:
        self.run_id = str(run_id)
        self.data_dir = data_dir
        self.flush_rows = max(1, int(flush_rows))
        self.flush_interval_seconds = None if flush_interval_seconds is None else max(0.1, float(flush_interval_seconds))
        self.event_bus = event_bus or default_event_bus
        self.metadata_store = MetadataStore(data_dir)
        self.writer = ParquetWriter(data_dir, self.run_id)
        self.metadata_store.initialize_run(
            run_id=self.run_id,
            config=config or {},
            variable_dictionary=variable_dictionary or [],
            formulas=formulas or {},
            metadata=metadata or {},
        )
        self._async_writer = bool(async_writer)
        self._queue: queue.Queue[Any] = queue.Queue(maxsize=100_000)
        self._stop = Event()
        self._thread: Thread | None = None
        if self._async_writer:
            self._thread = Thread(target=self._writer_loop, name=f"marl-dashboard-writer-{self.run_id}", daemon=True)
            self._thread.start()

    def _writer_loop(self) -> None:
        buffers: dict[str, list[dict[str, Any]]] = {}
        last_flush = monotonic()
        while not self._stop.is_set() or not self._queue.empty():
            try:
                item = self._queue.get(timeout=0.1)
            except queue.Empty:
                if (
                    self.flush_interval_seconds is not None
                    and buffers
                    and monotonic() - last_flush >= self.flush_interval_seconds
                ):
                    self._flush_buffers(buffers)
                    last_flush = monotonic()
                continue
            if item is None:
                self._queue.task_done()
                break
            table, rows = item
            if table == _FLUSH_SENTINEL:
                self._flush_buffers(buffers)
                last_flush = monotonic()
                rows.set()
                self._queue.task_done()
                continue
            buffers.setdefault(table, []).extend(rows)
            if len(buffers[table]) >= self.flush_rows:
                self.writer.write_records(table, buffers.pop(table))
                last_flush = monotonic()
            self._queue.task_done()
        self._flush_buffers(buffers)

    def _flush_buffers(self, buffers: dict[str, list[dict[str, Any]]]) -> None:
        for table, rows in list(buffers.items()):
            if rows:
                self.writer.write_records(table, rows)
        buffers.clear()

    def _write(self, table: str, rows: list[dict[str, Any]]) -> None:
        if not rows:
            return
        if self._async_writer:
            try:
                self._queue.put_nowait((table, rows))
            except queue.Full:
                self.event_bus.publish(
                    {
                        "run_id": self.run_id,
                        "event_type": "logger_warning",
                        "message": "dashboard logger queue is full; dropped rows",
                        "table": table,
                        "timestamp": _now(),
                    }
                )
        else:
            self.writer.write_records(table, rows)
        semantic_event = semantic_live_event(run_id=self.run_id, table=table, rows=rows[-10:])
        if semantic_event is not None:
            self.event_bus.publish(semantic_event)
        for row in rows[-10:]:
            self.event_bus.publish({"run_id": self.run_id, "table": table, **row})

    def _metric_rows(
        self,
        *,
        metric_group: str,
        values: dict[str, Any],
        units: dict[str, str] | None = None,
        formulas: dict[str, str] | None = None,
        descriptions: dict[str, str] | None = None,
        **context: Any,
    ) -> list[dict[str, Any]]:
        units = units or {}
        formulas = formulas or {}
        descriptions = descriptions or {}
        rows = []
        for metric_name, value in values.items():
            rows.append(
                {
                    "run_id": self.run_id,
                    "logged_at": _now(),
                    "metric_group": metric_group,
                    "metric_name": str(metric_name),
                    "value": value,
                    "unit": units.get(metric_name),
                    "formula_latex": formulas.get(metric_name),
                    "description": descriptions.get(metric_name),
                    **context,
                }
            )
        return rows

    def log_dataset(
        self,
        *,
        values: dict[str, Any],
        units: dict[str, str] | None = None,
        formulas: dict[str, str] | None = None,
        descriptions: dict[str, str] | None = None,
        **context: Any,
    ) -> None:
        self._write(
            "dataset_timeseries",
            self._metric_rows(
                metric_group="dataset",
                values=values,
                units=units,
                formulas=formulas,
                descriptions=descriptions,
                **context,
            ),
        )

    def log_reward_terms(
        self,
        *,
        terms: dict[str, Any],
        units: dict[str, str] | None = None,
        formulas: dict[str, str] | None = None,
        descriptions: dict[str, str] | None = None,
        **context: Any,
    ) -> None:
        total = terms.get("total_reward")
        rows = self._metric_rows(
            metric_group="reward",
            values=terms,
            units=units,
            formulas=formulas,
            descriptions=descriptions,
            **context,
        )
        for row in rows:
            row["term_name"] = row["metric_name"]
            row["sign_convention"] = "larger_is_better"
            row["component_ratio"] = 1.0 if row["metric_name"] == "total_reward" else _safe_ratio(row.get("value"), total)
        self._write("reward_terms", rows)

    def log_cost_terms(
        self,
        *,
        terms: dict[str, Any],
        units: dict[str, str] | None = None,
        formulas: dict[str, str] | None = None,
        descriptions: dict[str, str] | None = None,
        **context: Any,
    ) -> None:
        total = terms.get("total_cost")
        rows = self._metric_rows(
            metric_group="cost",
            values=terms,
            units=units,
            formulas=formulas,
            descriptions=descriptions,
            **context,
        )
        for row in rows:
            row["term_name"] = row["metric_name"]
            row["sign_convention"] = "smaller_is_better"
            row["component_ratio"] = 1.0 if row["metric_name"] == "total_cost" else _safe_ratio(row.get("value"), total)
        self._write("cost_terms", rows)

    def log_loss_terms(
        self,
        *,
        terms: dict[str, Any],
        units: dict[str, str] | None = None,
        formulas: dict[str, str] | None = None,
        descriptions: dict[str, str] | None = None,
        optimizer_name: str | None = None,
        network_name: str | None = None,
        **context: Any,
    ) -> None:
        total = terms.get("total_loss")
        rows = self._metric_rows(
            metric_group="loss",
            values=terms,
            units=units or {name: "scalar" for name in terms},
            formulas=formulas,
            descriptions=descriptions,
            optimizer_name=optimizer_name,
            network_name=network_name,
            **context,
        )
        for row in rows:
            row["term_name"] = row["metric_name"]
            row["component_ratio"] = 1.0 if row["metric_name"] == "total_loss" else _safe_ratio(row.get("value"), total)
        self._write("loss_terms", rows)

    def log_scalar(self, metric_name: str, value: Any, **context: Any) -> None:
        self._write(
            "scalar_metrics",
            self._metric_rows(metric_group="scalar", values={metric_name: value}, units={metric_name: "scalar"}, **context),
        )

    def log_event(self, event_type: str, payload: dict[str, Any] | None = None, **context: Any) -> None:
        payload = payload or {}
        row = {
            "run_id": self.run_id,
            "logged_at": _now(),
            "metric_group": "event",
            "metric_name": str(event_type),
            "event_type": str(event_type),
            "value": payload.get("message", str(payload)),
            "payload": payload,
            **context,
        }
        self._write("events", [row])

    def flush(self) -> None:
        if self._async_writer:
            flushed = Event()
            self._queue.put((_FLUSH_SENTINEL, flushed))
            self._queue.join()
            flushed.wait(timeout=10)

    def close(self, status: str = "finished") -> None:
        if self._async_writer:
            self._stop.set()
            self._queue.put(None)
            if self._thread is not None:
                self._thread.join(timeout=10)
        self.metadata_store.update_status(self.run_id, status)

    def __enter__(self) -> "ExperimentLogger":
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        self.close(status="error" if exc is not None else "finished")
