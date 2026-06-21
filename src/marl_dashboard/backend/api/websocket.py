from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from pathlib import Path
from time import monotonic
from typing import Any

from fastapi import FastAPI, WebSocket, WebSocketDisconnect

from marl_dashboard.backend.storage.query_service import METRIC_QUERY_COLUMNS
from marl_dashboard.logging.event_bus import default_event_bus
from marl_dashboard.logging.live_events import metric_group_for_table, semantic_live_event

LIVE_PUSH_INTERVAL_SECONDS = 0.2
PERSISTENT_LIVE_POLL_SECONDS = 2.0
PERSISTENT_LIVE_TABLES = (
    "dataset_timeseries",
    "reward_terms",
    "cost_terms",
    "loss_terms",
    "scalar_metrics",
    "events",
)


def table_signature_snapshot(query_service: Any, run_id: str) -> dict[str, tuple[int, int, int]]:
    return {
        table: query_service.duckdb_store.table_signature(run_id, table)
        for table in PERSISTENT_LIVE_TABLES
    }


def _metric_group_for_table(table: str) -> str:
    return metric_group_for_table(table)


def _records(frame: Any) -> list[dict[str, Any]]:
    if frame.empty:
        return []
    clean = frame.where(frame.notna(), None)
    return clean.to_dict(orient="records")


def _latest_persistent_rows(
    query_service: Any,
    run_id: str,
    table: str,
    *,
    max_files: int = 8,
    max_rows: int = 25,
) -> list[dict[str, Any]]:
    files = query_service.duckdb_store.table_files(run_id, table)
    if not files:
        return []
    recent_files = sorted(
        files,
        key=lambda file_name: Path(file_name).stat().st_mtime_ns if Path(file_name).exists() else 0,
        reverse=True,
    )[: max(1, int(max_files))]
    if not recent_files:
        return []
    preferred_columns = [
        *METRIC_QUERY_COLUMNS,
        "term_name",
        "event_type",
        "payload",
        "sign_convention",
    ]
    latest_order_columns = [
        "logged_at",
        "timestamp",
        "epoch_id",
        "episode_id",
        "gradient_step",
        "global_env_step",
        "date",
        "time_index",
    ]
    try:
        with query_service.duckdb_store._duckdb.connect(database=":memory:") as conn:
            available_columns = query_service.duckdb_store._table_columns(conn, recent_files)
            selected_columns = [column for column in preferred_columns if column in available_columns]
            selected_columns.extend(sorted(column for column in available_columns if column not in set(selected_columns)))
            if not selected_columns:
                return []
            order_columns = [column for column in latest_order_columns if column in available_columns]
            select_sql = ", ".join(f'"{column}"' for column in selected_columns)
            order_sql = ", ".join(f'"{column}" DESC' for column in order_columns) if order_columns else "1"
            frame = conn.execute(
                f"""
                SELECT {select_sql}
                FROM read_parquet(?, union_by_name=true)
                ORDER BY {order_sql}
                LIMIT ?
                """,
                [recent_files, max(1, int(max_rows))],
            ).fetchdf()
    except Exception:
        return []
    return _records(frame)


def persistent_live_update_events(
    query_service: Any,
    run_id: str,
    previous: dict[str, tuple[int, int, int]],
) -> tuple[list[dict[str, Any]], dict[str, tuple[int, int, int]]]:
    current = table_signature_snapshot(query_service, run_id)
    timestamp = datetime.now(timezone.utc).isoformat()
    events: list[dict[str, Any]] = []
    for table, signature in current.items():
        previous_signature = previous.get(table, (0, 0, 0))
        if signature == previous_signature or signature[0] <= 0:
            continue
        events.append(
            {
                "run_id": run_id,
                "event_type": "persistent_table_update",
                "table": table,
                "metric_group": _metric_group_for_table(table),
                "file_count": signature[0],
                "latest_mtime_ns": signature[1],
                "total_size_bytes": signature[2],
                "timestamp": timestamp,
            }
        )
        rows = _latest_persistent_rows(query_service, run_id, table)
        semantic_event = semantic_live_event(
            run_id=run_id,
            table=table,
            rows=rows,
            timestamp=timestamp,
            newest_first=True,
        )
        if semantic_event is not None:
            events.append(semantic_event)
    return events, current


def _live_event_key(event: dict[str, Any]) -> tuple[Any, ...]:
    if event.get("metric_name") in (None, ""):
        return (
            "event",
            event.get("run_id"),
            event.get("table"),
            event.get("event_type"),
            event.get("timestamp"),
            event.get("logged_at"),
            event.get("message"),
        )
    return (
        "metric",
        event.get("run_id"),
        event.get("table"),
        event.get("metric_group"),
        event.get("metric_name"),
        event.get("vpp_id"),
        event.get("agent_id"),
        event.get("policy_id"),
    )


def coalesce_live_events(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    latest: dict[tuple[Any, ...], dict[str, Any]] = {}
    for event in events:
        latest[_live_event_key(event)] = event
    return list(latest.values())


def register_websocket_routes(app: FastAPI) -> None:
    @app.websocket("/ws/runs/{run_id}/live")
    async def run_live(websocket: WebSocket, run_id: str) -> None:
        await websocket.accept()
        run_summary = next((run for run in app.state.query_service.runs() if run.get("run_id") == run_id), None)
        await websocket.send_json(
            {
                "run_id": run_id,
                "event_type": "run_status",
                "table": "metadata",
                "status": (run_summary or {}).get("status", "unknown"),
                "algorithm": (run_summary or {}).get("algorithm"),
                "environment": (run_summary or {}).get("environment"),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
        )
        for event in default_event_bus.recent(run_id=run_id, limit=50):
            await websocket.send_json(event)
        queue = default_event_bus.subscribe()
        try:
            pending: list[dict[str, Any]] = []
            last_flush = monotonic()
            last_persistent_poll = monotonic()
            persistent_signatures = table_signature_snapshot(app.state.query_service, run_id)
            while True:
                timeout = max(0.0, LIVE_PUSH_INTERVAL_SECONDS - (monotonic() - last_flush)) if pending else None
                try:
                    event = await asyncio.wait_for(queue.get(), timeout=timeout or PERSISTENT_LIVE_POLL_SECONDS)
                except asyncio.TimeoutError:
                    event = None
                if event is not None and event.get("run_id") == run_id:
                    pending.append(event)
                now = monotonic()
                if now - last_persistent_poll >= PERSISTENT_LIVE_POLL_SECONDS:
                    persistent_events, persistent_signatures = persistent_live_update_events(
                        app.state.query_service,
                        run_id,
                        persistent_signatures,
                    )
                    pending.extend(persistent_events)
                    last_persistent_poll = now
                if pending and monotonic() - last_flush >= LIVE_PUSH_INTERVAL_SECONDS:
                    for pending_event in coalesce_live_events(pending):
                        await websocket.send_json(pending_event)
                    pending.clear()
                    last_flush = monotonic()
        except WebSocketDisconnect:
            pass
        finally:
            default_event_bus.unsubscribe(queue)
