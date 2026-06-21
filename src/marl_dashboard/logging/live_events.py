from __future__ import annotations

from datetime import datetime, timezone
from typing import Any


TABLE_LIVE_EVENT_TYPES = {
    "dataset_timeseries": "latest_dataset_point",
    "reward_terms": "latest_reward_terms",
    "cost_terms": "latest_cost_terms",
    "loss_terms": "latest_loss_terms",
    "scalar_metrics": "latest_scalar_metrics",
    "events": "latest_event",
}

TABLE_METRIC_GROUPS = {
    "dataset_timeseries": "dataset",
    "reward_terms": "reward",
    "cost_terms": "cost",
    "loss_terms": "loss",
    "scalar_metrics": "scalar",
    "events": "event",
}

LIVE_CONTEXT_FIELDS = (
    "epoch_id",
    "episode_id",
    "batch_id",
    "gradient_step",
    "global_env_step",
    "env_id",
    "vpp_id",
    "agent_id",
    "policy_id",
    "date",
    "time_index",
    "timestamp",
)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def live_event_type_for_table(table: str) -> str | None:
    return TABLE_LIVE_EVENT_TYPES.get(str(table))


def metric_group_for_table(table: str) -> str:
    return TABLE_METRIC_GROUPS.get(str(table), str(table))


def semantic_live_event(
    *,
    run_id: str,
    table: str,
    rows: list[dict[str, Any]],
    timestamp: str | None = None,
    newest_first: bool = False,
) -> dict[str, Any] | None:
    event_type = live_event_type_for_table(table)
    if event_type is None or not rows:
        return None
    clean_rows = [dict(row) for row in rows if row]
    if not clean_rows:
        return None
    latest_row = clean_rows[0] if newest_first else clean_rows[-1]
    latest_context = {
        field: latest_row.get(field)
        for field in LIVE_CONTEXT_FIELDS
        if latest_row.get(field) is not None
    }
    metric_names = [
        str(row["metric_name"])
        for row in clean_rows
        if row.get("metric_name") is not None and str(row.get("metric_name")) != ""
    ]
    return {
        "run_id": run_id,
        "event_type": event_type,
        "table": table,
        "metric_group": metric_group_for_table(table),
        "timestamp": timestamp or _now(),
        "latest_context": latest_context,
        "metric_names": list(dict.fromkeys(metric_names)),
        "rows": clean_rows,
    }
