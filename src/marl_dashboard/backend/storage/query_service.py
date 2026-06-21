from __future__ import annotations

from collections import OrderedDict
from copy import deepcopy
from pathlib import Path
import re
from typing import Any

import pandas as pd

from marl_dashboard.backend.storage.duckdb_store import DuckDBStore
from marl_dashboard.backend.storage.metadata_store import MetadataStore
from marl_dashboard.backend.storage.variable_enrichment import (
    enrich_formula_value,
    enrich_metric_record,
    should_use_default_formula,
    variable_defaults,
)

SYNTHETIC_DATE_PREFIX = "profile_day_"

TABLE_BY_SCOPE = {
    "dataset": "dataset_timeseries",
    "reward": "reward_terms",
    "cost": "cost_terms",
    "loss": "loss_terms",
    "scalar": "scalar_metrics",
    "event": "events",
}

DEDUPLICATED_METRIC_TABLES = {"dataset_timeseries", "reward_terms", "cost_terms", "loss_terms"}
LARGE_UNFILTERED_DATASET_FILE_THRESHOLD = 1000

POLICY_ID_ALIASES = {
    "happo": "happo_sequential_ctde",
    "hatrpo": "hatrpo_trust_region_ctde",
    "matd3": "matd3_continuous_dispatch",
    "hasac": "hasac_continuous_dispatch",
}

RUN_STATUS_POLICY_IDS = {"paper_long", "paper_long_watchdog"}
RUN_STATUS_METRIC_NAMES = {
    "progress_rows",
    "episode_rows",
    "loss_rows",
    "reward_rows",
    "cost_rows",
    "dataset_rows",
    "latest_episode",
    "latest_step",
    "latest_reward_so_far",
    "latest_total_cost_so_far",
}

POLICY_ID_REVERSE_ALIASES: dict[str, list[str]] = {}
for _alias, _canonical in POLICY_ID_ALIASES.items():
    POLICY_ID_REVERSE_ALIASES.setdefault(_canonical, [_canonical]).append(_alias)

METRIC_QUERY_COLUMNS = [
    "run_id",
    "logged_at",
    "metric_group",
    "metric_name",
    "value",
    "unit",
    "formula_latex",
    "description",
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
    "component_ratio",
    "optimizer_name",
    "network_name",
    "group",
]

SEMANTIC_DEDUP_COLUMNS = [
    "run_id",
    "metric_group",
    "metric_name",
    "value",
    "unit",
    "formula_latex",
    "description",
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
    "component_ratio",
    "optimizer_name",
    "network_name",
]

DATASET_DISPLAY_DEDUP_COLUMNS = [
    "run_id",
    "metric_group",
    "metric_name",
    "value",
    "unit",
    "epoch_id",
    "episode_id",
    "batch_id",
    "gradient_step",
    "vpp_id",
    "agent_id",
    "policy_id",
    "date",
    "time_index",
    "timestamp",
]

INTEGER_CONTEXT_COLUMNS = (
    "epoch_id",
    "episode_id",
    "gradient_step",
    "global_env_step",
    "time_index",
)


def _as_list(value: str | list[str] | None) -> list[str] | None:
    if value is None or value == "":
        return None
    if isinstance(value, list):
        return value
    return [item.strip() for item in str(value).split(",") if item.strip()]


def _records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    if frame.empty:
        return []
    clean = frame.where(pd.notna(frame), None)
    return clean.to_dict(orient="records")


def _metric_records(
    frame: pd.DataFrame,
    *,
    synthetic_steps_per_day: int | None = None,
    synthetic_date_labels: list[str] | None = None,
) -> list[dict[str, Any]]:
    return [
        enrich_metric_record(_normalize_context(_with_synthetic_date(row, synthetic_steps_per_day, synthetic_date_labels)))
        for row in _records(frame)
    ]


def _normalize_metric_frame(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame
    normalized = frame.copy()
    if "env_id" in normalized.columns:
        normalized["env_id"] = normalized["env_id"].map(_normalize_env_id_value)
    if "policy_id" in normalized.columns:
        normalized["policy_id"] = normalized["policy_id"].map(_canonical_policy_id_value)
    return normalized


def _deduplicate_metric_frame(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame
    columns = [column for column in SEMANTIC_DEDUP_COLUMNS if column in frame.columns]
    if not columns:
        return frame
    return frame.drop_duplicates(subset=columns, keep="last")


def _deduplicate_dataset_display_frame(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame
    columns = [column for column in DATASET_DISPLAY_DEDUP_COLUMNS if column in frame.columns]
    if not columns:
        return frame
    return frame.drop_duplicates(subset=columns, keep="last")


def _normalize_context(row: dict[str, Any]) -> dict[str, Any]:
    return _normalize_env_context(_normalize_integer_context(row))


def _normalize_integer_context(row: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(row)
    for column in INTEGER_CONTEXT_COLUMNS:
        if column not in normalized:
            continue
        value = normalized.get(column)
        parsed = _optional_int(value)
        if parsed is not None:
            normalized[column] = parsed
        elif value is None or str(value).strip().lower() in {"", "nan", "none", "unknown"}:
            normalized[column] = None
    return normalized


def _normalize_env_context(row: dict[str, Any]) -> dict[str, Any]:
    if "env_id" not in row:
        return row
    normalized = dict(row)
    normalized["env_id"] = _normalize_env_id_value(normalized.get("env_id"))
    return normalized


def _canonical_policy_id_value(value: Any) -> Any:
    if value is None or pd.isna(value):
        return value
    text = str(value)
    return POLICY_ID_ALIASES.get(text, text)


def _policy_filter_values(policy_id: str | None) -> str | list[str] | None:
    if policy_id is None or policy_id == "":
        return None
    canonical = str(_canonical_policy_id_value(policy_id))
    values = POLICY_ID_REVERSE_ALIASES.get(canonical, [canonical])
    return values[0] if len(values) == 1 else values


def _is_aggregate_run_status_policy_row(row: dict[str, Any]) -> bool:
    policy_id = str(row.get("policy_id") or "")
    vpp_id = str(row.get("vpp_id") or "")
    metric_name = str(row.get("metric_name") or "")
    return (
        policy_id in RUN_STATUS_POLICY_IDS
        and vpp_id == "aggregate"
        and metric_name in RUN_STATUS_METRIC_NAMES
    )


def _date_statuses(dates: list[str], counts: dict[str, int], expected_time_slots: int | None) -> list[dict[str, Any]]:
    if not dates or not counts:
        return []
    expected = max(1, int(expected_time_slots or max(counts.values(), default=1)))
    statuses = []
    for date in dates:
        observed = int(counts.get(str(date), 0))
        complete = observed >= expected
        statuses.append(
            {
                "date": str(date),
                "observed_time_slots": observed,
                "expected_time_slots": expected,
                "complete": complete,
                "status": "complete" if complete else "partial",
            }
        )
    return statuses


def _normalize_env_id_value(value: Any) -> Any:
    if value is None:
        return None
    text = str(value)
    match = re.fullmatch(r"(worker_\d+)\.0", text)
    return match.group(1) if match else value


def _deduplicate_compare_frame(frame: pd.DataFrame, group_by: str) -> pd.DataFrame:
    if frame.empty or "metric_name" not in frame or group_by not in frame:
        return frame
    sort_columns = [
        column
        for column in ("logged_at", "epoch_id", "episode_id", "gradient_step", "global_env_step", "date", "time_index")
        if column in frame.columns
    ]
    sorted_frame = frame.sort_values(sort_columns) if sort_columns else frame
    return sorted_frame.drop_duplicates(subset=[group_by, "metric_name"], keep="last")


def _sort_metric_frame_for_display(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame
    sort_columns = [
        column
        for column in (
            "date",
            "time_index",
            "timestamp",
            "gradient_step",
            "global_env_step",
            "epoch_id",
            "episode_id",
            "env_id",
            "vpp_id",
            "agent_id",
            "policy_id",
            "metric_name",
        )
        if column in frame.columns
    ]
    if not sort_columns:
        return frame
    return frame.sort_values(sort_columns, kind="mergesort", na_position="last")


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            return None
        if numeric.is_integer():
            return int(numeric)
        return None


def _synthetic_date_label(day_index: int) -> str:
    return f"{SYNTHETIC_DATE_PREFIX}{day_index + 1:03d}"


def _parse_synthetic_date(value: str | None) -> int | None:
    if not value or not str(value).startswith(SYNTHETIC_DATE_PREFIX):
        return None
    try:
        return int(str(value).removeprefix(SYNTHETIC_DATE_PREFIX)) - 1
    except ValueError:
        return None


def _infer_steps_per_day(values: list[Any]) -> int | None:
    time_indices = []
    for value in values:
        numeric = _optional_int(value)
        if numeric is not None and numeric >= 0:
            time_indices.append(numeric)
    if not time_indices:
        return None
    max_index = max(time_indices)
    span = max_index + 1
    if span > 96 and span % 96 == 0:
        return 96
    if span > 24 and span % 24 == 0:
        return 24
    if max_index >= 95:
        return 96
    if max_index >= 23:
        return 24
    return None


def _has_real_values(values: list[Any]) -> bool:
    return bool(QueryService._distinct_values(values, str))


def _local_time_indices(values: list[Any], steps_per_day: int) -> list[int]:
    local_values = []
    for value in values:
        numeric = _optional_int(value)
        if numeric is not None and numeric >= 0:
            local_values.append(numeric % steps_per_day)
    return QueryService._distinct_values(local_values, int)


def _synthetic_dates(values: list[Any], steps_per_day: int) -> list[str]:
    day_indices = []
    time_indices = []
    for value in values:
        numeric = _optional_int(value)
        if numeric is not None and numeric >= 0:
            time_indices.append(numeric)
            day_indices.append(numeric // steps_per_day)
    if time_indices:
        max_time_index = max(time_indices)
        boundary_day_index = max_time_index // steps_per_day
        if max_time_index > 0 and max_time_index % steps_per_day == 0:
            day_indices = [day_index for day_index in day_indices if day_index != boundary_day_index]
    return [_synthetic_date_label(day_index) for day_index in QueryService._distinct_values(day_indices, int)]


def _with_synthetic_date(
    row: dict[str, Any],
    steps_per_day: int | None,
    date_labels: list[str] | None = None,
) -> dict[str, Any]:
    if steps_per_day is None or row.get("date") not in (None, ""):
        return row
    time_index = _optional_int(row.get("time_index"))
    if time_index is None or time_index < 0:
        return row
    enriched = dict(row)
    day_index = time_index // steps_per_day
    enriched["date"] = (
        date_labels[day_index] if date_labels is not None and 0 <= day_index < len(date_labels) else _synthetic_date_label(day_index)
    )
    enriched["time_index"] = time_index % steps_per_day
    if enriched.get("global_env_step") in (None, ""):
        enriched["global_env_step"] = time_index
    return enriched


def _resolve_synthetic_filters(
    *,
    date: str | None,
    start_time_index: int | None,
    end_time_index: int | None,
    steps_per_day: int | None,
) -> tuple[str | None, int | None, int | None]:
    day_index = _parse_synthetic_date(date)
    if day_index is None or steps_per_day is None:
        return date, start_time_index, end_time_index
    day_start = day_index * steps_per_day
    local_start = 0 if start_time_index is None else int(start_time_index)
    local_end = steps_per_day - 1 if end_time_index is None else int(end_time_index)
    return None, day_start + local_start, day_start + local_end


def _resolve_real_date_as_synthetic_filter(
    *,
    date: str | None,
    start_time_index: int | None,
    end_time_index: int | None,
    steps_per_day: int | None,
    date_labels: list[str] | None,
) -> tuple[str | None, int | None, int | None, bool]:
    if date is None or steps_per_day is None or not date_labels or _parse_synthetic_date(date) is not None:
        return date, start_time_index, end_time_index, True
    if date not in date_labels:
        return date, start_time_index, end_time_index, True
    day_index = date_labels.index(date)
    day_start = day_index * steps_per_day
    local_start = 0 if start_time_index is None else int(start_time_index)
    local_end = steps_per_day - 1 if end_time_index is None else int(end_time_index)
    return None, day_start + local_start, day_start + local_end, True


def _metadata_count(metadata: dict[str, Any], *keys: str) -> int | None:
    nested = metadata.get("metadata")
    nested_metadata = nested if isinstance(nested, dict) else {}
    for key in keys:
        count = _optional_int(metadata.get(key))
        if count is not None:
            return count
        count = _optional_int(nested_metadata.get(key))
        if count is not None:
            return count
    return None


class QueryService:
    def __init__(self, data_dir: str | Path) -> None:
        self.data_dir = Path(data_dir).expanduser().resolve()
        self.metadata_store = MetadataStore(self.data_dir)
        self.duckdb_store = DuckDBStore(self.data_dir)
        self._selectors_cache: dict[str, tuple[tuple[tuple[str, tuple[int, int, int]], ...], dict[str, Any]]] = {}
        self._metric_query_cache: OrderedDict[tuple[Any, ...], dict[str, Any]] = OrderedDict()
        self._metric_query_cache_max_size = 128

    def runs(self) -> list[dict[str, Any]]:
        rows = []
        for metadata in self.metadata_store.list_runs():
            run_id = str(metadata.get("run_id", ""))
            rows.append(
                {
                    "run_id": run_id,
                    "status": metadata.get("status", "unknown"),
                    "started_at": metadata.get("started_at"),
                    "ended_at": metadata.get("ended_at"),
                    "algorithm": metadata.get("algorithm"),
                    "environment": metadata.get("environment"),
                    "vpp_count": _metadata_count(metadata, "vpp_count", "num_vpps", "n_vpps"),
                    "epoch_count": _metadata_count(metadata, "epoch_count", "epochs", "num_epochs"),
                    "metadata": metadata.get("metadata", {}),
                }
            )
        return rows

    def metadata(self, run_id: str) -> dict[str, Any]:
        payload = self.metadata_store.metadata(run_id)
        payload["config"] = self.metadata_store.config(run_id)
        payload["selectors"] = self.selectors(run_id)
        return payload

    def variables(self, run_id: str) -> list[dict[str, Any]]:
        return self.metadata_store.variables(run_id)

    def formulas(self, run_id: str) -> dict[str, Any]:
        return self.metadata_store.formulas(run_id)

    def selectors(self, run_id: str) -> dict[str, Any]:
        tables = ("dataset_timeseries", "reward_terms", "cost_terms", "loss_terms")
        signature = tuple((table, self.duckdb_store.table_signature(run_id, table)) for table in tables)
        cached = self._selectors_cache.get(run_id)
        if cached is not None and cached[0] == signature:
            return cached[1]
        selector_columns = ("date", "vpp_id", "agent_id", "policy_id", "epoch_id", "episode_id", "time_index")
        values_by_column: dict[str, list[Any]] = {column: [] for column in selector_columns}
        dataset_time_values: list[Any] = []
        for table in tables:
            batch = self.duckdb_store.distinct_values_for_columns(run_id, table, list(selector_columns))
            for column in selector_columns:
                values_by_column[column].extend(batch.get(column, []))
            if table == "dataset_timeseries":
                dataset_time_values.extend(batch.get("time_index", []))
            values_by_column["vpp_id"].extend(self.duckdb_store.partition_values(run_id, table, "vpp_id"))
            values_by_column["epoch_id"].extend(self.duckdb_store.partition_values(run_id, table, "epoch_id"))
        date_values = self._distinct_values(values_by_column["date"], str)
        synthetic_steps_per_day = None if date_values else _infer_steps_per_day(dataset_time_values)
        real_date_steps_per_day = _infer_steps_per_day(dataset_time_values) if date_values else None
        time_indices = (
            _local_time_indices(dataset_time_values, synthetic_steps_per_day)
            if synthetic_steps_per_day is not None
            else (
                _local_time_indices(dataset_time_values, real_date_steps_per_day)
                if real_date_steps_per_day is not None
                else self._distinct_values(values_by_column["time_index"], int)
            )
        )
        expected_time_slots = len(time_indices) if time_indices else None
        date_statuses = _date_statuses(
            date_values,
            self.duckdb_store.date_time_index_counts(run_id, "dataset_timeseries"),
            expected_time_slots,
        )
        payload = {
            "run_id": run_id,
            "dates": date_values if synthetic_steps_per_day is None else _synthetic_dates(dataset_time_values, synthetic_steps_per_day),
            "vpp_ids": self._distinct_values(values_by_column["vpp_id"], str),
            "agent_ids": self._distinct_values(values_by_column["agent_id"], str),
            "policy_ids": self._selector_policy_ids(run_id, tables),
            "epoch_ids": self._distinct_values(values_by_column["epoch_id"], int),
            "episode_ids": self._distinct_values(values_by_column["episode_id"], int),
            "time_indices": time_indices,
            "date_statuses": date_statuses,
        }
        self._selectors_cache[run_id] = (signature, payload)
        return payload

    def _selector_policy_ids(self, run_id: str, tables: tuple[str, ...]) -> list[str]:
        values: list[Any] = []
        for table in tables:
            rows = self.duckdb_store.distinct_rows_for_columns(run_id, table, ["policy_id", "vpp_id", "metric_name"])
            for row in rows:
                policy_id = row.get("policy_id")
                if policy_id is None:
                    continue
                if _is_aggregate_run_status_policy_row(row):
                    continue
                values.append(_canonical_policy_id_value(policy_id))
        return self._distinct_values(values, str)

    @staticmethod
    def _distinct(frame: pd.DataFrame, column: str, cast: type) -> list[Any]:
        if column not in frame:
            return []
        return QueryService._distinct_values(frame[column].dropna().tolist(), cast)

    @staticmethod
    def _distinct_values(raw_values: list[Any], cast: type) -> list[Any]:
        values = []
        seen: set[Any] = set()
        for value in raw_values:
            if str(value).lower() in {"", "nan", "none", "unknown"}:
                continue
            try:
                cast_value = cast(value)
            except (TypeError, ValueError):
                if cast is int:
                    optional_int = _optional_int(value)
                    cast_value = optional_int if optional_int is not None else value
                else:
                    cast_value = value
            if cast_value in seen:
                continue
            seen.add(cast_value)
            values.append(cast_value)
        return sorted(values)

    def _distinct_across_tables(self, run_id: str, tables: tuple[str, ...], column: str, cast: type) -> list[Any]:
        values: list[Any] = []
        for table in tables:
            values.extend(self.duckdb_store.distinct_values(run_id, table, column))
        return self._distinct_values(values, cast)

    def _distinct_partition_and_column(self, run_id: str, tables: tuple[str, ...], column: str, cast: type) -> list[Any]:
        values: list[Any] = []
        for table in tables:
            values.extend(self.duckdb_store.partition_values(run_id, table, column))
            values.extend(self.duckdb_store.distinct_values(run_id, table, column))
        return self._distinct_values(values, cast)

    def _synthetic_steps_per_day(self, run_id: str, table: str, date: str | None) -> int | None:
        values = self.duckdb_store.distinct_values_for_columns(run_id, table, ["date", "time_index"])
        if _has_real_values(values.get("date", [])):
            return None
        steps_per_day = _infer_steps_per_day(values.get("time_index", []))
        if steps_per_day is None and _parse_synthetic_date(date) is not None:
            return 96
        return steps_per_day

    def _dataset_date_labels(self, run_id: str) -> list[str]:
        values = self.duckdb_store.distinct_values_for_columns(run_id, "dataset_timeseries", ["date"])
        return self._distinct_values(values.get("date", []), str)

    def _dataset_steps_per_day(self, run_id: str) -> int | None:
        values = self.duckdb_store.distinct_values_for_columns(run_id, "dataset_timeseries", ["time_index"])
        return _infer_steps_per_day(values.get("time_index", []))

    def _query_filters(
        self,
        *,
        run_id: str,
        table: str,
        date: str | None,
        start_time_index: int | None,
        end_time_index: int | None,
    ) -> tuple[str | None, int | None, int | None, int | None, bool, list[str] | None]:
        synthetic_steps_per_day = self._synthetic_steps_per_day(run_id, table, date)
        synthetic_date_labels: list[str] | None = None
        valid_synthetic_date = True
        if _parse_synthetic_date(date) is not None:
            values = self.duckdb_store.distinct_values_for_columns(run_id, table, ["time_index"])
            valid_synthetic_date = (
                synthetic_steps_per_day is not None
                and str(date) in set(_synthetic_dates(values.get("time_index", []), synthetic_steps_per_day))
            )
        query_date, query_start, query_end = _resolve_synthetic_filters(
            date=date,
            start_time_index=start_time_index,
            end_time_index=end_time_index,
            steps_per_day=synthetic_steps_per_day,
        )
        if synthetic_steps_per_day is not None and _parse_synthetic_date(date) is None:
            dataset_date_labels = self._dataset_date_labels(run_id)
            if dataset_date_labels:
                synthetic_date_labels = dataset_date_labels
                if date in dataset_date_labels:
                    query_date, query_start, query_end, valid_synthetic_date = _resolve_real_date_as_synthetic_filter(
                        date=date,
                        start_time_index=start_time_index,
                        end_time_index=end_time_index,
                        steps_per_day=synthetic_steps_per_day,
                        date_labels=dataset_date_labels,
                    )
        return query_date, query_start, query_end, synthetic_steps_per_day, valid_synthetic_date, synthetic_date_labels

    def _metric_query_cache_key(
        self,
        *,
        run_id: str,
        table: str,
        metrics: str | list[str] | None,
        epoch_id: int | None,
        episode_id: int | None,
        date: str | None,
        vpp_id: str | None,
        agent_id: str | None,
        start_time_index: int | None,
        end_time_index: int | None,
        policy_id: str | None,
        gradient_step: int | None,
        start_gradient_step: int | None,
        end_gradient_step: int | None,
        max_points: int,
        latest_first: bool,
    ) -> tuple[Any, ...]:
        metric_names = tuple(_as_list(metrics) or ())
        return (
            "metric_query",
            run_id,
            table,
            self.duckdb_store.table_signature(run_id, table),
            metric_names,
            epoch_id,
            episode_id,
            date,
            vpp_id,
            agent_id,
            start_time_index,
            end_time_index,
            policy_id,
            gradient_step,
            start_gradient_step,
            end_gradient_step,
            max_points,
            latest_first,
        )

    @staticmethod
    def _requires_narrow_dataset_filter(
        *,
        table: str,
        table_signature: tuple[int, int, int],
        epoch_id: int | None,
        episode_id: int | None,
        date: str | None,
        vpp_id: str | None,
        agent_id: str | None,
        start_time_index: int | None,
        end_time_index: int | None,
    ) -> bool:
        if table != "dataset_timeseries" or table_signature[0] < LARGE_UNFILTERED_DATASET_FILE_THRESHOLD:
            return False
        return not any(
            value is not None
            for value in (
                epoch_id,
                episode_id,
                date,
                vpp_id,
                agent_id,
                start_time_index,
                end_time_index,
            )
        )

    def _cached_metric_query(self, key: tuple[Any, ...]) -> dict[str, Any] | None:
        cached = self._metric_query_cache.get(key)
        if cached is None:
            return None
        self._metric_query_cache.move_to_end(key)
        return deepcopy(cached)

    def _store_metric_query_cache(self, key: tuple[Any, ...], result: dict[str, Any]) -> dict[str, Any]:
        self._metric_query_cache[key] = deepcopy(result)
        self._metric_query_cache.move_to_end(key)
        while len(self._metric_query_cache) > self._metric_query_cache_max_size:
            self._metric_query_cache.popitem(last=False)
        return result

    def _filtered_frame(
        self,
        *,
        run_id: str,
        table: str,
        metrics: str | list[str] | None = None,
        epoch_id: int | None = None,
        episode_id: int | None = None,
        date: str | None = None,
        vpp_id: str | None = None,
        agent_id: str | None = None,
        start_time_index: int | None = None,
        end_time_index: int | None = None,
        policy_id: str | None = None,
        gradient_step: int | None = None,
        start_gradient_step: int | None = None,
        end_gradient_step: int | None = None,
    ) -> pd.DataFrame:
        return self.duckdb_store.filtered_rows(
            run_id=run_id,
            table=table,
            metrics=metrics,
            epoch_id=epoch_id,
            episode_id=episode_id,
            date=date,
            vpp_id=vpp_id,
            agent_id=agent_id,
            start_time_index=start_time_index,
            end_time_index=end_time_index,
            policy_id=policy_id,
            gradient_step=gradient_step,
            start_gradient_step=start_gradient_step,
            end_gradient_step=end_gradient_step,
        )

    def query_metric_table(
        self,
        *,
        run_id: str,
        table: str,
        metrics: str | list[str] | None = None,
        epoch_id: int | None = None,
        episode_id: int | None = None,
        date: str | None = None,
        vpp_id: str | None = None,
        agent_id: str | None = None,
        start_time_index: int | None = None,
        end_time_index: int | None = None,
        policy_id: str | None = None,
        gradient_step: int | None = None,
        start_gradient_step: int | None = None,
        end_gradient_step: int | None = None,
        max_points: int = 2000,
        latest_first: bool = False,
    ) -> dict[str, Any]:
        max_points = max(1, int(max_points))
        latest_order_columns = ["logged_at", "timestamp", "epoch_id", "episode_id", "time_index"]
        cache_key = self._metric_query_cache_key(
            run_id=run_id,
            table=table,
            metrics=metrics,
            epoch_id=epoch_id,
            episode_id=episode_id,
            date=date,
            vpp_id=vpp_id,
            agent_id=agent_id,
            start_time_index=start_time_index,
            end_time_index=end_time_index,
            policy_id=policy_id,
            gradient_step=gradient_step,
            start_gradient_step=start_gradient_step,
            end_gradient_step=end_gradient_step,
            max_points=max_points,
            latest_first=latest_first,
        )
        cached = self._cached_metric_query(cache_key)
        if cached is not None:
            return cached
        table_signature = cache_key[3]
        if self._requires_narrow_dataset_filter(
            table=table,
            table_signature=table_signature,
            epoch_id=epoch_id,
            episode_id=episode_id,
            date=date,
            vpp_id=vpp_id,
            agent_id=agent_id,
            start_time_index=start_time_index,
            end_time_index=end_time_index,
        ):
            return self._store_metric_query_cache(
                cache_key,
                {
                    "chart_series": [],
                    "table_rows": [],
                    "units": {},
                    "formulas": {},
                    "summary": {
                        "row_count": 0,
                        "requires_filter": True,
                        "reason": "large_unfiltered_dataset_query",
                        "file_count": int(table_signature[0]),
                    },
                },
            )
        (
            query_date,
            query_start_time_index,
            query_end_time_index,
            synthetic_steps_per_day,
            valid_synthetic_date,
            synthetic_date_labels,
        ) = self._query_filters(
            run_id=run_id,
            table=table,
            date=date,
            start_time_index=start_time_index,
            end_time_index=end_time_index,
        )
        query_policy_id = _policy_filter_values(policy_id)
        if not valid_synthetic_date:
            return self._store_metric_query_cache(
                cache_key,
                {"chart_series": [], "table_rows": [], "units": {}, "formulas": {}, "summary": {"row_count": 0}},
            )
        row_count = self.duckdb_store.filtered_count(
            run_id=run_id,
            table=table,
            metrics=metrics,
            epoch_id=epoch_id,
            episode_id=episode_id,
            date=query_date,
            vpp_id=vpp_id,
            agent_id=agent_id,
            start_time_index=query_start_time_index,
            end_time_index=query_end_time_index,
            policy_id=query_policy_id,
            gradient_step=gradient_step,
            start_gradient_step=start_gradient_step,
            end_gradient_step=end_gradient_step,
        )
        effective_vpp_id = vpp_id
        summary_extra: dict[str, Any] = {}
        if row_count == 0 and table == "loss_terms" and vpp_id not in (None, "", "aggregate"):
            aggregate_count = self.duckdb_store.filtered_count(
                run_id=run_id,
                table=table,
                metrics=metrics,
                epoch_id=epoch_id,
                episode_id=episode_id,
                date=query_date,
                vpp_id="aggregate",
                agent_id=agent_id,
                start_time_index=query_start_time_index,
                end_time_index=query_end_time_index,
                policy_id=query_policy_id,
                gradient_step=gradient_step,
                start_gradient_step=start_gradient_step,
                end_gradient_step=end_gradient_step,
            )
            if aggregate_count > 0:
                row_count = aggregate_count
                effective_vpp_id = "aggregate"
                summary_extra = {
                    "vpp_filter_fallback": "aggregate_shared_loss",
                    "requested_vpp_id": str(vpp_id),
                    "effective_vpp_id": "aggregate",
                }
        if row_count == 0:
            return self._store_metric_query_cache(
                cache_key,
                {"chart_series": [], "table_rows": [], "units": {}, "formulas": {}, "summary": {"row_count": 0}},
            )
        chart_frame = self.duckdb_store.filtered_rows(
            run_id=run_id,
            table=table,
            metrics=metrics,
            epoch_id=epoch_id,
            episode_id=episode_id,
            date=query_date,
            vpp_id=effective_vpp_id,
            agent_id=agent_id,
            start_time_index=query_start_time_index,
            end_time_index=query_end_time_index,
            policy_id=query_policy_id,
            gradient_step=gradient_step,
            start_gradient_step=start_gradient_step,
            end_gradient_step=end_gradient_step,
            columns=METRIC_QUERY_COLUMNS,
            max_rows=max_points,
            sampled=not latest_first,
            sample_by=["metric_name", "vpp_id", "policy_id"] if not latest_first else None,
            order_by=latest_order_columns if latest_first else None,
            order_desc=latest_first,
        )
        table_frame = self.duckdb_store.filtered_rows(
            run_id=run_id,
            table=table,
            metrics=metrics,
            epoch_id=epoch_id,
            episode_id=episode_id,
            date=query_date,
            vpp_id=effective_vpp_id,
            agent_id=agent_id,
            start_time_index=query_start_time_index,
            end_time_index=query_end_time_index,
            policy_id=query_policy_id,
            gradient_step=gradient_step,
            start_gradient_step=start_gradient_step,
            end_gradient_step=end_gradient_step,
            columns=METRIC_QUERY_COLUMNS,
            max_rows=max_points,
            sampled=False,
            order_by=latest_order_columns if latest_first else None,
            order_desc=latest_first,
        )
        chart_frame = _normalize_metric_frame(chart_frame)
        table_frame = _normalize_metric_frame(table_frame)
        if table in DEDUPLICATED_METRIC_TABLES:
            chart_frame = _deduplicate_metric_frame(chart_frame)
            table_frame = _deduplicate_metric_frame(table_frame)
        if table == "dataset_timeseries":
            chart_frame = _deduplicate_dataset_display_frame(chart_frame)
            table_frame = _deduplicate_dataset_display_frame(table_frame)
        summary_row_count = len(table_frame) if row_count <= max_points else int(row_count)
        metadata_frame = self.duckdb_store.filtered_metadata(
            run_id=run_id,
            table=table,
            metrics=metrics,
            epoch_id=epoch_id,
            episode_id=episode_id,
            date=query_date,
            vpp_id=effective_vpp_id,
            agent_id=agent_id,
            start_time_index=query_start_time_index,
            end_time_index=query_end_time_index,
            policy_id=query_policy_id,
            gradient_step=gradient_step,
            start_gradient_step=start_gradient_step,
            end_gradient_step=end_gradient_step,
        )
        units = {}
        if not metadata_frame.empty and "unit" in metadata_frame:
            units = {
                str(row["metric_name"]): str(row["unit"])
                for _, row in metadata_frame.dropna(subset=["metric_name"]).iterrows()
                if row.get("unit") is not None
            }
        formulas = {}
        if not metadata_frame.empty and "formula_latex" in metadata_frame:
            for _, row in metadata_frame.dropna(subset=["metric_name"]).iterrows():
                formula = row.get("formula_latex")
                if formula is None or pd.isna(formula) or str(formula) == "":
                    continue
                formulas[str(row["metric_name"])] = str(formula)
        metric_names = set()
        for frame in (metadata_frame, chart_frame, table_frame):
            if not frame.empty and "metric_name" in frame:
                metric_names.update(str(name) for name in frame["metric_name"].dropna().unique())
        for metric_name in metric_names:
            defaults = variable_defaults(metric_name)
            if not defaults:
                continue
            if metric_name not in units and defaults.get("unit") not in (None, ""):
                units[metric_name] = str(defaults["unit"])
            default_formula = defaults.get("formula_latex")
            if default_formula not in (None, "") and (
                metric_name not in formulas or should_use_default_formula(formulas.get(metric_name))
            ):
                formulas[metric_name] = str(default_formula)
            elif metric_name in formulas:
                formulas[metric_name] = str(enrich_formula_value(metric_name, formulas[metric_name]))
        chart_series = []
        for keys, group in chart_frame.groupby([col for col in ("metric_name", "vpp_id", "policy_id") if col in chart_frame], dropna=False):
            if not isinstance(keys, tuple):
                keys = (keys,)
            key_cols = [col for col in ("metric_name", "vpp_id", "policy_id") if col in chart_frame]
            label_parts = [str(item) for item in keys if item is not None and str(item) != "nan"]
            chart_series.append(
                {
                    **{key_cols[i]: (None if str(keys[i]) == "nan" else keys[i]) for i in range(len(key_cols))},
                    "name": " / ".join(label_parts),
                    "unit": units.get(str(keys[0]), "") if keys else "",
                    "points": _metric_records(
                        _sort_metric_frame_for_display(group),
                        synthetic_steps_per_day=synthetic_steps_per_day,
                        synthetic_date_labels=synthetic_date_labels,
                    ),
                }
            )
        return self._store_metric_query_cache(cache_key, {
            "chart_series": chart_series,
            "table_rows": _metric_records(
                _sort_metric_frame_for_display(table_frame),
                synthetic_steps_per_day=synthetic_steps_per_day,
                synthetic_date_labels=synthetic_date_labels,
            ),
            "units": units,
            "formulas": formulas,
            "summary": {
                "row_count": int(summary_row_count),
                "returned_points": int(len(chart_frame)),
                "returned_table_rows": int(len(table_frame)),
                "table_rows_limited": bool(row_count > max_points),
                **summary_extra,
            },
        })

    def compare(
        self,
        *,
        run_id: str,
        table: str,
        metric_names: str | list[str],
        group_by: str,
        group_values: str | list[str] | None = None,
        fixed_epoch_id: int | None = None,
        fixed_episode_id: int | None = None,
        fixed_date: str | None = None,
        fixed_time_index: int | None = None,
        max_points: int = 2000,
    ) -> dict[str, Any]:
        max_points = max(1, int(max_points))
        (
            query_date,
            query_time_index,
            query_end_time_index,
            synthetic_steps_per_day,
            valid_synthetic_date,
            synthetic_date_labels,
        ) = self._query_filters(
            run_id=run_id,
            table=table,
            date=fixed_date,
            start_time_index=fixed_time_index,
            end_time_index=fixed_time_index,
        )
        if not valid_synthetic_date:
            return {"chart_series": [], "table_rows": [], "units": {}, "formulas": {}, "summary": {"row_count": 0}}
        row_count = self.duckdb_store.filtered_count(
            run_id=run_id,
            table=table,
            metrics=metric_names,
            epoch_id=fixed_epoch_id,
            episode_id=fixed_episode_id,
            date=query_date,
            start_time_index=query_time_index,
            end_time_index=query_end_time_index,
        )
        frame = self.duckdb_store.filtered_rows(
            run_id=run_id,
            table=table,
            metrics=metric_names,
            epoch_id=fixed_epoch_id,
            episode_id=fixed_episode_id,
            date=query_date,
            start_time_index=query_time_index,
            end_time_index=query_end_time_index,
            max_rows=None if fixed_date is not None and fixed_time_index is not None else max_points,
        )
        if frame.empty or group_by not in frame:
            return {"chart_series": [], "table_rows": [], "units": {}, "formulas": {}, "summary": {"row_count": 0}}
        frame = _normalize_metric_frame(frame)
        source_row_count = int(row_count)
        selected_group_values = _as_list(group_values) or []
        if selected_group_values:
            selected_group_set = {str(value) for value in selected_group_values}
            frame = frame[frame[group_by].map(lambda value: str(value) in selected_group_set)]
            row_count = int(len(frame))
            if frame.empty:
                return {
                    "chart_series": [],
                    "table_rows": [],
                    "units": {},
                    "formulas": {},
                    "summary": {
                        "row_count": 0,
                        "raw_row_count": source_row_count,
                        "group_by": group_by,
                        "selected_group_values": selected_group_values,
                    },
                }
        raw_frame_count = int(len(frame))
        frame = _deduplicate_compare_frame(frame, group_by)
        comparison_row_count = int(len(frame))
        table_rows_limited = comparison_row_count > max_points
        if len(frame) > max_points:
            frame = frame.head(max_points)
        rows = []
        for _, row in frame.sort_values([group_by, "metric_name"]).iterrows():
            rows.append(
                {
                    "group": row.get(group_by),
                    group_by: row.get(group_by),
                    "vpp_id": row.get("vpp_id"),
                    "agent_id": row.get("agent_id"),
                    "policy_id": row.get("policy_id"),
                    "epoch_id": row.get("epoch_id"),
                    "episode_id": row.get("episode_id"),
                    "date": row.get("date"),
                    "time_index": row.get("time_index"),
                    "timestamp": row.get("timestamp"),
                    "metric_name": row.get("metric_name"),
                    "value": row.get("value"),
                    "unit": row.get("unit"),
                }
            )
        table_rows = _metric_records(
            pd.DataFrame(rows),
            synthetic_steps_per_day=synthetic_steps_per_day,
            synthetic_date_labels=synthetic_date_labels,
        )
        units = {}
        if "metric_name" in frame and "unit" in frame:
            for _, row in frame.iterrows():
                unit = row.get("unit")
                if unit is None or pd.isna(unit):
                    continue
                units[str(row["metric_name"])] = str(unit)
        formulas = {}
        if "metric_name" in frame and "formula_latex" in frame:
            for _, row in frame.iterrows():
                formula = row.get("formula_latex")
                if formula is None or pd.isna(formula) or str(formula) == "":
                    continue
                formulas[str(row["metric_name"])] = str(formula)
        metric_names_seen = {str(name) for name in frame["metric_name"].dropna().unique()} if "metric_name" in frame else set()
        for metric_name in metric_names_seen:
            defaults = variable_defaults(metric_name)
            if not defaults:
                continue
            if metric_name not in units and defaults.get("unit") not in (None, ""):
                units[metric_name] = str(defaults["unit"])
            default_formula = defaults.get("formula_latex")
            if default_formula not in (None, "") and (
                metric_name not in formulas or should_use_default_formula(formulas.get(metric_name))
            ):
                formulas[metric_name] = str(default_formula)
        chart_series = []
        table_frame = pd.DataFrame(table_rows)
        if not table_frame.empty and "metric_name" in table_frame and group_by in table_frame:
            for keys, group in table_frame.groupby(["metric_name", group_by], dropna=False):
                metric_name, group_value = keys
                label_parts = [str(metric_name), str(group_value)]
                chart_series.append(
                    {
                        "metric_name": None if pd.isna(metric_name) else metric_name,
                        group_by: None if pd.isna(group_value) else group_value,
                        "name": " / ".join(label_parts),
                        "unit": units.get(str(metric_name), ""),
                        "points": group.to_dict(orient="records"),
                    }
                )
        return {
            "chart_series": chart_series,
            "table_rows": table_rows,
            "units": units,
            "formulas": formulas,
            "summary": {
                "row_count": int(row_count),
                "raw_row_count": source_row_count,
                "comparison_row_count": comparison_row_count,
                "returned_table_rows": int(len(table_rows)),
                "table_rows_limited": bool(table_rows_limited),
                "group_by": group_by,
                "fixed_episode_id": fixed_episode_id,
                "selected_group_values": selected_group_values,
                "deduplicated_rows": int(max(0, raw_frame_count - comparison_row_count)),
            },
        }
