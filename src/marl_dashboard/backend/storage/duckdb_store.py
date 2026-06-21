from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from marl_dashboard.backend.storage.parquet_writer import effective_partition_files


def _epoch_partition(epoch_id: Any) -> str:
    if epoch_id is None or epoch_id == "":
        return "epoch_id=unknown"
    try:
        return f"epoch_id={int(epoch_id):06d}"
    except (TypeError, ValueError):
        return f"epoch_id={str(epoch_id)}"


def _safe_partition(name: str, value: Any) -> str:
    text = "unknown" if value is None or value == "" else str(value)
    safe = "".join(ch if ch.isalnum() or ch in {"-", "_", "."} else "_" for ch in text)
    return f"{name}={safe}"


def _as_list(value: str | list[str] | None) -> list[str] | None:
    if value is None or value == "":
        return None
    if isinstance(value, list):
        return value
    return [item.strip() for item in str(value).split(",") if item.strip()]


class DuckDBStore:
    def __init__(self, data_dir: str | Path) -> None:
        try:
            import duckdb  # type: ignore
        except ModuleNotFoundError as exc:  # pragma: no cover - exercised when dependency is absent
            raise RuntimeError(
                "DuckDB is required for marl_dashboard queries. Install with "
                '`pip install -e ".[dashboard]"` or `pip install duckdb`.'
            ) from exc
        self._duckdb = duckdb
        self.data_dir = Path(data_dir).expanduser().resolve()

    def table_files(self, run_id: str, table: str, *, epoch_id: int | None = None, vpp_id: str | None = None) -> list[str]:
        root = self.data_dir / str(run_id) / "tables" / table
        if not root.exists():
            return []
        if epoch_id is not None and vpp_id is not None:
            root = root / _epoch_partition(epoch_id) / _safe_partition("vpp_id", vpp_id)
            return [str(path) for path in effective_partition_files(root)] if root.exists() else []
        if epoch_id is not None:
            root = root / _epoch_partition(epoch_id)
            return [str(path) for path in self._effective_table_files(root)] if root.exists() else []
        if vpp_id is not None:
            vpp_part = _safe_partition("vpp_id", vpp_id)
            return [str(path) for path in self._effective_table_files(root, vpp_part=vpp_part)]
        return [str(path) for path in self._effective_table_files(root)]

    @staticmethod
    def _effective_table_files(root: Path, *, vpp_part: str | None = None) -> list[Path]:
        partition_dirs: set[Path] = set()
        for path in root.rglob("*.parquet"):
            if "_compacted_parts" in path.parts:
                continue
            if vpp_part is not None and path.parent.name != vpp_part:
                continue
            partition_dirs.add(path.parent)
        files: list[Path] = []
        for partition_dir in sorted(partition_dirs):
            files.extend(effective_partition_files(partition_dir))
        return sorted(files)

    def table_signature(self, run_id: str, table: str) -> tuple[int, int, int]:
        count = 0
        latest_mtime_ns = 0
        total_size = 0
        for file_name in self.table_files(run_id, table):
            try:
                stat = Path(file_name).stat()
            except FileNotFoundError:
                continue
            count += 1
            latest_mtime_ns = max(latest_mtime_ns, int(stat.st_mtime_ns))
            total_size += int(stat.st_size)
        return count, latest_mtime_ns, total_size

    def partition_values(self, run_id: str, table: str, partition: str) -> list[str]:
        root = self.data_dir / str(run_id) / "tables" / table
        if not root.exists():
            return []
        prefix = f"{partition}="
        values: set[str] = set()
        for path in root.rglob(f"{prefix}*"):
            if path.is_dir() and path.name.startswith(prefix):
                values.add(path.name.removeprefix(prefix))
        return sorted(values)

    def distinct_values(self, run_id: str, table: str, column: str) -> list[Any]:
        files = self.table_files(run_id, table)
        if not files:
            return []
        try:
            with self._duckdb.connect(database=":memory:") as conn:
                frame = conn.execute(
                    f'SELECT DISTINCT "{column}" AS value FROM read_parquet(?, union_by_name=true) WHERE "{column}" IS NOT NULL',
                    [files],
                ).fetchdf()
        except Exception:
            return []
        if frame.empty:
            return []
        return frame["value"].dropna().tolist()

    def distinct_values_for_columns(self, run_id: str, table: str, columns: list[str]) -> dict[str, list[Any]]:
        files = self.table_files(run_id, table)
        if not files:
            return {column: [] for column in columns}
        try:
            with self._duckdb.connect(database=":memory:") as conn:
                available_columns = self._table_columns(conn, files)
                selected_columns = [column for column in columns if column in available_columns]
                if not selected_columns:
                    return {column: [] for column in columns}
                cast_columns = ", ".join(f'CAST("{column}" AS VARCHAR) AS "{column}"' for column in selected_columns)
                unpivot_columns = ", ".join(f'"{column}"' for column in selected_columns)
                frame = conn.execute(
                    f"""
                    SELECT column_name, value
                    FROM (
                        SELECT {cast_columns}
                        FROM read_parquet(?, union_by_name=true)
                    )
                    UNPIVOT (value FOR column_name IN ({unpivot_columns}))
                    WHERE value IS NOT NULL
                      AND LOWER(value) NOT IN ('', 'nan', 'none', 'unknown')
                    GROUP BY column_name, value
                    ORDER BY column_name, value
                    """,
                    [files],
                ).fetchdf()
        except Exception:
            return {column: [] for column in columns}
        result = {column: [] for column in columns}
        if frame.empty:
            return result
        for column, group in frame.groupby("column_name"):
            result[str(column)] = group["value"].dropna().tolist()
        return result

    def distinct_rows_for_columns(self, run_id: str, table: str, columns: list[str]) -> list[dict[str, Any]]:
        files = self.table_files(run_id, table)
        if not files:
            return []
        try:
            with self._duckdb.connect(database=":memory:") as conn:
                available_columns = self._table_columns(conn, files)
                selected_columns = [column for column in columns if column in available_columns]
                if not selected_columns:
                    return []
                select_columns = ", ".join(f'CAST("{column}" AS VARCHAR) AS "{column}"' for column in selected_columns)
                not_null_clauses = " OR ".join(f'"{column}" IS NOT NULL' for column in selected_columns)
                frame = conn.execute(
                    f"""
                    SELECT DISTINCT {select_columns}
                    FROM read_parquet(?, union_by_name=true)
                    WHERE {not_null_clauses}
                    ORDER BY {", ".join(f'"{column}"' for column in selected_columns)}
                    """,
                    [files],
                ).fetchdf()
        except Exception:
            return []
        if frame.empty:
            return []
        return frame.to_dict("records")

    def date_time_index_counts(self, run_id: str, table: str) -> dict[str, int]:
        files = self.table_files(run_id, table)
        if not files:
            return {}
        try:
            with self._duckdb.connect(database=":memory:") as conn:
                columns = self._table_columns(conn, files)
                if "date" not in columns or "time_index" not in columns:
                    return {}
                frame = conn.execute(
                    """
                    SELECT
                        CAST("date" AS VARCHAR) AS "date",
                        COUNT(DISTINCT TRY_CAST("time_index" AS BIGINT)) AS "time_slots"
                    FROM read_parquet(?, union_by_name=true)
                    WHERE "date" IS NOT NULL
                      AND LOWER(CAST("date" AS VARCHAR)) NOT IN ('', 'nan', 'none', 'unknown')
                      AND TRY_CAST("time_index" AS BIGINT) IS NOT NULL
                    GROUP BY "date"
                    ORDER BY "date"
                    """,
                    [files],
                ).fetchdf()
        except Exception:
            return {}
        if frame.empty:
            return {}
        return {str(row["date"]): int(row["time_slots"]) for _, row in frame.iterrows()}

    def filtered_count(
        self,
        run_id: str,
        table: str,
        *,
        metrics: str | list[str] | None = None,
        epoch_id: int | None = None,
        episode_id: int | None = None,
        date: str | None = None,
        vpp_id: str | None = None,
        agent_id: str | None = None,
        start_time_index: int | None = None,
        end_time_index: int | None = None,
        policy_id: str | list[str] | None = None,
        gradient_step: int | None = None,
        start_gradient_step: int | None = None,
        end_gradient_step: int | None = None,
    ) -> int:
        files = self.table_files(run_id, table, epoch_id=epoch_id, vpp_id=vpp_id)
        if not files:
            return 0
        with self._duckdb.connect(database=":memory:") as conn:
            columns = self._table_columns(conn, files)
            where_sql, params = self._where_clause(
                columns,
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
            value = conn.execute(
                f"SELECT COUNT(*) AS row_count FROM read_parquet(?, union_by_name=true){where_sql}",
                [files, *params],
            ).fetchone()[0]
        return int(value or 0)

    def filtered_metadata(
        self,
        run_id: str,
        table: str,
        *,
        metrics: str | list[str] | None = None,
        epoch_id: int | None = None,
        episode_id: int | None = None,
        date: str | None = None,
        vpp_id: str | None = None,
        agent_id: str | None = None,
        start_time_index: int | None = None,
        end_time_index: int | None = None,
        policy_id: str | list[str] | None = None,
        gradient_step: int | None = None,
        start_gradient_step: int | None = None,
        end_gradient_step: int | None = None,
    ) -> pd.DataFrame:
        files = self.table_files(run_id, table, epoch_id=epoch_id, vpp_id=vpp_id)
        if not files:
            return pd.DataFrame()
        with self._duckdb.connect(database=":memory:") as conn:
            columns = self._table_columns(conn, files)
            if "metric_name" not in columns:
                return pd.DataFrame()
            select_parts = ['"metric_name"']
            if "unit" in columns:
                select_parts.append('ANY_VALUE("unit") AS "unit"')
            if "formula_latex" in columns:
                select_parts.append('ANY_VALUE("formula_latex") AS "formula_latex"')
            where_sql, params = self._where_clause(
                columns,
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
            return conn.execute(
                f"""
                SELECT {", ".join(select_parts)}
                FROM read_parquet(?, union_by_name=true)
                {where_sql}
                GROUP BY "metric_name"
                ORDER BY "metric_name"
                """,
                [files, *params],
            ).fetchdf()

    def filtered_rows(
        self,
        run_id: str,
        table: str,
        *,
        metrics: str | list[str] | None = None,
        epoch_id: int | None = None,
        episode_id: int | None = None,
        date: str | None = None,
        vpp_id: str | None = None,
        agent_id: str | None = None,
        start_time_index: int | None = None,
        end_time_index: int | None = None,
        policy_id: str | list[str] | None = None,
        gradient_step: int | None = None,
        start_gradient_step: int | None = None,
        end_gradient_step: int | None = None,
        columns: list[str] | None = None,
        max_rows: int | None = None,
        sampled: bool = False,
        sample_by: list[str] | None = None,
        order_by: list[str] | None = None,
        order_desc: bool = False,
    ) -> pd.DataFrame:
        files = self.table_files(run_id, table, epoch_id=epoch_id, vpp_id=vpp_id)
        if not files:
            return pd.DataFrame()
        with self._duckdb.connect(database=":memory:") as conn:
            available_columns = self._table_columns(conn, files)
            if columns is None:
                selected_columns = list(available_columns)
            else:
                selected_columns = [column for column in columns if column in available_columns]
            if not selected_columns:
                return pd.DataFrame()
            select_sql = ", ".join(f'"{column}"' for column in selected_columns)
            where_sql, params = self._where_clause(
                available_columns,
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
            requested_order_columns = order_by or ["epoch_id", "episode_id", "date", "time_index", "gradient_step", "vpp_id", "metric_name"]
            order_columns = [column for column in requested_order_columns if column in available_columns]
            order_direction = " DESC" if order_desc else ""
            order_sql = ", ".join(f'"{column}"{order_direction}' for column in order_columns) if order_columns else "1"
            base_sql = f"SELECT {select_sql} FROM read_parquet(?, union_by_name=true){where_sql}"
            query_params: list[Any] = [files, *params]
            if max_rows is not None:
                max_rows = max(1, int(max_rows))
                if sampled:
                    sample_columns = [column for column in (sample_by or []) if column in available_columns]
                    if sample_columns:
                        partition_sql = ", ".join(f'"{column}"' for column in sample_columns)
                        sql = f"""
                        WITH filtered AS (
                            {base_sql}
                        ),
                        grouped AS (
                            SELECT
                                *,
                                DENSE_RANK() OVER (ORDER BY {partition_sql}) AS __group_id
                            FROM filtered
                        ),
                        numbered AS (
                            SELECT
                                *,
                                ROW_NUMBER() OVER (PARTITION BY {partition_sql} ORDER BY {order_sql}) AS __rn,
                                COUNT(*) OVER (PARTITION BY {partition_sql}) AS __group_total,
                                MAX(__group_id) OVER () AS __group_count
                            FROM grouped
                        ),
                        sampled AS (
                            SELECT
                                *,
                                GREATEST(1, CAST(FLOOR(?::DOUBLE / __group_count) AS BIGINT)) AS __per_group_limit
                            FROM numbered
                        )
                        SELECT {select_sql}
                        FROM sampled
                        WHERE ((__rn - 1) % GREATEST(1, CAST(CEIL(__group_total::DOUBLE / __per_group_limit) AS BIGINT))) = 0
                        ORDER BY {order_sql}
                        LIMIT ?
                        """
                        query_params.extend([max_rows, max_rows])
                    else:
                        sql = f"""
                        WITH filtered AS (
                            {base_sql}
                        ),
                        numbered AS (
                            SELECT
                                *,
                                ROW_NUMBER() OVER (ORDER BY {order_sql}) AS __rn,
                                COUNT(*) OVER () AS __total
                            FROM filtered
                        )
                        SELECT {select_sql}
                        FROM numbered
                        WHERE ((__rn - 1) % GREATEST(1, CAST(CEIL(__total::DOUBLE / ?) AS BIGINT))) = 0
                        ORDER BY {order_sql}
                        LIMIT ?
                        """
                        query_params.extend([max_rows, max_rows])
                else:
                    sql = f"{base_sql} ORDER BY {order_sql} LIMIT ?"
                    query_params.append(max_rows)
            else:
                sql = f"{base_sql} ORDER BY {order_sql}"
            return conn.execute(sql, query_params).fetchdf()

    @staticmethod
    def _table_columns(conn: Any, files: list[str]) -> set[str]:
        frame = conn.execute("SELECT * FROM read_parquet(?, union_by_name=true) LIMIT 0", [files]).fetchdf()
        return set(str(column) for column in frame.columns)

    @staticmethod
    def _where_clause(
        columns: set[str],
        *,
        metrics: str | list[str] | None = None,
        epoch_id: int | None = None,
        episode_id: int | None = None,
        date: str | None = None,
        vpp_id: str | None = None,
        agent_id: str | None = None,
        start_time_index: int | None = None,
        end_time_index: int | None = None,
        policy_id: str | list[str] | None = None,
        gradient_step: int | None = None,
        start_gradient_step: int | None = None,
        end_gradient_step: int | None = None,
    ) -> tuple[str, list[Any]]:
        clauses: list[str] = []
        params: list[Any] = []
        metric_names = _as_list(metrics)
        if metric_names and "metric_name" in columns:
            placeholders = ", ".join("?" for _ in metric_names)
            clauses.append(f'"metric_name" IN ({placeholders})')
            params.extend(metric_names)
        if epoch_id is not None and "epoch_id" in columns:
            clauses.append('TRY_CAST("epoch_id" AS BIGINT) = ?')
            params.append(int(epoch_id))
        if episode_id is not None and "episode_id" in columns:
            clauses.append('TRY_CAST("episode_id" AS BIGINT) = ?')
            params.append(int(episode_id))
        if date is not None and "date" in columns:
            clauses.append('CAST("date" AS VARCHAR) = ?')
            params.append(str(date))
        if vpp_id is not None and "vpp_id" in columns:
            clauses.append('CAST("vpp_id" AS VARCHAR) = ?')
            params.append(str(vpp_id))
        if agent_id is not None and "agent_id" in columns:
            clauses.append('CAST("agent_id" AS VARCHAR) = ?')
            params.append(str(agent_id))
        policy_ids = _as_list(policy_id)
        if policy_ids and "policy_id" in columns:
            placeholders = ", ".join("?" for _ in policy_ids)
            clauses.append(f'CAST("policy_id" AS VARCHAR) IN ({placeholders})')
            params.extend(policy_ids)
        if start_time_index is not None and "time_index" in columns:
            clauses.append('TRY_CAST("time_index" AS BIGINT) >= ?')
            params.append(int(start_time_index))
        if end_time_index is not None and "time_index" in columns:
            clauses.append('TRY_CAST("time_index" AS BIGINT) <= ?')
            params.append(int(end_time_index))
        if gradient_step is not None and "gradient_step" in columns:
            clauses.append('TRY_CAST("gradient_step" AS BIGINT) = ?')
            params.append(int(gradient_step))
        if start_gradient_step is not None and "gradient_step" in columns:
            clauses.append('TRY_CAST("gradient_step" AS BIGINT) >= ?')
            params.append(int(start_gradient_step))
        if end_gradient_step is not None and "gradient_step" in columns:
            clauses.append('TRY_CAST("gradient_step" AS BIGINT) <= ?')
            params.append(int(end_gradient_step))
        if not clauses:
            return "", params
        return " WHERE " + " AND ".join(clauses), params

    def read_table(self, run_id: str, table: str) -> pd.DataFrame:
        files = self.table_files(run_id, table)
        if not files:
            return pd.DataFrame()
        with self._duckdb.connect(database=":memory:") as conn:
            return conn.execute(
                "SELECT * FROM read_parquet(?, union_by_name=true)",
                [files],
            ).fetchdf()
