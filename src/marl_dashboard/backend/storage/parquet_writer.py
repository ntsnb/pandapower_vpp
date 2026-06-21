from __future__ import annotations

from datetime import datetime, timezone
import math
from pathlib import Path
from typing import Any

import pandas as pd


def _clean_scalar(value: Any) -> Any:
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    if isinstance(value, dict | list | tuple):
        return str(value)
    return value


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


class ParquetWriter:
    def __init__(self, data_dir: str | Path, run_id: str) -> None:
        self.data_dir = Path(data_dir).expanduser().resolve()
        self.run_id = str(run_id)
        self._parts: dict[tuple[str, str, str], int] = {}

    @property
    def run_dir(self) -> Path:
        return self.data_dir / self.run_id

    def write_records(self, table: str, records: list[dict[str, Any]]) -> list[Path]:
        if not records:
            return []
        grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
        for record in records:
            epoch_part = _epoch_partition(record.get("epoch_id"))
            vpp_part = _safe_partition("vpp_id", record.get("vpp_id"))
            grouped.setdefault((epoch_part, vpp_part), []).append(
                {str(key): _clean_scalar(value) for key, value in record.items()}
            )

        written: list[Path] = []
        for (epoch_part, vpp_part), rows in grouped.items():
            partition_dir = self.run_dir / "tables" / table / epoch_part / vpp_part
            partition_dir.mkdir(parents=True, exist_ok=True)
            part_key = (table, epoch_part, vpp_part)
            if part_key not in self._parts:
                self._parts[part_key] = _next_part_index(partition_dir)
            part_index = self._parts[part_key]
            self._parts[part_key] += 1
            path = partition_dir / f"part-{part_index:06d}.parquet"
            frame = pd.DataFrame(rows)
            frame.to_parquet(path, index=False)
            written.append(path)
        return written


def effective_partition_files(partition_dir: Path) -> list[Path]:
    if not partition_dir.exists():
        return []
    compact_files = sorted(partition_dir.glob("compact-*.parquet"))
    part_files = sorted(partition_dir.glob("part-*.parquet"))
    if not compact_files:
        return part_files
    latest_compact = max(compact_files, key=lambda path: (path.stat().st_mtime_ns, path.name))
    latest_mtime = latest_compact.stat().st_mtime_ns
    newer_parts = [path for path in part_files if path.stat().st_mtime_ns > latest_mtime]
    return [latest_compact, *newer_parts]


def compact_partition(partition_dir: Path, *, min_part_files: int = 64, archive_inputs: bool = True) -> Path | None:
    partition_dir = Path(partition_dir).expanduser().resolve()
    if not partition_dir.exists():
        return None
    part_files = sorted(partition_dir.glob("part-*.parquet"))
    input_files = effective_partition_files(partition_dir)
    compact_files = [path for path in input_files if path.name.startswith("compact-")]
    new_part_files = [path for path in input_files if path.name.startswith("part-")]
    if compact_files:
        if len(new_part_files) < max(1, int(min_part_files)):
            return None
    elif len(part_files) < max(1, int(min_part_files)):
        return None
    if not input_files:
        return None
    frames = [pd.read_parquet(path) for path in input_files if path.exists()]
    if not frames:
        return None
    frame = pd.concat(frames, ignore_index=True, sort=False)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
    compact_path = partition_dir / f"compact-{timestamp}.parquet"
    tmp_path = partition_dir / f".{compact_path.stem}.tmp.parquet"
    frame.to_parquet(tmp_path, index=False)
    tmp_path.replace(compact_path)

    if archive_inputs:
        archive_dir = partition_dir / "_compacted_parts"
        archive_dir.mkdir(exist_ok=True)
        for path in input_files:
            if not path.exists() or path == compact_path:
                continue
            target = archive_dir / path.name
            if target.exists():
                target = archive_dir / f"{timestamp}-{path.name}"
            path.replace(target)
    return compact_path


def _next_part_index(partition_dir: Path) -> int:
    max_index = -1
    for path in partition_dir.glob("part-*.parquet"):
        try:
            max_index = max(max_index, int(path.stem.split("-", 1)[1]))
        except (IndexError, ValueError):
            continue
    return max_index + 1
