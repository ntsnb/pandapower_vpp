from __future__ import annotations

from datetime import datetime, timezone
import json
import math
from pathlib import Path
from typing import Any

from marl_dashboard.backend.storage.variable_enrichment import (
    default_formula_dictionary,
    enrich_formula_dictionary,
    enrich_variable_dictionary,
)


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def clean_json_value(value: Any) -> Any:
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    if isinstance(value, dict):
        return {str(key): clean_json_value(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [clean_json_value(item) for item in value]
    return value


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(
        json.dumps(clean_json_value(payload), ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    tmp_path.replace(path)


def read_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


class MetadataStore:
    def __init__(self, data_dir: str | Path) -> None:
        self.data_dir = Path(data_dir).expanduser().resolve()

    def run_dir(self, run_id: str) -> Path:
        return self.data_dir / str(run_id)

    def initialize_run(
        self,
        *,
        run_id: str,
        config: dict[str, Any] | None = None,
        variable_dictionary: list[dict[str, Any]] | dict[str, Any] | None = None,
        formulas: dict[str, Any] | list[dict[str, Any]] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        run_dir = self.run_dir(run_id)
        run_dir.mkdir(parents=True, exist_ok=True)
        now = utc_now_iso()
        base_metadata = {
            "run_id": run_id,
            "status": "running",
            "started_at": now,
            "ended_at": None,
            "schema_version": "marl_dashboard.v1",
            "algorithm": (config or {}).get("algorithm"),
            "environment": (config or {}).get("environment"),
            "metadata": metadata or {},
        }
        write_json(run_dir / "metadata.json", base_metadata)
        write_json(run_dir / "config.json", config or {})
        write_json(run_dir / "variable_dictionary.json", variable_dictionary or [])
        write_json(run_dir / "formulas.json", formulas or {})

    def update_status(self, run_id: str, status: str, **extra: Any) -> None:
        path = self.run_dir(run_id) / "metadata.json"
        metadata = read_json(path, {"run_id": run_id, "started_at": None})
        metadata["status"] = status
        metadata.update(extra)
        if status in {"finished", "error", "stopped"}:
            metadata.setdefault("ended_at", utc_now_iso())
            metadata["ended_at"] = metadata["ended_at"] or utc_now_iso()
        write_json(path, metadata)

    def list_runs(self) -> list[dict[str, Any]]:
        if not self.data_dir.exists():
            return []
        rows = []
        for metadata_path in sorted(self.data_dir.glob("*/metadata.json")):
            rows.append(read_json(metadata_path, {}))
        return rows

    def metadata(self, run_id: str) -> dict[str, Any]:
        return read_json(self.run_dir(run_id) / "metadata.json", {})

    def config(self, run_id: str) -> dict[str, Any]:
        return read_json(self.run_dir(run_id) / "config.json", {})

    def variables(self, run_id: str) -> list[dict[str, Any]]:
        payload = read_json(self.run_dir(run_id) / "variable_dictionary.json", [])
        if isinstance(payload, dict):
            return enrich_variable_dictionary(list(payload.values()))
        return enrich_variable_dictionary(list(payload))

    def formulas(self, run_id: str) -> dict[str, Any]:
        payload = read_json(self.run_dir(run_id) / "formulas.json", {})
        formulas = default_formula_dictionary()
        if isinstance(payload, list):
            formulas.update(enrich_formula_dictionary({
                str(item.get("name")): item.get("formula_latex", "")
                for item in payload
                if isinstance(item, dict) and item.get("name")
            }))
            return formulas
        formulas.update(enrich_formula_dictionary(dict(payload)))
        return formulas
