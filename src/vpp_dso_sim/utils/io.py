from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


def ensure_dir(path: str | Path) -> Path:
    resolved = Path(path)
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved


def write_json(path: str | Path, data: dict[str, Any]) -> None:
    resolved = Path(path)
    ensure_dir(resolved.parent)
    with resolved.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def write_dataframe(path: str | Path, frame: pd.DataFrame) -> None:
    resolved = Path(path)
    ensure_dir(resolved.parent)
    frame.to_csv(resolved, index=True)

