from __future__ import annotations

from pathlib import Path
from typing import Any
from importlib.resources import files

import yaml


def project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def load_yaml(path: str | Path) -> dict[str, Any]:
    resolved = Path(path)
    if not resolved.is_absolute():
        resolved = project_root() / resolved
    if resolved.exists():
        with resolved.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
    else:
        resource = files("vpp_dso_sim.resources.configs").joinpath(resolved.name)
        if not resource.is_file():
            raise FileNotFoundError(f"YAML file not found: {path}")
        data = yaml.safe_load(resource.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise ValueError(f"YAML root must be a mapping: {resolved}")
    return data


def resolve_project_path(path: str | Path) -> Path:
    resolved = Path(path)
    if resolved.is_absolute():
        return resolved
    return project_root() / resolved
