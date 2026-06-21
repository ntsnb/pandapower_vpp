from __future__ import annotations

from importlib.resources import files
from pathlib import Path
from typing import Any

import yaml


def project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def config_registry_path() -> Path:
    return project_root() / "configs" / "registry.yaml"


def _load_config_registry() -> dict[str, str]:
    path = config_registry_path()
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    aliases = data.get("aliases", data)
    if not isinstance(aliases, dict):
        raise ValueError(f"Config registry aliases must be a mapping: {path}")
    return {str(key): str(value) for key, value in aliases.items()}


def resolve_config_path(path: str | Path, *, child_path: Path | None = None) -> Path:
    requested = Path(path)
    if requested.is_absolute():
        if requested.exists():
            return requested.resolve()
        raise FileNotFoundError(f"YAML file not found: {path}")

    candidates: list[Path] = []
    if child_path is not None:
        candidates.append(child_path.parent / requested)
    candidates.append(project_root() / requested)

    registry = _load_config_registry()
    alias = registry.get(str(path)) or registry.get(requested.as_posix()) or registry.get(requested.stem)
    if alias is not None:
        alias_path = Path(alias)
        candidates.append(alias_path if alias_path.is_absolute() else project_root() / alias_path)

    attempts: list[Path] = []
    for candidate in candidates:
        attempts.append(candidate)
        if candidate.exists():
            return candidate.resolve()

    resource = files("vpp_dso_sim.resources.configs").joinpath(requested.name)
    if resource.is_file():
        resource_path = Path(str(resource))
        if resource_path.exists():
            return resource_path.resolve()

    attempted = ", ".join(str(item) for item in attempts)
    raise FileNotFoundError(f"YAML file not found: {path}; attempted: {attempted}")


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if key == "extends":
            continue
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(dict(merged[key]), value)
        else:
            merged[key] = value
    return merged


def _resolve_extends_path(parent: str | Path, *, child_path: Path) -> Path:
    return resolve_config_path(parent, child_path=child_path)


def load_yaml(path: str | Path) -> dict[str, Any]:
    resolved = resolve_config_path(path)
    with resolved.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"YAML root must be a mapping: {resolved}")
    if "extends" in data:
        parent = load_yaml(_resolve_extends_path(data["extends"], child_path=resolved))
        data = _deep_merge(parent, data)
    return data


def resolve_project_path(path: str | Path) -> Path:
    resolved = Path(path)
    if resolved.is_absolute():
        return resolved
    return project_root() / resolved
