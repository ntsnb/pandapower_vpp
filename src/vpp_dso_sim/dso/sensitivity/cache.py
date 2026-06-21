from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from vpp_dso_sim.dso.envelope.schemas import ActionUnitState, NetworkObjectState, SensitivityEdgeTensor


@dataclass
class SensitivityCacheEntry:
    tensor: SensitivityEdgeTensor
    step: int


@dataclass(frozen=True)
class SensitivityRefreshDecision:
    """Decision for refreshing cached ActionUnit x NetworkObject sensitivities."""

    refresh: bool
    reasons: tuple[str, ...]
    priority_action_unit_ids: tuple[str, ...]


class SensitivityCache:
    """Small step-indexed cache for raw sensitivity tensors."""

    def __init__(self, ttl_steps: int = 8, confidence_decay: float = 0.98) -> None:
        self.ttl_steps = int(ttl_steps)
        self.confidence_decay = float(confidence_decay)
        self._entry: SensitivityCacheEntry | None = None

    def set(self, tensor: SensitivityEdgeTensor, step: int) -> None:
        self._entry = SensitivityCacheEntry(tensor=tensor, step=int(step))

    def peek(self) -> SensitivityCacheEntry | None:
        """Return the raw cache entry without applying TTL policy."""

        return self._entry

    def get(self, step: int) -> SensitivityEdgeTensor | None:
        if self._entry is None:
            return None
        if int(step) - self._entry.step > self.ttl_steps:
            return None
        return self._entry.tensor


def grid_state_snapshot(grid_state: dict[str, Any] | None) -> dict[str, float]:
    """Extract cache-relevant grid-state scalars used by refresh triggers."""

    state = dict(grid_state or {})
    return {
        "min_vm_pu": float(state.get("min_vm_pu", 1.0)),
        "max_vm_pu": float(state.get("max_vm_pu", 1.0)),
        "max_line_loading_percent": float(state.get("max_line_loading_percent", 0.0)),
        "max_trafo_loading_percent": float(state.get("max_trafo_loading_percent", 0.0)),
    }


def sensitivity_cache_metadata(
    *,
    step: int,
    action_units: Sequence[ActionUnitState],
    network_objects: Sequence[NetworkObjectState],
    grid_state: dict[str, Any] | None,
    refresh_reasons: Sequence[str] = (),
    priority_action_unit_ids: Sequence[str] = (),
) -> dict[str, object]:
    """Build metadata needed to decide future cache reuse and refresh priority."""

    return {
        "cache_step": int(step),
        "grid_state_snapshot": grid_state_snapshot(grid_state),
        "fr_width_by_action_unit": {
            unit.id.action_unit_id: float(unit.hard_width_mw())
            for unit in action_units
        },
        "projection_gap_by_action_unit": {
            unit.id.action_unit_id: float(abs(unit.projection_gap_hist_mw))
            for unit in action_units
        },
        "network_object_ids": tuple(obj.id.object_id for obj in network_objects),
        "refresh_reasons": tuple(str(reason) for reason in refresh_reasons),
        "priority_action_unit_ids": tuple(str(action_id) for action_id in priority_action_unit_ids),
    }


def _cfg_int(config: dict[str, Any], key: str, default: int) -> int:
    value = config.get(key, default)
    return int(value) if value is not None else int(default)


def _cfg_float(config: dict[str, Any], key: str, default: float) -> float:
    value = config.get(key, default)
    return float(value) if value is not None else float(default)


def _cached_step(raw: SensitivityEdgeTensor) -> int:
    return int(raw.metadata.get("cache_step", 0))


def _cached_widths(raw: SensitivityEdgeTensor) -> dict[str, float]:
    payload = raw.metadata.get("fr_width_by_action_unit", {})
    if not isinstance(payload, dict):
        return {}
    return {str(key): float(value) for key, value in payload.items()}


def _cached_grid_snapshot(raw: SensitivityEdgeTensor) -> dict[str, float]:
    payload = raw.metadata.get("grid_state_snapshot", {})
    if not isinstance(payload, dict):
        return {}
    return {str(key): float(value) for key, value in payload.items()}


def _confidence_by_action_unit(raw: SensitivityEdgeTensor) -> dict[str, float]:
    if "sensitivity_confidence" not in raw.channel_names:
        return {}
    confidence_idx = raw.channel_names.index("sensitivity_confidence")
    result: dict[str, float] = {}
    for action_index, action_id in enumerate(raw.action_unit_ids):
        valid = raw.edge_valid_mask[:, action_index]
        if not bool(valid.any()):
            result[str(action_id)] = 0.0
            continue
        result[str(action_id)] = float(raw.values[:, action_index, confidence_idx][valid].mean())
    return result


def _append_reason(reasons: list[str], reason: str) -> None:
    if reason not in reasons:
        reasons.append(reason)


def decide_sensitivity_refresh(
    raw: SensitivityEdgeTensor | None,
    *,
    step: int,
    action_units: Sequence[ActionUnitState],
    network_object_ids: Sequence[str],
    grid_state: dict[str, Any] | None,
    sensitivity_cfg: dict[str, Any] | None = None,
) -> SensitivityRefreshDecision:
    """Decide whether raw sensitivities should be refreshed.

    Refresh triggers follow the project prompt:
    update period, TTL, voltage/loading drift, FR width drift, projection gap
    history, missing ActionUnits, and missing NetworkObjects. Priority ActionUnits
    are capped by `max_perturbed_action_units_per_update` and sorted by missing
    status, projection gap, FR width change, low confidence, and headroom.
    """

    cfg = dict(sensitivity_cfg or {})
    max_priority = max(1, _cfg_int(cfg, "max_perturbed_action_units_per_update", max(1, len(action_units))))
    action_ids = [unit.id.action_unit_id for unit in action_units]
    if raw is None:
        priority = sorted(action_units, key=lambda unit: (-unit.hard_width_mw(), unit.id.action_unit_id))
        return SensitivityRefreshDecision(
            refresh=True,
            reasons=("cache_empty",),
            priority_action_unit_ids=tuple(unit.id.action_unit_id for unit in priority[:max_priority]),
        )

    reasons: list[str] = []
    cached_step = _cached_step(raw)
    age = int(step) - cached_step
    update_period = _cfg_int(cfg, "update_period_steps", 4)
    ttl_steps = _cfg_int(cfg, "cache_ttl_steps", 8)
    if ttl_steps >= 0 and age > ttl_steps:
        _append_reason(reasons, "cache_ttl_expired")
    if update_period > 0 and age >= update_period:
        _append_reason(reasons, "update_period_elapsed")

    missing_action_units = [action_id for action_id in action_ids if action_id not in set(raw.action_unit_ids)]
    missing_objects = [object_id for object_id in network_object_ids if object_id not in set(raw.network_object_ids)]
    if missing_action_units:
        _append_reason(reasons, "missing_action_units")
    if missing_objects:
        _append_reason(reasons, "missing_network_objects")

    current_snapshot = grid_state_snapshot(grid_state)
    cached_snapshot = _cached_grid_snapshot(raw)
    voltage_threshold = _cfg_float(cfg, "refresh_if_voltage_delta_pu_gt", 0.005)
    loading_threshold = _cfg_float(cfg, "refresh_if_loading_delta_pct_gt", 5.0)
    if cached_snapshot:
        voltage_delta = max(
            abs(current_snapshot["min_vm_pu"] - cached_snapshot.get("min_vm_pu", current_snapshot["min_vm_pu"])),
            abs(current_snapshot["max_vm_pu"] - cached_snapshot.get("max_vm_pu", current_snapshot["max_vm_pu"])),
        )
        loading_delta = max(
            abs(
                current_snapshot["max_line_loading_percent"]
                - cached_snapshot.get("max_line_loading_percent", current_snapshot["max_line_loading_percent"])
            ),
            abs(
                current_snapshot["max_trafo_loading_percent"]
                - cached_snapshot.get("max_trafo_loading_percent", current_snapshot["max_trafo_loading_percent"])
            ),
        )
        if voltage_delta > voltage_threshold:
            _append_reason(reasons, "voltage_delta")
        if loading_delta > loading_threshold:
            _append_reason(reasons, "loading_delta")

    cached_widths = _cached_widths(raw)
    width_threshold = _cfg_float(cfg, "refresh_if_fr_width_change_ratio_gt", 0.20)
    gap_threshold = _cfg_float(cfg, "refresh_if_projection_gap_hist_gt_mw", 0.10)
    confidence_threshold = _cfg_float(cfg, "refresh_if_confidence_lt", 0.0)
    confidence_by_action = _confidence_by_action_unit(raw)
    scored_units: list[tuple[float, str]] = []
    width_changed = False
    gap_triggered = False
    low_confidence = False
    for unit in action_units:
        action_id = unit.id.action_unit_id
        width = float(unit.hard_width_mw())
        cached_width = cached_widths.get(action_id)
        width_ratio = 0.0
        if cached_width is not None:
            denominator = max(abs(cached_width), 1e-9)
            width_ratio = abs(width - cached_width) / denominator
            width_changed = width_changed or width_ratio > width_threshold
        projection_gap = abs(float(unit.projection_gap_hist_mw))
        gap_triggered = gap_triggered or projection_gap > gap_threshold
        confidence = confidence_by_action.get(action_id, 1.0)
        low_confidence = low_confidence or confidence < confidence_threshold
        missing_score = 10000.0 if action_id in missing_action_units else 0.0
        gap_score = 1000.0 * projection_gap if projection_gap > gap_threshold else 0.0
        width_score = 500.0 * width_ratio if width_ratio > width_threshold else 0.0
        confidence_score = 200.0 * (confidence_threshold - confidence) if confidence < confidence_threshold else 0.0
        headroom_score = width
        scored_units.append((missing_score + gap_score + width_score + confidence_score + headroom_score, action_id))
    if width_changed:
        _append_reason(reasons, "fr_width_change")
    if gap_triggered:
        _append_reason(reasons, "projection_gap_hist")
    if low_confidence:
        _append_reason(reasons, "low_confidence")

    if not reasons:
        return SensitivityRefreshDecision(refresh=False, reasons=(), priority_action_unit_ids=())

    scored_units.sort(key=lambda item: (-item[0], item[1]))
    priority_ids = tuple(action_id for _, action_id in scored_units[:max_priority])
    return SensitivityRefreshDecision(
        refresh=True,
        reasons=tuple(reasons),
        priority_action_unit_ids=priority_ids,
    )


def active_sensitivity_slice(
    raw: SensitivityEdgeTensor,
    *,
    network_object_ids: list[str],
    action_unit_ids: list[str],
) -> SensitivityEdgeTensor:
    """Return active `M_raw[K_t, A_t, :]` slice without re-estimating physics."""

    object_lookup = {object_id: index for index, object_id in enumerate(raw.network_object_ids)}
    action_lookup = {action_id: index for index, action_id in enumerate(raw.action_unit_ids)}
    object_indices = [object_lookup[object_id] for object_id in network_object_ids if object_id in object_lookup]
    action_indices = [action_lookup[action_id] for action_id in action_unit_ids if action_id in action_lookup]
    values = raw.values[object_indices, :, :][:, action_indices, :]
    mask = raw.edge_valid_mask[object_indices, :][:, action_indices]
    return SensitivityEdgeTensor(
        values=values,
        channel_names=raw.channel_names,
        edge_valid_mask=mask,
        q_channel_mask=raw.q_channel_mask,
        action_unit_ids=tuple(raw.action_unit_ids[index] for index in action_indices),
        network_object_ids=tuple(raw.network_object_ids[index] for index in object_indices),
        metadata={**raw.metadata, "active_slice": True, "source": "raw_sensitivity_cache"},
    )


def merge_sensitivity_update(
    raw: SensitivityEdgeTensor,
    update: SensitivityEdgeTensor,
    *,
    metadata: dict[str, object] | None = None,
) -> SensitivityEdgeTensor:
    """Merge a priority ActionUnit refresh back into the raw sensitivity cache."""

    if raw.channel_names != update.channel_names:
        raise ValueError("Cannot merge sensitivity tensors with different channel schemas.")
    action_ids = list(raw.action_unit_ids)
    for action_id in update.action_unit_ids:
        if action_id not in action_ids:
            action_ids.append(action_id)
    object_ids = list(raw.network_object_ids)
    for object_id in update.network_object_ids:
        if object_id not in object_ids:
            object_ids.append(object_id)
    values = np.zeros((len(object_ids), len(action_ids), len(raw.channel_names)), dtype=np.float32)
    mask = np.zeros((len(object_ids), len(action_ids)), dtype=bool)
    object_lookup = {object_id: idx for idx, object_id in enumerate(object_ids)}
    action_lookup = {action_id: idx for idx, action_id in enumerate(action_ids)}

    def place(tensor: SensitivityEdgeTensor) -> None:
        for source_object_idx, object_id in enumerate(tensor.network_object_ids):
            target_object_idx = object_lookup[object_id]
            for source_action_idx, action_id in enumerate(tensor.action_unit_ids):
                target_action_idx = action_lookup[action_id]
                values[target_object_idx, target_action_idx, :] = tensor.values[source_object_idx, source_action_idx, :]
                mask[target_object_idx, target_action_idx] = bool(tensor.edge_valid_mask[source_object_idx, source_action_idx])

    place(raw)
    place(update)
    return SensitivityEdgeTensor(
        values=values,
        channel_names=raw.channel_names,
        edge_valid_mask=mask,
        q_channel_mask=bool(raw.q_channel_mask and update.q_channel_mask),
        action_unit_ids=tuple(action_ids),
        network_object_ids=tuple(object_ids),
        metadata={
            **raw.metadata,
            **update.metadata,
            **(metadata or {}),
            "partial_refresh_action_unit_ids": tuple(update.action_unit_ids),
        },
    )
