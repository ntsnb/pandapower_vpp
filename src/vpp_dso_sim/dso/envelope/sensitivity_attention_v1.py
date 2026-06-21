from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from vpp_dso_sim.dso.envelope.safe_decoder import decode_operating_envelope
from vpp_dso_sim.dso.models.bipartite_attention_actor import BipartiteSensitivityDSOActor
from vpp_dso_sim.dso.observation.structured_bipartite import encode_dso_observation_structured
from vpp_dso_sim.dso.sensitivity.cache import (
    SensitivityCache,
    active_sensitivity_slice,
    decide_sensitivity_refresh,
    merge_sensitivity_update,
    sensitivity_cache_metadata,
)
from vpp_dso_sim.dso.sensitivity.finite_difference import compute_finite_difference_sensitivity_tensor
from vpp_dso_sim.dso.sensitivity.selectors import build_action_units, select_critical_network_objects
from vpp_dso_sim.optimization.feasibility_region import current_power_by_fr_scope, scalar_target_to_vector_targets


def _clip01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _direction_index_from_service(service_request: str) -> int:
    text = str(service_request).lower()
    if "absorb" in text or "charge" in text:
        return 0
    if "export" in text or "inject" in text or "reduce" in text:
        return 2
    return 1


def _rule_targets_for_action_units(rule_envelope: dict[str, Any], action_units) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    p_min = float(rule_envelope.get("p_min_mw", 0.0))
    p_max = float(rule_envelope.get("p_max_mw", p_min))
    hard_width = max(1e-9, p_max - p_min)
    center = _clip01((float(rule_envelope.get("preferred_target_p_mw", p_min)) - p_min) / hard_width)
    width = _clip01(
        (
            float(rule_envelope.get("preferred_p_max_mw", p_max))
            - float(rule_envelope.get("preferred_p_min_mw", p_min))
        )
        / hard_width
    )
    direction_index = _direction_index_from_service(str(rule_envelope.get("service_request", "")))
    direction = np.zeros((len(action_units), 3), dtype=np.float32)
    if len(action_units):
        direction[:, direction_index] = 1.0
    return (
        np.full((len(action_units),), center, dtype=np.float32),
        np.full((len(action_units),), width, dtype=np.float32),
        direction,
        np.ones((len(action_units),), dtype=np.float32),
    )


def _normalize_direction_rows(values: np.ndarray) -> np.ndarray:
    rows = np.asarray(values, dtype=np.float32)
    if rows.ndim == 1:
        rows = rows.reshape(-1, 3) if rows.size % 3 == 0 else rows.reshape(1, -1)
    if rows.shape[-1] != 3:
        fallback = np.zeros((rows.shape[0], 3), dtype=np.float32)
        fallback[:, 1] = 1.0
        return fallback
    rows = np.clip(rows, 0.0, None)
    denominator = np.maximum(rows.sum(axis=1, keepdims=True), 1e-9)
    return rows / denominator


def _softmax_rows(logits: np.ndarray) -> np.ndarray:
    rows = np.asarray(logits, dtype=np.float32)
    if rows.ndim == 1:
        rows = rows.reshape(-1, 3) if rows.size % 3 == 0 else rows.reshape(1, -1)
    if rows.shape[-1] != 3:
        fallback = np.zeros((rows.shape[0], 3), dtype=np.float32)
        fallback[:, 1] = 1.0
        return fallback
    shifted = rows - rows.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / np.maximum(exp.sum(axis=1, keepdims=True), 1e-9)


def _select_override_array(
    actor_override: dict[str, Any],
    key: str,
    action_units,
    default: np.ndarray,
) -> np.ndarray:
    default_array = np.asarray(default, dtype=np.float32)
    raw = actor_override.get(key)
    if raw is None:
        return default_array.copy()

    unit_ids = [unit.id.action_unit_id for unit in action_units]
    if isinstance(raw, dict):
        selected = [
            raw.get(unit_id, raw.get(str(index), default_array[index]))
            for index, unit_id in enumerate(unit_ids)
        ]
        return np.asarray(selected, dtype=np.float32)

    values = np.asarray(raw, dtype=np.float32)
    if default_array.ndim == 2 and values.ndim == 1 and values.size % default_array.shape[1] == 0:
        values = values.reshape(-1, default_array.shape[1])
    payload_ids = [str(value) for value in actor_override.get("action_unit_ids", [])]
    if payload_ids and values.shape[0] == len(payload_ids):
        by_id = {unit_id: values[index] for index, unit_id in enumerate(payload_ids)}
        selected = [
            by_id.get(unit_id, default_array[index])
            for index, unit_id in enumerate(unit_ids)
        ]
        return np.asarray(selected, dtype=np.float32)

    result = default_array.copy()
    take = min(result.shape[0], values.shape[0])
    if take:
        result[:take] = values[:take]
    return result


def _legacy_targets_to_action_unit_arrays(
    actor_override: dict[str, Any],
    *,
    vpp,
    fr,
    action_units,
    min_width_ratio: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None:
    targets = actor_override.get("legacy_targets_by_vpp")
    if not isinstance(targets, dict) or str(vpp.id) not in targets:
        return None
    aggregate_target = float(targets[str(vpp.id)])
    vector_targets = scalar_target_to_vector_targets(vpp, fr, aggregate_target)
    center: list[float] = []
    width: list[float] = []
    strength: list[float] = []
    directions: list[list[float]] = []
    current_by_scope = current_power_by_fr_scope(vpp, fr)
    for unit in action_units:
        if fr.scope == "bus_vector":
            scope_key = f"bus_{int(unit.id.bus_id)}"
        elif unit.id.pcc_id:
            scope_key = str(unit.id.pcc_id)
        else:
            scope_key = str(unit.id.action_unit_id)
        target = float(vector_targets.get(scope_key, aggregate_target / max(1, len(action_units))))
        hard_width = max(1e-9, float(unit.p_max_mw) - float(unit.p_min_mw))
        center.append(_clip01((target - float(unit.p_min_mw)) / hard_width))
        width.append(_clip01(float(min_width_ratio)))
        strength.append(1.0)
        current = float(current_by_scope.get(scope_key, unit.p_cur_mw))
        if target < current - 1e-9:
            directions.append([1.0, 0.0, 0.0])
        elif target > current + 1e-9:
            directions.append([0.0, 0.0, 1.0])
        else:
            directions.append([0.0, 1.0, 0.0])
    return (
        np.asarray(center, dtype=np.float32),
        np.asarray(width, dtype=np.float32),
        np.asarray(directions, dtype=np.float32),
        np.asarray(strength, dtype=np.float32),
    )


class SensitivityAttentionEnvelopePolicy:
    """Structured DSO envelope policy descriptor.

    The trainable actor lives in `dso.models.bipartite_attention_actor`; this class
    is the routing object used by config-level policy switches.
    """

    policy_name = "sensitivity_attention_v1"

    def __init__(self, config: dict | None = None) -> None:
        self.config = dict(config or {})
        self._actor = None
        self._sensitivity_cache: SensitivityCache | None = None
        self._actor_checkpoint_loaded = False
        self._actor_checkpoint_path: str | None = None
        self._actor_checkpoint_source = "random_initialization"

    def _dso_cfg(self) -> dict[str, Any]:
        return dict(self.config.get("dso", self.config))

    def _selector_cfg(self) -> dict[str, Any]:
        return dict(self.config.get("selector", self._dso_cfg().get("selector", {})))

    def _sensitivity_cfg(self) -> dict[str, Any]:
        return dict(self.config.get("sensitivity", self._dso_cfg().get("sensitivity", {})))

    def _actor_cfg(self) -> dict[str, Any]:
        return dict(self._dso_cfg().get("actor", {}))

    def _residual_schedule_eta(self, dso_cfg: dict[str, Any], step: int) -> tuple[float, int]:
        if not bool(dso_cfg.get("enable_rule_warmstart", False)):
            return 1.0, int(step)
        warmstart_steps = max(0, int(dso_cfg.get("warmstart_steps", 0)))
        residual_steps = max(0, int(dso_cfg.get("residual_schedule_steps", 0)))
        explicit_progress = (
            dso_cfg.get("residual_progress_step")
            if "residual_progress_step" in dso_cfg
            else dso_cfg.get("training_step", dso_cfg.get("global_step"))
        )
        if explicit_progress is None:
            progress_step = (
                warmstart_steps + residual_steps
                if self._actor_checkpoint_loaded
                else int(step)
            )
        else:
            progress_step = int(explicit_progress)
        if progress_step < warmstart_steps:
            return 0.0, progress_step
        if residual_steps <= 0:
            return 1.0, progress_step
        eta = (progress_step - warmstart_steps) / float(residual_steps)
        return _clip01(eta), progress_step

    def _actor_checkpoint_path_from_config(self) -> str | None:
        actor_cfg = self._actor_cfg()
        dso_cfg = self._dso_cfg()
        raw_path = actor_cfg.get("checkpoint_path", dso_cfg.get("actor_checkpoint_path"))
        return str(raw_path) if raw_path else None

    def _cache_for(self, sensitivity_cfg: dict[str, Any]) -> SensitivityCache:
        ttl_steps = int(sensitivity_cfg.get("cache_ttl_steps", 8))
        confidence_decay = float(sensitivity_cfg.get("confidence_decay", 0.98))
        if (
            self._sensitivity_cache is None
            or self._sensitivity_cache.ttl_steps != ttl_steps
            or self._sensitivity_cache.confidence_decay != confidence_decay
        ):
            self._sensitivity_cache = SensitivityCache(
                ttl_steps=ttl_steps,
                confidence_decay=confidence_decay,
            )
        return self._sensitivity_cache

    def _projection_gap_history_by_scope(
        self,
        simulator,
        vpp,
        step: int,
        sensitivity_cfg: dict[str, Any],
    ) -> dict[str, float]:
        lookback_steps = max(1, int(sensitivity_cfg.get("projection_gap_history_lookback_steps", 16)))
        lower_step = int(step) - lookback_steps
        result: dict[str, float] = {}
        for row in getattr(simulator, "records", {}).get("projection_trace", []):
            if str(row.get("vpp_id")) != str(vpp.id):
                continue
            row_step = int(row.get("step", -1))
            if row_step >= int(step) or row_step < lower_step:
                continue
            gap = abs(float(row.get("delta_p_mw", 0.0)))
            if gap <= 1e-12:
                continue
            scope_type = str(row.get("scope_type", ""))
            scope_id = str(row.get("scope_id", ""))
            if scope_type == "bus":
                key = f"bus_{scope_id}"
            elif scope_type == "pcc":
                key = f"pcc_{scope_id}"
            elif scope_type == "der":
                key = scope_id
            else:
                key = f"pcc_{int(vpp.pcc_bus)}"
            result[key] = max(result.get(key, 0.0), gap)
        return result

    def _actor_for(self, observation) -> BipartiteSensitivityDSOActor:
        import torch

        actor_cfg = self._actor_cfg()
        d_model = int(actor_cfg.get("d_model", 64))
        torch.manual_seed(int(actor_cfg.get("init_seed", self._dso_cfg().get("seed", 0))))
        if self._actor is None:
            self._actor = BipartiteSensitivityDSOActor(
                global_feature_dim=int(observation.global_features.shape[-1]),
                action_token_dim=int(observation.action_tokens.shape[-1]),
                object_token_dim=int(observation.object_tokens.shape[-1]),
                edge_feature_dim=int(observation.sensitivity_edges.shape[-1]),
                d_model=d_model,
                num_heads=int(actor_cfg.get("num_heads", 4)),
                num_layers=int(actor_cfg.get("num_layers", 1)),
                action_self_attention_layers=int(actor_cfg.get("action_self_attention_layers", 1)),
                dropout=float(actor_cfg.get("dropout", 0.0)),
                min_width_ratio=float(actor_cfg.get("min_width_ratio", 0.10)),
                max_width_ratio=float(actor_cfg.get("max_width_ratio", 1.00)),
            )
            checkpoint_path = self._actor_checkpoint_path_from_config()
            if checkpoint_path:
                self._load_actor_checkpoint(self._actor, checkpoint_path, torch=torch)
            self._actor.eval()
        return self._actor

    def _load_actor_checkpoint(self, actor: BipartiteSensitivityDSOActor, checkpoint_path: str, *, torch) -> None:
        path = Path(checkpoint_path)
        checkpoint = torch.load(path, map_location="cpu")
        state_payload = checkpoint.get("actor_state_dict", checkpoint.get("state_dict", checkpoint))
        if not isinstance(state_payload, dict):
            raise ValueError(f"DSO actor checkpoint at {path} does not contain a state dict.")
        actor_keys = set(actor.state_dict().keys())
        extracted: dict[str, Any] = {}
        prefixes = (
            "dso_actor.attention_actor.",
            "attention_actor.",
            "dso_actor.",
            "",
        )
        for key, value in state_payload.items():
            for prefix in prefixes:
                if not str(key).startswith(prefix):
                    continue
                candidate = str(key)[len(prefix) :]
                if candidate in actor_keys:
                    extracted[candidate] = value
                    break
        if not extracted:
            raise ValueError(
                "No BipartiteSensitivityDSOActor weights found in checkpoint. "
                "Expected keys like 'dso_actor.attention_actor.center_head.weight'."
            )
        actor.load_state_dict(extracted, strict=True)
        self._actor_checkpoint_loaded = True
        self._actor_checkpoint_path = str(path)
        self._actor_checkpoint_source = str(
            checkpoint.get("dso_actor_type", checkpoint.get("algorithm", "checkpoint_actor_state_dict"))
        )

    def build(
        self,
        simulator,
        vpp,
        step: int,
        bid: dict[str, Any],
        fr,
        price: float,
        *,
        grid_state: dict[str, Any] | None = None,
        actor_override: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        import torch

        dso_cfg = self._dso_cfg()
        selector_cfg = self._selector_cfg()
        sensitivity_cfg = self._sensitivity_cfg()
        projection_gap_hist_by_scope = self._projection_gap_history_by_scope(simulator, vpp, step, sensitivity_cfg)
        action_units = build_action_units(
            vpp,
            fr,
            t=step,
            granularity=str(dso_cfg.get("action_unit_granularity", "vpp_bus")),
            projection_gap_hist_by_scope=projection_gap_hist_by_scope,
        )
        network_objects = select_critical_network_objects(
            simulator.scenario.net,
            voltage_limits=simulator.scenario.dso.voltage_limits,
            line_loading_limit_percent=simulator.scenario.dso.line_loading_limit_percent,
            trafo_loading_limit_percent=simulator.scenario.dso.trafo_loading_limit_percent,
            topk_low_voltage_buses=int(selector_cfg.get("topk_low_voltage_buses", 5)),
            topk_high_voltage_buses=int(selector_cfg.get("topk_high_voltage_buses", 5)),
            topk_lines=int(selector_cfg.get("topk_lines", 5)),
            topk_trafos=int(selector_cfg.get("topk_trafos", 3)),
        )
        action_unit_ids = [unit.id.action_unit_id for unit in action_units]
        selected_objects = [obj.id.object_id for obj in network_objects]
        cache_hit = False
        sensitivity_source = "finite_difference_recomputed"
        sensitivity_cache_step = None
        sensitivity_refresh_reasons: tuple[str, ...] = ()
        sensitivity_priority_action_units: tuple[str, ...] = ()
        cached_for_partial_refresh = None
        sensitivity = None
        if bool(sensitivity_cfg.get("cache_enabled", True)):
            cache = self._cache_for(sensitivity_cfg)
            cached_entry = cache.peek()
            cached = cached_entry.tensor if cached_entry is not None else None
            refresh_decision = decide_sensitivity_refresh(
                cached,
                step=step,
                action_units=action_units,
                network_object_ids=selected_objects,
                grid_state=grid_state,
                sensitivity_cfg=sensitivity_cfg,
            )
            sensitivity_refresh_reasons = refresh_decision.reasons
            sensitivity_priority_action_units = refresh_decision.priority_action_unit_ids
            if cached is not None and not refresh_decision.refresh:
                sensitivity = active_sensitivity_slice(
                    cached,
                    network_object_ids=selected_objects,
                    action_unit_ids=action_unit_ids,
                )
                cache_hit = True
                sensitivity_source = str(sensitivity.metadata.get("source", "raw_sensitivity_cache"))
                sensitivity_cache_step = int(cached.metadata.get("cache_step", step))
            elif cached is not None and set(action_unit_ids).issubset(cached.action_unit_ids) and set(selected_objects).issubset(cached.network_object_ids):
                cached_for_partial_refresh = cached
        else:
            sensitivity_refresh_reasons = ("cache_disabled",)
        if sensitivity is None:
            refresh_units = list(action_units)
            use_partial_refresh = False
            if cached_for_partial_refresh is not None and sensitivity_priority_action_units:
                priority_set = set(sensitivity_priority_action_units)
                selected_units = [unit for unit in action_units if unit.id.action_unit_id in priority_set]
                if selected_units and len(selected_units) < len(action_units):
                    refresh_units = selected_units
                    use_partial_refresh = True
            sensitivity = compute_finite_difference_sensitivity_tensor(
                simulator.scenario.net,
                refresh_units,
                network_objects,
                enable_q_channels=bool(dso_cfg.get("enable_q_channels", sensitivity_cfg.get("enable_q_channels", False))),
                epsilon_p_min_mw=float(sensitivity_cfg.get("epsilon_p_min_mw", 0.005)),
                epsilon_p_max_mw=float(sensitivity_cfg.get("epsilon_p_max_mw", 0.05)),
                epsilon_p_width_ratio=float(sensitivity_cfg.get("epsilon_p_width_ratio", 0.02)),
            )
            metadata = sensitivity_cache_metadata(
                step=step,
                action_units=action_units,
                network_objects=network_objects,
                grid_state=grid_state,
                refresh_reasons=sensitivity_refresh_reasons,
                priority_action_unit_ids=sensitivity_priority_action_units,
            )
            metadata = {
                **metadata,
                "source": "finite_difference_recomputed",
                "partial_priority_refresh": bool(use_partial_refresh),
            }
            if use_partial_refresh and cached_for_partial_refresh is not None:
                refreshed = merge_sensitivity_update(cached_for_partial_refresh, sensitivity, metadata=metadata)
                if bool(sensitivity_cfg.get("cache_enabled", True)):
                    self._cache_for(sensitivity_cfg).set(refreshed, step)
                sensitivity = active_sensitivity_slice(
                    refreshed,
                    network_object_ids=selected_objects,
                    action_unit_ids=action_unit_ids,
                )
            else:
                sensitivity.metadata = {**sensitivity.metadata, **metadata}
                if bool(sensitivity_cfg.get("cache_enabled", True)):
                    self._cache_for(sensitivity_cfg).set(sensitivity, step)
            sensitivity_cache_step = int(step)
        if bool(dso_cfg.get("ablation_no_sensitivity_edges", False)) or not bool(sensitivity_cfg.get("enabled", True)):
            sensitivity.values = np.zeros_like(sensitivity.values)
            sensitivity.metadata = {**sensitivity.metadata, "ablation_no_sensitivity_edges": True}
        max_action_units = int(dso_cfg.get("max_action_units", max(1, len(action_units))))
        max_network_objects = int(dso_cfg.get("max_network_objects", max(1, len(network_objects))))
        observation = encode_dso_observation_structured(
            step=step,
            dt_hours=simulator.scenario.dt_hours,
            voltage_limits=simulator.scenario.dso.voltage_limits,
            line_loading_limit_percent=simulator.scenario.dso.line_loading_limit_percent,
            trafo_loading_limit_percent=simulator.scenario.dso.trafo_loading_limit_percent,
            action_units=action_units,
            network_objects=network_objects,
            sensitivity_edges=sensitivity,
            max_action_units=max_action_units,
            max_network_objects=max_network_objects,
        )
        real_count = len(action_units)
        override_used = bool(actor_override)
        actor_override_source = str((actor_override or {}).get("source", "")) if override_used else ""
        if override_used:
            default_center = np.full((real_count,), 0.5, dtype=np.float32)
            default_width = np.full(
                (real_count,),
                float(self._actor_cfg().get("min_width_ratio", 0.10)),
                dtype=np.float32,
            )
            default_strength = np.ones((real_count,), dtype=np.float32)
            default_directions = np.zeros((real_count, 3), dtype=np.float32)
            if real_count:
                default_directions[:, 1] = 1.0
            legacy_arrays = _legacy_targets_to_action_unit_arrays(
                actor_override or {},
                vpp=vpp,
                fr=fr,
                action_units=action_units,
                min_width_ratio=float(self._actor_cfg().get("min_width_ratio", 0.10)),
            )
            if legacy_arrays is not None:
                center, width, directions, strength = legacy_arrays
            else:
                center = np.clip(
                    _select_override_array(actor_override or {}, "center_ratio", action_units, default_center),
                    0.0,
                    1.0,
                )
                width = np.clip(
                    _select_override_array(actor_override or {}, "width_ratio", action_units, default_width),
                    0.0,
                    1.0,
                )
                strength = np.clip(
                    _select_override_array(actor_override or {}, "guidance_strength", action_units, default_strength),
                    0.0,
                    1.0,
                )
                if "direction_probs" in (actor_override or {}):
                    directions = _normalize_direction_rows(
                        _select_override_array(actor_override or {}, "direction_probs", action_units, default_directions)
                    )
                else:
                    directions = _softmax_rows(
                        _select_override_array(actor_override or {}, "direction_logits", action_units, default_directions)
                    )
        else:
            actor = self._actor_for(observation)
            with torch.no_grad():
                outputs = actor(
                    global_features=torch.tensor(observation.global_features, dtype=torch.float32).unsqueeze(0),
                    action_tokens=torch.tensor(observation.action_tokens, dtype=torch.float32).unsqueeze(0),
                    object_tokens=torch.tensor(observation.object_tokens, dtype=torch.float32).unsqueeze(0),
                    sensitivity_edges=torch.tensor(observation.sensitivity_edges, dtype=torch.float32).unsqueeze(0),
                    action_mask=torch.tensor(observation.action_mask, dtype=torch.bool).unsqueeze(0),
                    object_mask=torch.tensor(observation.object_mask, dtype=torch.bool).unsqueeze(0),
                    edge_mask=torch.tensor(observation.edge_mask, dtype=torch.bool).unsqueeze(0),
                )
            center = outputs["center_ratio"].squeeze(0).cpu().numpy()[:real_count]
            width = outputs["width_ratio"].squeeze(0).cpu().numpy()[:real_count]
            directions = outputs["direction_probs"].squeeze(0).cpu().numpy()[:real_count]
            strength = outputs["guidance_strength"].squeeze(0).cpu().numpy()[:real_count]
        residual_eta, residual_progress_step = self._residual_schedule_eta(dso_cfg, step)
        residual_rule_blend_enabled = bool(dso_cfg.get("enable_rule_warmstart", False))
        rule_envelope = None
        rule_warmstart_role = "fallback_disabled"
        fallback_reason = ""
        if residual_rule_blend_enabled:
            rule_envelope = simulator._build_dso_operating_envelope(
                vpp,
                step,
                bid,
                fr,
                price,
                grid_state=grid_state,
            )
            rule_warmstart_role = "teacher_reference_only" if override_used else "fallback_reference_available"
        if residual_rule_blend_enabled and residual_eta < 1.0 and not override_used:
            if rule_envelope is None:
                rule_envelope = simulator._build_dso_operating_envelope(
                    vpp,
                    step,
                    bid,
                    fr,
                    price,
                    grid_state=grid_state,
                )
            rule_center, rule_width, rule_direction, rule_strength = _rule_targets_for_action_units(
                rule_envelope,
                action_units,
            )
            center = (1.0 - residual_eta) * rule_center + residual_eta * center
            width = (1.0 - residual_eta) * rule_width + residual_eta * width
            directions = (1.0 - residual_eta) * rule_direction + residual_eta * directions
            direction_denominator = np.maximum(directions.sum(axis=1, keepdims=True), 1e-9)
            directions = directions / direction_denominator
            strength = (1.0 - residual_eta) * rule_strength + residual_eta * strength
            rule_warmstart_role = "fallback_blend_without_unified_actor_action"
            fallback_reason = "no_external_unified_actor_override_before_residual_schedule_completion"
        records = decode_operating_envelope(
            action_unit_ids=[unit.id.action_unit_id for unit in action_units],
            vpp_ids=[unit.id.vpp_id for unit in action_units],
            pcc_ids=[unit.id.pcc_id for unit in action_units],
            bus_ids=[unit.id.bus_id for unit in action_units],
            p_hard_min_mw=[unit.p_min_mw for unit in action_units],
            p_hard_max_mw=[unit.p_max_mw for unit in action_units],
            center_ratio=center,
            width_ratio=width,
            direction_probs=directions,
            guidance_strength=strength,
        )
        p_min = float(sum(record.p_hard_min_mw for record in records))
        p_max = float(sum(record.p_hard_max_mw for record in records))
        preferred_low = float(sum(record.p_pref_lo_mw for record in records))
        preferred_high = float(sum(record.p_pref_hi_mw for record in records))
        preferred_target = float(sum(record.p_pref_target_mw for record in records))
        bounds_by_scope = {
            key: {
                "p_min_mw": float(item.p_min_mw),
                "p_max_mw": float(item.p_max_mw),
                "q_min_mvar": float(item.q_min_mvar),
                "q_max_mvar": float(item.q_max_mvar),
            }
            for key, item in fr.bounds.items()
        }
        preferred_target_by_scope = {}
        for record in records:
            key = f"bus_{record.bus_id}" if fr.scope == "bus_vector" else record.pcc_id or record.action_unit_id
            preferred_target_by_scope[str(key)] = float(record.p_pref_target_mw)
        confidence_idx = sensitivity.channel_names.index("sensitivity_confidence")
        confidence_values = sensitivity.values[:, :, confidence_idx]
        valid_confidence = confidence_values[sensitivity.edge_valid_mask]
        confidence_mean = float(np.mean(valid_confidence)) if valid_confidence.size else 0.0
        min_vm = float((grid_state or {}).get("min_vm_pu", 1.0))
        max_vm = float((grid_state or {}).get("max_vm_pu", 1.0))
        max_line = float((grid_state or {}).get("max_line_loading_percent", 0.0))
        max_trafo = float((grid_state or {}).get("max_trafo_loading_percent", 0.0))
        return {
            "step": int(step),
            "time_label": simulator._time_label(step),
            "vpp_id": vpp.id,
            "fr_id": fr.fr_id,
            "source_bid": "vpp_day_ahead_bid",
            "source_policy": self.policy_name,
            "p_min_mw": p_min,
            "p_max_mw": p_max,
            "q_min_mvar": float(fr.aggregate_bounds().q_min_mvar),
            "q_max_mvar": float(fr.aggregate_bounds().q_max_mvar),
            "preferred_p_min_mw": preferred_low,
            "preferred_p_max_mw": preferred_high,
            "preferred_target_p_mw": preferred_target,
            "service_request": "sensitivity_attention_guidance",
            "price": float(price),
            "grid_pressure_mode": "sensitivity_attention_v1",
            "ac_aware_grid_pressure_mode": "sensitivity_attention_v1_post_ac_required",
            "grid_priority_over_price": True,
            "network_min_vm_pu": min_vm,
            "network_max_vm_pu": max_vm,
            "network_max_line_loading_percent": max_line,
            "network_max_trafo_loading_percent": max_trafo,
            "pre_dispatch_powerflow_converged": bool((grid_state or {}).get("pre_dispatch_powerflow_converged", True)),
            "pcc_vm_pu": None,
            "bid_price_up": float(bid.get("bid_price_up", price)),
            "bid_price_down": float(bid.get("bid_price_down", price)),
            "confidence": confidence_mean,
            "dso_intent": "sensitivity_attention_v1_trainable_guidance_inside_fr_doe",
            "dso_decision_interface": (
                "sensitivity_attention_v1_unified_actor"
                if override_used
                else "sensitivity_attention_v1_internal_actor"
            ),
            "actor_override_source": actor_override_source,
            "rule_warmstart_role": rule_warmstart_role,
            "fallback_reason": fallback_reason,
            "rule_teacher_target_p_mw": (
                float(rule_envelope.get("preferred_target_p_mw"))
                if isinstance(rule_envelope, dict) and "preferred_target_p_mw" in rule_envelope
                else None
            ),
            "actor_target_p_mw": preferred_target,
            "final_target_p_mw": preferred_target,
            "fr_scope": str(fr.scope),
            "bounds_by_scope": bounds_by_scope,
            "current_p_by_scope": current_power_by_fr_scope(vpp, fr),
            "preferred_target_by_scope": preferred_target_by_scope,
            "vector_doe_enabled": bool(fr.scope != "pcc"),
            "award_status": "envelope_guidance",
            "action_units": action_unit_ids,
            "selected_network_objects": selected_objects,
            "active_sensitivity_edges_shape": tuple(int(v) for v in observation.sensitivity_edges.shape),
            "dso_actor_checkpoint_loaded": bool(self._actor_checkpoint_loaded),
            "dso_actor_checkpoint_path": self._actor_checkpoint_path,
            "dso_actor_checkpoint_source": self._actor_checkpoint_source,
            "residual_rule_blend_enabled": bool(residual_rule_blend_enabled),
            "residual_schedule_eta": float(residual_eta),
            "residual_schedule_progress_step": int(residual_progress_step),
            "warmstart_steps": int(dso_cfg.get("warmstart_steps", 0)),
            "residual_schedule_steps": int(dso_cfg.get("residual_schedule_steps", 0)),
            "sensitivity_cache_hit": bool(cache_hit),
            "sensitivity_source": sensitivity_source,
            "sensitivity_cache_step": sensitivity_cache_step,
            "sensitivity_refresh_reasons": sensitivity_refresh_reasons,
            "sensitivity_priority_action_units": sensitivity_priority_action_units,
            "sensitivity_partial_priority_refresh": bool(sensitivity.metadata.get("partial_priority_refresh", False)),
            "sensitivity_partial_refresh_action_unit_ids": tuple(
                str(action_id) for action_id in sensitivity.metadata.get("partial_refresh_action_unit_ids", ())
            ),
            "sensitivity_update_period_steps": int(sensitivity_cfg.get("update_period_steps", 4)),
            "sensitivity_cache_ttl_steps": int(sensitivity_cfg.get("cache_ttl_steps", 8)),
            "sensitivity_confidence": confidence_mean,
            "sensitivity_allocation_mode": str(
                sensitivity.metadata.get("sensitivity_allocation_mode", "")
            ),
            "sensitivity_allocation_weights": dict(
                sensitivity.metadata.get("sensitivity_allocation_weights", {})
            ),
            "dso_actor_raw_outputs": {
                "center_ratio": [float(v) for v in center],
                "width_ratio": [float(v) for v in width],
                "guidance_strength": [float(v) for v in strength],
            },
            "decoded_operating_envelope": [record.to_dict() for record in records],
            "direction_probs": [[float(x) for x in row] for row in directions],
            "guidance_strength_lambda": float(np.mean(strength)) if len(strength) else 0.0,
        }
