from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from vpp_dso_sim.dso.observation.structured_bipartite import encode_dso_observation_structured
from vpp_dso_sim.dso.sensitivity.finite_difference import compute_finite_difference_sensitivity_tensor
from vpp_dso_sim.dso.sensitivity.selectors import build_action_units, select_critical_network_objects
from vpp_dso_sim.network.powerflow import run_powerflow
from vpp_dso_sim.optimization.feasibility_region import compute_static_feasible_region


@dataclass(frozen=True)
class StructuredDSOFlatSpec:
    global_dim: int
    action_token_dim: int
    object_token_dim: int
    edge_feature_dim: int
    max_action_units: int
    max_network_objects: int
    action_unit_vpp_indices: tuple[int, ...]
    vpp_ids: tuple[str, ...]
    action_unit_ids: tuple[str, ...] = ()
    field_names: tuple[str, ...] = ()
    privacy_boundary: str = "dso_execution_actor_no_private_vpp_fields"

    @property
    def flat_dim(self) -> int:
        return int(
            self.global_dim
            + self.max_action_units * self.action_token_dim
            + self.max_network_objects * self.object_token_dim
            + self.max_network_objects * self.max_action_units * self.edge_feature_dim
            + self.max_action_units
            + self.max_network_objects
            + self.max_network_objects * self.max_action_units
        )

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["flat_dim"] = self.flat_dim
        return payload


def _structured_flat_field_names(spec: StructuredDSOFlatSpec) -> tuple[str, ...]:
    names: list[str] = []
    names.extend(f"global_feature_{index}" for index in range(spec.global_dim))
    for action_index in range(spec.max_action_units):
        names.extend(
            f"action_unit_{action_index}_feature_{feature_index}"
            for feature_index in range(spec.action_token_dim)
        )
    for object_index in range(spec.max_network_objects):
        names.extend(
            f"network_object_{object_index}_feature_{feature_index}"
            for feature_index in range(spec.object_token_dim)
        )
    for object_index in range(spec.max_network_objects):
        for action_index in range(spec.max_action_units):
            names.extend(
                f"sensitivity_object_{object_index}_action_{action_index}_channel_{channel_index}"
                for channel_index in range(spec.edge_feature_dim)
            )
    names.extend(f"action_mask_{index}" for index in range(spec.max_action_units))
    names.extend(f"object_mask_{index}" for index in range(spec.max_network_objects))
    for object_index in range(spec.max_network_objects):
        names.extend(
            f"edge_mask_object_{object_index}_action_{action_index}"
            for action_index in range(spec.max_action_units)
        )
    return tuple(names)


def flatten_structured_dso_observation(observation) -> np.ndarray:
    return np.concatenate(
        [
            observation.global_features.reshape(-1),
            observation.action_tokens.reshape(-1),
            observation.object_tokens.reshape(-1),
            observation.sensitivity_edges.reshape(-1),
            observation.action_mask.astype(np.float32).reshape(-1),
            observation.object_mask.astype(np.float32).reshape(-1),
            observation.edge_mask.astype(np.float32).reshape(-1),
        ],
        dtype=np.float32,
    )


def build_happo_structured_dso_observation(
    scenario,
    *,
    step: int,
    config: dict[str, Any],
) -> tuple[np.ndarray, StructuredDSOFlatSpec]:
    """Build a flattened structured DSO observation for HAPPO storage.

    The flat vector is only a storage representation. The actor reconstructs
    typed tensors before applying bipartite sensitivity attention.
    """

    run_powerflow(scenario.net)
    dso_cfg = dict(config.get("dso", {}))
    selector_cfg = dict(config.get("selector", {}))
    sensitivity_cfg = dict(config.get("sensitivity", {}))
    vpp_ids = tuple(str(vpp.id) for vpp in scenario.vpps)
    action_units = []
    action_unit_vpp_indices: list[int] = []
    for vpp_index, vpp in enumerate(scenario.vpps):
        fr = compute_static_feasible_region(vpp, step)
        units = build_action_units(
            vpp,
            fr,
            t=step,
            granularity=str(dso_cfg.get("action_unit_granularity", "vpp_bus")),
        )
        action_units.extend(units)
        action_unit_vpp_indices.extend([vpp_index] * len(units))
    network_objects = select_critical_network_objects(
        scenario.net,
        voltage_limits=scenario.dso.voltage_limits,
        line_loading_limit_percent=scenario.dso.line_loading_limit_percent,
        trafo_loading_limit_percent=scenario.dso.trafo_loading_limit_percent,
        topk_low_voltage_buses=int(selector_cfg.get("topk_low_voltage_buses", 5)),
        topk_high_voltage_buses=int(selector_cfg.get("topk_high_voltage_buses", 5)),
        topk_lines=int(selector_cfg.get("topk_lines", 5)),
        topk_trafos=int(selector_cfg.get("topk_trafos", 3)),
    )
    edges = compute_finite_difference_sensitivity_tensor(
        scenario.net,
        action_units,
        network_objects,
        enable_q_channels=bool(dso_cfg.get("enable_q_channels", sensitivity_cfg.get("enable_q_channels", False))),
        epsilon_p_min_mw=float(sensitivity_cfg.get("epsilon_p_min_mw", 0.005)),
        epsilon_p_max_mw=float(sensitivity_cfg.get("epsilon_p_max_mw", 0.05)),
        epsilon_p_width_ratio=float(sensitivity_cfg.get("epsilon_p_width_ratio", 0.02)),
    )
    if bool(dso_cfg.get("ablation_no_sensitivity_edges", False)) or not bool(sensitivity_cfg.get("enabled", True)):
        edges.values = np.zeros_like(edges.values)
        edges.metadata = {**edges.metadata, "ablation_no_sensitivity_edges": True}
    max_action_units = int(dso_cfg.get("max_action_units", max(1, len(action_units))))
    max_network_objects = int(dso_cfg.get("max_network_objects", max(1, len(network_objects))))
    observation = encode_dso_observation_structured(
        step=step,
        dt_hours=scenario.dt_hours,
        voltage_limits=scenario.dso.voltage_limits,
        line_loading_limit_percent=scenario.dso.line_loading_limit_percent,
        trafo_loading_limit_percent=scenario.dso.trafo_loading_limit_percent,
        action_units=action_units,
        network_objects=network_objects,
        sensitivity_edges=edges,
        max_action_units=max_action_units,
        max_network_objects=max_network_objects,
    )
    padded_indices = tuple(
        action_unit_vpp_indices[:max_action_units]
        + [-1] * max(0, max_action_units - len(action_unit_vpp_indices))
    )
    padded_action_unit_ids = tuple(
        [unit.id.action_unit_id for unit in action_units[:max_action_units]]
        + [""] * max(0, max_action_units - len(action_units))
    )
    spec = StructuredDSOFlatSpec(
        global_dim=int(observation.global_features.shape[-1]),
        action_token_dim=int(observation.action_tokens.shape[-1]),
        object_token_dim=int(observation.object_tokens.shape[-1]),
        edge_feature_dim=int(observation.sensitivity_edges.shape[-1]),
        max_action_units=max_action_units,
        max_network_objects=max_network_objects,
        action_unit_vpp_indices=padded_indices,
        vpp_ids=vpp_ids,
        action_unit_ids=padded_action_unit_ids,
        field_names=(),
    )
    spec = StructuredDSOFlatSpec(
        global_dim=spec.global_dim,
        action_token_dim=spec.action_token_dim,
        object_token_dim=spec.object_token_dim,
        edge_feature_dim=spec.edge_feature_dim,
        max_action_units=spec.max_action_units,
        max_network_objects=spec.max_network_objects,
        action_unit_vpp_indices=spec.action_unit_vpp_indices,
        vpp_ids=spec.vpp_ids,
        action_unit_ids=spec.action_unit_ids,
        field_names=_structured_flat_field_names(spec),
    )
    return flatten_structured_dso_observation(observation), spec
