from __future__ import annotations

import math

import numpy as np

from vpp_dso_sim.dso.envelope.schemas import (
    ActionUnitState,
    NetworkObjectState,
    SensitivityEdgeTensor,
    StructuredDSOObservation,
)


ACTION_TOKEN_FIELDS: tuple[str, ...] = (
    "unit_type_vpp_pcc",
    "unit_type_vpp_bus",
    "unit_type_der",
    "bus_id_norm",
    "p_cur_norm",
    "p_min_norm",
    "p_max_norm",
    "h_up_norm",
    "h_down_norm",
    "bid_up_norm",
    "bid_down_norm",
    "projection_gap_hist_norm",
    "q_control_available",
)

OBJECT_TOKEN_FIELDS: tuple[str, ...] = (
    "object_type_bus",
    "object_type_line",
    "object_type_trafo",
    "primary_id_norm",
    "endpoint_mean_bus_norm",
    "value_norm",
    "lower_margin_norm",
    "upper_margin_norm",
    "low_or_thermal_severity",
    "high_severity",
)


def _scale_power(value: float) -> float:
    return float(value) / 10.0


def _action_token(unit: ActionUnitState) -> list[float]:
    width = max(1e-9, unit.p_max_mw - unit.p_min_mw)
    return [
        1.0 if unit.id.unit_type == "vpp_pcc" else 0.0,
        1.0 if unit.id.unit_type == "vpp_bus" else 0.0,
        1.0 if unit.id.unit_type == "der" else 0.0,
        float(unit.id.bus_id) / 100.0,
        _scale_power(unit.p_cur_mw),
        _scale_power(unit.p_min_mw),
        _scale_power(unit.p_max_mw),
        _scale_power(max(0.0, unit.p_max_mw - unit.p_cur_mw)),
        _scale_power(max(0.0, unit.p_cur_mw - unit.p_min_mw)),
        float(unit.bid_up or 0.0) / 100.0,
        float(unit.bid_down or 0.0) / 100.0,
        _scale_power(unit.projection_gap_hist_mw),
        1.0 if unit.q_control_available else 0.0,
    ]


def _object_token(
    obj: NetworkObjectState,
    *,
    voltage_limits: tuple[float, float],
    line_loading_limit_percent: float,
    trafo_loading_limit_percent: float,
) -> list[float]:
    endpoint_mean = float(sum(obj.id.endpoint_bus_ids) / max(1, len(obj.id.endpoint_bus_ids))) / 100.0
    if obj.id.object_type == "bus":
        vmin, vmax = float(voltage_limits[0]), float(voltage_limits[1])
        lower_margin = float(obj.value - vmin)
        upper_margin = float(vmax - obj.value)
        return [
            1.0,
            0.0,
            0.0,
            float(obj.id.primary_id) / 100.0,
            endpoint_mean,
            float(obj.value),
            lower_margin / max(1e-6, vmax - vmin),
            upper_margin / max(1e-6, vmax - vmin),
            max(0.0, -lower_margin) / max(1e-6, vmax - vmin),
            max(0.0, -upper_margin) / max(1e-6, vmax - vmin),
        ]
    limit = float(line_loading_limit_percent if obj.id.object_type == "line" else trafo_loading_limit_percent)
    margin = float(limit - obj.value)
    severity = max(0.0, -margin) / max(1e-6, limit)
    return [
        0.0,
        1.0 if obj.id.object_type == "line" else 0.0,
        1.0 if obj.id.object_type == "trafo" else 0.0,
        float(obj.id.primary_id) / 100.0,
        endpoint_mean,
        float(obj.value) / max(1e-6, limit),
        0.0,
        margin / max(1e-6, limit),
        severity,
        0.0,
    ]


def _pad_2d(rows: list[list[float]], max_rows: int, width: int) -> np.ndarray:
    out = np.zeros((max_rows, width), dtype=np.float32)
    for index, row in enumerate(rows[:max_rows]):
        out[index, :width] = np.asarray(row[:width], dtype=np.float32)
    return out


def encode_dso_observation_structured(
    *,
    step: int,
    dt_hours: float,
    voltage_limits: tuple[float, float],
    line_loading_limit_percent: float,
    trafo_loading_limit_percent: float,
    action_units: list[ActionUnitState],
    network_objects: list[NetworkObjectState],
    sensitivity_edges: SensitivityEdgeTensor,
    max_action_units: int,
    max_network_objects: int,
) -> StructuredDSOObservation:
    """Build padded structured DSO actor observation with privacy-safe fields only."""

    real_actions = min(len(action_units), int(max_action_units))
    real_objects = min(len(network_objects), int(max_network_objects))
    action_tokens = _pad_2d([_action_token(unit) for unit in action_units], int(max_action_units), len(ACTION_TOKEN_FIELDS))
    object_tokens = _pad_2d(
        [
            _object_token(
                obj,
                voltage_limits=voltage_limits,
                line_loading_limit_percent=line_loading_limit_percent,
                trafo_loading_limit_percent=trafo_loading_limit_percent,
            )
            for obj in network_objects
        ],
        int(max_network_objects),
        len(OBJECT_TOKEN_FIELDS),
    )
    edge_width = len(sensitivity_edges.channel_names)
    edge_values = np.zeros((int(max_network_objects), int(max_action_units), edge_width), dtype=np.float32)
    edge_mask = np.zeros((int(max_network_objects), int(max_action_units)), dtype=bool)
    edge_values[:real_objects, :real_actions, :] = sensitivity_edges.values[:real_objects, :real_actions, :]
    edge_mask[:real_objects, :real_actions] = sensitivity_edges.edge_valid_mask[:real_objects, :real_actions]
    action_mask = np.zeros(int(max_action_units), dtype=bool)
    object_mask = np.zeros(int(max_network_objects), dtype=bool)
    action_mask[:real_actions] = True
    object_mask[:real_objects] = True

    hour = float(step) * float(dt_hours)
    global_features = np.asarray(
        [
            float(step) / 288.0,
            math.sin(2.0 * math.pi * hour / 24.0),
            math.cos(2.0 * math.pi * hour / 24.0),
            float(real_actions) / max(1.0, float(max_action_units)),
            float(real_objects) / max(1.0, float(max_network_objects)),
            1.0 if sensitivity_edges.metadata.get("status") == "ok" else 0.0,
        ],
        dtype=np.float32,
    )
    field_names = list(ACTION_TOKEN_FIELDS) + list(OBJECT_TOKEN_FIELDS) + list(sensitivity_edges.channel_names)
    return StructuredDSOObservation(
        global_features=global_features,
        action_tokens=action_tokens,
        object_tokens=object_tokens,
        sensitivity_edges=edge_values,
        action_mask=action_mask,
        object_mask=object_mask,
        edge_mask=edge_mask,
        metadata={
            "field_names": tuple(field_names),
            "contains_private_true_cost": False,
            "normalization": "power/10MW, price/100, bus_id/100, loading/limit",
            "padding": "zero rows with false masks",
        },
    )
