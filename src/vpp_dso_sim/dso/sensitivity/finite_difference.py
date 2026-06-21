from __future__ import annotations

from copy import deepcopy

import numpy as np

from vpp_dso_sim.dso.envelope.schemas import ActionUnitState, NetworkObjectState, SensitivityEdgeTensor
from vpp_dso_sim.network.powerflow import run_powerflow


SENSITIVITY_CHANNELS: tuple[str, ...] = (
    "dV_dP",
    "dV_dQ",
    "dLineLoading_dP",
    "dLineLoading_dQ",
    "dTrafoLoading_dP",
    "dTrafoLoading_dQ",
    "sensitivity_confidence",
    "sensitivity_age",
    "edge_valid_mask",
)


def _allocation_weights_for_unit(action_unit: ActionUnitState) -> dict[str, float]:
    refs = list(action_unit.id.pp_element_refs)
    if not refs:
        return {}
    weight = 1.0 / float(len(refs))
    return {f"{ref.element_type}:{ref.element_index}": weight for ref in refs}


def _apply_internal_p_delta_to_net(net, action_unit: ActionUnitState, delta_p_mw: float) -> None:
    refs = list(action_unit.id.pp_element_refs)
    if not refs:
        return
    share = float(delta_p_mw) / float(len(refs))
    for ref in refs:
        table = getattr(net, ref.element_type)
        if ref.element_index not in table.index:
            continue
        if ref.element_type == "sgen":
            table.at[ref.element_index, "p_mw"] = float(table.at[ref.element_index, "p_mw"]) + share
        elif ref.element_type == "load":
            table.at[ref.element_index, "p_mw"] = max(0.0, float(table.at[ref.element_index, "p_mw"]) - share)
        elif ref.element_type == "storage":
            table.at[ref.element_index, "p_mw"] = float(table.at[ref.element_index, "p_mw"]) - share


def _metric(net, obj: NetworkObjectState) -> float:
    if obj.id.object_type == "bus":
        return float(net.res_bus.at[obj.id.primary_id, "vm_pu"])
    if obj.id.object_type == "line":
        return float(net.res_line.at[obj.id.primary_id, "loading_percent"])
    if obj.id.object_type == "trafo" and hasattr(net, "res_trafo") and obj.id.primary_id in net.res_trafo.index:
        return float(net.res_trafo.at[obj.id.primary_id, "loading_percent"])
    return 0.0


def _channel_for_object(obj: NetworkObjectState) -> str:
    if obj.id.object_type == "bus":
        return "dV_dP"
    if obj.id.object_type == "line":
        return "dLineLoading_dP"
    return "dTrafoLoading_dP"


def _epsilon_for_unit(
    unit: ActionUnitState,
    *,
    epsilon_p_min_mw: float,
    epsilon_p_max_mw: float,
    epsilon_p_width_ratio: float,
) -> float:
    width = unit.hard_width_mw()
    return float(min(epsilon_p_max_mw, max(epsilon_p_min_mw, epsilon_p_width_ratio * width)))


def compute_finite_difference_sensitivity_tensor(
    net,
    action_units: list[ActionUnitState],
    network_objects: list[NetworkObjectState],
    *,
    enable_q_channels: bool = False,
    epsilon_p_min_mw: float = 0.005,
    epsilon_p_max_mw: float = 0.05,
    epsilon_p_width_ratio: float = 0.02,
) -> SensitivityEdgeTensor:
    """Estimate `M[K, A, C]` by finite-difference AC power flow.

    Q channels are included in the channel schema. If `enable_q_channels` is false,
    they remain zero and `q_channel_mask` is false.
    """

    allocation_weights = {
        unit.id.action_unit_id: _allocation_weights_for_unit(unit)
        for unit in action_units
    }
    base = deepcopy(net)
    if not run_powerflow(base):
        values = np.zeros((len(network_objects), len(action_units), len(SENSITIVITY_CHANNELS)), dtype=np.float32)
        mask = np.zeros((len(network_objects), len(action_units)), dtype=bool)
        return SensitivityEdgeTensor(
            values=values,
            channel_names=SENSITIVITY_CHANNELS,
            edge_valid_mask=mask,
            q_channel_mask=bool(enable_q_channels),
            action_unit_ids=tuple(unit.id.action_unit_id for unit in action_units),
            network_object_ids=tuple(obj.id.object_id for obj in network_objects),
            metadata={
                "status": "base_powerflow_failed",
                "sensitivity_allocation_mode": "equal_pp_element_refs",
                "sensitivity_allocation_weights": allocation_weights,
            },
        )

    values = np.zeros((len(network_objects), len(action_units), len(SENSITIVITY_CHANNELS)), dtype=np.float32)
    mask = np.zeros((len(network_objects), len(action_units)), dtype=bool)
    confidence_idx = SENSITIVITY_CHANNELS.index("sensitivity_confidence")
    age_idx = SENSITIVITY_CHANNELS.index("sensitivity_age")
    valid_idx = SENSITIVITY_CHANNELS.index("edge_valid_mask")

    for action_index, unit in enumerate(action_units):
        eps = _epsilon_for_unit(
            unit,
            epsilon_p_min_mw=epsilon_p_min_mw,
            epsilon_p_max_mw=epsilon_p_max_mw,
            epsilon_p_width_ratio=epsilon_p_width_ratio,
        )
        h_up = max(0.0, float(unit.p_max_mw) - float(unit.p_cur_mw))
        h_down = max(0.0, float(unit.p_cur_mw) - float(unit.p_min_mw))
        plus_delta = min(eps, 0.5 * h_up) if h_up > 1e-9 else 0.0
        minus_delta = min(eps, 0.5 * h_down) if h_down > 1e-9 else 0.0
        perturbed: list[tuple[float, object]] = []
        for delta in (plus_delta, -minus_delta):
            if abs(delta) <= 1e-9:
                continue
            work = deepcopy(net)
            _apply_internal_p_delta_to_net(work, unit, delta)
            if run_powerflow(work):
                perturbed.append((delta, work))
        if not perturbed:
            continue

        confidence = 1.0 if len(perturbed) == 2 else 0.5
        for object_index, obj in enumerate(network_objects):
            if len(perturbed) == 2 and perturbed[0][0] > 0.0 and perturbed[1][0] < 0.0:
                plus, plus_net = perturbed[0]
                minus, minus_net = perturbed[1]
                sensitivity = (_metric(plus_net, obj) - _metric(minus_net, obj)) / (plus - minus)
            else:
                delta, work = perturbed[0]
                sensitivity = (_metric(work, obj) - _metric(base, obj)) / delta
            channel_idx = SENSITIVITY_CHANNELS.index(_channel_for_object(obj))
            values[object_index, action_index, channel_idx] = float(sensitivity)
            values[object_index, action_index, confidence_idx] = float(confidence)
            values[object_index, action_index, age_idx] = 0.0
            values[object_index, action_index, valid_idx] = 1.0
            mask[object_index, action_index] = True

    if not enable_q_channels:
        for channel in ("dV_dQ", "dLineLoading_dQ", "dTrafoLoading_dQ"):
            values[:, :, SENSITIVITY_CHANNELS.index(channel)] = 0.0

    return SensitivityEdgeTensor(
        values=values,
        channel_names=SENSITIVITY_CHANNELS,
        edge_valid_mask=mask,
        q_channel_mask=bool(enable_q_channels),
        action_unit_ids=tuple(unit.id.action_unit_id for unit in action_units),
        network_object_ids=tuple(obj.id.object_id for obj in network_objects),
        metadata={
            "status": "ok",
            "estimator": "finite_difference_ac_powerflow",
            "sensitivity_allocation_mode": "equal_pp_element_refs",
            "sensitivity_allocation_weights": allocation_weights,
        },
    )
