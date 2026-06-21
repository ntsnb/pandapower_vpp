from __future__ import annotations

from pathlib import Path

import numpy as np

from vpp_dso_sim.dso.envelope.schemas import ActionUnitId, ActionUnitState, SensitivityEdgeTensor
from vpp_dso_sim.dso.sensitivity.cache import active_sensitivity_slice
from vpp_dso_sim.dso.sensitivity.cache import decide_sensitivity_refresh
from vpp_dso_sim.dso.sensitivity.cache import merge_sensitivity_update
from vpp_dso_sim.dso.sensitivity.finite_difference import (
    SENSITIVITY_CHANNELS,
    compute_finite_difference_sensitivity_tensor,
)
from vpp_dso_sim.dso.sensitivity.selectors import build_action_units, select_critical_network_objects
from vpp_dso_sim.network.powerflow import run_powerflow
from vpp_dso_sim.optimization.feasibility_region import compute_static_feasible_region
from vpp_dso_sim.simulation.scenario import load_scenario


def test_sensitivity_tensor_shape_and_q_mask_when_q_disabled() -> None:
    scenario = load_scenario(Path("configs") / "european_lv_benchmark_v2.yaml")
    assert run_powerflow(scenario.net)
    vpp = scenario.vpps[0]
    fr = compute_static_feasible_region(vpp, 0, scope="bus_vector")
    units = build_action_units(vpp, fr, t=0, granularity="vpp_bus")[:2]
    objects = select_critical_network_objects(
        scenario.net,
        voltage_limits=scenario.dso.voltage_limits,
        line_loading_limit_percent=scenario.dso.line_loading_limit_percent,
        trafo_loading_limit_percent=scenario.dso.trafo_loading_limit_percent,
        topk_low_voltage_buses=1,
        topk_high_voltage_buses=1,
        topk_lines=1,
        topk_trafos=1,
    )

    tensor = compute_finite_difference_sensitivity_tensor(
        scenario.net,
        units,
        objects,
        enable_q_channels=False,
    )

    assert tensor.values.shape == (len(objects), len(units), len(SENSITIVITY_CHANNELS))
    assert tensor.edge_valid_mask.shape == (len(objects), len(units))
    assert tensor.q_channel_mask is False
    for channel in ("dV_dQ", "dLineLoading_dQ", "dTrafoLoading_dQ"):
        idx = SENSITIVITY_CHANNELS.index(channel)
        assert np.allclose(tensor.values[:, :, idx], 0.0)


def test_active_sensitivity_edges_are_slice_of_raw_tensor() -> None:
    raw_values = np.arange(4 * 3 * len(SENSITIVITY_CHANNELS), dtype=np.float32).reshape(
        4,
        3,
        len(SENSITIVITY_CHANNELS),
    )
    raw_mask = np.ones((4, 3), dtype=bool)
    raw = SensitivityEdgeTensor(
        values=raw_values,
        channel_names=SENSITIVITY_CHANNELS,
        edge_valid_mask=raw_mask,
        q_channel_mask=False,
        action_unit_ids=("a0", "a1", "a2"),
        network_object_ids=("k0", "k1", "k2", "k3"),
    )

    active = active_sensitivity_slice(raw, network_object_ids=["k3", "k1"], action_unit_ids=["a2", "a0"])

    assert active.values.shape == (2, 2, len(SENSITIVITY_CHANNELS))
    assert np.array_equal(active.values, raw_values[[3, 1]][:, [2, 0], :])
    assert active.metadata["source"] == "raw_sensitivity_cache"


def _dummy_action_unit(action_unit_id: str, *, width: float = 1.0, gap: float = 0.0) -> ActionUnitState:
    return ActionUnitState(
        id=ActionUnitId(
            action_unit_id=action_unit_id,
            vpp_id="vpp_0",
            unit_type="vpp_bus",
            pcc_id="pcc_1",
            bus_id=1,
            pp_element_refs=(),
        ),
        p_cur_mw=0.0,
        q_cur_mvar=0.0,
        p_min_mw=-0.5 * float(width),
        p_max_mw=0.5 * float(width),
        q_min_mvar=0.0,
        q_max_mvar=0.0,
        bid_up=None,
        bid_down=None,
        projection_gap_hist_mw=float(gap),
        q_control_available=False,
    )


def _dummy_raw_sensitivity(
    *,
    action_unit_ids: tuple[str, ...] = ("a0",),
    network_object_ids: tuple[str, ...] = ("k0",),
    cache_step: int = 0,
    metadata: dict[str, object] | None = None,
) -> SensitivityEdgeTensor:
    values = np.zeros((len(network_object_ids), len(action_unit_ids), len(SENSITIVITY_CHANNELS)), dtype=np.float32)
    values[:, :, SENSITIVITY_CHANNELS.index("sensitivity_confidence")] = 1.0
    values[:, :, SENSITIVITY_CHANNELS.index("edge_valid_mask")] = 1.0
    return SensitivityEdgeTensor(
        values=values,
        channel_names=SENSITIVITY_CHANNELS,
        edge_valid_mask=np.ones((len(network_object_ids), len(action_unit_ids)), dtype=bool),
        q_channel_mask=False,
        action_unit_ids=action_unit_ids,
        network_object_ids=network_object_ids,
        metadata={
            "cache_step": cache_step,
            "grid_state_snapshot": {
                "min_vm_pu": 1.0,
                "max_vm_pu": 1.0,
                "max_line_loading_percent": 50.0,
                "max_trafo_loading_percent": 20.0,
            },
            "fr_width_by_action_unit": {action_id: 1.0 for action_id in action_unit_ids},
            "projection_gap_by_action_unit": {action_id: 0.0 for action_id in action_unit_ids},
            **(metadata or {}),
        },
    )


def test_sensitivity_refresh_decision_keeps_cache_before_period_when_state_is_stable() -> None:
    raw = _dummy_raw_sensitivity(cache_step=0)

    decision = decide_sensitivity_refresh(
        raw,
        step=3,
        action_units=[_dummy_action_unit("a0", width=1.0)],
        network_object_ids=["k0"],
        grid_state={"min_vm_pu": 1.0, "max_vm_pu": 1.0, "max_line_loading_percent": 50.0, "max_trafo_loading_percent": 20.0},
        sensitivity_cfg={
            "update_period_steps": 4,
            "cache_ttl_steps": 8,
            "refresh_if_voltage_delta_pu_gt": 0.005,
            "refresh_if_loading_delta_pct_gt": 5.0,
            "refresh_if_fr_width_change_ratio_gt": 0.20,
            "refresh_if_projection_gap_hist_gt_mw": 0.10,
        },
    )

    assert decision.refresh is False
    assert decision.reasons == ()
    assert decision.priority_action_unit_ids == ()


def test_sensitivity_refresh_decision_triggers_on_period_ttl_and_grid_state_changes() -> None:
    raw = _dummy_raw_sensitivity(cache_step=0)

    periodic = decide_sensitivity_refresh(
        raw,
        step=4,
        action_units=[_dummy_action_unit("a0", width=1.0)],
        network_object_ids=["k0"],
        grid_state={"min_vm_pu": 1.0, "max_vm_pu": 1.0, "max_line_loading_percent": 50.0, "max_trafo_loading_percent": 20.0},
        sensitivity_cfg={"update_period_steps": 4, "cache_ttl_steps": 8},
    )
    stale = decide_sensitivity_refresh(
        raw,
        step=9,
        action_units=[_dummy_action_unit("a0", width=1.0)],
        network_object_ids=["k0"],
        grid_state={"min_vm_pu": 1.0, "max_vm_pu": 1.0, "max_line_loading_percent": 50.0, "max_trafo_loading_percent": 20.0},
        sensitivity_cfg={"update_period_steps": 100, "cache_ttl_steps": 8},
    )
    voltage_changed = decide_sensitivity_refresh(
        raw,
        step=1,
        action_units=[_dummy_action_unit("a0", width=1.0)],
        network_object_ids=["k0"],
        grid_state={"min_vm_pu": 0.993, "max_vm_pu": 1.0, "max_line_loading_percent": 50.0, "max_trafo_loading_percent": 20.0},
        sensitivity_cfg={"update_period_steps": 100, "cache_ttl_steps": 8, "refresh_if_voltage_delta_pu_gt": 0.005},
    )
    loading_changed = decide_sensitivity_refresh(
        raw,
        step=1,
        action_units=[_dummy_action_unit("a0", width=1.0)],
        network_object_ids=["k0"],
        grid_state={"min_vm_pu": 1.0, "max_vm_pu": 1.0, "max_line_loading_percent": 56.0, "max_trafo_loading_percent": 20.0},
        sensitivity_cfg={"update_period_steps": 100, "cache_ttl_steps": 8, "refresh_if_loading_delta_pct_gt": 5.0},
    )

    assert periodic.refresh is True
    assert "update_period_elapsed" in periodic.reasons
    assert stale.refresh is True
    assert "cache_ttl_expired" in stale.reasons
    assert voltage_changed.refresh is True
    assert "voltage_delta" in voltage_changed.reasons
    assert loading_changed.refresh is True
    assert "loading_delta" in loading_changed.reasons


def test_sensitivity_refresh_decision_prioritizes_width_and_projection_gap_units() -> None:
    raw = _dummy_raw_sensitivity(
        action_unit_ids=("a0", "a1", "a2"),
        network_object_ids=("k0",),
        metadata={
            "fr_width_by_action_unit": {"a0": 1.0, "a1": 1.0, "a2": 1.0},
            "projection_gap_by_action_unit": {"a0": 0.0, "a1": 0.0, "a2": 0.0},
        },
    )

    decision = decide_sensitivity_refresh(
        raw,
        step=1,
        action_units=[
            _dummy_action_unit("a0", width=1.0, gap=0.0),
            _dummy_action_unit("a1", width=1.35, gap=0.0),
            _dummy_action_unit("a2", width=1.0, gap=0.25),
        ],
        network_object_ids=["k0"],
        grid_state={"min_vm_pu": 1.0, "max_vm_pu": 1.0, "max_line_loading_percent": 50.0, "max_trafo_loading_percent": 20.0},
        sensitivity_cfg={
            "update_period_steps": 100,
            "cache_ttl_steps": 8,
            "refresh_if_fr_width_change_ratio_gt": 0.20,
            "refresh_if_projection_gap_hist_gt_mw": 0.10,
            "max_perturbed_action_units_per_update": 2,
        },
    )

    assert decision.refresh is True
    assert "fr_width_change" in decision.reasons
    assert "projection_gap_hist" in decision.reasons
    assert decision.priority_action_unit_ids == ("a2", "a1")


def test_merge_sensitivity_update_overwrites_priority_action_units_without_losing_raw_cache() -> None:
    raw = _dummy_raw_sensitivity(action_unit_ids=("a0", "a1"), network_object_ids=("k0", "k1"))
    raw.values[:, 0, SENSITIVITY_CHANNELS.index("dV_dP")] = 1.0
    raw.values[:, 1, SENSITIVITY_CHANNELS.index("dV_dP")] = 2.0
    update = _dummy_raw_sensitivity(action_unit_ids=("a1",), network_object_ids=("k0", "k1"))
    update.values[:, 0, SENSITIVITY_CHANNELS.index("dV_dP")] = 9.0

    merged = merge_sensitivity_update(raw, update, metadata={"cache_step": 5, "source": "partial_priority_refresh"})

    assert merged.action_unit_ids == ("a0", "a1")
    assert merged.network_object_ids == ("k0", "k1")
    assert np.allclose(merged.values[:, 0, SENSITIVITY_CHANNELS.index("dV_dP")], 1.0)
    assert np.allclose(merged.values[:, 1, SENSITIVITY_CHANNELS.index("dV_dP")], 9.0)
    assert merged.metadata["source"] == "partial_priority_refresh"
    assert merged.metadata["partial_refresh_action_unit_ids"] == ("a1",)
