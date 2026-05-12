from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandapower as pp

from vpp_dso_sim.entities.vpp import VPPAggregator
from vpp_dso_sim.network.constraints import (
    ConstraintReport,
    check_network_constraints,
    violation_penalties,
)
from vpp_dso_sim.network.powerflow import run_powerflow
from vpp_dso_sim.network.result_extractor import extract_network_snapshot
from vpp_dso_sim.network.sensitivity import create_representative_data_for_vpp


@dataclass
class DSO:
    net: pp.pandapowerNet
    voltage_limits: tuple[float, float] = (0.95, 1.05)
    line_loading_limit_percent: float = 100.0
    trafo_loading_limit_percent: float = 100.0
    vpp_registry: dict[str, VPPAggregator] = field(default_factory=dict)
    market_price_profile: list[float] = field(default_factory=list)
    reward_privacy_mode: str = "oracle_system_cost"
    reward_component_weights: dict[str, float] = field(default_factory=dict)
    dso_reward_cost_scale: float = 0.05
    security_violation_count_penalty: float = 0.0
    reward_component_scales: dict[str, float] = field(default_factory=dict)
    reward_component_clip: float = 10.0

    def _reward_weight(self, component: str) -> float:
        return float(self.reward_component_weights.get(component, 1.0))

    def _reward_scale(self, component: str) -> float:
        default_scales = {
            "operation_cost": 1000.0,
            "target_tracking_error_penalty": 10.0,
            "action_projection_penalty": 10.0,
            "comfort_violation_penalty": 100.0,
            "soc_violation_penalty": 100.0,
            "security_violation_count_penalty": 1.0,
            "post_ac_violation_magnitude_penalty": 1.0,
            "voltage_violation_penalty": 1.0,
            "line_overload_penalty": 100.0,
            "transformer_overload_penalty": 100.0,
            "powerflow_penalty": 1000.0,
        }
        return max(1e-9, float(self.reward_component_scales.get(component, default_scales.get(component, 1.0))))

    def _scaled_component(self, component: str, weighted_value: float) -> float:
        normalized = max(0.0, float(weighted_value)) / self._reward_scale(component)
        return min(float(self.reward_component_clip), normalized)

    def register_vpp(self, vpp: VPPAggregator) -> None:
        self.vpp_registry[vpp.id] = vpp

    def run_powerflow(self) -> bool:
        return run_powerflow(self.net)

    def check_security_constraints(self) -> ConstraintReport:
        return check_network_constraints(
            self.net,
            voltage_limits=self.voltage_limits,
            line_loading_limit_percent=self.line_loading_limit_percent,
            trafo_loading_limit_percent=self.trafo_loading_limit_percent,
        )

    def compute_network_state(self) -> dict[str, Any]:
        state = extract_network_snapshot(self.net)
        state["converged"] = bool(getattr(self.net, "converged", False))
        return state

    def compute_representative_data(self, t: int) -> dict[str, dict[str, object]]:
        return {
            vpp_id: create_representative_data_for_vpp(self, vpp, t)
            for vpp_id, vpp in self.vpp_registry.items()
        }

    def issue_day_ahead_targets(self, t: int) -> dict[str, float]:
        return {
            vpp_id: 0.5 * (bounds[0] + bounds[1])
            for vpp_id, vpp in self.vpp_registry.items()
            for bounds in [vpp.aggregate_flexibility(t)]
        }

    def issue_realtime_regulation_targets(self, t: int) -> dict[str, float]:
        targets: dict[str, float] = {}
        state = self.compute_network_state()
        min_v = state.get("min_vm_pu", 1.0)
        max_v = state.get("max_vm_pu", 1.0)
        for vpp_id, vpp in self.vpp_registry.items():
            p_min, p_max, _, _ = vpp.aggregate_flexibility(t)
            current = vpp.current_power_mw()
            if max_v > self.voltage_limits[1]:
                targets[vpp_id] = max(p_min, current - 0.10 * (p_max - p_min))
            elif min_v < self.voltage_limits[0]:
                targets[vpp_id] = min(p_max, current + 0.10 * (p_max - p_min))
            else:
                targets[vpp_id] = current
        return targets

    def calculate_reward_or_cost(
        self,
        report: ConstraintReport | None = None,
        target_tracking_error: float = 0.0,
        t: int = 0,
        action_projection_gap_mw: float = 0.0,
        local_bounds_projection_gap_mw: float | None = None,
        ac_aware_projection_gap_mw: float = 0.0,
        ac_certified_projection_gap_mw: float = 0.0,
        ac_certificate_failed_count: int = 0,
        action_projection_count: int = 0,
    ) -> dict[str, float]:
        report = self.check_security_constraints() if report is None else report
        raw_network_penalties = violation_penalties(report)
        private_operation_cost = float(sum(vpp.operating_cost() for vpp in self.vpp_registry.values()))
        price = (
            float(self.market_price_profile[t % len(self.market_price_profile)])
            if self.market_price_profile
            else 80.0
        )
        ext_grid_p_mw = 0.0
        if hasattr(self.net, "res_ext_grid") and len(self.net.res_ext_grid) and "p_mw" in self.net.res_ext_grid:
            ext_grid_p_mw = float(self.net.res_ext_grid["p_mw"].sum())
        procurement_proxy_cost = abs(ext_grid_p_mw) * price
        operation_cost = (
            procurement_proxy_cost
            if self.reward_privacy_mode == "privacy_preserving_proxy"
            else private_operation_cost
        )
        comfort_penalty = float(sum(vpp.comfort_penalty(t) for vpp in self.vpp_registry.values()))
        soc_penalty = float(sum(vpp.soc_violation_penalty(t) for vpp in self.vpp_registry.values()))
        projection_gap = max(0.0, float(action_projection_gap_mw))
        local_projection_gap = (
            projection_gap
            if local_bounds_projection_gap_mw is None
            else max(0.0, float(local_bounds_projection_gap_mw))
        )
        ac_projection_gap = max(0.0, float(ac_aware_projection_gap_mw))
        ac_certified_gap = max(0.0, float(ac_certified_projection_gap_mw))
        ac_certificate_failures = max(0, int(ac_certificate_failed_count))
        projection_count = max(0, int(action_projection_count))
        certified_penalty_gap = max(projection_gap, local_projection_gap + ac_projection_gap + ac_certified_gap)
        raw_action_projection_penalty = 250.0 * certified_penalty_gap * certified_penalty_gap + 2.0 * projection_count
        raw_target_tracking_penalty = 100.0 * target_tracking_error * target_tracking_error
        post_ac_violation_records = report.to_records()
        post_ac_violation_count = len(post_ac_violation_records)
        post_ac_voltage_violation_count = sum(
            1 for record in post_ac_violation_records if str(record.get("kind", "")).startswith("bus_voltage")
        )
        post_ac_line_overload_count = sum(
            1 for record in post_ac_violation_records if record.get("kind") == "line_overload"
        )
        post_ac_trafo_overload_count = sum(
            1 for record in post_ac_violation_records if record.get("kind") == "trafo_overload"
        )
        post_ac_powerflow_failed = 0 if report.converged else 1
        post_ac_violation_magnitude = float(
            sum(abs(float(record.get("magnitude", 0.0))) for record in post_ac_violation_records)
        )
        raw_security_violation_count_penalty = (
            float(post_ac_violation_count) * float(self.security_violation_count_penalty)
        )
        raw_components = {
            "operation_cost": operation_cost,
            "target_tracking_error_penalty": raw_target_tracking_penalty,
            "action_projection_penalty": raw_action_projection_penalty,
            "comfort_violation_penalty": comfort_penalty,
            "soc_violation_penalty": soc_penalty,
            "security_violation_count_penalty": raw_security_violation_count_penalty,
            "post_ac_violation_magnitude_penalty": post_ac_violation_magnitude,
            **raw_network_penalties,
        }
        weighted_components = {
            key: float(value) * self._reward_weight(key)
            for key, value in raw_components.items()
        }
        scaled_components = {
            f"scaled_{key}": self._scaled_component(key, value)
            for key, value in weighted_components.items()
        }
        components = {
            "reward_privacy_mode_code": 1.0 if self.reward_privacy_mode == "privacy_preserving_proxy" else 0.0,
            "private_operation_cost_reference": private_operation_cost,
            "procurement_proxy_cost": procurement_proxy_cost,
            "action_projection_gap_mw": projection_gap,
            "local_bounds_projection_gap_mw": local_projection_gap,
            "ac_aware_projection_gap_mw": ac_projection_gap,
            "ac_certified_projection_gap_mw": ac_certified_gap,
            "ac_certificate_failed_count": float(ac_certificate_failures),
            "action_projection_count": float(projection_count),
            "constraint_violation_count": float(post_ac_violation_count),
            "post_ac_violation_count": float(post_ac_violation_count),
            "post_ac_voltage_violation_count": float(post_ac_voltage_violation_count),
            "post_ac_line_overload_count": float(post_ac_line_overload_count),
            "post_ac_trafo_overload_count": float(post_ac_trafo_overload_count),
            "post_ac_powerflow_failed": float(post_ac_powerflow_failed),
            "post_ac_violation_magnitude": post_ac_violation_magnitude,
            **weighted_components,
            **scaled_components,
        }
        for key, value in raw_components.items():
            components[f"raw_{key}"] = float(value)
            components[f"{key}_weight"] = self._reward_weight(key)
            components[f"{key}_scale"] = self._reward_scale(key)
        cost_keys = [
            "operation_cost",
            "target_tracking_error_penalty",
            "action_projection_penalty",
            "comfort_violation_penalty",
            "soc_violation_penalty",
            "security_violation_count_penalty",
            "post_ac_violation_magnitude_penalty",
            "voltage_violation_penalty",
            "line_overload_penalty",
            "transformer_overload_penalty",
            "powerflow_penalty",
        ]
        components["total_cost"] = float(sum(components[key] for key in cost_keys))
        components["scaled_total_cost"] = float(sum(components[f"scaled_{key}"] for key in cost_keys))
        components["raw_objective_reward"] = -components["total_cost"]
        components["scaled_objective_reward"] = -components["scaled_total_cost"]
        components["post_ac_security_penalty"] = float(
            components["security_violation_count_penalty"]
            + components["post_ac_violation_magnitude_penalty"]
            + components["voltage_violation_penalty"]
            + components["line_overload_penalty"]
            + components["transformer_overload_penalty"]
            + components["powerflow_penalty"]
        )
        components["scaled_post_ac_security_penalty"] = float(
            components["scaled_security_violation_count_penalty"]
            + components["scaled_post_ac_violation_magnitude_penalty"]
            + components["scaled_voltage_violation_penalty"]
            + components["scaled_line_overload_penalty"]
            + components["scaled_transformer_overload_penalty"]
            + components["scaled_powerflow_penalty"]
        )
        components["feasibility_bonus"] = 1.0 if report.ok else 0.0
        components["tracking_bonus"] = 0.25 / (1.0 + target_tracking_error)
        components["dso_total_cost"] = components["total_cost"]
        components["dso_scaled_total_cost"] = components["scaled_total_cost"]
        components["dso_reward_cost_basis"] = components["scaled_total_cost"]
        components["reward_scaling_version"] = 2.0
        components["dso_reward"] = (
            -float(self.dso_reward_cost_scale) * components["dso_reward_cost_basis"]
            + components["feasibility_bonus"]
            + components["tracking_bonus"]
        )
        components["scaled_reward"] = components["dso_reward"]
        # Backward-compatible single-agent reward. Multi-agent environments
        # split this into DSO, VPP dispatch and VPP portfolio rewards.
        components["reward"] = components["dso_reward"]
        return components

    def reset(self) -> None:
        for vpp in self.vpp_registry.values():
            vpp.reset()
