from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any

import pandapower as pp

from vpp_dso_sim.entities.vpp import VPPAggregator
from vpp_dso_sim.learning.reward_config import RewardConfig
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
    reward_config: RewardConfig = field(default_factory=RewardConfig)

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
            "envelope_width_penalty": 1.0,
            "smoothness_penalty": 1.0,
        }
        return max(1e-9, float(self.reward_component_scales.get(component, default_scales.get(component, 1.0))))

    def _scaled_component(self, component: str, weighted_value: float) -> float:
        normalized = max(0.0, float(weighted_value)) / self._reward_scale(component)
        return min(float(self.reward_component_clip), normalized)

    def _bounded_training_penalty(self, value: float) -> float:
        return min(float(self.reward_config.dso.component_clip), max(0.0, float(value)))

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
        mean_envelope_width_ratio: float = 0.0,
        envelope_smoothness_mw: float = 0.0,
        mean_guidance_strength_lambda: float = 0.0,
        effective_response_score: float | None = None,
        vpp_settlement_summaries: dict[str, dict[str, Any]] | None = None,
        raw_action_voltage_violation_cost: float = 0.0,
        raw_action_line_overload_cost: float = 0.0,
        raw_action_trafo_overload_cost: float = 0.0,
        raw_action_powerflow_failed: float = 0.0,
        projected_action_voltage_violation_cost: float | None = None,
        projected_action_line_overload_cost: float | None = None,
        projected_action_trafo_overload_cost: float | None = None,
        projected_action_powerflow_failed: float | None = None,
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
        raw_envelope_width_penalty = max(0.0, float(mean_envelope_width_ratio))
        raw_smoothness_penalty = max(0.0, float(envelope_smoothness_mw))
        effective_response = (
            max(0.0, min(1.0, float(effective_response_score)))
            if effective_response_score is not None
            else 1.0 / (1.0 + max(0.0, float(target_tracking_error)))
        )
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
            "envelope_width_penalty": raw_envelope_width_penalty,
            "smoothness_penalty": raw_smoothness_penalty,
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
            "mean_envelope_width_ratio": float(mean_envelope_width_ratio),
            "envelope_smoothness_mw": float(envelope_smoothness_mw),
            "mean_guidance_strength_lambda": float(mean_guidance_strength_lambda),
            "effective_response_bonus": float(effective_response),
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
            "envelope_width_penalty",
            "smoothness_penalty",
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
        components.update(
            self._dso_v2_diagnostics(
                report=report,
                procurement_proxy_cost=procurement_proxy_cost,
                projection_gap=projection_gap,
                ac_projection_gap=ac_projection_gap,
                ac_certified_gap=ac_certified_gap,
                mean_envelope_width_ratio=mean_envelope_width_ratio,
                envelope_smoothness_mw=envelope_smoothness_mw,
                post_ac_violation_count=post_ac_violation_count,
                post_ac_powerflow_failed=post_ac_powerflow_failed,
                raw_network_penalties=raw_network_penalties,
            )
        )
        components["reward_version_code"] = 2.0 if self.reward_config.is_v2_minimal else 1.0
        components["reward_scaling_version"] = 2.0
        if self.reward_config.is_v3_market_safety:
            components.update(
                self._dso_v3_market_safety_components(
                    report=report,
                    raw_network_penalties=raw_network_penalties,
                    vpp_settlement_summaries=vpp_settlement_summaries,
                    raw_action_voltage_violation_cost=raw_action_voltage_violation_cost,
                    raw_action_line_overload_cost=raw_action_line_overload_cost,
                    raw_action_trafo_overload_cost=raw_action_trafo_overload_cost,
                    raw_action_powerflow_failed=raw_action_powerflow_failed,
                    projected_action_voltage_violation_cost=projected_action_voltage_violation_cost,
                    projected_action_line_overload_cost=projected_action_line_overload_cost,
                    projected_action_trafo_overload_cost=projected_action_trafo_overload_cost,
                    projected_action_powerflow_failed=projected_action_powerflow_failed,
                )
            )
            components["reward_version_code"] = 3.1
            components["reward_scaling_version"] = 3.1
        elif self.reward_config.is_v2_minimal:
            dso_cfg = self.reward_config.dso
            components["dso_reward_cost_basis"] = components["dso_v2_training_cost_basis"]
            tracking_bonus = components["tracking_bonus"] if dso_cfg.enable_tracking_bonus else 0.0
            effective_bonus = components["effective_response_bonus"] if dso_cfg.enable_effective_response_bonus else 0.0
            feasibility_bonus = components["feasibility_bonus"] * float(dso_cfg.feasibility_bonus_weight)
            components["tracking_bonus_diagnostic"] = components["tracking_bonus"]
            components["effective_response_bonus_diagnostic"] = components["effective_response_bonus"]
            components["target_tracking_error_penalty_train_included"] = (
                1.0 if dso_cfg.enable_target_tracking_cost else 0.0
            )
            components["comfort_violation_penalty_train_included"] = float(dso_cfg.comfort_violation_weight)
            components["soc_violation_penalty_train_included"] = float(dso_cfg.soc_violation_weight)
            components["envelope_width_penalty_train_included"] = float(dso_cfg.envelope_width_penalty_weight)
            components["dso_reward_env"] = (
                feasibility_bonus
                + tracking_bonus
                + effective_bonus
                + components["dso_safe_capacity_utilization_reward"]
                - components["dso_flex_procurement_cost"]
                - components["dso_loss_cost"]
                - components["dso_curtailment_cost"]
                - components["dso_smoothness_penalty"]
                - components["dso_safety_margin_penalty"]
                - components["dso_hard_violation_penalty"]
                - components["dso_powerflow_failure_penalty"]
                - components["dso_responsible_projection_penalty"]
            )
            components["dso_reward_train"] = components["dso_reward_env"]
            components["dso_reward_critic_scaled"] = (
                components["dso_reward_train"] * float(self.reward_config.critic_reward_scale)
            )
            components["dso_reward"] = components["dso_reward_env"]
        else:
            components["dso_reward_cost_basis"] = components["scaled_total_cost"]
            components["tracking_bonus_diagnostic"] = components["tracking_bonus"]
            components["effective_response_bonus_diagnostic"] = components["effective_response_bonus"]
            components["target_tracking_error_penalty_train_included"] = 1.0
            components["comfort_violation_penalty_train_included"] = self._reward_weight("comfort_violation_penalty")
            components["soc_violation_penalty_train_included"] = self._reward_weight("soc_violation_penalty")
            components["envelope_width_penalty_train_included"] = self._reward_weight("envelope_width_penalty")
            components["dso_reward"] = (
                -float(self.dso_reward_cost_scale) * components["dso_reward_cost_basis"]
                + components["feasibility_bonus"]
                + components["tracking_bonus"]
                + components["effective_response_bonus"]
            )
            components["dso_reward_env"] = components["dso_reward"]
            components["dso_reward_train"] = components["dso_reward"]
            components["dso_reward_critic_scaled"] = (
                components["dso_reward_train"] * float(self.reward_config.critic_reward_scale)
            )
        components["scaled_reward"] = components["dso_reward"]
        # Backward-compatible single-agent reward. Multi-agent environments
        # split this into DSO, VPP dispatch and VPP portfolio rewards.
        components["reward"] = components["dso_reward"]
        return components

    def _dso_v3_market_safety_components(
        self,
        *,
        report: ConstraintReport,
        raw_network_penalties: dict[str, float],
        vpp_settlement_summaries: dict[str, dict[str, Any]] | None,
        raw_action_voltage_violation_cost: float,
        raw_action_line_overload_cost: float,
        raw_action_trafo_overload_cost: float,
        raw_action_powerflow_failed: float,
        projected_action_voltage_violation_cost: float | None,
        projected_action_line_overload_cost: float | None,
        projected_action_trafo_overload_cost: float | None,
        projected_action_powerflow_failed: float | None,
    ) -> dict[str, float]:
        cfg = self.reward_config.dso
        raw_voltage = max(0.0, float(raw_action_voltage_violation_cost))
        raw_line = max(0.0, float(raw_action_line_overload_cost))
        raw_trafo = max(0.0, float(raw_action_trafo_overload_cost))
        raw_pf = max(0.0, float(raw_action_powerflow_failed))
        projected_voltage = max(
            0.0,
            float(
                raw_network_penalties.get("voltage_violation_penalty", 0.0)
                if projected_action_voltage_violation_cost is None
                else projected_action_voltage_violation_cost
            ),
        )
        projected_line = max(
            0.0,
            float(
                raw_network_penalties.get("line_overload_penalty", 0.0)
                if projected_action_line_overload_cost is None
                else projected_action_line_overload_cost
            ),
        )
        projected_trafo = max(
            0.0,
            float(
                raw_network_penalties.get("transformer_overload_penalty", 0.0)
                if projected_action_trafo_overload_cost is None
                else projected_action_trafo_overload_cost
            ),
        )
        projected_pf = max(
            0.0,
            float((0.0 if report.converged else 1.0) if projected_action_powerflow_failed is None else projected_action_powerflow_failed),
        )
        raw_total = raw_voltage + raw_line + raw_trafo + raw_pf
        projected_total = projected_voltage + projected_line + projected_trafo + projected_pf
        raw_penalty_input = raw_total
        if raw_total > float(cfg.raw_safety_epsilon):
            raw_penalty_input += float(cfg.min_raw_unsafe_penalty)
        mode = str(cfg.safety_gate_input_mode).lower()
        if mode == "sum_raw_projected":
            gate_input = raw_penalty_input + projected_total
        else:
            gate_input = max(raw_penalty_input, projected_total)
        safety_gate = math.exp(-float(cfg.soft_safety_gate_kappa) * gate_input)

        summaries = dict(vpp_settlement_summaries or {})
        welfare_raw = 0.0
        transfer_excluded = 0.0
        audit_complete_values: list[float] = []
        balance_ok_values: list[float] = []
        for summary in summaries.values():
            welfare_raw += float(summary.get("operational_surplus", summary.get("private_profit", 0.0)))
            transfer_excluded += float(summary.get("dso_transfer_payment_cost", 0.0))
            transfer_excluded += float(summary.get("service_payment", 0.0))
            transfer_excluded += float(summary.get("availability_payment", 0.0))
            audit_complete_values.append(float(summary.get("settlement_audit_complete", 0.0)))
            balance_ok_values.append(float(summary.get("settlement_power_balance_ok", 0.0)))
        baseline_std = max(1.0e-9, float(cfg.welfare_baseline_std))
        welfare_normalized = (welfare_raw - float(cfg.welfare_baseline_mean)) / baseline_std
        welfare_clipped = min(float(cfg.welfare_clip), max(-float(cfg.welfare_clip), welfare_normalized))
        welfare_reward = float(cfg.welfare_weight) * safety_gate * welfare_clipped
        raw_safety_penalty = float(cfg.raw_action_safety_weight) * raw_penalty_input
        projected_safety_penalty = float(cfg.projected_action_safety_weight) * projected_total
        reward = welfare_reward - raw_safety_penalty - projected_safety_penalty
        audit_complete = sum(audit_complete_values) / len(audit_complete_values) if audit_complete_values else 0.0
        balance_ok = sum(balance_ok_values) / len(balance_ok_values) if balance_ok_values else 0.0
        return {
            "dso_reward_cost_basis": float(gate_input),
            "dso_reward_train": float(reward),
            "dso_reward_env": float(reward),
            "dso_reward": float(reward),
            "dso_reward_critic_scaled": float(reward) * float(self.reward_config.critic_reward_scale),
            "dso_vpp_welfare_raw": float(welfare_raw),
            "dso_vpp_welfare_normalized": float(welfare_normalized),
            "dso_vpp_welfare_clipped": float(welfare_clipped),
            "dso_vpp_welfare_reward": float(welfare_reward),
            "dso_transfer_payment_excluded": float(transfer_excluded),
            "dso_safety_gate": float(safety_gate),
            "dso_safety_gate_input": float(gate_input),
            "raw_action_safety_cost_norm": float(raw_total),
            "raw_action_safety_penalty_input": float(raw_penalty_input),
            "dso_raw_action_safety_penalty": float(raw_safety_penalty),
            "projected_action_safety_cost_norm": float(projected_total),
            "dso_projected_action_safety_penalty": float(projected_safety_penalty),
            "raw_action_voltage_violation_cost": float(raw_voltage),
            "raw_action_line_overload_cost": float(raw_line),
            "raw_action_trafo_overload_cost": float(raw_trafo),
            "raw_action_powerflow_failed": float(raw_pf),
            "projected_action_voltage_violation_cost": float(projected_voltage),
            "projected_action_line_overload_cost": float(projected_line),
            "projected_action_trafo_overload_cost": float(projected_trafo),
            "projected_action_powerflow_failed": float(projected_pf),
            "cmdp_cost_voltage": float(max(raw_voltage, projected_voltage)),
            "cmdp_cost_line_overload": float(max(raw_line, projected_line)),
            "cmdp_cost_trafo_overload": float(max(raw_trafo, projected_trafo)),
            "cmdp_cost_powerflow_failed": float(max(raw_pf, projected_pf)),
            "cmdp_cost_safety_total": float(max(raw_voltage, projected_voltage) + max(raw_line, projected_line) + max(raw_trafo, projected_trafo) + max(raw_pf, projected_pf)),
            "settlement_audit_complete": float(audit_complete),
            "settlement_power_balance_ok": float(balance_ok),
        }

    def _dso_v2_diagnostics(
        self,
        *,
        report: ConstraintReport,
        procurement_proxy_cost: float,
        projection_gap: float,
        ac_projection_gap: float,
        ac_certified_gap: float,
        mean_envelope_width_ratio: float,
        envelope_smoothness_mw: float,
        post_ac_violation_count: int,
        post_ac_powerflow_failed: int,
        raw_network_penalties: dict[str, float],
    ) -> dict[str, float]:
        cfg = self.reward_config.dso
        voltage_guard, line_guard, trafo_guard, voltage_margin, line_margin, trafo_margin = self._guard_band_penalties()
        safety_margin_raw = voltage_guard + line_guard + trafo_guard
        hard_violation_raw = (
            float(post_ac_violation_count)
            + float(raw_network_penalties.get("voltage_violation_penalty", 0.0))
            + float(raw_network_penalties.get("line_overload_penalty", 0.0)) / 100.0
            + float(raw_network_penalties.get("transformer_overload_penalty", 0.0)) / 100.0
        )
        line_loss_cost = self._network_loss_cost()
        safe_capacity_utilization = max(0.0, min(1.0, float(mean_envelope_width_ratio)))
        over_conservative = max(0.0, 1.0 - safe_capacity_utilization) if report.ok else 0.0
        dso_projection_gap = max(0.0, float(ac_projection_gap) + float(ac_certified_gap))
        dso_projection_penalty = (
            float(cfg.projection_gap_weight)
            * self._bounded_training_penalty(5.0 * dso_projection_gap + 10.0 * dso_projection_gap * dso_projection_gap)
        )
        flex_procurement = (
            float(cfg.flex_procurement_cost_weight)
            * self._bounded_training_penalty(float(procurement_proxy_cost) / 1000.0)
        )
        loss_cost = (
            float(cfg.loss_cost_weight)
            * self._bounded_training_penalty(float(line_loss_cost) / 1000.0)
        )
        curtailment_cost = float(cfg.curtailment_cost_weight) * over_conservative
        smoothness_penalty = (
            float(cfg.smoothness_weight)
            * self._bounded_training_penalty(float(envelope_smoothness_mw))
        )
        safety_margin_penalty = (
            float(cfg.safety_margin_weight)
            * self._bounded_training_penalty(safety_margin_raw)
        )
        hard_violation_penalty = (
            float(cfg.hard_violation_weight)
            * self._bounded_training_penalty(hard_violation_raw)
        )
        powerflow_failure_penalty = (
            float(cfg.powerflow_failure_weight)
            * float(post_ac_powerflow_failed)
        )
        safe_capacity_reward = float(cfg.safe_capacity_utilization_weight) * safe_capacity_utilization
        training_cost_basis = (
            flex_procurement
            + loss_cost
            + curtailment_cost
            + smoothness_penalty
            + safety_margin_penalty
            + hard_violation_penalty
            + powerflow_failure_penalty
            + dso_projection_penalty
        )
        return {
            "dso_safety_margin_unclipped": float(safety_margin_raw),
            "dso_safety_margin_penalty": float(safety_margin_penalty),
            "dso_voltage_guard_penalty": float(voltage_guard),
            "dso_line_guard_penalty": float(line_guard),
            "dso_trafo_guard_penalty": float(trafo_guard),
            "dso_voltage_min_margin_pu": float(voltage_margin),
            "dso_line_min_margin_percent": float(line_margin),
            "dso_trafo_min_margin_percent": float(trafo_margin),
            "dso_hard_violation_unclipped": float(hard_violation_raw),
            "dso_hard_violation_penalty": float(hard_violation_penalty),
            "dso_powerflow_failure_penalty": float(powerflow_failure_penalty),
            "dso_flex_procurement_cost": float(flex_procurement),
            "dso_loss_cost": float(loss_cost),
            "dso_curtailment_cost": float(curtailment_cost),
            "dso_safe_capacity_utilization": float(safe_capacity_utilization),
            "dso_safe_capacity_utilization_reward": float(safe_capacity_reward),
            "dso_over_conservative_curtailment_penalty": float(over_conservative),
            "dso_smoothness_penalty": float(smoothness_penalty),
            "dso_responsible_projection_gap_mw": float(dso_projection_gap),
            "dso_responsible_projection_penalty": float(dso_projection_penalty),
            "dso_v2_training_cost_basis": float(training_cost_basis),
            "target_tracking_error_to_raw_target": 0.0,
            "target_tracking_error_to_projected_target": 0.0,
        }

    def _guard_band_penalties(self) -> tuple[float, float, float, float, float, float]:
        cfg = self.reward_config.dso
        vmin, vmax = self.voltage_limits
        voltage_guard = 0.0
        line_guard = 0.0
        trafo_guard = 0.0
        min_voltage_margin = float("inf")
        min_line_margin = float("inf")
        min_trafo_margin = float("inf")
        if hasattr(self.net, "res_bus") and "vm_pu" in self.net.res_bus:
            for vm_pu in self.net.res_bus["vm_pu"]:
                value = float(vm_pu)
                margin = min(value - float(vmin), float(vmax) - value)
                min_voltage_margin = min(min_voltage_margin, margin)
                voltage_guard += max(0.0, float(cfg.voltage_guard_band_pu) - margin) ** 2
        if hasattr(self.net, "res_line") and "loading_percent" in self.net.res_line:
            for loading in self.net.res_line["loading_percent"]:
                value = float(loading)
                margin = float(self.line_loading_limit_percent) - value
                min_line_margin = min(min_line_margin, margin)
                line_guard += max(0.0, float(cfg.line_guard_band_percent) - margin) ** 2
        if hasattr(self.net, "res_trafo") and "loading_percent" in self.net.res_trafo:
            for loading in self.net.res_trafo["loading_percent"]:
                value = float(loading)
                margin = float(self.trafo_loading_limit_percent) - value
                min_trafo_margin = min(min_trafo_margin, margin)
                trafo_guard += max(0.0, float(cfg.trafo_guard_band_percent) - margin) ** 2
        return (
            float(voltage_guard),
            float(line_guard),
            float(trafo_guard),
            0.0 if min_voltage_margin == float("inf") else float(min_voltage_margin),
            0.0 if min_line_margin == float("inf") else float(min_line_margin),
            0.0 if min_trafo_margin == float("inf") else float(min_trafo_margin),
        )

    def _network_loss_cost(self) -> float:
        price = float(self.market_price_profile[0]) if self.market_price_profile else 80.0
        losses = 0.0
        if hasattr(self.net, "res_line") and "pl_mw" in self.net.res_line:
            losses += float(self.net.res_line["pl_mw"].sum())
        if hasattr(self.net, "res_trafo") and "pl_mw" in self.net.res_trafo:
            losses += float(self.net.res_trafo["pl_mw"].sum())
        return max(0.0, losses) * price

    def reset(self) -> None:
        for vpp in self.vpp_registry.values():
            vpp.reset()
