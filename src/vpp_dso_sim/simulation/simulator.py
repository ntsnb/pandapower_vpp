from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd
import numpy as np

from vpp_dso_sim.der.evcs import EVCSModel
from vpp_dso_sim.der.hvac import HVACModel
from vpp_dso_sim.der.storage import StorageModel
from vpp_dso_sim.network.constraints import ConstraintReport
from vpp_dso_sim.network.powerflow import scale_base_loads
from vpp_dso_sim.network.sensitivity import compute_vpp_active_power_sensitivity
from vpp_dso_sim.optimization.ac_security_projection import certify_or_repair_dispatch
from vpp_dso_sim.optimization.baselines import price_driven_target
from vpp_dso_sim.optimization.feasibility_region import (
    compute_static_feasible_region,
    scalar_target_to_vector_targets,
    current_power_by_fr_scope,
    project_scalar_target_to_feasible_region,
    project_vector_target_to_feasible_region,
)
from vpp_dso_sim.simulation.portfolio_events import apply_portfolio_event
from vpp_dso_sim.simulation.scenario import SimulationScenario
from vpp_dso_sim.utils.io import ensure_dir, write_json


@dataclass
class Simulator:
    scenario: SimulationScenario
    records: dict[str, list[dict[str, Any]]] = field(default_factory=dict)
    current_step: int = 0
    _initial_net_tables: dict[str, pd.DataFrame] = field(init=False, repr=False)
    _initial_der_states: dict[str, dict[str, Any]] = field(init=False, repr=False)
    _initial_vpp_der_ids: dict[str, list[str]] = field(init=False, repr=False)
    _der_objects_by_id: dict[str, Any] = field(init=False, repr=False)
    _initial_vpp_metadata: dict[str, dict[str, Any]] = field(init=False, repr=False)
    _applied_portfolio_event_ids: set[str] = field(default_factory=set, init=False, repr=False)

    def __post_init__(self) -> None:
        self._capture_initial_state()

    def _capture_initial_state(self) -> None:
        self._initial_net_tables = {}
        for table_name in (
            "load",
            "sgen",
            "storage",
            "gen",
            "ext_grid",
            "bus",
            "line",
            "trafo",
            "res_bus",
            "res_line",
            "res_trafo",
            "res_ext_grid",
            "res_load",
            "res_sgen",
            "res_storage",
        ):
            if hasattr(self.scenario.net, table_name):
                self._initial_net_tables[table_name] = getattr(self.scenario.net, table_name).copy(deep=True)
        self._initial_der_states = {}
        self._initial_vpp_der_ids = {}
        self._der_objects_by_id = {}
        self._initial_vpp_metadata = {}
        for vpp in self.scenario.vpps:
            self._initial_vpp_der_ids[vpp.id] = [der.id for der in vpp.der_list]
            self._initial_vpp_metadata[vpp.id] = deepcopy(vpp.metadata)
            for der in vpp.der_list:
                self._initial_der_states[der.id] = deepcopy(der.__dict__)
                self._der_objects_by_id[der.id] = der

    def _restore_initial_state(self) -> None:
        for table_name, frame in self._initial_net_tables.items():
            self.scenario.net[table_name] = frame.copy(deep=True)
        for vpp in self.scenario.vpps:
            vpp.der_list = [
                self._der_objects_by_id[der_id]
                for der_id in self._initial_vpp_der_ids.get(vpp.id, [])
                if der_id in self._der_objects_by_id
            ]
            vpp.metadata = deepcopy(self._initial_vpp_metadata.get(vpp.id, {}))
            for der in vpp.der_list:
                state = self._initial_der_states.get(der.id)
                if state is not None:
                    der.__dict__.clear()
                    der.__dict__.update(deepcopy(state))
        self.scenario.net.converged = False
        self._applied_portfolio_event_ids.clear()

    def reset(self) -> None:
        self._restore_initial_state()
        self.current_step = 0
        self.records = {
            "bus_voltage": [],
            "line_loading": [],
            "trafo_loading": [],
            "edge_power_flow": [],
            "profile_state": [],
            "vpp_portfolio_history": [],
            "portfolio_change_log": [],
            "vpp_day_ahead_bid": [],
            "dso_operating_envelope": [],
            "fr_envelope_state": [],
            "projection_trace": [],
            "vpp_rl_disaggregation": [],
            "vpp_power": [],
            "der_dispatch": [],
            "storage_soc": [],
            "evcs_soc": [],
            "hvac_temperature": [],
            "constraint_violations": [],
            "reward_components": [],
        }
        self.scenario.dso.reset()

    def _profile_value(self, values: list[float], t: int) -> float:
        return float(values[t % len(values)])

    def step(self, t: int | None = None, actions: dict[str, float] | None = None) -> dict[str, Any]:
        if not self.records:
            self.reset()
        step = self.current_step if t is None else t
        self.current_step = step + 1

        scenario = self.scenario
        price = self._profile_value(scenario.price_profile, step)
        load_scale = self._profile_value(scenario.load_profile, step)
        pv_forecast_factor = self._profile_value(scenario.pv_profile, step)
        scale_base_loads(scenario.net, load_scale)
        self._apply_portfolio_events(step)
        self._record_portfolio_history(step)
        # Build the DSO envelope from the latest known grid state, not only
        # from price and local VPP bounds. This pre-dispatch run is the
        # benchmark control layer's view of the system before new instructions.
        pre_dispatch_powerflow_converged = scenario.dso.run_powerflow()
        pre_dispatch_grid_state = scenario.dso.compute_network_state()
        pre_dispatch_grid_state["pre_dispatch_powerflow_converged"] = bool(pre_dispatch_powerflow_converged)

        tracking_error = 0.0
        action_projection_gap_mw = 0.0
        local_bounds_projection_gap_mw = 0.0
        ac_aware_projection_gap_mw = 0.0
        ac_certified_projection_gap_mw = 0.0
        action_projection_count = 0
        powerflow_projection_rows: list[dict[str, Any]] = []
        candidate_dispatch_by_vpp: dict[str, dict[str, float]] = {}
        candidate_rows: list[dict[str, Any]] = []
        for vpp in scenario.vpps:
            bid = vpp.day_ahead_bid(step, price_hint=price)
            fr = compute_static_feasible_region(vpp, step)
            envelope = self._build_dso_operating_envelope(
                vpp,
                step,
                bid,
                fr,
                price,
                grid_state=pre_dispatch_grid_state,
            )
            self.records["vpp_day_ahead_bid"].append({**bid, "time_label": self._time_label(step)})
            self.records["dso_operating_envelope"].append(envelope)
            (
                raw_target,
                command_source,
                der_actions,
                action_mode,
                pre_projection_gap,
                pre_projection_count,
                explicit_der_dispatch,
                explicit_scope_targets,
            ) = self._resolve_vpp_action(
                vpp=vpp,
                step=step,
                price=price,
                envelope=envelope,
                actions=actions,
            )
            if explicit_scope_targets is not None and fr.scope != "pcc":
                vector_raw = current_power_by_fr_scope(vpp, fr)
                vector_raw.update(explicit_scope_targets)
                projected = project_vector_target_to_feasible_region(fr, vector_raw)
                vector_targets = {key: float(value[0]) for key, value in projected.items()}
                local_target = float(sum(vector_targets.values()))
                ac_projected_target = local_target
                raw_target = float(sum(vector_raw.get(key, 0.0) for key in fr.bounds))
                local_projection_gap = float(
                    sum(abs(float(vector_targets[key]) - float(vector_raw.get(key, 0.0))) for key in fr.bounds)
                )
                ac_projection_gap = 0.0
                target_projection_gap = local_projection_gap
            else:
                local_target, _ = project_scalar_target_to_feasible_region(fr, raw_target, 0.0)
                ac_projected_target = self._project_target_to_ac_aware_envelope(local_target, envelope)
                local_projection_gap = abs(float(local_target) - float(raw_target))
                ac_projection_gap = abs(float(ac_projected_target) - float(local_target))
                target_projection_gap = abs(float(ac_projected_target) - float(raw_target))
                vector_targets = scalar_target_to_vector_targets(vpp, fr, ac_projected_target)
            action_projection_gap_mw += float(pre_projection_gap) + target_projection_gap
            local_bounds_projection_gap_mw += float(pre_projection_gap) + local_projection_gap
            ac_aware_projection_gap_mw += ac_projection_gap
            action_projection_count += int(pre_projection_count) + int(target_projection_gap > 1e-9)
            self._record_projection_trace_prefix(
                step=step,
                vpp=vpp,
                fr_id=fr.fr_id,
                command_source=command_source,
                raw_target=raw_target,
                projected_target=local_target,
                ac_projected_target=ac_projected_target,
                envelope=envelope,
            )
            self._record_vector_projection_trace(
                step=step,
                vpp=vpp,
                fr=fr,
                command_source=command_source,
                targets_by_scope=vector_targets,
            )
            if explicit_der_dispatch is not None:
                dispatch = self._clip_explicit_der_dispatch(vpp, explicit_der_dispatch, step)
                target = float(sum(dispatch.values()))
                der_actions = dispatch
            else:
                target = ac_projected_target
            if explicit_der_dispatch is None and fr.scope == "pcc":
                dispatch = vpp.disaggregate_power_target(target, 0.0, step, der_actions=der_actions)
            elif explicit_der_dispatch is None:
                dispatch = vpp.disaggregate_power_targets_by_scope(vector_targets, step, der_actions=der_actions)
            candidate_dispatch_by_vpp[str(vpp.id)] = dispatch
            self._record_fr_envelope_state(step, vpp, fr)
            candidate_rows.append(
                {
                    "vpp": vpp,
                    "fr": fr,
                    "command_source": command_source,
                    "action_mode": action_mode,
                    "target": target,
                    "candidate_dispatch": dispatch,
                    "der_actions": der_actions,
                }
            )

        certificate = certify_or_repair_dispatch(
            base_net=scenario.net,
            vpps=scenario.vpps,
            candidate_dispatch_by_vpp=candidate_dispatch_by_vpp,
            t=step,
            voltage_limits=scenario.dso.voltage_limits,
            line_loading_limit_percent=scenario.dso.line_loading_limit_percent,
            trafo_loading_limit_percent=scenario.dso.trafo_loading_limit_percent,
        )
        ac_certified_projection_gap_mw = float(certificate.repair_gap_mw)
        if ac_certified_projection_gap_mw > 1e-9:
            action_projection_gap_mw += ac_certified_projection_gap_mw
            action_projection_count += 1

        for item in candidate_rows:
            vpp = item["vpp"]
            fr = item["fr"]
            command_source = str(item["command_source"])
            target = float(item["target"])
            repaired_dispatch = certificate.dispatch_by_vpp.get(str(vpp.id), item["candidate_dispatch"])
            vpp.apply_dispatch_to_net(scenario.net, repaired_dispatch, step)
            self._record_vpp_rl_disaggregation(
                step=step,
                vpp=vpp,
                command_source=command_source,
                action_mode=str(item["action_mode"]),
                target=target,
                dispatch=repaired_dispatch,
                der_actions=item["der_actions"],
            )
            self._record_projection_trace_writes(
                step=step,
                vpp=vpp,
                fr_id=fr.fr_id,
                command_source=command_source,
                dispatch=repaired_dispatch,
            )
            self._record_ac_certificate_trace(
                step=step,
                vpp=vpp,
                fr_id=fr.fr_id,
                command_source=command_source,
                candidate_dispatch=item["candidate_dispatch"],
                repaired_dispatch=repaired_dispatch,
                certificate=certificate.to_dict(),
            )
            actual = float(sum(repaired_dispatch.get(der.id, der.p_mw) for der in vpp.der_list))
            tracking_error += abs(actual - target)
            powerflow_projection_rows.append(
                self._projection_trace_row(
                    step=step,
                    vpp_id=vpp.id,
                    fr_id=fr.fr_id,
                    command_source=command_source,
                    stage_order=8,
                    stage_name="powerflow_result",
                    p_mw=actual,
                    q_mvar=vpp.current_reactive_power_mvar(),
                    p_lower_mw=fr.aggregate_bounds().p_min_mw,
                    p_upper_mw=fr.aggregate_bounds().p_max_mw,
                    q_lower_mvar=fr.aggregate_bounds().q_min_mvar,
                    q_upper_mvar=fr.aggregate_bounds().q_max_mvar,
                    active_constraint="",
                    projection_reason="post-power-flow aggregate delivered response after AC-certified dispatch",
                    projection_gap_scope="post_ac_security_audit",
                )
            )

        scenario.dso.run_powerflow()
        report = scenario.dso.check_security_constraints()
        post_ac_violation_records = report.to_records(step)
        for row in powerflow_projection_rows:
            row["post_ac_violation_count"] = float(len(post_ac_violation_records))
            row["post_ac_security_ok"] = bool(report.ok)
            row["post_ac_powerflow_converged"] = bool(report.converged)
            row.update(certificate.to_dict())
        self.records["projection_trace"].extend(powerflow_projection_rows)

        for vpp in scenario.vpps:
            vpp.update_dynamic_states(step, scenario.dt_hours)

        reward_components = scenario.dso.calculate_reward_or_cost(
            report,
            tracking_error,
            step,
            action_projection_gap_mw=action_projection_gap_mw,
            local_bounds_projection_gap_mw=local_bounds_projection_gap_mw,
            ac_aware_projection_gap_mw=ac_aware_projection_gap_mw,
            ac_certified_projection_gap_mw=ac_certified_projection_gap_mw,
            ac_certificate_failed_count=int(not certificate.ac_safe),
            action_projection_count=action_projection_count,
        )
        self._record_step(step, report, reward_components, price, load_scale, pv_forecast_factor)
        return {
            "step": step,
            "price": price,
            "converged": report.converged,
            "violations": report.to_records(step),
            "reward_components": reward_components,
        }

    def _time_label(self, step: int) -> str:
        return f"{float(step) * float(self.scenario.dt_hours):05.2f} h"

    def _project_target_to_ac_aware_envelope(self, target_p_mw: float, envelope: dict[str, Any]) -> float:
        return float(
            max(
                float(envelope.get("p_min_mw", target_p_mw)),
                min(float(envelope.get("p_max_mw", target_p_mw)), float(target_p_mw)),
            )
        )

    def _tighten_dso_envelope_with_ac_sensitivity(
        self,
        *,
        vpp,
        step: int,
        p_min: float,
        p_max: float,
        preferred_low: float,
        preferred_high: float,
        preferred_target: float,
        min_vm: float,
        max_vm: float,
        max_line: float,
        low_warning: float,
        high_warning: float,
        line_warning: float,
    ) -> tuple[float, float, float, float, float, dict[str, Any]]:
        original_p_min = float(p_min)
        original_p_max = float(p_max)
        current_p = float(vpp.current_power_mw())
        near_voltage = min_vm < low_warning or max_vm > high_warning
        near_line = max_line > line_warning
        audit: dict[str, Any] = {
            "ac_aware_enabled": False,
            "ac_aware_status": "not_near_ac_limit",
            "ac_aware_reason": "",
            "ac_aware_original_p_min_mw": original_p_min,
            "ac_aware_original_p_max_mw": original_p_max,
            "ac_aware_shrink_lower_mw": 0.0,
            "ac_aware_shrink_upper_mw": 0.0,
            "ac_aware_projection_method": "finite_difference_sensitivity",
        }
        if not (near_voltage or near_line):
            return p_min, p_max, preferred_low, preferred_high, preferred_target, audit

        sensitivity = compute_vpp_active_power_sensitivity(self.scenario.net, vpp, t=step)
        audit.update({f"ac_aware_{key}": value for key, value in sensitivity.items() if not isinstance(value, dict)})
        if sensitivity.get("status") != "ok":
            audit["ac_aware_status"] = str(sensitivity.get("status", "sensitivity_unavailable"))
            return p_min, p_max, preferred_low, preferred_high, preferred_target, audit

        reasons: list[str] = []
        guard = 1.10
        eps = 1e-9
        max_single_step_shrink = 0.50 * max(0.0, original_p_max - original_p_min)

        def tighten_lower(required_min: float, reason: str) -> None:
            nonlocal p_min
            conservative_cap = original_p_min + max_single_step_shrink
            bounded = min(conservative_cap, original_p_max, max(original_p_min, float(required_min)))
            if bounded > p_min + 1e-9:
                p_min = bounded
                reasons.append(reason)

        def tighten_upper(required_max: float, reason: str) -> None:
            nonlocal p_max
            conservative_floor = original_p_max - max_single_step_shrink
            bounded = max(conservative_floor, original_p_min, min(original_p_max, float(required_max)))
            if bounded < p_max - 1e-9:
                p_max = bounded
                reasons.append(reason)

        def sensitivity_value(*keys: str) -> float:
            values = [float(sensitivity.get(key, 0.0) or 0.0) for key in keys]
            return max(values, key=lambda value: abs(value)) if values else 0.0

        if min_vm < low_warning:
            sensitivity_up = sensitivity_value(
                "increase_min_bus_vm_pu_per_mw",
                "increase_min_connection_bus_vm_pu_per_mw",
                "increase_pcc_bus_vm_pu_per_mw",
            )
            sensitivity_down = sensitivity_value(
                "decrease_min_bus_vm_pu_per_mw",
                "decrease_min_connection_bus_vm_pu_per_mw",
                "decrease_pcc_bus_vm_pu_per_mw",
            )
            required_voltage_lift = float(low_warning - min_vm)
            if sensitivity_up > eps:
                tighten_lower(
                    current_p + guard * required_voltage_lift / sensitivity_up,
                    "low_voltage_requires_more_injection_or_less_load",
                )
            elif sensitivity_down < -eps:
                tighten_upper(
                    current_p - guard * required_voltage_lift / abs(sensitivity_down),
                    "low_voltage_requires_less_injection_or_more_load",
                )

        if max_vm > high_warning:
            sensitivity_up = sensitivity_value(
                "increase_max_bus_vm_pu_per_mw",
                "increase_max_connection_bus_vm_pu_per_mw",
                "increase_pcc_bus_vm_pu_per_mw",
            )
            sensitivity_down = sensitivity_value(
                "decrease_max_bus_vm_pu_per_mw",
                "decrease_max_connection_bus_vm_pu_per_mw",
                "decrease_pcc_bus_vm_pu_per_mw",
            )
            required_voltage_drop = float(max_vm - high_warning)
            if sensitivity_down > eps:
                tighten_upper(
                    current_p - guard * required_voltage_drop / sensitivity_down,
                    "high_voltage_requires_less_injection_or_more_load",
                )
            elif sensitivity_up < -eps:
                tighten_lower(
                    current_p + guard * required_voltage_drop / abs(sensitivity_up),
                    "high_voltage_requires_more_injection_or_less_load",
                )

        if max_line > line_warning:
            sensitivity_up = float(sensitivity.get("increase_critical_line_loading_percent_per_mw", 0.0))
            if abs(sensitivity_up) > eps:
                available_margin = float(line_warning - max_line)
                if sensitivity_up > 0.0:
                    tighten_upper(
                        current_p + available_margin / sensitivity_up / guard,
                        "thermal_limit_caps_more_injection_or_less_load",
                    )
                else:
                    tighten_lower(
                        current_p + available_margin / sensitivity_up / guard,
                        "thermal_limit_caps_less_injection_or_more_load",
                    )

        if p_min > p_max:
            collapse = max(original_p_min, min(original_p_max, 0.5 * (p_min + p_max)))
            p_min = p_max = collapse
            reasons.append("conflicting_ac_sensitivity_limits_collapsed_envelope")

        preferred_low = max(p_min, min(p_max, preferred_low))
        preferred_high = max(p_min, min(p_max, preferred_high))
        if preferred_low > preferred_high:
            midpoint = max(p_min, min(p_max, preferred_target))
            preferred_low = preferred_high = midpoint
        preferred_target = max(preferred_low, min(preferred_high, preferred_target))

        audit.update(
            {
                "ac_aware_enabled": bool(reasons),
                "ac_aware_status": "tightened" if reasons else "sensitivity_no_tightening",
                "ac_aware_reason": ";".join(reasons),
                "ac_aware_p_min_mw": float(p_min),
                "ac_aware_p_max_mw": float(p_max),
                "ac_aware_shrink_lower_mw": float(max(0.0, p_min - original_p_min)),
                "ac_aware_shrink_upper_mw": float(max(0.0, original_p_max - p_max)),
            }
        )
        return p_min, p_max, preferred_low, preferred_high, preferred_target, audit

    def _build_dso_operating_envelope(
        self,
        vpp,
        step: int,
        bid: dict[str, Any],
        fr,
        price: float,
        grid_state: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        bounds = fr.aggregate_bounds()
        p_min = float(max(bounds.p_min_mw, float(bid.get("p_min_mw", bounds.p_min_mw))))
        p_max = float(min(bounds.p_max_mw, float(bid.get("p_max_mw", bounds.p_max_mw))))
        if p_min > p_max:
            p_min = p_max = 0.5 * (p_min + p_max)
        span = max(0.0, p_max - p_min)
        state = grid_state or self.scenario.dso.compute_network_state()
        pre_dispatch_converged = bool(
            state.get("pre_dispatch_powerflow_converged", state.get("converged", True))
        )
        vmin_limit, vmax_limit = self.scenario.dso.voltage_limits
        low_warning = float(vmin_limit) + 0.010
        high_warning = float(vmax_limit) - 0.010
        line_warning = 0.92 * float(self.scenario.dso.line_loading_limit_percent)
        trafo_warning = 0.92 * float(self.scenario.dso.trafo_loading_limit_percent)
        min_vm = float(state.get("min_vm_pu", 1.0))
        max_vm = float(state.get("max_vm_pu", 1.0))
        max_line = float(state.get("max_line_loading_percent", 0.0))
        max_trafo = float(state.get("max_trafo_loading_percent", 0.0))
        if not pre_dispatch_converged:
            min_vm = max_vm = 1.0
            max_line = max_trafo = 0.0
        pcc_vm = None
        if pre_dispatch_converged and hasattr(self.scenario.net, "res_bus") and int(vpp.pcc_bus) in self.scenario.net.res_bus.index:
            pcc_vm = float(self.scenario.net.res_bus.at[int(vpp.pcc_bus), "vm_pu"])

        grid_pressure_mode = "price_economic"
        grid_priority_over_price = False
        if not pre_dispatch_converged:
            grid_pressure_mode = "pre_dispatch_powerflow_failed"
            service_request = "hold_current_dispatch"
            current = max(p_min, min(p_max, float(vpp.current_power_mw())))
            preferred_low = preferred_high = preferred_target = current
            grid_priority_over_price = True
        elif min_vm < low_warning:
            grid_pressure_mode = "low_voltage_support"
            service_request = "export_or_reduce_load"
            preferred_low = p_min + 0.65 * span
            preferred_high = p_max
            preferred_target = p_min + 0.85 * span
            grid_priority_over_price = True
        elif max_vm > high_warning:
            grid_pressure_mode = "high_voltage_absorption"
            service_request = "absorb_or_charge"
            preferred_low = p_min
            preferred_high = p_min + 0.35 * span
            preferred_target = p_min + 0.20 * span
            grid_priority_over_price = True
        elif max_line > line_warning or max_trafo > trafo_warning:
            grid_pressure_mode = "thermal_congestion_tighten"
            service_request = "balanced_operation"
            preferred_low = p_min + 0.40 * span
            preferred_high = p_min + 0.60 * span
            preferred_target = p_min + 0.50 * span
            grid_priority_over_price = True
        else:
            service_request = ""
            preferred_low = preferred_high = preferred_target = 0.0

        if not grid_priority_over_price and price <= 55.0:
            service_request = "absorb_or_charge"
            preferred_low = p_min
            preferred_high = p_min + 0.35 * span
            preferred_target = p_min + 0.20 * span
        elif not grid_priority_over_price and price >= 100.0:
            service_request = "export_or_reduce_load"
            preferred_low = p_min + 0.65 * span
            preferred_high = p_max
            preferred_target = p_min + 0.85 * span
        elif not grid_priority_over_price:
            service_request = "balanced_operation"
            preferred_low = p_min + 0.35 * span
            preferred_high = p_min + 0.65 * span
            preferred_target = p_min + 0.50 * span

        if pre_dispatch_converged:
            p_min, p_max, preferred_low, preferred_high, preferred_target, ac_aware_audit = (
                self._tighten_dso_envelope_with_ac_sensitivity(
                    vpp=vpp,
                    step=step,
                    p_min=p_min,
                    p_max=p_max,
                    preferred_low=preferred_low,
                    preferred_high=preferred_high,
                    preferred_target=preferred_target,
                    min_vm=min_vm,
                    max_vm=max_vm,
                    max_line=max_line,
                    low_warning=low_warning,
                    high_warning=high_warning,
                    line_warning=line_warning,
                )
            )
        else:
            ac_aware_audit = {
                "ac_aware_enabled": False,
                "ac_aware_status": "pre_dispatch_powerflow_failed",
                "ac_aware_reason": "stale_pre_dispatch_results_not_used",
                "ac_aware_original_p_min_mw": float(p_min),
                "ac_aware_original_p_max_mw": float(p_max),
                "ac_aware_shrink_lower_mw": 0.0,
                "ac_aware_shrink_upper_mw": 0.0,
                "ac_aware_projection_method": "disabled_due_to_pre_dispatch_powerflow_failure",
            }
        grid_priority_over_price = bool(grid_priority_over_price or ac_aware_audit.get("ac_aware_enabled", False))
        bounds_by_scope = {
            key: {
                "p_min_mw": float(item.p_min_mw),
                "p_max_mw": float(item.p_max_mw),
                "q_min_mvar": float(item.q_min_mvar),
                "q_max_mvar": float(item.q_max_mvar),
            }
            for key, item in fr.bounds.items()
        }
        current_p_by_scope = current_power_by_fr_scope(vpp, fr)
        preferred_target_by_scope = scalar_target_to_vector_targets(vpp, fr, preferred_target)
        return {
            "step": int(step),
            "time_label": self._time_label(step),
            "vpp_id": vpp.id,
            "fr_id": fr.fr_id,
            "source_bid": "vpp_day_ahead_bid",
            "p_min_mw": p_min,
            "p_max_mw": p_max,
            "q_min_mvar": float(bounds.q_min_mvar),
            "q_max_mvar": float(bounds.q_max_mvar),
            "preferred_p_min_mw": float(preferred_low),
            "preferred_p_max_mw": float(preferred_high),
            "preferred_target_p_mw": float(preferred_target),
            "service_request": service_request,
            "price": float(price),
            "grid_pressure_mode": grid_pressure_mode,
            "ac_aware_grid_pressure_mode": (
                f"ac_aware_{grid_pressure_mode}"
                if ac_aware_audit.get("ac_aware_enabled")
                else grid_pressure_mode
            ),
            "grid_priority_over_price": bool(grid_priority_over_price),
            "network_min_vm_pu": min_vm,
            "network_max_vm_pu": max_vm,
            "network_max_line_loading_percent": max_line,
            "network_max_trafo_loading_percent": max_trafo,
            "pre_dispatch_powerflow_converged": bool(pre_dispatch_converged),
            "pcc_vm_pu": pcc_vm,
            "voltage_low_warning_pu": low_warning,
            "voltage_high_warning_pu": high_warning,
            "bid_price_up": float(bid.get("bid_price_up", price)),
            "bid_price_down": float(bid.get("bid_price_down", price)),
            "confidence": float(bid.get("confidence", 0.0)),
            "dso_intent": (
                "ac_aware_security_tightened_envelope"
                if ac_aware_audit.get("ac_aware_enabled")
                else "grid_security_overrides_price_when_near_limits"
                if grid_priority_over_price
                else "envelope_guidance_from_bid_price_and_local_bounds"
            ),
            "fr_scope": str(fr.scope),
            "bounds_by_scope": bounds_by_scope,
            "current_p_by_scope": current_p_by_scope,
            "preferred_target_by_scope": preferred_target_by_scope,
            "vector_doe_enabled": bool(fr.scope != "pcc"),
            **ac_aware_audit,
        }

    def _resolve_vpp_action(
        self,
        vpp,
        step: int,
        price: float,
        envelope: dict[str, Any],
        actions: dict[str, Any] | None,
    ) -> tuple[
        float,
        str,
        dict[str, float] | list[float] | tuple[float, ...] | None,
        str,
        float,
        int,
        dict[str, float] | None,
        dict[str, float] | None,
    ]:
        if actions and vpp.id in actions:
            raw = actions[vpp.id]
            if isinstance(raw, dict):
                explicit = raw.get("der_dispatch_p_mw")
                explicit_dispatch = (
                    {str(key): float(value) for key, value in explicit.items()}
                    if isinstance(explicit, dict)
                    else None
                )
                scope_raw = raw.get("selected_p_by_scope", raw.get("target_by_bus"))
                scope_targets = self._normalize_scope_targets(scope_raw) if isinstance(scope_raw, dict) else None
                target = float(
                    raw.get(
                        "selected_p_mw",
                        raw.get(
                            "target_p_mw",
                            sum(scope_targets.values())
                            if scope_targets is not None
                            else raw.get("preferred_target_p_mw", envelope["preferred_target_p_mw"]),
                        ),
                    )
                )
                der_actions = raw.get("der_actions")
                command_source = str(raw.get("command_source", "vpp_rl_envelope_action"))
                action_mode = str(
                    raw.get(
                        "action_mode",
                        "explicit_der_dispatch"
                        if explicit_dispatch is not None
                        else "vector_scope_target"
                        if scope_targets is not None
                        else "learned_der_disaggregation"
                        if der_actions is not None
                        else "aggregate_target",
                    )
                )
                pre_gap = float(raw.get("pre_projection_gap_mw", 0.0))
                pre_count = int(bool(raw.get("pre_projection_clipped", False)))
                return target, command_source, der_actions, action_mode, pre_gap, pre_count, explicit_dispatch, scope_targets
            return float(raw), "external_action", None, "legacy_scalar_target", 0.0, 0, None, None
        return (
            float(envelope.get("preferred_target_p_mw", price_driven_target(vpp, step, price))),
            "price_driven_baseline",
            None,
            "baseline_envelope_target",
            0.0,
            0,
            None,
            None,
        )

    def _normalize_scope_targets(self, raw: dict[str, Any]) -> dict[str, float]:
        targets: dict[str, float] = {}
        for key, value in raw.items():
            text = str(key)
            normalized = text if "_" in text else f"bus_{int(float(text))}"
            targets[normalized] = float(value)
        return targets

    def _clip_explicit_der_dispatch(self, vpp, dispatch: dict[str, float], step: int) -> dict[str, float]:
        clipped: dict[str, float] = {}
        for der in vpp.der_list:
            der.metadata["current_t"] = step
            p_min, p_max, _, _ = der.get_bounds(step)
            requested = float(dispatch.get(str(der.id), dispatch.get(der.id, der.p_mw)))
            clipped[str(der.id)] = float(np.clip(requested, float(p_min), float(p_max)))
        return clipped

    def _record_vpp_rl_disaggregation(
        self,
        step: int,
        vpp,
        command_source: str,
        action_mode: str,
        target: float,
        dispatch: dict[str, float],
        der_actions: dict[str, float] | list[float] | tuple[float, ...] | None,
    ) -> None:
        action_by_der: dict[str, float | None] = {}
        if isinstance(der_actions, dict):
            action_by_der = {str(key): float(value) for key, value in der_actions.items()}
        elif der_actions is not None:
            action_by_der = {
                der.id: float(der_actions[index])
                for index, der in enumerate(vpp.der_list)
                if index < len(der_actions)
            }
        aggregate_dispatch = float(sum(dispatch.get(der.id, der.p_mw) for der in vpp.der_list))
        projection_gap = abs(aggregate_dispatch - float(target))
        action_values = [float(value) for value in action_by_der.values() if value is not None]
        saturation_rate = (
            float(sum(abs(value) >= 0.999 for value in action_values) / len(action_values))
            if action_values
            else 0.0
        )
        for der in vpp.der_list:
            p_min, p_max, _, _ = der.get_bounds(step)
            target_p = float(dispatch.get(der.id, der.p_mw))
            self.records["vpp_rl_disaggregation"].append(
                {
                    "step": int(step),
                    "time_label": self._time_label(step),
                    "vpp_id": vpp.id,
                    "der_id": der.id,
                    "command_source": command_source,
                    "action_mode": action_mode,
                    "aggregate_target_p_mw": float(target),
                    "aggregate_dispatch_p_mw": aggregate_dispatch,
                    "projection_gap_mw": projection_gap,
                    "local_bounds_projection_gap_mw": projection_gap,
                    "projection_gap_scope": "local_der_bounds_not_ac_security",
                    "normalized_der_action": action_by_der.get(der.id),
                    "action_saturation_rate": saturation_rate,
                    "der_target_p_mw": target_p,
                    "p_min_mw": float(p_min),
                    "p_max_mw": float(p_max),
                    "is_learned_der_action": der_actions is not None,
                    "fallback_triggered": False,
                }
            )

    def _projection_trace_row(
        self,
        step: int,
        vpp_id: str,
        fr_id: str,
        command_source: str,
        stage_order: int,
        stage_name: str,
        p_mw: float,
        q_mvar: float = 0.0,
        p_lower_mw: float | None = None,
        p_upper_mw: float | None = None,
        q_lower_mvar: float | None = None,
        q_upper_mvar: float | None = None,
        delta_p_mw: float = 0.0,
        delta_q_mvar: float = 0.0,
        was_projected: bool = False,
        active_constraint: str = "",
        projection_reason: str = "",
        projection_gap_scope: str | None = None,
        scope_type: str = "vpp",
        scope_id: str = "",
        pp_element_type: str = "",
        pp_element_index: int | None = None,
    ) -> dict[str, Any]:
        return {
            "trace_id": f"{vpp_id}_{step}_{stage_order}_{scope_id or stage_name}",
            "step": int(step),
            "time_label": self._time_label(step),
            "vpp_id": vpp_id,
            "fr_id": fr_id,
            "command_source": command_source,
            "stage_order": int(stage_order),
            "stage_name": stage_name,
            "scope_type": scope_type,
            "scope_id": scope_id or vpp_id,
            "p_mw": float(p_mw),
            "q_mvar": float(q_mvar),
            "p_lower_mw": p_lower_mw,
            "p_upper_mw": p_upper_mw,
            "q_lower_mvar": q_lower_mvar,
            "q_upper_mvar": q_upper_mvar,
            "delta_p_mw": float(delta_p_mw),
            "delta_q_mvar": float(delta_q_mvar),
            "was_projected": bool(was_projected),
            "active_constraint": active_constraint,
            "projection_reason": projection_reason,
            "projection_gap_scope": projection_gap_scope or "local_bounds_not_ac_security",
            "pp_element_type": pp_element_type,
            "pp_element_index": pp_element_index,
        }

    def _record_projection_trace_prefix(
        self,
        step: int,
        vpp,
        fr_id: str,
        command_source: str,
        raw_target: float,
        projected_target: float,
        ac_projected_target: float | None = None,
        envelope: dict[str, Any] | None = None,
    ) -> None:
        p_min, p_max, q_min, q_max = vpp.aggregate_flexibility(step)
        device_target = max(p_min, min(p_max, raw_target))
        aggregate = compute_static_feasible_region(vpp, step).aggregate_bounds()
        active_constraint = ""
        if abs(projected_target - raw_target) > 1e-9:
            active_constraint = "p_max_mw" if projected_target < raw_target else "p_min_mw"
        rows = [
            self._projection_trace_row(
                step,
                vpp.id,
                fr_id,
                command_source,
                1,
                "raw_action",
                raw_target,
                projection_reason="raw DSO/RL/baseline target before projection",
            ),
            self._projection_trace_row(
                step,
                vpp.id,
                fr_id,
                command_source,
                2,
                "device_bounds",
                device_target,
                p_lower_mw=p_min,
                p_upper_mw=p_max,
                q_lower_mvar=q_min,
                q_upper_mvar=q_max,
                delta_p_mw=device_target - raw_target,
                was_projected=abs(device_target - raw_target) > 1e-9,
                active_constraint="device_p_bounds" if abs(device_target - raw_target) > 1e-9 else "",
                projection_reason="aggregate local DER bounds from current device states",
            ),
            self._projection_trace_row(
                step,
                vpp.id,
                fr_id,
                command_source,
                3,
                "fr_doe",
                projected_target,
                p_lower_mw=aggregate.p_min_mw,
                p_upper_mw=aggregate.p_max_mw,
                q_lower_mvar=aggregate.q_min_mvar,
                q_upper_mvar=aggregate.q_max_mvar,
                delta_p_mw=projected_target - raw_target,
                was_projected=abs(projected_target - raw_target) > 1e-9,
                active_constraint=active_constraint,
                projection_reason="static v0 FR/DOE from local DER bounds",
            ),
        ]
        if ac_projected_target is not None and envelope is not None:
            ac_delta = float(ac_projected_target) - float(projected_target)
            ac_active_constraint = ""
            if abs(ac_delta) > 1e-9:
                ac_active_constraint = "ac_aware_p_max_mw" if ac_projected_target < projected_target else "ac_aware_p_min_mw"
            rows.append(
                self._projection_trace_row(
                    step,
                    vpp.id,
                    fr_id,
                    command_source,
                    4,
                    "ac_aware_doe",
                    float(ac_projected_target),
                    p_lower_mw=float(envelope.get("p_min_mw", aggregate.p_min_mw)),
                    p_upper_mw=float(envelope.get("p_max_mw", aggregate.p_max_mw)),
                    q_lower_mvar=aggregate.q_min_mvar,
                    q_upper_mvar=aggregate.q_max_mvar,
                    delta_p_mw=ac_delta,
                    was_projected=abs(ac_delta) > 1e-9,
                    active_constraint=ac_active_constraint,
                    projection_reason=str(envelope.get("ac_aware_reason", "AC-aware DOE envelope projection")),
                    projection_gap_scope="ac_aware_heuristic_not_certificate",
                )
            )
        self.records["projection_trace"].extend(rows)

    def _record_vector_projection_trace(
        self,
        *,
        step: int,
        vpp,
        fr,
        command_source: str,
        targets_by_scope: dict[str, float],
    ) -> None:
        if fr.scope == "pcc":
            return
        current = current_power_by_fr_scope(vpp, fr)
        rows = []
        for scope_key, bounds in fr.bounds.items():
            target = float(targets_by_scope.get(scope_key, current.get(scope_key, 0.0)))
            if scope_key.startswith("bus_"):
                scope_type = "bus"
                scope_id = scope_key.removeprefix("bus_")
            elif scope_key.startswith("zone_"):
                scope_type = "zone"
                scope_id = scope_key.removeprefix("zone_")
            elif scope_key.startswith("pcc_"):
                scope_type = "pcc"
                scope_id = scope_key.removeprefix("pcc_")
            else:
                scope_type = "der"
                scope_id = scope_key
            rows.append(
                self._projection_trace_row(
                    step=step,
                    vpp_id=vpp.id,
                    fr_id=fr.fr_id,
                    command_source=command_source,
                    stage_order=5,
                    stage_name="bus_vector_doe",
                    p_mw=target,
                    p_lower_mw=float(bounds.p_min_mw),
                    p_upper_mw=float(bounds.p_max_mw),
                    q_lower_mvar=float(bounds.q_min_mvar),
                    q_upper_mvar=float(bounds.q_max_mvar),
                    delta_p_mw=target - float(current.get(scope_key, 0.0)),
                    was_projected=True,
                    active_constraint="bus_vector_scope",
                    projection_reason="multi-node FR/DOE scalar request distributed into physical bus/zone scope before DER dispatch",
                    projection_gap_scope="bus_vector_fr_doe_not_aggregate_scalar",
                    scope_type=scope_type,
                    scope_id=scope_id,
                )
            )
        self.records["projection_trace"].extend(rows)

    def _record_ac_certificate_trace(
        self,
        *,
        step: int,
        vpp,
        fr_id: str,
        command_source: str,
        candidate_dispatch: dict[str, float],
        repaired_dispatch: dict[str, float],
        certificate: dict[str, Any],
    ) -> None:
        candidate_total = float(sum(candidate_dispatch.get(der.id, der.p_mw) for der in vpp.der_list))
        repaired_total = float(sum(repaired_dispatch.get(der.id, der.p_mw) for der in vpp.der_list))
        row = self._projection_trace_row(
            step=step,
            vpp_id=vpp.id,
            fr_id=fr_id,
            command_source=command_source,
            stage_order=6,
            stage_name="ac_pf_certificate",
            p_mw=repaired_total,
            q_mvar=vpp.current_reactive_power_mvar(),
            delta_p_mw=repaired_total - candidate_total,
            was_projected=abs(repaired_total - candidate_total) > 1e-9,
            active_constraint=str(certificate.get("ac_certificate_status", "")),
            projection_reason=(
                "joint VPP dispatch replayed through pandapower AC power flow; "
                "candidate accepted if safe, otherwise backed off toward current dispatch"
            ),
            projection_gap_scope="ac_powerflow_verified_dispatch_shield",
        )
        row.update(certificate)
        self.records["projection_trace"].append(row)

    def _record_projection_trace_writes(
        self,
        step: int,
        vpp,
        fr_id: str,
        command_source: str,
        dispatch: dict[str, float],
    ) -> None:
        for der in vpp.der_list:
            p_min, p_max, q_min, q_max = der.get_bounds(step)
            target = float(dispatch.get(der.id, der.p_mw))
            self.records["projection_trace"].append(
                self._projection_trace_row(
                    step=step,
                    vpp_id=vpp.id,
                    fr_id=fr_id,
                    command_source=command_source,
                    stage_order=5,
                    stage_name="pandapower_write",
                    p_mw=target,
                    q_mvar=der.q_mvar,
                    p_lower_mw=p_min,
                    p_upper_mw=p_max,
                    q_lower_mvar=q_min,
                    q_upper_mvar=q_max,
                    was_projected=False,
                    projection_reason="DER target written to its true pandapower element",
                    scope_type="der",
                    scope_id=der.id,
                    pp_element_type=der.pp_element_type,
                    pp_element_index=der.pp_element_index,
                )
            )

    def _record_fr_envelope_state(self, step: int, vpp, fr) -> None:
        physical_mode = str(fr.metadata.get("physical_mode", vpp.physical_mode()))
        der_by_bus: dict[int, list[Any]] = {}
        for der in vpp.der_list:
            der_by_bus.setdefault(int(der.bus), []).append(der)
        for element_id, bounds in fr.bounds.items():
            if element_id.startswith("bus_"):
                scope_type = "bus"
                scope_id = element_id.removeprefix("bus_")
                bus_id = int(scope_id)
                ders = der_by_bus.get(bus_id, [])
            elif element_id.startswith("pcc_"):
                scope_type = "pcc"
                scope_id = element_id.removeprefix("pcc_")
                bus_id = int(scope_id)
                ders = vpp.der_list
            elif element_id.startswith("zone_"):
                scope_type = "zone"
                scope_id = element_id.removeprefix("zone_")
                bus_id = None
                ders = [der for der in vpp.der_list if str(der.metadata.get("zone_id", f"bus_{der.bus}")) == scope_id]
            else:
                scope_type = "der"
                scope_id = element_id
                bus_id = None
                ders = [der for der in vpp.der_list if der.id == element_id]
            p_value = float(sum(der.p_mw for der in ders))
            q_value = float(sum(der.q_mvar for der in ders))
            for variable, lower, upper, value, unit in [
                ("p_mw", bounds.p_min_mw, bounds.p_max_mw, p_value, "MW"),
                ("q_mvar", bounds.q_min_mvar, bounds.q_max_mvar, q_value, "MVAr"),
            ]:
                self.records["fr_envelope_state"].append(
                    {
                        "fr_id": fr.fr_id,
                        "step": int(step),
                        "time_label": self._time_label(step),
                        "vpp_id": vpp.id,
                        "portfolio_version": fr.portfolio_version,
                        "physical_mode": physical_mode,
                        "representation": fr.representation,
                        "source_method": fr.source_method,
                        "scope_type": scope_type,
                        "scope_id": scope_id,
                        "bus_id": bus_id,
                        "zone_id": scope_id if scope_type == "zone" else "",
                        "variable": variable,
                        "lower_bound": float(lower),
                        "upper_bound": float(upper),
                        "current_value": value,
                        "projected_value": value,
                        "unit": unit,
                        "safety_margin": fr.safety_margin_mw if variable == "p_mw" else fr.safety_margin_mvar,
                        "bound_basis": "local_der_bounds",
                        "is_binding": abs(value - lower) <= 1e-6 or abs(value - upper) <= 1e-6,
                    }
                )

    def _apply_portfolio_events(self, step: int) -> None:
        for event in self.scenario.portfolio_events:
            if event.effective_step != int(step) or event.event_id in self._applied_portfolio_event_ids:
                continue
            row = apply_portfolio_event(self.scenario.vpps, event, step)
            row["time_label"] = self._time_label(step)
            self.records["portfolio_change_log"].append(row)
            self._applied_portfolio_event_ids.add(event.event_id)

    def _record_portfolio_history(self, step: int) -> None:
        for vpp in self.scenario.vpps:
            portfolio = vpp.portfolio(step)
            self.records["vpp_portfolio_history"].append(
                {
                    "step": int(step),
                    "time_label": self._time_label(step),
                    "vpp_id": portfolio.vpp_id,
                    "portfolio_version": portfolio.portfolio_version,
                    "physical_mode": portfolio.physical_mode,
                    "pcc_bus_id": portfolio.pcc_bus_id,
                    "connection_buses": ",".join(str(bus) for bus in portfolio.connection_buses),
                    "zone_ids": ",".join(portfolio.zone_ids),
                    "der_ids": ",".join(portfolio.der_ids),
                    "der_count": len(portfolio.der_ids),
                    "connection_bus_count": len(portfolio.connection_buses),
                    "max_import_mw": portfolio.max_import_mw,
                    "max_export_mw": portfolio.max_export_mw,
                }
            )

    def run_timeseries(self, horizon_steps: int | None = None) -> dict[str, pd.DataFrame]:
        self.reset()
        horizon = self.scenario.horizon_steps if horizon_steps is None else horizon_steps
        for step in range(horizon):
            self.step(step)
        return self.collect_results()

    def _record_step(
        self,
        step: int,
        report: ConstraintReport,
        reward_components: dict[str, float],
        price: float,
        load_scale: float,
        pv_forecast_factor: float,
    ) -> None:
        net = self.scenario.net
        self.records["profile_state"].append(
            {
                "step": step,
                "time_hours": float(step) * float(self.scenario.dt_hours),
                "time_label": f"{float(step) * float(self.scenario.dt_hours):05.2f} h",
                "price": price,
                "load_scale": load_scale,
                "pv_forecast_factor": pv_forecast_factor,
            }
        )
        bus_row = {"step": step}
        if hasattr(net, "res_bus") and len(net.res_bus):
            bus_row.update({f"bus_{idx}": float(vm) for idx, vm in net.res_bus["vm_pu"].items()})
        self.records["bus_voltage"].append(bus_row)

        line_row = {"step": step}
        if hasattr(net, "res_line") and len(net.res_line):
            line_row.update(
                {f"line_{idx}": float(value) for idx, value in net.res_line["loading_percent"].items()}
            )
        self.records["line_loading"].append(line_row)

        trafo_row = {"step": step}
        if len(net.trafo) and hasattr(net, "res_trafo") and len(net.res_trafo):
            trafo_row.update(
                {f"trafo_{idx}": float(value) for idx, value in net.res_trafo["loading_percent"].items()}
            )
        self.records["trafo_loading"].append(trafo_row)

        if hasattr(net, "res_line") and len(net.res_line):
            for idx, row in net.res_line.iterrows():
                self.records["edge_power_flow"].append(
                    {
                        "step": step,
                        "edge_id": f"line_{int(idx)}",
                        "edge_type": "line",
                        "pp_index": int(idx),
                        "p_from_mw": float(row.get("p_from_mw", 0.0)),
                        "q_from_mvar": float(row.get("q_from_mvar", 0.0)),
                        "p_to_mw": float(row.get("p_to_mw", 0.0)),
                        "q_to_mvar": float(row.get("q_to_mvar", 0.0)),
                        "active_loss_mw": float(row.get("pl_mw", 0.0)),
                        "reactive_loss_mvar": float(row.get("ql_mvar", 0.0)),
                    }
                )
        if len(net.trafo) and hasattr(net, "res_trafo") and len(net.res_trafo):
            for idx, row in net.res_trafo.iterrows():
                self.records["edge_power_flow"].append(
                    {
                        "step": step,
                        "edge_id": f"trafo_{int(idx)}",
                        "edge_type": "trafo",
                        "pp_index": int(idx),
                        "p_from_mw": float(row.get("p_hv_mw", 0.0)),
                        "q_from_mvar": float(row.get("q_hv_mvar", 0.0)),
                        "p_to_mw": float(row.get("p_lv_mw", 0.0)),
                        "q_to_mvar": float(row.get("q_lv_mvar", 0.0)),
                        "active_loss_mw": float(row.get("pl_mw", 0.0)),
                        "reactive_loss_mvar": float(row.get("ql_mvar", 0.0)),
                    }
                )

        for vpp in self.scenario.vpps:
            self.records["vpp_power"].append(
                {
                    "step": step,
                    "vpp_id": vpp.id,
                    "p_mw": vpp.current_power_mw(),
                    "q_mvar": vpp.current_reactive_power_mvar(),
                    **{
                        key: value
                        for key, value in zip(
                            ["p_min_mw", "p_max_mw", "q_min_mvar", "q_max_mvar"],
                            vpp.aggregate_flexibility(step),
                        )
                    },
                }
            )
            for der in vpp.der_list:
                p_min, p_max, q_min, q_max = der.get_bounds(step)
                available_power = (
                    float(der.available_power(step)) if hasattr(der, "available_power") else None
                )
                self.records["der_dispatch"].append(
                    {
                        "step": step,
                        "vpp_id": vpp.id,
                        "der_id": der.id,
                        "type": der.__class__.__name__,
                        "p_mw": der.p_mw,
                        "q_mvar": der.q_mvar,
                        "p_min_mw": p_min,
                        "p_max_mw": p_max,
                        "q_min_mvar": q_min,
                        "q_max_mvar": q_max,
                        "available_p_mw": available_power,
                    }
                )
                if isinstance(der, StorageModel):
                    self.records["storage_soc"].append(
                        {"step": step, "vpp_id": vpp.id, "der_id": der.id, "soc": der.soc}
                    )
                elif isinstance(der, EVCSModel):
                    self.records["evcs_soc"].append(
                        {"step": step, "vpp_id": vpp.id, "der_id": der.id, "average_soc": der.average_soc()}
                    )
                elif isinstance(der, HVACModel):
                    self.records["hvac_temperature"].append(
                        {
                            "step": step,
                            "vpp_id": vpp.id,
                            "der_id": der.id,
                            "indoor_temp": der.indoor_temp,
                        }
                    )

        self.records["constraint_violations"].extend(report.to_records(step))
        self.records["reward_components"].append({"step": step, **reward_components})

    def collect_results(self) -> dict[str, pd.DataFrame]:
        return {name: pd.DataFrame(rows) for name, rows in self.records.items()}

    def export_results(self, output_dir: str | Path = "outputs") -> dict[str, Path]:
        output_path = ensure_dir(output_dir)
        results = self.collect_results()
        empty_schema = {
            "constraint_violations": ["step", "kind", "element", "value", "limit", "magnitude"],
            "portfolio_change_log": [
                "event_id",
                "step",
                "time_label",
                "der_id",
                "from_vpp_id",
                "to_vpp_id",
                "reason",
                "status",
            ],
        }
        paths: dict[str, Path] = {}
        for name, frame in results.items():
            path = output_path / f"{name}.csv"
            if frame.empty and name in empty_schema:
                frame = pd.DataFrame(columns=empty_schema[name])
            frame.to_csv(path, index=False)
            paths[name] = path

        summary = self._summary(results)
        write_json(output_path / "summary.json", summary)
        paths["summary"] = output_path / "summary.json"
        return paths

    def _summary(self, results: dict[str, pd.DataFrame]) -> dict[str, Any]:
        bus = results.get("bus_voltage", pd.DataFrame())
        line = results.get("line_loading", pd.DataFrame())
        reward = results.get("reward_components", pd.DataFrame())
        bus_values = bus.drop(columns=["step"], errors="ignore")
        line_values = line.drop(columns=["step"], errors="ignore")
        return {
            "steps": int(len(bus)),
            "min_voltage_vm_pu": float(bus_values.min().min()) if not bus_values.empty else None,
            "max_voltage_vm_pu": float(bus_values.max().max()) if not bus_values.empty else None,
            "max_line_loading_percent": float(line_values.max().max()) if not line_values.empty else None,
            "total_cost": float(reward["total_cost"].sum()) if "total_cost" in reward else None,
        }
