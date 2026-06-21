from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from vpp_dso_sim.envs.observations import build_actor_observation, build_critic_global_state
from vpp_dso_sim.envs.reward_design import PortfolioWindowTracker, build_role_reward_maps
from vpp_dso_sim.learning.agent_roles import build_agent_role_map, build_encoder_role_map
from vpp_dso_sim.learning.ctde_interface import (
    CTDEInterfaceContract,
    build_ctde_interface_contract,
    validate_multi_agent_actions,
)
from vpp_dso_sim.optimization.feasibility_region import compute_static_feasible_region
from vpp_dso_sim.simulation.scenario import load_scenario
from vpp_dso_sim.simulation.simulator import Simulator


DSO_ENVELOPE_GUIDANCE_ACTION_KEY = "__dso_envelope_guidance__"


@dataclass
class DSOAgent:
    agent_id: str = "dso"

    def action_description(self) -> dict[str, str]:
        return {
            "target_pq": "active/reactive targets for each VPP",
            "price_signal": "optional price signal per VPP",
            "operating_envelope": "optional PCC power limits",
        }


@dataclass
class VPPAgent:
    agent_id: str

    def action_description(self) -> dict[str, str]:
        return {
            "der_dispatch": "internal DER set points",
            "storage": "ESS charge/discharge action",
            "flexible_load": "DR load response",
            "evcs": "EV charging power",
            "pv": "curtailment and reactive-power command",
        }


class HierarchicalVPPDSOInterface:
    """Structural placeholder for CTDE and hierarchical RL.

    TODO(v0.3): implement a PettingZoo/RLlib-compatible adapter using dict
    observations/actions with agent ids: dso, vpp_0, vpp_1, ...
    """

    def __init__(self, simulator):
        self.simulator = simulator
        self.upper_policy = None
        self.lower_policy = None
        self.baseline_lower_policy = "rule_based_disaggregation"

    def observation_space_description(self) -> dict[str, Any]:
        return {
            "dso_global_guidance": "network state plus VPP reports and bidirectional embeddings",
            "vpp_i_dispatch": "own DER state plus own FR/DOE, local flex signal and representative DSO data",
            "vpp_i_portfolio": "slow-loop own portfolio, profit, reliability and service-call history",
            "critic_global_state": "centralized-training state; not exposed to decentralized VPP actors",
        }

    def action_space_description(self) -> dict[str, Any]:
        return {
            "dso_global_guidance": DSOAgent().action_description(),
            "vpp_i_dispatch": VPPAgent("vpp_i").action_description(),
            "vpp_i_portfolio": {
                "portfolio_update": "slow-loop keep/add/remove/reweight proposal without changing DER physical buses"
            },
        }

    def agent_role_map(self) -> list[dict[str, Any]]:
        return [role.to_dict() for role in build_agent_role_map(self.simulator.scenario.vpps)]

    def encoder_role_map(self) -> list[dict[str, Any]]:
        return [role.to_dict() for role in build_encoder_role_map()]

    def actor_observations(self, t: int) -> dict[str, dict[str, Any]]:
        return {vpp.id: build_actor_observation(vpp, t) for vpp in self.simulator.scenario.vpps}

    def critic_global_state(self, t: int) -> dict[str, Any]:
        return build_critic_global_state(self.simulator.scenario, t)

    def ctde_interface_contract(
        self,
        policy_layout: str = "independent_actor_scaffold",
        share_dispatch_policy: bool = False,
        share_portfolio_policy: bool = False,
    ) -> CTDEInterfaceContract:
        return build_ctde_interface_contract(
            self.simulator.scenario.vpps,
            policy_layout=policy_layout,
            share_dispatch_policy=share_dispatch_policy,
            share_portfolio_policy=share_portfolio_policy,
        )


class MultiAgentVPPDSOEnv:
    """Minimal parallel multi-agent interface for heterogeneous DSO/VPP research.

    The return contract follows the PettingZoo parallel-env shape without adding
    PettingZoo as a hard dependency:

    `reset() -> (observations, infos)`
    `step(actions) -> (observations, rewards, terminations, truncations, infos)`

    v0 action handling:
    - `dso_global_guidance` provides envelope-preference targets per VPP.
    - `{vpp_id}_dispatch` may provide `selected_p_mw` plus `der_actions`
      for learned DER-level disaggregation. Legacy `normalized_setpoint_bias`
      remains accepted for backward-compatible smoke tests.
    - `{vpp_id}_portfolio` actions are recorded for the slow-loop portfolio
      agent; physical portfolio events still come from scenario
      `portfolio_events` so tests stay deterministic.
    """

    def __init__(
        self,
        config_path: str | Path | None = None,
        scenario=None,
        horizon_steps: int | None = None,
    ) -> None:
        self.scenario = scenario if scenario is not None else load_scenario(config_path)
        if horizon_steps is not None:
            self.scenario.horizon_steps = int(horizon_steps)
        self.simulator = Simulator(self.scenario)
        self.current_step = 0
        self.agents = self._build_agents()
        self.possible_agents = list(self.agents)
        self.last_info: dict[str, Any] = {}
        self.default_ctde_contract = self.ctde_interface_contract()
        self.portfolio_window_tracker = PortfolioWindowTracker(self.scenario.dso.reward_config)

    def policy_compatibility_signature(self) -> dict[str, Any]:
        der_counts = {vpp.id: len(vpp.der_list) for vpp in self.scenario.vpps}
        return {
            "vpp_ids": [vpp.id for vpp in self.scenario.vpps],
            "vpp_count": len(self.scenario.vpps),
            "physical_modes": {vpp.id: vpp.physical_mode() for vpp in self.scenario.vpps},
            "der_counts": der_counts,
            "max_der_per_vpp": max(der_counts.values(), default=0),
            "agent_ids": list(self.agents),
        }

    def _build_agents(self) -> list[str]:
        agents = ["dso_global_guidance"]
        for vpp in self.scenario.vpps:
            agents.append(f"{vpp.id}_dispatch")
        for vpp in self.scenario.vpps:
            agents.append(f"{vpp.id}_portfolio")
        return agents

    def ctde_interface_contract(
        self,
        policy_layout: str = "independent_actor_scaffold",
        share_dispatch_policy: bool = False,
        share_portfolio_policy: bool = False,
    ) -> CTDEInterfaceContract:
        return build_ctde_interface_contract(
            self.scenario.vpps,
            policy_layout=policy_layout,
            share_dispatch_policy=share_dispatch_policy,
            share_portfolio_policy=share_portfolio_policy,
        )

    def _vpp_by_agent(self, agent_id: str):
        for vpp in self.scenario.vpps:
            if agent_id in {f"{vpp.id}_dispatch", f"{vpp.id}_portfolio"}:
                return vpp
        return None

    def _dso_observation(self, t: int) -> dict[str, Any]:
        network_state = self.scenario.dso.compute_network_state()
        return {
            "agent_id": "dso_global_guidance",
            "observation_type": "dso_guidance_observation",
            "time_index": int(t),
            "network_state": network_state,
            "vpp_reports": {vpp.id: vpp.report_to_dso(t) for vpp in self.scenario.vpps},
            "role": "global target / price / envelope guidance",
        }

    def _portfolio_observation(self, vpp, t: int) -> dict[str, Any]:
        actor_obs = build_actor_observation(vpp, t, operating_envelope=self._operating_envelope(vpp, t))
        portfolio = actor_obs["portfolio"]
        return {
            "agent_id": f"{vpp.id}_portfolio",
            "observation_type": "vpp_portfolio_observation",
            "time_index": int(t),
            "portfolio": portfolio,
            "slow_loop_options": ["keep", "reweight", "propose_membership_change"],
            "trainable_action_current_version": True,
            "physical_change_allowed": False,
            "portfolio_action_semantics": {
                "keep": "preserve current commercial aggregation weights",
                "reweight": "adjust commercial preference/priority without moving physical DER buses",
                "propose_membership_change": "request a slow commercial membership change for a gated future event",
            },
            "note": (
                "The portfolio action is trainable, but physical DER membership changes are still gated by "
                "deterministic scenario portfolio_events so an RL action cannot silently move pandapower elements."
            ),
        }

    def _operating_envelope(self, vpp, t: int) -> dict[str, Any]:
        price = self.simulator._profile_value(self.scenario.price_profile, t)
        bid = vpp.day_ahead_bid(t, price_hint=price)
        fr = compute_static_feasible_region(vpp, t)
        return self.simulator._build_dso_operating_envelope_for_policy(vpp, t, bid, fr, price)

    def _observations(self, t: int) -> dict[str, dict[str, Any]]:
        obs: dict[str, dict[str, Any]] = {"dso_global_guidance": self._dso_observation(t)}
        for vpp in self.scenario.vpps:
            envelope = self._operating_envelope(vpp, t)
            obs[f"{vpp.id}_dispatch"] = build_actor_observation(
                vpp,
                t,
                include_private_cost=True,
                operating_envelope=envelope,
                service_signal={
                    "service_request": envelope.get("service_request", ""),
                    "preferred_p_min_mw": envelope.get("preferred_p_min_mw", 0.0),
                    "preferred_p_max_mw": envelope.get("preferred_p_max_mw", 0.0),
                    "price": envelope.get("price", 0.0),
                },
                dispatch_award={
                    "awarded_p_mw": envelope.get("preferred_target_p_mw", 0.0),
                    "settlement_price": envelope.get("price", 0.0),
                    "award_status": "envelope_guidance",
                },
            )
            obs[f"{vpp.id}_portfolio"] = self._portfolio_observation(vpp, t)
        return obs

    def _decode_dso_targets(self, actions: dict[str, Any] | None) -> dict[str, float]:
        if not actions:
            return {}
        raw = actions.get("dso_global_guidance", {})
        if isinstance(raw, dict):
            if "envelope_action" in raw:
                return {}
            if "targets" in raw and isinstance(raw["targets"], dict):
                return {str(key): float(value) for key, value in raw["targets"].items()}
            return {
                str(key): float(value)
                for key, value in raw.items()
                if str(key) in {vpp.id for vpp in self.scenario.vpps}
            }
        if isinstance(raw, (list, tuple)):
            return {
                vpp.id: float(raw[index])
                for index, vpp in enumerate(self.scenario.vpps)
                if index < len(raw)
            }
        try:
            import numpy as np

            array = np.asarray(raw, dtype=float).reshape(-1)
            return {
                vpp.id: float(array[index])
                for index, vpp in enumerate(self.scenario.vpps)
                if index < len(array)
            }
        except Exception:
            return {}

    def _decode_dso_envelope_action(self, actions: dict[str, Any] | None) -> dict[str, Any]:
        if not actions:
            return {}
        raw = actions.get("dso_global_guidance", {})
        if not isinstance(raw, dict):
            return {}
        envelope_action = raw.get("envelope_action")
        return dict(envelope_action) if isinstance(envelope_action, dict) else {}

    def _uses_sensitivity_attention_envelope(self) -> bool:
        dso_cfg = dict(self.scenario.config.get("dso", {}))
        return str(dso_cfg.get("envelope_policy", "")) == "sensitivity_attention_v1"

    def _legacy_targets_to_unified_envelope_action(self, dso_targets: dict[str, float]) -> dict[str, Any]:
        if not dso_targets:
            return {}
        return {
            "source": "legacy_targets_converted_to_unified_envelope",
            "legacy_targets_by_vpp": {str(key): float(value) for key, value in dso_targets.items()},
        }

    def _decode_dispatch_targets(
        self,
        dso_targets: dict[str, float],
        actions: dict[str, Any] | None,
    ) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
        simulator_actions: dict[str, dict[str, Any]] = {
            vpp_id: {
                "selected_p_mw": float(target),
                "command_source": "dso_envelope_target",
                "action_mode": "aggregate_target_without_der_rl",
            }
            for vpp_id, target in dso_targets.items()
        }
        if not actions:
            return simulator_actions, {}

        adjusted = dict(dso_targets)
        audit: dict[str, dict[str, Any]] = {}
        for vpp in self.scenario.vpps:
            agent_id = f"{vpp.id}_dispatch"
            raw = actions.get(agent_id)
            if raw is None and vpp.id not in adjusted:
                continue
            try:
                der_actions = None
                if isinstance(raw, dict):
                    der_actions = raw.get("der_actions", raw.get("normalized_der_actions"))
                    if "selected_p_mw" in raw:
                        bias_target = float(raw["selected_p_mw"])
                        bias = 0.0
                    elif "target_p_mw" in raw:
                        bias_target = float(raw["target_p_mw"])
                        bias = 0.0
                    else:
                        bias = float(raw.get("normalized_setpoint_bias", raw.get("response_bias", 0.0)))
                        p_min, p_max, _, _ = vpp.aggregate_flexibility(self.current_step)
                        base = float(adjusted.get(vpp.id, vpp.current_power_mw()))
                        bias_target = base + max(-1.0, min(1.0, bias)) * 0.25 * max(1e-9, p_max - p_min)
                else:
                    bias = float(raw)
                    p_min, p_max, _, _ = vpp.aggregate_flexibility(self.current_step)
                    base = float(adjusted.get(vpp.id, vpp.current_power_mw()))
                    bias_target = base + max(-1.0, min(1.0, bias)) * 0.25 * max(1e-9, p_max - p_min)
            except (TypeError, ValueError):
                continue

            p_min, p_max, _, _ = vpp.aggregate_flexibility(self.current_step)
            projected = max(float(p_min), min(float(p_max), float(bias_target)))
            previous = float(adjusted.get(vpp.id, vpp.current_power_mw()))
            adjusted[vpp.id] = projected
            projection_gap = abs(projected - float(bias_target))
            simulator_actions[vpp.id] = {
                "selected_p_mw": projected,
                "der_actions": der_actions,
                "command_source": "vpp_rl_envelope_action" if der_actions is not None else "vpp_rl_aggregate_action",
                "action_mode": "learned_der_disaggregation" if der_actions is not None else "aggregate_target_bias",
                "raw_action_norm": float(bias),
                "decoded_target_p_mw": float(bias_target),
                "raw_target_p_mw": float(bias_target),
                "pre_projection_gap_mw": projection_gap,
                "pre_projection_clipped": bool(projection_gap > 1e-9),
            }
            audit[vpp.id] = {
                "dso_target_p_mw": previous,
                "baseline_p_mw": previous,
                "dispatch_bias": float(bias),
                "raw_action_norm": float(bias),
                "raw_target_p_mw": float(bias_target),
                "decoded_target_p_mw": float(bias_target),
                "dispatch_adjusted_target_p_mw": float(bias_target),
                "projected_target_p_mw": projected,
                "projection_gap_mw": projection_gap,
                "local_bounds_projection_gap_mw": projection_gap,
                "projection_gap_scope": "local_der_bounds_not_ac_security",
                "projection_clipped": bool(projection_gap > 1e-9),
                "p_min_mw": float(p_min),
                "p_max_mw": float(p_max),
                "der_action_count": len(der_actions) if isinstance(der_actions, (list, tuple, dict)) else 0,
                "uses_learned_der_actions": der_actions is not None,
            }
        return simulator_actions, audit

    def _action_landing_audit_from_records(self, step: int) -> dict[str, dict[str, Any]]:
        """Summarize simulator projection stages into per-VPP landing fields."""

        stage_rows: dict[str, dict[str, dict[str, Any]]] = {}
        for row in self.simulator.records.get("projection_trace", []):
            try:
                row_step = int(float(row.get("step", -1)))
            except (TypeError, ValueError):
                continue
            if row_step != int(step):
                continue
            vpp_id = str(row.get("vpp_id", ""))
            stage_name = str(row.get("stage_name", ""))
            if not vpp_id or not stage_name:
                continue
            stage_rows.setdefault(vpp_id, {})[stage_name] = dict(row)

        landing: dict[str, dict[str, Any]] = {}
        for vpp_id, by_stage in stage_rows.items():
            raw = self._stage_p_mw(by_stage, "raw_action", 0.0)
            device = self._stage_p_mw(by_stage, "device_bounds", raw)
            pre_ac = self._stage_p_mw(by_stage, "fr_doe", device)
            ac_projected = self._stage_p_mw(by_stage, "ac_aware_doe", pre_ac)
            ac_certified = self._stage_p_mw(by_stage, "ac_pf_certificate", ac_projected)
            actual = self._stage_p_mw(by_stage, "powerflow_result", ac_certified)
            landing[vpp_id] = {
                "raw_target_p_mw": float(raw),
                "device_feasible_target_p_mw": float(device),
                "pre_ac_target_p_mw": float(pre_ac),
                "ac_projected_target_p_mw": float(ac_projected),
                "ac_certified_target_p_mw": float(ac_certified),
                "actual_target_p_mw": float(actual),
                "raw_to_device_gap_mw": abs(float(device) - float(raw)),
                "device_to_ac_gap_mw": abs(float(ac_projected) - float(device)),
                "ac_to_actual_gap_mw": abs(float(actual) - float(ac_projected)),
            }
        return landing

    @staticmethod
    def _stage_p_mw(by_stage: dict[str, dict[str, Any]], stage_name: str, default: float) -> float:
        row = by_stage.get(stage_name, {})
        try:
            return float(row.get("p_mw", default))
        except (TypeError, ValueError):
            return float(default)

    def _envelopes_from_records(self, step: int) -> dict[str, dict[str, Any]]:
        envelopes: dict[str, dict[str, Any]] = {}
        for row in self.simulator.records.get("dso_operating_envelope", []):
            if int(row.get("step", -1)) == int(step):
                envelopes[str(row.get("vpp_id"))] = dict(row)
        return envelopes

    def _portfolio_actions_by_vpp(self, actions: dict[str, Any] | None) -> dict[str, str]:
        portfolio_actions: dict[str, str] = {}
        for vpp in self.scenario.vpps:
            raw = (actions or {}).get(f"{vpp.id}_portfolio", {})
            if isinstance(raw, dict):
                portfolio_actions[vpp.id] = str(raw.get("action", "keep"))
            elif raw is not None:
                portfolio_actions[vpp.id] = str(raw)
            else:
                portfolio_actions[vpp.id] = "keep"
        return portfolio_actions

    def _agent_rewards(
        self,
        reward_components: dict[str, Any],
        dispatch_adjustments: dict[str, dict[str, Any]],
        actions: dict[str, Any] | None,
        step: int,
    ) -> tuple[dict[str, float], dict[str, dict[str, float]]]:
        dispatch_audit = {
            str(vpp_id): dict(payload)
            for vpp_id, payload in dict(dispatch_adjustments or {}).items()
        }
        for vpp_id, landing_payload in self._action_landing_audit_from_records(step).items():
            target = dispatch_audit.setdefault(str(vpp_id), {})
            for key, value in landing_payload.items():
                if key in {"raw_target_p_mw", "decoded_target_p_mw"} and key in target:
                    continue
                target[key] = value
        for vpp_id, summary in self._vpp_settlement_summaries_from_records(step).items():
            dispatch_audit.setdefault(str(vpp_id), {}).update(summary)
        rewards, components = build_role_reward_maps(
            vpps=self.scenario.vpps,
            envelopes_by_vpp=self._envelopes_from_records(step),
            dispatch_audit=dispatch_audit,
            portfolio_actions_by_vpp=self._portfolio_actions_by_vpp(actions),
            dso_components=reward_components,
            dt_hours=float(self.scenario.dt_hours),
            t=int(step),
            reward_config=self.scenario.dso.reward_config,
            portfolio_tracker=self.portfolio_window_tracker,
        )
        return (
            {agent: float(rewards.get(agent, 0.0)) for agent in self.agents},
            {agent: components.get(agent, {}) for agent in self.agents},
        )

    def _vpp_settlement_summaries_from_records(self, step: int) -> dict[str, dict[str, Any]]:
        summaries: dict[str, dict[str, Any]] = {}
        for row in self.simulator.records.get("vpp_settlement_summary", []):
            try:
                row_step = int(float(row.get("step", -1)))
            except (TypeError, ValueError):
                continue
            if row_step == int(step):
                summaries[str(row.get("vpp_id"))] = dict(row)
        return summaries

    def validate_action_payload(self, actions: dict[str, Any] | None) -> dict[str, Any]:
        return validate_multi_agent_actions(actions, self.scenario.vpps).to_dict()

    def reset(
        self,
        seed: int | None = None,
        start_step: int = 0,
    ) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
        if seed is not None:
            self.scenario.seed = int(seed)
        self.simulator.reset()
        self.portfolio_window_tracker.reset()
        safe_horizon = max(1, int(self.scenario.horizon_steps))
        self.current_step = int(start_step) % safe_horizon
        self.simulator.current_step = int(self.current_step)
        infos = {
            agent: {
                "agent_role_map": [role.to_dict() for role in build_agent_role_map(self.scenario.vpps)],
                "encoder_role_map": [role.to_dict() for role in build_encoder_role_map()],
                "ctde_actor_spec": self.default_ctde_contract.actor_spec_for(agent).to_dict()
                if self.default_ctde_contract.actor_spec_for(agent) is not None
                else None,
                "centralized_critic_spec": self.default_ctde_contract.centralized_critic.to_dict()
                if agent == "dso_global_guidance"
                else None,
            }
            for agent in self.agents
        }
        return self._observations(self.current_step), infos

    def step(
        self,
        actions: dict[str, Any] | None,
    ) -> tuple[
        dict[str, dict[str, Any]],
        dict[str, float],
        dict[str, bool],
        dict[str, bool],
        dict[str, dict[str, Any]],
    ]:
        action_validation = validate_multi_agent_actions(actions, self.scenario.vpps)
        validated_actions = action_validation.normalized_actions
        dso_envelope_action = self._decode_dso_envelope_action(validated_actions)
        dso_targets = {} if dso_envelope_action else self._decode_dso_targets(validated_actions)
        simulator_dso_targets = dso_targets
        if not dso_envelope_action and dso_targets and self._uses_sensitivity_attention_envelope():
            dso_envelope_action = self._legacy_targets_to_unified_envelope_action(dso_targets)
            simulator_dso_targets = {}
        simulator_actions, dispatch_adjustments = self._decode_dispatch_targets(simulator_dso_targets, validated_actions)
        if dso_envelope_action:
            simulator_actions[DSO_ENVELOPE_GUIDANCE_ACTION_KEY] = dso_envelope_action
        decoded_numeric_targets = {
            vpp_id: float(payload.get("selected_p_mw", 0.0))
            for vpp_id, payload in simulator_actions.items()
            if isinstance(payload, dict) and vpp_id != DSO_ENVELOPE_GUIDANCE_ACTION_KEY
        }
        result = self.simulator.step(self.current_step, actions=simulator_actions or None)
        self.current_step = int(result["step"]) + 1
        done = self.current_step >= int(self.scenario.horizon_steps)
        reward_components = result.get("reward_components", {})
        observations = self._observations(self.current_step)
        rewards, agent_reward_components = self._agent_rewards(
            reward_components,
            dispatch_adjustments,
            validated_actions,
            int(result["step"]),
        )
        terminations = {agent: False for agent in self.agents}
        truncations = {agent: bool(done) for agent in self.agents}
        critic_state = build_critic_global_state(self.scenario, self.current_step)
        infos = {
            agent: {
                "step": int(result["step"]),
                "decoded_dso_targets": dso_targets,
                "decoded_dso_envelope_action": dso_envelope_action,
                "decoded_simulator_targets": decoded_numeric_targets,
                "decoded_simulator_action_payload": simulator_actions,
                "decoded_vpp_dispatch_adjustments": dispatch_adjustments,
                "raw_action": (actions or {}).get(agent),
                "validated_action": validated_actions.get(agent),
                "action_validation": action_validation.to_dict(),
                "critic_global_state": critic_state if agent == "dso_global_guidance" else None,
                "training_only_critic_state": critic_state if agent == "dso_global_guidance" else None,
                "critic_state_visibility": "training_only_not_actor_observation",
                "reward_components": reward_components,
                "agent_reward_components": agent_reward_components.get(agent, {}),
                "violations": result.get("violations", []),
            }
            for agent in self.agents
        }
        self.last_info = infos
        return observations, rewards, terminations, truncations, infos

    def close(self) -> None:
        return None
