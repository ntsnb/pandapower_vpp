from __future__ import annotations

from dataclasses import asdict, dataclass
from math import isfinite
from typing import Any


def _is_numeric(value: Any) -> bool:
    try:
        return isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _as_float(value: Any) -> float:
    return float(value)


@dataclass(frozen=True)
class ActionFieldSpec:
    name: str
    value_kind: str
    required: bool
    description: str
    aliases: tuple[str, ...] = ()
    choices: tuple[str, ...] = ()
    min_value: float | None = None
    max_value: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ActorInterfaceSpec:
    agent_id: str
    role_type: str
    owner_id: str
    observation_type: str
    action_schema_id: str
    policy_module_id: str
    execution_mode: str
    current_implementation: str
    fields: tuple[ActionFieldSpec, ...]

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["fields"] = [field.to_dict() for field in self.fields]
        return data


@dataclass(frozen=True)
class PolicyModuleSpec:
    module_id: str
    module_type: str
    shared_parameters: bool
    trainable: bool
    consumes_observation_type: str
    produces_action_schema: str
    notes: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class CentralizedCriticSpec:
    module_id: str
    observation_type: str
    builder_function: str
    visible_to_decentralized_actors: bool
    current_consumer: str
    target_upgrade: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ValidationIssue:
    level: str
    agent_id: str
    field: str
    message: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ActionValidationReport:
    normalized_actions: dict[str, Any]
    errors: tuple[ValidationIssue, ...]
    warnings: tuple[ValidationIssue, ...]

    @property
    def is_valid(self) -> bool:
        return not self.errors

    def to_dict(self) -> dict[str, Any]:
        return {
            "is_valid": self.is_valid,
            "normalized_actions": self.normalized_actions,
            "errors": [issue.to_dict() for issue in self.errors],
            "warnings": [issue.to_dict() for issue in self.warnings],
        }


@dataclass(frozen=True)
class CTDEInterfaceContract:
    policy_layout: str
    actor_specs: tuple[ActorInterfaceSpec, ...]
    policy_modules: tuple[PolicyModuleSpec, ...]
    centralized_critic: CentralizedCriticSpec

    def actor_spec_for(self, agent_id: str) -> ActorInterfaceSpec | None:
        for spec in self.actor_specs:
            if spec.agent_id == agent_id:
                return spec
        return None

    def to_dict(self) -> dict[str, Any]:
        return {
            "policy_layout": self.policy_layout,
            "actor_specs": [spec.to_dict() for spec in self.actor_specs],
            "policy_modules": [module.to_dict() for module in self.policy_modules],
            "centralized_critic": self.centralized_critic.to_dict(),
        }


def _dso_action_fields() -> tuple[ActionFieldSpec, ...]:
    return (
        ActionFieldSpec(
            name="targets",
            value_kind="mapping[str,float]",
            required=False,
            description="Per-VPP aggregate active-power targets in MW.",
        ),
    )


def _dispatch_action_fields() -> tuple[ActionFieldSpec, ...]:
    return (
        ActionFieldSpec(
            name="selected_p_mw",
            value_kind="float",
            required=False,
            description="Aggregate active-power target selected inside the DSO envelope.",
            aliases=("target_p_mw",),
        ),
        ActionFieldSpec(
            name="normalized_setpoint_bias",
            value_kind="float",
            required=False,
            description="Legacy normalized aggregate bias around the current or DSO-provided target.",
            aliases=("response_bias",),
            min_value=-1.0,
            max_value=1.0,
        ),
        ActionFieldSpec(
            name="der_actions",
            value_kind="mapping[str,float]",
            required=False,
            description="Per-DER normalized dispatch actions.",
            aliases=("normalized_der_actions",),
            min_value=-1.0,
            max_value=1.0,
        ),
        ActionFieldSpec(
            name="policy_version",
            value_kind="str",
            required=False,
            description="Optional policy identifier for tracing.",
        ),
    )


def _portfolio_action_fields() -> tuple[ActionFieldSpec, ...]:
    return (
        ActionFieldSpec(
            name="action",
            value_kind="enum",
            required=True,
            description="Slow-loop commercial portfolio proposal.",
            choices=("keep", "reweight", "propose_membership_change"),
        ),
        ActionFieldSpec(
            name="policy_version",
            value_kind="str",
            required=False,
            description="Optional policy identifier for tracing.",
        ),
    )


def build_ctde_interface_contract(
    vpps,
    policy_layout: str = "independent_actor_scaffold",
    share_dispatch_policy: bool = False,
    share_portfolio_policy: bool = False,
) -> CTDEInterfaceContract:
    if policy_layout not in {"shared_actor_critic", "independent_actor_scaffold"}:
        raise ValueError(f"Unsupported policy_layout: {policy_layout}")

    actor_specs: list[ActorInterfaceSpec] = []
    policy_modules: list[PolicyModuleSpec] = []

    if policy_layout == "shared_actor_critic":
        policy_modules.extend(
            [
                PolicyModuleSpec(
                    module_id="shared_actor_critic:dso_head",
                    module_type="dso_actor_head",
                    shared_parameters=True,
                    trainable=True,
                    consumes_observation_type="dso_guidance_observation",
                    produces_action_schema="dso_guidance_targets_v1",
                    notes="Current PyTorch trainer uses a shared backbone and a DSO target head.",
                ),
                PolicyModuleSpec(
                    module_id="shared_actor_critic:vpp_dispatch_head",
                    module_type="vpp_dispatch_actor_head",
                    shared_parameters=True,
                    trainable=True,
                    consumes_observation_type="actor_observation_i",
                    produces_action_schema="vpp_dispatch_action_v1",
                    notes="Current PyTorch trainer uses a shared backbone and emits selected_p_mw plus DER actions.",
                ),
                PolicyModuleSpec(
                    module_id="shared_actor_critic:vpp_portfolio_head",
                    module_type="vpp_portfolio_actor_head",
                    shared_parameters=True,
                    trainable=True,
                    consumes_observation_type="vpp_portfolio_observation",
                    produces_action_schema="vpp_portfolio_action_v1",
                    notes="Current trainer includes portfolio categorical actions in the policy-gradient loss; physical membership changes remain gated by scenario events.",
                ),
            ]
        )
    else:
        policy_modules.append(
            PolicyModuleSpec(
                module_id="dso_actor",
                module_type="dso_actor",
                shared_parameters=False,
                trainable=True,
                consumes_observation_type="dso_guidance_observation",
                produces_action_schema="dso_guidance_targets_v1",
                    notes="Default scaffold for the independent DSO actor under the hierarchical CTDE/HAPPO design.",
            )
        )
        if share_dispatch_policy:
            policy_modules.append(
                PolicyModuleSpec(
                    module_id="vpp_dispatch_actor",
                    module_type="vpp_dispatch_actor",
                    shared_parameters=True,
                    trainable=True,
                    consumes_observation_type="actor_observation_i",
                    produces_action_schema="vpp_dispatch_action_v1",
                    notes="Template/shared VPP dispatch actor for decentralized execution.",
                )
            )
        if share_portfolio_policy:
            policy_modules.append(
                PolicyModuleSpec(
                    module_id="vpp_portfolio_actor",
                    module_type="vpp_portfolio_actor",
                    shared_parameters=True,
                    trainable=True,
                    consumes_observation_type="vpp_portfolio_observation",
                    produces_action_schema="vpp_portfolio_action_v1",
                    notes="Template/shared slow-loop VPP portfolio actor scaffold.",
                )
            )

    actor_specs.append(
        ActorInterfaceSpec(
            agent_id="dso_global_guidance",
            role_type="dso_guidance_agent",
            owner_id="dso",
            observation_type="dso_guidance_observation",
            action_schema_id="dso_guidance_targets_v1",
            policy_module_id="shared_actor_critic:dso_head" if policy_layout == "shared_actor_critic" else "dso_actor",
            execution_mode="centralized_training_decentralized_execution_ready",
            current_implementation="trainable_shared_head" if policy_layout == "shared_actor_critic" else "independent_actor_scaffold",
            fields=_dso_action_fields(),
        )
    )

    for vpp in vpps:
        if policy_layout == "shared_actor_critic":
            dispatch_module_id = "shared_actor_critic:vpp_dispatch_head"
            portfolio_module_id = "shared_actor_critic:vpp_portfolio_head"
        else:
            dispatch_module_id = "vpp_dispatch_actor" if share_dispatch_policy else f"{vpp.id}_dispatch_actor"
            portfolio_module_id = "vpp_portfolio_actor" if share_portfolio_policy else f"{vpp.id}_portfolio_actor"
            if not share_dispatch_policy:
                policy_modules.append(
                    PolicyModuleSpec(
                        module_id=dispatch_module_id,
                        module_type="vpp_dispatch_actor",
                        shared_parameters=False,
                        trainable=True,
                        consumes_observation_type="actor_observation_i",
                        produces_action_schema="vpp_dispatch_action_v1",
                        notes=f"Per-VPP dispatch scaffold for {vpp.id}.",
                    )
                )
            if not share_portfolio_policy:
                policy_modules.append(
                    PolicyModuleSpec(
                        module_id=portfolio_module_id,
                        module_type="vpp_portfolio_actor",
                        shared_parameters=False,
                        trainable=True,
                        consumes_observation_type="vpp_portfolio_observation",
                        produces_action_schema="vpp_portfolio_action_v1",
                        notes=f"Per-VPP portfolio scaffold for {vpp.id}.",
                    )
                )

        actor_specs.append(
            ActorInterfaceSpec(
                agent_id=f"{vpp.id}_dispatch",
                role_type="vpp_dispatch_agent",
                owner_id=vpp.id,
                observation_type="actor_observation_i",
                action_schema_id="vpp_dispatch_action_v1",
                policy_module_id=dispatch_module_id,
                execution_mode="decentralized_execution",
                current_implementation="shared_head_with_local_action_schema"
                if policy_layout == "shared_actor_critic"
                else "independent_actor_scaffold",
                fields=_dispatch_action_fields(),
            )
        )
        actor_specs.append(
            ActorInterfaceSpec(
                agent_id=f"{vpp.id}_portfolio",
                role_type="vpp_portfolio_agent",
                owner_id=vpp.id,
                observation_type="vpp_portfolio_observation",
                action_schema_id="vpp_portfolio_action_v1",
                policy_module_id=portfolio_module_id,
                execution_mode="decentralized_execution",
                current_implementation="trainable_slow_loop_head_with_physical_change_gate"
                if policy_layout == "shared_actor_critic"
                else "independent_actor_scaffold",
                fields=_portfolio_action_fields(),
            )
        )

    critic = CentralizedCriticSpec(
        module_id="centralized_critic",
        observation_type="critic_global_state",
        builder_function="vpp_dso_sim.envs.observations.build_critic_global_state",
        visible_to_decentralized_actors=False,
        current_consumer="advanced HAPPO and CTDE trainers consume critic_global_state during training only; decentralized actors do not receive it.",
        target_upgrade="Topology-aware or OPF-certified critics can replace this contract without changing actor observation semantics.",
    )
    return CTDEInterfaceContract(
        policy_layout=policy_layout,
        actor_specs=tuple(actor_specs),
        policy_modules=tuple(policy_modules),
        centralized_critic=critic,
    )


def validate_multi_agent_actions(actions: dict[str, Any] | None, vpps) -> ActionValidationReport:
    normalized_actions: dict[str, Any] = {}
    errors: list[ValidationIssue] = []
    warnings: list[ValidationIssue] = []
    if not actions:
        return ActionValidationReport(normalized_actions={}, errors=(), warnings=())

    vpp_ids = [str(vpp.id) for vpp in vpps]
    vpp_lookup = {str(vpp.id): vpp for vpp in vpps}
    allowed_agents = {"dso_global_guidance"}
    allowed_agents.update(f"{vpp_id}_dispatch" for vpp_id in vpp_ids)
    allowed_agents.update(f"{vpp_id}_portfolio" for vpp_id in vpp_ids)

    for agent_id in actions:
        if agent_id not in allowed_agents:
            errors.append(
                ValidationIssue(
                    level="error",
                    agent_id=str(agent_id),
                    field="agent_id",
                    message="Unknown agent id for this environment instance.",
                )
            )

    if "dso_global_guidance" in actions:
        normalized, issue_list = _validate_dso_action(actions["dso_global_guidance"], vpp_ids)
        normalized_actions["dso_global_guidance"] = normalized
        for issue in issue_list:
            (errors if issue.level == "error" else warnings).append(issue)

    for vpp_id in vpp_ids:
        dispatch_agent = f"{vpp_id}_dispatch"
        if dispatch_agent in actions:
            normalized, issue_list = _validate_dispatch_action(actions[dispatch_agent], vpp_lookup[vpp_id], dispatch_agent)
            normalized_actions[dispatch_agent] = normalized
            for issue in issue_list:
                (errors if issue.level == "error" else warnings).append(issue)
        portfolio_agent = f"{vpp_id}_portfolio"
        if portfolio_agent in actions:
            normalized, issue_list = _validate_portfolio_action(actions[portfolio_agent], portfolio_agent)
            normalized_actions[portfolio_agent] = normalized
            for issue in issue_list:
                (errors if issue.level == "error" else warnings).append(issue)

    normalized_actions = {agent_id: payload for agent_id, payload in normalized_actions.items() if payload}
    return ActionValidationReport(
        normalized_actions=normalized_actions,
        errors=tuple(errors),
        warnings=tuple(warnings),
    )


def _validate_dso_action(raw: Any, vpp_ids: list[str]) -> tuple[dict[str, Any], list[ValidationIssue]]:
    issues: list[ValidationIssue] = []
    if raw is None:
        return {}, issues
    if isinstance(raw, dict):
        if "envelope_action" in raw:
            envelope_action = raw.get("envelope_action")
            if not isinstance(envelope_action, dict):
                issues.append(
                    ValidationIssue(
                        level="error",
                        agent_id="dso_global_guidance",
                        field="envelope_action",
                        message="DSO envelope_action must be a dict payload.",
                    )
                )
                return {}, issues
            return {"envelope_action": dict(envelope_action)}, issues
        target_source = raw.get("targets", raw)
        if not isinstance(target_source, dict):
            issues.append(
                ValidationIssue(
                    level="error",
                    agent_id="dso_global_guidance",
                    field="targets",
                    message="DSO guidance must map VPP ids to numeric active-power targets.",
                )
            )
            return {}, issues
        targets: dict[str, float] = {}
        for key, value in target_source.items():
            key_text = str(key)
            if key_text not in vpp_ids:
                issues.append(
                    ValidationIssue(
                        level="warning",
                        agent_id="dso_global_guidance",
                        field=key_text,
                        message="Ignoring target for unknown VPP id.",
                    )
                )
                continue
            if not _is_numeric(value):
                issues.append(
                    ValidationIssue(
                        level="error",
                        agent_id="dso_global_guidance",
                        field=key_text,
                        message="DSO target must be numeric.",
                    )
                )
                continue
            targets[key_text] = _as_float(value)
        return {"targets": targets}, issues
    if isinstance(raw, (list, tuple)):
        values = list(raw)
    else:
        try:
            import numpy as np

            values = list(np.asarray(raw, dtype=float).reshape(-1))
        except Exception:
            issues.append(
                ValidationIssue(
                    level="error",
                    agent_id="dso_global_guidance",
                    field="targets",
                    message="Unsupported DSO target payload; expected dict, list/tuple or numeric array.",
                )
            )
            return {}, issues
    targets: dict[str, float] = {}
    for index, vpp_id in enumerate(vpp_ids[: len(values)]):
        if not _is_numeric(values[index]):
            issues.append(
                ValidationIssue(
                    level="error",
                    agent_id="dso_global_guidance",
                    field=vpp_id,
                    message="DSO target must be numeric.",
                )
            )
            continue
        targets[vpp_id] = _as_float(values[index])
    return {"targets": targets}, issues


def _validate_dispatch_action(raw: Any, vpp, agent_id: str) -> tuple[dict[str, Any], list[ValidationIssue]]:
    issues: list[ValidationIssue] = []
    if raw is None:
        return {}, issues
    if _is_numeric(raw):
        return {"normalized_setpoint_bias": _as_float(raw)}, issues
    if not isinstance(raw, dict):
        issues.append(
            ValidationIssue(
                level="error",
                agent_id=agent_id,
                field="payload",
                message="Dispatch action must be a number or dict payload.",
            )
        )
        return {}, issues

    normalized: dict[str, Any] = {}
    if "selected_p_mw" in raw or "target_p_mw" in raw:
        value = raw["selected_p_mw"] if "selected_p_mw" in raw else raw["target_p_mw"]
        if not _is_numeric(value):
            issues.append(
                ValidationIssue(
                    level="error",
                    agent_id=agent_id,
                    field="selected_p_mw",
                    message="selected_p_mw must be numeric.",
                )
            )
        else:
            normalized["selected_p_mw"] = _as_float(value)

    if "normalized_setpoint_bias" in raw or "response_bias" in raw:
        value = raw["normalized_setpoint_bias"] if "normalized_setpoint_bias" in raw else raw["response_bias"]
        if not _is_numeric(value):
            issues.append(
                ValidationIssue(
                    level="error",
                    agent_id=agent_id,
                    field="normalized_setpoint_bias",
                    message="normalized_setpoint_bias must be numeric.",
                )
            )
        else:
            bias_value = _as_float(value)
            normalized["normalized_setpoint_bias"] = bias_value
            if bias_value < -1.0 or bias_value > 1.0:
                issues.append(
                    ValidationIssue(
                        level="warning",
                        agent_id=agent_id,
                        field="normalized_setpoint_bias",
                        message="Bias is outside [-1, 1]; env projection will clip it.",
                    )
                )

    if "policy_version" in raw:
        normalized["policy_version"] = str(raw["policy_version"])

    der_actions_raw = raw.get("der_actions", raw.get("normalized_der_actions"))
    if der_actions_raw is not None:
        if not isinstance(der_actions_raw, dict):
            issues.append(
                ValidationIssue(
                    level="error",
                    agent_id=agent_id,
                    field="der_actions",
                    message="der_actions must be a dict keyed by DER id.",
                )
            )
        else:
            valid_der_ids = {str(der.id) for der in vpp.der_list}
            der_actions: dict[str, float] = {}
            for der_id, value in der_actions_raw.items():
                der_key = str(der_id)
                if der_key not in valid_der_ids:
                    issues.append(
                        ValidationIssue(
                            level="error",
                            agent_id=agent_id,
                            field=der_key,
                            message="Unknown DER id for this VPP dispatch agent.",
                        )
                    )
                    continue
                if not _is_numeric(value):
                    issues.append(
                        ValidationIssue(
                            level="error",
                            agent_id=agent_id,
                            field=der_key,
                            message="DER action must be numeric.",
                        )
                    )
                    continue
                der_value = _as_float(value)
                der_actions[der_key] = der_value
                if der_value < -1.0 or der_value > 1.0:
                    issues.append(
                        ValidationIssue(
                            level="warning",
                            agent_id=agent_id,
                            field=der_key,
                            message="DER action is outside [-1, 1]; downstream projection should clamp it.",
                        )
                    )
            normalized["der_actions"] = der_actions
    return normalized, issues


def _validate_portfolio_action(raw: Any, agent_id: str) -> tuple[dict[str, Any], list[ValidationIssue]]:
    issues: list[ValidationIssue] = []
    normalized: dict[str, Any] = {}
    if raw is None:
        return normalized, issues
    if isinstance(raw, str):
        raw = {"action": raw}
    if not isinstance(raw, dict):
        issues.append(
            ValidationIssue(
                level="error",
                agent_id=agent_id,
                field="payload",
                message="Portfolio action must be a dict or action string.",
            )
        )
        return normalized, issues
    action = raw.get("action")
    allowed = {"keep", "reweight", "propose_membership_change"}
    if action not in allowed:
        issues.append(
            ValidationIssue(
                level="error",
                agent_id=agent_id,
                field="action",
                message=f"Portfolio action must be one of {sorted(allowed)}.",
            )
        )
    else:
        normalized["action"] = str(action)
    if "policy_version" in raw:
        normalized["policy_version"] = str(raw["policy_version"])
    return normalized, issues
