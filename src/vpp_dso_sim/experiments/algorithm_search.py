from __future__ import annotations

import importlib
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from vpp_dso_sim.utils.io import ensure_dir, write_json


@dataclass(frozen=True)
class AlgorithmSearchConfig:
    """Configuration for the lightweight MARL algorithm idea search.

    The search is intentionally metadata based. It ranks candidates for the
    current DSO/VPP role-specific general-sum setting without launching long
    training jobs or making performance claims.
    """

    output_dir: str | Path = "outputs/algorithm_search"
    top_k: int = 5
    min_candidates: int = 20
    registry_module: str = "vpp_dso_sim.learning.advanced_marl"
    proxy_budget_label: str = "metadata_only_short_budget_v1"


@dataclass(frozen=True)
class AlgorithmCandidate:
    algorithm_id: str
    family: str
    idea: str
    action_space: str
    reward_mode: str
    privacy_mode: str
    heterogeneity_model: str
    engineering_stage: str
    tags: tuple[str, ...]
    registry_source: str = "curated_builtin_search_space"
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        row = asdict(self)
        row["tags"] = ",".join(self.tags)
        return row


@dataclass(frozen=True)
class RegistryLoadReport:
    registry_module: str
    registry_available: bool
    registry_candidate_count: int
    fallback_candidate_count: int
    registry_error: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


SCORE_COLUMNS = (
    "reward_fit",
    "privacy_fit",
    "continuous_action_fit",
    "heterogeneity_fit",
    "risk_penalty",
    "expected_engineering_lift",
)

SCORE_WEIGHTS = {
    "reward_fit": 0.30,
    "privacy_fit": 0.25,
    "continuous_action_fit": 0.20,
    "heterogeneity_fit": 0.15,
    "risk_penalty": -0.07,
    "expected_engineering_lift": -0.03,
}


def _candidate(
    algorithm_id: str,
    family: str,
    idea: str,
    *,
    action_space: str,
    reward_mode: str,
    privacy_mode: str,
    heterogeneity_model: str,
    engineering_stage: str,
    tags: tuple[str, ...],
    notes: str = "",
) -> AlgorithmCandidate:
    return AlgorithmCandidate(
        algorithm_id=algorithm_id,
        family=family,
        idea=idea,
        action_space=action_space,
        reward_mode=reward_mode,
        privacy_mode=privacy_mode,
        heterogeneity_model=heterogeneity_model,
        engineering_stage=engineering_stage,
        tags=tuple(sorted(set(tags))),
        notes=notes,
    )


def fallback_algorithm_candidates() -> list[AlgorithmCandidate]:
    """Return built-in candidates used when learning.advanced_marl is absent."""

    return [
        _candidate(
            "privacy_separated_ctde_actor_critic",
            "CTDE actor-critic",
            "Keep the current privacy-separated CTDE baseline and tune it first.",
            action_space="continuous_gaussian",
            reward_mode="role_specific_general_sum",
            privacy_mode="local_actor_centralized_critic",
            heterogeneity_model="deep_sets_vpp_dispatch",
            engineering_stage="implemented",
            tags=(
                "ctde",
                "local_actor",
                "continuous",
                "general_sum",
                "role_specific_reward",
                "set_encoder",
                "implemented",
                "low_lift",
            ),
        ),
        _candidate(
            "role_specific_mappo_set_encoder",
            "MAPPO",
            "MAPPO with local VPP actors, role-specific rewards and set-encoded DER tokens.",
            action_space="continuous_gaussian",
            reward_mode="role_specific_general_sum",
            privacy_mode="local_actor_centralized_critic",
            heterogeneity_model="shared_set_actor_per_role",
            engineering_stage="near_term",
            tags=(
                "ctde",
                "local_actor",
                "continuous",
                "general_sum",
                "role_specific_reward",
                "set_encoder",
                "centralized_critic",
            ),
        ),
        _candidate(
            "constrained_mappo_projection_penalty",
            "Constrained MAPPO",
            "MAPPO variant that learns from FR/DOE projection gap and safety penalties.",
            action_space="continuous_gaussian",
            reward_mode="role_specific_general_sum_with_constraints",
            privacy_mode="local_actor_centralized_critic",
            heterogeneity_model="set_actor_plus_safety_features",
            engineering_stage="near_term",
            tags=(
                "ctde",
                "local_actor",
                "continuous",
                "general_sum",
                "role_specific_reward",
                "constraint_aware",
                "projection_penalty",
                "set_encoder",
            ),
        ),
        _candidate(
            "gnn_critic_local_actor",
            "Graph critic CTDE",
            "Use a grid graph critic during training while execution actors stay local.",
            action_space="continuous_gaussian",
            reward_mode="role_specific_general_sum",
            privacy_mode="local_actor_graph_critic",
            heterogeneity_model="graph_critic_and_local_set_actor",
            engineering_stage="medium_lift",
            tags=(
                "ctde",
                "local_actor",
                "continuous",
                "general_sum",
                "graph_critic",
                "set_encoder",
                "medium_lift",
            ),
        ),
        _candidate(
            "mappo_graph_critic_set_actor",
            "MAPPO graph critic",
            "MAPPO with graph-aware centralized critic and local set-encoded actors.",
            action_space="continuous_gaussian",
            reward_mode="role_specific_general_sum",
            privacy_mode="local_actor_graph_critic",
            heterogeneity_model="graph_critic_set_actor",
            engineering_stage="medium_lift",
            tags=(
                "ctde",
                "local_actor",
                "continuous",
                "general_sum",
                "graph_critic",
                "set_encoder",
                "role_specific_reward",
            ),
        ),
        _candidate(
            "safe_rl_lagrangian_ctde",
            "Constrained CTDE",
            "Add Lagrangian multipliers for voltage, loading and projection penalties.",
            action_space="continuous_gaussian",
            reward_mode="constrained_role_specific_general_sum",
            privacy_mode="local_actor_centralized_critic",
            heterogeneity_model="role_specific_actor_heads",
            engineering_stage="medium_lift",
            tags=(
                "ctde",
                "local_actor",
                "continuous",
                "general_sum",
                "constraint_aware",
                "safety",
                "medium_lift",
            ),
        ),
        _candidate(
            "matd3_continuous_vpp_dispatch",
            "MATD3",
            "Off-policy deterministic continuous control for VPP dispatch actors.",
            action_space="continuous_deterministic",
            reward_mode="role_specific_general_sum",
            privacy_mode="local_actor_centralized_critic",
            heterogeneity_model="per_role_actor_heads",
            engineering_stage="medium_lift",
            tags=(
                "ctde",
                "local_actor",
                "continuous",
                "general_sum",
                "off_policy",
                "deterministic_actor",
                "medium_lift",
            ),
        ),
        _candidate(
            "masac_entropy_regularized_bidding",
            "MASAC",
            "Entropy-regularized continuous VPP dispatch and bid-markup policies.",
            action_space="continuous_stochastic",
            reward_mode="settlement_aware_general_sum",
            privacy_mode="local_actor_centralized_critic",
            heterogeneity_model="per_role_actor_heads",
            engineering_stage="medium_lift",
            tags=(
                "ctde",
                "local_actor",
                "continuous",
                "general_sum",
                "settlement",
                "market_bid",
                "off_policy",
                "medium_lift",
            ),
        ),
        _candidate(
            "facmac_continuous_value_factorization",
            "FACMAC",
            "Continuous value factorization for coordinated VPP responses.",
            action_space="continuous_deterministic",
            reward_mode="general_sum_with_factorized_value",
            privacy_mode="local_actor_mixing_critic",
            heterogeneity_model="factorized_vpp_mixer",
            engineering_stage="medium_lift",
            tags=(
                "ctde",
                "local_actor",
                "continuous",
                "value_factorization",
                "general_sum",
                "medium_lift",
            ),
        ),
        _candidate(
            "happo_role_factorized",
            "HAPPO",
            "Heterogeneous-agent PPO with separate DSO, VPP dispatch and portfolio roles.",
            action_space="continuous_gaussian",
            reward_mode="role_specific_general_sum",
            privacy_mode="local_actor_centralized_critic",
            heterogeneity_model="role_factorized_policies",
            engineering_stage="medium_lift",
            tags=(
                "ctde",
                "local_actor",
                "continuous",
                "general_sum",
                "role_factorized",
                "heterogeneous_roles",
                "medium_lift",
            ),
        ),
        _candidate(
            "hatrpo_role_factorized",
            "HATRPO",
            "Trust-region heterogeneous-agent policy optimization for stable updates.",
            action_space="continuous_gaussian",
            reward_mode="role_specific_general_sum",
            privacy_mode="local_actor_centralized_critic",
            heterogeneity_model="role_factorized_policies",
            engineering_stage="high_lift",
            tags=(
                "ctde",
                "local_actor",
                "continuous",
                "general_sum",
                "role_factorized",
                "trust_region",
                "high_lift",
            ),
        ),
        _candidate(
            "ippo_local_actor_deep_sets",
            "IPPO",
            "Independent local PPO actors with DER set encoders and no privileged actor input.",
            action_space="continuous_gaussian",
            reward_mode="role_specific_local_rewards",
            privacy_mode="decentralized_local_actor",
            heterogeneity_model="deep_sets_per_vpp",
            engineering_stage="low_lift",
            tags=(
                "local_actor",
                "continuous",
                "role_specific_reward",
                "set_encoder",
                "low_lift",
            ),
        ),
        _candidate(
            "mean_field_actor_critic_vpps",
            "Mean-field MARL",
            "Approximate many VPP interactions with aggregate peer response statistics.",
            action_space="continuous_gaussian",
            reward_mode="role_specific_general_sum",
            privacy_mode="local_actor_mean_field_context",
            heterogeneity_model="mean_field_vpp_context",
            engineering_stage="medium_lift",
            tags=(
                "ctde",
                "local_actor",
                "continuous",
                "general_sum",
                "mean_field",
                "scalable",
                "medium_lift",
            ),
        ),
        _candidate(
            "hierarchical_options_vpp_dispatch",
            "HRL options",
            "High-level VPP service options with inner DER disaggregation policies.",
            action_space="hybrid_continuous_options",
            reward_mode="role_specific_general_sum",
            privacy_mode="local_actor_centralized_critic",
            heterogeneity_model="hierarchical_vpp_roles",
            engineering_stage="high_lift",
            tags=(
                "ctde",
                "local_actor",
                "continuous",
                "general_sum",
                "hierarchical",
                "options",
                "high_lift",
            ),
        ),
        _candidate(
            "transformer_critic_role_attention",
            "Attention CTDE",
            "Transformer critic over role tokens and VPP capability summaries.",
            action_space="continuous_gaussian",
            reward_mode="role_specific_general_sum",
            privacy_mode="local_actor_attention_critic",
            heterogeneity_model="role_attention_tokens",
            engineering_stage="high_lift",
            tags=(
                "ctde",
                "local_actor",
                "continuous",
                "general_sum",
                "attention",
                "role_factorized",
                "high_lift",
            ),
        ),
        _candidate(
            "multi_objective_pareto_ctde",
            "Multi-objective CTDE",
            "Learn Pareto-conditioned policies for DSO cost, VPP profit and fairness.",
            action_space="continuous_gaussian",
            reward_mode="multi_objective_general_sum",
            privacy_mode="local_actor_centralized_critic",
            heterogeneity_model="role_conditioned_actor_critic",
            engineering_stage="high_lift",
            tags=(
                "ctde",
                "local_actor",
                "continuous",
                "general_sum",
                "multi_objective",
                "fairness",
                "high_lift",
            ),
        ),
        _candidate(
            "offline_bc_from_rule_traces",
            "Behavior cloning",
            "Clone safe rule-based traces to warm-start local continuous actors.",
            action_space="continuous_gaussian",
            reward_mode="supervised_warmstart",
            privacy_mode="local_actor_trace_dataset",
            heterogeneity_model="set_encoder_behavior_clone",
            engineering_stage="low_lift",
            tags=(
                "local_actor",
                "continuous",
                "imitation",
                "warmstart",
                "set_encoder",
                "low_lift",
            ),
        ),
        _candidate(
            "cql_offline_local_actor",
            "Offline RL",
            "Conservative offline RL from simulator traces before online fine-tuning.",
            action_space="continuous_gaussian",
            reward_mode="role_specific_general_sum",
            privacy_mode="local_actor_offline_dataset",
            heterogeneity_model="set_encoder_offline_actor",
            engineering_stage="medium_lift",
            tags=(
                "local_actor",
                "continuous",
                "offline_rl",
                "general_sum",
                "set_encoder",
                "medium_lift",
            ),
        ),
        _candidate(
            "contextual_bandit_bid_markup",
            "Contextual bandit",
            "Learn VPP bid markup and reserve preference without full temporal credit assignment.",
            action_space="continuous_bid_markup",
            reward_mode="settlement_aware_local_profit",
            privacy_mode="local_actor_no_peer_private_state",
            heterogeneity_model="per_vpp_context_features",
            engineering_stage="low_lift",
            tags=(
                "local_actor",
                "continuous",
                "market_bid",
                "settlement",
                "low_lift",
            ),
        ),
        _candidate(
            "model_based_mpc_warmstart_policy",
            "Model-based RL",
            "Use short-horizon MPC or safety projection traces to warm-start actors.",
            action_space="continuous_gaussian",
            reward_mode="role_specific_general_sum",
            privacy_mode="local_actor_with_public_safety_model",
            heterogeneity_model="model_based_warmstart",
            engineering_stage="high_lift",
            tags=(
                "local_actor",
                "continuous",
                "general_sum",
                "model_based",
                "warmstart",
                "safety",
                "high_lift",
            ),
        ),
        _candidate(
            "qmix_discrete_flex_awards",
            "QMIX",
            "Discrete flex-award bins with monotonic value mixing.",
            action_space="discrete_bins",
            reward_mode="shared_or_factorized_reward",
            privacy_mode="local_actor_mixing_critic",
            heterogeneity_model="homogeneous_value_mixer",
            engineering_stage="low_lift",
            tags=("ctde", "local_actor", "discrete", "value_factorization", "low_lift"),
        ),
        _candidate(
            "vdn_discrete_dispatch",
            "VDN",
            "Value decomposition baseline over coarse dispatch bins.",
            action_space="discrete_bins",
            reward_mode="shared_or_factorized_reward",
            privacy_mode="local_actor_mixing_critic",
            heterogeneity_model="homogeneous_value_sum",
            engineering_stage="low_lift",
            tags=("local_actor", "discrete", "value_factorization", "low_lift"),
        ),
        _candidate(
            "maddpg_peer_critic_baseline",
            "MADDPG",
            "Continuous actor baseline with centralized critic over joint action summaries.",
            action_space="continuous_deterministic",
            reward_mode="general_sum",
            privacy_mode="local_actor_peer_action_critic",
            heterogeneity_model="per_role_actor_heads",
            engineering_stage="medium_lift",
            tags=(
                "ctde",
                "local_actor",
                "continuous",
                "general_sum",
                "peer_action_critic",
                "medium_lift",
            ),
        ),
        _candidate(
            "shared_backbone_actor_critic_benchmark",
            "Shared-backbone ablation",
            "Retain the old shared-backbone actor-critic as an explicit ablation only.",
            action_space="continuous_gaussian",
            reward_mode="shared_global_reward",
            privacy_mode="shared_latent_actor",
            heterogeneity_model="flat_shared_mlp",
            engineering_stage="implemented_ablation",
            tags=("continuous", "shared_backbone", "centralized_actor", "privacy_risk", "implemented"),
        ),
        _candidate(
            "oracle_centralized_opf_imitation",
            "Oracle imitation",
            "Imitate a future full-information OPF oracle for gap analysis only.",
            action_space="continuous_gaussian",
            reward_mode="oracle_full_information",
            privacy_mode="oracle_actor_not_deployable",
            heterogeneity_model="centralized_grid_model",
            engineering_stage="research_baseline_high_lift",
            tags=(
                "continuous",
                "oracle",
                "centralized_actor",
                "privacy_risk",
                "high_lift",
                "baseline_only",
            ),
        ),
    ]


def _normalize_tags(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        raw_items = value.replace(";", ",").replace("|", ",").split(",")
    else:
        try:
            raw_items = list(value)
        except TypeError:
            raw_items = [value]
    tags = [str(item).strip().lower() for item in raw_items if str(item).strip()]
    return tuple(sorted(set(tags)))


def _first_present(mapping: Mapping[str, Any], keys: tuple[str, ...], default: str = "") -> str:
    for key in keys:
        value = mapping.get(key)
        if value not in (None, ""):
            return str(value)
    return default


def _join_nonempty(*values: Any, separator: str = " ") -> str:
    return separator.join(str(value) for value in values if value not in (None, ""))


def _inferred_tags(*parts: str) -> tuple[str, ...]:
    text = " ".join(part.lower() for part in parts)
    tags: set[str] = set()
    if "ctde" in text or "centralized critic" in text:
        tags.add("ctde")
    if "centralized_training_only" in text or "centralized training" in text:
        tags.add("ctde")
        tags.add("centralized_critic")
    if "local actor" in text or "local_actor_observation" in text or "decentralized" in text:
        tags.add("local_actor")
    if "continuous" in text or "gaussian" in text or "deterministic" in text:
        tags.add("continuous")
    if "discrete" in text or "qmix" in text or "vdn" in text:
        tags.add("discrete")
    if "general-sum" in text or "general_sum" in text:
        tags.add("general_sum")
    if "role-specific" in text or "role_specific" in text:
        tags.add("role_specific_reward")
    if "graph" in text or "gnn" in text:
        tags.add("graph_critic")
    if "set" in text or "deep sets" in text:
        tags.add("set_encoder")
    if "native" in text and "heterogeneous" in text:
        tags.add("heterogeneous_roles")
    if "attention" in text or "transformer" in text:
        tags.add("attention")
    if "oracle" in text:
        tags.add("oracle")
    if "shared" in text and "backbone" in text:
        tags.add("shared_backbone")
    if "centralized actor" in text:
        tags.add("centralized_actor")
    return tuple(sorted(tags))


def _candidate_from_mapping(data: Mapping[str, Any], source: str) -> AlgorithmCandidate | None:
    algorithm_id = _first_present(data, ("algorithm_id", "key", "id", "name", "algorithm", "slug"))
    if not algorithm_id:
        return None
    family = _first_present(data, ("family", "algorithm_family", "class_name"), "registry_candidate")
    idea = _first_present(
        data,
        ("idea", "description", "summary", "display_name", "recommended_role"),
        family,
    )
    action_support = data.get("action_support")
    if isinstance(action_support, (list, tuple, set)):
        action_space_default = "_".join(str(item) for item in action_support)
    else:
        action_space_default = "unknown"
    action_space = _first_present(data, ("action_space", "action_type"), action_space_default)
    reward_mode = _first_present(
        data,
        ("reward_mode", "reward_model", "reward", "objective"),
        "unknown",
    )
    privacy_default = _join_nonempty(
        data.get("actor_privacy_scope"),
        data.get("critic_scope"),
        separator="_",
    )
    privacy_mode = _first_present(
        data,
        ("privacy_mode", "privacy", "execution_privacy"),
        privacy_default or "unknown",
    )
    heterogeneity_model = _first_present(
        data,
        (
            "heterogeneity_model",
            "heterogeneity_support",
            "heterogeneity",
            "encoder",
            "critic",
        ),
        "unknown",
    )
    engineering_stage = _first_present(
        data,
        ("engineering_stage", "stage", "status", "implementation_status"),
        "registry_candidate",
    )
    tags = set(_normalize_tags(data.get("tags") or data.get("features") or data.get("capabilities")))
    if isinstance(action_support, (list, tuple, set)):
        tags.update(str(item).strip().lower() for item in action_support if str(item).strip())
    training_pattern = str(data.get("training_pattern", "")).lower()
    if training_pattern == "ctde":
        tags.add("ctde")
    if str(data.get("actor_privacy_scope", "")).lower() == "local_actor_observation":
        tags.add("local_actor")
    if str(data.get("critic_scope", "")).lower() == "centralized_training_only":
        tags.add("centralized_critic")
    if str(data.get("reward_model", "")).lower() == "general_sum":
        tags.add("general_sum")
    implementation_risk = data.get("implementation_risk")
    if implementation_risk not in (None, ""):
        try:
            risk_value = float(implementation_risk)
        except (TypeError, ValueError):
            risk_value = 0.0
        if risk_value >= 0.65:
            tags.add("high_lift")
        elif risk_value >= 0.50:
            tags.add("medium_lift")
        elif risk_value <= 0.35:
            tags.add("low_lift")
    heterogeneity_support = str(data.get("heterogeneity_support", "")).lower()
    if heterogeneity_support == "native":
        tags.add("heterogeneous_roles")
    elif heterogeneity_support == "role_specific":
        tags.add("role_factorized")
    policy_type = str(data.get("policy_type", "")).lower()
    if "deterministic" in policy_type:
        tags.add("deterministic_actor")
    if "stochastic" in policy_type:
        tags.add("stochastic_actor")
    tags.update(
        _inferred_tags(
            algorithm_id,
            family,
            idea,
            action_space,
            reward_mode,
            privacy_mode,
            heterogeneity_model,
            engineering_stage,
            training_pattern,
            str(data.get("actor_privacy_scope", "")),
            str(data.get("critic_scope", "")),
            str(data.get("policy_type", "")),
            str(data.get("recommended_role", "")),
            str(data.get("notes", "")),
        )
    )
    notes = _first_present(data, ("notes", "rationale", "claim_boundary"), "")
    return AlgorithmCandidate(
        algorithm_id=algorithm_id,
        family=family,
        idea=idea,
        action_space=action_space,
        reward_mode=reward_mode,
        privacy_mode=privacy_mode,
        heterogeneity_model=heterogeneity_model,
        engineering_stage=engineering_stage,
        tags=tuple(sorted(tags)),
        registry_source=source,
        notes=notes,
    )


def _coerce_registry_item(item: Any, source: str) -> AlgorithmCandidate | None:
    if isinstance(item, AlgorithmCandidate):
        data = item.to_dict()
        data["tags"] = item.tags
        return _candidate_from_mapping(data, source)
    if hasattr(item, "to_dict") and callable(item.to_dict):
        item = item.to_dict()
    elif hasattr(item, "__dataclass_fields__"):
        item = asdict(item)
    if isinstance(item, Mapping):
        return _candidate_from_mapping(item, source)
    return None


def _registry_items(raw_registry: Any) -> list[Any]:
    if raw_registry is None:
        return []
    if callable(raw_registry):
        raw_registry = raw_registry()
    if isinstance(raw_registry, Mapping):
        items: list[Any] = []
        for key, value in raw_registry.items():
            if isinstance(value, Mapping):
                merged = dict(value)
                merged.setdefault("algorithm_id", key)
                items.append(merged)
            else:
                items.append({"algorithm_id": key, "description": str(value)})
        return items
    try:
        return list(raw_registry)
    except TypeError:
        return []


def _load_registry_candidates(module_name: str) -> tuple[list[AlgorithmCandidate], str]:
    try:
        module = importlib.import_module(module_name)
    except Exception as exc:  # pragma: no cover - exact import failure depends on environment
        return [], str(exc)

    attr_names = (
        "ALGORITHM_REGISTRY",
        "ADVANCED_MARL_REGISTRY",
        "REGISTRY",
        "registry",
        "get_algorithm_registry",
        "get_registry",
        "build_registry",
    )
    raw_items: list[Any] = []
    for attr_name in attr_names:
        if not hasattr(module, attr_name):
            continue
        raw_items.extend(_registry_items(getattr(module, attr_name)))

    candidates_by_id: dict[str, AlgorithmCandidate] = {}
    for item in raw_items:
        candidate = _coerce_registry_item(item, source=module_name)
        if candidate is not None and candidate.algorithm_id not in candidates_by_id:
            candidates_by_id[candidate.algorithm_id] = candidate
    if not raw_items:
        return [], "registry module imported but no known registry attribute was found"
    return list(candidates_by_id.values()), ""


def load_algorithm_candidates(
    registry_module: str = "vpp_dso_sim.learning.advanced_marl",
    *,
    min_candidates: int = 20,
) -> tuple[list[AlgorithmCandidate], RegistryLoadReport]:
    """Load optional advanced MARL registry and fill with fallback candidates."""

    registry_candidates, registry_error = _load_registry_candidates(registry_module)
    fallback_candidates = fallback_algorithm_candidates()
    by_id: dict[str, AlgorithmCandidate] = {}
    for candidate in registry_candidates:
        by_id[candidate.algorithm_id] = candidate
    fallback_used = 0
    target_count = max(1, min_candidates)
    if registry_candidates:
        for candidate in fallback_candidates:
            if len(by_id) >= target_count:
                break
            if candidate.algorithm_id not in by_id:
                by_id[candidate.algorithm_id] = candidate
                fallback_used += 1
    else:
        for candidate in fallback_candidates:
            if candidate.algorithm_id not in by_id:
                by_id[candidate.algorithm_id] = candidate
                fallback_used += 1
    candidates = list(by_id.values())
    report = RegistryLoadReport(
        registry_module=registry_module,
        registry_available=bool(registry_candidates),
        registry_candidate_count=len(registry_candidates),
        fallback_candidate_count=fallback_used,
        registry_error=registry_error,
    )
    return candidates, report


def _clip(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _tagset(candidate: AlgorithmCandidate) -> set[str]:
    tags = set(candidate.tags)
    tags.update(
        _inferred_tags(
            candidate.algorithm_id,
            candidate.family,
            candidate.idea,
            candidate.action_space,
            candidate.reward_mode,
            candidate.privacy_mode,
            candidate.heterogeneity_model,
            candidate.engineering_stage,
        )
    )
    return tags


def _score_candidate(candidate: AlgorithmCandidate) -> dict[str, float]:
    tags = _tagset(candidate)
    reward_fit = 0.45
    if "general_sum" in tags:
        reward_fit += 0.20
    if "role_specific_reward" in tags:
        reward_fit += 0.15
    if "settlement" in tags or "market_bid" in tags:
        reward_fit += 0.08
    if "constraint_aware" in tags or "projection_penalty" in tags or "safety" in tags:
        reward_fit += 0.08
    if "shared_backbone" in tags or "shared_global_reward" in tags:
        reward_fit -= 0.14
    if "oracle" in tags:
        reward_fit -= 0.12

    privacy_fit = 0.42
    if "local_actor" in tags:
        privacy_fit += 0.25
    if "ctde" in tags:
        privacy_fit += 0.18
    if "centralized_critic" in tags or "graph_critic" in tags or "attention" in tags:
        privacy_fit += 0.05
    if "privacy_risk" in tags:
        privacy_fit -= 0.35
    if "centralized_actor" in tags or "shared_backbone" in tags:
        privacy_fit -= 0.28
    if "oracle" in tags:
        privacy_fit -= 0.30

    continuous_action_fit = 0.30
    if "continuous" in tags:
        continuous_action_fit += 0.50
    if "deterministic_actor" in tags or "off_policy" in tags:
        continuous_action_fit += 0.05
    if "market_bid" in tags:
        continuous_action_fit += 0.05
    if "discrete" in tags and "continuous" not in tags:
        continuous_action_fit -= 0.25
    if "hybrid" in tags or "options" in tags:
        continuous_action_fit += 0.08

    heterogeneity_fit = 0.35
    if "heterogeneous_roles" in tags:
        heterogeneity_fit += 0.35
    if "role_factorized" in tags:
        heterogeneity_fit += 0.35
    for tag in (
        "set_encoder",
        "graph_critic",
        "attention",
        "hierarchical",
        "mean_field",
        "multi_objective",
    ):
        if tag in tags:
            heterogeneity_fit += 0.08
    if "flat_shared_mlp" in tags or "shared_backbone" in tags:
        heterogeneity_fit -= 0.12

    risk_penalty = 0.18
    for tag in ("high_lift", "trust_region", "attention", "model_based", "hierarchical"):
        if tag in tags:
            risk_penalty += 0.10
    for tag in ("off_policy", "graph_critic", "multi_objective", "oracle"):
        if tag in tags:
            risk_penalty += 0.07
    if "privacy_risk" in tags or "centralized_actor" in tags:
        risk_penalty += 0.20
    if "implemented" in tags:
        risk_penalty -= 0.08
    if "low_lift" in tags:
        risk_penalty -= 0.06

    engineering_lift = 0.30
    if "implemented" in tags:
        engineering_lift -= 0.16
    if "low_lift" in tags:
        engineering_lift -= 0.10
    if "medium_lift" in tags:
        engineering_lift += 0.12
    if "high_lift" in tags:
        engineering_lift += 0.30
    for tag in ("graph_critic", "attention", "off_policy", "trust_region", "model_based"):
        if tag in tags:
            engineering_lift += 0.05
    if "oracle" in tags:
        engineering_lift += 0.08

    return {
        "reward_fit": round(_clip(reward_fit), 4),
        "privacy_fit": round(_clip(privacy_fit), 4),
        "continuous_action_fit": round(_clip(continuous_action_fit), 4),
        "heterogeneity_fit": round(_clip(heterogeneity_fit), 4),
        "risk_penalty": round(_clip(risk_penalty), 4),
        "expected_engineering_lift": round(_clip(engineering_lift), 4),
    }


def _proxy_score(scores: Mapping[str, float]) -> float:
    value = sum(float(scores[key]) * weight for key, weight in SCORE_WEIGHTS.items())
    return round(value, 4)


def _rejection_reason(row: Mapping[str, Any]) -> str:
    if float(row["privacy_fit"]) < 0.60:
        return "privacy_fit_below_current_ctde_bar"
    if float(row["continuous_action_fit"]) < 0.60:
        return "weak_fit_for_continuous_vpp_dispatch_actions"
    if float(row["risk_penalty"]) > 0.55:
        return "engineering_or_claim_risk_too_high_for_next_iteration"
    if float(row["expected_engineering_lift"]) > 0.60:
        return "implementation_lift_too_high_for_short_budget"
    return "below_top_k_proxy_score_for_current_iteration"


def _keep_reason(row: Mapping[str, Any]) -> str:
    strengths: list[str] = []
    if float(row["privacy_fit"]) >= 0.80:
        strengths.append("strong privacy boundary fit")
    if float(row["continuous_action_fit"]) >= 0.80:
        strengths.append("continuous action fit")
    if float(row["reward_fit"]) >= 0.80:
        strengths.append("role-specific general-sum reward fit")
    if float(row["heterogeneity_fit"]) >= 0.70:
        strengths.append("heterogeneous DSO/VPP role fit")
    if float(row["risk_penalty"]) <= 0.35:
        strengths.append("manageable short-term engineering risk")
    if not strengths:
        strengths.append("highest proxy score under the current search weights")
    return "; ".join(strengths)


def score_algorithm_candidates(
    candidates: list[AlgorithmCandidate],
    *,
    top_k: int = 5,
    proxy_budget_label: str = "metadata_only_short_budget_v1",
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for candidate in candidates:
        scores = _score_candidate(candidate)
        rows.append(
            {
                **candidate.to_dict(),
                **scores,
                "proxy_score": _proxy_score(scores),
                "proxy_budget_label": proxy_budget_label,
            }
        )
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    frame = frame.sort_values(
        by=["proxy_score", "privacy_fit", "expected_engineering_lift", "algorithm_id"],
        ascending=[False, False, True, True],
    ).reset_index(drop=True)
    frame.insert(0, "rank", range(1, len(frame) + 1))
    cutoff = max(0, min(int(top_k), len(frame)))
    frame["recommendation_status"] = [
        "recommended" if int(rank) <= cutoff else "rejected_for_now"
        for rank in frame["rank"]
    ]
    frame["is_top_k_recommendation"] = frame["rank"] <= cutoff
    frame["rejection_reason"] = [
        "" if status == "recommended" else _rejection_reason(row)
        for status, row in zip(frame["recommendation_status"], frame.to_dict(orient="records"))
    ]
    frame["keep_reason"] = [
        _keep_reason(row) if status == "recommended" else ""
        for status, row in zip(frame["recommendation_status"], frame.to_dict(orient="records"))
    ]
    ordered_columns = [
        "rank",
        "algorithm_id",
        "family",
        "recommendation_status",
        "is_top_k_recommendation",
        "proxy_score",
        *SCORE_COLUMNS,
        "keep_reason",
        "rejection_reason",
        "idea",
        "action_space",
        "reward_mode",
        "privacy_mode",
        "heterogeneity_model",
        "engineering_stage",
        "tags",
        "registry_source",
        "proxy_budget_label",
        "notes",
    ]
    return frame[ordered_columns]


def _summary_from_scores(
    *,
    cfg: AlgorithmSearchConfig,
    scores: pd.DataFrame,
    registry_report: RegistryLoadReport,
    candidate_scores_path: Path,
) -> dict[str, Any]:
    top = scores[scores["recommendation_status"] == "recommended"].copy()
    rejected = scores[scores["recommendation_status"] == "rejected_for_now"].copy()
    return {
        "experiment": "algorithm_search",
        "proxy_budget_label": cfg.proxy_budget_label,
        "candidate_count": int(len(scores)),
        "top_k": int(min(cfg.top_k, len(scores))),
        "recommended_count": int(len(top)),
        "rejected_for_now_count": int(len(rejected)),
        "candidate_scores_csv": str(candidate_scores_path),
        "registry": registry_report.to_dict(),
        "score_columns": list(SCORE_COLUMNS),
        "score_weights": SCORE_WEIGHTS,
        "score_semantics": {
            "reward_fit": "Fit to role-specific general-sum DSO/VPP reward design.",
            "privacy_fit": "Fit to local actor execution and CTDE privacy boundaries.",
            "continuous_action_fit": "Fit to continuous VPP dispatch, bid and DER action heads.",
            "heterogeneity_fit": "Fit to mixed DSO/VPP roles and heterogeneous DER portfolios.",
            "risk_penalty": "Penalty for privacy, stability or research-claim risk.",
            "expected_engineering_lift": "Estimated implementation effort; lower is better.",
        },
        "top_recommendations": top[
            [
                "rank",
                "algorithm_id",
                "family",
                "proxy_score",
                "reward_fit",
                "privacy_fit",
                "continuous_action_fit",
                "heterogeneity_fit",
                "risk_penalty",
                "expected_engineering_lift",
            ]
        ].to_dict(orient="records"),
        "claim_boundary": (
            "This is a lightweight algorithm-idea search. It is not a training result, "
            "not an oracle comparison, and not paper-level evidence."
        ),
    }


def run_algorithm_search(config: AlgorithmSearchConfig | None = None) -> dict[str, Any]:
    cfg = config or AlgorithmSearchConfig()
    out = ensure_dir(cfg.output_dir)
    candidates, registry_report = load_algorithm_candidates(
        cfg.registry_module,
        min_candidates=cfg.min_candidates,
    )
    scores = score_algorithm_candidates(
        candidates,
        top_k=cfg.top_k,
        proxy_budget_label=cfg.proxy_budget_label,
    )
    candidate_scores_path = out / "candidate_scores.csv"
    scores.to_csv(candidate_scores_path, index=False)
    summary = _summary_from_scores(
        cfg=cfg,
        scores=scores,
        registry_report=registry_report,
        candidate_scores_path=candidate_scores_path,
    )
    summary_path = out / "summary.json"
    write_json(summary_path, summary)
    return {
        "output_dir": out,
        "candidate_scores": scores,
        "summary": summary,
        "candidate_scores_path": candidate_scores_path,
        "summary_path": summary_path,
        "registry_report": registry_report,
    }
