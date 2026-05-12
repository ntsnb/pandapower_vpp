from __future__ import annotations

import copy
from dataclasses import asdict, dataclass
import importlib.util
import os
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np
import pandas as pd


TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None
TQDM_AVAILABLE = importlib.util.find_spec("tqdm") is not None


def _episode_progress(iterable: Any, *, total: int, desc: str) -> tuple[Any, bool]:
    """Return a tqdm episode iterator when running in an interactive terminal."""

    if not TQDM_AVAILABLE or not sys.stderr.isatty():
        return iterable, False
    from tqdm.auto import tqdm

    position = int(os.environ.get("VPP_DSO_TQDM_TRAIN_POSITION", "0"))
    return (
        tqdm(
            iterable,
            total=int(total),
            desc=desc,
            unit="ep",
            dynamic_ncols=True,
            leave=False,
            position=position,
        ),
        True,
    )


def _set_episode_postfix(progress: Any, enabled: bool, **metrics: Any) -> None:
    if not enabled:
        return
    progress.set_postfix(metrics, refresh=False)


def _require_torch():
    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch is required for advanced MARL research trainers. Install torch first.")
    import torch
    import torch.optim as optim
    from torch.distributions import Categorical, Normal

    return torch, optim, Normal, Categorical


@dataclass(frozen=True)
class AlgorithmCandidate:
    key: str
    display_name: str
    family: str
    training_pattern: str
    actor_privacy_scope: str
    critic_scope: str
    action_support: tuple[str, ...]
    policy_type: str
    heterogeneity_support: str
    reward_model: str
    implementation_risk: float
    recommended_role: str
    notes: str

    def __post_init__(self) -> None:
        if not 0.0 <= self.implementation_risk <= 1.0:
            raise ValueError("implementation_risk must be in [0, 1].")

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["action_support"] = list(self.action_support)
        return payload


@dataclass(frozen=True)
class ScoringWeights:
    privacy_fit: float = 0.28
    continuous_action_fit: float = 0.20
    heterogeneity_fit: float = 0.18
    general_sum_fit: float = 0.14
    ctde_fit: float = 0.10
    implementation_risk_fit: float = 0.10

    def normalized(self) -> "ScoringWeights":
        total = (
            self.privacy_fit
            + self.continuous_action_fit
            + self.heterogeneity_fit
            + self.general_sum_fit
            + self.ctde_fit
            + self.implementation_risk_fit
        )
        if total <= 0.0:
            raise ValueError("At least one scoring weight must be positive.")
        return ScoringWeights(
            privacy_fit=self.privacy_fit / total,
            continuous_action_fit=self.continuous_action_fit / total,
            heterogeneity_fit=self.heterogeneity_fit / total,
            general_sum_fit=self.general_sum_fit / total,
            ctde_fit=self.ctde_fit / total,
            implementation_risk_fit=self.implementation_risk_fit / total,
        )


@dataclass(frozen=True)
class AlgorithmScore:
    candidate_key: str
    display_name: str
    total_score: float
    privacy_fit: float
    continuous_action_fit: float
    heterogeneity_fit: float
    general_sum_fit: float
    ctde_fit: float
    implementation_risk: float
    risk_adjusted_fit: float
    registry_order: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class TwinCriticSpec:
    state_dim: int
    joint_action_dim: int
    hidden_dims: tuple[int, ...] = (256, 256)
    activation: str = "relu"
    layer_norm: bool = True
    output_dim: int = 1
    algorithm_style: str = "td3_matd3_twin_q"
    critic_scope: str = "centralized_training_only"
    input_contract: str = "critic_global_state + joint_action_summary"

    def __post_init__(self) -> None:
        if self.state_dim <= 0:
            raise ValueError("state_dim must be positive.")
        if self.joint_action_dim <= 0:
            raise ValueError("joint_action_dim must be positive.")
        if not self.hidden_dims or any(dim <= 0 for dim in self.hidden_dims):
            raise ValueError("hidden_dims must contain positive dimensions.")
        if self.output_dim <= 0:
            raise ValueError("output_dim must be positive.")

    @property
    def input_dim(self) -> int:
        return self.state_dim + self.joint_action_dim

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["hidden_dims"] = list(self.hidden_dims)
        payload["input_dim"] = self.input_dim
        return payload


@dataclass(frozen=True)
class MultiHeadValueCriticSpec:
    state_dim: int
    joint_action_dim: int
    head_names: tuple[str, ...]
    hidden_dims: tuple[int, ...] = (256, 256)
    activation: str = "relu"
    layer_norm: bool = True
    algorithm_style: str = "happo_centralized_value"
    critic_scope: str = "centralized_training_only"
    input_contract: str = "critic_global_state + joint_action_summary"

    def __post_init__(self) -> None:
        if self.state_dim <= 0:
            raise ValueError("state_dim must be positive.")
        if self.joint_action_dim <= 0:
            raise ValueError("joint_action_dim must be positive.")
        if not self.hidden_dims or any(dim <= 0 for dim in self.hidden_dims):
            raise ValueError("hidden_dims must contain positive dimensions.")
        if not self.head_names:
            raise ValueError("head_names must not be empty.")

    @property
    def input_dim(self) -> int:
        return self.state_dim + self.joint_action_dim

    @property
    def output_dim(self) -> int:
        return len(self.head_names)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["head_names"] = list(self.head_names)
        payload["hidden_dims"] = list(self.hidden_dims)
        payload["input_dim"] = self.input_dim
        payload["output_dim"] = self.output_dim
        return payload


@dataclass(frozen=True)
class HAPPOConfig:
    algorithm: str = "happo_sequential_ctde"
    horizon_steps: int = 8
    episodes: int = 3
    gamma: float = 0.97
    gae_lambda: float = 0.95
    ppo_clip_ratio: float = 0.20
    ppo_epochs: int = 1
    actor_learning_rate: float = 3e-4
    critic_learning_rate: float = 3e-4
    hidden_dim: int = 64
    entropy_coef: float = 0.01
    value_coef: float = 0.50
    max_grad_norm: float = 1.0
    seed: int = 42
    action_clip: float = 1.0
    importance_correction_clip: float = 2.0
    share_vpp_dispatch_parameters: bool = False
    share_vpp_portfolio_parameters: bool = False
    portfolio_decision_interval_steps: int = 24
    portfolio_force_keep_between_decisions: bool = True
    dso_shield_intervention_penalty_coef: float = 1.0
    dispatch_shield_intervention_penalty_coef: float = 1.0
    reward_scale: float = 0.01
    value_target_clip: float | None = 1_000.0
    critic_use_action_summary: bool = False
    importance_correction_total_clip: float = 10.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class HASACConfig:
    algorithm: str = "hasac_continuous_dispatch"
    horizon_steps: int = 8
    episodes: int = 3
    gamma: float = 0.97
    actor_learning_rate: float = 3e-4
    critic_learning_rate: float = 3e-4
    alpha_learning_rate: float = 3e-4
    hidden_dim: int = 64
    batch_size: int = 32
    replay_capacity: int = 20_000
    warmup_steps: int = 16
    tau: float = 0.01
    seed: int = 42
    action_clip: float = 1.0
    target_entropy_dso: float | None = None
    target_entropy_dispatch: float | None = None
    target_entropy_multiplier: float = 1.0
    init_log_alpha_dso: float = -4.0
    init_log_alpha_dispatch: float = -4.0
    log_alpha_min: float = -8.0
    log_alpha_max: float = 3.0
    share_vpp_dispatch_parameters: bool = False
    dso_shield_intervention_penalty_coef: float = 1.0
    dispatch_shield_intervention_penalty_coef: float = 1.0
    reward_scale: float = 0.01
    target_q_clip: float | None = 1_000.0
    critic_grad_clip: float = 1.0
    actor_grad_clip: float = 1.0
    alpha_grad_clip: float = 1.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def build_matd3_twin_critic_spec(
    *,
    critic_global_state_dim: int,
    joint_action_dim: int,
    hidden_dims: tuple[int, ...] = (256, 256),
    activation: str = "relu",
    layer_norm: bool = True,
) -> TwinCriticSpec:
    return TwinCriticSpec(
        state_dim=critic_global_state_dim,
        joint_action_dim=joint_action_dim,
        hidden_dims=hidden_dims,
        activation=activation,
        layer_norm=layer_norm,
        algorithm_style="matd3_centralized_twin_q",
        critic_scope="centralized_training_only",
        input_contract="critic_global_state + all_agent_continuous_actions",
    )


ALGORITHM_REGISTRY: tuple[AlgorithmCandidate, ...] = (
    AlgorithmCandidate(
        key="ippo",
        display_name="Independent PPO",
        family="policy_gradient",
        training_pattern="independent",
        actor_privacy_scope="local_actor_observation",
        critic_scope="local_value",
        action_support=("continuous", "discrete"),
        policy_type="stochastic",
        heterogeneity_support="role_specific",
        reward_model="general_sum",
        implementation_risk=0.25,
        recommended_role="simple decentralized dispatch baseline",
        notes="Strong privacy baseline; no centralized critic coordination signal.",
    ),
    AlgorithmCandidate(
        key="mappo",
        display_name="Multi-Agent PPO",
        family="policy_gradient",
        training_pattern="ctde",
        actor_privacy_scope="local_actor_observation",
        critic_scope="centralized_training_only",
        action_support=("continuous", "discrete"),
        policy_type="stochastic",
        heterogeneity_support="role_specific",
        reward_model="general_sum",
        implementation_risk=0.35,
        recommended_role="first CTDE upgrade from current actor-critic",
        notes="Practical default when on-policy stability matters more than sample efficiency.",
    ),
    AlgorithmCandidate(
        key="happo",
        display_name="Heterogeneous-Agent PPO",
        family="policy_gradient",
        training_pattern="ctde",
        actor_privacy_scope="local_actor_observation",
        critic_scope="centralized_training_only",
        action_support=("continuous", "discrete"),
        policy_type="stochastic",
        heterogeneity_support="native",
        reward_model="general_sum",
        implementation_risk=0.42,
        recommended_role="role-specific DSO/VPP actor upgrade",
        notes="Fits distinct DSO, VPP dispatch, and portfolio roles without forcing sharing.",
    ),
    AlgorithmCandidate(
        key="hatrpo",
        display_name="Heterogeneous-Agent TRPO",
        family="trust_region_policy_gradient",
        training_pattern="ctde",
        actor_privacy_scope="local_actor_observation",
        critic_scope="centralized_training_only",
        action_support=("continuous", "discrete"),
        policy_type="stochastic_trust_region",
        heterogeneity_support="native",
        reward_model="general_sum",
        implementation_risk=0.58,
        recommended_role="trust-region stability candidate after HAPPO reward/critic fixes",
        notes="Natural follow-up to HAPPO when clipped PPO updates remain unstable under long-horizon DSO/VPP coupling.",
    ),
    AlgorithmCandidate(
        key="hasac",
        display_name="Heterogeneous-Agent Soft Actor-Critic",
        family="maximum_entropy_actor_critic",
        training_pattern="ctde",
        actor_privacy_scope="local_actor_observation",
        critic_scope="centralized_training_only",
        action_support=("continuous",),
        policy_type="stochastic_entropy_regularized",
        heterogeneity_support="native",
        reward_model="general_sum",
        implementation_risk=0.55,
        recommended_role="continuous control candidate after MAPPO/HAPPO smoke tests",
        notes="Good fit for continuous VPP dispatch; higher tuning and replay-buffer risk.",
    ),
    AlgorithmCandidate(
        key="maddpg",
        display_name="Multi-Agent DDPG",
        family="deterministic_actor_critic",
        training_pattern="ctde",
        actor_privacy_scope="local_actor_observation",
        critic_scope="centralized_training_only",
        action_support=("continuous",),
        policy_type="deterministic",
        heterogeneity_support="role_specific",
        reward_model="general_sum",
        implementation_risk=0.50,
        recommended_role="off-policy deterministic dispatch baseline",
        notes="Natural CTDE critic but sensitive to critic overestimation and exploration.",
    ),
    AlgorithmCandidate(
        key="matd3",
        display_name="Multi-Agent TD3",
        family="deterministic_actor_critic",
        training_pattern="ctde",
        actor_privacy_scope="local_actor_observation",
        critic_scope="centralized_training_only",
        action_support=("continuous",),
        policy_type="deterministic",
        heterogeneity_support="role_specific",
        reward_model="general_sum",
        implementation_risk=0.47,
        recommended_role="twin-critic continuous dispatch candidate",
        notes="Twin critics reduce overestimation for continuous joint-action Q estimates.",
    ),
    AlgorithmCandidate(
        key="masac",
        display_name="Multi-Agent Soft Actor-Critic",
        family="maximum_entropy_actor_critic",
        training_pattern="ctde",
        actor_privacy_scope="local_actor_observation",
        critic_scope="centralized_training_only",
        action_support=("continuous", "discrete"),
        policy_type="stochastic_entropy_regularized",
        heterogeneity_support="role_specific",
        reward_model="general_sum",
        implementation_risk=0.52,
        recommended_role="sample-efficient stochastic continuous-control baseline",
        notes="Useful if entropy improves exploration under tight FR/DOE projection.",
    ),
    AlgorithmCandidate(
        key="mad4pg",
        display_name="Multi-Agent D4PG",
        family="distributional_deterministic_actor_critic",
        training_pattern="ctde",
        actor_privacy_scope="local_actor_observation",
        critic_scope="centralized_training_only",
        action_support=("continuous",),
        policy_type="deterministic_distributional_critic",
        heterogeneity_support="role_specific",
        reward_model="general_sum",
        implementation_risk=0.70,
        recommended_role="advanced off-policy benchmark after MATD3",
        notes="Distributional critic is attractive but adds implementation and tuning burden.",
    ),
    AlgorithmCandidate(
        key="facmac",
        display_name="FACMAC",
        family="factorized_actor_critic",
        training_pattern="ctde",
        actor_privacy_scope="local_actor_observation",
        critic_scope="factorized_centralized_training",
        action_support=("continuous",),
        policy_type="deterministic",
        heterogeneity_support="moderate",
        reward_model="cooperative",
        implementation_risk=0.65,
        recommended_role="cooperative continuous-control ablation",
        notes="Better for shared team objectives than role-specific general-sum rewards.",
    ),
    AlgorithmCandidate(
        key="maac",
        display_name="Multi-Actor-Attention-Critic",
        family="attention_actor_critic",
        training_pattern="ctde",
        actor_privacy_scope="local_actor_observation",
        critic_scope="centralized_training_only",
        action_support=("continuous", "discrete"),
        policy_type="stochastic",
        heterogeneity_support="moderate",
        reward_model="general_sum",
        implementation_risk=0.58,
        recommended_role="attention critic for many-VPP studies",
        notes="Attention can expose which VPP interactions matter during centralized training.",
    ),
    AlgorithmCandidate(
        key="coma",
        display_name="Counterfactual Multi-Agent Policy Gradients",
        family="counterfactual_policy_gradient",
        training_pattern="ctde",
        actor_privacy_scope="local_actor_observation",
        critic_scope="centralized_training_only",
        action_support=("discrete",),
        policy_type="stochastic",
        heterogeneity_support="moderate",
        reward_model="cooperative",
        implementation_risk=0.62,
        recommended_role="discrete-action credit-assignment ablation",
        notes="Counterfactual baseline is less aligned with continuous VPP dispatch.",
    ),
    AlgorithmCandidate(
        key="qmix",
        display_name="QMIX",
        family="value_decomposition",
        training_pattern="ctde",
        actor_privacy_scope="local_actor_observation",
        critic_scope="monotonic_mixing_network",
        action_support=("discrete",),
        policy_type="value_based",
        heterogeneity_support="homogeneous",
        reward_model="cooperative",
        implementation_risk=0.45,
        recommended_role="discrete cooperative baseline only",
        notes="Monotonic team-value assumption mismatches general-sum DSO/VPP incentives.",
    ),
    AlgorithmCandidate(
        key="vdn",
        display_name="Value Decomposition Networks",
        family="value_decomposition",
        training_pattern="ctde",
        actor_privacy_scope="local_actor_observation",
        critic_scope="additive_value_decomposition",
        action_support=("discrete",),
        policy_type="value_based",
        heterogeneity_support="homogeneous",
        reward_model="cooperative",
        implementation_risk=0.32,
        recommended_role="simple discrete cooperative ablation",
        notes="Low implementation risk but weak fit for continuous role-specific rewards.",
    ),
    AlgorithmCandidate(
        key="qtran",
        display_name="QTRAN",
        family="value_decomposition",
        training_pattern="ctde",
        actor_privacy_scope="local_actor_observation",
        critic_scope="transformed_joint_action_value",
        action_support=("discrete",),
        policy_type="value_based",
        heterogeneity_support="moderate",
        reward_model="cooperative",
        implementation_risk=0.68,
        recommended_role="discrete non-monotonic value-decomposition ablation",
        notes="Can represent richer joint values than QMIX but remains discrete-action heavy.",
    ),
    AlgorithmCandidate(
        key="qplex",
        display_name="QPLEX",
        family="duplex_dueling_value_decomposition",
        training_pattern="ctde",
        actor_privacy_scope="local_actor_observation",
        critic_scope="duplex_dueling_mixing",
        action_support=("discrete",),
        policy_type="value_based",
        heterogeneity_support="moderate",
        reward_model="cooperative",
        implementation_risk=0.64,
        recommended_role="advanced discrete value-decomposition baseline",
        notes="Better expressiveness than QMIX; still not a primary continuous-control fit.",
    ),
    AlgorithmCandidate(
        key="wqmix",
        display_name="Weighted QMIX",
        family="value_decomposition",
        training_pattern="ctde",
        actor_privacy_scope="local_actor_observation",
        critic_scope="weighted_monotonic_mixing",
        action_support=("discrete",),
        policy_type="value_based",
        heterogeneity_support="homogeneous",
        reward_model="cooperative",
        implementation_risk=0.58,
        recommended_role="QMIX robustness ablation",
        notes="Weighted loss may reduce QMIX bias but does not solve continuous dispatch.",
    ),
    AlgorithmCandidate(
        key="continuous_qmix_relaxation",
        display_name="Continuous-Relaxed QMIX",
        family="value_decomposition",
        training_pattern="ctde",
        actor_privacy_scope="local_actor_observation",
        critic_scope="monotonic_mixing_network",
        action_support=("continuous_relaxed", "discrete"),
        policy_type="relaxed_value_based",
        heterogeneity_support="homogeneous",
        reward_model="cooperative",
        implementation_risk=0.72,
        recommended_role="research ablation, not a primary candidate",
        notes="Continuous relaxation adds complexity while retaining cooperative assumptions.",
    ),
    AlgorithmCandidate(
        key="hysteretic_iql",
        display_name="Hysteretic Independent Q-Learning",
        family="independent_value_based",
        training_pattern="independent",
        actor_privacy_scope="local_actor_observation",
        critic_scope="local_value",
        action_support=("discrete",),
        policy_type="value_based",
        heterogeneity_support="homogeneous",
        reward_model="cooperative",
        implementation_risk=0.30,
        recommended_role="legacy discrete independent baseline",
        notes="Useful only as a low-cost sanity baseline for discretized actions.",
    ),
    AlgorithmCandidate(
        key="maven",
        display_name="MAVEN",
        family="latent_exploration_value_decomposition",
        training_pattern="ctde",
        actor_privacy_scope="local_actor_observation",
        critic_scope="latent_conditioned_mixing",
        action_support=("discrete",),
        policy_type="value_based_latent_exploration",
        heterogeneity_support="homogeneous",
        reward_model="cooperative",
        implementation_risk=0.74,
        recommended_role="exploration ablation under sparse service calls",
        notes="Latent exploration may help sparse calls but weakly fits continuous controls.",
    ),
    AlgorithmCandidate(
        key="roma",
        display_name="RODE/ROMA-Style Role-Oriented MARL",
        family="role_decomposition",
        training_pattern="ctde",
        actor_privacy_scope="local_actor_observation",
        critic_scope="role_conditioned_centralized_training",
        action_support=("discrete", "continuous_relaxed"),
        policy_type="role_conditioned",
        heterogeneity_support="native",
        reward_model="mixed",
        implementation_risk=0.69,
        recommended_role="role-discovery research candidate",
        notes="Conceptually aligned with DSO/VPP roles but heavier than explicit role specs.",
    ),
    AlgorithmCandidate(
        key="tarmac",
        display_name="TarMAC",
        family="communication_learning",
        training_pattern="ctde",
        actor_privacy_scope="learned_messages_only",
        critic_scope="centralized_training_only",
        action_support=("continuous", "discrete"),
        policy_type="stochastic_communication",
        heterogeneity_support="moderate",
        reward_model="mixed",
        implementation_risk=0.78,
        recommended_role="communication-policy research extension",
        notes="Any learned messages need explicit privacy audits before execution use.",
    ),
    AlgorithmCandidate(
        key="mean_field_actor_critic",
        display_name="Mean-Field Actor-Critic",
        family="mean_field_marl",
        training_pattern="ctde",
        actor_privacy_scope="local_actor_observation",
        critic_scope="mean_field_neighbor_summary",
        action_support=("continuous", "discrete"),
        policy_type="stochastic",
        heterogeneity_support="moderate",
        reward_model="mixed",
        implementation_risk=0.57,
        recommended_role="large-number-of-VPP scaling baseline",
        notes="Scales to many agents but compresses detailed VPP interactions.",
    ),
    AlgorithmCandidate(
        key="mappo_gnn_critic",
        display_name="MAPPO with Graph Centralized Critic",
        family="graph_policy_gradient",
        training_pattern="ctde",
        actor_privacy_scope="local_actor_observation",
        critic_scope="centralized_training_only",
        action_support=("continuous", "discrete"),
        policy_type="stochastic",
        heterogeneity_support="role_specific",
        reward_model="general_sum",
        implementation_risk=0.63,
        recommended_role="topology-aware centralized critic upgrade",
        notes="Good paper-grade direction if critic graph features do not leak to actors.",
    ),
    AlgorithmCandidate(
        key="matd3_gnn_critic",
        display_name="MATD3 with Graph Twin Critics",
        family="graph_deterministic_actor_critic",
        training_pattern="ctde",
        actor_privacy_scope="local_actor_observation",
        critic_scope="centralized_training_only",
        action_support=("continuous",),
        policy_type="deterministic",
        heterogeneity_support="role_specific",
        reward_model="general_sum",
        implementation_risk=0.67,
        recommended_role="advanced topology-aware twin-critic candidate",
        notes="Best reserved until plain MATD3 twin critics are stable.",
    ),
    AlgorithmCandidate(
        key="transformer_ctde",
        display_name="Transformer CTDE Actor-Critic",
        family="attention_set_transformer",
        training_pattern="ctde",
        actor_privacy_scope="local_actor_observation",
        critic_scope="centralized_training_only",
        action_support=("continuous", "discrete"),
        policy_type="stochastic",
        heterogeneity_support="native",
        reward_model="general_sum",
        implementation_risk=0.76,
        recommended_role="paper-grade heterogeneous sequence/set critic",
        notes="Expressive for DER/VPP tokens but high data and interpretability burden.",
    ),
)

_REGISTRY_BY_KEY = {candidate.key: candidate for candidate in ALGORITHM_REGISTRY}
_REGISTRY_ORDER = {candidate.key: index for index, candidate in enumerate(ALGORITHM_REGISTRY)}


def get_algorithm_registry() -> tuple[AlgorithmCandidate, ...]:
    return ALGORITHM_REGISTRY


def candidate_by_key(key: str) -> AlgorithmCandidate:
    try:
        return _REGISTRY_BY_KEY[key]
    except KeyError as exc:
        raise KeyError(f"Unknown advanced MARL candidate: {key}") from exc


def _privacy_fit(candidate: AlgorithmCandidate) -> float:
    actor_score = {
        "local_actor_observation": 0.75,
        "learned_messages_only": 0.55,
        "local_plus_public_messages": 0.65,
        "shared_global_observation": 0.25,
        "oracle_actor": 0.05,
    }.get(candidate.actor_privacy_scope, 0.35)
    critic_bonus = {
        "centralized_training_only": 0.20,
        "factorized_centralized_training": 0.17,
        "role_conditioned_centralized_training": 0.17,
        "monotonic_mixing_network": 0.13,
        "weighted_monotonic_mixing": 0.13,
        "additive_value_decomposition": 0.10,
        "transformed_joint_action_value": 0.13,
        "duplex_dueling_mixing": 0.13,
        "latent_conditioned_mixing": 0.13,
        "mean_field_neighbor_summary": 0.12,
        "local_value": 0.08,
    }.get(candidate.critic_scope, 0.08)
    return min(1.0, actor_score + critic_bonus)


def _continuous_action_fit(candidate: AlgorithmCandidate) -> float:
    support = set(candidate.action_support)
    if "continuous" in support:
        return 1.0
    if "continuous_relaxed" in support:
        return 0.55
    return 0.18


def _heterogeneity_fit(candidate: AlgorithmCandidate) -> float:
    return {
        "native": 1.0,
        "role_specific": 0.86,
        "moderate": 0.62,
        "homogeneous": 0.30,
    }.get(candidate.heterogeneity_support, 0.45)


def _general_sum_fit(candidate: AlgorithmCandidate) -> float:
    return {
        "general_sum": 1.0,
        "mixed": 0.62,
        "cooperative": 0.25,
    }.get(candidate.reward_model, 0.40)


def _ctde_fit(candidate: AlgorithmCandidate) -> float:
    return {
        "ctde": 1.0,
        "independent": 0.58,
        "centralized": 0.30,
    }.get(candidate.training_pattern, 0.50)


def score_algorithm_candidate(
    candidate: AlgorithmCandidate,
    *,
    weights: ScoringWeights | None = None,
    registry_order: int | None = None,
) -> AlgorithmScore:
    scoring_weights = (weights or ScoringWeights()).normalized()
    privacy = _privacy_fit(candidate)
    continuous = _continuous_action_fit(candidate)
    heterogeneity = _heterogeneity_fit(candidate)
    general_sum = _general_sum_fit(candidate)
    ctde = _ctde_fit(candidate)
    risk_adjusted = 1.0 - candidate.implementation_risk
    total = (
        scoring_weights.privacy_fit * privacy
        + scoring_weights.continuous_action_fit * continuous
        + scoring_weights.heterogeneity_fit * heterogeneity
        + scoring_weights.general_sum_fit * general_sum
        + scoring_weights.ctde_fit * ctde
        + scoring_weights.implementation_risk_fit * risk_adjusted
    )
    order = _REGISTRY_ORDER.get(candidate.key, 10_000) if registry_order is None else registry_order
    return AlgorithmScore(
        candidate_key=candidate.key,
        display_name=candidate.display_name,
        total_score=round(float(total), 6),
        privacy_fit=round(float(privacy), 6),
        continuous_action_fit=round(float(continuous), 6),
        heterogeneity_fit=round(float(heterogeneity), 6),
        general_sum_fit=round(float(general_sum), 6),
        ctde_fit=round(float(ctde), 6),
        implementation_risk=round(float(candidate.implementation_risk), 6),
        risk_adjusted_fit=round(float(risk_adjusted), 6),
        registry_order=order,
    )


def rank_algorithm_candidates(
    candidates: tuple[AlgorithmCandidate, ...] | list[AlgorithmCandidate] | None = None,
    *,
    weights: ScoringWeights | None = None,
) -> tuple[AlgorithmScore, ...]:
    candidate_list = list(ALGORITHM_REGISTRY if candidates is None else candidates)
    scored = [
        score_algorithm_candidate(candidate, weights=weights, registry_order=index)
        for index, candidate in enumerate(candidate_list)
    ]
    return tuple(
        sorted(
            scored,
            key=lambda score: (-score.total_score, score.registry_order, score.candidate_key),
        )
    )


def build_twin_critic(
    spec: TwinCriticSpec,
    *,
    torch_module: Any | None = None,
    require_torch: bool = False,
) -> Any | None:
    if torch_module is None:
        try:
            import torch as torch_module  # type: ignore[no-redef]
        except ModuleNotFoundError:
            if require_torch:
                raise
            return None

    torch = torch_module
    nn = torch.nn

    def activation_layer() -> Any:
        if spec.activation == "relu":
            return nn.ReLU()
        if spec.activation == "tanh":
            return nn.Tanh()
        if spec.activation == "elu":
            return nn.ELU()
        raise ValueError(f"Unsupported activation: {spec.activation}")

    def make_q_network() -> Any:
        layers: list[Any] = []
        input_dim = spec.input_dim
        for hidden_dim in spec.hidden_dims:
            if spec.layer_norm:
                layers.append(nn.LayerNorm(input_dim))
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(activation_layer())
            input_dim = hidden_dim
        layers.append(nn.Linear(input_dim, spec.output_dim))
        return nn.Sequential(*layers)

    class TwinCritic(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.spec = spec
            self.q1 = make_q_network()
            self.q2 = make_q_network()

        def forward(self, state: Any, joint_action: Any) -> tuple[Any, Any]:
            value_input = torch.cat([state, joint_action], dim=-1)
            return self.q1(value_input), self.q2(value_input)

        def q1_value(self, state: Any, joint_action: Any) -> Any:
            value_input = torch.cat([state, joint_action], dim=-1)
            return self.q1(value_input)

    return TwinCritic()


def build_multi_head_value_critic(
    spec: MultiHeadValueCriticSpec,
    *,
    torch_module: Any | None = None,
    require_torch: bool = False,
) -> Any | None:
    if torch_module is None:
        try:
            import torch as torch_module  # type: ignore[no-redef]
        except ModuleNotFoundError:
            if require_torch:
                raise
            return None

    torch = torch_module
    nn = torch.nn

    def activation_layer() -> Any:
        if spec.activation == "relu":
            return nn.ReLU()
        if spec.activation == "tanh":
            return nn.Tanh()
        if spec.activation == "elu":
            return nn.ELU()
        raise ValueError(f"Unsupported activation: {spec.activation}")

    layers: list[Any] = []
    input_dim = spec.input_dim
    for hidden_dim in spec.hidden_dims:
        if spec.layer_norm:
            layers.append(nn.LayerNorm(input_dim))
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(activation_layer())
        input_dim = hidden_dim
    layers.append(nn.Linear(input_dim, spec.output_dim))

    class MultiHeadValueCritic(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.spec = spec
            self.value_net = nn.Sequential(*layers)

        def forward(self, state: Any, joint_action: Any) -> Any:
            value_input = torch.cat([state, joint_action], dim=-1)
            return self.value_net(value_input)

        def forward_heads(self, state: Any, joint_action: Any) -> dict[str, Any]:
            values = self.forward(state, joint_action)
            return {
                head_name: values[..., index]
                for index, head_name in enumerate(self.spec.head_names)
            }

    return MultiHeadValueCritic()


class OffPolicyReplayBuffer:
    """Minimal NumPy replay buffer for off-policy multi-agent research loops."""

    def __init__(self, capacity: int, seed: int = 0) -> None:
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        self.capacity = int(capacity)
        self.rng = np.random.default_rng(seed)
        self.storage: list[dict[str, Any]] = []
        self.position = 0

    def __len__(self) -> int:
        return len(self.storage)

    def add(self, transition: dict[str, Any]) -> None:
        if len(self.storage) < self.capacity:
            self.storage.append(transition)
        else:
            self.storage[self.position] = transition
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size: int) -> dict[str, np.ndarray]:
        if len(self.storage) < batch_size:
            raise ValueError("not enough transitions to sample")
        indices = self.rng.choice(len(self.storage), size=int(batch_size), replace=False)
        keys = self.storage[0].keys()
        batch: dict[str, list[Any]] = {key: [] for key in keys}
        for index in indices:
            item = self.storage[int(index)]
            for key in keys:
                batch[key].append(item[key])
        return {key: np.asarray(values) for key, values in batch.items()}


def _soft_update(source: Any, target: Any, tau: float) -> None:
    for source_param, target_param in zip(source.parameters(), target.parameters()):
        target_param.data.mul_(1.0 - float(tau)).add_(source_param.data, alpha=float(tau))


def _normalize_advantages(advantages: Any, torch: Any) -> Any:
    if advantages.numel() <= 1:
        return advantages
    return (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)


def _sample_squashed_gaussian(mean: Any, log_std: Any, *, action_clip: float, torch: Any) -> tuple[Any, Any, Any]:
    _, _, Normal, _ = _require_torch()
    bounded_log_std = torch.clamp(log_std, -5.0, 2.0)
    dist = Normal(mean, bounded_log_std.exp())
    raw_action = dist.rsample()
    squashed = torch.tanh(raw_action)
    scaled_action = squashed * float(action_clip)
    log_prob = dist.log_prob(raw_action) - torch.log1p(-squashed.pow(2) + 1e-6)
    log_prob = log_prob.reshape(log_prob.shape[0], -1).sum(dim=-1)
    return scaled_action, log_prob, raw_action


def _sample_squashed_gaussian_with_log_prob_dims(
    mean: Any,
    log_std: Any,
    *,
    action_clip: float,
    torch: Any,
) -> tuple[Any, Any, Any]:
    _, _, Normal, _ = _require_torch()
    bounded_log_std = torch.clamp(log_std, -5.0, 2.0)
    dist = Normal(mean, bounded_log_std.exp())
    raw_action = dist.rsample()
    squashed = torch.tanh(raw_action)
    scaled_action = squashed * float(action_clip)
    log_prob_dims = dist.log_prob(raw_action) - torch.log1p(-squashed.pow(2) + 1e-6)
    return scaled_action, log_prob_dims, raw_action


def _dso_gaussian_stats(actor: Any, obs_tensor: Any, raw_action_tensor: Any, Normal: Any) -> tuple[Any, Any]:
    mean, log_std = actor(obs_tensor)
    dist = Normal(mean, log_std.exp())
    log_prob = dist.log_prob(raw_action_tensor).reshape(raw_action_tensor.shape[0], -1).sum(dim=-1)
    entropy = dist.entropy().reshape(raw_action_tensor.shape[0], -1).sum(dim=-1)
    return log_prob, entropy


def _dispatch_gaussian_stats(
    actor: Any,
    obs_tensor: Any,
    raw_aggregate_tensor: Any,
    raw_der_tensor: Any,
    used_der_counts: Any,
    Normal: Any,
    torch: Any,
) -> tuple[Any, Any]:
    aggregate_mean, aggregate_log_std, der_mean, der_log_std = actor(obs_tensor)
    aggregate_dist = Normal(aggregate_mean, aggregate_log_std.exp())
    der_dist = Normal(der_mean, der_log_std.exp())
    aggregate_log_prob = aggregate_dist.log_prob(raw_aggregate_tensor).reshape(raw_aggregate_tensor.shape[0], -1).sum(dim=-1)
    aggregate_entropy = aggregate_dist.entropy().reshape(raw_aggregate_tensor.shape[0], -1).sum(dim=-1)
    der_log_prob_full = der_dist.log_prob(raw_der_tensor)
    der_entropy_full = der_dist.entropy()
    der_mask = (
        torch.arange(raw_der_tensor.shape[1], device=raw_der_tensor.device).unsqueeze(0)
        < used_der_counts.unsqueeze(1)
    ).float()
    der_log_prob = (der_log_prob_full * der_mask).sum(dim=-1)
    der_entropy = (der_entropy_full * der_mask).sum(dim=-1)
    return aggregate_log_prob + der_log_prob, aggregate_entropy + der_entropy


def _portfolio_categorical_stats(actor: Any, obs_tensor: Any, action_idx_tensor: Any, Categorical: Any) -> tuple[Any, Any]:
    logits = actor(obs_tensor)
    dist = Categorical(logits=logits)
    log_prob = dist.log_prob(action_idx_tensor)
    entropy = dist.entropy()
    return log_prob, entropy


def _happo_role_loss(
    *,
    log_probs: Any,
    old_log_probs: Any,
    advantages: Any,
    correction: Any,
    clip_ratio: float,
    entropy: Any,
    entropy_coef: float,
    torch: Any,
) -> tuple[Any, Any]:
    normalized_advantages = _normalize_advantages(advantages, torch).detach()
    ratios = torch.exp(log_probs - old_log_probs.detach())
    detached_correction = correction.detach()
    unclipped = ratios * detached_correction * normalized_advantages
    clipped = torch.clamp(ratios, 1.0 - float(clip_ratio), 1.0 + float(clip_ratio)) * detached_correction * normalized_advantages
    policy_loss = -torch.min(unclipped, clipped).mean()
    total_loss = policy_loss - float(entropy_coef) * entropy.mean()
    return total_loss, ratios


def _continuous_step_observations(
    observations: dict[str, dict[str, Any]],
    *,
    env: Any,
    vpp_ids: list[str],
    max_der_per_vpp: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    from vpp_dso_sim.envs.observations import build_critic_global_state
    from vpp_dso_sim.learning.deep_rl import (
        encode_critic_global_state,
        encode_dso_observation,
        encode_vpp_dispatch_observation,
    )

    dso_obs = encode_dso_observation(observations["dso_global_guidance"], vpp_ids)
    vpp_obs = np.stack(
        [
            encode_vpp_dispatch_observation(observations[f"{vpp_id}_dispatch"], max_der_per_vpp)
            for vpp_id in vpp_ids
        ],
        axis=0,
    ).astype(np.float32)
    critic_state = encode_critic_global_state(
        build_critic_global_state(env.scenario, env.current_step),
        vpp_ids,
    )
    return dso_obs.astype(np.float32), vpp_obs.astype(np.float32), critic_state.astype(np.float32)


def _continuous_payload_from_actions(
    *,
    dso_action: np.ndarray,
    aggregate_actions: np.ndarray,
    der_actions: np.ndarray,
    dso_obs: dict[str, Any],
    vpp_observations: dict[str, dict[str, Any]],
    vpp_ids: list[str],
    der_ids_by_vpp: dict[str, list[str]],
    action_clip: float,
    policy_version: str,
) -> dict[str, Any]:
    from vpp_dso_sim.learning.deep_rl import _target_from_normalized_scalar, _targets_from_normalized_actions

    dso_targets = _targets_from_normalized_actions(dso_action, dso_obs, vpp_ids, action_clip)
    payload: dict[str, Any] = {"dso_global_guidance": {"targets": dso_targets}}
    for index, vpp_id in enumerate(vpp_ids):
        selected_target = _target_from_normalized_scalar(
            float(aggregate_actions[index]),
            vpp_observations[f"{vpp_id}_dispatch"],
            action_clip,
        )
        der_values = der_actions[index].reshape(-1)
        payload[f"{vpp_id}_dispatch"] = {
            "selected_p_mw": float(selected_target),
            "der_actions": {
                der_id: float(der_values[der_index])
                for der_index, der_id in enumerate(der_ids_by_vpp[vpp_id])
                if der_index < len(der_values)
            },
            "policy_version": policy_version,
        }
        payload[f"{vpp_id}_portfolio"] = {
            "action": "keep",
            "policy_version": f"{policy_version}_continuous_only_portfolio_hold",
        }
    return payload


def train_happo(
    config_path: str | Path | None = None,
    output_dir: str | Path = "outputs/happo",
    config: HAPPOConfig | None = None,
) -> dict[str, Any]:
    """Train a runnable HAPPO-style research skeleton with sequential role updates.

    Scope:
    - DSO Gaussian actor with centralized value baseline.
    - Per-VPP dispatch Gaussian actors updated sequentially after DSO by default.
    - Per-VPP slow-loop portfolio categorical actors updated after dispatch by default.
    - Importance correction accumulates across the role update order.

    Parameter sharing is still available as an explicit ablation through
    ``share_vpp_dispatch_parameters`` and ``share_vpp_portfolio_parameters``.
    Physical DER/VPP membership changes remain gated by the scenario layer.
    """

    from vpp_dso_sim.envs.multi_agent_env import MultiAgentVPPDSOEnv
    from vpp_dso_sim.envs.observations import build_critic_global_state
    from vpp_dso_sim.learning.deep_rl import (
        _gae_returns_advantages,
        _build_privacy_separated_networks,
        _target_from_normalized_scalar,
        _targets_from_normalized_actions,
        encode_critic_global_state,
        encode_dso_observation,
        encode_joint_action_summary,
        encode_vpp_dispatch_observation,
        encode_vpp_portfolio_observation,
    )
    from vpp_dso_sim.learning.reward_contracts import shield_intervention_metrics
    from vpp_dso_sim.utils.io import ensure_dir, write_json

    cfg = config or HAPPOConfig()
    torch, optim, Normal, Categorical = _require_torch()
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    out = ensure_dir(output_dir)

    env_probe = MultiAgentVPPDSOEnv(config_path=config_path, horizon_steps=cfg.horizon_steps)
    observations, _ = env_probe.reset(seed=cfg.seed)
    policy_signature = env_probe.policy_compatibility_signature()
    vpp_ids = [vpp.id for vpp in env_probe.scenario.vpps]
    der_ids_by_vpp = {vpp.id: [der.id for der in vpp.der_list] for vpp in env_probe.scenario.vpps}
    max_der_per_vpp = max(1, max((len(ids) for ids in der_ids_by_vpp.values()), default=1))
    dso_input_dim = int(len(encode_dso_observation(observations["dso_global_guidance"], vpp_ids)))
    first_vpp_id = vpp_ids[0]
    vpp_input_dim = int(len(encode_vpp_dispatch_observation(observations[f"{first_vpp_id}_dispatch"], max_der_per_vpp)))
    portfolio_input_dim = int(len(encode_vpp_portfolio_observation(observations[f"{first_vpp_id}_portfolio"])))
    critic_input_dim = int(len(encode_critic_global_state(build_critic_global_state(env_probe.scenario, 0), vpp_ids)))
    critic_action_dim = int(len(encode_joint_action_summary(
        normalized_dso_action=np.zeros(len(vpp_ids), dtype=np.float32),
        vpp_ids=vpp_ids,
        normalized_aggregate_actions={vpp_id: 0.0 for vpp_id in vpp_ids},
        normalized_der_actions={vpp_id: np.zeros(max_der_per_vpp, dtype=np.float32) for vpp_id in vpp_ids},
        portfolio_action_indices={vpp_id: 0 for vpp_id in vpp_ids},
        max_der_per_vpp=max_der_per_vpp,
        action_clip=cfg.action_clip,
    )))
    value_action_dim = critic_action_dim if bool(cfg.critic_use_action_summary) else 1
    env_probe.close()

    modules, architecture_meta = _build_privacy_separated_networks(
        dso_input_dim=dso_input_dim,
        vpp_input_dim=vpp_input_dim,
        portfolio_input_dim=portfolio_input_dim,
        critic_input_dim=critic_input_dim,
        critic_action_dim=critic_action_dim,
        action_dim=len(vpp_ids),
        der_action_dim=max_der_per_vpp,
        hidden_dim=cfg.hidden_dim,
    )
    actor_modules = torch.nn.ModuleDict({"dso_actor": modules["dso_actor"]})
    if cfg.share_vpp_dispatch_parameters:
        actor_modules["vpp_dispatch_actor"] = modules["vpp_dispatch_actor"]
    else:
        for vpp_id in vpp_ids:
            actor_modules[f"{vpp_id}_dispatch_actor"] = copy.deepcopy(modules["vpp_dispatch_actor"])
    if cfg.share_vpp_portfolio_parameters:
        actor_modules["vpp_portfolio_actor"] = modules["vpp_portfolio_actor"]
    else:
        for vpp_id in vpp_ids:
            actor_modules[f"{vpp_id}_portfolio_actor"] = copy.deepcopy(modules["vpp_portfolio_actor"])
    head_names = (
        "dso_global_guidance",
        *[f"{vpp_id}_dispatch" for vpp_id in vpp_ids],
        *[f"{vpp_id}_portfolio" for vpp_id in vpp_ids],
    )
    value_spec = MultiHeadValueCriticSpec(
        state_dim=critic_input_dim,
        joint_action_dim=value_action_dim,
        head_names=head_names,
        hidden_dims=(cfg.hidden_dim, cfg.hidden_dim),
        algorithm_style=(
            "happo_centralized_action_conditioned_multi_head_value"
            if bool(cfg.critic_use_action_summary)
            else "happo_centralized_state_only_multi_head_value"
        ),
    )
    value_critic = build_multi_head_value_critic(value_spec, require_torch=True)
    dso_optimizer = optim.Adam(actor_modules["dso_actor"].parameters(), lr=float(cfg.actor_learning_rate))
    dispatch_optimizers = (
        {"shared_vpp_dispatch": optim.Adam(actor_modules["vpp_dispatch_actor"].parameters(), lr=float(cfg.actor_learning_rate))}
        if cfg.share_vpp_dispatch_parameters
        else {
            vpp_id: optim.Adam(actor_modules[f"{vpp_id}_dispatch_actor"].parameters(), lr=float(cfg.actor_learning_rate))
            for vpp_id in vpp_ids
        }
    )
    portfolio_optimizers = (
        {"shared_vpp_portfolio": optim.Adam(actor_modules["vpp_portfolio_actor"].parameters(), lr=float(cfg.actor_learning_rate))}
        if cfg.share_vpp_portfolio_parameters
        else {
            vpp_id: optim.Adam(actor_modules[f"{vpp_id}_portfolio_actor"].parameters(), lr=float(cfg.actor_learning_rate))
            for vpp_id in vpp_ids
        }
    )
    critic_optimizer = optim.Adam(value_critic.parameters(), lr=float(cfg.critic_learning_rate))

    def dispatch_actor_for(vpp_id: str) -> Any:
        return (
            actor_modules["vpp_dispatch_actor"]
            if cfg.share_vpp_dispatch_parameters
            else actor_modules[f"{vpp_id}_dispatch_actor"]
        )

    def portfolio_actor_for(vpp_id: str) -> Any:
        return (
            actor_modules["vpp_portfolio_actor"]
            if cfg.share_vpp_portfolio_parameters
            else actor_modules[f"{vpp_id}_portfolio_actor"]
        )

    def is_portfolio_decision_step(step: int) -> bool:
        interval = max(1, int(cfg.portfolio_decision_interval_steps))
        return int(step) % interval == 0

    episode_rows: list[dict[str, Any]] = []
    step_rows: list[dict[str, Any]] = []
    update_rows: list[dict[str, Any]] = []
    total_role_updates = 0
    best_episode_reward = float("-inf")
    best_episode_index = -1
    best_checkpoint_state: dict[str, Any] | None = None
    initial_params = torch.cat(
        [
            param.detach().flatten().cpu()
            for param in [*actor_modules.parameters(), *value_critic.parameters()]
        ]
    )

    last_progress_print = 0.0
    progress_interval_seconds = 60.0
    episode_iter, has_tqdm_progress = _episode_progress(range(cfg.episodes), total=cfg.episodes, desc="HAPPO")
    for episode in episode_iter:
        env = MultiAgentVPPDSOEnv(config_path=config_path, horizon_steps=cfg.horizon_steps)
        observations, _ = env.reset(seed=cfg.seed + episode)
        rollout: dict[str, list[Any]] = {
            "critic_state": [],
            "action_summary": [],
            "dso_obs": [],
            "dso_raw_action": [],
            "dso_log_prob": [],
            "dso_entropy": [],
            "dispatch_obs": [],
            "dispatch_raw_aggregate": [],
            "dispatch_raw_der": [],
            "dispatch_log_prob": [],
            "dispatch_entropy": [],
            "dispatch_der_count": [],
            "portfolio_obs": [],
            "portfolio_action_idx": [],
            "portfolio_log_prob": [],
            "portfolio_entropy": [],
            "portfolio_update_mask": [],
            "rewards": [],
        }
        episode_reward = 0.0
        total_cost = 0.0
        violation_count = 0
        projection_gap_total = 0.0
        ac_projection_gap_total = 0.0
        local_projection_gap_total = 0.0
        shield_intervention_penalty_total = 0.0
        shield_intervention_count = 0

        for step in range(cfg.horizon_steps):
            dso_obs = observations["dso_global_guidance"]
            dso_obs_vec = encode_dso_observation(dso_obs, vpp_ids).astype(np.float32)
            dso_tensor = torch.tensor(dso_obs_vec, dtype=torch.float32).unsqueeze(0)
            critic_state = encode_critic_global_state(
                build_critic_global_state(env.scenario, env.current_step),
                vpp_ids,
            ).astype(np.float32)
            dso_mean, dso_log_std = actor_modules["dso_actor"](dso_tensor)
            dso_dist = Normal(dso_mean, dso_log_std.exp())
            raw_dso_action = dso_dist.rsample()
            normalized_dso_action = torch.clamp(raw_dso_action, -cfg.action_clip, cfg.action_clip)
            dso_targets = _targets_from_normalized_actions(
                normalized_dso_action.detach().cpu().numpy().reshape(-1),
                dso_obs,
                vpp_ids,
                cfg.action_clip,
            )

            action_payload: dict[str, Any] = {"dso_global_guidance": {"targets": dso_targets}}
            dispatch_obs_rows: list[np.ndarray] = []
            dispatch_raw_aggregate_rows: list[np.ndarray] = []
            dispatch_raw_der_rows: list[np.ndarray] = []
            dispatch_log_prob_rows: list[float] = []
            dispatch_entropy_rows: list[float] = []
            dispatch_der_counts: list[int] = []
            normalized_aggregate_actions: dict[str, float] = {}
            normalized_der_actions: dict[str, np.ndarray] = {}
            portfolio_obs_rows: list[np.ndarray] = []
            portfolio_action_rows: list[int] = []
            portfolio_log_prob_rows: list[float] = []
            portfolio_entropy_rows: list[float] = []
            portfolio_update_mask_rows: list[float] = []
            portfolio_action_indices: dict[str, int] = {}
            portfolio_decision_step = is_portfolio_decision_step(step)

            for vpp_id in vpp_ids:
                vpp_obs = observations[f"{vpp_id}_dispatch"]
                encoded_vpp_obs = encode_vpp_dispatch_observation(vpp_obs, max_der_per_vpp).astype(np.float32)
                vpp_tensor = torch.tensor(encoded_vpp_obs, dtype=torch.float32).unsqueeze(0)
                aggregate_mean, aggregate_log_std, der_mean, der_log_std = dispatch_actor_for(vpp_id)(vpp_tensor)
                aggregate_dist = Normal(aggregate_mean, aggregate_log_std.exp())
                der_dist = Normal(der_mean, der_log_std.exp())
                raw_aggregate = aggregate_dist.rsample()
                raw_der = der_dist.rsample()
                normalized_aggregate = torch.clamp(raw_aggregate, -cfg.action_clip, cfg.action_clip)
                normalized_der = torch.clamp(raw_der, -cfg.action_clip, cfg.action_clip)
                selected_target = _target_from_normalized_scalar(
                    float(normalized_aggregate.detach().cpu().item()),
                    vpp_obs,
                    cfg.action_clip,
                )
                normalized_aggregate_actions[vpp_id] = float(normalized_aggregate.detach().cpu().item())
                normalized_der_actions[vpp_id] = normalized_der.detach().cpu().numpy().reshape(-1).copy()
                der_ids = der_ids_by_vpp[vpp_id]
                action_payload[f"{vpp_id}_dispatch"] = {
                    "selected_p_mw": float(selected_target),
                    "der_actions": {
                        der_id: float(normalized_der.detach().cpu().numpy().reshape(-1)[index])
                        for index, der_id in enumerate(der_ids)
                    },
                    "policy_version": cfg.algorithm,
                }
                dispatch_obs_rows.append(encoded_vpp_obs)
                dispatch_raw_aggregate_rows.append(raw_aggregate.detach().cpu().numpy().reshape(1))
                dispatch_raw_der_rows.append(raw_der.detach().cpu().numpy().reshape(-1))
                dispatch_der_counts.append(len(der_ids))
                dispatch_log_prob_rows.append(
                    float(aggregate_dist.log_prob(raw_aggregate).sum().detach().cpu().item())
                    + float(der_dist.log_prob(raw_der).reshape(-1)[: len(der_ids)].sum().detach().cpu().item())
                )
                dispatch_entropy_rows.append(
                    float(aggregate_dist.entropy().sum().detach().cpu().item())
                    + float(der_dist.entropy().reshape(-1)[: len(der_ids)].sum().detach().cpu().item())
                )

                portfolio_obs = observations[f"{vpp_id}_portfolio"]
                encoded_portfolio_obs = encode_vpp_portfolio_observation(portfolio_obs).astype(np.float32)
                portfolio_tensor = torch.tensor(encoded_portfolio_obs, dtype=torch.float32).unsqueeze(0)
                if portfolio_decision_step:
                    logits = portfolio_actor_for(vpp_id)(portfolio_tensor).squeeze(0)
                    portfolio_dist = Categorical(logits=logits)
                    action_idx = int(portfolio_dist.sample().item())
                    portfolio_log_prob = float(portfolio_dist.log_prob(torch.tensor(action_idx)).detach().cpu().item())
                    portfolio_entropy = float(portfolio_dist.entropy().detach().cpu().item())
                    portfolio_update_mask = 1.0
                else:
                    action_idx = 0
                    portfolio_log_prob = 0.0
                    portfolio_entropy = 0.0
                    portfolio_update_mask = 0.0
                portfolio_label = ("keep", "reweight", "propose_membership_change")[action_idx]
                portfolio_action_indices[vpp_id] = action_idx
                action_payload[f"{vpp_id}_portfolio"] = {
                    "action": portfolio_label,
                    "policy_version": cfg.algorithm if portfolio_decision_step else f"{cfg.algorithm}_slow_loop_hold",
                }
                portfolio_obs_rows.append(encoded_portfolio_obs)
                portfolio_action_rows.append(action_idx)
                portfolio_log_prob_rows.append(portfolio_log_prob)
                portfolio_entropy_rows.append(portfolio_entropy)
                portfolio_update_mask_rows.append(portfolio_update_mask)

            action_summary = encode_joint_action_summary(
                normalized_dso_action=normalized_dso_action.detach().cpu().numpy().reshape(-1),
                vpp_ids=vpp_ids,
                normalized_aggregate_actions=normalized_aggregate_actions,
                normalized_der_actions=normalized_der_actions,
                portfolio_action_indices=portfolio_action_indices,
                max_der_per_vpp=max_der_per_vpp,
                action_clip=cfg.action_clip,
            ).astype(np.float32)

            next_observations, reward_map, _, truncations, infos = env.step(action_payload)
            reward_components = infos["dso_global_guidance"].get("reward_components", {})
            shield_metrics = shield_intervention_metrics(reward_components)
            violations = infos["dso_global_guidance"].get("violations", [])
            projection_audit = infos["dso_global_guidance"].get("decoded_vpp_dispatch_adjustments", {})
            decoded_projection_gap = sum(
                abs(
                    float(item.get("projected_target_p_mw", 0.0))
                    - float(item.get("dispatch_adjusted_target_p_mw", 0.0))
                )
                for item in projection_audit.values()
            )
            projection_gap = float(shield_metrics["shield_intervention_gap_mw"] or decoded_projection_gap)
            shield_penalty = float(shield_metrics["shield_intervention_penalty"])
            raw_dso_reward = float(reward_map["dso_global_guidance"])
            raw_dispatch_rewards = np.asarray([float(reward_map[f"{vpp_id}_dispatch"]) for vpp_id in vpp_ids], dtype=np.float32)
            dso_reward = raw_dso_reward - float(cfg.dso_shield_intervention_penalty_coef) * shield_penalty
            dispatch_rewards = raw_dispatch_rewards - float(cfg.dispatch_shield_intervention_penalty_coef) * shield_penalty
            portfolio_rewards = np.asarray([float(reward_map[f"{vpp_id}_portfolio"]) for vpp_id in vpp_ids], dtype=np.float32)
            if not portfolio_decision_step:
                portfolio_rewards = np.zeros_like(portfolio_rewards)
            reward_row = np.concatenate(
                [
                    np.asarray([dso_reward], dtype=np.float32),
                    dispatch_rewards,
                    portfolio_rewards,
                ],
                axis=0,
            )
            learning_reward = float(dso_reward + dispatch_rewards.mean() + portfolio_rewards.mean())

            rollout["critic_state"].append(critic_state)
            rollout["action_summary"].append(action_summary)
            rollout["dso_obs"].append(dso_obs_vec)
            rollout["dso_raw_action"].append(raw_dso_action.detach().cpu().numpy().reshape(-1))
            rollout["dso_log_prob"].append(float(dso_dist.log_prob(raw_dso_action).sum().detach().cpu().item()))
            rollout["dso_entropy"].append(float(dso_dist.entropy().sum().detach().cpu().item()))
            rollout["dispatch_obs"].append(np.asarray(dispatch_obs_rows, dtype=np.float32))
            rollout["dispatch_raw_aggregate"].append(np.asarray(dispatch_raw_aggregate_rows, dtype=np.float32))
            rollout["dispatch_raw_der"].append(np.asarray(dispatch_raw_der_rows, dtype=np.float32))
            rollout["dispatch_log_prob"].append(np.asarray(dispatch_log_prob_rows, dtype=np.float32))
            rollout["dispatch_entropy"].append(np.asarray(dispatch_entropy_rows, dtype=np.float32))
            rollout["dispatch_der_count"].append(np.asarray(dispatch_der_counts, dtype=np.int64))
            rollout["portfolio_obs"].append(np.asarray(portfolio_obs_rows, dtype=np.float32))
            rollout["portfolio_action_idx"].append(np.asarray(portfolio_action_rows, dtype=np.int64))
            rollout["portfolio_log_prob"].append(np.asarray(portfolio_log_prob_rows, dtype=np.float32))
            rollout["portfolio_entropy"].append(np.asarray(portfolio_entropy_rows, dtype=np.float32))
            rollout["portfolio_update_mask"].append(np.asarray(portfolio_update_mask_rows, dtype=np.float32))
            rollout["rewards"].append(reward_row * float(cfg.reward_scale))

            episode_reward += learning_reward
            total_cost += float(reward_components.get("total_cost", -dso_reward))
            violation_count += int(len(violations))
            projection_gap_total += float(projection_gap)
            ac_projection_gap_total += float(shield_metrics["ac_aware_projection_gap_mw"])
            local_projection_gap_total += float(shield_metrics["local_bounds_projection_gap_mw"])
            shield_intervention_penalty_total += shield_penalty
            shield_intervention_count += int(shield_metrics["shield_intervention_count"] > 0.0)
            step_rows.append(
                {
                    "episode": int(episode),
                    "step": int(step),
                    "algorithm": cfg.algorithm,
                    "reward": learning_reward,
                    "dso_reward": dso_reward,
                    "mean_dispatch_reward": float(dispatch_rewards.mean()) if dispatch_rewards.size else 0.0,
                    "mean_portfolio_reward": float(portfolio_rewards.mean()) if portfolio_rewards.size else 0.0,
                    "portfolio_decision_step": bool(portfolio_decision_step),
                    "projection_gap_mw": float(projection_gap),
                    "decoded_projection_gap_mw": float(decoded_projection_gap),
                    "action_projection_gap_mw": float(shield_metrics["action_projection_gap_mw"]),
                    "local_bounds_projection_gap_mw": float(shield_metrics["local_bounds_projection_gap_mw"]),
                    "ac_aware_projection_gap_mw": float(shield_metrics["ac_aware_projection_gap_mw"]),
                    "shield_intervention_gap_mw": float(shield_metrics["shield_intervention_gap_mw"]),
                    "shield_intervention_penalty": shield_penalty,
                    "shield_intervention_count": int(shield_metrics["shield_intervention_count"] > 0.0),
                    "raw_dso_reward_before_shield_penalty": raw_dso_reward,
                    "total_cost": float(reward_components.get("total_cost", -dso_reward)),
                    "violation_count": int(len(violations)),
                }
            )

            observations = next_observations
            if all(truncations.values()):
                break

        critic_state_tensor = torch.tensor(np.asarray(rollout["critic_state"]), dtype=torch.float32)
        action_summary_tensor = torch.tensor(np.asarray(rollout["action_summary"]), dtype=torch.float32)
        value_action_tensor = (
            action_summary_tensor
            if bool(cfg.critic_use_action_summary)
            else torch.zeros((critic_state_tensor.shape[0], value_action_dim), dtype=torch.float32)
        )
        value_matrix = value_critic(critic_state_tensor, value_action_tensor)
        rewards_matrix = np.asarray(rollout["rewards"], dtype=np.float32)
        returns_by_head: list[Any] = []
        advantages_by_head: list[Any] = []
        for head_index in range(rewards_matrix.shape[1]):
            head_returns, head_advantages = _gae_returns_advantages(
                rewards=rewards_matrix[:, head_index].tolist(),
                values=value_matrix[:, head_index],
                gamma=cfg.gamma,
                gae_lambda=cfg.gae_lambda,
                torch=torch,
            )
            returns_by_head.append(head_returns)
            advantages_by_head.append(head_advantages)
        returns_tensor = torch.stack(returns_by_head, dim=-1)
        advantages_tensor = torch.stack(advantages_by_head, dim=-1)
        if cfg.value_target_clip is not None:
            returns_tensor = torch.clamp(
                returns_tensor,
                -float(cfg.value_target_clip),
                float(cfg.value_target_clip),
            )
        critic_loss = (value_matrix - returns_tensor.detach()).pow(2).mean()
        critic_optimizer.zero_grad()
        critic_loss.backward()
        critic_grad_norm = float(torch.nn.utils.clip_grad_norm_(value_critic.parameters(), cfg.max_grad_norm))
        critic_optimizer.step()

        dso_obs_tensor = torch.tensor(np.asarray(rollout["dso_obs"]), dtype=torch.float32)
        dso_raw_action_tensor = torch.tensor(np.asarray(rollout["dso_raw_action"]), dtype=torch.float32)
        dso_old_log_prob_tensor = torch.tensor(np.asarray(rollout["dso_log_prob"]), dtype=torch.float32)
        dispatch_obs_tensor = torch.tensor(np.asarray(rollout["dispatch_obs"]), dtype=torch.float32)
        dispatch_raw_aggregate_tensor = torch.tensor(np.asarray(rollout["dispatch_raw_aggregate"]), dtype=torch.float32)
        dispatch_raw_der_tensor = torch.tensor(np.asarray(rollout["dispatch_raw_der"]), dtype=torch.float32)
        dispatch_old_log_prob_tensor = torch.tensor(np.asarray(rollout["dispatch_log_prob"]), dtype=torch.float32)
        dispatch_der_count_tensor = torch.tensor(np.asarray(rollout["dispatch_der_count"]), dtype=torch.int64)
        portfolio_obs_tensor = torch.tensor(np.asarray(rollout["portfolio_obs"]), dtype=torch.float32)
        portfolio_action_idx_tensor = torch.tensor(np.asarray(rollout["portfolio_action_idx"]), dtype=torch.int64)
        portfolio_old_log_prob_tensor = torch.tensor(np.asarray(rollout["portfolio_log_prob"]), dtype=torch.float32)
        portfolio_update_mask_tensor = torch.tensor(np.asarray(rollout["portfolio_update_mask"]), dtype=torch.bool)
        correction = torch.ones(dso_old_log_prob_tensor.shape[0], dtype=torch.float32)

        dso_policy_loss_value = 0.0
        dispatch_policy_loss_value = 0.0
        portfolio_policy_loss_value = 0.0
        dso_correction_mean = 1.0
        dispatch_correction_mean = 1.0
        portfolio_correction_mean = 1.0

        for epoch in range(cfg.ppo_epochs):
            total_correction_min = 1.0 / max(1e-6, float(cfg.importance_correction_total_clip))
            total_correction_max = float(cfg.importance_correction_total_clip)
            dso_log_prob, dso_entropy = _dso_gaussian_stats(
                actor_modules["dso_actor"],
                dso_obs_tensor,
                dso_raw_action_tensor,
                Normal,
            )
            dso_loss, dso_ratios = _happo_role_loss(
                log_probs=dso_log_prob,
                old_log_probs=dso_old_log_prob_tensor,
                advantages=advantages_tensor[:, 0],
                correction=correction,
                clip_ratio=cfg.ppo_clip_ratio,
                entropy=dso_entropy,
                entropy_coef=cfg.entropy_coef,
                torch=torch,
            )
            dso_optimizer.zero_grad()
            dso_loss.backward()
            dso_grad_norm = float(torch.nn.utils.clip_grad_norm_(actor_modules["dso_actor"].parameters(), cfg.max_grad_norm))
            dso_optimizer.step()
            dso_policy_loss_value = float(dso_loss.detach().cpu().item())
            total_role_updates += 1
            with torch.no_grad():
                updated_dso_log_prob, _ = _dso_gaussian_stats(
                    actor_modules["dso_actor"],
                    dso_obs_tensor,
                    dso_raw_action_tensor,
                    Normal,
                )
                correction = correction * torch.clamp(
                    torch.exp(updated_dso_log_prob - dso_old_log_prob_tensor),
                    1.0 / float(cfg.importance_correction_clip),
                    float(cfg.importance_correction_clip),
                )
                correction = torch.clamp(correction, total_correction_min, total_correction_max)
                dso_correction_mean = float(correction.mean().cpu().item())
            update_rows.append(
                {
                    "episode": int(episode),
                    "epoch": int(epoch),
                    "role": "dso_global_guidance",
                    "policy_loss": dso_policy_loss_value,
                    "ratio_mean": float(dso_ratios.detach().mean().cpu().item()),
                    "correction_mean": dso_correction_mean,
                    "grad_norm": dso_grad_norm,
                }
            )

            if cfg.share_vpp_dispatch_parameters:
                flat_dispatch_obs = dispatch_obs_tensor.reshape(-1, dispatch_obs_tensor.shape[-1])
                flat_dispatch_raw_aggregate = dispatch_raw_aggregate_tensor.reshape(-1, 1)
                flat_dispatch_raw_der = dispatch_raw_der_tensor.reshape(-1, dispatch_raw_der_tensor.shape[-1])
                flat_dispatch_counts = dispatch_der_count_tensor.reshape(-1)
                flat_dispatch_old_log_prob = dispatch_old_log_prob_tensor.reshape(-1)
                flat_dispatch_advantages = advantages_tensor[:, 1 : 1 + len(vpp_ids)].reshape(-1)
                dispatch_log_prob, dispatch_entropy = _dispatch_gaussian_stats(
                    actor_modules["vpp_dispatch_actor"],
                    flat_dispatch_obs,
                    flat_dispatch_raw_aggregate,
                    flat_dispatch_raw_der,
                    flat_dispatch_counts,
                    Normal,
                    torch,
                )
                dispatch_loss, dispatch_ratios = _happo_role_loss(
                    log_probs=dispatch_log_prob,
                    old_log_probs=flat_dispatch_old_log_prob,
                    advantages=flat_dispatch_advantages,
                    correction=correction.unsqueeze(1).repeat(1, len(vpp_ids)).reshape(-1),
                    clip_ratio=cfg.ppo_clip_ratio,
                    entropy=dispatch_entropy,
                    entropy_coef=cfg.entropy_coef,
                    torch=torch,
                )
                dispatch_optimizers["shared_vpp_dispatch"].zero_grad()
                dispatch_loss.backward()
                dispatch_grad_norm = float(
                    torch.nn.utils.clip_grad_norm_(actor_modules["vpp_dispatch_actor"].parameters(), cfg.max_grad_norm)
                )
                dispatch_optimizers["shared_vpp_dispatch"].step()
                dispatch_policy_loss_value = float(dispatch_loss.detach().cpu().item())
                total_role_updates += 1
                with torch.no_grad():
                    updated_dispatch_log_prob, _ = _dispatch_gaussian_stats(
                        actor_modules["vpp_dispatch_actor"],
                        flat_dispatch_obs,
                        flat_dispatch_raw_aggregate,
                        flat_dispatch_raw_der,
                        flat_dispatch_counts,
                        Normal,
                        torch,
                    )
                    dispatch_step_ratio = torch.exp(
                        updated_dispatch_log_prob.reshape(-1, len(vpp_ids)) - dispatch_old_log_prob_tensor
                    )
                    dispatch_step_ratio = torch.clamp(
                        dispatch_step_ratio,
                        1.0 / float(cfg.importance_correction_clip),
                        float(cfg.importance_correction_clip),
                    )
                    correction = correction * dispatch_step_ratio.prod(dim=1)
                    correction = torch.clamp(correction, total_correction_min, total_correction_max)
                    dispatch_correction_mean = float(correction.mean().cpu().item())
                update_rows.append(
                    {
                        "episode": int(episode),
                        "epoch": int(epoch),
                        "role": "shared_vpp_dispatch",
                        "policy_loss": dispatch_policy_loss_value,
                        "ratio_mean": float(dispatch_ratios.detach().mean().cpu().item()),
                        "correction_mean": dispatch_correction_mean,
                        "grad_norm": dispatch_grad_norm,
                    }
                )
            else:
                dispatch_losses_for_episode: list[float] = []
                for vpp_index, vpp_id in enumerate(vpp_ids):
                    dispatch_log_prob, dispatch_entropy = _dispatch_gaussian_stats(
                        actor_modules[f"{vpp_id}_dispatch_actor"],
                        dispatch_obs_tensor[:, vpp_index, :],
                        dispatch_raw_aggregate_tensor[:, vpp_index, :],
                        dispatch_raw_der_tensor[:, vpp_index, :],
                        dispatch_der_count_tensor[:, vpp_index],
                        Normal,
                        torch,
                    )
                    dispatch_loss, dispatch_ratios = _happo_role_loss(
                        log_probs=dispatch_log_prob,
                        old_log_probs=dispatch_old_log_prob_tensor[:, vpp_index],
                        advantages=advantages_tensor[:, 1 + vpp_index],
                        correction=correction,
                        clip_ratio=cfg.ppo_clip_ratio,
                        entropy=dispatch_entropy,
                        entropy_coef=cfg.entropy_coef,
                        torch=torch,
                    )
                    dispatch_optimizers[vpp_id].zero_grad()
                    dispatch_loss.backward()
                    dispatch_grad_norm = float(
                        torch.nn.utils.clip_grad_norm_(
                            actor_modules[f"{vpp_id}_dispatch_actor"].parameters(),
                            cfg.max_grad_norm,
                        )
                    )
                    dispatch_optimizers[vpp_id].step()
                    dispatch_policy_loss_value = float(dispatch_loss.detach().cpu().item())
                    dispatch_losses_for_episode.append(dispatch_policy_loss_value)
                    total_role_updates += 1
                    with torch.no_grad():
                        updated_dispatch_log_prob, _ = _dispatch_gaussian_stats(
                            actor_modules[f"{vpp_id}_dispatch_actor"],
                            dispatch_obs_tensor[:, vpp_index, :],
                            dispatch_raw_aggregate_tensor[:, vpp_index, :],
                            dispatch_raw_der_tensor[:, vpp_index, :],
                            dispatch_der_count_tensor[:, vpp_index],
                            Normal,
                            torch,
                        )
                        dispatch_step_ratio = torch.clamp(
                            torch.exp(updated_dispatch_log_prob - dispatch_old_log_prob_tensor[:, vpp_index]),
                            1.0 / float(cfg.importance_correction_clip),
                            float(cfg.importance_correction_clip),
                        )
                        correction = correction * dispatch_step_ratio
                        correction = torch.clamp(correction, total_correction_min, total_correction_max)
                        dispatch_correction_mean = float(correction.mean().cpu().item())
                    update_rows.append(
                        {
                            "episode": int(episode),
                            "epoch": int(epoch),
                            "role": f"{vpp_id}_dispatch",
                            "target_vpp_id": vpp_id,
                            "policy_loss": dispatch_policy_loss_value,
                            "ratio_mean": float(dispatch_ratios.detach().mean().cpu().item()),
                            "correction_mean": dispatch_correction_mean,
                            "grad_norm": dispatch_grad_norm,
                        }
                    )
                if dispatch_losses_for_episode:
                    dispatch_policy_loss_value = float(np.mean(dispatch_losses_for_episode))

            if cfg.share_vpp_portfolio_parameters:
                flat_portfolio_mask = portfolio_update_mask_tensor.reshape(-1)
                if bool(flat_portfolio_mask.any()):
                    flat_portfolio_obs = portfolio_obs_tensor.reshape(-1, portfolio_obs_tensor.shape[-1])[flat_portfolio_mask]
                    flat_portfolio_action_idx = portfolio_action_idx_tensor.reshape(-1)[flat_portfolio_mask]
                    flat_portfolio_old_log_prob = portfolio_old_log_prob_tensor.reshape(-1)[flat_portfolio_mask]
                    flat_portfolio_advantages = advantages_tensor[:, 1 + len(vpp_ids) :].reshape(-1)[flat_portfolio_mask]
                    flat_portfolio_correction = correction.unsqueeze(1).repeat(1, len(vpp_ids)).reshape(-1)[flat_portfolio_mask]
                    portfolio_log_prob, portfolio_entropy = _portfolio_categorical_stats(
                        actor_modules["vpp_portfolio_actor"],
                        flat_portfolio_obs,
                        flat_portfolio_action_idx,
                        Categorical,
                    )
                    portfolio_loss, portfolio_ratios = _happo_role_loss(
                        log_probs=portfolio_log_prob,
                        old_log_probs=flat_portfolio_old_log_prob,
                        advantages=flat_portfolio_advantages,
                        correction=flat_portfolio_correction,
                        clip_ratio=cfg.ppo_clip_ratio,
                        entropy=portfolio_entropy,
                        entropy_coef=cfg.entropy_coef,
                        torch=torch,
                    )
                    portfolio_optimizers["shared_vpp_portfolio"].zero_grad()
                    portfolio_loss.backward()
                    portfolio_grad_norm = float(
                        torch.nn.utils.clip_grad_norm_(actor_modules["vpp_portfolio_actor"].parameters(), cfg.max_grad_norm)
                    )
                    portfolio_optimizers["shared_vpp_portfolio"].step()
                    portfolio_policy_loss_value = float(portfolio_loss.detach().cpu().item())
                    total_role_updates += 1
                    with torch.no_grad():
                        full_portfolio_obs = portfolio_obs_tensor.reshape(-1, portfolio_obs_tensor.shape[-1])
                        full_portfolio_action_idx = portfolio_action_idx_tensor.reshape(-1)
                        updated_portfolio_log_prob, _ = _portfolio_categorical_stats(
                            actor_modules["vpp_portfolio_actor"],
                            full_portfolio_obs,
                            full_portfolio_action_idx,
                            Categorical,
                        )
                        portfolio_step_ratio = torch.exp(
                            updated_portfolio_log_prob.reshape(-1, len(vpp_ids)) - portfolio_old_log_prob_tensor
                        )
                        portfolio_step_ratio = torch.clamp(
                            portfolio_step_ratio,
                            1.0 / float(cfg.importance_correction_clip),
                            float(cfg.importance_correction_clip),
                        )
                        masked_ratio = torch.where(portfolio_update_mask_tensor, portfolio_step_ratio, torch.ones_like(portfolio_step_ratio))
                        correction = correction * masked_ratio.prod(dim=1)
                        correction = torch.clamp(correction, total_correction_min, total_correction_max)
                        portfolio_correction_mean = float(correction.mean().cpu().item())
                    update_rows.append(
                        {
                            "episode": int(episode),
                            "epoch": int(epoch),
                            "role": "shared_vpp_portfolio",
                            "policy_loss": portfolio_policy_loss_value,
                            "ratio_mean": float(portfolio_ratios.detach().mean().cpu().item()),
                            "correction_mean": portfolio_correction_mean,
                            "grad_norm": portfolio_grad_norm,
                        }
                    )
            else:
                portfolio_losses_for_episode: list[float] = []
                for vpp_index, vpp_id in enumerate(vpp_ids):
                    mask = portfolio_update_mask_tensor[:, vpp_index]
                    if not bool(mask.any()):
                        continue
                    portfolio_log_prob, portfolio_entropy = _portfolio_categorical_stats(
                        actor_modules[f"{vpp_id}_portfolio_actor"],
                        portfolio_obs_tensor[:, vpp_index, :][mask],
                        portfolio_action_idx_tensor[:, vpp_index][mask],
                        Categorical,
                    )
                    portfolio_loss, portfolio_ratios = _happo_role_loss(
                        log_probs=portfolio_log_prob,
                        old_log_probs=portfolio_old_log_prob_tensor[:, vpp_index][mask],
                        advantages=advantages_tensor[:, 1 + len(vpp_ids) + vpp_index][mask],
                        correction=correction[mask],
                        clip_ratio=cfg.ppo_clip_ratio,
                        entropy=portfolio_entropy,
                        entropy_coef=cfg.entropy_coef,
                        torch=torch,
                    )
                    portfolio_optimizers[vpp_id].zero_grad()
                    portfolio_loss.backward()
                    portfolio_grad_norm = float(
                        torch.nn.utils.clip_grad_norm_(
                            actor_modules[f"{vpp_id}_portfolio_actor"].parameters(),
                            cfg.max_grad_norm,
                        )
                    )
                    portfolio_optimizers[vpp_id].step()
                    portfolio_policy_loss_value = float(portfolio_loss.detach().cpu().item())
                    portfolio_losses_for_episode.append(portfolio_policy_loss_value)
                    total_role_updates += 1
                    with torch.no_grad():
                        updated_portfolio_log_prob, _ = _portfolio_categorical_stats(
                            actor_modules[f"{vpp_id}_portfolio_actor"],
                            portfolio_obs_tensor[:, vpp_index, :][mask],
                            portfolio_action_idx_tensor[:, vpp_index][mask],
                            Categorical,
                        )
                        portfolio_step_ratio = torch.clamp(
                            torch.exp(updated_portfolio_log_prob - portfolio_old_log_prob_tensor[:, vpp_index][mask]),
                            1.0 / float(cfg.importance_correction_clip),
                            float(cfg.importance_correction_clip),
                        )
                        next_correction = correction.clone()
                        next_correction[mask] = next_correction[mask] * portfolio_step_ratio
                        correction = torch.clamp(next_correction, total_correction_min, total_correction_max)
                        portfolio_correction_mean = float(correction.mean().cpu().item())
                    update_rows.append(
                        {
                            "episode": int(episode),
                            "epoch": int(epoch),
                            "role": f"{vpp_id}_portfolio",
                            "target_vpp_id": vpp_id,
                            "policy_loss": portfolio_policy_loss_value,
                            "ratio_mean": float(portfolio_ratios.detach().mean().cpu().item()),
                            "correction_mean": portfolio_correction_mean,
                            "grad_norm": portfolio_grad_norm,
                            "slow_loop_updates": int(mask.sum().cpu().item()),
                        }
                    )
                if portfolio_losses_for_episode:
                    portfolio_policy_loss_value = float(np.mean(portfolio_losses_for_episode))

        episode_rows.append(
            {
                "episode": int(episode),
                "algorithm": cfg.algorithm,
                "episode_reward": float(episode_reward),
                "episode_cost": float(total_cost),
                "violation_count": int(violation_count),
                "projection_gap_mw": float(projection_gap_total),
                "local_bounds_projection_gap_mw": float(local_projection_gap_total),
                "ac_aware_projection_gap_mw": float(ac_projection_gap_total),
                "shield_intervention_gap_mw": float(projection_gap_total),
                "shield_intervention_penalty": float(shield_intervention_penalty_total),
                "shield_intervention_count": int(shield_intervention_count),
                "critic_loss": float(critic_loss.detach().cpu().item()),
                "critic_grad_norm": critic_grad_norm,
                "dso_policy_loss": dso_policy_loss_value,
                "dispatch_policy_loss": dispatch_policy_loss_value,
                "portfolio_policy_loss": portfolio_policy_loss_value,
                "dso_correction_mean": dso_correction_mean,
                "dispatch_correction_mean": dispatch_correction_mean,
                "portfolio_correction_mean": portfolio_correction_mean,
            }
        )
        if float(episode_reward) > best_episode_reward:
            best_episode_reward = float(episode_reward)
            best_episode_index = int(episode)
            best_checkpoint_state = {
                "actor_state_dict": copy.deepcopy(actor_modules.state_dict()),
                "value_critic_state_dict": copy.deepcopy(value_critic.state_dict()),
            }
        _set_episode_postfix(
            episode_iter,
            has_tqdm_progress,
            reward=f"{episode_reward:.2f}",
            cost=f"{total_cost:.1f}",
            viol=int(violation_count),
            gap=f"{projection_gap_total:.4f}",
            critic=f"{float(critic_loss.detach().cpu().item()):.3f}",
        )
        now = time.monotonic()
        if (
            not has_tqdm_progress
            and (episode == 0 or episode == cfg.episodes - 1 or now - last_progress_print >= progress_interval_seconds)
        ):
            last_progress_print = now
            print(
                "[HAPPO] "
                f"episode={episode + 1}/{cfg.episodes} "
                f"reward={episode_reward:.4f} "
                f"cost={total_cost:.4f} "
                f"violations={violation_count} "
                f"projection_gap_mw={projection_gap_total:.6f} "
                f"critic_loss={float(critic_loss.detach().cpu().item()):.6f} "
                f"dso_loss={dso_policy_loss_value:.6f} "
                f"dispatch_loss={dispatch_policy_loss_value:.6f} "
                f"portfolio_loss={portfolio_policy_loss_value:.6f}",
                flush=True,
            )
        env.close()

    episode_metrics = pd.DataFrame(episode_rows)
    step_metrics = pd.DataFrame(step_rows)
    update_metrics = pd.DataFrame(update_rows)
    actor_update_order = ",".join(
        [
            "dso_global_guidance",
            *(
                ["shared_vpp_dispatch"]
                if cfg.share_vpp_dispatch_parameters
                else [f"{vpp_id}_dispatch" for vpp_id in vpp_ids]
            ),
            *(
                ["shared_vpp_portfolio"]
                if cfg.share_vpp_portfolio_parameters
                else [f"{vpp_id}_portfolio" for vpp_id in vpp_ids]
            ),
        ]
    )
    final_params = torch.cat(
        [
            param.detach().flatten().cpu()
            for param in [*actor_modules.parameters(), *value_critic.parameters()]
        ]
    )
    param_delta_l2 = float(torch.norm(final_params - initial_params).item())
    checkpoint_path = out / "happo_checkpoint.pt"
    best_checkpoint_path = out / "happo_best_checkpoint.pt"

    def checkpoint_payload(actor_state: Any, value_state: Any) -> dict[str, Any]:
        return {
            "config": cfg.to_dict(),
            "actor_state_dict": actor_state,
            "value_critic_state_dict": value_state,
            "value_spec": value_spec.to_dict(),
            "vpp_ids": vpp_ids,
            "der_ids_by_vpp": der_ids_by_vpp,
            "max_der_per_vpp": max_der_per_vpp,
            "dso_input_dim": dso_input_dim,
            "vpp_input_dim": vpp_input_dim,
            "portfolio_input_dim": portfolio_input_dim,
            "critic_input_dim": critic_input_dim,
            "critic_action_dim": value_action_dim,
            "joint_action_summary_dim": critic_action_dim,
            "critic_use_action_summary": bool(cfg.critic_use_action_summary),
            "policy_signature": policy_signature,
            "architecture_meta": architecture_meta,
            "selection_metric": "episode_reward",
        }

    torch.save(
        checkpoint_payload(actor_modules.state_dict(), value_critic.state_dict()),
        checkpoint_path,
    )
    if best_checkpoint_state is None:
        best_checkpoint_state = {
            "actor_state_dict": copy.deepcopy(actor_modules.state_dict()),
            "value_critic_state_dict": copy.deepcopy(value_critic.state_dict()),
        }
        best_episode_index = int(episode_metrics["episode"].iloc[-1]) if not episode_metrics.empty else -1
    torch.save(
        checkpoint_payload(
            best_checkpoint_state["actor_state_dict"],
            best_checkpoint_state["value_critic_state_dict"],
        ),
        best_checkpoint_path,
    )
    summary = {
        "algorithm": cfg.algorithm,
        "status": "completed",
        "is_deep_rl": True,
        "deep_learning_framework": "torch",
        "training_pattern": "on_policy_ctde",
        "sequential_role_update": True,
        "importance_correction": True,
        "actor_update_order": actor_update_order,
        "value_head_names": ",".join(head_names),
        "value_head_count": int(len(head_names)),
        "per_vpp_dispatch_actors": not bool(cfg.share_vpp_dispatch_parameters),
        "per_vpp_portfolio_actors": not bool(cfg.share_vpp_portfolio_parameters),
        "shared_dispatch_parameters": bool(cfg.share_vpp_dispatch_parameters),
        "shared_portfolio_parameters": bool(cfg.share_vpp_portfolio_parameters),
        "portfolio_agent_timescale": "slow_loop",
        "portfolio_decision_interval_steps": int(cfg.portfolio_decision_interval_steps),
        "portfolio_decision_step_count": int(step_metrics["portfolio_decision_step"].sum())
        if not step_metrics.empty and "portfolio_decision_step" in step_metrics
        else 0,
        "portfolio_force_keep_between_decisions": bool(cfg.portfolio_force_keep_between_decisions),
        "ppo_epochs": int(cfg.ppo_epochs),
        "ppo_clip_ratio": float(cfg.ppo_clip_ratio),
        "gae_lambda": float(cfg.gae_lambda),
        "importance_correction_clip": float(cfg.importance_correction_clip),
        "importance_correction_total_clip": float(cfg.importance_correction_total_clip),
        "reward_scale": float(cfg.reward_scale),
        "value_target_clip": None if cfg.value_target_clip is None else float(cfg.value_target_clip),
        "critic_use_action_summary": bool(cfg.critic_use_action_summary),
        "critic_baseline_type": "action_conditioned" if bool(cfg.critic_use_action_summary) else "state_only",
        "joint_action_summary_dim": int(critic_action_dim),
        "critic_action_dim": int(value_action_dim),
        "shield_intervention_penalty_in_role_rewards": True,
        "dso_shield_intervention_penalty_coef": float(cfg.dso_shield_intervention_penalty_coef),
        "dispatch_shield_intervention_penalty_coef": float(cfg.dispatch_shield_intervention_penalty_coef),
        "total_shield_intervention_gap_mw": float(episode_metrics["shield_intervention_gap_mw"].sum())
        if not episode_metrics.empty and "shield_intervention_gap_mw" in episode_metrics
        else 0.0,
        "total_shield_intervention_penalty": float(episode_metrics["shield_intervention_penalty"].sum())
        if not episode_metrics.empty and "shield_intervention_penalty" in episode_metrics
        else 0.0,
        "shield_intervention_step_count": int(episode_metrics["shield_intervention_count"].sum())
        if not episode_metrics.empty and "shield_intervention_count" in episode_metrics
        else 0,
        "episodes": int(cfg.episodes),
        "horizon_steps": int(cfg.horizon_steps),
        "role_update_steps": int(total_role_updates),
        "param_delta_l2": param_delta_l2,
        "best_episode_reward": float(episode_metrics["episode_reward"].max()) if not episode_metrics.empty else None,
        "final_episode_reward": float(episode_metrics["episode_reward"].iloc[-1]) if not episode_metrics.empty else None,
        "final_checkpoint": str(checkpoint_path),
        "best_checkpoint": str(best_checkpoint_path),
        "best_checkpoint_episode": int(best_episode_index),
        "selected_checkpoint_policy": "train_best_episode_reward",
        "checkpoint": str(best_checkpoint_path),
        "claim_boundary": (
            "Target hierarchical HAPPO scaffold: centralized multi-head state-value critic, DSO update, per-VPP dispatch "
            "updates, slow-loop per-VPP portfolio updates and cumulative importance correction are implemented. "
            "Physical portfolio membership changes remain gated by the scenario layer, so the portfolio actor learns "
            "commercial configuration proposals rather than silently moving pandapower elements."
        ),
    }
    episode_metrics.to_csv(out / "happo_episode_metrics.csv", index=False)
    step_metrics.to_csv(out / "happo_step_metrics.csv", index=False)
    update_metrics.to_csv(out / "happo_update_metrics.csv", index=False)
    pd.DataFrame([summary]).to_csv(out / "happo_training_summary.csv", index=False)
    write_json(out / "happo_training_summary.json", summary)
    write_json(out / "happo_config.json", cfg.to_dict())
    return {
        "summary": summary,
        "episode_metrics": episode_metrics,
        "step_metrics": step_metrics,
        "update_metrics": update_metrics,
        "checkpoint": best_checkpoint_path,
        "final_checkpoint": checkpoint_path,
        "best_checkpoint": best_checkpoint_path,
        "output_dir": out,
    }


def train_hasac(
    config_path: str | Path | None = None,
    output_dir: str | Path = "outputs/hasac",
    config: HASACConfig | None = None,
) -> dict[str, Any]:
    """Train a runnable HASAC-style continuous-control scaffold.

    Scope:
    - DSO continuous guidance actor and per-VPP dispatch actors by default.
    - Centralized multi-head twin soft Q critics.
    - Off-policy replay, entropy temperature tuning and soft target backups.
    - Slow discrete portfolio action remains fixed at `keep`.
    """

    from vpp_dso_sim.envs.multi_agent_env import MultiAgentVPPDSOEnv
    from vpp_dso_sim.learning.deep_rl import _build_privacy_separated_networks
    from vpp_dso_sim.learning.reward_contracts import shield_intervention_metrics
    from vpp_dso_sim.utils.io import ensure_dir, write_json

    cfg = config or HASACConfig()
    torch, optim, _, _ = _require_torch()
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    out = ensure_dir(output_dir)

    env_probe = MultiAgentVPPDSOEnv(config_path=config_path, horizon_steps=cfg.horizon_steps)
    observations, _ = env_probe.reset(seed=cfg.seed)
    policy_signature = env_probe.policy_compatibility_signature()
    vpp_ids = [vpp.id for vpp in env_probe.scenario.vpps]
    der_ids_by_vpp = {vpp.id: [der.id for der in vpp.der_list] for vpp in env_probe.scenario.vpps}
    max_der_per_vpp = max(1, max((len(ids) for ids in der_ids_by_vpp.values()), default=1))
    dso_obs_vec, vpp_obs_mat, critic_vec = _continuous_step_observations(
        observations,
        env=env_probe,
        vpp_ids=vpp_ids,
        max_der_per_vpp=max_der_per_vpp,
    )
    dso_input_dim = int(len(dso_obs_vec))
    vpp_input_dim = int(vpp_obs_mat.shape[-1])
    critic_input_dim = int(len(critic_vec))
    joint_action_dim = len(vpp_ids) * (2 + max_der_per_vpp)
    env_probe.close()

    modules, architecture_meta = _build_privacy_separated_networks(
        dso_input_dim=dso_input_dim,
        vpp_input_dim=vpp_input_dim,
        portfolio_input_dim=9,
        critic_input_dim=critic_input_dim,
        critic_action_dim=1,
        action_dim=len(vpp_ids),
        der_action_dim=max_der_per_vpp,
        hidden_dim=cfg.hidden_dim,
    )
    actor_modules = torch.nn.ModuleDict({"dso_actor": modules["dso_actor"]})
    if cfg.share_vpp_dispatch_parameters:
        actor_modules["vpp_dispatch_actor"] = modules["vpp_dispatch_actor"]
    else:
        for vpp_id in vpp_ids:
            actor_modules[f"{vpp_id}_dispatch_actor"] = copy.deepcopy(modules["vpp_dispatch_actor"])
    critic_head_names = ["dso_global_guidance", *[f"{vpp_id}_dispatch" for vpp_id in vpp_ids]]
    critic_spec = TwinCriticSpec(
        state_dim=critic_input_dim,
        joint_action_dim=joint_action_dim,
        hidden_dims=(cfg.hidden_dim, cfg.hidden_dim),
        output_dim=len(critic_head_names),
        algorithm_style="hasac_centralized_twin_soft_q",
        input_contract="critic_global_state + differentiable_flat_joint_continuous_actions",
    )
    role_critic = build_twin_critic(critic_spec, require_torch=True)
    target_role_critic = copy.deepcopy(role_critic)
    dso_actor_optimizer = optim.Adam(actor_modules["dso_actor"].parameters(), lr=float(cfg.actor_learning_rate))
    dispatch_actor_params = [
        param
        for name, module in actor_modules.items()
        if name != "dso_actor"
        for param in module.parameters()
    ]
    dispatch_actor_optimizer = optim.Adam(dispatch_actor_params, lr=float(cfg.actor_learning_rate))
    critic_optimizer = optim.Adam(role_critic.parameters(), lr=float(cfg.critic_learning_rate))
    replay = OffPolicyReplayBuffer(capacity=cfg.replay_capacity, seed=cfg.seed)

    dso_action_dim = len(vpp_ids)
    dispatch_action_dim = 1 + max_der_per_vpp
    target_entropy_dso = float(
        cfg.target_entropy_dso
        if cfg.target_entropy_dso is not None
        else -float(dso_action_dim) * float(cfg.target_entropy_multiplier)
    )
    target_entropy_dispatch = float(
        cfg.target_entropy_dispatch
        if cfg.target_entropy_dispatch is not None
        else -float(dispatch_action_dim) * float(cfg.target_entropy_multiplier)
    )
    log_alpha_dso = torch.tensor(float(cfg.init_log_alpha_dso), dtype=torch.float32, requires_grad=True)
    log_alpha_dispatch = torch.tensor(float(cfg.init_log_alpha_dispatch), dtype=torch.float32, requires_grad=True)
    alpha_optimizer = optim.Adam([log_alpha_dso, log_alpha_dispatch], lr=float(cfg.alpha_learning_rate))

    def dispatch_actor_for(vpp_id: str) -> Any:
        return (
            actor_modules["vpp_dispatch_actor"]
            if cfg.share_vpp_dispatch_parameters
            else actor_modules[f"{vpp_id}_dispatch_actor"]
        )

    der_count_tensor = torch.tensor(
        [len(der_ids_by_vpp[vpp_id]) for vpp_id in vpp_ids],
        dtype=torch.long,
    )

    def sample_dispatch_batch(vpp_obs_tensor: Any) -> tuple[Any, Any, Any]:
        aggregate_actions: list[Any] = []
        der_actions: list[Any] = []
        dispatch_log_probs: list[Any] = []
        for vpp_index, vpp_id in enumerate(vpp_ids):
            aggregate_mean, aggregate_log_std, der_mean, der_log_std = dispatch_actor_for(vpp_id)(
                vpp_obs_tensor[:, vpp_index, :]
            )
            aggregate_action, aggregate_log_prob_dims, _ = _sample_squashed_gaussian_with_log_prob_dims(
                aggregate_mean,
                aggregate_log_std,
                action_clip=cfg.action_clip,
                torch=torch,
            )
            der_action, der_log_prob_dims, _ = _sample_squashed_gaussian_with_log_prob_dims(
                der_mean,
                der_log_std,
                action_clip=cfg.action_clip,
                torch=torch,
            )
            der_mask = (
                torch.arange(der_action.shape[-1], device=der_action.device).unsqueeze(0)
                < int(der_count_tensor[vpp_index].item())
            ).float()
            aggregate_log_prob = aggregate_log_prob_dims.reshape(aggregate_log_prob_dims.shape[0], -1).sum(dim=-1)
            der_log_prob = (der_log_prob_dims * der_mask).reshape(der_log_prob_dims.shape[0], -1).sum(dim=-1)
            aggregate_actions.append(aggregate_action)
            der_actions.append(der_action.unsqueeze(1))
            dispatch_log_probs.append((aggregate_log_prob + der_log_prob).unsqueeze(1))
        return torch.cat(aggregate_actions, dim=1), torch.cat(der_actions, dim=1), torch.cat(dispatch_log_probs, dim=1)

    episode_rows: list[dict[str, Any]] = []
    step_rows: list[dict[str, Any]] = []
    update_rows: list[dict[str, Any]] = []
    total_env_steps = 0
    critic_updates = 0
    actor_updates = 0
    alpha_updates = 0
    best_episode_reward = float("-inf")
    best_episode_index = -1
    best_checkpoint_state: dict[str, Any] | None = None

    last_progress_print = 0.0
    progress_interval_seconds = 60.0
    episode_iter, has_tqdm_progress = _episode_progress(range(cfg.episodes), total=cfg.episodes, desc="HASAC")
    for episode in episode_iter:
        env = MultiAgentVPPDSOEnv(config_path=config_path, horizon_steps=cfg.horizon_steps)
        observations, _ = env.reset(seed=cfg.seed + episode)
        dso_obs_vec, vpp_obs_mat, critic_vec = _continuous_step_observations(
            observations,
            env=env,
            vpp_ids=vpp_ids,
            max_der_per_vpp=max_der_per_vpp,
        )
        episode_reward = 0.0
        total_cost = 0.0
        violation_count = 0
        projection_gap_total = 0.0
        ac_projection_gap_total = 0.0
        local_projection_gap_total = 0.0
        shield_intervention_penalty_total = 0.0
        shield_intervention_count = 0

        for step in range(cfg.horizon_steps):
            with torch.no_grad():
                dso_obs_tensor = torch.tensor(dso_obs_vec, dtype=torch.float32).unsqueeze(0)
                vpp_obs_tensor = torch.tensor(vpp_obs_mat, dtype=torch.float32).unsqueeze(0)
                dso_mean, dso_log_std = actor_modules["dso_actor"](dso_obs_tensor)
                dso_action_t, _, _ = _sample_squashed_gaussian(
                    dso_mean,
                    dso_log_std,
                    action_clip=cfg.action_clip,
                    torch=torch,
                )
                aggregate_action_t, der_action_t, _ = sample_dispatch_batch(vpp_obs_tensor)
                batch_size = int(dso_action_t.shape[0])
                joint_action_t = torch.cat(
                    [dso_action_t, aggregate_action_t, der_action_t.reshape(batch_size, -1)],
                    dim=-1,
                )

            action_payload = _continuous_payload_from_actions(
                dso_action=dso_action_t.cpu().numpy().reshape(-1),
                aggregate_actions=aggregate_action_t.cpu().numpy().reshape(len(vpp_ids)),
                der_actions=der_action_t.cpu().numpy().reshape(len(vpp_ids), max_der_per_vpp),
                dso_obs=observations["dso_global_guidance"],
                vpp_observations=observations,
                vpp_ids=vpp_ids,
                der_ids_by_vpp=der_ids_by_vpp,
                action_clip=cfg.action_clip,
                policy_version=cfg.algorithm,
            )
            next_observations, reward_map, _, truncations, infos = env.step(action_payload)
            reward_components = infos["dso_global_guidance"].get("reward_components", {})
            shield_metrics = shield_intervention_metrics(reward_components)
            violations = infos["dso_global_guidance"].get("violations", [])
            projection_audit = infos["dso_global_guidance"].get("decoded_vpp_dispatch_adjustments", {})
            decoded_projection_gap = sum(
                abs(
                    float(item.get("projected_target_p_mw", 0.0))
                    - float(item.get("dispatch_adjusted_target_p_mw", 0.0))
                )
                for item in projection_audit.values()
            )
            projection_gap = float(shield_metrics["shield_intervention_gap_mw"] or decoded_projection_gap)
            shield_penalty = float(shield_metrics["shield_intervention_penalty"])
            raw_dso_reward = float(reward_map["dso_global_guidance"])
            raw_dispatch_rewards = np.asarray([float(reward_map[f"{vpp_id}_dispatch"]) for vpp_id in vpp_ids], dtype=np.float32)
            dso_reward = raw_dso_reward - float(cfg.dso_shield_intervention_penalty_coef) * shield_penalty
            dispatch_rewards = raw_dispatch_rewards - float(cfg.dispatch_shield_intervention_penalty_coef) * shield_penalty
            dispatch_reward_mean = float(dispatch_rewards.mean()) if dispatch_rewards.size else 0.0
            reward_vector = np.concatenate(
                [
                    np.asarray([dso_reward], dtype=np.float32),
                    dispatch_rewards,
                ],
                axis=0,
            )
            next_dso_obs_vec, next_vpp_obs_mat, next_critic_vec = _continuous_step_observations(
                next_observations,
                env=env,
                vpp_ids=vpp_ids,
                max_der_per_vpp=max_der_per_vpp,
            )
            done = bool(all(truncations.values()))
            replay.add(
                {
                    "dso_obs": dso_obs_vec,
                    "vpp_obs": vpp_obs_mat,
                    "critic_state": critic_vec,
                    "joint_action": joint_action_t.cpu().numpy().reshape(-1),
                    "next_dso_obs": next_dso_obs_vec,
                    "next_vpp_obs": next_vpp_obs_mat,
                    "next_critic_state": next_critic_vec,
                    "reward_vector": reward_vector,
                    "shield_intervention_gap_mw": float(shield_metrics["shield_intervention_gap_mw"]),
                    "shield_intervention_penalty": shield_penalty,
                    "done": float(done),
                }
            )

            episode_reward += float(dso_reward + dispatch_reward_mean)
            total_cost += float(reward_components.get("total_cost", -dso_reward))
            violation_count += int(len(violations))
            projection_gap_total += float(projection_gap)
            ac_projection_gap_total += float(shield_metrics["ac_aware_projection_gap_mw"])
            local_projection_gap_total += float(shield_metrics["local_bounds_projection_gap_mw"])
            shield_intervention_penalty_total += shield_penalty
            shield_intervention_count += int(shield_metrics["shield_intervention_count"] > 0.0)
            total_env_steps += 1
            step_rows.append(
                {
                    "episode": int(episode),
                    "step": int(step),
                    "global_step": int(total_env_steps),
                    "algorithm": cfg.algorithm,
                    "reward": float(dso_reward + dispatch_reward_mean),
                    "dso_reward": dso_reward,
                    "vpp_dispatch_reward": dispatch_reward_mean,
                    "projection_gap_mw": float(projection_gap),
                    "decoded_projection_gap_mw": float(decoded_projection_gap),
                    "action_projection_gap_mw": float(shield_metrics["action_projection_gap_mw"]),
                    "local_bounds_projection_gap_mw": float(shield_metrics["local_bounds_projection_gap_mw"]),
                    "ac_aware_projection_gap_mw": float(shield_metrics["ac_aware_projection_gap_mw"]),
                    "shield_intervention_gap_mw": float(shield_metrics["shield_intervention_gap_mw"]),
                    "shield_intervention_penalty": shield_penalty,
                    "shield_intervention_count": int(shield_metrics["shield_intervention_count"] > 0.0),
                    "raw_dso_reward_before_shield_penalty": raw_dso_reward,
                    "total_cost": float(reward_components.get("total_cost", -dso_reward)),
                    "violation_count": int(len(violations)),
                    "replay_size": int(len(replay)),
                    "action_min": float(joint_action_t.min().cpu().item()),
                    "action_max": float(joint_action_t.max().cpu().item()),
                }
            )

            if len(replay) >= int(cfg.batch_size) and total_env_steps >= int(cfg.warmup_steps):
                batch = replay.sample(cfg.batch_size)
                critic_state_b = torch.tensor(batch["critic_state"], dtype=torch.float32)
                joint_action_b = torch.tensor(batch["joint_action"], dtype=torch.float32)
                next_critic_state_b = torch.tensor(batch["next_critic_state"], dtype=torch.float32)
                next_dso_obs_b = torch.tensor(batch["next_dso_obs"], dtype=torch.float32)
                next_vpp_obs_b = torch.tensor(batch["next_vpp_obs"], dtype=torch.float32)
                reward_vector_b = torch.tensor(batch["reward_vector"], dtype=torch.float32)
                shield_gap_b = torch.tensor(batch["shield_intervention_gap_mw"], dtype=torch.float32)
                shield_penalty_b = torch.tensor(batch["shield_intervention_penalty"], dtype=torch.float32)
                done_b = torch.tensor(batch["done"], dtype=torch.float32)

                with torch.no_grad():
                    next_dso_mean, next_dso_log_std = actor_modules["dso_actor"](next_dso_obs_b)
                    next_dso_action_b, next_dso_log_prob_b, _ = _sample_squashed_gaussian(
                        next_dso_mean,
                        next_dso_log_std,
                        action_clip=cfg.action_clip,
                        torch=torch,
                    )
                    batch_size, n_vpps, _ = next_vpp_obs_b.shape
                    next_aggregate_action_b, next_der_action_b, next_dispatch_log_prob_b = sample_dispatch_batch(next_vpp_obs_b)
                    next_joint_action_b = torch.cat(
                        [next_dso_action_b, next_aggregate_action_b, next_der_action_b.reshape(batch_size, -1)],
                        dim=-1,
                    )
                    target_q1, target_q2 = target_role_critic(next_critic_state_b, next_joint_action_b)
                    min_target_q = torch.min(target_q1, target_q2)
                    alpha_dso = log_alpha_dso.exp().detach()
                    alpha_dispatch = log_alpha_dispatch.exp().detach()
                    entropy_term = torch.cat(
                        [
                            (alpha_dso * next_dso_log_prob_b).unsqueeze(-1),
                            alpha_dispatch * next_dispatch_log_prob_b,
                        ],
                        dim=-1,
                    )
                    scaled_reward_vector_b = reward_vector_b * float(cfg.reward_scale)
                    target_q = scaled_reward_vector_b + (1.0 - done_b.unsqueeze(-1)) * float(cfg.gamma) * (
                        min_target_q - entropy_term
                    )
                    if cfg.target_q_clip is not None:
                        target_q = torch.clamp(target_q, -float(cfg.target_q_clip), float(cfg.target_q_clip))

                current_q1, current_q2 = role_critic(critic_state_b, joint_action_b)
                critic_loss = ((current_q1 - target_q) ** 2).mean() + ((current_q2 - target_q) ** 2).mean()
                critic_optimizer.zero_grad()
                critic_loss.backward()
                critic_grad_norm = float(torch.nn.utils.clip_grad_norm_(role_critic.parameters(), cfg.critic_grad_clip))
                critic_optimizer.step()
                critic_updates += 1

                current_dso_obs_b = torch.tensor(batch["dso_obs"], dtype=torch.float32)
                current_vpp_obs_b = torch.tensor(batch["vpp_obs"], dtype=torch.float32)
                current_dso_mean, current_dso_log_std = actor_modules["dso_actor"](current_dso_obs_b)
                current_dso_action_b, current_dso_log_prob_b, _ = _sample_squashed_gaussian(
                    current_dso_mean,
                    current_dso_log_std,
                    action_clip=cfg.action_clip,
                    torch=torch,
                )
                batch_size, n_vpps, _ = current_vpp_obs_b.shape
                current_aggregate_action_b, current_der_action_b, current_dispatch_log_prob_b = sample_dispatch_batch(current_vpp_obs_b)
                current_joint_action_b = torch.cat(
                    [current_dso_action_b, current_aggregate_action_b, current_der_action_b.reshape(batch_size, -1)],
                    dim=-1,
                )
                q1_heads = role_critic.q1_value(critic_state_b, current_joint_action_b)
                alpha_dso = log_alpha_dso.exp()
                alpha_dispatch = log_alpha_dispatch.exp()
                dso_joint_action_b = torch.cat(
                    [
                        current_dso_action_b,
                        current_aggregate_action_b.detach(),
                        current_der_action_b.detach().reshape(batch_size, -1),
                    ],
                    dim=-1,
                )
                dso_q_heads = role_critic.q1_value(critic_state_b, dso_joint_action_b)
                dso_actor_loss = -(dso_q_heads[:, 0] - alpha_dso * current_dso_log_prob_b).mean()
                dso_actor_optimizer.zero_grad()
                dso_actor_loss.backward()
                dso_actor_grad_norm = float(
                    torch.nn.utils.clip_grad_norm_(actor_modules["dso_actor"].parameters(), cfg.actor_grad_clip)
                )
                dso_actor_optimizer.step()

                current_dso_mean_2, current_dso_log_std_2 = actor_modules["dso_actor"](current_dso_obs_b)
                current_dso_action_b_2, _, _ = _sample_squashed_gaussian(
                    current_dso_mean_2,
                    current_dso_log_std_2,
                    action_clip=cfg.action_clip,
                    torch=torch,
                )
                current_aggregate_action_b_2, current_der_action_b_2, current_dispatch_log_prob_b_2 = sample_dispatch_batch(
                    current_vpp_obs_b
                )
                dispatch_joint_action_b = torch.cat(
                    [
                        current_dso_action_b_2.detach(),
                        current_aggregate_action_b_2,
                        current_der_action_b_2.reshape(batch_size, -1),
                    ],
                    dim=-1,
                )
                dispatch_q_heads = role_critic.q1_value(critic_state_b, dispatch_joint_action_b)
                dispatch_actor_loss = -(
                    dispatch_q_heads[:, 1:].mean(dim=-1)
                    - alpha_dispatch * current_dispatch_log_prob_b_2.mean(dim=-1)
                ).mean()
                dispatch_actor_optimizer.zero_grad()
                dispatch_actor_loss.backward()
                dispatch_actor_grad_norm = float(
                    torch.nn.utils.clip_grad_norm_(dispatch_actor_params, cfg.actor_grad_clip)
                )
                dispatch_actor_optimizer.step()
                actor_loss = dso_actor_loss + dispatch_actor_loss
                actor_grad_norm = max(dso_actor_grad_norm, dispatch_actor_grad_norm)
                actor_updates += 1

                alpha_loss = -(
                    log_alpha_dso * (current_dso_log_prob_b.detach() + target_entropy_dso).mean()
                    + log_alpha_dispatch * (current_dispatch_log_prob_b.detach().mean(dim=-1) + target_entropy_dispatch).mean()
                )
                alpha_optimizer.zero_grad()
                alpha_loss.backward()
                alpha_grad_norm = float(torch.nn.utils.clip_grad_norm_([log_alpha_dso, log_alpha_dispatch], cfg.alpha_grad_clip))
                alpha_optimizer.step()
                with torch.no_grad():
                    log_alpha_dso.clamp_(float(cfg.log_alpha_min), float(cfg.log_alpha_max))
                    log_alpha_dispatch.clamp_(float(cfg.log_alpha_min), float(cfg.log_alpha_max))
                alpha_updates += 1
                _soft_update(role_critic, target_role_critic, cfg.tau)

                update_rows.append(
                    {
                        "global_step": int(total_env_steps),
                        "critic_update": int(critic_updates),
                        "actor_update": int(actor_updates),
                        "alpha_update": int(alpha_updates),
                        "critic_loss": float(critic_loss.detach().cpu().item()),
                        "actor_loss": float(actor_loss.detach().cpu().item()),
                        "dso_actor_loss": float(dso_actor_loss.detach().cpu().item()),
                        "dispatch_actor_loss": float(dispatch_actor_loss.detach().cpu().item()),
                        "alpha_loss": float(alpha_loss.detach().cpu().item()),
                        "alpha_dso": float(log_alpha_dso.exp().detach().cpu().item()),
                        "alpha_dispatch": float(log_alpha_dispatch.exp().detach().cpu().item()),
                        "critic_grad_norm": critic_grad_norm,
                        "actor_grad_norm": actor_grad_norm,
                        "dso_actor_grad_norm": dso_actor_grad_norm,
                        "dispatch_actor_grad_norm": dispatch_actor_grad_norm,
                        "alpha_grad_norm": alpha_grad_norm,
                        "reward_scale": float(cfg.reward_scale),
                        "target_q_abs_max": float(target_q.detach().abs().max().cpu().item()),
                        "shield_intervention_gap_mw_mean": float(shield_gap_b.mean().detach().cpu().item()),
                        "shield_intervention_penalty_mean": float(shield_penalty_b.mean().detach().cpu().item()),
                    }
                )

            observations = next_observations
            dso_obs_vec = next_dso_obs_vec
            vpp_obs_mat = next_vpp_obs_mat
            critic_vec = next_critic_vec
            if done:
                break

        episode_rows.append(
            {
                "episode": int(episode),
                "algorithm": cfg.algorithm,
                "episode_reward": float(episode_reward),
                "episode_cost": float(total_cost),
                "violation_count": int(violation_count),
                "projection_gap_mw": float(projection_gap_total),
                "local_bounds_projection_gap_mw": float(local_projection_gap_total),
                "ac_aware_projection_gap_mw": float(ac_projection_gap_total),
                "shield_intervention_gap_mw": float(projection_gap_total),
                "shield_intervention_penalty": float(shield_intervention_penalty_total),
                "shield_intervention_count": int(shield_intervention_count),
                "replay_size": int(len(replay)),
                "critic_updates": int(critic_updates),
                "actor_updates": int(actor_updates),
                "alpha_updates": int(alpha_updates),
            }
        )
        if float(episode_reward) > best_episode_reward:
            best_episode_reward = float(episode_reward)
            best_episode_index = int(episode)
            best_checkpoint_state = {
                "actor_state_dict": copy.deepcopy(actor_modules.state_dict()),
                "role_critic_state_dict": copy.deepcopy(role_critic.state_dict()),
                "target_role_critic_state_dict": copy.deepcopy(target_role_critic.state_dict()),
                "log_alpha_dso": float(log_alpha_dso.detach().cpu().item()),
                "log_alpha_dispatch": float(log_alpha_dispatch.detach().cpu().item()),
            }
        _set_episode_postfix(
            episode_iter,
            has_tqdm_progress,
            reward=f"{episode_reward:.2f}",
            cost=f"{total_cost:.1f}",
            viol=int(violation_count),
            gap=f"{projection_gap_total:.4f}",
            replay=int(len(replay)),
            q=int(critic_updates),
            pi=int(actor_updates),
        )
        now = time.monotonic()
        if (
            not has_tqdm_progress
            and (episode == 0 or episode == cfg.episodes - 1 or now - last_progress_print >= progress_interval_seconds)
        ):
            last_progress_print = now
            print(
                "[HASAC] "
                f"episode={episode + 1}/{cfg.episodes} "
                f"reward={episode_reward:.4f} "
                f"cost={total_cost:.4f} "
                f"violations={violation_count} "
                f"projection_gap_mw={projection_gap_total:.6f} "
                f"replay={len(replay)} "
                f"critic_updates={critic_updates} "
                f"actor_updates={actor_updates} "
                f"alpha_updates={alpha_updates}",
                flush=True,
            )
        env.close()

    episode_metrics = pd.DataFrame(episode_rows)
    step_metrics = pd.DataFrame(step_rows)
    update_metrics = pd.DataFrame(update_rows)
    checkpoint_path = out / "hasac_checkpoint.pt"
    best_checkpoint_path = out / "hasac_best_checkpoint.pt"

    def checkpoint_payload(
        actor_state: Any,
        critic_state: Any,
        target_critic_state: Any,
        alpha_dso_value: float,
        alpha_dispatch_value: float,
    ) -> dict[str, Any]:
        return {
            "config": cfg.to_dict(),
            "actor_state_dict": actor_state,
            "role_critic_state_dict": critic_state,
            "target_role_critic_state_dict": target_critic_state,
            "log_alpha_dso": float(alpha_dso_value),
            "log_alpha_dispatch": float(alpha_dispatch_value),
            "target_entropy_dso": float(target_entropy_dso),
            "target_entropy_dispatch": float(target_entropy_dispatch),
            "target_entropy_multiplier": float(cfg.target_entropy_multiplier),
            "log_alpha_min": float(cfg.log_alpha_min),
            "log_alpha_max": float(cfg.log_alpha_max),
            "critic_head_names": critic_head_names,
            "critic_spec": critic_spec.to_dict(),
            "dso_input_dim": dso_input_dim,
            "vpp_input_dim": vpp_input_dim,
            "critic_input_dim": critic_input_dim,
            "joint_action_dim": joint_action_dim,
            "max_der_per_vpp": max_der_per_vpp,
            "vpp_ids": vpp_ids,
            "der_ids_by_vpp": der_ids_by_vpp,
            "policy_signature": policy_signature,
            "architecture_meta": architecture_meta,
            "selection_metric": "episode_reward",
        }

    torch.save(
        checkpoint_payload(
            actor_modules.state_dict(),
            role_critic.state_dict(),
            target_role_critic.state_dict(),
            float(log_alpha_dso.detach().cpu().item()),
            float(log_alpha_dispatch.detach().cpu().item()),
        ),
        checkpoint_path,
    )
    if best_checkpoint_state is None:
        best_checkpoint_state = {
            "actor_state_dict": copy.deepcopy(actor_modules.state_dict()),
            "role_critic_state_dict": copy.deepcopy(role_critic.state_dict()),
            "target_role_critic_state_dict": copy.deepcopy(target_role_critic.state_dict()),
            "log_alpha_dso": float(log_alpha_dso.detach().cpu().item()),
            "log_alpha_dispatch": float(log_alpha_dispatch.detach().cpu().item()),
        }
        best_episode_index = int(episode_metrics["episode"].iloc[-1]) if not episode_metrics.empty else -1
    torch.save(
        checkpoint_payload(
            best_checkpoint_state["actor_state_dict"],
            best_checkpoint_state["role_critic_state_dict"],
            best_checkpoint_state["target_role_critic_state_dict"],
            best_checkpoint_state["log_alpha_dso"],
            best_checkpoint_state["log_alpha_dispatch"],
        ),
        best_checkpoint_path,
    )
    summary = {
        "algorithm": cfg.algorithm,
        "status": "completed",
        "is_deep_rl": True,
        "deep_learning_framework": "torch",
        "training_pattern": "off_policy_ctde",
        "soft_actor_critic": True,
        "twin_soft_q": True,
        "off_policy_replay": True,
        "automatic_entropy_tuning": True,
        "reward_scale": float(cfg.reward_scale),
        "target_q_clip": None if cfg.target_q_clip is None else float(cfg.target_q_clip),
        "critic_grad_clip": float(cfg.critic_grad_clip),
        "actor_grad_clip": float(cfg.actor_grad_clip),
        "alpha_grad_clip": float(cfg.alpha_grad_clip),
        "dso_dispatch_actor_objectives_separated": True,
        "valid_der_action_log_prob_mask": True,
        "shield_intervention_penalty_in_role_rewards": True,
        "dso_shield_intervention_penalty_coef": float(cfg.dso_shield_intervention_penalty_coef),
        "dispatch_shield_intervention_penalty_coef": float(cfg.dispatch_shield_intervention_penalty_coef),
        "total_shield_intervention_gap_mw": float(episode_metrics["shield_intervention_gap_mw"].sum())
        if not episode_metrics.empty and "shield_intervention_gap_mw" in episode_metrics
        else 0.0,
        "total_shield_intervention_penalty": float(episode_metrics["shield_intervention_penalty"].sum())
        if not episode_metrics.empty and "shield_intervention_penalty" in episode_metrics
        else 0.0,
        "shield_intervention_step_count": int(episode_metrics["shield_intervention_count"].sum())
        if not episode_metrics.empty and "shield_intervention_count" in episode_metrics
        else 0,
        "per_vpp_dispatch_actors": not bool(cfg.share_vpp_dispatch_parameters),
        "shared_dispatch_parameters": bool(cfg.share_vpp_dispatch_parameters),
        "per_vpp_dispatch_q_heads": True,
        "portfolio_scope": "held_keep_discrete_slow_loop_not_hasac",
        "episodes": int(cfg.episodes),
        "horizon_steps": int(cfg.horizon_steps),
        "total_env_steps": int(total_env_steps),
        "critic_updates": int(critic_updates),
        "actor_updates": int(actor_updates),
        "alpha_updates": int(alpha_updates),
        "final_replay_size": int(len(replay)),
        "alpha_dso": float(log_alpha_dso.exp().detach().cpu().item()),
        "alpha_dispatch": float(log_alpha_dispatch.exp().detach().cpu().item()),
        "target_entropy_dso": float(target_entropy_dso),
        "target_entropy_dispatch": float(target_entropy_dispatch),
        "target_entropy_multiplier": float(cfg.target_entropy_multiplier),
        "log_alpha_min": float(cfg.log_alpha_min),
        "log_alpha_max": float(cfg.log_alpha_max),
        "critic_head_names": ",".join(critic_head_names),
        "target_network_tau": float(cfg.tau),
        "best_episode_reward": float(episode_metrics["episode_reward"].max()) if not episode_metrics.empty else None,
        "final_episode_reward": float(episode_metrics["episode_reward"].iloc[-1]) if not episode_metrics.empty else None,
        "final_checkpoint": str(checkpoint_path),
        "best_checkpoint": str(best_checkpoint_path),
        "best_checkpoint_episode": int(best_episode_index),
        "selected_checkpoint_policy": "train_best_episode_reward",
        "checkpoint": str(best_checkpoint_path),
        "claim_boundary": (
            "Runnable HASAC research scaffold: stochastic continuous actors, centralized multi-head twin soft Q "
            "critics, off-policy replay, entropy temperature tuning and soft target backup are present. The slow "
            "discrete portfolio action remains outside this continuous-control scope."
        ),
    }
    episode_metrics.to_csv(out / "hasac_episode_metrics.csv", index=False)
    step_metrics.to_csv(out / "hasac_step_metrics.csv", index=False)
    update_metrics.to_csv(out / "hasac_update_metrics.csv", index=False)
    pd.DataFrame([summary]).to_csv(out / "hasac_training_summary.csv", index=False)
    write_json(out / "hasac_training_summary.json", summary)
    write_json(out / "hasac_config.json", cfg.to_dict())
    return {
        "summary": summary,
        "episode_metrics": episode_metrics,
        "step_metrics": step_metrics,
        "update_metrics": update_metrics,
        "checkpoint": best_checkpoint_path,
        "final_checkpoint": checkpoint_path,
        "best_checkpoint": best_checkpoint_path,
        "output_dir": out,
    }


def evaluate_hasac_checkpoint(
    *,
    config_path: str | Path | None,
    checkpoint_path: str | Path,
    output_dir: str | Path,
    horizon_steps: int | None = None,
    seed: int = 42,
) -> dict[str, Any]:
    from vpp_dso_sim.envs.multi_agent_env import MultiAgentVPPDSOEnv
    from vpp_dso_sim.learning.deep_rl import _build_privacy_separated_networks
    from vpp_dso_sim.learning.reward_contracts import shield_intervention_metrics
    from vpp_dso_sim.utils.io import ensure_dir, write_json

    torch, _, _, _ = _require_torch()
    checkpoint = torch.load(Path(checkpoint_path), map_location="cpu")
    cfg = checkpoint.get("config", {})
    out = ensure_dir(output_dir)
    eval_horizon = int(horizon_steps or cfg.get("horizon_steps", 8))
    vpp_ids = list(checkpoint["vpp_ids"])
    max_der_per_vpp = int(checkpoint["max_der_per_vpp"])
    der_ids_by_vpp = dict(checkpoint["der_ids_by_vpp"])
    modules, _ = _build_privacy_separated_networks(
        dso_input_dim=int(checkpoint["dso_input_dim"]),
        vpp_input_dim=int(checkpoint["vpp_input_dim"]),
        portfolio_input_dim=9,
        critic_input_dim=int(checkpoint["critic_input_dim"]),
        critic_action_dim=1,
        action_dim=len(vpp_ids),
        der_action_dim=max_der_per_vpp,
        hidden_dim=int(cfg.get("hidden_dim", 64)),
    )
    actor_modules = torch.nn.ModuleDict({"dso_actor": modules["dso_actor"]})
    if bool(cfg.get("share_vpp_dispatch_parameters", False)):
        actor_modules["vpp_dispatch_actor"] = modules["vpp_dispatch_actor"]
    else:
        for vpp_id in vpp_ids:
            actor_modules[f"{vpp_id}_dispatch_actor"] = copy.deepcopy(modules["vpp_dispatch_actor"])
    actor_modules.load_state_dict(checkpoint["actor_state_dict"])
    actor_modules.eval()

    def dispatch_actor_for(vpp_id: str) -> Any:
        return (
            actor_modules["vpp_dispatch_actor"]
            if bool(cfg.get("share_vpp_dispatch_parameters", False))
            else actor_modules[f"{vpp_id}_dispatch_actor"]
        )

    def deterministic_dispatch_batch(vpp_obs_tensor: Any) -> tuple[Any, Any]:
        aggregate_actions: list[Any] = []
        der_actions: list[Any] = []
        for vpp_index, vpp_id in enumerate(vpp_ids):
            aggregate_mean, _, der_mean, _ = dispatch_actor_for(vpp_id)(vpp_obs_tensor[:, vpp_index, :])
            aggregate_actions.append(
                torch.clamp(
                    torch.tanh(aggregate_mean),
                    -float(cfg.get("action_clip", 1.0)),
                    float(cfg.get("action_clip", 1.0)),
                )
            )
            der_actions.append(
                torch.clamp(
                    torch.tanh(der_mean),
                    -float(cfg.get("action_clip", 1.0)),
                    float(cfg.get("action_clip", 1.0)),
                ).unsqueeze(1)
            )
        return torch.cat(aggregate_actions, dim=1), torch.cat(der_actions, dim=1)

    env = MultiAgentVPPDSOEnv(config_path=config_path, horizon_steps=eval_horizon)
    observations, _ = env.reset(seed=seed)
    step_rows: list[dict[str, Any]] = []
    total_reward = 0.0
    total_cost = 0.0
    total_violations = 0

    with torch.no_grad():
        for step in range(eval_horizon):
            dso_obs_vec, vpp_obs_mat, _ = _continuous_step_observations(
                observations,
                env=env,
                vpp_ids=vpp_ids,
                max_der_per_vpp=max_der_per_vpp,
            )
            dso_obs_tensor = torch.tensor(dso_obs_vec, dtype=torch.float32).unsqueeze(0)
            vpp_obs_tensor = torch.tensor(vpp_obs_mat, dtype=torch.float32).unsqueeze(0)
            dso_mean, _ = actor_modules["dso_actor"](dso_obs_tensor)
            dso_action_t = torch.clamp(
                torch.tanh(dso_mean),
                -float(cfg.get("action_clip", 1.0)),
                float(cfg.get("action_clip", 1.0)),
            )
            aggregate_action_t, der_action_t = deterministic_dispatch_batch(vpp_obs_tensor)
            action_payload = _continuous_payload_from_actions(
                dso_action=dso_action_t.cpu().numpy().reshape(-1),
                aggregate_actions=aggregate_action_t.cpu().numpy().reshape(len(vpp_ids)),
                der_actions=der_action_t.cpu().numpy().reshape(len(vpp_ids), max_der_per_vpp),
                dso_obs=observations["dso_global_guidance"],
                vpp_observations=observations,
                vpp_ids=vpp_ids,
                der_ids_by_vpp=der_ids_by_vpp,
                action_clip=float(cfg.get("action_clip", 1.0)),
                policy_version="frozen_hasac",
            )
            observations, reward_map, _, truncations, infos = env.step(action_payload)
            reward_components = infos["dso_global_guidance"].get("reward_components", {})
            shield_metrics = shield_intervention_metrics(reward_components)
            shield_penalty = float(shield_metrics["shield_intervention_penalty"])
            raw_dso_reward = float(reward_map["dso_global_guidance"])
            raw_dispatch_reward = float(
                np.mean([float(reward_map[f"{vpp_id}_dispatch"]) for vpp_id in vpp_ids]) if vpp_ids else 0.0
            )
            dso_reward = raw_dso_reward - float(cfg.get("dso_shield_intervention_penalty_coef", 1.0)) * shield_penalty
            dispatch_reward = raw_dispatch_reward - float(cfg.get("dispatch_shield_intervention_penalty_coef", 1.0)) * shield_penalty
            violations = infos["dso_global_guidance"].get("violations", [])
            reward = dso_reward + dispatch_reward
            total_reward += reward
            total_cost += float(reward_components.get("total_cost", -dso_reward))
            total_violations += len(violations)
            step_rows.append(
                {
                    "step": int(step),
                    "algorithm": "hasac_continuous_dispatch",
                    "evaluation_mode": "frozen_mean_actor",
                    "reward": float(reward),
                    "dso_reward": float(dso_reward),
                    "vpp_dispatch_reward": float(dispatch_reward),
                    "raw_dso_reward_before_shield_penalty": float(raw_dso_reward),
                    "raw_dispatch_reward_before_shield_penalty": float(raw_dispatch_reward),
                    "raw_objective_reward": float(reward_components.get("raw_objective_reward", -float(reward_components.get("total_cost", 0.0)))),
                    "total_cost": float(reward_components.get("total_cost", -dso_reward)),
                    "violation_count": int(len(violations)),
                    "post_ac_violation_count": float(reward_components.get("post_ac_violation_count", len(violations))),
                    "post_ac_voltage_violation_count": float(reward_components.get("post_ac_voltage_violation_count", 0.0)),
                    "post_ac_line_overload_count": float(reward_components.get("post_ac_line_overload_count", 0.0)),
                    "post_ac_trafo_overload_count": float(reward_components.get("post_ac_trafo_overload_count", 0.0)),
                    "post_ac_powerflow_failed": float(reward_components.get("post_ac_powerflow_failed", 0.0)),
                    "post_ac_violation_magnitude": float(reward_components.get("post_ac_violation_magnitude", 0.0)),
                    "shield_intervention_penalty": shield_penalty,
                }
            )
            if all(truncations.values()):
                break

    env.simulator.export_results(out / "simulator_results")
    step_metrics = pd.DataFrame(step_rows)
    step_metrics.to_csv(out / "hasac_frozen_eval_step_metrics.csv", index=False)
    summary = {
        "algorithm": "hasac_continuous_dispatch",
        "evaluation_mode": "frozen_mean_actor",
        "checkpoint": str(checkpoint_path),
        "horizon_steps": int(eval_horizon),
        "seed": int(seed),
        "total_reward": float(total_reward),
        "total_cost": float(total_cost),
        "total_violation_count": int(total_violations),
    }
    write_json(out / "hasac_frozen_eval_summary.json", summary)
    env.close()
    return {"summary": summary, "step_metrics": step_metrics, "output_dir": out}


def evaluate_happo_checkpoint(
    *,
    config_path: str | Path | None,
    checkpoint_path: str | Path,
    output_dir: str | Path,
    horizon_steps: int | None = None,
    seed: int = 42,
) -> dict[str, Any]:
    """Evaluate a saved HAPPO actor set without exploration.

    The evaluator keeps the same privacy boundary as training: DSO, each VPP
    dispatch actor, and each slow portfolio actor are reconstructed from the
    checkpoint. Continuous actors use their Gaussian means; portfolio actors
    use argmax only on configured slow-loop decision steps.
    """

    from vpp_dso_sim.envs.multi_agent_env import MultiAgentVPPDSOEnv
    from vpp_dso_sim.learning.deep_rl import (
        _build_privacy_separated_networks,
        encode_vpp_portfolio_observation,
    )
    from vpp_dso_sim.learning.reward_contracts import shield_intervention_metrics
    from vpp_dso_sim.utils.io import ensure_dir, write_json

    torch, _, _, _ = _require_torch()
    checkpoint = torch.load(Path(checkpoint_path), map_location="cpu")
    cfg = checkpoint.get("config", {})
    out = ensure_dir(output_dir)
    eval_horizon = int(horizon_steps or cfg.get("horizon_steps", 8))
    vpp_ids = list(checkpoint["vpp_ids"])
    max_der_per_vpp = int(checkpoint["max_der_per_vpp"])
    der_ids_by_vpp = dict(checkpoint["der_ids_by_vpp"])

    modules, _ = _build_privacy_separated_networks(
        dso_input_dim=int(checkpoint["dso_input_dim"]),
        vpp_input_dim=int(checkpoint["vpp_input_dim"]),
        portfolio_input_dim=int(checkpoint.get("portfolio_input_dim", 9)),
        critic_input_dim=int(checkpoint["critic_input_dim"]),
        critic_action_dim=1,
        action_dim=len(vpp_ids),
        der_action_dim=max_der_per_vpp,
        hidden_dim=int(cfg.get("hidden_dim", 64)),
    )
    actor_modules = torch.nn.ModuleDict({"dso_actor": modules["dso_actor"]})
    if bool(cfg.get("share_vpp_dispatch_parameters", False)):
        actor_modules["vpp_dispatch_actor"] = modules["vpp_dispatch_actor"]
    else:
        for vpp_id in vpp_ids:
            actor_modules[f"{vpp_id}_dispatch_actor"] = copy.deepcopy(modules["vpp_dispatch_actor"])
    if bool(cfg.get("share_vpp_portfolio_parameters", False)):
        actor_modules["vpp_portfolio_actor"] = modules["vpp_portfolio_actor"]
    else:
        for vpp_id in vpp_ids:
            actor_modules[f"{vpp_id}_portfolio_actor"] = copy.deepcopy(modules["vpp_portfolio_actor"])
    actor_modules.load_state_dict(checkpoint["actor_state_dict"])
    actor_modules.eval()

    def dispatch_actor_for(vpp_id: str) -> Any:
        return (
            actor_modules["vpp_dispatch_actor"]
            if bool(cfg.get("share_vpp_dispatch_parameters", False))
            else actor_modules[f"{vpp_id}_dispatch_actor"]
        )

    def portfolio_actor_for(vpp_id: str) -> Any:
        return (
            actor_modules["vpp_portfolio_actor"]
            if bool(cfg.get("share_vpp_portfolio_parameters", False))
            else actor_modules[f"{vpp_id}_portfolio_actor"]
        )

    def portfolio_decision_step(step: int) -> bool:
        interval = max(1, int(cfg.get("portfolio_decision_interval_steps", 24)))
        return int(step) % interval == 0

    def deterministic_dispatch_batch(vpp_obs_tensor: Any) -> tuple[Any, Any]:
        aggregate_actions: list[Any] = []
        der_actions: list[Any] = []
        action_clip = float(cfg.get("action_clip", 1.0))
        for vpp_index, vpp_id in enumerate(vpp_ids):
            aggregate_mean, _, der_mean, _ = dispatch_actor_for(vpp_id)(vpp_obs_tensor[:, vpp_index, :])
            aggregate_actions.append(torch.clamp(aggregate_mean, -action_clip, action_clip))
            der_actions.append(torch.clamp(der_mean, -action_clip, action_clip).unsqueeze(1))
        return torch.cat(aggregate_actions, dim=1), torch.cat(der_actions, dim=1)

    env = MultiAgentVPPDSOEnv(config_path=config_path, horizon_steps=eval_horizon)
    observations, _ = env.reset(seed=seed)
    step_rows: list[dict[str, Any]] = []
    total_reward = 0.0
    total_cost = 0.0
    total_violations = 0
    portfolio_action_counts = {"keep": 0, "reweight": 0, "propose_membership_change": 0}

    with torch.no_grad():
        for step in range(eval_horizon):
            dso_obs_vec, vpp_obs_mat, _ = _continuous_step_observations(
                observations,
                env=env,
                vpp_ids=vpp_ids,
                max_der_per_vpp=max_der_per_vpp,
            )
            dso_obs_tensor = torch.tensor(dso_obs_vec, dtype=torch.float32).unsqueeze(0)
            vpp_obs_tensor = torch.tensor(vpp_obs_mat, dtype=torch.float32).unsqueeze(0)
            dso_mean, _ = actor_modules["dso_actor"](dso_obs_tensor)
            action_clip = float(cfg.get("action_clip", 1.0))
            dso_action_t = torch.clamp(dso_mean, -action_clip, action_clip)
            aggregate_action_t, der_action_t = deterministic_dispatch_batch(vpp_obs_tensor)
            action_payload = _continuous_payload_from_actions(
                dso_action=dso_action_t.cpu().numpy().reshape(-1),
                aggregate_actions=aggregate_action_t.cpu().numpy().reshape(len(vpp_ids)),
                der_actions=der_action_t.cpu().numpy().reshape(len(vpp_ids), max_der_per_vpp),
                dso_obs=observations["dso_global_guidance"],
                vpp_observations=observations,
                vpp_ids=vpp_ids,
                der_ids_by_vpp=der_ids_by_vpp,
                action_clip=action_clip,
                policy_version="frozen_happo",
            )
            decision_step = portfolio_decision_step(step)
            for vpp_id in vpp_ids:
                portfolio_action = "keep"
                if decision_step:
                    portfolio_vec = encode_vpp_portfolio_observation(observations[f"{vpp_id}_portfolio"])
                    logits = portfolio_actor_for(vpp_id)(torch.tensor(portfolio_vec, dtype=torch.float32).unsqueeze(0))
                    portfolio_action = ("keep", "reweight", "propose_membership_change")[
                        int(torch.argmax(logits.squeeze(0)).item())
                    ]
                portfolio_action_counts[portfolio_action] += 1
                action_payload[f"{vpp_id}_portfolio"] = {
                    "action": portfolio_action,
                    "policy_version": "frozen_happo" if decision_step else "frozen_happo_slow_loop_hold",
                }

            observations, reward_map, _, truncations, infos = env.step(action_payload)
            raw_dso_reward = float(reward_map["dso_global_guidance"])
            raw_dispatch_reward = float(
                np.mean([float(reward_map[f"{vpp_id}_dispatch"]) for vpp_id in vpp_ids]) if vpp_ids else 0.0
            )
            raw_portfolio_reward = float(
                np.mean([float(reward_map[f"{vpp_id}_portfolio"]) for vpp_id in vpp_ids]) if vpp_ids else 0.0
            )
            reward_components = infos["dso_global_guidance"].get("reward_components", {})
            shield_metrics = shield_intervention_metrics(reward_components)
            shield_penalty = float(shield_metrics["shield_intervention_penalty"])
            dso_reward = raw_dso_reward - float(cfg.get("dso_shield_intervention_penalty_coef", 1.0)) * shield_penalty
            dispatch_reward = raw_dispatch_reward - float(cfg.get("dispatch_shield_intervention_penalty_coef", 1.0)) * shield_penalty
            portfolio_reward = raw_portfolio_reward if decision_step else 0.0
            violations = infos["dso_global_guidance"].get("violations", [])
            reward = dso_reward + dispatch_reward + portfolio_reward
            total_reward += reward
            total_cost += float(reward_components.get("total_cost", -dso_reward))
            total_violations += len(violations)
            step_rows.append(
                {
                    "step": int(step),
                    "algorithm": "happo_sequential_ctde",
                    "evaluation_mode": "frozen_mean_argmax_actor",
                    "reward": float(reward),
                    "dso_reward": float(dso_reward),
                    "vpp_dispatch_reward": float(dispatch_reward),
                    "vpp_portfolio_reward": float(portfolio_reward),
                    "raw_dso_reward_before_shield_penalty": float(raw_dso_reward),
                    "raw_dispatch_reward_before_shield_penalty": float(raw_dispatch_reward),
                    "raw_portfolio_reward_before_decision_mask": float(raw_portfolio_reward),
                    "raw_objective_reward": float(reward_components.get("raw_objective_reward", -float(reward_components.get("total_cost", 0.0)))),
                    "total_cost": float(reward_components.get("total_cost", -dso_reward)),
                    "violation_count": int(len(violations)),
                    "post_ac_violation_count": float(reward_components.get("post_ac_violation_count", len(violations))),
                    "post_ac_voltage_violation_count": float(reward_components.get("post_ac_voltage_violation_count", 0.0)),
                    "post_ac_line_overload_count": float(reward_components.get("post_ac_line_overload_count", 0.0)),
                    "post_ac_trafo_overload_count": float(reward_components.get("post_ac_trafo_overload_count", 0.0)),
                    "post_ac_powerflow_failed": float(reward_components.get("post_ac_powerflow_failed", 0.0)),
                    "post_ac_violation_magnitude": float(reward_components.get("post_ac_violation_magnitude", 0.0)),
                    "shield_intervention_penalty": shield_penalty,
                    "portfolio_decision_step": bool(decision_step),
                }
            )
            if all(truncations.values()):
                break

    env.simulator.export_results(out / "simulator_results")
    step_metrics = pd.DataFrame(step_rows)
    step_metrics.to_csv(out / "happo_frozen_eval_step_metrics.csv", index=False)
    summary = {
        "algorithm": "happo_sequential_ctde",
        "evaluation_mode": "frozen_mean_argmax_actor",
        "checkpoint": str(checkpoint_path),
        "horizon_steps": int(eval_horizon),
        "seed": int(seed),
        "total_reward": float(total_reward),
        "total_cost": float(total_cost),
        "total_violation_count": int(total_violations),
        "portfolio_action_counts": dict(portfolio_action_counts),
    }
    write_json(out / "happo_frozen_eval_summary.json", summary)
    pd.DataFrame([summary | {"portfolio_action_counts": str(portfolio_action_counts)}]).to_csv(
        out / "happo_frozen_eval_summary.csv",
        index=False,
    )
    env.close()
    return {"summary": summary, "step_metrics": step_metrics, "output_dir": out}


def torch_available() -> bool:
    return TORCH_AVAILABLE


__all__ = [
    "ALGORITHM_REGISTRY",
    "AlgorithmCandidate",
    "AlgorithmScore",
    "HAPPOConfig",
    "HASACConfig",
    "MultiHeadValueCriticSpec",
    "OffPolicyReplayBuffer",
    "ScoringWeights",
    "TwinCriticSpec",
    "build_matd3_twin_critic_spec",
    "build_multi_head_value_critic",
    "build_twin_critic",
    "candidate_by_key",
    "evaluate_happo_checkpoint",
    "evaluate_hasac_checkpoint",
    "get_algorithm_registry",
    "rank_algorithm_candidates",
    "score_algorithm_candidate",
    "torch_available",
    "train_happo",
    "train_hasac",
]
