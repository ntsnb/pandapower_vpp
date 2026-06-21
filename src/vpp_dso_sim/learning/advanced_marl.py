from __future__ import annotations

import copy
from dataclasses import asdict, dataclass, replace
import hashlib
import importlib.util
import os
from pathlib import Path
import sys
import time
from typing import Any, Callable

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


def _resolve_torch_device(torch: Any, requested_device: str | None) -> tuple[Any, dict[str, Any]]:
    """Resolve a trainer device while making CUDA failures explicit."""

    requested = str(requested_device or "auto").strip().lower() or "auto"
    cuda_available = bool(torch.cuda.is_available())
    cuda_device_count = int(torch.cuda.device_count()) if cuda_available else 0
    if requested == "auto":
        resolved = "cuda" if cuda_available else "cpu"
    elif requested == "cpu":
        resolved = "cpu"
    elif requested == "cuda" or requested.startswith("cuda:"):
        if not cuda_available:
            raise RuntimeError(
                "HAPPO device is set to CUDA, but torch.cuda.is_available() is false. "
                "Check that the command is running outside a GPU-isolated sandbox, "
                "that /dev/nvidia* devices are visible, and that the NVIDIA driver matches the installed CUDA build."
            )
        if ":" in requested:
            device_index = int(requested.split(":", 1)[1])
            if device_index < 0 or device_index >= cuda_device_count:
                raise ValueError(
                    f"HAPPO requested {requested!r}, but PyTorch sees {cuda_device_count} CUDA device(s)."
                )
        resolved = requested
    else:
        raise ValueError("HAPPO device must be 'auto', 'cpu', 'cuda', or 'cuda:<index>'.")

    cuda_device_name = None
    if resolved.startswith("cuda"):
        cuda_device_name = torch.cuda.get_device_name(torch.device(resolved))
    return (
        torch.device(resolved),
        {
            "requested_device": requested,
            "resolved_device": str(torch.device(resolved)),
            "cuda_available": cuda_available,
            "cuda_device_count": cuda_device_count,
            "cuda_device_name": cuda_device_name,
        },
    )


def _state_dict_to_cpu(state_dict: Any) -> dict[str, Any]:
    """Save checkpoints that can be loaded on CPU-only machines."""

    return {
        key: value.detach().cpu().clone() if hasattr(value, "detach") else copy.deepcopy(value)
        for key, value in state_dict.items()
    }


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
    target_kl: float | None = None
    normalize_observations: bool = False
    normalize_advantages: bool = True
    nan_guard: bool = True
    device: str = "auto"
    dispatch_actor_encoder_type: str = "deepset_v1"
    shared_rollout_enabled: bool = False
    shared_rollout_workers: int = 1
    shared_rollout_backend: str = "serial"
    rollout_fragment_steps: int | None = None
    rollout_policy_version_check: bool = True
    reward_dynamic_reports: bool = True
    reward_dynamic_report_every_episodes: int = 1
    reward_dynamic_report_all_workers: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _config_hash(path: str | Path | None) -> str | None:
    if path is None:
        return None
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()[:12]


def _happo_config_from_yaml(path: str | Path | None) -> HAPPOConfig:
    if path is None:
        return HAPPOConfig()
    from vpp_dso_sim.utils.config import load_yaml

    payload = load_yaml(path)
    simulation = dict(payload.get("simulation", {}))
    trainer = dict(payload.get("trainer", {}))
    updates: dict[str, Any] = {}
    if "horizon_steps" in simulation:
        updates["horizon_steps"] = int(simulation["horizon_steps"])
    if "seed" in simulation:
        updates["seed"] = int(simulation["seed"])
    reward = dict(payload.get("reward", {}))
    if "critic_reward_scale" in reward:
        updates["reward_scale"] = float(reward["critic_reward_scale"])
    direct_fields = set(HAPPOConfig.__dataclass_fields__)
    for key, value in trainer.items():
        if key in {"name"}:
            continue
        if key == "clip_param":
            updates["ppo_clip_ratio"] = float(value)
            continue
        if key == "learning_rate":
            updates["actor_learning_rate"] = float(value)
            updates["critic_learning_rate"] = float(value)
            continue
        if key in direct_fields:
            updates[key] = value
    return HAPPOConfig(**updates)


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
    device: str = "auto"
    dispatch_actor_encoder_type: str = "deepset_v1"

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


def _normalize_observation_array(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32)
    mean = arr.mean(axis=-1, keepdims=True)
    std = arr.std(axis=-1, keepdims=True)
    return ((arr - mean) / (std + 1e-6)).astype(np.float32)


def _append_observation_std(stats: dict[str, list[float]], key: str, values: np.ndarray) -> None:
    arr = np.asarray(values, dtype=np.float32)
    if arr.size:
        stats.setdefault(key, []).append(float(arr.std()))


def _summarize_observation_normalization_stats(stats: dict[str, list[float]]) -> dict[str, float]:
    summary: dict[str, float] = {}
    for key, values in stats.items():
        summary[f"{key}_std_mean"] = float(np.mean(values)) if values else 0.0
    return summary


def _nan_guard_tensors(torch: Any, name: str, *tensors: Any) -> None:
    for tensor in tensors:
        if tensor is None:
            continue
        if not bool(torch.isfinite(tensor).all().detach().cpu().item()):
            raise FloatingPointError(f"NaN/Inf detected in {name}.")


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
    normalize_advantages: bool = True,
) -> tuple[Any, Any]:
    normalized_advantages = (
        _normalize_advantages(advantages, torch) if bool(normalize_advantages) else advantages
    ).detach()
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
    structured_dso_spec: Any | None = None,
    dso_actor_cfg: dict[str, Any] | None = None,
) -> dict[str, Any]:
    from vpp_dso_sim.learning.deep_rl import _target_from_normalized_scalar, _targets_from_normalized_actions
    from vpp_dso_sim.dso.models.structured_happo_actor import normalized_envelope_action_to_payload

    if structured_dso_spec is not None:
        actor_cfg = dict(dso_actor_cfg or {})
        payload: dict[str, Any] = {
            "dso_global_guidance": {
                "envelope_action": normalized_envelope_action_to_payload(
                    dso_action,
                    structured_dso_spec,
                    action_clip=action_clip,
                    min_width_ratio=float(actor_cfg.get("min_width_ratio", 0.10)),
                    max_width_ratio=float(actor_cfg.get("max_width_ratio", 1.00)),
                    direction_logit_scale=float(actor_cfg.get("direction_logit_scale", 5.0)),
                    source=f"{policy_version}_sensitivity_attention_v1_unified_actor",
                )
            }
        }
    else:
        dso_targets = _targets_from_normalized_actions(dso_action, dso_obs, vpp_ids, action_clip)
        payload = {"dso_global_guidance": {"targets": dso_targets}}
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
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
    progress_step_interval: int = 24,
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
    from vpp_dso_sim.dso.models.structured_happo_actor import (
        StructuredDSOGaussianActor,
        normalized_envelope_action_to_payload,
        structured_envelope_action_dim,
    )
    from vpp_dso_sim.dso.observation.happo_structured import build_happo_structured_dso_observation
    from vpp_dso_sim.learning.deep_rl import (
        _gae_returns_advantages,
        _gae_returns_advantages_bootstrap,
        _build_privacy_separated_networks,
        _target_from_normalized_scalar,
        _targets_from_normalized_actions,
        encode_critic_global_state,
        encode_dso_observation,
        encode_joint_action_summary,
        encode_vpp_dispatch_observation,
        encode_vpp_portfolio_observation,
    )
    from vpp_dso_sim.learning.reward_config import write_reward_config_artifacts
    from vpp_dso_sim.learning.reward_contracts import shield_intervention_metrics
    from vpp_dso_sim.learning.reward_trace import dispatch_private_profit_trace_rows
    from vpp_dso_sim.learning.shared_rollout_workers import SharedRolloutWorkerSpec, SubprocessSharedRolloutWorker
    from vpp_dso_sim.utils.io import ensure_dir, write_json

    cfg = config or _happo_config_from_yaml(config_path)
    config_path_text = str(config_path) if config_path is not None else None
    config_hash = _config_hash(config_path) if config_path is not None else None
    shared_rollout_enabled = bool(cfg.shared_rollout_enabled) or int(cfg.shared_rollout_workers) > 1
    shared_rollout_workers = max(1, int(cfg.shared_rollout_workers))
    shared_rollout_backend = str(cfg.shared_rollout_backend)
    supported_shared_rollout_backends = {"serial", "subprocess"}
    if shared_rollout_enabled and shared_rollout_backend not in supported_shared_rollout_backends:
        raise NotImplementedError(
            "Unsupported HAPPO shared rollout backend "
            f"{shared_rollout_backend!r}; expected one of {sorted(supported_shared_rollout_backends)}."
        )
    torch, optim, Normal, Categorical = _require_torch()
    device, device_meta = _resolve_torch_device(torch, cfg.device)
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    out = ensure_dir(output_dir)

    env_probe = MultiAgentVPPDSOEnv(config_path=config_path, horizon_steps=cfg.horizon_steps)
    observations, _ = env_probe.reset(seed=cfg.seed)
    policy_signature = env_probe.policy_compatibility_signature()
    probe_scenario_config = env_probe.scenario.config
    resolved_reward_config = env_probe.scenario.dso.reward_config
    cfg = replace(
        cfg,
        reward_scale=float(resolved_reward_config.critic_reward_scale),
        dso_shield_intervention_penalty_coef=float(resolved_reward_config.shield.dso_penalty_coef),
        dispatch_shield_intervention_penalty_coef=float(resolved_reward_config.shield.dispatch_penalty_coef),
    )
    reward_artifacts = write_reward_config_artifacts(out, resolved_reward_config)
    print(
        "[RewardConfig] "
        f"version={resolved_reward_config.version} "
        f"critic_reward_scale={resolved_reward_config.critic_reward_scale} "
        f"shield_dso_penalty_coef={resolved_reward_config.shield.dso_penalty_coef} "
        f"shield_dispatch_penalty_coef={resolved_reward_config.shield.dispatch_penalty_coef} "
        f"shield_portfolio_future_penalty_coef={resolved_reward_config.shield.portfolio_future_penalty_coef} "
        f"hash={reward_artifacts['reward_config_hash']}",
        flush=True,
    )
    vpp_ids = [vpp.id for vpp in env_probe.scenario.vpps]
    der_ids_by_vpp = {vpp.id: [der.id for der in vpp.der_list] for vpp in env_probe.scenario.vpps}
    max_der_per_vpp = max(1, max((len(ids) for ids in der_ids_by_vpp.values()), default=1))
    scenario_dso_cfg = dict(probe_scenario_config.get("dso", {}))
    dso_actor_cfg = dict(scenario_dso_cfg.get("actor", {}))
    dso_actor_observation_mode = str(scenario_dso_cfg.get("observation_mode", "legacy_flat"))
    use_structured_dso_actor = bool(
        dso_actor_observation_mode == "structured_bipartite"
        or str(scenario_dso_cfg.get("envelope_policy", "")) == "sensitivity_attention_v1"
    )
    structured_dso_spec = None
    if use_structured_dso_actor:
        dso_probe_vec, structured_dso_spec = build_happo_structured_dso_observation(
            env_probe.scenario,
            step=0,
            config=probe_scenario_config,
        )
        dso_input_dim = int(len(dso_probe_vec))
    else:
        dso_input_dim = int(len(encode_dso_observation(observations["dso_global_guidance"], vpp_ids)))
    first_vpp_id = vpp_ids[0]
    vpp_input_dim = int(len(encode_vpp_dispatch_observation(observations[f"{first_vpp_id}_dispatch"], max_der_per_vpp)))
    portfolio_input_dim = int(len(encode_vpp_portfolio_observation(observations[f"{first_vpp_id}_portfolio"])))
    critic_input_dim = int(len(encode_critic_global_state(build_critic_global_state(env_probe.scenario, 0), vpp_ids)))
    dso_action_probe_dim = (
        structured_envelope_action_dim(structured_dso_spec)
        if use_structured_dso_actor and structured_dso_spec is not None
        else len(vpp_ids)
    )
    critic_action_dim = int(len(encode_joint_action_summary(
        normalized_dso_action=np.zeros(dso_action_probe_dim, dtype=np.float32),
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
        dispatch_actor_encoder_type=cfg.dispatch_actor_encoder_type,
    )
    if use_structured_dso_actor:
        if structured_dso_spec is None:
            raise RuntimeError("structured_dso_spec was not initialized for structured DSO actor mode")
        modules["dso_actor"] = StructuredDSOGaussianActor(
            spec=structured_dso_spec,
            d_model=int(dso_actor_cfg.get("d_model", cfg.hidden_dim)),
            num_heads=int(dso_actor_cfg.get("num_heads", 4)),
            num_layers=int(dso_actor_cfg.get("num_layers", 1)),
            action_self_attention_layers=int(dso_actor_cfg.get("action_self_attention_layers", 1)),
            dropout=float(dso_actor_cfg.get("dropout", 0.0)),
            min_width_ratio=float(dso_actor_cfg.get("min_width_ratio", 0.10)),
            max_width_ratio=float(dso_actor_cfg.get("max_width_ratio", 1.00)),
        )
        architecture_meta = {
            **architecture_meta,
            "dso_actor_type": "sensitivity_attention_v1_structured_happo",
            "dso_actor_observation_mode": "structured_bipartite",
            "structured_dso_flat_spec": structured_dso_spec.to_dict(),
        }
    else:
        architecture_meta = {
            **architecture_meta,
            "dso_actor_type": "legacy_mlp_gaussian",
            "dso_actor_observation_mode": "legacy_flat",
        }
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
    actor_modules.to(device)
    value_critic.to(device)
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
    dispatch_profit_trace_rows: list[dict[str, Any]] = []
    update_rows: list[dict[str, Any]] = []
    observation_norm_stats: dict[str, list[float]] = {
        "dso_obs": [],
        "dispatch_obs": [],
        "portfolio_obs": [],
        "critic_state": [],
    }
    total_role_updates = 0
    kl_early_stop_count = 0
    shared_rollout_batches = 0
    shared_rollout_policy_version_mismatch_count = 0
    shared_rollout_total_samples = 0
    shared_rollout_fragment_cut_count = 0
    shared_rollout_bootstrap_values: list[float] = []
    shared_rollout_policy_versions: dict[int, str] = {}
    shared_rollout_worker_counts: dict[int, int] = {}
    shared_rollout_fragment_steps_by_episode: dict[int, int] = {}
    shared_rollout_worker_start_offsets: dict[int, int] = {}
    shared_rollout_worker_terminal_reset_count = 0
    shared_rollout_subprocess_workers: dict[int, SubprocessSharedRolloutWorker] = {}
    shared_rollout_subprocess_worker_pids: dict[int, int] = {}
    shared_rollout_perf_by_episode: dict[int, dict[str, Any]] = {}
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
    step_progress_interval = max(1, int(progress_step_interval))

    def maybe_report_train_step(step_index: int) -> None:
        if progress_callback is None:
            return
        completed = int(step_index) + 1
        if completed < int(cfg.horizon_steps) and completed % step_progress_interval != 0:
            return
        progress_callback(
            {
                "phase": "train_step",
                "message": "HAPPO train step progress",
                "episode": int(episode) + 1,
                "episodes": int(cfg.episodes),
                "step": completed,
                "horizon_steps": int(cfg.horizon_steps),
                "step_progress_pct": float(completed / max(1, int(cfg.horizon_steps))),
                "episode_progress_pct": float((int(episode) + completed / max(1, int(cfg.horizon_steps))) / max(1, int(cfg.episodes))),
                "reward_so_far": float(episode_reward),
                "total_cost_so_far": float(total_cost),
                "violations_so_far": int(violation_count),
                "projection_gap_mw": float(projection_gap_total),
            }
        )

    def maybe_report_shared_rollout_step(
        *,
        episode_index: int,
        worker_index: int,
        worker_count: int,
        worker_start_step: int,
        local_step_index: int,
        env_step: int,
        fragment_steps: int,
        policy_version: str,
        worker_reward: float,
        worker_cost: float,
        worker_violation_count: int,
        worker_projection_gap: float,
    ) -> None:
        if progress_callback is None:
            return
        completed = int(local_step_index) + 1
        if completed < int(fragment_steps) and completed % step_progress_interval != 0:
            return
        total_worker_steps = max(1, int(worker_count) * int(fragment_steps))
        completed_worker_steps = int(worker_index) * int(fragment_steps) + completed
        progress_callback(
            {
                "phase": "shared_rollout_step",
                "message": "HAPPO shared rollout worker progress",
                "episode": int(episode_index) + 1,
                "episodes": int(cfg.episodes),
                "worker_index": int(worker_index),
                "worker_count": int(worker_count),
                "worker_start_step": int(worker_start_step),
                "step": int(env_step) + 1,
                "local_step": completed,
                "fragment_steps": int(fragment_steps),
                "horizon_steps": int(cfg.horizon_steps),
                "step_progress_pct": float(completed / max(1, int(fragment_steps))),
                "episode_progress_pct": float(
                    (int(episode_index) + completed_worker_steps / total_worker_steps)
                    / max(1, int(cfg.episodes))
                ),
                "reward_so_far": float(worker_reward),
                "total_cost_so_far": float(worker_cost),
                "violations_so_far": int(worker_violation_count),
                "projection_gap_mw": float(worker_projection_gap),
                "policy_version": str(policy_version),
            }
        )

    def build_shared_rollout_action_bundle(
        *,
        worker_state: dict[str, Any],
        policy_version: str,
    ) -> dict[str, Any]:
        observations = worker_state["observations"]
        step = int(worker_state.get("current_step", 0))
        dso_obs = observations["dso_global_guidance"]
        if "dso_obs_vec" in worker_state:
            dso_obs_vec = np.asarray(worker_state["dso_obs_vec"], dtype=np.float32)
        elif use_structured_dso_actor:
            dso_obs_vec, _current_structured_spec = build_happo_structured_dso_observation(
                worker_state["env"].scenario,
                step=step,
                config=worker_state["env"].scenario.config,
            )
            if structured_dso_spec is not None and int(len(dso_obs_vec)) != int(structured_dso_spec.flat_dim):
                raise RuntimeError(
                    "Structured DSO observation shape changed during HAPPO shared rollout: "
                    f"got {len(dso_obs_vec)}, expected {structured_dso_spec.flat_dim}."
                )
            dso_obs_vec = dso_obs_vec.astype(np.float32)
        else:
            dso_obs_vec = encode_dso_observation(dso_obs, vpp_ids).astype(np.float32)
        if "critic_state_vec" in worker_state:
            critic_state = np.asarray(worker_state["critic_state_vec"], dtype=np.float32)
        else:
            critic_state = encode_critic_global_state(
                build_critic_global_state(worker_state["env"].scenario, step),
                vpp_ids,
            ).astype(np.float32)
        _append_observation_std(observation_norm_stats, "dso_obs", dso_obs_vec)
        _append_observation_std(observation_norm_stats, "critic_state", critic_state)
        if cfg.normalize_observations:
            dso_obs_vec = _normalize_observation_array(dso_obs_vec)
            critic_state = _normalize_observation_array(critic_state)
        dso_tensor = torch.as_tensor(dso_obs_vec, dtype=torch.float32, device=device).unsqueeze(0)
        dso_mean, dso_log_std = actor_modules["dso_actor"](dso_tensor)
        dso_dist = Normal(dso_mean, dso_log_std.exp())
        raw_dso_action = dso_dist.rsample()
        normalized_dso_action = torch.clamp(raw_dso_action, -cfg.action_clip, cfg.action_clip)
        normalized_dso_np = normalized_dso_action.detach().cpu().numpy().reshape(-1)
        if use_structured_dso_actor:
            if structured_dso_spec is None:
                raise RuntimeError("structured_dso_spec missing while building shared DSO envelope action")
            action_payload: dict[str, Any] = {
                "dso_global_guidance": {
                    "envelope_action": normalized_envelope_action_to_payload(
                        normalized_dso_np,
                        structured_dso_spec,
                        action_clip=cfg.action_clip,
                        min_width_ratio=float(dso_actor_cfg.get("min_width_ratio", 0.10)),
                        max_width_ratio=float(dso_actor_cfg.get("max_width_ratio", 1.00)),
                        direction_logit_scale=float(dso_actor_cfg.get("direction_logit_scale", 5.0)),
                        source="happo_shared_rollout_sensitivity_attention_v1_actor",
                    )
                }
            }
        else:
            dso_targets = _targets_from_normalized_actions(
                normalized_dso_np,
                dso_obs,
                vpp_ids,
                cfg.action_clip,
            )
            action_payload = {"dso_global_guidance": {"targets": dso_targets}}

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
            _append_observation_std(observation_norm_stats, "dispatch_obs", encoded_vpp_obs)
            if cfg.normalize_observations:
                encoded_vpp_obs = _normalize_observation_array(encoded_vpp_obs)
            vpp_tensor = torch.as_tensor(encoded_vpp_obs, dtype=torch.float32, device=device).unsqueeze(0)
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
                "policy_version": policy_version,
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
            _append_observation_std(observation_norm_stats, "portfolio_obs", encoded_portfolio_obs)
            if cfg.normalize_observations:
                encoded_portfolio_obs = _normalize_observation_array(encoded_portfolio_obs)
            portfolio_tensor = torch.as_tensor(encoded_portfolio_obs, dtype=torch.float32, device=device).unsqueeze(0)
            if portfolio_decision_step:
                logits = portfolio_actor_for(vpp_id)(portfolio_tensor).squeeze(0)
                portfolio_dist = Categorical(logits=logits)
                action_idx = int(portfolio_dist.sample().item())
                portfolio_log_prob = float(
                    portfolio_dist.log_prob(torch.tensor(action_idx, device=device)).detach().cpu().item()
                )
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
                "policy_version": policy_version if portfolio_decision_step else f"{policy_version}_slow_loop_hold",
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
        return {
            "step": int(step),
            "action_payload": action_payload,
            "critic_state": critic_state,
            "action_summary": action_summary,
            "dso_obs_vec": dso_obs_vec,
            "raw_dso_action": raw_dso_action.detach().cpu().numpy().reshape(-1),
            "dso_log_prob": float(dso_dist.log_prob(raw_dso_action).sum().detach().cpu().item()),
            "dso_entropy": float(dso_dist.entropy().sum().detach().cpu().item()),
            "dispatch_obs_rows": np.asarray(dispatch_obs_rows, dtype=np.float32),
            "dispatch_raw_aggregate_rows": np.asarray(dispatch_raw_aggregate_rows, dtype=np.float32),
            "dispatch_raw_der_rows": np.asarray(dispatch_raw_der_rows, dtype=np.float32),
            "dispatch_log_prob_rows": np.asarray(dispatch_log_prob_rows, dtype=np.float32),
            "dispatch_entropy_rows": np.asarray(dispatch_entropy_rows, dtype=np.float32),
            "dispatch_der_counts": np.asarray(dispatch_der_counts, dtype=np.int64),
            "normalized_aggregate_actions": normalized_aggregate_actions,
            "normalized_der_actions": normalized_der_actions,
            "portfolio_obs_rows": np.asarray(portfolio_obs_rows, dtype=np.float32),
            "portfolio_action_rows": np.asarray(portfolio_action_rows, dtype=np.int64),
            "portfolio_log_prob_rows": np.asarray(portfolio_log_prob_rows, dtype=np.float32),
            "portfolio_entropy_rows": np.asarray(portfolio_entropy_rows, dtype=np.float32),
            "portfolio_update_mask_rows": np.asarray(portfolio_update_mask_rows, dtype=np.float32),
            "portfolio_decision_step": bool(portfolio_decision_step),
        }

    def append_shared_rollout_transition(
        *,
        episode_index: int,
        worker_index: int,
        worker_start_step: int,
        policy_version: str,
        action_bundle: dict[str, Any],
        transition: dict[str, Any],
        rollout: dict[str, list[Any]],
    ) -> dict[str, Any]:
        step = int(action_bundle["step"])
        reward_map = transition["reward_map"]
        truncations = transition["truncations"]
        infos = transition["infos"]
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
        dispatch_components = [
            infos[f"{vpp_id}_dispatch"].get("agent_reward_components", {})
            for vpp_id in vpp_ids
        ]
        portfolio_components = [
            infos[f"{vpp_id}_portfolio"].get("agent_reward_components", {})
            for vpp_id in vpp_ids
        ]
        raw_dispatch_rewards = np.asarray([float(reward_map[f"{vpp_id}_dispatch"]) for vpp_id in vpp_ids], dtype=np.float32)
        dso_attr_penalty = float(reward_components.get("dso_responsible_projection_penalty", shield_penalty))
        dispatch_attr_penalties = np.asarray(
            [
                float(component.get("dispatch_projection_penalty", shield_penalty))
                for component in dispatch_components
            ],
            dtype=np.float32,
        )
        dso_reward = raw_dso_reward - float(cfg.dso_shield_intervention_penalty_coef) * dso_attr_penalty
        dispatch_rewards = raw_dispatch_rewards - float(cfg.dispatch_shield_intervention_penalty_coef) * dispatch_attr_penalties
        portfolio_rewards = np.asarray([float(reward_map[f"{vpp_id}_portfolio"]) for vpp_id in vpp_ids], dtype=np.float32)
        if not bool(action_bundle["portfolio_decision_step"]):
            portfolio_rewards = np.zeros_like(portfolio_rewards)
        dispatch_trace_rows = dispatch_private_profit_trace_rows(
            episode=int(episode_index),
            step=int(step),
            algorithm=cfg.algorithm,
            vpp_ids=vpp_ids,
            dispatch_components=dispatch_components,
            raw_dispatch_rewards=raw_dispatch_rewards,
            train_dispatch_rewards=dispatch_rewards,
        )
        normalized_aggregate_actions = action_bundle["normalized_aggregate_actions"]
        normalized_der_actions = action_bundle["normalized_der_actions"]
        for row in dispatch_trace_rows:
            row["worker_index"] = int(worker_index)
            row["worker_start_step"] = int(worker_start_step)
            row["policy_version"] = str(policy_version)
            row["policy_normalized_aggregate_action"] = float(
                normalized_aggregate_actions.get(str(row.get("vpp_id")), 0.0)
            )
            der_action = np.asarray(
                normalized_der_actions.get(str(row.get("vpp_id")), np.zeros(0, dtype=np.float32)),
                dtype=np.float32,
            ).reshape(-1)
            row["policy_normalized_der_action_mean"] = float(np.mean(der_action)) if der_action.size else 0.0
            row["policy_normalized_der_action_std"] = float(np.std(der_action)) if der_action.size else 0.0
        dispatch_profit_trace_rows.extend(dispatch_trace_rows)
        reward_row = np.concatenate(
            [
                np.asarray([dso_reward], dtype=np.float32),
                dispatch_rewards,
                portfolio_rewards,
            ],
            axis=0,
        )
        learning_reward = float(dso_reward + dispatch_rewards.mean() + portfolio_rewards.mean())
        terminal = bool(all(truncations.values()))

        rollout["critic_state"].append(action_bundle["critic_state"])
        rollout["action_summary"].append(action_bundle["action_summary"])
        rollout["dso_obs"].append(action_bundle["dso_obs_vec"])
        rollout["dso_raw_action"].append(action_bundle["raw_dso_action"])
        rollout["dso_log_prob"].append(float(action_bundle["dso_log_prob"]))
        rollout["dso_entropy"].append(float(action_bundle["dso_entropy"]))
        rollout["dispatch_obs"].append(action_bundle["dispatch_obs_rows"])
        rollout["dispatch_raw_aggregate"].append(action_bundle["dispatch_raw_aggregate_rows"])
        rollout["dispatch_raw_der"].append(action_bundle["dispatch_raw_der_rows"])
        rollout["dispatch_log_prob"].append(action_bundle["dispatch_log_prob_rows"])
        rollout["dispatch_entropy"].append(action_bundle["dispatch_entropy_rows"])
        rollout["dispatch_der_count"].append(action_bundle["dispatch_der_counts"])
        rollout["portfolio_obs"].append(action_bundle["portfolio_obs_rows"])
        rollout["portfolio_action_idx"].append(action_bundle["portfolio_action_rows"])
        rollout["portfolio_log_prob"].append(action_bundle["portfolio_log_prob_rows"])
        rollout["portfolio_entropy"].append(action_bundle["portfolio_entropy_rows"])
        rollout["portfolio_update_mask"].append(action_bundle["portfolio_update_mask_rows"])
        rollout["rewards"].append(reward_row * float(cfg.reward_scale))
        rollout["worker_index"].append(int(worker_index))
        rollout["terminal"].append(terminal)
        rollout["policy_version"].append(policy_version)

        def component_mean(rows: list[dict[str, Any]], key: str) -> float:
            values = [float(row.get(key, 0.0)) for row in rows]
            return float(np.mean(values)) if values else 0.0

        step_rows.append(
            {
                "episode": int(episode_index),
                "step": int(step),
                "worker_index": int(worker_index),
                "worker_start_step": int(worker_start_step),
                "policy_version": policy_version,
                "algorithm": cfg.algorithm,
                "reward": learning_reward,
                "dso_reward": dso_reward,
                "dso_reward_env": raw_dso_reward,
                "dso_reward_train": dso_reward,
                "dso_reward_critic_scaled": dso_reward * float(cfg.reward_scale),
                "mean_dispatch_reward": float(dispatch_rewards.mean()) if dispatch_rewards.size else 0.0,
                "mean_dispatch_reward_env": float(raw_dispatch_rewards.mean()) if raw_dispatch_rewards.size else 0.0,
                "mean_dispatch_reward_train": float(dispatch_rewards.mean()) if dispatch_rewards.size else 0.0,
                "mean_portfolio_reward": float(portfolio_rewards.mean()) if portfolio_rewards.size else 0.0,
                "mean_portfolio_reward_env": float(np.asarray([float(reward_map[f"{vpp_id}_portfolio"]) for vpp_id in vpp_ids], dtype=np.float32).mean()) if vpp_ids else 0.0,
                "mean_portfolio_reward_train": float(portfolio_rewards.mean()) if portfolio_rewards.size else 0.0,
                "portfolio_decision_step": bool(action_bundle["portfolio_decision_step"]),
                "private_profit_proxy": component_mean(dispatch_components, "private_profit_proxy"),
                "economic_operational_surplus": component_mean(dispatch_components, "economic_operational_surplus"),
                "quality_adjusted_operational_surplus": component_mean(dispatch_components, "quality_adjusted_operational_surplus"),
                "service_quality_penalty_total": component_mean(dispatch_components, "service_quality_penalty_total"),
                "dispatch_projection_penalty": component_mean(dispatch_components, "dispatch_projection_penalty"),
                "settlement_audit_complete": component_mean(dispatch_components, "settlement_audit_complete"),
                "settlement_power_balance_ok": component_mean(dispatch_components, "settlement_power_balance_ok"),
                "service_payment": component_mean(dispatch_components, "service_payment"),
                "availability_payment": component_mean(dispatch_components, "availability_payment"),
                "storage_potential_shaping_reward": component_mean(dispatch_components, "storage_potential_shaping_reward"),
                "portfolio_window_profit": component_mean(portfolio_components, "portfolio_window_profit"),
                "portfolio_switching_cost": component_mean(portfolio_components, "portfolio_switching_cost"),
                "projection_gap_mw": float(projection_gap),
                "decoded_projection_gap_mw": float(decoded_projection_gap),
                "action_projection_gap_mw": float(shield_metrics["action_projection_gap_mw"]),
                "local_bounds_projection_gap_mw": float(shield_metrics["local_bounds_projection_gap_mw"]),
                "ac_aware_projection_gap_mw": float(shield_metrics["ac_aware_projection_gap_mw"]),
                "ac_certified_projection_gap_mw": float(shield_metrics["ac_certified_projection_gap_mw"]),
                "raw_action_violation_rate": float(shield_metrics["shield_intervention_count"] > 0.0),
                "certificate_repair_rate": float(shield_metrics["ac_certified_projection_gap_mw"] > 1e-9),
                "shield_intervention_gap_mw": float(shield_metrics["shield_intervention_gap_mw"]),
                "shield_intervention_penalty": shield_penalty,
                "shield_intervention_count": int(shield_metrics["shield_intervention_count"] > 0.0),
                "dso_safety_gate": float(reward_components.get("dso_safety_gate", 1.0)),
                "dso_vpp_welfare_raw": float(reward_components.get("dso_vpp_welfare_raw", 0.0)),
                "raw_action_safety_cost_norm": float(reward_components.get("raw_action_safety_cost_norm", 0.0)),
                "projected_action_safety_cost_norm": float(reward_components.get("projected_action_safety_cost_norm", 0.0)),
                "post_ac_violation_count": float(reward_components.get("post_ac_violation_count", len(violations))),
                "post_ac_security_penalty": float(reward_components.get("post_ac_security_penalty", 0.0)),
                "total_cost": float(reward_components.get("total_cost", -dso_reward)),
                "violation_count": int(len(violations)),
            }
        )
        return {
            "episode_reward": learning_reward,
            "total_cost": float(reward_components.get("total_cost", -dso_reward)),
            "violation_count": int(len(violations)),
            "projection_gap_mw": float(projection_gap),
            "ac_projection_gap_mw": float(shield_metrics["ac_aware_projection_gap_mw"]),
            "local_projection_gap_mw": float(shield_metrics["local_bounds_projection_gap_mw"]),
            "shield_intervention_penalty": shield_penalty,
            "shield_intervention_count": int(shield_metrics["shield_intervention_count"] > 0.0),
            "terminal": terminal,
        }

    def collect_shared_worker_rollout(
        *,
        episode_index: int,
        worker_index: int,
        worker_env: Any,
        worker_observations: dict[str, dict[str, Any]],
        worker_start_step: int,
        rollout: dict[str, list[Any]],
        policy_version: str,
        fragment_steps: int,
    ) -> dict[str, Any]:
        env = worker_env
        observations = worker_observations
        worker_reward = 0.0
        worker_cost = 0.0
        worker_violation_count = 0
        worker_projection_gap = 0.0
        worker_ac_projection_gap = 0.0
        worker_local_projection_gap = 0.0
        worker_shield_penalty = 0.0
        worker_shield_count = 0
        terminal_reached = False
        steps_collected = 0

        for local_step in range(fragment_steps):
            step = int(env.current_step)
            dso_obs = observations["dso_global_guidance"]
            if use_structured_dso_actor:
                dso_obs_vec, _current_structured_spec = build_happo_structured_dso_observation(
                    env.scenario,
                    step=env.current_step,
                    config=env.scenario.config,
                )
                if structured_dso_spec is not None and int(len(dso_obs_vec)) != int(structured_dso_spec.flat_dim):
                    raise RuntimeError(
                        "Structured DSO observation shape changed during shared HAPPO rollout: "
                        f"got {len(dso_obs_vec)}, expected {structured_dso_spec.flat_dim}."
                    )
                dso_obs_vec = dso_obs_vec.astype(np.float32)
            else:
                dso_obs_vec = encode_dso_observation(dso_obs, vpp_ids).astype(np.float32)
            critic_state = encode_critic_global_state(
                build_critic_global_state(env.scenario, env.current_step),
                vpp_ids,
            ).astype(np.float32)
            _append_observation_std(observation_norm_stats, "dso_obs", dso_obs_vec)
            _append_observation_std(observation_norm_stats, "critic_state", critic_state)
            if cfg.normalize_observations:
                dso_obs_vec = _normalize_observation_array(dso_obs_vec)
                critic_state = _normalize_observation_array(critic_state)
            dso_tensor = torch.as_tensor(dso_obs_vec, dtype=torch.float32, device=device).unsqueeze(0)
            dso_mean, dso_log_std = actor_modules["dso_actor"](dso_tensor)
            dso_dist = Normal(dso_mean, dso_log_std.exp())
            raw_dso_action = dso_dist.rsample()
            normalized_dso_action = torch.clamp(raw_dso_action, -cfg.action_clip, cfg.action_clip)
            normalized_dso_np = normalized_dso_action.detach().cpu().numpy().reshape(-1)
            if use_structured_dso_actor:
                if structured_dso_spec is None:
                    raise RuntimeError("structured_dso_spec missing while building shared DSO envelope action")
                action_payload: dict[str, Any] = {
                    "dso_global_guidance": {
                        "envelope_action": normalized_envelope_action_to_payload(
                            normalized_dso_np,
                            structured_dso_spec,
                            action_clip=cfg.action_clip,
                            min_width_ratio=float(dso_actor_cfg.get("min_width_ratio", 0.10)),
                            max_width_ratio=float(dso_actor_cfg.get("max_width_ratio", 1.00)),
                            direction_logit_scale=float(dso_actor_cfg.get("direction_logit_scale", 5.0)),
                            source="happo_shared_rollout_sensitivity_attention_v1_actor",
                        )
                    }
                }
            else:
                dso_targets = _targets_from_normalized_actions(
                    normalized_dso_np,
                    dso_obs,
                    vpp_ids,
                    cfg.action_clip,
                )
                action_payload = {"dso_global_guidance": {"targets": dso_targets}}

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
                _append_observation_std(observation_norm_stats, "dispatch_obs", encoded_vpp_obs)
                if cfg.normalize_observations:
                    encoded_vpp_obs = _normalize_observation_array(encoded_vpp_obs)
                vpp_tensor = torch.as_tensor(encoded_vpp_obs, dtype=torch.float32, device=device).unsqueeze(0)
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
                    "policy_version": policy_version,
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
                _append_observation_std(observation_norm_stats, "portfolio_obs", encoded_portfolio_obs)
                if cfg.normalize_observations:
                    encoded_portfolio_obs = _normalize_observation_array(encoded_portfolio_obs)
                portfolio_tensor = torch.as_tensor(encoded_portfolio_obs, dtype=torch.float32, device=device).unsqueeze(0)
                if portfolio_decision_step:
                    logits = portfolio_actor_for(vpp_id)(portfolio_tensor).squeeze(0)
                    portfolio_dist = Categorical(logits=logits)
                    action_idx = int(portfolio_dist.sample().item())
                    portfolio_log_prob = float(
                        portfolio_dist.log_prob(torch.tensor(action_idx, device=device)).detach().cpu().item()
                    )
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
                    "policy_version": policy_version if portfolio_decision_step else f"{policy_version}_slow_loop_hold",
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
            dispatch_components = [
                infos[f"{vpp_id}_dispatch"].get("agent_reward_components", {})
                for vpp_id in vpp_ids
            ]
            portfolio_components = [
                infos[f"{vpp_id}_portfolio"].get("agent_reward_components", {})
                for vpp_id in vpp_ids
            ]
            raw_dispatch_rewards = np.asarray([float(reward_map[f"{vpp_id}_dispatch"]) for vpp_id in vpp_ids], dtype=np.float32)
            dso_attr_penalty = float(reward_components.get("dso_responsible_projection_penalty", shield_penalty))
            dispatch_attr_penalties = np.asarray(
                [
                    float(component.get("dispatch_projection_penalty", shield_penalty))
                    for component in dispatch_components
                ],
                dtype=np.float32,
            )
            dso_reward = raw_dso_reward - float(cfg.dso_shield_intervention_penalty_coef) * dso_attr_penalty
            dispatch_rewards = raw_dispatch_rewards - float(cfg.dispatch_shield_intervention_penalty_coef) * dispatch_attr_penalties
            portfolio_rewards = np.asarray([float(reward_map[f"{vpp_id}_portfolio"]) for vpp_id in vpp_ids], dtype=np.float32)
            if not portfolio_decision_step:
                portfolio_rewards = np.zeros_like(portfolio_rewards)
            dispatch_trace_rows = dispatch_private_profit_trace_rows(
                episode=int(episode_index),
                step=int(step),
                algorithm=cfg.algorithm,
                vpp_ids=vpp_ids,
                dispatch_components=dispatch_components,
                raw_dispatch_rewards=raw_dispatch_rewards,
                train_dispatch_rewards=dispatch_rewards,
            )
            for row in dispatch_trace_rows:
                row["worker_index"] = int(worker_index)
                row["worker_start_step"] = int(worker_start_step)
                row["policy_version"] = str(policy_version)
                row["policy_normalized_aggregate_action"] = float(
                    normalized_aggregate_actions.get(str(row.get("vpp_id")), 0.0)
                )
                der_action = np.asarray(
                    normalized_der_actions.get(str(row.get("vpp_id")), np.zeros(0, dtype=np.float32)),
                    dtype=np.float32,
                ).reshape(-1)
                row["policy_normalized_der_action_mean"] = float(np.mean(der_action)) if der_action.size else 0.0
                row["policy_normalized_der_action_std"] = float(np.std(der_action)) if der_action.size else 0.0
            dispatch_profit_trace_rows.extend(dispatch_trace_rows)
            reward_row = np.concatenate(
                [
                    np.asarray([dso_reward], dtype=np.float32),
                    dispatch_rewards,
                    portfolio_rewards,
                ],
                axis=0,
            )
            learning_reward = float(dso_reward + dispatch_rewards.mean() + portfolio_rewards.mean())
            terminal = bool(all(truncations.values()))

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
            rollout["worker_index"].append(int(worker_index))
            rollout["terminal"].append(terminal)
            rollout["policy_version"].append(policy_version)

            def component_mean(rows: list[dict[str, Any]], key: str) -> float:
                values = [float(row.get(key, 0.0)) for row in rows]
                return float(np.mean(values)) if values else 0.0

            step_rows.append(
                {
                    "episode": int(episode_index),
                    "step": int(step),
                    "worker_index": int(worker_index),
                    "worker_start_step": int(worker_start_step),
                    "policy_version": policy_version,
                    "algorithm": cfg.algorithm,
                    "reward": learning_reward,
                    "dso_reward": dso_reward,
                    "dso_reward_env": raw_dso_reward,
                    "dso_reward_train": dso_reward,
                    "dso_reward_critic_scaled": dso_reward * float(cfg.reward_scale),
                    "mean_dispatch_reward": float(dispatch_rewards.mean()) if dispatch_rewards.size else 0.0,
                    "mean_dispatch_reward_env": float(raw_dispatch_rewards.mean()) if raw_dispatch_rewards.size else 0.0,
                    "mean_dispatch_reward_train": float(dispatch_rewards.mean()) if dispatch_rewards.size else 0.0,
                    "mean_portfolio_reward": float(portfolio_rewards.mean()) if portfolio_rewards.size else 0.0,
                    "mean_portfolio_reward_env": float(np.asarray([float(reward_map[f"{vpp_id}_portfolio"]) for vpp_id in vpp_ids], dtype=np.float32).mean()) if vpp_ids else 0.0,
                    "mean_portfolio_reward_train": float(portfolio_rewards.mean()) if portfolio_rewards.size else 0.0,
                    "portfolio_decision_step": bool(portfolio_decision_step),
                    "private_profit_proxy": component_mean(dispatch_components, "private_profit_proxy"),
                    "economic_operational_surplus": component_mean(dispatch_components, "economic_operational_surplus"),
                    "quality_adjusted_operational_surplus": component_mean(dispatch_components, "quality_adjusted_operational_surplus"),
                    "service_quality_penalty_total": component_mean(dispatch_components, "service_quality_penalty_total"),
                    "dispatch_projection_penalty": component_mean(dispatch_components, "dispatch_projection_penalty"),
                    "settlement_audit_complete": component_mean(dispatch_components, "settlement_audit_complete"),
                    "settlement_power_balance_ok": component_mean(dispatch_components, "settlement_power_balance_ok"),
                    "service_payment": component_mean(dispatch_components, "service_payment"),
                    "availability_payment": component_mean(dispatch_components, "availability_payment"),
                    "storage_potential_shaping_reward": component_mean(dispatch_components, "storage_potential_shaping_reward"),
                    "portfolio_window_profit": component_mean(portfolio_components, "portfolio_window_profit"),
                    "portfolio_switching_cost": component_mean(portfolio_components, "portfolio_switching_cost"),
                    "projection_gap_mw": float(projection_gap),
                    "decoded_projection_gap_mw": float(decoded_projection_gap),
                    "action_projection_gap_mw": float(shield_metrics["action_projection_gap_mw"]),
                    "local_bounds_projection_gap_mw": float(shield_metrics["local_bounds_projection_gap_mw"]),
                    "ac_aware_projection_gap_mw": float(shield_metrics["ac_aware_projection_gap_mw"]),
                    "ac_certified_projection_gap_mw": float(shield_metrics["ac_certified_projection_gap_mw"]),
                    "raw_action_violation_rate": float(shield_metrics["shield_intervention_count"] > 0.0),
                    "certificate_repair_rate": float(shield_metrics["ac_certified_projection_gap_mw"] > 1e-9),
                    "shield_intervention_gap_mw": float(shield_metrics["shield_intervention_gap_mw"]),
                    "shield_intervention_penalty": shield_penalty,
                    "shield_intervention_count": int(shield_metrics["shield_intervention_count"] > 0.0),
                    "dso_safety_gate": float(reward_components.get("dso_safety_gate", 1.0)),
                    "dso_vpp_welfare_raw": float(reward_components.get("dso_vpp_welfare_raw", 0.0)),
                    "raw_action_safety_cost_norm": float(reward_components.get("raw_action_safety_cost_norm", 0.0)),
                    "projected_action_safety_cost_norm": float(reward_components.get("projected_action_safety_cost_norm", 0.0)),
                    "post_ac_violation_count": float(reward_components.get("post_ac_violation_count", len(violations))),
                    "post_ac_security_penalty": float(reward_components.get("post_ac_security_penalty", 0.0)),
                    "total_cost": float(reward_components.get("total_cost", -dso_reward)),
                    "violation_count": int(len(violations)),
                }
            )

            worker_reward += learning_reward
            worker_cost += float(reward_components.get("total_cost", -dso_reward))
            worker_violation_count += int(len(violations))
            worker_projection_gap += float(projection_gap)
            worker_ac_projection_gap += float(shield_metrics["ac_aware_projection_gap_mw"])
            worker_local_projection_gap += float(shield_metrics["local_bounds_projection_gap_mw"])
            worker_shield_penalty += shield_penalty
            worker_shield_count += int(shield_metrics["shield_intervention_count"] > 0.0)
            steps_collected += 1
            maybe_report_shared_rollout_step(
                episode_index=int(episode_index),
                worker_index=int(worker_index),
                worker_count=int(shared_rollout_workers),
                worker_start_step=int(worker_start_step),
                local_step_index=int(local_step),
                env_step=int(step),
                fragment_steps=int(fragment_steps),
                policy_version=str(policy_version),
                worker_reward=float(worker_reward),
                worker_cost=float(worker_cost),
                worker_violation_count=int(worker_violation_count),
                worker_projection_gap=float(worker_projection_gap),
            )
            observations = next_observations
            terminal_reached = terminal
            if terminal:
                break

        bootstrap_value = np.zeros(len(head_names), dtype=np.float32)
        fragment_cut = bool(steps_collected > 0 and not terminal_reached and env.current_step < int(cfg.horizon_steps))
        if fragment_cut:
            next_critic_state = encode_critic_global_state(
                build_critic_global_state(env.scenario, env.current_step),
                vpp_ids,
            ).astype(np.float32)
            if cfg.normalize_observations:
                next_critic_state = _normalize_observation_array(next_critic_state)
            next_state_tensor = torch.as_tensor(next_critic_state, dtype=torch.float32, device=device).unsqueeze(0)
            next_action_tensor = torch.zeros((1, value_action_dim), dtype=torch.float32, device=device)
            with torch.no_grad():
                bootstrap_value = value_critic(next_state_tensor, next_action_tensor).detach().cpu().numpy().reshape(-1).astype(np.float32)
        next_worker_observations = observations
        reset_after_terminal = False
        if terminal_reached:
            next_seed = int(cfg.seed) + (int(episode_index) + 1) * 1000 + int(worker_index)
            next_worker_observations, _ = env.reset(seed=next_seed, start_step=int(worker_start_step))
            reset_after_terminal = True
        return {
            "episode_reward": worker_reward,
            "total_cost": worker_cost,
            "violation_count": worker_violation_count,
            "projection_gap_mw": worker_projection_gap,
            "ac_projection_gap_mw": worker_ac_projection_gap,
            "local_projection_gap_mw": worker_local_projection_gap,
            "shield_intervention_penalty": worker_shield_penalty,
            "shield_intervention_count": worker_shield_count,
            "steps_collected": steps_collected,
            "fragment_cut": fragment_cut,
            "bootstrap_value": bootstrap_value,
            "next_observations": next_worker_observations,
            "reset_after_terminal": reset_after_terminal,
        }

    def collect_subprocess_shared_rollout(
        *,
        episode_index: int,
        rollout: dict[str, list[Any]],
        policy_version: str,
        fragment_steps: int,
    ) -> dict[str, Any]:
        batch_started = time.perf_counter()
        policy_forward_seconds = 0.0
        env_step_wall_seconds = 0.0
        wait_for_workers_seconds = 0.0
        worker_step_seconds: list[float] = []
        worker_step_seconds_by_worker: dict[int, list[float]] = {
            int(worker_index): []
            for worker_index in shared_worker_states
        }
        worker_metrics: dict[int, dict[str, Any]] = {
            int(worker_index): {
                "episode_reward": 0.0,
                "total_cost": 0.0,
                "violation_count": 0,
                "projection_gap_mw": 0.0,
                "ac_projection_gap_mw": 0.0,
                "local_projection_gap_mw": 0.0,
                "shield_intervention_penalty": 0.0,
                "shield_intervention_count": 0,
                "steps_collected": 0,
                "terminal_reached": False,
            }
            for worker_index in shared_worker_states
        }
        active_workers = set(int(worker_index) for worker_index in shared_worker_states)

        for local_step in range(fragment_steps):
            if not active_workers:
                break
            policy_started = time.perf_counter()
            action_bundles = {
                int(worker_index): build_shared_rollout_action_bundle(
                    worker_state=shared_worker_states[int(worker_index)],
                    policy_version=policy_version,
                )
                for worker_index in sorted(active_workers)
            }
            policy_forward_seconds += time.perf_counter() - policy_started

            step_wall_started = time.perf_counter()
            for worker_index, action_bundle in action_bundles.items():
                shared_rollout_subprocess_workers[int(worker_index)].step_async(action_bundle["action_payload"])
            wait_started = time.perf_counter()
            step_payloads = {
                int(worker_index): shared_rollout_subprocess_workers[int(worker_index)].recv()
                for worker_index in action_bundles
            }
            wait_for_workers_seconds += time.perf_counter() - wait_started
            env_step_wall_seconds += time.perf_counter() - step_wall_started

            for worker_index, payload in step_payloads.items():
                transition = payload["transition"]
                worker_duration = float(transition.get("worker_step_seconds", 0.0))
                worker_step_seconds.append(worker_duration)
                worker_step_seconds_by_worker[int(worker_index)].append(worker_duration)
                next_state = dict(payload["state"])
                next_state["worker_start_step"] = int(shared_worker_states[int(worker_index)]["worker_start_step"])
                shared_worker_states[int(worker_index)] = next_state
                action_bundle = action_bundles[int(worker_index)]
                step_metrics_payload = append_shared_rollout_transition(
                    episode_index=int(episode_index),
                    worker_index=int(worker_index),
                    worker_start_step=int(next_state["worker_start_step"]),
                    policy_version=str(policy_version),
                    action_bundle=action_bundle,
                    transition=transition,
                    rollout=rollout,
                )
                metrics = worker_metrics[int(worker_index)]
                metrics["episode_reward"] += float(step_metrics_payload["episode_reward"])
                metrics["total_cost"] += float(step_metrics_payload["total_cost"])
                metrics["violation_count"] += int(step_metrics_payload["violation_count"])
                metrics["projection_gap_mw"] += float(step_metrics_payload["projection_gap_mw"])
                metrics["ac_projection_gap_mw"] += float(step_metrics_payload["ac_projection_gap_mw"])
                metrics["local_projection_gap_mw"] += float(step_metrics_payload["local_projection_gap_mw"])
                metrics["shield_intervention_penalty"] += float(step_metrics_payload["shield_intervention_penalty"])
                metrics["shield_intervention_count"] += int(step_metrics_payload["shield_intervention_count"])
                metrics["steps_collected"] += 1
                metrics["terminal_reached"] = bool(step_metrics_payload["terminal"])
                maybe_report_shared_rollout_step(
                    episode_index=int(episode_index),
                    worker_index=int(worker_index),
                    worker_count=int(shared_rollout_workers),
                    worker_start_step=int(next_state["worker_start_step"]),
                    local_step_index=int(local_step),
                    env_step=int(action_bundle["step"]),
                    fragment_steps=int(fragment_steps),
                    policy_version=str(policy_version),
                    worker_reward=float(metrics["episode_reward"]),
                    worker_cost=float(metrics["total_cost"]),
                    worker_violation_count=int(metrics["violation_count"]),
                    worker_projection_gap=float(metrics["projection_gap_mw"]),
                )
                if bool(step_metrics_payload["terminal"]):
                    active_workers.discard(int(worker_index))

        bootstrap_values_by_worker: dict[int, np.ndarray] = {}
        reset_after_terminal_count = 0
        fragment_cut_count = 0
        for worker_index, metrics in worker_metrics.items():
            bootstrap_value = np.zeros(len(head_names), dtype=np.float32)
            if int(metrics["steps_collected"]) > 0 and not bool(metrics["terminal_reached"]):
                current_step = int(shared_worker_states[int(worker_index)].get("current_step", 0))
                if current_step < int(cfg.horizon_steps):
                    next_critic_state = np.asarray(
                        shared_worker_states[int(worker_index)]["critic_state_vec"],
                        dtype=np.float32,
                    )
                    if cfg.normalize_observations:
                        next_critic_state = _normalize_observation_array(next_critic_state)
                    next_state_tensor = torch.as_tensor(next_critic_state, dtype=torch.float32, device=device).unsqueeze(0)
                    next_action_tensor = torch.zeros((1, value_action_dim), dtype=torch.float32, device=device)
                    with torch.no_grad():
                        bootstrap_value = value_critic(next_state_tensor, next_action_tensor).detach().cpu().numpy().reshape(-1).astype(np.float32)
                    fragment_cut_count += 1
            if bool(metrics["terminal_reached"]):
                next_seed = int(cfg.seed) + (int(episode_index) + 1) * 1000 + int(worker_index)
                reset_payload = shared_rollout_subprocess_workers[int(worker_index)].reset(
                    seed=next_seed,
                    start_step=int(shared_worker_states[int(worker_index)]["worker_start_step"]),
                )
                reset_state = dict(reset_payload["state"])
                reset_state["worker_start_step"] = int(shared_worker_states[int(worker_index)]["worker_start_step"])
                shared_worker_states[int(worker_index)] = reset_state
                reset_after_terminal_count += 1
            bootstrap_values_by_worker[int(worker_index)] = bootstrap_value

        total_samples = int(sum(int(item["steps_collected"]) for item in worker_metrics.values()))
        slowest_worker_id = -1
        if worker_step_seconds_by_worker:
            worker_means = {
                int(worker_index): float(np.mean(values)) if values else 0.0
                for worker_index, values in worker_step_seconds_by_worker.items()
            }
            slowest_worker_id = max(worker_means, key=worker_means.get)
        rollout_collect_seconds = time.perf_counter() - batch_started
        return {
            "worker_metrics": list(worker_metrics.values()),
            "bootstrap_values_by_worker": bootstrap_values_by_worker,
            "fragment_cut_count": int(fragment_cut_count),
            "reset_after_terminal_count": int(reset_after_terminal_count),
            "performance": {
                "shared_rollout_backend": str(shared_rollout_backend),
                "num_workers": int(shared_rollout_workers),
                "rollout_fragment_steps": int(shared_rollout_fragment_steps_default),
                "rollout_collect_seconds": float(rollout_collect_seconds),
                "policy_forward_seconds": float(policy_forward_seconds),
                "env_step_wall_seconds": float(env_step_wall_seconds),
                "env_step_worker_mean_seconds": float(np.mean(worker_step_seconds)) if worker_step_seconds else 0.0,
                "env_step_worker_max_seconds": float(np.max(worker_step_seconds)) if worker_step_seconds else 0.0,
                "wait_for_workers_seconds": float(wait_for_workers_seconds),
                "samples_collected": int(total_samples),
                "samples_per_second": float(total_samples / max(1e-9, rollout_collect_seconds)),
                "slowest_worker_id": int(slowest_worker_id),
            },
        }

    shared_rollout_fragment_steps_default = max(1, int(cfg.rollout_fragment_steps or cfg.horizon_steps))
    shared_rollout_fragment_steps_default = min(shared_rollout_fragment_steps_default, int(cfg.horizon_steps))

    def shared_rollout_worker_start_step(worker_index: int) -> int:
        if not shared_rollout_enabled:
            return 0
        return int((int(worker_index) * int(shared_rollout_fragment_steps_default)) % max(1, int(cfg.horizon_steps)))

    shared_worker_states: dict[int, dict[str, Any]] = {}
    if shared_rollout_enabled:
        for worker_index in range(shared_rollout_workers):
            worker_start_step = shared_rollout_worker_start_step(int(worker_index))
            shared_rollout_worker_start_offsets[int(worker_index)] = int(worker_start_step)
            if shared_rollout_backend == "subprocess":
                worker = SubprocessSharedRolloutWorker(
                    SharedRolloutWorkerSpec(
                        worker_index=int(worker_index),
                        config_path=config_path_text,
                        horizon_steps=int(cfg.horizon_steps),
                        use_structured_dso_actor=bool(use_structured_dso_actor),
                        vpp_ids=tuple(str(vpp_id) for vpp_id in vpp_ids),
                        blas_threads=1,
                    )
                )
                worker.start()
                reset_payload = worker.reset(
                    seed=int(cfg.seed) + int(worker_index),
                    start_step=int(worker_start_step),
                )
                shared_rollout_subprocess_workers[int(worker_index)] = worker
                if worker.pid is not None:
                    shared_rollout_subprocess_worker_pids[int(worker_index)] = int(worker.pid)
                worker_state = dict(reset_payload["state"])
                worker_state["worker_start_step"] = int(worker_start_step)
                shared_worker_states[int(worker_index)] = worker_state
            else:
                worker_env = MultiAgentVPPDSOEnv(config_path=config_path, horizon_steps=cfg.horizon_steps)
                worker_observations, _ = worker_env.reset(
                    seed=int(cfg.seed) + int(worker_index),
                    start_step=int(worker_start_step),
                )
                shared_worker_states[int(worker_index)] = {
                    "env": worker_env,
                    "observations": worker_observations,
                    "current_step": int(worker_env.current_step),
                    "worker_start_step": int(worker_start_step),
                }

    episode_iter, has_tqdm_progress = _episode_progress(range(cfg.episodes), total=cfg.episodes, desc="HAPPO")
    for episode in episode_iter:
        env = None
        observations: dict[str, dict[str, Any]] = {}
        if not shared_rollout_enabled:
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
            "worker_index": [],
            "terminal": [],
            "policy_version": [],
            "bootstrap_values_by_worker": {},
        }
        episode_reward = 0.0
        total_cost = 0.0
        violation_count = 0
        projection_gap_total = 0.0
        ac_projection_gap_total = 0.0
        local_projection_gap_total = 0.0
        shield_intervention_penalty_total = 0.0
        shield_intervention_count = 0

        if shared_rollout_enabled:
            fragment_steps = max(1, int(cfg.rollout_fragment_steps or cfg.horizon_steps))
            fragment_steps = min(fragment_steps, int(cfg.horizon_steps))
            policy_version = f"{cfg.algorithm}:shared_rollout:episode={int(episode)}"
            shared_rollout_batches += 1
            shared_rollout_policy_versions[int(episode)] = policy_version
            shared_rollout_worker_counts[int(episode)] = int(shared_rollout_workers)
            shared_rollout_fragment_steps_by_episode[int(episode)] = int(fragment_steps)
            worker_metrics: list[dict[str, Any]] = []
            if shared_rollout_backend == "subprocess":
                subprocess_batch = collect_subprocess_shared_rollout(
                    episode_index=int(episode),
                    rollout=rollout,
                    policy_version=policy_version,
                    fragment_steps=int(fragment_steps),
                )
                worker_metrics = list(subprocess_batch["worker_metrics"])
                shared_rollout_perf_by_episode[int(episode)] = dict(subprocess_batch["performance"])
                for worker_index, bootstrap_value in subprocess_batch["bootstrap_values_by_worker"].items():
                    rollout["bootstrap_values_by_worker"][int(worker_index)] = bootstrap_value
                    shared_rollout_bootstrap_values.extend([float(value) for value in np.asarray(bootstrap_value).reshape(-1)])
                shared_rollout_total_samples += int(subprocess_batch["performance"]["samples_collected"])
                shared_rollout_fragment_cut_count += int(subprocess_batch["fragment_cut_count"])
                shared_rollout_worker_terminal_reset_count += int(subprocess_batch["reset_after_terminal_count"])
            else:
                serial_rollout_started = time.perf_counter()
                serial_worker_seconds: list[float] = []
                for worker_index in range(shared_rollout_workers):
                    worker_state = shared_worker_states[int(worker_index)]
                    worker_started = time.perf_counter()
                    metrics = collect_shared_worker_rollout(
                        episode_index=int(episode),
                        worker_index=int(worker_index),
                        worker_env=worker_state["env"],
                        worker_observations=worker_state["observations"],
                        worker_start_step=int(worker_state["worker_start_step"]),
                        rollout=rollout,
                        policy_version=policy_version,
                        fragment_steps=int(fragment_steps),
                    )
                    serial_worker_seconds.append(time.perf_counter() - worker_started)
                    worker_metrics.append(metrics)
                    worker_state["observations"] = metrics["next_observations"]
                    worker_state["current_step"] = int(worker_state["env"].current_step)
                    rollout["bootstrap_values_by_worker"][int(worker_index)] = metrics["bootstrap_value"]
                    shared_rollout_total_samples += int(metrics["steps_collected"])
                    if bool(metrics["fragment_cut"]):
                        shared_rollout_fragment_cut_count += 1
                    if bool(metrics["reset_after_terminal"]):
                        shared_rollout_worker_terminal_reset_count += 1
                    shared_rollout_bootstrap_values.extend([float(value) for value in np.asarray(metrics["bootstrap_value"]).reshape(-1)])
                serial_collect_seconds = time.perf_counter() - serial_rollout_started
                serial_samples = int(sum(int(item["steps_collected"]) for item in worker_metrics))
                shared_rollout_perf_by_episode[int(episode)] = {
                    "shared_rollout_backend": str(shared_rollout_backend),
                    "num_workers": int(shared_rollout_workers),
                    "rollout_fragment_steps": int(shared_rollout_fragment_steps_default),
                    "rollout_collect_seconds": float(serial_collect_seconds),
                    "policy_forward_seconds": 0.0,
                    "env_step_wall_seconds": float(serial_collect_seconds),
                    "env_step_worker_mean_seconds": float(np.mean(serial_worker_seconds)) if serial_worker_seconds else 0.0,
                    "env_step_worker_max_seconds": float(np.max(serial_worker_seconds)) if serial_worker_seconds else 0.0,
                    "wait_for_workers_seconds": 0.0,
                    "samples_collected": int(serial_samples),
                    "samples_per_second": float(serial_samples / max(1e-9, serial_collect_seconds)),
                    "slowest_worker_id": int(np.argmax(serial_worker_seconds)) if serial_worker_seconds else -1,
                }
            denominator = max(1, int(shared_rollout_workers))
            episode_reward = float(sum(float(item["episode_reward"]) for item in worker_metrics) / denominator)
            total_cost = float(sum(float(item["total_cost"]) for item in worker_metrics) / denominator)
            violation_count = int(sum(int(item["violation_count"]) for item in worker_metrics))
            projection_gap_total = float(sum(float(item["projection_gap_mw"]) for item in worker_metrics) / denominator)
            ac_projection_gap_total = float(sum(float(item["ac_projection_gap_mw"]) for item in worker_metrics) / denominator)
            local_projection_gap_total = float(sum(float(item["local_projection_gap_mw"]) for item in worker_metrics) / denominator)
            shield_intervention_penalty_total = float(
                sum(float(item["shield_intervention_penalty"]) for item in worker_metrics) / denominator
            )
            shield_intervention_count = int(sum(int(item["shield_intervention_count"]) for item in worker_metrics))
            if cfg.rollout_policy_version_check and len(set(str(value) for value in rollout["policy_version"])) != 1:
                shared_rollout_policy_version_mismatch_count += 1
                raise RuntimeError("Merged HAPPO shared rollout batch contains multiple behavior policy versions.")

        step_iter = range(0) if shared_rollout_enabled else range(cfg.horizon_steps)
        for step in step_iter:
            dso_obs = observations["dso_global_guidance"]
            if use_structured_dso_actor:
                dso_obs_vec, _current_structured_spec = build_happo_structured_dso_observation(
                    env.scenario,
                    step=env.current_step,
                    config=env.scenario.config,
                )
                if structured_dso_spec is not None and int(len(dso_obs_vec)) != int(structured_dso_spec.flat_dim):
                    raise RuntimeError(
                        "Structured DSO observation shape changed during HAPPO rollout: "
                        f"got {len(dso_obs_vec)}, expected {structured_dso_spec.flat_dim}."
                    )
                dso_obs_vec = dso_obs_vec.astype(np.float32)
            else:
                dso_obs_vec = encode_dso_observation(dso_obs, vpp_ids).astype(np.float32)
            critic_state = encode_critic_global_state(
                build_critic_global_state(env.scenario, env.current_step),
                vpp_ids,
            ).astype(np.float32)
            _append_observation_std(observation_norm_stats, "dso_obs", dso_obs_vec)
            _append_observation_std(observation_norm_stats, "critic_state", critic_state)
            if cfg.normalize_observations:
                dso_obs_vec = _normalize_observation_array(dso_obs_vec)
                critic_state = _normalize_observation_array(critic_state)
            dso_tensor = torch.as_tensor(dso_obs_vec, dtype=torch.float32, device=device).unsqueeze(0)
            dso_mean, dso_log_std = actor_modules["dso_actor"](dso_tensor)
            dso_dist = Normal(dso_mean, dso_log_std.exp())
            raw_dso_action = dso_dist.rsample()
            normalized_dso_action = torch.clamp(raw_dso_action, -cfg.action_clip, cfg.action_clip)
            normalized_dso_np = normalized_dso_action.detach().cpu().numpy().reshape(-1)
            if use_structured_dso_actor:
                if structured_dso_spec is None:
                    raise RuntimeError("structured_dso_spec missing while building unified DSO envelope action")
                action_payload: dict[str, Any] = {
                    "dso_global_guidance": {
                        "envelope_action": normalized_envelope_action_to_payload(
                            normalized_dso_np,
                            structured_dso_spec,
                            action_clip=cfg.action_clip,
                            min_width_ratio=float(dso_actor_cfg.get("min_width_ratio", 0.10)),
                            max_width_ratio=float(dso_actor_cfg.get("max_width_ratio", 1.00)),
                            direction_logit_scale=float(dso_actor_cfg.get("direction_logit_scale", 5.0)),
                            source="happo_sensitivity_attention_v1_unified_actor",
                        )
                    }
                }
            else:
                dso_targets = _targets_from_normalized_actions(
                    normalized_dso_np,
                    dso_obs,
                    vpp_ids,
                    cfg.action_clip,
                )
                action_payload = {"dso_global_guidance": {"targets": dso_targets}}
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
                _append_observation_std(observation_norm_stats, "dispatch_obs", encoded_vpp_obs)
                if cfg.normalize_observations:
                    encoded_vpp_obs = _normalize_observation_array(encoded_vpp_obs)
                vpp_tensor = torch.as_tensor(encoded_vpp_obs, dtype=torch.float32, device=device).unsqueeze(0)
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
                _append_observation_std(observation_norm_stats, "portfolio_obs", encoded_portfolio_obs)
                if cfg.normalize_observations:
                    encoded_portfolio_obs = _normalize_observation_array(encoded_portfolio_obs)
                portfolio_tensor = torch.as_tensor(encoded_portfolio_obs, dtype=torch.float32, device=device).unsqueeze(0)
                if portfolio_decision_step:
                    logits = portfolio_actor_for(vpp_id)(portfolio_tensor).squeeze(0)
                    portfolio_dist = Categorical(logits=logits)
                    action_idx = int(portfolio_dist.sample().item())
                    portfolio_log_prob = float(
                        portfolio_dist.log_prob(torch.tensor(action_idx, device=device)).detach().cpu().item()
                    )
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
            dispatch_components = [
                infos[f"{vpp_id}_dispatch"].get("agent_reward_components", {})
                for vpp_id in vpp_ids
            ]
            portfolio_components = [
                infos[f"{vpp_id}_portfolio"].get("agent_reward_components", {})
                for vpp_id in vpp_ids
            ]
            raw_dispatch_rewards = np.asarray([float(reward_map[f"{vpp_id}_dispatch"]) for vpp_id in vpp_ids], dtype=np.float32)
            dso_attr_penalty = float(
                reward_components.get("dso_responsible_projection_penalty", shield_penalty)
            )
            dispatch_attr_penalties = np.asarray(
                [
                    float(component.get("dispatch_projection_penalty", shield_penalty))
                    for component in dispatch_components
                ],
                dtype=np.float32,
            )
            dso_reward = raw_dso_reward - float(cfg.dso_shield_intervention_penalty_coef) * dso_attr_penalty
            dispatch_rewards = raw_dispatch_rewards - float(cfg.dispatch_shield_intervention_penalty_coef) * dispatch_attr_penalties
            portfolio_rewards = np.asarray([float(reward_map[f"{vpp_id}_portfolio"]) for vpp_id in vpp_ids], dtype=np.float32)
            if not portfolio_decision_step:
                portfolio_rewards = np.zeros_like(portfolio_rewards)
            dispatch_profit_trace_rows.extend(
                dispatch_private_profit_trace_rows(
                    episode=int(episode),
                    step=int(step),
                    algorithm=cfg.algorithm,
                    vpp_ids=vpp_ids,
                    dispatch_components=dispatch_components,
                    raw_dispatch_rewards=raw_dispatch_rewards,
                    train_dispatch_rewards=dispatch_rewards,
                )
            )
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
            def component_mean(rows: list[dict[str, Any]], key: str) -> float:
                values = [float(row.get(key, 0.0)) for row in rows]
                return float(np.mean(values)) if values else 0.0

            def percentile(values: np.ndarray, q: float) -> float:
                return float(np.percentile(values.astype(float), q)) if values.size else 0.0

            step_rows.append(
                {
                    "episode": int(episode),
                    "step": int(step),
                    "algorithm": cfg.algorithm,
                    "reward": learning_reward,
                    "dso_reward": dso_reward,
                    "dso_reward_env": raw_dso_reward,
                    "dso_reward_train": dso_reward,
                    "dso_reward_critic_scaled": dso_reward * float(cfg.reward_scale),
                    "mean_dispatch_reward": float(dispatch_rewards.mean()) if dispatch_rewards.size else 0.0,
                    "mean_dispatch_reward_env": float(raw_dispatch_rewards.mean()) if raw_dispatch_rewards.size else 0.0,
                    "mean_dispatch_reward_train": float(dispatch_rewards.mean()) if dispatch_rewards.size else 0.0,
                    "min_dispatch_reward_train": float(dispatch_rewards.min()) if dispatch_rewards.size else 0.0,
                    "p05_dispatch_reward_train": percentile(dispatch_rewards, 5),
                    "p95_dispatch_reward_train": percentile(dispatch_rewards, 95),
                    "mean_portfolio_reward": float(portfolio_rewards.mean()) if portfolio_rewards.size else 0.0,
                    "mean_portfolio_reward_env": float(np.asarray([float(reward_map[f"{vpp_id}_portfolio"]) for vpp_id in vpp_ids], dtype=np.float32).mean()) if vpp_ids else 0.0,
                    "mean_portfolio_reward_train": float(portfolio_rewards.mean()) if portfolio_rewards.size else 0.0,
                    "portfolio_decision_step": bool(portfolio_decision_step),
                    "projection_gap_mw": float(projection_gap),
                    "decoded_projection_gap_mw": float(decoded_projection_gap),
                    "action_projection_gap_mw": float(shield_metrics["action_projection_gap_mw"]),
                    "local_bounds_projection_gap_mw": float(shield_metrics["local_bounds_projection_gap_mw"]),
                    "ac_aware_projection_gap_mw": float(shield_metrics["ac_aware_projection_gap_mw"]),
                    "ac_certified_projection_gap_mw": float(shield_metrics["ac_certified_projection_gap_mw"]),
                    "raw_action_violation_rate": float(shield_metrics["shield_intervention_count"] > 0.0),
                    "certificate_repair_rate": float(shield_metrics["ac_certified_projection_gap_mw"] > 1e-9),
                    "shield_intervention_gap_mw": float(shield_metrics["shield_intervention_gap_mw"]),
                    "shield_intervention_penalty": shield_penalty,
                    "shield_intervention_count": int(shield_metrics["shield_intervention_count"] > 0.0),
                    "raw_dso_reward_before_shield_penalty": raw_dso_reward,
                    "raw_dispatch_reward_before_shield_penalty": float(raw_dispatch_rewards.mean()) if raw_dispatch_rewards.size else 0.0,
                    "raw_portfolio_reward_before_decision_mask": component_mean(portfolio_components, "vpp_portfolio_reward_env"),
                    "private_profit_proxy": component_mean(dispatch_components, "private_profit_proxy"),
                    "vpp_operational_surplus_ex_transfer": component_mean(dispatch_components, "vpp_operational_surplus_ex_transfer"),
                    "economic_operational_surplus": component_mean(dispatch_components, "economic_operational_surplus"),
                    "quality_adjusted_operational_surplus": component_mean(dispatch_components, "quality_adjusted_operational_surplus"),
                    "service_quality_penalty_total": component_mean(dispatch_components, "service_quality_penalty_total"),
                    "settlement_audit_complete": component_mean(dispatch_components, "settlement_audit_complete"),
                    "settlement_power_balance_ok": component_mean(dispatch_components, "settlement_power_balance_ok"),
                    "settlement_power_balance_error_mw": component_mean(dispatch_components, "settlement_power_balance_error_mw"),
                    "storage_potential_shaping_reward": component_mean(dispatch_components, "storage_potential_shaping_reward"),
                    "storage_potential_raw": component_mean(dispatch_components, "storage_potential_raw"),
                    "storage_value_spread_per_mwh": component_mean(dispatch_components, "storage_value_spread_per_mwh"),
                    "storage_charge_mwh": component_mean(dispatch_components, "storage_charge_mwh"),
                    "storage_discharge_mwh": component_mean(dispatch_components, "storage_discharge_mwh"),
                    "storage_anti_hoarding_pass": component_mean(dispatch_components, "storage_anti_hoarding_pass"),
                    "energy_market_revenue": component_mean(dispatch_components, "energy_market_revenue"),
                    "visible_energy_minus_operation_cost": component_mean(
                        dispatch_components,
                        "visible_energy_minus_operation_cost",
                    ),
                    "market_energy_margin_total": component_mean(dispatch_components, "market_energy_margin_total"),
                    "baseline_p_mw": component_mean(dispatch_components, "baseline_p_mw"),
                    "raw_action_norm": component_mean(dispatch_components, "raw_action_norm"),
                    "raw_target_p_mw": component_mean(dispatch_components, "raw_target_p_mw"),
                    "decoded_target_p_mw": component_mean(dispatch_components, "decoded_target_p_mw"),
                    "device_feasible_target_p_mw": component_mean(dispatch_components, "device_feasible_target_p_mw"),
                    "pre_ac_target_p_mw": component_mean(dispatch_components, "pre_ac_target_p_mw"),
                    "ac_projected_target_p_mw": component_mean(dispatch_components, "ac_projected_target_p_mw"),
                    "ac_certified_target_p_mw": component_mean(dispatch_components, "ac_certified_target_p_mw"),
                    "actual_target_p_mw": component_mean(dispatch_components, "actual_target_p_mw"),
                    "raw_delta_p_mw": component_mean(dispatch_components, "raw_delta_p_mw"),
                    "decoded_delta_p_mw": component_mean(dispatch_components, "decoded_delta_p_mw"),
                    "device_feasible_delta_p_mw": component_mean(dispatch_components, "device_feasible_delta_p_mw"),
                    "pre_ac_delta_p_mw": component_mean(dispatch_components, "pre_ac_delta_p_mw"),
                    "ac_projected_delta_p_mw": component_mean(dispatch_components, "ac_projected_delta_p_mw"),
                    "ac_certified_delta_p_mw": component_mean(dispatch_components, "ac_certified_delta_p_mw"),
                    "raw_to_device_gap_mw": component_mean(dispatch_components, "raw_to_device_gap_mw"),
                    "device_to_ac_gap_mw": component_mean(dispatch_components, "device_to_ac_gap_mw"),
                    "ac_to_actual_gap_mw": component_mean(dispatch_components, "ac_to_actual_gap_mw"),
                    "accepted_to_actual_gap_mw": component_mean(dispatch_components, "accepted_to_actual_gap_mw"),
                    "actual_delta_nonzero_flag": component_mean(dispatch_components, "actual_delta_nonzero_flag"),
                    "action_landing_ratio": component_mean(dispatch_components, "action_landing_ratio"),
                    "action_landing_drop_reason_code": component_mean(dispatch_components, "action_landing_drop_reason_code"),
                    "requested_delta_p_mw": component_mean(dispatch_components, "requested_delta_p_mw"),
                    "accepted_delta_p_mw": component_mean(dispatch_components, "accepted_delta_p_mw"),
                    "actual_delta_p_mw": component_mean(dispatch_components, "actual_delta_p_mw"),
                    "verified_delivery_mw": component_mean(dispatch_components, "verified_delivery_mw"),
                    "contract_shortfall_mw": component_mean(dispatch_components, "contract_shortfall_mw"),
                    "contract_delivery_penalty": component_mean(dispatch_components, "contract_delivery_penalty"),
                    "availability_payment": component_mean(dispatch_components, "availability_payment"),
                    "service_payment": component_mean(dispatch_components, "service_payment"),
                    "export_revenue_total": component_mean(dispatch_components, "export_revenue_total"),
                    "pv_export_revenue_total": component_mean(dispatch_components, "pv_export_revenue_total"),
                    "mt_export_revenue_total": component_mean(dispatch_components, "mt_export_revenue_total"),
                    "storage_discharge_revenue_total": component_mean(dispatch_components, "storage_discharge_revenue_total"),
                    "evcs_user_revenue_total": component_mean(dispatch_components, "evcs_user_revenue_total"),
                    "import_energy_cost_total": component_mean(dispatch_components, "import_energy_cost_total"),
                    "evcs_wholesale_cost_total": component_mean(dispatch_components, "evcs_wholesale_cost_total"),
                    "storage_charge_cost_total": component_mean(dispatch_components, "storage_charge_cost_total"),
                    "hvac_energy_cost_total": component_mean(dispatch_components, "hvac_energy_cost_total"),
                    "flex_energy_cost_total": component_mean(dispatch_components, "flex_energy_cost_total"),
                    "unclassified_import_cost_total": component_mean(dispatch_components, "unclassified_import_cost_total"),
                    "der_operating_cost_total": component_mean(dispatch_components, "der_operating_cost_total"),
                    "battery_degradation_cost_total": component_mean(dispatch_components, "battery_degradation_cost_total"),
                    "comfort_cost_total": component_mean(dispatch_components, "comfort_cost_total"),
                    "unserved_penalty_total": component_mean(dispatch_components, "unserved_penalty_total"),
                    "legacy_operational_surplus_with_service_quality": component_mean(
                        dispatch_components,
                        "legacy_operational_surplus_with_service_quality",
                    ),
                    "der_operation_cost": component_mean(dispatch_components, "der_operation_cost"),
                    "battery_degradation_cost": component_mean(dispatch_components, "battery_degradation_cost"),
                    "comfort_penalty": component_mean(dispatch_components, "comfort_penalty"),
                    "soc_penalty": component_mean(dispatch_components, "soc_penalty"),
                    "dispatch_responsible_projection_gap_mw": component_mean(dispatch_components, "dispatch_responsible_projection_gap_mw"),
                    "dispatch_projection_penalty": component_mean(dispatch_components, "dispatch_projection_penalty"),
                    "portfolio_window_profit": component_mean(portfolio_components, "portfolio_window_profit"),
                    "portfolio_window_contract_shortfall": component_mean(portfolio_components, "portfolio_window_contract_shortfall"),
                    "portfolio_window_shield_intervention": component_mean(portfolio_components, "portfolio_window_shield_intervention"),
                    "portfolio_window_projection_gap": component_mean(portfolio_components, "portfolio_window_projection_gap"),
                    "portfolio_window_comfort_soc_penalty": component_mean(portfolio_components, "portfolio_window_comfort_soc_penalty"),
                    "portfolio_window_verified_capacity": component_mean(portfolio_components, "portfolio_window_verified_capacity"),
                    "portfolio_switching_cost": component_mean(portfolio_components, "portfolio_switching_cost"),
                    "portfolio_action_type": component_mean(portfolio_components, "portfolio_action_type_code"),
                    "dso_safety_margin_penalty": float(reward_components.get("dso_safety_margin_penalty", 0.0)),
                    "dso_voltage_guard_penalty": float(reward_components.get("dso_voltage_guard_penalty", 0.0)),
                    "dso_line_guard_penalty": float(reward_components.get("dso_line_guard_penalty", 0.0)),
                    "dso_trafo_guard_penalty": float(reward_components.get("dso_trafo_guard_penalty", 0.0)),
                    "dso_powerflow_failure_penalty": float(reward_components.get("dso_powerflow_failure_penalty", 0.0)),
                    "dso_flex_procurement_cost": float(reward_components.get("dso_flex_procurement_cost", 0.0)),
                    "dso_loss_cost": float(reward_components.get("dso_loss_cost", 0.0)),
                    "dso_curtailment_cost": float(reward_components.get("dso_curtailment_cost", 0.0)),
                    "dso_safe_capacity_utilization_reward": float(reward_components.get("dso_safe_capacity_utilization_reward", 0.0)),
                    "dso_over_conservative_curtailment_penalty": float(reward_components.get("dso_over_conservative_curtailment_penalty", 0.0)),
                    "dso_responsible_projection_gap_mw": float(reward_components.get("dso_responsible_projection_gap_mw", 0.0)),
                    "dso_responsible_projection_penalty": float(reward_components.get("dso_responsible_projection_penalty", 0.0)),
                    "dso_safety_gate": float(reward_components.get("dso_safety_gate", 1.0)),
                    "dso_safety_gate_input": float(reward_components.get("dso_safety_gate_input", 0.0)),
                    "dso_vpp_welfare_raw": float(reward_components.get("dso_vpp_welfare_raw", 0.0)),
                    "dso_vpp_welfare_normalized": float(reward_components.get("dso_vpp_welfare_normalized", 0.0)),
                    "raw_action_safety_cost_norm": float(reward_components.get("raw_action_safety_cost_norm", 0.0)),
                    "projected_action_safety_cost_norm": float(reward_components.get("projected_action_safety_cost_norm", 0.0)),
                    "raw_action_safety_penalty_input": float(reward_components.get("raw_action_safety_penalty_input", 0.0)),
                    "dso_raw_action_safety_penalty": float(reward_components.get("dso_raw_action_safety_penalty", 0.0)),
                    "dso_projected_action_safety_penalty": float(reward_components.get("dso_projected_action_safety_penalty", 0.0)),
                    "cmdp_cost_voltage": float(reward_components.get("cmdp_cost_voltage", 0.0)),
                    "cmdp_cost_line_overload": float(reward_components.get("cmdp_cost_line_overload", 0.0)),
                    "cmdp_cost_trafo_overload": float(reward_components.get("cmdp_cost_trafo_overload", 0.0)),
                    "cmdp_cost_powerflow_failed": float(reward_components.get("cmdp_cost_powerflow_failed", 0.0)),
                    "tracking_bonus_diagnostic": float(reward_components.get("tracking_bonus_diagnostic", reward_components.get("tracking_bonus", 0.0))),
                    "effective_response_bonus_diagnostic": float(reward_components.get("effective_response_bonus_diagnostic", reward_components.get("effective_response_bonus", 0.0))),
                    "target_tracking_error_to_raw_target": float(reward_components.get("target_tracking_error_to_raw_target", 0.0)),
                    "target_tracking_error_to_projected_target": float(reward_components.get("target_tracking_error_to_projected_target", 0.0)),
                    "post_ac_violation_count": float(reward_components.get("post_ac_violation_count", len(violations))),
                    "post_ac_security_penalty": float(reward_components.get("post_ac_security_penalty", 0.0)),
                    "total_cost": float(reward_components.get("total_cost", -dso_reward)),
                    "violation_count": int(len(violations)),
                }
            )

            observations = next_observations
            maybe_report_train_step(step)
            if all(truncations.values()):
                break

        update_started = time.perf_counter()
        critic_state_tensor = torch.as_tensor(np.asarray(rollout["critic_state"]), dtype=torch.float32, device=device)
        action_summary_tensor = torch.as_tensor(np.asarray(rollout["action_summary"]), dtype=torch.float32, device=device)
        value_action_tensor = (
            action_summary_tensor
            if bool(cfg.critic_use_action_summary)
            else torch.zeros((critic_state_tensor.shape[0], value_action_dim), dtype=torch.float32, device=device)
        )
        value_matrix = value_critic(critic_state_tensor, value_action_tensor)
        rewards_matrix = np.asarray(rollout["rewards"], dtype=np.float32)
        returns_by_head: list[Any] = []
        advantages_by_head: list[Any] = []
        for head_index in range(rewards_matrix.shape[1]):
            if shared_rollout_enabled:
                worker_index_array = np.asarray(rollout["worker_index"], dtype=np.int64)
                terminal_array = np.asarray(rollout["terminal"], dtype=bool)
                head_returns_parts: list[Any] = []
                head_advantages_parts: list[Any] = []
                for worker_index in sorted(set(worker_index_array.tolist())):
                    mask = worker_index_array == int(worker_index)
                    indices = np.flatnonzero(mask)
                    bootstrap_values = rollout["bootstrap_values_by_worker"].get(int(worker_index))
                    next_value = (
                        float(np.asarray(bootstrap_values, dtype=np.float32).reshape(-1)[head_index])
                        if bootstrap_values is not None
                        else 0.0
                    )
                    worker_returns, worker_advantages = _gae_returns_advantages_bootstrap(
                        rewards=rewards_matrix[indices, head_index].tolist(),
                        values=value_matrix[torch.as_tensor(indices, dtype=torch.long, device=device), head_index],
                        next_value=torch.tensor(next_value, dtype=torch.float32, device=device),
                        terminals=terminal_array[indices].tolist(),
                        gamma=cfg.gamma,
                        gae_lambda=cfg.gae_lambda,
                        torch=torch,
                    )
                    head_returns_parts.append(worker_returns)
                    head_advantages_parts.append(worker_advantages)
                head_returns = torch.cat(head_returns_parts, dim=0)
                head_advantages = torch.cat(head_advantages_parts, dim=0)
            else:
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
        if cfg.nan_guard:
            _nan_guard_tensors(torch, "happo_critic_update", value_matrix, returns_tensor, critic_loss)
        critic_optimizer.zero_grad()
        critic_loss.backward()
        critic_grad_norm = float(torch.nn.utils.clip_grad_norm_(value_critic.parameters(), cfg.max_grad_norm))
        critic_optimizer.step()

        dso_obs_tensor = torch.as_tensor(np.asarray(rollout["dso_obs"]), dtype=torch.float32, device=device)
        dso_raw_action_tensor = torch.as_tensor(np.asarray(rollout["dso_raw_action"]), dtype=torch.float32, device=device)
        dso_old_log_prob_tensor = torch.as_tensor(np.asarray(rollout["dso_log_prob"]), dtype=torch.float32, device=device)
        dispatch_obs_tensor = torch.as_tensor(np.asarray(rollout["dispatch_obs"]), dtype=torch.float32, device=device)
        dispatch_raw_aggregate_tensor = torch.as_tensor(
            np.asarray(rollout["dispatch_raw_aggregate"]),
            dtype=torch.float32,
            device=device,
        )
        dispatch_raw_der_tensor = torch.as_tensor(np.asarray(rollout["dispatch_raw_der"]), dtype=torch.float32, device=device)
        dispatch_old_log_prob_tensor = torch.as_tensor(
            np.asarray(rollout["dispatch_log_prob"]),
            dtype=torch.float32,
            device=device,
        )
        dispatch_der_count_tensor = torch.as_tensor(np.asarray(rollout["dispatch_der_count"]), dtype=torch.int64, device=device)
        portfolio_obs_tensor = torch.as_tensor(np.asarray(rollout["portfolio_obs"]), dtype=torch.float32, device=device)
        portfolio_action_idx_tensor = torch.as_tensor(np.asarray(rollout["portfolio_action_idx"]), dtype=torch.int64, device=device)
        portfolio_old_log_prob_tensor = torch.as_tensor(
            np.asarray(rollout["portfolio_log_prob"]),
            dtype=torch.float32,
            device=device,
        )
        portfolio_update_mask_tensor = torch.as_tensor(
            np.asarray(rollout["portfolio_update_mask"]),
            dtype=torch.bool,
            device=device,
        )
        correction = torch.ones(dso_old_log_prob_tensor.shape[0], dtype=torch.float32, device=device)

        if shared_rollout_enabled:
            with torch.no_grad():
                old_log_prob_parts = [dso_old_log_prob_tensor.reshape(-1)]
                new_log_prob_parts: list[Any] = []
                ratio_parts: list[Any] = []
                initial_dso_log_prob, _ = _dso_gaussian_stats(
                    actor_modules["dso_actor"],
                    dso_obs_tensor,
                    dso_raw_action_tensor,
                    Normal,
                )
                new_log_prob_parts.append(initial_dso_log_prob.reshape(-1))
                ratio_parts.append(torch.exp(initial_dso_log_prob.reshape(-1) - dso_old_log_prob_tensor.reshape(-1)))
                if cfg.share_vpp_dispatch_parameters:
                    flat_dispatch_obs = dispatch_obs_tensor.reshape(-1, dispatch_obs_tensor.shape[-1])
                    flat_dispatch_raw_aggregate = dispatch_raw_aggregate_tensor.reshape(-1, 1)
                    flat_dispatch_raw_der = dispatch_raw_der_tensor.reshape(-1, dispatch_raw_der_tensor.shape[-1])
                    flat_dispatch_counts = dispatch_der_count_tensor.reshape(-1)
                    flat_dispatch_old_log_prob = dispatch_old_log_prob_tensor.reshape(-1)
                    initial_dispatch_log_prob, _ = _dispatch_gaussian_stats(
                        actor_modules["vpp_dispatch_actor"],
                        flat_dispatch_obs,
                        flat_dispatch_raw_aggregate,
                        flat_dispatch_raw_der,
                        flat_dispatch_counts,
                        Normal,
                        torch,
                    )
                    old_log_prob_parts.append(flat_dispatch_old_log_prob)
                    new_log_prob_parts.append(initial_dispatch_log_prob.reshape(-1))
                    ratio_parts.append(torch.exp(initial_dispatch_log_prob.reshape(-1) - flat_dispatch_old_log_prob))
                else:
                    for vpp_index, vpp_id in enumerate(vpp_ids):
                        initial_dispatch_log_prob, _ = _dispatch_gaussian_stats(
                            actor_modules[f"{vpp_id}_dispatch_actor"],
                            dispatch_obs_tensor[:, vpp_index, :],
                            dispatch_raw_aggregate_tensor[:, vpp_index, :],
                            dispatch_raw_der_tensor[:, vpp_index, :],
                            dispatch_der_count_tensor[:, vpp_index],
                            Normal,
                            torch,
                        )
                        old_dispatch = dispatch_old_log_prob_tensor[:, vpp_index].reshape(-1)
                        old_log_prob_parts.append(old_dispatch)
                        new_log_prob_parts.append(initial_dispatch_log_prob.reshape(-1))
                        ratio_parts.append(torch.exp(initial_dispatch_log_prob.reshape(-1) - old_dispatch))
                if cfg.share_vpp_portfolio_parameters:
                    flat_portfolio_mask = portfolio_update_mask_tensor.reshape(-1)
                    if bool(flat_portfolio_mask.any()):
                        flat_portfolio_obs = portfolio_obs_tensor.reshape(-1, portfolio_obs_tensor.shape[-1])[flat_portfolio_mask]
                        flat_portfolio_action_idx = portfolio_action_idx_tensor.reshape(-1)[flat_portfolio_mask]
                        flat_portfolio_old_log_prob = portfolio_old_log_prob_tensor.reshape(-1)[flat_portfolio_mask]
                        initial_portfolio_log_prob, _ = _portfolio_categorical_stats(
                            actor_modules["vpp_portfolio_actor"],
                            flat_portfolio_obs,
                            flat_portfolio_action_idx,
                            Categorical,
                        )
                        old_log_prob_parts.append(flat_portfolio_old_log_prob)
                        new_log_prob_parts.append(initial_portfolio_log_prob.reshape(-1))
                        ratio_parts.append(torch.exp(initial_portfolio_log_prob.reshape(-1) - flat_portfolio_old_log_prob))
                else:
                    for vpp_index, vpp_id in enumerate(vpp_ids):
                        mask = portfolio_update_mask_tensor[:, vpp_index]
                        if not bool(mask.any()):
                            continue
                        initial_portfolio_log_prob, _ = _portfolio_categorical_stats(
                            actor_modules[f"{vpp_id}_portfolio_actor"],
                            portfolio_obs_tensor[:, vpp_index, :][mask],
                            portfolio_action_idx_tensor[:, vpp_index][mask],
                            Categorical,
                        )
                        old_portfolio = portfolio_old_log_prob_tensor[:, vpp_index][mask].reshape(-1)
                        old_log_prob_parts.append(old_portfolio)
                        new_log_prob_parts.append(initial_portfolio_log_prob.reshape(-1))
                        ratio_parts.append(torch.exp(initial_portfolio_log_prob.reshape(-1) - old_portfolio))
                old_log_prob_all = torch.cat(old_log_prob_parts) if old_log_prob_parts else torch.zeros(1, device=device)
                new_log_prob_all = torch.cat(new_log_prob_parts) if new_log_prob_parts else torch.zeros(1, device=device)
                ratio_all = torch.cat(ratio_parts) if ratio_parts else torch.ones(1, device=device)
                perf_payload = shared_rollout_perf_by_episode.setdefault(int(episode), {})
                perf_payload.update(
                    {
                        "ratio_mean_before_first_update": float(ratio_all.mean().detach().cpu().item()),
                        "ratio_std_before_first_update": float(ratio_all.std(unbiased=False).detach().cpu().item()),
                        "old_log_prob_nan_count": int((~torch.isfinite(old_log_prob_all)).sum().detach().cpu().item()),
                        "new_log_prob_nan_count": int((~torch.isfinite(new_log_prob_all)).sum().detach().cpu().item()),
                        "advantage_mean": float(advantages_tensor.mean().detach().cpu().item()),
                        "advantage_std": float(advantages_tensor.std(unbiased=False).detach().cpu().item()),
                        "return_mean": float(returns_tensor.mean().detach().cpu().item()),
                        "return_std": float(returns_tensor.std(unbiased=False).detach().cpu().item()),
                    }
                )

        dso_policy_loss_value = 0.0
        dispatch_policy_loss_value = 0.0
        portfolio_policy_loss_value = 0.0
        dso_correction_mean = 1.0
        dispatch_correction_mean = 1.0
        portfolio_correction_mean = 1.0

        def target_kl_exceeded(approx_kl: float) -> bool:
            return cfg.target_kl is not None and abs(float(approx_kl)) > float(cfg.target_kl)

        for epoch in range(cfg.ppo_epochs):
            epoch_target_kl_exceeded = False
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
                normalize_advantages=cfg.normalize_advantages,
            )
            if cfg.nan_guard:
                _nan_guard_tensors(torch, "happo_dso_update", dso_log_prob, dso_entropy, dso_loss)
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
                dso_approx_kl = float((dso_old_log_prob_tensor - updated_dso_log_prob).mean().cpu().item())
            dso_target_kl_exceeded = target_kl_exceeded(dso_approx_kl)
            epoch_target_kl_exceeded = epoch_target_kl_exceeded or dso_target_kl_exceeded
            update_rows.append(
                {
                    "episode": int(episode),
                    "epoch": int(epoch),
                    "role": "dso_global_guidance",
                    "policy_loss": dso_policy_loss_value,
                    "ratio_mean": float(dso_ratios.detach().mean().cpu().item()),
                    "entropy_mean": float(dso_entropy.detach().mean().cpu().item()),
                    "approx_kl": dso_approx_kl,
                    "target_kl": None if cfg.target_kl is None else float(cfg.target_kl),
                    "target_kl_exceeded": bool(dso_target_kl_exceeded),
                    "nan_guard_triggered": False,
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
                    normalize_advantages=cfg.normalize_advantages,
                )
                if cfg.nan_guard:
                    _nan_guard_tensors(torch, "happo_shared_dispatch_update", dispatch_log_prob, dispatch_entropy, dispatch_loss)
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
                    dispatch_approx_kl = float(
                        (flat_dispatch_old_log_prob - updated_dispatch_log_prob).mean().cpu().item()
                    )
                dispatch_target_kl_exceeded = target_kl_exceeded(dispatch_approx_kl)
                epoch_target_kl_exceeded = epoch_target_kl_exceeded or dispatch_target_kl_exceeded
                update_rows.append(
                    {
                        "episode": int(episode),
                        "epoch": int(epoch),
                        "role": "shared_vpp_dispatch",
                        "policy_loss": dispatch_policy_loss_value,
                        "ratio_mean": float(dispatch_ratios.detach().mean().cpu().item()),
                        "entropy_mean": float(dispatch_entropy.detach().mean().cpu().item()),
                        "approx_kl": dispatch_approx_kl,
                        "target_kl": None if cfg.target_kl is None else float(cfg.target_kl),
                        "target_kl_exceeded": bool(dispatch_target_kl_exceeded),
                        "nan_guard_triggered": False,
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
                        normalize_advantages=cfg.normalize_advantages,
                    )
                    if cfg.nan_guard:
                        _nan_guard_tensors(torch, f"happo_{vpp_id}_dispatch_update", dispatch_log_prob, dispatch_entropy, dispatch_loss)
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
                        dispatch_approx_kl = float(
                            (dispatch_old_log_prob_tensor[:, vpp_index] - updated_dispatch_log_prob).mean().cpu().item()
                        )
                    dispatch_target_kl_exceeded = target_kl_exceeded(dispatch_approx_kl)
                    epoch_target_kl_exceeded = epoch_target_kl_exceeded or dispatch_target_kl_exceeded
                    update_rows.append(
                        {
                            "episode": int(episode),
                            "epoch": int(epoch),
                            "role": f"{vpp_id}_dispatch",
                            "target_vpp_id": vpp_id,
                            "policy_loss": dispatch_policy_loss_value,
                            "ratio_mean": float(dispatch_ratios.detach().mean().cpu().item()),
                            "entropy_mean": float(dispatch_entropy.detach().mean().cpu().item()),
                            "approx_kl": dispatch_approx_kl,
                            "target_kl": None if cfg.target_kl is None else float(cfg.target_kl),
                            "target_kl_exceeded": bool(dispatch_target_kl_exceeded),
                            "nan_guard_triggered": False,
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
                        normalize_advantages=cfg.normalize_advantages,
                    )
                    if cfg.nan_guard:
                        _nan_guard_tensors(torch, "happo_shared_portfolio_update", portfolio_log_prob, portfolio_entropy, portfolio_loss)
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
                        portfolio_approx_kl = float(
                            (
                                flat_portfolio_old_log_prob
                                - updated_portfolio_log_prob.reshape(-1)[flat_portfolio_mask]
                            ).mean().cpu().item()
                        )
                    portfolio_target_kl_exceeded = target_kl_exceeded(portfolio_approx_kl)
                    epoch_target_kl_exceeded = epoch_target_kl_exceeded or portfolio_target_kl_exceeded
                    update_rows.append(
                        {
                            "episode": int(episode),
                            "epoch": int(epoch),
                            "role": "shared_vpp_portfolio",
                            "policy_loss": portfolio_policy_loss_value,
                            "ratio_mean": float(portfolio_ratios.detach().mean().cpu().item()),
                            "entropy_mean": float(portfolio_entropy.detach().mean().cpu().item()),
                            "approx_kl": portfolio_approx_kl,
                            "target_kl": None if cfg.target_kl is None else float(cfg.target_kl),
                            "target_kl_exceeded": bool(portfolio_target_kl_exceeded),
                            "nan_guard_triggered": False,
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
                        normalize_advantages=cfg.normalize_advantages,
                    )
                    if cfg.nan_guard:
                        _nan_guard_tensors(torch, f"happo_{vpp_id}_portfolio_update", portfolio_log_prob, portfolio_entropy, portfolio_loss)
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
                        portfolio_approx_kl = float(
                            (
                                portfolio_old_log_prob_tensor[:, vpp_index][mask]
                                - updated_portfolio_log_prob
                            ).mean().cpu().item()
                        )
                    portfolio_target_kl_exceeded = target_kl_exceeded(portfolio_approx_kl)
                    epoch_target_kl_exceeded = epoch_target_kl_exceeded or portfolio_target_kl_exceeded
                    update_rows.append(
                        {
                            "episode": int(episode),
                            "epoch": int(epoch),
                            "role": f"{vpp_id}_portfolio",
                            "target_vpp_id": vpp_id,
                            "policy_loss": portfolio_policy_loss_value,
                            "ratio_mean": float(portfolio_ratios.detach().mean().cpu().item()),
                            "entropy_mean": float(portfolio_entropy.detach().mean().cpu().item()),
                            "approx_kl": portfolio_approx_kl,
                            "target_kl": None if cfg.target_kl is None else float(cfg.target_kl),
                            "target_kl_exceeded": bool(portfolio_target_kl_exceeded),
                            "nan_guard_triggered": False,
                            "correction_mean": portfolio_correction_mean,
                            "grad_norm": portfolio_grad_norm,
                            "slow_loop_updates": int(mask.sum().cpu().item()),
                        }
                    )
                if portfolio_losses_for_episode:
                    portfolio_policy_loss_value = float(np.mean(portfolio_losses_for_episode))

            if epoch_target_kl_exceeded:
                kl_early_stop_count += 1
                break

        update_seconds = time.perf_counter() - update_started
        if shared_rollout_enabled:
            perf_payload = shared_rollout_perf_by_episode.setdefault(int(episode), {})
            perf_payload["update_seconds"] = float(update_seconds)
            perf_payload["total_update_seconds"] = float(
                update_seconds + float(perf_payload.get("rollout_collect_seconds", 0.0))
            )

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
        if progress_callback is not None:
            progress_callback(
                {
                    "phase": "train_update",
                    "message": "HAPPO training update completed",
                    "episode": int(episode) + 1,
                    "episodes": int(cfg.episodes),
                    "step": int(cfg.horizon_steps),
                    "horizon_steps": int(cfg.horizon_steps),
                    "global_step": int((int(episode) + 1) * int(cfg.horizon_steps)),
                    "gradient_step": int(episode) + 1,
                    "episode_reward": float(episode_reward),
                    "episode_cost": float(total_cost),
                    "violation_count": int(violation_count),
                    "projection_gap_mw": float(projection_gap_total),
                    "critic_loss": float(critic_loss.detach().cpu().item()),
                    "critic_grad_norm": critic_grad_norm,
                    "dso_policy_loss": dso_policy_loss_value,
                    "dispatch_policy_loss": dispatch_policy_loss_value,
                    "portfolio_policy_loss": portfolio_policy_loss_value,
                }
            )
        if float(episode_reward) > best_episode_reward:
            best_episode_reward = float(episode_reward)
            best_episode_index = int(episode)
            best_checkpoint_state = {
                "actor_state_dict": _state_dict_to_cpu(actor_modules.state_dict()),
                "value_critic_state_dict": _state_dict_to_cpu(value_critic.state_dict()),
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
        episode_trace = pd.DataFrame(
            [row for row in dispatch_profit_trace_rows if int(row.get("episode", -1)) == int(episode)]
        )
        episode_trace.to_csv(
            out / f"happo_dispatch_private_profit_trace_episode_{int(episode):04d}.csv",
            index=False,
        )
        pd.DataFrame(dispatch_profit_trace_rows).to_csv(
            out / "happo_dispatch_private_profit_trace.csv",
            index=False,
        )
        report_every = max(1, int(cfg.reward_dynamic_report_every_episodes))
        should_write_reward_report = bool(cfg.reward_dynamic_reports) and (
            int(episode) == 0 or int(episode) == int(cfg.episodes) - 1 or int(episode) % report_every == 0
        )
        if should_write_reward_report:
            from vpp_dso_sim.visualization.reward_dynamic_report import write_reward_dynamic_episode_report

            episode_step_metrics = pd.DataFrame(
                [row for row in step_rows if int(row.get("episode", -1)) == int(episode)]
            )
            report_trace = episode_trace
            if (
                bool(shared_rollout_enabled)
                and not bool(cfg.reward_dynamic_report_all_workers)
                and "worker_index" in episode_step_metrics.columns
            ):
                episode_step_metrics = episode_step_metrics[episode_step_metrics["worker_index"].astype(int) == 0].copy()
                if "worker_index" in report_trace.columns:
                    report_trace = report_trace[report_trace["worker_index"].astype(int) == 0].copy()

            write_reward_dynamic_episode_report(
                output_dir=out / "reports" / "reward_dynamic_cards",
                algorithm=cfg.algorithm,
                episode=int(episode),
                step_metrics=episode_step_metrics,
                episode_metrics=pd.DataFrame([episode_rows[-1]]) if episode_rows else pd.DataFrame(),
                dispatch_trace=report_trace,
                update_metrics=pd.DataFrame(
                    [row for row in update_rows if int(row.get("episode", -1)) == int(episode)]
                ),
            )
        if env is not None:
            env.close()

    shared_rollout_subprocess_worker_exitcodes: dict[int, int | None] = {}
    if shared_rollout_enabled and shared_rollout_backend == "subprocess":
        for worker_index, worker in shared_rollout_subprocess_workers.items():
            worker.close()
            shared_rollout_subprocess_worker_exitcodes[int(worker_index)] = worker.exitcode
    else:
        for worker_state in shared_worker_states.values():
            if "env" in worker_state:
                worker_state["env"].close()

    episode_metrics = pd.DataFrame(episode_rows)
    step_metrics = pd.DataFrame(step_rows)
    dispatch_private_profit_trace = pd.DataFrame(dispatch_profit_trace_rows)
    update_metrics = pd.DataFrame(update_rows)
    if shared_rollout_enabled and not update_metrics.empty:
        update_metrics["policy_version"] = update_metrics["episode"].map(shared_rollout_policy_versions)
        update_metrics["worker_count"] = update_metrics["episode"].map(shared_rollout_worker_counts).fillna(shared_rollout_workers).astype(int)
        update_metrics["num_workers"] = update_metrics["worker_count"]
        update_metrics["rollout_fragment_steps"] = (
            update_metrics["episode"].map(shared_rollout_fragment_steps_by_episode).fillna(int(cfg.rollout_fragment_steps or cfg.horizon_steps)).astype(int)
        )
        update_metrics["shared_rollout_enabled"] = True
        update_metrics["bootstrap_value_mean"] = float(np.mean(shared_rollout_bootstrap_values)) if shared_rollout_bootstrap_values else 0.0
        update_metrics["fragment_cut_count"] = int(shared_rollout_fragment_cut_count)
        perf_fields = (
            "shared_rollout_backend",
            "num_workers",
            "rollout_collect_seconds",
            "policy_forward_seconds",
            "env_step_wall_seconds",
            "env_step_worker_mean_seconds",
            "env_step_worker_max_seconds",
            "wait_for_workers_seconds",
            "update_seconds",
            "total_update_seconds",
            "samples_collected",
            "samples_per_second",
            "slowest_worker_id",
            "ratio_mean_before_first_update",
            "ratio_std_before_first_update",
            "old_log_prob_nan_count",
            "new_log_prob_nan_count",
            "advantage_mean",
            "advantage_std",
            "return_mean",
            "return_std",
        )
        for field in perf_fields:
            update_metrics[field] = update_metrics["episode"].map(
                {
                    int(episode_index): payload.get(field)
                    for episode_index, payload in shared_rollout_perf_by_episode.items()
                }
            )
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
    observation_normalization_stats = _summarize_observation_normalization_stats(observation_norm_stats)
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
            "config_path": config_path_text,
            "config_hash": config_hash,
            "device_meta": dict(device_meta),
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
            "reward_config": resolved_reward_config.to_dict(),
            "reward_config_hash": reward_artifacts["reward_config_hash"],
            "dso_actor_observation_mode": architecture_meta.get("dso_actor_observation_mode", "legacy_flat"),
            "dso_actor_type": architecture_meta.get("dso_actor_type", "legacy_mlp_gaussian"),
            "structured_dso_flat_spec": (
                structured_dso_spec.to_dict() if structured_dso_spec is not None else None
            ),
            "selection_metric": "episode_reward",
        }

    torch.save(
        checkpoint_payload(_state_dict_to_cpu(actor_modules.state_dict()), _state_dict_to_cpu(value_critic.state_dict())),
        checkpoint_path,
    )
    if best_checkpoint_state is None:
        best_checkpoint_state = {
            "actor_state_dict": _state_dict_to_cpu(actor_modules.state_dict()),
            "value_critic_state_dict": _state_dict_to_cpu(value_critic.state_dict()),
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
        "config_path": config_path_text,
        "config_hash": config_hash,
        "is_deep_rl": True,
        "deep_learning_framework": "torch",
        "requested_device": str(device_meta["requested_device"]),
        "resolved_device": str(device_meta["resolved_device"]),
        "cuda_available": bool(device_meta["cuda_available"]),
        "cuda_device_count": int(device_meta["cuda_device_count"]),
        "cuda_device_name": device_meta["cuda_device_name"],
        "seed": int(cfg.seed),
        "training_pattern": "on_policy_ctde",
        "shared_rollout_enabled": bool(shared_rollout_enabled),
        "shared_rollout_workers": int(shared_rollout_workers),
        "shared_rollout_backend": str(shared_rollout_backend),
        "rollout_fragment_steps": None if cfg.rollout_fragment_steps is None else int(cfg.rollout_fragment_steps),
        "shared_rollout_batches": int(shared_rollout_batches),
        "shared_rollout_total_samples": int(shared_rollout_total_samples if shared_rollout_enabled else len(step_metrics)),
        "shared_rollout_fragment_cut_count": int(shared_rollout_fragment_cut_count),
        "shared_rollout_worker_terminal_reset_count": int(shared_rollout_worker_terminal_reset_count),
        "shared_rollout_worker_start_offsets": {
            str(worker_index): int(start_step)
            for worker_index, start_step in sorted(shared_rollout_worker_start_offsets.items())
        },
        "shared_rollout_subprocess_worker_pids": {
            str(worker_index): int(pid)
            for worker_index, pid in sorted(shared_rollout_subprocess_worker_pids.items())
        },
        "shared_rollout_subprocess_worker_exitcodes": {
            str(worker_index): (None if exitcode is None else int(exitcode))
            for worker_index, exitcode in sorted(shared_rollout_subprocess_worker_exitcodes.items())
        },
        "shared_rollout_rollout_collect_seconds_mean": (
            float(np.mean([float(payload.get("rollout_collect_seconds", 0.0)) for payload in shared_rollout_perf_by_episode.values()]))
            if shared_rollout_perf_by_episode
            else 0.0
        ),
        "shared_rollout_samples_per_second_mean": (
            float(np.mean([float(payload.get("samples_per_second", 0.0)) for payload in shared_rollout_perf_by_episode.values()]))
            if shared_rollout_perf_by_episode
            else 0.0
        ),
        "shared_rollout_policy_version_mismatch_count": int(shared_rollout_policy_version_mismatch_count),
        "shared_rollout_bootstrap_value_mean": float(np.mean(shared_rollout_bootstrap_values)) if shared_rollout_bootstrap_values else 0.0,
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
        "target_kl": None if cfg.target_kl is None else float(cfg.target_kl),
        "kl_early_stop_count": int(kl_early_stop_count),
        "gae_lambda": float(cfg.gae_lambda),
        "normalize_observations": bool(cfg.normalize_observations),
        "normalize_advantages": bool(cfg.normalize_advantages),
        "nan_guard": bool(cfg.nan_guard),
        "nan_guard_trigger_count": 0,
        "observation_normalization_stats": observation_normalization_stats,
        "importance_correction_clip": float(cfg.importance_correction_clip),
        "importance_correction_total_clip": float(cfg.importance_correction_total_clip),
        "reward_scale": float(cfg.reward_scale),
        "reward_version": str(resolved_reward_config.version),
        "critic_reward_scale": float(resolved_reward_config.critic_reward_scale),
        "resolved_reward_config": reward_artifacts["resolved_reward_config"],
        "reward_config_hash": reward_artifacts["reward_config_hash"],
        "reward_config_hash_path": reward_artifacts["reward_config_hash_path"],
        "value_target_clip": None if cfg.value_target_clip is None else float(cfg.value_target_clip),
        "critic_use_action_summary": bool(cfg.critic_use_action_summary),
        "critic_baseline_type": "action_conditioned" if bool(cfg.critic_use_action_summary) else "state_only",
        "dso_input_dim": int(dso_input_dim),
        "dso_actor_observation_mode": architecture_meta.get("dso_actor_observation_mode", "legacy_flat"),
        "dso_actor_type": architecture_meta.get("dso_actor_type", "legacy_mlp_gaussian"),
        "dispatch_actor_encoder_type": architecture_meta.get("dispatch_actor_encoder_type", cfg.dispatch_actor_encoder_type),
        "vpp_encoder_type": architecture_meta.get("vpp_encoder_type", ""),
        "architecture_version": architecture_meta.get("architecture_version", ""),
        "structured_dso_actor_trainable": bool(use_structured_dso_actor),
        "structured_dso_flat_spec": structured_dso_spec.to_dict() if structured_dso_spec is not None else None,
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
    dispatch_private_profit_trace.to_csv(out / "happo_dispatch_private_profit_trace.csv", index=False)
    update_metrics.to_csv(out / "happo_update_metrics.csv", index=False)
    pd.DataFrame([summary]).to_csv(out / "happo_training_summary.csv", index=False)
    write_json(out / "happo_training_summary.json", summary)
    write_json(out / "happo_config.json", cfg.to_dict())
    return {
        "summary": summary,
        "episode_metrics": episode_metrics,
        "step_metrics": step_metrics,
        "dispatch_private_profit_trace": dispatch_private_profit_trace,
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
    from vpp_dso_sim.learning.reward_config import write_reward_config_artifacts
    from vpp_dso_sim.utils.io import ensure_dir, write_json

    cfg = config or HASACConfig()
    torch, optim, _, _ = _require_torch()
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    out = ensure_dir(output_dir)
    device, device_meta = _resolve_torch_device(torch, cfg.device)

    env_probe = MultiAgentVPPDSOEnv(config_path=config_path, horizon_steps=cfg.horizon_steps)
    observations, _ = env_probe.reset(seed=cfg.seed)
    resolved_reward_config = env_probe.scenario.dso.reward_config
    cfg = replace(
        cfg,
        reward_scale=float(resolved_reward_config.critic_reward_scale),
        dso_shield_intervention_penalty_coef=float(resolved_reward_config.shield.dso_penalty_coef),
        dispatch_shield_intervention_penalty_coef=float(resolved_reward_config.shield.dispatch_penalty_coef),
    )
    reward_artifacts = write_reward_config_artifacts(out, resolved_reward_config)
    print(
        "[RewardConfig] "
        f"version={resolved_reward_config.version} "
        f"critic_reward_scale={resolved_reward_config.critic_reward_scale} "
        f"shield_dso_penalty_coef={resolved_reward_config.shield.dso_penalty_coef} "
        f"shield_dispatch_penalty_coef={resolved_reward_config.shield.dispatch_penalty_coef} "
        f"shield_portfolio_future_penalty_coef={resolved_reward_config.shield.portfolio_future_penalty_coef} "
        f"hash={reward_artifacts['reward_config_hash']}",
        flush=True,
    )
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
        dispatch_actor_encoder_type=cfg.dispatch_actor_encoder_type,
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
    actor_modules.to(device)
    role_critic.to(device)
    target_role_critic.to(device)
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
    log_alpha_dso = torch.tensor(
        float(cfg.init_log_alpha_dso),
        dtype=torch.float32,
        device=device,
        requires_grad=True,
    )
    log_alpha_dispatch = torch.tensor(
        float(cfg.init_log_alpha_dispatch),
        dtype=torch.float32,
        device=device,
        requires_grad=True,
    )
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
        device=device,
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
                dso_obs_tensor = torch.tensor(dso_obs_vec, dtype=torch.float32, device=device).unsqueeze(0)
                vpp_obs_tensor = torch.tensor(vpp_obs_mat, dtype=torch.float32, device=device).unsqueeze(0)
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
            dispatch_components = [
                infos[f"{vpp_id}_dispatch"].get("agent_reward_components", {})
                for vpp_id in vpp_ids
            ]
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
            def component_mean(rows: list[dict[str, Any]], key: str) -> float:
                values = [float(row.get(key, 0.0)) for row in rows]
                return float(np.mean(values)) if values else 0.0

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
                    "baseline_p_mw": component_mean(dispatch_components, "baseline_p_mw"),
                    "raw_action_norm": component_mean(dispatch_components, "raw_action_norm"),
                    "raw_target_p_mw": component_mean(dispatch_components, "raw_target_p_mw"),
                    "decoded_target_p_mw": component_mean(dispatch_components, "decoded_target_p_mw"),
                    "device_feasible_target_p_mw": component_mean(dispatch_components, "device_feasible_target_p_mw"),
                    "pre_ac_target_p_mw": component_mean(dispatch_components, "pre_ac_target_p_mw"),
                    "ac_projected_target_p_mw": component_mean(dispatch_components, "ac_projected_target_p_mw"),
                    "ac_certified_target_p_mw": component_mean(dispatch_components, "ac_certified_target_p_mw"),
                    "actual_target_p_mw": component_mean(dispatch_components, "actual_target_p_mw"),
                    "raw_delta_p_mw": component_mean(dispatch_components, "raw_delta_p_mw"),
                    "decoded_delta_p_mw": component_mean(dispatch_components, "decoded_delta_p_mw"),
                    "device_feasible_delta_p_mw": component_mean(dispatch_components, "device_feasible_delta_p_mw"),
                    "pre_ac_delta_p_mw": component_mean(dispatch_components, "pre_ac_delta_p_mw"),
                    "ac_projected_delta_p_mw": component_mean(dispatch_components, "ac_projected_delta_p_mw"),
                    "ac_certified_delta_p_mw": component_mean(dispatch_components, "ac_certified_delta_p_mw"),
                    "accepted_delta_p_mw": component_mean(dispatch_components, "accepted_delta_p_mw"),
                    "actual_delta_p_mw": component_mean(dispatch_components, "actual_delta_p_mw"),
                    "raw_to_device_gap_mw": component_mean(dispatch_components, "raw_to_device_gap_mw"),
                    "device_to_ac_gap_mw": component_mean(dispatch_components, "device_to_ac_gap_mw"),
                    "ac_to_actual_gap_mw": component_mean(dispatch_components, "ac_to_actual_gap_mw"),
                    "accepted_to_actual_gap_mw": component_mean(dispatch_components, "accepted_to_actual_gap_mw"),
                    "actual_delta_nonzero_flag": component_mean(dispatch_components, "actual_delta_nonzero_flag"),
                    "action_landing_ratio": component_mean(dispatch_components, "action_landing_ratio"),
                    "action_landing_drop_reason_code": component_mean(dispatch_components, "action_landing_drop_reason_code"),
                    "total_cost": float(reward_components.get("total_cost", -dso_reward)),
                    "violation_count": int(len(violations)),
                    "replay_size": int(len(replay)),
                    "action_min": float(joint_action_t.min().cpu().item()),
                    "action_max": float(joint_action_t.max().cpu().item()),
                }
            )

            if len(replay) >= int(cfg.batch_size) and total_env_steps >= int(cfg.warmup_steps):
                batch = replay.sample(cfg.batch_size)
                critic_state_b = torch.tensor(batch["critic_state"], dtype=torch.float32, device=device)
                joint_action_b = torch.tensor(batch["joint_action"], dtype=torch.float32, device=device)
                next_critic_state_b = torch.tensor(batch["next_critic_state"], dtype=torch.float32, device=device)
                next_dso_obs_b = torch.tensor(batch["next_dso_obs"], dtype=torch.float32, device=device)
                next_vpp_obs_b = torch.tensor(batch["next_vpp_obs"], dtype=torch.float32, device=device)
                reward_vector_b = torch.tensor(batch["reward_vector"], dtype=torch.float32, device=device)
                shield_gap_b = torch.tensor(batch["shield_intervention_gap_mw"], dtype=torch.float32, device=device)
                shield_penalty_b = torch.tensor(batch["shield_intervention_penalty"], dtype=torch.float32, device=device)
                done_b = torch.tensor(batch["done"], dtype=torch.float32, device=device)

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

                current_dso_obs_b = torch.tensor(batch["dso_obs"], dtype=torch.float32, device=device)
                current_vpp_obs_b = torch.tensor(batch["vpp_obs"], dtype=torch.float32, device=device)
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
                "actor_state_dict": _state_dict_to_cpu(actor_modules.state_dict()),
                "role_critic_state_dict": _state_dict_to_cpu(role_critic.state_dict()),
                "target_role_critic_state_dict": _state_dict_to_cpu(target_role_critic.state_dict()),
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
            "reward_config": resolved_reward_config.to_dict(),
            "reward_config_hash": reward_artifacts["reward_config_hash"],
            "selection_metric": "episode_reward",
        }

    torch.save(
        checkpoint_payload(
            _state_dict_to_cpu(actor_modules.state_dict()),
            _state_dict_to_cpu(role_critic.state_dict()),
            _state_dict_to_cpu(target_role_critic.state_dict()),
            float(log_alpha_dso.detach().cpu().item()),
            float(log_alpha_dispatch.detach().cpu().item()),
        ),
        checkpoint_path,
    )
    if best_checkpoint_state is None:
        best_checkpoint_state = {
            "actor_state_dict": _state_dict_to_cpu(actor_modules.state_dict()),
            "role_critic_state_dict": _state_dict_to_cpu(role_critic.state_dict()),
            "target_role_critic_state_dict": _state_dict_to_cpu(target_role_critic.state_dict()),
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
        "requested_device": str(device_meta["requested_device"]),
        "resolved_device": str(device_meta["resolved_device"]),
        "cuda_available": bool(device_meta["cuda_available"]),
        "cuda_device_count": int(device_meta["cuda_device_count"]),
        "cuda_device_name": device_meta["cuda_device_name"],
        "device_meta": dict(device_meta),
        "training_pattern": "off_policy_ctde",
        "soft_actor_critic": True,
        "twin_soft_q": True,
        "off_policy_replay": True,
        "automatic_entropy_tuning": True,
        "reward_scale": float(cfg.reward_scale),
        "reward_version": str(resolved_reward_config.version),
        "critic_reward_scale": float(resolved_reward_config.critic_reward_scale),
        "resolved_reward_config": reward_artifacts["resolved_reward_config"],
        "reward_config_hash": reward_artifacts["reward_config_hash"],
        "reward_config_hash_path": reward_artifacts["reward_config_hash_path"],
        "target_q_clip": None if cfg.target_q_clip is None else float(cfg.target_q_clip),
        "critic_grad_clip": float(cfg.critic_grad_clip),
        "actor_grad_clip": float(cfg.actor_grad_clip),
        "alpha_grad_clip": float(cfg.alpha_grad_clip),
        "dso_dispatch_actor_objectives_separated": True,
        "dispatch_actor_encoder_type": architecture_meta.get("dispatch_actor_encoder_type", cfg.dispatch_actor_encoder_type),
        "vpp_encoder_type": architecture_meta.get("vpp_encoder_type", ""),
        "architecture_version": architecture_meta.get("architecture_version", ""),
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
    device, device_meta = _resolve_torch_device(torch, cfg.get("device", "auto"))
    out = ensure_dir(output_dir)
    eval_horizon = int(horizon_steps or cfg.get("horizon_steps", 8))
    normalize_observations = bool(cfg.get("normalize_observations", False))
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
        dispatch_actor_encoder_type=str(cfg.get("dispatch_actor_encoder_type", "deepset_v1")),
    )
    actor_modules = torch.nn.ModuleDict({"dso_actor": modules["dso_actor"]})
    if bool(cfg.get("share_vpp_dispatch_parameters", False)):
        actor_modules["vpp_dispatch_actor"] = modules["vpp_dispatch_actor"]
    else:
        for vpp_id in vpp_ids:
            actor_modules[f"{vpp_id}_dispatch_actor"] = copy.deepcopy(modules["vpp_dispatch_actor"])
    actor_modules.load_state_dict(checkpoint["actor_state_dict"])
    actor_modules.to(device)
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
            dso_obs_tensor = torch.as_tensor(dso_obs_vec, dtype=torch.float32, device=device).unsqueeze(0)
            vpp_obs_tensor = torch.as_tensor(vpp_obs_mat, dtype=torch.float32, device=device).unsqueeze(0)
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
    from vpp_dso_sim.dso.models.structured_happo_actor import StructuredDSOGaussianActor
    from vpp_dso_sim.dso.observation.happo_structured import StructuredDSOFlatSpec, build_happo_structured_dso_observation
    from vpp_dso_sim.learning.deep_rl import (
        _build_privacy_separated_networks,
        encode_vpp_portfolio_observation,
    )
    from vpp_dso_sim.learning.reward_contracts import shield_intervention_metrics
    from vpp_dso_sim.utils.io import ensure_dir, write_json

    torch, _, _, _ = _require_torch()
    checkpoint = torch.load(Path(checkpoint_path), map_location="cpu")
    cfg = checkpoint.get("config", {})
    device, device_meta = _resolve_torch_device(torch, cfg.get("device", "auto"))
    out = ensure_dir(output_dir)
    eval_horizon = int(horizon_steps or cfg.get("horizon_steps", 8))
    normalize_observations = bool(cfg.get("normalize_observations", False))
    vpp_ids = list(checkpoint["vpp_ids"])
    max_der_per_vpp = int(checkpoint["max_der_per_vpp"])
    der_ids_by_vpp = dict(checkpoint["der_ids_by_vpp"])
    dso_actor_observation_mode = str(checkpoint.get("dso_actor_observation_mode", "legacy_flat"))
    dso_actor_type = str(checkpoint.get("dso_actor_type", "legacy_mlp_gaussian"))
    structured_dso_spec_payload = checkpoint.get("structured_dso_flat_spec")
    use_structured_dso_actor = bool(
        dso_actor_observation_mode == "structured_bipartite"
        or dso_actor_type == "sensitivity_attention_v1_structured_happo"
    )
    structured_dso_spec = None
    if use_structured_dso_actor:
        if not structured_dso_spec_payload:
            raise ValueError("Structured HAPPO checkpoint is missing structured_dso_flat_spec.")
        structured_dso_spec = StructuredDSOFlatSpec(
            global_dim=int(structured_dso_spec_payload["global_dim"]),
            action_token_dim=int(structured_dso_spec_payload["action_token_dim"]),
            object_token_dim=int(structured_dso_spec_payload["object_token_dim"]),
            edge_feature_dim=int(structured_dso_spec_payload["edge_feature_dim"]),
            max_action_units=int(structured_dso_spec_payload["max_action_units"]),
            max_network_objects=int(structured_dso_spec_payload["max_network_objects"]),
            action_unit_vpp_indices=tuple(int(value) for value in structured_dso_spec_payload["action_unit_vpp_indices"]),
            vpp_ids=tuple(str(value) for value in structured_dso_spec_payload["vpp_ids"]),
            action_unit_ids=tuple(str(value) for value in structured_dso_spec_payload.get("action_unit_ids", ())),
            field_names=tuple(str(value) for value in structured_dso_spec_payload.get("field_names", ())),
            privacy_boundary=str(
                structured_dso_spec_payload.get("privacy_boundary", "dso_execution_actor_no_private_vpp_fields")
            ),
        )
    probe_env = MultiAgentVPPDSOEnv(config_path=config_path, horizon_steps=eval_horizon)
    probe_scenario_config = probe_env.scenario.config
    probe_env.close()

    modules, _ = _build_privacy_separated_networks(
        dso_input_dim=int(checkpoint["dso_input_dim"]),
        vpp_input_dim=int(checkpoint["vpp_input_dim"]),
        portfolio_input_dim=int(checkpoint.get("portfolio_input_dim", 9)),
        critic_input_dim=int(checkpoint["critic_input_dim"]),
        critic_action_dim=1,
        action_dim=len(vpp_ids),
        der_action_dim=max_der_per_vpp,
        hidden_dim=int(cfg.get("hidden_dim", 64)),
        dispatch_actor_encoder_type=str(cfg.get("dispatch_actor_encoder_type", "deepset_v1")),
    )
    if use_structured_dso_actor:
        if structured_dso_spec is None:
            raise RuntimeError("structured_dso_spec was not initialized for structured HAPPO frozen eval")
        actor_cfg = dict(probe_scenario_config.get("dso", {}).get("actor", {}))
        modules["dso_actor"] = StructuredDSOGaussianActor(
            spec=structured_dso_spec,
            d_model=int(actor_cfg.get("d_model", cfg.get("hidden_dim", 64))),
            num_heads=int(actor_cfg.get("num_heads", 4)),
            num_layers=int(actor_cfg.get("num_layers", 1)),
            action_self_attention_layers=int(actor_cfg.get("action_self_attention_layers", 1)),
            dropout=float(actor_cfg.get("dropout", 0.0)),
            min_width_ratio=float(actor_cfg.get("min_width_ratio", 0.10)),
            max_width_ratio=float(actor_cfg.get("max_width_ratio", 1.00)),
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
    actor_modules.to(device)
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
            if use_structured_dso_actor:
                dso_obs_vec, current_structured_spec = build_happo_structured_dso_observation(
                    env.scenario,
                    step=env.current_step,
                    config=env.scenario.config,
                )
                if structured_dso_spec is not None and int(len(dso_obs_vec)) != int(structured_dso_spec.flat_dim):
                    raise RuntimeError(
                        "Structured DSO frozen-eval observation shape changed: "
                        f"got {len(dso_obs_vec)}, expected {structured_dso_spec.flat_dim}."
                    )
                if structured_dso_spec is not None and tuple(current_structured_spec.vpp_ids) != tuple(structured_dso_spec.vpp_ids):
                    raise RuntimeError(
                        "Structured DSO frozen-eval VPP ids do not match checkpoint: "
                        f"got {current_structured_spec.vpp_ids}, expected {structured_dso_spec.vpp_ids}."
                    )
            if normalize_observations:
                dso_obs_vec = _normalize_observation_array(dso_obs_vec)
                vpp_obs_mat = _normalize_observation_array(vpp_obs_mat)
            dso_obs_tensor = torch.as_tensor(dso_obs_vec, dtype=torch.float32, device=device).unsqueeze(0)
            vpp_obs_tensor = torch.as_tensor(vpp_obs_mat, dtype=torch.float32, device=device).unsqueeze(0)
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
                structured_dso_spec=structured_dso_spec if use_structured_dso_actor else None,
                dso_actor_cfg=dict(env.scenario.config.get("dso", {}).get("actor", {})),
            )
            decision_step = portfolio_decision_step(step)
            for vpp_id in vpp_ids:
                portfolio_action = "keep"
                if decision_step:
                    portfolio_vec = encode_vpp_portfolio_observation(observations[f"{vpp_id}_portfolio"])
                    if normalize_observations:
                        portfolio_vec = _normalize_observation_array(portfolio_vec)
                    portfolio_tensor = torch.as_tensor(portfolio_vec, dtype=torch.float32, device=device).unsqueeze(0)
                    logits = portfolio_actor_for(vpp_id)(portfolio_tensor)
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
        "dso_actor_observation_mode": dso_actor_observation_mode,
        "dso_actor_type": dso_actor_type,
        "structured_dso_actor_loaded": bool(use_structured_dso_actor),
        "normalize_observations": bool(normalize_observations),
        "requested_device": str(device_meta["requested_device"]),
        "resolved_device": str(device_meta["resolved_device"]),
        "cuda_available": bool(device_meta["cuda_available"]),
        "cuda_device_count": int(device_meta["cuda_device_count"]),
        "cuda_device_name": device_meta["cuda_device_name"],
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
