from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import importlib.util
import math

import numpy as np
import pandas as pd

from vpp_dso_sim.envs.multi_agent_env import MultiAgentVPPDSOEnv
from vpp_dso_sim.envs.observations import build_critic_global_state
from vpp_dso_sim.learning.ctde_networks import (
    VPP_DISPATCH_CONTEXT_DIM,
    VPP_DISPATCH_TOKEN_DIM,
    build_privacy_separated_ctde_modules,
    encode_joint_action_summary as _encode_joint_action_summary,
    joint_action_summary_dim,
)
from vpp_dso_sim.utils.io import ensure_dir, write_json


TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None


@dataclass(frozen=True)
class DeepRLConfig:
    algorithm: str = "shared_actor_critic_benchmark"
    horizon_steps: int = 8
    episodes: int = 3
    gamma: float = 0.97
    learning_rate: float = 3e-4
    hidden_dim: int = 64
    entropy_coef: float = 0.01
    value_coef: float = 0.50
    max_grad_norm: float = 1.0
    seed: int = 42
    action_clip: float = 1.0
    portfolio_reward_coef: float = 0.20

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class PrivacySeparatedCTDEConfig:
    algorithm: str = "privacy_separated_ctde_actor_critic"
    horizon_steps: int = 8
    episodes: int = 3
    gamma: float = 0.97
    gae_lambda: float = 0.95
    ppo_clip_ratio: float = 0.20
    ppo_epochs: int = 1
    use_gae: bool = True
    learning_rate: float = 3e-4
    hidden_dim: int = 64
    entropy_coef: float = 0.01
    value_coef: float = 0.50
    dso_loss_coef: float = 1.0
    vpp_dispatch_loss_coef: float = 1.0
    portfolio_loss_coef: float = 0.25
    max_grad_norm: float = 1.0
    seed: int = 42
    action_clip: float = 1.0
    portfolio_reward_coef: float = 0.20

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _require_torch():
    if not TORCH_AVAILABLE:
        raise ImportError(
            "PyTorch is required for deep RL training. Install torch or run the lightweight baseline instead."
        )
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.distributions import Categorical, Normal

    return torch, nn, optim, Normal, Categorical


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None or (isinstance(value, float) and math.isnan(value)):
            return float(default)
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def encode_dso_observation(obs: dict[str, Any], vpp_ids: list[str], max_vpps: int | None = None) -> np.ndarray:
    """Encode DSO observation into a stable numeric vector for neural policies."""

    if max_vpps is None:
        max_vpps = len(vpp_ids)
    network = obs.get("network_state", {})
    features: list[float] = [
        _as_float(obs.get("time_index", 0)) / 288.0,
        _as_float(network.get("min_vm_pu", 1.0), 1.0),
        _as_float(network.get("max_vm_pu", 1.0), 1.0),
        _as_float(network.get("max_line_loading_percent", 0.0)) / 100.0,
        _as_float(network.get("max_trafo_loading_percent", 0.0)) / 100.0,
    ]
    reports = obs.get("vpp_reports", {})
    for vpp_id in vpp_ids[:max_vpps]:
        report = reports.get(vpp_id, {})
        features.extend(
            [
                _as_float(report.get("p_mw", 0.0)),
                _as_float(report.get("q_mvar", 0.0)),
                _as_float(report.get("p_min_mw", 0.0)),
                _as_float(report.get("p_max_mw", 0.0)),
                _as_float(report.get("q_min_mvar", 0.0)),
                _as_float(report.get("q_max_mvar", 0.0)),
                1.0 if str(report.get("physical_mode", "")) == "single_pcc" else 0.0,
            ]
        )
    missing = max(0, max_vpps - len(vpp_ids))
    features.extend([0.0] * missing * 7)
    return np.asarray(features, dtype=np.float32)


def _der_type_code(der_type: str) -> list[float]:
    text = str(der_type).lower()
    return [
        1.0 if "pv" in text else 0.0,
        1.0 if "storage" in text else 0.0,
        1.0 if "micro" in text or "turbine" in text else 0.0,
        1.0 if "flexible" in text or "load" in text else 0.0,
        1.0 if "ev" in text else 0.0,
        1.0 if "hvac" in text else 0.0,
    ]


VPP_CONTEXT_FEATURES = VPP_DISPATCH_CONTEXT_DIM
DER_TOKEN_FEATURES = VPP_DISPATCH_TOKEN_DIM


def encode_vpp_dispatch_observation(obs: dict[str, Any], max_der_per_vpp: int) -> np.ndarray:
    """Encode one VPP's private dispatch observation.

    This encoder deliberately uses only the owning VPP's local fields plus the
    DSO envelope addressed to that VPP. It does not read `network_state`,
    `vpp_reports`, `critic_global_state` or any other VPP's private assets.
    """

    bounds = obs.get("aggregate_bounds", {})
    current = obs.get("current_power", {})
    envelope = obs.get("operating_envelope", {})
    signal = obs.get("service_signal", {})
    award = obs.get("dispatch_award", {})
    portfolio = obs.get("portfolio", {})
    service_text = str(signal.get("service_request", envelope.get("service_request", ""))).lower()
    assets = list(obs.get("local_assets", []))[:max_der_per_vpp]
    features: list[float] = [
        _as_float(obs.get("time_index", 0)) / 288.0,
        _as_float(bounds.get("p_min_mw", 0.0)),
        _as_float(bounds.get("p_max_mw", 0.0)),
        _as_float(bounds.get("q_min_mvar", 0.0)),
        _as_float(bounds.get("q_max_mvar", 0.0)),
        _as_float(current.get("p_mw", 0.0)),
        _as_float(current.get("q_mvar", 0.0)),
        _as_float(envelope.get("preferred_target_p_mw", award.get("awarded_p_mw", 0.0))),
        _as_float(envelope.get("preferred_p_min_mw", envelope.get("p_min_mw", 0.0))),
        _as_float(envelope.get("preferred_p_max_mw", envelope.get("p_max_mw", 0.0))),
        _as_float(envelope.get("price", signal.get("price", 0.0))) / 100.0,
        1.0 if "absorb" in service_text or "import" in service_text else 0.0,
        1.0 if "export" in service_text or "inject" in service_text else 0.0,
        1.0 if str(portfolio.get("physical_mode", "")) == "single_pcc" else 0.0,
        float(len(portfolio.get("connection_buses", []))) / 20.0,
        float(len(assets)) / max(1.0, float(max_der_per_vpp)),
    ]
    for asset in assets:
        cost = asset.get("cost_coefficients") or (0.0, 0.0, 0.0)
        try:
            c0, c1, c2 = list(cost)[:3]
        except TypeError:
            c0, c1, c2 = 0.0, 0.0, 0.0
        features.extend(
            [
                *_der_type_code(str(asset.get("der_type", ""))),
                _as_float(asset.get("bus_id", 0.0)) / 100.0,
                _as_float(asset.get("p_min_mw", 0.0)),
                _as_float(asset.get("p_max_mw", 0.0)),
                _as_float(asset.get("q_min_mvar", 0.0)),
                _as_float(asset.get("q_max_mvar", 0.0)),
                1.0 if bool(asset.get("controllable", True)) else 0.0,
                _as_float(c0, 0.0),
                _as_float(c1, 0.0),
                _as_float(c2, 0.0),
                _as_float(asset.get("current_p_mw", 0.0)),
                _as_float(asset.get("current_q_mvar", 0.0)),
                _as_float(asset.get("available_p_mw", 0.0)),
                _as_float(asset.get("soc", 0.0)),
                _as_float(asset.get("soc_min", 0.0)),
                _as_float(asset.get("soc_max", 0.0)),
                _as_float(asset.get("average_soc", 0.0)),
                _as_float(asset.get("indoor_temp", 0.0)) / 40.0,
                _as_float(asset.get("temp_min", 0.0)) / 40.0,
                _as_float(asset.get("temp_max", 0.0)) / 40.0,
                _as_float(asset.get("comfort_penalty", 0.0)) / 100.0,
            ]
        )
    per_der_features = DER_TOKEN_FEATURES
    missing = max(0, max_der_per_vpp - len(assets))
    features.extend([0.0] * missing * per_der_features)
    return np.asarray(features, dtype=np.float32)


def encode_joint_action_summary(
    *,
    normalized_dso_action: np.ndarray,
    vpp_ids: list[str],
    normalized_aggregate_actions: dict[str, float],
    normalized_der_actions: dict[str, np.ndarray],
    portfolio_action_indices: dict[str, int],
    max_der_per_vpp: int,
    action_clip: float,
) -> np.ndarray:
    """Encode a compact training-only joint-action summary for the critic."""

    scale = max(1e-6, float(action_clip))
    normalized_dso = np.clip(
        np.asarray(normalized_dso_action, dtype=np.float32).reshape(-1) / scale,
        -1.0,
        1.0,
    )
    normalized_aggregate = {
        vpp_id: float(np.clip(float(value) / scale, -1.0, 1.0))
        for vpp_id, value in normalized_aggregate_actions.items()
    }
    normalized_der = {
        vpp_id: np.clip(np.asarray(values, dtype=np.float32).reshape(-1) / scale, -1.0, 1.0)
        for vpp_id, values in normalized_der_actions.items()
    }
    return _encode_joint_action_summary(
        vpp_ids=vpp_ids,
        dso_action=normalized_dso,
        vpp_aggregate_actions=normalized_aggregate,
        der_actions_by_vpp=normalized_der,
        portfolio_action_indices=portfolio_action_indices,
    )


def encode_vpp_portfolio_observation(obs: dict[str, Any]) -> np.ndarray:
    portfolio = obs.get("portfolio", {})
    der_ids = list(portfolio.get("der_ids", []))
    connection_buses = list(portfolio.get("connection_buses", []))
    return np.asarray(
        [
            _as_float(obs.get("time_index", 0)) / 288.0,
            1.0 if str(portfolio.get("physical_mode", "")) == "single_pcc" else 0.0,
            _as_float(portfolio.get("pcc_bus_id", 0.0)) / 100.0,
            float(len(connection_buses)) / 20.0,
            float(len(der_ids)) / 50.0,
            _as_float(portfolio.get("max_import_mw", 0.0)),
            _as_float(portfolio.get("max_export_mw", 0.0)),
            1.0 if bool(obs.get("trainable_action_current_version", True)) else 0.0,
            1.0 if bool(obs.get("physical_change_allowed", False)) else 0.0,
        ],
        dtype=np.float32,
    )


def encode_critic_global_state(state: dict[str, Any], vpp_ids: list[str], max_vpps: int | None = None) -> np.ndarray:
    if max_vpps is None:
        max_vpps = len(vpp_ids)
    network = state.get("network_state", {})
    features: list[float] = [
        _as_float(state.get("time_index", 0)) / 288.0,
        _as_float(network.get("min_vm_pu", 1.0), 1.0),
        _as_float(network.get("max_vm_pu", 1.0), 1.0),
        _as_float(network.get("max_line_loading_percent", 0.0)) / 100.0,
        _as_float(network.get("max_trafo_loading_percent", 0.0)) / 100.0,
    ]
    reports = state.get("vpp_reports", {})
    actor_obs = state.get("vpp_actor_observations", {})
    for vpp_id in vpp_ids[:max_vpps]:
        report = reports.get(vpp_id, {})
        local = actor_obs.get(vpp_id, {})
        assets = list(local.get("local_assets", []))
        private_cost_sum = 0.0
        for asset in assets:
            cost = asset.get("cost_coefficients") or (0.0, 0.0, 0.0)
            try:
                private_cost_sum += sum(float(value) for value in list(cost)[:3])
            except (TypeError, ValueError):
                private_cost_sum += 0.0
        features.extend(
            [
                _as_float(report.get("p_mw", 0.0)),
                _as_float(report.get("q_mvar", 0.0)),
                _as_float(report.get("p_min_mw", 0.0)),
                _as_float(report.get("p_max_mw", 0.0)),
                _as_float(report.get("q_min_mvar", 0.0)),
                _as_float(report.get("q_max_mvar", 0.0)),
                1.0 if str(report.get("physical_mode", "")) == "single_pcc" else 0.0,
                float(len(assets)) / 50.0,
                private_cost_sum / max(1.0, float(len(assets))),
            ]
        )
    missing = max(0, max_vpps - len(vpp_ids))
    features.extend([0.0] * missing * 9)
    return np.asarray(features, dtype=np.float32)


def _targets_from_normalized_actions(
    action: np.ndarray,
    dso_obs: dict[str, Any],
    vpp_ids: list[str],
    action_clip: float,
) -> dict[str, float]:
    reports = dso_obs.get("vpp_reports", {})
    clipped = np.clip(action, -float(action_clip), float(action_clip))
    targets: dict[str, float] = {}
    for index, vpp_id in enumerate(vpp_ids):
        report = reports.get(vpp_id, {})
        p_min = _as_float(report.get("p_min_mw", -0.1), -0.1)
        p_max = _as_float(report.get("p_max_mw", 0.1), 0.1)
        center = 0.5 * (p_min + p_max)
        halfspan = max(1e-6, 0.5 * (p_max - p_min))
        targets[vpp_id] = float(center + float(clipped[index]) * halfspan)
    return targets


def _target_from_normalized_scalar(normalized_action: float, vpp_obs: dict[str, Any], action_clip: float) -> float:
    bounds = vpp_obs.get("aggregate_bounds", {})
    envelope = vpp_obs.get("operating_envelope", {})
    p_min = _as_float(envelope.get("preferred_p_min_mw", bounds.get("p_min_mw", -0.1)), -0.1)
    p_max = _as_float(envelope.get("preferred_p_max_mw", bounds.get("p_max_mw", 0.1)), 0.1)
    if p_min > p_max:
        p_min, p_max = p_max, p_min
    clipped = max(-float(action_clip), min(float(action_clip), float(normalized_action)))
    center = 0.5 * (p_min + p_max)
    halfspan = max(1e-6, 0.5 * (p_max - p_min))
    return float(center + clipped * halfspan)


def _portfolio_action_from_logits(logits, categorical_cls) -> tuple[str, Any, Any, int]:
    dist = categorical_cls(logits=logits)
    sample = dist.sample()
    action_idx = int(sample.item())
    label = ("keep", "reweight", "propose_membership_change")[action_idx]
    return label, dist.log_prob(sample), dist.entropy(), action_idx


def _portfolio_proxy_reward(label: str, dso_obs: dict[str, Any], vpp_id: str) -> float:
    """Small action-dependent learning signal for the slow portfolio head.

    Portfolio actions are commercial configuration proposals in v0: they must
    not move DER physical buses during a fast simulation step. This proxy reward
    lets the portfolio head train while keeping physical membership changes
    gated by deterministic scenario events. It prefers:
    - `keep` when the grid is calm and current flexibility is adequate.
    - `reweight` when the grid/price context suggests medium stress.
    - `propose_membership_change` only when stress is high or flexibility is thin.
    """

    reports = dso_obs.get("vpp_reports", {})
    report = reports.get(vpp_id, {})
    network = dso_obs.get("network_state", {})
    p_min = _as_float(report.get("p_min_mw", 0.0))
    p_max = _as_float(report.get("p_max_mw", 0.0))
    flex_span = max(0.0, p_max - p_min)
    line_stress = max(0.0, _as_float(network.get("max_line_loading_percent", 0.0)) / 100.0 - 0.80)
    voltage_low_stress = max(0.0, 0.95 - _as_float(network.get("min_vm_pu", 1.0), 1.0))
    voltage_high_stress = max(0.0, _as_float(network.get("max_vm_pu", 1.0), 1.0) - 1.05)
    stress = line_stress + 10.0 * (voltage_low_stress + voltage_high_stress)
    thin_flex = max(0.0, 0.10 - flex_span)
    multi_node_bonus = 0.01 if str(report.get("physical_mode", "")) == "multi_node" else 0.0
    if label == "keep":
        return 0.025 - 0.020 * stress - 0.030 * thin_flex
    if label == "reweight":
        return 0.010 + 0.030 * min(stress, 1.0) + multi_node_bonus - 0.010 * thin_flex
    if label == "propose_membership_change":
        return -0.010 + 0.060 * min(stress + thin_flex, 1.0) + multi_node_bonus
    return -0.05


def _build_networks(input_dim: int, action_dim: int, der_action_dim: int, hidden_dim: int):
    torch, nn, _, _, _ = _require_torch()

    class GaussianActorCritic(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.backbone = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.Tanh(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.Tanh(),
            )
            self.mean = nn.Linear(hidden_dim, action_dim)
            self.value = nn.Linear(hidden_dim, 1)
            self.dso_log_std = nn.Parameter(torch.full((action_dim,), -0.7))
            self.vpp_target_head = nn.Linear(hidden_dim, action_dim)
            self.vpp_target_log_std = nn.Parameter(torch.full((action_dim,), -0.8))
            self.der_dispatch_head = nn.Linear(hidden_dim, der_action_dim)
            self.der_dispatch_log_std = nn.Parameter(torch.full((der_action_dim,), -0.8))
            self.portfolio_head = nn.Linear(hidden_dim, action_dim * 3)

        def forward(self, x):
            latent = self.backbone(x)
            mean = torch.tanh(self.mean(latent))
            value = self.value(latent).squeeze(-1)
            dso_log_std = self.dso_log_std.expand_as(mean)
            vpp_target_mean = torch.tanh(self.vpp_target_head(latent))
            vpp_target_log_std = self.vpp_target_log_std.expand_as(vpp_target_mean)
            der_dispatch_mean = torch.tanh(self.der_dispatch_head(latent))
            der_dispatch_log_std = self.der_dispatch_log_std.expand_as(der_dispatch_mean)
            portfolio_logits = self.portfolio_head(latent).reshape(-1, action_dim, 3)
            return (
                mean,
                dso_log_std,
                value,
                vpp_target_mean,
                vpp_target_log_std,
                der_dispatch_mean,
                der_dispatch_log_std,
                portfolio_logits,
            )

    return GaussianActorCritic()


def _build_privacy_separated_networks(
    *,
    dso_input_dim: int,
    vpp_input_dim: int,
    portfolio_input_dim: int,
    critic_input_dim: int,
    critic_action_dim: int,
    action_dim: int,
    der_action_dim: int,
    hidden_dim: int,
):
    torch, nn, _, _, _ = _require_torch()
    return build_privacy_separated_ctde_modules(
        torch=torch,
        nn=nn,
        dso_input_dim=dso_input_dim,
        vpp_input_dim=vpp_input_dim,
        portfolio_input_dim=portfolio_input_dim,
        critic_input_dim=critic_input_dim,
        action_summary_dim=critic_action_dim,
        action_dim=action_dim,
        der_action_dim=der_action_dim,
        hidden_dim=hidden_dim,
    )


def _discounted_returns(rewards: list[float], gamma: float) -> list[float]:
    values: list[float] = []
    running = 0.0
    for reward in reversed(rewards):
        running = float(reward) + float(gamma) * running
        values.append(running)
    return list(reversed(values))


def _normalize_tensor(tensor: Any, torch: Any) -> Any:
    if tensor.numel() <= 1:
        return tensor
    return (tensor - tensor.mean()) / (tensor.std(unbiased=False) + 1e-8)


def _gae_returns_advantages(
    *,
    rewards: list[float],
    values: Any,
    gamma: float,
    gae_lambda: float,
    torch: Any,
) -> tuple[Any, Any]:
    """Compute terminal-bootstrap GAE for one finite rollout.

    The simulator rollout has a fixed finite horizon, so the bootstrap value
    after the last step is zero. This keeps the function dependency-free and
    makes the training contract explicit for tests and reports.
    """

    reward_tensor = torch.tensor(rewards, dtype=torch.float32)
    detached_values = values.detach()
    advantages = torch.zeros_like(reward_tensor)
    running_advantage = torch.tensor(0.0, dtype=torch.float32)
    next_value = torch.tensor(0.0, dtype=torch.float32)
    for index in range(len(rewards) - 1, -1, -1):
        delta = reward_tensor[index] + float(gamma) * next_value - detached_values[index]
        running_advantage = delta + float(gamma) * float(gae_lambda) * running_advantage
        advantages[index] = running_advantage
        next_value = detached_values[index]
    returns = advantages + detached_values
    return returns, advantages


def _ppo_clipped_policy_loss(
    *,
    log_probs: Any,
    old_log_probs: Any,
    advantages: Any,
    clip_ratio: float,
    torch: Any,
) -> tuple[Any, Any]:
    normalized_advantages = _normalize_tensor(advantages, torch).detach()
    ratios = torch.exp(log_probs - old_log_probs.detach())
    unclipped = ratios * normalized_advantages
    clipped = torch.clamp(ratios, 1.0 - float(clip_ratio), 1.0 + float(clip_ratio)) * normalized_advantages
    return -torch.min(unclipped, clipped).mean(), ratios


def _vpp_der_slices(vpps) -> tuple[dict[str, list[str]], dict[str, slice], int]:
    der_ids_by_vpp: dict[str, list[str]] = {}
    slices: dict[str, slice] = {}
    start = 0
    for vpp in vpps:
        ids = [der.id for der in vpp.der_list]
        der_ids_by_vpp[vpp.id] = ids
        end = start + len(ids)
        slices[vpp.id] = slice(start, end)
        start = end
    return der_ids_by_vpp, slices, start


def train_deep_rl_actor_critic(
    config_path: str | Path | None = None,
    output_dir: str | Path = "outputs/deep_rl",
    config: DeepRLConfig | None = None,
) -> dict[str, Any]:
    """Train a small PyTorch actor-critic policy against the pandapower simulator.

    This is a real gradient-based RL loop: the neural policy samples DSO
    envelope-preference targets, VPP aggregate choices inside envelopes, and
    DER-level normalized disaggregation actions. Those actions drive
    `Simulator.step()`, FR/DOE projection and pandapower power flow, then the
    log-probabilities are updated from realized rewards. The slow portfolio
    head is also trained as a categorical commercial-configuration proposal;
    deterministic scenario portfolio events still gate physical membership
    changes so learned actions cannot silently move grid elements.
    """

    cfg = config or DeepRLConfig()
    torch, _, optim, Normal, Categorical = _require_torch()
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    out = ensure_dir(output_dir)
    env_probe = MultiAgentVPPDSOEnv(config_path=config_path, horizon_steps=cfg.horizon_steps)
    obs, _ = env_probe.reset(seed=cfg.seed)
    vpp_ids = [vpp.id for vpp in env_probe.scenario.vpps]
    der_ids_by_vpp, der_slices, der_action_dim = _vpp_der_slices(env_probe.scenario.vpps)
    input_dim = int(len(encode_dso_observation(obs["dso_global_guidance"], vpp_ids)))
    action_dim = int(len(vpp_ids))
    env_probe.close()

    policy = _build_networks(
        input_dim=input_dim,
        action_dim=action_dim,
        der_action_dim=der_action_dim,
        hidden_dim=cfg.hidden_dim,
    )
    optimizer = optim.Adam(policy.parameters(), lr=cfg.learning_rate)
    initial_params = torch.cat([param.detach().flatten().cpu() for param in policy.parameters()])
    optimizer_steps = 0

    episode_rows: list[dict[str, Any]] = []
    step_rows: list[dict[str, Any]] = []
    trajectory_rows: list[dict[str, Any]] = []
    loss_rows: list[dict[str, Any]] = []

    for episode in range(cfg.episodes):
        env = MultiAgentVPPDSOEnv(config_path=config_path, horizon_steps=cfg.horizon_steps)
        observations, _ = env.reset(seed=cfg.seed + episode)
        log_probs = []
        values = []
        entropies = []
        dso_rewards: list[float] = []
        dispatch_rewards: list[float] = []
        portfolio_rewards: list[float] = []
        learning_rewards: list[float] = []
        total_cost = 0.0
        violation_count = 0
        clipping_count = 0
        portfolio_proxy_reward_total = 0.0
        portfolio_action_counts = {"keep": 0, "reweight": 0, "propose_membership_change": 0}

        for step in range(cfg.horizon_steps):
            dso_obs = observations["dso_global_guidance"]
            encoded = torch.tensor(
                encode_dso_observation(dso_obs, vpp_ids),
                dtype=torch.float32,
            ).unsqueeze(0)
            (
                mean,
                dso_log_std,
                value,
                vpp_target_mean,
                vpp_target_log_std,
                der_dispatch_mean,
                der_dispatch_log_std,
                portfolio_logits,
            ) = policy(encoded)
            dso_dist = Normal(mean, dso_log_std.exp())
            raw_sample = dso_dist.rsample()
            normalized_action = torch.clamp(raw_sample, -cfg.action_clip, cfg.action_clip)
            dso_log_prob = dso_dist.log_prob(raw_sample).sum(dim=-1)
            entropy = dso_dist.entropy().sum(dim=-1)

            vpp_target_dist = Normal(vpp_target_mean, vpp_target_log_std.exp())
            raw_vpp_target = vpp_target_dist.rsample()
            normalized_vpp_target = torch.clamp(raw_vpp_target, -cfg.action_clip, cfg.action_clip)
            vpp_target_log_prob = vpp_target_dist.log_prob(raw_vpp_target).sum(dim=-1)
            vpp_target_entropy = vpp_target_dist.entropy().sum(dim=-1)

            der_dist = Normal(der_dispatch_mean, der_dispatch_log_std.exp())
            raw_der_dispatch = der_dist.rsample()
            normalized_der_dispatch = torch.clamp(raw_der_dispatch, -cfg.action_clip, cfg.action_clip)
            der_log_prob = der_dist.log_prob(raw_der_dispatch).sum(dim=-1)
            der_entropy = der_dist.entropy().sum(dim=-1)

            action_np = normalized_action.detach().cpu().numpy().reshape(-1)
            dso_targets = _targets_from_normalized_actions(action_np, dso_obs, vpp_ids, cfg.action_clip)
            vpp_target_np = normalized_vpp_target.detach().cpu().numpy().reshape(-1)
            selected_targets = _targets_from_normalized_actions(vpp_target_np, dso_obs, vpp_ids, cfg.action_clip)
            der_action_np = normalized_der_dispatch.detach().cpu().numpy().reshape(-1)

            action_payload: dict[str, Any] = {"dso_global_guidance": {"targets": dso_targets}}
            portfolio_log_prob_sum = None
            portfolio_entropy_sum = None
            step_portfolio_proxy_reward = 0.0
            step_portfolio_actions: dict[str, str] = {}
            for index, vpp_id in enumerate(vpp_ids):
                der_slice = der_slices[vpp_id]
                der_values = der_action_np[der_slice]
                der_actions = {
                    der_id: float(der_values[der_index])
                    for der_index, der_id in enumerate(der_ids_by_vpp[vpp_id])
                }
                action_payload[f"{vpp_id}_dispatch"] = {
                    "selected_p_mw": float(selected_targets[vpp_id]),
                    "der_actions": der_actions,
                    "policy_version": cfg.algorithm,
                }
                portfolio_label, portfolio_log_prob, portfolio_entropy, _ = _portfolio_action_from_logits(
                    portfolio_logits[0, index],
                    Categorical,
                )
                portfolio_log_prob_sum = (
                    portfolio_log_prob
                    if portfolio_log_prob_sum is None
                    else portfolio_log_prob_sum + portfolio_log_prob
                )
                portfolio_entropy_sum = (
                    portfolio_entropy
                    if portfolio_entropy_sum is None
                    else portfolio_entropy_sum + portfolio_entropy
                )
                step_portfolio_actions[vpp_id] = portfolio_label
                portfolio_action_counts[portfolio_label] += 1
                step_portfolio_proxy_reward += _portfolio_proxy_reward(portfolio_label, dso_obs, vpp_id)
                action_payload[f"{vpp_id}_portfolio"] = {
                    "action": portfolio_label,
                    "policy_version": cfg.algorithm,
                }

            observations, reward_map, _, truncations, infos = env.step(action_payload)
            dso_reward = float(reward_map["dso_global_guidance"])
            step_dispatch_reward = float(
                np.mean([reward_map[f"{vpp_id}_dispatch"] for vpp_id in vpp_ids])
                if vpp_ids
                else 0.0
            )
            step_portfolio_reward = float(
                np.mean([reward_map[f"{vpp_id}_portfolio"] for vpp_id in vpp_ids])
                if vpp_ids
                else 0.0
            )
            reward_components = infos["dso_global_guidance"].get("reward_components", {})
            violations = infos["dso_global_guidance"].get("violations", [])
            reward_total_cost = float(reward_components.get("total_cost", -dso_reward))
            learning_reward = dso_reward + step_dispatch_reward + float(cfg.portfolio_reward_coef) * step_portfolio_reward
            total_cost += reward_total_cost
            portfolio_proxy_reward_total += step_portfolio_reward
            violation_count += int(len(violations))
            clipped_now = bool(np.any(np.abs(action_np) >= cfg.action_clip - 1e-7))
            clipping_count += int(clipped_now)

            if portfolio_log_prob_sum is None:
                portfolio_log_prob_sum = torch.tensor(0.0, dtype=torch.float32)
            if portfolio_entropy_sum is None:
                portfolio_entropy_sum = torch.tensor(0.0, dtype=torch.float32)
            combined_log_prob = dso_log_prob + vpp_target_log_prob + der_log_prob + portfolio_log_prob_sum
            combined_entropy = entropy + vpp_target_entropy + der_entropy + portfolio_entropy_sum
            log_probs.append(combined_log_prob.squeeze(0))
            values.append(value.squeeze(0))
            entropies.append(combined_entropy.squeeze(0))
            dso_rewards.append(dso_reward)
            dispatch_rewards.append(step_dispatch_reward)
            portfolio_rewards.append(step_portfolio_reward)
            learning_rewards.append(learning_reward)

            step_rows.append(
                {
                    "episode": episode,
                    "step": step,
                    "algorithm": cfg.algorithm,
                    "reward": learning_reward,
                    "environment_reward": dso_reward,
                    "dso_reward": dso_reward,
                    "vpp_dispatch_reward": step_dispatch_reward,
                    "vpp_portfolio_reward": step_portfolio_reward,
                    "portfolio_proxy_reward": step_portfolio_reward,
                    "total_cost": reward_total_cost,
                    "violation_count": len(violations),
                    "value_estimate": float(value.detach().cpu().item()),
                    "policy_entropy": float(combined_entropy.detach().cpu().item()),
                    "portfolio_entropy": float(portfolio_entropy_sum.detach().cpu().item()),
                    "portfolio_keep_count": sum(1 for action in step_portfolio_actions.values() if action == "keep"),
                    "portfolio_reweight_count": sum(1 for action in step_portfolio_actions.values() if action == "reweight"),
                    "portfolio_membership_change_count": sum(
                        1 for action in step_portfolio_actions.values() if action == "propose_membership_change"
                    ),
                    "action_min": float(np.min(action_np)),
                    "action_mean": float(np.mean(action_np)),
                    "action_max": float(np.max(action_np)),
                    "vpp_target_action_min": float(np.min(vpp_target_np)),
                    "vpp_target_action_mean": float(np.mean(vpp_target_np)),
                    "vpp_target_action_max": float(np.max(vpp_target_np)),
                    "der_action_min": float(np.min(der_action_np)) if der_action_dim else 0.0,
                    "der_action_mean": float(np.mean(der_action_np)) if der_action_dim else 0.0,
                    "der_action_max": float(np.max(der_action_np)) if der_action_dim else 0.0,
                    "projection_clipping": clipped_now,
                    "reward_architecture": "role_specific_general_sum",
                }
            )
            for vpp_id in vpp_ids:
                trajectory_rows.append(
                    {
                        "episode": episode,
                        "step": step,
                        "agent_id": "dso_global_guidance",
                        "target_vpp_id": vpp_id,
                        "dso_target_p_mw": dso_targets[vpp_id],
                        "selected_vpp_target_p_mw": selected_targets[vpp_id],
                        "portfolio_action": step_portfolio_actions.get(vpp_id, ""),
                        "der_action_count": len(der_ids_by_vpp[vpp_id]),
                        "reward": learning_reward,
                        "environment_reward": dso_reward,
                        "vpp_dispatch_reward": float(reward_map[f"{vpp_id}_dispatch"]),
                        "vpp_portfolio_reward": float(reward_map[f"{vpp_id}_portfolio"]),
                        "total_cost": reward_total_cost,
                    }
                )
            if all(truncations.values()):
                break

        returns = torch.tensor(_discounted_returns(learning_rewards, cfg.gamma), dtype=torch.float32)
        if len(returns) > 1:
            normalized_returns = (returns - returns.mean()) / (returns.std(unbiased=False) + 1e-8)
        else:
            normalized_returns = returns
        value_tensor = torch.stack(values)
        log_prob_tensor = torch.stack(log_probs)
        entropy_tensor = torch.stack(entropies)
        advantages = normalized_returns - value_tensor.detach()

        policy_loss = -(log_prob_tensor * advantages).mean()
        value_loss = (value_tensor - normalized_returns).pow(2).mean()
        entropy_loss = -entropy_tensor.mean()
        loss = policy_loss + cfg.value_coef * value_loss + cfg.entropy_coef * entropy_loss

        optimizer.zero_grad()
        loss.backward()
        grad_norm = float(torch.nn.utils.clip_grad_norm_(policy.parameters(), cfg.max_grad_norm))
        optimizer.step()
        optimizer_steps += 1

        episode_reward = float(sum(learning_rewards))
        episode_rows.append(
            {
                "episode": episode,
                "algorithm": cfg.algorithm,
                "episode_reward": episode_reward,
                "dso_episode_reward": float(sum(dso_rewards)),
                "vpp_dispatch_episode_reward": float(sum(dispatch_rewards)),
                "vpp_portfolio_episode_reward": float(sum(portfolio_rewards)),
                "episode_cost": float(total_cost),
                "mean_step_reward": episode_reward / max(1, len(learning_rewards)),
                "violation_count": int(violation_count),
                "projection_clipping_rate": clipping_count / max(1, len(learning_rewards)),
                "portfolio_proxy_reward": float(portfolio_proxy_reward_total),
                "portfolio_keep_rate": portfolio_action_counts["keep"] / max(1, sum(portfolio_action_counts.values())),
                "portfolio_reweight_rate": portfolio_action_counts["reweight"] / max(1, sum(portfolio_action_counts.values())),
                "portfolio_membership_change_rate": portfolio_action_counts["propose_membership_change"]
                / max(1, sum(portfolio_action_counts.values())),
                "policy_loss": float(policy_loss.detach().cpu().item()),
                "value_loss": float(value_loss.detach().cpu().item()),
                "entropy": float(entropy_tensor.mean().detach().cpu().item()),
                "grad_norm": grad_norm,
                "learning_rate": cfg.learning_rate,
            }
        )
        loss_rows.append(
            {
                "episode": episode,
                "algorithm": cfg.algorithm,
                "policy_loss": float(policy_loss.detach().cpu().item()),
                "value_loss": float(value_loss.detach().cpu().item()),
                "entropy_loss": float(entropy_loss.detach().cpu().item()),
                "total_loss": float(loss.detach().cpu().item()),
                "grad_norm": grad_norm,
                "optimizer_step": optimizer_steps,
            }
        )
        env.close()

    episode_metrics = pd.DataFrame(episode_rows)
    step_metrics = pd.DataFrame(step_rows)
    trajectory = pd.DataFrame(trajectory_rows)
    loss_metrics = pd.DataFrame(loss_rows)
    final_params = torch.cat([param.detach().flatten().cpu() for param in policy.parameters()])
    param_delta_l2 = float(torch.norm(final_params - initial_params).item())

    checkpoint_path = out / "actor_critic_checkpoint.pt"
    torch.save(
        {
            "model_state_dict": policy.state_dict(),
            "config": cfg.to_dict(),
            "input_dim": input_dim,
            "action_dim": action_dim,
            "der_action_dim": der_action_dim,
            "vpp_ids": vpp_ids,
            "der_ids_by_vpp": der_ids_by_vpp,
        },
        checkpoint_path,
    )

    summary = {
        "algorithm": cfg.algorithm,
        "status": "completed",
        "is_deep_rl": True,
        "deep_learning_framework": "torch",
        "episodes": cfg.episodes,
        "horizon_steps": cfg.horizon_steps,
        "hidden_dim": cfg.hidden_dim,
        "optimizer_steps": int(optimizer_steps),
        "param_delta_l2": param_delta_l2,
        "dso_actor_trainable": True,
        "vpp_dispatch_trainable": True,
        "vpp_der_disaggregation_trainable": True,
        "dispatch_action_type": "der_level_normalized_setpoints",
        "portfolio_trainable": True,
        "portfolio_action_type": "slow_loop_commercial_configuration_proposal",
        "portfolio_physical_change_gated": True,
        "reward_architecture": "role_specific_general_sum",
        "dso_reward_type": "grid_safety_procurement_tracking",
        "vpp_dispatch_reward_type": "self_interested_settlement_delivery",
        "vpp_portfolio_reward_type": "long_horizon_profit_reliability_localized_dso_alignment",
        "portfolio_reward_coef": float(cfg.portfolio_reward_coef),
        "portfolio_proxy_reward_total": float(episode_metrics["portfolio_proxy_reward"].sum())
        if not episode_metrics.empty and "portfolio_proxy_reward" in episode_metrics
        else None,
        "portfolio_keep_rate": float(episode_metrics["portfolio_keep_rate"].mean())
        if not episode_metrics.empty and "portfolio_keep_rate" in episode_metrics
        else None,
        "portfolio_reweight_rate": float(episode_metrics["portfolio_reweight_rate"].mean())
        if not episode_metrics.empty and "portfolio_reweight_rate" in episode_metrics
        else None,
        "portfolio_membership_change_rate": float(episode_metrics["portfolio_membership_change_rate"].mean())
        if not episode_metrics.empty and "portfolio_membership_change_rate" in episode_metrics
        else None,
        "best_episode_reward": float(episode_metrics["episode_reward"].max()) if not episode_metrics.empty else None,
        "final_episode_reward": float(episode_metrics["episode_reward"].iloc[-1]) if not episode_metrics.empty else None,
        "mean_projection_clipping_rate": float(episode_metrics["projection_clipping_rate"].mean())
        if not episode_metrics.empty
        else None,
        "total_violation_count": int(episode_metrics["violation_count"].sum()) if not episode_metrics.empty else None,
        "checkpoint": str(checkpoint_path),
        "note": (
            "DSO neural actor emits envelope-preference targets. VPP dispatch actors now learn both selected "
            "aggregate VPP power and DER-level normalized setpoint proposals inside the DSO envelope. A safety "
            "projection keeps aggregate power inside FR/DOE. Rewards are role-specific: DSO uses a grid-security "
            "reward, VPP dispatch uses a self-interested settlement/delivery reward, and the slow portfolio head "
            "uses a long-horizon profit/reliability reward with localized DSO-alignment credit. Physical DER "
            "membership changes remain gated by deterministic scenario portfolio events."
        ),
    }

    episode_metrics.to_csv(out / "deep_rl_episode_metrics.csv", index=False)
    step_metrics.to_csv(out / "deep_rl_step_metrics.csv", index=False)
    trajectory.to_csv(out / "deep_rl_trajectory.csv", index=False)
    loss_metrics.to_csv(out / "deep_rl_loss_metrics.csv", index=False)
    pd.DataFrame([summary]).to_csv(out / "deep_rl_training_summary.csv", index=False)
    write_json(out / "deep_rl_training_summary.json", summary)
    write_json(out / "deep_rl_config.json", cfg.to_dict())

    return {
        "episode_metrics": episode_metrics,
        "step_metrics": step_metrics,
        "trajectory": trajectory,
        "loss_metrics": loss_metrics,
        "summary": summary,
        "output_dir": out,
        "checkpoint": checkpoint_path,
    }


def train_privacy_separated_ctde(
    config_path: str | Path | None = None,
    output_dir: str | Path = "outputs/deep_rl",
    config: PrivacySeparatedCTDEConfig | None = None,
) -> dict[str, Any]:
    """Train the privacy-separated CTDE actor-critic stack.

    This is the first executable version of the target architecture documented
    in `rl_target_ctde_architecture`: the DSO actor, VPP dispatch actor and VPP
    portfolio actor use separate execution encoders. The VPP dispatch actor is
    parameter-shared across homogeneous VPPs but receives only one VPP's local
    observation at a time. The centralized critic consumes
    `critic_global_state` for training only.
    """

    cfg = config or PrivacySeparatedCTDEConfig()
    torch, _, optim, Normal, Categorical = _require_torch()
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
    vpp_input_dim = int(
        len(encode_vpp_dispatch_observation(observations[f"{first_vpp_id}_dispatch"], max_der_per_vpp))
    )
    portfolio_input_dim = int(len(encode_vpp_portfolio_observation(observations[f"{first_vpp_id}_portfolio"])))
    critic_input_dim = int(
        len(encode_critic_global_state(build_critic_global_state(env_probe.scenario, 0), vpp_ids))
    )
    critic_action_dim = joint_action_summary_dim(len(vpp_ids))
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
    optimizer = optim.Adam(modules.parameters(), lr=cfg.learning_rate)
    initial_params = torch.cat([param.detach().flatten().cpu() for param in modules.parameters()])
    optimizer_steps = 0

    episode_rows: list[dict[str, Any]] = []
    step_rows: list[dict[str, Any]] = []
    trajectory_rows: list[dict[str, Any]] = []
    loss_rows: list[dict[str, Any]] = []

    for episode in range(cfg.episodes):
        env = MultiAgentVPPDSOEnv(config_path=config_path, horizon_steps=cfg.horizon_steps)
        observations, _ = env.reset(seed=cfg.seed + episode)
        dso_log_probs = []
        dispatch_log_probs = []
        portfolio_log_probs = []
        dso_values = []
        dispatch_values = []
        portfolio_values = []
        entropies = []
        dso_rewards: list[float] = []
        dispatch_rewards: list[float] = []
        portfolio_rewards: list[float] = []
        learning_rewards: list[float] = []
        total_cost = 0.0
        violation_count = 0
        clipping_count = 0
        portfolio_proxy_reward_total = 0.0
        portfolio_action_counts = {"keep": 0, "reweight": 0, "propose_membership_change": 0}

        for step in range(cfg.horizon_steps):
            dso_obs = observations["dso_global_guidance"]
            dso_tensor = torch.tensor(
                encode_dso_observation(dso_obs, vpp_ids),
                dtype=torch.float32,
            ).unsqueeze(0)
            critic_tensor = torch.tensor(
                encode_critic_global_state(
                    build_critic_global_state(env.scenario, env.current_step),
                    vpp_ids,
                ),
                dtype=torch.float32,
            ).unsqueeze(0)
            dso_mean, dso_log_std = modules["dso_actor"](dso_tensor)
            dso_dist = Normal(dso_mean, dso_log_std.exp())
            raw_dso_action = dso_dist.rsample()
            normalized_dso_action = torch.clamp(raw_dso_action, -cfg.action_clip, cfg.action_clip)
            dso_targets = _targets_from_normalized_actions(
                normalized_dso_action.detach().cpu().numpy().reshape(-1),
                dso_obs,
                vpp_ids,
                cfg.action_clip,
            )
            step_dso_log_prob = dso_dist.log_prob(raw_dso_action).sum(dim=-1).squeeze(0)
            step_entropy = dso_dist.entropy().sum(dim=-1).squeeze(0)
            step_dispatch_log_prob = torch.tensor(0.0, dtype=torch.float32)
            step_portfolio_log_prob = torch.tensor(0.0, dtype=torch.float32)

            action_payload: dict[str, Any] = {"dso_global_guidance": {"targets": dso_targets}}
            step_portfolio_proxy_reward = 0.0
            step_portfolio_actions: dict[str, str] = {}
            selected_targets: dict[str, float] = {}
            normalized_aggregate_actions: dict[str, float] = {}
            normalized_der_actions: dict[str, np.ndarray] = {}
            portfolio_action_indices: dict[str, int] = {}
            action_min = 0.0
            action_max = 0.0
            all_policy_actions: list[float] = list(normalized_dso_action.detach().cpu().numpy().reshape(-1))

            for vpp_id in vpp_ids:
                vpp_obs = observations[f"{vpp_id}_dispatch"]
                vpp_tensor = torch.tensor(
                    encode_vpp_dispatch_observation(vpp_obs, max_der_per_vpp),
                    dtype=torch.float32,
                ).unsqueeze(0)
                aggregate_mean, aggregate_log_std, der_mean, der_log_std = modules["vpp_dispatch_actor"](vpp_tensor)
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
                selected_targets[vpp_id] = selected_target
                der_values = normalized_der.detach().cpu().numpy().reshape(-1)
                normalized_aggregate_actions[vpp_id] = float(normalized_aggregate.detach().cpu().item())
                normalized_der_actions[vpp_id] = der_values.copy()
                der_ids = der_ids_by_vpp[vpp_id]
                der_actions = {der_id: float(der_values[index]) for index, der_id in enumerate(der_ids)}
                action_payload[f"{vpp_id}_dispatch"] = {
                    "selected_p_mw": float(selected_target),
                    "der_actions": der_actions,
                    "policy_version": cfg.algorithm,
                }
                used_der_log_prob = der_dist.log_prob(raw_der).reshape(-1)[: len(der_ids)].sum()
                used_der_entropy = der_dist.entropy().reshape(-1)[: len(der_ids)].sum()
                step_dispatch_log_prob = (
                    step_dispatch_log_prob + aggregate_dist.log_prob(raw_aggregate).sum() + used_der_log_prob
                )
                step_entropy = step_entropy + aggregate_dist.entropy().sum() + used_der_entropy
                all_policy_actions.append(float(normalized_aggregate.detach().cpu().item()))
                all_policy_actions.extend(float(value) for value in der_values[: len(der_ids)])

                portfolio_obs = observations[f"{vpp_id}_portfolio"]
                portfolio_tensor = torch.tensor(
                    encode_vpp_portfolio_observation(portfolio_obs),
                    dtype=torch.float32,
                ).unsqueeze(0)
                logits = modules["vpp_portfolio_actor"](portfolio_tensor).squeeze(0)
                portfolio_label, portfolio_log_prob, portfolio_entropy, portfolio_index = _portfolio_action_from_logits(
                    logits,
                    Categorical,
                )
                step_portfolio_log_prob = step_portfolio_log_prob + portfolio_log_prob
                step_entropy = step_entropy + portfolio_entropy
                step_portfolio_actions[vpp_id] = portfolio_label
                portfolio_action_indices[vpp_id] = int(portfolio_index)
                portfolio_action_counts[portfolio_label] += 1
                step_portfolio_proxy_reward += _portfolio_proxy_reward(portfolio_label, dso_obs, vpp_id)
                action_payload[f"{vpp_id}_portfolio"] = {
                    "action": portfolio_label,
                    "policy_version": cfg.algorithm,
                }

            action_summary_array = encode_joint_action_summary(
                normalized_dso_action=normalized_dso_action.detach().cpu().numpy().reshape(-1),
                vpp_ids=vpp_ids,
                normalized_aggregate_actions=normalized_aggregate_actions,
                normalized_der_actions=normalized_der_actions,
                portfolio_action_indices=portfolio_action_indices,
                max_der_per_vpp=max_der_per_vpp,
                action_clip=cfg.action_clip,
            )
            action_summary_tensor = torch.tensor(action_summary_array, dtype=torch.float32).unsqueeze(0)
            if hasattr(modules["centralized_critic"], "forward_heads"):
                value_heads = modules["centralized_critic"].forward_heads(
                    critic_tensor,
                    action_summary_tensor,
                )
            else:
                scalar_value = modules["centralized_critic"](critic_tensor, action_summary_tensor)
                value_heads = {
                    "dso": scalar_value,
                    "dispatch": scalar_value,
                    "portfolio": scalar_value,
                }
            dso_value = value_heads["dso"]
            dispatch_value = value_heads["dispatch"]
            portfolio_value = value_heads["portfolio"]

            if all_policy_actions:
                action_min = float(np.min(all_policy_actions))
                action_max = float(np.max(all_policy_actions))
            observations, reward_map, _, truncations, infos = env.step(action_payload)
            dso_reward = float(reward_map["dso_global_guidance"])
            reward_components = infos["dso_global_guidance"].get("reward_components", {})
            violations = infos["dso_global_guidance"].get("violations", [])
            reward_total_cost = float(reward_components.get("total_cost", -dso_reward))
            projection_audit = infos["dso_global_guidance"].get("decoded_vpp_dispatch_adjustments", {})
            agent_reward_components = {
                agent_id: info.get("agent_reward_components", {})
                for agent_id, info in infos.items()
            }
            step_dispatch_reward = float(
                np.mean([reward_map[f"{vpp_id}_dispatch"] for vpp_id in vpp_ids])
                if vpp_ids
                else 0.0
            )
            step_portfolio_reward = float(
                np.mean([reward_map[f"{vpp_id}_portfolio"] for vpp_id in vpp_ids])
                if vpp_ids
                else 0.0
            )
            projection_gap = sum(
                abs(
                    float(item.get("projected_target_p_mw", 0.0))
                    - float(item.get("dispatch_adjusted_target_p_mw", 0.0))
                )
                for item in projection_audit.values()
            )
            learning_reward = dso_reward + step_dispatch_reward + float(cfg.portfolio_reward_coef) * step_portfolio_reward
            total_cost += reward_total_cost
            portfolio_proxy_reward_total += step_portfolio_reward
            violation_count += int(len(violations))
            clipped_now = bool(action_min <= -cfg.action_clip + 1e-7 or action_max >= cfg.action_clip - 1e-7)
            clipping_count += int(clipped_now)

            dso_log_probs.append(step_dso_log_prob)
            dispatch_log_probs.append(step_dispatch_log_prob)
            portfolio_log_probs.append(step_portfolio_log_prob)
            dso_values.append(dso_value.squeeze(0))
            dispatch_values.append(dispatch_value.squeeze(0))
            portfolio_values.append(portfolio_value.squeeze(0))
            entropies.append(step_entropy)
            dso_rewards.append(dso_reward)
            dispatch_rewards.append(step_dispatch_reward)
            portfolio_rewards.append(step_portfolio_reward)
            learning_rewards.append(learning_reward)

            step_rows.append(
                {
                    "episode": episode,
                    "step": step,
                    "algorithm": cfg.algorithm,
                    "reward": learning_reward,
                    "environment_reward": dso_reward,
                    "dso_reward": dso_reward,
                    "vpp_dispatch_reward": step_dispatch_reward,
                    "vpp_portfolio_reward": step_portfolio_reward,
                    "portfolio_proxy_reward": step_portfolio_reward,
                    "projection_gap_mw": projection_gap,
                    "total_cost": reward_total_cost,
                    "violation_count": len(violations),
                    "architecture_version": architecture_meta["architecture_version"],
                    "vpp_encoder_type": architecture_meta["vpp_encoder_type"],
                    "critic_type": architecture_meta["critic_type"],
                    "critic_head_type": architecture_meta.get("critic_head_type", "single_value_head"),
                    "action_conditioned_critic": architecture_meta["action_conditioned_critic"],
                    "value_estimate": float(dso_value.detach().cpu().item()),
                    "dso_value_estimate": float(dso_value.detach().cpu().item()),
                    "dispatch_value_estimate": float(dispatch_value.detach().cpu().item()),
                    "portfolio_value_estimate": float(portfolio_value.detach().cpu().item()),
                    "critic_action_summary_dim": critic_action_dim,
                    "critic_action_summary_l2": float(np.linalg.norm(action_summary_array)),
                    "policy_entropy": float(step_entropy.detach().cpu().item()),
                    "dso_log_prob": float(step_dso_log_prob.detach().cpu().item()),
                    "vpp_dispatch_log_prob": float(step_dispatch_log_prob.detach().cpu().item()),
                    "portfolio_log_prob": float(step_portfolio_log_prob.detach().cpu().item()),
                    "portfolio_keep_count": sum(1 for action in step_portfolio_actions.values() if action == "keep"),
                    "portfolio_reweight_count": sum(1 for action in step_portfolio_actions.values() if action == "reweight"),
                    "portfolio_membership_change_count": sum(
                        1 for action in step_portfolio_actions.values() if action == "propose_membership_change"
                    ),
                    "action_min": action_min,
                    "action_max": action_max,
                    "projection_clipping": clipped_now,
                    "reward_architecture": "role_specific_general_sum",
                }
            )
            for vpp_id in vpp_ids:
                dispatch_components = agent_reward_components.get(f"{vpp_id}_dispatch", {})
                portfolio_components = agent_reward_components.get(f"{vpp_id}_portfolio", {})
                trajectory_rows.append(
                    {
                        "episode": episode,
                        "step": step,
                        "agent_id": f"{vpp_id}_dispatch",
                        "target_vpp_id": vpp_id,
                        "dso_target_p_mw": dso_targets[vpp_id],
                        "selected_vpp_target_p_mw": selected_targets[vpp_id],
                        "portfolio_action": step_portfolio_actions.get(vpp_id, ""),
                        "der_action_count": len(der_ids_by_vpp[vpp_id]),
                        "reward": learning_reward,
                        "environment_reward": dso_reward,
                        "vpp_dispatch_reward": float(reward_map[f"{vpp_id}_dispatch"]),
                        "vpp_portfolio_reward": float(reward_map[f"{vpp_id}_portfolio"]),
                        "private_profit_proxy": float(dispatch_components.get("private_profit_proxy", 0.0)),
                        "localized_dso_alignment_reward": float(
                            portfolio_components.get("localized_dso_alignment_reward", 0.0)
                        ),
                        "total_cost": reward_total_cost,
                        "privacy_scope": "own_vpp_local_observation_only",
                        "architecture_version": architecture_meta["architecture_version"],
                        "vpp_encoder_type": architecture_meta["vpp_encoder_type"],
                        "critic_type": architecture_meta["critic_type"],
                    }
                )
            if all(truncations.values()):
                break

        dso_value_tensor = torch.stack(dso_values)
        dispatch_value_tensor = torch.stack(dispatch_values)
        portfolio_value_tensor = torch.stack(portfolio_values)
        if cfg.use_gae:
            dso_returns, dso_advantages = _gae_returns_advantages(
                rewards=dso_rewards,
                values=dso_value_tensor,
                gamma=cfg.gamma,
                gae_lambda=cfg.gae_lambda,
                torch=torch,
            )
            dispatch_returns, dispatch_advantages = _gae_returns_advantages(
                rewards=dispatch_rewards,
                values=dispatch_value_tensor,
                gamma=cfg.gamma,
                gae_lambda=cfg.gae_lambda,
                torch=torch,
            )
            portfolio_returns, portfolio_advantages = _gae_returns_advantages(
                rewards=portfolio_rewards,
                values=portfolio_value_tensor,
                gamma=cfg.gamma,
                gae_lambda=cfg.gae_lambda,
                torch=torch,
            )
        else:
            dso_returns = torch.tensor(_discounted_returns(dso_rewards, cfg.gamma), dtype=torch.float32)
            dispatch_returns = torch.tensor(_discounted_returns(dispatch_rewards, cfg.gamma), dtype=torch.float32)
            portfolio_returns = torch.tensor(_discounted_returns(portfolio_rewards, cfg.gamma), dtype=torch.float32)
            dso_advantages = dso_returns - dso_value_tensor.detach()
            dispatch_advantages = dispatch_returns - dispatch_value_tensor.detach()
            portfolio_advantages = portfolio_returns - portfolio_value_tensor.detach()
        normalized_dso_returns = _normalize_tensor(dso_returns, torch)
        normalized_dispatch_returns = _normalize_tensor(dispatch_returns, torch)
        normalized_portfolio_returns = _normalize_tensor(portfolio_returns, torch)
        dso_log_prob_tensor = torch.stack(dso_log_probs)
        dispatch_log_prob_tensor = torch.stack(dispatch_log_probs)
        portfolio_log_prob_tensor = torch.stack(portfolio_log_probs)
        entropy_tensor = torch.stack(entropies)

        dso_policy_loss, dso_ratio = _ppo_clipped_policy_loss(
            log_probs=dso_log_prob_tensor,
            old_log_probs=dso_log_prob_tensor.detach(),
            advantages=dso_advantages,
            clip_ratio=cfg.ppo_clip_ratio,
            torch=torch,
        )
        vpp_dispatch_policy_loss, dispatch_ratio = _ppo_clipped_policy_loss(
            log_probs=dispatch_log_prob_tensor,
            old_log_probs=dispatch_log_prob_tensor.detach(),
            advantages=dispatch_advantages,
            clip_ratio=cfg.ppo_clip_ratio,
            torch=torch,
        )
        portfolio_policy_loss, portfolio_ratio = _ppo_clipped_policy_loss(
            log_probs=portfolio_log_prob_tensor,
            old_log_probs=portfolio_log_prob_tensor.detach(),
            advantages=portfolio_advantages,
            clip_ratio=cfg.ppo_clip_ratio,
            torch=torch,
        )
        dso_value_loss = (dso_value_tensor - normalized_dso_returns).pow(2).mean()
        dispatch_value_loss = (dispatch_value_tensor - normalized_dispatch_returns).pow(2).mean()
        portfolio_value_loss = (portfolio_value_tensor - normalized_portfolio_returns).pow(2).mean()
        value_loss = (dso_value_loss + dispatch_value_loss + portfolio_value_loss) / 3.0
        entropy_loss = -entropy_tensor.mean()
        policy_loss = (
            cfg.dso_loss_coef * dso_policy_loss
            + cfg.vpp_dispatch_loss_coef * vpp_dispatch_policy_loss
            + cfg.portfolio_loss_coef * portfolio_policy_loss
        )
        loss = policy_loss + cfg.value_coef * value_loss + cfg.entropy_coef * entropy_loss

        optimizer.zero_grad()
        loss.backward()
        grad_norm = float(torch.nn.utils.clip_grad_norm_(modules.parameters(), cfg.max_grad_norm))
        optimizer.step()
        optimizer_steps += 1

        episode_reward = float(sum(learning_rewards))
        action_total = max(1, sum(portfolio_action_counts.values()))
        episode_rows.append(
            {
                "episode": episode,
                "algorithm": cfg.algorithm,
                "episode_reward": episode_reward,
                "dso_episode_reward": float(sum(dso_rewards)),
                "vpp_dispatch_episode_reward": float(sum(dispatch_rewards)),
                "vpp_portfolio_episode_reward": float(sum(portfolio_rewards)),
                "episode_cost": float(total_cost),
                "mean_step_reward": episode_reward / max(1, len(learning_rewards)),
                "violation_count": int(violation_count),
                "projection_clipping_rate": clipping_count / max(1, len(learning_rewards)),
                "portfolio_proxy_reward": float(portfolio_proxy_reward_total),
                "portfolio_keep_rate": portfolio_action_counts["keep"] / action_total,
                "portfolio_reweight_rate": portfolio_action_counts["reweight"] / action_total,
                "portfolio_membership_change_rate": portfolio_action_counts["propose_membership_change"] / action_total,
                "policy_loss": float(policy_loss.detach().cpu().item()),
                "dso_policy_loss": float(dso_policy_loss.detach().cpu().item()),
                "vpp_dispatch_policy_loss": float(vpp_dispatch_policy_loss.detach().cpu().item()),
                "portfolio_policy_loss": float(portfolio_policy_loss.detach().cpu().item()),
                "value_loss": float(value_loss.detach().cpu().item()),
                "dso_value_loss": float(dso_value_loss.detach().cpu().item()),
                "dispatch_value_loss": float(dispatch_value_loss.detach().cpu().item()),
                "portfolio_value_loss": float(portfolio_value_loss.detach().cpu().item()),
                "gae_lambda": float(cfg.gae_lambda),
                "ppo_clip_ratio": float(cfg.ppo_clip_ratio),
                "ppo_epochs": int(cfg.ppo_epochs),
                "policy_update_rule": "mappo_happo_lite_gae_single_epoch_clipped_surrogate",
                "dso_ratio_mean": float(dso_ratio.detach().mean().cpu().item()),
                "dispatch_ratio_mean": float(dispatch_ratio.detach().mean().cpu().item()),
                "portfolio_ratio_mean": float(portfolio_ratio.detach().mean().cpu().item()),
                "entropy": float(entropy_tensor.mean().detach().cpu().item()),
                "grad_norm": grad_norm,
                "learning_rate": cfg.learning_rate,
            }
        )
        loss_rows.append(
            {
                "episode": episode,
                "algorithm": cfg.algorithm,
                "dso_policy_loss": float(dso_policy_loss.detach().cpu().item()),
                "vpp_dispatch_policy_loss": float(vpp_dispatch_policy_loss.detach().cpu().item()),
                "portfolio_policy_loss": float(portfolio_policy_loss.detach().cpu().item()),
                "policy_loss": float(policy_loss.detach().cpu().item()),
                "value_loss": float(value_loss.detach().cpu().item()),
                "dso_value_loss": float(dso_value_loss.detach().cpu().item()),
                "dispatch_value_loss": float(dispatch_value_loss.detach().cpu().item()),
                "portfolio_value_loss": float(portfolio_value_loss.detach().cpu().item()),
                "entropy_loss": float(entropy_loss.detach().cpu().item()),
                "total_loss": float(loss.detach().cpu().item()),
                "gae_lambda": float(cfg.gae_lambda),
                "ppo_clip_ratio": float(cfg.ppo_clip_ratio),
                "ppo_epochs": int(cfg.ppo_epochs),
                "policy_update_rule": "mappo_happo_lite_gae_single_epoch_clipped_surrogate",
                "dso_ratio_mean": float(dso_ratio.detach().mean().cpu().item()),
                "dispatch_ratio_mean": float(dispatch_ratio.detach().mean().cpu().item()),
                "portfolio_ratio_mean": float(portfolio_ratio.detach().mean().cpu().item()),
                "dso_advantage_mean": float(dso_advantages.detach().mean().cpu().item()),
                "dispatch_advantage_mean": float(dispatch_advantages.detach().mean().cpu().item()),
                "portfolio_advantage_mean": float(portfolio_advantages.detach().mean().cpu().item()),
                "grad_norm": grad_norm,
                "optimizer_step": optimizer_steps,
            }
        )
        env.close()

    episode_metrics = pd.DataFrame(episode_rows)
    step_metrics = pd.DataFrame(step_rows)
    trajectory = pd.DataFrame(trajectory_rows)
    loss_metrics = pd.DataFrame(loss_rows)
    final_params = torch.cat([param.detach().flatten().cpu() for param in modules.parameters()])
    param_delta_l2 = float(torch.norm(final_params - initial_params).item())

    checkpoint_path = out / "privacy_separated_ctde_checkpoint.pt"
    torch.save(
        {
            "model_state_dict": modules.state_dict(),
                "config": cfg.to_dict(),
                "dso_input_dim": dso_input_dim,
                "vpp_input_dim": vpp_input_dim,
                "portfolio_input_dim": portfolio_input_dim,
                "critic_input_dim": critic_input_dim,
                "critic_action_dim": critic_action_dim,
                "max_der_per_vpp": max_der_per_vpp,
                "vpp_ids": vpp_ids,
                "der_ids_by_vpp": der_ids_by_vpp,
                "architecture_meta": architecture_meta,
                "policy_signature": policy_signature,
        },
        checkpoint_path,
    )

    summary = {
        "algorithm": cfg.algorithm,
        "status": "completed",
        "is_deep_rl": True,
        "deep_learning_framework": "torch",
        "episodes": cfg.episodes,
        "horizon_steps": cfg.horizon_steps,
        "hidden_dim": cfg.hidden_dim,
        "architecture_version": architecture_meta["architecture_version"],
        "dso_encoder_type": "privacy_scoped_mlp",
        "vpp_encoder_type": architecture_meta["vpp_encoder_type"],
        "vpp_context_features": architecture_meta["vpp_dispatch_context_dim"],
        "der_token_features": architecture_meta["vpp_dispatch_token_dim"],
        "critic_type": architecture_meta["critic_type"],
        "critic_head_type": architecture_meta.get("critic_head_type", "single_value_head"),
        "critic_value_heads": architecture_meta.get("critic_value_heads", "dso"),
        "role_aware_value_baselines": True,
        "mappo_happo_lite_ready": bool(architecture_meta.get("mappo_happo_lite_ready", False)),
        "action_conditioned_critic": architecture_meta["action_conditioned_critic"],
        "critic_action_dim": critic_action_dim,
        "critic_action_summary_dim": architecture_meta["critic_action_summary_dim"],
        "critic_action_summary_type": architecture_meta["critic_action_summary_type"],
        "policy_update_rule": "mappo_happo_lite_gae_single_epoch_clipped_surrogate",
        "gae_lambda": float(cfg.gae_lambda),
        "ppo_clip_ratio": float(cfg.ppo_clip_ratio),
        "ppo_epochs": int(cfg.ppo_epochs),
        "use_gae": bool(cfg.use_gae),
        "optimizer_steps": int(optimizer_steps),
        "param_delta_l2": param_delta_l2,
        "dso_actor_trainable": True,
        "vpp_dispatch_trainable": True,
        "vpp_der_disaggregation_trainable": True,
        "dispatch_action_type": "privacy_scoped_der_level_normalized_setpoints",
        "portfolio_trainable": True,
        "portfolio_action_type": "slow_loop_commercial_configuration_proposal",
        "portfolio_physical_change_gated": True,
        "reward_architecture": "role_specific_general_sum",
        "dso_reward_type": "grid_safety_procurement_tracking",
        "vpp_dispatch_reward_type": "self_interested_settlement_delivery",
        "vpp_portfolio_reward_type": "long_horizon_profit_reliability_localized_dso_alignment",
        "privacy_separated_execution": True,
        "dso_vpp_shared_encoder": False,
        "homogeneous_vpp_parameter_sharing": True,
        "centralized_critic_uses_global_state": True,
        "critic_visible_to_decentralized_actors": False,
        "target_ctde_primary_trainer": True,
        "policy_signature": policy_signature,
        "reward_privacy_mode": str(env_probe.scenario.dso.reward_privacy_mode),
        "dso_input_dim": dso_input_dim,
        "vpp_input_dim": vpp_input_dim,
        "portfolio_input_dim": portfolio_input_dim,
        "critic_input_dim": critic_input_dim,
        "max_der_per_vpp": max_der_per_vpp,
        "portfolio_reward_coef": float(cfg.portfolio_reward_coef),
        "portfolio_proxy_reward_total": float(episode_metrics["portfolio_proxy_reward"].sum())
        if not episode_metrics.empty and "portfolio_proxy_reward" in episode_metrics
        else None,
        "best_episode_reward": float(episode_metrics["episode_reward"].max()) if not episode_metrics.empty else None,
        "final_episode_reward": float(episode_metrics["episode_reward"].iloc[-1]) if not episode_metrics.empty else None,
        "mean_projection_clipping_rate": float(episode_metrics["projection_clipping_rate"].mean())
        if not episode_metrics.empty
        else None,
        "total_violation_count": int(episode_metrics["violation_count"].sum()) if not episode_metrics.empty else None,
        "checkpoint": str(checkpoint_path),
        "note": (
            "Privacy-separated CTDE trainer: DSO actor, VPP local dispatch actor, VPP portfolio actor and "
            "centralized critic are separate neural modules. VPP dispatch uses a Deep Sets style DER token "
            "encoder with shared token MLP plus masked pooling over local assets instead of a flat padded MLP. "
            "Homogeneous VPP dispatch/portfolio actors share parameters for sample efficiency, but raw "
            "observations remain local. Rewards are role-specific general-sum signals: DSO reward trains the "
            "global guidance actor, VPP dispatch rewards train self-interested DER disaggregation, and VPP "
            "portfolio rewards include a localized DSO-alignment credit rather than raw global reward sharing. "
            "The centralized critic consumes critic_global_state plus a compact joint action summary for training "
            "only and is not exposed to decentralized execution. The current update rule is MAPPO/HAPPO-lite: "
            "role-aware value heads, GAE-lambda advantages and a single-epoch PPO clipped surrogate. It is not yet "
            "a full algorithm-specific MATD3/HAPPO/HASAC implementation."
        ),
    }

    episode_metrics.to_csv(out / "deep_rl_episode_metrics.csv", index=False)
    step_metrics.to_csv(out / "deep_rl_step_metrics.csv", index=False)
    trajectory.to_csv(out / "deep_rl_trajectory.csv", index=False)
    loss_metrics.to_csv(out / "deep_rl_loss_metrics.csv", index=False)
    pd.DataFrame([summary]).to_csv(out / "deep_rl_training_summary.csv", index=False)
    write_json(out / "deep_rl_training_summary.json", summary)
    write_json(out / "deep_rl_config.json", cfg.to_dict())

    return {
        "episode_metrics": episode_metrics,
        "step_metrics": step_metrics,
        "trajectory": trajectory,
        "loss_metrics": loss_metrics,
        "summary": summary,
        "output_dir": out,
        "checkpoint": checkpoint_path,
    }


def evaluate_privacy_separated_ctde_checkpoint(
    *,
    config_path: str | Path | None,
    checkpoint_path: str | Path,
    output_dir: str | Path,
    horizon_steps: int | None = None,
    seed: int = 42,
) -> dict[str, Any]:
    """Run a frozen privacy-separated CTDE checkpoint on one evaluation split.

    This function is deliberately deterministic: actor means are used instead
    of stochastic samples, and the portfolio head uses argmax. It is the bridge
    between training and benchmark evaluation, so train/eval/holdout runs can
    share the same metrics protocol instead of mixing training rollouts with
    frozen-policy evidence.
    """

    torch, _, _, _, _ = _require_torch()
    checkpoint = torch.load(Path(checkpoint_path), map_location="cpu")
    checkpoint_config = checkpoint.get("config", {})
    eval_horizon = int(horizon_steps or checkpoint_config.get("horizon_steps", 8))
    out = ensure_dir(output_dir)

    env = MultiAgentVPPDSOEnv(config_path=config_path, horizon_steps=eval_horizon)
    observations, _ = env.reset(seed=seed)
    vpp_ids = [vpp.id for vpp in env.scenario.vpps]
    eval_policy_signature = env.policy_compatibility_signature()
    checkpoint_vpp_ids = list(checkpoint.get("vpp_ids", []))
    if checkpoint_vpp_ids and checkpoint_vpp_ids != vpp_ids:
        env.close()
        raise ValueError(
            "Checkpoint VPP layout does not match evaluation scenario. "
            f"checkpoint_vpps={checkpoint_vpp_ids}, eval_vpps={vpp_ids}. "
            "Topology transfer needs an adapter or compatible policy head."
        )

    der_ids_by_vpp = {vpp.id: [der.id for der in vpp.der_list] for vpp in env.scenario.vpps}
    max_der_per_vpp = int(checkpoint.get("max_der_per_vpp", max(1, max(len(ids) for ids in der_ids_by_vpp.values()))))
    hidden_dim = int(checkpoint_config.get("hidden_dim", 64))
    modules, architecture_meta = _build_privacy_separated_networks(
        dso_input_dim=int(checkpoint["dso_input_dim"]),
        vpp_input_dim=int(checkpoint["vpp_input_dim"]),
        portfolio_input_dim=int(checkpoint["portfolio_input_dim"]),
        critic_input_dim=int(checkpoint["critic_input_dim"]),
        critic_action_dim=int(checkpoint["critic_action_dim"]),
        action_dim=len(vpp_ids),
        der_action_dim=max_der_per_vpp,
        hidden_dim=hidden_dim,
    )
    load_result = modules.load_state_dict(checkpoint["model_state_dict"], strict=False)
    modules.eval()

    step_rows: list[dict[str, Any]] = []
    trajectory_rows: list[dict[str, Any]] = []
    total_reward = 0.0
    total_cost = 0.0
    total_violations = 0
    total_projection_gap = 0.0

    with torch.no_grad():
        for step in range(eval_horizon):
            dso_obs = observations["dso_global_guidance"]
            dso_tensor = torch.tensor(
                encode_dso_observation(dso_obs, vpp_ids),
                dtype=torch.float32,
            ).unsqueeze(0)
            critic_tensor = torch.tensor(
                encode_critic_global_state(
                    build_critic_global_state(env.scenario, env.current_step),
                    vpp_ids,
                ),
                dtype=torch.float32,
            ).unsqueeze(0)
            dso_mean, _ = modules["dso_actor"](dso_tensor)
            normalized_dso_action = torch.clamp(
                dso_mean,
                -float(checkpoint_config.get("action_clip", 1.0)),
                float(checkpoint_config.get("action_clip", 1.0)),
            )
            dso_targets = _targets_from_normalized_actions(
                normalized_dso_action.cpu().numpy().reshape(-1),
                dso_obs,
                vpp_ids,
                float(checkpoint_config.get("action_clip", 1.0)),
            )

            action_payload: dict[str, Any] = {"dso_global_guidance": {"targets": dso_targets}}
            normalized_aggregate_actions: dict[str, float] = {}
            normalized_der_actions: dict[str, np.ndarray] = {}
            portfolio_action_indices: dict[str, int] = {}
            selected_targets: dict[str, float] = {}
            step_portfolio_actions: dict[str, str] = {}

            for vpp_id in vpp_ids:
                vpp_obs = observations[f"{vpp_id}_dispatch"]
                vpp_tensor = torch.tensor(
                    encode_vpp_dispatch_observation(vpp_obs, max_der_per_vpp),
                    dtype=torch.float32,
                ).unsqueeze(0)
                aggregate_mean, _, der_mean, _ = modules["vpp_dispatch_actor"](vpp_tensor)
                normalized_aggregate = torch.clamp(
                    aggregate_mean,
                    -float(checkpoint_config.get("action_clip", 1.0)),
                    float(checkpoint_config.get("action_clip", 1.0)),
                )
                normalized_der = torch.clamp(
                    der_mean,
                    -float(checkpoint_config.get("action_clip", 1.0)),
                    float(checkpoint_config.get("action_clip", 1.0)),
                )
                selected_target = _target_from_normalized_scalar(
                    float(normalized_aggregate.cpu().item()),
                    vpp_obs,
                    float(checkpoint_config.get("action_clip", 1.0)),
                )
                selected_targets[vpp_id] = selected_target
                der_values = normalized_der.cpu().numpy().reshape(-1)
                normalized_aggregate_actions[vpp_id] = float(normalized_aggregate.cpu().item())
                normalized_der_actions[vpp_id] = der_values.copy()
                der_ids = der_ids_by_vpp[vpp_id]
                action_payload[f"{vpp_id}_dispatch"] = {
                    "selected_p_mw": float(selected_target),
                    "der_actions": {
                        der_id: float(der_values[index])
                        for index, der_id in enumerate(der_ids)
                        if index < len(der_values)
                    },
                    "policy_version": "frozen_privacy_separated_ctde",
                }

                portfolio_obs = observations[f"{vpp_id}_portfolio"]
                portfolio_tensor = torch.tensor(
                    encode_vpp_portfolio_observation(portfolio_obs),
                    dtype=torch.float32,
                ).unsqueeze(0)
                logits = modules["vpp_portfolio_actor"](portfolio_tensor).squeeze(0)
                portfolio_index = int(torch.argmax(logits).item())
                portfolio_label = ("keep", "reweight", "propose_membership_change")[portfolio_index]
                portfolio_action_indices[vpp_id] = portfolio_index
                step_portfolio_actions[vpp_id] = portfolio_label
                action_payload[f"{vpp_id}_portfolio"] = {
                    "action": portfolio_label,
                    "policy_version": "frozen_privacy_separated_ctde",
                }

            action_summary_array = encode_joint_action_summary(
                normalized_dso_action=normalized_dso_action.cpu().numpy().reshape(-1),
                vpp_ids=vpp_ids,
                normalized_aggregate_actions=normalized_aggregate_actions,
                normalized_der_actions=normalized_der_actions,
                portfolio_action_indices=portfolio_action_indices,
                max_der_per_vpp=max_der_per_vpp,
                action_clip=float(checkpoint_config.get("action_clip", 1.0)),
            )
            action_summary_tensor = torch.tensor(action_summary_array, dtype=torch.float32).unsqueeze(0)
            value = modules["centralized_critic"](critic_tensor, action_summary_tensor)
            if hasattr(modules["centralized_critic"], "forward_heads"):
                value_heads = modules["centralized_critic"].forward_heads(
                    critic_tensor,
                    action_summary_tensor,
                )
                value = value_heads["dso"]

            observations, reward_map, _, truncations, infos = env.step(action_payload)
            reward = float(reward_map["dso_global_guidance"])
            reward_components = infos["dso_global_guidance"].get("reward_components", {})
            violations = infos["dso_global_guidance"].get("violations", [])
            agent_reward_components = {
                agent_id: info.get("agent_reward_components", {})
                for agent_id, info in infos.items()
            }
            projection_audit = infos["dso_global_guidance"].get("decoded_vpp_dispatch_adjustments", {})
            projection_gap = sum(
                abs(
                    float(item.get("projected_target_p_mw", 0.0))
                    - float(item.get("dispatch_adjusted_target_p_mw", 0.0))
                )
                for item in projection_audit.values()
            )
            total_reward += reward
            total_cost += float(reward_components.get("total_cost", -reward))
            total_violations += int(len(violations))
            total_projection_gap += float(projection_gap)
            step_rows.append(
                {
                    "step": int(step),
                    "algorithm": "privacy_separated_ctde_actor_critic",
                    "evaluation_mode": "frozen_deterministic_mean_policy",
                    "reward": reward,
                    "total_cost": float(reward_components.get("total_cost", -reward)),
                    "violation_count": int(len(violations)),
                    "projection_gap_mw": float(projection_gap),
                    "critic_value_estimate": float(value.cpu().item()),
                    "critic_action_summary_l2": float(np.linalg.norm(action_summary_array)),
                    "architecture_version": architecture_meta["architecture_version"],
                    "vpp_encoder_type": architecture_meta["vpp_encoder_type"],
                    "critic_type": architecture_meta["critic_type"],
                    "critic_head_type": architecture_meta.get("critic_head_type", "single_value_head"),
                }
            )
            for vpp_id in vpp_ids:
                dispatch_components = agent_reward_components.get(f"{vpp_id}_dispatch", {})
                portfolio_components = agent_reward_components.get(f"{vpp_id}_portfolio", {})
                trajectory_rows.append(
                    {
                        "step": int(step),
                        "agent_id": f"{vpp_id}_dispatch",
                        "target_vpp_id": vpp_id,
                        "dso_target_p_mw": float(dso_targets[vpp_id]),
                        "selected_vpp_target_p_mw": float(selected_targets[vpp_id]),
                        "portfolio_action": step_portfolio_actions.get(vpp_id, ""),
                        "der_action_count": len(der_ids_by_vpp[vpp_id]),
                        "vpp_dispatch_reward": float(reward_map.get(f"{vpp_id}_dispatch", 0.0)),
                        "vpp_portfolio_reward": float(reward_map.get(f"{vpp_id}_portfolio", 0.0)),
                        "private_profit_proxy": float(dispatch_components.get("private_profit_proxy", 0.0)),
                        "energy_market_revenue": float(dispatch_components.get("energy_market_revenue", 0.0)),
                        "flexibility_service_payment": float(
                            dispatch_components.get("flexibility_service_payment", 0.0)
                        ),
                        "availability_payment": float(dispatch_components.get("availability_payment", 0.0)),
                        "der_operation_cost": float(dispatch_components.get("der_operation_cost", 0.0)),
                        "localized_dso_alignment_reward": float(
                            portfolio_components.get("localized_dso_alignment_reward", 0.0)
                        ),
                        "privacy_scope": "own_vpp_local_observation_only",
                    }
                )
            if all(truncations.values()):
                break

    simulator_results = env.simulator.collect_results()
    env.simulator.export_results(out / "simulator_results")
    step_metrics = pd.DataFrame(step_rows)
    trajectory = pd.DataFrame(trajectory_rows)
    step_metrics.to_csv(out / "frozen_eval_step_metrics.csv", index=False)
    trajectory.to_csv(out / "frozen_eval_trajectory.csv", index=False)
    private_profit_series = pd.to_numeric(
        trajectory.get("private_profit_proxy", pd.Series(dtype=float)),
        errors="coerce",
    ).fillna(0.0)
    eval_total_private_profit_proxy = float(private_profit_series.sum())
    eval_positive_private_profit_step_rate = (
        float((private_profit_series > 0.0).mean()) if not private_profit_series.empty else 0.0
    )
    summary = {
        "algorithm": "privacy_separated_ctde_actor_critic",
        "evaluation_mode": "frozen_deterministic_mean_policy",
        "checkpoint": str(checkpoint_path),
        "seed": int(seed),
        "horizon_steps": int(eval_horizon),
        "total_reward": float(total_reward),
        "total_cost": float(total_cost),
        "total_violation_count": int(total_violations),
        "total_projection_gap_mw": float(total_projection_gap),
        "eval_total_private_profit_proxy": eval_total_private_profit_proxy,
        "eval_positive_private_profit_step_rate": eval_positive_private_profit_step_rate,
        "architecture_version": architecture_meta["architecture_version"],
        "vpp_encoder_type": architecture_meta["vpp_encoder_type"],
        "critic_type": architecture_meta["critic_type"],
        "critic_head_type": architecture_meta.get("critic_head_type", "single_value_head"),
        "critic_value_heads": architecture_meta.get("critic_value_heads", "dso"),
        "checkpoint_missing_keys": list(load_result.missing_keys),
        "checkpoint_unexpected_keys": list(load_result.unexpected_keys),
        "policy_signature": checkpoint.get("policy_signature", {}),
        "eval_policy_signature": eval_policy_signature,
        "reward_privacy_mode": str(env.scenario.dso.reward_privacy_mode),
    }
    pd.DataFrame([summary]).to_csv(out / "frozen_eval_summary.csv", index=False)
    write_json(out / "frozen_eval_summary.json", summary)

    return {
        "summary": summary,
        "step_metrics": step_metrics,
        "trajectory": trajectory,
        "simulator_results": simulator_results,
        "scenario": env.scenario,
        "output_dir": out,
    }


def torch_available() -> bool:
    return TORCH_AVAILABLE
