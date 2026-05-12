from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import importlib.util

import numpy as np


TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None


@dataclass(frozen=True)
class HAPPOConfig:
    algorithm: str = "happo_sequential_ctde"
    clip_ratio: float = 0.20
    entropy_coef: float = 0.01
    value_coef: float = 0.50
    max_grad_norm: float = 0.50
    role_order: tuple[str, ...] = ("dso_global_guidance", "vpp_dispatch", "vpp_portfolio")

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["role_order"] = list(self.role_order)
        return payload


@dataclass(frozen=True)
class HASACConfig:
    algorithm: str = "hasac_soft_actor_critic"
    gamma: float = 0.97
    tau: float = 0.01
    actor_learning_rate: float = 3e-4
    critic_learning_rate: float = 3e-4
    alpha_learning_rate: float = 3e-4
    hidden_dim: int = 128
    replay_capacity: int = 50_000
    batch_size: int = 128
    init_log_alpha: float = -1.0
    target_entropy: float | None = None
    seed: int = 42

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _require_torch() -> Any:
    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch is required for HAPPO/HASAC training utilities. Install torch first.")
    import torch

    return torch


def happo_sequential_surrogate_loss(
    *,
    new_log_probs_by_role: dict[str, Any],
    old_log_probs_by_role: dict[str, Any],
    advantages_by_role: dict[str, Any],
    role_order: tuple[str, ...] | list[str],
    clip_ratio: float,
    torch_module: Any | None = None,
) -> tuple[Any, dict[str, Any]]:
    """Compute HAPPO-style sequential clipped losses with importance correction.

    The correction product uses roles that have already been updated in the
    specified order. This is the key difference from a simultaneous MAPPO
    update, where every role sees the same uncorrected advantage.
    """

    torch = torch_module or _require_torch()
    correction = None
    losses: list[Any] = []
    diagnostics: dict[str, Any] = {
        "algorithm": "happo_sequential_ctde",
        "role_order": list(role_order),
        "sequential_update": True,
        "importance_correction": True,
    }
    for position, role in enumerate(role_order):
        if role not in new_log_probs_by_role:
            continue
        new_log_prob = new_log_probs_by_role[role]
        old_log_prob = old_log_probs_by_role[role].detach()
        ratio = torch.exp(new_log_prob - old_log_prob)
        advantage = advantages_by_role[role].detach()
        if correction is None:
            corrected_advantage = advantage
            correction_for_role = torch.ones_like(ratio)
        else:
            correction_for_role = correction.detach()
            corrected_advantage = advantage * correction_for_role
        clipped_ratio = torch.clamp(ratio, 1.0 - float(clip_ratio), 1.0 + float(clip_ratio))
        unclipped = ratio * corrected_advantage
        clipped = clipped_ratio * corrected_advantage
        role_loss = -torch.min(unclipped, clipped).mean()
        losses.append(role_loss)
        correction = ratio.detach() if correction is None else correction * ratio.detach()
        diagnostics[f"{role}_loss"] = role_loss
        diagnostics[f"{role}_ratio_mean"] = ratio.detach().mean()
        diagnostics[f"{role}_correction_mean"] = correction_for_role.detach().mean()
        diagnostics[f"{role}_position"] = position
    if not losses:
        raise ValueError("No matching role log probabilities were provided.")
    total_loss = torch.stack(losses).mean()
    diagnostics["total_loss"] = total_loss
    diagnostics["final_importance_correction_mean"] = correction.detach().mean() if correction is not None else None
    return total_loss, diagnostics


class HASACReplayBuffer:
    """Replay buffer for heterogeneous SAC with per-role reward heads."""

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


def build_squashed_gaussian_actor(
    *,
    obs_dim: int,
    action_dim: int,
    hidden_dim: int = 128,
    torch_module: Any | None = None,
) -> Any:
    torch = torch_module or _require_torch()
    nn = torch.nn

    class SquashedGaussianActor(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(obs_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
            )
            self.mean = nn.Linear(hidden_dim, action_dim)
            self.log_std = nn.Linear(hidden_dim, action_dim)

        def forward(self, obs: Any) -> tuple[Any, Any]:
            hidden = self.net(obs)
            return self.mean(hidden), self.log_std(hidden).clamp(-5.0, 2.0)

        def sample(self, obs: Any) -> tuple[Any, Any, Any]:
            mean, log_std = self(obs)
            std = log_std.exp()
            normal = torch.distributions.Normal(mean, std)
            raw = normal.rsample()
            action = torch.tanh(raw)
            log_prob = normal.log_prob(raw) - torch.log(1.0 - action.pow(2) + 1e-6)
            return action, log_prob.sum(dim=-1, keepdim=True), torch.tanh(mean)

    return SquashedGaussianActor()


def build_hasac_twin_soft_q(
    *,
    state_dim: int,
    joint_action_dim: int,
    output_dim: int,
    hidden_dim: int = 128,
    torch_module: Any | None = None,
) -> Any:
    torch = torch_module or _require_torch()
    nn = torch.nn

    def make_q() -> Any:
        return nn.Sequential(
            nn.Linear(state_dim + joint_action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    class HASACTwinSoftQ(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.q1 = make_q()
            self.q2 = make_q()
            self.output_dim = output_dim

        def forward(self, state: Any, joint_action: Any) -> tuple[Any, Any]:
            value_input = torch.cat([state, joint_action], dim=-1)
            return self.q1(value_input), self.q2(value_input)

        def min_q(self, state: Any, joint_action: Any) -> Any:
            q1, q2 = self(state, joint_action)
            return torch.min(q1, q2)

    return HASACTwinSoftQ()


def hasac_soft_critic_loss(
    *,
    critic: Any,
    target_critic: Any,
    state: Any,
    action: Any,
    reward: Any,
    next_state: Any,
    next_action: Any,
    next_log_prob: Any,
    done: Any,
    alpha: Any,
    gamma: float,
    torch_module: Any | None = None,
) -> tuple[Any, Any]:
    torch = torch_module or _require_torch()
    if reward.ndim == 1:
        reward = reward.unsqueeze(-1)
    with torch.no_grad():
        next_q1, next_q2 = target_critic(next_state, next_action)
        entropy_term = alpha.detach() * next_log_prob
        target_q = reward + (1.0 - done.unsqueeze(-1)) * float(gamma) * (torch.min(next_q1, next_q2) - entropy_term)
    q1, q2 = critic(state, action)
    loss = torch.nn.functional.mse_loss(q1, target_q) + torch.nn.functional.mse_loss(q2, target_q)
    return loss, target_q


def hasac_actor_alpha_loss(
    *,
    critic: Any,
    state: Any,
    sampled_action: Any,
    log_prob: Any,
    log_alpha: Any,
    target_entropy: float,
    torch_module: Any | None = None,
) -> tuple[Any, Any, Any]:
    torch = torch_module or _require_torch()
    alpha = log_alpha.exp()
    q_value = critic.min_q(state, sampled_action)
    actor_loss = (alpha.detach() * log_prob - q_value.mean(dim=-1, keepdim=True)).mean()
    alpha_loss = -(log_alpha * (log_prob + float(target_entropy)).detach()).mean()
    return actor_loss, alpha_loss, alpha


def advanced_algorithm_capability_rows() -> list[dict[str, Any]]:
    """Structured metadata consumed by reports and dashboard CSV exporters."""

    return [
        {
            "algorithm": "matd3_continuous_dispatch",
            "algorithm_zh": "MATD3 连续调度",
            "implemented_mechanisms": "replay_buffer,twin_q,target_networks,target_policy_smoothing,delayed_actor_update,per_vpp_q_heads",
            "implemented_mechanisms_zh": "经验回放、双 Q、目标网络、目标策略平滑、延迟 actor 更新、每个 VPP 独立 Q 头",
            "critic_heads": "dso_global_guidance + one dispatch Q head per VPP",
            "critic_heads_zh": "DSO 全局引导 Q 头 + 每个 VPP 一个 dispatch Q 头",
            "claim_boundary": "continuous dispatch only with per-VPP dispatch actors by default; slow discrete portfolio remains outside MATD3",
            "claim_boundary_zh": "默认使用每个 VPP 独立调度 actor，只覆盖连续调度；慢时间尺度离散组合配置不放进 MATD3",
        },
        {
            "algorithm": "happo_sequential_ctde",
            "algorithm_zh": "HAPPO 顺序 CTDE",
            "implemented_mechanisms": "sequential_role_update,importance_correction,ppo_clipped_surrogate,role_order",
            "implemented_mechanisms_zh": "按角色顺序更新、累计重要性校正、PPO 裁剪目标、固定角色顺序",
            "critic_heads": "role-aware value baselines from CTDE trainer",
            "critic_heads_zh": "来自 CTDE trainer 的角色感知 value baseline",
            "claim_boundary": "dedicated smoke-capable env trainer is present; default HAPPO uses per-VPP dispatch and per-VPP slow-loop portfolio actors",
            "claim_boundary_zh": "已有可运行 smoke 级环境训练器；默认 HAPPO 使用每个 VPP 独立调度 actor 和慢周期组合配置 actor",
        },
        {
            "algorithm": "hatrpo_trust_region_ctde",
            "algorithm_zh": "HATRPO 信赖域 CTDE",
            "implemented_mechanisms": "conjugate_gradient,fisher_vector_product,kl_trust_region,line_search,centralized_state_value",
            "implemented_mechanisms_zh": "共轭梯度、Fisher 向量积、KL 信赖域、回溯线搜索、集中式状态值函数",
            "critic_heads": "centralized state-value baselines by role; actors keep local observations",
            "critic_heads_zh": "按角色的集中式状态值函数；actor 仍只使用本地观测",
            "claim_boundary": "trust-region on-policy candidate for DSO/VPP stability; still relies on environment safety projection and post-AC validation",
            "claim_boundary_zh": "面向 DSO/VPP 稳定性的 on-policy 信赖域候选；仍依赖环境安全投影和 post-AC 校核",
        },
        {
            "algorithm": "hasac_soft_actor_critic",
            "algorithm_zh": "HASAC 最大熵 Actor-Critic",
            "implemented_mechanisms": "squashed_gaussian_actor,entropy_temperature,twin_soft_q,soft_bellman_backup,off_policy_replay",
            "implemented_mechanisms_zh": "tanh 高斯 actor、熵温度、双 soft Q、soft Bellman 备份、离策略经验回放",
            "critic_heads": "general-sum soft Q heads; compatible with DSO and per-VPP reward vectors",
            "critic_heads_zh": "general-sum soft Q 多头，可对应 DSO 和每个 VPP 的 reward 向量",
            "claim_boundary": "dedicated continuous-dispatch env trainer is present with per-VPP dispatch actors by default; slow discrete portfolio remains outside HASAC",
            "claim_boundary_zh": "已有连续调度环境训练器，默认每个 VPP 独立调度 actor；慢时间尺度离散组合配置仍不放进 HASAC",
        },
    ]


def torch_available() -> bool:
    return TORCH_AVAILABLE


__all__ = [
    "HAPPOConfig",
    "HASACConfig",
    "HASACReplayBuffer",
    "advanced_algorithm_capability_rows",
    "build_hasac_twin_soft_q",
    "build_squashed_gaussian_actor",
    "happo_sequential_surrogate_loss",
    "hasac_actor_alpha_loss",
    "hasac_soft_critic_loss",
    "torch_available",
]
