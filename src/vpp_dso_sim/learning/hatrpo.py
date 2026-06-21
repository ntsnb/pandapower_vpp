from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import importlib.util
import math
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd


TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None


@dataclass(frozen=True)
class HATRPOConfig:
    """Configuration for the minimal CTDE HATRPO trainer.

    The trainer keeps decentralized execution semantics: actors consume only
    role-local observations, while the value baseline consumes
    ``critic_global_state`` during training.
    """

    algorithm: str = "hatrpo_trust_region_ctde"
    horizon_steps: int = 8
    episodes: int = 3
    gamma: float = 0.97
    gae_lambda: float = 0.95
    max_kl: float = 0.02
    cg_iters: int = 10
    cg_damping: float = 0.10
    residual_tol: float = 1e-10
    line_search_steps: int = 8
    line_search_backtrack: float = 0.50
    value_learning_rate: float = 3e-4
    value_epochs: int = 3
    hidden_dim: int = 64
    seed: int = 42
    action_clip: float = 1.0
    entropy_coef: float = 0.0
    max_grad_norm: float = 1.0
    portfolio_decision_interval_steps: int = 24
    portfolio_force_keep_between_decisions: bool = True
    dso_shield_intervention_penalty_coef: float = 1.0
    dispatch_shield_intervention_penalty_coef: float = 1.0
    reward_scale: float = 0.01
    value_target_clip: float | None = 1_000.0
    device: str = "auto"
    dispatch_actor_encoder_type: str = "deepset_v1"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def torch_available() -> bool:
    return bool(TORCH_AVAILABLE)


def _require_torch() -> tuple[Any, Any, Any]:
    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch is required for HATRPO training utilities. Install torch first.")
    import torch
    from torch.distributions import Categorical, Normal

    return torch, Normal, Categorical


def _module_state_dict_to_cpu(module: Any) -> dict[str, Any]:
    return {
        key: value.detach().cpu().clone() if hasattr(value, "detach") else value
        for key, value in module.state_dict().items()
    }


def build_hatrpo_gaussian_policy(
    *,
    obs_dim: int,
    action_dim: int,
    hidden_dim: int = 64,
    torch_module: Any | None = None,
) -> Any:
    """Build a small Gaussian actor used by DSO and VPP dispatch roles."""

    torch = torch_module or _require_torch()[0]
    nn = torch.nn

    class GaussianRolePolicy(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.net = nn.Sequential(
                nn.LayerNorm(obs_dim),
                nn.Linear(obs_dim, hidden_dim),
                nn.Tanh(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.Tanh(),
            )
            self.mean = nn.Linear(hidden_dim, action_dim)
            self.log_std = nn.Parameter(torch.full((action_dim,), -0.7))

        def forward(self, obs: Any) -> tuple[Any, Any]:
            hidden = self.net(obs)
            mean = torch.tanh(self.mean(hidden))
            return mean, self.log_std.expand_as(mean)

    return GaussianRolePolicy()


def build_hatrpo_dispatch_policy(
    *,
    obs_dim: int,
    max_der_per_vpp: int,
    hidden_dim: int = 64,
    dispatch_actor_encoder_type: str = "deepset_v1",
    torch_module: Any | None = None,
) -> Any:
    """Build a HATRPO-compatible dispatch policy from the shared VPP actor family."""

    torch = torch_module or _require_torch()[0]
    nn = torch.nn
    from vpp_dso_sim.learning.ctde_networks import build_privacy_separated_ctde_modules

    modules, architecture_meta = build_privacy_separated_ctde_modules(
        torch=torch,
        nn=nn,
        dso_input_dim=1,
        vpp_input_dim=obs_dim,
        portfolio_input_dim=1,
        critic_input_dim=1,
        action_summary_dim=1,
        action_dim=1,
        der_action_dim=max_der_per_vpp,
        hidden_dim=hidden_dim,
        dispatch_actor_encoder_type=dispatch_actor_encoder_type,
    )
    dispatch_actor = modules["vpp_dispatch_actor"]

    class HATRPODispatchPolicy(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.dispatch_actor = dispatch_actor
            self.architecture_meta = dict(architecture_meta)

        def forward(self, obs: Any) -> tuple[Any, Any]:
            aggregate_mean, aggregate_log_std, der_mean, der_log_std = self.dispatch_actor(obs)
            mean = torch.cat([aggregate_mean, der_mean], dim=-1)
            log_std = torch.cat([aggregate_log_std, der_log_std], dim=-1)
            return mean, log_std

    return HATRPODispatchPolicy()


def build_hatrpo_categorical_policy(
    *,
    obs_dim: int,
    categories: int = 3,
    hidden_dim: int = 64,
    torch_module: Any | None = None,
) -> Any:
    """Build a categorical actor for slow-loop portfolio proposals."""

    torch = torch_module or _require_torch()[0]
    nn = torch.nn

    class CategoricalRolePolicy(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.net = nn.Sequential(
                nn.LayerNorm(obs_dim),
                nn.Linear(obs_dim, hidden_dim),
                nn.Tanh(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.Tanh(),
                nn.Linear(hidden_dim, categories),
            )

        def forward(self, obs: Any) -> Any:
            return self.net(obs)

    return CategoricalRolePolicy()


def build_hatrpo_centralized_value_critic(
    *,
    input_dim: int,
    hidden_dim: int = 64,
    output_dim: int = 3,
    torch_module: Any | None = None,
) -> Any:
    """Build the training-only centralized value baseline."""

    torch = torch_module or _require_torch()[0]
    nn = torch.nn

    return nn.Sequential(
        nn.LayerNorm(input_dim),
        nn.Linear(input_dim, hidden_dim),
        nn.Tanh(),
        nn.Linear(hidden_dim, hidden_dim),
        nn.Tanh(),
        nn.Linear(hidden_dim, output_dim),
    )


def conjugate_gradient(
    fisher_vector_product: Callable[[Any], Any],
    b: Any,
    *,
    cg_iters: int = 10,
    residual_tol: float = 1e-10,
) -> tuple[Any, dict[str, Any]]:
    """Solve ``Ax=b`` using conjugate gradients without materializing ``A``."""

    x = b.new_zeros(b.shape)
    r = b.detach().clone()
    p = r.clone()
    rdotr = torch_dot = (r * r).sum()
    iterations = 0
    for index in range(int(cg_iters)):
        fisher_p = fisher_vector_product(p)
        alpha = rdotr / ((p * fisher_p).sum() + 1e-8)
        x = x + alpha * p
        r = r - alpha * fisher_p
        new_rdotr = (r * r).sum()
        iterations = index + 1
        if float(new_rdotr.detach().cpu().item()) < float(residual_tol):
            rdotr = new_rdotr
            break
        beta = new_rdotr / (rdotr + 1e-8)
        p = r + beta * p
        rdotr = new_rdotr
    diagnostics = {
        "cg_iterations": int(iterations),
        "cg_residual": float(rdotr.detach().cpu().item()),
        "initial_residual": float(torch_dot.detach().cpu().item()),
    }
    return x, diagnostics


def _flat_params(module: Any, torch: Any) -> Any:
    params = [param.detach().reshape(-1) for param in module.parameters()]
    if not params:
        return torch.zeros(0, dtype=torch.float32)
    return torch.cat(params)


def _set_flat_params(module: Any, flat_params: Any) -> None:
    offset = 0
    for param in module.parameters():
        count = param.numel()
        value = flat_params[offset : offset + count].view_as(param)
        param.data.copy_(value)
        offset += count


def _flat_grad(
    output: Any,
    module: Any,
    torch: Any,
    *,
    retain_graph: bool = False,
    create_graph: bool = False,
) -> Any:
    params = [param for param in module.parameters() if param.requires_grad]
    grads = torch.autograd.grad(
        output,
        params,
        retain_graph=retain_graph,
        create_graph=create_graph,
        allow_unused=True,
    )
    flat: list[Any] = []
    for param, grad in zip(params, grads):
        if grad is None:
            flat.append(torch.zeros_like(param).reshape(-1))
        else:
            flat.append(grad.reshape(-1))
    if not flat:
        return torch.zeros(0, dtype=torch.float32)
    return torch.cat(flat)


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
    detached_values = values.detach()
    device = detached_values.device
    reward_tensor = torch.tensor(rewards, dtype=torch.float32, device=device)
    advantages = torch.zeros_like(reward_tensor)
    running_advantage = torch.tensor(0.0, dtype=torch.float32, device=device)
    next_value = torch.tensor(0.0, dtype=torch.float32, device=device)
    for index in range(len(rewards) - 1, -1, -1):
        delta = reward_tensor[index] + float(gamma) * next_value - detached_values[index]
        running_advantage = delta + float(gamma) * float(gae_lambda) * running_advantage
        advantages[index] = running_advantage
        next_value = detached_values[index]
    return advantages + detached_values, advantages


def _gaussian_stats(policy: Any, obs: Any, actions: Any, mask: Any | None, normal_cls: Any) -> tuple[Any, Any, Any]:
    mean, log_std = policy(obs)
    dist = normal_cls(mean, log_std.exp())
    log_prob_by_dim = dist.log_prob(actions)
    entropy_by_dim = dist.entropy()
    if mask is None:
        log_prob = log_prob_by_dim.sum(dim=-1)
        entropy = entropy_by_dim.sum(dim=-1)
    else:
        log_prob = (log_prob_by_dim * mask).sum(dim=-1)
        entropy = (entropy_by_dim * mask).sum(dim=-1)
    return log_prob, entropy, dist


def _categorical_stats(policy: Any, obs: Any, actions: Any, categorical_cls: Any) -> tuple[Any, Any, Any]:
    dist = categorical_cls(logits=policy(obs))
    return dist.log_prob(actions), dist.entropy(), dist


def hatrpo_trust_region_update(
    *,
    policy: Any,
    obs: Any,
    actions: Any,
    old_log_probs: Any,
    advantages: Any,
    max_kl: float,
    cg_iters: int,
    cg_damping: float,
    line_search_steps: int,
    line_search_backtrack: float,
    residual_tol: float = 1e-10,
    entropy_coef: float = 0.0,
    action_mask: Any | None = None,
    distribution: str = "gaussian",
    torch_module: Any | None = None,
    role: str = "policy",
) -> dict[str, Any]:
    """Apply one TRPO update using conjugate gradient and an FVP closure."""

    torch, Normal, Categorical = _require_torch()
    if torch_module is not None:
        torch = torch_module

    if obs.numel() == 0 or actions.numel() == 0:
        return {
            "role": role,
            "update_accepted": False,
            "skip_reason": "empty_batch",
            "fisher_vector_product": True,
            "conjugate_gradient": True,
        }

    if distribution not in {"gaussian", "categorical"}:
        raise ValueError(f"Unsupported HATRPO distribution: {distribution}")

    normalized_advantages = _normalize_tensor(advantages.float(), torch).detach()

    with torch.no_grad():
        if distribution == "gaussian":
            old_mean, old_log_std = policy(obs)
            old_mean = old_mean.detach()
            old_log_std = old_log_std.detach()
            old_logits = None
        else:
            old_logits = policy(obs).detach()
            old_mean = None
            old_log_std = None

    def surrogate_kl_entropy() -> tuple[Any, Any, Any, Any]:
        if distribution == "gaussian":
            log_probs, entropy, new_dist = _gaussian_stats(policy, obs, actions, action_mask, Normal)
            old_dist = Normal(old_mean, old_log_std.exp())
            kl_by_dim = torch.distributions.kl.kl_divergence(old_dist, new_dist)
            kl = kl_by_dim.sum(dim=-1).mean() if action_mask is None else (kl_by_dim * action_mask).sum(dim=-1).mean()
        else:
            log_probs, entropy, new_dist = _categorical_stats(policy, obs, actions, Categorical)
            old_dist = Categorical(logits=old_logits)
            kl = torch.distributions.kl.kl_divergence(old_dist, new_dist).mean()
        ratios = torch.exp(log_probs - old_log_probs.detach())
        surrogate = (ratios * normalized_advantages).mean() + float(entropy_coef) * entropy.mean()
        return surrogate, kl, entropy.mean(), ratios

    old_surrogate, old_kl, old_entropy, old_ratios = surrogate_kl_entropy()
    policy_gradient = _flat_grad(old_surrogate, policy, torch, retain_graph=True)
    grad_norm = float(torch.linalg.vector_norm(policy_gradient).detach().cpu().item())
    if not math.isfinite(grad_norm) or grad_norm <= 1e-12:
        return {
            "role": role,
            "update_accepted": False,
            "skip_reason": "zero_policy_gradient",
            "fisher_vector_product": True,
            "conjugate_gradient": True,
            "mean_kl": float(old_kl.detach().cpu().item()),
            "surrogate_before": float(old_surrogate.detach().cpu().item()),
            "surrogate_after": float(old_surrogate.detach().cpu().item()),
            "grad_norm": grad_norm,
        }

    def fisher_vector_product(vector: Any) -> Any:
        _, kl, _, _ = surrogate_kl_entropy()
        flat_kl_grad = _flat_grad(kl, policy, torch, create_graph=True, retain_graph=True)
        kl_vector_product = (flat_kl_grad * vector).sum()
        flat_hessian_vector = _flat_grad(kl_vector_product, policy, torch, retain_graph=True)
        return flat_hessian_vector + float(cg_damping) * vector

    step_direction, cg_diagnostics = conjugate_gradient(
        fisher_vector_product,
        policy_gradient.detach(),
        cg_iters=int(cg_iters),
        residual_tol=float(residual_tol),
    )
    fvp_step = fisher_vector_product(step_direction)
    step_curvature = (step_direction * fvp_step).sum()
    if float(step_curvature.detach().cpu().item()) <= 0.0:
        return {
            "role": role,
            "update_accepted": False,
            "skip_reason": "non_positive_curvature",
            "fisher_vector_product": True,
            "conjugate_gradient": True,
            "mean_kl": float(old_kl.detach().cpu().item()),
            "surrogate_before": float(old_surrogate.detach().cpu().item()),
            "surrogate_after": float(old_surrogate.detach().cpu().item()),
            "grad_norm": grad_norm,
            **cg_diagnostics,
        }

    step_scale = torch.sqrt(2.0 * float(max_kl) / (step_curvature + 1e-8))
    full_step = step_direction * step_scale
    expected_improvement = (policy_gradient * full_step).sum()
    old_params = _flat_params(policy, torch)
    accepted = False
    accepted_fraction = 0.0
    accepted_kl = old_kl
    accepted_surrogate = old_surrogate
    line_search_used = 0

    for search_index in range(int(line_search_steps)):
        fraction = float(line_search_backtrack) ** search_index
        _set_flat_params(policy, old_params + fraction * full_step)
        new_surrogate, new_kl, _, _ = surrogate_kl_entropy()
        improvement = new_surrogate - old_surrogate
        line_search_used = search_index + 1
        if (
            torch.isfinite(new_kl)
            and torch.isfinite(new_surrogate)
            and float(new_kl.detach().cpu().item()) <= float(max_kl)
            and float(improvement.detach().cpu().item()) >= -1e-8
        ):
            accepted = True
            accepted_fraction = fraction
            accepted_kl = new_kl.detach()
            accepted_surrogate = new_surrogate.detach()
            break

    if not accepted:
        _set_flat_params(policy, old_params)
        accepted_kl = old_kl.detach()
        accepted_surrogate = old_surrogate.detach()

    new_params = _flat_params(policy, torch)
    param_delta = torch.linalg.vector_norm(new_params - old_params)
    return {
        "role": role,
        "policy_update_rule": "trpo_conjugate_gradient_fisher_vector_product",
        "trust_region": True,
        "fisher_vector_product": True,
        "conjugate_gradient": True,
        "update_accepted": bool(accepted),
        "line_search_steps_used": int(line_search_used),
        "line_search_fraction": float(accepted_fraction),
        "mean_kl": float(accepted_kl.cpu().item()),
        "max_kl": float(max_kl),
        "surrogate_before": float(old_surrogate.detach().cpu().item()),
        "surrogate_after": float(accepted_surrogate.cpu().item()),
        "entropy_before": float(old_entropy.detach().cpu().item()),
        "ratio_mean_before": float(old_ratios.detach().mean().cpu().item()),
        "expected_improvement": float(expected_improvement.detach().cpu().item()),
        "grad_norm": grad_norm,
        "param_delta_l2": float(param_delta.detach().cpu().item()),
        **cg_diagnostics,
    }


def _portfolio_label(action_idx: int) -> str:
    return ("keep", "reweight", "propose_membership_change")[int(action_idx)]


def _portfolio_decision_step(step: int, cfg: HATRPOConfig) -> bool:
    interval = max(1, int(cfg.portfolio_decision_interval_steps))
    return int(step) % interval == 0


def _all_trainable_params(torch: Any, modules: list[Any]) -> Any:
    params = [param.detach().reshape(-1).cpu() for module in modules for param in module.parameters()]
    if not params:
        return torch.zeros(0, dtype=torch.float32)
    return torch.cat(params)


def train_hatrpo(
    config_path: str | Path | None = None,
    output_dir: str | Path = "outputs/hatrpo",
    config: HATRPOConfig | None = None,
) -> dict[str, Any]:
    """Train a minimal HATRPO CTDE policy against ``MultiAgentVPPDSOEnv``.

    Scope:
    - DSO Gaussian policy over envelope-preference targets.
    - Shared VPP dispatch Gaussian policy over aggregate target plus DER actions.
    - Shared portfolio categorical policy over keep/reweight/change proposals.
    - Centralized training-only value baseline over ``critic_global_state``.
    - TRPO trust-region updates using CG/FVP per heterogeneous role.
    """

    from vpp_dso_sim.envs.multi_agent_env import MultiAgentVPPDSOEnv
    from vpp_dso_sim.envs.observations import build_critic_global_state
    from vpp_dso_sim.learning.deep_rl import (
        _target_from_normalized_scalar,
        _targets_from_normalized_actions,
        encode_critic_global_state,
        encode_dso_observation,
        encode_joint_action_summary,
        encode_vpp_dispatch_observation,
        encode_vpp_portfolio_observation,
    )
    from vpp_dso_sim.learning.reward_contracts import shield_intervention_metrics
    from vpp_dso_sim.learning.reward_config import write_reward_config_artifacts
    from vpp_dso_sim.learning.reward_trace import dispatch_private_profit_trace_rows
    from vpp_dso_sim.learning.advanced_marl import _resolve_torch_device
    from vpp_dso_sim.utils.io import ensure_dir, write_json

    cfg = config or HATRPOConfig()
    torch, Normal, Categorical = _require_torch()
    device, device_meta = _resolve_torch_device(torch, cfg.device)
    torch.manual_seed(cfg.seed)
    if bool(device_meta["cuda_available"]):
        torch.cuda.manual_seed_all(cfg.seed)
    np.random.seed(cfg.seed)
    out = ensure_dir(output_dir)

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
    vpp_ids = [vpp.id for vpp in env_probe.scenario.vpps]
    der_ids_by_vpp = {vpp.id: [der.id for der in vpp.der_list] for vpp in env_probe.scenario.vpps}
    max_der_per_vpp = max(1, max((len(ids) for ids in der_ids_by_vpp.values()), default=1))
    dso_input_dim = int(len(encode_dso_observation(observations["dso_global_guidance"], vpp_ids)))
    first_vpp_id = vpp_ids[0]
    dispatch_input_dim = int(
        len(encode_vpp_dispatch_observation(observations[f"{first_vpp_id}_dispatch"], max_der_per_vpp))
    )
    portfolio_input_dim = int(len(encode_vpp_portfolio_observation(observations[f"{first_vpp_id}_portfolio"])))
    critic_input_dim = int(
        len(encode_critic_global_state(build_critic_global_state(env_probe.scenario, 0), vpp_ids))
    )
    policy_signature = env_probe.policy_compatibility_signature()
    env_probe.close()

    dso_policy = build_hatrpo_gaussian_policy(
        obs_dim=dso_input_dim,
        action_dim=len(vpp_ids),
        hidden_dim=cfg.hidden_dim,
        torch_module=torch,
    ).to(device)
    dispatch_policy = build_hatrpo_dispatch_policy(
        obs_dim=dispatch_input_dim,
        max_der_per_vpp=max_der_per_vpp,
        hidden_dim=cfg.hidden_dim,
        dispatch_actor_encoder_type=cfg.dispatch_actor_encoder_type,
        torch_module=torch,
    ).to(device)
    portfolio_policy = build_hatrpo_categorical_policy(
        obs_dim=portfolio_input_dim,
        categories=3,
        hidden_dim=cfg.hidden_dim,
        torch_module=torch,
    ).to(device)
    value_critic = build_hatrpo_centralized_value_critic(
        input_dim=critic_input_dim,
        hidden_dim=cfg.hidden_dim,
        output_dim=3,
        torch_module=torch,
    ).to(device)
    value_optimizer = torch.optim.Adam(value_critic.parameters(), lr=float(cfg.value_learning_rate))
    trainable_modules = [dso_policy, dispatch_policy, portfolio_policy, value_critic]
    initial_params = _all_trainable_params(torch, trainable_modules)

    episode_rows: list[dict[str, Any]] = []
    step_rows: list[dict[str, Any]] = []
    dispatch_profit_trace_rows: list[dict[str, Any]] = []
    update_rows: list[dict[str, Any]] = []
    total_actor_update_attempts = 0
    accepted_actor_updates = 0
    critic_updates = 0

    for episode in range(int(cfg.episodes)):
        env = MultiAgentVPPDSOEnv(config_path=config_path, horizon_steps=cfg.horizon_steps)
        observations, _ = env.reset(seed=cfg.seed + episode)
        rollout: dict[str, list[Any]] = {
            "critic_state": [],
            "action_summary": [],
            "dso_obs": [],
            "dso_action": [],
            "dso_log_prob": [],
            "dso_rewards": [],
            "dispatch_obs": [],
            "dispatch_action": [],
            "dispatch_mask": [],
            "dispatch_log_prob": [],
            "dispatch_rewards": [],
            "portfolio_obs": [],
            "portfolio_action": [],
            "portfolio_log_prob": [],
            "portfolio_mask": [],
            "portfolio_rewards": [],
        }
        episode_reward = 0.0
        total_cost = 0.0
        violation_count = 0
        projection_gap_total = 0.0
        shield_penalty_total = 0.0

        for step in range(int(cfg.horizon_steps)):
            dso_obs = observations["dso_global_guidance"]
            dso_obs_vec = encode_dso_observation(dso_obs, vpp_ids).astype(np.float32)
            critic_vec = encode_critic_global_state(
                build_critic_global_state(env.scenario, env.current_step),
                vpp_ids,
            ).astype(np.float32)
            dso_tensor = torch.tensor(dso_obs_vec, dtype=torch.float32, device=device).unsqueeze(0)
            with torch.no_grad():
                dso_mean, dso_log_std = dso_policy(dso_tensor)
                dso_dist = Normal(dso_mean, dso_log_std.exp())
                raw_dso_action = dso_dist.sample()
                dso_log_prob = dso_dist.log_prob(raw_dso_action).sum(dim=-1)
            normalized_dso = torch.clamp(raw_dso_action, -cfg.action_clip, cfg.action_clip)
            dso_targets = _targets_from_normalized_actions(
                normalized_dso.detach().cpu().numpy().reshape(-1),
                dso_obs,
                vpp_ids,
                cfg.action_clip,
            )
            action_payload: dict[str, Any] = {"dso_global_guidance": {"targets": dso_targets}}
            normalized_aggregate_actions: dict[str, float] = {}
            normalized_der_actions: dict[str, np.ndarray] = {}
            portfolio_action_indices: dict[str, int] = {}
            step_dispatch_obs: list[np.ndarray] = []
            step_dispatch_actions: list[np.ndarray] = []
            step_dispatch_masks: list[np.ndarray] = []
            step_dispatch_log_probs: list[float] = []
            step_portfolio_obs: list[np.ndarray] = []
            step_portfolio_actions: list[int] = []
            step_portfolio_log_probs: list[float] = []
            step_portfolio_masks: list[float] = []
            portfolio_decision_step = _portfolio_decision_step(step, cfg)

            for vpp_id in vpp_ids:
                dispatch_obs = observations[f"{vpp_id}_dispatch"]
                dispatch_obs_vec = encode_vpp_dispatch_observation(dispatch_obs, max_der_per_vpp).astype(np.float32)
                dispatch_tensor = torch.tensor(dispatch_obs_vec, dtype=torch.float32, device=device).unsqueeze(0)
                der_ids = der_ids_by_vpp[vpp_id]
                dispatch_mask_np = np.zeros(1 + max_der_per_vpp, dtype=np.float32)
                dispatch_mask_np[: 1 + len(der_ids)] = 1.0
                dispatch_mask = torch.tensor(dispatch_mask_np, dtype=torch.float32, device=device).unsqueeze(0)
                with torch.no_grad():
                    dispatch_mean, dispatch_log_std = dispatch_policy(dispatch_tensor)
                    dispatch_dist = Normal(dispatch_mean, dispatch_log_std.exp())
                    raw_dispatch = dispatch_dist.sample()
                    dispatch_log_prob = (dispatch_dist.log_prob(raw_dispatch) * dispatch_mask).sum(dim=-1)
                normalized_dispatch = torch.clamp(raw_dispatch, -cfg.action_clip, cfg.action_clip)
                dispatch_values = normalized_dispatch.detach().cpu().numpy().reshape(-1)
                selected_target = _target_from_normalized_scalar(
                    float(dispatch_values[0]),
                    dispatch_obs,
                    cfg.action_clip,
                )
                der_values = dispatch_values[1 : 1 + len(der_ids)]
                action_payload[f"{vpp_id}_dispatch"] = {
                    "selected_p_mw": float(selected_target),
                    "der_actions": {
                        der_id: float(der_values[index])
                        for index, der_id in enumerate(der_ids)
                    },
                    "policy_version": cfg.algorithm,
                }
                normalized_aggregate_actions[vpp_id] = float(dispatch_values[0])
                normalized_der_actions[vpp_id] = np.asarray(der_values, dtype=np.float32)
                step_dispatch_obs.append(dispatch_obs_vec)
                step_dispatch_actions.append(raw_dispatch.detach().cpu().numpy().reshape(-1))
                step_dispatch_masks.append(dispatch_mask_np)
                step_dispatch_log_probs.append(float(dispatch_log_prob.detach().cpu().item()))

                portfolio_obs = observations[f"{vpp_id}_portfolio"]
                portfolio_obs_vec = encode_vpp_portfolio_observation(portfolio_obs).astype(np.float32)
                portfolio_tensor = torch.tensor(portfolio_obs_vec, dtype=torch.float32, device=device).unsqueeze(0)
                if portfolio_decision_step:
                    with torch.no_grad():
                        portfolio_dist = Categorical(logits=portfolio_policy(portfolio_tensor))
                        portfolio_action = portfolio_dist.sample()
                        portfolio_log_prob = portfolio_dist.log_prob(portfolio_action)
                    action_idx = int(portfolio_action.detach().cpu().item())
                    portfolio_mask = 1.0
                    log_prob_value = float(portfolio_log_prob.detach().cpu().item())
                else:
                    action_idx = 0
                    portfolio_mask = 0.0
                    log_prob_value = 0.0
                portfolio_action_indices[vpp_id] = action_idx
                action_payload[f"{vpp_id}_portfolio"] = {
                    "action": _portfolio_label(action_idx),
                    "policy_version": cfg.algorithm
                    if portfolio_decision_step
                    else f"{cfg.algorithm}_slow_loop_hold",
                }
                step_portfolio_obs.append(portfolio_obs_vec)
                step_portfolio_actions.append(action_idx)
                step_portfolio_log_probs.append(log_prob_value)
                step_portfolio_masks.append(portfolio_mask)

            action_summary = encode_joint_action_summary(
                normalized_dso_action=normalized_dso.detach().cpu().numpy().reshape(-1),
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
            dispatch_rewards = np.asarray(
                [float(reward_map[f"{vpp_id}_dispatch"]) for vpp_id in vpp_ids],
                dtype=np.float32,
            )
            raw_dispatch_rewards = dispatch_rewards.copy()
            portfolio_rewards = np.asarray(
                [float(reward_map[f"{vpp_id}_portfolio"]) for vpp_id in vpp_ids],
                dtype=np.float32,
            )
            raw_portfolio_rewards = portfolio_rewards.copy()
            dso_attr_penalty = float(reward_components.get("dso_responsible_projection_penalty", shield_penalty))
            dispatch_attr_penalties = np.asarray(
                [
                    float(component.get("dispatch_projection_penalty", shield_penalty))
                    for component in dispatch_components
                ],
                dtype=np.float32,
            )
            dso_reward = raw_dso_reward - float(cfg.dso_shield_intervention_penalty_coef) * dso_attr_penalty
            dispatch_rewards = dispatch_rewards - float(cfg.dispatch_shield_intervention_penalty_coef) * dispatch_attr_penalties
            if cfg.portfolio_force_keep_between_decisions and not portfolio_decision_step:
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
            learning_reward = float(dso_reward + dispatch_rewards.mean() + portfolio_rewards.mean())

            rollout["critic_state"].append(critic_vec)
            rollout["action_summary"].append(action_summary)
            rollout["dso_obs"].append(dso_obs_vec)
            rollout["dso_action"].append(raw_dso_action.detach().cpu().numpy().reshape(-1))
            rollout["dso_log_prob"].append(float(dso_log_prob.detach().cpu().item()))
            rollout["dso_rewards"].append(float(dso_reward) * float(cfg.reward_scale))
            rollout["dispatch_obs"].append(np.asarray(step_dispatch_obs, dtype=np.float32))
            rollout["dispatch_action"].append(np.asarray(step_dispatch_actions, dtype=np.float32))
            rollout["dispatch_mask"].append(np.asarray(step_dispatch_masks, dtype=np.float32))
            rollout["dispatch_log_prob"].append(np.asarray(step_dispatch_log_probs, dtype=np.float32))
            rollout["dispatch_rewards"].append(dispatch_rewards * float(cfg.reward_scale))
            rollout["portfolio_obs"].append(np.asarray(step_portfolio_obs, dtype=np.float32))
            rollout["portfolio_action"].append(np.asarray(step_portfolio_actions, dtype=np.int64))
            rollout["portfolio_log_prob"].append(np.asarray(step_portfolio_log_probs, dtype=np.float32))
            rollout["portfolio_mask"].append(np.asarray(step_portfolio_masks, dtype=np.float32))
            rollout["portfolio_rewards"].append(portfolio_rewards * float(cfg.reward_scale))

            total_cost += float(reward_components.get("total_cost", -dso_reward))
            projection_gap = float(shield_metrics["shield_intervention_gap_mw"])
            projection_gap_total += projection_gap
            shield_penalty_total += shield_penalty
            violation_count += int(len(infos["dso_global_guidance"].get("violations", [])))
            episode_reward += learning_reward
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
                    "dso_reward": float(dso_reward),
                    "dso_reward_env": float(raw_dso_reward),
                    "dso_reward_train": float(dso_reward),
                    "dso_reward_critic_scaled": float(dso_reward) * float(cfg.reward_scale),
                    "mean_dispatch_reward": float(dispatch_rewards.mean()) if dispatch_rewards.size else 0.0,
                    "mean_dispatch_reward_env": float(raw_dispatch_rewards.mean()) if raw_dispatch_rewards.size else 0.0,
                    "mean_dispatch_reward_train": float(dispatch_rewards.mean()) if dispatch_rewards.size else 0.0,
                    "min_dispatch_reward_train": float(dispatch_rewards.min()) if dispatch_rewards.size else 0.0,
                    "p05_dispatch_reward_train": percentile(dispatch_rewards, 5),
                    "p95_dispatch_reward_train": percentile(dispatch_rewards, 95),
                    "mean_portfolio_reward": float(portfolio_rewards.mean()) if portfolio_rewards.size else 0.0,
                    "mean_portfolio_reward_env": float(raw_portfolio_rewards.mean()) if raw_portfolio_rewards.size else 0.0,
                    "mean_portfolio_reward_train": float(portfolio_rewards.mean()) if portfolio_rewards.size else 0.0,
                    "portfolio_decision_step": bool(portfolio_decision_step),
                    "shield_intervention_penalty": shield_penalty,
                    "projection_gap_mw": projection_gap,
                    "shield_intervention_gap_mw": projection_gap,
                    "action_projection_gap_mw": float(shield_metrics["action_projection_gap_mw"]),
                    "local_bounds_projection_gap_mw": float(shield_metrics["local_bounds_projection_gap_mw"]),
                    "ac_aware_projection_gap_mw": float(shield_metrics["ac_aware_projection_gap_mw"]),
                    "ac_certified_projection_gap_mw": float(shield_metrics["ac_certified_projection_gap_mw"]),
                    "raw_action_violation_rate": float(shield_metrics["shield_intervention_count"] > 0.0),
                    "certificate_repair_rate": float(shield_metrics["ac_certified_projection_gap_mw"] > 1e-9),
                    "raw_dso_reward_before_shield_penalty": float(raw_dso_reward),
                    "raw_dispatch_reward_before_shield_penalty": float(raw_dispatch_rewards.mean()) if raw_dispatch_rewards.size else 0.0,
                    "raw_portfolio_reward_before_decision_mask": float(raw_portfolio_rewards.mean()) if raw_portfolio_rewards.size else 0.0,
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
                    "export_revenue_total": component_mean(dispatch_components, "export_revenue_total"),
                    "pv_export_revenue_total": component_mean(dispatch_components, "pv_export_revenue_total"),
                    "mt_export_revenue_total": component_mean(dispatch_components, "mt_export_revenue_total"),
                    "storage_discharge_revenue_total": component_mean(
                        dispatch_components,
                        "storage_discharge_revenue_total",
                    ),
                    "evcs_user_revenue_total": component_mean(dispatch_components, "evcs_user_revenue_total"),
                    "import_energy_cost_total": component_mean(dispatch_components, "import_energy_cost_total"),
                    "evcs_wholesale_cost_total": component_mean(dispatch_components, "evcs_wholesale_cost_total"),
                    "storage_charge_cost_total": component_mean(dispatch_components, "storage_charge_cost_total"),
                    "hvac_energy_cost_total": component_mean(dispatch_components, "hvac_energy_cost_total"),
                    "flex_energy_cost_total": component_mean(dispatch_components, "flex_energy_cost_total"),
                    "unclassified_import_cost_total": component_mean(
                        dispatch_components,
                        "unclassified_import_cost_total",
                    ),
                    "der_operating_cost_total": component_mean(dispatch_components, "der_operating_cost_total"),
                    "battery_degradation_cost_total": component_mean(
                        dispatch_components,
                        "battery_degradation_cost_total",
                    ),
                    "comfort_cost_total": component_mean(dispatch_components, "comfort_cost_total"),
                    "unserved_penalty_total": component_mean(dispatch_components, "unserved_penalty_total"),
                    "legacy_operational_surplus_with_service_quality": component_mean(
                        dispatch_components,
                        "legacy_operational_surplus_with_service_quality",
                    ),
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
                    "service_payment": component_mean(dispatch_components, "service_payment"),
                    "contract_shortfall_mw": component_mean(dispatch_components, "contract_shortfall_mw"),
                    "contract_delivery_penalty": component_mean(dispatch_components, "contract_delivery_penalty"),
                    "availability_payment": component_mean(dispatch_components, "availability_payment"),
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
                    "post_ac_violation_count": float(reward_components.get("post_ac_violation_count", len(infos["dso_global_guidance"].get("violations", [])))),
                    "post_ac_security_penalty": float(reward_components.get("post_ac_security_penalty", 0.0)),
                    "total_cost": float(reward_components.get("total_cost", -dso_reward)),
                    "violation_count": int(len(infos["dso_global_guidance"].get("violations", []))),
                    "privacy_scope": "own_vpp_local_observation_only",
                    "critic_state_visibility": "training_only_not_actor_observation",
                    "critic_action_summary_l2": float(np.linalg.norm(action_summary)),
                }
            )
            observations = next_observations
            if all(truncations.values()):
                break

        env.close()
        if not rollout["critic_state"]:
            continue

        critic_state_tensor = torch.tensor(np.asarray(rollout["critic_state"]), dtype=torch.float32, device=device)
        value_matrix = value_critic(critic_state_tensor)
        dispatch_rewards_matrix = np.asarray(rollout["dispatch_rewards"], dtype=np.float32)
        portfolio_rewards_matrix = np.asarray(rollout["portfolio_rewards"], dtype=np.float32)

        dso_returns, dso_advantages = _gae_returns_advantages(
            rewards=[float(value) for value in rollout["dso_rewards"]],
            values=value_matrix[:, 0],
            gamma=cfg.gamma,
            gae_lambda=cfg.gae_lambda,
            torch=torch,
        )
        dispatch_returns_by_vpp: list[Any] = []
        dispatch_advantages_by_vpp: list[Any] = []
        portfolio_returns_by_vpp: list[Any] = []
        portfolio_advantages_by_vpp: list[Any] = []
        for index, _vpp_id in enumerate(vpp_ids):
            dispatch_returns, dispatch_advantages = _gae_returns_advantages(
                rewards=dispatch_rewards_matrix[:, index].astype(float).tolist(),
                values=value_matrix[:, 1],
                gamma=cfg.gamma,
                gae_lambda=cfg.gae_lambda,
                torch=torch,
            )
            portfolio_returns, portfolio_advantages = _gae_returns_advantages(
                rewards=portfolio_rewards_matrix[:, index].astype(float).tolist(),
                values=value_matrix[:, 2],
                gamma=cfg.gamma,
                gae_lambda=cfg.gae_lambda,
                torch=torch,
            )
            dispatch_returns_by_vpp.append(dispatch_returns)
            dispatch_advantages_by_vpp.append(dispatch_advantages)
            portfolio_returns_by_vpp.append(portfolio_returns)
            portfolio_advantages_by_vpp.append(portfolio_advantages)

        target_value_matrix = torch.stack(
            [
                dso_returns.detach(),
                torch.stack(dispatch_returns_by_vpp, dim=1).mean(dim=1).detach(),
                torch.stack(portfolio_returns_by_vpp, dim=1).mean(dim=1).detach(),
            ],
            dim=1,
        )
        if cfg.value_target_clip is not None:
            target_value_matrix = torch.clamp(
                target_value_matrix,
                -float(cfg.value_target_clip),
                float(cfg.value_target_clip),
            )
        critic_loss_value = 0.0
        critic_grad_norm = 0.0
        for _ in range(int(cfg.value_epochs)):
            predicted_values = value_critic(critic_state_tensor)
            critic_loss = torch.nn.functional.mse_loss(predicted_values, target_value_matrix)
            value_optimizer.zero_grad()
            critic_loss.backward()
            critic_grad_norm = float(torch.nn.utils.clip_grad_norm_(value_critic.parameters(), cfg.max_grad_norm))
            value_optimizer.step()
            critic_updates += 1
            critic_loss_value = float(critic_loss.detach().cpu().item())

        dso_diag = hatrpo_trust_region_update(
            policy=dso_policy,
            obs=torch.tensor(np.asarray(rollout["dso_obs"]), dtype=torch.float32, device=device),
            actions=torch.tensor(np.asarray(rollout["dso_action"]), dtype=torch.float32, device=device),
            old_log_probs=torch.tensor(np.asarray(rollout["dso_log_prob"]), dtype=torch.float32, device=device),
            advantages=dso_advantages.detach(),
            max_kl=cfg.max_kl,
            cg_iters=cfg.cg_iters,
            cg_damping=cfg.cg_damping,
            line_search_steps=cfg.line_search_steps,
            line_search_backtrack=cfg.line_search_backtrack,
            residual_tol=cfg.residual_tol,
            entropy_coef=cfg.entropy_coef,
            distribution="gaussian",
            torch_module=torch,
            role="dso_global_guidance",
        )
        total_actor_update_attempts += 1
        accepted_actor_updates += int(bool(dso_diag.get("update_accepted", False)))
        update_rows.append({"episode": int(episode), **dso_diag})

        dispatch_advantages_tensor = torch.stack(dispatch_advantages_by_vpp, dim=1).reshape(-1)
        dispatch_diag = hatrpo_trust_region_update(
            policy=dispatch_policy,
            obs=torch.tensor(np.asarray(rollout["dispatch_obs"]), dtype=torch.float32, device=device).reshape(
                -1,
                dispatch_input_dim,
            ),
            actions=torch.tensor(np.asarray(rollout["dispatch_action"]), dtype=torch.float32, device=device).reshape(
                -1,
                1 + max_der_per_vpp,
            ),
            old_log_probs=torch.tensor(
                np.asarray(rollout["dispatch_log_prob"]),
                dtype=torch.float32,
                device=device,
            ).reshape(-1),
            advantages=dispatch_advantages_tensor.detach(),
            max_kl=cfg.max_kl,
            cg_iters=cfg.cg_iters,
            cg_damping=cfg.cg_damping,
            line_search_steps=cfg.line_search_steps,
            line_search_backtrack=cfg.line_search_backtrack,
            residual_tol=cfg.residual_tol,
            entropy_coef=cfg.entropy_coef,
            action_mask=torch.tensor(np.asarray(rollout["dispatch_mask"]), dtype=torch.float32, device=device).reshape(
                -1,
                1 + max_der_per_vpp,
            ),
            distribution="gaussian",
            torch_module=torch,
            role="shared_vpp_dispatch",
        )
        total_actor_update_attempts += 1
        accepted_actor_updates += int(bool(dispatch_diag.get("update_accepted", False)))
        update_rows.append({"episode": int(episode), **dispatch_diag})

        portfolio_mask_tensor = torch.tensor(
            np.asarray(rollout["portfolio_mask"]),
            dtype=torch.bool,
            device=device,
        ).reshape(-1)
        portfolio_obs_tensor = torch.tensor(
            np.asarray(rollout["portfolio_obs"]),
            dtype=torch.float32,
            device=device,
        ).reshape(
            -1,
            portfolio_input_dim,
        )
        portfolio_action_tensor = torch.tensor(
            np.asarray(rollout["portfolio_action"]),
            dtype=torch.int64,
            device=device,
        ).reshape(-1)
        portfolio_log_prob_tensor = torch.tensor(
            np.asarray(rollout["portfolio_log_prob"]),
            dtype=torch.float32,
            device=device,
        ).reshape(-1)
        portfolio_advantages_tensor = torch.stack(portfolio_advantages_by_vpp, dim=1).reshape(-1)
        if bool(portfolio_mask_tensor.any().item()):
            portfolio_diag = hatrpo_trust_region_update(
                policy=portfolio_policy,
                obs=portfolio_obs_tensor[portfolio_mask_tensor],
                actions=portfolio_action_tensor[portfolio_mask_tensor],
                old_log_probs=portfolio_log_prob_tensor[portfolio_mask_tensor],
                advantages=portfolio_advantages_tensor.detach()[portfolio_mask_tensor],
                max_kl=cfg.max_kl,
                cg_iters=cfg.cg_iters,
                cg_damping=cfg.cg_damping,
                line_search_steps=cfg.line_search_steps,
                line_search_backtrack=cfg.line_search_backtrack,
                residual_tol=cfg.residual_tol,
                entropy_coef=cfg.entropy_coef,
                distribution="categorical",
                torch_module=torch,
                role="shared_vpp_portfolio",
            )
        else:
            portfolio_diag = {
                "role": "shared_vpp_portfolio",
                "update_accepted": False,
                "skip_reason": "no_portfolio_decision_samples",
                "fisher_vector_product": True,
                "conjugate_gradient": True,
            }
        total_actor_update_attempts += 1
        accepted_actor_updates += int(bool(portfolio_diag.get("update_accepted", False)))
        update_rows.append({"episode": int(episode), **portfolio_diag})

        episode_rows.append(
            {
                "episode": int(episode),
                "algorithm": cfg.algorithm,
                "episode_reward": float(episode_reward),
                "reward": float(episode_reward),
                "episode_cost": float(total_cost),
                "total_cost": float(total_cost),
                "violation_count": int(violation_count),
                "projection_gap_mw": float(projection_gap_total),
                "shield_intervention_penalty": float(shield_penalty_total),
                "critic_loss": float(critic_loss_value),
                "critic_grad_norm": float(critic_grad_norm),
                "actor_update_attempts": 3,
                "accepted_actor_updates": int(
                    bool(dso_diag.get("update_accepted", False))
                    + bool(dispatch_diag.get("update_accepted", False))
                    + bool(portfolio_diag.get("update_accepted", False))
                ),
            }
        )
        episode_trace = pd.DataFrame(
            [row for row in dispatch_profit_trace_rows if int(row.get("episode", -1)) == int(episode)]
        )
        episode_trace.to_csv(
            out / f"hatrpo_dispatch_private_profit_trace_episode_{int(episode):04d}.csv",
            index=False,
        )
        pd.DataFrame(dispatch_profit_trace_rows).to_csv(
            out / "hatrpo_dispatch_private_profit_trace.csv",
            index=False,
        )
        from vpp_dso_sim.visualization.reward_dynamic_report import write_reward_dynamic_episode_report

        write_reward_dynamic_episode_report(
            output_dir=out / "reports" / "reward_dynamic_cards",
            algorithm=cfg.algorithm,
            episode=int(episode),
            step_metrics=pd.DataFrame(
                [row for row in step_rows if int(row.get("episode", -1)) == int(episode)]
            ),
            episode_metrics=pd.DataFrame([episode_rows[-1]]) if episode_rows else pd.DataFrame(),
            dispatch_trace=episode_trace,
            update_metrics=pd.DataFrame(
                [row for row in update_rows if int(row.get("episode", -1)) == int(episode)]
            ),
        )

    final_params = _all_trainable_params(torch, trainable_modules)
    param_delta = torch.linalg.vector_norm(final_params - initial_params).detach().cpu().item()
    update_df = pd.DataFrame(update_rows)
    episode_df = pd.DataFrame(episode_rows)
    step_df = pd.DataFrame(step_rows)
    dispatch_private_profit_trace = pd.DataFrame(dispatch_profit_trace_rows)
    max_observed_kl = float(update_df["mean_kl"].dropna().max()) if "mean_kl" in update_df else 0.0
    summary = {
        "algorithm": cfg.algorithm,
        "hatrpo_complete_core": True,
        "trust_region_surrogate_update": True,
        "conjugate_gradient": True,
        "fisher_vector_product": True,
        "line_search": True,
        "max_kl": float(cfg.max_kl),
        "max_observed_kl": max_observed_kl,
        "centralized_critic_uses_global_state": True,
        "critic_visible_to_decentralized_actors": False,
        "actor_privacy_scope": "local_actor_observation",
        "ctde_observation_encoding": True,
        "dso_policy_distribution": "gaussian",
        "dispatch_policy_distribution": "gaussian",
        "dispatch_actor_encoder_type": str(cfg.dispatch_actor_encoder_type),
        "vpp_encoder_type": str(getattr(dispatch_policy, "architecture_meta", {}).get("vpp_encoder_type", "")),
        "architecture_version": str(getattr(dispatch_policy, "architecture_meta", {}).get("architecture_version", "")),
        "portfolio_policy_distribution": "categorical",
        "shared_vpp_dispatch_policy": True,
        "shared_vpp_portfolio_policy": True,
        "dispatch_action_dim": int(1 + max_der_per_vpp),
        "critic_head_type": "centralized_role_value_baseline",
        "critic_value_heads": "dso,dispatch,portfolio",
        "reward_scale": float(cfg.reward_scale),
        "requested_device": str(device_meta["requested_device"]),
        "resolved_device": str(device_meta["resolved_device"]),
        "cuda_available": bool(device_meta["cuda_available"]),
        "cuda_device_count": int(device_meta["cuda_device_count"]),
        "cuda_device_name": device_meta["cuda_device_name"],
        "device_meta": dict(device_meta),
        "reward_version": resolved_reward_config.version,
        "critic_reward_scale": float(resolved_reward_config.critic_reward_scale),
        "reward_config_hash": reward_artifacts["reward_config_hash"],
        "resolved_reward_config": resolved_reward_config.to_dict(),
        "resolved_reward_config_path": str(reward_artifacts["resolved_reward_config"]),
        "reward_config_hash_path": str(reward_artifacts["reward_config_hash_path"]),
        "shield_intervention_penalty_in_role_rewards": True,
        "dso_shield_intervention_penalty_coef": float(cfg.dso_shield_intervention_penalty_coef),
        "dispatch_shield_intervention_penalty_coef": float(cfg.dispatch_shield_intervention_penalty_coef),
        "value_target_clip": None if cfg.value_target_clip is None else float(cfg.value_target_clip),
        "actor_update_attempts": int(total_actor_update_attempts),
        "accepted_actor_updates": int(accepted_actor_updates),
        "critic_updates": int(critic_updates),
        "episodes": int(cfg.episodes),
        "horizon_steps": int(cfg.horizon_steps),
        "param_delta_l2": float(param_delta),
        "best_episode_reward": float(episode_df["reward"].max()) if not episode_df.empty else None,
        "final_episode_reward": float(episode_df["reward"].iloc[-1]) if not episode_df.empty else None,
        "vpp_count": int(len(vpp_ids)),
        "policy_signature": policy_signature,
        "checkpoint": str(out / "hatrpo_checkpoint.pt"),
        "final_checkpoint": str(out / "hatrpo_checkpoint.pt"),
        "best_checkpoint": str(out / "hatrpo_checkpoint.pt"),
        "selected_checkpoint_policy": "final_checkpoint_only",
    }

    episode_df.to_csv(out / "hatrpo_episode_metrics.csv", index=False)
    step_df.to_csv(out / "hatrpo_step_metrics.csv", index=False)
    dispatch_private_profit_trace.to_csv(out / "hatrpo_dispatch_private_profit_trace.csv", index=False)
    update_df.to_csv(out / "hatrpo_update_metrics.csv", index=False)
    write_json(out / "hatrpo_training_summary.json", summary)
    write_json(out / "hatrpo_config.json", cfg.to_dict())
    checkpoint = out / "hatrpo_checkpoint.pt"
    torch.save(
        {
            "config": cfg.to_dict(),
            "summary": summary,
            "reward_config": resolved_reward_config.to_dict(),
            "reward_config_hash": reward_artifacts["reward_config_hash"],
            "reward_version": resolved_reward_config.version,
            "model_state_dict": {
                "dso_policy": _module_state_dict_to_cpu(dso_policy),
                "dispatch_policy": _module_state_dict_to_cpu(dispatch_policy),
                "portfolio_policy": _module_state_dict_to_cpu(portfolio_policy),
                "centralized_value_critic": _module_state_dict_to_cpu(value_critic),
            },
            "architecture_meta": {
                "algorithm": cfg.algorithm,
                "actor_privacy_scope": "local_actor_observation",
                "critic_scope": "centralized_training_only",
                "critic_input_contract": "critic_global_state",
                "execution_action_contract": "dso targets + selected_p_mw + der_actions + portfolio proposal",
                "trust_region_update": "conjugate_gradient_fisher_vector_product",
                "dispatch_actor_encoder_type": str(cfg.dispatch_actor_encoder_type),
                "vpp_encoder_type": str(getattr(dispatch_policy, "architecture_meta", {}).get("vpp_encoder_type", "")),
                "dispatch_actor_architecture_version": str(
                    getattr(dispatch_policy, "architecture_meta", {}).get("architecture_version", "")
                ),
            },
            "vpp_ids": vpp_ids,
            "der_ids_by_vpp": der_ids_by_vpp,
            "max_der_per_vpp": max_der_per_vpp,
            "dso_input_dim": dso_input_dim,
            "dispatch_input_dim": dispatch_input_dim,
            "portfolio_input_dim": portfolio_input_dim,
            "critic_input_dim": critic_input_dim,
        },
        checkpoint,
    )
    return {
        "summary": summary,
        "episode_metrics": episode_df,
        "step_metrics": step_df,
        "dispatch_private_profit_trace": dispatch_private_profit_trace,
        "update_metrics": update_df,
        "checkpoint": checkpoint,
        "final_checkpoint": checkpoint,
        "best_checkpoint": checkpoint,
    }


def evaluate_hatrpo_checkpoint(
    *,
    config_path: str | Path | None,
    checkpoint_path: str | Path,
    output_dir: str | Path,
    horizon_steps: int | None = None,
    seed: int = 42,
) -> dict[str, Any]:
    """Evaluate a HATRPO checkpoint with deterministic decentralized actors."""

    from vpp_dso_sim.envs.multi_agent_env import MultiAgentVPPDSOEnv
    from vpp_dso_sim.learning.deep_rl import (
        _target_from_normalized_scalar,
        _targets_from_normalized_actions,
        encode_dso_observation,
        encode_vpp_dispatch_observation,
        encode_vpp_portfolio_observation,
    )
    from vpp_dso_sim.learning.reward_contracts import shield_intervention_metrics
    from vpp_dso_sim.learning.advanced_marl import _resolve_torch_device
    from vpp_dso_sim.utils.io import ensure_dir, write_json

    torch, _, _ = _require_torch()
    checkpoint = torch.load(Path(checkpoint_path), map_location="cpu")
    cfg = checkpoint.get("config", {})
    device, device_meta = _resolve_torch_device(torch, cfg.get("device", "auto"))
    out = ensure_dir(output_dir)
    eval_horizon = int(horizon_steps or cfg.get("horizon_steps", 8))
    vpp_ids = list(checkpoint["vpp_ids"])
    der_ids_by_vpp = dict(checkpoint["der_ids_by_vpp"])
    max_der_per_vpp = int(checkpoint["max_der_per_vpp"])
    hidden_dim = int(cfg.get("hidden_dim", 64))
    action_clip = float(cfg.get("action_clip", 1.0))

    dso_policy = build_hatrpo_gaussian_policy(
        obs_dim=int(checkpoint["dso_input_dim"]),
        action_dim=len(vpp_ids),
        hidden_dim=hidden_dim,
        torch_module=torch,
    ).to(device)
    dispatch_policy = build_hatrpo_dispatch_policy(
        obs_dim=int(checkpoint["dispatch_input_dim"]),
        max_der_per_vpp=max_der_per_vpp,
        hidden_dim=hidden_dim,
        dispatch_actor_encoder_type=str(cfg.get("dispatch_actor_encoder_type", "deepset_v1")),
        torch_module=torch,
    ).to(device)
    portfolio_policy = build_hatrpo_categorical_policy(
        obs_dim=int(checkpoint["portfolio_input_dim"]),
        categories=3,
        hidden_dim=hidden_dim,
        torch_module=torch,
    ).to(device)
    state = checkpoint["model_state_dict"]
    dso_policy.load_state_dict(state["dso_policy"])
    dispatch_policy.load_state_dict(state["dispatch_policy"])
    portfolio_policy.load_state_dict(state["portfolio_policy"])
    dso_policy.eval()
    dispatch_policy.eval()
    portfolio_policy.eval()

    def portfolio_decision_step(step: int) -> bool:
        interval = max(1, int(cfg.get("portfolio_decision_interval_steps", 24)))
        return int(step) % interval == 0

    env = MultiAgentVPPDSOEnv(config_path=config_path, horizon_steps=eval_horizon)
    observations, _ = env.reset(seed=seed)
    step_rows: list[dict[str, Any]] = []
    total_reward = 0.0
    total_cost = 0.0
    total_violations = 0
    total_projection_gap = 0.0
    total_shield_penalty = 0.0
    portfolio_action_counts = {"keep": 0, "reweight": 0, "propose_membership_change": 0}

    with torch.no_grad():
        for step in range(eval_horizon):
            dso_obs = observations["dso_global_guidance"]
            dso_vec = encode_dso_observation(dso_obs, vpp_ids).astype(np.float32)
            dso_mean, _ = dso_policy(torch.tensor(dso_vec, dtype=torch.float32, device=device).unsqueeze(0))
            dso_action = torch.clamp(dso_mean, -action_clip, action_clip).cpu().numpy().reshape(-1)
            action_payload: dict[str, Any] = {
                "dso_global_guidance": {
                    "targets": _targets_from_normalized_actions(dso_action, dso_obs, vpp_ids, action_clip)
                }
            }
            for vpp_id in vpp_ids:
                dispatch_obs = observations[f"{vpp_id}_dispatch"]
                dispatch_vec = encode_vpp_dispatch_observation(dispatch_obs, max_der_per_vpp).astype(np.float32)
                dispatch_mean, _ = dispatch_policy(
                    torch.tensor(dispatch_vec, dtype=torch.float32, device=device).unsqueeze(0)
                )
                dispatch_action = torch.clamp(dispatch_mean, -action_clip, action_clip).cpu().numpy().reshape(-1)
                der_ids = der_ids_by_vpp[vpp_id]
                action_payload[f"{vpp_id}_dispatch"] = {
                    "selected_p_mw": float(_target_from_normalized_scalar(dispatch_action[0], dispatch_obs, action_clip)),
                    "der_actions": {
                        der_id: float(dispatch_action[1 + index])
                        for index, der_id in enumerate(der_ids)
                        if 1 + index < dispatch_action.size
                    },
                    "policy_version": "frozen_hatrpo",
                }

                portfolio_obs = observations[f"{vpp_id}_portfolio"]
                if portfolio_decision_step(step):
                    portfolio_vec = encode_vpp_portfolio_observation(portfolio_obs).astype(np.float32)
                    logits = portfolio_policy(
                        torch.tensor(portfolio_vec, dtype=torch.float32, device=device).unsqueeze(0)
                    ).squeeze(0)
                    action_idx = int(torch.argmax(logits).cpu().item())
                else:
                    action_idx = 0
                action_label = _portfolio_label(action_idx)
                portfolio_action_counts[action_label] += 1
                action_payload[f"{vpp_id}_portfolio"] = {
                    "action": action_label,
                    "policy_version": "frozen_hatrpo" if portfolio_decision_step(step) else "frozen_hatrpo_slow_loop_hold",
                }

            observations, reward_map, _, truncations, infos = env.step(action_payload)
            reward_components = infos["dso_global_guidance"].get("reward_components", {})
            shield_metrics = shield_intervention_metrics(reward_components)
            violations = infos["dso_global_guidance"].get("violations", [])
            raw_dso_reward = float(reward_map["dso_global_guidance"])
            raw_dispatch_reward = float(np.mean([float(reward_map[f"{vpp_id}_dispatch"]) for vpp_id in vpp_ids])) if vpp_ids else 0.0
            raw_portfolio_reward = float(np.mean([float(reward_map[f"{vpp_id}_portfolio"]) for vpp_id in vpp_ids])) if vpp_ids else 0.0
            shield_penalty = float(shield_metrics["shield_intervention_penalty"])
            dso_reward = raw_dso_reward - float(cfg.get("dso_shield_intervention_penalty_coef", 1.0)) * shield_penalty
            dispatch_reward = raw_dispatch_reward - float(cfg.get("dispatch_shield_intervention_penalty_coef", 1.0)) * shield_penalty
            portfolio_reward = raw_portfolio_reward if portfolio_decision_step(step) else 0.0
            reward = dso_reward + dispatch_reward + portfolio_reward
            total_reward += reward
            step_cost = float(reward_components.get("total_cost", -dso_reward))
            total_cost += step_cost
            total_violations += len(violations)
            total_projection_gap += float(shield_metrics["shield_intervention_gap_mw"])
            total_shield_penalty += float(shield_metrics["shield_intervention_penalty"])
            step_rows.append(
                {
                    "step": int(step),
                    "algorithm": "hatrpo_trust_region_ctde",
                    "evaluation_mode": "frozen_mean_argmax_actor",
                    "reward": float(reward),
                    "dso_reward": dso_reward,
                    "vpp_dispatch_reward": dispatch_reward,
                    "vpp_portfolio_reward": portfolio_reward,
                    "raw_dso_reward_before_shield_penalty": float(raw_dso_reward),
                    "raw_dispatch_reward_before_shield_penalty": float(raw_dispatch_reward),
                    "raw_portfolio_reward_before_decision_mask": float(raw_portfolio_reward),
                    "raw_objective_reward": float(reward_components.get("raw_objective_reward", -float(reward_components.get("total_cost", 0.0)))),
                    "total_cost": step_cost,
                    "violation_count": int(len(violations)),
                    "post_ac_violation_count": float(reward_components.get("post_ac_violation_count", len(violations))),
                    "post_ac_voltage_violation_count": float(reward_components.get("post_ac_voltage_violation_count", 0.0)),
                    "post_ac_line_overload_count": float(reward_components.get("post_ac_line_overload_count", 0.0)),
                    "post_ac_trafo_overload_count": float(reward_components.get("post_ac_trafo_overload_count", 0.0)),
                    "post_ac_powerflow_failed": float(reward_components.get("post_ac_powerflow_failed", 0.0)),
                    "post_ac_violation_magnitude": float(reward_components.get("post_ac_violation_magnitude", 0.0)),
                    "shield_intervention_gap_mw": float(shield_metrics["shield_intervention_gap_mw"]),
                    "shield_intervention_penalty": shield_penalty,
                    "privacy_scope": "own_vpp_local_observation_only",
                }
            )
            if all(truncations.values()):
                break

    env.simulator.export_results(out / "simulator_results")
    step_metrics = pd.DataFrame(step_rows)
    step_metrics.to_csv(out / "hatrpo_frozen_eval_step_metrics.csv", index=False)
    summary = {
        "algorithm": "hatrpo_trust_region_ctde",
        "evaluation_mode": "frozen_mean_argmax_actor",
        "checkpoint": str(checkpoint_path),
        "requested_device": str(device_meta["requested_device"]),
        "resolved_device": str(device_meta["resolved_device"]),
        "cuda_available": bool(device_meta["cuda_available"]),
        "cuda_device_count": int(device_meta["cuda_device_count"]),
        "cuda_device_name": device_meta["cuda_device_name"],
        "device_meta": dict(device_meta),
        "horizon_steps": int(eval_horizon),
        "seed": int(seed),
        "total_reward": float(total_reward),
        "total_cost": float(total_cost),
        "total_violation_count": int(total_violations),
        "total_shield_intervention_gap_mw": float(total_projection_gap),
        "total_shield_intervention_penalty": float(total_shield_penalty),
        "portfolio_action_counts": portfolio_action_counts,
    }
    write_json(out / "hatrpo_frozen_eval_summary.json", summary)
    pd.DataFrame([summary]).to_csv(out / "hatrpo_frozen_eval_summary.csv", index=False)
    env.close()
    return {"summary": summary, "step_metrics": step_metrics, "output_dir": out}


__all__ = [
    "HATRPOConfig",
    "build_hatrpo_categorical_policy",
    "build_hatrpo_centralized_value_critic",
    "build_hatrpo_gaussian_policy",
    "conjugate_gradient",
    "evaluate_hatrpo_checkpoint",
    "hatrpo_trust_region_update",
    "torch_available",
    "train_hatrpo",
]
