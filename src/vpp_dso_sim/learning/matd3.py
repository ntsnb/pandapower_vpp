from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import copy
import importlib.util
import os
import sys
import time

import numpy as np
import pandas as pd

from vpp_dso_sim.envs.multi_agent_env import MultiAgentVPPDSOEnv
from vpp_dso_sim.envs.observations import build_critic_global_state
from vpp_dso_sim.learning.advanced_marl import TwinCriticSpec, build_twin_critic
from vpp_dso_sim.learning.deep_rl import (
    _build_privacy_separated_networks,
    _target_from_normalized_scalar,
    _targets_from_normalized_actions,
    encode_critic_global_state,
    encode_dso_observation,
    encode_vpp_dispatch_observation,
)
from vpp_dso_sim.learning.reward_contracts import shield_intervention_metrics
from vpp_dso_sim.utils.io import ensure_dir, write_json


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


@dataclass(frozen=True)
class MATD3Config:
    """Multi-agent TD3 configuration for the continuous DSO/VPP dispatch subproblem."""

    algorithm: str = "matd3_continuous_dispatch"
    horizon_steps: int = 24
    episodes: int = 3
    gamma: float = 0.97
    actor_learning_rate: float = 3e-4
    critic_learning_rate: float = 3e-4
    hidden_dim: int = 64
    batch_size: int = 32
    replay_capacity: int = 20_000
    warmup_steps: int = 16
    exploration_noise: float = 0.15
    policy_noise: float = 0.10
    noise_clip: float = 0.30
    policy_delay: int = 2
    tau: float = 0.01
    seed: int = 42
    action_clip: float = 1.0
    dso_actor_loss_coef: float = 1.0
    dispatch_actor_loss_coef: float = 1.0
    share_vpp_dispatch_parameters: bool = False
    dso_shield_intervention_penalty_coef: float = 1.0
    dispatch_shield_intervention_penalty_coef: float = 1.0
    reward_scale: float = 0.01
    target_q_clip: float | None = 1_000.0
    critic_grad_clip: float = 1.0
    actor_grad_clip: float = 1.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _require_torch():
    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch is required for MATD3 training. Install torch first.")
    import torch
    import torch.optim as optim

    return torch, optim


class MATD3ReplayBuffer:
    """Replay buffer for deterministic continuous DSO/VPP dispatch training."""

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


def _actor_action_tensors(
    *,
    modules: Any,
    dso_obs_tensor: Any,
    vpp_obs_tensor: Any,
    action_clip: float,
    torch: Any,
    vpp_ids: list[str] | None = None,
) -> tuple[Any, Any, Any, Any]:
    dso_mean, _ = modules["dso_actor"](dso_obs_tensor)
    dso_action = torch.clamp(dso_mean, -float(action_clip), float(action_clip))
    batch_size, n_vpps, vpp_dim = vpp_obs_tensor.shape
    if "vpp_dispatch_actor" in modules:
        flat_vpp = vpp_obs_tensor.reshape(batch_size * n_vpps, vpp_dim)
        aggregate_mean, _, der_mean, _ = modules["vpp_dispatch_actor"](flat_vpp)
        aggregate_action = torch.clamp(aggregate_mean.reshape(batch_size, n_vpps), -float(action_clip), float(action_clip))
        der_action = torch.clamp(
            der_mean.reshape(batch_size, n_vpps, -1),
            -float(action_clip),
            float(action_clip),
        )
    else:
        if vpp_ids is None:
            raise ValueError("vpp_ids are required when MATD3 uses per-VPP dispatch actors.")
        aggregate_actions: list[Any] = []
        der_actions: list[Any] = []
        for vpp_index, vpp_id in enumerate(vpp_ids):
            aggregate_mean, _, der_mean, _ = modules[f"{vpp_id}_dispatch_actor"](vpp_obs_tensor[:, vpp_index, :])
            aggregate_actions.append(torch.clamp(aggregate_mean, -float(action_clip), float(action_clip)))
            der_actions.append(torch.clamp(der_mean, -float(action_clip), float(action_clip)).unsqueeze(1))
        aggregate_action = torch.cat(aggregate_actions, dim=1)
        der_action = torch.cat(der_actions, dim=1)
    joint_action = torch.cat(
        [
            dso_action,
            aggregate_action,
            der_action.reshape(batch_size, -1),
        ],
        dim=-1,
    )
    return dso_action, aggregate_action, der_action, joint_action


def _add_clipped_noise(action: Any, *, std: float, clip: float, action_clip: float, torch: Any) -> Any:
    if std <= 0.0:
        return torch.clamp(action, -float(action_clip), float(action_clip))
    noise = torch.randn_like(action) * float(std)
    noise = torch.clamp(noise, -float(clip), float(clip))
    return torch.clamp(action + noise, -float(action_clip), float(action_clip))


def _encode_step_observations(
    observations: dict[str, dict[str, Any]],
    *,
    env: MultiAgentVPPDSOEnv,
    vpp_ids: list[str],
    max_der_per_vpp: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
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


def _payload_from_actions(
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


def _critic_loss(
    *,
    critic: Any,
    target_critic: Any,
    critic_state: Any,
    action: Any,
    next_critic_state: Any,
    target_next_action: Any,
    reward: Any,
    done: Any,
    gamma: float,
    reward_scale: float,
    target_q_clip: float | None,
    torch: Any,
) -> tuple[Any, float, float, Any]:
    with torch.no_grad():
        target_q1, target_q2 = target_critic(next_critic_state, target_next_action)
        if reward.ndim == 1:
            reward = reward.unsqueeze(-1)
        target_q = reward * float(reward_scale) + (1.0 - done.unsqueeze(-1)) * float(gamma) * torch.min(target_q1, target_q2)
        if target_q_clip is not None:
            target_q = torch.clamp(target_q, -float(target_q_clip), float(target_q_clip))
    q1, q2 = critic(critic_state, action)
    q1_head_loss = ((q1 - target_q) ** 2).mean(dim=0)
    q2_head_loss = ((q2 - target_q) ** 2).mean(dim=0)
    head_loss = q1_head_loss + q2_head_loss
    loss = head_loss.mean()
    return loss, float(q1.detach().mean().cpu().item()), float(q2.detach().mean().cpu().item()), head_loss.detach()


def train_matd3(
    config_path: str | Path | None = None,
    output_dir: str | Path = "outputs/matd3",
    config: MATD3Config | None = None,
) -> dict[str, Any]:
    """Train a true off-policy MATD3 loop for continuous DSO/VPP dispatch actions.

    Scope: DSO envelope preferences, VPP aggregate dispatch and DER-level
    continuous setpoint proposals. The discrete slow portfolio action remains
    held at `keep`; it should be trained by the HAPPO/MAPPO path rather than
    forced into deterministic TD3.
    """

    cfg = config or MATD3Config()
    torch, optim = _require_torch()
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
    vpp_input_dim = int(
        len(encode_vpp_dispatch_observation(observations[f"{vpp_ids[0]}_dispatch"], max_der_per_vpp))
    )
    critic_input_dim = int(
        len(encode_critic_global_state(build_critic_global_state(env_probe.scenario, 0), vpp_ids))
    )
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
    target_actor_modules = copy.deepcopy(actor_modules)
    critic_spec = TwinCriticSpec(
        state_dim=critic_input_dim,
        joint_action_dim=joint_action_dim,
        hidden_dims=(cfg.hidden_dim, cfg.hidden_dim),
        output_dim=1 + len(vpp_ids),
        algorithm_style="matd3_centralized_twin_q",
        input_contract="critic_global_state + differentiable_flat_joint_continuous_actions",
    )
    critic_head_names = ["dso_global_guidance", *[f"{vpp_id}_dispatch" for vpp_id in vpp_ids]]
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
    role_critic_optimizer = optim.Adam(role_critic.parameters(), lr=float(cfg.critic_learning_rate))
    replay = MATD3ReplayBuffer(capacity=cfg.replay_capacity, seed=cfg.seed)

    episode_rows: list[dict[str, Any]] = []
    step_rows: list[dict[str, Any]] = []
    update_rows: list[dict[str, Any]] = []
    trajectory_rows: list[dict[str, Any]] = []
    total_env_steps = 0
    critic_updates = 0
    actor_updates = 0
    best_episode_reward = float("-inf")
    best_episode_index = -1
    best_checkpoint_state: dict[str, Any] | None = None

    last_progress_print = 0.0
    progress_interval_seconds = 60.0
    episode_iter, has_tqdm_progress = _episode_progress(range(cfg.episodes), total=cfg.episodes, desc="MATD3")
    for episode in episode_iter:
        env = MultiAgentVPPDSOEnv(config_path=config_path, horizon_steps=cfg.horizon_steps)
        observations, _ = env.reset(seed=cfg.seed + episode)
        dso_obs_vec, vpp_obs_mat, critic_vec = _encode_step_observations(
            observations,
            env=env,
            vpp_ids=vpp_ids,
            max_der_per_vpp=max_der_per_vpp,
        )
        episode_reward = 0.0
        dso_episode_reward = 0.0
        dispatch_episode_reward = 0.0
        total_cost = 0.0
        violation_count = 0
        projection_gap_total = 0.0
        ac_projection_gap_total = 0.0
        local_projection_gap_total = 0.0
        shield_intervention_penalty_total = 0.0
        shield_intervention_count = 0

        for step in range(cfg.horizon_steps):
            with torch.no_grad():
                dso_tensor = torch.tensor(dso_obs_vec, dtype=torch.float32).unsqueeze(0)
                vpp_tensor = torch.tensor(vpp_obs_mat, dtype=torch.float32).unsqueeze(0)
                dso_action_t, aggregate_t, der_t, joint_t = _actor_action_tensors(
                    modules=actor_modules,
                    dso_obs_tensor=dso_tensor,
                    vpp_obs_tensor=vpp_tensor,
                    action_clip=cfg.action_clip,
                    torch=torch,
                    vpp_ids=vpp_ids,
                )
                dso_action_t = _add_clipped_noise(
                    dso_action_t,
                    std=cfg.exploration_noise,
                    clip=cfg.noise_clip,
                    action_clip=cfg.action_clip,
                    torch=torch,
                )
                aggregate_t = _add_clipped_noise(
                    aggregate_t,
                    std=cfg.exploration_noise,
                    clip=cfg.noise_clip,
                    action_clip=cfg.action_clip,
                    torch=torch,
                )
                der_t = _add_clipped_noise(
                    der_t,
                    std=cfg.exploration_noise,
                    clip=cfg.noise_clip,
                    action_clip=cfg.action_clip,
                    torch=torch,
                )
                joint_t = torch.cat([dso_action_t, aggregate_t, der_t.reshape(1, -1)], dim=-1)

            dso_action = dso_action_t.cpu().numpy().reshape(-1)
            aggregate_actions = aggregate_t.cpu().numpy().reshape(len(vpp_ids))
            der_actions = der_t.cpu().numpy().reshape(len(vpp_ids), max_der_per_vpp)
            action_payload = _payload_from_actions(
                dso_action=dso_action,
                aggregate_actions=aggregate_actions,
                der_actions=der_actions,
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
            raw_vpp_dispatch_rewards = np.asarray(
                [float(reward_map[f"{vpp_id}_dispatch"]) for vpp_id in vpp_ids],
                dtype=np.float32,
            )
            dso_reward = raw_dso_reward - float(cfg.dso_shield_intervention_penalty_coef) * shield_penalty
            vpp_dispatch_rewards = (
                raw_vpp_dispatch_rewards - float(cfg.dispatch_shield_intervention_penalty_coef) * shield_penalty
            )
            dispatch_reward = float(np.mean(vpp_dispatch_rewards) if len(vpp_dispatch_rewards) else 0.0)
            learning_reward = dso_reward + dispatch_reward
            next_dso_obs_vec, next_vpp_obs_mat, next_critic_vec = _encode_step_observations(
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
                    "joint_action": joint_t.cpu().numpy().reshape(-1),
                    "next_dso_obs": next_dso_obs_vec,
                    "next_vpp_obs": next_vpp_obs_mat,
                    "next_critic_state": next_critic_vec,
                    "dso_reward": dso_reward,
                    "dispatch_reward": dispatch_reward,
                    "role_rewards": np.concatenate(
                        [np.asarray([dso_reward], dtype=np.float32), vpp_dispatch_rewards],
                        axis=0,
                    ),
                    "shield_intervention_gap_mw": float(shield_metrics["shield_intervention_gap_mw"]),
                    "shield_intervention_penalty": shield_penalty,
                    "done": float(done),
                }
            )
            episode_reward += learning_reward
            dso_episode_reward += dso_reward
            dispatch_episode_reward += dispatch_reward
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
                    "reward": float(learning_reward),
                    "dso_reward": float(dso_reward),
                    "vpp_dispatch_reward": float(dispatch_reward),
                    "total_cost": float(reward_components.get("total_cost", -dso_reward)),
                    "violation_count": int(len(violations)),
                    "projection_gap_mw": float(projection_gap),
                    "decoded_projection_gap_mw": float(decoded_projection_gap),
                    "action_projection_gap_mw": float(shield_metrics["action_projection_gap_mw"]),
                    "local_bounds_projection_gap_mw": float(shield_metrics["local_bounds_projection_gap_mw"]),
                    "ac_aware_projection_gap_mw": float(shield_metrics["ac_aware_projection_gap_mw"]),
                    "shield_intervention_gap_mw": float(shield_metrics["shield_intervention_gap_mw"]),
                    "shield_intervention_penalty": shield_penalty,
                    "shield_intervention_count": int(shield_metrics["shield_intervention_count"] > 0.0),
                    "raw_dso_reward_before_shield_penalty": raw_dso_reward,
                    "replay_size": int(len(replay)),
                    "joint_action_dim": int(joint_action_dim),
                    "action_min": float(joint_t.min().cpu().item()),
                    "action_max": float(joint_t.max().cpu().item()),
                    "portfolio_policy": "held_keep_not_matd3_discrete",
                }
            )
            for vpp_id in vpp_ids:
                trajectory_rows.append(
                    {
                        "episode": int(episode),
                        "step": int(step),
                        "agent_id": f"{vpp_id}_dispatch",
                        "target_vpp_id": vpp_id,
                        "algorithm": cfg.algorithm,
                        "dso_action": float(dso_action[vpp_ids.index(vpp_id)]),
                        "aggregate_action": float(aggregate_actions[vpp_ids.index(vpp_id)]),
                        "der_action_count": len(der_ids_by_vpp[vpp_id]),
                        "privacy_scope": "own_vpp_local_observation_only",
                    }
                )

            if len(replay) >= int(cfg.batch_size) and total_env_steps >= int(cfg.warmup_steps):
                batch = replay.sample(cfg.batch_size)
                critic_state_b = torch.tensor(batch["critic_state"], dtype=torch.float32)
                action_b = torch.tensor(batch["joint_action"], dtype=torch.float32)
                next_critic_state_b = torch.tensor(batch["next_critic_state"], dtype=torch.float32)
                next_dso_obs_b = torch.tensor(batch["next_dso_obs"], dtype=torch.float32)
                next_vpp_obs_b = torch.tensor(batch["next_vpp_obs"], dtype=torch.float32)
                role_reward_b = torch.tensor(batch["role_rewards"], dtype=torch.float32)
                shield_gap_b = torch.tensor(batch["shield_intervention_gap_mw"], dtype=torch.float32)
                shield_penalty_b = torch.tensor(batch["shield_intervention_penalty"], dtype=torch.float32)
                done_b = torch.tensor(batch["done"], dtype=torch.float32)

                with torch.no_grad():
                    _, _, _, next_action_b = _actor_action_tensors(
                        modules=target_actor_modules,
                        dso_obs_tensor=next_dso_obs_b,
                        vpp_obs_tensor=next_vpp_obs_b,
                        action_clip=cfg.action_clip,
                        torch=torch,
                        vpp_ids=vpp_ids,
                    )
                    next_action_b = _add_clipped_noise(
                        next_action_b,
                        std=cfg.policy_noise,
                        clip=cfg.noise_clip,
                        action_clip=cfg.action_clip,
                        torch=torch,
                    )

                role_loss, role_q1_mean, role_q2_mean, role_head_loss = _critic_loss(
                    critic=role_critic,
                    target_critic=target_role_critic,
                    critic_state=critic_state_b,
                    action=action_b,
                    next_critic_state=next_critic_state_b,
                    target_next_action=next_action_b,
                    reward=role_reward_b,
                    done=done_b,
                    gamma=cfg.gamma,
                    reward_scale=cfg.reward_scale,
                    target_q_clip=cfg.target_q_clip,
                    torch=torch,
                )
                role_critic_optimizer.zero_grad()
                role_loss.backward()
                critic_grad_norm = float(torch.nn.utils.clip_grad_norm_(role_critic.parameters(), cfg.critic_grad_clip))
                role_critic_optimizer.step()
                critic_updates += 1

                actor_loss_value = None
                dso_actor_loss_value = None
                dispatch_actor_loss_value = None
                dso_actor_grad_norm = None
                dispatch_actor_grad_norm = None
                if critic_updates % int(cfg.policy_delay) == 0:
                    dso_obs_b = torch.tensor(batch["dso_obs"], dtype=torch.float32)
                    vpp_obs_b = torch.tensor(batch["vpp_obs"], dtype=torch.float32)
                    current_dso_action_b, current_aggregate_action_b, current_der_action_b, _ = _actor_action_tensors(
                        modules=actor_modules,
                        dso_obs_tensor=dso_obs_b,
                        vpp_obs_tensor=vpp_obs_b,
                        action_clip=cfg.action_clip,
                        torch=torch,
                        vpp_ids=vpp_ids,
                    )
                    batch_size = int(current_dso_action_b.shape[0])
                    dso_joint_action_b = torch.cat(
                        [
                            current_dso_action_b,
                            current_aggregate_action_b.detach(),
                            current_der_action_b.detach().reshape(batch_size, -1),
                        ],
                        dim=-1,
                    )
                    current_q_heads = role_critic.q1_value(critic_state_b, dso_joint_action_b)
                    dso_actor_loss = -current_q_heads[:, 0].mean()
                    dso_actor_optimizer.zero_grad()
                    (float(cfg.dso_actor_loss_coef) * dso_actor_loss).backward()
                    dso_actor_grad_norm = float(
                        torch.nn.utils.clip_grad_norm_(actor_modules["dso_actor"].parameters(), cfg.actor_grad_clip)
                    )
                    dso_actor_optimizer.step()

                    current_dso_action_b_2, current_aggregate_action_b_2, current_der_action_b_2, _ = _actor_action_tensors(
                        modules=actor_modules,
                        dso_obs_tensor=dso_obs_b,
                        vpp_obs_tensor=vpp_obs_b,
                        action_clip=cfg.action_clip,
                        torch=torch,
                        vpp_ids=vpp_ids,
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
                    dispatch_actor_loss = (
                        -dispatch_q_heads[:, 1:].mean()
                        if dispatch_q_heads.shape[1] > 1
                        else torch.tensor(0.0, dtype=torch.float32)
                    )
                    dispatch_actor_optimizer.zero_grad()
                    (float(cfg.dispatch_actor_loss_coef) * dispatch_actor_loss).backward()
                    dispatch_actor_grad_norm = float(
                        torch.nn.utils.clip_grad_norm_(dispatch_actor_params, cfg.actor_grad_clip)
                    )
                    dispatch_actor_optimizer.step()
                    actor_loss = (
                        float(cfg.dso_actor_loss_coef) * dso_actor_loss
                        + float(cfg.dispatch_actor_loss_coef) * dispatch_actor_loss
                    )
                    _soft_update(actor_modules, target_actor_modules, cfg.tau)
                    _soft_update(role_critic, target_role_critic, cfg.tau)
                    actor_updates += 1
                    actor_loss_value = float(actor_loss.detach().cpu().item())
                    dso_actor_loss_value = float(dso_actor_loss.detach().cpu().item())
                    dispatch_actor_loss_value = float(dispatch_actor_loss.detach().cpu().item())

                role_head_loss_np = role_head_loss.cpu().numpy().reshape(-1)
                dso_head_loss = float(role_head_loss_np[0]) if len(role_head_loss_np) else float("nan")
                dispatch_head_loss = (
                    float(np.mean(role_head_loss_np[1:])) if len(role_head_loss_np) > 1 else 0.0
                )
                update_rows.append(
                    {
                        "global_step": int(total_env_steps),
                        "critic_update": int(critic_updates),
                        "actor_update": int(actor_updates),
                        "role_critic_loss": float(role_loss.detach().cpu().item()),
                        "dso_critic_loss": dso_head_loss,
                        "dispatch_critic_loss": dispatch_head_loss,
                        "actor_loss": actor_loss_value,
                        "dso_actor_loss": dso_actor_loss_value,
                        "dispatch_actor_loss": dispatch_actor_loss_value,
                        "critic_grad_norm": critic_grad_norm,
                        "dso_actor_grad_norm": dso_actor_grad_norm,
                        "dispatch_actor_grad_norm": dispatch_actor_grad_norm,
                        "reward_scale": float(cfg.reward_scale),
                        "target_q_clip": None if cfg.target_q_clip is None else float(cfg.target_q_clip),
                        "dso_dispatch_actor_objectives_separated": True,
                        "role_q1_mean": role_q1_mean,
                        "role_q2_mean": role_q2_mean,
                        "dso_q1_mean": float(role_q1_mean),
                        "dso_q2_mean": float(role_q2_mean),
                        "dispatch_q1_mean": float(role_q1_mean),
                        "dispatch_q2_mean": float(role_q2_mean),
                        "critic_head_count": int(len(critic_head_names)),
                        "critic_head_names": ",".join(critic_head_names),
                        "per_vpp_dispatch_q_heads": True,
                        "policy_delay": int(cfg.policy_delay),
                        "target_network_tau": float(cfg.tau),
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
                "dso_episode_reward": float(dso_episode_reward),
                "vpp_dispatch_episode_reward": float(dispatch_episode_reward),
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
            }
        )
        if float(episode_reward) > best_episode_reward:
            best_episode_reward = float(episode_reward)
            best_episode_index = int(episode)
            best_checkpoint_state = {
                "actor_state_dict": copy.deepcopy(actor_modules.state_dict()),
                "target_actor_state_dict": copy.deepcopy(target_actor_modules.state_dict()),
                "role_critic_state_dict": copy.deepcopy(role_critic.state_dict()),
                "target_role_critic_state_dict": copy.deepcopy(target_role_critic.state_dict()),
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
                "[MATD3] "
                f"episode={episode + 1}/{cfg.episodes} "
                f"reward={episode_reward:.4f} "
                f"cost={total_cost:.4f} "
                f"violations={violation_count} "
                f"projection_gap_mw={projection_gap_total:.6f} "
                f"replay={len(replay)} "
                f"critic_updates={critic_updates} "
                f"actor_updates={actor_updates}",
                flush=True,
            )
        env.close()

    episode_metrics = pd.DataFrame(episode_rows)
    step_metrics = pd.DataFrame(step_rows)
    update_metrics = pd.DataFrame(update_rows)
    trajectory = pd.DataFrame(trajectory_rows)
    checkpoint_path = out / "matd3_checkpoint.pt"
    best_checkpoint_path = out / "matd3_best_checkpoint.pt"

    def checkpoint_payload(
        actor_state: Any,
        target_actor_state: Any,
        critic_state: Any,
        target_critic_state: Any,
    ) -> dict[str, Any]:
        return {
            "config": cfg.to_dict(),
            "actor_state_dict": actor_state,
            "target_actor_state_dict": target_actor_state,
            "role_critic_state_dict": critic_state,
            "target_role_critic_state_dict": target_critic_state,
            "dso_input_dim": dso_input_dim,
            "vpp_input_dim": vpp_input_dim,
            "critic_input_dim": critic_input_dim,
            "joint_action_dim": joint_action_dim,
            "max_der_per_vpp": max_der_per_vpp,
            "vpp_ids": vpp_ids,
            "der_ids_by_vpp": der_ids_by_vpp,
            "policy_signature": policy_signature,
            "critic_spec": critic_spec.to_dict(),
            "critic_head_names": critic_head_names,
            "architecture_meta": architecture_meta,
            "selection_metric": "episode_reward",
        }

    torch.save(
        checkpoint_payload(
            actor_modules.state_dict(),
            target_actor_modules.state_dict(),
            role_critic.state_dict(),
            target_role_critic.state_dict(),
        ),
        checkpoint_path,
    )
    if best_checkpoint_state is None:
        best_checkpoint_state = {
            "actor_state_dict": copy.deepcopy(actor_modules.state_dict()),
            "target_actor_state_dict": copy.deepcopy(target_actor_modules.state_dict()),
            "role_critic_state_dict": copy.deepcopy(role_critic.state_dict()),
            "target_role_critic_state_dict": copy.deepcopy(target_role_critic.state_dict()),
        }
        best_episode_index = int(episode_metrics["episode"].iloc[-1]) if not episode_metrics.empty else -1
    torch.save(
        checkpoint_payload(
            best_checkpoint_state["actor_state_dict"],
            best_checkpoint_state["target_actor_state_dict"],
            best_checkpoint_state["role_critic_state_dict"],
            best_checkpoint_state["target_role_critic_state_dict"],
        ),
        best_checkpoint_path,
    )
    summary = {
        "algorithm": cfg.algorithm,
        "status": "completed",
        "is_deep_rl": True,
        "deep_learning_framework": "torch",
        "training_pattern": "off_policy_ctde",
        "matd3_complete_core": True,
        "continuous_control_scope": "dso_envelope_vpp_aggregate_der_dispatch",
        "portfolio_scope": "held_keep_discrete_slow_loop_not_matd3",
        "episodes": int(cfg.episodes),
        "horizon_steps": int(cfg.horizon_steps),
        "total_env_steps": int(total_env_steps),
        "critic_updates": int(critic_updates),
        "actor_updates": int(actor_updates),
        "batch_size": int(cfg.batch_size),
        "replay_capacity": int(cfg.replay_capacity),
        "final_replay_size": int(len(replay)),
        "policy_delay": int(cfg.policy_delay),
        "target_network_tau": float(cfg.tau),
        "twin_critics": True,
        "target_networks": True,
        "target_policy_smoothing": True,
        "delayed_actor_updates": True,
        "reward_scale": float(cfg.reward_scale),
        "target_q_clip": None if cfg.target_q_clip is None else float(cfg.target_q_clip),
        "critic_grad_clip": float(cfg.critic_grad_clip),
        "actor_grad_clip": float(cfg.actor_grad_clip),
        "dso_dispatch_actor_objectives_separated": True,
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
        "mean_dispatch_reward_critic": False,
        "per_vpp_dispatch_q_heads": True,
        "per_vpp_dispatch_actors": not bool(cfg.share_vpp_dispatch_parameters),
        "shared_dispatch_parameters": bool(cfg.share_vpp_dispatch_parameters),
        "general_sum_reward_heads": True,
        "critic_head_type": "role_multi_head_twin_q",
        "critic_head_count": int(len(critic_head_names)),
        "critic_head_names": ",".join(critic_head_names),
        "dso_input_dim": int(dso_input_dim),
        "vpp_input_dim": int(vpp_input_dim),
        "critic_input_dim": int(critic_input_dim),
        "joint_action_dim": int(joint_action_dim),
        "max_der_per_vpp": int(max_der_per_vpp),
        "best_episode_reward": float(episode_metrics["episode_reward"].max()) if not episode_metrics.empty else None,
        "final_episode_reward": float(episode_metrics["episode_reward"].iloc[-1]) if not episode_metrics.empty else None,
        "total_violation_count": int(episode_metrics["violation_count"].sum()) if not episode_metrics.empty else 0,
        "final_checkpoint": str(checkpoint_path),
        "best_checkpoint": str(best_checkpoint_path),
        "best_checkpoint_episode": int(best_episode_index),
        "selected_checkpoint_policy": "train_best_episode_reward",
        "checkpoint": str(best_checkpoint_path),
        "claim_boundary": (
            "This is a full MATD3-style off-policy implementation for the continuous DSO/VPP dispatch subproblem: "
            "centralized multi-head twin Q critics, per-VPP dispatch reward heads, target networks, replay buffer, "
            "target policy smoothing and delayed actor updates are present. The discrete slow portfolio action remains "
            "outside MATD3 and should be trained by HAPPO/MAPPO-style stochastic policies."
        ),
    }
    episode_metrics.to_csv(out / "matd3_episode_metrics.csv", index=False)
    step_metrics.to_csv(out / "matd3_step_metrics.csv", index=False)
    update_metrics.to_csv(out / "matd3_update_metrics.csv", index=False)
    trajectory.to_csv(out / "matd3_trajectory.csv", index=False)
    pd.DataFrame([summary]).to_csv(out / "matd3_training_summary.csv", index=False)
    write_json(out / "matd3_training_summary.json", summary)
    write_json(out / "matd3_config.json", cfg.to_dict())
    return {
        "summary": summary,
        "episode_metrics": episode_metrics,
        "step_metrics": step_metrics,
        "update_metrics": update_metrics,
        "trajectory": trajectory,
        "checkpoint": best_checkpoint_path,
        "final_checkpoint": checkpoint_path,
        "best_checkpoint": best_checkpoint_path,
        "output_dir": out,
    }


def evaluate_matd3_checkpoint(
    *,
    config_path: str | Path | None,
    checkpoint_path: str | Path,
    output_dir: str | Path,
    horizon_steps: int | None = None,
    seed: int = 42,
) -> dict[str, Any]:
    torch, _ = _require_torch()
    checkpoint = torch.load(Path(checkpoint_path), map_location="cpu")
    cfg = checkpoint.get("config", {})
    eval_horizon = int(horizon_steps or cfg.get("horizon_steps", 24))
    out = ensure_dir(output_dir)
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

    env = MultiAgentVPPDSOEnv(config_path=config_path, horizon_steps=eval_horizon)
    observations, _ = env.reset(seed=seed)
    total_reward = 0.0
    total_cost = 0.0
    total_violations = 0
    step_rows: list[dict[str, Any]] = []
    with torch.no_grad():
        for step in range(eval_horizon):
            dso_obs_vec, vpp_obs_mat, _ = _encode_step_observations(
                observations,
                env=env,
                vpp_ids=vpp_ids,
                max_der_per_vpp=max_der_per_vpp,
            )
            dso_action_t, aggregate_t, der_t, joint_t = _actor_action_tensors(
                modules=actor_modules,
                dso_obs_tensor=torch.tensor(dso_obs_vec, dtype=torch.float32).unsqueeze(0),
                vpp_obs_tensor=torch.tensor(vpp_obs_mat, dtype=torch.float32).unsqueeze(0),
                action_clip=float(cfg.get("action_clip", 1.0)),
                torch=torch,
                vpp_ids=vpp_ids,
            )
            action_payload = _payload_from_actions(
                dso_action=dso_action_t.cpu().numpy().reshape(-1),
                aggregate_actions=aggregate_t.cpu().numpy().reshape(len(vpp_ids)),
                der_actions=der_t.cpu().numpy().reshape(len(vpp_ids), max_der_per_vpp),
                dso_obs=observations["dso_global_guidance"],
                vpp_observations=observations,
                vpp_ids=vpp_ids,
                der_ids_by_vpp=der_ids_by_vpp,
                action_clip=float(cfg.get("action_clip", 1.0)),
                policy_version="frozen_matd3",
            )
            observations, reward_map, _, truncations, infos = env.step(action_payload)
            reward_components = infos["dso_global_guidance"].get("reward_components", {})
            shield_metrics = shield_intervention_metrics(reward_components)
            shield_penalty = float(shield_metrics["shield_intervention_penalty"])
            raw_dso_reward = float(reward_map["dso_global_guidance"])
            raw_dispatch_reward = float(
                np.mean([reward_map[f"{vpp_id}_dispatch"] for vpp_id in vpp_ids])
                if vpp_ids
                else 0.0
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
                    "algorithm": "matd3_continuous_dispatch",
                    "evaluation_mode": "frozen_deterministic_actor",
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
                    "joint_action_l2": float(torch.linalg.vector_norm(joint_t).cpu().item()),
                }
            )
            if all(truncations.values()):
                break
    env.simulator.export_results(out / "simulator_results")
    step_metrics = pd.DataFrame(step_rows)
    step_metrics.to_csv(out / "matd3_frozen_eval_step_metrics.csv", index=False)
    summary = {
        "algorithm": "matd3_continuous_dispatch",
        "evaluation_mode": "frozen_deterministic_actor",
        "checkpoint": str(checkpoint_path),
        "horizon_steps": int(eval_horizon),
        "seed": int(seed),
        "total_reward": float(total_reward),
        "total_cost": float(total_cost),
        "total_violation_count": int(total_violations),
    }
    write_json(out / "matd3_frozen_eval_summary.json", summary)
    pd.DataFrame([summary]).to_csv(out / "matd3_frozen_eval_summary.csv", index=False)
    env.close()
    return {"summary": summary, "step_metrics": step_metrics, "output_dir": out}


def torch_available() -> bool:
    return TORCH_AVAILABLE


__all__ = [
    "MATD3Config",
    "MATD3ReplayBuffer",
    "evaluate_matd3_checkpoint",
    "torch_available",
    "train_matd3",
]
