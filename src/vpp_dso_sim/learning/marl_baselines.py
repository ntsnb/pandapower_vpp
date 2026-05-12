from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from vpp_dso_sim.envs.gym_env import VPPDSOEnv
from vpp_dso_sim.learning.agent_roles import build_agent_role_map, build_encoder_role_map
from vpp_dso_sim.learning.encoders import encode_node_need
from vpp_dso_sim.utils.io import ensure_dir, write_json


CLASSIC_BASELINES = ("ippo", "mappo", "maddpg", "qmix")


@dataclass(frozen=True)
class BaselineConfig:
    algorithm: str
    horizon_steps: int = 8
    episodes: int = 2
    action_scale: float = 0.10
    exploration_noise: float = 0.02
    seed: int = 42

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _env_signal(env: VPPDSOEnv) -> dict[str, float | str]:
    state = env.scenario.dso.compute_network_state()
    encoded = encode_node_need(state)
    price = float(env.scenario.price_profile[env.current_step % len(env.scenario.price_profile)])
    pv = float(env.scenario.pv_profile[env.current_step % len(env.scenario.pv_profile)])
    load = float(env.scenario.load_profile[env.current_step % len(env.scenario.load_profile)])
    return {**state, **encoded, "price": price, "pv_forecast_factor": pv, "load_scale": load}


def _algorithm_action(
    algorithm: str,
    env: VPPDSOEnv,
    rng: np.random.Generator,
    action_scale: float,
    exploration_noise: float,
) -> np.ndarray:
    signal = _env_signal(env)
    n = env.n_vpps
    noise = rng.normal(0.0, exploration_noise, size=n)
    price = float(signal.get("price", 70.0))
    load = float(signal.get("load_scale", 1.0))
    pv = float(signal.get("pv_forecast_factor", 0.0))
    voltage_low_need = float(signal.get("voltage_low_need", 0.0))
    voltage_high_need = float(signal.get("voltage_high_need", 0.0))
    congestion_need = float(signal.get("congestion_need", 0.0))

    if algorithm == "ippo":
        # Independent VPP actors: each sees a different local bias plus shared price.
        price_signal = np.clip((price - 75.0) / 50.0, -1.0, 1.0)
        local_bias = np.linspace(-0.4, 0.4, n)
        action = action_scale * (price_signal + 0.2 * local_bias) + noise
    elif algorithm == "mappo":
        # Shared centralized-training signal: all actors respond to global stress.
        stress = voltage_low_need - voltage_high_need - 0.5 * congestion_need
        action = np.full(n, action_scale * np.clip(stress, -1.0, 1.0)) + noise
    elif algorithm == "maddpg":
        # Deterministic actor with critic-like global features.
        deterministic = 0.6 * (load - 1.0) - 0.4 * pv + voltage_low_need - voltage_high_need
        action = np.full(n, action_scale * np.clip(deterministic, -1.0, 1.0)) + 0.5 * noise
    elif algorithm == "qmix":
        # Discrete joint-action mixing fallback.
        candidates = np.array([-action_scale, 0.0, action_scale])
        if voltage_low_need > voltage_high_need:
            chosen = action_scale
        elif voltage_high_need > 0.0 or congestion_need > 0.0:
            chosen = -action_scale
        else:
            chosen = candidates[int(rng.integers(0, len(candidates)))]
        action = np.full(n, chosen) + 0.25 * noise
    else:
        raise ValueError(f"Unsupported MARL baseline algorithm: {algorithm}")
    return np.clip(action, -1.0, 1.0).astype(np.float32)


def run_single_baseline(config: BaselineConfig, config_path: str | Path | None = None) -> dict[str, pd.DataFrame | dict[str, Any]]:
    rng = np.random.default_rng(config.seed)
    episode_rows: list[dict[str, Any]] = []
    step_rows: list[dict[str, Any]] = []
    for episode in range(config.episodes):
        env = VPPDSOEnv(config_path=config_path, horizon_steps=config.horizon_steps)
        _, reset_info = env.reset(seed=config.seed + episode)
        total_reward = 0.0
        total_cost = 0.0
        violation_count = 0
        for step in range(config.horizon_steps):
            action = _algorithm_action(
                config.algorithm,
                env,
                rng,
                config.action_scale,
                config.exploration_noise,
            )
            _, reward, _, truncated, info = env.step(action)
            components = info.get("reward_components", {})
            violations = info.get("violations", [])
            total_reward += float(reward)
            total_cost += float(components.get("total_cost", 0.0))
            violation_count += len(violations)
            step_rows.append(
                {
                    "algorithm": config.algorithm,
                    "episode": episode,
                    "step": step,
                    "env_step": int(info.get("step", step + 1)),
                    "reward": float(reward),
                    "total_cost": float(components.get("total_cost", 0.0)),
                    "violation_count": len(violations),
                    "action_mean": float(np.mean(action)),
                    "action_min": float(np.min(action)),
                    "action_max": float(np.max(action)),
                    "reset_step": int(reset_info.get("step", 0)),
                }
            )
            if truncated:
                break
        episode_rows.append(
            {
                "algorithm": config.algorithm,
                "episode": episode,
                "episode_reward": float(total_reward),
                "episode_cost": float(total_cost),
                "violation_count": int(violation_count),
                "mean_step_reward": float(total_reward / max(1, config.horizon_steps)),
                "action_scale": float(config.action_scale),
                "exploration_noise": float(config.exploration_noise),
                "seed": int(config.seed),
            }
        )
        env.close()
    return {
        "episode_metrics": pd.DataFrame(episode_rows),
        "step_metrics": pd.DataFrame(step_rows),
        "config": config.to_dict(),
    }


def run_marl_baselines(
    config_path: str | Path | None = None,
    output_dir: str | Path = "outputs/marl_baselines",
    algorithms: tuple[str, ...] = CLASSIC_BASELINES,
    horizon_steps: int = 8,
    episodes: int = 2,
    action_scale: float = 0.10,
    exploration_noise: float = 0.02,
    seed: int = 42,
) -> dict[str, Any]:
    out = ensure_dir(output_dir)
    all_episode: list[pd.DataFrame] = []
    all_step: list[pd.DataFrame] = []
    env_for_roles = VPPDSOEnv(config_path=config_path, horizon_steps=1)
    agent_roles = pd.DataFrame([role.to_dict() for role in build_agent_role_map(env_for_roles.scenario.vpps)])
    encoder_roles = pd.DataFrame([role.to_dict() for role in build_encoder_role_map()])
    env_for_roles.close()

    for index, algorithm in enumerate(algorithms):
        result = run_single_baseline(
            BaselineConfig(
                algorithm=algorithm,
                horizon_steps=horizon_steps,
                episodes=episodes,
                action_scale=action_scale,
                exploration_noise=exploration_noise,
                seed=seed + index * 100,
            ),
            config_path=config_path,
        )
        all_episode.append(result["episode_metrics"])
        all_step.append(result["step_metrics"])

    episode_metrics = pd.concat(all_episode, ignore_index=True) if all_episode else pd.DataFrame()
    step_metrics = pd.concat(all_step, ignore_index=True) if all_step else pd.DataFrame()
    summary = {
        "algorithms": list(algorithms),
        "horizon_steps": horizon_steps,
        "episodes": episodes,
        "best_algorithm": None,
        "best_episode_reward": None,
        "status": "no_runs",
    }
    if not episode_metrics.empty:
        best_idx = episode_metrics["episode_reward"].idxmax()
        summary.update(
            {
                "best_algorithm": str(episode_metrics.at[best_idx, "algorithm"]),
                "best_episode_reward": float(episode_metrics.at[best_idx, "episode_reward"]),
                "status": "completed",
            }
        )

    episode_metrics.to_csv(out / "episode_metrics.csv", index=False)
    step_metrics.to_csv(out / "step_metrics.csv", index=False)
    agent_roles.to_csv(out / "agent_role_map.csv", index=False)
    encoder_roles.to_csv(out / "encoder_role_map.csv", index=False)
    write_json(out / "baseline_summary.json", summary)
    return {
        "episode_metrics": episode_metrics,
        "step_metrics": step_metrics,
        "agent_role_map": agent_roles,
        "encoder_role_map": encoder_roles,
        "summary": summary,
        "output_dir": out,
    }

