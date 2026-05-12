from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from vpp_dso_sim.envs.gym_env import VPPDSOEnv


def main() -> None:
    env = VPPDSOEnv(PROJECT_ROOT / "configs" / "ieee33_multi_vpp.yaml", horizon_steps=10)
    obs, info = env.reset(seed=42)
    print(f"reset_obs_shape={obs.shape} info={info}")
    for step in range(10):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        print(
            f"step={step} obs_shape={obs.shape} reward={reward:.3f} "
            f"truncated={truncated} total_cost={info['reward_components']['total_cost']:.3f}"
        )
        if terminated or truncated:
            break
    env.close()


if __name__ == "__main__":
    main()

