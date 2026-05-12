from __future__ import annotations

from vpp_dso_sim.envs.gym_env import VPPDSOEnv


def test_gym_env_reset_and_step():
    env = VPPDSOEnv(horizon_steps=2)
    obs, info = env.reset(seed=123)
    assert obs.shape == env.observation_space.shape
    assert info["step"] == 0
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    assert obs.shape == env.observation_space.shape
    assert isinstance(reward, float)
    assert terminated is False
    assert "reward_components" in info
    assert {
        "raw_objective_reward",
        "feasibility_bonus",
        "tracking_bonus",
        "action_projection_gap_mw",
        "action_projection_penalty",
    }.issubset(info["reward_components"])
