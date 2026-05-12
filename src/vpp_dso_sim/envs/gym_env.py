from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

try:
    import gymnasium as gym
    from gymnasium import spaces
    GYMNASIUM_AVAILABLE = True
except ImportError:  # pragma: no cover - used only when optional dependency is absent.
    GYMNASIUM_AVAILABLE = False

    class _FallbackEnv:
        metadata: dict[str, Any] = {}

        def __init__(self):
            return None

        def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
            return None

    class _FallbackBox:
        def __init__(self, low, high, shape, dtype):
            self.low = low
            self.high = high
            self.shape = shape
            self.dtype = dtype

        def sample(self):
            return np.random.uniform(self.low, self.high, self.shape).astype(self.dtype)

    class _FallbackSpaces:
        Box = _FallbackBox

    class _FallbackGym:
        Env = _FallbackEnv

    gym = _FallbackGym()
    spaces = _FallbackSpaces()

from vpp_dso_sim.der.evcs import EVCSModel
from vpp_dso_sim.der.hvac import HVACModel
from vpp_dso_sim.der.storage import StorageModel
from vpp_dso_sim.simulation.scenario import load_scenario
from vpp_dso_sim.simulation.simulator import Simulator


class VPPDSOEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(self, config_path: str | Path | None = None, horizon_steps: int | None = None):
        super().__init__()
        self.config_path = config_path
        self.horizon_override = horizon_steps
        self.scenario = load_scenario(config_path)
        if horizon_steps is not None:
            self.scenario.horizon_steps = horizon_steps
        self.simulator = Simulator(self.scenario)
        self.scenario.dso.run_powerflow()
        self.current_step = 0
        self.n_vpps = len(self.scenario.vpps)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(self.n_vpps,), dtype=np.float32)
        obs = self._get_observation()
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=obs.shape,
            dtype=np.float32,
        )

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        if GYMNASIUM_AVAILABLE:
            super().reset(seed=seed)
        elif seed is not None:
            np.random.seed(seed)
        self.scenario = load_scenario(self.config_path)
        if self.horizon_override is not None:
            self.scenario.horizon_steps = self.horizon_override
        self.simulator = Simulator(self.scenario)
        self.simulator.reset()
        self.current_step = 0
        self.scenario.dso.run_powerflow()
        return self._get_observation(), {"step": self.current_step}

    def step(self, action):
        action_array = np.asarray(action, dtype=float).reshape(-1)
        targets: dict[str, float] = {}
        for i, vpp in enumerate(self.scenario.vpps):
            delta = float(action_array[i]) if i < len(action_array) else 0.0
            p_min, p_max, _, _ = vpp.aggregate_flexibility(self.current_step)
            targets[vpp.id] = float(np.clip(vpp.current_power_mw() + delta, p_min, p_max))

        result = self.simulator.step(self.current_step, targets)
        self.current_step += 1
        obs = self._get_observation()
        reward = float(result["reward_components"]["reward"])
        terminated = False
        truncated = self.current_step >= self.scenario.horizon_steps
        info = {
            "step": self.current_step,
            "reward_components": result["reward_components"],
            "violations": result["violations"],
        }
        return obs, reward, terminated, truncated, info

    def _get_observation(self) -> np.ndarray:
        net = self.scenario.net
        values: list[float] = [float(self.current_step)]
        if hasattr(net, "res_bus") and len(net.res_bus):
            values.extend(net.res_bus["vm_pu"].fillna(1.0).astype(float).to_list())
        else:
            values.extend([1.0] * len(net.bus))
        if hasattr(net, "res_line") and len(net.res_line):
            values.extend(net.res_line["loading_percent"].fillna(0.0).astype(float).to_list())
        else:
            values.extend([0.0] * len(net.line))
        if len(net.trafo) and hasattr(net, "res_trafo") and len(net.res_trafo):
            values.extend(net.res_trafo["loading_percent"].fillna(0.0).astype(float).to_list())
        else:
            values.extend([0.0] * len(net.trafo))

        for vpp in self.scenario.vpps:
            p_min, p_max, _, _ = vpp.aggregate_flexibility(self.current_step)
            values.extend([vpp.current_power_mw(), vpp.current_reactive_power_mvar(), p_min, p_max])

        for vpp in self.scenario.vpps:
            for der in vpp.der_list:
                if isinstance(der, StorageModel):
                    values.append(der.soc)
                elif isinstance(der, EVCSModel):
                    values.append(der.average_soc())
                elif isinstance(der, HVACModel):
                    values.append(der.indoor_temp)

        values.append(float(self.scenario.price_profile[self.current_step % len(self.scenario.price_profile)]))
        values.append(float(self.scenario.pv_profile[self.current_step % len(self.scenario.pv_profile)]))
        values.append(float(self.scenario.load_profile[self.current_step % len(self.scenario.load_profile)]))
        return np.asarray(values, dtype=np.float32)

    def render(self):
        state = self.scenario.dso.compute_network_state()
        print(state)

    def close(self):
        return None
