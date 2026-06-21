from __future__ import annotations

from dataclasses import dataclass
import multiprocessing as mp
import os
from pathlib import Path
import time
import traceback
from typing import Any

import numpy as np


@dataclass(frozen=True)
class SharedRolloutWorkerSpec:
    worker_index: int
    config_path: str | None
    horizon_steps: int
    use_structured_dso_actor: bool
    vpp_ids: tuple[str, ...]
    blas_threads: int = 1


def _worker_thread_env(blas_threads: int) -> None:
    value = str(max(1, int(blas_threads)))
    for key in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
    ):
        os.environ[key] = value


def _build_worker_policy_state(env: Any, observations: dict[str, dict[str, Any]], spec: SharedRolloutWorkerSpec) -> dict[str, Any]:
    from vpp_dso_sim.dso.observation.happo_structured import build_happo_structured_dso_observation
    from vpp_dso_sim.envs.observations import build_critic_global_state
    from vpp_dso_sim.learning.deep_rl import encode_critic_global_state, encode_dso_observation

    current_step = int(env.current_step)
    if bool(spec.use_structured_dso_actor):
        dso_obs_vec, structured_spec = build_happo_structured_dso_observation(
            env.scenario,
            step=current_step,
            config=env.scenario.config,
        )
        structured_vpp_ids = tuple(str(vpp_id) for vpp_id in structured_spec.vpp_ids)
        if structured_vpp_ids != tuple(spec.vpp_ids):
            raise RuntimeError(
                "Worker structured DSO observation VPP ids changed: "
                f"got {structured_vpp_ids}, expected {spec.vpp_ids}."
            )
    else:
        dso_obs_vec = encode_dso_observation(observations["dso_global_guidance"], list(spec.vpp_ids))
    critic_state_vec = encode_critic_global_state(
        build_critic_global_state(env.scenario, current_step),
        list(spec.vpp_ids),
    )
    return {
        "observations": observations,
        "current_step": current_step,
        "dso_obs_vec": np.asarray(dso_obs_vec, dtype=np.float32),
        "critic_state_vec": np.asarray(critic_state_vec, dtype=np.float32),
    }


def _worker_loop(conn: Any, spec: SharedRolloutWorkerSpec) -> None:
    _worker_thread_env(spec.blas_threads)
    env = None
    try:
        from vpp_dso_sim.envs.multi_agent_env import MultiAgentVPPDSOEnv

        env = MultiAgentVPPDSOEnv(
            config_path=Path(spec.config_path) if spec.config_path is not None else None,
            horizon_steps=int(spec.horizon_steps),
        )
        while True:
            message = conn.recv()
            command = str(message.get("cmd", ""))
            if command == "reset":
                observations, infos = env.reset(
                    seed=message.get("seed"),
                    start_step=int(message.get("start_step", 0)),
                )
                conn.send(
                    {
                        "ok": True,
                        "cmd": "reset",
                        "worker_index": int(spec.worker_index),
                        "state": _build_worker_policy_state(env, observations, spec),
                        "infos": infos,
                    }
                )
            elif command == "step":
                step_started = time.perf_counter()
                next_observations, reward_map, terminations, truncations, infos = env.step(message.get("action_payload"))
                step_seconds = time.perf_counter() - step_started
                conn.send(
                    {
                        "ok": True,
                        "cmd": "step",
                        "worker_index": int(spec.worker_index),
                        "transition": {
                            "next_observations": next_observations,
                            "reward_map": reward_map,
                            "terminations": terminations,
                            "truncations": truncations,
                            "infos": infos,
                            "worker_step_seconds": float(step_seconds),
                        },
                        "state": _build_worker_policy_state(env, next_observations, spec),
                    }
                )
            elif command == "close":
                if env is not None:
                    env.close()
                conn.send({"ok": True, "cmd": "close", "worker_index": int(spec.worker_index)})
                break
            else:
                raise ValueError(f"Unknown shared rollout worker command: {command!r}")
    except KeyboardInterrupt:
        try:
            conn.send(
                {
                    "ok": False,
                    "cmd": "interrupted",
                    "worker_index": int(spec.worker_index),
                    "error": "KeyboardInterrupt",
                    "traceback": traceback.format_exc(),
                }
            )
        except Exception:
            pass
    except BaseException as exc:
        try:
            conn.send(
                {
                    "ok": False,
                    "cmd": "error",
                    "worker_index": int(spec.worker_index),
                    "error": repr(exc),
                    "traceback": traceback.format_exc(),
                }
            )
        except Exception:
            pass
    finally:
        if env is not None:
            try:
                env.close()
            except Exception:
                pass
        conn.close()


class SubprocessSharedRolloutWorker:
    def __init__(self, spec: SharedRolloutWorkerSpec) -> None:
        self.spec = spec
        self._ctx = mp.get_context("spawn")
        self._parent_conn, child_conn = self._ctx.Pipe()
        self._process = self._ctx.Process(
            target=_worker_loop,
            args=(child_conn, spec),
            name=f"happo-shared-rollout-worker-{int(spec.worker_index)}",
        )

    @property
    def pid(self) -> int | None:
        return self._process.pid

    @property
    def exitcode(self) -> int | None:
        return self._process.exitcode

    def start(self) -> None:
        self._process.start()

    def reset(self, *, seed: int, start_step: int) -> dict[str, Any]:
        self._parent_conn.send({"cmd": "reset", "seed": int(seed), "start_step": int(start_step)})
        return self.recv()

    def step_async(self, action_payload: dict[str, Any]) -> None:
        self._parent_conn.send({"cmd": "step", "action_payload": action_payload})

    def recv(self) -> dict[str, Any]:
        payload = self._parent_conn.recv()
        if not bool(payload.get("ok", False)):
            raise RuntimeError(
                "HAPPO shared rollout subprocess worker failed: "
                f"worker={payload.get('worker_index')} error={payload.get('error')}\n"
                f"{payload.get('traceback', '')}"
            )
        return payload

    def close(self, *, timeout: float = 5.0) -> None:
        if self._process.is_alive():
            try:
                self._parent_conn.send({"cmd": "close"})
                self.recv()
            except Exception:
                pass
            self._process.join(timeout=float(timeout))
        if self._process.is_alive():
            self._process.terminate()
            self._process.join(timeout=float(timeout))
        self._parent_conn.close()
