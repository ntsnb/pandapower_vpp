from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any
from uuid import uuid4

import importlib.util
import pandas as pd

from vpp_dso_sim.learning.marl_baselines import CLASSIC_BASELINES, run_single_baseline, BaselineConfig
from vpp_dso_sim.utils.io import ensure_dir, write_json


@dataclass(frozen=True)
class TuningConfig:
    algorithms: tuple[str, ...] = CLASSIC_BASELINES
    action_scales: tuple[float, ...] = (0.05, 0.10, 0.20)
    exploration_noises: tuple[float, ...] = (0.00, 0.02)
    horizon_steps: int = 8
    episodes_per_trial: int = 2
    seed: int = 42
    min_reward_improvement: float = 0.0
    max_violation_count: int = 0


class TrainingSupervisor:
    """Experiment-level tuning supervisor.

    v0 uses lightweight baseline policies to keep experiments reproducible and
    fast. If convergence fails, the returned summary explicitly asks the main
    thread / algorithm agent for review.
    """

    def __init__(self, config: TuningConfig | None = None):
        self.config = config or TuningConfig()
        self.deep_learning_available = importlib.util.find_spec("torch") is not None

    def run(
        self,
        config_path: str | Path | None = None,
        output_dir: str | Path = "outputs/marl_baselines",
    ) -> dict[str, Any]:
        out = ensure_dir(output_dir)
        run_id = f"marl_{uuid4().hex[:8]}"
        rows: list[dict[str, Any]] = []
        trial_id = 0
        for algorithm in self.config.algorithms:
            for action_scale in self.config.action_scales:
                for noise in self.config.exploration_noises:
                    result = run_single_baseline(
                        BaselineConfig(
                            algorithm=algorithm,
                            horizon_steps=self.config.horizon_steps,
                            episodes=self.config.episodes_per_trial,
                            action_scale=action_scale,
                            exploration_noise=noise,
                            seed=self.config.seed + trial_id,
                        ),
                        config_path=config_path,
                    )
                    episode = result["episode_metrics"]
                    mean_reward = float(episode["episode_reward"].mean()) if not episode.empty else float("-inf")
                    best_reward = float(episode["episode_reward"].max()) if not episode.empty else float("-inf")
                    reward_std = float(episode["episode_reward"].std(ddof=0)) if len(episode) > 1 else 0.0
                    violations = int(episode["violation_count"].sum()) if not episode.empty else 0
                    trial_converged = bool(violations <= self.config.max_violation_count)
                    rows.append(
                        {
                            "run_id": run_id,
                            "trial_id": trial_id,
                            "algorithm": algorithm,
                            "action_scale": action_scale,
                            "exploration_noise": noise,
                            "mean_reward": mean_reward,
                            "best_reward": best_reward,
                            "reward_std": reward_std,
                            "violation_count": violations,
                            "converged": trial_converged,
                            "status": "converged" if trial_converged else "needs_algorithm_review",
                            "needs_algorithm_review": not trial_converged,
                            "deep_learning_available": bool(self.deep_learning_available),
                        }
                    )
                    trial_id += 1

        trials = pd.DataFrame(rows)
        if trials.empty:
            summary = {
                "run_id": run_id,
                "status": "failed",
                "converged": False,
                "needs_algorithm_review": True,
                "reason": "no_trials_executed",
                "handoff_target": "main_thread",
                "handoff_message": "No tuning trials executed; return to algorithm agent.",
                "deep_learning_available": self.deep_learning_available,
            }
        else:
            best_idx = trials["mean_reward"].idxmax()
            best = trials.loc[best_idx].to_dict()
            converged = bool(best["violation_count"] <= self.config.max_violation_count)
            handoff = not converged
            summary = {
                "run_id": run_id,
                "status": "converged" if converged else "needs_algorithm_review",
                "converged": converged,
                "needs_algorithm_review": handoff,
                "reason": "ok" if converged else "constraint_violations_remain_or_reward_unstable",
                "handoff_target": "" if converged else "main_thread",
                "handoff_message": ""
                if converged
                else "Training did not meet convergence or safety criteria; return to main thread and algorithm agent for review.",
                "best_trial": best,
                "deep_learning_available": self.deep_learning_available,
                "config": asdict(self.config),
            }
        trials.to_csv(out / "tuning_trials.csv", index=False)
        training_summary = _flatten_training_summary(summary, trials)
        pd.DataFrame([training_summary]).to_csv(out / "training_summary.csv", index=False)
        write_json(out / "training_summary.json", summary)
        return {
            "tuning_trials": trials,
            "training_summary": pd.DataFrame([training_summary]),
            "summary": summary,
            "output_dir": out,
        }


def _flatten_training_summary(summary: dict[str, Any], trials: pd.DataFrame) -> dict[str, Any]:
    best = summary.get("best_trial", {}) if isinstance(summary.get("best_trial", {}), dict) else {}
    config = summary.get("config", {}) if isinstance(summary.get("config", {}), dict) else {}
    return {
        "run_id": summary.get("run_id", ""),
        "status": summary.get("status", ""),
        "converged": bool(summary.get("converged", False)),
        "needs_algorithm_review": bool(summary.get("needs_algorithm_review", True)),
        "reason": summary.get("reason", ""),
        "best_trial_id": best.get("trial_id", ""),
        "best_algorithm": best.get("algorithm", ""),
        "best_action_scale": best.get("action_scale", ""),
        "best_exploration_noise": best.get("exploration_noise", ""),
        "best_mean_reward": best.get("mean_reward", ""),
        "best_reward": best.get("best_reward", ""),
        "best_violation_count": best.get("violation_count", ""),
        "trial_count": int(len(trials)),
        "horizon_steps": config.get("horizon_steps", ""),
        "episodes_per_trial": config.get("episodes_per_trial", ""),
        "deep_learning_available": bool(summary.get("deep_learning_available", False)),
        "handoff_target": summary.get("handoff_target", ""),
        "handoff_message": summary.get("handoff_message", ""),
    }
