from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from vpp_dso_sim.learning.deep_rl import (
    DeepRLConfig,
    PrivacySeparatedCTDEConfig,
    torch_available,
    train_deep_rl_actor_critic,
    train_privacy_separated_ctde,
)
from vpp_dso_sim.visualization.report_refresh import refresh_visualization_outputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a small PyTorch actor-critic policy.")
    parser.add_argument(
        "--config",
        default=str(PROJECT_ROOT / "configs" / "european_lv_mixed_vpp.yaml"),
        help="Scenario YAML path.",
    )
    parser.add_argument("--output-dir", default=str(PROJECT_ROOT / "outputs" / "deep_rl"))
    parser.add_argument("--report-output-dir", default=str(PROJECT_ROOT / "outputs"))
    parser.add_argument(
        "--skip-report-refresh",
        action="store_true",
        help="Train only. By default, reports and dashboard CSVs are refreshed after training.",
    )
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--horizon-steps", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--algorithm",
        choices=("ctde", "shared"),
        default="ctde",
        help="ctde runs the privacy-separated target trainer; shared runs the old shared-backbone benchmark.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not torch_available():
        raise SystemExit("PyTorch is not installed. Install torch before running deep RL training.")
    if args.algorithm == "shared":
        result = train_deep_rl_actor_critic(
            config_path=args.config,
            output_dir=args.output_dir,
            config=DeepRLConfig(
                episodes=args.episodes,
                horizon_steps=args.horizon_steps,
                learning_rate=args.learning_rate,
                seed=args.seed,
            ),
        )
    else:
        result = train_privacy_separated_ctde(
            config_path=args.config,
            output_dir=args.output_dir,
            config=PrivacySeparatedCTDEConfig(
                episodes=args.episodes,
                horizon_steps=args.horizon_steps,
                learning_rate=args.learning_rate,
                seed=args.seed,
            ),
        )
    summary = result["summary"]
    print(f"output_dir={Path(result['output_dir']).resolve()}")
    print(f"algorithm={summary['algorithm']}")
    print(f"episodes={summary['episodes']}")
    print(f"horizon_steps={summary['horizon_steps']}")
    print(f"best_episode_reward={summary['best_episode_reward']:.3f}")
    print(f"final_episode_reward={summary['final_episode_reward']:.3f}")
    print(f"total_violation_count={summary['total_violation_count']}")
    print(f"checkpoint={summary['checkpoint']}")
    if not args.skip_report_refresh:
        report_paths = refresh_visualization_outputs(
            config_path=args.config,
            output_dir=args.report_output_dir,
            deep_rl_dir=args.output_dir,
        )
        print("visualization_refresh=completed")
        print(f"interactive_report={report_paths['interactive_report']}")
        print(f"rl_architecture_report={report_paths['rl_architecture_report']}")
        print(f"vpp_first_person_report={report_paths['first_person_reports']['index']}")
        print(f"model_update_summary={report_paths['model_update_summary']}")


if __name__ == "__main__":
    main()
