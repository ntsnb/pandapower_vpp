from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from vpp_dso_sim.experiments.deep_rl_campaign import (  # noqa: E402
    DeepRLCandidateCampaignConfig,
    run_deep_rl_candidate_campaign,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a deep-RL candidate campaign with explicit implementation-boundary labels."
    )
    parser.add_argument(
        "--preset",
        choices=("pilot_1d", "long_7d", "long_14d"),
        default="pilot_1d",
        help=(
            "Training budget preset. pilot_1d is a quick engineering check; "
            "long_7d/long_14d are the recommended research-scale starting points."
        ),
    )
    parser.add_argument("--config", default=str(PROJECT_ROOT / "configs" / "european_lv_benchmark_v2.yaml"))
    parser.add_argument(
        "--output-dir",
        default=str(PROJECT_ROOT / "outputs" / "deep_rl_candidate_campaign"),
    )
    parser.add_argument("--episodes", type=int, default=2)
    parser.add_argument("--horizon-steps", type=int, default=96)
    parser.add_argument("--eval-horizon-steps", type=int, default=96)
    parser.add_argument("--train-top-k", type=int, default=3)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--seed", type=int, action="append", default=None)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument(
        "--train-all-adapters",
        action="store_true",
        help="Train every candidate that the current CTDE adapter can execute.",
    )
    parser.add_argument(
        "--plan-only",
        action="store_true",
        help="Write plan/status tables without launching PyTorch training.",
    )
    return parser.parse_args()


def _apply_preset(args: argparse.Namespace) -> tuple[int, int, int, tuple[int, ...]]:
    seeds = tuple(args.seed) if args.seed else ()
    if args.preset == "long_7d":
        return (
            max(args.episodes, 100),
            max(args.horizon_steps, 672),
            max(args.eval_horizon_steps, 672),
            seeds or (6101, 6102, 6103, 6104, 6105),
        )
    if args.preset == "long_14d":
        return (
            max(args.episodes, 100),
            max(args.horizon_steps, 1344),
            max(args.eval_horizon_steps, 1344),
            seeds or (6101, 6102, 6103, 6104, 6105),
        )
    return args.episodes, args.horizon_steps, args.eval_horizon_steps, seeds or (7401,)


def main() -> None:
    args = parse_args()
    episodes, horizon_steps, eval_horizon_steps, seeds = _apply_preset(args)
    result = run_deep_rl_candidate_campaign(
        DeepRLCandidateCampaignConfig(
            config_path=args.config,
            output_dir=args.output_dir,
            top_k=args.top_k,
            train_top_k=args.train_top_k,
            train_all_adapters=args.train_all_adapters,
            execute_training=not args.plan_only,
            episodes=episodes,
            horizon_steps=horizon_steps,
            eval_horizon_steps=eval_horizon_steps,
            seeds=seeds,
            hidden_dim=args.hidden_dim,
            learning_rate=args.learning_rate,
        )
    )
    summary = result["summary"]
    print(f"output_dir={Path(result['output_dir']).resolve()}")
    print(f"preset={args.preset}")
    print(f"episodes={episodes}")
    print(f"horizon_steps={horizon_steps}")
    print(f"eval_horizon_steps={eval_horizon_steps}")
    print(f"seeds={','.join(str(seed) for seed in seeds)}")
    print(f"candidate_count={summary['candidate_count']}")
    print(f"trained_count={summary['trained_count']}")
    print(f"queued_count={summary['queued_count']}")
    print(f"not_yet_implemented_count={summary['not_yet_implemented_count']}")
    print(f"best_candidate_by_positive_profit={summary['best_candidate_by_positive_profit']}")
    print(
        "best_positive_private_profit_step_rate="
        f"{summary['best_positive_private_profit_step_rate']:.4f}"
    )
    print(f"campaign_report={result['report']}")


if __name__ == "__main__":
    main()
