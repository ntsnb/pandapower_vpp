from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from vpp_dso_sim.learning.matd3 import (  # noqa: E402
    MATD3Config,
    evaluate_matd3_checkpoint,
    torch_available,
    train_matd3,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train MATD3 for continuous DSO/VPP dispatch.")
    parser.add_argument("--config", default=str(PROJECT_ROOT / "configs" / "european_lv_mixed_vpp.yaml"))
    parser.add_argument("--output-dir", default=str(PROJECT_ROOT / "outputs" / "matd3"))
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--horizon-steps", type=int, default=24)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--warmup-steps", type=int, default=16)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eval", action="store_true", help="Run frozen deterministic evaluation after training.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not torch_available():
        raise SystemExit("PyTorch is not installed. Install torch before running MATD3 training.")
    result = train_matd3(
        config_path=args.config,
        output_dir=args.output_dir,
        config=MATD3Config(
            episodes=args.episodes,
            horizon_steps=args.horizon_steps,
            batch_size=args.batch_size,
            warmup_steps=args.warmup_steps,
            hidden_dim=args.hidden_dim,
            seed=args.seed,
        ),
    )
    summary = result["summary"]
    print(f"output_dir={Path(result['output_dir']).resolve()}")
    print(f"algorithm={summary['algorithm']}")
    print(f"total_env_steps={summary['total_env_steps']}")
    print(f"critic_updates={summary['critic_updates']}")
    print(f"actor_updates={summary['actor_updates']}")
    print(f"final_episode_reward={summary['final_episode_reward']:.3f}")
    print(f"checkpoint={summary['checkpoint']}")
    if args.eval:
        eval_result = evaluate_matd3_checkpoint(
            config_path=args.config,
            checkpoint_path=result["checkpoint"],
            output_dir=Path(args.output_dir) / "frozen_eval",
            horizon_steps=args.horizon_steps,
            seed=args.seed + 10_000,
        )
        print(f"eval_total_reward={eval_result['summary']['total_reward']:.3f}")


if __name__ == "__main__":
    main()
