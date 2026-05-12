from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from vpp_dso_sim.learning.advanced_marl import (  # noqa: E402
    HAPPOConfig,
    HASACConfig,
    evaluate_happo_checkpoint,
    evaluate_hasac_checkpoint,
    torch_available,
    train_happo,
    train_hasac,
)
from vpp_dso_sim.learning.hatrpo import HATRPOConfig, evaluate_hatrpo_checkpoint, train_hatrpo  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train HAPPO, HATRPO or HASAC research scaffolds.")
    parser.add_argument("--algorithm", choices=("happo", "hatrpo", "hasac"), default="happo")
    parser.add_argument("--config", default=str(PROJECT_ROOT / "configs" / "european_lv_mixed_vpp.yaml"))
    parser.add_argument("--output-dir", default=str(PROJECT_ROOT / "outputs" / "advanced_marl"))
    parser.add_argument("--episodes", type=int, default=2)
    parser.add_argument("--horizon-steps", type=int, default=8)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--warmup-steps", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eval", action="store_true", help="Run frozen eval after training.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not torch_available():
        raise SystemExit("PyTorch is not installed. Install torch before running advanced MARL training.")
    output_dir = Path(args.output_dir) / args.algorithm
    if args.algorithm == "happo":
        result = train_happo(
            config_path=args.config,
            output_dir=output_dir,
            config=HAPPOConfig(
                episodes=args.episodes,
                horizon_steps=args.horizon_steps,
                hidden_dim=args.hidden_dim,
                seed=args.seed,
            ),
        )
    elif args.algorithm == "hatrpo":
        result = train_hatrpo(
            config_path=args.config,
            output_dir=output_dir,
            config=HATRPOConfig(
                episodes=args.episodes,
                horizon_steps=args.horizon_steps,
                hidden_dim=args.hidden_dim,
                seed=args.seed,
            ),
        )
    else:
        result = train_hasac(
            config_path=args.config,
            output_dir=output_dir,
            config=HASACConfig(
                episodes=args.episodes,
                horizon_steps=args.horizon_steps,
                hidden_dim=args.hidden_dim,
                batch_size=args.batch_size,
                warmup_steps=args.warmup_steps,
                seed=args.seed,
            ),
        )
    summary = result["summary"]
    print(f"output_dir={Path(result['output_dir']).resolve()}")
    print(f"algorithm={summary['algorithm']}")
    print(f"final_episode_reward={summary['final_episode_reward']}")
    print(f"checkpoint={summary['checkpoint']}")
    if args.eval:
        if args.algorithm == "happo":
            eval_result = evaluate_happo_checkpoint(
                config_path=args.config,
                checkpoint_path=result["checkpoint"],
                output_dir=output_dir / "frozen_eval",
                horizon_steps=args.horizon_steps,
                seed=args.seed + 10_000,
            )
        elif args.algorithm == "hatrpo":
            eval_result = evaluate_hatrpo_checkpoint(
                config_path=args.config,
                checkpoint_path=result["checkpoint"],
                output_dir=output_dir / "frozen_eval",
                horizon_steps=args.horizon_steps,
                seed=args.seed + 10_000,
            )
        else:
            eval_result = evaluate_hasac_checkpoint(
                config_path=args.config,
                checkpoint_path=result["checkpoint"],
                output_dir=output_dir / "frozen_eval",
                horizon_steps=args.horizon_steps,
                seed=args.seed + 10_000,
            )
        print(f"eval_total_reward={eval_result['summary']['total_reward']}")


if __name__ == "__main__":
    main()
