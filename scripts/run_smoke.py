#!/usr/bin/env python3
from __future__ import annotations

import argparse

from vpp_dso_sim.experiments.dso_sensitivity_attention import run_smoke_rollout


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a deterministic DSO envelope smoke rollout.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument("--output-dir")
    args = parser.parse_args()
    summary = run_smoke_rollout(
        config_path=args.config,
        seed=args.seed,
        steps=args.steps,
        output_dir=args.output_dir,
    )
    print(summary)


if __name__ == "__main__":
    main()
