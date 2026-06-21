#!/usr/bin/env python3
from __future__ import annotations

import argparse

from vpp_dso_sim.experiments.dso_sensitivity_attention import run_short_training_sanity


def main() -> None:
    parser = argparse.ArgumentParser(description="Run short BC warm-start sanity for DSO sensitivity actor.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=256)
    parser.add_argument("--output-dir")
    args = parser.parse_args()
    summary = run_short_training_sanity(
        config_path=args.config,
        seed=args.seed,
        steps=args.steps,
        output_dir=args.output_dir,
    )
    print(summary)


if __name__ == "__main__":
    main()
