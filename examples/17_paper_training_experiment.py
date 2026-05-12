from __future__ import annotations

import argparse
from dataclasses import replace
import os
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
BOOT_CACHE_DIR = PROJECT_ROOT / "outputs" / ".cache"
(BOOT_CACHE_DIR / "matplotlib").mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(BOOT_CACHE_DIR / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(BOOT_CACHE_DIR))

from vpp_dso_sim.experiments.paper_training import (  # noqa: E402
    paper_training_preset,
    run_paper_training_experiment,
)


def _csv_tuple(value: str | None, cast=str):
    if not value:
        return None
    return tuple(cast(item.strip()) for item in value.split(",") if item.strip())


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run the formal DSO/VPP paper-training campaign with split profiles, "
            "baselines, trainable MARL algorithms, TensorBoard scalars and static HTML."
        )
    )
    parser.add_argument("--preset", default="smoke", choices=["smoke", "pilot", "paper_lite", "paper_long"])
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--config-path", default=None)
    parser.add_argument(
        "--algorithms",
        default=None,
        help="Comma list, e.g. rule_based,no_flex,static_fr_price_extreme_proxy,ac_validated_search_reference,happo",
    )
    parser.add_argument("--seeds", default=None, help="Comma list of integer seeds")
    parser.add_argument("--hparam-cases", default=None, help="Comma list: base,lower_lr,higher_entropy,larger_network")
    parser.add_argument("--horizon-steps", type=int, default=None)
    parser.add_argument("--eval-horizon-steps", type=int, default=None)
    parser.add_argument("--train-episodes", type=int, default=None)
    parser.add_argument("--gamma", type=float, default=None)
    parser.add_argument("--checkpoint-selection", default=None, choices=["final", "train_best", "both"])
    parser.add_argument("--data-source", default=None, choices=["smart_ds", "synthetic"])
    parser.add_argument(
        "--ac-reference-max-candidates",
        type=int,
        default=None,
        help="Candidate budget per step for ac_validated_search_reference. Larger values improve the reference but cost more AC power-flow solves.",
    )
    parser.add_argument("--progress-interval-seconds", type=float, default=None)
    parser.add_argument("--verbose-progress", action="store_true")
    parser.add_argument(
        "--resume-completed",
        action="store_true",
        help="Reuse existing completed train/eval artifacts in the output directory. Off by default to avoid stale-result contamination.",
    )
    parser.add_argument("--no-html", action="store_true")
    parser.add_argument("--no-tensorboard", action="store_true")
    args = parser.parse_args()

    cfg = paper_training_preset(args.preset)
    updates = {}
    if args.output_dir:
        updates["output_dir"] = args.output_dir
    if args.config_path:
        updates["config_path"] = args.config_path
    parsed_algorithms = _csv_tuple(args.algorithms)
    if parsed_algorithms:
        updates["algorithms"] = parsed_algorithms
    parsed_seeds = _csv_tuple(args.seeds, int)
    if parsed_seeds:
        updates["seeds"] = parsed_seeds
    parsed_cases = _csv_tuple(args.hparam_cases)
    if parsed_cases:
        updates["hparam_cases"] = parsed_cases
    if args.horizon_steps is not None:
        updates["horizon_steps"] = int(args.horizon_steps)
    if args.eval_horizon_steps is not None:
        updates["eval_horizon_steps"] = int(args.eval_horizon_steps)
    if args.train_episodes is not None:
        updates["train_episodes"] = int(args.train_episodes)
    if args.gamma is not None:
        updates["gamma"] = float(args.gamma)
    if args.checkpoint_selection is not None:
        updates["checkpoint_selection"] = args.checkpoint_selection
    if args.data_source is not None:
        updates["data_source"] = args.data_source
    if args.ac_reference_max_candidates is not None:
        updates["ac_reference_max_candidates"] = int(args.ac_reference_max_candidates)
    if args.progress_interval_seconds is not None:
        updates["progress_interval_seconds"] = float(args.progress_interval_seconds)
    if args.verbose_progress:
        updates["verbose_progress"] = True
    if args.resume_completed:
        updates["resume_completed"] = True
    if args.no_html:
        updates["export_html"] = False
    if args.no_tensorboard:
        updates["tensorboard"] = False
    cfg = replace(cfg, **updates)

    result = run_paper_training_experiment(cfg)
    print(f"Output directory: {result['output_dir']}")
    print(f"HTML report: {result['html_path']}")
    print(f"Runs: {len(result['run_index'])}")
    print(f"Evaluation rows: {len(result['evaluation_seed_metrics'])}")


if __name__ == "__main__":
    main()
