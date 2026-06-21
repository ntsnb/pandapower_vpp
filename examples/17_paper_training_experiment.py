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

from vpp_dso_sim.utils.runtime import configure_numeric_thread_limits  # noqa: E402

configure_numeric_thread_limits(default_threads=8)

from vpp_dso_sim.experiments.paper_training import (  # noqa: E402
    _validate_trainable_cuda_requirement,
    paper_training_preset,
    run_paper_training_experiment,
)


def _csv_tuple(value: str | None, cast=str):
    if not value:
        return None
    return tuple(cast(item.strip()) for item in value.split(",") if item.strip())


def _default_dashboard_data_dir(output_dir: str | Path) -> Path:
    resolved = Path(output_dir)
    return resolved.parent / f"{resolved.name}_dashboard_runs"


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run the formal DSO/VPP paper-training campaign with split profiles, "
            "baselines, trainable MARL algorithms, TensorBoard scalars and static HTML."
        )
    )
    parser.add_argument(
        "--preset",
        default="smoke",
        choices=[
            "smoke",
            "pilot",
            "paper_lite",
            "paper_long",
            "paper_long_sensitivity_v1",
            "paper_long_sensitivity_v1_reward_v3_1_market_safety",
        ],
    )
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
    parser.add_argument("--happo-shared-rollout", action="store_true")
    parser.add_argument("--happo-shared-rollout-workers", type=int, default=None)
    parser.add_argument("--happo-rollout-fragment-steps", type=int, default=None)
    parser.add_argument("--happo-shared-rollout-backend", choices=["serial", "subprocess"], default=None)
    parser.add_argument("--no-happo-reward-dynamic-reports", action="store_true")
    parser.add_argument("--happo-reward-dynamic-report-every-episodes", type=int, default=None)
    parser.add_argument("--happo-reward-dynamic-report-all-workers", action="store_true")
    parser.add_argument(
        "--resume-completed",
        action="store_true",
        help="Reuse existing completed train/eval artifacts in the output directory. Off by default to avoid stale-result contamination.",
    )
    parser.add_argument("--no-html", action="store_true")
    parser.add_argument("--no-tensorboard", action="store_true")
    parser.add_argument(
        "--dashboard",
        action="store_true",
        help="Start the local dashboard during the run and export paper-training summary metrics after completion.",
    )
    parser.add_argument("--dashboard-data-dir", default=None)
    parser.add_argument("--dashboard-host", default="127.0.0.1")
    parser.add_argument("--dashboard-port", type=int, default=8765)
    parser.add_argument("--dashboard-auto-port", action="store_true")
    parser.add_argument("--dashboard-open-browser", action="store_true")
    parser.add_argument(
        "--dashboard-keep-alive",
        action="store_true",
        help="Keep the dashboard process alive after training completes until Ctrl+C.",
    )
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
    if args.happo_shared_rollout:
        updates["happo_shared_rollout_enabled"] = True
    if args.happo_shared_rollout_workers is not None:
        updates["happo_shared_rollout_workers"] = int(args.happo_shared_rollout_workers)
    if args.happo_rollout_fragment_steps is not None:
        updates["happo_rollout_fragment_steps"] = int(args.happo_rollout_fragment_steps)
    if args.happo_shared_rollout_backend is not None:
        updates["happo_shared_rollout_backend"] = str(args.happo_shared_rollout_backend)
    if args.no_happo_reward_dynamic_reports:
        updates["happo_reward_dynamic_reports"] = False
    if args.happo_reward_dynamic_report_every_episodes is not None:
        updates["happo_reward_dynamic_report_every_episodes"] = int(
            args.happo_reward_dynamic_report_every_episodes
        )
    if args.happo_reward_dynamic_report_all_workers:
        updates["happo_reward_dynamic_report_all_workers"] = True
    if args.resume_completed:
        updates["resume_completed"] = True
    if args.no_html:
        updates["export_html"] = False
    if args.no_tensorboard:
        updates["tensorboard"] = False
    cfg = replace(cfg, **updates)
    trainable_algorithms = {"happo", "hatrpo", "matd3", "hasac"}
    if any(algorithm in trainable_algorithms for algorithm in cfg.algorithms):
        _validate_trainable_cuda_requirement(cfg, algorithm=",".join(cfg.algorithms))

    dashboard_handle = None
    dashboard_data_dir = Path(args.dashboard_data_dir) if args.dashboard_data_dir else _default_dashboard_data_dir(cfg.output_dir)
    if args.dashboard:
        from marl_dashboard.backend.server import start_dashboard  # noqa: E402

        dashboard_handle = start_dashboard(
            data_dir=dashboard_data_dir,
            host=args.dashboard_host,
            port=args.dashboard_port,
            auto_port=args.dashboard_auto_port,
            open_browser=args.dashboard_open_browser,
            background=True,
        )
    try:
        result = run_paper_training_experiment(cfg)
        if args.dashboard:
            from marl_dashboard.integrations.paper_training import export_paper_training_dashboard  # noqa: E402

            dashboard_run_id = export_paper_training_dashboard(result, data_dir=dashboard_data_dir)
            print(f"Dashboard data dir: {dashboard_data_dir}")
            print(f"Dashboard run: {dashboard_run_id}")
            if dashboard_handle is not None:
                print(f"Dashboard URL: {dashboard_handle.url}")
            if args.dashboard_keep_alive and dashboard_handle is not None:
                print("Dashboard keep-alive enabled. Press Ctrl+C to stop.")
                try:
                    while True:
                        import time

                        time.sleep(3600)
                except KeyboardInterrupt:
                    pass
    finally:
        if dashboard_handle is not None and not args.dashboard_keep_alive:
            dashboard_handle.stop()
    print(f"Output directory: {result['output_dir']}")
    print(f"HTML report: {result['html_path']}")
    print(f"Runs: {len(result['run_index'])}")
    print(f"Evaluation rows: {len(result['evaluation_seed_metrics'])}")


if __name__ == "__main__":
    main()
