from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from vpp_dso_sim.experiments import BenchmarkExperimentConfig, run_benchmark_experiment
from vpp_dso_sim.visualization.report_refresh import refresh_visualization_outputs


def _tuple_arg(value: str) -> tuple[str, ...]:
    return tuple(item.strip() for item in value.split(",") if item.strip())


def _seed_arg(value: str) -> tuple[int, ...]:
    return tuple(int(item.strip()) for item in value.split(",") if item.strip())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the benchmark-v2 DSO/VPP experiment suite.")
    parser.add_argument("--config", default=str(PROJECT_ROOT / "configs" / "european_lv_benchmark_v2.yaml"))
    parser.add_argument(
        "--topology-holdout-config",
        default=str(PROJECT_ROOT / "configs" / "ieee33_multi_vpp.yaml"),
    )
    parser.add_argument(
        "--safety-tight-config",
        default=str(PROJECT_ROOT / "configs" / "european_lv_benchmark_v2_safety_tight.yaml"),
    )
    parser.add_argument("--sanity-config", default=str(PROJECT_ROOT / "configs" / "lv_taiqu_demo.yaml"))
    parser.add_argument("--output-dir", default=str(PROJECT_ROOT / "outputs" / "benchmark_v2"))
    parser.add_argument(
        "--report-output-dir",
        default=str(PROJECT_ROOT / "outputs"),
        help="Only used with --refresh-root-report. Benchmark-aware HTML is always written to --output-dir by default.",
    )
    parser.add_argument("--horizon-steps", type=int, default=288)
    parser.add_argument("--seeds", default="3101,3102,3103,3104,3105")
    parser.add_argument("--train-variants", default="train_mixed")
    parser.add_argument("--eval-variants", default="holdout_peak,holdout_cloudy,holdout_reverseflow")
    parser.add_argument("--topology-holdout-variants", default="holdout_peak")
    parser.add_argument(
        "--variants",
        default=None,
        help="Deprecated compatibility option: first value maps to train variant, remaining values map to eval variants.",
    )
    parser.add_argument("--algorithms", default="rule_based")
    parser.add_argument("--ctde-train-episodes", type=int, default=3)
    parser.add_argument("--ctde-train-horizon-steps", type=int, default=96)
    parser.add_argument("--ctde-eval-horizon-steps", type=int, default=None)
    parser.add_argument("--ctde-hidden-dim", type=int, default=64)
    parser.add_argument("--ctde-learning-rate", type=float, default=3e-4)
    parser.add_argument("--skip-topology-holdout", action="store_true")
    parser.add_argument("--skip-safety-tight", action="store_true")
    parser.add_argument("--include-sanity", action="store_true")
    parser.add_argument(
        "--skip-report-refresh",
        action="store_true",
        help="Run benchmark CSVs only. By default, benchmark-aware HTML and dashboard CSVs are refreshed in --output-dir.",
    )
    parser.add_argument(
        "--refresh-root-report",
        action="store_true",
        help="Also refresh the generic root outputs/interactive_report.html from a fresh rollout. This is not the benchmark report.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = BenchmarkExperimentConfig(
        config_path=args.config,
        safety_tight_config_path=args.safety_tight_config,
        topology_holdout_config_path=args.topology_holdout_config,
        sanity_config_path=args.sanity_config,
        output_dir=args.output_dir,
        horizon_steps=args.horizon_steps,
        seeds=_seed_arg(args.seeds),
        train_variants=_tuple_arg(args.train_variants),
        eval_variants=_tuple_arg(args.eval_variants),
        topology_holdout_variants=_tuple_arg(args.topology_holdout_variants),
        variants=_tuple_arg(args.variants) if args.variants else None,
        algorithms=_tuple_arg(args.algorithms),
        ctde_train_episodes=args.ctde_train_episodes,
        ctde_train_horizon_steps=args.ctde_train_horizon_steps,
        ctde_eval_horizon_steps=args.ctde_eval_horizon_steps,
        ctde_hidden_dim=args.ctde_hidden_dim,
        ctde_learning_rate=args.ctde_learning_rate,
        include_topology_holdout=not args.skip_topology_holdout,
        include_safety_tight=not args.skip_safety_tight,
        include_sanity=args.include_sanity,
        export_visualizations=not args.skip_report_refresh,
    )
    result = run_benchmark_experiment(cfg)
    metrics = result["seed_metrics"]
    print(f"output_dir={Path(result['output_dir']).resolve()}")
    print(f"runs={len(metrics)}")
    print(f"splits={sorted(metrics['split'].unique().tolist())}")
    print(f"algorithms={sorted(metrics['algorithm'].unique().tolist())}")
    print(f"min_voltage={metrics['min_voltage_vm_pu'].min():.4f}")
    print(f"max_line_loading={metrics['max_line_loading_percent'].max():.2f}")
    print(f"max_trafo_loading={metrics['max_trafo_loading_percent'].max():.2f}")
    print(f"security_pass_rate={metrics['security_pass'].mean():.3f}")
    print(f"report={result['report']}")
    print(
        "files=seed_metrics.csv,aggregate_metrics.csv,profile_quality.csv,experiment_manifest.json,"
        "benchmark_report.html,interactive_report.html,rl_architecture.html,vpp_first_person/index.html"
    )
    visualizations = result.get("visualizations", {})
    if visualizations:
        print("benchmark_visualization_refresh=completed")
        print(f"benchmark_interactive_report={visualizations['interactive_report']}")
        print(f"benchmark_rl_architecture_report={visualizations['rl_architecture_report']}")
        print(f"benchmark_vpp_first_person_report={visualizations['first_person_reports']['index']}")
        print(f"benchmark_model_update_summary={visualizations['model_update_summary']}")
    if args.refresh_root_report:
        first_seed = _seed_arg(args.seeds)[0]
        ctde_training_dir = Path(result["output_dir"]) / "training" / f"ctde_seed_{first_seed}"
        deep_rl_dir = ctde_training_dir if ctde_training_dir.exists() else None
        report_paths = refresh_visualization_outputs(
            config_path=args.config,
            output_dir=args.report_output_dir,
            deep_rl_dir=deep_rl_dir,
        )
        print("visualization_refresh=completed")
        print(f"interactive_report={report_paths['interactive_report']}")
        print(f"rl_architecture_report={report_paths['rl_architecture_report']}")
        print(f"vpp_first_person_report={report_paths['first_person_reports']['index']}")
        print(f"model_update_summary={report_paths['model_update_summary']}")


if __name__ == "__main__":
    main()
