from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from vpp_dso_sim.experiments.algorithm_search import (  # noqa: E402
    AlgorithmSearchConfig,
    run_algorithm_search,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a lightweight MARL algorithm idea search without long training."
    )
    parser.add_argument(
        "--output-dir",
        default=str(PROJECT_ROOT / "outputs" / "algorithm_search"),
    )
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--min-candidates", type=int, default=20)
    parser.add_argument(
        "--registry-module",
        default="vpp_dso_sim.learning.advanced_marl",
        help="Optional registry module. Falls back to local candidates if absent.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run_algorithm_search(
        AlgorithmSearchConfig(
            output_dir=args.output_dir,
            top_k=args.top_k,
            min_candidates=args.min_candidates,
            registry_module=args.registry_module,
        )
    )
    scores = result["candidate_scores"]
    summary = result["summary"]
    top = scores[scores["recommendation_status"] == "recommended"]

    print(f"output_dir={Path(result['output_dir']).resolve()}")
    print(f"candidate_count={summary['candidate_count']}")
    print(f"top_k={summary['top_k']}")
    print(f"rejected_for_now_count={summary['rejected_for_now_count']}")
    print(f"candidate_scores={result['candidate_scores_path']}")
    print(f"summary={result['summary_path']}")
    print("training_executed=false")
    print("claim_boundary=metadata_proxy_search_not_deep_rl_training")
    print("top_recommendations=" + ",".join(top["algorithm_id"].astype(str).tolist()))


if __name__ == "__main__":
    main()
