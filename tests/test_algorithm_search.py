from __future__ import annotations

import json
import sys
from pathlib import Path
from types import ModuleType

import pandas as pd

from vpp_dso_sim.experiments.algorithm_search import (
    AlgorithmSearchConfig,
    SCORE_COLUMNS,
    load_algorithm_candidates,
    run_algorithm_search,
    score_algorithm_candidates,
)


def test_algorithm_search_writes_required_outputs_and_fields():
    output_dir = Path("outputs") / "test_algorithm_search"
    result = run_algorithm_search(
        AlgorithmSearchConfig(
            output_dir=output_dir,
            top_k=5,
            min_candidates=20,
        )
    )

    scores = result["candidate_scores"]
    required_columns = {
        "rank",
        "algorithm_id",
        "family",
        "recommendation_status",
        "proxy_score",
        "keep_reason",
        "rejection_reason",
        *SCORE_COLUMNS,
    }

    assert len(scores) >= 20
    assert required_columns.issubset(scores.columns)
    assert (output_dir / "candidate_scores.csv").exists()
    assert (output_dir / "summary.json").exists()

    loaded_scores = pd.read_csv(output_dir / "candidate_scores.csv")
    with (output_dir / "summary.json").open(encoding="utf-8") as handle:
        summary = json.load(handle)

    assert len(loaded_scores) >= 20
    assert required_columns.issubset(loaded_scores.columns)
    assert summary["candidate_count"] == len(scores)
    assert summary["top_k"] == 5
    assert len(summary["top_recommendations"]) == 5
    assert set(scores["recommendation_status"]) == {"recommended", "rejected_for_now"}
    assert int((scores["recommendation_status"] == "recommended").sum()) == 5
    assert int((scores["recommendation_status"] == "rejected_for_now").sum()) >= 15
    assert scores.loc[scores["recommendation_status"] == "recommended", "keep_reason"].ne("").all()
    assert scores.loc[scores["recommendation_status"] == "rejected_for_now", "rejection_reason"].ne("").all()


def test_algorithm_search_reads_optional_advanced_marl_registry(monkeypatch):
    module_name = "vpp_dso_sim.learning.advanced_marl"
    fake_module = ModuleType(module_name)
    fake_module.ALGORITHM_REGISTRY = {
        "registry_test_mappo": {
            "family": "MAPPO",
            "idea": "Registry candidate with local actors and a centralized critic.",
            "action_space": "continuous_gaussian",
            "reward_mode": "role_specific_general_sum",
            "privacy_mode": "local_actor_centralized_critic",
            "heterogeneity_model": "set_encoder",
            "engineering_stage": "registry_candidate",
            "tags": ("ctde", "local_actor", "continuous", "general_sum", "set_encoder"),
        }
    }
    monkeypatch.setitem(sys.modules, module_name, fake_module)

    candidates, report = load_algorithm_candidates(module_name, min_candidates=20)

    assert report.registry_available is True
    assert report.registry_candidate_count == 1
    assert len(candidates) >= 20
    assert "registry_test_mappo" in {candidate.algorithm_id for candidate in candidates}


def test_algorithm_search_reads_project_advanced_marl_registry():
    candidates, report = load_algorithm_candidates(
        "vpp_dso_sim.learning.advanced_marl",
        min_candidates=20,
    )

    candidate_ids = {candidate.algorithm_id for candidate in candidates}
    assert report.registry_available is True
    assert report.registry_candidate_count >= 20
    assert report.fallback_candidate_count == 0
    assert {"mappo", "happo", "matd3", "mappo_gnn_critic"}.issubset(candidate_ids)


def test_project_registry_candidates_keep_ctde_privacy_and_continuous_tags():
    candidates, _ = load_algorithm_candidates(
        "vpp_dso_sim.learning.advanced_marl",
        min_candidates=20,
    )
    by_id = {candidate.algorithm_id: candidate for candidate in candidates}
    scores = score_algorithm_candidates([by_id["mappo"], by_id["happo"], by_id["matd3"]])

    assert scores["privacy_fit"].min() >= 0.80
    assert scores["continuous_action_fit"].min() >= 0.80
    assert scores["heterogeneity_fit"].min() >= 0.70
