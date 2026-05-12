from __future__ import annotations

from pathlib import Path

from vpp_dso_sim.experiments.deep_rl_campaign import (
    DeepRLCandidateCampaignConfig,
    run_deep_rl_candidate_campaign,
)


def test_deep_rl_candidate_campaign_writes_full_plan_without_training():
    output_dir = Path("outputs") / "test_deep_rl_candidate_campaign"
    result = run_deep_rl_candidate_campaign(
        DeepRLCandidateCampaignConfig(
            output_dir=output_dir,
            min_candidates=20,
            top_k=5,
            train_top_k=3,
            execute_training=False,
            episodes=1,
            horizon_steps=4,
            eval_horizon_steps=4,
            seeds=(8101,),
        )
    )

    plan = result["plan"]
    results = result["results"]
    summary = result["summary"]

    assert len(plan) >= 20
    assert {"algorithm_id", "selected_for_training", "truth_level"}.issubset(plan.columns)
    assert {"candidate_id", "status", "truth_level", "reason"}.issubset(results.columns)
    assert int(summary["candidate_count"]) == len(plan)
    assert int(summary["trained_count"]) == 0
    assert (output_dir / "candidate_training_plan.csv").exists()
    assert (output_dir / "candidate_training_results.csv").exists()
    assert (output_dir / "campaign_summary.json").exists()
    assert (output_dir / "deep_rl_candidate_campaign.html").exists()
    assert "ctde_adapter_training" in set(plan["truth_level"])
