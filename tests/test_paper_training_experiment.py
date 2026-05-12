from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path

import pytest

from vpp_dso_sim.experiments.paper_training import (
    _baseline_comparison,
    paper_training_preset,
    run_paper_training_experiment,
)
from vpp_dso_sim.optimization import oracle_baseline
from vpp_dso_sim.optimization.oracle_baseline import build_ac_validated_search_actions
from vpp_dso_sim.learning.advanced_marl import (
    HAPPOConfig,
    evaluate_happo_checkpoint,
    torch_available,
    train_happo,
)
from vpp_dso_sim.simulation.scenario import load_scenario
from vpp_dso_sim.simulation.profiles import smart_ds_austin_profile_pack


def test_smart_ds_profile_pack_returns_long_profiles_or_explicit_fallback():
    pack = smart_ds_austin_profile_pack(horizon_steps=12, seed=123, variant="holdout_peak")

    assert len(pack["load"]) == 12
    assert len(pack["pv"]) == 12
    assert len(pack["price"]) == 12
    assert "metadata" in pack
    assert "source" in pack["metadata"]


def test_paper_training_baseline_smoke_writes_static_report():
    cfg = replace(
        paper_training_preset("smoke"),
        output_dir=Path("outputs") / "test_paper_training_baseline_smoke",
        algorithms=("rule_based", "no_flex", "static_fr_price_extreme_proxy", "ac_validated_search_reference"),
        seeds=(9701,),
        horizon_steps=2,
        eval_horizon_steps=2,
        tensorboard=False,
    )

    result = run_paper_training_experiment(cfg)

    assert result["html_path"].exists()
    assert (Path(result["output_dir"]) / "long_training_report_data.json").exists()
    assert not result["evaluation_seed_metrics"].empty
    assert {"rule_based", "no_flex", "static_fr_price_extreme_proxy", "ac_validated_search_reference"}.issubset(
        set(result["evaluation_seed_metrics"]["algorithm"])
    )
    ac_rows = result["evaluation_seed_metrics"][
        result["evaluation_seed_metrics"]["algorithm"] == "ac_validated_search_reference"
    ]
    assert bool(ac_rows["is_ac_validated"].all())
    assert not bool(ac_rows["is_upper_bound_claim_allowed"].any())
    assert set(ac_rows["baseline_role"]) == {"ac_validated_best_found_dispatch_reference"}
    assert int(ac_rows["search_budget"].sum()) > 0
    assert (Path(result["output_dir"]) / "architecture_diagnostics.csv").exists()
    assert (Path(result["output_dir"]) / "claim_guardrails.csv").exists()
    assert (Path(result["output_dir"]) / "claim_readiness.json").exists()
    assert "ac_certificate_safe_rate" in result["evaluation_seed_metrics"]
    assert "post_ac_security_pass_rate" in result["evaluation_seed_metrics"]
    readiness = result["claim_readiness"]
    assert readiness["execution_ready"] is True
    assert readiness["paper_claim_ready"] is False
    assert "optimal_or_upper_bound_claim" in set(result["claim_guardrails"]["rule_id"])


def test_paper_long_defaults_to_checkpoint_both():
    cfg = paper_training_preset("paper_long")

    assert cfg.checkpoint_selection == "both"
    assert cfg.happo_critic_use_action_summary is True


def test_baseline_comparison_keeps_checkpoint_labels_separate():
    frame = pytest.importorskip("pandas").DataFrame(
        [
            {"algorithm": "rule_based", "checkpoint_label": "baseline", "eval_total_reward": 10.0, "eval_total_cost": 5.0, "total_violation_cells": 0},
            {"algorithm": "happo", "checkpoint_label": "final", "eval_total_reward": 11.0, "eval_total_cost": 4.0, "total_violation_cells": 0},
            {"algorithm": "happo", "checkpoint_label": "train_best", "eval_total_reward": 12.0, "eval_total_cost": 3.0, "total_violation_cells": 0},
        ]
    )

    comparison = _baseline_comparison(frame)

    reward_rows = comparison[(comparison["algorithm"] == "happo") & (comparison["metric"] == "eval_total_reward")]
    assert set(reward_rows["checkpoint_label"]) == {"final", "train_best"}


def test_paper_training_rejects_legacy_oracle_output_manifest(tmp_path):
    (tmp_path / "experiment_manifest.json").write_text(
        json.dumps({"schema_version": "paper_training_v1", "config": {"algorithms": ["opf_oracle_proxy"]}}),
        encoding="utf-8",
    )
    cfg = replace(
        paper_training_preset("smoke"),
        output_dir=tmp_path,
        algorithms=("rule_based",),
        seeds=(9702,),
        horizon_steps=1,
        eval_horizon_steps=1,
        tensorboard=False,
        export_html=False,
    )

    with pytest.raises(RuntimeError, match="legacy baseline artifacts"):
        run_paper_training_experiment(cfg)


def test_paper_long_rejects_nonempty_output_without_resume(tmp_path):
    (tmp_path / "stale.csv").write_text("old,result\n", encoding="utf-8")
    cfg = replace(
        paper_training_preset("paper_long"),
        output_dir=tmp_path,
        algorithms=("rule_based",),
        seeds=(9703,),
        horizon_steps=1,
        eval_horizon_steps=1,
        train_episodes=1,
        tensorboard=False,
        export_html=False,
    )

    with pytest.raises(RuntimeError, match="not empty"):
        run_paper_training_experiment(cfg)


def test_ac_validated_reference_returns_explicit_der_dispatch_actions():
    scenario = load_scenario(Path("configs") / "european_lv_benchmark_v2.yaml")
    result = build_ac_validated_search_actions(scenario, step=0, price=float(scenario.price_profile[0]), max_candidates=4)

    assert result.metadata["baseline_role"] == "ac_validated_best_found_dispatch_reference"
    assert result.metadata["is_ac_validated"] is True
    for vpp in scenario.vpps:
        action = result.actions[str(vpp.id)]
        assert action["action_mode"] == "ac_validated_best_found_explicit_der_dispatch"
        assert set(action["der_dispatch_p_mw"]).issuperset({str(der.id) for der in vpp.der_list})


def test_ac_validated_reference_fallback_is_not_marked_ac_validated(monkeypatch):
    class UnsafeCertificate:
        ac_safe = False

    monkeypatch.setattr(
        oracle_baseline,
        "certify_or_repair_dispatch",
        lambda *args, **kwargs: UnsafeCertificate(),
    )
    scenario = load_scenario(Path("configs") / "european_lv_benchmark_v2.yaml")

    result = oracle_baseline.build_ac_validated_search_actions(
        scenario,
        step=0,
        price=float(scenario.price_profile[0]),
        max_candidates=2,
    )

    assert result.metadata["is_ac_validated"] is False
    assert result.metadata["fallback_to_current_dispatch"] is True
    assert result.metadata["feasible_candidate_count"] == 0
    assert result.metadata["reference_scope"] == "fallback_current_dispatch_no_ac_feasible_candidate"


@pytest.mark.skipif(not torch_available(), reason="PyTorch is not installed")
def test_happo_checkpoint_frozen_eval_runs():
    train_dir = Path("outputs") / "test_happo_eval_smoke" / "train"
    eval_dir = Path("outputs") / "test_happo_eval_smoke" / "eval"
    train = train_happo(
        config_path=Path("configs") / "european_lv_mixed_vpp.yaml",
        output_dir=train_dir,
        config=HAPPOConfig(
            episodes=1,
            horizon_steps=2,
            hidden_dim=16,
            ppo_epochs=1,
            seed=73,
        ),
    )

    eval_result = evaluate_happo_checkpoint(
        config_path=Path("configs") / "european_lv_mixed_vpp.yaml",
        checkpoint_path=train["checkpoint"],
        output_dir=eval_dir,
        horizon_steps=2,
        seed=74,
    )

    assert eval_result["summary"]["evaluation_mode"] == "frozen_mean_argmax_actor"
    assert eval_result["summary"]["total_violation_count"] >= 0
    assert not eval_result["step_metrics"].empty
    assert (eval_dir / "happo_frozen_eval_summary.json").exists()
