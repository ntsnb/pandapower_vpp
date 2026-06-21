from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path

import pandas as pd
import pytest

from vpp_dso_sim.experiments.paper_training import (
    PaperTrainingExperimentConfig,
    _baseline_comparison,
    _baseline_safety_gate_diagnostics,
    _print_progress,
    _validate_trainable_cuda_requirement,
    _run_baseline_rollout,
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


def test_happo_shared_rollout_defaults_are_off():
    happo_cfg = HAPPOConfig()
    paper_cfg = PaperTrainingExperimentConfig()

    assert happo_cfg.shared_rollout_enabled is False
    assert happo_cfg.shared_rollout_workers == 1
    assert happo_cfg.shared_rollout_backend == "serial"
    assert happo_cfg.rollout_fragment_steps is None
    assert happo_cfg.rollout_policy_version_check is True
    assert paper_cfg.happo_shared_rollout_enabled is False
    assert paper_cfg.happo_shared_rollout_workers == 1
    assert paper_cfg.happo_shared_rollout_backend == "serial"
    assert paper_cfg.happo_rollout_fragment_steps is None


def test_train_update_progress_csv_preserves_loss_and_step_fields(tmp_path):
    _print_progress(
        tmp_path,
        {
            "phase": "train_update",
            "message": "HAPPO training update completed",
            "run_id": "happo_base_train_mixed_seed_9401",
            "algorithm": "happo",
            "episode": 1,
            "step": 672,
            "global_step": 672,
            "gradient_step": 1,
            "critic_loss": 0.014767788350582123,
            "critic_grad_norm": 0.5569129586219788,
            "dso_policy_loss": -1.3803623914718628,
            "dispatch_policy_loss": -0.03383631951042584,
            "portfolio_policy_loss": -0.009964794984885625,
        },
        print_event=False,
    )

    progress = pd.read_csv(tmp_path / "experiment_progress.csv")

    row = progress.iloc[0]
    assert row["phase"] == "train_update"
    assert row["global_step"] == 672
    assert row["gradient_step"] == 1
    assert row["critic_loss"] == pytest.approx(0.014767788350582123)
    assert row["critic_grad_norm"] == pytest.approx(0.5569129586219788)
    assert row["dso_policy_loss"] == pytest.approx(-1.3803623914718628)
    assert row["dispatch_policy_loss"] == pytest.approx(-0.03383631951042584)
    assert row["portfolio_policy_loss"] == pytest.approx(-0.009964794984885625)


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
    assert (Path(result["output_dir"]) / "baseline_safety_gate.csv").exists()
    assert (
        Path(result["output_dir"])
        / "runs"
        / "ac_validated_search_reference_holdout_peak_seed_9701"
        / "simulator_results"
        / "ac_validated_search_metadata.csv"
    ).exists()
    assert "ac_certificate_safe_rate" in result["evaluation_seed_metrics"]
    assert "post_ac_security_pass_rate" in result["evaluation_seed_metrics"]
    readiness = result["claim_readiness"]
    assert readiness["execution_ready"] is True
    assert readiness["paper_claim_ready"] is False
    assert "optimal_or_upper_bound_claim" in set(result["claim_guardrails"]["rule_id"])


def test_baseline_rollout_reports_step_progress(tmp_path):
    events: list[dict[str, object]] = []

    _run_baseline_rollout(
        algorithm="rule_based",
        config_path=Path("configs") / "european_lv_mixed_vpp.yaml",
        output_dir=tmp_path / "baseline_progress",
        seed=9710,
        variant="holdout_peak",
        split="eval_profile",
        scenario_name="european_lv_mixed_vpp",
        horizon_steps=2,
        experiment_level="smoke",
        progress_callback=events.append,
        progress_step_interval=1,
    )

    assert [event["phase"] for event in events] == ["baseline_step", "baseline_step"]
    assert [event["step"] for event in events] == [1, 2]
    assert all(event["horizon_steps"] == 2 for event in events)


def test_paper_long_defaults_to_checkpoint_both():
    cfg = paper_training_preset("paper_long")

    assert cfg.checkpoint_selection == "both"
    assert cfg.happo_critic_use_action_summary is True


def test_paper_long_sensitivity_v1_preset_uses_structured_happo_config():
    cfg = paper_training_preset("paper_long_sensitivity_v1")

    assert cfg.preset == "paper_long_sensitivity_v1"
    assert str(cfg.config_path).endswith(
        "configs/european_lv_benchmark_v2_sensitivity_attention_v1_reward_v3_1_market_safety.yaml"
    )
    assert cfg.algorithms == ("rule_based", "no_flex", "ac_validated_search_reference", "happo")
    assert cfg.seeds == (9401, 9402, 9403, 9404, 9405)
    assert cfg.eval_variants == ("holdout_peak", "holdout_cloudy", "holdout_reverseflow")
    assert cfg.horizon_steps == 672
    assert cfg.eval_horizon_steps == 672
    assert cfg.train_episodes == 120
    assert cfg.checkpoint_selection == "both"
    assert cfg.happo_critic_use_action_summary is True
    assert cfg.happo_use_yaml_trainer_settings is True
    assert cfg.dispatch_actor_encoder_type == "set_attention_v1"
    assert cfg.require_cuda_for_trainable is True


def test_paper_long_sensitivity_v1_reward_v3_1_alias_matches_default_latest_reward():
    default_cfg = paper_training_preset("paper_long_sensitivity_v1")
    explicit_cfg = paper_training_preset("paper_long_sensitivity_v1_reward_v3_1_market_safety")

    assert default_cfg.config_path == explicit_cfg.config_path
    assert explicit_cfg.preset == "paper_long_sensitivity_v1_reward_v3_1_market_safety"

    scenario = load_scenario(explicit_cfg.config_path)

    assert scenario.dso.reward_config.is_v3_market_safety


def test_trainable_cuda_guard_blocks_paper_long_cpu_fallback():
    cfg = PaperTrainingExperimentConfig(require_cuda_for_trainable=True)

    with pytest.raises(RuntimeError, match="CUDA is required"):
        _validate_trainable_cuda_requirement(cfg, algorithm="happo", cuda_available=False)


def test_trainable_cuda_guard_allows_smoke_cpu_fallback():
    cfg = PaperTrainingExperimentConfig(require_cuda_for_trainable=False)

    _validate_trainable_cuda_requirement(cfg, algorithm="happo", cuda_available=False)


def test_sensitivity_v1_large_benchmark_config_extends_paper_scenario():
    scenario = load_scenario(Path("configs") / "european_lv_benchmark_v2_sensitivity_attention_v1.yaml")

    assert len(scenario.vpps) == 7
    assert scenario.config["network"]["type"] == "european_lv_benchmark_v2"
    assert scenario.config["dso"]["envelope_policy"] == "sensitivity_attention_v1"
    assert scenario.config["dso"]["observation_mode"] == "structured_bipartite"
    assert scenario.config["dso"]["max_action_units"] >= 25
    assert scenario.config["dso"]["max_network_objects"] >= 16
    assert scenario.config["trainer"]["target_kl"] == 0.02
    assert scenario.config["trainer"]["normalize_observations"] is True


def test_baseline_safety_gate_blocks_unsafe_ac_reference():
    pd = pytest.importorskip("pandas")
    cfg = paper_training_preset("paper_long")
    metrics = pd.DataFrame(
        [
            {
                "algorithm": "ac_validated_search_reference",
                "profile_variant": "holdout_reverseflow",
                "seed": 9401,
                "total_violation_cells": 12,
                "post_ac_violation_count": 12,
                "post_ac_powerflow_failed": 0,
                "certificate_failed_no_ac_safe_recovery_rate": 0.0,
                "fallback_to_current_dispatch_step_count": 0,
                "is_ac_validated": True,
            }
        ]
    )

    gate = _baseline_safety_gate_diagnostics(cfg=cfg, evaluation_seed_metrics=metrics)

    assert "ac_reference_post_ac_unsafe" in set(gate["check_id"])
    assert bool(gate["block_execution"].any())


def test_baseline_safety_gate_passes_clean_ac_reference():
    pd = pytest.importorskip("pandas")
    cfg = paper_training_preset("paper_long")
    metrics = pd.DataFrame(
        [
            {
                "algorithm": "ac_validated_search_reference",
                "profile_variant": "holdout_peak",
                "seed": 9401,
                "total_violation_cells": 0,
                "post_ac_violation_count": 0,
                "post_ac_powerflow_failed": 0,
                "certificate_failed_no_ac_safe_recovery_rate": 0.0,
                "fallback_to_current_dispatch_step_count": 0,
                "is_ac_validated": True,
            }
        ]
    )

    gate = _baseline_safety_gate_diagnostics(cfg=cfg, evaluation_seed_metrics=metrics)

    assert set(gate["check_id"]) == {"ac_reference_baseline_gate_passed"}
    assert not bool(gate["block_execution"].any())


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


def test_paper_long_sensitivity_v1_rejects_nonempty_output_without_resume(tmp_path):
    (tmp_path / "stale.csv").write_text("old,result\n", encoding="utf-8")
    cfg = replace(
        paper_training_preset("paper_long_sensitivity_v1"),
        output_dir=tmp_path,
        algorithms=("rule_based",),
        seeds=(9705,),
        horizon_steps=1,
        eval_horizon_steps=1,
        train_episodes=1,
        tensorboard=False,
        export_html=False,
    )

    with pytest.raises(RuntimeError, match="not empty"):
        run_paper_training_experiment(cfg)


@pytest.mark.skipif(not torch_available(), reason="PyTorch is not installed")
def test_paper_training_structured_happo_sensitivity_smoke():
    output_dir = Path("outputs") / "test_paper_training_structured_happo_sensitivity_smoke"
    cfg = replace(
        paper_training_preset("smoke"),
        config_path=Path("configs") / "european_lv_benchmark_v2_sensitivity_attention_v1.yaml",
        output_dir=output_dir,
        algorithms=("happo",),
        seeds=(9704,),
        horizon_steps=2,
        eval_horizon_steps=2,
        train_episodes=1,
        hidden_dim=16,
        batch_size=4,
        ppo_epochs=1,
        tensorboard=False,
        export_html=False,
        checkpoint_selection="final",
        happo_critic_use_action_summary=True,
        happo_use_yaml_trainer_settings=True,
    )

    result = run_paper_training_experiment(cfg)

    summary_path = (
        Path(result["output_dir"])
        / "runs"
        / "happo_base_train_mixed_seed_9704"
        / "train"
        / "happo_training_summary.json"
    )
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["dso_actor_type"] == "sensitivity_attention_v1_structured_happo"
    assert summary["dso_actor_observation_mode"] == "structured_bipartite"
    assert summary["target_kl"] == 0.02
    assert summary["normalize_observations"] is True
    assert summary["normalize_advantages"] is True
    assert summary["nan_guard"] is True
    assert summary["critic_use_action_summary"] is True
    assert (Path(result["output_dir"]) / "training_episode_metrics.csv").exists()
    assert (Path(result["output_dir"]) / "training_loss_metrics.csv").exists()
    progress = pd.read_csv(Path(result["output_dir"]) / "experiment_progress.csv")
    train_step_rows = progress[progress["phase"] == "train_step"]
    assert not train_step_rows.empty
    assert train_step_rows["run_id"].eq("happo_base_train_mixed_seed_9704").all()
    assert train_step_rows["step"].max() == 2
    assert "reward_so_far" in train_step_rows.columns


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
