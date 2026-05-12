from __future__ import annotations

from pathlib import Path

import pytest

from vpp_dso_sim.experiments import BenchmarkExperimentConfig, run_benchmark_experiment
from vpp_dso_sim.learning.deep_rl import torch_available
from vpp_dso_sim.network.builder import build_network
from vpp_dso_sim.network.european_lv import build_european_lv_benchmark_network
from vpp_dso_sim.network.powerflow import run_powerflow
from vpp_dso_sim.simulation.profiles import benchmark_profile_pack, profile_quality_summary
from vpp_dso_sim.simulation.scenario import load_scenario


def test_benchmark_v2_network_is_branched_and_converges():
    net = build_european_lv_benchmark_network()

    assert len(net.bus) == 123
    assert len(net.line) == 121
    assert set(net.line["line_section_type"]) == {"trunk", "lateral"}
    assert int((net.line["line_section_type"] == "lateral").sum()) > 20
    assert {"branch_id", "phase_hint", "zone_id"}.issubset(net.bus.columns)
    assert run_powerflow(net)
    assert bool(net.converged)


def test_benchmark_v2_profile_pack_is_not_repeated_day_replay():
    pack = benchmark_profile_pack(288, seed=3101, variant="train_mixed")
    reverse = benchmark_profile_pack(288, seed=3101, variant="holdout_reverseflow")
    quality = profile_quality_summary(pack["load"], pack["pv"], pack["price"])

    assert len(pack["load"]) == 288
    assert len(quality) == 3
    assert quality["load_mean"].nunique() > 1
    assert quality["pv_energy_pu_h"].nunique() > 1
    assert sum(reverse["pv"]) > sum(pack["pv"])
    assert sum(reverse["load"]) < sum(pack["load"])


def test_benchmark_v2_config_has_single_and_multi_node_vpps():
    scenario = load_scenario(Path("configs") / "european_lv_benchmark_v2.yaml")
    safety = load_scenario(Path("configs") / "european_lv_benchmark_v2_safety_tight.yaml")
    modes = {vpp.physical_mode() for vpp in scenario.vpps}

    assert modes == {"single_pcc", "multi_node"}
    assert len(scenario.vpps) >= 7
    assert scenario.dso.reward_privacy_mode == "privacy_preserving_proxy"
    assert safety.dso.voltage_limits == (0.95, 1.05)
    assert len(safety.vpps) == len(scenario.vpps)
    assert scenario.config["profiles"]["profile_pack"] == "benchmark_3day_v1"
    assert scenario.load_profile[:96] != scenario.load_profile[96:192]


def test_european_lv_network_kwargs_passthrough():
    net_low = build_network({"network": {"type": "european_lv", "base_load_scale": 0.8}})
    net_high = build_network({"network": {"type": "european_lv", "base_load_scale": 1.3}})

    assert float(net_high.load["p_mw"].sum()) > float(net_low.load["p_mw"].sum()) * 1.5


def test_european_lv_config_ignores_dso_constraint_keys_in_builder():
    scenario = load_scenario(Path("configs") / "european_lv_mixed_vpp.yaml")

    assert len(scenario.net.bus) == 123
    assert run_powerflow(scenario.net)


def test_benchmark_v2_feeder_resource_coverage():
    scenario = load_scenario(Path("configs") / "european_lv_benchmark_v2.yaml")
    feeders_with_der = set()
    for vpp in scenario.vpps:
        for der in vpp.der_list:
            feeder_id = str(scenario.net.bus.at[der.bus, "feeder_id"])
            if feeder_id.startswith("F"):
                feeders_with_der.add(feeder_id)

    assert {"F1", "F2", "F3", "F4", "F5", "F6"}.issubset(feeders_with_der)


def test_benchmark_runner_writes_metrics():
    output_dir = Path("outputs") / "test_benchmark_v2_runner"
    result = run_benchmark_experiment(
        BenchmarkExperimentConfig(
            horizon_steps=4,
            seeds=(101,),
            train_variants=("train_mixed",),
            eval_variants=("holdout_peak",),
            topology_holdout_variants=("holdout_cloudy",),
            output_dir=output_dir,
        )
    )

    assert len(result["seed_metrics"]) == 4
    assert set(result["seed_metrics"]["split"]) == {
        "train_profile",
        "eval_profile",
        "safety_tight_limits",
        "topology_holdout",
    }
    assert {"algorithm", "split", "min_voltage_vm_pu", "max_line_loading_percent", "powerflow_fail_count"}.issubset(
        result["seed_metrics"].columns
    )
    assert (output_dir / "seed_metrics.csv").exists()
    assert (output_dir / "aggregate_metrics.csv").exists()
    assert (output_dir / "profile_quality.csv").exists()
    assert (output_dir / "experiment_manifest.json").exists()
    assert result["report"].exists()
    assert (output_dir / "interactive_report.html").exists()
    assert (output_dir / "rl_architecture.html").exists()
    assert (output_dir / "vpp_first_person" / "index.html").exists()
    assert (output_dir / "dashboard_data" / "benchmark_seed_metrics.csv").exists()
    html = (output_dir / "interactive_report.html").read_text(encoding="utf-8")
    assert "Benchmark-aware UI" in html
    assert "seed_metrics.csv" in html
    assert "dashboard_data/benchmark_seed_metrics.csv" in html


@pytest.mark.skipif(not torch_available(), reason="PyTorch is not installed")
def test_benchmark_runner_can_train_then_frozen_eval_ctde():
    output_dir = Path("outputs") / "test_benchmark_v2_ctde_runner"
    result = run_benchmark_experiment(
        BenchmarkExperimentConfig(
            horizon_steps=2,
            seeds=(202,),
            train_variants=("train_mixed",),
            eval_variants=("holdout_peak",),
            algorithms=("privacy_separated_ctde_actor_critic",),
            include_topology_holdout=True,
            ctde_train_episodes=1,
            ctde_train_horizon_steps=2,
            ctde_eval_horizon_steps=2,
            ctde_hidden_dim=16,
            output_dir=output_dir,
        )
    )

    metrics = result["seed_metrics"]
    assert len(metrics) == 3
    assert set(metrics["algorithm"]) == {"privacy_separated_ctde_actor_critic"}
    assert set(metrics["split"]) == {"train_profile", "eval_profile", "safety_tight_limits"}
    assert "checkpoint_path" in metrics.columns
    assert "frozen_eval_total_reward" in metrics.columns
    assert (output_dir / "training" / "ctde_seed_202" / "privacy_separated_ctde_checkpoint.pt").exists()
    assert (output_dir / "dashboard_data" / "model_update_summary.csv").exists()
    arch_html = (output_dir / "rl_architecture.html").read_text(encoding="utf-8")
    assert "train-then-frozen-eval" in arch_html
    assert "frozen_deterministic_mean_policy" in arch_html
    first_person_html = (output_dir / "vpp_first_person" / "index.html").read_text(encoding="utf-8")
    assert "Benchmark First-Person Replay" in first_person_html
