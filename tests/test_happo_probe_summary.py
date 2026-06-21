from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from scripts.summarize_happo_probe import summarize_probe_root


def test_summarize_happo_probe_reads_episode_and_update_metrics(tmp_path):
    run = tmp_path / "seed_9401_base" / "runs" / "happo_base_train_mixed_seed_9401" / "train"
    run.mkdir(parents=True)
    pd.DataFrame(
        [
            {
                "episode": 0,
                "episode_reward": -10.0,
                "total_cost": 100.0,
                "violation_count": 0,
                "projection_gap_mw": 0.0,
                "critic_loss": 0.30,
            },
            {
                "episode": 1,
                "episode_reward": -8.0,
                "total_cost": 95.0,
                "violation_count": 0,
                "projection_gap_mw": 0.0,
                "critic_loss": 0.20,
            },
        ]
    ).to_csv(run / "happo_episode_metrics.csv", index=False)
    pd.DataFrame(
        [
            {
                "episode": 1,
                "role": "dso_global_guidance",
                "policy_loss": -1.0,
                "entropy_mean": 2.0,
                "approx_kl": 0.01,
            },
            {
                "episode": 1,
                "role": "vpp_1_dispatch",
                "policy_loss": -0.2,
                "entropy_mean": 1.5,
                "approx_kl": 0.02,
            },
        ]
    ).to_csv(run / "happo_update_metrics.csv", index=False)

    summary = summarize_probe_root(tmp_path)

    assert len(summary) == 1
    row = summary.iloc[0]
    assert int(row["seed"]) == 9401
    assert row["hparam_case"] == "base"
    assert float(row["final_reward"]) == -8.0
    assert float(row["mean_reward"]) == -9.0
    assert float(row["reward_std"]) == pytest.approx(1.0)
    assert bool(row["completed"]) is True
    assert bool(row["nan_or_inf"]) is False
    assert float(row["mean_policy_loss"]) == pytest.approx(-0.6)
    assert float(row["mean_entropy"]) == pytest.approx(1.75)
    assert float(row["mean_approx_kl"]) == pytest.approx(0.015)


def test_summarize_happo_probe_writes_csv_and_json(tmp_path):
    run = tmp_path / "seed_9402_lower_lr" / "runs" / "happo_lower_lr_train_mixed_seed_9402" / "train"
    run.mkdir(parents=True)
    pd.DataFrame(
        [
            {
                "episode": 0,
                "episode_reward": -5.0,
                "total_cost": 50.0,
                "violation_count": 1,
                "projection_gap_mw": 0.2,
            }
        ]
    ).to_csv(run / "happo_episode_metrics.csv", index=False)
    output_dir = tmp_path / "summary"

    summary = summarize_probe_root(tmp_path, output_dir=output_dir)

    assert len(summary) == 1
    assert (output_dir / "happo_probe_summary.csv").exists()
    json_path = output_dir / "happo_probe_summary.json"
    assert json_path.exists()
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload[0]["seed"] == 9402
    assert payload[0]["hparam_case"] == "lower_lr"
    assert np.isfinite(pd.read_csv(output_dir / "happo_probe_summary.csv")["final_reward"]).all()


def test_summarize_happo_probe_falls_back_to_stdout_log(tmp_path):
    log_dir = tmp_path / "seed_9403_higher_entropy" / "logs"
    log_dir.mkdir(parents=True)
    (log_dir / "paper_long_stdout.log").write_text(
        "\n".join(
            [
                "[HAPPO] episode=1/120 reward=-604.6740 cost=30335.7458 violations=0 projection_gap_mw=0.000000 critic_loss=0.010492 dso_loss=-4.141086 dispatch_loss=-0.113549 portfolio_loss=-0.033067",
                "[HAPPO] episode=2/120 reward=-597.5586 cost=29986.5085 violations=0 projection_gap_mw=0.000000 critic_loss=0.009071 dso_loss=-4.142814 dispatch_loss=-0.528545 portfolio_loss=-0.036857",
                "[HAPPO] episode=3/120 reward=-600.8508 cost=29918.9463 violations=0 projection_gap_mw=0.000000 critic_loss=0.007944 dso_loss=-4.144542 dispatch_loss=-0.507027 portfolio_loss=-0.027632",
            ]
        ),
        encoding="utf-8",
    )

    summary = summarize_probe_root(tmp_path)

    assert len(summary) == 1
    row = summary.iloc[0]
    assert row["source"] == "stdout_log"
    assert int(row["seed"]) == 9403
    assert row["hparam_case"] == "higher_entropy"
    assert int(row["episode_count"]) == 3
    assert float(row["final_reward"]) == pytest.approx(-600.8508)
    assert float(row["mean_reward"]) == pytest.approx((-604.6740 - 597.5586 - 600.8508) / 3)
    assert float(row["final_total_cost"]) == pytest.approx(29918.9463)
    assert float(row["total_violations"]) == 0.0
    assert float(row["projection_gap_mw_sum"]) == 0.0
    assert float(row["mean_critic_loss"]) == pytest.approx((0.010492 + 0.009071 + 0.007944) / 3)
    assert float(row["mean_dso_loss"]) == pytest.approx((-4.141086 - 4.142814 - 4.144542) / 3)
    assert float(row["mean_dispatch_loss"]) == pytest.approx((-0.113549 - 0.528545 - 0.507027) / 3)
