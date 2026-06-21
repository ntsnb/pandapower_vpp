from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd
import pytest

from vpp_dso_sim.learning.advanced_marl import (
    HAPPOConfig,
    _happo_config_from_yaml,
    evaluate_happo_checkpoint,
    torch_available,
    train_happo,
)


@pytest.mark.skipif(not torch_available() or importlib.util.find_spec("torch") is None, reason="PyTorch is not installed")
def test_structured_dso_actor_outputs_action_unit_envelope_parameters() -> None:
    import torch

    from vpp_dso_sim.dso.models.structured_happo_actor import StructuredDSOGaussianActor
    from vpp_dso_sim.dso.observation.happo_structured import StructuredDSOFlatSpec

    spec = StructuredDSOFlatSpec(
        global_dim=6,
        action_token_dim=13,
        object_token_dim=10,
        edge_feature_dim=9,
        max_action_units=3,
        max_network_objects=2,
        action_unit_vpp_indices=(0, 0, 1),
        vpp_ids=("vpp_a", "vpp_b"),
    )
    model = StructuredDSOGaussianActor(
        spec=spec,
        d_model=32,
        num_heads=4,
        num_layers=1,
        action_self_attention_layers=1,
        dropout=0.0,
    )
    x = torch.zeros(1, spec.flat_dim)
    offset = spec.global_dim
    offset += spec.max_action_units * spec.action_token_dim
    offset += spec.max_network_objects * spec.object_token_dim
    offset += spec.max_network_objects * spec.max_action_units * spec.edge_feature_dim
    x[:, offset : offset + spec.max_action_units] = 1.0
    offset += spec.max_action_units
    x[:, offset : offset + spec.max_network_objects] = 1.0
    offset += spec.max_network_objects
    x[:, offset : offset + spec.max_network_objects * spec.max_action_units] = 1.0

    mean, log_std = model(x)

    assert mean.shape == (1, spec.max_action_units * 6)
    assert log_std.shape == mean.shape
    assert model.envelope_channels == (
        "center_ratio",
        "width_ratio",
        "guidance_strength",
        "direction_absorb_logit",
        "direction_balanced_logit",
        "direction_inject_logit",
    )


@pytest.mark.skipif(not torch_available() or importlib.util.find_spec("torch") is None, reason="PyTorch is not installed")
def test_happo_uses_structured_dso_actor_when_config_requests_it(tmp_path) -> None:
    result = train_happo(
        config_path="configs/happo_sensitivity_attention_v1.yaml",
        output_dir=tmp_path,
        config=HAPPOConfig(
            horizon_steps=2,
            episodes=1,
            hidden_dim=32,
            ppo_epochs=1,
            seed=0,
            critic_use_action_summary=True,
            normalize_observations=True,
        ),
    )

    summary = result["summary"]
    assert summary["dso_actor_observation_mode"] == "structured_bipartite"
    assert summary["dso_actor_type"] == "sensitivity_attention_v1_structured_happo"
    assert summary["dso_input_dim"] > 26
    assert summary["structured_dso_actor_trainable"] is True
    update_metrics = pd.read_csv(tmp_path / "happo_update_metrics.csv")
    assert "dso_global_guidance" in set(update_metrics["role"])
    assert update_metrics["policy_loss"].notna().all()
    assert {"entropy_mean", "approx_kl"}.issubset(update_metrics.columns)
    assert update_metrics[["entropy_mean", "approx_kl"]].notna().all().all()


@pytest.mark.skipif(not torch_available() or importlib.util.find_spec("torch") is None, reason="PyTorch is not installed")
def test_happo_stability_fields_are_configured_and_logged(tmp_path) -> None:
    result = train_happo(
        config_path="configs/happo_sensitivity_attention_v1.yaml",
        output_dir=tmp_path,
        config=HAPPOConfig(
            horizon_steps=2,
            episodes=1,
            hidden_dim=32,
            ppo_epochs=2,
            seed=0,
            critic_use_action_summary=True,
            target_kl=0.02,
            normalize_observations=True,
            normalize_advantages=True,
            nan_guard=True,
        ),
    )

    summary = result["summary"]
    assert summary["target_kl"] == 0.02
    assert summary["normalize_observations"] is True
    assert summary["normalize_advantages"] is True
    assert summary["nan_guard"] is True
    assert "observation_normalization_stats" in summary
    assert summary["observation_normalization_stats"]["dso_obs_std_mean"] > 0.0
    update_metrics = pd.read_csv(tmp_path / "happo_update_metrics.csv")
    assert {"approx_kl", "target_kl", "target_kl_exceeded", "nan_guard_triggered"}.issubset(update_metrics.columns)
    assert update_metrics["target_kl"].notna().all()
    assert not update_metrics["nan_guard_triggered"].fillna(False).astype(bool).any()


@pytest.mark.skipif(not torch_available() or importlib.util.find_spec("torch") is None, reason="PyTorch is not installed")
def test_structured_happo_checkpoint_frozen_eval_runs(tmp_path) -> None:
    train_dir = tmp_path / "train"
    eval_dir = tmp_path / "eval"
    train = train_happo(
        config_path="configs/happo_sensitivity_attention_v1.yaml",
        output_dir=train_dir,
        config=HAPPOConfig(
            horizon_steps=2,
            episodes=1,
            hidden_dim=32,
            ppo_epochs=1,
            seed=0,
            critic_use_action_summary=True,
            normalize_observations=True,
        ),
    )

    eval_result = evaluate_happo_checkpoint(
        config_path="configs/happo_sensitivity_attention_v1.yaml",
        checkpoint_path=train["checkpoint"],
        output_dir=eval_dir,
        horizon_steps=2,
        seed=1,
    )

    assert eval_result["summary"]["evaluation_mode"] == "frozen_mean_argmax_actor"
    assert eval_result["summary"]["dso_actor_observation_mode"] == "structured_bipartite"
    assert eval_result["summary"]["dso_actor_type"] == "sensitivity_attention_v1_structured_happo"
    assert eval_result["summary"]["normalize_observations"] is True
    assert not eval_result["step_metrics"].empty
    assert (eval_dir / "happo_frozen_eval_summary.json").exists()


@pytest.mark.skipif(not torch_available() or importlib.util.find_spec("torch") is None, reason="PyTorch is not installed")
def test_happo_reads_trainer_stability_fields_from_yaml_when_config_not_passed(tmp_path) -> None:
    base = Path(
        "configs/algorithms/dso_sensitivity_attention/v1/happo_sensitivity_attention_v1.yaml"
    ).read_text(encoding="utf-8")
    tuned = (
        base.replace("horizon_steps: 96", "horizon_steps: 2")
        .replace("seed: 0", "seed: 7", 1)
        .replace("  target_kl: 0.02", "  target_kl: 0.017")
        .replace("  max_grad_norm: 0.5", "  max_grad_norm: 0.4")
    )
    tuned = tuned.replace(
        "  nan_guard: true\n",
        "  nan_guard: true\n  episodes: 1\n  ppo_epochs: 2\n  hidden_dim: 32\n  critic_use_action_summary: true\n",
    )
    config_path = tmp_path / "yaml_driven_happo.yaml"
    config_path.write_text(tuned, encoding="utf-8")

    result = train_happo(config_path=config_path, output_dir=tmp_path / "train")

    summary = result["summary"]
    assert summary["episodes"] == 1
    assert summary["horizon_steps"] == 2
    assert summary["seed"] == 7
    assert summary["target_kl"] == 0.017
    assert summary["normalize_observations"] is True
    assert summary["normalize_advantages"] is True
    assert summary["nan_guard"] is True
    assert summary["config_hash"]
    assert summary["config_path"] == str(config_path)
    update_metrics = pd.read_csv(tmp_path / "train" / "happo_update_metrics.csv")
    assert update_metrics["target_kl"].dropna().eq(0.017).all()


def test_happo_reads_trainer_device_from_yaml(tmp_path) -> None:
    base = Path(
        "configs/algorithms/dso_sensitivity_attention/v1/happo_sensitivity_attention_v1.yaml"
    ).read_text(encoding="utf-8")
    tuned = base.replace("  device: auto\n", "  device: cpu\n")
    config_path = tmp_path / "device_happo.yaml"
    config_path.write_text(tuned, encoding="utf-8")

    cfg = _happo_config_from_yaml(config_path)

    assert cfg.device == "cpu"


@pytest.mark.skipif(not torch_available() or importlib.util.find_spec("torch") is None, reason="PyTorch is not installed")
def test_happo_training_and_frozen_eval_record_resolved_device(tmp_path) -> None:
    train_dir = tmp_path / "train"
    eval_dir = tmp_path / "eval"
    train = train_happo(
        config_path="configs/happo_sensitivity_attention_v1.yaml",
        output_dir=train_dir,
        config=HAPPOConfig(
            horizon_steps=2,
            episodes=1,
            hidden_dim=32,
            ppo_epochs=1,
            seed=0,
            critic_use_action_summary=True,
            normalize_observations=True,
            device="cpu",
        ),
    )

    assert train["summary"]["requested_device"] == "cpu"
    assert train["summary"]["resolved_device"] == "cpu"
    saved_config = pd.read_json(train_dir / "happo_config.json", typ="series")
    assert saved_config["device"] == "cpu"

    eval_result = evaluate_happo_checkpoint(
        config_path="configs/happo_sensitivity_attention_v1.yaml",
        checkpoint_path=train["checkpoint"],
        output_dir=eval_dir,
        horizon_steps=2,
        seed=1,
    )

    assert eval_result["summary"]["requested_device"] == "cpu"
    assert eval_result["summary"]["resolved_device"] == "cpu"


@pytest.mark.skipif(not torch_available() or importlib.util.find_spec("torch") is None, reason="PyTorch is not installed")
def test_happo_applies_reward_shield_coefficients_from_yaml(tmp_path) -> None:
    config_path = tmp_path / "shield_coeff_happo.yaml"
    config_path.write_text(
        "\n".join(
            [
                "extends: configs/european_lv_benchmark_v2_sensitivity_attention_v1_reward_v2_minimal.yaml",
                "reward:",
                "  shield:",
                "    dso_penalty_coef: 0.25",
                "    dispatch_penalty_coef: 0.50",
            ]
        ),
        encoding="utf-8",
    )

    train = train_happo(
        config_path=config_path,
        output_dir=tmp_path / "shield_coeff_train",
        config=HAPPOConfig(
            horizon_steps=1,
            episodes=1,
            hidden_dim=16,
            ppo_epochs=1,
            seed=19,
            device="cpu",
        ),
    )

    assert train["summary"]["dso_shield_intervention_penalty_coef"] == pytest.approx(0.25)
    assert train["summary"]["dispatch_shield_intervention_penalty_coef"] == pytest.approx(0.50)


@pytest.mark.skipif(not torch_available() or importlib.util.find_spec("torch") is None, reason="PyTorch is not installed")
def test_happo_reports_step_progress(tmp_path) -> None:
    events: list[dict[str, object]] = []

    train_happo(
        config_path="configs/happo_sensitivity_attention_v1.yaml",
        output_dir=tmp_path,
        config=HAPPOConfig(
            horizon_steps=2,
            episodes=1,
            hidden_dim=32,
            ppo_epochs=1,
            seed=0,
            critic_use_action_summary=True,
            normalize_observations=True,
            device="cpu",
        ),
        progress_callback=events.append,
        progress_step_interval=1,
    )

    assert [event["phase"] for event in events] == ["train_step", "train_step"]
    assert [event["step"] for event in events] == [1, 2]
    assert all(event["horizon_steps"] == 2 for event in events)
    assert all(event["episodes"] == 1 for event in events)
    assert all("reward_so_far" in event for event in events)
