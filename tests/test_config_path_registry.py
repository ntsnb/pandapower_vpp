from pathlib import Path

import pytest

from vpp_dso_sim.utils.config import load_yaml, resolve_config_path


def test_legacy_and_canonical_demo_config_load_same_network_type():
    legacy = load_yaml("configs/ieee33_multi_vpp.yaml")
    canonical = load_yaml("configs/scenarios/demo/ieee33_multi_vpp.yaml")

    assert legacy["network"]["type"] == "ieee33"
    assert canonical["network"]["type"] == "ieee33"
    assert legacy["network"] == canonical["network"]


def test_registry_alias_loads_config():
    cfg = load_yaml("happo_sensitivity_attention_v1")

    assert cfg["name"] == "happo_sensitivity_attention_v1"
    assert cfg["dso"]["envelope_policy"] == "sensitivity_attention_v1"


def test_nested_extends_resolves_from_canonical_config():
    cfg = load_yaml(
        "configs/experiments/paper_long/sensitivity_attention_v1/"
        "european_lv_benchmark_v2_sensitivity_attention_v1_reward_v2_minimal.yaml"
    )

    assert cfg["network"]["type"] == "european_lv_benchmark_v2"
    assert cfg["reward"]["version"] == "v2_minimal"
    assert cfg["dso"]["envelope_policy"] == "sensitivity_attention_v1"


def test_paper_long_sensitivity_default_alias_uses_latest_reward_v3_1():
    cfg = load_yaml("paper_long_sensitivity_v1")

    assert cfg["reward"]["version"] == "v3_market_safety"


def test_resolve_config_path_reports_alias_target():
    resolved = resolve_config_path("reward_v2_minimal")

    assert resolved == Path("configs/rewards/v2_minimal/reward_v2_minimal.yaml").resolve()


def test_unknown_config_path_error_mentions_request():
    with pytest.raises(FileNotFoundError, match="does_not_exist_config"):
        load_yaml("does_not_exist_config")
