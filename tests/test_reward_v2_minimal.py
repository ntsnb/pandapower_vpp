from __future__ import annotations

import math
from pathlib import Path

import pytest

from vpp_dso_sim.entities.dso import DSO
from vpp_dso_sim.envs.reward_design import (
    RewardConfig,
    PortfolioWindowTracker,
    attribute_projection_gaps,
    vpp_dispatch_reward_components,
)
from vpp_dso_sim.network.constraints import ConstraintReport
from vpp_dso_sim.utils.config import load_yaml


class _RewardProbeVPP:
    def __init__(
        self,
        *,
        delivered_p_mw: float = 0.0,
        cost: float = 0.0,
        comfort: float = 0.0,
        soc: float = 0.0,
        battery_power_mw: float = 0.0,
    ) -> None:
        self._delivered_p_mw = float(delivered_p_mw)
        self._cost = float(cost)
        self._comfort = float(comfort)
        self._soc = float(soc)
        self.der_list = [_BatteryProbe(battery_power_mw)]

    def current_power_mw(self) -> float:
        return self._delivered_p_mw

    def operating_cost(self) -> float:
        return self._cost

    def comfort_penalty(self, _t: int) -> float:
        return self._comfort

    def soc_violation_penalty(self, _t: int) -> float:
        return self._soc


class _BatteryProbe:
    def __init__(self, p_mw: float) -> None:
        self.p_mw = float(p_mw)
        self.capacity_mwh = 1.0


def test_vpp_yaml_weights_are_effective() -> None:
    envelope = {
        "price": 100.0,
        "p_min_mw": -1.0,
        "p_max_mw": 1.0,
        "preferred_target_p_mw": 0.0,
        "service_request": "export_or_reduce_load",
    }
    audit = {"baseline_p_mw": 0.1, "projected_target_p_mw": 0.3}
    base_cfg = RewardConfig.from_dict(
        {
            "version": "v2_minimal",
            "vpp": {"dispatch": {"private_profit_weight": 0.02}},
        }
    )
    stronger_cfg = RewardConfig.from_dict(
        {
            "version": "v2_minimal",
            "vpp": {"dispatch": {"private_profit_weight": 0.20}},
        }
    )

    base = vpp_dispatch_reward_components(
        vpp=_RewardProbeVPP(delivered_p_mw=0.3, cost=0.0),
        envelope=envelope,
        audit=audit,
        dt_hours=0.25,
        t=0,
        reward_config=base_cfg,
    )
    stronger = vpp_dispatch_reward_components(
        vpp=_RewardProbeVPP(delivered_p_mw=0.3, cost=0.0),
        envelope=envelope,
        audit=audit,
        dt_hours=0.25,
        t=0,
        reward_config=stronger_cfg,
    )

    assert base["private_profit_weight"] == pytest.approx(0.02)
    assert stronger["private_profit_weight"] == pytest.approx(0.20)
    assert stronger["vpp_dispatch_reward"] > base["vpp_dispatch_reward"]


def test_dispatch_private_profit_components_expose_formula_inputs() -> None:
    cfg = RewardConfig.from_dict(
        {
            "version": "v2_minimal",
            "vpp": {"dispatch": {"private_profit_weight": 0.02}},
        }
    )
    components = vpp_dispatch_reward_components(
        vpp=_RewardProbeVPP(delivered_p_mw=-0.4, cost=7.0),
        envelope={
            "price": 90.0,
            "p_min_mw": -1.0,
            "p_max_mw": 1.0,
            "preferred_target_p_mw": -0.4,
        },
        audit={"baseline_p_mw": 0.0, "projected_target_p_mw": -0.4},
        dt_hours=0.25,
        t=0,
        reward_config=cfg,
    )

    assert components["market_price"] == pytest.approx(90.0)
    assert components["dt_hours"] == pytest.approx(0.25)
    assert components["energy_market_revenue"] == pytest.approx(90.0 * -0.4 * 0.25)
    assert components["der_operation_cost"] == pytest.approx(7.0 * 0.25)
    assert components["private_profit_proxy"] == pytest.approx(
        components["energy_market_revenue"] - components["der_operation_cost"]
    )
    assert components["dispatch_private_profit_reward"] == pytest.approx(
        components["private_profit_weight"] * components["private_profit_proxy"]
    )


def test_dso_tracking_disabled_in_v2() -> None:
    cfg = RewardConfig.from_dict(
        {
            "version": "v2_minimal",
            "dso": {
                "enable_tracking_bonus": False,
                "enable_effective_response_bonus": False,
                "enable_target_tracking_cost": False,
                "comfort_violation_weight": 0.0,
                "soc_violation_weight": 0.0,
            },
        }
    )
    dso = DSO(net={}, reward_config=cfg)
    safe = ConstraintReport(converged=True, violations=[])

    low_error = dso.calculate_reward_or_cost(report=safe, target_tracking_error=0.0)
    high_error = dso.calculate_reward_or_cost(report=safe, target_tracking_error=5.0)

    assert low_error["reward_version_code"] == pytest.approx(2.0)
    assert low_error["tracking_bonus_diagnostic"] == pytest.approx(0.25)
    assert high_error["tracking_bonus_diagnostic"] < low_error["tracking_bonus_diagnostic"]
    assert low_error["target_tracking_error_penalty_train_included"] == 0.0
    assert high_error["target_tracking_error_penalty_train_included"] == 0.0
    assert high_error["dso_reward_train"] == pytest.approx(low_error["dso_reward_train"])


def test_dso_comfort_soc_excluded_in_v2() -> None:
    cfg = RewardConfig.from_dict({"version": "v2_minimal"})
    dso_clean = DSO(net={}, reward_config=cfg)
    dso_penalized = DSO(net={}, reward_config=cfg)
    dso_penalized.vpp_registry["vpp"] = _RewardProbeVPP(comfort=10_000.0, soc=5_000.0)
    safe = ConstraintReport(converged=True, violations=[])

    clean = dso_clean.calculate_reward_or_cost(report=safe)
    penalized = dso_penalized.calculate_reward_or_cost(report=safe)

    assert penalized["raw_comfort_violation_penalty"] > clean["raw_comfort_violation_penalty"]
    assert penalized["raw_soc_violation_penalty"] > clean["raw_soc_violation_penalty"]
    assert clean["comfort_violation_penalty_train_included"] == 0.0
    assert clean["soc_violation_penalty_train_included"] == 0.0
    assert penalized["dso_reward_train"] == pytest.approx(clean["dso_reward_train"])


def test_dispatch_contract_delivery_uses_baseline_service_payment() -> None:
    cfg = RewardConfig.from_dict(
        {
            "version": "v2_minimal",
            "vpp": {
                "dispatch": {
                    "private_profit_weight": 0.0,
                    "service_payment_weight": 1.0,
                    "availability_payment_weight": 0.0,
                    "contract_delivery_weight": 10.0,
                    "comfort_soc_weight": 0.0,
                    "projection_linear_weight": 0.0,
                    "projection_quadratic_weight": 0.0,
                    "battery_degradation_weight": 0.0,
                }
            },
        }
    )
    envelope = {
        "price": 100.0,
        "p_min_mw": -1.0,
        "p_max_mw": 1.0,
        "preferred_target_p_mw": 0.5,
        "service_request": "export_or_reduce_load",
    }
    full = vpp_dispatch_reward_components(
        vpp=_RewardProbeVPP(delivered_p_mw=0.5),
        envelope=envelope,
        audit={"baseline_p_mw": 0.2, "projected_target_p_mw": 0.5},
        dt_hours=0.25,
        t=0,
        reward_config=cfg,
    )
    short = vpp_dispatch_reward_components(
        vpp=_RewardProbeVPP(delivered_p_mw=0.35),
        envelope=envelope,
        audit={"baseline_p_mw": 0.2, "projected_target_p_mw": 0.5},
        dt_hours=0.25,
        t=0,
        reward_config=cfg,
    )

    assert full["baseline_p_mw"] == pytest.approx(0.2)
    assert full["accepted_delta_p_mw"] == pytest.approx(0.3)
    assert full["actual_delta_p_mw"] == pytest.approx(0.3)
    assert full["verified_delivery_mw"] == pytest.approx(0.3)
    assert full["contract_shortfall_mw"] == pytest.approx(0.0)
    assert full["service_payment"] == pytest.approx(100.0 * 0.3 * 0.25)
    assert short["verified_delivery_mw"] == pytest.approx(0.15)
    assert short["contract_shortfall_mw"] == pytest.approx(0.15)
    assert short["contract_delivery_penalty"] == pytest.approx(10.0 * 0.15 * 0.15)


def test_portfolio_window_reward_only_settles_on_decision_step() -> None:
    cfg = RewardConfig.from_dict(
        {
            "version": "v2_minimal",
            "vpp": {
                "portfolio": {
                    "decision_interval_steps": 2,
                    "long_horizon_profit_weight": 1.0,
                    "verified_capacity_weight": 1.0,
                    "delivery_reliability_weight": 1.0,
                    "future_shield_penalty_weight": 1.0,
                    "future_projection_penalty_weight": 1.0,
                    "future_comfort_soc_weight": 1.0,
                    "switching_reweight_cost": 0.05,
                }
            },
        }
    )
    tracker = PortfolioWindowTracker(cfg)
    tracker.update(
        "vpp_1",
        {
            "private_profit_proxy": 2.0,
            "contract_shortfall_mw": 0.1,
            "dispatch_responsible_projection_gap_mw": 0.2,
            "scaled_comfort_soc_penalty": 0.3,
            "verified_delivery_mw": 0.4,
        },
        shield_intervention_gap_mw=0.5,
    )

    non_decision = tracker.settle_if_decision("vpp_1", step=1, action="reweight")
    decision = tracker.settle_if_decision("vpp_1", step=2, action="reweight")

    assert non_decision["vpp_portfolio_reward"] == 0.0
    assert non_decision["portfolio_decision_step"] == 0.0
    assert decision["portfolio_decision_step"] == 1.0
    assert decision["portfolio_window_profit"] == pytest.approx(2.0)
    assert decision["portfolio_window_contract_shortfall"] == pytest.approx(0.1)
    assert decision["portfolio_window_shield_intervention"] == pytest.approx(0.5)
    assert decision["portfolio_window_projection_gap"] == pytest.approx(0.2)
    assert decision["portfolio_window_comfort_soc_penalty"] == pytest.approx(0.3)
    assert decision["portfolio_window_verified_capacity"] == pytest.approx(0.4)
    expected = 2.0 + 0.4 - 0.1 - 0.5 - 0.2 - 0.3 - 0.05
    assert decision["vpp_portfolio_reward"] == pytest.approx(expected)


def test_projection_gap_attribution_outputs_actor_responsibility() -> None:
    result = attribute_projection_gaps(
        action_projection_gap_mw=0.7,
        local_bounds_projection_gap_mw=0.2,
        ac_aware_projection_gap_mw=0.1,
        ac_certified_projection_gap_mw=0.4,
        dispatch_audit={
            "vpp_1": {
                "projection_gap_mw": 0.2,
                "projection_gap_scope": "local_der_bounds_not_ac_security",
            },
            "vpp_2": {"projection_gap_mw": 0.0},
        },
    )

    assert result["dispatch_responsible_projection_gap_mw"]["vpp_1"] == pytest.approx(0.2)
    assert result["dso_responsible_projection_gap_mw"] == pytest.approx(0.1 + 0.4)
    assert result["attribution_method"] == "heuristic_scope_and_ac_gap"


def test_reward_no_nan_inf_in_v2_components() -> None:
    cfg = RewardConfig.from_dict({"version": "v2_minimal"})
    dso = DSO(net={}, reward_config=cfg)
    components = dso.calculate_reward_or_cost(report=ConstraintReport(converged=True, violations=[]))
    dispatch = vpp_dispatch_reward_components(
        vpp=_RewardProbeVPP(delivered_p_mw=0.0),
        envelope={"price": 80.0, "p_min_mw": -1.0, "p_max_mw": 1.0},
        audit={},
        dt_hours=0.25,
        t=0,
        reward_config=cfg,
    )

    for mapping in (components, dispatch):
        for key, value in mapping.items():
            if isinstance(value, (int, float)):
                assert math.isfinite(float(value)), key


def test_reward_v2_matrix_configs_resolve_expected_overrides() -> None:
    base = Path("configs") / "european_lv_benchmark_v2_sensitivity_attention_v1_reward_v2_minimal.yaml"
    legacy = Path("configs") / "european_lv_benchmark_v2_sensitivity_attention_v1_legacy_v1_reward.yaml"
    no_shield = Path("configs") / "european_lv_benchmark_v2_sensitivity_attention_v1_reward_v2_minimal_no_shield_eval.yaml"
    no_portfolio = (
        Path("configs")
        / "european_lv_benchmark_v2_sensitivity_attention_v1_reward_v2_minimal_no_portfolio_window_penalty.yaml"
    )
    preferred = (
        Path("configs")
        / "european_lv_benchmark_v2_sensitivity_attention_v1_reward_v2_minimal_preferred_bonus_0p05.yaml"
    )
    weight_5 = (
        Path("configs")
        / "european_lv_benchmark_v2_sensitivity_attention_v1_reward_v2_minimal_contract_delivery_weight_5.yaml"
    )
    weight_10 = (
        Path("configs")
        / "european_lv_benchmark_v2_sensitivity_attention_v1_reward_v2_minimal_contract_delivery_weight_10.yaml"
    )
    weight_20 = (
        Path("configs")
        / "european_lv_benchmark_v2_sensitivity_attention_v1_reward_v2_minimal_contract_delivery_weight_20.yaml"
    )

    assert RewardConfig.from_dict(load_yaml(base)["reward"]).version == "v2_minimal"
    assert RewardConfig.from_dict(load_yaml(legacy)["reward"]).version == "v1_legacy"

    no_shield_cfg = RewardConfig.from_dict(load_yaml(no_shield)["reward"])
    assert no_shield_cfg.shield.dso_penalty_coef == pytest.approx(0.0)
    assert no_shield_cfg.shield.dispatch_penalty_coef == pytest.approx(0.0)
    assert no_shield_cfg.shield.portfolio_future_penalty_coef == pytest.approx(0.0)
    assert no_shield_cfg.vpp.portfolio.future_shield_penalty_weight == pytest.approx(0.0)
    assert no_shield_cfg.vpp.portfolio.future_projection_penalty_weight == pytest.approx(0.0)

    no_portfolio_cfg = RewardConfig.from_dict(load_yaml(no_portfolio)["reward"])
    assert no_portfolio_cfg.vpp.portfolio.delivery_reliability_weight == pytest.approx(0.0)
    assert no_portfolio_cfg.vpp.portfolio.future_shield_penalty_weight == pytest.approx(0.0)
    assert no_portfolio_cfg.vpp.portfolio.future_projection_penalty_weight == pytest.approx(0.0)
    assert no_portfolio_cfg.vpp.portfolio.future_comfort_soc_weight == pytest.approx(0.0)

    preferred_cfg = RewardConfig.from_dict(load_yaml(preferred)["reward"])
    assert preferred_cfg.vpp.dispatch.preferred_region_bonus_weight == pytest.approx(0.05)

    assert RewardConfig.from_dict(load_yaml(weight_5)["reward"]).vpp.dispatch.contract_delivery_weight == pytest.approx(5.0)
    assert RewardConfig.from_dict(load_yaml(weight_10)["reward"]).vpp.dispatch.contract_delivery_weight == pytest.approx(10.0)
    assert RewardConfig.from_dict(load_yaml(weight_20)["reward"]).vpp.dispatch.contract_delivery_weight == pytest.approx(20.0)
