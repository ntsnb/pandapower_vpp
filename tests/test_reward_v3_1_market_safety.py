from __future__ import annotations

from pathlib import Path

import pytest

from vpp_dso_sim.der.evcs import EVCSModel
from vpp_dso_sim.der.hvac import HVACModel
from vpp_dso_sim.der.pv import PVModel
from vpp_dso_sim.der.storage import StorageModel
from vpp_dso_sim.entities.dso import DSO
from vpp_dso_sim.entities.vpp import VPPAggregator
from vpp_dso_sim.envs.reward_design import vpp_dispatch_reward_components
from vpp_dso_sim.learning.reward_config import RewardConfig
from vpp_dso_sim.network.constraints import ConstraintReport, ConstraintViolation
from vpp_dso_sim.optimization.ac_security_projection import ACDispatchCertificate
from vpp_dso_sim.simulation.scenario import load_scenario
from vpp_dso_sim.simulation.simulator import Simulator
from vpp_dso_sim.simulation.settlement import build_settlement_audit


class _RewardProbeVPP:
    def __init__(self, delivered_p_mw: float = 0.0) -> None:
        self._delivered_p_mw = float(delivered_p_mw)

    def current_power_mw(self) -> float:
        return self._delivered_p_mw

    def operating_cost(self) -> float:
        return 0.0

    def comfort_penalty(self, _t: int) -> float:
        return 0.0

    def soc_violation_penalty(self, _t: int) -> float:
        return 0.0


def _v3_config(**overrides) -> RewardConfig:
    payload = {
        "version": "v3_market_safety",
        "dso": {
            "raw_action_safety_weight": 10.0,
            "projected_action_safety_weight": 5.0,
            "min_raw_unsafe_penalty": 0.1,
            "raw_safety_epsilon": 1.0e-5,
            "welfare_baseline_mean": 0.0,
            "welfare_baseline_std": 10.0,
            "welfare_clip": 5.0,
            "soft_safety_gate_kappa": 2.0,
            **overrides.pop("dso", {}),
        },
        "vpp": {
            "dispatch": {
                "private_profit_weight": 1.0,
                **overrides.pop("dispatch", {}),
            }
        },
        **overrides,
    }
    return RewardConfig.from_dict(payload)


def test_v3_config_defaults_disable_proxy_market_terms() -> None:
    cfg = RewardConfig.from_dict({"version": "v3_market_safety"})

    assert cfg.is_v3_market_safety
    assert cfg.vpp.dispatch.service_payment_weight == 0.0
    assert cfg.vpp.dispatch.availability_payment_weight == 0.0
    assert cfg.vpp.dispatch.contract_delivery_weight == 0.0
    assert cfg.vpp.dispatch.service_payment_source == "disabled"
    assert cfg.vpp.dispatch.availability_payment_source == "disabled"
    assert cfg.vpp.dispatch.contract_settlement_source == "disabled"
    assert cfg.vpp.dispatch.storage_potential_shaping_weight == pytest.approx(0.02)
    assert cfg.vpp.dispatch.storage_terminal_value_weight == 0.0
    assert cfg.vpp.dispatch.storage_terminal_soc_reference_mode == "disabled"
    assert cfg.dso.raw_action_safety_weight == pytest.approx(10.0)
    assert cfg.dso.projected_action_safety_weight == pytest.approx(5.0)
    assert cfg.dso.min_raw_unsafe_penalty == pytest.approx(0.1)


def test_v3_dispatch_reward_uses_settlement_surplus_and_ignores_disabled_proxy_payments() -> None:
    cfg = _v3_config()
    components = vpp_dispatch_reward_components(
        vpp=_RewardProbeVPP(delivered_p_mw=-0.20),
        envelope={"price": 80.0, "p_min_mw": -1.0, "p_max_mw": 1.0},
        audit={
            "operational_surplus": 12.5,
            "service_payment": 100.0,
            "availability_payment": 50.0,
            "contract_penalty": 25.0,
            "settlement_audit_complete": 1.0,
            "settlement_power_balance_ok": 1.0,
        },
        dt_hours=0.25,
        t=0,
        reward_config=cfg,
    )

    assert components["private_profit_proxy"] == pytest.approx(12.5)
    assert components["vpp_operational_surplus_ex_transfer"] == pytest.approx(12.5)
    assert components["service_payment"] == 0.0
    assert components["availability_payment"] == 0.0
    assert components["contract_delivery_penalty"] == 0.0
    assert components["storage_potential_shaping_weight"] == pytest.approx(
        cfg.vpp.dispatch.storage_potential_shaping_weight
    )
    assert components["vpp_dispatch_reward"] == pytest.approx(12.5)
    assert components["settlement_audit_complete"] == 1.0
    assert components["settlement_power_balance_ok"] == 1.0


def test_der_settlement_distinguishes_evcs_import_from_storage_charge() -> None:
    pv = PVModel(
        id="pv_a",
        name="pv_a",
        bus=1,
        owner_vpp_id="vpp_a",
        p_mw=0.10,
        p_max_mw=0.20,
        forecast_profile=[1.0],
    )
    evcs = EVCSModel(
        id="evcs_a",
        name="evcs_a",
        bus=2,
        owner_vpp_id="vpp_a",
        p_mw=-0.08,
        n_evs=2,
        p_charge_max_mw=0.10,
    )
    storage = StorageModel(
        id="ess_a",
        name="ess_a",
        bus=3,
        owner_vpp_id="vpp_a",
        p_mw=-0.04,
        capacity_mwh=1.0,
        soc=0.50,
        p_charge_max_mw=0.10,
        p_discharge_max_mw=0.10,
    )
    evcs.metadata["evcs_retail_price_multiplier"] = 1.5
    vpp = VPPAggregator(id="vpp_a", name="vpp_a", pcc_bus=1, der_list=[pv, evcs, storage])

    der_rows, summaries = build_settlement_audit(
        vpps=[vpp],
        t=0,
        dt_hours=0.25,
        market_price=80.0,
        before_state={
            "evcs_a": {"average_soc": 0.25, "connected_evs": 2},
            "ess_a": {"soc": 0.50},
        },
        settlement_power_balance_tolerance_mw=1.0e-9,
    )
    summary = summaries["vpp_a"]

    assert len(der_rows) == 3
    assert summary["vpp_delivered_p_mw"] == pytest.approx(-0.02)
    assert summary["audit_reconstructed_p_mw"] == pytest.approx(-0.02)
    assert summary["settlement_power_balance_ok"] == 1.0
    assert summary["pv_export_revenue_total"] == pytest.approx(0.10 * 0.25 * 80.0)
    assert summary["evcs_user_revenue_total"] == pytest.approx(0.08 * 0.25 * 120.0)
    assert summary["evcs_wholesale_cost_total"] == pytest.approx(0.08 * 0.25 * 80.0)
    assert summary["storage_charge_cost_total"] == pytest.approx(0.04 * 0.25 * 80.0)
    assert summary["export_revenue_total"] == pytest.approx(summary["pv_export_revenue_total"])


def test_v3_settlement_keeps_raw_service_quality_penalties_out_of_economic_surplus() -> None:
    hvac = HVACModel(
        id="hvac_a",
        name="hvac_a",
        bus=2,
        owner_vpp_id="vpp_a",
        p_mw=-0.01,
        rated_power_mw=0.03,
        indoor_temp=50.0,
        setpoint_profile=[24.0],
    )
    vpp = VPPAggregator(id="vpp_a", name="vpp_a", pcc_bus=1, der_list=[hvac])

    _, summaries = build_settlement_audit(
        vpps=[vpp],
        t=0,
        dt_hours=0.25,
        market_price=80.0,
        settlement_power_balance_tolerance_mw=1.0e-9,
    )
    summary = summaries["vpp_a"]

    energy_cost = 0.01 * 0.25 * 80.0
    assert summary["comfort_cost_total"] > 0.0
    assert summary["service_quality_penalty_total"] == pytest.approx(summary["comfort_cost_total"])
    assert summary["economic_operational_surplus"] == pytest.approx(-energy_cost)
    assert summary["operational_surplus"] == pytest.approx(summary["economic_operational_surplus"])
    assert summary["quality_adjusted_operational_surplus"] == pytest.approx(
        summary["economic_operational_surplus"] - summary["service_quality_penalty_total"]
    )


def test_v3_dso_safety_gate_penalizes_raw_unsafe_even_when_projected_safe() -> None:
    cfg = _v3_config()
    report = ConstraintReport(converged=True, violations=[])
    dso = DSO(net={}, reward_config=cfg)

    safe = dso.calculate_reward_or_cost(
        report=report,
        vpp_settlement_summaries={
            "vpp_a": {"operational_surplus": 20.0, "dso_transfer_payment_cost": 999.0}
        },
        raw_action_voltage_violation_cost=0.0,
        projected_action_voltage_violation_cost=0.0,
    )
    raw_unsafe = dso.calculate_reward_or_cost(
        report=report,
        vpp_settlement_summaries={
            "vpp_a": {"operational_surplus": 20.0, "dso_transfer_payment_cost": 999.0}
        },
        raw_action_voltage_violation_cost=0.2,
        projected_action_voltage_violation_cost=0.0,
    )

    assert safe["reward_version_code"] == pytest.approx(3.1)
    assert safe["dso_transfer_payment_excluded"] == pytest.approx(999.0)
    assert safe["dso_vpp_welfare_raw"] == pytest.approx(20.0)
    assert safe["dso_safety_gate"] == pytest.approx(1.0)
    assert raw_unsafe["dso_raw_action_safety_penalty"] > 0.0
    assert raw_unsafe["dso_safety_gate"] < safe["dso_safety_gate"]
    assert raw_unsafe["dso_reward_train"] < safe["dso_reward_train"]


def test_v3_projected_unsafe_reduces_safety_gate() -> None:
    cfg = _v3_config()
    dso = DSO(net={}, reward_config=cfg)
    report = ConstraintReport(
        converged=True,
        violations=[
            ConstraintViolation(
                kind="bus_voltage_high",
                element="3",
                value=1.06,
                limit=1.05,
                magnitude=0.01,
            )
        ],
    )

    components = dso.calculate_reward_or_cost(
        report=report,
        vpp_settlement_summaries={"vpp_a": {"operational_surplus": 10.0}},
        raw_action_voltage_violation_cost=0.0,
        projected_action_voltage_violation_cost=0.5,
    )

    assert components["projected_action_safety_cost_norm"] > 0.0
    assert components["dso_safety_gate_input"] == pytest.approx(
        max(
            components["raw_action_safety_penalty_input"],
            components["projected_action_safety_cost_norm"],
        )
    )
    assert components["dso_safety_gate"] < 1.0
    assert components["cmdp_cost_voltage"] > 0.0


def test_v3_config_alias_loads_paper_long_sensitivity_config() -> None:
    scenario = load_scenario(
        Path("configs/experiments/paper_long/sensitivity_attention_v1")
        / "european_lv_benchmark_v2_sensitivity_attention_v1_reward_v3_1_market_safety.yaml"
    )

    assert scenario.dso.reward_config.is_v3_market_safety
    assert scenario.dso.reward_config.vpp.dispatch.service_payment_weight == 0.0
    assert scenario.dso.reward_config.vpp.dispatch.availability_payment_weight == 0.0
    assert scenario.dso.reward_config.vpp.dispatch.contract_delivery_weight == 0.0


def test_v3_simulator_feeds_ac_candidate_raw_safety_into_dso_reward(monkeypatch) -> None:
    scenario = load_scenario(
        Path("configs/experiments/paper_long/sensitivity_attention_v1")
        / "european_lv_benchmark_v2_sensitivity_attention_v1_reward_v3_1_market_safety.yaml"
    )
    scenario.horizon_steps = 1

    def fake_certificate(**kwargs):
        return ACDispatchCertificate(
            status="repaired_by_ac_powerflow_backoff",
            dispatch_by_vpp=kwargs["candidate_dispatch_by_vpp"],
            accepted_alpha=0.5,
            candidate_violation_count=2,
            repaired_violation_count=0,
            candidate_powerflow_converged=True,
            repaired_powerflow_converged=True,
            repair_gap_mw=0.05,
            candidate_voltage_violation_cost=0.20,
            candidate_line_overload_cost=0.30,
            candidate_trafo_overload_cost=0.0,
            candidate_powerflow_failure_cost=0.0,
            repaired_voltage_violation_cost=0.0,
            repaired_line_overload_cost=0.0,
            repaired_trafo_overload_cost=0.0,
            repaired_powerflow_failure_cost=0.0,
        )

    monkeypatch.setattr(
        "vpp_dso_sim.simulation.simulator.certify_or_repair_dispatch",
        fake_certificate,
    )

    result = Simulator(scenario).step(0)
    components = result["reward_components"]

    assert components["raw_action_voltage_violation_cost"] == pytest.approx(0.20)
    assert components["raw_action_line_overload_cost"] == pytest.approx(0.30)
    assert components["raw_action_safety_cost_norm"] == pytest.approx(0.50)
    assert components["raw_action_safety_penalty_input"] == pytest.approx(0.60)
    assert components["dso_safety_gate"] < 1.0
    assert components["projected_action_safety_cost_norm"] == pytest.approx(0.0)


def test_v3_dispatch_reward_rejects_incomplete_required_settlement_audit() -> None:
    cfg = _v3_config()

    with pytest.raises(ValueError, match="incomplete DER-level settlement audit"):
        vpp_dispatch_reward_components(
            vpp=_RewardProbeVPP(delivered_p_mw=-0.20),
            envelope={"price": 80.0, "p_min_mw": -1.0, "p_max_mw": 1.0},
            audit={
                "operational_surplus": 12.5,
                "settlement_audit_complete": 0.0,
                "settlement_power_balance_ok": 1.0,
            },
            dt_hours=0.25,
            t=0,
            reward_config=cfg,
        )
