from __future__ import annotations

from pathlib import Path

from vpp_dso_sim.network.constraints import check_network_constraints
from vpp_dso_sim.optimization import ac_security_projection
from vpp_dso_sim.optimization.ac_security_projection import certify_or_repair_dispatch
from vpp_dso_sim.optimization.oracle_baseline import build_ac_validated_search_actions
from vpp_dso_sim.simulation.scenario import load_scenario


def test_ac_certificate_dispatch_matches_final_apply_semantics():
    scenario = load_scenario(Path("configs") / "european_lv_benchmark_v2.yaml")
    reference = build_ac_validated_search_actions(
        scenario,
        step=0,
        price=float(scenario.price_profile[0]),
        max_candidates=4,
    )
    dispatch_by_vpp = {
        str(vpp_id): {str(der_id): float(p_mw) for der_id, p_mw in action["der_dispatch_p_mw"].items()}
        for vpp_id, action in reference.actions.items()
    }

    certificate = certify_or_repair_dispatch(
        base_net=scenario.net,
        vpps=scenario.vpps,
        candidate_dispatch_by_vpp=dispatch_by_vpp,
        t=0,
        voltage_limits=scenario.dso.voltage_limits,
        line_loading_limit_percent=scenario.dso.line_loading_limit_percent,
        trafo_loading_limit_percent=scenario.dso.trafo_loading_limit_percent,
    )

    assert certificate.ac_safe
    for vpp in scenario.vpps:
        vpp.apply_dispatch_to_net(scenario.net, certificate.dispatch_by_vpp[str(vpp.id)], 0)
    assert scenario.dso.run_powerflow()
    report = check_network_constraints(
        scenario.net,
        voltage_limits=scenario.dso.voltage_limits,
        line_loading_limit_percent=scenario.dso.line_loading_limit_percent,
        trafo_loading_limit_percent=scenario.dso.trafo_loading_limit_percent,
    )
    assert report.ok


class _DummyDER:
    def __init__(self, der_id: str, p_mw: float, p_min: float, p_max: float):
        self.id = der_id
        self.p_mw = p_mw
        self.metadata = {}
        self._p_min = p_min
        self._p_max = p_max

    def get_bounds(self, t: int):
        return self._p_min, self._p_max, 0.0, 0.0


class _DummyVPP:
    def __init__(self, vpp_id: str, ders: list[_DummyDER]):
        self.id = vpp_id
        self.der_list = ders


def test_ac_certificate_uses_emergency_recovery_when_current_is_unsafe(monkeypatch):
    vpps = [_DummyVPP("vpp", [_DummyDER("der", p_mw=1.0, p_min=-1.0, p_max=1.0)])]

    def fake_validate(*, dispatch):
        value = float(dispatch["vpp"]["der"])
        return value <= -0.5, 0 if value <= -0.5 else 3, True

    def validate_dispatch(base_net, vpps, dispatch, t, **kwargs):
        return fake_validate(dispatch=dispatch)

    monkeypatch.setattr(ac_security_projection, "_validate_dispatch", validate_dispatch)

    certificate = certify_or_repair_dispatch(
        base_net=None,
        vpps=vpps,
        candidate_dispatch_by_vpp={"vpp": {"der": 1.0}},
        t=0,
        voltage_limits=(0.94, 1.06),
        line_loading_limit_percent=100.0,
        trafo_loading_limit_percent=100.0,
        max_backoff_iterations=8,
    )

    assert certificate.ac_safe
    assert certificate.status.startswith("repaired_by_ac_powerflow_emergency_recovery")
    assert certificate.dispatch_by_vpp["vpp"]["der"] <= -0.5
    assert certificate.repair_gap_mw > 0.0


def test_ac_certificate_reports_hard_failure_when_no_recovery_is_safe(monkeypatch):
    vpps = [_DummyVPP("vpp", [_DummyDER("der", p_mw=1.0, p_min=-1.0, p_max=1.0)])]

    monkeypatch.setattr(
        ac_security_projection,
        "_validate_dispatch",
        lambda *args, **kwargs: (False, 2, True),
    )

    certificate = certify_or_repair_dispatch(
        base_net=None,
        vpps=vpps,
        candidate_dispatch_by_vpp={"vpp": {"der": 1.0}},
        t=0,
        voltage_limits=(0.94, 1.06),
        line_loading_limit_percent=100.0,
        trafo_loading_limit_percent=100.0,
    )

    assert not certificate.ac_safe
    assert certificate.status == "certificate_failed_no_ac_safe_recovery"
