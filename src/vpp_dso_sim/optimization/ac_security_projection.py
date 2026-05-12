from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Any

import pandapower as pp

from vpp_dso_sim.network.constraints import check_network_constraints
from vpp_dso_sim.network.powerflow import run_powerflow


DispatchByVPP = dict[str, dict[str, float]]


@dataclass(frozen=True)
class ACDispatchCertificate:
    status: str
    dispatch_by_vpp: DispatchByVPP
    accepted_alpha: float
    candidate_violation_count: int
    repaired_violation_count: int
    candidate_powerflow_converged: bool
    repaired_powerflow_converged: bool
    repair_gap_mw: float
    checked_constraints: str = "bus_voltage,line_loading,trafo_loading,powerflow_convergence"

    @property
    def ac_safe(self) -> bool:
        return self.repaired_powerflow_converged and self.repaired_violation_count == 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "ac_certificate_status": self.status,
            "ac_certificate_safe": bool(self.ac_safe),
            "ac_certificate_accepted_alpha": float(self.accepted_alpha),
            "ac_certificate_candidate_violation_count": int(self.candidate_violation_count),
            "ac_certificate_repaired_violation_count": int(self.repaired_violation_count),
            "ac_certificate_candidate_powerflow_converged": bool(self.candidate_powerflow_converged),
            "ac_certificate_repaired_powerflow_converged": bool(self.repaired_powerflow_converged),
            "ac_certified_projection_gap_mw": float(self.repair_gap_mw),
            "ac_certificate_checked_constraints": self.checked_constraints,
        }


def current_dispatch_by_vpp(vpps: list[Any]) -> DispatchByVPP:
    return {
        str(vpp.id): {str(der.id): float(der.p_mw) for der in vpp.der_list}
        for vpp in vpps
    }


def dispatch_gap_mw(a: DispatchByVPP, b: DispatchByVPP) -> float:
    gap = 0.0
    for vpp_id, dispatch in a.items():
        other = b.get(vpp_id, {})
        for der_id, value in dispatch.items():
            gap += abs(float(value) - float(other.get(der_id, value)))
    return float(gap)


def interpolate_dispatch(current: DispatchByVPP, candidate: DispatchByVPP, alpha: float) -> DispatchByVPP:
    alpha = max(0.0, min(1.0, float(alpha)))
    merged: DispatchByVPP = {}
    for vpp_id, current_dispatch in current.items():
        candidate_dispatch = candidate.get(vpp_id, {})
        merged[vpp_id] = {}
        for der_id, current_value in current_dispatch.items():
            target = float(candidate_dispatch.get(der_id, current_value))
            merged[vpp_id][der_id] = float(current_value + alpha * (target - float(current_value)))
    return merged


def _write_der_to_net_without_state(net: pp.pandapowerNet, der: Any, p_mw: float, t: int) -> None:
    if der.pp_element_index is None:
        return
    der_preview = deepcopy(der)
    der_preview.metadata = dict(getattr(der, "metadata", {}))
    der_preview.metadata["current_t"] = int(t)
    # Match VPPAggregator.apply_dispatch_to_net semantics exactly: active
    # dispatch is written through the DER model with q_mvar=0.0. The copied DER
    # prevents SOC/temperature/ramp state mutation in the live simulator.
    der_preview.set_power(net, float(p_mw), 0.0)


def _dispatch_to_net_copy(base_net: pp.pandapowerNet, vpps: list[Any], dispatch: DispatchByVPP, t: int) -> pp.pandapowerNet:
    work_net = deepcopy(base_net)
    for vpp in vpps:
        for der in vpp.der_list:
            p = dispatch.get(str(vpp.id), {}).get(str(der.id), float(der.p_mw))
            _write_der_to_net_without_state(work_net, der, p, t)
    return work_net


def _validate_dispatch(
    base_net: pp.pandapowerNet,
    vpps: list[Any],
    dispatch: DispatchByVPP,
    t: int,
    *,
    voltage_limits: tuple[float, float],
    line_loading_limit_percent: float,
    trafo_loading_limit_percent: float,
) -> tuple[bool, int, bool]:
    work_net = _dispatch_to_net_copy(base_net, vpps, dispatch, t)
    converged = run_powerflow(work_net)
    report = check_network_constraints(
        work_net,
        voltage_limits=voltage_limits,
        line_loading_limit_percent=line_loading_limit_percent,
        trafo_loading_limit_percent=trafo_loading_limit_percent,
    )
    return bool(converged and report.ok), len(report.to_records(t)), bool(report.converged)


def certify_or_repair_dispatch(
    *,
    base_net: pp.pandapowerNet,
    vpps: list[Any],
    candidate_dispatch_by_vpp: DispatchByVPP,
    t: int,
    voltage_limits: tuple[float, float],
    line_loading_limit_percent: float,
    trafo_loading_limit_percent: float,
    max_backoff_iterations: int = 12,
) -> ACDispatchCertificate:
    """Verify a joint candidate dispatch with AC power flow and repair by backoff.

    This is not an AC-OPF envelope proof. It is a per-step AC power-flow replay
    certificate for the dispatch that is about to be written to pandapower.
    """

    current = current_dispatch_by_vpp(vpps)
    candidate_ok, candidate_violations, candidate_converged = _validate_dispatch(
        base_net,
        vpps,
        candidate_dispatch_by_vpp,
        t,
        voltage_limits=voltage_limits,
        line_loading_limit_percent=line_loading_limit_percent,
        trafo_loading_limit_percent=trafo_loading_limit_percent,
    )
    if candidate_ok:
        return ACDispatchCertificate(
            status="accepted_candidate_ac_safe",
            dispatch_by_vpp=candidate_dispatch_by_vpp,
            accepted_alpha=1.0,
            candidate_violation_count=candidate_violations,
            repaired_violation_count=0,
            candidate_powerflow_converged=candidate_converged,
            repaired_powerflow_converged=True,
            repair_gap_mw=0.0,
        )

    current_ok, current_violations, current_converged = _validate_dispatch(
        base_net,
        vpps,
        current,
        t,
        voltage_limits=voltage_limits,
        line_loading_limit_percent=line_loading_limit_percent,
        trafo_loading_limit_percent=trafo_loading_limit_percent,
    )
    if not current_ok:
        return ACDispatchCertificate(
            status="certificate_failed_current_dispatch_insecure",
            dispatch_by_vpp=current,
            accepted_alpha=0.0,
            candidate_violation_count=candidate_violations,
            repaired_violation_count=current_violations,
            candidate_powerflow_converged=candidate_converged,
            repaired_powerflow_converged=current_converged,
            repair_gap_mw=dispatch_gap_mw(candidate_dispatch_by_vpp, current),
        )

    low = 0.0
    high = 1.0
    best = current
    best_violations = current_violations
    best_converged = current_converged
    for _ in range(max(1, int(max_backoff_iterations))):
        mid = 0.5 * (low + high)
        trial = interpolate_dispatch(current, candidate_dispatch_by_vpp, mid)
        ok, violations, converged = _validate_dispatch(
            base_net,
            vpps,
            trial,
            t,
            voltage_limits=voltage_limits,
            line_loading_limit_percent=line_loading_limit_percent,
            trafo_loading_limit_percent=trafo_loading_limit_percent,
        )
        if ok:
            low = mid
            best = trial
            best_violations = violations
            best_converged = converged
        else:
            high = mid

    status = "repaired_by_ac_powerflow_backoff" if low > 1e-9 else "rolled_back_to_current_safe_dispatch"
    return ACDispatchCertificate(
        status=status,
        dispatch_by_vpp=best,
        accepted_alpha=float(low),
        candidate_violation_count=candidate_violations,
        repaired_violation_count=best_violations,
        candidate_powerflow_converged=candidate_converged,
        repaired_powerflow_converged=best_converged,
        repair_gap_mw=dispatch_gap_mw(candidate_dispatch_by_vpp, best),
    )
