from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Any

import pandapower as pp

from vpp_dso_sim.network.constraints import check_network_constraints, violation_penalties
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
    candidate_voltage_violation_cost: float = 0.0
    candidate_line_overload_cost: float = 0.0
    candidate_trafo_overload_cost: float = 0.0
    candidate_powerflow_failure_cost: float = 0.0
    repaired_voltage_violation_cost: float = 0.0
    repaired_line_overload_cost: float = 0.0
    repaired_trafo_overload_cost: float = 0.0
    repaired_powerflow_failure_cost: float = 0.0

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
            "ac_candidate_voltage_violation_cost": float(self.candidate_voltage_violation_cost),
            "ac_candidate_line_overload_cost": float(self.candidate_line_overload_cost),
            "ac_candidate_trafo_overload_cost": float(self.candidate_trafo_overload_cost),
            "ac_candidate_powerflow_failure_cost": float(self.candidate_powerflow_failure_cost),
            "ac_repaired_voltage_violation_cost": float(self.repaired_voltage_violation_cost),
            "ac_repaired_line_overload_cost": float(self.repaired_line_overload_cost),
            "ac_repaired_trafo_overload_cost": float(self.repaired_trafo_overload_cost),
            "ac_repaired_powerflow_failure_cost": float(self.repaired_powerflow_failure_cost),
        }


@dataclass(frozen=True)
class ACDispatchValidation:
    ok: bool
    violation_count: int
    converged: bool
    voltage_violation_cost: float = 0.0
    line_overload_cost: float = 0.0
    trafo_overload_cost: float = 0.0
    powerflow_failure_cost: float = 0.0


def _as_validation(value: ACDispatchValidation | tuple[Any, ...]) -> ACDispatchValidation:
    """Accept legacy tuple validation results used by older tests/hooks."""

    if isinstance(value, ACDispatchValidation):
        return value
    if isinstance(value, tuple) and len(value) >= 3:
        return ACDispatchValidation(
            ok=bool(value[0]),
            violation_count=int(value[1]),
            converged=bool(value[2]),
        )
    raise TypeError(f"Unsupported AC dispatch validation result: {type(value).__name__}")


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


def _der_bounds(der: Any, t: int) -> tuple[float, float]:
    old_t = der.metadata.get("current_t") if hasattr(der, "metadata") else None
    had_old_t = hasattr(der, "metadata") and "current_t" in der.metadata
    if hasattr(der, "metadata"):
        der.metadata["current_t"] = int(t)
    try:
        p_min, p_max, _, _ = der.get_bounds(t)
    finally:
        if hasattr(der, "metadata"):
            if had_old_t:
                der.metadata["current_t"] = old_t
            else:
                der.metadata.pop("current_t", None)
    return float(p_min), float(p_max)


def _dispatch_from_bound_selector(vpps: list[Any], t: int, selector: str) -> DispatchByVPP:
    dispatch: DispatchByVPP = {}
    for vpp in vpps:
        dispatch[str(vpp.id)] = {}
        for der in vpp.der_list:
            p_min, p_max = _der_bounds(der, t)
            if selector == "min":
                value = p_min
            elif selector == "max":
                value = p_max
            elif selector == "zero":
                value = max(p_min, min(p_max, 0.0))
            else:
                value = 0.5 * (p_min + p_max)
            dispatch[str(vpp.id)][str(der.id)] = float(value)
    return dispatch


def _emergency_recovery_anchors(vpps: list[Any], t: int) -> list[tuple[str, DispatchByVPP]]:
    """Deterministic non-optimizing recovery anchors for unsafe current states."""

    anchors = [
        ("all_min_absorb_or_curtail", _dispatch_from_bound_selector(vpps, t, "min")),
        ("all_zero_neutral", _dispatch_from_bound_selector(vpps, t, "zero")),
        ("all_mid_capability", _dispatch_from_bound_selector(vpps, t, "mid")),
        ("all_max_generate_or_discharge", _dispatch_from_bound_selector(vpps, t, "max")),
    ]
    unique: list[tuple[str, DispatchByVPP]] = []
    seen: set[tuple[tuple[str, tuple[tuple[str, float], ...]], ...]] = set()
    for name, dispatch in anchors:
        key = tuple(
            sorted(
                (vpp_id, tuple(sorted((der_id, round(float(value), 9)) for der_id, value in der_dispatch.items())))
                for vpp_id, der_dispatch in dispatch.items()
            )
        )
        if key in seen:
            continue
        seen.add(key)
        unique.append((name, dispatch))
    return unique


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


def _search_from_unsafe_current_to_safe_anchor(
    *,
    base_net: pp.pandapowerNet,
    vpps: list[Any],
    current: DispatchByVPP,
    safe_anchor: DispatchByVPP,
    t: int,
    voltage_limits: tuple[float, float],
    line_loading_limit_percent: float,
    trafo_loading_limit_percent: float,
    max_backoff_iterations: int,
) -> tuple[DispatchByVPP, float, ACDispatchValidation]:
    low = 0.0
    high = 1.0
    best = safe_anchor
    best_validation = ACDispatchValidation(ok=True, violation_count=0, converged=True)
    for _ in range(max(1, int(max_backoff_iterations))):
        mid = 0.5 * (low + high)
        trial = interpolate_dispatch(current, safe_anchor, mid)
        validation = _as_validation(
            _validate_dispatch(
                base_net,
                vpps,
                trial,
                t,
                voltage_limits=voltage_limits,
                line_loading_limit_percent=line_loading_limit_percent,
                trafo_loading_limit_percent=trafo_loading_limit_percent,
            )
        )
        if validation.ok:
            high = mid
            best = trial
            best_validation = validation
        else:
            low = mid
    return best, float(high), best_validation


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
) -> ACDispatchValidation:
    work_net = _dispatch_to_net_copy(base_net, vpps, dispatch, t)
    converged = run_powerflow(work_net)
    report = check_network_constraints(
        work_net,
        voltage_limits=voltage_limits,
        line_loading_limit_percent=line_loading_limit_percent,
        trafo_loading_limit_percent=trafo_loading_limit_percent,
    )
    penalties = violation_penalties(report)
    return ACDispatchValidation(
        ok=bool(converged and report.ok),
        violation_count=len(report.to_records(t)),
        converged=bool(report.converged),
        voltage_violation_cost=float(penalties.get("voltage_violation_penalty", 0.0)),
        line_overload_cost=float(penalties.get("line_overload_penalty", 0.0)),
        trafo_overload_cost=float(penalties.get("transformer_overload_penalty", 0.0)),
        powerflow_failure_cost=float(penalties.get("powerflow_penalty", 0.0)),
    )


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
    candidate_validation = _as_validation(
        _validate_dispatch(
            base_net,
            vpps,
            candidate_dispatch_by_vpp,
            t,
            voltage_limits=voltage_limits,
            line_loading_limit_percent=line_loading_limit_percent,
            trafo_loading_limit_percent=trafo_loading_limit_percent,
        )
    )
    if candidate_validation.ok:
        return ACDispatchCertificate(
            status="accepted_candidate_ac_safe",
            dispatch_by_vpp=candidate_dispatch_by_vpp,
            accepted_alpha=1.0,
            candidate_violation_count=candidate_validation.violation_count,
            repaired_violation_count=0,
            candidate_powerflow_converged=candidate_validation.converged,
            repaired_powerflow_converged=True,
            repair_gap_mw=0.0,
            candidate_voltage_violation_cost=candidate_validation.voltage_violation_cost,
            candidate_line_overload_cost=candidate_validation.line_overload_cost,
            candidate_trafo_overload_cost=candidate_validation.trafo_overload_cost,
            candidate_powerflow_failure_cost=candidate_validation.powerflow_failure_cost,
        )

    current_validation = _as_validation(
        _validate_dispatch(
            base_net,
            vpps,
            current,
            t,
            voltage_limits=voltage_limits,
            line_loading_limit_percent=line_loading_limit_percent,
            trafo_loading_limit_percent=trafo_loading_limit_percent,
        )
    )
    if not current_validation.ok:
        best_recovery: dict[str, Any] | None = None
        for anchor_name, anchor in _emergency_recovery_anchors(vpps, t):
            anchor_validation = _as_validation(
                _validate_dispatch(
                    base_net,
                    vpps,
                    anchor,
                    t,
                    voltage_limits=voltage_limits,
                    line_loading_limit_percent=line_loading_limit_percent,
                    trafo_loading_limit_percent=trafo_loading_limit_percent,
                )
            )
            if not anchor_validation.ok:
                continue
            repaired, beta, repaired_validation = _search_from_unsafe_current_to_safe_anchor(
                base_net=base_net,
                vpps=vpps,
                current=current,
                safe_anchor=anchor,
                t=t,
                voltage_limits=voltage_limits,
                line_loading_limit_percent=line_loading_limit_percent,
                trafo_loading_limit_percent=trafo_loading_limit_percent,
                max_backoff_iterations=max_backoff_iterations,
            )
            gap = dispatch_gap_mw(candidate_dispatch_by_vpp, repaired)
            if best_recovery is None or gap < float(best_recovery["gap"]):
                best_recovery = {
                    "anchor_name": anchor_name,
                    "dispatch": repaired,
                    "accepted_beta": beta,
                    "validation": repaired_validation,
                    "gap": gap,
                    "anchor_validation": anchor_validation,
                }
        if best_recovery is not None:
            repaired_validation = best_recovery["validation"]
            return ACDispatchCertificate(
                status=f"repaired_by_ac_powerflow_emergency_recovery:{best_recovery['anchor_name']}",
                dispatch_by_vpp=best_recovery["dispatch"],
                accepted_alpha=float(best_recovery["accepted_beta"]),
                candidate_violation_count=candidate_validation.violation_count,
                repaired_violation_count=int(repaired_validation.violation_count),
                candidate_powerflow_converged=candidate_validation.converged,
                repaired_powerflow_converged=bool(repaired_validation.converged),
                repair_gap_mw=float(best_recovery["gap"]),
                candidate_voltage_violation_cost=candidate_validation.voltage_violation_cost,
                candidate_line_overload_cost=candidate_validation.line_overload_cost,
                candidate_trafo_overload_cost=candidate_validation.trafo_overload_cost,
                candidate_powerflow_failure_cost=candidate_validation.powerflow_failure_cost,
                repaired_voltage_violation_cost=repaired_validation.voltage_violation_cost,
                repaired_line_overload_cost=repaired_validation.line_overload_cost,
                repaired_trafo_overload_cost=repaired_validation.trafo_overload_cost,
                repaired_powerflow_failure_cost=repaired_validation.powerflow_failure_cost,
            )
        return ACDispatchCertificate(
            status="certificate_failed_no_ac_safe_recovery",
            dispatch_by_vpp=current,
            accepted_alpha=0.0,
            candidate_violation_count=candidate_validation.violation_count,
            repaired_violation_count=current_validation.violation_count,
            candidate_powerflow_converged=candidate_validation.converged,
            repaired_powerflow_converged=current_validation.converged,
            repair_gap_mw=dispatch_gap_mw(candidate_dispatch_by_vpp, current),
            candidate_voltage_violation_cost=candidate_validation.voltage_violation_cost,
            candidate_line_overload_cost=candidate_validation.line_overload_cost,
            candidate_trafo_overload_cost=candidate_validation.trafo_overload_cost,
            candidate_powerflow_failure_cost=candidate_validation.powerflow_failure_cost,
            repaired_voltage_violation_cost=current_validation.voltage_violation_cost,
            repaired_line_overload_cost=current_validation.line_overload_cost,
            repaired_trafo_overload_cost=current_validation.trafo_overload_cost,
            repaired_powerflow_failure_cost=current_validation.powerflow_failure_cost,
        )

    low = 0.0
    high = 1.0
    best = current
    best_validation = current_validation
    for _ in range(max(1, int(max_backoff_iterations))):
        mid = 0.5 * (low + high)
        trial = interpolate_dispatch(current, candidate_dispatch_by_vpp, mid)
        validation = _as_validation(
            _validate_dispatch(
                base_net,
                vpps,
                trial,
                t,
                voltage_limits=voltage_limits,
                line_loading_limit_percent=line_loading_limit_percent,
                trafo_loading_limit_percent=trafo_loading_limit_percent,
            )
        )
        if validation.ok:
            low = mid
            best = trial
            best_validation = validation
        else:
            high = mid

    status = "repaired_by_ac_powerflow_backoff" if low > 1e-9 else "rolled_back_to_current_safe_dispatch"
    return ACDispatchCertificate(
        status=status,
        dispatch_by_vpp=best,
        accepted_alpha=float(low),
        candidate_violation_count=candidate_validation.violation_count,
        repaired_violation_count=best_validation.violation_count,
        candidate_powerflow_converged=candidate_validation.converged,
        repaired_powerflow_converged=best_validation.converged,
        repair_gap_mw=dispatch_gap_mw(candidate_dispatch_by_vpp, best),
        candidate_voltage_violation_cost=candidate_validation.voltage_violation_cost,
        candidate_line_overload_cost=candidate_validation.line_overload_cost,
        candidate_trafo_overload_cost=candidate_validation.trafo_overload_cost,
        candidate_powerflow_failure_cost=candidate_validation.powerflow_failure_cost,
        repaired_voltage_violation_cost=best_validation.voltage_violation_cost,
        repaired_line_overload_cost=best_validation.line_overload_cost,
        repaired_trafo_overload_cost=best_validation.trafo_overload_cost,
        repaired_powerflow_failure_cost=best_validation.powerflow_failure_cost,
    )
