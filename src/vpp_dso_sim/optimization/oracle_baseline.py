from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from vpp_dso_sim.optimization.ac_security_projection import certify_or_repair_dispatch
from vpp_dso_sim.optimization.baselines import price_driven_target
from vpp_dso_sim.optimization.feasibility_region import (
    compute_static_feasible_region,
    scalar_target_to_vector_targets,
)


@dataclass(frozen=True)
class ACValidatedSearchResult:
    actions: dict[str, dict[str, Any]]
    metadata: dict[str, Any]


def _vpp_bounds(vpp: Any, step: int) -> tuple[float, float, float]:
    bounds = compute_static_feasible_region(vpp, step).aggregate_bounds()
    p_min = float(bounds.p_min_mw)
    p_max = float(bounds.p_max_mw)
    return p_min, 0.5 * (p_min + p_max), p_max


def _candidate_target_vectors(
    scenario: Any,
    step: int,
    price: float,
    *,
    max_candidates: int,
) -> list[dict[str, float]]:
    vpps = list(scenario.vpps)
    current = {vpp.id: float(vpp.current_power_mw()) for vpp in vpps}
    price_rule = {vpp.id: float(price_driven_target(vpp, step, price)) for vpp in vpps}
    low_all = {vpp.id: _vpp_bounds(vpp, step)[0] for vpp in vpps}
    mid_all = {vpp.id: _vpp_bounds(vpp, step)[1] for vpp in vpps}
    high_all = {vpp.id: _vpp_bounds(vpp, step)[2] for vpp in vpps}

    candidates: list[dict[str, float]] = [current, price_rule, low_all, mid_all, high_all]
    for vpp in vpps:
        p_min, p_mid, p_max = _vpp_bounds(vpp, step)
        for value in (p_min, p_mid, p_max):
            candidate = dict(price_rule)
            candidate[vpp.id] = float(value)
            candidates.append(candidate)

    unique: list[dict[str, float]] = []
    seen: set[tuple[tuple[str, float], ...]] = set()
    for candidate in candidates:
        key = tuple(sorted((str(k), round(float(v), 9)) for k, v in candidate.items()))
        if key in seen:
            continue
        seen.add(key)
        unique.append(candidate)
        if len(unique) >= max(1, int(max_candidates)):
            break
    return unique


def _dispatch_from_targets(scenario: Any, step: int, targets: dict[str, float]) -> dict[str, dict[str, float]]:
    dispatch_by_vpp: dict[str, dict[str, float]] = {}
    for vpp in scenario.vpps:
        fr = compute_static_feasible_region(vpp, step)
        target = float(targets.get(vpp.id, vpp.current_power_mw()))
        if fr.scope == "pcc":
            dispatch = vpp.disaggregate_power_target(target, 0.0, step)
        else:
            vector_targets = scalar_target_to_vector_targets(vpp, fr, target)
            dispatch = vpp.disaggregate_power_targets_by_scope(vector_targets, step)
        dispatch_by_vpp[str(vpp.id)] = dispatch
    return dispatch_by_vpp


def _dispatch_cost(scenario: Any, dispatch_by_vpp: dict[str, dict[str, float]]) -> float:
    cost = 0.0
    for vpp in scenario.vpps:
        dispatch = dispatch_by_vpp.get(str(vpp.id), {})
        for der in vpp.der_list:
            cost += float(der.operating_cost(float(dispatch.get(str(der.id), der.p_mw))))
    return float(cost)


def build_ac_validated_search_actions(
    scenario: Any,
    step: int,
    price: float,
    *,
    max_candidates: int = 32,
) -> ACValidatedSearchResult:
    """Return a best-found AC-validated dispatch reference for one step.

    This is intentionally named a search reference, not an OPF oracle. It
    evaluates a bounded candidate set through the same pandapower AC security
    certificate used by the simulator and selects the lowest-cost AC-safe
    repaired dispatch found within that budget.
    """

    candidate_targets = _candidate_target_vectors(scenario, step, price, max_candidates=max_candidates)
    best: dict[str, Any] | None = None
    feasible_count = 0
    repaired_count = 0
    for index, targets in enumerate(candidate_targets):
        dispatch = _dispatch_from_targets(scenario, step, targets)
        certificate = certify_or_repair_dispatch(
            base_net=scenario.net,
            vpps=scenario.vpps,
            candidate_dispatch_by_vpp=dispatch,
            t=step,
            voltage_limits=scenario.dso.voltage_limits,
            line_loading_limit_percent=scenario.dso.line_loading_limit_percent,
            trafo_loading_limit_percent=scenario.dso.trafo_loading_limit_percent,
        )
        if not certificate.ac_safe:
            continue
        feasible_count += 1
        if certificate.status != "accepted_candidate_ac_safe":
            repaired_count += 1
        cost = _dispatch_cost(scenario, certificate.dispatch_by_vpp)
        if best is None or cost < float(best["cost"]):
            best = {
                "candidate_index": int(index),
                "targets": targets,
                "dispatch": certificate.dispatch_by_vpp,
                "cost": float(cost),
                "certificate": certificate,
            }

    if best is None:
        fallback_targets = {vpp.id: float(vpp.current_power_mw()) for vpp in scenario.vpps}
        fallback_dispatch = _dispatch_from_targets(scenario, step, fallback_targets)
        best = {
            "candidate_index": -1,
            "targets": fallback_targets,
            "dispatch": fallback_dispatch,
            "cost": _dispatch_cost(scenario, fallback_dispatch),
            "certificate": None,
        }

    actions: dict[str, dict[str, Any]] = {}
    for vpp in scenario.vpps:
        dispatch = best["dispatch"].get(str(vpp.id), {})
        selected = float(sum(dispatch.get(str(der.id), der.p_mw) for der in vpp.der_list))
        actions[str(vpp.id)] = {
            "selected_p_mw": selected,
            "der_dispatch_p_mw": {str(der_id): float(p_mw) for der_id, p_mw in dispatch.items()},
            "command_source": "ac_validated_search_reference",
            "action_mode": "ac_validated_best_found_explicit_der_dispatch",
        }

    certificate = best.get("certificate")
    fallback_to_current_dispatch = certificate is None
    metadata = {
        "baseline_role": "ac_validated_best_found_dispatch_reference",
        "is_ac_validated": bool(not fallback_to_current_dispatch),
        "is_search_based": True,
        "is_upper_bound_claim_allowed": False,
        "reference_scope": "bounded_candidate_search_not_exhaustive_opf"
        if not fallback_to_current_dispatch
        else "fallback_current_dispatch_no_ac_feasible_candidate",
        "search_budget": int(max_candidates),
        "candidate_count": int(len(candidate_targets)),
        "feasible_candidate_count": int(feasible_count),
        "repaired_candidate_count": int(repaired_count),
        "fallback_to_current_dispatch": bool(fallback_to_current_dispatch),
        "best_candidate_index": int(best["candidate_index"]),
        "best_feasible_cost": float(best["cost"]),
        "certificate_status": "" if certificate is None else certificate.status,
        "accepted_alpha": 0.0 if certificate is None else float(certificate.accepted_alpha),
    }
    return ACValidatedSearchResult(actions=actions, metadata=metadata)
