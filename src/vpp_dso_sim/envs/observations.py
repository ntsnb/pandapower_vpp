from __future__ import annotations

from typing import Any

from vpp_dso_sim.entities.schemas import DERSpec, VPPPortfolio, schema_visibility


def build_actor_observation(
    vpp,
    t: int,
    representative_data: dict[str, Any] | None = None,
    operating_envelope: dict[str, Any] | None = None,
    service_signal: dict[str, Any] | None = None,
    dispatch_award: dict[str, Any] | None = None,
    include_private_cost: bool = False,
) -> dict[str, Any]:
    """Build a privacy-scoped VPP actor observation.

    The observation includes the VPP's own portfolio and DER state, plus DSO
    representative data explicitly addressed to this VPP. It does not include
    other VPP private state or the full network topology.
    """

    portfolio = VPPPortfolio.from_vpp(vpp, t)
    p_min, p_max, q_min, q_max = vpp.aggregate_flexibility(t)
    assets = []
    for der in vpp.der_list:
        spec = DERSpec.from_der(der, t, include_private_cost=include_private_cost).to_dict()
        if not include_private_cost:
            spec.pop("cost_coefficients", None)
            spec.pop("metadata", None)
        assets.append(spec)
    return {
        "agent_id": str(vpp.id),
        "observation_type": "actor_observation_i",
        "privacy_mode": str(vpp.privacy_mode),
        "time_index": int(t),
        "portfolio": portfolio.to_dict(),
        "aggregate_bounds": {
            "p_min_mw": float(p_min),
            "p_max_mw": float(p_max),
            "q_min_mvar": float(q_min),
            "q_max_mvar": float(q_max),
        },
        "current_power": {
            "p_mw": float(vpp.current_power_mw()),
            "q_mvar": float(vpp.current_reactive_power_mvar()),
        },
        "local_assets": assets,
        "der_mask": [bool(der.controllable) for der in vpp.der_list],
        "operating_envelope": operating_envelope or {},
        "service_signal": service_signal or {},
        "dispatch_award": dispatch_award or {},
        "representative_grid_data": representative_data or {},
    }


def build_critic_global_state(scenario, t: int) -> dict[str, Any]:
    """Build centralized-training state.

    This is intended for critic / oracle diagnostics, not decentralized VPP
    actor execution.
    """

    dso = scenario.dso
    return {
        "agent_id": "critic",
        "observation_type": "critic_global_state",
        "time_index": int(t),
        "network_state": dso.compute_network_state(),
        "vpp_reports": {vpp.id: vpp.report_to_dso(t) for vpp in scenario.vpps},
        "vpp_actor_observations": {
            vpp.id: build_actor_observation(vpp, t, include_private_cost=True)
            for vpp in scenario.vpps
        },
    }


def privacy_visibility_records() -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for schema_cls in (DERSpec, VPPPortfolio):
        for row in schema_visibility(schema_cls):
            records.append({"schema": schema_cls.__name__, **row})
    return records
