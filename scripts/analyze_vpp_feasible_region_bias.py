from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pandas as pd

from vpp_dso_sim.network.powerflow import scale_base_loads
from vpp_dso_sim.optimization.feasibility_region import compute_static_feasible_region
from vpp_dso_sim.simulation.scenario import load_scenario
from vpp_dso_sim.simulation.simulator import Simulator
from vpp_dso_sim.utils.io import ensure_dir, write_json


VARIANT_FACTORS: dict[str, tuple[float, float, bool]] = {
    "baseline": (1.0, 1.0, True),
    "load_scale_1p2": (1.2, 1.0, True),
    "load_scale_1p5": (1.5, 1.0, True),
    "pv_scale_0p8": (1.0, 0.8, True),
    "pv_scale_1p2": (1.0, 1.2, True),
    "no_ac_aware": (1.0, 1.0, False),
}


def _variant_factors(name: str) -> tuple[float, float, bool]:
    if name not in VARIANT_FACTORS:
        raise ValueError(f"Unsupported feasible-region bias diagnostic variant: {name}")
    return VARIANT_FACTORS[name]


def _apply_profile_variant(scenario, *, load_factor: float, pv_factor: float) -> None:
    scenario.load_profile = [float(value) * float(load_factor) for value in scenario.load_profile]
    scenario.pv_profile = [float(value) * float(pv_factor) for value in scenario.pv_profile]
    for vpp in scenario.vpps:
        for der in vpp.der_list:
            if hasattr(der, "forecast_profile"):
                der.forecast_profile = [float(value) * float(pv_factor) for value in der.forecast_profile]


def _calm_grid_state(grid_state: dict[str, Any]) -> dict[str, Any]:
    calm = dict(grid_state)
    calm.update(
        {
            "min_vm_pu": 1.0,
            "max_vm_pu": 1.0,
            "max_line_loading_percent": 0.0,
            "max_trafo_loading_percent": 0.0,
            "pre_dispatch_powerflow_converged": bool(grid_state.get("pre_dispatch_powerflow_converged", True)),
        }
    )
    return calm


def _resource_mix(vpp) -> dict[str, int]:
    counts = {
        "pv_der_count": 0,
        "storage_der_count": 0,
        "load_der_count": 0,
        "evcs_der_count": 0,
        "other_der_count": 0,
    }
    for der in vpp.der_list:
        name = der.__class__.__name__.lower()
        if "pv" in name:
            counts["pv_der_count"] += 1
        elif "storage" in name or "battery" in name:
            counts["storage_der_count"] += 1
        elif "evcs" in name or "ev" in name:
            counts["evcs_der_count"] += 1
        elif "load" in name or "hvac" in name:
            counts["load_der_count"] += 1
        else:
            counts["other_der_count"] += 1
    return counts


def _rows_for_variant(
    *,
    config_path: str | Path,
    variant: str,
    horizon_steps: int,
) -> list[dict[str, Any]]:
    load_factor, pv_factor, ac_aware = _variant_factors(variant)
    scenario = load_scenario(config_path)
    _apply_profile_variant(scenario, load_factor=load_factor, pv_factor=pv_factor)
    simulator = Simulator(scenario)
    simulator.reset()
    rows: list[dict[str, Any]] = []
    for step in range(int(horizon_steps)):
        load_scale = simulator._profile_value(scenario.load_profile, step)
        scale_base_loads(scenario.net, load_scale)
        for vpp in scenario.vpps:
            for der in vpp.der_list:
                der.metadata["current_t"] = int(step)
        pre_dispatch_converged = scenario.dso.run_powerflow()
        grid_state = scenario.dso.compute_network_state()
        grid_state["pre_dispatch_powerflow_converged"] = bool(pre_dispatch_converged)
        envelope_grid_state = grid_state if ac_aware else _calm_grid_state(grid_state)
        price = simulator._profile_value(scenario.price_profile, step)
        for vpp in scenario.vpps:
            fr = compute_static_feasible_region(vpp, step)
            bounds = fr.aggregate_bounds()
            p_min = float(bounds.p_min_mw)
            p_max = float(bounds.p_max_mw)
            midpoint = 0.5 * (p_min + p_max)
            current_p = float(vpp.current_power_mw())
            bid = vpp.day_ahead_bid(step, price_hint=price)
            envelope = simulator._build_dso_operating_envelope_for_policy(
                vpp,
                step,
                bid,
                fr,
                price,
                grid_state=envelope_grid_state,
            )
            preferred = float(envelope.get("preferred_target_p_mw", midpoint))
            rows.append(
                {
                    "step": int(step),
                    "vpp_id": str(vpp.id),
                    "variant": variant,
                    "load_factor": float(load_factor),
                    "pv_factor": float(pv_factor),
                    "ac_aware_requested": bool(ac_aware),
                    "fr_scope": str(fr.scope),
                    "physical_mode": str(vpp.physical_mode()),
                    "connection_bus_count": int(len(vpp.connection_buses())),
                    "der_count": int(len(vpp.der_list)),
                    **_resource_mix(vpp),
                    "p_min_mw": p_min,
                    "p_max_mw": p_max,
                    "midpoint_mw": float(midpoint),
                    "span_mw": float(max(0.0, p_max - p_min)),
                    "current_p_mw": current_p,
                    "all_negative": bool(p_max < 0.0),
                    "crosses_zero": bool(p_min <= 0.0 <= p_max),
                    "midpoint_negative": bool(midpoint < 0.0),
                    "preferred_target_p_mw": preferred,
                    "preferred_target_negative": bool(preferred < 0.0),
                    "injection_headroom_mw": float(max(0.0, p_max - current_p)),
                    "absorption_headroom_mw": float(max(0.0, current_p - p_min)),
                    "network_min_vm_pu": float(grid_state.get("min_vm_pu", 1.0)),
                    "network_max_vm_pu": float(grid_state.get("max_vm_pu", 1.0)),
                    "max_line_loading_percent": float(grid_state.get("max_line_loading_percent", 0.0)),
                    "max_trafo_loading_percent": float(grid_state.get("max_trafo_loading_percent", 0.0)),
                    "pre_dispatch_powerflow_converged": bool(pre_dispatch_converged),
                    "ac_aware_enabled": bool(envelope.get("ac_aware_enabled", False)),
                    "ac_aware_reason": str(envelope.get("ac_aware_reason", "")),
                    "source_policy": str(envelope.get("source_policy", "")),
                    "dso_decision_interface": str(envelope.get("dso_decision_interface", "")),
                }
            )
    return rows


def run_bias_diagnostic(
    *,
    config_path: str | Path,
    output_dir: str | Path,
    horizon_steps: int = 24,
    variants: tuple[str, ...] = tuple(VARIANT_FACTORS),
) -> dict[str, Path]:
    out = ensure_dir(output_dir)
    rows: list[dict[str, Any]] = []
    for variant in variants:
        rows.extend(
            _rows_for_variant(
                config_path=config_path,
                variant=str(variant),
                horizon_steps=int(horizon_steps),
            )
        )
    detail = pd.DataFrame(rows)
    detail_csv = out / "vpp_feasible_region_bias_detail.csv"
    summary_csv = out / "vpp_feasible_region_bias_summary.csv"
    detail.to_csv(detail_csv, index=False)
    summary = (
        detail.groupby(["variant", "vpp_id"], as_index=False)
        .agg(
            rows=("step", "count"),
            negative_midpoint_rate=("midpoint_negative", "mean"),
            preferred_target_negative_rate=("preferred_target_negative", "mean"),
            all_negative_rate=("all_negative", "mean"),
            crosses_zero_rate=("crosses_zero", "mean"),
            mean_p_min_mw=("p_min_mw", "mean"),
            mean_p_max_mw=("p_max_mw", "mean"),
            mean_midpoint_mw=("midpoint_mw", "mean"),
            mean_preferred_target_p_mw=("preferred_target_p_mw", "mean"),
            mean_injection_headroom_mw=("injection_headroom_mw", "mean"),
            mean_absorption_headroom_mw=("absorption_headroom_mw", "mean"),
            ac_aware_enabled_rate=("ac_aware_enabled", "mean"),
        )
    )
    summary.to_csv(summary_csv, index=False)
    metadata_path = out / "vpp_feasible_region_bias_metadata.json"
    write_json(
        metadata_path,
        {
            "config_path": str(config_path),
            "horizon_steps": int(horizon_steps),
            "variants": list(variants),
            "detail_csv": str(detail_csv),
            "summary_csv": str(summary_csv),
        },
    )
    return {"detail_csv": detail_csv, "summary_csv": summary_csv, "metadata_json": metadata_path}


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze whether VPP feasible regions are biased toward negative P.")
    parser.add_argument("--config", required=True, help="Scenario/config path.")
    parser.add_argument("--output-dir", required=True, help="Output directory.")
    parser.add_argument("--horizon-steps", type=int, default=24)
    parser.add_argument(
        "--variants",
        default=",".join(VARIANT_FACTORS),
        help="Comma-separated variants: baseline,load_scale_1p2,load_scale_1p5,pv_scale_0p8,pv_scale_1p2,no_ac_aware.",
    )
    args = parser.parse_args()
    result = run_bias_diagnostic(
        config_path=args.config,
        output_dir=args.output_dir,
        horizon_steps=args.horizon_steps,
        variants=tuple(item.strip() for item in str(args.variants).split(",") if item.strip()),
    )
    print(f"detail_csv={result['detail_csv']}")
    print(f"summary_csv={result['summary_csv']}")


if __name__ == "__main__":
    main()
