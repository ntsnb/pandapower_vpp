from __future__ import annotations

import math
from typing import Any

import pandapower as pp


FEEDER_SIZES = (22, 21, 20, 20, 19, 19)


def _feeder_scale(scales: dict[str, Any] | None, feeder_id: str, default: float = 1.0) -> float:
    if not scales:
        return float(default)
    return float(scales.get(feeder_id, scales.get(feeder_id.lower(), default)))


def _lv_base_load_mw(feeder_index: int, depth: int, *, scale: float = 1.0) -> tuple[float, float]:
    """Return a deterministic residential/commercial LV load pattern.

    The values are intentionally moderate so the 123-bus demo remains robust in
    a laptop smoke test while still producing visible voltage and flow changes.
    """

    daily_mix = 0.0022 + 0.00035 * ((depth + feeder_index) % 4)
    commercial_boost = 0.0009 if feeder_index in {1, 4} and depth % 5 in {0, 1} else 0.0
    ev_ready_boost = 0.0007 if feeder_index in {2, 5} and depth % 6 == 0 else 0.0
    p_mw = (daily_mix + commercial_boost + ev_ready_boost) * scale
    q_mvar = p_mw * 0.32
    return p_mw, q_mvar


def _set_bus_metadata(
    net: pp.pandapowerNet,
    bus: int,
    feeder_id: str,
    depth: int,
    zone_id: str,
    *,
    branch_id: str = "trunk",
    phase_hint: str = "ABC",
) -> None:
    net.bus.at[bus, "feeder_id"] = feeder_id
    net.bus.at[bus, "feeder_depth"] = int(depth)
    net.bus.at[bus, "zone_id"] = zone_id
    net.bus.at[bus, "branch_id"] = branch_id
    net.bus.at[bus, "phase_hint"] = phase_hint


def _benchmark_parent(
    *,
    seq: int,
    feeder_size: int,
    lv_main: int,
    bus_by_feeder_depth: dict[tuple[int, int], int],
    feeder_index: int,
) -> tuple[int, int, str, str]:
    trunk_length = max(8, int(math.ceil(feeder_size * 0.46)))
    if seq == 1:
        return lv_main, 1, "trunk", "ABC"
    if seq <= trunk_length:
        return bus_by_feeder_depth[(feeder_index, seq - 1)], seq, "trunk", "ABC"

    lateral_seq = seq - trunk_length
    root_depth = 2 + ((lateral_seq - 1) % max(1, trunk_length - 2))
    lateral_layer = 1 + ((lateral_seq - 1) // max(1, trunk_length - 2))
    parent_depth = root_depth if lateral_layer == 1 else root_depth + lateral_layer - 1
    parent_depth = min(parent_depth, seq - 1)
    phase_cycle = ("A", "B", "C")
    phase_hint = phase_cycle[(lateral_seq + feeder_index) % len(phase_cycle)]
    return (
        bus_by_feeder_depth[(feeder_index, parent_depth)],
        parent_depth + 1,
        f"lateral_{root_depth:02d}",
        phase_hint,
    )


def build_european_lv_demo_network(
    *,
    topology_variant: str = "radial_chain",
    base_load_scale: float = 1.0,
    transformer_sn_mva: float = 2.5,
    line_resistance_scale: float = 1.0,
    line_reactance_scale: float = 1.0,
    feeder_load_scale: dict[str, Any] | None = None,
    line_capacity_scale_by_feeder: dict[str, Any] | None = None,
) -> pp.pandapowerNet:
    """Build a 123-bus European-LV-style transformer-area test feeder.

    This is not a verbatim copy of an IEEE or CIGRE benchmark. It is a
    reproducible balanced equivalent with a 10 kV upstream source, one 10/0.4 kV
    transformer, six long radial LV feeders, and 121 downstream service buses.
    The topology is deliberately large enough for VPP portfolio and UI tests,
    but small enough for repeated pytest and dashboard generation.
    """

    is_benchmark = topology_variant in {"branched_benchmark", "benchmark_v2"}
    net = pp.create_empty_network(
        name="european_lv_123_branched_benchmark" if is_benchmark else "european_lv_123_multi_vpp_demo",
        sn_mva=max(0.5, float(transformer_sn_mva)),
    )
    mv_bus = pp.create_bus(net, vn_kv=10.0, name="mv_source_bus")
    lv_main = pp.create_bus(net, vn_kv=0.4, name="lv_main_busbar")
    _set_bus_metadata(net, mv_bus, "mv", 0, "substation")
    _set_bus_metadata(net, lv_main, "lv_main", 0, "substation")

    pp.create_ext_grid(net, bus=mv_bus, vm_pu=1.03, name="upstream_grid")
    pp.create_transformer_from_parameters(
        net,
        hv_bus=mv_bus,
        lv_bus=lv_main,
        sn_mva=max(0.5, float(transformer_sn_mva)),
        vn_hv_kv=10.0,
        vn_lv_kv=0.4,
        vk_percent=4.5,
        vkr_percent=0.9,
        pfe_kw=2.2,
        i0_percent=0.25,
        shift_degree=0.0,
        name="10kV_0.4kV_2.5MVA_transformer",
    )

    previous_by_feeder: dict[int, int] = {}
    bus_by_feeder_depth: dict[tuple[int, int], int] = {}
    for feeder_index, feeder_size in enumerate(FEEDER_SIZES):
        feeder_id = f"F{feeder_index + 1}"
        feeder_load_multiplier = _feeder_scale(feeder_load_scale, feeder_id)
        feeder_capacity_multiplier = max(0.05, _feeder_scale(line_capacity_scale_by_feeder, feeder_id))
        previous = lv_main
        previous_by_feeder[feeder_index] = previous
        for depth in range(1, feeder_size + 1):
            if is_benchmark:
                parent, electrical_depth, branch_id, phase_hint = _benchmark_parent(
                    seq=depth,
                    feeder_size=feeder_size,
                    lv_main=lv_main,
                    bus_by_feeder_depth=bus_by_feeder_depth,
                    feeder_index=feeder_index,
                )
            else:
                parent, electrical_depth, branch_id, phase_hint = previous, depth, "trunk", "ABC"
            zone_id = f"{feeder_id}_zone_{1 + (electrical_depth - 1) // 4 if is_benchmark else 1 + (depth - 1) // 6}"
            bus = pp.create_bus(
                net,
                vn_kv=0.4,
                name=f"{feeder_id}_bus_{depth:02d}",
            )
            _set_bus_metadata(
                net,
                bus,
                feeder_id,
                electrical_depth,
                zone_id,
                branch_id=branch_id,
                phase_hint=phase_hint,
            )
            bus_by_feeder_depth[(feeder_index, depth)] = bus

            section_is_lateral = branch_id != "trunk"
            length_km = (
                0.010 + 0.0015 * ((depth + feeder_index) % 5)
                if is_benchmark and not section_is_lateral
                else 0.0055 + 0.001 * ((depth + feeder_index) % 4)
            )
            pp.create_line_from_parameters(
                net,
                from_bus=parent,
                to_bus=bus,
                length_km=length_km,
                r_ohm_per_km=0.443 * float(line_resistance_scale) * (1.18 if section_is_lateral else 1.0),
                x_ohm_per_km=0.078 * float(line_reactance_scale) * (1.10 if section_is_lateral else 1.0),
                c_nf_per_km=0.0,
                max_i_ka=(
                    ((0.52 if electrical_depth <= 4 else 0.34) if is_benchmark else (0.55 if depth <= 4 else 0.40))
                    * feeder_capacity_multiplier
                ),
                name=f"{feeder_id}_line_{depth:02d}",
            )
            line_idx = int(net.line.index[-1])
            net.line.at[line_idx, "line_section_type"] = "lateral" if section_is_lateral else "trunk"
            net.line.at[line_idx, "feeder_id"] = feeder_id
            net.line.at[line_idx, "branch_id"] = branch_id

            p_mw, q_mvar = _lv_base_load_mw(
                feeder_index,
                electrical_depth,
                scale=float(base_load_scale) * feeder_load_multiplier,
            )
            load_idx = pp.create_load(
                net,
                bus=bus,
                p_mw=p_mw,
                q_mvar=q_mvar,
                name=f"{feeder_id}_base_load_{depth:02d}",
            )
            net.load.at[load_idx, "base_p_mw"] = p_mw
            net.load.at[load_idx, "base_q_mvar"] = q_mvar
            net.load.at[load_idx, "is_base_load"] = True
            net.load.at[load_idx, "feeder_id"] = feeder_id
            net.load.at[load_idx, "zone_id"] = zone_id
            net.load.at[load_idx, "phase_hint"] = phase_hint
            net.load.at[load_idx, "customer_class"] = (
                "commercial" if feeder_index in {1, 4} and electrical_depth % 5 in {0, 1} else
                "ev_ready" if feeder_index in {2, 5} and electrical_depth % 6 == 0 else
                "residential"
            )

            previous = bus
            previous_by_feeder[feeder_index] = previous

    # Normally-open tie switches are represented as bus-bus switches so topology
    # metadata is visible without closing meshes in the baseline power-flow run.
    tie_pairs = [((0, 12), (1, 10)), ((2, 13), (3, 11)), ((4, 10), (5, 9))]
    for idx, ((f1, d1), (f2, d2)) in enumerate(tie_pairs):
        pp.create_switch(
            net,
            bus=bus_by_feeder_depth[(f1, d1)],
            element=bus_by_feeder_depth[(f2, d2)],
            et="b",
            closed=False,
            type="LBS",
            name=f"normally_open_tie_{idx + 1}",
        )
        net.switch.at[int(net.switch.index[-1]), "switch_role"] = "normally_open_tie"

    # Store deterministic drawing hints for downstream visualizations.
    for bus_idx, row in net.bus.iterrows():
        feeder_id = str(row.get("feeder_id", ""))
        depth = int(row.get("feeder_depth", 0) or 0)
        if feeder_id.startswith("F"):
            feeder_number = int(feeder_id[1:])
            net.bus.at[bus_idx, "schematic_x"] = float(depth)
            net.bus.at[bus_idx, "schematic_y"] = float((feeder_number - 1) * 3.0 + 0.15 * math.sin(depth))
        else:
            net.bus.at[bus_idx, "schematic_x"] = 0.0
            net.bus.at[bus_idx, "schematic_y"] = -1.5 if feeder_id == "mv" else 7.5

    return net


def build_european_lv_benchmark_network(**kwargs) -> pp.pandapowerNet:
    """Build the branch-rich benchmark-v2 variant used for second experiments."""

    return build_european_lv_demo_network(
        topology_variant="benchmark_v2",
        base_load_scale=float(kwargs.get("base_load_scale", 1.12)),
        transformer_sn_mva=float(kwargs.get("transformer_sn_mva", 2.2)),
        line_resistance_scale=float(kwargs.get("line_resistance_scale", 1.08)),
        line_reactance_scale=float(kwargs.get("line_reactance_scale", 1.04)),
        feeder_load_scale=kwargs.get("feeder_load_scale"),
        line_capacity_scale_by_feeder=kwargs.get("line_capacity_scale_by_feeder"),
    )
