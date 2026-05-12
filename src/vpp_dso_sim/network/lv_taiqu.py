from __future__ import annotations

import pandapower as pp


def build_lv_taiqu_demo_network() -> pp.pandapowerNet:
    """Build a simple 10 kV / 0.4 kV transformer-area demonstration network."""

    net = pp.create_empty_network(name="lv_taiqu_demo", sn_mva=1.0)
    hv_bus = pp.create_bus(net, vn_kv=10.0, name="10kV_source_bus")
    lv_main = pp.create_bus(net, vn_kv=0.4, name="0.4kV_main_bus")
    feeder_buses = [
        pp.create_bus(net, vn_kv=0.4, name=f"lv_feeder_{i}") for i in range(1, 7)
    ]

    pp.create_ext_grid(net, bus=hv_bus, vm_pu=1.02, name="upstream_grid")
    pp.create_transformer_from_parameters(
        net,
        hv_bus=hv_bus,
        lv_bus=lv_main,
        sn_mva=0.63,
        vn_hv_kv=10.0,
        vn_lv_kv=0.4,
        vk_percent=4.0,
        vkr_percent=0.8,
        pfe_kw=1.0,
        i0_percent=0.2,
        shift_degree=0.0,
        name="10kV_0.4kV_distribution_trafo",
    )

    for i, bus in enumerate(feeder_buses):
        pp.create_line_from_parameters(
            net,
            from_bus=lv_main,
            to_bus=bus,
            length_km=0.08 + 0.02 * i,
            r_ohm_per_km=0.642,
            x_ohm_per_km=0.083,
            c_nf_per_km=0.0,
            max_i_ka=0.30,
            name=f"lv_line_{i}",
        )

    base_loads = [
        (feeder_buses[0], 0.055, 0.018, "residential_load_a"),
        (feeder_buses[1], 0.060, 0.020, "residential_load_b"),
        (feeder_buses[2], 0.085, 0.030, "commercial_load"),
        (feeder_buses[3], 0.040, 0.012, "public_service_load"),
        (feeder_buses[4], 0.035, 0.010, "community_load"),
        (feeder_buses[5], 0.025, 0.008, "spare_feeder_load"),
    ]
    for bus, p_mw, q_mvar, name in base_loads:
        idx = pp.create_load(net, bus=bus, p_mw=p_mw, q_mvar=q_mvar, name=name)
        net.load.at[idx, "base_p_mw"] = p_mw
        net.load.at[idx, "base_q_mvar"] = q_mvar
        net.load.at[idx, "is_base_load"] = True

    return net

