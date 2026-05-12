from __future__ import annotations

import pandapower as pp


IEEE33_BRANCHES = [
    (0, 1, 0.0922, 0.0470),
    (1, 2, 0.4930, 0.2511),
    (2, 3, 0.3660, 0.1864),
    (3, 4, 0.3811, 0.1941),
    (4, 5, 0.8190, 0.7070),
    (5, 6, 0.1872, 0.6188),
    (6, 7, 0.7114, 0.2351),
    (7, 8, 1.0300, 0.7400),
    (8, 9, 1.0440, 0.7400),
    (9, 10, 0.1966, 0.0650),
    (10, 11, 0.3744, 0.1238),
    (11, 12, 1.4680, 1.1550),
    (12, 13, 0.5416, 0.7129),
    (13, 14, 0.5910, 0.5260),
    (14, 15, 0.7463, 0.5450),
    (15, 16, 1.2890, 1.7210),
    (16, 17, 0.7320, 0.5740),
    (1, 18, 0.1640, 0.1565),
    (18, 19, 1.5042, 1.3554),
    (19, 20, 0.4095, 0.4784),
    (20, 21, 0.7089, 0.9373),
    (2, 22, 0.4512, 0.3083),
    (22, 23, 0.8980, 0.7091),
    (23, 24, 0.8960, 0.7011),
    (5, 25, 0.2030, 0.1034),
    (25, 26, 0.2842, 0.1447),
    (26, 27, 1.0590, 0.9337),
    (27, 28, 0.8042, 0.7006),
    (28, 29, 0.5075, 0.2585),
    (29, 30, 0.9744, 0.9630),
    (30, 31, 0.3105, 0.3619),
    (31, 32, 0.3410, 0.5302),
]

IEEE33_LOADS = [
    (1, 0.100, 0.060),
    (2, 0.090, 0.040),
    (3, 0.120, 0.080),
    (4, 0.060, 0.030),
    (5, 0.060, 0.020),
    (6, 0.200, 0.100),
    (7, 0.200, 0.100),
    (8, 0.060, 0.020),
    (9, 0.060, 0.020),
    (10, 0.045, 0.030),
    (11, 0.060, 0.035),
    (12, 0.060, 0.035),
    (13, 0.120, 0.080),
    (14, 0.060, 0.010),
    (15, 0.060, 0.020),
    (16, 0.060, 0.020),
    (17, 0.090, 0.040),
    (18, 0.090, 0.040),
    (19, 0.090, 0.040),
    (20, 0.090, 0.040),
    (21, 0.090, 0.040),
    (22, 0.090, 0.050),
    (23, 0.420, 0.200),
    (24, 0.420, 0.200),
    (25, 0.060, 0.025),
    (26, 0.060, 0.025),
    (27, 0.060, 0.020),
    (28, 0.120, 0.070),
    (29, 0.200, 0.100),
    (30, 0.150, 0.070),
    (31, 0.210, 0.100),
    (32, 0.060, 0.040),
]


def build_ieee33_network(load_scale: float = 0.60) -> pp.pandapowerNet:
    """Build a balanced IEEE 33-bus style radial feeder.

    The electrical values are a compact software-test model inspired by common
    IEEE 33-bus data. The default load scale keeps the v0.1 demo comfortably
    solvable while still producing meaningful voltage drops.
    """

    net = pp.create_empty_network(name="ieee33_multi_vpp_demo", sn_mva=10.0)
    for i in range(33):
        pp.create_bus(net, vn_kv=12.66, name=f"bus_{i}")

    pp.create_ext_grid(net, bus=0, vm_pu=1.02, name="grid_connection")

    for idx, (from_bus, to_bus, r_ohm, x_ohm) in enumerate(IEEE33_BRANCHES):
        pp.create_line_from_parameters(
            net,
            from_bus=from_bus,
            to_bus=to_bus,
            length_km=1.0,
            r_ohm_per_km=r_ohm,
            x_ohm_per_km=x_ohm,
            c_nf_per_km=0.0,
            max_i_ka=0.60,
            name=f"line_{idx}_{from_bus}_{to_bus}",
        )

    for bus, p_mw, q_mvar in IEEE33_LOADS:
        p = p_mw * load_scale
        q = q_mvar * load_scale
        idx = pp.create_load(net, bus=bus, p_mw=p, q_mvar=q, name=f"base_load_bus_{bus}")
        net.load.at[idx, "base_p_mw"] = p
        net.load.at[idx, "base_q_mvar"] = q
        net.load.at[idx, "is_base_load"] = True

    return net

