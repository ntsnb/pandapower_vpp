from __future__ import annotations

from copy import deepcopy

import numpy as np
import pandapower as pp

from vpp_dso_sim.network.constraints import check_network_constraints
from vpp_dso_sim.network.powerflow import run_powerflow


def aggregate_flexibility_fast(vpp, t: int) -> tuple[float, float, float, float]:
    p_min = p_max = q_min = q_max = 0.0
    for der in vpp.der_list:
        der.metadata["current_t"] = t
        bounds = der.get_bounds(t)
        p_min += bounds[0]
        p_max += bounds[1]
        q_min += bounds[2]
        q_max += bounds[3]
    return (p_min, p_max, q_min, q_max)


def _write_sample_to_net(sample_net: pp.pandapowerNet, der, p_mw: float, q_mvar: float) -> None:
    if der.pp_element_index is None:
        return
    if der.pp_element_type == "sgen":
        sample_net.sgen.at[der.pp_element_index, "p_mw"] = p_mw
        sample_net.sgen.at[der.pp_element_index, "q_mvar"] = q_mvar
    elif der.pp_element_type == "load":
        sample_net.load.at[der.pp_element_index, "p_mw"] = max(0.0, -p_mw)
        sample_net.load.at[der.pp_element_index, "q_mvar"] = max(0.0, -q_mvar)
    elif der.pp_element_type == "storage":
        sample_net.storage.at[der.pp_element_index, "p_mw"] = -p_mw
        sample_net.storage.at[der.pp_element_index, "q_mvar"] = q_mvar


def aggregate_flexibility_by_sampling(vpp, t: int, net: pp.pandapowerNet, n_samples: int = 50):
    rng = np.random.default_rng(vpp.metadata.get("seed", 0) if hasattr(vpp, "metadata") else 0)
    feasible: list[tuple[float, float]] = []
    for _ in range(n_samples):
        sample_net = deepcopy(net)
        total_p = 0.0
        total_q = 0.0
        for der in vpp.der_list:
            der.metadata["current_t"] = t
            p_min, p_max, q_min, q_max = der.get_bounds(t)
            p = float(rng.uniform(p_min, p_max)) if p_max > p_min else p_min
            q = float(rng.uniform(q_min, q_max)) if q_max > q_min else q_min
            total_p += p
            total_q += q
            _write_sample_to_net(sample_net, der, p, q)
        run_powerflow(sample_net)
        if check_network_constraints(sample_net).ok:
            feasible.append((total_p, total_q))

    if not feasible:
        return {
            "p_min_mw": 0.0,
            "p_max_mw": 0.0,
            "q_min_mvar": 0.0,
            "q_max_mvar": 0.0,
            "n_feasible": 0,
        }

    data = np.asarray(feasible)
    return {
        "p_min_mw": float(data[:, 0].min()),
        "p_max_mw": float(data[:, 0].max()),
        "q_min_mvar": float(data[:, 1].min()),
        "q_max_mvar": float(data[:, 1].max()),
        "n_feasible": len(feasible),
    }

