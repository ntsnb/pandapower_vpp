from __future__ import annotations

from typing import Any

from vpp_dso_sim.network.european_lv import build_european_lv_benchmark_network, build_european_lv_demo_network
from vpp_dso_sim.network.ieee33 import build_ieee33_network
from vpp_dso_sim.network.lv_taiqu import build_lv_taiqu_demo_network


_EUROPEAN_LV_BUILDER_KEYS = {
    "topology_variant",
    "base_load_scale",
    "transformer_sn_mva",
    "line_resistance_scale",
    "line_reactance_scale",
    "feeder_load_scale",
    "line_capacity_scale_by_feeder",
}


def build_network(config: dict[str, Any] | None = None):
    network_config = (config or {}).get("network", config or {})
    network_type = network_config.get("type", "ieee33")
    european_lv_kwargs = {
        key: value
        for key, value in network_config.items()
        if key in _EUROPEAN_LV_BUILDER_KEYS
    }
    if network_type == "ieee33":
        return build_ieee33_network()
    if network_type in {"lv_taiqu", "taiqu", "lv"}:
        return build_lv_taiqu_demo_network()
    if network_type in {"european_lv", "ieee_european_lv", "european_lv_123", "lv_123"}:
        return build_european_lv_demo_network(**european_lv_kwargs)
    if network_type in {"european_lv_benchmark", "european_lv_benchmark_v2", "lv_123_benchmark"}:
        return build_european_lv_benchmark_network(**european_lv_kwargs)
    raise ValueError(f"Unsupported network type: {network_type}")
