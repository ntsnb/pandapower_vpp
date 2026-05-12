from __future__ import annotations

import pandas as pd
import pandapower as pp


def extract_network_snapshot(net: pp.pandapowerNet) -> dict[str, float]:
    snapshot: dict[str, float] = {}
    if hasattr(net, "res_bus") and len(net.res_bus):
        snapshot["min_vm_pu"] = float(net.res_bus["vm_pu"].min())
        snapshot["max_vm_pu"] = float(net.res_bus["vm_pu"].max())
    if hasattr(net, "res_line") and len(net.res_line):
        snapshot["max_line_loading_percent"] = float(net.res_line["loading_percent"].max())
    else:
        snapshot["max_line_loading_percent"] = 0.0
    if len(net.trafo) and hasattr(net, "res_trafo") and len(net.res_trafo):
        snapshot["max_trafo_loading_percent"] = float(net.res_trafo["loading_percent"].max())
    else:
        snapshot["max_trafo_loading_percent"] = 0.0
    if hasattr(net, "res_ext_grid") and len(net.res_ext_grid):
        snapshot["pcc_p_mw"] = float(net.res_ext_grid["p_mw"].sum())
        snapshot["pcc_q_mvar"] = float(net.res_ext_grid["q_mvar"].sum())
    return snapshot


def result_frames_for_step(net: pp.pandapowerNet, step: int) -> dict[str, pd.DataFrame]:
    bus = pd.DataFrame({"step": step, "vm_pu": net.res_bus["vm_pu"]})
    line = pd.DataFrame({"step": step, "loading_percent": net.res_line["loading_percent"]})
    if len(net.trafo):
        trafo = pd.DataFrame({"step": step, "loading_percent": net.res_trafo["loading_percent"]})
    else:
        trafo = pd.DataFrame(columns=["step", "loading_percent"])
    return {"bus": bus, "line": line, "trafo": trafo}

