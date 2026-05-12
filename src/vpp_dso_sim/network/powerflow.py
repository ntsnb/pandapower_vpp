from __future__ import annotations

import pandapower as pp
import pandapower.auxiliary as pp_auxiliary
import pandapower.build_branch as pp_build_branch


def _disable_numba_helpers_for_server_env() -> None:
    """Avoid broken external numba dispatchers leaking into pandapower helpers."""

    pure_get_values = getattr(pp_auxiliary, "_get_values", None)
    if pure_get_values is not None:
        pp_auxiliary.get_values = pure_get_values
        pp_build_branch.get_values = pure_get_values


def run_powerflow(net: pp.pandapowerNet, algorithm: str = "bfsw") -> bool:
    """Run a robust distribution-grid power flow."""

    _disable_numba_helpers_for_server_env()
    for candidate in (algorithm, "nr"):
        try:
            pp.runpp(
                net,
                algorithm=candidate,
                init="auto",
                max_iteration=100,
                calculate_voltage_angles=False,
                numba=False,
            )
            if bool(getattr(net, "converged", False)):
                return True
        except Exception as exc:
            net["last_powerflow_error"] = f"{candidate}: {exc.__class__.__name__}: {exc}"
            net["converged"] = False
        if candidate == "nr":
            break
    return bool(getattr(net, "converged", False))


def scale_base_loads(net: pp.pandapowerNet, scale: float) -> None:
    if len(net.load) == 0:
        return
    if "is_base_load" not in net.load.columns:
        return
    mask = net.load["is_base_load"].eq(True)
    if "base_p_mw" in net.load.columns:
        net.load.loc[mask, "p_mw"] = net.load.loc[mask, "base_p_mw"].astype(float) * scale
    if "base_q_mvar" in net.load.columns:
        net.load.loc[mask, "q_mvar"] = net.load.loc[mask, "base_q_mvar"].astype(float) * scale
