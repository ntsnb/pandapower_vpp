from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from vpp_dso_sim.der.flexible_load import FlexibleLoadModel
from vpp_dso_sim.der.pv import PVModel
from vpp_dso_sim.der.storage import StorageModel
from vpp_dso_sim.network.ieee33 import build_ieee33_network
from vpp_dso_sim.network.powerflow import run_powerflow


def main() -> None:
    net = build_ieee33_network()
    pv = PVModel(
        id="demo_pv",
        name="demo_pv",
        bus=10,
        p_max_mw=0.5,
        q_min_mvar=-0.2,
        q_max_mvar=0.2,
        apparent_power_mva=0.55,
        forecast_profile=[0.8],
    )
    ess = StorageModel(
        id="demo_ess",
        name="demo_ess",
        bus=18,
        capacity_mwh=1.0,
        soc=0.5,
        p_charge_max_mw=0.2,
        p_discharge_max_mw=0.2,
    )
    flex = FlexibleLoadModel(
        id="demo_flex",
        name="demo_flex",
        bus=20,
        baseline_p_mw=0.12,
        p_min_load_mw=0.06,
        p_max_load_mw=0.18,
    )
    for der in (pv, ess, flex):
        der.attach_to_net(net)
    pv.set_power(net, 0.4, 0.0)
    ess.set_storage_power(net, 0.10, 0.0)
    flex.set_power(net, -0.10, 0.0)
    converged = run_powerflow(net)
    print(f"converged={converged}")
    print(f"min_vm_pu={net.res_bus.vm_pu.min():.4f}")
    print(f"max_line_loading_percent={net.res_line.loading_percent.max():.2f}")


if __name__ == "__main__":
    main()

