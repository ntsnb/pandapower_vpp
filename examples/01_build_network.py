from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from vpp_dso_sim.network.ieee33 import build_ieee33_network
from vpp_dso_sim.network.powerflow import run_powerflow


def main() -> None:
    net = build_ieee33_network()
    converged = run_powerflow(net)
    print(f"converged={converged}")
    print(f"bus={len(net.bus)} line={len(net.line)} load={len(net.load)}")
    print(f"sgen={len(net.sgen)} storage={len(net.storage)} trafo={len(net.trafo)}")
    print(f"min_vm_pu={net.res_bus.vm_pu.min():.4f}")
    print(f"max_line_loading_percent={net.res_line.loading_percent.max():.2f}")


if __name__ == "__main__":
    main()

