from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from vpp_dso_sim.utils.io import ensure_dir


def _save(fig, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_timeseries_results(results: dict[str, pd.DataFrame], output_dir: str | Path = "outputs/figures") -> dict[str, Path]:
    out = ensure_dir(output_dir)
    paths: dict[str, Path] = {}

    bus = results.get("bus_voltage", pd.DataFrame())
    if not bus.empty:
        fig, ax = plt.subplots(figsize=(9, 4))
        cols = [c for c in bus.columns if c != "step"]
        ax.plot(bus["step"], bus[cols].min(axis=1), label="min bus voltage")
        ax.plot(bus["step"], bus[cols].max(axis=1), label="max bus voltage")
        ax.axhline(0.95, color="tab:red", linestyle="--", linewidth=1)
        ax.axhline(1.05, color="tab:red", linestyle="--", linewidth=1)
        ax.set_xlabel("step")
        ax.set_ylabel("vm_pu")
        ax.legend()
        paths["voltage_profile"] = out / "voltage_profile.png"
        _save(fig, paths["voltage_profile"])

    line = results.get("line_loading", pd.DataFrame())
    if not line.empty:
        fig, ax = plt.subplots(figsize=(9, 4))
        cols = [c for c in line.columns if c != "step"]
        ax.plot(line["step"], line[cols].max(axis=1), label="max line loading")
        ax.axhline(100.0, color="tab:red", linestyle="--", linewidth=1)
        ax.set_xlabel("step")
        ax.set_ylabel("loading_percent")
        ax.legend()
        paths["max_line_loading"] = out / "max_line_loading.png"
        _save(fig, paths["max_line_loading"])

    vpp = results.get("vpp_power", pd.DataFrame())
    if not vpp.empty:
        fig, ax = plt.subplots(figsize=(9, 4))
        pivot = vpp.pivot(index="step", columns="vpp_id", values="p_mw")
        pivot.plot(ax=ax)
        ax.set_xlabel("step")
        ax.set_ylabel("P MW, injection positive")
        paths["vpp_power_timeseries"] = out / "vpp_power_timeseries.png"
        _save(fig, paths["vpp_power_timeseries"])

    storage = results.get("storage_soc", pd.DataFrame())
    if not storage.empty:
        fig, ax = plt.subplots(figsize=(9, 4))
        pivot = storage.pivot(index="step", columns="der_id", values="soc")
        pivot.plot(ax=ax)
        ax.set_xlabel("step")
        ax.set_ylabel("SOC")
        paths["storage_soc"] = out / "storage_soc.png"
        _save(fig, paths["storage_soc"])

    reward = results.get("reward_components", pd.DataFrame())
    if not reward.empty:
        fig, ax = plt.subplots(figsize=(9, 4))
        cols = [
            c
            for c in reward.columns
            if c.endswith("_penalty") or c in {"operation_cost", "total_cost"}
        ]
        reward.set_index("step")[cols].plot(ax=ax)
        ax.set_xlabel("step")
        ax.set_ylabel("cost")
        paths["reward_components"] = out / "reward_components.png"
        _save(fig, paths["reward_components"])

    return paths

