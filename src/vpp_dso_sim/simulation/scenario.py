from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandapower as pp

from vpp_dso_sim.der.evcs import EVCSModel
from vpp_dso_sim.der.flexible_load import FlexibleLoadModel
from vpp_dso_sim.der.hvac import HVACModel
from vpp_dso_sim.der.microturbine import MicroTurbineModel
from vpp_dso_sim.der.pv import PVModel
from vpp_dso_sim.der.storage import StorageModel
from vpp_dso_sim.entities.dso import DSO
from vpp_dso_sim.entities.vpp import VPPAggregator
from vpp_dso_sim.network.builder import build_network
from vpp_dso_sim.simulation.portfolio_events import PortfolioEvent, normalize_portfolio_events
from vpp_dso_sim.simulation.profiles import (
    benchmark_profile_pack,
    default_load_profile,
    default_price_profile,
    default_pv_profile,
    load_profile_csv,
)
from vpp_dso_sim.utils.config import load_yaml


@dataclass
class SimulationScenario:
    net: pp.pandapowerNet
    dso: DSO
    vpps: list[VPPAggregator]
    load_profile: list[float]
    pv_profile: list[float]
    price_profile: list[float]
    horizon_steps: int
    dt_hours: float
    seed: int
    config: dict[str, Any]
    portfolio_events: list[PortfolioEvent]


def _profile_from_config(config: dict[str, Any], key: str, default_factory, horizon_steps: int) -> list[float]:
    profiles = config.get("profiles", {})
    path = profiles.get(f"{key}_csv")
    if path:
        try:
            return load_profile_csv(path, horizon_steps)
        except FileNotFoundError:
            return default_factory(horizon_steps)
    return default_factory(horizon_steps)


def _profiles_from_config(
    config: dict[str, Any],
    *,
    horizon_steps: int,
    dt_hours: float,
    seed: int,
) -> tuple[list[float], list[float], list[float]]:
    profiles = config.get("profiles", {})
    profile_pack = profiles.get("profile_pack")
    if profile_pack:
        generated = benchmark_profile_pack(
            horizon_steps,
            dt_hours=dt_hours,
            seed=int(profiles.get("seed", seed)),
            variant=str(profiles.get("variant", profile_pack)),
        )
        return generated["load"], generated["pv"], generated["price"]
    return (
        _profile_from_config(config, "load_profile", default_load_profile, horizon_steps),
        _profile_from_config(config, "pv_profile", default_pv_profile, horizon_steps),
        _profile_from_config(config, "price_profile", default_price_profile, horizon_steps),
    )


def _apply_der_metadata(der, cfg: dict[str, Any]):
    metadata = dict(cfg.get("metadata", {}))
    for key in ("zone_id", "phase", "feeder_id", "asset_group", "portfolio_role"):
        if key in cfg:
            metadata[key] = cfg[key]
    der.metadata.update(metadata)
    return der


def _make_pv(
    vpp_id: str,
    cfg: dict[str, Any],
    pv_profile: list[float],
    *,
    p_scale: float = 1.0,
    apparent_scale: float | None = None,
) -> PVModel:
    p_max = float(cfg.get("p_max_mw", 0.2)) * float(p_scale)
    apparent_multiplier = float(apparent_scale if apparent_scale is not None else p_scale)
    apparent = float(cfg.get("apparent_power_mva", max(float(cfg.get("p_max_mw", 0.2)), 1e-6))) * apparent_multiplier
    return _apply_der_metadata(PVModel(
        id=str(cfg["id"]),
        name=str(cfg.get("name", cfg["id"])),
        bus=int(cfg["bus"]),
        owner_vpp_id=vpp_id,
        p_max_mw=p_max,
        q_min_mvar=-apparent,
        q_max_mvar=apparent,
        cost_coefficients=(0.0, 0.0, 0.0),
        forecast_profile=pv_profile,
        curtailment_rate=float(cfg.get("curtailment_rate", 1.0)),
        apparent_power_mva=apparent,
    ), cfg)


def _make_storage(vpp_id: str, cfg: dict[str, Any]) -> StorageModel:
    return _apply_der_metadata(StorageModel(
        id=str(cfg["id"]),
        name=str(cfg.get("name", cfg["id"])),
        bus=int(cfg["bus"]),
        owner_vpp_id=vpp_id,
        p_mw=0.0,
        q_min_mvar=-float(cfg.get("q_max_mvar", 0.0)),
        q_max_mvar=float(cfg.get("q_max_mvar", 0.0)),
        cost_coefficients=(0.0, 5.0, 0.0),
        capacity_mwh=float(cfg.get("capacity_mwh", 1.0)),
        soc=float(cfg.get("soc", 0.5)),
        soc_min=float(cfg.get("soc_min", 0.1)),
        soc_max=float(cfg.get("soc_max", 0.9)),
        p_charge_max_mw=float(cfg.get("p_charge_max_mw", 0.2)),
        p_discharge_max_mw=float(cfg.get("p_discharge_max_mw", 0.2)),
    ), cfg)


def _make_microturbine(vpp_id: str, cfg: dict[str, Any]) -> MicroTurbineModel:
    p_min = float(cfg.get("p_min_mw", 0.0))
    return _apply_der_metadata(MicroTurbineModel(
        id=str(cfg["id"]),
        name=str(cfg.get("name", cfg["id"])),
        bus=int(cfg["bus"]),
        owner_vpp_id=vpp_id,
        p_mw=p_min,
        p_min_mw=p_min,
        p_max_mw=float(cfg.get("p_max_mw", 0.3)),
        q_min_mvar=float(cfg.get("q_min_mvar", -0.1)),
        q_max_mvar=float(cfg.get("q_max_mvar", 0.1)),
        cost_coefficients=tuple(cfg.get("cost_coefficients", (0.2, 60.0, 0.0))),
        ramp_up_mw_per_step=float(cfg.get("ramp_up_mw_per_step", 0.1)),
        ramp_down_mw_per_step=float(cfg.get("ramp_down_mw_per_step", 0.1)),
    ), cfg)


def _make_flexible_load(vpp_id: str, cfg: dict[str, Any]) -> FlexibleLoadModel:
    baseline = float(cfg.get("baseline_p_mw", 0.1))
    return _apply_der_metadata(FlexibleLoadModel(
        id=str(cfg["id"]),
        name=str(cfg.get("name", cfg["id"])),
        bus=int(cfg["bus"]),
        owner_vpp_id=vpp_id,
        cost_coefficients=(0.0, 30.0, 0.0),
        baseline_p_mw=baseline,
        p_min_load_mw=float(cfg.get("p_min_mw", 0.5 * baseline)),
        p_max_load_mw=float(cfg.get("p_max_mw", 1.3 * baseline)),
    ), cfg)


def _make_hvac(vpp_id: str, cfg: dict[str, Any], horizon_steps: int) -> HVACModel:
    hours = [i / 4.0 for i in range(horizon_steps)]
    outdoor = [28.0 + 5.0 * max(0.0, __import__("math").sin((h - 7.0) / 24.0 * 6.283185307)) for h in hours]
    return _apply_der_metadata(HVACModel(
        id=str(cfg["id"]),
        name=str(cfg.get("name", cfg["id"])),
        bus=int(cfg["bus"]),
        owner_vpp_id=vpp_id,
        cost_coefficients=(0.0, 35.0, 0.0),
        rated_power_mw=float(cfg.get("rated_power_mw", 0.2)),
        indoor_temp=float(cfg.get("indoor_temp", 24.0)),
        outdoor_temp_profile=outdoor,
        setpoint_profile=[float(cfg.get("setpoint", 24.0))] * horizon_steps,
        temp_min=float(cfg.get("temp_min", 22.0)),
        temp_max=float(cfg.get("temp_max", 26.0)),
    ), cfg)


def _make_evcs(vpp_id: str, cfg: dict[str, Any]) -> EVCSModel:
    return _apply_der_metadata(EVCSModel(
        id=str(cfg["id"]),
        name=str(cfg.get("name", cfg["id"])),
        bus=int(cfg["bus"]),
        owner_vpp_id=vpp_id,
        cost_coefficients=(0.0, 25.0, 0.0),
        n_evs=int(cfg.get("n_evs", 10)),
        p_charge_max_mw=float(cfg.get("p_charge_max_mw", 0.15)),
    ), cfg)


def _build_vpps(config: dict[str, Any], pv_profile: list[float], horizon_steps: int) -> list[VPPAggregator]:
    vpps: list[VPPAggregator] = []
    asset_scaling = config.get("asset_scaling", {})
    pv_p_scale = float(asset_scaling.get("pv_p_max_multiplier", 1.0))
    pv_apparent_scale = float(asset_scaling.get("pv_apparent_power_multiplier", pv_p_scale))
    for vpp_cfg in config.get("vpps", []):
        vpp_id = str(vpp_cfg["id"])
        assets = vpp_cfg.get("assets", {})
        der_list = []
        for cfg in assets.get("pv", []):
            der_list.append(
                _make_pv(
                    vpp_id,
                    cfg,
                    pv_profile,
                    p_scale=pv_p_scale,
                    apparent_scale=pv_apparent_scale,
                )
            )
        for cfg in assets.get("storage", []):
            der_list.append(_make_storage(vpp_id, cfg))
        for cfg in assets.get("microturbine", []):
            der_list.append(_make_microturbine(vpp_id, cfg))
        for cfg in assets.get("flexible_load", []):
            der_list.append(_make_flexible_load(vpp_id, cfg))
        for cfg in assets.get("hvac_aggregator", []):
            der_list.append(_make_hvac(vpp_id, cfg, horizon_steps))
        for cfg in assets.get("evcs", []):
            der_list.append(_make_evcs(vpp_id, cfg))
        vpps.append(
            VPPAggregator(
                id=vpp_id,
                name=str(vpp_cfg.get("name", vpp_id)),
                pcc_bus=int(vpp_cfg["pcc_bus"]),
                der_list=der_list,
                privacy_mode=str(vpp_cfg.get("privacy_mode", "full_information")),
                metadata={
                    **dict(vpp_cfg.get("metadata", {})),
                    "portfolio_version": str(vpp_cfg.get("portfolio_version", "v0")),
                    "zone_ids": list(vpp_cfg.get("zone_ids", [])),
                },
            )
        )
    return vpps


def load_scenario(config_path: str | Path | None = None) -> SimulationScenario:
    if config_path is None:
        config_path = "configs/ieee33_multi_vpp.yaml"
    config = load_yaml(config_path)
    sim_config = config.get("simulation", {})
    horizon_steps = int(sim_config.get("horizon_steps", 288))
    dt_hours = float(sim_config.get("dt_hours", 0.25))
    seed = int(sim_config.get("seed", 42))
    load_profile, pv_profile, price_profile = _profiles_from_config(
        config,
        horizon_steps=horizon_steps,
        dt_hours=dt_hours,
        seed=seed,
    )

    net = build_network(config)
    network_config = config.get("network", {})
    reward_config = config.get("reward", {})
    dso = DSO(
        net=net,
        voltage_limits=tuple(network_config.get("voltage_limits", [0.95, 1.05])),
        line_loading_limit_percent=float(network_config.get("line_loading_limit_percent", 100.0)),
        trafo_loading_limit_percent=float(network_config.get("trafo_loading_limit_percent", 100.0)),
        market_price_profile=price_profile,
        reward_privacy_mode=str(reward_config.get("privacy_mode", "oracle_system_cost")),
        reward_component_weights={
            str(key): float(value)
            for key, value in dict(reward_config.get("component_weights", {})).items()
        },
        dso_reward_cost_scale=float(reward_config.get("dso_reward_cost_scale", 0.05)),
        security_violation_count_penalty=float(reward_config.get("security_violation_count_penalty", 0.0)),
        reward_component_scales={
            str(key): float(value)
            for key, value in dict(reward_config.get("component_scales", {})).items()
        },
        reward_component_clip=float(reward_config.get("component_clip", 10.0)),
    )
    vpps = _build_vpps(config, pv_profile, horizon_steps)
    for vpp in vpps:
        vpp.attach_assets_to_net(net)
        dso.register_vpp(vpp)

    return SimulationScenario(
        net=net,
        dso=dso,
        vpps=vpps,
        load_profile=load_profile,
        pv_profile=pv_profile,
        price_profile=price_profile,
        horizon_steps=horizon_steps,
        dt_hours=dt_hours,
        seed=seed,
        config=config,
        portfolio_events=normalize_portfolio_events(config),
    )
