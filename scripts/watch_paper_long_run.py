#!/usr/bin/env python3
from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
import json
import math
import os
from pathlib import Path
import re
import signal
import time
from typing import Any

import pandas as pd


LOSS_PROGRESS_HINTS = ("loss", "grad_norm", "entropy", "alpha")
PHYSICAL_DATASET_TRACE_STATE_PREFIX = "physical_dataset_v2_calendar"
UPDATE_METRIC_HINTS = (
    "loss",
    "entropy",
    "kl",
    "grad_norm",
    "surrogate",
    "expected_improvement",
    "ratio_mean",
    "param_delta_l2",
    "cg_residual",
    "initial_residual",
    "alpha",
)
UPDATE_METRIC_CONTEXT_COLUMNS = {
    "episode",
    "epoch",
    "global_step",
    "critic_update",
    "actor_update",
    "role",
    "target_vpp_id",
    "policy_update_rule",
    "policy_version",
    "worker_count",
    "num_workers",
    "shared_rollout_enabled",
}

DISPATCH_TRACE_DATASET_METRICS: dict[str, tuple[str, str, str]] = {
    "electricity_price": (
        "market_price",
        "currency/MWh",
        "Electricity price seen by the dispatch trace. In this run it comes from the configured price profile.",
    ),
    "delivered_p_mw": (
        "delivered_p_mw",
        "MW",
        "VPP actual net delivered active power in the dispatch trace; negative values mean net import/absorption.",
    ),
    "baseline_p_mw": ("baseline_p_mw", "MW", "Baseline VPP active power before the dispatch action."),
    "requested_delta_p_mw": ("requested_delta_p_mw", "MW", "DSO-requested VPP active-power adjustment."),
    "accepted_delta_p_mw": ("accepted_delta_p_mw", "MW", "VPP accepted active-power adjustment."),
    "actual_delta_p_mw": ("actual_delta_p_mw", "MW", "Actual VPP active-power adjustment after decoding and projection."),
    "actual_target_p_mw": ("actual_target_p_mw", "MW", "Actual VPP active-power target after decoding and projection."),
    "raw_target_p_mw": ("raw_target_p_mw", "MW", "Raw dispatch action converted to a VPP active-power target."),
    "decoded_target_p_mw": ("decoded_target_p_mw", "MW", "Decoded VPP active-power target before feasibility/projection stages."),
    "device_feasible_target_p_mw": ("device_feasible_target_p_mw", "MW", "Device-feasible VPP active-power target."),
    "ac_projected_target_p_mw": ("ac_projected_target_p_mw", "MW", "AC-security projected VPP active-power target."),
    "action_landing_ratio": (
        "action_landing_ratio",
        "ratio",
        "Absolute actual delta divided by absolute decoded delta plus epsilon.",
    ),
    "policy_normalized_aggregate_action": (
        "policy_normalized_aggregate_action",
        "normalized",
        "Normalized aggregate action emitted by the dispatch policy.",
    ),
    "policy_normalized_der_action_mean": (
        "policy_normalized_der_action_mean",
        "normalized",
        "Mean normalized DER-level action emitted by the dispatch policy.",
    ),
    "policy_normalized_der_action_std": (
        "policy_normalized_der_action_std",
        "normalized",
        "Standard deviation of normalized DER-level actions emitted by the dispatch policy.",
    ),
}

DERIVED_PROFILE_DATASET_METRICS: dict[str, tuple[str, str, str]] = {
    "ev_charging_load": (
        "ev_charging_load",
        "MW",
        "EV charging load derived from EVCS wholesale cost, market price, and step duration.",
    ),
    "storage_power": (
        "storage_power",
        "MW",
        "Storage net power derived from storage discharge revenue minus storage charge cost; positive means discharge.",
    ),
    "storage_soc": (
        "storage_soc",
        "%",
        "Initial or configured VPP storage SOC from scenario_config.yaml, energy-capacity weighted when multiple storage assets exist.",
    ),
    "pv_power": (
        "pv_power",
        "MW",
        "VPP PV generation proxy from configured PV capacity multiplied by the PV profile value.",
    ),
    "wind_power": (
        "wind_power",
        "MW",
        "VPP wind generation proxy. It is 0.0 when no wind asset is configured for the VPP.",
    ),
    "base_load": (
        "base_load",
        "MW",
        "VPP controllable base-load proxy from flexible-load baselines and HVAC rated powers multiplied by the load profile value.",
    ),
    "net_load": (
        "net_load",
        "MW",
        "VPP net-load proxy: base_load + ev_charging_load - pv_power - wind_power - storage_power.",
    ),
}

DISPATCH_TRACE_REWARD_METRICS: dict[str, tuple[str, str, str]] = {
    "dispatch_reward_train": (
        "dispatch_reward_train",
        "score",
        "Per-dispatch-agent reward used for training after algorithm-side processing.",
    ),
    "dispatch_reward_env": (
        "dispatch_reward_env",
        "score",
        "Per-dispatch-agent reward returned by the environment before algorithm-side processing.",
    ),
    "dispatch_private_profit_reward": (
        "dispatch_private_profit_reward",
        "score",
        "Private-profit reward component for the VPP dispatch agent.",
    ),
    "private_profit_proxy": ("private_profit_proxy", "currency", "Private-profit proxy before reward weighting."),
    "private_profit_weight": (
        "private_profit_weight",
        "dimensionless",
        "Reward weight applied to the private-profit or settlement surplus term.",
    ),
    "energy_market_revenue": (
        "energy_market_revenue",
        "currency",
        "Energy-market revenue; can be negative when the VPP is a net importer.",
    ),
    "visible_energy_minus_operation_cost": (
        "visible_energy_minus_operation_cost",
        "currency",
        "Energy-market revenue minus aggregate DER operation cost.",
    ),
    "market_energy_margin_total": (
        "market_energy_margin_total",
        "currency",
        "Export revenue plus EV user revenue minus import energy cost.",
    ),
    "economic_operational_surplus": (
        "economic_operational_surplus",
        "currency",
        "Economic operating surplus from settlement breakdown.",
    ),
    "quality_adjusted_operational_surplus": (
        "quality_adjusted_operational_surplus",
        "currency",
        "Economic operating surplus after service-quality penalties.",
    ),
    "flexibility_service_payment": ("flexibility_service_payment", "currency", "Flexibility service payment."),
    "service_payment": ("service_payment", "currency", "Accepted flexibility service payment."),
    "service_payment_weight": (
        "service_payment_weight",
        "dimensionless",
        "Reward weight applied to accepted flexibility service payment.",
    ),
    "availability_payment": ("availability_payment", "currency", "Availability payment."),
    "availability_payment_weight": (
        "availability_payment_weight",
        "dimensionless",
        "Reward weight applied to availability payment.",
    ),
    "preferred_region_bonus": ("preferred_region_bonus", "score", "Preferred operating-region reward bonus."),
    "storage_potential_raw": (
        "storage_potential_raw",
        "currency",
        "Unscaled storage future-value potential before reward shaping weight.",
    ),
    "storage_potential_shaping_reward": (
        "storage_potential_shaping_reward",
        "score",
        "Storage potential shaping contribution after reward weighting.",
    ),
    "storage_potential_shaping_weight": (
        "storage_potential_shaping_weight",
        "dimensionless",
        "Reward weight applied to unscaled storage potential.",
    ),
    "export_revenue_total": ("export_revenue_total", "currency", "Total DER export revenue."),
    "pv_export_revenue_total": ("pv_export_revenue_total", "currency", "PV export revenue."),
    "mt_export_revenue_total": ("mt_export_revenue_total", "currency", "Microturbine export revenue."),
    "storage_discharge_revenue_total": (
        "storage_discharge_revenue_total",
        "currency",
        "Storage discharge revenue.",
    ),
    "evcs_user_revenue_total": ("evcs_user_revenue_total", "currency", "EV charging user revenue."),
}

DISPATCH_TRACE_COST_METRICS: dict[str, tuple[str, str, str]] = {
    "import_energy_cost_total": ("import_energy_cost_total", "currency", "Total import-energy cost."),
    "evcs_wholesale_cost_total": ("evcs_wholesale_cost_total", "currency", "EV charging wholesale energy cost."),
    "storage_charge_cost_total": ("storage_charge_cost_total", "currency", "Storage charging energy cost."),
    "hvac_energy_cost_total": ("hvac_energy_cost_total", "currency", "HVAC energy cost."),
    "flex_energy_cost_total": ("flex_energy_cost_total", "currency", "Flexible load energy cost."),
    "unclassified_import_cost_total": (
        "unclassified_import_cost_total",
        "currency",
        "Import cost not assigned to a more specific DER class.",
    ),
    "der_operation_cost": ("der_operation_cost", "currency", "Aggregate DER operation cost."),
    "der_operating_cost_total": ("der_operating_cost_total", "currency", "DER operating cost from settlement."),
    "battery_degradation_cost_total": (
        "battery_degradation_cost_total",
        "currency",
        "Battery degradation cost from settlement.",
    ),
    "comfort_cost_total": ("comfort_cost_total", "currency", "HVAC comfort penalty/cost."),
    "unserved_penalty_total": ("unserved_penalty_total", "currency", "Unserved EV/load penalty."),
    "contract_delivery_penalty": ("contract_delivery_penalty", "currency", "Contract delivery shortfall penalty."),
    "dispatch_projection_penalty": ("dispatch_projection_penalty", "score", "Dispatch action projection penalty."),
    "scaled_comfort_soc_penalty": ("scaled_comfort_soc_penalty", "score", "Scaled comfort/SOC penalty."),
    "battery_degradation_cost": ("battery_degradation_cost", "currency", "Battery degradation cost component."),
    "reward_scaled_contract_delivery_penalty": (
        "reward_scaled_contract_delivery_penalty",
        "score",
        "Contract delivery penalty after reward weight.",
    ),
    "reward_scaled_dispatch_projection_penalty": (
        "reward_scaled_dispatch_projection_penalty",
        "score",
        "Dispatch projection penalty as subtracted in the reward.",
    ),
    "reward_scaled_training_projection_penalty": (
        "reward_scaled_training_projection_penalty",
        "score",
        "Additional algorithm-side projection penalty subtracted from the training reward.",
    ),
    "reward_scaled_total_projection_penalty": (
        "reward_scaled_total_projection_penalty",
        "score",
        "Environment plus algorithm-side projection penalty affecting the training reward.",
    ),
    "reward_scaled_comfort_soc_penalty": (
        "reward_scaled_comfort_soc_penalty",
        "score",
        "Comfort/SOC penalty after reward weight.",
    ),
    "reward_scaled_battery_degradation_penalty": (
        "reward_scaled_battery_degradation_penalty",
        "score",
        "Battery degradation penalty after reward weight.",
    ),
}

DISPATCH_TRACE_FORMULA_COLUMNS = {
    "energy_market_revenue": "energy_market_revenue_formula",
    "der_operation_cost": "der_operation_cost_formula",
    "economic_operational_surplus": "economic_operational_surplus_formula",
    "quality_adjusted_operational_surplus": "quality_adjusted_surplus_formula",
    "service_quality_penalty_total": "service_quality_penalty_formula",
    "private_profit_proxy": "private_profit_proxy_formula",
    "dispatch_private_profit_reward": "dispatch_private_profit_reward_formula",
    "action_landing_ratio": "action_landing_ratio_formula",
}

DEFAULT_FORMULAS = {
    "ev_charging_load": "evcs_wholesale_cost_total / (market_price * dt_hours)",
    "storage_power": "(storage_discharge_revenue_total - storage_charge_cost_total) / (market_price * dt_hours)",
    "storage_soc": "weighted_mean(storage.soc, storage.capacity_mwh) * 100",
    "pv_power": "sum(pv.p_max_mw) * pv_profile[time_index]",
    "wind_power": "sum(wind.p_max_mw) * wind_profile[time_index]",
    "base_load": "(sum(flexible_load.baseline_p_mw) + sum(hvac.rated_power_mw)) * load_profile[time_index]",
    "net_load": "base_load + ev_charging_load - pv_power - wind_power - storage_power",
}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _process_alive(pid: int) -> bool:
    try:
        os.kill(int(pid), 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    try:
        return pd.read_csv(path, low_memory=False)
    except pd.errors.ParserError:
        return pd.read_csv(path, engine="python", on_bad_lines="skip")


def _finite_series(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame:
        return pd.Series(dtype=float)
    return pd.to_numeric(frame[column], errors="coerce").dropna()


def _has_non_finite_numeric_value(frame: pd.DataFrame) -> bool:
    for column in frame.columns:
        numeric = pd.to_numeric(frame[column], errors="coerce").dropna()
        if not numeric.empty and not numeric.map(math.isfinite).all():
            return True
    return False


def _has_missing_required_numeric(frame: pd.DataFrame, column: str) -> bool:
    if column not in frame:
        return False
    raw = frame[column]
    if raw.isna().any():
        return True
    numeric = pd.to_numeric(raw, errors="coerce")
    return bool(numeric.isna().any())


def _reward_collapse_reason(episode_metrics: pd.DataFrame, *, min_points: int, collapse_ratio: float) -> str | None:
    rewards = _finite_series(episode_metrics, "episode_reward")
    if len(rewards) < int(min_points):
        return None
    window = max(3, min(10, len(rewards) // 4))
    first = float(rewards.head(window).mean())
    last = float(rewards.tail(window).mean())
    if not math.isfinite(first) or not math.isfinite(last):
        return "non_finite_episode_reward"
    if first < 0 and last < first * float(collapse_ratio):
        return f"reward_collapse:first_mean={first:.6g}:last_mean={last:.6g}:ratio={collapse_ratio}"
    if first >= 0 and last < -abs(first) * float(collapse_ratio):
        return f"reward_crossed_negative_collapse:first_mean={first:.6g}:last_mean={last:.6g}:ratio={collapse_ratio}"
    return None


def audit_once(
    *,
    output_dir: Path,
    pid: int,
    stop_on_anomaly: bool,
    min_reward_points: int,
    collapse_ratio: float,
) -> dict[str, Any]:
    progress = _read_csv(output_dir / "experiment_progress.csv")
    episodes = _read_csv(output_dir / "training_episode_metrics.csv")
    losses = _read_csv(output_dir / "training_loss_metrics.csv")
    latest_progress = progress.tail(1).to_dict(orient="records")
    reasons: list[str] = []

    for frame_name, frame in (("episode", episodes), ("loss", losses)):
        if frame.empty:
            continue
        if _has_non_finite_numeric_value(frame):
            reasons.append(f"{frame_name}_metric_non_finite_detected")
    if _has_missing_required_numeric(episodes, "episode_reward"):
        reasons.append("episode_reward_nan_detected")

    rewards = _finite_series(episodes, "episode_reward")
    loss_values = pd.concat(
        [_finite_series(losses, column) for column in losses.columns if "loss" in str(column).lower()],
        ignore_index=True,
    ) if not losses.empty else pd.Series(dtype=float)
    if not rewards.empty and float(rewards.abs().max()) > 1e9:
        reasons.append("episode_reward_abs_gt_1e9")
    if not loss_values.empty and float(loss_values.abs().max()) > 1e9:
        reasons.append("loss_abs_gt_1e9")
    collapse = _reward_collapse_reason(episodes, min_points=min_reward_points, collapse_ratio=collapse_ratio)
    if collapse:
        reasons.append(collapse)

    status = "ok"
    alive = _process_alive(pid)
    if not alive:
        status = "process_finished"
    elif reasons:
        status = "anomaly"
        if stop_on_anomaly:
            os.kill(pid, signal.SIGTERM)
            status = "stopped_on_anomaly"

    return {
        "timestamp": _utc_now(),
        "output_dir": str(output_dir),
        "pid": int(pid),
        "process_alive": alive,
        "status": status,
        "reasons": reasons,
        "progress_rows": int(len(progress)),
        "episode_rows": int(len(episodes)),
        "loss_rows": int(len(losses)),
        "latest_progress": latest_progress[0] if latest_progress else {},
        "reward_tail_mean": None if rewards.empty else float(rewards.tail(min(10, len(rewards))).mean()),
        "loss_tail_mean": None if loss_values.empty else float(loss_values.tail(min(100, len(loss_values))).mean()),
    }


def _write_audit(path: Path, record: dict[str, Any]) -> None:
    clean = _json_clean(record)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(clean, ensure_ascii=False, default=str, allow_nan=False) + "\n")
    latest_path = path.with_name("latest_audit.json")
    latest_path.write_text(json.dumps(clean, indent=2, ensure_ascii=False, default=str, allow_nan=False), encoding="utf-8")


def _json_clean(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_clean(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_clean(item) for item in value]
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    return value


def _dashboard_variable_dictionary() -> list[dict[str, Any]]:
    return [
        {
            "name": "step_progress_pct",
            "display_name": "Baseline step progress",
            "symbol": "p_{step}",
            "unit": "ratio",
            "group": "paper_long_watchdog",
            "physical_meaning": "Latest baseline/eval rollout step progress reported by paper_training.py.",
            "source": "scripts/watch_paper_long_run.py",
            "notes": "Watchdog mirror metric; it does not alter training.",
        },
        {
            "name": "progress_rows",
            "display_name": "Progress event rows",
            "symbol": "N_{progress}",
            "unit": "count",
            "group": "paper_long_watchdog",
            "physical_meaning": "Number of rows currently present in experiment_progress.csv.",
            "source": "scripts/watch_paper_long_run.py",
        },
        {
            "name": "episode_rows",
            "display_name": "Episode metric rows",
            "symbol": "N_{episode}",
            "unit": "count",
            "group": "paper_long_watchdog",
            "physical_meaning": "Number of rows currently present in training_episode_metrics.csv.",
            "source": "scripts/watch_paper_long_run.py",
        },
        {
            "name": "loss_rows",
            "display_name": "Loss metric rows",
            "symbol": "N_{loss}",
            "unit": "count",
            "group": "paper_long_watchdog",
            "physical_meaning": "Number of rows currently present in training_loss_metrics.csv.",
            "source": "scripts/watch_paper_long_run.py",
        },
        {
            "name": "global_env_step",
            "display_name": "Latest reported environment step",
            "symbol": "t",
            "unit": "step",
            "group": "paper_long_watchdog",
            "physical_meaning": "Latest rollout step reported by paper_training.py progress events.",
            "source": "scripts/watch_paper_long_run.py",
        },
        {
            "name": "reward_so_far",
            "display_name": "Episode reward so far",
            "symbol": "R_{episode,sofar}",
            "unit": "score",
            "group": "training_progress",
            "physical_meaning": "Cumulative reward accumulated so far inside the currently running episode.",
            "source": "experiment_progress.csv",
            "notes": "This is a training progress metric, not a physical VPP variable.",
        },
        {
            "name": "electricity_price",
            "display_name": "Electricity price",
            "symbol": "pi_t",
            "unit": "currency/MWh",
            "group": "dataset",
            "physical_meaning": "Electricity price used by the dispatch trace at the current time index.",
            "source": "dispatch_private_profit_trace.market_price",
            "notes": "In the current paper run this is copied onto each VPP trace row from the configured price profile.",
        },
        {
            "name": "delivered_p_mw",
            "display_name": "Delivered active power",
            "symbol": "P^{delivered}_{vpp,t}",
            "unit": "MW",
            "group": "dataset",
            "physical_meaning": "Actual VPP net delivered active power; negative values mean net import/absorption.",
            "source": "dispatch_private_profit_trace.delivered_p_mw",
        },
        {
            "name": "actual_delta_p_mw",
            "display_name": "Actual dispatch adjustment",
            "symbol": "Delta P^{actual}_{vpp,t}",
            "unit": "MW",
            "group": "dataset",
            "physical_meaning": "Actual VPP active-power adjustment after action decoding, feasibility handling, and AC projection.",
            "source": "dispatch_private_profit_trace.actual_delta_p_mw",
        },
        {
            "name": "actual_target_p_mw",
            "display_name": "Actual dispatch target",
            "symbol": "P^{target,actual}_{vpp,t}",
            "unit": "MW",
            "group": "dataset",
            "physical_meaning": "Actual VPP active-power target after action decoding, feasibility handling, and AC projection.",
            "source": "dispatch_private_profit_trace.actual_target_p_mw",
        },
        {
            "name": "dispatch_reward_train",
            "display_name": "Dispatch training reward",
            "symbol": "r^{train}_{dispatch,vpp,t}",
            "unit": "score",
            "group": "reward",
            "physical_meaning": "Per-VPP dispatch-agent reward used by the learner for training.",
            "source": "dispatch_private_profit_trace.dispatch_reward_train",
        },
        {
            "name": "energy_market_revenue",
            "display_name": "Energy market revenue",
            "symbol": "pi_t P^{delivered}_{vpp,t} Delta t",
            "unit": "currency",
            "group": "reward",
            "physical_meaning": "Energy-market revenue for a VPP step; negative values indicate net import cost under the sign convention.",
            "formula_latex": "pi_t P^{delivered}_{vpp,t} Delta t",
            "source": "dispatch_private_profit_trace.energy_market_revenue",
        },
        {
            "name": "evcs_user_revenue_total",
            "display_name": "EV charging user revenue",
            "symbol": "Rev^{EVCS}_{user}",
            "unit": "currency",
            "group": "reward",
            "physical_meaning": "Revenue from EV charging users in the dispatch settlement trace.",
            "source": "dispatch_private_profit_trace.evcs_user_revenue_total",
        },
        {
            "name": "storage_discharge_revenue_total",
            "display_name": "Storage discharge revenue",
            "symbol": "Rev^{ESS}_{discharge}",
            "unit": "currency",
            "group": "reward",
            "physical_meaning": "Revenue attributed to storage discharge in the dispatch settlement trace.",
            "source": "dispatch_private_profit_trace.storage_discharge_revenue_total",
        },
        {
            "name": "pv_export_revenue_total",
            "display_name": "PV export revenue",
            "symbol": "Rev^{PV}_{export}",
            "unit": "currency",
            "group": "reward",
            "physical_meaning": "Revenue attributed to PV export in the dispatch settlement trace.",
            "source": "dispatch_private_profit_trace.pv_export_revenue_total",
            "notes": "This is a settlement/revenue term, not a direct PV generation MW curve.",
        },
        {
            "name": "ev_charging_load",
            "display_name": "EV charging load",
            "symbol": "P^{EVCS}_{vpp,t}",
            "unit": "MW",
            "group": "dataset",
            "physical_meaning": "EV charging power derived from EVCS settlement energy cost, market price, and time-step duration.",
            "formula_latex": "Cost^{EVCS}_{wholesale} / (pi_t Delta t)",
            "source": "dispatch_private_profit_trace + scenario profile adapter",
        },
        {
            "name": "storage_power",
            "display_name": "Storage power",
            "symbol": "P^{ESS}_{vpp,t}",
            "unit": "MW",
            "group": "dataset",
            "physical_meaning": "Storage net power; positive means discharge and negative means charging.",
            "formula_latex": "(Rev^{ESS}_{discharge} - Cost^{ESS}_{charge}) / (pi_t Delta t)",
            "source": "dispatch_private_profit_trace + scenario profile adapter",
        },
        {
            "name": "storage_soc",
            "display_name": "Storage SOC",
            "symbol": "SOC^{ESS}_{vpp}",
            "unit": "%",
            "group": "dataset",
            "physical_meaning": "Configured initial storage state of charge for the VPP, energy-capacity weighted when multiple storage assets exist.",
            "source": "scenario_config.yaml",
            "notes": "Current live trace does not yet expose dynamic SOC by time step; this value is the configured initial SOC.",
        },
        {
            "name": "pv_power",
            "display_name": "PV power",
            "symbol": "P^{PV}_{vpp,t}",
            "unit": "MW",
            "group": "dataset",
            "physical_meaning": "PV generation proxy from configured VPP PV capacity and the PV profile value at the selected time index.",
            "formula_latex": "sum(P^{PV,max}_{vpp}) x PVProfile_t",
            "source": "scenario_config.yaml + pv_profile.csv",
        },
        {
            "name": "wind_power",
            "display_name": "Wind power",
            "symbol": "P^{WT}_{vpp,t}",
            "unit": "MW",
            "group": "dataset",
            "physical_meaning": "Wind generation proxy. It is 0 when the scenario has no configured wind assets for that VPP.",
            "source": "scenario_config.yaml",
        },
        {
            "name": "base_load",
            "display_name": "Base load",
            "symbol": "P^{load}_{vpp,t}",
            "unit": "MW",
            "group": "dataset",
            "physical_meaning": "Controllable base-load proxy from flexible-load baseline and HVAC rated power scaled by the load profile.",
            "formula_latex": "(sum(P^{flex,base}) + sum(P^{HVAC,rated})) x LoadProfile_t",
            "source": "scenario_config.yaml + load_profile.csv",
        },
        {
            "name": "net_load",
            "display_name": "Net load",
            "symbol": "P^{net}_{vpp,t}",
            "unit": "MW",
            "group": "dataset",
            "physical_meaning": "Net-load proxy combining base load, EV charging, PV, wind, and storage net power.",
            "formula_latex": "P^{load} + P^{EVCS} - P^{PV} - P^{WT} - P^{ESS}",
            "source": "scenario_config.yaml + profile CSVs + dispatch trace",
        },
        {
            "name": "storage_charge_cost_total",
            "display_name": "Storage charging cost",
            "symbol": "Cost^{ESS}_{charge}",
            "unit": "currency",
            "group": "cost",
            "physical_meaning": "Energy cost attributed to storage charging in the dispatch settlement trace.",
            "source": "dispatch_private_profit_trace.storage_charge_cost_total",
        },
        {
            "name": "evcs_wholesale_cost_total",
            "display_name": "EV charging wholesale cost",
            "symbol": "Cost^{EVCS}_{wholesale}",
            "unit": "currency",
            "group": "cost",
            "physical_meaning": "Wholesale energy cost attributed to EV charging in the dispatch settlement trace.",
            "source": "dispatch_private_profit_trace.evcs_wholesale_cost_total",
        },
        {
            "name": "policy_loss",
            "display_name": "Policy loss",
            "symbol": "L_{policy}",
            "unit": "scalar",
            "group": "loss",
            "physical_meaning": "Actor/policy optimization objective value for a policy role or VPP-specific policy update.",
            "source": "*_update_metrics.csv",
        },
        {
            "name": "critic_loss",
            "display_name": "Critic loss",
            "symbol": "L_{critic}",
            "unit": "scalar",
            "group": "loss",
            "physical_meaning": "Value/Q critic optimization loss.",
            "source": "experiment_progress.csv or *_update_metrics.csv",
        },
        {
            "name": "entropy_mean",
            "display_name": "Policy entropy",
            "symbol": "H(pi)",
            "unit": "scalar",
            "group": "loss",
            "physical_meaning": "Mean action-distribution entropy for a policy update; useful for exploration diagnostics.",
            "source": "*_update_metrics.csv",
        },
        {
            "name": "approx_kl",
            "display_name": "Approximate KL",
            "symbol": "D_{KL}^{approx}",
            "unit": "scalar",
            "group": "loss",
            "physical_meaning": "Approximate KL divergence between old and updated policy distributions.",
            "source": "*_update_metrics.csv",
        },
        {
            "name": "mean_kl",
            "display_name": "Mean KL",
            "symbol": "D_{KL}",
            "unit": "scalar",
            "group": "loss",
            "physical_meaning": "Mean KL divergence used by trust-region policy updates.",
            "source": "*_update_metrics.csv",
        },
        {
            "name": "grad_norm",
            "display_name": "Gradient norm",
            "symbol": "||g||_2",
            "unit": "scalar",
            "group": "loss",
            "physical_meaning": "Gradient norm recorded for the policy or network update.",
            "source": "*_update_metrics.csv",
        },
    ]


def _maybe_float(value: Any) -> float | None:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return numeric if math.isfinite(numeric) else None


def _text_or_none(value: Any) -> str | None:
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "null"}:
        return None
    return text


def _worker_index_label(value: Any) -> str | None:
    numeric = _maybe_float(value)
    if numeric is not None and float(numeric).is_integer():
        return str(int(numeric))
    return _text_or_none(value)


def _progress_loss_terms(progress: dict[str, Any]) -> dict[str, float]:
    terms: dict[str, float] = {}
    for name, value in progress.items():
        if any(hint in str(name).lower() for hint in LOSS_PROGRESS_HINTS):
            numeric = _maybe_float(value)
            if numeric is not None:
                terms[str(name)] = numeric
    return terms


def _safe_ratio(value: Any, total: Any) -> float | None:
    numeric = _maybe_float(value)
    denominator = _maybe_float(total)
    if numeric is None or denominator in (None, 0.0):
        return None
    return numeric / denominator


def _dispatch_trace_files(output_dir: Path) -> list[Path]:
    files = sorted(
        {
            *output_dir.glob("runs/*/train/*_dispatch_private_profit_trace_episode_*.csv"),
            *output_dir.glob("*_dispatch_private_profit_trace_episode_*.csv"),
        },
        key=lambda path: str(path),
    )
    if files:
        return files
    return sorted(
        {
            *output_dir.glob("runs/*/train/*_dispatch_private_profit_trace.csv"),
            *output_dir.glob("*_dispatch_private_profit_trace.csv"),
        },
        key=lambda path: str(path),
    )


def _trace_state_key(output_dir: Path, path: Path) -> str:
    try:
        relative = path.resolve().relative_to(output_dir.resolve())
    except ValueError:
        relative = path.resolve()
    stat = path.stat()
    return f"{relative}:{stat.st_size}:{stat.st_mtime_ns}"


def _trace_path_id(output_dir: Path, path: Path) -> str:
    try:
        return str(path.resolve().relative_to(output_dir.resolve()))
    except ValueError:
        return str(path.resolve())


def _trace_row_state_prefix(output_dir: Path, path: Path, kind: str) -> str:
    return f"dispatch_trace_rows:{kind}:{_trace_path_id(output_dir, path)}:"


def _trace_processed_rows(mirrored_trace_keys: set[str], output_dir: Path, path: Path, kind: str) -> int | None:
    prefix = _trace_row_state_prefix(output_dir, path, kind)
    processed: list[int] = []
    for item in mirrored_trace_keys:
        if not item.startswith(prefix):
            continue
        try:
            processed.append(int(item.removeprefix(prefix)))
        except ValueError:
            continue
    return max(processed) if processed else None


def _replace_trace_processed_rows(
    mirrored_trace_keys: set[str],
    output_dir: Path,
    path: Path,
    kind: str,
    processed_rows: int,
) -> None:
    prefix = _trace_row_state_prefix(output_dir, path, kind)
    mirrored_trace_keys.difference_update({item for item in mirrored_trace_keys if item.startswith(prefix)})
    mirrored_trace_keys.add(f"{prefix}{int(processed_rows)}")


def _physical_dataset_trace_state_key(output_dir: Path, path: Path) -> str:
    return f"{PHYSICAL_DATASET_TRACE_STATE_PREFIX}:{_trace_state_key(output_dir, path)}"


def _read_trace_state(path: Path | None) -> set[str]:
    if path is None or not path.exists():
        return set()
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return set()
    if not isinstance(payload, list):
        return set()
    return {str(item) for item in payload}


def _write_trace_state(path: Path | None, mirrored_trace_keys: set[str]) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(sorted(mirrored_trace_keys), indent=2, ensure_ascii=False), encoding="utf-8")


def _representative_dispatch_trace(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame
    if "worker_index" not in frame:
        return frame.copy()
    worker_index = pd.to_numeric(frame["worker_index"], errors="coerce")
    if (worker_index == 0).any():
        return frame[worker_index == 0].copy()
    return frame.copy()


def _dispatch_trace_total_rows(path: Path) -> int:
    frame = _read_csv(path)
    if frame.empty or "vpp_id" not in frame:
        return 0
    return int(len(_representative_dispatch_trace(frame)))


def _source_run_id_from_trace_path(trace_path: Path) -> str:
    return trace_path.parent.parent.name if trace_path.parent.name == "train" else trace_path.stem


def _profile_values(path: Path | None) -> list[float]:
    if path is None or not path.exists():
        return []
    frame = _read_csv(path)
    if frame.empty:
        return []
    column = "value" if "value" in frame else str(frame.columns[0])
    values = pd.to_numeric(frame[column], errors="coerce").dropna().tolist()
    return [float(value) for value in values]


def _profile_value(values: list[float], step: int) -> float | None:
    if not values:
        return None
    return float(values[int(step) % len(values)])


def _resolve_profile_csv(profile_dir: Path, config: dict[str, Any], name: str) -> Path | None:
    profiles = config.get("profiles", {}) if isinstance(config.get("profiles"), dict) else {}
    raw = profiles.get(name)
    if raw:
        path = Path(str(raw)).expanduser()
        return path if path.is_absolute() else profile_dir / path
    fallback = profile_dir / name
    return fallback if fallback.exists() else None


def _profile_dir_for_source(
    output_dir: Path,
    *,
    source_name: str | None = None,
    seed: int | None = None,
    variant: str | None = None,
) -> Path | None:
    profiles_root = output_dir / "profiles"
    if not profiles_root.exists():
        return None
    source_text = str(source_name or "")
    if seed is None:
        seed_match = re.search(r"_seed_(\d+)", source_text)
        seed = int(seed_match.group(1)) if seed_match else None
    if variant is None:
        for candidate in ("train_mixed", "holdout_peak", "holdout_cloudy", "holdout_reverseflow"):
            if candidate in source_text:
                variant = candidate
                break
    candidates = [path for path in sorted(profiles_root.iterdir()) if path.is_dir()]
    if seed is not None:
        candidates = [path for path in candidates if f"seed_{int(seed)}" in path.name]
    if variant:
        variant_matches = [path for path in candidates if str(variant) in path.name]
        if variant_matches:
            candidates = variant_matches
    candidates = [path for path in candidates if (path / "scenario_config.yaml").exists()]
    return candidates[0] if candidates else None


def _profile_dir_for_trace(output_dir: Path, trace_path: Path) -> Path | None:
    return _profile_dir_for_source(output_dir, source_name=_source_run_id_from_trace_path(trace_path))


def _profile_dir_for_progress_row(output_dir: Path, row: dict[str, Any]) -> Path | None:
    seed_value = _maybe_float(row.get("seed"))
    variant = None
    for key in ("train_variant", "profile_variant", "eval_variant"):
        variant = _text_or_none(row.get(key))
        if variant:
            break
    return _profile_dir_for_source(
        output_dir,
        source_name=_text_or_none(row.get("run_id")),
        seed=int(seed_value) if seed_value is not None else None,
        variant=variant,
    )


def _load_profile_context(output_dir: Path, trace_path: Path) -> dict[str, Any] | None:
    profile_dir = _profile_dir_for_trace(output_dir, trace_path)
    return _load_profile_context_from_dir(profile_dir)


def _load_profile_context_for_progress_row(output_dir: Path, row: dict[str, Any]) -> dict[str, Any] | None:
    profile_dir = _profile_dir_for_progress_row(output_dir, row)
    return _load_profile_context_from_dir(profile_dir)


def _load_profile_context_from_dir(profile_dir: Path | None) -> dict[str, Any] | None:
    if profile_dir is None:
        return None
    try:
        import yaml
    except ModuleNotFoundError:
        return None
    config_path = profile_dir / "scenario_config.yaml"
    try:
        config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    except Exception:
        return None
    if not isinstance(config, dict):
        return None
    metadata_path = profile_dir / "profile_metadata.json"
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.exists() else {}
    except json.JSONDecodeError:
        metadata = {}
    return {
        "profile_dir": profile_dir,
        "config": config,
        "metadata": metadata if isinstance(metadata, dict) else {},
        "load_profile": _profile_values(_resolve_profile_csv(profile_dir, config, "load_profile_csv")),
        "pv_profile": _profile_values(_resolve_profile_csv(profile_dir, config, "pv_profile_csv")),
        "price_profile": _profile_values(_resolve_profile_csv(profile_dir, config, "price_profile_csv")),
        "vpps": _vpp_asset_summaries(config),
        "dt_hours": _maybe_float((config.get("simulation") or {}).get("dt_hours")) if isinstance(config.get("simulation"), dict) else None,
    }


def _parse_profile_datetime(value: Any) -> datetime | None:
    text = _text_or_none(value)
    if text is None:
        return None
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _profile_calendar_year(metadata: dict[str, Any]) -> int | None:
    for value in (metadata.get("calendar_year"), metadata.get("year")):
        numeric = _maybe_float(value)
        if numeric is not None and 1900 <= int(numeric) <= 2200:
            return int(numeric)
    profiles_root = _text_or_none(metadata.get("profiles_root")) or ""
    match = re.search(r"(?:^|/)(19\d{2}|20\d{2})(?:/|$)", profiles_root)
    if match:
        return int(match.group(1))
    source = _text_or_none(metadata.get("source")) or ""
    if "smart_ds" in source.lower():
        return 2018
    return None


def _profile_calendar_start(profile_context: dict[str, Any] | None) -> datetime | None:
    if profile_context is None:
        return None
    metadata = profile_context.get("metadata", {})
    metadata = metadata if isinstance(metadata, dict) else {}
    config = profile_context.get("config", {})
    config = config if isinstance(config, dict) else {}
    simulation = config.get("simulation", {}) if isinstance(config.get("simulation"), dict) else {}
    for key in ("start_timestamp", "start_datetime", "start_time", "base_timestamp", "calendar_start"):
        parsed = _parse_profile_datetime(metadata.get(key))
        if parsed is not None:
            return parsed
        parsed = _parse_profile_datetime(simulation.get(key))
        if parsed is not None:
            return parsed
    year = _profile_calendar_year(metadata)
    if year is None:
        return None
    return datetime(year, 1, 1, tzinfo=timezone.utc)


def _calendar_fields(row: dict[str, Any], profile_context: dict[str, Any] | None) -> dict[str, Any]:
    step = _maybe_float(row.get("step"))
    if step is None:
        return {}
    base = _profile_calendar_start(profile_context)
    if base is None:
        return {}
    dt_hours = _maybe_float(row.get("dt_hours"))
    if dt_hours is None and profile_context is not None:
        dt_hours = _maybe_float(profile_context.get("dt_hours"))
    dt_hours = 0.25 if dt_hours is None else float(dt_hours)
    steps_per_day = max(1, int(round(24.0 / max(dt_hours, 1e-9))))
    step_index = int(step)
    timestamp = base + timedelta(hours=dt_hours * step_index)
    return {
        "date": timestamp.date().isoformat(),
        "time_index": int(step_index % steps_per_day),
        "timestamp": timestamp.isoformat().replace("+00:00", "Z"),
        "calendar_source": "profile_metadata" if profile_context else "unknown",
    }


def _asset_list(assets: dict[str, Any], key: str) -> list[dict[str, Any]]:
    value = assets.get(key, [])
    if isinstance(value, list):
        return [item for item in value if isinstance(item, dict)]
    return []


def _sum_numeric(items: list[dict[str, Any]], key: str) -> float:
    total = 0.0
    for item in items:
        numeric = _maybe_float(item.get(key))
        if numeric is not None:
            total += numeric
    return total


def _weighted_storage_soc(storages: list[dict[str, Any]]) -> float | None:
    weighted = 0.0
    total_capacity = 0.0
    plain_values: list[float] = []
    for storage in storages:
        soc = _maybe_float(storage.get("soc"))
        if soc is None:
            continue
        plain_values.append(soc)
        capacity = _maybe_float(storage.get("capacity_mwh"))
        if capacity is not None and capacity > 0:
            weighted += soc * capacity
            total_capacity += capacity
    if total_capacity > 0:
        return 100.0 * weighted / total_capacity
    if plain_values:
        return 100.0 * sum(plain_values) / len(plain_values)
    return None


def _vpp_asset_summaries(config: dict[str, Any]) -> dict[str, dict[str, float | None]]:
    rows: dict[str, dict[str, float | None]] = {}
    vpps = config.get("vpps", [])
    if not isinstance(vpps, list):
        return rows
    for vpp in vpps:
        if not isinstance(vpp, dict) or not vpp.get("id"):
            continue
        assets = vpp.get("assets", {}) if isinstance(vpp.get("assets"), dict) else {}
        pv_assets = _asset_list(assets, "pv")
        wind_assets = [
            *_asset_list(assets, "wind"),
            *_asset_list(assets, "wind_turbine"),
            *_asset_list(assets, "wind_turbines"),
        ]
        storage_assets = _asset_list(assets, "storage")
        flexible_loads = _asset_list(assets, "flexible_load")
        hvac_assets = _asset_list(assets, "hvac_aggregator")
        evcs_assets = _asset_list(assets, "evcs")
        rows[str(vpp["id"])] = {
            "pv_capacity_mw": _sum_numeric(pv_assets, "p_max_mw"),
            "wind_capacity_mw": _sum_numeric(wind_assets, "p_max_mw"),
            "base_load_nominal_mw": _sum_numeric(flexible_loads, "baseline_p_mw")
            + _sum_numeric(hvac_assets, "rated_power_mw"),
            "evcs_capacity_mw": _sum_numeric(evcs_assets, "p_charge_max_mw"),
            "storage_soc_pct": _weighted_storage_soc(storage_assets),
        }
    return rows


def _power_from_money(row: dict[str, Any], column: str, *, price: float | None, dt_hours: float | None) -> float | None:
    value = _maybe_float(row.get(column))
    if value is None or price in (None, 0.0) or dt_hours in (None, 0.0):
        return None
    return value / (float(price) * float(dt_hours))


def _derived_profile_dataset_values(row: dict[str, Any], profile_context: dict[str, Any] | None) -> dict[str, float]:
    if profile_context is None:
        return {}
    step = _maybe_float(row.get("step"))
    vpp_id = _text_or_none(row.get("vpp_id"))
    if step is None or not vpp_id:
        return {}
    step_index = int(step)
    vpp_summary = profile_context.get("vpps", {}).get(vpp_id, {})
    if not isinstance(vpp_summary, dict):
        return {}
    load_factor = _profile_value(profile_context.get("load_profile", []), step_index)
    pv_factor = _profile_value(profile_context.get("pv_profile", []), step_index)
    profile_price = _profile_value(profile_context.get("price_profile", []), step_index)
    price = _maybe_float(row.get("market_price")) or profile_price
    dt_hours = _maybe_float(row.get("dt_hours")) or _maybe_float(profile_context.get("dt_hours"))

    pv_capacity = _maybe_float(vpp_summary.get("pv_capacity_mw")) or 0.0
    wind_capacity = _maybe_float(vpp_summary.get("wind_capacity_mw")) or 0.0
    base_nominal = _maybe_float(vpp_summary.get("base_load_nominal_mw")) or 0.0
    evcs_capacity = _maybe_float(vpp_summary.get("evcs_capacity_mw")) or 0.0

    pv_power = pv_capacity * pv_factor if pv_factor is not None else None
    wind_power = 0.0 if wind_capacity == 0.0 else None
    base_load = base_nominal * load_factor if load_factor is not None else None
    ev_charging_load = _power_from_money(row, "evcs_wholesale_cost_total", price=price, dt_hours=dt_hours)
    if ev_charging_load is None and load_factor is not None:
        ev_charging_load = evcs_capacity * load_factor
    storage_discharge_power = _power_from_money(
        row,
        "storage_discharge_revenue_total",
        price=price,
        dt_hours=dt_hours,
    )
    storage_charge_power = _power_from_money(row, "storage_charge_cost_total", price=price, dt_hours=dt_hours)
    storage_power = None
    if storage_discharge_power is not None or storage_charge_power is not None:
        storage_power = float(storage_discharge_power or 0.0) - float(storage_charge_power or 0.0)

    values = {
        "pv_power": pv_power,
        "wind_power": wind_power,
        "base_load": base_load,
        "ev_charging_load": ev_charging_load,
        "storage_power": storage_power,
        "storage_soc": _maybe_float(vpp_summary.get("storage_soc_pct")),
    }
    if all(values.get(name) is not None for name in ("base_load", "ev_charging_load", "pv_power", "wind_power", "storage_power")):
        values["net_load"] = (
            float(values["base_load"])
            + float(values["ev_charging_load"])
            - float(values["pv_power"])
            - float(values["wind_power"])
            - float(values["storage_power"])
        )
    return {name: float(value) for name, value in values.items() if value is not None}


def _values_from_trace_row(row: dict[str, Any], specs: dict[str, tuple[str, str, str]]) -> dict[str, float]:
    values: dict[str, float] = {}
    for metric_name, (source_column, _unit, _description) in specs.items():
        numeric = _maybe_float(row.get(source_column))
        if numeric is not None:
            values[metric_name] = numeric
    return values


def _units_from_specs(specs: dict[str, tuple[str, str, str]], values: dict[str, float]) -> dict[str, str]:
    return {metric_name: specs[metric_name][1] for metric_name in values}


def _descriptions_from_specs(specs: dict[str, tuple[str, str, str]], values: dict[str, float]) -> dict[str, str]:
    return {metric_name: specs[metric_name][2] for metric_name in values}


def _formulas_from_trace_row(row: dict[str, Any], values: dict[str, float]) -> dict[str, str]:
    formulas: dict[str, str] = {}
    for metric_name in values:
        formula_column = DISPATCH_TRACE_FORMULA_COLUMNS.get(metric_name)
        if not formula_column:
            default_formula = DEFAULT_FORMULAS.get(metric_name)
            if default_formula:
                formulas[metric_name] = default_formula
            continue
        formula = _text_or_none(row.get(formula_column))
        if formula:
            formulas[metric_name] = formula
        elif metric_name in DEFAULT_FORMULAS:
            formulas[metric_name] = DEFAULT_FORMULAS[metric_name]
    return formulas


def _trace_context(row: dict[str, Any], *, trace_path: Path, profile_context: dict[str, Any] | None = None) -> dict[str, Any]:
    step = _maybe_float(row.get("step"))
    episode = _maybe_float(row.get("episode"))
    worker_index = _worker_index_label(row.get("worker_index"))
    env_id = f"worker_{worker_index}" if worker_index is not None else "dispatch_trace"
    calendar = _calendar_fields(row, profile_context)
    worker_start_step = _maybe_float(row.get("worker_start_step"))
    if step is not None and worker_start_step is not None:
        global_env_step = int(worker_start_step + step)
    else:
        global_env_step = int(step) if step is not None else None
    context = {
        "epoch_id": 0,
        "episode_id": int(episode) + 1 if episode is not None else None,
        "time_index": calendar.get("time_index", int(step) if step is not None else None),
        "global_env_step": global_env_step,
        "env_id": env_id,
        "vpp_id": _text_or_none(row.get("vpp_id")),
        "agent_id": _text_or_none(row.get("agent_id")),
        "policy_id": _text_or_none(row.get("algorithm")) or _text_or_none(row.get("policy_version")) or "dispatch_trace",
        "timestamp": calendar.get("timestamp"),
        "date": calendar.get("date"),
        "source_run_id": trace_path.parent.parent.name if trace_path.parent.name == "train" else trace_path.stem,
        "phase": "dispatch_private_profit_trace",
        "source_file": str(trace_path),
    }
    if calendar.get("calendar_source"):
        context["calendar_source"] = calendar["calendar_source"]
    policy_version = _text_or_none(row.get("policy_version"))
    if policy_version:
        context["policy_version"] = policy_version
    return context


def _progress_profile_context(row: dict[str, Any], *, vpp_id: str, profile_context: dict[str, Any]) -> dict[str, Any]:
    step = _maybe_float(row.get("step"))
    episode = _maybe_float(row.get("episode"))
    worker_index = _worker_index_label(row.get("worker_index"))
    calendar = _calendar_fields(row, profile_context)
    worker_start_step = _maybe_float(row.get("worker_start_step"))
    if step is not None and worker_start_step is not None:
        global_env_step = int(worker_start_step + step)
    else:
        global_env_step = int(step) if step is not None else None
    context = {
        "epoch_id": 0,
        "episode_id": int(episode) if episode is not None else None,
        "time_index": calendar.get("time_index", int(step) if step is not None else None),
        "global_env_step": global_env_step,
        "env_id": f"worker_{worker_index}" if worker_index is not None else "profile_physical_fallback",
        "vpp_id": vpp_id,
        "agent_id": f"{vpp_id}_dispatch",
        "policy_id": _text_or_none(row.get("algorithm")) or "paper_long",
        "timestamp": calendar.get("timestamp") or _text_or_none(row.get("timestamp")),
        "date": calendar.get("date"),
        "source_run_id": _text_or_none(row.get("run_id")),
        "phase": "profile_physical_fallback",
        "source_file": str(profile_context.get("profile_dir", "")),
    }
    if calendar.get("calendar_source"):
        context["calendar_source"] = calendar["calendar_source"]
    policy_version = _text_or_none(row.get("policy_version"))
    if policy_version:
        context["policy_version"] = policy_version
    return context


def _profile_dataset_values_for_progress_row(
    row: dict[str, Any],
    *,
    vpp_id: str,
    profile_context: dict[str, Any],
) -> dict[str, float]:
    step = _maybe_float(row.get("step"))
    if step is None:
        return {}
    step_index = int(step)
    profile_price = _profile_value(profile_context.get("price_profile", []), step_index)
    synthetic_row = {**row, "vpp_id": vpp_id}
    if profile_price is not None:
        synthetic_row["market_price"] = profile_price
    values = _derived_profile_dataset_values(synthetic_row, profile_context)
    if profile_price is not None:
        values["electricity_price"] = float(profile_price)
    return values


def _profile_horizon_steps(row: dict[str, Any], profile_context: dict[str, Any]) -> int:
    row_horizon = _maybe_float(row.get("horizon_steps"))
    if row_horizon is not None and row_horizon > 0:
        return int(row_horizon)
    config = profile_context.get("config", {})
    simulation = config.get("simulation", {}) if isinstance(config, dict) and isinstance(config.get("simulation"), dict) else {}
    config_horizon = _maybe_float(simulation.get("horizon_steps"))
    if config_horizon is not None and config_horizon > 0:
        return int(config_horizon)
    profile_lengths = [
        len(profile_context.get("load_profile", []) or []),
        len(profile_context.get("pv_profile", []) or []),
        len(profile_context.get("price_profile", []) or []),
    ]
    return max(profile_lengths) if profile_lengths else 0


def _dense_profile_dataset_state_key(row: dict[str, Any], profile_context: dict[str, Any], horizon_steps: int) -> str:
    episode = _maybe_float(row.get("episode"))
    seed = _maybe_float(row.get("seed"))
    profile_dir = _text_or_none(profile_context.get("profile_dir")) or "unknown_profile"
    source_run_id = _text_or_none(row.get("run_id")) or "unknown_run"
    variant = _text_or_none(row.get("train_variant")) or _text_or_none(row.get("profile_variant")) or "unknown_variant"
    return (
        "profile_dense_dataset:"
        f"profile={profile_dir}:"
        f"run={source_run_id}:"
        f"variant={variant}:"
        f"seed={int(seed) if seed is not None else 'unknown'}:"
        f"episode={int(episode) if episode is not None else 'unknown'}:"
        f"horizon={int(horizon_steps)}"
    )


def _mirror_profile_physical_dataset_progress_row_to_dashboard(
    logger: Any,
    *,
    output_dir: Path,
    row: dict[str, Any],
) -> int:
    profile_context = _load_profile_context_for_progress_row(output_dir, row)
    if profile_context is None:
        return 0
    vpp_summaries = profile_context.get("vpps", {})
    if not isinstance(vpp_summaries, dict) or not vpp_summaries:
        return 0
    dataset_specs = {**DISPATCH_TRACE_DATASET_METRICS, **DERIVED_PROFILE_DATASET_METRICS}
    dataset_rows: list[dict[str, Any]] = []
    for vpp_id in sorted(str(item) for item in vpp_summaries):
        values = _profile_dataset_values_for_progress_row(row, vpp_id=vpp_id, profile_context=profile_context)
        if not values:
            continue
        dataset_rows.extend(
            _metric_rows(
                logger=logger,
                metric_group="dataset",
                values=values,
                units=_units_from_specs(dataset_specs, values),
                descriptions=_descriptions_from_specs(dataset_specs, values),
                formulas=_formulas_from_trace_row({**row, "vpp_id": vpp_id}, values),
                context=_progress_profile_context(row, vpp_id=vpp_id, profile_context=profile_context),
            )
        )
    logger._write("dataset_timeseries", dataset_rows)
    return len(dataset_rows)


def _mirror_dense_profile_physical_dataset_progress_row_to_dashboard(
    logger: Any,
    *,
    output_dir: Path,
    row: dict[str, Any],
    mirrored_dense_profile_keys: set[str],
    dense_profile_state_path: Path | None,
) -> int:
    profile_context = _load_profile_context_for_progress_row(output_dir, row)
    if profile_context is None:
        return 0
    vpp_summaries = profile_context.get("vpps", {})
    if not isinstance(vpp_summaries, dict) or not vpp_summaries:
        return 0
    horizon_steps = _profile_horizon_steps(row, profile_context)
    if horizon_steps <= 0:
        return 0
    state_key = _dense_profile_dataset_state_key(row, profile_context, horizon_steps)
    if state_key in mirrored_dense_profile_keys:
        return 0

    dataset_specs = {**DISPATCH_TRACE_DATASET_METRICS, **DERIVED_PROFILE_DATASET_METRICS}
    dataset_rows: list[dict[str, Any]] = []
    for step_index in range(horizon_steps):
        step_row = dict(row)
        step_row["step"] = int(step_index)
        if step_row.get("dt_hours") in (None, "") and profile_context.get("dt_hours") is not None:
            step_row["dt_hours"] = profile_context["dt_hours"]
        for vpp_id in sorted(str(item) for item in vpp_summaries):
            values = _profile_dataset_values_for_progress_row(step_row, vpp_id=vpp_id, profile_context=profile_context)
            if not values:
                continue
            context = _progress_profile_context(step_row, vpp_id=vpp_id, profile_context=profile_context)
            context["phase"] = "profile_physical_dense"
            dataset_rows.extend(
                _metric_rows(
                    logger=logger,
                    metric_group="dataset",
                    values=values,
                    units=_units_from_specs(dataset_specs, values),
                    descriptions=_descriptions_from_specs(dataset_specs, values),
                    formulas=_formulas_from_trace_row({**step_row, "vpp_id": vpp_id}, values),
                    context=context,
                )
            )

    if not dataset_rows:
        return 0
    logger._write("dataset_timeseries", dataset_rows)
    mirrored_dense_profile_keys.add(state_key)
    _write_trace_state(dense_profile_state_path, mirrored_dense_profile_keys)
    return len(dataset_rows)


def _mirror_progress_reward_cost_terms_per_vpp_to_dashboard(
    logger: Any,
    *,
    output_dir: Path,
    row: dict[str, Any],
) -> int:
    profile_context = _load_profile_context_for_progress_row(output_dir, row)
    if profile_context is None:
        return 0
    vpp_summaries = profile_context.get("vpps", {})
    if not isinstance(vpp_summaries, dict) or not vpp_summaries:
        return 0
    reward_terms = {
        name: numeric
        for name in ("reward_sum", "final_episode_reward", "reward_so_far")
        if (numeric := _maybe_float(row.get(name))) is not None
    }
    if reward_terms:
        total_reward = reward_terms.get("reward_sum") or reward_terms.get("final_episode_reward") or reward_terms.get("reward_so_far")
        reward_terms.setdefault("total_reward", total_reward)
    cost_terms = {
        name: numeric
        for name in ("total_cost", "total_cost_so_far")
        if (numeric := _maybe_float(row.get(name))) is not None
    }
    if cost_terms:
        cost_terms.setdefault("total_cost", cost_terms.get("total_cost_so_far"))
    if not reward_terms and not cost_terms:
        return 0

    reward_descriptions = {
        "reward_sum": "Completed rollout reward mirrored from paper_training.py progress to every VPP as a fallback until dispatch trace rows arrive.",
        "final_episode_reward": "Completed training episode reward mirrored from paper_training.py progress to every VPP as a fallback until dispatch trace rows arrive.",
        "reward_so_far": "In-progress episode reward mirrored from paper_training.py progress to every VPP as a fallback; dispatch trace rows provide true per-VPP reward terms.",
        "total_reward": "Dashboard-compatible total reward fallback mirrored to each VPP from the latest progress event.",
    }
    cost_descriptions = {
        "total_cost": "Completed or in-progress rollout cost mirrored from paper_training.py progress to every VPP as a fallback until dispatch trace rows arrive.",
        "total_cost_so_far": "In-progress episode cost mirrored from paper_training.py progress to every VPP as a fallback; dispatch trace rows provide true per-VPP cost terms.",
    }
    reward_rows: list[dict[str, Any]] = []
    cost_rows: list[dict[str, Any]] = []
    written_rows = 0
    for vpp_id in sorted(str(item) for item in vpp_summaries):
        context = _progress_profile_context(row, vpp_id=vpp_id, profile_context=profile_context)
        context["phase"] = "profile_reward_cost_fallback"
        if reward_terms:
            rows = _metric_rows(
                logger=logger,
                metric_group="reward",
                values=reward_terms,
                units={name: "score" for name in reward_terms},
                descriptions=reward_descriptions,
                formulas={},
                context=context,
            )
            total_reward = reward_terms.get("total_reward")
            for metric_row in rows:
                metric_row["term_name"] = metric_row["metric_name"]
                metric_row["sign_convention"] = "larger_is_better"
                metric_row["component_ratio"] = 1.0 if metric_row["metric_name"] == "total_reward" else _safe_ratio(metric_row["value"], total_reward)
            reward_rows.extend(rows)
            written_rows += len(reward_terms)
        if cost_terms:
            rows = _metric_rows(
                logger=logger,
                metric_group="cost",
                values=cost_terms,
                units={name: "cost" for name in cost_terms},
                descriptions=cost_descriptions,
                formulas={},
                context=context,
            )
            total_cost = cost_terms.get("total_cost")
            for metric_row in rows:
                metric_row["term_name"] = metric_row["metric_name"]
                metric_row["sign_convention"] = "smaller_is_better"
                metric_row["component_ratio"] = 1.0 if metric_row["metric_name"] == "total_cost" else _safe_ratio(metric_row["value"], total_cost)
            cost_rows.extend(rows)
            written_rows += len(cost_terms)
    logger._write("reward_terms", reward_rows)
    logger._write("cost_terms", cost_rows)
    return written_rows


def _metric_rows(
    *,
    logger: Any,
    metric_group: str,
    values: dict[str, float],
    units: dict[str, str],
    descriptions: dict[str, str],
    formulas: dict[str, str],
    context: dict[str, Any],
) -> list[dict[str, Any]]:
    logged_at = _utc_now()
    rows: list[dict[str, Any]] = []
    for metric_name, value in values.items():
        rows.append(
            {
                "run_id": logger.run_id,
                "logged_at": logged_at,
                "metric_group": metric_group,
                "metric_name": metric_name,
                "value": value,
                "unit": units.get(metric_name),
                "formula_latex": formulas.get(metric_name),
                "description": descriptions.get(metric_name),
                **context,
            }
        )
    return rows


def _mirror_dispatch_trace_file_to_dashboard(
    logger: Any,
    trace_path: Path,
    *,
    output_dir: Path,
    include_core_metrics: bool = True,
    include_physical_dataset: bool = True,
    start_row: int = 0,
) -> tuple[int, int]:
    frame = _read_csv(trace_path)
    if frame.empty or "vpp_id" not in frame:
        return 0, 0
    frame = _representative_dispatch_trace(frame)
    total_rows = int(len(frame))
    start_row = max(0, min(int(start_row), total_rows))
    frame = frame.iloc[start_row:].copy()
    if frame.empty:
        return 0, total_rows
    profile_context = _load_profile_context(output_dir, trace_path) if include_physical_dataset else None
    dataset_specs = {**DISPATCH_TRACE_DATASET_METRICS, **DERIVED_PROFILE_DATASET_METRICS}
    dataset_rows: list[dict[str, Any]] = []
    reward_rows: list[dict[str, Any]] = []
    cost_rows: list[dict[str, Any]] = []

    for raw_row in frame.to_dict(orient="records"):
        context = _trace_context(raw_row, trace_path=trace_path, profile_context=profile_context)
        if not context.get("vpp_id"):
            continue
        include_dataset_core_metrics = include_core_metrics or include_physical_dataset
        dataset_values = _values_from_trace_row(raw_row, DISPATCH_TRACE_DATASET_METRICS) if include_dataset_core_metrics else {}
        if include_physical_dataset:
            dataset_values.update(_derived_profile_dataset_values(raw_row, profile_context))
        reward_values = _values_from_trace_row(raw_row, DISPATCH_TRACE_REWARD_METRICS) if include_core_metrics else {}
        cost_values = _values_from_trace_row(raw_row, DISPATCH_TRACE_COST_METRICS) if include_core_metrics else {}
        if dataset_values:
            dataset_rows.extend(
                _metric_rows(
                    logger=logger,
                    metric_group="dataset",
                    values=dataset_values,
                    units=_units_from_specs(dataset_specs, dataset_values),
                    descriptions=_descriptions_from_specs(dataset_specs, dataset_values),
                    formulas=_formulas_from_trace_row(raw_row, dataset_values),
                    context=context,
                )
            )
        if reward_values:
            rows = _metric_rows(
                logger=logger,
                metric_group="reward",
                values=reward_values,
                units=_units_from_specs(DISPATCH_TRACE_REWARD_METRICS, reward_values),
                descriptions=_descriptions_from_specs(DISPATCH_TRACE_REWARD_METRICS, reward_values),
                formulas=_formulas_from_trace_row(raw_row, reward_values),
                context=context,
            )
            total_reward = reward_values.get("dispatch_reward_train")
            for row in rows:
                row["term_name"] = row["metric_name"]
                row["sign_convention"] = "larger_is_better"
                row["component_ratio"] = 1.0 if row["metric_name"] == "dispatch_reward_train" else _safe_ratio(row["value"], total_reward)
            reward_rows.extend(rows)
        if cost_values:
            rows = _metric_rows(
                logger=logger,
                metric_group="cost",
                values=cost_values,
                units=_units_from_specs(DISPATCH_TRACE_COST_METRICS, cost_values),
                descriptions=_descriptions_from_specs(DISPATCH_TRACE_COST_METRICS, cost_values),
                formulas=_formulas_from_trace_row(raw_row, cost_values),
                context=context,
            )
            total_cost = sum(cost_values.values())
            for row in rows:
                row["term_name"] = row["metric_name"]
                row["sign_convention"] = "smaller_is_better"
                row["component_ratio"] = _safe_ratio(row["value"], total_cost)
            cost_rows.extend(rows)

    logger._write("dataset_timeseries", dataset_rows)
    logger._write("reward_terms", reward_rows)
    logger._write("cost_terms", cost_rows)
    return int(len(frame)), total_rows


def _mirror_dispatch_traces_to_dashboard(
    logger: Any,
    *,
    output_dir: Path,
    mirrored_trace_keys: set[str],
    trace_state_path: Path | None,
) -> set[str]:
    changed = False
    for trace_path in _dispatch_trace_files(output_dir):
        key = _trace_state_key(output_dir, trace_path)
        physical_key = _physical_dataset_trace_state_key(output_dir, trace_path)
        core_processed_rows = _trace_processed_rows(mirrored_trace_keys, output_dir, trace_path, "core")
        physical_processed_rows = _trace_processed_rows(mirrored_trace_keys, output_dir, trace_path, "physical")
        include_core_metrics = key not in mirrored_trace_keys
        include_physical_dataset = physical_key not in mirrored_trace_keys
        if core_processed_rows is not None:
            include_core_metrics = True
        if physical_processed_rows is not None:
            include_physical_dataset = True
        if not include_core_metrics and not include_physical_dataset:
            total_rows = _dispatch_trace_total_rows(trace_path)
            if total_rows > 0:
                if core_processed_rows is None and key in mirrored_trace_keys:
                    _replace_trace_processed_rows(mirrored_trace_keys, output_dir, trace_path, "core", total_rows)
                    changed = True
                if physical_processed_rows is None and physical_key in mirrored_trace_keys:
                    _replace_trace_processed_rows(mirrored_trace_keys, output_dir, trace_path, "physical", total_rows)
                    changed = True
            continue
        start_candidates = []
        if include_core_metrics:
            start_candidates.append(core_processed_rows or 0)
        if include_physical_dataset:
            start_candidates.append(physical_processed_rows or 0)
        start_row = min(start_candidates) if start_candidates else 0
        try:
            mirrored_rows, total_rows = _mirror_dispatch_trace_file_to_dashboard(
                logger,
                trace_path,
                output_dir=output_dir,
                include_core_metrics=include_core_metrics,
                include_physical_dataset=include_physical_dataset,
                start_row=start_row,
            )
        except Exception as exc:  # pragma: no cover - defensive failure isolation for live training watchdogs
            print(f"dashboard dispatch trace mirror failed for {trace_path}: {exc}", flush=True)
            continue
        if total_rows > 0:
            if include_core_metrics:
                mirrored_trace_keys.add(key)
                _replace_trace_processed_rows(mirrored_trace_keys, output_dir, trace_path, "core", total_rows)
            elif core_processed_rows is None and key in mirrored_trace_keys:
                _replace_trace_processed_rows(mirrored_trace_keys, output_dir, trace_path, "core", total_rows)
            if include_physical_dataset:
                mirrored_trace_keys.add(physical_key)
                _replace_trace_processed_rows(mirrored_trace_keys, output_dir, trace_path, "physical", total_rows)
            elif physical_processed_rows is None and physical_key in mirrored_trace_keys:
                _replace_trace_processed_rows(mirrored_trace_keys, output_dir, trace_path, "physical", total_rows)
        if mirrored_rows > 0:
            changed = True
    if changed:
        _write_trace_state(trace_state_path, mirrored_trace_keys)
    return mirrored_trace_keys


def _update_metric_files(output_dir: Path) -> list[Path]:
    return sorted(
        {
            *output_dir.glob("runs/*/train/*_update_metrics.csv"),
            *output_dir.glob("*_update_metrics.csv"),
        },
        key=lambda path: str(path),
    )


def _algorithm_from_update_metrics_path(path: Path) -> str:
    name = path.name
    if name.endswith("_update_metrics.csv"):
        return name[: -len("_update_metrics.csv")]
    return name.split("_", 1)[0]


def _update_metric_state_key(output_dir: Path, path: Path) -> str:
    try:
        relative = path.resolve().relative_to(output_dir.resolve())
    except ValueError:
        relative = path.resolve()
    stat = path.stat()
    return f"{relative}:{stat.st_size}:{stat.st_mtime_ns}"


def _vpp_id_from_update_metric_row(row: dict[str, Any]) -> str:
    target_vpp_id = _text_or_none(row.get("target_vpp_id"))
    if target_vpp_id:
        return target_vpp_id
    role = _text_or_none(row.get("role")) or ""
    for suffix in ("_dispatch", "_portfolio"):
        if role.startswith("vpp_") and role.endswith(suffix):
            return role[: -len(suffix)]
    return "aggregate"


def _loss_terms_from_update_metric_row(row: dict[str, Any]) -> dict[str, float]:
    terms: dict[str, float] = {}
    for name, value in row.items():
        lower = str(name).lower()
        if lower in UPDATE_METRIC_CONTEXT_COLUMNS:
            continue
        if not any(hint in lower for hint in UPDATE_METRIC_HINTS):
            continue
        numeric = _maybe_float(value)
        if numeric is not None:
            terms[str(name)] = numeric
    return terms


def _gradient_step_from_update_metric_row(row: dict[str, Any], index: int) -> int:
    for name in ("global_step", "critic_update", "actor_update", "epoch"):
        numeric = _maybe_float(row.get(name))
        if numeric is not None:
            return int(numeric)
    return int(index) + 1


def _mirror_update_metrics_file_to_dashboard(logger: Any, path: Path, *, output_dir: Path) -> int:
    frame = _read_csv(path)
    if frame.empty:
        return 0
    rows: list[dict[str, Any]] = []
    algorithm = _algorithm_from_update_metrics_path(path)
    source_run_id = path.parent.parent.name if path.parent.name == "train" else path.stem
    logged_at = _utc_now()
    for index, raw_row in enumerate(frame.to_dict(orient="records")):
        terms = _loss_terms_from_update_metric_row(raw_row)
        if not terms:
            continue
        episode = _maybe_float(raw_row.get("episode"))
        epoch = _maybe_float(raw_row.get("epoch"))
        role = _text_or_none(raw_row.get("role")) or algorithm
        context = {
            "epoch_id": int(epoch) if epoch is not None else 0,
            "episode_id": int(episode) + 1 if episode is not None else None,
            "gradient_step": _gradient_step_from_update_metric_row(raw_row, index),
            "global_env_step": int(_maybe_float(raw_row.get("global_step")) or 0) or None,
            "env_id": "update_metrics",
            "vpp_id": _vpp_id_from_update_metric_row(raw_row),
            "agent_id": role,
            "policy_id": role,
            "timestamp": None,
            "source_run_id": source_run_id,
            "phase": "update_metrics",
            "source_file": str(path),
            "optimizer_name": _text_or_none(raw_row.get("policy_update_rule")) or algorithm,
            "network_name": role,
        }
        policy_version = _text_or_none(raw_row.get("policy_version"))
        if policy_version:
            context["policy_version"] = policy_version
        total_loss = terms.get("total_loss")
        for metric_name, value in terms.items():
            rows.append(
                {
                    "run_id": logger.run_id,
                    "logged_at": logged_at,
                    "metric_group": "loss",
                    "metric_name": metric_name,
                    "term_name": metric_name,
                    "value": value,
                    "unit": "scalar",
                    "formula_latex": None,
                    "description": "Training update metric mirrored from *_update_metrics.csv.",
                    "component_ratio": 1.0 if metric_name == "total_loss" else _safe_ratio(value, total_loss),
                    **context,
                }
            )
    logger._write("loss_terms", rows)
    return int(len(frame))


def _mirror_update_metrics_to_dashboard(
    logger: Any,
    *,
    output_dir: Path,
    mirrored_update_keys: set[str],
    update_state_path: Path | None,
) -> set[str]:
    changed = False
    for path in _update_metric_files(output_dir):
        key = _update_metric_state_key(output_dir, path)
        if key in mirrored_update_keys:
            continue
        try:
            mirrored_rows = _mirror_update_metrics_file_to_dashboard(logger, path, output_dir=output_dir)
        except Exception as exc:  # pragma: no cover - defensive failure isolation for live training watchdogs
            print(f"dashboard update metrics mirror failed for {path}: {exc}", flush=True)
            continue
        if mirrored_rows > 0:
            mirrored_update_keys.add(key)
            changed = True
    if changed:
        _write_trace_state(update_state_path, mirrored_update_keys)
    return mirrored_update_keys


def _mirror_audit_to_dashboard(logger: Any, record: dict[str, Any], *, include_progress_terms: bool = True) -> None:
    latest = record.get("latest_progress", {})
    step = _maybe_float(latest.get("step"))
    gradient_step = _maybe_float(latest.get("gradient_step", latest.get("global_step")))
    context = {
        "epoch_id": 0,
        "episode_id": _maybe_float(latest.get("episode")),
        "time_index": int(step) if step is not None else None,
        "global_env_step": int(step) if step is not None else None,
        "env_id": _text_or_none(latest.get("profile_variant")) or _text_or_none(latest.get("eval_variant")) or "paper_long",
        "vpp_id": "aggregate",
        "agent_id": "aggregate",
        "policy_id": _text_or_none(latest.get("algorithm")) or "paper_long",
        "timestamp": record.get("timestamp"),
        "source_run_id": latest.get("run_id"),
        "phase": latest.get("phase"),
    }
    if gradient_step is not None:
        context["gradient_step"] = int(gradient_step)
    logger.log_event(
        "watchdog_audit",
        {
            "message": record.get("status"),
            "reasons": record.get("reasons", []),
            "latest_phase": latest.get("phase"),
            "latest_run_id": latest.get("run_id"),
        },
        **context,
    )
    scalar_values = {
        "progress_rows": record.get("progress_rows"),
        "episode_rows": record.get("episode_rows"),
        "loss_rows": record.get("loss_rows"),
        "step_progress_pct": latest.get("step_progress_pct"),
        "reward_tail_mean": record.get("reward_tail_mean"),
        "loss_tail_mean": record.get("loss_tail_mean"),
        "violations": latest.get("violations"),
        "violations_so_far": latest.get("violations_so_far"),
        "projection_gap_mw": latest.get("projection_gap_mw"),
    }
    units = {
        "progress_rows": "count",
        "episode_rows": "count",
        "loss_rows": "count",
        "step_progress_pct": "ratio",
        "reward_tail_mean": "score",
        "loss_tail_mean": "scalar",
        "violations": "count",
        "violations_so_far": "count",
        "projection_gap_mw": "MW",
    }
    for metric_name, value in scalar_values.items():
        numeric = _maybe_float(value)
        if numeric is not None:
            logger.log_scalar(metric_name, numeric, unit=units.get(metric_name, "scalar"), **context)
    dataset_values = {
        "global_env_step": step,
        "step_progress_pct": latest.get("step_progress_pct"),
        "progress_rows": record.get("progress_rows"),
        "episode_rows": record.get("episode_rows"),
        "loss_rows": record.get("loss_rows"),
        "violations": latest.get("violations"),
        "violations_so_far": latest.get("violations_so_far"),
        "projection_gap_mw": latest.get("projection_gap_mw"),
    }
    dataset_values = {name: numeric for name, value in dataset_values.items() if (numeric := _maybe_float(value)) is not None}
    if dataset_values:
        logger.log_dataset(
            values=dataset_values,
            units={
                "global_env_step": "step",
                "step_progress_pct": "ratio",
                "progress_rows": "count",
                "episode_rows": "count",
                "loss_rows": "count",
                "violations": "count",
                "violations_so_far": "count",
                "projection_gap_mw": "MW",
            },
            descriptions={
                "global_env_step": "Latest progress step mirrored by the watchdog.",
                "step_progress_pct": "Latest progress percentage for the active baseline/eval rollout.",
                "progress_rows": "Rows currently present in experiment_progress.csv.",
                "episode_rows": "Rows currently present in training_episode_metrics.csv.",
                "loss_rows": "Rows currently present in training_loss_metrics.csv.",
                "violations": "Completed rollout violation count from paper_training.py progress.",
                "violations_so_far": "In-progress rollout violation count from paper_training.py progress.",
                "projection_gap_mw": "Latest projected action gap from paper_training.py progress.",
            },
            **context,
        )
    reward_terms = {
        name: numeric
        for name in ("reward_sum", "final_episode_reward", "reward_so_far")
        if (numeric := _maybe_float(latest.get(name))) is not None
    }
    if include_progress_terms and reward_terms:
        total_reward = reward_terms.get("reward_sum") or reward_terms.get("final_episode_reward") or reward_terms.get("reward_so_far")
        reward_terms.setdefault("total_reward", total_reward)
        logger.log_reward_terms(
            terms=reward_terms,
            units={name: "score" for name in reward_terms},
            descriptions={
                "reward_sum": "Completed rollout reward from paper_training.py progress.",
                "final_episode_reward": "Completed training episode reward from paper_training.py progress.",
                "reward_so_far": "In-progress episode reward from paper_training.py progress.",
                "total_reward": "Dashboard-compatible total reward mirror for the latest progress event.",
            },
            **context,
        )
    cost_terms = {
        name: numeric
        for name in ("total_cost", "total_cost_so_far")
        if (numeric := _maybe_float(latest.get(name))) is not None
    }
    if include_progress_terms and cost_terms:
        cost_terms.setdefault("total_cost", cost_terms.get("total_cost_so_far"))
        logger.log_cost_terms(
            terms=cost_terms,
            units={name: "cost" for name in cost_terms},
            descriptions={
                "total_cost": "Completed rollout cost from paper_training.py progress.",
                "total_cost_so_far": "In-progress episode cost from paper_training.py progress.",
            },
            **context,
        )
    loss_terms = _progress_loss_terms(latest)
    if include_progress_terms and loss_terms:
        logger.log_loss_terms(
            terms=loss_terms,
            units={name: "scalar" for name in loss_terms},
            descriptions={
                name: "Live training loss/update metric mirrored from paper_training.py progress."
                for name in loss_terms
            },
            optimizer_name=str(latest.get("optimizer_name") or "unknown"),
            network_name=str(latest.get("network_name") or latest.get("algorithm") or "paper_training"),
            **context,
        )


def _mirror_progress_rows_to_dashboard(
    logger: Any,
    *,
    output_dir: Path,
    record: dict[str, Any],
    mirrored_progress_rows: int,
    mirrored_dense_profile_keys: set[str],
    dense_profile_state_path: Path | None,
) -> int:
    progress = _read_csv(output_dir / "experiment_progress.csv")
    if progress.empty:
        _mirror_audit_to_dashboard(logger, record)
        return 0

    if mirrored_progress_rows > len(progress):
        mirrored_progress_rows = 0

    new_rows = progress.iloc[mirrored_progress_rows:].to_dict(orient="records")
    if not new_rows:
        _mirror_audit_to_dashboard(logger, record, include_progress_terms=False)
        return int(len(progress))

    for row in new_rows:
        progress_record = dict(record)
        progress_record["latest_progress"] = _json_clean(row)
        _mirror_audit_to_dashboard(logger, progress_record)
        clean_row = _json_clean(row)
        _mirror_profile_physical_dataset_progress_row_to_dashboard(logger, output_dir=output_dir, row=clean_row)
        _mirror_dense_profile_physical_dataset_progress_row_to_dashboard(
            logger,
            output_dir=output_dir,
            row=clean_row,
            mirrored_dense_profile_keys=mirrored_dense_profile_keys,
            dense_profile_state_path=dense_profile_state_path,
        )
        _mirror_progress_reward_cost_terms_per_vpp_to_dashboard(logger, output_dir=output_dir, row=clean_row)
    return int(len(progress))


def _compact_dashboard_run_partitions(
    *,
    dashboard_data_dir: Path,
    dashboard_run_id: str,
    min_part_files: int,
) -> int:
    run_tables_dir = dashboard_data_dir / dashboard_run_id / "tables"
    if not run_tables_dir.exists():
        return 0
    from marl_dashboard.backend.storage.parquet_writer import compact_partition

    created = 0
    partition_dirs = sorted(
        {
            path.parent
            for path in run_tables_dir.rglob("part-*.parquet")
            if "_compacted_parts" not in path.parts
        }
    )
    for partition_dir in partition_dirs:
        try:
            compact_path = compact_partition(
                partition_dir,
                min_part_files=max(1, int(min_part_files)),
                archive_inputs=False,
            )
        except Exception as exc:  # pragma: no cover - compaction is best-effort for live watchdogs
            print(f"dashboard partition compaction failed for {partition_dir}: {exc}", flush=True)
            continue
        if compact_path is not None:
            created += 1
    return created


def main() -> None:
    parser = argparse.ArgumentParser(description="Hourly paper-long watchdog audit.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--pid", type=int, required=True)
    parser.add_argument("--audit-log", required=True)
    parser.add_argument("--interval-seconds", type=float, default=3600.0)
    parser.add_argument("--stop-on-anomaly", action="store_true")
    parser.add_argument("--min-reward-points", type=int, default=10)
    parser.add_argument("--collapse-ratio", type=float, default=5.0)
    parser.add_argument("--once", action="store_true")
    parser.add_argument(
        "--dashboard-heartbeat-seconds",
        type=float,
        default=None,
        help="Mirror the latest progress to dashboard at this cadence while keeping formal audit cadence at --interval-seconds.",
    )
    parser.add_argument("--max-heartbeats", type=int, default=None, help="Exit after N heartbeats; mainly for smoke tests.")
    parser.add_argument("--dashboard-data-dir", default=None)
    parser.add_argument("--dashboard-run-id", default=None)
    parser.add_argument(
        "--dashboard-skip-existing-progress",
        action="store_true",
        help="When restarting a dashboard watchdog, do not replay progress rows that already exist at startup.",
    )
    parser.add_argument(
        "--dashboard-compact-every-heartbeats",
        type=int,
        default=30,
        help="Best-effort compact dashboard parquet partitions every N heartbeats; set 0 to disable.",
    )
    parser.add_argument(
        "--dashboard-compact-min-part-files",
        type=int,
        default=128,
        help="Minimum new part files in one partition before live-safe dashboard compaction is attempted.",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir).expanduser().resolve()
    audit_log = Path(args.audit_log).expanduser().resolve()
    formal_interval = max(1.0, float(args.interval_seconds))
    heartbeat_interval = max(1.0, float(args.dashboard_heartbeat_seconds or args.interval_seconds))
    if args.dashboard_heartbeat_seconds is not None:
        heartbeat_interval = max(0.1, float(args.dashboard_heartbeat_seconds))
    max_heartbeats = None if args.max_heartbeats is None else max(1, int(args.max_heartbeats))
    dashboard_compact_every = max(0, int(args.dashboard_compact_every_heartbeats))
    dashboard_compact_min_part_files = max(1, int(args.dashboard_compact_min_part_files))
    heartbeat_count = 0
    last_formal_audit_monotonic: float | None = None
    dashboard_logger = None
    mirrored_progress_rows = 0
    mirrored_trace_keys: set[str] = set()
    mirrored_update_keys: set[str] = set()
    mirrored_dense_profile_keys: set[str] = set()
    trace_state_path: Path | None = None
    update_state_path: Path | None = None
    dense_profile_state_path: Path | None = None
    if args.dashboard_data_dir:
        from marl_dashboard.logging import ExperimentLogger

        dashboard_run_id = str(args.dashboard_run_id or f"{output_dir.name}_watchdog")
        dashboard_data_dir = Path(args.dashboard_data_dir).expanduser().resolve()
        trace_state_path = dashboard_data_dir / dashboard_run_id / "mirrored_dispatch_trace_files.json"
        update_state_path = dashboard_data_dir / dashboard_run_id / "mirrored_update_metric_files.json"
        dense_profile_state_path = dashboard_data_dir / dashboard_run_id / "mirrored_dense_profile_dataset.json"
        mirrored_trace_keys = _read_trace_state(trace_state_path)
        mirrored_update_keys = _read_trace_state(update_state_path)
        mirrored_dense_profile_keys = _read_trace_state(dense_profile_state_path)
        dashboard_logger = ExperimentLogger(
            run_id=dashboard_run_id,
            data_dir=str(dashboard_data_dir),
            config={"algorithm": "paper_long_watchdog", "environment": "paper_training"},
            variable_dictionary=_dashboard_variable_dictionary(),
            formulas={},
            metadata={"source": "watch_paper_long_run.py", "output_dir": str(output_dir), "watched_pid": int(args.pid)},
            async_writer=False,
        )
        if bool(args.dashboard_skip_existing_progress):
            mirrored_progress_rows = int(len(_read_csv(output_dir / "experiment_progress.csv")))
    while True:
        now = time.monotonic()
        record = audit_once(
            output_dir=output_dir,
            pid=int(args.pid),
            stop_on_anomaly=bool(args.stop_on_anomaly),
            min_reward_points=int(args.min_reward_points),
            collapse_ratio=float(args.collapse_ratio),
        )
        heartbeat_count += 1
        if dashboard_logger is not None:
            mirrored_progress_rows = _mirror_progress_rows_to_dashboard(
                dashboard_logger,
                output_dir=output_dir,
                record=record,
                mirrored_progress_rows=mirrored_progress_rows,
                mirrored_dense_profile_keys=mirrored_dense_profile_keys,
                dense_profile_state_path=dense_profile_state_path,
            )
            mirrored_trace_keys = _mirror_dispatch_traces_to_dashboard(
                dashboard_logger,
                output_dir=output_dir,
                mirrored_trace_keys=mirrored_trace_keys,
                trace_state_path=trace_state_path,
            )
            mirrored_update_keys = _mirror_update_metrics_to_dashboard(
                dashboard_logger,
                output_dir=output_dir,
                mirrored_update_keys=mirrored_update_keys,
                update_state_path=update_state_path,
            )
            if (
                dashboard_compact_every > 0
                and heartbeat_count % dashboard_compact_every == 0
                and args.dashboard_data_dir
            ):
                compacted = _compact_dashboard_run_partitions(
                    dashboard_data_dir=dashboard_data_dir,
                    dashboard_run_id=dashboard_run_id,
                    min_part_files=dashboard_compact_min_part_files,
                )
                if compacted > 0:
                    print(
                        json.dumps(
                            {
                                "event": "dashboard_compaction",
                                "run_id": dashboard_run_id,
                                "compacted_partitions": compacted,
                            },
                            ensure_ascii=False,
                            default=str,
                            allow_nan=False,
                        ),
                        flush=True,
                    )
        terminal = record["status"] in {"process_finished", "stopped_on_anomaly"}
        formal_due = (
            bool(args.once)
            or terminal
            or last_formal_audit_monotonic is None
            or now - last_formal_audit_monotonic >= formal_interval
        )
        if formal_due:
            _write_audit(audit_log, record)
            print(json.dumps(_json_clean(record), ensure_ascii=False, default=str, allow_nan=False), flush=True)
            last_formal_audit_monotonic = now
        if args.once or terminal or (max_heartbeats is not None and heartbeat_count >= max_heartbeats):
            if dashboard_logger is not None and terminal:
                dashboard_logger.close("stopped" if record["status"] == "stopped_on_anomaly" else "finished")
            break
        time.sleep(heartbeat_interval)


if __name__ == "__main__":
    main()
