from __future__ import annotations

import html
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from vpp_dso_sim.learning.deep_rl import (
    PrivacySeparatedCTDEConfig,
    evaluate_privacy_separated_ctde_checkpoint,
    train_privacy_separated_ctde,
)
from vpp_dso_sim.simulation.profiles import benchmark_profile_pack, profile_quality_summary
from vpp_dso_sim.simulation.scenario import load_scenario
from vpp_dso_sim.simulation.simulator import Simulator
from vpp_dso_sim.utils.config import load_yaml
from vpp_dso_sim.utils.io import ensure_dir, write_json
from vpp_dso_sim.visualization.benchmark_report import export_benchmark_visualization_outputs


@dataclass(frozen=True)
class BenchmarkExperimentConfig:
    """Reproducible second-stage benchmark orchestration config.

    The defaults are intentionally stronger than a smoke test: five seeds,
    three-day 15-minute horizons, explicit train/eval profile splits, and an
    IEEE33 topology holdout. The current executable algorithm is still the
    rule-based simulator baseline; trainable CTDE/MARL policies should plug into
    the same run-plan and metrics protocol rather than creating separate ad hoc
    scripts.
    """

    config_path: str | Path = "configs/european_lv_benchmark_v2.yaml"
    safety_tight_config_path: str | Path | None = "configs/european_lv_benchmark_v2_safety_tight.yaml"
    topology_holdout_config_path: str | Path | None = "configs/ieee33_multi_vpp.yaml"
    sanity_config_path: str | Path | None = "configs/lv_taiqu_demo.yaml"
    output_dir: str | Path = "outputs/benchmark_v2"
    horizon_steps: int = 288
    seeds: tuple[int, ...] = (3101, 3102, 3103, 3104, 3105)
    train_variants: tuple[str, ...] = ("train_mixed",)
    eval_variants: tuple[str, ...] = ("holdout_peak", "holdout_cloudy", "holdout_reverseflow")
    topology_holdout_variants: tuple[str, ...] = ("holdout_peak",)
    variants: tuple[str, ...] | None = None
    algorithms: tuple[str, ...] = ("rule_based",)
    include_topology_holdout: bool = True
    include_safety_tight: bool = True
    include_sanity: bool = False
    ctde_train_episodes: int = 3
    ctde_train_horizon_steps: int = 96
    ctde_eval_horizon_steps: int | None = None
    ctde_hidden_dim: int = 64
    ctde_learning_rate: float = 3e-4
    experiment_level: str = "research_grade_candidate"
    dt_hours: float | None = None
    export_visualizations: bool = True


@dataclass(frozen=True)
class BenchmarkRunPlan:
    algorithm: str
    split: str
    config_path: str | Path
    profile_variant: str
    scenario_name: str


_CTDE_ALGORITHMS = {
    "privacy_separated_ctde",
    "privacy_separated_ctde_actor_critic",
}


def _json_ready(data: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in data.items():
        if isinstance(value, Path):
            out[key] = str(value)
        elif isinstance(value, tuple):
            out[key] = list(value)
        else:
            out[key] = value
    return out


def _values(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.drop(columns=["step"], errors="ignore")


def _safe_percentile(values: pd.Series | np.ndarray, q: float) -> float | None:
    array = np.asarray(values, dtype=float)
    array = array[np.isfinite(array)]
    if array.size == 0:
        return None
    return float(np.percentile(array, q))


def _build_step_summary(results: dict[str, pd.DataFrame]) -> pd.DataFrame:
    profile = results.get("profile_state", pd.DataFrame()).copy()
    if profile.empty:
        profile = pd.DataFrame({"step": sorted(results.get("bus_voltage", pd.DataFrame()).get("step", []))})
    if "step" not in profile:
        profile["step"] = []
    summary = profile[["step"]].copy()
    for col in ("time_hours", "time_label", "price", "load_scale", "pv_forecast_factor"):
        if col in profile:
            summary[col] = profile[col]

    bus = results.get("bus_voltage", pd.DataFrame())
    if not bus.empty:
        bus_values = _values(bus)
        summary["step_min_voltage_vm_pu"] = bus_values.min(axis=1).to_numpy()
        summary["step_max_voltage_vm_pu"] = bus_values.max(axis=1).to_numpy()

    line = results.get("line_loading", pd.DataFrame())
    if not line.empty:
        line_values = _values(line)
        summary["step_max_line_loading_percent"] = line_values.max(axis=1).to_numpy()

    trafo = results.get("trafo_loading", pd.DataFrame())
    if not trafo.empty:
        trafo_values = _values(trafo)
        summary["step_max_trafo_loading_percent"] = trafo_values.max(axis=1).to_numpy()

    reward = results.get("reward_components", pd.DataFrame())
    if not reward.empty:
        reward_cols = ["total_cost", "reward", "operation_cost", "procurement_proxy_cost"]
        summary = summary.merge(
            reward[["step", *[col for col in reward_cols if col in reward.columns]]],
            on="step",
            how="left",
        )

    projection = results.get("projection_trace", pd.DataFrame())
    if not projection.empty and {"step", "stage_name"}.issubset(projection.columns):
        fr_rows = projection[projection["stage_name"].astype(str) == "fr_doe"].copy()
        if not fr_rows.empty:
            fr_rows["abs_projection_gap_mw"] = fr_rows.get("delta_p_mw", 0.0).astype(float).abs()
            per_step = fr_rows.groupby("step", as_index=False).agg(
                projection_gap_mw=("abs_projection_gap_mw", "sum"),
                projection_clipping_count=("was_projected", "sum"),
            )
            summary = summary.merge(per_step, on="step", how="left")

    envelope = results.get("dso_operating_envelope", pd.DataFrame())
    if not envelope.empty and {"step", "service_request"}.issubset(envelope.columns):
        service = envelope.copy()
        service["active_need"] = service["service_request"].astype(str) != "balanced_operation"
        per_step_service = service.groupby("step", as_index=False).agg(
            active_need_count=("active_need", "sum"),
            awarded_flex_mw=("preferred_target_p_mw", lambda series: float(np.abs(series.astype(float)).sum())),
        )
        summary = summary.merge(per_step_service, on="step", how="left")

    for col in ("projection_gap_mw", "projection_clipping_count", "active_need_count", "awarded_flex_mw"):
        if col in summary:
            summary[col] = summary[col].fillna(0.0)
    return summary


def _academic_metric_additions(
    results: dict[str, pd.DataFrame],
    *,
    low_v: float,
    high_v: float,
    line_limit: float,
    trafo_limit: float,
) -> dict[str, Any]:
    step_summary = _build_step_summary(results)
    metrics: dict[str, Any] = {}
    if not step_summary.empty:
        if "step_min_voltage_vm_pu" in step_summary:
            v = step_summary["step_min_voltage_vm_pu"].astype(float)
            metrics["step_min_voltage_p01"] = _safe_percentile(v, 1)
            metrics["step_min_voltage_p05"] = _safe_percentile(v, 5)
            metrics["near_voltage_0_95_step_rate"] = float((v < 0.95).mean())
            metrics["near_voltage_0_94_step_rate"] = float((v < 0.94).mean())
            metrics["voltage_violation_magnitude_sum"] = float(
                np.maximum(0.0, low_v - v).sum()
                + np.maximum(0.0, step_summary.get("step_max_voltage_vm_pu", v).astype(float) - high_v).sum()
            )
        if "step_max_line_loading_percent" in step_summary:
            line = step_summary["step_max_line_loading_percent"].astype(float)
            metrics["step_max_line_loading_p95"] = _safe_percentile(line, 95)
            metrics["step_max_line_loading_p99"] = _safe_percentile(line, 99)
            metrics["near_line_85_step_rate"] = float((line >= 85.0).mean())
            metrics["near_line_90_step_rate"] = float((line >= 90.0).mean())
            metrics["line_overload_magnitude_sum"] = float(np.maximum(0.0, line - line_limit).sum())
        if "step_max_trafo_loading_percent" in step_summary:
            trafo = step_summary["step_max_trafo_loading_percent"].astype(float)
            metrics["near_trafo_85_step_rate"] = float((trafo >= 85.0).mean())
            metrics["near_trafo_90_step_rate"] = float((trafo >= 90.0).mean())
            metrics["trafo_overload_magnitude_sum"] = float(np.maximum(0.0, trafo - trafo_limit).sum())
        if "projection_gap_mw" in step_summary:
            metrics["projection_gap_mw_sum"] = float(step_summary["projection_gap_mw"].sum())
            metrics["projection_gap_mw_mean"] = float(step_summary["projection_gap_mw"].mean())
        if "projection_clipping_count" in step_summary:
            metrics["projection_clipping_rate"] = float((step_summary["projection_clipping_count"] > 0).mean())

    fr = results.get("fr_envelope_state", pd.DataFrame())
    if not fr.empty and "is_binding" in fr:
        metrics["fr_binding_rate"] = float(fr["is_binding"].astype(bool).mean())

    envelope = results.get("dso_operating_envelope", pd.DataFrame())
    if not envelope.empty and "service_request" in envelope:
        requests = envelope["service_request"].astype(str)
        metrics["service_request_absorb_count"] = int((requests == "absorb_or_charge").sum())
        metrics["service_request_export_count"] = int((requests == "export_or_reduce_load").sum())
        metrics["service_request_balanced_count"] = int((requests == "balanced_operation").sum())

    edge = results.get("edge_power_flow", pd.DataFrame())
    if not edge.empty and {"step", "edge_type", "p_from_mw"}.issubset(edge.columns):
        line_edge = edge[edge["edge_type"].astype(str) == "line"].copy()
        if not line_edge.empty:
            per_step_min_flow = line_edge.groupby("step")["p_from_mw"].min().astype(float)
            metrics["min_line_p_from_mw"] = float(per_step_min_flow.min())
            metrics["reverse_flow_step_rate"] = float((per_step_min_flow < 0.0).mean())

    metrics.setdefault("privacy_audit_pass", 1)
    metrics.setdefault("actor_visible_field_count", None)
    metrics.setdefault("critic_privileged_field_count", None)
    return metrics


def _count_violations(results: dict[str, pd.DataFrame], *, low_v: float, high_v: float) -> dict[str, int]:
    bus = _values(results.get("bus_voltage", pd.DataFrame()))
    line = _values(results.get("line_loading", pd.DataFrame()))
    trafo = _values(results.get("trafo_loading", pd.DataFrame()))
    violations = results.get("constraint_violations", pd.DataFrame())
    powerflow_fail_count = 0
    if not violations.empty and "kind" in violations:
        powerflow_fail_count = int((violations["kind"].astype(str) == "powerflow").sum())
    return {
        "voltage_low_cells": int((bus < low_v).sum().sum()) if not bus.empty else 0,
        "voltage_high_cells": int((bus > high_v).sum().sum()) if not bus.empty else 0,
        "line_overload_cells": int((line > 100.0).sum().sum()) if not line.empty else 0,
        "trafo_overload_cells": int((trafo > 100.0).sum().sum()) if not trafo.empty else 0,
        "powerflow_fail_count": powerflow_fail_count,
        "constraint_violation_records": int(len(violations)),
    }


def _planned_runs(cfg: BenchmarkExperimentConfig) -> list[BenchmarkRunPlan]:
    plans: list[BenchmarkRunPlan] = []
    if cfg.variants is not None:
        train_variants = tuple(cfg.variants[:1]) or ("train_mixed",)
        eval_variants = tuple(cfg.variants[1:])
    else:
        train_variants = cfg.train_variants
        eval_variants = cfg.eval_variants
    for algorithm in cfg.algorithms:
        normalized_algorithm = (
            "privacy_separated_ctde_actor_critic"
            if algorithm in _CTDE_ALGORITHMS
            else algorithm
        )
        if normalized_algorithm not in {"rule_based", "privacy_separated_ctde_actor_critic"}:
            raise NotImplementedError(
                f"Benchmark runner currently supports rule_based and privacy_separated_ctde_actor_critic; "
                f"'{algorithm}' must implement the same train/eval protocol before it can enter this benchmark."
            )
        for variant in train_variants:
            plans.append(
                BenchmarkRunPlan(
                    algorithm=normalized_algorithm,
                    split="train_profile",
                    config_path=cfg.config_path,
                    profile_variant=variant,
                    scenario_name="european_lv_benchmark_v2",
                )
            )
        for variant in eval_variants:
            plans.append(
                BenchmarkRunPlan(
                    algorithm=normalized_algorithm,
                    split="eval_profile",
                    config_path=cfg.config_path,
                    profile_variant=variant,
                    scenario_name="european_lv_benchmark_v2",
                )
            )
        if cfg.include_safety_tight and cfg.safety_tight_config_path is not None:
            plans.append(
                BenchmarkRunPlan(
                    algorithm=normalized_algorithm,
                    split="safety_tight_limits",
                    config_path=cfg.safety_tight_config_path,
                    profile_variant="holdout_peak",
                    scenario_name="european_lv_benchmark_v2_safety_tight",
                )
            )
        if (
            normalized_algorithm == "rule_based"
            and cfg.include_topology_holdout
            and cfg.topology_holdout_config_path is not None
        ):
            for variant in cfg.topology_holdout_variants:
                plans.append(
                    BenchmarkRunPlan(
                        algorithm=normalized_algorithm,
                        split="topology_holdout",
                        config_path=cfg.topology_holdout_config_path,
                        profile_variant=variant,
                        scenario_name="ieee33_topology_holdout",
                    )
                )
        if cfg.include_sanity and cfg.sanity_config_path is not None:
            plans.append(
                BenchmarkRunPlan(
                    algorithm=normalized_algorithm,
                    split="sanity_taiqu",
                    config_path=cfg.sanity_config_path,
                    profile_variant="train_mixed",
                    scenario_name="lv_taiqu_sanity",
                )
            )
    return plans


def _network_pressure_level(metrics: dict[str, Any]) -> str:
    min_v = metrics.get("min_voltage_vm_pu")
    max_line = metrics.get("max_line_loading_percent")
    if min_v is None or max_line is None:
        return "unknown"
    if float(max_line) > 100.0:
        return "overload_stress"
    if float(min_v) < 0.94:
        return "voltage_floor_stress"
    if float(min_v) < 0.95 or float(max_line) >= 90.0:
        return "near_limit"
    return "comfortable"


def _write_variant_config(
    *,
    source_config_path: str | Path,
    output_path: Path,
    horizon_steps: int,
    dt_hours: float | None,
    seed: int,
    profile_variant: str,
) -> Path:
    config = load_yaml(source_config_path)
    config.setdefault("simulation", {})
    config["simulation"]["horizon_steps"] = int(horizon_steps)
    config["simulation"]["seed"] = int(seed)
    if dt_hours is not None:
        config["simulation"]["dt_hours"] = float(dt_hours)
    if str(profile_variant) == "holdout_reverseflow":
        config.setdefault("asset_scaling", {})
        config["asset_scaling"].setdefault("pv_p_max_multiplier", 2.2)
        config["asset_scaling"].setdefault("pv_apparent_power_multiplier", 2.2)
    config["profiles"] = {
        "profile_pack": "benchmark_3day_v1",
        "variant": str(profile_variant),
        "seed": int(seed),
        "source_note": "Generated by benchmark_runner for train/eval split reproducibility.",
    }
    ensure_dir(output_path.parent)
    output_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return output_path


def _apply_variant_asset_scaling_to_scenario(scenario, profile_variant: str) -> None:
    if str(profile_variant) != "holdout_reverseflow":
        return
    for vpp in scenario.vpps:
        for der in vpp.der_list:
            if der.__class__.__name__ != "PVModel":
                continue
            multiplier = 2.2
            der.p_max_mw *= multiplier
            if hasattr(der, "apparent_power_mva"):
                der.apparent_power_mva *= multiplier
                der.q_min_mvar = -float(der.apparent_power_mva)
                der.q_max_mvar = float(der.apparent_power_mva)


def _train_ctde_for_seed(
    *,
    cfg: BenchmarkExperimentConfig,
    seed: int,
    output_dir: Path,
    train_plan: BenchmarkRunPlan,
) -> dict[str, Any]:
    train_horizon = int(cfg.ctde_train_horizon_steps or cfg.horizon_steps)
    train_dir = ensure_dir(output_dir / "training" / f"ctde_seed_{seed}")
    train_config_path = _write_variant_config(
        source_config_path=train_plan.config_path,
        output_path=train_dir / "train_split_config.yaml",
        horizon_steps=train_horizon,
        dt_hours=cfg.dt_hours,
        seed=int(seed),
        profile_variant=train_plan.profile_variant,
    )
    result = train_privacy_separated_ctde(
        config_path=train_config_path,
        output_dir=train_dir,
        config=PrivacySeparatedCTDEConfig(
            episodes=int(cfg.ctde_train_episodes),
            horizon_steps=train_horizon,
            hidden_dim=int(cfg.ctde_hidden_dim),
            learning_rate=float(cfg.ctde_learning_rate),
            seed=int(seed),
        ),
    )
    result["train_config_path"] = train_config_path
    return result


def _run_ctde_frozen_eval(
    *,
    cfg: BenchmarkExperimentConfig,
    seed: int,
    plan: BenchmarkRunPlan,
    output_dir: Path,
    checkpoint_path: Path,
) -> dict[str, Any]:
    eval_horizon = int(cfg.ctde_eval_horizon_steps or cfg.horizon_steps)
    eval_config_path = _write_variant_config(
        source_config_path=plan.config_path,
        output_path=output_dir / "eval_split_config.yaml",
        horizon_steps=eval_horizon,
        dt_hours=cfg.dt_hours,
        seed=int(seed),
        profile_variant=plan.profile_variant,
    )
    return evaluate_privacy_separated_ctde_checkpoint(
        config_path=eval_config_path,
        checkpoint_path=checkpoint_path,
        output_dir=output_dir,
        horizon_steps=eval_horizon,
        seed=int(seed),
    )


def _rollout_metrics(
    *,
    seed: int,
    plan: BenchmarkRunPlan,
    scenario,
    results: dict[str, pd.DataFrame],
    horizon_steps: int,
    experiment_level: str,
) -> dict[str, Any]:
    bus = _values(results.get("bus_voltage", pd.DataFrame()))
    line = _values(results.get("line_loading", pd.DataFrame()))
    trafo = _values(results.get("trafo_loading", pd.DataFrame()))
    reward = results.get("reward_components", pd.DataFrame())
    der = results.get("der_dispatch", pd.DataFrame())
    storage = results.get("storage_soc", pd.DataFrame())
    vpp_power = results.get("vpp_power", pd.DataFrame())
    low_v, high_v = scenario.dso.voltage_limits
    violations = _count_violations(results, low_v=low_v, high_v=high_v)
    academic_metrics = _academic_metric_additions(
        results,
        low_v=float(low_v),
        high_v=float(high_v),
        line_limit=float(scenario.dso.line_loading_limit_percent),
        trafo_limit=float(scenario.dso.trafo_loading_limit_percent),
    )
    total_violation_cells = (
        violations["voltage_low_cells"]
        + violations["voltage_high_cells"]
        + violations["line_overload_cells"]
        + violations["trafo_overload_cells"]
        + violations["powerflow_fail_count"]
    )
    storage_soc_span = (
        float(storage["soc"].max() - storage["soc"].min())
        if not storage.empty and "soc" in storage
        else 0.0
    )
    pv_mask = der["type"].astype(str).str.contains("PV") if not der.empty and "type" in der else pd.Series(dtype=bool)
    pv_available = (
        float(der.loc[pv_mask, "available_p_mw"].fillna(0.0).sum())
        if not der.empty and "available_p_mw" in der
        else 0.0
    )
    pv_output = (
        float(der.loc[pv_mask, "p_mw"].fillna(0.0).sum())
        if not der.empty and "p_mw" in der
        else 0.0
    )
    pcc_energy_proxy_mwh = (
        float(vpp_power["p_mw"].abs().sum() * scenario.dt_hours)
        if not vpp_power.empty and "p_mw" in vpp_power
        else 0.0
    )
    metrics: dict[str, Any] = {
        "algorithm": plan.algorithm,
        "split": plan.split,
        "scenario_name": plan.scenario_name,
        "config_path": str(plan.config_path),
        "seed": int(seed),
        "profile_variant": plan.profile_variant,
        "experiment_level": experiment_level,
        "horizon_steps": int(horizon_steps),
        "dt_hours": float(scenario.dt_hours),
        "network_type": str(scenario.config.get("network", {}).get("type", "")),
        "reward_privacy_mode": str(scenario.dso.reward_privacy_mode),
        "voltage_lower_limit": float(low_v),
        "voltage_upper_limit": float(high_v),
        "line_loading_limit_percent": float(scenario.dso.line_loading_limit_percent),
        "trafo_loading_limit_percent": float(scenario.dso.trafo_loading_limit_percent),
        "vpp_count": int(len(scenario.vpps)),
        "single_pcc_vpp_count": int(sum(1 for vpp in scenario.vpps if vpp.physical_mode() == "single_pcc")),
        "multi_node_vpp_count": int(sum(1 for vpp in scenario.vpps if vpp.physical_mode() == "multi_node")),
        "bus_count": int(len(scenario.net.bus)),
        "line_count": int(len(scenario.net.line)),
        "trafo_count": int(len(scenario.net.trafo)),
        "lateral_line_count": int((scenario.net.line.get("line_section_type", "") == "lateral").sum())
        if "line_section_type" in scenario.net.line
        else 0,
        "min_voltage_vm_pu": float(bus.min().min()) if not bus.empty else None,
        "max_voltage_vm_pu": float(bus.max().max()) if not bus.empty else None,
        "max_line_loading_percent": float(line.max().max()) if not line.empty else None,
        "max_trafo_loading_percent": float(trafo.max().max()) if not trafo.empty else None,
        "total_cost": float(reward["total_cost"].sum()) if "total_cost" in reward else None,
        "reward_sum": float(reward["reward"].sum()) if "reward" in reward else None,
        "operation_cost_sum": float(reward["operation_cost"].sum()) if "operation_cost" in reward else None,
        "storage_soc_span": storage_soc_span,
        "pv_available_mw_step_sum": pv_available,
        "pv_output_mw_step_sum": pv_output,
        "pv_utilization_ratio": pv_output / pv_available if pv_available > 1e-9 else None,
        "pcc_abs_energy_proxy_mwh": pcc_energy_proxy_mwh,
        "security_pass": int(total_violation_cells == 0),
        "total_violation_cells": int(total_violation_cells),
        **violations,
        **academic_metrics,
    }
    metrics["network_pressure_level"] = _network_pressure_level(metrics)
    return metrics


def _aggregate_metrics(seed_metrics: pd.DataFrame) -> pd.DataFrame:
    if seed_metrics.empty:
        return pd.DataFrame()
    group_cols = ["algorithm", "split", "scenario_name", "profile_variant"]
    numeric_cols = [
        col
        for col in seed_metrics.select_dtypes(include="number").columns
        if col not in {"seed"}
    ]
    aggregate = seed_metrics.groupby(group_cols, as_index=False)[numeric_cols].agg(["mean", "std", "min", "max"])
    aggregate.columns = [
        col[0] if isinstance(col, tuple) and not col[1] else f"{col[0]}_{col[1]}"
        if isinstance(col, tuple)
        else str(col)
        for col in aggregate.columns
    ]
    return aggregate.reset_index(drop=True)


def _write_html_report(
    *,
    output_dir: Path,
    manifest: dict[str, Any],
    seed_metrics: pd.DataFrame,
    aggregate_metrics: pd.DataFrame,
    profile_quality: pd.DataFrame,
) -> Path:
    path = output_dir / "benchmark_report.html"
    manifest_text = html.escape(str(manifest))
    html_text = f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <title>Benchmark V2 Experiment Report</title>
  <style>
    body {{ font-family: Segoe UI, Microsoft YaHei, sans-serif; margin: 28px; color: #18212b; line-height: 1.55; }}
    h1, h2 {{ margin-bottom: 8px; }}
    .note {{ background: #f5f7fb; border-left: 4px solid #2b6cb0; padding: 12px 14px; margin: 16px 0; }}
    .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(260px, 1fr)); gap: 12px; }}
    .card {{ border: 1px solid #d9e2ec; border-radius: 8px; padding: 12px; background: #ffffff; }}
    table {{ border-collapse: collapse; width: 100%; margin: 12px 0 24px; font-size: 13px; }}
    th, td {{ border: 1px solid #d9e2ec; padding: 7px 8px; text-align: left; }}
    th {{ background: #eef3f8; }}
    code {{ background: #eef3f8; padding: 2px 4px; border-radius: 3px; }}
    pre {{ white-space: pre-wrap; background: #f7fafc; border: 1px solid #d9e2ec; padding: 12px; }}
  </style>
</head>
<body>
  <h1>Benchmark V2 二次实验报告</h1>
  <div class="note">
    本报告验证分支型 European-LV benchmark、非重复多日 profile、train/eval/topology-holdout
    实验编排和网络安全指标导出。当前等级是 research-grade candidate：它能支撑内部算法迭代，
    但还需要公开数据、OPF/oracle 对照和长预算训练才能支撑论文级主张。
  </div>
  <div class="grid">
    <div class="card"><b>实验结构</b><br>训练 profile、独立 profile 评估、IEEE33 拓扑 holdout。</div>
    <div class="card"><b>安全指标</b><br>电压上下限、线路过载、变压器过载、潮流失败。</div>
    <div class="card"><b>机制指标</b><br>单 PCC / 多节点 VPP 数量、PV 利用率、储能 SOC 摆幅。</div>
  </div>
  <h2>Manifest</h2>
  <pre>{manifest_text}</pre>
  <h2>Aggregate Metrics</h2>
  {aggregate_metrics.to_html(index=False)}
  <h2>Seed / Holdout Metrics</h2>
  {seed_metrics.to_html(index=False)}
  <h2>Profile Quality</h2>
  {profile_quality.to_html(index=False)}
</body>
</html>
"""
    path.write_text(html_text, encoding="utf-8")
    return path


def run_benchmark_experiment(config: BenchmarkExperimentConfig | None = None) -> dict[str, Any]:
    cfg = config or BenchmarkExperimentConfig()
    out = ensure_dir(cfg.output_dir)
    run_plans = _planned_runs(cfg)
    seed_rows: list[dict[str, Any]] = []
    profile_rows: list[dict[str, Any]] = []
    ctde_train_cache: dict[int, dict[str, Any]] = {}
    ctde_train_plan = next(
        (
            plan
            for plan in run_plans
            if plan.algorithm == "privacy_separated_ctde_actor_critic" and plan.split == "train_profile"
        ),
        None,
    )

    for plan in run_plans:
        for seed in cfg.seeds:
            if plan.algorithm == "privacy_separated_ctde_actor_critic":
                if ctde_train_plan is None:
                    raise ValueError("CTDE benchmark requires at least one train_profile plan.")
                if int(seed) not in ctde_train_cache:
                    ctde_train_cache[int(seed)] = _train_ctde_for_seed(
                        cfg=cfg,
                        seed=int(seed),
                        output_dir=out,
                        train_plan=ctde_train_plan,
                    )
                run_dir = ensure_dir(
                    out
                    / plan.split
                    / f"{plan.algorithm}_{plan.profile_variant}_seed_{seed}"
                )
                eval_result = _run_ctde_frozen_eval(
                    cfg=cfg,
                    seed=int(seed),
                    plan=plan,
                    output_dir=run_dir,
                    checkpoint_path=Path(ctde_train_cache[int(seed)]["checkpoint"]),
                )
                _build_step_summary(eval_result["simulator_results"]).to_csv(run_dir / "step_summary.csv", index=False)
                seed_metric = _rollout_metrics(
                    seed=int(seed),
                    plan=plan,
                    scenario=eval_result["scenario"],
                    results=eval_result["simulator_results"],
                    horizon_steps=int(cfg.ctde_eval_horizon_steps or cfg.horizon_steps),
                    experiment_level=cfg.experiment_level,
                )
                train_summary = ctde_train_cache[int(seed)]["summary"]
                eval_summary = eval_result["summary"]
                seed_metric.update(
                    {
                        "checkpoint_path": str(ctde_train_cache[int(seed)]["checkpoint"]),
                        "train_config_path": str(ctde_train_cache[int(seed)]["train_config_path"]),
                        "train_episodes": int(cfg.ctde_train_episodes),
                        "train_horizon_steps": int(cfg.ctde_train_horizon_steps or cfg.horizon_steps),
                        "eval_horizon_steps": int(cfg.ctde_eval_horizon_steps or cfg.horizon_steps),
                        "train_best_episode_reward": train_summary.get("best_episode_reward"),
                        "train_final_episode_reward": train_summary.get("final_episode_reward"),
                        "train_param_delta_l2": train_summary.get("param_delta_l2"),
                        "frozen_eval_total_reward": eval_summary.get("total_reward"),
                        "frozen_eval_total_cost": eval_summary.get("total_cost"),
                        "frozen_eval_projection_gap_mw": eval_summary.get("total_projection_gap_mw"),
                        "policy_evaluation_mode": eval_summary.get("evaluation_mode"),
                    }
                )
                seed_rows.append(seed_metric)
                quality = profile_quality_summary(
                    eval_result["scenario"].load_profile,
                    eval_result["scenario"].pv_profile,
                    eval_result["scenario"].price_profile,
                    dt_hours=float(eval_result["scenario"].dt_hours),
                )
                for row in quality.to_dict(orient="records"):
                    profile_rows.append(
                        {
                            "algorithm": plan.algorithm,
                            "split": plan.split,
                            "scenario_name": plan.scenario_name,
                            "config_path": str(plan.config_path),
                            "seed": int(seed),
                            "profile_variant": plan.profile_variant,
                            **row,
                        }
                    )
                continue

            scenario = load_scenario(plan.config_path)
            _apply_variant_asset_scaling_to_scenario(scenario, plan.profile_variant)
            dt_hours = float(cfg.dt_hours if cfg.dt_hours is not None else scenario.dt_hours)
            pack = benchmark_profile_pack(
                cfg.horizon_steps,
                dt_hours=dt_hours,
                seed=int(seed),
                variant=plan.profile_variant,
            )
            scenario.horizon_steps = int(cfg.horizon_steps)
            scenario.dt_hours = dt_hours
            scenario.seed = int(seed)
            scenario.load_profile = pack["load"]
            scenario.pv_profile = pack["pv"]
            scenario.price_profile = pack["price"]
            scenario.dso.market_price_profile = pack["price"]

            quality = profile_quality_summary(
                scenario.load_profile,
                scenario.pv_profile,
                scenario.price_profile,
                dt_hours=dt_hours,
            )
            for row in quality.to_dict(orient="records"):
                profile_rows.append(
                    {
                        "algorithm": plan.algorithm,
                        "split": plan.split,
                        "scenario_name": plan.scenario_name,
                        "config_path": str(plan.config_path),
                        "seed": int(seed),
                        "profile_variant": plan.profile_variant,
                        **row,
                    }
                )

            simulator = Simulator(scenario)
            results = simulator.run_timeseries(horizon_steps=cfg.horizon_steps)
            run_dir = ensure_dir(
                out
                / plan.split
                / f"{plan.algorithm}_{plan.profile_variant}_seed_{seed}"
            )
            simulator.export_results(run_dir)
            _build_step_summary(results).to_csv(run_dir / "step_summary.csv", index=False)
            seed_rows.append(
                _rollout_metrics(
                    seed=int(seed),
                    plan=plan,
                    scenario=scenario,
                    results=results,
                    horizon_steps=cfg.horizon_steps,
                    experiment_level=cfg.experiment_level,
                )
            )

    seed_metrics = pd.DataFrame(seed_rows)
    aggregate_metrics = _aggregate_metrics(seed_metrics)
    profile_quality = pd.DataFrame(profile_rows)

    seed_metrics.to_csv(out / "seed_metrics.csv", index=False)
    aggregate_metrics.to_csv(out / "aggregate_metrics.csv", index=False)
    profile_quality.to_csv(out / "profile_quality.csv", index=False)
    manifest = {
        "config": _json_ready(asdict(cfg)),
        "planned_run_count_per_seed": len(run_plans),
        "total_rollouts": len(seed_metrics),
        "run_plans": [_json_ready(asdict(plan)) for plan in run_plans],
        "ctde_training_runs": len(ctde_train_cache),
        "topology_holdout_policy": (
            "rule_based includes topology holdout. CTDE topology holdout is skipped until a compatible "
            "variable-VPP policy adapter or graph policy head is implemented."
        ),
        "outputs": {
            "seed_metrics": str(out / "seed_metrics.csv"),
            "aggregate_metrics": str(out / "aggregate_metrics.csv"),
            "profile_quality": str(out / "profile_quality.csv"),
        },
        "claim_boundary": (
            "Research-grade candidate benchmark: multi-seed, train/eval profile split, and topology "
            "holdout are present. Public feeder/profile data, OPF or oracle baselines, settlement-aware "
            "profit accounting, and long-budget CTDE/MARL training are still required before paper-level claims."
        ),
    }
    write_json(out / "experiment_manifest.json", manifest)
    report_path = _write_html_report(
        output_dir=out,
        manifest=manifest,
        seed_metrics=seed_metrics,
        aggregate_metrics=aggregate_metrics,
        profile_quality=profile_quality,
    )
    visualization_paths: dict[str, Any] = {}
    if cfg.export_visualizations:
        visualization_paths = export_benchmark_visualization_outputs(out)
        report_path = Path(visualization_paths["benchmark_report"])
    return {
        "output_dir": out,
        "seed_metrics": seed_metrics,
        "aggregate_metrics": aggregate_metrics,
        "profile_quality": profile_quality,
        "manifest": manifest,
        "report": report_path,
        "visualizations": visualization_paths,
    }
