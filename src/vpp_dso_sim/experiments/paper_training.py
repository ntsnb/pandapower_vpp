from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from datetime import datetime
import hashlib
import html
import importlib.metadata as importlib_metadata
import importlib.util
import json
import os
from pathlib import Path
import sys
import time
from typing import Any, Callable

from vpp_dso_sim.utils.runtime import configure_numeric_thread_limits

NUMERIC_THREAD_LIMITS = configure_numeric_thread_limits(default_threads=8)

_BOOT_CACHE_DIR = Path(os.environ.get("VPP_DSO_BOOT_CACHE_DIR", "/tmp/pandapower_vpp_cache"))
(_BOOT_CACHE_DIR / "matplotlib").mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_BOOT_CACHE_DIR / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(_BOOT_CACHE_DIR))

import numpy as np
import pandas as pd
import yaml

from vpp_dso_sim.experiments.benchmark_runner import (
    BenchmarkRunPlan,
    _build_step_summary,
    _json_ready,
    _rollout_metrics,
)
from vpp_dso_sim.optimization.oracle_baseline import build_ac_validated_search_actions
from vpp_dso_sim.optimization.feasibility_region import compute_static_feasible_region
from vpp_dso_sim.simulation.profiles import (
    benchmark_profile_pack,
    profile_quality_summary,
    smart_ds_austin_profile_pack,
)
from vpp_dso_sim.simulation.scenario import load_scenario
from vpp_dso_sim.simulation.simulator import Simulator
from vpp_dso_sim.utils.config import load_yaml
from vpp_dso_sim.utils.io import ensure_dir, write_json


TQDM_AVAILABLE = importlib.util.find_spec("tqdm") is not None


@dataclass(frozen=True)
class PaperTrainingExperimentConfig:
    """Formal training/evaluation campaign for the DSO/VPP MARL stack.

    The presets intentionally separate a fast executable smoke run from the
    heavier budgets that should be launched for paper tables. Training rewards
    are never treated as final performance; trainable policies are evaluated
    through frozen actors on independent profile variants.
    """

    config_path: str | Path = "configs/european_lv_benchmark_v2.yaml"
    output_dir: str | Path = "outputs/paper_training"
    preset: str = "paper_lite"
    algorithms: tuple[str, ...] = (
        "rule_based",
        "no_flex",
        "ac_validated_search_reference",
        "happo",
        "hatrpo",
        "matd3",
        "hasac",
    )
    seeds: tuple[int, ...] = (9301, 9302, 9303)
    train_variants: tuple[str, ...] = ("train_mixed",)
    eval_variants: tuple[str, ...] = ("holdout_peak", "holdout_cloudy")
    hparam_cases: tuple[str, ...] = ("base", "lower_lr", "higher_entropy")
    data_source: str = "smart_ds"
    horizon_steps: int = 288
    eval_horizon_steps: int | None = None
    train_episodes: int = 30
    dt_hours: float | None = 0.25
    hidden_dim: int = 128
    learning_rate: float = 3e-4
    gamma: float = 0.97
    batch_size: int = 128
    replay_capacity: int = 100_000
    warmup_steps: int = 256
    ppo_epochs: int = 3
    tensorboard: bool = True
    export_html: bool = True
    resume_completed: bool = False
    checkpoint_selection: str = "final"
    progress_interval_seconds: float = 60.0
    verbose_progress: bool = False
    ac_reference_max_candidates: int = 16
    happo_critic_use_action_summary: bool = False
    happo_use_yaml_trainer_settings: bool = False
    dispatch_actor_encoder_type: str = "deepset_v1"
    happo_shared_rollout_enabled: bool = False
    happo_shared_rollout_workers: int = 1
    happo_shared_rollout_backend: str = "serial"
    happo_rollout_fragment_steps: int | None = None
    happo_reward_dynamic_reports: bool = True
    happo_reward_dynamic_report_every_episodes: int = 1
    happo_reward_dynamic_report_all_workers: bool = False
    require_cuda_for_trainable: bool = False

    def to_dict(self) -> dict[str, Any]:
        return _json_ready(asdict(self))


def _configure_local_plot_cache(output_dir: Path) -> None:
    """Keep long-run plotting/font caches inside the experiment output tree."""

    cache_dir = ensure_dir(output_dir / ".cache")
    os.environ.setdefault("MPLCONFIGDIR", str(ensure_dir(cache_dir / "matplotlib")))
    os.environ.setdefault("XDG_CACHE_HOME", str(cache_dir))


def _file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _profile_data_hash(config_path: str | Path) -> str:
    config = load_yaml(config_path)
    profiles = config.get("profiles", {}) if isinstance(config, dict) else {}
    digest = hashlib.sha256()
    for key in ("load_profile_csv", "pv_profile_csv", "price_profile_csv"):
        profile_path = profiles.get(key)
        if not profile_path:
            continue
        path = Path(str(profile_path))
        digest.update(key.encode("utf-8"))
        digest.update(_file_sha256(path).encode("ascii"))
    if digest.digest() == hashlib.sha256().digest():
        return _file_sha256(config_path)
    return digest.hexdigest()


def _runtime_versions() -> dict[str, Any]:
    packages = {}
    for name in ("torch", "pandapower", "gymnasium", "numpy", "pandas"):
        try:
            packages[name] = importlib_metadata.version(name)
        except importlib_metadata.PackageNotFoundError:
            packages[name] = None
    return {
        "python": sys.version.split()[0],
        "packages": packages,
    }


def paper_training_preset(name: str) -> PaperTrainingExperimentConfig:
    normalized = str(name).replace("-", "_").lower()
    if normalized == "smoke":
        return PaperTrainingExperimentConfig(
            output_dir="outputs/paper_training_smoke",
            preset="smoke",
            seeds=(9401,),
            train_variants=("train_mixed",),
            eval_variants=("holdout_peak",),
            hparam_cases=("base",),
            horizon_steps=8,
            eval_horizon_steps=8,
            train_episodes=1,
            hidden_dim=16,
            batch_size=4,
            replay_capacity=128,
            warmup_steps=2,
            ppo_epochs=1,
            ac_reference_max_candidates=8,
        )
    if normalized == "pilot":
        return PaperTrainingExperimentConfig(
            output_dir="outputs/paper_training_pilot",
            preset="pilot",
            seeds=(9401, 9402),
            train_variants=("train_mixed",),
            eval_variants=("holdout_peak",),
            hparam_cases=("base", "lower_lr"),
            horizon_steps=96,
            eval_horizon_steps=96,
            train_episodes=8,
            hidden_dim=64,
            batch_size=32,
            replay_capacity=10_000,
            warmup_steps=24,
            ppo_epochs=2,
            ac_reference_max_candidates=12,
        )
    if normalized == "paper_long":
        return PaperTrainingExperimentConfig(
            output_dir="outputs/paper_training_long",
            preset="paper_long",
            seeds=(9401, 9402, 9403, 9404, 9405),
            train_variants=("train_mixed",),
            eval_variants=("holdout_peak", "holdout_cloudy", "holdout_reverseflow"),
            hparam_cases=("base", "lower_lr", "higher_entropy", "larger_network"),
            horizon_steps=672,
            eval_horizon_steps=672,
            train_episodes=120,
            hidden_dim=256,
            gamma=0.995,
            batch_size=256,
            replay_capacity=300_000,
            warmup_steps=2_000,
            ppo_epochs=4,
            checkpoint_selection="both",
            ac_reference_max_candidates=16,
            happo_critic_use_action_summary=True,
            require_cuda_for_trainable=True,
        )
    if normalized in {"paper_long_sensitivity_v1", "paper_long_sensitivity_v1_reward_v3_1_market_safety"}:
        return PaperTrainingExperimentConfig(
            config_path="configs/european_lv_benchmark_v2_sensitivity_attention_v1_reward_v3_1_market_safety.yaml",
            output_dir="outputs/paper_training_long_sensitivity_v1",
            preset=normalized,
            algorithms=("rule_based", "no_flex", "ac_validated_search_reference", "happo"),
            seeds=(9401, 9402, 9403, 9404, 9405),
            train_variants=("train_mixed",),
            eval_variants=("holdout_peak", "holdout_cloudy", "holdout_reverseflow"),
            hparam_cases=("base", "lower_lr", "higher_entropy", "larger_network"),
            horizon_steps=672,
            eval_horizon_steps=672,
            train_episodes=120,
            hidden_dim=256,
            gamma=0.995,
            batch_size=256,
            replay_capacity=300_000,
            warmup_steps=2_000,
            ppo_epochs=4,
            checkpoint_selection="both",
            ac_reference_max_candidates=16,
            happo_critic_use_action_summary=True,
            happo_use_yaml_trainer_settings=True,
            dispatch_actor_encoder_type="set_attention_v1",
            require_cuda_for_trainable=True,
        )
    if normalized != "paper_lite":
        raise ValueError(f"Unknown paper training preset: {name}")
    return PaperTrainingExperimentConfig()


def _profile_pack(
    *,
    cfg: PaperTrainingExperimentConfig,
    horizon_steps: int,
    seed: int,
    variant: str,
) -> dict[str, Any]:
    dt_hours = float(cfg.dt_hours or 0.25)
    if str(cfg.data_source).lower() == "smart_ds":
        return dict(
            smart_ds_austin_profile_pack(
                horizon_steps,
                dt_hours=dt_hours,
                seed=int(seed),
                variant=variant,
            )
        )
    pack = benchmark_profile_pack(horizon_steps, dt_hours=dt_hours, seed=int(seed), variant=variant)
    return {
        **pack,
        "metadata": {
            "source": "synthetic_benchmark_profile_pack",
            "variant": variant,
            "price_source": "synthetic_scarcity_proxy",
        },
    }


def _write_profile_config(
    *,
    cfg: PaperTrainingExperimentConfig,
    output_dir: Path,
    seed: int,
    variant: str,
    horizon_steps: int,
    run_id: str,
) -> tuple[Path, dict[str, Any], pd.DataFrame]:
    profile_dir = ensure_dir(output_dir / "profiles" / run_id)
    pack = _profile_pack(cfg=cfg, horizon_steps=horizon_steps, seed=seed, variant=variant)
    load_path = profile_dir / "load_profile.csv"
    pv_path = profile_dir / "pv_profile.csv"
    price_path = profile_dir / "price_profile.csv"
    pd.DataFrame({"value": pack["load"]}).to_csv(load_path, index=False)
    pd.DataFrame({"value": pack["pv"]}).to_csv(pv_path, index=False)
    pd.DataFrame({"value": pack["price"]}).to_csv(price_path, index=False)

    config = load_yaml(cfg.config_path)
    config.setdefault("simulation", {})
    config["simulation"]["horizon_steps"] = int(horizon_steps)
    config["simulation"]["seed"] = int(seed)
    if cfg.dt_hours is not None:
        config["simulation"]["dt_hours"] = float(cfg.dt_hours)
    if str(variant) == "holdout_reverseflow":
        config.setdefault("asset_scaling", {})
        config["asset_scaling"].setdefault("pv_p_max_multiplier", 2.2)
        config["asset_scaling"].setdefault("pv_apparent_power_multiplier", 2.2)
    config["profiles"] = {
        "load_profile_csv": str(load_path.resolve()),
        "pv_profile_csv": str(pv_path.resolve()),
        "price_profile_csv": str(price_path.resolve()),
        "source": str(pack.get("metadata", {}).get("source", cfg.data_source)),
        "variant": variant,
        "seed": int(seed),
        "price_source": str(pack.get("metadata", {}).get("price_source", "")),
        "source_note": (
            "Profiles are written by paper_training.py. SMART-DS load/PV shapes are used when "
            "available; price is a derived scarcity proxy unless a market dataset is configured."
        ),
    }
    config_path = profile_dir / "scenario_config.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    quality = profile_quality_summary(
        list(pack["load"]),
        list(pack["pv"]),
        list(pack["price"]),
        dt_hours=float(cfg.dt_hours or 0.25),
    )
    metadata = dict(pack.get("metadata", {}))
    metadata.update(
        {
            "run_id": run_id,
            "seed": int(seed),
            "variant": variant,
            "horizon_steps": int(horizon_steps),
            "config_path": str(config_path),
        }
    )
    write_json(profile_dir / "profile_metadata.json", _make_json_safe(metadata))
    return config_path.resolve(), metadata, quality


def _make_json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _make_json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_make_json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    return value


def _sample_frame_for_report(frame: pd.DataFrame, max_rows: int = 4000) -> pd.DataFrame:
    """Keep long HTML responsive while preserving start/middle/end curve shape."""

    if frame.empty or len(frame) <= max_rows:
        return frame
    group_cols = [col for col in ("run_id", "algorithm", "seed", "hparam_case") if col in frame.columns]
    if not group_cols:
        positions = np.linspace(0, len(frame) - 1, max_rows, dtype=int)
        return frame.iloc[np.unique(positions)].copy()
    groups = list(frame.groupby(group_cols, dropna=False, sort=False))
    per_group = max(8, int(np.ceil(max_rows / max(1, len(groups)))))
    pieces: list[pd.DataFrame] = []
    for _, group in groups:
        if len(group) <= per_group:
            pieces.append(group)
            continue
        positions = np.linspace(0, len(group) - 1, per_group, dtype=int)
        pieces.append(group.iloc[np.unique(positions)])
    sampled = pd.concat(pieces, ignore_index=False).sort_index()
    if len(sampled) > max_rows:
        positions = np.linspace(0, len(sampled) - 1, max_rows, dtype=int)
        sampled = sampled.iloc[np.unique(positions)]
    return sampled.copy()


def _checkpoint_choices(train: dict[str, Any], selection: str) -> list[tuple[str, Path]]:
    final = Path(str(train.get("final_checkpoint", train.get("checkpoint", ""))))
    best = Path(str(train.get("best_checkpoint", train.get("checkpoint", final))))
    if selection == "train_best":
        return [("train_best", best)]
    if selection == "both":
        return [("final", final), ("train_best", best)]
    return [("final", final)]


def _as_bool_float(series: pd.Series) -> pd.Series:
    if series.empty:
        return pd.Series(dtype=float)
    if pd.api.types.is_bool_dtype(series):
        return series.astype(float)
    if pd.api.types.is_numeric_dtype(series):
        return series.fillna(0.0).astype(float).clip(lower=0.0, upper=1.0)
    normalized = series.astype(str).str.strip().str.lower()
    return normalized.isin({"true", "1", "yes", "y", "safe", "ok"}).astype(float)


def _certificate_step_frame_from_projection(projection: pd.DataFrame) -> pd.DataFrame:
    if projection.empty or {"step", "stage_name"}.difference(projection.columns):
        return pd.DataFrame()
    cert = projection[projection["stage_name"] == "ac_pf_certificate"].copy()
    if cert.empty:
        return pd.DataFrame()

    cert["step"] = pd.to_numeric(cert["step"], errors="coerce").astype("Int64")
    cert = cert.dropna(subset=["step"]).copy()
    if cert.empty:
        return pd.DataFrame()
    cert["step"] = cert["step"].astype(int)

    trace_rows = cert.groupby("step", as_index=False).size().rename(columns={"size": "ac_certificate_trace_rows"})
    per_step = cert.sort_values(["step", "trace_id"] if "trace_id" in cert.columns else ["step"]).groupby(
        "step", as_index=False
    ).last()
    status = per_step.get("ac_certificate_status", pd.Series("", index=per_step.index)).astype(str)
    safe = (
        _as_bool_float(per_step["ac_certificate_safe"])
        if "ac_certificate_safe" in per_step
        else pd.Series(0.0, index=per_step.index)
    )
    accepted_alpha = (
        pd.to_numeric(per_step["ac_certificate_accepted_alpha"], errors="coerce").fillna(0.0)
        if "ac_certificate_accepted_alpha" in per_step
        else pd.Series(0.0, index=per_step.index)
    )
    gap = (
        pd.to_numeric(per_step["ac_certified_projection_gap_mw"], errors="coerce").fillna(0.0)
        if "ac_certified_projection_gap_mw" in per_step
        else pd.Series(0.0, index=per_step.index)
    )

    frame = pd.DataFrame(
        {
            "step": per_step["step"].astype(int),
            "ac_certificate_status": status,
            "ac_certificate_safe_rate": safe.astype(float),
            "accepted_candidate_ac_safe_rate": status.eq("accepted_candidate_ac_safe").astype(float),
            "repaired_by_ac_powerflow_backoff_rate": status.eq("repaired_by_ac_powerflow_backoff").astype(float),
            "repaired_by_ac_powerflow_emergency_recovery_rate": status.str.startswith(
                "repaired_by_ac_powerflow_emergency_recovery"
            ).astype(float),
            "rolled_back_to_current_safe_dispatch_rate": status.eq("rolled_back_to_current_safe_dispatch").astype(float),
            "certificate_failed_current_dispatch_insecure_rate": status.eq(
                "certificate_failed_current_dispatch_insecure"
            ).astype(float),
            "certificate_failed_no_ac_safe_recovery_rate": status.eq(
                "certificate_failed_no_ac_safe_recovery"
            ).astype(float),
            "mean_ac_certificate_accepted_alpha": accepted_alpha.astype(float),
            "mean_ac_certified_projection_gap_mw": gap.astype(float),
            "ac_certificate_backoff_count": status.isin(
                {"repaired_by_ac_powerflow_backoff", "rolled_back_to_current_safe_dispatch"}
            ).astype(float)
            + status.str.startswith(
                "repaired_by_ac_powerflow_emergency_recovery"
            ).astype(float),
        }
    )
    return frame.merge(trace_rows, on="step", how="left")


def _certificate_summary_from_projection(projection: pd.DataFrame) -> dict[str, Any]:
    cert_steps = _certificate_step_frame_from_projection(projection)
    if cert_steps.empty:
        return {
            "ac_certificate_safe_rate": 0.0,
            "accepted_candidate_ac_safe_rate": 0.0,
            "repaired_by_ac_powerflow_backoff_rate": 0.0,
            "repaired_by_ac_powerflow_emergency_recovery_rate": 0.0,
            "rolled_back_to_current_safe_dispatch_rate": 0.0,
            "certificate_failed_current_dispatch_insecure_rate": 0.0,
            "certificate_failed_no_ac_safe_recovery_rate": 0.0,
            "mean_ac_certificate_accepted_alpha": 0.0,
            "mean_ac_certified_projection_gap_mw": 0.0,
            "ac_certificate_backoff_count": 0,
            "ac_certificate_trace_rows": 0,
        }
    numeric = cert_steps.select_dtypes(include="number")
    def mean_or_zero(column: str) -> float:
        if column not in numeric:
            return 0.0
        value = float(numeric[column].fillna(0.0).mean())
        return value if np.isfinite(value) else 0.0

    def sum_or_zero(column: str) -> float:
        if column not in numeric:
            return 0.0
        value = float(numeric[column].fillna(0.0).sum())
        return value if np.isfinite(value) else 0.0

    return {
        "ac_certificate_safe_rate": mean_or_zero("ac_certificate_safe_rate"),
        "accepted_candidate_ac_safe_rate": mean_or_zero("accepted_candidate_ac_safe_rate"),
        "repaired_by_ac_powerflow_backoff_rate": mean_or_zero("repaired_by_ac_powerflow_backoff_rate"),
        "repaired_by_ac_powerflow_emergency_recovery_rate": mean_or_zero(
            "repaired_by_ac_powerflow_emergency_recovery_rate"
        ),
        "rolled_back_to_current_safe_dispatch_rate": mean_or_zero("rolled_back_to_current_safe_dispatch_rate"),
        "certificate_failed_current_dispatch_insecure_rate": mean_or_zero(
            "certificate_failed_current_dispatch_insecure_rate"
        ),
        "certificate_failed_no_ac_safe_recovery_rate": mean_or_zero(
            "certificate_failed_no_ac_safe_recovery_rate"
        ),
        "mean_ac_certificate_accepted_alpha": mean_or_zero("mean_ac_certificate_accepted_alpha"),
        "mean_ac_certified_projection_gap_mw": mean_or_zero("mean_ac_certified_projection_gap_mw"),
        "ac_certificate_backoff_count": int(sum_or_zero("ac_certificate_backoff_count")),
        "ac_certificate_trace_rows": int(sum_or_zero("ac_certificate_trace_rows")),
    }


def _post_ac_summary_from_reward(reward: pd.DataFrame) -> dict[str, Any]:
    if reward.empty:
        return {
            "post_ac_security_pass_rate": 0.0,
            "post_ac_powerflow_converged_rate": 0.0,
        }
    violation = (
        pd.to_numeric(reward["post_ac_violation_count"], errors="coerce").fillna(0.0)
        if "post_ac_violation_count" in reward
        else pd.Series(0.0, index=reward.index)
    )
    pf_failed = (
        pd.to_numeric(reward["post_ac_powerflow_failed"], errors="coerce").fillna(0.0)
        if "post_ac_powerflow_failed" in reward
        else pd.Series(0.0, index=reward.index)
    )
    return {
        "post_ac_security_pass_rate": float((violation <= 0.0).mean()),
        "post_ac_powerflow_converged_rate": float((pf_failed <= 0.0).mean()),
    }


def _augment_step_metrics_from_simulator_results(step_metrics: pd.DataFrame, output_dir: Path | str | None) -> pd.DataFrame:
    if step_metrics.empty or output_dir is None:
        return step_metrics
    results_dir = Path(output_dir) / "simulator_results"
    if not results_dir.exists():
        return step_metrics
    augmented = step_metrics.copy()
    reward_path = results_dir / "reward_components.csv"
    if reward_path.exists():
        reward = pd.read_csv(reward_path, low_memory=False)
        cols = [
            col
            for col in (
                "step",
                "ac_certified_projection_gap_mw",
                "ac_certificate_failed_count",
                "action_projection_gap_mw",
                "local_bounds_projection_gap_mw",
                "ac_aware_projection_gap_mw",
            )
            if col in reward.columns
        ]
        if "step" in cols:
            augmented = augmented.merge(reward[cols], on="step", how="left", suffixes=("", "_reward_components"))
    projection_path = results_dir / "projection_trace.csv"
    if projection_path.exists():
        projection = pd.read_csv(projection_path, low_memory=False)
        cert_summary = _certificate_step_frame_from_projection(projection)
        if not cert_summary.empty:
            augmented = augmented.merge(cert_summary, on="step", how="left")
    return augmented


def _append_progress_event(output_dir: Path, event: dict[str, Any]) -> None:
    payload = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        **_make_json_safe(event),
    }
    jsonl_path = output_dir / "experiment_progress.jsonl"
    with jsonl_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")
    csv_path = output_dir / "experiment_progress.csv"
    columns = [
        "timestamp",
        "phase",
        "message",
        "run_id",
        "algorithm",
        "seed",
        "profile_variant",
        "eval_variant",
        "hparam_case",
        "train_variant",
        "horizon_steps",
        "episodes",
        "reward_sum",
        "final_episode_reward",
        "total_cost",
        "violations",
        "checkpoint",
        "html_path",
        "preset",
        "algorithms",
        "seeds",
        "eval_horizon_steps",
        "train_episodes",
        "run_count",
        "eval_rows",
        "profile_seed",
        "episode",
        "episode_progress_pct",
        "step",
        "global_step",
        "step_progress_pct",
        "gradient_step",
        "worker_index",
        "worker_count",
        "worker_start_step",
        "local_step",
        "fragment_steps",
        "policy_version",
        "reward_so_far",
        "episode_reward",
        "total_cost_so_far",
        "episode_cost",
        "violations_so_far",
        "violation_count",
        "projection_gap_mw",
        "critic_loss",
        "critic_grad_norm",
        "dso_policy_loss",
        "dispatch_policy_loss",
        "portfolio_policy_loss",
    ]
    row = {column: payload.get(column, "") for column in columns}
    expected_header = ",".join(columns)
    if csv_path.exists():
        with csv_path.open("r", encoding="utf-8") as f:
            current_header = f.readline().strip()
        if current_header != expected_header:
            rows = []
            with jsonl_path.open("r", encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    item = json.loads(line)
                    rows.append({column: item.get(column, "") for column in columns})
            pd.DataFrame(rows, columns=columns).to_csv(csv_path, index=False)
            return
    pd.DataFrame([row], columns=columns).to_csv(
        csv_path,
        mode="a",
        header=not csv_path.exists(),
        index=False,
    )


def _progress_bar(done: int, total: int, width: int = 24) -> str:
    if total <= 0:
        return "[" + "-" * width + "]"
    ratio = max(0.0, min(1.0, float(done) / float(total)))
    filled = int(round(ratio * width))
    return "[" + "#" * filled + "-" * (width - filled) + f"] {ratio * 100:5.1f}%"


def _build_tqdm_bars(state: dict[str, Any]) -> dict[str, Any] | None:
    if not TQDM_AVAILABLE or not sys.stderr.isatty():
        return None
    from tqdm.auto import tqdm

    os.environ.setdefault("VPP_DSO_TQDM_TRAIN_POSITION", "1")
    totals = state.get("totals", {})
    return {
        "overall": tqdm(
            total=int(totals.get("overall", 0)),
            desc="Campaign",
            unit="run",
            dynamic_ncols=True,
            position=0,
            leave=True,
        ),
    }


def _update_tqdm_bars(state: dict[str, Any]) -> bool:
    bars = state.get("_tqdm_bars")
    if not bars:
        return False
    done = state.get("done", {})
    latest = state.get("latest", {})
    previous = state.setdefault("_tqdm_last_done", {})
    totals = state.get("totals", {})
    postfix = {
        "phase": str(latest.get("phase", ""))[:18],
        "B": f"{done.get('baseline', 0)}/{totals.get('baseline', 0)}",
        "T": f"{done.get('train', 0)}/{totals.get('train', 0)}",
        "E": f"{done.get('eval', 0)}/{totals.get('eval', 0)}",
    }
    for stage, bar in bars.items():
        current = int(done.get(stage, 0))
        delta = current - int(previous.get(stage, 0))
        if delta > 0:
            bar.update(delta)
        previous[stage] = current
        if stage == "overall":
            bar.set_postfix(postfix, refresh=False)
    return True


def _tqdm_write(message: str) -> None:
    if TQDM_AVAILABLE and sys.stderr.isatty():
        from tqdm.auto import tqdm

        tqdm.write(message)
    else:
        print(message, flush=True)


def _close_tqdm_bars(state: dict[str, Any]) -> None:
    bars = state.get("_tqdm_bars")
    if not bars:
        return
    for bar in bars.values():
        bar.close()
    state["_tqdm_bars"] = None


def _write_live_progress_dashboard(output_dir: Path, state: dict[str, Any]) -> Path:
    latest = state.get("latest", {})
    totals = state.get("totals", {})
    done = state.get("done", {})

    def percent(stage: str) -> float:
        total = int(totals.get(stage, 0))
        if total <= 0:
            return 0.0
        return max(0.0, min(100.0, 100.0 * int(done.get(stage, 0)) / total))

    rows = []
    for stage, label_zh, note in (
        ("baseline", "Baseline 基线", "rule-based / no-flex / static-FR / AC-validated reference"),
        ("train", "RL 训练", "HAPPO / MATD3 / HASAC checkpoint"),
        ("eval", "冻结评估", "holdout profiles with frozen policy"),
    ):
        stage_done = int(done.get(stage, 0))
        stage_total = int(totals.get(stage, 0))
        stage_percent = percent(stage)
        rows.append(
            f"""<section class="stage">
              <div class="stage-head">
                <div><h3>{html.escape(label_zh)}</h3><p>{html.escape(note)}</p></div>
                <strong>{stage_done}/{stage_total}</strong>
              </div>
              <div class="meter"><span style="width: {stage_percent:.2f}%"></span></div>
              <div class="percent">{stage_percent:.1f}%</div>
            </section>"""
        )
    latest_rows = "".join(
        f"<tr><td>{html.escape(str(key))}</td><td>{html.escape(str(value))}</td></tr>"
        for key, value in latest.items()
    )
    path = output_dir / "live_progress.html"
    overall_percent = percent("overall")
    path.write_text(
        f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta http-equiv="refresh" content="30">
  <title>Paper Training Live Progress</title>
  <style>
    :root {{ --ink:#152130; --muted:#607083; --line:#d8e1ea; --paper:#ffffff; --bg:#f4f7fb; --accent:#1f7a8c; --accent2:#f2a65a; }}
    * {{ box-sizing: border-box; }}
    body {{ font-family: Segoe UI, Microsoft YaHei, sans-serif; margin:0; color:var(--ink); background:var(--bg); }}
    main {{ max-width:1180px; margin:0 auto; padding:28px 24px 40px; }}
    .hero {{ display:flex; justify-content:space-between; gap:24px; align-items:flex-end; padding:22px 0 18px; border-bottom:1px solid var(--line); }}
    h1 {{ margin:0; font-size:28px; letter-spacing:0; }}
    h2 {{ margin:0 0 12px; font-size:18px; }}
    h3 {{ margin:0; font-size:16px; }}
    p {{ margin:6px 0 0; color:var(--muted); }}
    .badge {{ border:1px solid var(--line); border-radius:999px; padding:8px 12px; background:#fff; color:var(--muted); white-space:nowrap; }}
    .grid {{ display:grid; grid-template-columns:1.15fr .85fr; gap:16px; margin-top:18px; }}
    .card {{ background:var(--paper); border:1px solid var(--line); border-radius:10px; padding:18px; box-shadow:0 8px 22px rgba(20,34,48,.06); }}
    .overall-number {{ font-size:42px; font-weight:700; line-height:1; margin:6px 0 12px; }}
    .meter {{ width:100%; height:14px; background:#e8eef5; border-radius:999px; overflow:hidden; border:1px solid #dae4ee; }}
    .meter span {{ display:block; height:100%; background:linear-gradient(90deg, var(--accent), #51a3a3, var(--accent2)); border-radius:999px; transition:width .35s ease; }}
    .stage-list {{ display:grid; gap:12px; }}
    .stage {{ border:1px solid #e1e8ef; border-radius:8px; padding:14px; background:#fbfdff; }}
    .stage-head {{ display:flex; align-items:flex-start; justify-content:space-between; gap:14px; margin-bottom:10px; }}
    .stage-head strong {{ font-size:17px; white-space:nowrap; }}
    .percent {{ margin-top:6px; color:var(--muted); font-variant-numeric:tabular-nums; }}
    table {{ border-collapse:collapse; width:100%; background:white; table-layout:fixed; }}
    td {{ border-bottom:1px solid #e5ecf3; padding:9px 10px; text-align:left; vertical-align:top; overflow-wrap:anywhere; }}
    td:first-child {{ width:180px; color:var(--muted); }}
    code {{ font-family:Consolas, monospace; font-size:13px; }}
    @media (max-width:820px) {{ .hero, .grid {{ display:block; }} .badge {{ display:inline-block; margin-top:12px; }} .card {{ margin-top:14px; }} }}
  </style>
</head>
<body>
<main>
  <section class="hero">
    <div>
      <h1>Paper Training Live Progress / 长周期训练实时进度</h1>
      <p>页面每 30 秒自动刷新。详细事件写入 <code>experiment_progress.jsonl</code> 和 <code>experiment_progress.csv</code>。</p>
    </div>
    <div class="badge">tqdm terminal bars + static live dashboard</div>
  </section>
  <section class="grid">
    <div class="card">
      <h2>Overall / 总体</h2>
      <div class="overall-number">{overall_percent:.1f}%</div>
      <div class="meter"><span style="width: {overall_percent:.2f}%"></span></div>
      <p>{int(done.get("overall", 0))} / {int(totals.get("overall", 0))} runs finished</p>
    </div>
    <div class="card">
      <h2>Latest / 最近事件</h2>
      <table><tbody>{latest_rows}</tbody></table>
    </div>
  </section>
  <section class="card" style="margin-top:16px">
    <h2>Stages / 阶段</h2>
    <div class="stage-list">{''.join(rows)}</div>
  </section>
</main>
</body>
</html>
""",
        encoding="utf-8",
    )
    return path


def _emit_progress_summary(
    output_dir: Path,
    state: dict[str, Any],
    *,
    force: bool = False,
) -> None:
    now = time.monotonic()
    interval = float(state.get("interval_seconds", 60.0))
    last_print = float(state.get("last_print_monotonic", 0.0))
    _write_live_progress_dashboard(output_dir, state)
    has_tqdm = _update_tqdm_bars(state)
    if not force and now - last_print < interval:
        return
    state["last_print_monotonic"] = now
    totals = state.get("totals", {})
    done = state.get("done", {})
    latest = state.get("latest", {})
    summary_line = (
        "[paper_training:summary] "
        f"overall {done.get('overall', 0)}/{totals.get('overall', 0)} "
        f"{_progress_bar(int(done.get('overall', 0)), int(totals.get('overall', 0)))} "
        f"| baseline {done.get('baseline', 0)}/{totals.get('baseline', 0)} "
        f"| train {done.get('train', 0)}/{totals.get('train', 0)} "
        f"| eval {done.get('eval', 0)}/{totals.get('eval', 0)} "
        f"| latest_phase={latest.get('phase', '')} "
        f"run={latest.get('run_id', '')} "
        f"reward={latest.get('reward_sum', latest.get('final_episode_reward', ''))} "
        f"cost={latest.get('total_cost', '')} "
        f"violations={latest.get('violations', '')} "
        f"| live=live_progress.html"
    )
    if has_tqdm:
        if bool(state.get("verbose_progress", False)):
            _tqdm_write(summary_line)
    else:
        print(summary_line, flush=True)


def _print_progress(output_dir: Path, event: dict[str, Any], *, print_event: bool = True) -> None:
    phase = str(event.get("phase", "progress"))
    message = str(event.get("message", ""))
    details = " ".join(
        f"{key}={value}"
        for key, value in event.items()
        if key not in {"phase", "message"}
    )
    if print_event:
        _tqdm_write(f"[paper_training:{phase}] {message} {details}".rstrip())
    _append_progress_event(output_dir, event)


BASELINE_CLAIM_FIELDS = (
    "baseline_role",
    "is_ac_validated",
    "is_search_based",
    "is_upper_bound_claim_allowed",
    "reference_scope",
    "execution_has_ac_dispatch_shield",
    "feasible_candidate_count",
    "search_budget",
    "best_feasible_cost",
    "fallback_to_current_dispatch_step_count",
    "all_steps_have_ac_feasible_candidate",
)


def _baseline_claim_metadata(algorithm: str, results: dict[str, pd.DataFrame] | None = None) -> dict[str, Any]:
    if algorithm == "rule_based":
        metadata: dict[str, Any] = {
            "baseline_role": "rule_based_dispatch_policy",
            "is_ac_validated": False,
            "is_search_based": False,
            "is_upper_bound_claim_allowed": False,
            "reference_scope": "deterministic_rule_policy_with_post_ac_metrics",
            "execution_has_ac_dispatch_shield": True,
            "feasible_candidate_count": 0,
            "search_budget": 0,
            "best_feasible_cost": 0.0,
            "fallback_to_current_dispatch_step_count": 0,
            "all_steps_have_ac_feasible_candidate": False,
        }
    elif algorithm == "no_flex":
        metadata = {
            "baseline_role": "no_flex_hold_current_power",
            "is_ac_validated": False,
            "is_search_based": False,
            "is_upper_bound_claim_allowed": False,
            "reference_scope": "hold_current_dispatch_with_post_ac_metrics",
            "execution_has_ac_dispatch_shield": True,
            "feasible_candidate_count": 0,
            "search_budget": 0,
            "best_feasible_cost": 0.0,
            "fallback_to_current_dispatch_step_count": 0,
            "all_steps_have_ac_feasible_candidate": False,
        }
    elif algorithm == "ac_validated_search_reference":
        metadata = {
            "baseline_role": "ac_validated_best_found_dispatch_reference",
            "is_ac_validated": True,
            "is_search_based": True,
            "is_upper_bound_claim_allowed": False,
            "reference_scope": "bounded_candidate_search_not_exhaustive_opf",
            "execution_has_ac_dispatch_shield": True,
            "feasible_candidate_count": 0,
            "search_budget": 0,
            "best_feasible_cost": 0.0,
            "fallback_to_current_dispatch_step_count": 0,
            "all_steps_have_ac_feasible_candidate": False,
        }
        rows = pd.DataFrame() if results is None else results.get("ac_validated_search_metadata", pd.DataFrame())
        if not rows.empty:
            feasible_series = rows.get("feasible_candidate_count", pd.Series(dtype=float)).fillna(0)
            fallback_series = rows.get("fallback_to_current_dispatch", pd.Series(dtype=bool)).fillna(False)
            fallback_count = int(_as_bool_float(fallback_series).sum())
            metadata.update(
                {
                    "is_ac_validated": bool(fallback_count == 0 and (feasible_series > 0).all()),
                    "reference_scope": "bounded_candidate_search_ac_feasible_every_step_not_exhaustive_opf"
                    if fallback_count == 0
                    else "bounded_search_with_current_dispatch_fallback_steps",
                    "feasible_candidate_count": int(feasible_series.sum()),
                    "search_budget": int(rows.get("search_budget", pd.Series(dtype=float)).fillna(0).sum()),
                    "best_feasible_cost": float(rows.get("best_feasible_cost", pd.Series(dtype=float)).fillna(0.0).sum()),
                    "fallback_to_current_dispatch_step_count": fallback_count,
                    "all_steps_have_ac_feasible_candidate": bool(fallback_count == 0 and (feasible_series > 0).all()),
                }
            )
    else:
        metadata = {
            "baseline_role": "legacy_static_fr_price_extreme_proxy"
            if algorithm == "opf_oracle_proxy"
            else "static_fr_price_extreme_proxy",
            "is_ac_validated": False,
            "is_search_based": False,
            "is_upper_bound_claim_allowed": False,
            "reference_scope": "static_feasible_region_price_extreme_not_opf",
            "execution_has_ac_dispatch_shield": True,
            "feasible_candidate_count": 0,
            "search_budget": 0,
            "best_feasible_cost": 0.0,
            "fallback_to_current_dispatch_step_count": 0,
            "all_steps_have_ac_feasible_candidate": False,
        }
    return metadata


def _baseline_target(vpp: Any, algorithm: str, step: int, price: float) -> tuple[float, str, str]:
    if algorithm == "no_flex":
        return (
            float(vpp.current_power_mw()),
            "no_flex_hold_current_power",
            "hold_current_power_no_learned_der_action",
        )
    fr = compute_static_feasible_region(vpp, step)
    bounds = fr.aggregate_bounds()
    p_min = float(bounds.p_min_mw)
    p_max = float(bounds.p_max_mw)
    if price >= 105.0:
        target = p_max
        intent = "full_information_high_price_export_or_discharge"
    elif price <= 55.0:
        target = p_min
        intent = "full_information_low_price_absorb_or_charge"
    else:
        target = 0.5 * (p_min + p_max)
        intent = "full_information_mid_price_neutral_midpoint"
    source = "legacy_opf_oracle_proxy_alias_not_opf" if algorithm == "opf_oracle_proxy" else "static_fr_price_extreme_proxy"
    return target, source, intent


def _run_baseline_rollout(
    *,
    algorithm: str,
    config_path: Path,
    output_dir: Path,
    seed: int,
    variant: str,
    split: str,
    scenario_name: str,
    horizon_steps: int,
    experiment_level: str,
    reuse_existing: bool = False,
    ac_reference_max_candidates: int = 16,
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
    progress_step_interval: int = 24,
) -> dict[str, Any]:
    scenario = load_scenario(config_path)
    simulator = Simulator(scenario)
    existing_results_dir = output_dir / "simulator_results"
    step_interval = max(1, int(progress_step_interval))

    def maybe_report_step(step_index: int) -> None:
        completed = int(step_index) + 1
        if progress_callback is None:
            return
        if completed < int(horizon_steps) and completed % step_interval != 0:
            return
        progress_callback(
            {
                "phase": "baseline_step",
                "message": "baseline step progress",
                "step": completed,
                "horizon_steps": int(horizon_steps),
                "step_progress_pct": float(completed / max(1, int(horizon_steps))),
            }
        )

    if reuse_existing and (existing_results_dir / "summary.json").exists():
        results = {
            path.stem: pd.read_csv(path, low_memory=False)
            for path in existing_results_dir.glob("*.csv")
        }
    elif algorithm == "rule_based":
        simulator.reset()
        for step in range(horizon_steps):
            simulator.step(step)
            maybe_report_step(step)
        results = simulator.collect_results()
    else:
        simulator.reset()
        ac_search_rows: list[dict[str, Any]] = []
        if algorithm == "ac_validated_search_reference":

            def ac_reference_action_factory(**kwargs: Any) -> dict[str, dict[str, Any]]:
                current_step = int(kwargs["step"])
                current_price = float(kwargs["price"])
                reference = build_ac_validated_search_actions(
                    scenario,
                    current_step,
                    current_price,
                    max_candidates=int(ac_reference_max_candidates),
                )
                ac_search_rows.append({"step": current_step, **reference.metadata})
                return reference.actions

        for step in range(horizon_steps):
            price = float(scenario.price_profile[step % len(scenario.price_profile)])
            if algorithm == "ac_validated_search_reference":
                actions = ac_reference_action_factory
            else:
                actions = {}
                for vpp in scenario.vpps:
                    target, source, mode = _baseline_target(vpp, algorithm, step, price)
                    actions[vpp.id] = {
                        "selected_p_mw": target,
                        "command_source": source,
                        "action_mode": mode,
                    }
            simulator.step(actions=actions)
            maybe_report_step(step)
        results = simulator.collect_results()
        if ac_search_rows:
            results["ac_validated_search_metadata"] = pd.DataFrame(ac_search_rows)

    if not reuse_existing or not (existing_results_dir / "summary.json").exists():
        simulator.export_results(output_dir / "simulator_results")
        if "ac_validated_search_metadata" in results:
            results["ac_validated_search_metadata"].to_csv(
                output_dir / "simulator_results" / "ac_validated_search_metadata.csv",
                index=False,
            )
        _build_step_summary(results).to_csv(output_dir / "step_summary.csv", index=False)
    plan = BenchmarkRunPlan(
        algorithm=algorithm,
        split=split,
        config_path=str(config_path),
        profile_variant=variant,
        scenario_name=scenario_name,
    )
    metrics = _rollout_metrics(
        seed=int(seed),
        plan=plan,
        scenario=scenario,
        results=results,
        horizon_steps=horizon_steps,
        experiment_level=experiment_level,
    )
    metrics.update(_baseline_claim_metadata(algorithm, results))
    metrics.update({"run_id": output_dir.name, "hparam_case": "baseline"})
    return {"metrics": metrics, "results": results, "output_dir": output_dir}


def _case_overrides(case: str) -> dict[str, float | int]:
    if case == "lower_lr":
        return {"learning_rate_multiplier": 0.33}
    if case == "higher_entropy":
        return {
            "entropy_coef": 0.03,
            "exploration_noise": 0.22,
            "hasac_target_entropy_multiplier": 1.50,
            "hasac_init_log_alpha": -3.0,
        }
    if case == "larger_network":
        return {"hidden_dim_multiplier": 2}
    return {"learning_rate_multiplier": 1.0}


def _is_paper_long_family(cfg: PaperTrainingExperimentConfig) -> bool:
    return str(cfg.preset).replace("-", "_").startswith("paper_long")


def _validate_trainable_cuda_requirement(
    cfg: PaperTrainingExperimentConfig,
    *,
    algorithm: str,
    cuda_available: bool | None = None,
) -> None:
    if not bool(cfg.require_cuda_for_trainable):
        return
    if cuda_available is None:
        try:
            import torch

            cuda_available = bool(torch.cuda.is_available())
            if cuda_available:
                try:
                    torch.empty(1, device="cuda").cpu()
                except Exception:
                    cuda_available = False
        except Exception:
            cuda_available = False
    if not bool(cuda_available):
        raise RuntimeError(
            "CUDA is required for this paper-long trainable experiment but PyTorch cannot use a CUDA device. "
            f"algorithm={algorithm}, preset={cfg.preset}, output_dir={cfg.output_dir}. "
            "Fix the NVIDIA driver/device-node permissions or disable require_cuda_for_trainable only for "
            "explicit CPU debugging runs."
        )


def _algorithm_label(algorithm: str) -> str:
    return {
        "happo": "happo_sequential_ctde",
        "hatrpo": "hatrpo_trust_region_ctde",
        "matd3": "matd3_continuous_dispatch",
        "hasac": "hasac_continuous_dispatch",
        "opf_oracle_proxy": "opf_oracle_proxy",
    }.get(algorithm, algorithm)


def _algorithm_file_prefix(algorithm: str) -> str:
    if algorithm in {"happo", "happo_sequential_ctde"}:
        return "happo"
    if algorithm in {"hatrpo", "hatrpo_trust_region_ctde"}:
        return "hatrpo"
    if algorithm in {"matd3", "matd3_continuous_dispatch"}:
        return "matd3"
    if algorithm in {"hasac", "hasac_continuous_dispatch"}:
        return "hasac"
    raise ValueError(f"Unsupported trainable algorithm: {algorithm}")


def _load_completed_training(algorithm: str, run_dir: Path) -> dict[str, Any] | None:
    prefix = _algorithm_file_prefix(algorithm)
    train_dir = run_dir / "train"
    checkpoint = train_dir / f"{prefix}_checkpoint.pt"
    episode_path = train_dir / f"{prefix}_episode_metrics.csv"
    update_path = train_dir / f"{prefix}_update_metrics.csv"
    summary_path = train_dir / f"{prefix}_training_summary.json"
    if not checkpoint.exists() or not episode_path.exists() or not summary_path.exists():
        return None
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    selected_checkpoint = Path(str(summary.get("checkpoint", checkpoint)))
    if not selected_checkpoint.exists():
        selected_checkpoint = checkpoint
    return {
        "checkpoint": selected_checkpoint,
        "final_checkpoint": checkpoint,
        "best_checkpoint": Path(str(summary.get("best_checkpoint", selected_checkpoint))),
        "episode_metrics": pd.read_csv(episode_path),
        "update_metrics": pd.read_csv(update_path) if update_path.exists() else pd.DataFrame(),
        "summary": summary,
        "output_dir": train_dir,
    }


def _load_completed_eval(algorithm: str, eval_dir: Path) -> dict[str, Any] | None:
    prefix = _algorithm_file_prefix(algorithm)
    step_path = eval_dir / f"{prefix}_frozen_eval_step_metrics.csv"
    summary_path = eval_dir / f"{prefix}_frozen_eval_summary.json"
    if not step_path.exists() or not summary_path.exists():
        return None
    return {
        "summary": json.loads(summary_path.read_text(encoding="utf-8")),
        "step_metrics": pd.read_csv(step_path),
        "output_dir": eval_dir,
    }


def _train_algorithm(
    *,
    algorithm: str,
    cfg: PaperTrainingExperimentConfig,
    train_config_path: Path,
    eval_config_path: Path,
    run_dir: Path,
    seed: int,
    case: str,
    eval_horizon_steps: int,
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
    progress_step_interval: int = 24,
) -> dict[str, Any]:
    _validate_trainable_cuda_requirement(cfg, algorithm=algorithm)
    overrides = _case_overrides(case)
    hidden_dim = int(cfg.hidden_dim * int(overrides.get("hidden_dim_multiplier", 1)))
    lr = float(cfg.learning_rate) * float(overrides.get("learning_rate_multiplier", 1.0))
    train_out = ensure_dir(run_dir / "train")
    eval_out = run_dir / "frozen_eval"
    run_initial_eval = str(cfg.checkpoint_selection) != "both"
    if algorithm == "happo":
        from vpp_dso_sim.learning.advanced_marl import (
            HAPPOConfig,
            _happo_config_from_yaml,
            evaluate_happo_checkpoint,
            train_happo,
        )

        happo_config = (
            _happo_config_from_yaml(train_config_path)
            if bool(cfg.happo_use_yaml_trainer_settings)
            else HAPPOConfig()
        )
        happo_config = replace(
            happo_config,
            episodes=int(cfg.train_episodes),
            horizon_steps=int(cfg.horizon_steps),
            gamma=float(cfg.gamma),
            hidden_dim=hidden_dim,
            actor_learning_rate=lr,
            critic_learning_rate=lr,
            ppo_epochs=int(cfg.ppo_epochs),
            entropy_coef=float(overrides.get("entropy_coef", happo_config.entropy_coef)),
            critic_use_action_summary=bool(
                cfg.happo_critic_use_action_summary or happo_config.critic_use_action_summary
            ),
            dispatch_actor_encoder_type=str(
                overrides.get(
                    "dispatch_actor_encoder_type",
                    happo_config.dispatch_actor_encoder_type
                    if happo_config.dispatch_actor_encoder_type != "deepset_v1"
                    else cfg.dispatch_actor_encoder_type,
                )
            ),
            shared_rollout_enabled=bool(cfg.happo_shared_rollout_enabled),
            shared_rollout_workers=int(cfg.happo_shared_rollout_workers),
            shared_rollout_backend=str(cfg.happo_shared_rollout_backend),
            rollout_fragment_steps=(
                None
                if cfg.happo_rollout_fragment_steps is None
                else int(cfg.happo_rollout_fragment_steps)
            ),
            reward_dynamic_reports=bool(cfg.happo_reward_dynamic_reports),
            reward_dynamic_report_every_episodes=int(cfg.happo_reward_dynamic_report_every_episodes),
            reward_dynamic_report_all_workers=bool(cfg.happo_reward_dynamic_report_all_workers),
            seed=int(seed),
        )
        train = train_happo(
            config_path=train_config_path,
            output_dir=train_out,
            config=happo_config,
            progress_callback=progress_callback,
            progress_step_interval=progress_step_interval,
        )
        eval_result = None
        if run_initial_eval:
            eval_checkpoint = Path(train.get("final_checkpoint", train["checkpoint"])) if cfg.checkpoint_selection == "final" else Path(train["checkpoint"])
            eval_result = evaluate_happo_checkpoint(
                config_path=eval_config_path,
                checkpoint_path=eval_checkpoint,
                output_dir=ensure_dir(eval_out),
                horizon_steps=eval_horizon_steps,
                seed=int(seed) + 10_000,
            )
    elif algorithm == "hatrpo":
        from vpp_dso_sim.learning.hatrpo import HATRPOConfig, evaluate_hatrpo_checkpoint, train_hatrpo

        train = train_hatrpo(
            config_path=train_config_path,
            output_dir=train_out,
            config=HATRPOConfig(
                episodes=int(cfg.train_episodes),
                horizon_steps=int(cfg.horizon_steps),
                gamma=float(cfg.gamma),
                hidden_dim=hidden_dim,
                value_learning_rate=lr,
                entropy_coef=float(overrides.get("entropy_coef", 0.0)),
                dispatch_actor_encoder_type=str(
                    overrides.get("dispatch_actor_encoder_type", cfg.dispatch_actor_encoder_type)
                ),
                seed=int(seed),
            ),
        )
        eval_result = None
        if run_initial_eval:
            eval_checkpoint = Path(train.get("final_checkpoint", train["checkpoint"])) if cfg.checkpoint_selection == "final" else Path(train["checkpoint"])
            eval_result = evaluate_hatrpo_checkpoint(
                config_path=eval_config_path,
                checkpoint_path=eval_checkpoint,
                output_dir=ensure_dir(eval_out),
                horizon_steps=eval_horizon_steps,
                seed=int(seed) + 10_000,
            )
    elif algorithm == "matd3":
        from vpp_dso_sim.learning.matd3 import MATD3Config, evaluate_matd3_checkpoint, train_matd3

        train = train_matd3(
            config_path=train_config_path,
            output_dir=train_out,
            config=MATD3Config(
                episodes=int(cfg.train_episodes),
                horizon_steps=int(cfg.horizon_steps),
                gamma=float(cfg.gamma),
                hidden_dim=hidden_dim,
                actor_learning_rate=lr,
                critic_learning_rate=lr,
                batch_size=int(cfg.batch_size),
                replay_capacity=int(cfg.replay_capacity),
                warmup_steps=int(cfg.warmup_steps),
                exploration_noise=float(overrides.get("exploration_noise", 0.15)),
                dispatch_actor_encoder_type=str(
                    overrides.get("dispatch_actor_encoder_type", cfg.dispatch_actor_encoder_type)
                ),
                seed=int(seed),
            ),
        )
        eval_result = None
        if run_initial_eval:
            eval_checkpoint = Path(train.get("final_checkpoint", train["checkpoint"])) if cfg.checkpoint_selection == "final" else Path(train["checkpoint"])
            eval_result = evaluate_matd3_checkpoint(
                config_path=eval_config_path,
                checkpoint_path=eval_checkpoint,
                output_dir=ensure_dir(eval_out),
                horizon_steps=eval_horizon_steps,
                seed=int(seed) + 10_000,
            )
    elif algorithm == "hasac":
        from vpp_dso_sim.learning.advanced_marl import (
            HASACConfig,
            evaluate_hasac_checkpoint,
            train_hasac,
        )

        train = train_hasac(
            config_path=train_config_path,
            output_dir=train_out,
            config=HASACConfig(
                episodes=int(cfg.train_episodes),
                horizon_steps=int(cfg.horizon_steps),
                gamma=float(cfg.gamma),
                hidden_dim=hidden_dim,
                actor_learning_rate=lr,
                critic_learning_rate=lr,
                alpha_learning_rate=lr,
                batch_size=int(cfg.batch_size),
                replay_capacity=int(cfg.replay_capacity),
                warmup_steps=int(cfg.warmup_steps),
                target_entropy_multiplier=float(overrides.get("hasac_target_entropy_multiplier", 1.0)),
                init_log_alpha_dso=float(overrides.get("hasac_init_log_alpha", -4.0)),
                init_log_alpha_dispatch=float(overrides.get("hasac_init_log_alpha", -4.0)),
                dispatch_actor_encoder_type=str(
                    overrides.get("dispatch_actor_encoder_type", cfg.dispatch_actor_encoder_type)
                ),
                seed=int(seed),
            ),
        )
        eval_result = None
        if run_initial_eval:
            eval_checkpoint = Path(train.get("final_checkpoint", train["checkpoint"])) if cfg.checkpoint_selection == "final" else Path(train["checkpoint"])
            eval_result = evaluate_hasac_checkpoint(
                config_path=eval_config_path,
                checkpoint_path=eval_checkpoint,
                output_dir=ensure_dir(eval_out),
                horizon_steps=eval_horizon_steps,
                seed=int(seed) + 10_000,
            )
    else:
        raise ValueError(f"Unsupported trainable algorithm: {algorithm}")
    return {"train": train, "eval": eval_result, "hidden_dim": hidden_dim, "learning_rate": lr}


def _evaluate_algorithm_checkpoint(
    *,
    algorithm: str,
    eval_config_path: Path,
    checkpoint_path: Path,
    eval_output_dir: Path,
    eval_horizon_steps: int,
    seed: int,
) -> dict[str, Any]:
    if algorithm == "happo":
        from vpp_dso_sim.learning.advanced_marl import evaluate_happo_checkpoint

        return evaluate_happo_checkpoint(
            config_path=eval_config_path,
            checkpoint_path=checkpoint_path,
            output_dir=eval_output_dir,
            horizon_steps=eval_horizon_steps,
            seed=int(seed) + 10_000,
        )
    if algorithm == "hatrpo":
        from vpp_dso_sim.learning.hatrpo import evaluate_hatrpo_checkpoint

        return evaluate_hatrpo_checkpoint(
            config_path=eval_config_path,
            checkpoint_path=checkpoint_path,
            output_dir=eval_output_dir,
            horizon_steps=eval_horizon_steps,
            seed=int(seed) + 10_000,
        )
    if algorithm == "matd3":
        from vpp_dso_sim.learning.matd3 import evaluate_matd3_checkpoint

        return evaluate_matd3_checkpoint(
            config_path=eval_config_path,
            checkpoint_path=checkpoint_path,
            output_dir=eval_output_dir,
            horizon_steps=eval_horizon_steps,
            seed=int(seed) + 10_000,
        )
    if algorithm == "hasac":
        from vpp_dso_sim.learning.advanced_marl import evaluate_hasac_checkpoint

        return evaluate_hasac_checkpoint(
            config_path=eval_config_path,
            checkpoint_path=checkpoint_path,
            output_dir=eval_output_dir,
            horizon_steps=eval_horizon_steps,
            seed=int(seed) + 10_000,
        )
    raise ValueError(f"Unsupported trainable algorithm: {algorithm}")


def _step_metric_summary(
    *,
    run_id: str,
    algorithm: str,
    seed: int,
    split: str,
    profile_variant: str,
    hparam_case: str,
    profile_seed: int | None = None,
    profile_config_path: Path | None = None,
    profile_hash: str | None = None,
    checkpoint_path: Path | None = None,
    checkpoint_label: str = "",
    checkpoint_selection: str = "",
    step_metrics: pd.DataFrame,
) -> dict[str, Any]:
    profile_fields = {
        "profile_seed": None if profile_seed is None else int(profile_seed),
        "profile_config_path": "" if profile_config_path is None else str(profile_config_path),
        "profile_hash": "" if profile_hash is None else str(profile_hash),
        "checkpoint_path": "" if checkpoint_path is None else str(checkpoint_path),
        "checkpoint_label": str(checkpoint_label),
        "checkpoint_selection": str(checkpoint_selection),
    }
    if step_metrics.empty:
        return {
            "run_id": run_id,
            "algorithm": algorithm,
            "seed": int(seed),
            "split": split,
            "profile_variant": profile_variant,
            "hparam_case": hparam_case,
            "eval_total_reward": 0.0,
            "dso_reward_sum": 0.0,
            "dispatch_reward_sum": 0.0,
            "portfolio_reward_sum": 0.0,
            "total_agent_reward_sum": 0.0,
            "raw_objective_reward_sum": 0.0,
            "eval_total_cost": 0.0,
            "total_violation_cells": 0,
            "post_ac_violation_count": 0,
            "post_ac_voltage_violation_count": 0,
            "post_ac_line_overload_count": 0,
            "post_ac_trafo_overload_count": 0,
            "post_ac_powerflow_failed": 0,
            "post_ac_violation_magnitude": 0.0,
            "post_ac_security_pass_rate": 0.0,
            "post_ac_powerflow_converged_rate": 0.0,
            "ac_certified_projection_gap_mw": 0.0,
            "mean_ac_certified_projection_gap_mw": 0.0,
            "ac_certificate_failed_count": 0,
            "ac_certificate_backoff_count": 0,
            "ac_certificate_trace_rows": 0,
            "ac_certificate_safe_rate": 0.0,
            "accepted_candidate_ac_safe_rate": 0.0,
            "repaired_by_ac_powerflow_backoff_rate": 0.0,
            "repaired_by_ac_powerflow_emergency_recovery_rate": 0.0,
            "rolled_back_to_current_safe_dispatch_rate": 0.0,
            "certificate_failed_current_dispatch_insecure_rate": 0.0,
            "certificate_failed_no_ac_safe_recovery_rate": 0.0,
            "mean_ac_certificate_accepted_alpha": 0.0,
            **profile_fields,
        }
    def col_sum(*names: str) -> float:
        for name in names:
            if name in step_metrics:
                return float(step_metrics[name].fillna(0.0).sum())
        return 0.0

    def col_mean(*names: str) -> float:
        for name in names:
            if name in step_metrics:
                return float(step_metrics[name].fillna(0.0).mean())
        return 0.0

    total_cost = col_sum("total_cost")
    total_reward = col_sum("reward")
    raw_objective = col_sum("raw_objective_reward")
    if abs(raw_objective) < 1e-12 and total_cost:
        raw_objective = -total_cost
    return {
        "run_id": run_id,
        "algorithm": algorithm,
        "seed": int(seed),
        "split": split,
        "profile_variant": profile_variant,
        "hparam_case": hparam_case,
        "horizon_steps": int(len(step_metrics)),
        "eval_total_reward": total_reward,
        "eval_mean_reward": col_mean("reward"),
        "dso_reward_sum": col_sum("dso_reward"),
        "dispatch_reward_sum": col_sum("vpp_dispatch_reward", "mean_dispatch_reward"),
        "portfolio_reward_sum": col_sum("vpp_portfolio_reward", "mean_portfolio_reward"),
        "total_agent_reward_sum": total_reward,
        "raw_objective_reward_sum": raw_objective,
        "eval_total_cost": total_cost,
        "total_violation_cells": int(col_sum("violation_count")),
        "post_ac_violation_count": int(col_sum("post_ac_violation_count")),
        "post_ac_voltage_violation_count": int(col_sum("post_ac_voltage_violation_count")),
        "post_ac_line_overload_count": int(col_sum("post_ac_line_overload_count")),
        "post_ac_trafo_overload_count": int(col_sum("post_ac_trafo_overload_count")),
        "post_ac_powerflow_failed": int(col_sum("post_ac_powerflow_failed")),
        "post_ac_violation_magnitude": col_sum("post_ac_violation_magnitude"),
        "post_ac_security_pass_rate": float(
            (step_metrics["post_ac_violation_count"].fillna(0.0) <= 0.0).mean()
        )
        if "post_ac_violation_count" in step_metrics
        else float(col_sum("violation_count") == 0),
        "post_ac_powerflow_converged_rate": float(
            (step_metrics["post_ac_powerflow_failed"].fillna(0.0) <= 0.0).mean()
        )
        if "post_ac_powerflow_failed" in step_metrics
        else 1.0,
        "ac_certified_projection_gap_mw": col_sum("ac_certified_projection_gap_mw"),
        "mean_ac_certified_projection_gap_mw": col_mean(
            "mean_ac_certified_projection_gap_mw", "ac_certified_projection_gap_mw"
        ),
        "ac_certificate_failed_count": int(col_sum("ac_certificate_failed_count")),
        "ac_certificate_backoff_count": int(col_sum("ac_certificate_backoff_count")),
        "ac_certificate_trace_rows": int(col_sum("ac_certificate_trace_rows")),
        "ac_certificate_safe_rate": col_mean("ac_certificate_safe_rate"),
        "accepted_candidate_ac_safe_rate": col_mean("accepted_candidate_ac_safe_rate"),
        "repaired_by_ac_powerflow_backoff_rate": col_mean("repaired_by_ac_powerflow_backoff_rate"),
        "repaired_by_ac_powerflow_emergency_recovery_rate": col_mean(
            "repaired_by_ac_powerflow_emergency_recovery_rate"
        ),
        "rolled_back_to_current_safe_dispatch_rate": col_mean("rolled_back_to_current_safe_dispatch_rate"),
        "certificate_failed_current_dispatch_insecure_rate": col_mean(
            "certificate_failed_current_dispatch_insecure_rate"
        ),
        "certificate_failed_no_ac_safe_recovery_rate": col_mean("certificate_failed_no_ac_safe_recovery_rate"),
        "mean_ac_certificate_accepted_alpha": col_mean("mean_ac_certificate_accepted_alpha"),
        "security_pass": int(col_sum("violation_count") == 0),
        **profile_fields,
    }


def _baseline_eval_summary_row(
    *,
    run_id: str,
    algorithm: str,
    seed: int,
    profile_variant: str,
    horizon_steps: int,
    profile_seed: int,
    profile_config_path: Path,
    profile_hash: str,
    metrics: dict[str, Any],
    results: dict[str, pd.DataFrame],
) -> dict[str, Any]:
    reward = results.get("reward_components", pd.DataFrame())

    def reward_sum(column: str) -> float:
        if reward.empty or column not in reward:
            return 0.0
        return float(reward[column].fillna(0.0).sum())

    total_cost = float(metrics.get("total_cost", reward_sum("total_cost")) or 0.0)
    dso_reward = float(metrics.get("reward_sum", reward_sum("reward")) or 0.0)
    raw_objective = reward_sum("raw_objective_reward")
    if abs(raw_objective) < 1e-12 and total_cost:
        raw_objective = -total_cost
    violation_count = int(metrics.get("total_violation_cells", reward_sum("post_ac_violation_count")) or 0)
    projection = results.get("projection_trace", pd.DataFrame())
    certificate_summary = _certificate_summary_from_projection(projection)
    post_ac_summary = _post_ac_summary_from_reward(reward)
    return {
        "run_id": run_id,
        "algorithm": algorithm,
        "seed": int(seed),
        "split": "eval_profile",
        "profile_variant": profile_variant,
        "hparam_case": "baseline",
        "horizon_steps": int(horizon_steps),
        "eval_total_reward": dso_reward,
        "eval_mean_reward": dso_reward / max(1, int(horizon_steps)),
        "dso_reward_sum": dso_reward,
        "dispatch_reward_sum": 0.0,
        "portfolio_reward_sum": 0.0,
        "total_agent_reward_sum": dso_reward,
        "raw_objective_reward_sum": raw_objective,
        "eval_total_cost": total_cost,
        "total_violation_cells": violation_count,
        "post_ac_violation_count": int(reward_sum("post_ac_violation_count")),
        "post_ac_voltage_violation_count": int(reward_sum("post_ac_voltage_violation_count")),
        "post_ac_line_overload_count": int(reward_sum("post_ac_line_overload_count")),
        "post_ac_trafo_overload_count": int(reward_sum("post_ac_trafo_overload_count")),
        "post_ac_powerflow_failed": int(reward_sum("post_ac_powerflow_failed")),
        "post_ac_violation_magnitude": reward_sum("post_ac_violation_magnitude"),
        "post_ac_security_pass_rate": post_ac_summary["post_ac_security_pass_rate"],
        "post_ac_powerflow_converged_rate": post_ac_summary["post_ac_powerflow_converged_rate"],
        "ac_certified_projection_gap_mw": reward_sum("ac_certified_projection_gap_mw"),
        "mean_ac_certified_projection_gap_mw": certificate_summary["mean_ac_certified_projection_gap_mw"],
        "ac_certificate_failed_count": int(reward_sum("ac_certificate_failed_count")),
        **certificate_summary,
        "security_pass": int(metrics.get("security_pass", violation_count == 0)),
        "profile_seed": int(profile_seed),
        "profile_config_path": str(profile_config_path),
        "profile_hash": str(profile_hash),
        "checkpoint_path": "",
        "checkpoint_label": "baseline",
        "checkpoint_selection": "baseline",
        **{field: metrics.get(field, _baseline_claim_metadata(algorithm).get(field)) for field in BASELINE_CLAIM_FIELDS},
    }


def _write_tensorboard_scalars(
    *,
    output_dir: Path,
    run_id: str,
    episode_metrics: pd.DataFrame,
    update_metrics: pd.DataFrame,
    eval_step_metrics: pd.DataFrame,
    write_train: bool = True,
    write_eval: bool = True,
) -> Path | None:
    try:
        from torch.utils.tensorboard import SummaryWriter
    except Exception:
        return None

    log_dir = ensure_dir(output_dir / "tensorboard" / run_id)
    writer = SummaryWriter(log_dir=str(log_dir))
    if write_train:
        for _, row in episode_metrics.iterrows():
            episode = int(row.get("episode", 0))
            for col in ("episode_reward", "dso_episode_reward", "vpp_dispatch_episode_reward", "episode_cost", "violation_count", "projection_gap_mw"):
                if col in row and pd.notna(row[col]):
                    writer.add_scalar(f"train/{col}", float(row[col]), episode)
        for _, row in update_metrics.iterrows():
            step = int(row.get("global_step", row.get("critic_update", 0)))
            for col in ("critic_loss", "role_critic_loss", "actor_loss", "alpha_loss", "dso_actor_loss", "dispatch_actor_loss", "actor_grad_norm", "critic_grad_norm"):
                if col in row and pd.notna(row[col]):
                    writer.add_scalar(f"loss/{col}", float(row[col]), step)
    if write_eval:
        for _, row in eval_step_metrics.iterrows():
            step = int(row.get("step", 0))
            for col in ("reward", "dso_reward", "vpp_dispatch_reward", "vpp_portfolio_reward", "total_cost", "violation_count"):
                if col in row and pd.notna(row[col]):
                    writer.add_scalar(f"eval/{col}", float(row[col]), step)
    writer.flush()
    writer.close()
    return log_dir


def _export_training_images(
    *,
    output_dir: Path,
    episode_metrics: pd.DataFrame,
    loss_metrics: pd.DataFrame,
    evaluation_seed_metrics: pd.DataFrame,
) -> pd.DataFrame:
    image_dir = ensure_dir(output_dir / "tensorboard_images")
    rows: list[dict[str, Any]] = []
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return pd.DataFrame(rows)

    def save(fig: Any, name: str, title_en: str, title_zh: str) -> None:
        path = image_dir / name
        fig.tight_layout()
        fig.savefig(path, dpi=160)
        plt.close(fig)
        rows.append(
            {
                "tag": name.replace(".png", ""),
                "image_path": str(path.relative_to(output_dir)),
                "title_en": title_en,
                "title_zh": title_zh,
            }
        )

    if not episode_metrics.empty and "episode_reward" in episode_metrics:
        fig, ax = plt.subplots(figsize=(8.5, 4.2))
        for (algorithm, case), group in episode_metrics.groupby(["algorithm", "hparam_case"], dropna=False):
            curve = group.groupby("episode")["episode_reward"].mean()
            ax.plot(curve.index, curve.values, marker="o", linewidth=1.8, label=f"{algorithm}/{case}")
        ax.set_xlabel("Episode")
        ax.set_ylabel("Reward")
        ax.set_title("Training reward by algorithm")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)
        save(fig, "training_reward_curve.png", "Training Reward Curve", "训练奖励曲线")

    if not loss_metrics.empty:
        loss_col = "critic_loss" if "critic_loss" in loss_metrics else "role_critic_loss"
        if loss_col in loss_metrics:
            fig, ax = plt.subplots(figsize=(8.5, 4.2))
            grouping = ["algorithm", "hparam_case"] if "hparam_case" in loss_metrics else ["algorithm"]
            for group_key, group in loss_metrics.groupby(grouping, dropna=False):
                curve = group.groupby("global_step")[loss_col].mean().tail(300)
                label = "/".join(str(part) for part in group_key) if isinstance(group_key, tuple) else str(group_key)
                ax.plot(curve.index, curve.values, linewidth=1.4, label=label)
            ax.set_xlabel("Global step")
            ax.set_ylabel(loss_col)
            ax.set_title("Critic/value loss trace")
            ax.grid(alpha=0.25)
            ax.legend(fontsize=8)
            save(fig, "loss_trace.png", "Loss Trace", "损失曲线")

    if not evaluation_seed_metrics.empty:
        for metric, name, zh in (
            ("eval_total_reward", "eval_reward_bar.png", "冻结评估总奖励"),
            ("eval_total_cost", "eval_cost_bar.png", "冻结评估总成本"),
            ("total_violation_cells", "eval_violations_bar.png", "冻结评估越限数"),
        ):
            if metric not in evaluation_seed_metrics:
                continue
            fig, ax = plt.subplots(figsize=(8.8, 4.2))
            values = evaluation_seed_metrics.groupby("algorithm")[metric].agg(["mean", "std"]).fillna(0.0)
            ax.bar(values.index.astype(str), values["mean"], yerr=values["std"], capsize=4)
            ax.set_ylabel(metric)
            ax.set_title(metric)
            ax.tick_params(axis="x", labelrotation=25)
            ax.grid(axis="y", alpha=0.25)
            save(fig, name, metric, zh)

    image_index = pd.DataFrame(rows)
    image_index.to_csv(output_dir / "tensorboard_assets.csv", index=False)
    return image_index


def _aggregate_eval_metrics(evaluation_seed_metrics: pd.DataFrame) -> pd.DataFrame:
    if evaluation_seed_metrics.empty:
        return pd.DataFrame()
    group_cols = ["algorithm", "split", "profile_variant", "hparam_case"]
    if "checkpoint_label" in evaluation_seed_metrics.columns:
        group_cols.append("checkpoint_label")
    numeric_cols = [
        col
        for col in evaluation_seed_metrics.select_dtypes(include="number").columns
        if col not in {"seed"}
    ]
    grouped = evaluation_seed_metrics.groupby(group_cols, as_index=False)[numeric_cols].agg(["mean", "std", "min", "max", "count"])
    grouped.columns = [
        col[0] if isinstance(col, tuple) and not col[1] else f"{col[0]}_{col[1]}"
        if isinstance(col, tuple)
        else str(col)
        for col in grouped.columns
    ]
    frame = grouped.reset_index(drop=True)
    for metric in ("eval_total_reward", "eval_total_cost", "total_violation_cells"):
        mean_col = f"{metric}_mean"
        std_col = f"{metric}_std"
        count_col = f"{metric}_count"
        if {mean_col, std_col, count_col}.issubset(frame.columns):
            ci = 1.96 * frame[std_col].fillna(0.0) / np.sqrt(frame[count_col].clip(lower=1))
            frame[f"{metric}_ci95_low"] = frame[mean_col] - ci
            frame[f"{metric}_ci95_high"] = frame[mean_col] + ci
    return frame


def _baseline_comparison(evaluation_seed_metrics: pd.DataFrame) -> pd.DataFrame:
    if evaluation_seed_metrics.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    baseline = evaluation_seed_metrics[evaluation_seed_metrics["algorithm"] == "rule_based"]
    group_cols = ["algorithm"]
    if "checkpoint_label" in evaluation_seed_metrics:
        group_cols.append("checkpoint_label")
    for metric in ("eval_total_reward", "eval_total_cost", "total_violation_cells"):
        if metric not in evaluation_seed_metrics:
            continue
        base_mean = float(baseline[metric].mean()) if not baseline.empty else float("nan")
        for group_key, group in evaluation_seed_metrics.groupby(group_cols, dropna=False):
            key_tuple = group_key if isinstance(group_key, tuple) else (group_key,)
            group_identity = dict(zip(group_cols, key_tuple))
            algorithm = str(group_identity.get("algorithm", ""))
            checkpoint_label = str(group_identity.get("checkpoint_label", ""))
            value_mean = float(group[metric].mean())
            delta = value_mean - base_mean if np.isfinite(base_mean) else float("nan")
            claim_context = {
                field: (
                    group[field].dropna().iloc[0]
                    if field in group.columns and not group[field].dropna().empty
                    else ""
                )
                for field in BASELINE_CLAIM_FIELDS
            }
            if metric in {"eval_total_cost", "total_violation_cells"}:
                win_rate = float((group[metric] < base_mean).mean()) if np.isfinite(base_mean) else None
                direction = "lower_is_better"
            else:
                win_rate = float((group[metric] > base_mean).mean()) if np.isfinite(base_mean) else None
                direction = "higher_is_better"
            rows.append(
                {
                    "algorithm": algorithm,
                    "checkpoint_label": checkpoint_label,
                    "baseline_algorithm": "rule_based",
                    "metric": metric,
                    "direction": direction,
                    "value_mean": value_mean,
                    "baseline_mean": base_mean,
                    "delta_abs": delta,
                    "delta_pct": delta / abs(base_mean) if abs(base_mean) > 1e-9 else float("nan"),
                    "win_rate": win_rate,
                    **claim_context,
                }
            )
    return pd.DataFrame(rows)


def _window_mean(values: pd.Series, start_frac: float, end_frac: float) -> float:
    clean = pd.to_numeric(values, errors="coerce").dropna()
    if clean.empty:
        return 0.0
    n = len(clean)
    start = min(n - 1, max(0, int(np.floor(n * start_frac))))
    end = min(n, max(start + 1, int(np.ceil(n * end_frac))))
    return float(clean.iloc[start:end].mean())


def _convergence_summary(episode_metrics: pd.DataFrame, loss_metrics: pd.DataFrame) -> pd.DataFrame:
    if episode_metrics.empty or "episode_reward" not in episode_metrics:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    group_cols = [col for col in ("run_id", "algorithm", "seed", "profile_variant", "hparam_case") if col in episode_metrics]
    for key, group in episode_metrics.groupby(group_cols, dropna=False, sort=False):
        key_tuple = key if isinstance(key, tuple) else (key,)
        base = dict(zip(group_cols, key_tuple))
        ordered = group.sort_values("episode") if "episode" in group else group
        rewards = pd.to_numeric(ordered["episode_reward"], errors="coerce").dropna()
        if rewards.empty:
            continue
        loss_group = pd.DataFrame()
        if not loss_metrics.empty and "run_id" in loss_metrics and "run_id" in base:
            loss_group = loss_metrics[loss_metrics["run_id"] == base["run_id"]]
        critic_loss_col = "critic_loss" if "critic_loss" in loss_group else "role_critic_loss"
        final_mean = _window_mean(rewards, 0.8, 1.0)
        best_reward = float(rewards.max())
        final_best_gap_abs = float(best_reward - final_mean)
        gap_scale = max(1.0, abs(best_reward), abs(final_mean))
        rows.append(
            {
                **base,
                "episode_count": int(len(rewards)),
                "first_episode_reward": float(rewards.iloc[0]),
                "early_reward_mean": _window_mean(rewards, 0.0, 0.2),
                "middle_reward_mean": _window_mean(rewards, 0.4, 0.6),
                "final_reward_mean": final_mean,
                "best_episode_reward": best_reward,
                "final_best_gap_abs": final_best_gap_abs,
                "final_best_gap_ratio": float(final_best_gap_abs / gap_scale),
                "reward_slope_first_to_final": float(final_mean - _window_mean(rewards, 0.0, 0.2)),
                "late_collapse_flag": bool(final_best_gap_abs / gap_scale > 0.25),
                "mean_violation_count": float(group["violation_count"].fillna(0.0).mean())
                if "violation_count" in group
                else 0.0,
                "mean_shield_intervention_gap_mw": float(group["shield_intervention_gap_mw"].fillna(0.0).mean())
                if "shield_intervention_gap_mw" in group
                else 0.0,
                "final_critic_loss_mean": float(
                    pd.to_numeric(loss_group[critic_loss_col], errors="coerce").dropna().tail(50).mean()
                )
                if not loss_group.empty and critic_loss_col in loss_group
                else 0.0,
            }
        )
    return pd.DataFrame(rows)


def _claim_guardrails(
    *,
    cfg: PaperTrainingExperimentConfig,
    evaluation_seed_metrics: pd.DataFrame,
    diagnostics: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    def add(rule_id: str, status: str, severity: str, blocked_claim: str, allowed_claim: str, reason: str) -> None:
        rows.append(
            {
                "rule_id": rule_id,
                "status": status,
                "severity": severity,
                "blocked_claim": blocked_claim,
                "allowed_claim": allowed_claim,
                "reason": reason,
            }
        )

    ac_rows = pd.DataFrame()
    if not evaluation_seed_metrics.empty and "is_ac_validated" in evaluation_seed_metrics:
        ac_mask = evaluation_seed_metrics["is_ac_validated"].map(lambda value: bool(value) if pd.notna(value) else False)
        ac_rows = evaluation_seed_metrics[ac_mask]
    upper_bound_allowed = False
    if not ac_rows.empty and "is_upper_bound_claim_allowed" in ac_rows:
        upper_bound_allowed = bool(
            ac_rows["is_upper_bound_claim_allowed"].map(lambda value: bool(value) if pd.notna(value) else False).any()
        )
    add(
        "optimal_or_upper_bound_claim",
        "allowed" if upper_bound_allowed else "blocked",
        "high" if not upper_bound_allowed else "info",
        "Do not call bounded baselines oracle, OPF optimum, or upper-bound.",
        "Call ac_validated_search_reference an AC-feasible bounded search reference.",
        "No true exhaustive AC OPF/MILP reference is present." if not upper_bound_allowed else "A reference explicitly permits upper-bound claims.",
    )

    price_is_proxy = str(cfg.data_source).lower() == "smart_ds"
    add(
        "real_market_profit_claim",
        "blocked" if price_is_proxy else "review",
        "medium",
        "Do not claim real market settlement profit when prices are scarcity proxies.",
        "Report proxy economic cost/reward under documented synthetic/scarcity prices.",
        "SMART-DS supplies load/PV shapes; this protocol still derives price signals.",
    )
    add(
        "learned_physical_membership_claim",
        "blocked",
        "medium",
        "Do not claim RL physically moves DER membership between VPPs.",
        "Report slow-loop portfolio recommendations under scenario-gated membership events.",
        "Portfolio membership changes are still executed through scenario events and approval gates.",
    )
    has_security_violations = (
        not evaluation_seed_metrics.empty
        and "total_violation_cells" in evaluation_seed_metrics
        and float(evaluation_seed_metrics["total_violation_cells"].fillna(0.0).max()) > 0.0
    )
    add(
        "guaranteed_safe_operation_claim",
        "blocked" if has_security_violations else "guarded",
        "high" if has_security_violations else "medium",
        "Do not claim guaranteed safe operation from RL alone.",
        "Report post-AC security pass rate and certificate/backoff statistics.",
        "Frozen evaluation includes violations." if has_security_violations else "Safety is enforced by an AC-aware shield and must be reported as shielded execution.",
    )
    severe_diagnostics = (
        diagnostics["severity"].astype(str).str.lower().eq("high").sum()
        if not diagnostics.empty and "severity" in diagnostics
        else 0
    )
    readiness = {
        "run_mode": "experiment",
        "paper_claim_ready": bool(
            not has_security_violations
            and upper_bound_allowed
            and not price_is_proxy
            and int(severe_diagnostics) == 0
        ),
        "execution_ready": True,
        "blocked_claim_count": int(sum(1 for row in rows if row["status"] == "blocked")),
        "high_diagnostic_count": int(severe_diagnostics),
        "summary": (
            "Executable for paper-long convergence observation; not paper-claim-ready until blocked claims are resolved."
        ),
    }
    return pd.DataFrame(rows), readiness


def _guard_output_protocol(output_dir: Path, cfg: PaperTrainingExperimentConfig) -> None:
    manifest_path = output_dir / "experiment_manifest.json"
    previous: dict[str, Any] = {}
    if manifest_path.exists():
        try:
            loaded = json.loads(manifest_path.read_text(encoding="utf-8"))
            previous = loaded if isinstance(loaded, dict) else {}
        except Exception:
            previous = {}
        previous_config = previous.get("config", {}) if isinstance(previous, dict) else {}
        previous_algorithms = set(previous_config.get("algorithms", []))
        legacy_algorithms = {"opf_oracle_proxy"}
        if previous_algorithms.intersection(legacy_algorithms):
            raise RuntimeError(
                f"Output directory {output_dir} contains legacy baseline artifacts {sorted(previous_algorithms.intersection(legacy_algorithms))}. "
                "Use a fresh output directory for paper-long, or archive/remove the old directory before rerunning."
            )
        if bool(cfg.resume_completed) and str(previous.get("schema_version", "")) != "paper_training_v1":
            raise RuntimeError(
                f"Output directory {output_dir} has an incompatible manifest schema. "
                "Disable --resume-completed or use a fresh output directory."
            )
    if _is_paper_long_family(cfg) and not bool(cfg.resume_completed):
        allowed_preflight = {".cache", "logs"}
        existing = [child.name for child in output_dir.iterdir() if child.name not in allowed_preflight]
        if existing:
            preview = ", ".join(sorted(existing)[:8])
            raise RuntimeError(
                f"Paper-long output directory {output_dir} is not empty ({preview}). "
                "Use a fresh output directory, or pass --resume-completed only for a deliberate compatible resume."
            )


def _architecture_diagnostics(
    *,
    cfg: PaperTrainingExperimentConfig,
    evaluation_seed_metrics: pd.DataFrame,
    run_index: pd.DataFrame,
) -> pd.DataFrame:
    rows = [
        {
            "severity": "medium",
            "check_id": "baseline_claim_boundary",
            "component": "baseline",
            "finding": (
                "static_fr_price_extreme_proxy and the legacy opf_oracle_proxy alias are static FR heuristics, "
                "not AC OPF, MILP market clearing, or an upper bound. ac_validated_search_reference is only a "
                "bounded AC power-flow screened reference."
            ),
            "finding_zh": (
                "static_fr_price_extreme_proxy 以及旧 opf_oracle_proxy 别名只是静态 FR 启发式，"
                "不是 AC OPF、MILP 市场出清或性能上界。ac_validated_search_reference 也只是有预算的 AC 潮流校验搜索参考。"
            ),
            "recommendation": "Keep upper-bound or optimality claims blocked until a true AC OPF/MILP/reference solver is added.",
            "status": "guarded",
        },
        {
            "severity": "medium",
            "check_id": "smart_ds_price_proxy",
            "component": "dataset",
            "finding": "SMART-DS supplies load/PV shapes; price remains a derived scarcity proxy.",
            "finding_zh": "SMART-DS 提供负荷/PV形状，电价仍是派生稀缺性代理。",
            "recommendation": "Attach tariff/market price data such as OpenEI/CAISO or a documented local tariff.",
            "status": "tracked",
        },
        {
            "severity": "medium",
            "check_id": "portfolio_physical_gate",
            "component": "portfolio_agent",
            "finding": "Slow portfolio actions are evaluated, but physical DER membership changes are still gated by scenario events.",
            "finding_zh": "慢周期组合智能体动作已参与评估，但物理 DER 归属变更仍由场景事件门控。",
            "recommendation": "Add executable membership-change constraints, settlement, and approval workflow.",
            "status": "open",
        },
    ]
    if not evaluation_seed_metrics.empty and "total_violation_cells" in evaluation_seed_metrics:
        worst = int(evaluation_seed_metrics["total_violation_cells"].max())
        if worst > 0:
            rows.append(
                {
                    "severity": "high",
                    "check_id": "security_violations_present",
                    "component": "safety_projection",
                    "finding": f"At least one frozen-eval run had {worst} violation cells.",
                    "finding_zh": f"至少一个冻结评估 run 出现 {worst} 个越限单元。",
                    "recommendation": "Tighten DSO envelopes, add AC-OPF projection, and increase violation penalties only after checking physical feasibility.",
                    "status": "needs_follow_up",
                }
            )
    if evaluation_seed_metrics.empty or "is_ac_validated" not in evaluation_seed_metrics:
        rows.append(
            {
                "severity": "high",
                "check_id": "paper_claim_blocked",
                "component": "baseline",
                "finding": "No AC-validated reference baseline metadata is present in evaluation_seed_metrics.",
                "finding_zh": "evaluation_seed_metrics 中没有 AC 校验参考基线元数据。",
                "recommendation": "Include ac_validated_search_reference before paper-long comparisons that discuss AC feasibility.",
                "status": "blocked",
            }
        )
    else:
        is_ac_validated = evaluation_seed_metrics["is_ac_validated"].map(
            lambda value: bool(value) if pd.notna(value) else False
        )
        if not bool(is_ac_validated.any()):
            rows.append(
                {
                    "severity": "high",
                    "check_id": "paper_claim_blocked",
                    "component": "baseline",
                    "finding": "No baseline row is marked as AC-validated.",
                    "finding_zh": "没有任何 baseline 行被标记为 AC 校验参考。",
                    "recommendation": "Enable ac_validated_search_reference and keep static FR proxies out of safety/optimality claims.",
                    "status": "blocked",
                }
            )
        if "is_upper_bound_claim_allowed" in evaluation_seed_metrics:
            ac_rows = evaluation_seed_metrics[is_ac_validated]
            upper_bound_allowed = ac_rows["is_upper_bound_claim_allowed"].map(
                lambda value: bool(value) if pd.notna(value) else False
            )
            if not ac_rows.empty and not bool(upper_bound_allowed.any()):
                rows.append(
                    {
                        "severity": "medium",
                        "check_id": "ac_reference_not_upper_bound",
                        "component": "baseline",
                        "finding": "AC-validated search reference exists, but it is not exhaustive and cannot support upper-bound claims.",
                        "finding_zh": "已经有 AC 校验搜索参考，但它不是穷举/优化器结果，不能支撑上界声明。",
                        "recommendation": "Use it as a feasibility sanity reference only; add AC OPF/MILP for upper-bound language.",
                        "status": "tracked",
                    }
                )
    reward_alignment_col = (
        "raw_objective_reward_sum"
        if "raw_objective_reward_sum" in evaluation_seed_metrics.columns
        else "eval_total_reward"
    )
    if not evaluation_seed_metrics.empty and {"algorithm", reward_alignment_col, "eval_total_cost"}.issubset(
        evaluation_seed_metrics.columns
    ):
        grouped = evaluation_seed_metrics.groupby("algorithm")[[reward_alignment_col, "eval_total_cost"]].mean()
        if not grouped.empty:
            best_reward_algorithm = str(grouped[reward_alignment_col].idxmax())
            min_cost_algorithm = str(grouped["eval_total_cost"].idxmin())
            best_cost = float(grouped.loc[best_reward_algorithm, "eval_total_cost"])
            min_cost = float(grouped.loc[min_cost_algorithm, "eval_total_cost"])
            if min_cost > 0.0 and best_cost > 1.05 * min_cost:
                rows.append(
                    {
                        "severity": "high",
                        "check_id": "reward_cost_misalignment",
                        "component": "reward_design",
                        "finding": (
                            f"Best reward algorithm {best_reward_algorithm} has mean cost {best_cost:.3f}, "
                            f"while lowest-cost algorithm {min_cost_algorithm} has {min_cost:.3f}."
                        ),
                        "finding_zh": (
                            f"奖励最高的算法 {best_reward_algorithm} 平均成本为 {best_cost:.3f}，"
                            f"但最低成本算法 {min_cost_algorithm} 只有 {min_cost:.3f}。"
                        ),
                        "recommendation": (
                            "Inspect raw objective, DSO reward, VPP profit, and total social cost separately; tune "
                            "reward weights only after confirming the intended economic objective."
                        ),
                        "status": "needs_reward_review",
                    }
                )
            for proxy_name in ("opf_oracle_proxy", "static_fr_price_extreme_proxy"):
                if {proxy_name, "rule_based"}.issubset(grouped.index):
                    reward_gap = abs(
                        float(grouped.loc[proxy_name, reward_alignment_col])
                        - float(grouped.loc["rule_based", reward_alignment_col])
                    )
                    proxy_group = evaluation_seed_metrics[evaluation_seed_metrics["algorithm"] == proxy_name]
                    rule_group = evaluation_seed_metrics[evaluation_seed_metrics["algorithm"] == "rule_based"]
                    proxy_cost = float(proxy_group["eval_total_cost"].mean())
                    rule_cost = float(rule_group["eval_total_cost"].mean())
                    proxy_violations = float(proxy_group.get("total_violation_cells", pd.Series(dtype=float)).mean())
                    rule_violations = float(rule_group.get("total_violation_cells", pd.Series(dtype=float)).mean())
                    if proxy_cost > rule_cost + 1e-9 or proxy_violations > rule_violations + 1e-9:
                        rows.append(
                            {
                                "severity": "high",
                                "check_id": "oracle_proxy_not_upper_bound",
                                "component": "oracle_baseline",
                                "finding": (
                                    f"{proxy_name} is worse than rule_based on mean cost or violations "
                                    f"(cost {proxy_cost:.3f} vs {rule_cost:.3f}; violations {proxy_violations:.3f} vs {rule_violations:.3f})."
                                ),
                                "finding_zh": (
                                    f"{proxy_name} 在平均成本或越限数上劣于 rule_based "
                                    f"(成本 {proxy_cost:.3f} vs {rule_cost:.3f}; 越限 {proxy_violations:.3f} vs {rule_violations:.3f})。"
                                ),
                                "recommendation": "Do not label this proxy as oracle, OPF, or upper bound; report it only as a static FR heuristic.",
                                "status": "blocked",
                            }
                        )
                    if reward_gap < 1e-6:
                        rows.append(
                            {
                                "severity": "medium",
                                "check_id": "oracle_proxy_not_distinct",
                                "component": "oracle_baseline",
                                "finding": f"{proxy_name} is numerically identical to rule_based in this campaign.",
                                "finding_zh": f"{proxy_name} 在本次实验中与 rule_based 数值完全相同。",
                                "recommendation": "Keep it as an auditable static heuristic or replace it with a true AC-validated optimizer.",
                                "status": "open",
                            }
                        )
    if len(cfg.seeds) < 5 and cfg.preset not in {"smoke", "pilot"}:
        rows.append(
            {
                "severity": "medium",
                "check_id": "seed_count_below_paper_protocol",
                "component": "statistics",
                "finding": f"Configured seed count is {len(cfg.seeds)}, below the 5-seed minimum target.",
                "finding_zh": f"当前 seed 数为 {len(cfg.seeds)}，低于论文级 5-seed 目标。",
                "recommendation": "Use the paper_long preset or override --seeds with at least five seeds.",
                "status": "open",
            }
        )
    if bool(cfg.resume_completed):
        rows.append(
            {
                "severity": "medium",
                "check_id": "resume_completed_enabled",
                "component": "experiment_cache",
                "finding": "Existing completed artifacts may be reused from the output directory.",
                "finding_zh": "当前允许复用输出目录中的已完成训练/评估产物。",
                "recommendation": "Use a fresh output directory after code/config changes, or leave --resume-completed off for clean paper-long campaigns.",
                "status": "tracked",
            }
        )
    return pd.DataFrame(rows)


def _baseline_safety_gate_diagnostics(
    *,
    cfg: PaperTrainingExperimentConfig,
    evaluation_seed_metrics: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if "ac_validated_search_reference" not in set(cfg.algorithms):
        return pd.DataFrame(rows)

    def add(
        *,
        severity: str,
        check_id: str,
        finding: str,
        finding_zh: str,
        recommendation: str,
        status: str,
        block_execution: bool,
    ) -> None:
        rows.append(
            {
                "severity": severity,
                "check_id": check_id,
                "component": "baseline_safety_gate",
                "finding": finding,
                "finding_zh": finding_zh,
                "recommendation": recommendation,
                "status": status,
                "block_execution": bool(block_execution),
            }
        )

    if evaluation_seed_metrics.empty or "algorithm" not in evaluation_seed_metrics:
        add(
            severity="high",
            check_id="ac_reference_missing",
            finding="No evaluation rows are available for the AC-validated reference baseline.",
            finding_zh="没有可用于 AC 校验参考基线的评估行。",
            recommendation="Complete baseline rollouts before launching paper-long RL training.",
            status="blocked",
            block_execution=True,
        )
        return pd.DataFrame(rows)

    ac_rows = evaluation_seed_metrics[
        evaluation_seed_metrics["algorithm"].astype(str).eq("ac_validated_search_reference")
    ]
    if ac_rows.empty:
        add(
            severity="high",
            check_id="ac_reference_missing",
            finding="ac_validated_search_reference is configured but no completed baseline rows were found.",
            finding_zh="配置中包含 ac_validated_search_reference，但没有找到已完成的基线结果行。",
            recommendation="Run the AC reference baseline for every evaluation variant and seed before RL training.",
            status="blocked",
            block_execution=True,
        )
        return pd.DataFrame(rows)

    def max_numeric(column: str) -> float:
        if column not in ac_rows:
            return 0.0
        values = pd.to_numeric(ac_rows[column], errors="coerce").fillna(0.0)
        return float(values.max()) if not values.empty else 0.0

    def sum_numeric(column: str) -> float:
        if column not in ac_rows:
            return 0.0
        values = pd.to_numeric(ac_rows[column], errors="coerce").fillna(0.0)
        return float(values.sum()) if not values.empty else 0.0

    violation_total = max(sum_numeric("total_violation_cells"), sum_numeric("post_ac_violation_count"))
    powerflow_failures = sum_numeric("post_ac_powerflow_failed")
    if violation_total > 0.0 or powerflow_failures > 0.0:
        add(
            severity="high",
            check_id="ac_reference_post_ac_unsafe",
            finding=(
                "ac_validated_search_reference produced post-AC violations or power-flow failures "
                f"(violations={violation_total:.0f}, powerflow_failures={powerflow_failures:.0f})."
            ),
            finding_zh=(
                "ac_validated_search_reference 出现 post-AC 越限或潮流失败 "
                f"(越限={violation_total:.0f}, 潮流失败={powerflow_failures:.0f})。"
            ),
            recommendation=(
                "Do not enter paper-long RL training until the AC reference baseline is safe on all "
                "evaluation variants and seeds."
            ),
            status="blocked",
            block_execution=True,
        )

    fallback_steps = sum_numeric("fallback_to_current_dispatch_step_count")
    current_insecure_rate = max_numeric("certificate_failed_current_dispatch_insecure_rate")
    no_safe_recovery_rate = max_numeric("certificate_failed_no_ac_safe_recovery_rate")
    if fallback_steps > 0.0 or current_insecure_rate > 0.0 or no_safe_recovery_rate > 0.0:
        add(
            severity="high",
            check_id="ac_reference_certificate_hard_failures",
            finding=(
                "AC reference certification had hard failures "
                f"(fallback_steps={fallback_steps:.0f}, current_insecure_rate={current_insecure_rate:.3f}, "
                f"no_safe_recovery_rate={no_safe_recovery_rate:.3f})."
            ),
            finding_zh=(
                "AC 参考基线存在证书硬失败 "
                f"(回退步数={fallback_steps:.0f}, current 不安全率={current_insecure_rate:.3f}, "
                f"无安全恢复率={no_safe_recovery_rate:.3f})。"
            ),
            recommendation=(
                "Fix the emergency recovery/reference search path before using this baseline in paper-long "
                "comparisons."
            ),
            status="blocked",
            block_execution=True,
        )

    if "is_ac_validated" in ac_rows:
        validated = ac_rows["is_ac_validated"].map(lambda value: bool(value) if pd.notna(value) else False)
        if not bool(validated.all()):
            add(
                severity="high",
                check_id="ac_reference_not_validated_every_seed",
                finding="At least one AC reference baseline row is not marked AC-validated.",
                finding_zh="至少一个 AC 参考基线结果行没有被标记为 AC 校验通过。",
                recommendation="Require every AC reference seed/variant to have feasible candidates and no unsafe fallback.",
                status="blocked",
                block_execution=True,
            )

    if not rows:
        add(
            severity="info",
            check_id="ac_reference_baseline_gate_passed",
            finding="AC reference baseline passed post-AC safety and certification checks.",
            finding_zh="AC 参考基线通过了 post-AC 安全性和证书检查。",
            recommendation="Proceed to trainable RL algorithms; keep reporting certificate and post-AC statistics.",
            status="passed",
            block_execution=False,
        )
    return pd.DataFrame(rows)


def _write_baseline_phase_artifacts(
    *,
    output_dir: Path,
    cfg: PaperTrainingExperimentConfig,
    run_rows: list[dict[str, Any]],
    seed_rows: list[dict[str, Any]],
    eval_rows: list[dict[str, Any]],
    profile_rows: list[dict[str, Any]],
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    run_index = pd.DataFrame(run_rows)
    seed_metrics = pd.DataFrame(seed_rows)
    evaluation_seed_metrics = pd.DataFrame(eval_rows)
    profile_quality = pd.DataFrame(profile_rows)
    aggregate_metrics = _aggregate_eval_metrics(evaluation_seed_metrics)
    baseline_comparison = _baseline_comparison(evaluation_seed_metrics)
    diagnostics = _architecture_diagnostics(cfg=cfg, evaluation_seed_metrics=evaluation_seed_metrics, run_index=run_index)
    baseline_gate = _baseline_safety_gate_diagnostics(cfg=cfg, evaluation_seed_metrics=evaluation_seed_metrics)
    if not baseline_gate.empty:
        diagnostics = pd.concat([diagnostics, baseline_gate], ignore_index=True, sort=False)
    claim_guardrails, claim_readiness = _claim_guardrails(
        cfg=cfg,
        evaluation_seed_metrics=evaluation_seed_metrics,
        diagnostics=diagnostics,
    )
    blockers = (
        baseline_gate["block_execution"].fillna(False).map(bool)
        if not baseline_gate.empty and "block_execution" in baseline_gate
        else pd.Series(dtype=bool)
    )
    if bool(blockers.any()):
        claim_readiness = {
            **claim_readiness,
            "execution_ready": False,
            "summary": "Baseline safety gate failed; paper-long RL training should not start from this baseline set.",
        }

    run_index.to_csv(output_dir / "baseline_run_index.csv", index=False)
    seed_metrics.to_csv(output_dir / "baseline_seed_metrics.csv", index=False)
    evaluation_seed_metrics.to_csv(output_dir / "baseline_evaluation_seed_metrics.csv", index=False)
    aggregate_metrics.to_csv(output_dir / "baseline_aggregate_metrics.csv", index=False)
    baseline_comparison.to_csv(output_dir / "baseline_comparison.csv", index=False)
    profile_quality.to_csv(output_dir / "baseline_profile_quality.csv", index=False)
    diagnostics.to_csv(output_dir / "baseline_architecture_diagnostics.csv", index=False)
    claim_guardrails.to_csv(output_dir / "baseline_claim_guardrails.csv", index=False)
    baseline_gate.to_csv(output_dir / "baseline_safety_gate.csv", index=False)
    write_json(output_dir / "baseline_claim_readiness.json", _make_json_safe(claim_readiness))

    manifest = {
        "schema_version": "paper_training_v1",
        "phase": "baseline_complete",
        "config": cfg.to_dict(),
        "runtime": _runtime_versions(),
        "source_config_hash": _file_sha256(cfg.config_path),
        "artifacts": {
            "baseline_run_index": str(output_dir / "baseline_run_index.csv"),
            "baseline_seed_metrics": str(output_dir / "baseline_seed_metrics.csv"),
            "baseline_evaluation_seed_metrics": str(output_dir / "baseline_evaluation_seed_metrics.csv"),
            "baseline_safety_gate": str(output_dir / "baseline_safety_gate.csv"),
            "baseline_claim_guardrails": str(output_dir / "baseline_claim_guardrails.csv"),
            "baseline_claim_readiness": str(output_dir / "baseline_claim_readiness.json"),
        },
    }
    write_json(output_dir / "experiment_manifest.json", _make_json_safe(manifest))
    return baseline_gate, diagnostics, claim_readiness


def _write_long_training_report(
    *,
    output_dir: Path,
    manifest: dict[str, Any],
    run_index: pd.DataFrame,
    episode_metrics: pd.DataFrame,
    loss_metrics: pd.DataFrame,
    evaluation_seed_metrics: pd.DataFrame,
    aggregate_metrics: pd.DataFrame,
    baseline_comparison: pd.DataFrame,
    convergence_summary: pd.DataFrame,
    profile_quality: pd.DataFrame,
    diagnostics: pd.DataFrame,
    claim_guardrails: pd.DataFrame,
    claim_readiness: dict[str, Any],
    image_index: pd.DataFrame,
) -> Path:
    data = {
        "manifest": manifest,
        "run_index": run_index.to_dict(orient="records"),
        "episode_metrics": _sample_frame_for_report(episode_metrics).to_dict(orient="records"),
        "loss_metrics": _sample_frame_for_report(loss_metrics).to_dict(orient="records"),
        "evaluation_seed_metrics": evaluation_seed_metrics.to_dict(orient="records"),
        "aggregate_metrics": aggregate_metrics.to_dict(orient="records"),
        "baseline_comparison": baseline_comparison.to_dict(orient="records"),
        "convergence_summary": convergence_summary.to_dict(orient="records"),
        "profile_quality": profile_quality.to_dict(orient="records"),
        "diagnostics": diagnostics.to_dict(orient="records"),
        "claim_guardrails": claim_guardrails.to_dict(orient="records"),
        "claim_readiness": claim_readiness,
        "images": image_index.to_dict(orient="records"),
    }
    (output_dir / "long_training_report_data.json").write_text(
        json.dumps(_make_json_safe(data), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    data_json = html.escape(json.dumps(_make_json_safe(data), ensure_ascii=False))
    image_cards = "\n".join(
        f"""<figure class="image-card"><img src="{html.escape(str(row['image_path']))}" alt="{html.escape(str(row['title_en']))}"><figcaption data-en="{html.escape(str(row['title_en']))}" data-zh="{html.escape(str(row['title_zh']))}">{html.escape(str(row['title_zh']))}</figcaption></figure>"""
        for row in image_index.to_dict(orient="records")
    )
    path = output_dir / "long_training_report.html"
    path.write_text(
        f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <title>DSO/VPP Long Training Report</title>
  <style>
    body {{ font-family: Segoe UI, Microsoft YaHei, sans-serif; margin: 0; background: #f4f7fb; color: #16202a; }}
    header {{ padding: 24px 32px; background: #102033; color: white; }}
    main {{ padding: 24px 32px 48px; }}
    button {{ border: 1px solid #7b8da3; background: white; border-radius: 6px; padding: 7px 11px; cursor: pointer; }}
    .toolbar {{ display: flex; gap: 10px; align-items: center; margin-top: 12px; }}
    .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(240px, 1fr)); gap: 14px; }}
    .card, section {{ background: white; border: 1px solid #d7e0ea; border-radius: 8px; padding: 16px; margin-bottom: 18px; }}
    .card b {{ display: block; font-size: 13px; color: #526174; margin-bottom: 4px; }}
    .card span {{ font-size: 24px; font-weight: 700; }}
    table {{ width: 100%; border-collapse: collapse; font-size: 13px; }}
    th, td {{ border: 1px solid #d9e2ec; padding: 7px 8px; text-align: left; vertical-align: top; }}
    th {{ background: #edf3f8; position: sticky; top: 0; }}
    .table-wrap {{ max-height: 420px; overflow: auto; border: 1px solid #d9e2ec; }}
    .image-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(360px, 1fr)); gap: 14px; }}
    .image-card {{ background: #fbfdff; border: 1px solid #d9e2ec; border-radius: 8px; margin: 0; padding: 10px; }}
    .image-card img {{ width: 100%; height: auto; display: block; }}
    .image-card figcaption {{ font-size: 13px; color: #44566c; margin-top: 8px; }}
    .note {{ background: #fff7df; border-left: 4px solid #d69e2e; padding: 11px 13px; }}
    .hidden {{ display: none; }}
  </style>
</head>
<body>
<header>
  <h1 data-en="DSO/VPP Long Training Report" data-zh="DSO/VPP 长周期训练报告">DSO/VPP 长周期训练报告</h1>
  <div class="toolbar">
    <button onclick="setLang('zh')">中文</button>
    <button onclick="setLang('en')">English</button>
    <button onclick="toggleAllTables()" data-en="Toggle tables" data-zh="切换表格显示">切换表格显示</button>
  </div>
</header>
<main>
  <section class="note">
    <b data-en="Claim boundary" data-zh="结论边界">结论边界</b>
    <span data-en="This report is an executable experiment dashboard. Smoke/pilot runs validate the pipeline; paper claims require the paper_long preset, real split audits, and a strict AC OPF or MILP reference before any optimality or upper-bound language."
          data-zh="本报告是可执行实验看板。smoke/pilot 用于验证流水线；论文结论需要 paper_long preset、真实 split 审计；若要写最优性或上界，还需要严格 AC OPF 或 MILP 参考。">本报告是可执行实验看板。smoke/pilot 用于验证流水线；论文结论需要 paper_long preset、真实 split 审计；若要写最优性或上界，还需要严格 AC OPF 或 MILP 参考。</span>
  </section>
  <section>
    <h2 data-en="Claim Guardrails" data-zh="论文声明门禁">论文声明门禁</h2>
    <div class="table-wrap" id="claim-table"></div>
  </section>
  <section>
    <h2 data-en="Overview" data-zh="实验总览">实验总览</h2>
    <div class="grid" id="overview"></div>
  </section>
  <section>
    <h2 data-en="TensorBoard Exported Images" data-zh="TensorBoard/训练图像导出">TensorBoard/训练图像导出</h2>
    <div class="image-grid">{image_cards}</div>
  </section>
  <section>
    <h2 data-en="Frozen Evaluation Metrics" data-zh="冻结评估指标">冻结评估指标</h2>
    <div class="table-wrap" id="eval-table"></div>
  </section>
  <section>
    <h2 data-en="Baseline Comparison" data-zh="Baseline 对比">Baseline 对比</h2>
    <div class="table-wrap" id="baseline-table"></div>
  </section>
  <section>
    <h2 data-en="Convergence Summary" data-zh="收敛摘要">收敛摘要</h2>
    <div class="table-wrap" id="convergence-table"></div>
  </section>
  <section>
    <h2 data-en="Architecture Diagnostics" data-zh="架构与实验诊断">架构与实验诊断</h2>
    <div class="table-wrap" id="diagnostics-table"></div>
  </section>
  <section>
    <h2 data-en="Run Index" data-zh="运行索引">运行索引</h2>
    <div class="table-wrap" id="run-table"></div>
  </section>
</main>
<script id="payload" type="application/json">{data_json}</script>
<script>
const payload = JSON.parse(document.getElementById('payload').textContent);
let lang = localStorage.getItem('paperTrainingLang') || 'zh';
function text(en, zh) {{ return lang === 'en' ? en : zh; }}
function setLang(next) {{
  lang = next; localStorage.setItem('paperTrainingLang', lang);
  document.querySelectorAll('[data-en]').forEach(el => el.textContent = el.dataset[lang]);
  render();
}}
function fmt(v) {{
  if (typeof v === 'number') return Math.abs(v) >= 100 ? v.toFixed(1) : v.toFixed(4);
  if (v === null || v === undefined) return '';
  return String(v);
}}
function renderTable(id, rows, maxRows=200) {{
  const el = document.getElementById(id);
  if (!rows || rows.length === 0) {{ el.innerHTML = '<p>No rows.</p>'; return; }}
  const keys = Object.keys(rows[0]);
  const body = rows.slice(0, maxRows).map(r => '<tr>' + keys.map(k => '<td>' + fmt(r[k]) + '</td>').join('') + '</tr>').join('');
  el.innerHTML = '<table><thead><tr>' + keys.map(k => '<th>' + k + '</th>').join('') + '</tr></thead><tbody>' + body + '</tbody></table>';
}}
function render() {{
  const m = payload.manifest.config;
  const cards = [
    [text('Preset','Preset'), m.preset],
    [text('Algorithms','算法'), (m.algorithms || []).join(', ')],
    [text('Seeds','Seed 数'), (m.seeds || []).length],
    [text('Train episodes','训练 episode'), m.train_episodes],
    [text('Horizon steps','仿真步数'), m.horizon_steps],
    [text('Data source','数据源'), m.data_source],
    [text('Claim ready','论文声明就绪'), payload.claim_readiness.paper_claim_ready],
  ];
  document.getElementById('overview').innerHTML = cards.map(c => `<div class="card"><b>${{c[0]}}</b><span>${{fmt(c[1])}}</span></div>`).join('');
  renderTable('claim-table', payload.claim_guardrails);
  renderTable('eval-table', payload.evaluation_seed_metrics);
  renderTable('baseline-table', payload.baseline_comparison);
  renderTable('convergence-table', payload.convergence_summary);
  renderTable('diagnostics-table', payload.diagnostics);
  renderTable('run-table', payload.run_index);
}}
function toggleAllTables() {{
  document.querySelectorAll('.table-wrap').forEach(el => el.classList.toggle('hidden'));
}}
setLang(lang);
</script>
</body>
</html>
""",
        encoding="utf-8",
    )
    return path


def run_paper_training_experiment(config: PaperTrainingExperimentConfig | None = None) -> dict[str, Any]:
    cfg = config or PaperTrainingExperimentConfig()
    if _is_paper_long_family(cfg) and not bool(cfg.require_cuda_for_trainable):
        cfg = replace(cfg, require_cuda_for_trainable=True)
    out = ensure_dir(cfg.output_dir)
    _guard_output_protocol(out, cfg)
    _configure_local_plot_cache(out)
    trainable_algorithm_names = {"happo", "hatrpo", "matd3", "hasac"}
    baseline_total = len(cfg.seeds) * len(cfg.eval_variants) * len([alg for alg in cfg.algorithms if alg not in trainable_algorithm_names])
    train_total = (
        len(cfg.seeds)
        * len(cfg.train_variants)
        * len([alg for alg in cfg.algorithms if alg in trainable_algorithm_names])
        * len(cfg.hparam_cases)
    )
    checkpoint_eval_multiplier = len(_checkpoint_choices({"checkpoint": "checkpoint.pt"}, str(cfg.checkpoint_selection)))
    eval_total = train_total * len(cfg.eval_variants) * checkpoint_eval_multiplier
    progress_state: dict[str, Any] = {
        "interval_seconds": float(cfg.progress_interval_seconds),
        "last_print_monotonic": 0.0,
        "totals": {
            "baseline": baseline_total,
            "train": train_total,
            "eval": eval_total,
            "overall": baseline_total + train_total + eval_total,
        },
        "done": {"baseline": 0, "train": 0, "eval": 0, "overall": 0},
        "latest": {},
        "verbose_progress": bool(cfg.verbose_progress),
    }
    progress_state["_tqdm_bars"] = _build_tqdm_bars(progress_state)
    _print_progress(
        out,
        {
            "phase": "start",
            "message": "campaign started",
            "preset": cfg.preset,
            "algorithms": ",".join(cfg.algorithms),
            "seeds": ",".join(str(seed) for seed in cfg.seeds),
            "horizon_steps": int(cfg.horizon_steps),
            "eval_horizon_steps": int(cfg.eval_horizon_steps or cfg.horizon_steps),
            "train_episodes": int(cfg.train_episodes),
        },
        print_event=bool(cfg.verbose_progress),
    )
    progress_state["latest"] = {"phase": "start", "run_id": "", "message": "campaign started"}
    _emit_progress_summary(out, progress_state, force=True)
    trainable = trainable_algorithm_names
    baseline_algorithms = [alg for alg in cfg.algorithms if alg not in trainable]
    train_algorithms = [alg for alg in cfg.algorithms if alg in trainable]
    eval_horizon = int(cfg.eval_horizon_steps or cfg.horizon_steps)

    run_rows: list[dict[str, Any]] = []
    seed_rows: list[dict[str, Any]] = []
    eval_rows: list[dict[str, Any]] = []
    profile_rows: list[dict[str, Any]] = []
    episode_frames: list[pd.DataFrame] = []
    loss_frames: list[pd.DataFrame] = []
    tb_rows: list[dict[str, Any]] = []

    for seed in cfg.seeds:
        for variant in (*cfg.train_variants, *cfg.eval_variants):
            run_id = f"profile_{variant}_seed_{seed}"
            horizon = int(cfg.horizon_steps if variant in cfg.train_variants else eval_horizon)
            profile_seed = int(seed) if variant in cfg.train_variants else int(seed) + 10_000
            config_path, metadata, quality = _write_profile_config(
                cfg=cfg,
                output_dir=out,
                seed=profile_seed,
                variant=variant,
                horizon_steps=horizon,
                run_id=run_id,
            )
            for quality_row in quality.to_dict(orient="records"):
                profile_rows.append(
                    {
                        "seed": int(seed),
                        "profile_variant": variant,
                        "data_source": metadata.get("source", cfg.data_source),
                        "profile_seed": int(profile_seed),
                        **quality_row,
                    }
                )
            if variant not in cfg.eval_variants:
                continue
            for algorithm in baseline_algorithms:
                baseline_run_id = f"{algorithm}_{variant}_seed_{seed}"
                run_dir = ensure_dir(out / "runs" / baseline_run_id)
                baseline_profile_hash = _profile_data_hash(config_path)
                _print_progress(
                    out,
                    {
                        "phase": "baseline_start",
                        "message": "baseline rollout",
                        "run_id": baseline_run_id,
                        "algorithm": algorithm,
                        "seed": int(seed),
                        "profile_variant": variant,
                        "horizon_steps": eval_horizon,
                        "profile_seed": int(profile_seed),
                    },
                    print_event=bool(cfg.verbose_progress),
                )

                def baseline_progress_callback(event: dict[str, Any]) -> None:
                    progress_event = {
                        **event,
                        "run_id": baseline_run_id,
                        "algorithm": algorithm,
                        "seed": int(seed),
                        "profile_variant": variant,
                    }
                    _print_progress(out, progress_event, print_event=False)
                    progress_state["latest"] = progress_event
                    _emit_progress_summary(out, progress_state)

                baseline = _run_baseline_rollout(
                    algorithm=algorithm,
                    config_path=config_path,
                    output_dir=run_dir,
                    seed=int(seed),
                    variant=variant,
                    split="eval_profile",
                    scenario_name="european_lv_benchmark_v2",
                    horizon_steps=eval_horizon,
                    experiment_level=cfg.preset,
                    reuse_existing=bool(cfg.resume_completed),
                    ac_reference_max_candidates=int(cfg.ac_reference_max_candidates),
                    progress_callback=baseline_progress_callback,
                    progress_step_interval=6,
                )
                _print_progress(
                    out,
                    {
                        "phase": "baseline_done",
                        "message": "baseline completed",
                        "run_id": baseline_run_id,
                        "algorithm": algorithm,
                        "seed": int(seed),
                        "profile_variant": variant,
                        "reward_sum": baseline["metrics"].get("reward_sum"),
                        "total_cost": baseline["metrics"].get("total_cost"),
                        "violations": baseline["metrics"].get("total_violation_cells"),
                    },
                    print_event=bool(cfg.verbose_progress),
                )
                progress_state["done"]["baseline"] += 1
                progress_state["done"]["overall"] += 1
                progress_state["latest"] = {
                    "phase": "baseline_done",
                    "run_id": baseline_run_id,
                    "algorithm": algorithm,
                    "seed": int(seed),
                    "profile_variant": variant,
                    "reward_sum": baseline["metrics"].get("reward_sum"),
                    "total_cost": baseline["metrics"].get("total_cost"),
                    "violations": baseline["metrics"].get("total_violation_cells"),
                }
                _emit_progress_summary(out, progress_state)
                seed_rows.append(baseline["metrics"])
                eval_rows.append(
                    _baseline_eval_summary_row(
                        run_id=baseline_run_id,
                        algorithm=algorithm,
                        seed=int(seed),
                        profile_variant=variant,
                        horizon_steps=eval_horizon,
                        profile_seed=int(profile_seed),
                        profile_config_path=config_path,
                        profile_hash=baseline_profile_hash,
                        metrics=baseline["metrics"],
                        results=baseline["results"],
                    )
                )
                run_rows.append(
                    {
                        "run_id": baseline_run_id,
                        "algorithm": algorithm,
                        "seed": int(seed),
                        "split": "eval_profile",
                        "profile_variant": variant,
                        "hparam_case": "baseline",
                        "status": "completed",
                        "run_dir": str(run_dir),
                        "checkpoint_path": "",
                        "tensorboard_dir": "",
                        "config_path": str(config_path),
                        "profile_seed": int(profile_seed),
                        "profile_hash": baseline_profile_hash,
                    }
                )

    baseline_gate, _, _ = _write_baseline_phase_artifacts(
        output_dir=out,
        cfg=cfg,
        run_rows=run_rows,
        seed_rows=seed_rows,
        eval_rows=eval_rows,
        profile_rows=profile_rows,
    )
    baseline_blocked = (
        not baseline_gate.empty
        and "block_execution" in baseline_gate
        and bool(baseline_gate["block_execution"].fillna(False).map(bool).any())
    )
    if baseline_blocked and _is_paper_long_family(cfg):
        blocking = baseline_gate[baseline_gate["block_execution"].fillna(False).map(bool)]
        blocking_ids = ",".join(blocking["check_id"].astype(str).tolist())
        _print_progress(
            out,
            {
                "phase": "baseline_gate_failed",
                "message": "paper-long training stopped before RL because baseline safety gate failed",
                "run_id": "baseline_safety_gate",
                "hparam_case": "baseline",
                "violations": int(
                    pd.to_numeric(
                        pd.DataFrame(eval_rows).get("total_violation_cells", pd.Series(dtype=float)),
                        errors="coerce",
                    )
                    .fillna(0.0)
                    .sum()
                ),
            },
            print_event=True,
        )
        _close_tqdm_bars(progress_state)
        raise RuntimeError(
            "Baseline safety gate failed before paper-long RL training. "
            f"Blocking checks: {blocking_ids}. See {out / 'baseline_safety_gate.csv'}."
        )

    for seed in cfg.seeds:
        for train_variant in cfg.train_variants:
            train_profile_id = f"train_{train_variant}_seed_{seed}"
            train_config_path, _, train_quality = _write_profile_config(
                cfg=cfg,
                output_dir=out,
                seed=int(seed),
                variant=train_variant,
                horizon_steps=int(cfg.horizon_steps),
                run_id=train_profile_id,
            )
            for algorithm in train_algorithms:
                for case in cfg.hparam_cases:
                    first_eval_variant = cfg.eval_variants[0]
                    first_eval_profile_id = f"eval_{first_eval_variant}_seed_{seed}_{algorithm}_{case}"
                    first_eval_config_path, _, _ = _write_profile_config(
                        cfg=cfg,
                        output_dir=out,
                        seed=int(seed) + 10_000,
                        variant=first_eval_variant,
                        horizon_steps=eval_horizon,
                        run_id=first_eval_profile_id,
                    )
                    train_run_id = f"{algorithm}_{case}_{train_variant}_seed_{seed}"
                    run_dir = ensure_dir(out / "runs" / train_run_id)
                    existing_train = _load_completed_training(algorithm, run_dir) if cfg.resume_completed else None
                    if existing_train is None:
                        _print_progress(
                            out,
                            {
                                "phase": "train_start",
                                "message": "training started",
                                "run_id": train_run_id,
                                "algorithm": algorithm,
                                "seed": int(seed),
                                "hparam_case": case,
                                "train_variant": train_variant,
                                "episodes": int(cfg.train_episodes),
                                "horizon_steps": int(cfg.horizon_steps),
                            },
                            print_event=bool(cfg.verbose_progress),
                        )

                        def train_progress_callback(event: dict[str, Any]) -> None:
                            progress_event = {
                                **event,
                                "run_id": train_run_id,
                                "algorithm": algorithm,
                                "seed": int(seed),
                                "hparam_case": case,
                                "train_variant": train_variant,
                            }
                            _print_progress(out, progress_event, print_event=False)
                            progress_state["latest"] = progress_event
                            _emit_progress_summary(out, progress_state)

                        trained = _train_algorithm(
                            algorithm=algorithm,
                            cfg=cfg,
                            train_config_path=train_config_path,
                            eval_config_path=first_eval_config_path,
                            run_dir=run_dir,
                            seed=int(seed),
                            case=case,
                            eval_horizon_steps=eval_horizon,
                            progress_callback=train_progress_callback,
                            progress_step_interval=24,
                        )
                        train = trained["train"]
                        first_eval_result = trained["eval"]
                        final_reward = (
                            float(train["episode_metrics"]["episode_reward"].iloc[-1])
                            if not train["episode_metrics"].empty
                            and "episode_reward" in train["episode_metrics"]
                            else None
                        )
                        _print_progress(
                            out,
                            {
                                "phase": "train_done",
                                "message": "training completed",
                                "run_id": train_run_id,
                                "algorithm": algorithm,
                                "seed": int(seed),
                                "hparam_case": case,
                                "final_episode_reward": final_reward,
                                "checkpoint": str(train["checkpoint"]),
                            },
                            print_event=bool(cfg.verbose_progress),
                        )
                        progress_state["done"]["train"] += 1
                        progress_state["done"]["overall"] += 1
                        progress_state["latest"] = {
                            "phase": "train_done",
                            "run_id": train_run_id,
                            "algorithm": algorithm,
                            "seed": int(seed),
                            "hparam_case": case,
                            "final_episode_reward": final_reward,
                        }
                        _emit_progress_summary(out, progress_state, force=True)
                    else:
                        _print_progress(
                            out,
                            {
                                "phase": "train_resume_skip",
                                "message": "existing checkpoint reused",
                                "run_id": train_run_id,
                                "algorithm": algorithm,
                                "seed": int(seed),
                                "hparam_case": case,
                                "checkpoint": str(existing_train["checkpoint"]),
                            },
                            print_event=bool(cfg.verbose_progress),
                        )
                        progress_state["done"]["train"] += 1
                        progress_state["done"]["overall"] += 1
                        progress_state["latest"] = {
                            "phase": "train_resume_skip",
                            "run_id": train_run_id,
                            "algorithm": algorithm,
                            "seed": int(seed),
                            "hparam_case": case,
                        }
                        _emit_progress_summary(out, progress_state)
                        trained = {
                            "train": existing_train,
                            "hidden_dim": int(cfg.hidden_dim * int(_case_overrides(case).get("hidden_dim_multiplier", 1))),
                            "learning_rate": float(cfg.learning_rate)
                            * float(_case_overrides(case).get("learning_rate_multiplier", 1.0)),
                        }
                        train = existing_train
                        first_eval_result = None
                    checkpoint_choices = _checkpoint_choices(train, str(cfg.checkpoint_selection))
                    selected_checkpoint = checkpoint_choices[0][1]
                    ep = train["episode_metrics"].copy()
                    ep["run_id"] = train_run_id
                    ep["algorithm"] = _algorithm_label(algorithm)
                    ep["seed"] = int(seed)
                    ep["profile_variant"] = train_variant
                    ep["hparam_case"] = case
                    episode_frames.append(ep)
                    updates = train.get("update_metrics", pd.DataFrame()).copy()
                    if not updates.empty:
                        updates["run_id"] = train_run_id
                        updates["algorithm"] = _algorithm_label(algorithm)
                        updates["seed"] = int(seed)
                        updates["hparam_case"] = case
                        loss_frames.append(updates)
                    train_tensorboard_dir = ""
                    if cfg.tensorboard:
                        tb_dir = _write_tensorboard_scalars(
                            output_dir=out,
                            run_id=train_run_id,
                            episode_metrics=ep,
                            update_metrics=updates,
                            eval_step_metrics=pd.DataFrame(),
                            write_train=True,
                            write_eval=False,
                        )
                        train_tensorboard_dir = str(tb_dir) if tb_dir is not None else ""
                        if tb_dir is not None:
                            tb_rows.append(
                                {
                                    "run_id": train_run_id,
                                    "algorithm": _algorithm_label(algorithm),
                                    "seed": int(seed),
                                    "split": "train_profile",
                                    "tensorboard_dir": train_tensorboard_dir,
                                }
                            )
                    run_rows.append(
                        {
                            "run_id": train_run_id,
                            "algorithm": _algorithm_label(algorithm),
                            "seed": int(seed),
                            "split": "train_profile",
                            "profile_variant": train_variant,
                            "hparam_case": case,
                            "status": "completed",
                            "run_dir": str(run_dir),
                            "checkpoint_path": str(selected_checkpoint),
                            "final_checkpoint_path": str(train.get("final_checkpoint", selected_checkpoint)),
                            "best_checkpoint_path": str(train.get("best_checkpoint", selected_checkpoint)),
                            "checkpoint_selection": str(cfg.checkpoint_selection),
                            "checkpoint_label": checkpoint_choices[0][0],
                            "tensorboard_dir": train_tensorboard_dir,
                            "config_path": str(train_config_path),
                            "profile_seed": int(seed),
                            "profile_hash": _profile_data_hash(train_config_path),
                            "hidden_dim": int(trained["hidden_dim"]),
                            "learning_rate": float(trained["learning_rate"]),
                        }
                    )

                    for eval_variant in cfg.eval_variants:
                        eval_profile_id = f"eval_{eval_variant}_seed_{seed}_{algorithm}_{case}"
                        eval_config_path, _, _ = _write_profile_config(
                            cfg=cfg,
                            output_dir=out,
                            seed=int(seed) + 10_000,
                            variant=eval_variant,
                            horizon_steps=eval_horizon,
                            run_id=eval_profile_id,
                        )
                        eval_profile_hash = _profile_data_hash(eval_config_path)
                        for checkpoint_label, checkpoint_path in checkpoint_choices:
                            checkpoint_suffix = (
                                f"_ckpt_{checkpoint_label}" if str(cfg.checkpoint_selection) == "both" else ""
                            )
                            eval_run_id = f"{train_run_id}{checkpoint_suffix}_eval_{eval_variant}"
                            _print_progress(
                                out,
                                {
                                    "phase": "eval_start",
                                    "message": "frozen eval started",
                                    "run_id": eval_run_id,
                                    "algorithm": algorithm,
                                    "seed": int(seed),
                                    "hparam_case": case,
                                    "eval_variant": eval_variant,
                                    "checkpoint_label": checkpoint_label,
                                    "horizon_steps": eval_horizon,
                                },
                                print_event=bool(cfg.verbose_progress),
                            )
                            default_eval_dir = "frozen_eval" if eval_variant == first_eval_variant else f"frozen_eval_{eval_variant}"
                            eval_dir_name = (
                                default_eval_dir
                                if str(cfg.checkpoint_selection) != "both"
                                else f"{default_eval_dir}_{checkpoint_label}"
                            )
                            eval_run_dir = ensure_dir(run_dir / eval_dir_name)
                            use_first_eval = (
                                first_eval_result is not None
                                and eval_variant == first_eval_variant
                                and str(cfg.checkpoint_selection) != "both"
                            )
                            eval_result = first_eval_result if use_first_eval else None
                            if eval_result is None:
                                eval_result = _load_completed_eval(algorithm, eval_run_dir) if cfg.resume_completed else None
                            if eval_result is None:
                                eval_result = _evaluate_algorithm_checkpoint(
                                    algorithm=algorithm,
                                    eval_config_path=eval_config_path,
                                    checkpoint_path=checkpoint_path,
                                    eval_output_dir=eval_run_dir,
                                    eval_horizon_steps=eval_horizon,
                                    seed=int(seed),
                                )
                            step_metrics = _augment_step_metrics_from_simulator_results(
                                eval_result["step_metrics"].copy(),
                                eval_result.get("output_dir", eval_run_dir),
                            )
                            reward_sum = (
                                float(step_metrics.get("reward", pd.Series(dtype=float)).fillna(0.0).sum())
                                if not step_metrics.empty
                                else 0.0
                            )
                            total_cost = (
                                float(step_metrics.get("total_cost", pd.Series(dtype=float)).fillna(0.0).sum())
                                if not step_metrics.empty
                                else 0.0
                            )
                            violations = (
                                int(step_metrics.get("violation_count", pd.Series(dtype=float)).fillna(0).sum())
                                if not step_metrics.empty
                                else 0
                            )
                            _print_progress(
                                out,
                                {
                                    "phase": "eval_done",
                                    "message": "frozen eval completed",
                                    "run_id": eval_run_id,
                                    "algorithm": algorithm,
                                    "seed": int(seed),
                                    "hparam_case": case,
                                    "eval_variant": eval_variant,
                                    "checkpoint_label": checkpoint_label,
                                    "reward_sum": reward_sum,
                                    "total_cost": total_cost,
                                    "violations": violations,
                                },
                                print_event=bool(cfg.verbose_progress),
                            )
                            progress_state["done"]["eval"] += 1
                            progress_state["done"]["overall"] += 1
                            progress_state["latest"] = {
                                "phase": "eval_done",
                                "run_id": eval_run_id,
                                "algorithm": algorithm,
                                "seed": int(seed),
                                "hparam_case": case,
                                "eval_variant": eval_variant,
                                "checkpoint_label": checkpoint_label,
                                "reward_sum": reward_sum,
                                "total_cost": total_cost,
                                "violations": violations,
                            }
                            _emit_progress_summary(out, progress_state)
                            eval_rows.append(
                                _step_metric_summary(
                                    run_id=eval_run_id,
                                    algorithm=_algorithm_label(algorithm),
                                    seed=int(seed),
                                    split="eval_profile",
                                    profile_variant=eval_variant,
                                    hparam_case=case,
                                    profile_seed=int(seed) + 10_000,
                                    profile_config_path=eval_config_path,
                                    profile_hash=eval_profile_hash,
                                    checkpoint_path=checkpoint_path,
                                    checkpoint_label=checkpoint_label,
                                    checkpoint_selection=str(cfg.checkpoint_selection),
                                    step_metrics=step_metrics,
                                )
                            )
                            tensorboard_dir = ""
                            if cfg.tensorboard:
                                tb_dir = _write_tensorboard_scalars(
                                    output_dir=out,
                                    run_id=eval_run_id,
                                    episode_metrics=pd.DataFrame(),
                                    update_metrics=pd.DataFrame(),
                                    eval_step_metrics=step_metrics,
                                    write_train=False,
                                    write_eval=True,
                                )
                                tensorboard_dir = str(tb_dir) if tb_dir is not None else ""
                                if tb_dir is not None:
                                    tb_rows.append(
                                        {
                                            "run_id": eval_run_id,
                                            "algorithm": _algorithm_label(algorithm),
                                            "seed": int(seed),
                                            "split": "frozen_eval_profile",
                                            "checkpoint_label": checkpoint_label,
                                            "tensorboard_dir": tensorboard_dir,
                                        }
                                    )
                            run_rows.append(
                                {
                                    "run_id": eval_run_id,
                                    "algorithm": _algorithm_label(algorithm),
                                    "seed": int(seed),
                                    "split": "frozen_eval_profile",
                                    "profile_variant": f"{train_variant}->{eval_variant}",
                                    "hparam_case": case,
                                    "status": "completed",
                                    "run_dir": str(eval_run_dir),
                                    "checkpoint_path": str(checkpoint_path),
                                    "final_checkpoint_path": str(train.get("final_checkpoint", selected_checkpoint)),
                                    "best_checkpoint_path": str(train.get("best_checkpoint", selected_checkpoint)),
                                    "checkpoint_selection": str(cfg.checkpoint_selection),
                                    "checkpoint_label": checkpoint_label,
                                    "tensorboard_dir": tensorboard_dir,
                                    "config_path": str(eval_config_path),
                                    "train_config_path": str(train_config_path),
                                    "profile_seed": int(seed) + 10_000,
                                    "profile_hash": eval_profile_hash,
                                    "hidden_dim": int(trained["hidden_dim"]),
                                    "learning_rate": float(trained["learning_rate"]),
                                }
                            )

    run_index = pd.DataFrame(run_rows)
    seed_metrics = pd.DataFrame(seed_rows)
    evaluation_seed_metrics = pd.DataFrame(eval_rows)
    episode_clean = [frame.dropna(axis=1, how="all") for frame in episode_frames if not frame.empty]
    loss_clean = [frame.dropna(axis=1, how="all") for frame in loss_frames if not frame.empty]
    episode_metrics = pd.concat(episode_clean, ignore_index=True) if episode_clean else pd.DataFrame()
    loss_metrics = pd.concat(loss_clean, ignore_index=True) if loss_clean else pd.DataFrame()
    profile_quality = pd.DataFrame(profile_rows)
    aggregate_metrics = _aggregate_eval_metrics(evaluation_seed_metrics)
    baseline_comparison = _baseline_comparison(evaluation_seed_metrics)
    convergence_summary = _convergence_summary(episode_metrics, loss_metrics)
    diagnostics = _architecture_diagnostics(cfg=cfg, evaluation_seed_metrics=evaluation_seed_metrics, run_index=run_index)
    baseline_gate = _baseline_safety_gate_diagnostics(cfg=cfg, evaluation_seed_metrics=evaluation_seed_metrics)
    if not baseline_gate.empty:
        diagnostics = pd.concat([diagnostics, baseline_gate], ignore_index=True, sort=False)
    claim_guardrails, claim_readiness = _claim_guardrails(
        cfg=cfg,
        evaluation_seed_metrics=evaluation_seed_metrics,
        diagnostics=diagnostics,
    )
    image_index = _export_training_images(
        output_dir=out,
        episode_metrics=episode_metrics,
        loss_metrics=loss_metrics,
        evaluation_seed_metrics=evaluation_seed_metrics,
    )

    manifest = {
        "schema_version": "paper_training_v1",
        "config": cfg.to_dict(),
        "runtime": _runtime_versions(),
        "source_config_hash": _file_sha256(cfg.config_path),
        "claim_boundary": (
            "Executable long-training protocol with train/eval profile split, multi-seed aggregation, "
            "rule/no-flex/static-FR and AC-validated-search reference baselines, TensorBoard scalar logs, "
            "exported images, and static HTML. Static FR proxies are not OPF/oracle/upper-bound baselines; "
            "the AC-validated search reference is a bounded feasibility sanity reference, not an optimality proof."
        ),
        "artifacts": {
            "run_index": str(out / "run_index.csv"),
            "evaluation_seed_metrics": str(out / "evaluation_seed_metrics.csv"),
            "aggregate_metrics": str(out / "aggregate_metrics.csv"),
            "baseline_comparison": str(out / "baseline_comparison.csv"),
            "convergence_summary": str(out / "convergence_summary.csv"),
            "episode_metrics": str(out / "training_episode_metrics.csv"),
            "loss_metrics": str(out / "training_loss_metrics.csv"),
            "profile_quality": str(out / "profile_quality.csv"),
            "diagnostics": str(out / "architecture_diagnostics.csv"),
            "baseline_safety_gate": str(out / "baseline_safety_gate.csv"),
            "claim_guardrails": str(out / "claim_guardrails.csv"),
            "claim_readiness": str(out / "claim_readiness.json"),
            "tensorboard_assets": str(out / "tensorboard_assets.csv"),
        },
    }

    run_index.to_csv(out / "run_index.csv", index=False)
    seed_metrics.to_csv(out / "baseline_seed_metrics.csv", index=False)
    evaluation_seed_metrics.to_csv(out / "evaluation_seed_metrics.csv", index=False)
    aggregate_metrics.to_csv(out / "aggregate_metrics.csv", index=False)
    baseline_comparison.to_csv(out / "baseline_comparison.csv", index=False)
    convergence_summary.to_csv(out / "convergence_summary.csv", index=False)
    episode_metrics.to_csv(out / "training_episode_metrics.csv", index=False)
    loss_metrics.to_csv(out / "training_loss_metrics.csv", index=False)
    profile_quality.to_csv(out / "profile_quality.csv", index=False)
    diagnostics.to_csv(out / "architecture_diagnostics.csv", index=False)
    claim_guardrails.to_csv(out / "claim_guardrails.csv", index=False)
    write_json(out / "claim_readiness.json", _make_json_safe(claim_readiness))
    pd.DataFrame(tb_rows).to_csv(out / "tensorboard_runs.csv", index=False)
    write_json(out / "experiment_manifest.json", _make_json_safe(manifest))
    html_path = (
        _write_long_training_report(
            output_dir=out,
            manifest=manifest,
            run_index=run_index,
            episode_metrics=episode_metrics,
            loss_metrics=loss_metrics,
            evaluation_seed_metrics=evaluation_seed_metrics,
            aggregate_metrics=aggregate_metrics,
            baseline_comparison=baseline_comparison,
            convergence_summary=convergence_summary,
            profile_quality=profile_quality,
            diagnostics=diagnostics,
            claim_guardrails=claim_guardrails,
            claim_readiness=claim_readiness,
            image_index=image_index,
        )
        if cfg.export_html
        else None
    )
    _print_progress(
        out,
        {
            "phase": "done",
            "message": "campaign completed",
            "run_count": int(len(run_index)),
            "eval_rows": int(len(evaluation_seed_metrics)),
            "html_path": str(html_path) if html_path is not None else "",
        },
        print_event=True,
    )
    progress_state["latest"] = {
        "phase": "done",
        "run_id": "",
        "message": "campaign completed",
        "html_path": str(html_path) if html_path is not None else "",
    }
    _emit_progress_summary(out, progress_state, force=True)
    _close_tqdm_bars(progress_state)

    return {
        "manifest": manifest,
        "run_index": run_index,
        "baseline_seed_metrics": seed_metrics,
        "evaluation_seed_metrics": evaluation_seed_metrics,
        "aggregate_metrics": aggregate_metrics,
        "baseline_comparison": baseline_comparison,
        "convergence_summary": convergence_summary,
        "episode_metrics": episode_metrics,
        "loss_metrics": loss_metrics,
        "profile_quality": profile_quality,
        "diagnostics": diagnostics,
        "baseline_safety_gate": baseline_gate,
        "claim_guardrails": claim_guardrails,
        "claim_readiness": claim_readiness,
        "image_index": image_index,
        "html_path": html_path,
        "output_dir": out,
    }


__all__ = [
    "PaperTrainingExperimentConfig",
    "paper_training_preset",
    "run_paper_training_experiment",
]
