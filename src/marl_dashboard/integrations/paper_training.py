from __future__ import annotations

from pathlib import Path
import re
from typing import Any

import pandas as pd

from marl_dashboard.logging import ExperimentLogger


REWARD_COLUMNS = (
    "episode_reward",
    "dso_episode_reward",
    "vpp_dispatch_episode_reward",
    "vpp_portfolio_episode_reward",
    "eval_total_reward",
    "eval_mean_reward",
    "dso_reward_sum",
    "dispatch_reward_sum",
    "portfolio_reward_sum",
    "total_agent_reward_sum",
    "raw_objective_reward_sum",
)

COST_COLUMNS = (
    "episode_cost",
    "eval_total_cost",
    "total_cost",
)

SCALAR_COLUMNS = (
    "violation_count",
    "total_violation_cells",
    "post_ac_violation_count",
    "post_ac_voltage_violation_count",
    "post_ac_line_overload_count",
    "post_ac_trafo_overload_count",
    "post_ac_powerflow_failed",
    "post_ac_security_pass_rate",
    "post_ac_powerflow_converged_rate",
    "projection_gap_mw",
    "ac_certified_projection_gap_mw",
    "mean_ac_certified_projection_gap_mw",
    "ac_certificate_failed_count",
    "security_pass",
)

LOSS_HINTS = ("loss", "grad_norm", "entropy", "alpha")

UNITS = {
    "total_reward": "score",
    "episode_reward": "score",
    "dso_episode_reward": "score",
    "vpp_dispatch_episode_reward": "score",
    "vpp_portfolio_episode_reward": "score",
    "eval_total_reward": "score",
    "eval_mean_reward": "score/step",
    "dso_reward_sum": "score",
    "dispatch_reward_sum": "score",
    "portfolio_reward_sum": "score",
    "total_agent_reward_sum": "score",
    "raw_objective_reward_sum": "score",
    "total_cost": "cost",
    "episode_cost": "cost",
    "eval_total_cost": "cost",
    "violation_count": "count",
    "total_violation_cells": "count",
    "post_ac_violation_count": "count",
    "post_ac_voltage_violation_count": "count",
    "post_ac_line_overload_count": "count",
    "post_ac_trafo_overload_count": "count",
    "post_ac_powerflow_failed": "count",
    "post_ac_security_pass_rate": "ratio",
    "post_ac_powerflow_converged_rate": "ratio",
    "projection_gap_mw": "MW",
    "ac_certified_projection_gap_mw": "MW",
    "mean_ac_certified_projection_gap_mw": "MW",
    "ac_certificate_failed_count": "count",
    "security_pass": "bool",
}

FORMULAS = {
    "total_reward": "R_{episode}",
    "episode_reward": "R_{episode}",
    "eval_total_reward": "\\sum_t r_t^{eval}",
    "total_cost": "C_{total}",
    "episode_cost": "C_{episode}",
    "eval_total_cost": "\\sum_t C_t^{eval}",
}


def _slug(value: str) -> str:
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("_")
    return text or "paper_training"


def _frame(value: Any) -> pd.DataFrame:
    if isinstance(value, pd.DataFrame):
        return value
    if value is None:
        return pd.DataFrame()
    try:
        return pd.DataFrame(value)
    except Exception:
        return pd.DataFrame()


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and pd.notna(value)


def _value(row: pd.Series, column: str) -> float | int | None:
    if column not in row:
        return None
    value = row.get(column)
    if not _is_number(value):
        return None
    return float(value)


def _terms(row: pd.Series, columns: tuple[str, ...]) -> dict[str, float]:
    terms: dict[str, float] = {}
    for column in columns:
        value = _value(row, column)
        if value is not None:
            terms[column] = value
    return terms


def _loss_terms(row: pd.Series) -> dict[str, float]:
    terms: dict[str, float] = {}
    for column, value in row.items():
        if any(hint in str(column).lower() for hint in LOSS_HINTS) and _is_number(value):
            terms[str(column)] = float(value)
    return terms


def _int_value(row: pd.Series, column: str, default: int) -> int:
    value = row.get(column)
    if _is_number(value):
        return int(value)
    return int(default)


def _policy_id(row: pd.Series) -> str:
    algorithm = str(row.get("algorithm", "unknown_policy"))
    case = str(row.get("hparam_case", "") or "")
    return f"{algorithm}/{case}" if case else algorithm


def _base_context(row: pd.Series, *, row_index: int, split: str) -> dict[str, Any]:
    episode = _int_value(row, "episode", row_index)
    return {
        "epoch_id": 0,
        "episode_id": episode,
        "time_index": episode,
        "env_id": str(row.get("profile_variant", split) or split),
        "vpp_id": str(row.get("vpp_id", "aggregate") or "aggregate"),
        "agent_id": str(row.get("agent_id", "aggregate") or "aggregate"),
        "policy_id": _policy_id(row),
        "source_run_id": str(row.get("run_id", "")),
        "split": split,
        "seed": row.get("seed"),
        "hparam_case": row.get("hparam_case"),
    }


def _variable_dictionary() -> list[dict[str, Any]]:
    variables = []
    for name, unit in sorted(UNITS.items()):
        variables.append(
            {
                "name": name,
                "display_name": name.replace("_", " "),
                "symbol": name,
                "unit": unit,
                "group": "paper_training_summary",
                "physical_meaning": "Mapped from paper_training.py summary artifacts; confirm exact formula before publication claims.",
                "formula_latex": FORMULAS.get(name),
                "source": "src/vpp_dso_sim/experiments/paper_training.py",
                "notes": "Summary-level dashboard ingestion; not an env.step trace.",
            }
        )
    return variables


def export_paper_training_dashboard(
    result: dict[str, Any],
    *,
    data_dir: str | Path | None = None,
    run_id: str | None = None,
    async_writer: bool = False,
) -> str:
    """Export paper-training summary DataFrames to the dashboard schema.

    This adapter reads already-produced experiment artifacts. It does not call
    env.reset, env.step, optimizer.step, or mutate training objects.
    """

    output_dir = Path(result.get("output_dir", ".")).expanduser().resolve()
    manifest = dict(result.get("manifest", {}) or {})
    config = dict(manifest.get("config", {}) or {})
    preset = str(config.get("preset", "paper_training"))
    dashboard_run_id = str(run_id or f"paper_training_{_slug(preset)}")
    dashboard_data_dir = Path(data_dir or output_dir / "dashboard_runs").expanduser().resolve()

    logger = ExperimentLogger(
        run_id=dashboard_run_id,
        data_dir=str(dashboard_data_dir),
        config=config,
        variable_dictionary=_variable_dictionary(),
        formulas=FORMULAS,
        metadata={
            "source": "paper_training_summary_adapter",
            "output_dir": str(output_dir),
            "schema_version": manifest.get("schema_version", "unknown"),
            "artifact_level": "summary_csv",
        },
        async_writer=async_writer,
    )
    try:
        logger.log_event("paper_training_ingest_start", {"message": "paper training dashboard export started"})
        episode_metrics = _frame(result.get("episode_metrics"))
        for row_index, (_, row) in enumerate(episode_metrics.iterrows()):
            context = _base_context(row, row_index=row_index, split="train_profile")
            rewards = _terms(row, REWARD_COLUMNS)
            if "episode_reward" in rewards:
                rewards.setdefault("total_reward", rewards["episode_reward"])
            if rewards:
                logger.log_reward_terms(terms=rewards, units=UNITS, formulas=FORMULAS, **context)
            costs = _terms(row, COST_COLUMNS)
            if "episode_cost" in costs:
                costs.setdefault("total_cost", costs["episode_cost"])
            if costs:
                logger.log_cost_terms(terms=costs, units=UNITS, formulas=FORMULAS, **context)
            for metric_name, value in _terms(row, SCALAR_COLUMNS).items():
                logger.log_scalar(metric_name, value, unit=UNITS.get(metric_name), **context)

        loss_metrics = _frame(result.get("loss_metrics"))
        for row_index, (_, row) in enumerate(loss_metrics.iterrows()):
            gradient_step = _int_value(row, "global_step", _int_value(row, "critic_update", row_index))
            context = {
                **_base_context(row, row_index=row_index, split="train_profile"),
                "gradient_step": gradient_step,
                "time_index": gradient_step,
            }
            losses = _loss_terms(row)
            if losses:
                logger.log_loss_terms(terms=losses, optimizer_name="unknown", network_name="paper_training", **context)

        evaluation = _frame(result.get("evaluation_seed_metrics"))
        for row_index, (_, row) in enumerate(evaluation.iterrows()):
            context = _base_context(row, row_index=row_index, split="eval_profile")
            rewards = _terms(row, REWARD_COLUMNS)
            if "eval_total_reward" in rewards:
                rewards.setdefault("total_reward", rewards["eval_total_reward"])
            if rewards:
                logger.log_reward_terms(terms=rewards, units=UNITS, formulas=FORMULAS, **context)
            costs = _terms(row, COST_COLUMNS)
            if "eval_total_cost" in costs:
                costs.setdefault("total_cost", costs["eval_total_cost"])
            if costs:
                logger.log_cost_terms(terms=costs, units=UNITS, formulas=FORMULAS, **context)
            for metric_name, value in _terms(row, SCALAR_COLUMNS).items():
                logger.log_scalar(metric_name, value, unit=UNITS.get(metric_name), **context)

        run_index = _frame(result.get("run_index"))
        for row_index, (_, row) in enumerate(run_index.iterrows()):
            context = _base_context(row, row_index=row_index, split=str(row.get("split", "run_index")))
            logger.log_event(
                "paper_training_run_index",
                {
                    "message": str(row.get("status", "run_index")),
                    "source_run_id": str(row.get("run_id", "")),
                    "algorithm": str(row.get("algorithm", "")),
                    "split": str(row.get("split", "")),
                    "run_dir": str(row.get("run_dir", "")),
                },
                **context,
            )

        logger.log_event("paper_training_ingest_end", {"message": "paper training dashboard export finished"})
    except Exception:
        logger.close(status="error")
        raise
    else:
        logger.close(status="finished")
    return dashboard_run_id
