from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


REWARD_COMPONENT_TERMS: tuple[tuple[str, str], ...] = (
    ("dso_safety_margin_penalty", "dso"),
    ("dso_voltage_guard_penalty", "dso"),
    ("dso_line_guard_penalty", "dso"),
    ("dso_trafo_guard_penalty", "dso"),
    ("dso_powerflow_failure_penalty", "dso"),
    ("dso_flex_procurement_cost", "dso"),
    ("dso_loss_cost", "dso"),
    ("dso_curtailment_cost", "dso"),
    ("dso_safe_capacity_utilization_reward", "dso"),
    ("dso_over_conservative_curtailment_penalty", "dso"),
    ("dso_responsible_projection_penalty", "dso"),
    ("dispatch_private_profit_reward", "dispatch"),
    ("availability_payment", "dispatch"),
    ("service_payment", "dispatch"),
    ("contract_delivery_penalty", "dispatch"),
    ("dispatch_projection_penalty", "dispatch"),
    ("dispatch_comfort_soc_penalty", "dispatch"),
    ("dispatch_battery_degradation_penalty", "dispatch"),
    ("portfolio_profit_reward", "portfolio"),
    ("portfolio_verified_capacity_reward", "portfolio"),
    ("portfolio_contract_shortfall_penalty", "portfolio"),
    ("portfolio_future_shield_penalty", "portfolio"),
    ("portfolio_future_projection_penalty", "portfolio"),
    ("portfolio_future_comfort_soc_penalty", "portfolio"),
    ("portfolio_switching_cost", "portfolio"),
    ("shield_intervention_penalty", "safety"),
    ("shield_intervention_gap_mw", "safety"),
    ("action_projection_gap_mw", "safety"),
    ("local_bounds_projection_gap_mw", "safety"),
    ("ac_aware_projection_gap_mw", "safety"),
)

EVAL_COMPARE_METRICS: tuple[str, ...] = (
    "eval_total_reward",
    "eval_total_cost",
    "total_violation_cells",
    "post_ac_violation_count",
    "post_ac_powerflow_failed",
    "post_ac_security_pass_rate",
    "ac_certified_projection_gap_mw",
)


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path, low_memory=False)


def collect_step_metrics(run_dir: str | Path) -> pd.DataFrame:
    root = Path(run_dir)
    frames: list[pd.DataFrame] = []
    for path in sorted(root.glob("runs/*/train/*_step_metrics.csv")):
        frame = _read_csv(path)
        if frame.empty:
            continue
        frame["source_file"] = str(path)
        frame["run_id"] = path.parts[-3]
        frames.append(frame)
    return pd.concat(frames, ignore_index=True, sort=False) if frames else pd.DataFrame()


def _numeric(frame: pd.DataFrame, column: str, default: float = 0.0) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(default, index=frame.index, dtype="float64")
    return pd.to_numeric(frame[column], errors="coerce").fillna(default)


def _bounded_penalty(values: pd.Series, *, scale: float, clip: float) -> pd.Series:
    return (values.clip(lower=0.0) / float(scale)).clip(upper=float(clip))


def _with_effective_reward_terms(step_metrics: pd.DataFrame) -> pd.DataFrame:
    """Derive the terms that actually enter reward/loss from diagnostic columns."""

    if step_metrics.empty:
        return step_metrics
    frame = step_metrics.copy()

    private_profit_weight = _numeric(frame, "private_profit_weight", 0.02)
    frame["dispatch_private_profit_reward"] = private_profit_weight * _numeric(frame, "private_profit_proxy")

    if "dispatch_comfort_soc_penalty" not in frame.columns:
        if "scaled_comfort_soc_penalty" in frame.columns:
            scaled_comfort_soc = _numeric(frame, "scaled_comfort_soc_penalty")
        else:
            raw_comfort_soc = _numeric(frame, "comfort_penalty") + _numeric(frame, "soc_penalty")
            scaled_comfort_soc = _bounded_penalty(raw_comfort_soc, scale=100.0, clip=5.0)
        comfort_weight = _numeric(frame, "comfort_soc_weight", 0.02)
        frame["dispatch_comfort_soc_penalty"] = comfort_weight * scaled_comfort_soc

    if "dispatch_battery_degradation_penalty" not in frame.columns:
        frame["dispatch_battery_degradation_penalty"] = (
            _numeric(frame, "battery_degradation_weight", 0.01)
            * _numeric(frame, "battery_degradation_cost")
        )

    if "portfolio_profit_reward" not in frame.columns:
        frame["portfolio_profit_reward"] = 0.05 * _numeric(frame, "portfolio_window_profit")
    if "portfolio_verified_capacity_reward" not in frame.columns:
        frame["portfolio_verified_capacity_reward"] = 0.50 * _numeric(frame, "portfolio_window_verified_capacity")
    if "portfolio_contract_shortfall_penalty" not in frame.columns:
        frame["portfolio_contract_shortfall_penalty"] = _numeric(frame, "portfolio_window_contract_shortfall")
    if "portfolio_future_shield_penalty" not in frame.columns:
        frame["portfolio_future_shield_penalty"] = _numeric(frame, "portfolio_window_shield_intervention")
    if "portfolio_future_projection_penalty" not in frame.columns:
        frame["portfolio_future_projection_penalty"] = 0.50 * _numeric(frame, "portfolio_window_projection_gap")
    if "portfolio_future_comfort_soc_penalty" not in frame.columns:
        frame["portfolio_future_comfort_soc_penalty"] = 0.02 * _numeric(frame, "portfolio_window_comfort_soc_penalty")

    return frame


def reward_abs_share_table(step_metrics: pd.DataFrame) -> pd.DataFrame:
    if step_metrics.empty:
        return pd.DataFrame(columns=["algorithm", "source", "term", "mean", "mean_abs", "total_abs", "abs_share"])
    step_metrics = _with_effective_reward_terms(step_metrics)
    rows: list[dict[str, Any]] = []
    group_cols = ["algorithm"] if "algorithm" in step_metrics else []
    grouped = step_metrics.groupby(group_cols, dropna=False) if group_cols else [(None, step_metrics)]
    for group_key, group in grouped:
        algorithm = str(group_key[0] if isinstance(group_key, tuple) and len(group_key) == 1 else group_key) if group_cols else "all"
        available_terms = [(term, source) for term, source in REWARD_COMPONENT_TERMS if term in group.columns]
        total_abs = 0.0
        term_stats: list[dict[str, Any]] = []
        for term, source in available_terms:
            values = pd.to_numeric(group[term], errors="coerce").fillna(0.0)
            abs_sum = float(values.abs().sum())
            total_abs += abs_sum
            term_stats.append(
                {
                    "algorithm": algorithm,
                    "source": source,
                    "term": term,
                    "mean": float(values.mean()) if len(values) else 0.0,
                    "mean_abs": float(values.abs().mean()) if len(values) else 0.0,
                    "total_abs": abs_sum,
                }
            )
        for item in term_stats:
            item["abs_share"] = float(item["total_abs"] / total_abs) if total_abs > 0.0 else 0.0
            rows.append(item)
    return pd.DataFrame(rows).sort_values(["algorithm", "abs_share"], ascending=[True, False])


def compare_evaluation_metrics(*, legacy_dir: str | Path, new_dir: str | Path) -> pd.DataFrame:
    legacy = _read_csv(Path(legacy_dir) / "evaluation_seed_metrics.csv")
    new = _read_csv(Path(new_dir) / "evaluation_seed_metrics.csv")
    frames: list[pd.DataFrame] = []
    for label, frame in (("legacy", legacy), ("new", new)):
        if frame.empty:
            continue
        frame = frame.copy()
        frame["reward_audit_label"] = label
        frames.append(frame)
    if not frames:
        return pd.DataFrame(columns=["algorithm", "metric", "legacy_mean", "new_mean", "delta_new_minus_legacy"])
    merged = pd.concat(frames, ignore_index=True, sort=False)
    id_cols = ["reward_audit_label", "algorithm"]
    if "checkpoint_label" in merged.columns:
        id_cols.append("checkpoint_label")
    rows: list[dict[str, Any]] = []
    for metric in EVAL_COMPARE_METRICS:
        if metric not in merged.columns:
            continue
        summary = (
            merged[id_cols + [metric]]
            .assign(**{metric: pd.to_numeric(merged[metric], errors="coerce")})
            .groupby(id_cols, dropna=False)[metric]
            .mean()
            .reset_index()
        )
        pivot_cols = [col for col in id_cols if col != "reward_audit_label"]
        pivot = summary.pivot_table(index=pivot_cols, columns="reward_audit_label", values=metric, aggfunc="mean")
        for index, values in pivot.iterrows():
            index_values = index if isinstance(index, tuple) else (index,)
            row = {col: value for col, value in zip(pivot_cols, index_values)}
            legacy_value = float(values["legacy"]) if "legacy" in values and pd.notna(values["legacy"]) else None
            new_value = float(values["new"]) if "new" in values and pd.notna(values["new"]) else None
            row.update(
                {
                    "metric": metric,
                    "legacy_mean": legacy_value,
                    "new_mean": new_value,
                    "delta_new_minus_legacy": None
                    if legacy_value is None or new_value is None
                    else float(new_value - legacy_value),
                }
            )
            rows.append(row)
    return pd.DataFrame(rows)


def run_audit(*, new_dir: str | Path, output_dir: str | Path, legacy_dir: str | Path | None = None) -> dict[str, Any]:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    step_metrics = collect_step_metrics(new_dir)
    share = reward_abs_share_table(step_metrics)
    share_path = out / "reward_component_abs_share.csv"
    share.to_csv(share_path, index=False)

    comparison_path = None
    comparison_rows = 0
    if legacy_dir is not None:
        comparison = compare_evaluation_metrics(legacy_dir=legacy_dir, new_dir=new_dir)
        comparison_path = out / "legacy_vs_new_eval_comparison.csv"
        comparison.to_csv(comparison_path, index=False)
        comparison_rows = int(len(comparison))

    summary = {
        "new_dir": str(new_dir),
        "legacy_dir": None if legacy_dir is None else str(legacy_dir),
        "step_metric_rows": int(len(step_metrics)),
        "reward_component_abs_share_path": str(share_path),
        "reward_component_abs_share_rows": int(len(share)),
        "legacy_vs_new_eval_comparison_path": None if comparison_path is None else str(comparison_path),
        "legacy_vs_new_eval_comparison_rows": comparison_rows,
    }
    (out / "reward_v2_result_audit_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit reward_v2 runs: component abs-share and legacy/new metrics.")
    parser.add_argument("--new-dir", required=True)
    parser.add_argument("--legacy-dir", default=None)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    summary = run_audit(new_dir=args.new_dir, legacy_dir=args.legacy_dir, output_dir=args.output_dir)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
