from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


_SHARD_RE = re.compile(r"^seed_(?P<seed>\d+)_(?P<case>.+)$")
_RUN_RE = re.compile(r"^happo_(?P<case>.+)_train_.*_seed_(?P<seed>\d+)$")
_HAPPO_LOG_RE = re.compile(
    r"\[HAPPO\]\s+episode=(?P<episode>\d+)/(?P<episodes>\d+)\s+"
    r"reward=(?P<reward>[-+0-9.eE]+)\s+"
    r"cost=(?P<cost>[-+0-9.eE]+)\s+"
    r"violations=(?P<violations>\d+)\s+"
    r"projection_gap_mw=(?P<projection_gap_mw>[-+0-9.eE]+)\s+"
    r"critic_loss=(?P<critic_loss>[-+0-9.eE]+)\s+"
    r"dso_loss=(?P<dso_loss>[-+0-9.eE]+)\s+"
    r"dispatch_loss=(?P<dispatch_loss>[-+0-9.eE]+)\s+"
    r"portfolio_loss=(?P<portfolio_loss>[-+0-9.eE]+)"
)


def _finite_or_none(value: Any) -> float | None:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return numeric if math.isfinite(numeric) else None


def _column_mean(frame: pd.DataFrame, column: str) -> float | None:
    if column not in frame or frame.empty:
        return None
    values = pd.to_numeric(frame[column], errors="coerce")
    if values.dropna().empty:
        return None
    return _finite_or_none(values.mean())


def _column_sum(frame: pd.DataFrame, column: str) -> float | None:
    if column not in frame or frame.empty:
        return None
    values = pd.to_numeric(frame[column], errors="coerce")
    if values.dropna().empty:
        return None
    return _finite_or_none(values.sum())


def _numeric_has_nan_or_inf(frame: pd.DataFrame) -> bool:
    numeric = frame.select_dtypes(include=[np.number])
    if numeric.empty:
        return False
    values = numeric.to_numpy(dtype=float)
    return bool(np.isnan(values).any() or np.isinf(values).any())


def _parse_seed_case(metric_path: Path) -> tuple[int | None, str]:
    shard_dir = metric_path.parents[3].name if len(metric_path.parents) >= 4 else ""
    run_id = metric_path.parents[1].name if len(metric_path.parents) >= 2 else ""
    shard_match = _SHARD_RE.match(shard_dir)
    if shard_match:
        return int(shard_match.group("seed")), shard_match.group("case")
    run_match = _RUN_RE.match(run_id)
    if run_match:
        return int(run_match.group("seed")), run_match.group("case")
    return None, "unknown"


def _parse_seed_case_from_shard_dir(path: Path) -> tuple[int | None, str]:
    for part in reversed(path.parts):
        shard_match = _SHARD_RE.match(part)
        if shard_match:
            return int(shard_match.group("seed")), shard_match.group("case")
    return None, "unknown"


def _summarize_metric_file(metric_path: Path) -> dict[str, Any]:
    seed, hparam_case = _parse_seed_case(metric_path)
    run_dir = metric_path.parent
    run_id = metric_path.parents[1].name if len(metric_path.parents) >= 2 else run_dir.name
    episode_metrics = pd.read_csv(metric_path)
    update_path = run_dir / "happo_update_metrics.csv"
    update_metrics = pd.read_csv(update_path) if update_path.exists() else pd.DataFrame()

    rewards = pd.to_numeric(episode_metrics.get("episode_reward", pd.Series(dtype=float)), errors="coerce")
    final_reward = _finite_or_none(rewards.dropna().iloc[-1]) if not rewards.dropna().empty else None
    best_reward = _finite_or_none(rewards.max()) if not rewards.dropna().empty else None
    mean_reward = _finite_or_none(rewards.mean()) if not rewards.dropna().empty else None
    reward_std = _finite_or_none(rewards.std(ddof=0)) if len(rewards.dropna()) > 1 else 0.0
    final_cost = None
    if "total_cost" in episode_metrics and not episode_metrics["total_cost"].dropna().empty:
        final_cost = _finite_or_none(pd.to_numeric(episode_metrics["total_cost"], errors="coerce").dropna().iloc[-1])

    return {
        "run_id": run_id,
        "run_dir": str(run_dir),
        "source": "metric_csv",
        "seed": seed,
        "hparam_case": hparam_case,
        "completed": bool(len(episode_metrics) > 0),
        "episode_count": int(len(episode_metrics)),
        "final_reward": final_reward,
        "mean_reward": mean_reward,
        "reward_std": reward_std,
        "best_reward": best_reward,
        "final_total_cost": final_cost,
        "total_violations": _column_sum(episode_metrics, "violation_count"),
        "projection_gap_mw_sum": _column_sum(episode_metrics, "projection_gap_mw"),
        "mean_critic_loss": _column_mean(episode_metrics, "critic_loss"),
        "mean_policy_loss": _column_mean(update_metrics, "policy_loss"),
        "mean_entropy": _column_mean(update_metrics, "entropy_mean"),
        "mean_approx_kl": _column_mean(update_metrics, "approx_kl"),
        "nan_or_inf": bool(_numeric_has_nan_or_inf(episode_metrics) or _numeric_has_nan_or_inf(update_metrics)),
        "update_metric_rows": int(len(update_metrics)),
    }


def _summarize_log_file(log_path: Path) -> dict[str, Any] | None:
    seed, hparam_case = _parse_seed_case_from_shard_dir(log_path)
    rows: list[dict[str, Any]] = []
    for line in log_path.read_text(encoding="utf-8", errors="replace").splitlines():
        match = _HAPPO_LOG_RE.search(line)
        if not match:
            continue
        item = match.groupdict()
        rows.append(
            {
                "episode": int(item["episode"]),
                "episode_reward": float(item["reward"]),
                "total_cost": float(item["cost"]),
                "violation_count": int(item["violations"]),
                "projection_gap_mw": float(item["projection_gap_mw"]),
                "critic_loss": float(item["critic_loss"]),
                "dso_loss": float(item["dso_loss"]),
                "dispatch_loss": float(item["dispatch_loss"]),
                "portfolio_loss": float(item["portfolio_loss"]),
            }
        )
    if not rows:
        return None
    frame = pd.DataFrame(rows)
    rewards = pd.to_numeric(frame["episode_reward"], errors="coerce")
    return {
        "run_id": log_path.parent.parent.name,
        "run_dir": str(log_path.parent.parent),
        "source": "stdout_log",
        "seed": seed,
        "hparam_case": hparam_case,
        "completed": True,
        "episode_count": int(len(frame)),
        "final_reward": _finite_or_none(rewards.dropna().iloc[-1]) if not rewards.dropna().empty else None,
        "mean_reward": _finite_or_none(rewards.mean()),
        "reward_std": _finite_or_none(rewards.std(ddof=0)) if len(rewards.dropna()) > 1 else 0.0,
        "best_reward": _finite_or_none(rewards.max()),
        "final_total_cost": _finite_or_none(pd.to_numeric(frame["total_cost"], errors="coerce").dropna().iloc[-1]),
        "total_violations": _column_sum(frame, "violation_count"),
        "projection_gap_mw_sum": _column_sum(frame, "projection_gap_mw"),
        "mean_critic_loss": _column_mean(frame, "critic_loss"),
        "mean_dso_loss": _column_mean(frame, "dso_loss"),
        "mean_dispatch_loss": _column_mean(frame, "dispatch_loss"),
        "mean_portfolio_loss": _column_mean(frame, "portfolio_loss"),
        "mean_policy_loss": None,
        "mean_entropy": None,
        "mean_approx_kl": None,
        "nan_or_inf": _numeric_has_nan_or_inf(frame),
        "update_metric_rows": 0,
    }


def summarize_probe_root(root: str | Path, output_dir: str | Path | None = None) -> pd.DataFrame:
    """Summarize short HAPPO shard-probe outputs under ``root``."""

    root_path = Path(root)
    rows = [
        _summarize_metric_file(metric_path)
        for metric_path in sorted(root_path.glob("*/runs/*/train/happo_episode_metrics.csv"))
    ]
    metric_shards = {Path(str(row["run_dir"])).parents[2].name for row in rows if "run_dir" in row}
    for log_path in sorted(root_path.glob("*/logs/paper_long_stdout.log")):
        if log_path.parent.parent.name in metric_shards:
            continue
        row = _summarize_log_file(log_path)
        if row is not None:
            rows.append(row)
    summary = pd.DataFrame(rows)
    if output_dir is not None:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        summary.to_csv(out / "happo_probe_summary.csv", index=False)
        payload = json.loads(summary.to_json(orient="records"))
        (out / "happo_probe_summary.json").write_text(
            json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize short HAPPO paper-long shard probe outputs.")
    parser.add_argument("root", type=Path)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()
    output_dir = args.output_dir or (args.root / "probe_summary")
    summary = summarize_probe_root(args.root, output_dir=output_dir)
    print(summary.to_string(index=False) if not summary.empty else "No completed HAPPO episode metric files found.")
    print(f"Summary directory: {output_dir}")


if __name__ == "__main__":
    main()
