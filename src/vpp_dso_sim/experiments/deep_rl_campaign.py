from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from vpp_dso_sim.experiments.algorithm_search import (
    load_algorithm_candidates,
    score_algorithm_candidates,
)
from vpp_dso_sim.learning.deep_rl import (
    PrivacySeparatedCTDEConfig,
    evaluate_privacy_separated_ctde_checkpoint,
    torch_available,
    train_privacy_separated_ctde,
)
from vpp_dso_sim.utils.io import ensure_dir, write_json


_TRUE_IMPLEMENTED = {"privacy_separated_ctde_actor_critic"}
_CTDE_ADAPTER_CAPABLE = {
    "mappo",
    "happo",
    "maddpg",
    "matd3",
    "mappo_gnn_critic",
    "masac",
    "ippo",
    "mad4pg",
}


@dataclass(frozen=True)
class DeepRLCandidateCampaignConfig:
    """Longer-budget deep-RL campaign over the MARL candidate registry.

    The campaign is intentionally strict about claim boundaries. It can run a
    real PyTorch rollout for candidates through the current
    privacy-separated-CTDE training adapter, but it does not claim that a
    candidate such as MATD3/HAPPO/HASAC is fully implemented unless the exact
    algorithm-specific update rule is present.
    """

    config_path: str | Path = "configs/european_lv_benchmark_v2.yaml"
    output_dir: str | Path = "outputs/deep_rl_candidate_campaign"
    registry_module: str = "vpp_dso_sim.learning.advanced_marl"
    min_candidates: int = 20
    top_k: int = 5
    train_top_k: int = 3
    train_all_adapters: bool = False
    execute_training: bool = True
    episodes: int = 2
    horizon_steps: int = 96
    eval_horizon_steps: int = 96
    seeds: tuple[int, ...] = (7401,)
    hidden_dim: int = 64
    learning_rate: float = 3e-4
    entropy_coef: float = 0.01
    value_coef: float = 0.50
    portfolio_reward_coef: float = 0.20

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["config_path"] = str(self.config_path)
        payload["output_dir"] = str(self.output_dir)
        payload["seeds"] = list(self.seeds)
        return payload


def _adapter_truth(candidate_id: str) -> tuple[bool, str, str]:
    if candidate_id in _TRUE_IMPLEMENTED:
        return True, "true_implemented", "Exact trainer is implemented in deep_rl.py."
    if candidate_id in _CTDE_ADAPTER_CAPABLE:
        return (
            True,
            "ctde_adapter_training",
            (
                "Runs a real PyTorch privacy-separated CTDE training loop, but the "
                "candidate-specific update rule is not yet fully implemented."
            ),
        )
    return (
        False,
        "not_yet_implemented",
        "Candidate-specific neural update rule is not implemented yet; queued for future work.",
    )


def _candidate_training_config(
    *,
    base: DeepRLCandidateCampaignConfig,
    candidate_id: str,
    seed: int,
) -> PrivacySeparatedCTDEConfig:
    lr_scale = 1.0
    entropy_coef = base.entropy_coef
    portfolio_coef = base.portfolio_reward_coef
    if candidate_id in {"mappo", "happo", "ippo"}:
        entropy_coef = max(entropy_coef, 0.015)
    if candidate_id in {"matd3", "maddpg", "mad4pg"}:
        lr_scale = 0.75
        entropy_coef = min(entropy_coef, 0.008)
    if "gnn" in candidate_id:
        lr_scale = 0.80
    if candidate_id in {"masac", "hasac"}:
        entropy_coef = max(entropy_coef, 0.025)
    return PrivacySeparatedCTDEConfig(
        algorithm=f"{candidate_id}_ctde_training_adapter",
        horizon_steps=int(base.horizon_steps),
        episodes=int(base.episodes),
        learning_rate=float(base.learning_rate) * lr_scale,
        hidden_dim=int(base.hidden_dim),
        entropy_coef=float(entropy_coef),
        value_coef=float(base.value_coef),
        portfolio_reward_coef=float(portfolio_coef),
        seed=int(seed),
    )


def _safe_sum(frame: pd.DataFrame, column: str) -> float:
    if frame.empty or column not in frame:
        return 0.0
    return float(pd.to_numeric(frame[column], errors="coerce").fillna(0.0).sum())


def _safe_mean(frame: pd.DataFrame, column: str) -> float:
    if frame.empty or column not in frame:
        return 0.0
    series = pd.to_numeric(frame[column], errors="coerce").dropna()
    return float(series.mean()) if not series.empty else 0.0


def _training_result_row(
    *,
    candidate_id: str,
    seed: int,
    status: str,
    truth_level: str,
    reason: str,
    train_output_dir: Path | None = None,
    train_result: dict[str, Any] | None = None,
    eval_result: dict[str, Any] | None = None,
) -> dict[str, Any]:
    train_summary = (train_result or {}).get("summary", {})
    eval_summary = (eval_result or {}).get("summary", {})
    trajectory = (train_result or {}).get("trajectory", pd.DataFrame())
    episode_metrics = (train_result or {}).get("episode_metrics", pd.DataFrame())
    private_profit_total = _safe_sum(trajectory, "private_profit_proxy")
    private_profit_mean = _safe_mean(trajectory, "private_profit_proxy")
    positive_profit_rate = 0.0
    if not trajectory.empty and "private_profit_proxy" in trajectory:
        profit = pd.to_numeric(trajectory["private_profit_proxy"], errors="coerce").fillna(0.0)
        positive_profit_rate = float((profit > 0.0).mean())
    return {
        "candidate_id": candidate_id,
        "seed": int(seed),
        "status": status,
        "truth_level": truth_level,
        "reason": reason,
        "train_output_dir": str(train_output_dir or ""),
        "episodes": train_summary.get("episodes"),
        "horizon_steps": train_summary.get("horizon_steps"),
        "best_episode_reward": train_summary.get("best_episode_reward"),
        "final_episode_reward": train_summary.get("final_episode_reward"),
        "episode_reward_max": _safe_mean(episode_metrics, "episode_reward")
        if not episode_metrics.empty
        else None,
        "train_total_private_profit_proxy": private_profit_total,
        "mean_private_profit_proxy": private_profit_mean,
        "positive_private_profit_step_rate": positive_profit_rate,
        "eval_total_reward": eval_summary.get("total_reward"),
        "eval_total_cost": eval_summary.get("total_cost"),
        "eval_total_private_profit_proxy": eval_summary.get("eval_total_private_profit_proxy"),
        "eval_positive_private_profit_step_rate": eval_summary.get(
            "eval_positive_private_profit_step_rate"
        ),
        "eval_total_violation_count": eval_summary.get("total_violation_count"),
        "eval_total_projection_gap_mw": eval_summary.get("total_projection_gap_mw"),
        "checkpoint": train_summary.get("checkpoint"),
    }


def _status_class(status: str) -> str:
    if status == "trained":
        return "ok"
    if status == "queued":
        return "wait"
    return "hold"


def _write_campaign_html(
    *,
    output_dir: Path,
    plan: pd.DataFrame,
    results: pd.DataFrame,
    summary: dict[str, Any],
) -> Path:
    top_rows = results.sort_values(
        ["status", "positive_private_profit_step_rate", "final_episode_reward"],
        ascending=[True, False, False],
    ).head(20)
    plan_rows = plan.head(50)

    def table(frame: pd.DataFrame) -> str:
        if frame.empty:
            return "<p class='muted'>No rows / 暂无数据</p>"
        cols = list(frame.columns)
        head = "".join(f"<th>{col}</th>" for col in cols)
        body = []
        for _, row in frame.iterrows():
            cells = "".join(f"<td>{row.get(col, '')}</td>" for col in cols)
            klass = _status_class(str(row.get("status", "")))
            body.append(f"<tr class='{klass}'>{cells}</tr>")
        return f"<div class='table-wrap'><table><thead><tr>{head}</tr></thead><tbody>{''.join(body)}</tbody></table></div>"

    html_text = f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Deep RL Candidate Campaign</title>
  <style>
    body {{ margin:0; font-family:"Segoe UI","Microsoft YaHei",Arial,sans-serif; background:#f3f7fb; color:#132235; }}
    header {{ padding:28px 34px; color:white; background:linear-gradient(135deg,#062238,#185b87); }}
    main {{ padding:22px 30px 40px; }}
    .panel {{ background:white; border:1px solid #d9e5ef; border-radius:8px; padding:18px; margin-bottom:16px; box-shadow:0 12px 30px rgba(12,34,55,0.08); }}
    .metric-grid {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(210px,1fr)); gap:12px; }}
    .metric {{ background:white; border:1px solid #d9e5ef; border-radius:8px; padding:14px; }}
    .metric strong {{ display:block; font-size:24px; }}
    .metric span {{ color:#5d7084; font-size:12px; font-weight:800; }}
    .table-wrap {{ overflow:auto; border:1px solid #d9e5ef; border-radius:8px; }}
    table {{ border-collapse:collapse; width:100%; font-size:12px; background:white; }}
    th,td {{ padding:8px 9px; border-bottom:1px solid #e8eef4; text-align:left; vertical-align:top; }}
    th {{ background:#edf5fb; position:sticky; top:0; }}
    tr.ok td:first-child {{ border-left:5px solid #12805c; }}
    tr.wait td:first-child {{ border-left:5px solid #c98300; }}
    tr.hold td:first-child {{ border-left:5px solid #8796a8; }}
    .muted {{ color:#5d7084; }}
  </style>
</head>
<body>
  <header>
    <p>Deep RL Candidate Campaign / 深度强化学习候选算法实验</p>
    <h1>真实训练状态、收益探索与算法实现边界</h1>
    <p>本页面同步展示所有候选算法的训练状态。`ctde_adapter_training` 表示已经运行真实 PyTorch 训练闭环，但还不是该候选算法的完整专用实现。</p>
  </header>
  <main>
    <section class="metric-grid">
      <div class="metric"><span>候选总数</span><strong>{summary.get("candidate_count", 0)}</strong></div>
      <div class="metric"><span>已训练候选</span><strong>{summary.get("trained_count", 0)}</strong></div>
      <div class="metric"><span>暂未实现候选</span><strong>{summary.get("not_yet_implemented_count", 0)}</strong></div>
      <div class="metric"><span>最高正收益步占比</span><strong>{summary.get("best_positive_private_profit_step_rate", 0.0):.3f}</strong></div>
    </section>
    <section class="panel">
      <h2>训练结果 / Training Results</h2>
      {table(top_rows)}
    </section>
    <section class="panel">
      <h2>全候选训练计划 / Candidate Training Plan</h2>
      {table(plan_rows)}
    </section>
    <section class="panel">
      <h2>Claim Boundary / 结论边界</h2>
      <p>{summary.get("claim_boundary", "")}</p>
    </section>
  </main>
</body>
</html>
"""
    path = output_dir / "deep_rl_candidate_campaign.html"
    path.write_text(html_text, encoding="utf-8")
    return path


def run_deep_rl_candidate_campaign(
    config: DeepRLCandidateCampaignConfig | None = None,
) -> dict[str, Any]:
    cfg = config or DeepRLCandidateCampaignConfig()
    out = ensure_dir(cfg.output_dir)
    candidates, registry_report = load_algorithm_candidates(
        cfg.registry_module,
        min_candidates=cfg.min_candidates,
    )
    scores = score_algorithm_candidates(candidates, top_k=cfg.top_k)
    if scores.empty:
        raise ValueError("No MARL candidates were loaded for the deep-RL campaign.")

    plan_rows: list[dict[str, Any]] = []
    for row in scores.to_dict(orient="records"):
        candidate_id = str(row["algorithm_id"])
        can_train, truth_level, reason = _adapter_truth(candidate_id)
        selected_for_training = bool(can_train) and (
            bool(cfg.train_all_adapters) or int(row["rank"]) <= int(cfg.train_top_k)
        )
        plan_rows.append(
            {
                **row,
                "can_execute_current_training": bool(can_train),
                "selected_for_training": selected_for_training,
                "truth_level": truth_level,
                "training_adapter_reason": reason,
            }
        )
    plan = pd.DataFrame(plan_rows)
    plan.to_csv(out / "candidate_training_plan.csv", index=False)

    result_rows: list[dict[str, Any]] = []
    if not torch_available():
        for row in plan.to_dict(orient="records"):
            result_rows.append(
                _training_result_row(
                    candidate_id=str(row["algorithm_id"]),
                    seed=int(cfg.seeds[0]),
                    status="not_run",
                    truth_level=str(row["truth_level"]),
                    reason="PyTorch is not available in the active environment.",
                )
            )
    else:
        for row in plan.to_dict(orient="records"):
            candidate_id = str(row["algorithm_id"])
            if not bool(row["selected_for_training"]) or not cfg.execute_training:
                status = "queued" if bool(row["can_execute_current_training"]) else "not_implemented"
                result_rows.append(
                    _training_result_row(
                        candidate_id=candidate_id,
                        seed=int(cfg.seeds[0]),
                        status=status,
                        truth_level=str(row["truth_level"]),
                        reason=str(row["training_adapter_reason"]),
                    )
                )
                continue
            for seed in cfg.seeds:
                run_dir = ensure_dir(out / "runs" / f"{candidate_id}_seed_{seed}")
                train_cfg = _candidate_training_config(base=cfg, candidate_id=candidate_id, seed=int(seed))
                train_result = train_privacy_separated_ctde(
                    config_path=cfg.config_path,
                    output_dir=run_dir / "train",
                    config=train_cfg,
                )
                eval_result = evaluate_privacy_separated_ctde_checkpoint(
                    config_path=cfg.config_path,
                    checkpoint_path=train_result["checkpoint"],
                    output_dir=run_dir / "frozen_eval",
                    horizon_steps=int(cfg.eval_horizon_steps),
                    seed=int(seed) + 10_000,
                )
                result_rows.append(
                    _training_result_row(
                        candidate_id=candidate_id,
                        seed=int(seed),
                        status="trained",
                        truth_level=str(row["truth_level"]),
                        reason=str(row["training_adapter_reason"]),
                        train_output_dir=run_dir,
                        train_result=train_result,
                        eval_result=eval_result,
                    )
                )

    results = pd.DataFrame(result_rows)
    results.to_csv(out / "candidate_training_results.csv", index=False)
    trained = results[results.get("status", "") == "trained"] if not results.empty else pd.DataFrame()
    positive_rates = pd.to_numeric(
        trained.get("positive_private_profit_step_rate", pd.Series(dtype=float)),
        errors="coerce",
    ).fillna(0.0)
    summary = {
        "experiment": "deep_rl_candidate_campaign",
        "config": cfg.to_dict(),
        "registry": registry_report.to_dict(),
        "candidate_count": int(len(plan)),
        "trained_count": int((results.get("status", "") == "trained").sum()) if not results.empty else 0,
        "queued_count": int((results.get("status", "") == "queued").sum()) if not results.empty else 0,
        "not_yet_implemented_count": int((results.get("status", "") == "not_implemented").sum())
        if not results.empty
        else 0,
        "best_positive_private_profit_step_rate": float(positive_rates.max()) if not positive_rates.empty else 0.0,
        "best_candidate_by_positive_profit": str(
            trained.iloc[int(positive_rates.argmax())]["candidate_id"]
        )
        if not trained.empty and not positive_rates.empty
        else "",
        "claim_boundary": (
            "This campaign runs real PyTorch training only for candidates supported by the current "
            "privacy-separated CTDE adapter. Rows marked ctde_adapter_training are not full "
            "candidate-specific implementations such as true MATD3/HAPPO/HASAC."
        ),
    }
    write_json(out / "campaign_summary.json", summary)
    report_path = _write_campaign_html(output_dir=out, plan=plan, results=results, summary=summary)
    return {
        "output_dir": out,
        "plan": plan,
        "results": results,
        "summary": summary,
        "report": report_path,
    }


__all__ = ["DeepRLCandidateCampaignConfig", "run_deep_rl_candidate_campaign"]
