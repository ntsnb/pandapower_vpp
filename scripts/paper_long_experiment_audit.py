from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[1]
MAIN_RUN = ROOT / "outputs" / "paper_training_long_current"
ACGATE_RUN = ROOT / "outputs" / "paper_training_long_20260513_acgate"
SENS_PRE_RUN = ROOT / "outputs" / "paper_training_long_sensitivity_v1_preflight_smoke"
SENS_PROGRESS_RUN = ROOT / "outputs" / "paper_training_long_sensitivity_v1_20260528_thread8_progress"
OUT = ROOT / "outputs" / "paper_long_experiment_audit_20260528"
FIG = OUT / "figures"
TABLE = OUT / "tables"


def ensure_dirs() -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    TABLE.mkdir(parents=True, exist_ok=True)


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    try:
        return pd.read_csv(path, low_memory=False)
    except pd.errors.ParserError:
        return pd.read_csv(path, on_bad_lines="skip", engine="python")


def progress_text_summary(path: Path) -> dict[str, Any]:
    if not path.exists() or path.stat().st_size == 0:
        return {"rows": 0, "latest_phase": "", "latest_run": "", "has_done": False}
    lines = [line.rstrip("\n") for line in path.read_text(encoding="utf-8", errors="replace").splitlines() if line.strip()]
    data_lines = lines[1:] if lines and lines[0].startswith("timestamp,") else lines
    latest = data_lines[-1] if data_lines else ""
    fields = latest.split(",")
    return {
        "rows": len(data_lines),
        "latest_phase": fields[1] if len(fields) > 1 else "",
        "latest_run": fields[3] if len(fields) > 3 else "",
        "has_done": any(",done," in line for line in data_lines),
    }


def savefig(name: str) -> str:
    path = FIG / name
    plt.tight_layout()
    plt.savefig(path, dpi=180, bbox_inches="tight")
    plt.close()
    return str(path.relative_to(ROOT))


def summarize_run_dirs() -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for run_dir in sorted((ROOT / "outputs").glob("*")):
        if not run_dir.is_dir():
            continue
        name = run_dir.name
        if name.startswith("paper_long_experiment_audit_"):
            continue
        if not any(token in name for token in ("paper_training_long", "paper_long")):
            continue
        manifest = read_json(run_dir / "experiment_manifest.json")
        progress_summary = progress_text_summary(run_dir / "experiment_progress.csv")
        run_index = read_csv(run_dir / "run_index.csv")
        eval_metrics = read_csv(run_dir / "evaluation_seed_metrics.csv")
        episode_metrics = read_csv(run_dir / "training_episode_metrics.csv")
        loss_metrics_path = run_dir / "training_loss_metrics.csv"
        status = "unknown"
        if name == "paper_training_long_current":
            status = "completed_legacy_paper_long"
        elif name == "paper_training_long_20260513_acgate":
            status = "baseline_gate_manifest_only_or_empty_tables"
        elif "preflight_smoke" in name or "guardrail_smoke" in name:
            status = "smoke_not_paper_result"
        elif name in {"paper_training_long_20260512_fresh", "paper_training_long_20260512_rerun"}:
            status = "partial_incomplete"
        elif name == "paper_training_long_wrapper_test":
            status = "wrapper_test_not_campaign"
        elif "sensitivity_v1_20260528" in name:
            status = "incomplete_progress_only"
        elif not run_index.empty and not eval_metrics.empty:
            status = "completed_or_candidate"
        rows.append(
            {
                "run_dir": str(run_dir.relative_to(ROOT)),
                "status": status,
                "preset": manifest.get("config", {}).get("preset", manifest.get("preset", "")),
                "phase": manifest.get("phase", ""),
                "algorithms": ",".join(manifest.get("config", {}).get("algorithms", []))
                if manifest.get("config")
                else "",
                "seeds": ",".join(str(v) for v in manifest.get("config", {}).get("seeds", []))
                if manifest.get("config")
                else "",
                "horizon_steps": manifest.get("config", {}).get("horizon_steps", ""),
                "eval_horizon_steps": manifest.get("config", {}).get("eval_horizon_steps", ""),
                "train_episodes": manifest.get("config", {}).get("train_episodes", ""),
                "run_index_rows": max(0, len(run_index)),
                "eval_rows": max(0, len(eval_metrics)),
                "episode_rows": max(0, len(episode_metrics)),
                "loss_file_exists": loss_metrics_path.exists(),
                "loss_file_size_mb": round(loss_metrics_path.stat().st_size / 1024 / 1024, 2)
                if loss_metrics_path.exists()
                else 0.0,
                "progress_rows": int(progress_summary["rows"]),
                "latest_progress_phase": progress_summary["latest_phase"],
                "latest_progress_run": progress_summary["latest_run"],
                "progress_has_done_event": bool(progress_summary["has_done"]),
            }
        )
    frame = pd.DataFrame(rows)
    frame.to_csv(TABLE / "paper_long_run_inventory.csv", index=False)
    return frame


def make_episode_plots(episodes: pd.DataFrame) -> list[str]:
    paths: list[str] = []
    if episodes.empty:
        return paths
    episodes = episodes.copy()
    for col in ("episode", "episode_reward", "episode_cost", "critic_loss"):
        if col in episodes:
            episodes[col] = pd.to_numeric(episodes[col], errors="coerce")

    group_cols = ["algorithm", "hparam_case", "episode"]
    numeric_cols = [
        col
        for col in (
            "episode_reward",
            "episode_cost",
            "violation_count",
            "projection_gap_mw",
            "critic_loss",
            "dso_policy_loss",
            "dispatch_policy_loss",
            "portfolio_policy_loss",
        )
        if col in episodes
    ]
    curves = episodes.groupby(group_cols, dropna=False)[numeric_cols].mean().reset_index()
    curves.to_csv(TABLE / "training_episode_curves_by_algorithm_hparam.csv", index=False)

    plt.figure(figsize=(12, 7))
    for (alg, case), group in curves.groupby(["algorithm", "hparam_case"], dropna=False):
        if "episode_reward" not in group:
            continue
        series = group.sort_values("episode")
        y = series["episode_reward"].rolling(7, min_periods=1).mean()
        plt.plot(series["episode"], y, linewidth=1.6, label=f"{alg} / {case}")
    plt.xlabel("Episode")
    plt.ylabel("Mean episode reward, rolling-7")
    plt.title("Paper-long training reward curves")
    plt.grid(alpha=0.25)
    plt.legend(fontsize=7, ncol=2)
    paths.append(savefig("01_training_reward_curves.png"))

    loss_cols = [c for c in ("critic_loss", "dso_policy_loss", "dispatch_policy_loss", "portfolio_policy_loss") if c in curves]
    if loss_cols:
        fig, axes = plt.subplots(len(loss_cols), 1, figsize=(12, 2.8 * len(loss_cols)), sharex=True)
        if len(loss_cols) == 1:
            axes = [axes]
        alg_curves = curves.groupby(["algorithm", "episode"], dropna=False)[loss_cols].mean().reset_index()
        for ax, col in zip(axes, loss_cols):
            for alg, group in alg_curves.groupby("algorithm", dropna=False):
                series = group.sort_values("episode")
                ax.plot(series["episode"], series[col].rolling(7, min_periods=1).mean(), label=str(alg))
            ax.set_ylabel(col)
            ax.grid(alpha=0.25)
        axes[-1].set_xlabel("Episode")
        axes[0].set_title("Episode-level loss traces")
        axes[0].legend(fontsize=8, ncol=3)
        paths.append(savefig("02_episode_level_loss_traces.png"))

    return paths


def aggregate_loss_metrics(loss_path: Path) -> pd.DataFrame:
    if not loss_path.exists():
        return pd.DataFrame()
    header = pd.read_csv(loss_path, nrows=0).columns.tolist()
    wanted = [
        "episode",
        "global_step",
        "role",
        "policy_loss",
        "ratio_mean",
        "correction_mean",
        "grad_norm",
        "entropy_mean",
        "approx_kl",
        "target_kl_exceeded",
        "run_id",
        "algorithm",
        "seed",
        "hparam_case",
        "critic_update",
        "actor_update",
        "alpha_update",
        "role_critic_loss",
        "dso_critic_loss",
        "dispatch_critic_loss",
        "actor_loss",
        "dso_actor_loss",
        "dispatch_actor_loss",
        "critic_loss",
        "alpha_loss",
        "alpha_dso",
        "alpha_dispatch",
        "critic_grad_norm",
        "actor_grad_norm",
    ]
    usecols = [c for c in wanted if c in header]
    chunks: list[pd.DataFrame] = []
    for chunk in pd.read_csv(loss_path, usecols=usecols, chunksize=350_000, low_memory=False):
        for col in usecols:
            if col not in {"role", "run_id", "algorithm", "hparam_case"}:
                chunk[col] = pd.to_numeric(chunk[col], errors="coerce")
        if "role" in chunk:
            chunk["role_family"] = chunk["role"].astype(str)
            chunk.loc[chunk["role_family"].str.contains("_dispatch", na=False), "role_family"] = "vpp_dispatch"
            chunk.loc[chunk["role_family"].str.contains("_portfolio", na=False), "role_family"] = "vpp_portfolio"
        else:
            chunk["role_family"] = ""
        if "episode" in chunk:
            chunk["x_episode"] = chunk["episode"]
        elif "global_step" in chunk:
            chunk["x_episode"] = np.floor(chunk["global_step"] / 672.0)
        else:
            chunk["x_episode"] = np.nan
        group_cols = [c for c in ("algorithm", "hparam_case", "role_family", "x_episode") if c in chunk]
        num_cols = [
            c
            for c in (
                "policy_loss",
                "ratio_mean",
                "correction_mean",
                "grad_norm",
                "entropy_mean",
                "approx_kl",
                "target_kl_exceeded",
                "role_critic_loss",
                "dso_critic_loss",
                "dispatch_critic_loss",
                "actor_loss",
                "dso_actor_loss",
                "dispatch_actor_loss",
                "critic_loss",
                "alpha_loss",
                "alpha_dso",
                "alpha_dispatch",
                "critic_grad_norm",
                "actor_grad_norm",
            )
            if c in chunk
        ]
        if not group_cols or not num_cols:
            continue
        grouped = chunk.groupby(group_cols, dropna=False)[num_cols].agg(["sum", "count"]).reset_index()
        grouped.columns = [
            "_".join([str(part) for part in col if part])
            if isinstance(col, tuple)
            else str(col)
            for col in grouped.columns
        ]
        chunks.append(grouped)
    if not chunks:
        return pd.DataFrame()
    merged = pd.concat(chunks, ignore_index=True)
    key_cols = [c for c in ("algorithm", "hparam_case", "role_family", "x_episode") if c in merged]
    sum_cols = [c for c in merged.columns if c.endswith("_sum")]
    count_cols = [c for c in merged.columns if c.endswith("_count")]
    agg = merged.groupby(key_cols, dropna=False)[sum_cols + count_cols].sum().reset_index()
    for sum_col in sum_cols:
        base = sum_col[: -len("_sum")]
        count_col = f"{base}_count"
        if count_col in agg:
            agg[base] = agg[sum_col] / agg[count_col].replace(0, np.nan)
    keep = key_cols + [
        c
        for c in (
            "policy_loss",
            "ratio_mean",
            "correction_mean",
            "grad_norm",
            "entropy_mean",
            "approx_kl",
            "target_kl_exceeded",
            "role_critic_loss",
            "dso_critic_loss",
            "dispatch_critic_loss",
            "actor_loss",
            "dso_actor_loss",
            "dispatch_actor_loss",
            "critic_loss",
            "alpha_loss",
            "alpha_dso",
            "alpha_dispatch",
            "critic_grad_norm",
            "actor_grad_norm",
        )
        if c in agg
    ]
    result = agg[keep].sort_values(key_cols).reset_index(drop=True)
    result.to_csv(TABLE / "training_loss_curves_aggregated.csv", index=False)
    return result


def make_loss_plots(loss_curves: pd.DataFrame) -> list[str]:
    paths: list[str] = []
    if loss_curves.empty:
        return paths
    loss_curves = loss_curves.copy()
    loss_curves["x_episode"] = pd.to_numeric(loss_curves["x_episode"], errors="coerce")

    happo = loss_curves[loss_curves["algorithm"].astype(str).str.contains("happo", na=False)]
    if not happo.empty and "policy_loss" in happo:
        plt.figure(figsize=(12, 7))
        for (case, role), group in happo.groupby(["hparam_case", "role_family"], dropna=False):
            series = group.sort_values("x_episode")
            y = series["policy_loss"].rolling(7, min_periods=1).mean()
            plt.plot(series["x_episode"], y, linewidth=1.4, label=f"{case} / {role}")
        plt.xlabel("Episode")
        plt.ylabel("Mean HAPPO policy loss")
        plt.title("HAPPO role policy-loss curves")
        plt.grid(alpha=0.25)
        plt.legend(fontsize=7, ncol=2)
        paths.append(savefig("03_happo_role_policy_loss.png"))

    continuous = loss_curves[
        loss_curves["algorithm"].astype(str).str.contains("matd3|hasac", case=False, regex=True, na=False)
    ]
    cols = [c for c in ("critic_loss", "role_critic_loss", "actor_loss", "alpha_loss") if c in continuous]
    if not continuous.empty and cols:
        fig, axes = plt.subplots(len(cols), 1, figsize=(12, 2.8 * len(cols)), sharex=True)
        if len(cols) == 1:
            axes = [axes]
        for ax, col in zip(axes, cols):
            for (alg, case), group in continuous.groupby(["algorithm", "hparam_case"], dropna=False):
                series = group.sort_values("x_episode")
                ax.plot(series["x_episode"], series[col].rolling(7, min_periods=1).mean(), label=f"{alg}/{case}")
            ax.set_ylabel(col)
            ax.grid(alpha=0.25)
        axes[0].set_title("Off-policy continuous-control update losses")
        axes[0].legend(fontsize=7, ncol=2)
        axes[-1].set_xlabel("Approx. episode index")
        paths.append(savefig("04_matd3_hasac_update_losses.png"))

    stab_cols = [c for c in ("entropy_mean", "approx_kl", "grad_norm", "target_kl_exceeded") if c in happo]
    if not happo.empty and stab_cols:
        fig, axes = plt.subplots(len(stab_cols), 1, figsize=(12, 2.7 * len(stab_cols)), sharex=True)
        if len(stab_cols) == 1:
            axes = [axes]
        collapsed = happo.groupby(["hparam_case", "x_episode"], dropna=False)[stab_cols].mean().reset_index()
        for ax, col in zip(axes, stab_cols):
            for case, group in collapsed.groupby("hparam_case", dropna=False):
                series = group.sort_values("x_episode")
                ax.plot(series["x_episode"], series[col].rolling(7, min_periods=1).mean(), label=str(case))
            ax.set_ylabel(col)
            ax.grid(alpha=0.25)
        axes[0].set_title("HAPPO stability diagnostics")
        axes[0].legend(fontsize=8, ncol=2)
        axes[-1].set_xlabel("Episode")
        paths.append(savefig("05_happo_stability_diagnostics.png"))
    return paths


def make_eval_plots(eval_metrics: pd.DataFrame, aggregate: pd.DataFrame) -> list[str]:
    paths: list[str] = []
    if eval_metrics.empty:
        return paths
    frame = eval_metrics.copy()
    for col in ("eval_total_reward", "eval_total_cost", "total_violation_cells", "security_pass"):
        if col in frame:
            frame[col] = pd.to_numeric(frame[col], errors="coerce")
    summary = frame.groupby(["algorithm", "hparam_case"], dropna=False).agg(
        eval_total_reward_mean=("eval_total_reward", "mean"),
        eval_total_reward_std=("eval_total_reward", "std"),
        eval_total_reward_count=("eval_total_reward", "count"),
        eval_total_cost_mean=("eval_total_cost", "mean"),
        eval_total_cost_std=("eval_total_cost", "std"),
        eval_total_cost_count=("eval_total_cost", "count"),
        total_violation_cells_mean=("total_violation_cells", "mean"),
        total_violation_cells_std=("total_violation_cells", "std"),
        total_violation_cells_count=("total_violation_cells", "count"),
        security_pass_mean=("security_pass", "mean"),
        security_pass_std=("security_pass", "std"),
        security_pass_count=("security_pass", "count"),
    ).reset_index(
    )
    summary.to_csv(TABLE / "frozen_eval_summary_by_algorithm_hparam.csv", index=False)

    for metric, name, ylabel in (
        ("eval_total_reward", "06_frozen_eval_reward.png", "Mean eval total reward"),
        ("eval_total_cost", "07_frozen_eval_cost.png", "Mean eval total cost"),
        ("total_violation_cells", "08_frozen_eval_violations.png", "Mean violation cells"),
    ):
        plt.figure(figsize=(12, 6))
        means = frame.groupby(["algorithm", "hparam_case"], dropna=False)[metric].mean().sort_values()
        labels = [f"{idx[0]}\n{idx[1]}" for idx in means.index]
        plt.bar(range(len(means)), means.values)
        plt.xticks(range(len(means)), labels, rotation=45, ha="right", fontsize=8)
        plt.ylabel(ylabel)
        plt.title(metric)
        plt.grid(axis="y", alpha=0.25)
        paths.append(savefig(name))
    if not aggregate.empty:
        aggregate.to_csv(TABLE / "aggregate_metrics_copy.csv", index=False)
    return paths


def build_reward_hparam_table() -> pd.DataFrame:
    base_config = yaml.safe_load((ROOT / "configs" / "european_lv_benchmark_v2.yaml").read_text(encoding="utf-8"))
    sensitivity_config = yaml.safe_load(
        (ROOT / "configs" / "european_lv_benchmark_v2_sensitivity_attention_v1.yaml").read_text(encoding="utf-8")
    )
    rows = [
        {
            "scope": "VPP dispatch reward",
            "name": "FLEXIBILITY_SERVICE_PRICE_MULTIPLIER",
            "value": 1.00,
            "meaning_zh": "灵活性服务支付代理使用 price_profile 的倍数；当前不是真实市场结算价。",
            "source": "src/vpp_dso_sim/learning/reward_contracts.py",
        },
        {
            "scope": "VPP dispatch reward",
            "name": "AVAILABILITY_PAYMENT_RATE",
            "value": 0.02,
            "meaning_zh": "按可调范围宽度给 availability payment 的比例。",
            "source": "src/vpp_dso_sim/learning/reward_contracts.py",
        },
        {
            "scope": "VPP dispatch reward",
            "name": "DISPATCH_PRIVATE_PROFIT_WEIGHT",
            "value": 0.02,
            "meaning_zh": "private_profit_proxy 进入 VPP dispatch reward 的权重。",
            "source": "src/vpp_dso_sim/learning/reward_contracts.py",
        },
        {
            "scope": "VPP dispatch reward",
            "name": "DISPATCH_TRACKING_PENALTY_WEIGHT",
            "value": 25.0,
            "meaning_zh": "跟踪 DSO 推荐目标偏差的平方惩罚权重。",
            "source": "src/vpp_dso_sim/learning/reward_contracts.py",
        },
        {
            "scope": "VPP dispatch reward",
            "name": "DISPATCH_LINEAR_PROJECTION_PENALTY_WEIGHT",
            "value": 5.0,
            "meaning_zh": "动作被本地/安全投影修正时的一次惩罚权重。",
            "source": "src/vpp_dso_sim/learning/reward_contracts.py",
        },
        {
            "scope": "VPP dispatch reward",
            "name": "DISPATCH_QUADRATIC_PROJECTION_PENALTY_WEIGHT",
            "value": 10.0,
            "meaning_zh": "动作被投影修正时的二次惩罚权重。",
            "source": "src/vpp_dso_sim/learning/reward_contracts.py",
        },
        {
            "scope": "VPP dispatch reward",
            "name": "DISPATCH_COMFORT_SOC_PENALTY_WEIGHT",
            "value": 0.001,
            "meaning_zh": "舒适度/SOC 越界代理惩罚进入 reward 的权重。",
            "source": "src/vpp_dso_sim/learning/reward_contracts.py",
        },
        {
            "scope": "VPP dispatch reward",
            "name": "DISPATCH_PREFERRED_REGION_BONUS_WEIGHT",
            "value": 0.50,
            "meaning_zh": "实际出力落入 DSO 推荐运行区间时的奖励权重。",
            "source": "src/vpp_dso_sim/learning/reward_contracts.py",
        },
        {
            "scope": "VPP dispatch reward",
            "name": "comfort_soc_penalty_scale / clip",
            "value": "100.0 / 5.0",
            "meaning_zh": "舒适度/SOC 原始惩罚先除以 100，再截断到最多 5。",
            "source": "src/vpp_dso_sim/envs/reward_design.py",
        },
        {
            "scope": "Portfolio reward",
            "name": "PORTFOLIO_LOCALIZED_DSO_ALIGNMENT_WEIGHT",
            "value": 1.0,
            "meaning_zh": "慢周期 portfolio 接收局部化 DSO 对齐信号的权重；不是共享全局 DSO reward。",
            "source": "src/vpp_dso_sim/learning/reward_contracts.py",
        },
        {
            "scope": "Portfolio reward",
            "name": "PORTFOLIO_DECISION_MASK_DEFAULT_INTERVAL_STEPS",
            "value": 24,
            "meaning_zh": "默认每 24 个 15 分钟步才允许一次慢周期组合决策。",
            "source": "src/vpp_dso_sim/learning/reward_contracts.py",
        },
        {
            "scope": "DSO config reward",
            "name": "privacy_mode",
            "value": base_config.get("reward", {}).get("privacy_mode"),
            "meaning_zh": "DSO reward 使用隐私保护代理模式。",
            "source": "configs/european_lv_benchmark_v2.yaml",
        },
        {
            "scope": "DSO config reward",
            "name": "dso_reward_cost_scale",
            "value": base_config.get("reward", {}).get("dso_reward_cost_scale"),
            "meaning_zh": "DSO 总成本缩放为训练 reward 的比例。",
            "source": "configs/european_lv_benchmark_v2.yaml",
        },
    ]
    for name, value in base_config.get("reward", {}).get("component_weights", {}).items():
        rows.append(
            {
                "scope": "DSO config component_weights",
                "name": name,
                "value": value,
                "meaning_zh": "DSO 成本/安全分项权重，来自基准配置。",
                "source": "configs/european_lv_benchmark_v2.yaml",
            }
        )
    for name, value in sensitivity_config.get("reward", {}).get("dso", {}).items():
        rows.append(
            {
                "scope": "sensitivity_v1 DSO reward",
                "name": name,
                "value": value,
                "meaning_zh": "结构化 DSO actor 协议中的 DSO reward 权重。",
                "source": "configs/european_lv_benchmark_v2_sensitivity_attention_v1.yaml",
            }
        )
    for name, value in sensitivity_config.get("reward", {}).get("vpp", {}).items():
        rows.append(
            {
                "scope": "sensitivity_v1 VPP reward",
                "name": name,
                "value": value,
                "meaning_zh": "结构化协议中的 VPP reward 权重设置。",
                "source": "configs/european_lv_benchmark_v2_sensitivity_attention_v1.yaml",
            }
        )
    frame = pd.DataFrame(rows)
    frame.to_csv(TABLE / "reward_hyperparameters.csv", index=False)
    return frame


def training_hparam_table() -> pd.DataFrame:
    rows = [
        {"case": "base", "learning_rate_multiplier": 1.0, "entropy_coef": "default", "hidden_dim_multiplier": 1, "notes_zh": "默认学习率与网络规模。"},
        {"case": "lower_lr", "learning_rate_multiplier": 0.33, "entropy_coef": "default", "hidden_dim_multiplier": 1, "notes_zh": "学习率降为 0.33 倍。"},
        {"case": "higher_entropy", "learning_rate_multiplier": 1.0, "entropy_coef": 0.03, "hidden_dim_multiplier": 1, "notes_zh": "HAPPO/HATRPO 熵系数提高；MATD3 exploration_noise=0.22；HASAC target entropy multiplier=1.50, init_log_alpha=-3.0。"},
        {"case": "larger_network", "learning_rate_multiplier": 1.0, "entropy_coef": "default", "hidden_dim_multiplier": 2, "notes_zh": "hidden_dim 扩大 2 倍，paper_long 中 256 -> 512。"},
    ]
    frame = pd.DataFrame(rows)
    frame.to_csv(TABLE / "hparam_cases.csv", index=False)
    return frame


def build_report(
    *,
    inventory: pd.DataFrame,
    manifest: dict[str, Any],
    episodes: pd.DataFrame,
    eval_metrics: pd.DataFrame,
    reward_hparams: pd.DataFrame,
    hparams: pd.DataFrame,
    figure_paths: list[str],
) -> Path:
    config = manifest.get("config", {})
    run_index = read_csv(MAIN_RUN / "run_index.csv")
    progress_sens = read_csv(SENS_PROGRESS_RUN / "experiment_progress.csv")
    lines: list[str] = []
    lines.append("# Paper-long 级实验完成部分审计、训练曲线与 reward 超参数报告")
    lines.append("")
    lines.append("生成时间：2026-05-28。证据来源为当前仓库工作区中的 `outputs/`、`configs/`、`src/`、`README.md` 与实验文档。")
    lines.append("")
    lines.append("## 1. 结论先行")
    lines.append("")
    lines.append(
        "- 当前可完整提取训练曲线和冻结评估指标的 long 目录是 `outputs/paper_training_long_current`。它包含完整的训练/评估 CSV、TensorBoard 图片和 HTML 报告，但这是 2026-05-11 的旧 `paper_long` 协议，manifest 中仍包含 `opf_oracle_proxy`，因此只能作为“已完成的旧 long 训练输出”分析，不能直接升级为最终论文最优性结论。"
    )
    lines.append(
        "- `outputs/paper_training_long_20260513_acgate` 的 manifest 标记为 `baseline_complete`，但核心 baseline CSV 只有表头或为空；它提供了 claim guardrail 约束，不能作为训练曲线来源。"
    )
    lines.append(
        "- 2026-05-28 的 `paper_long_sensitivity_v1` 正式长跑目录目前只有早期进度记录，`thread8_progress` 停在第二个 baseline run 附近，没有训练 episode/loss/eval 汇总文件；因此本报告不把它写成已完成 paper-long 结果。"
    )
    lines.append("")
    lines.append("## 2. 已完成 long 输出目录清单")
    lines.append("")
    lines.append(inventory.to_markdown(index=False))
    lines.append("")
    lines.append("## 3. `paper_training_long_current` 实验协议")
    lines.append("")
    lines.append(f"- preset：`{config.get('preset')}`")
    lines.append(f"- 配置文件：`{config.get('config_path')}`")
    lines.append(f"- 算法：`{', '.join(config.get('algorithms', []))}`")
    lines.append(f"- seeds：`{config.get('seeds')}`")
    lines.append(f"- 训练 split：`{config.get('train_variants')}`；冻结评估 split：`{config.get('eval_variants')}`")
    lines.append(f"- horizon：训练 `{config.get('horizon_steps')}` steps，评估 `{config.get('eval_horizon_steps')}` steps；每步 `{config.get('dt_hours')}` h。")
    lines.append(f"- 训练 episode：`{config.get('train_episodes')}`；hidden_dim：`{config.get('hidden_dim')}`；learning_rate：`{config.get('learning_rate')}`；batch_size：`{config.get('batch_size')}`。")
    lines.append(f"- warmup_steps：`{config.get('warmup_steps')}`；replay_capacity：`{config.get('replay_capacity')}`；ppo_epochs：`{config.get('ppo_epochs')}`。")
    if not run_index.empty:
        counts = run_index.groupby(["split", "algorithm"], dropna=False).size().reset_index(name="count")
        lines.append("")
        lines.append("运行数量按 split/algorithm 汇总：")
        lines.append("")
        lines.append(counts.to_markdown(index=False))
    lines.append("")
    lines.append("## 4. 训练曲线与 loss 曲线")
    lines.append("")
    lines.append(
        f"`training_episode_metrics.csv` 行数为 `{len(episodes)}`；`training_loss_metrics.csv` 是大文件，脚本采用分块聚合生成 `tables/training_loss_curves_aggregated.csv`，避免把 1.3GB CSV 全量塞入报告。"
    )
    lines.append("")
    for path in figure_paths:
        root_relative = Path(path)
        absolute = ROOT / root_relative
        rel = Path("../..") / root_relative
        title = root_relative.stem.replace("_", " ")
        if absolute.exists():
            lines.append(f"![{title}]({rel.as_posix()})")
        else:
            lines.append(f"- 图文件缺失：`{root_relative.as_posix()}`")
        lines.append("")
    lines.append("## 5. Reward 结构与超参数")
    lines.append("")
    lines.append(
        "VPP dispatch reward 当前是 settlement-like proxy。它把能量收益代理、灵活性服务支付代理、可调范围可用性支付代理、DER 运行成本、目标跟踪误差、投影惩罚、舒适度/SOC 惩罚和推荐区间奖励组合成训练信号。它不是正式市场会计利润。"
    )
    lines.append("")
    lines.append(reward_hparams.to_markdown(index=False))
    lines.append("")
    lines.append("## 6. 训练超参数 case")
    lines.append("")
    lines.append(hparams.to_markdown(index=False))
    lines.append("")
    lines.append("## 7. 冻结评估指标文件")
    lines.append("")
    if not eval_metrics.empty:
        eval_summary = (
            eval_metrics.groupby(["algorithm", "hparam_case"], dropna=False)[
                ["eval_total_reward", "eval_total_cost", "total_violation_cells", "security_pass"]
            ]
            .mean()
            .reset_index()
        )
        lines.append(eval_summary.to_markdown(index=False))
    else:
        lines.append("未发现 evaluation_seed_metrics.csv。")
    lines.append("")
    lines.append("## 8. 证据文件索引")
    lines.append("")
    for rel in (
        "outputs/paper_training_long_current/experiment_manifest.json",
        "outputs/paper_training_long_current/training_episode_metrics.csv",
        "outputs/paper_training_long_current/training_loss_metrics.csv",
        "outputs/paper_training_long_current/evaluation_seed_metrics.csv",
        "outputs/paper_training_long_current/aggregate_metrics.csv",
        "outputs/paper_training_long_current/baseline_comparison.csv",
        "outputs/paper_training_long_current/architecture_diagnostics.csv",
        "outputs/paper_training_long_current/tensorboard_images/training_reward_curve.png",
        "outputs/paper_training_long_current/tensorboard_images/loss_trace.png",
        "src/vpp_dso_sim/experiments/paper_training.py",
        "src/vpp_dso_sim/envs/reward_design.py",
        "src/vpp_dso_sim/learning/reward_contracts.py",
        "src/vpp_dso_sim/learning/advanced_marl.py",
        "src/vpp_dso_sim/learning/matd3.py",
        "src/vpp_dso_sim/learning/hatrpo.py",
        "configs/european_lv_benchmark_v2.yaml",
        "configs/european_lv_benchmark_v2_sensitivity_attention_v1.yaml",
        "docs/experiments/paper_long_sensitivity_v1_protocol.md",
    ):
        lines.append(f"- `{rel}`")
    lines.append("")
    lines.append("## 9. 重要边界")
    lines.append("")
    lines.append("- 不把 `opf_oracle_proxy` 或 `ac_validated_search_reference` 写成全局最优上界。")
    lines.append("- 不把 `paper_training_long_sensitivity_v1_preflight_smoke` 写成论文级长周期结果。")
    lines.append("- 不把 `private_profit_proxy` 写成真实会计利润；它只是 reward proxy 与经济诊断指标。")
    lines.append("- 不把正在推进但未完成的 2026-05-28 sensitivity-v1 长跑写成已完成。")
    lines.append("")
    if not progress_sens.empty:
        lines.append("2026-05-28 sensitivity-v1 进度尾部：")
        lines.append("")
        lines.append(progress_sens.tail(8).to_markdown(index=False))
        lines.append("")
    report_path = OUT / "paper_long_experiment_summary_report.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path


def main() -> None:
    ensure_dirs()
    inventory = summarize_run_dirs()
    manifest = read_json(MAIN_RUN / "experiment_manifest.json")
    episodes = read_csv(MAIN_RUN / "training_episode_metrics.csv")
    eval_metrics = read_csv(MAIN_RUN / "evaluation_seed_metrics.csv")
    aggregate = read_csv(MAIN_RUN / "aggregate_metrics.csv")
    reward_hparams = build_reward_hparam_table()
    hparams = training_hparam_table()
    figure_paths: list[str] = []
    figure_paths.extend(make_episode_plots(episodes))
    loss_curves = aggregate_loss_metrics(MAIN_RUN / "training_loss_metrics.csv")
    figure_paths.extend(make_loss_plots(loss_curves))
    figure_paths.extend(make_eval_plots(eval_metrics, aggregate))
    report = build_report(
        inventory=inventory,
        manifest=manifest,
        episodes=episodes,
        eval_metrics=eval_metrics,
        reward_hparams=reward_hparams,
        hparams=hparams,
        figure_paths=figure_paths,
    )
    index = {
        "report": str(report.relative_to(ROOT)),
        "figures": figure_paths,
        "tables": [str(path.relative_to(ROOT)) for path in sorted(TABLE.glob("*.csv"))],
    }
    (OUT / "artifact_index.json").write_text(json.dumps(index, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(index, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
