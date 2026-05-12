from __future__ import annotations

from html import escape
import json
from pathlib import Path
from typing import Any

import pandas as pd

from vpp_dso_sim.utils.io import ensure_dir


CTDE_ALGORITHM_ID = "privacy_separated_ctde_actor_critic"

COLUMN_ZH = {
    "algorithm": "算法",
    "split": "实验划分",
    "scenario_name": "场景名称",
    "seed": "随机种子",
    "profile_variant": "负荷/PV/电价曲线变体",
    "security_pass": "安全通过",
    "min_voltage_vm_pu": "最低母线电压 pu",
    "max_voltage_vm_pu": "最高母线电压 pu",
    "max_line_loading_percent": "最高线路负载率 %",
    "max_trafo_loading_percent": "最高变压器负载率 %",
    "reward_sum": "累计奖励",
    "total_cost": "累计成本",
    "operation_cost_sum": "运行成本合计",
    "reward_privacy_mode": "奖励隐私模式",
    "near_voltage_0_95_step_rate": "接近/低于 0.95pu 的时步比例",
    "near_line_85_step_rate": "线路负载超过 85% 的时步比例",
    "reverse_flow_step_rate": "线路反向潮流时步比例",
    "fr_binding_rate": "FR/DOE 边界绑定比例",
    "projection_gap_mw_sum": "安全投影修正量合计 MW",
    "projection_clipping_rate": "安全投影裁剪比例",
    "policy_evaluation_mode": "策略评估模式",
    "frozen_eval_total_reward": "冻结策略评估累计奖励",
    "frozen_eval_total_cost": "冻结策略评估累计成本",
    "train_best_episode_reward": "训练期最好回合奖励",
    "train_final_episode_reward": "训练期最后回合奖励",
    "train_param_delta_l2": "训练前后参数变化 L2",
    "checkpoint_path": "模型 checkpoint 路径",
    "step": "时步",
    "time_hours": "仿真时间 h",
    "time_label": "时间标签",
    "price": "电价",
    "load_scale": "负荷倍率",
    "pv_forecast_factor": "PV 预测因子",
    "step_min_voltage_vm_pu": "本时步最低电压 pu",
    "step_max_voltage_vm_pu": "本时步最高电压 pu",
    "step_max_line_loading_percent": "本时步最高线路负载率 %",
    "step_max_trafo_loading_percent": "本时步最高变压器负载率 %",
    "projection_gap_mw": "本时步安全投影修正量 MW",
    "projection_clipping_count": "本时步投影裁剪次数",
    "active_need_count": "本时步有主动服务需求的 VPP 数",
    "awarded_flex_mw": "本时步调用/偏好灵活性 MW",
    "run_dir": "运行目录",
    "step_summary_path": "时步摘要路径",
    "frozen_eval_summary_path": "冻结评估摘要路径",
}

VALUE_ZH = {
    "rule_based": "规则基线",
    CTDE_ALGORITHM_ID: "隐私分离 CTDE Actor-Critic",
    "train_profile": "训练曲线",
    "eval_profile": "评估曲线",
    "safety_tight_limits": "安全裕度收紧",
    "topology_holdout": "拓扑泛化 holdout",
    "holdout_reverseflow": "反向潮流 holdout",
    "holdout_peak": "尖峰负荷 holdout",
    "holdout_cloudy": "阴云低 PV holdout",
    "train_mixed": "混合训练曲线",
    "privacy_preserving_proxy": "隐私保护代理奖励",
    "frozen_deterministic_mean_policy": "冻结确定性均值策略评估",
}


def _read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _rel(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root)).replace("\\", "/")
    except ValueError:
        return str(path)


def _lang(en: str, zh: str, *, block: bool = False) -> str:
    cls = "lang-block" if block else "lang-inline"
    return (
        f"<span class='lang-copy {cls} lang-en'>{en}</span>"
        f"<span class='lang-copy {cls} lang-zh'>{zh}</span>"
    )


def _value_text(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    text = str(value)
    return VALUE_ZH.get(text, text)


def _fmt(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def _header(column: str) -> str:
    zh = COLUMN_ZH.get(column, column.replace("_", " "))
    return f"{escape(column)}<br><small>{escape(zh)}</small>"


def _table(frame: pd.DataFrame, *, max_rows: int = 80, columns: list[str] | None = None) -> str:
    if frame.empty:
        return "<p class='muted'>No rows / 暂无数据。</p>"
    data = frame.copy()
    if columns is not None:
        data = data[[col for col in columns if col in data.columns]]
    if len(data) > max_rows:
        data = data.head(max_rows)
    head = "".join(f"<th>{_header(str(col))}</th>" for col in data.columns)
    rows = []
    for _, row in data.iterrows():
        cells = []
        for col in data.columns:
            raw = row[col]
            text = _fmt(raw)
            zh = _value_text(raw)
            if zh != text:
                cell = f"{escape(text)}<br><small>{escape(zh)}</small>"
            else:
                cell = escape(text)
            cells.append(f"<td>{cell}</td>")
        rows.append(f"<tr>{''.join(cells)}</tr>")
    return f"<div class='table-wrap'><table><thead><tr>{head}</tr></thead><tbody>{''.join(rows)}</tbody></table></div>"


def _metric_card(label_en: str, label_zh: str, value: object, note_en: str = "", note_zh: str = "") -> str:
    return (
        "<article class='metric-card'>"
        f"<span>{_lang(label_en, label_zh)}</span>"
        f"<strong>{escape(_fmt(value))}</strong>"
        f"<small>{_lang(note_en, note_zh)}</small>"
        "</article>"
    )


def _plot_html(kind: str, seed_metrics: pd.DataFrame, focus_step: pd.DataFrame) -> str:
    try:
        import plotly.graph_objects as go
        import plotly.io as pio
        from plotly.subplots import make_subplots
    except Exception:
        return "<div class='plot-fallback'>Plotly is not installed / 未安装 Plotly，图形降级为表格。</div>"

    if kind == "benchmark_overview" and not seed_metrics.empty:
        grouped = (
            seed_metrics.groupby(["algorithm", "split", "profile_variant"], dropna=False)
            .agg(
                security_pass_rate=("security_pass", "mean"),
                min_voltage=("min_voltage_vm_pu", "min"),
                max_line=("max_line_loading_percent", "max"),
            )
            .reset_index()
        )
        grouped["label"] = (
            grouped["algorithm"].astype(str)
            + "<br>"
            + grouped["split"].astype(str)
            + " / "
            + grouped["profile_variant"].astype(str)
        )
        fig = make_subplots(
            rows=2,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.16,
            subplot_titles=(
                "安全通过率 / Security pass rate",
                "最低电压与最高线路负载 / Min voltage and max line loading",
            ),
            specs=[[{"secondary_y": False}], [{"secondary_y": True}]],
        )
        fig.add_trace(
            go.Bar(
                x=grouped["label"],
                y=grouped["security_pass_rate"],
                name="安全通过率 / pass rate",
                marker_color="#2563eb",
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=grouped["label"],
                y=grouped["min_voltage"],
                name="最低电压 pu / min voltage",
                mode="lines+markers",
                line={"color": "#059669", "width": 3},
            ),
            row=2,
            col=1,
            secondary_y=False,
        )
        fig.add_trace(
            go.Scatter(
                x=grouped["label"],
                y=grouped["max_line"],
                name="最高线路负载 % / max line loading",
                mode="lines+markers",
                line={"color": "#dc2626", "width": 3},
            ),
            row=2,
            col=1,
            secondary_y=True,
        )
        fig.update_yaxes(title_text="安全通过率", range=[0, 1.05], row=1, col=1)
        fig.update_yaxes(title_text="电压 pu", row=2, col=1, secondary_y=False)
        fig.update_yaxes(title_text="线路负载 %", row=2, col=1, secondary_y=True)
        fig.update_layout(height=760, margin={"l": 50, "r": 50, "t": 80, "b": 130}, legend={"orientation": "h"})
        return pio.to_html(fig, include_plotlyjs="cdn", full_html=False)

    if kind == "security_scatter" and not seed_metrics.empty:
        fig = go.Figure()
        for algorithm, part in seed_metrics.groupby("algorithm"):
            fig.add_trace(
                go.Scatter(
                    x=part["max_line_loading_percent"],
                    y=part["min_voltage_vm_pu"],
                    mode="markers+text",
                    text=part["profile_variant"].astype(str),
                    textposition="top center",
                    name=str(algorithm),
                    marker={"size": 13},
                    customdata=part[["split", "seed", "security_pass"]].to_numpy(),
                    hovertemplate=(
                        "split=%{customdata[0]}<br>seed=%{customdata[1]}"
                        "<br>security_pass=%{customdata[2]}<br>line=%{x:.2f}%<br>voltage=%{y:.4f}pu<extra></extra>"
                    ),
                )
            )
        fig.add_hline(y=0.95, line_dash="dash", line_color="#f59e0b", annotation_text="0.95 pu")
        fig.add_vline(x=85.0, line_dash="dot", line_color="#64748b", annotation_text="85%")
        fig.add_vline(x=100.0, line_dash="dash", line_color="#dc2626", annotation_text="100%")
        fig.update_layout(
            title="安全裕度散点 / Security margin scatter",
            xaxis_title="最高线路负载率 % / max line loading",
            yaxis_title="最低电压 pu / min voltage",
            height=520,
            margin={"l": 60, "r": 30, "t": 80, "b": 60},
        )
        return pio.to_html(fig, include_plotlyjs=False, full_html=False)

    if kind == "focus_step" and not focus_step.empty:
        fig = make_subplots(
            rows=3,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.1,
            subplot_titles=(
                "电压轨迹 / Voltage trajectory",
                "线路与变压器负载 / Loading trajectory",
                "安全投影与灵活性调用 / Projection and flex award",
            ),
            specs=[[{}], [{"secondary_y": True}], [{"secondary_y": True}]],
        )
        x = focus_step["time_hours"] if "time_hours" in focus_step else focus_step["step"]
        fig.add_trace(
            go.Scatter(x=x, y=focus_step["step_min_voltage_vm_pu"], name="最低电压 pu", line={"color": "#059669", "width": 3}),
            row=1,
            col=1,
        )
        if "step_max_voltage_vm_pu" in focus_step:
            fig.add_trace(
                go.Scatter(x=x, y=focus_step["step_max_voltage_vm_pu"], name="最高电压 pu", line={"color": "#16a34a", "dash": "dot"}),
                row=1,
                col=1,
            )
        fig.add_hline(y=0.95, row=1, col=1, line_dash="dash", line_color="#f59e0b")
        fig.add_trace(
            go.Scatter(
                x=x,
                y=focus_step["step_max_line_loading_percent"],
                name="最高线路负载 %",
                line={"color": "#dc2626", "width": 3},
            ),
            row=2,
            col=1,
            secondary_y=False,
        )
        if "step_max_trafo_loading_percent" in focus_step:
            fig.add_trace(
                go.Scatter(
                    x=x,
                    y=focus_step["step_max_trafo_loading_percent"],
                    name="最高变压器负载 %",
                    line={"color": "#7c3aed", "width": 2},
                ),
                row=2,
                col=1,
                secondary_y=True,
            )
        if "projection_gap_mw" in focus_step:
            fig.add_trace(
                go.Bar(x=x, y=focus_step["projection_gap_mw"], name="安全投影修正 MW", marker_color="#f97316"),
                row=3,
                col=1,
                secondary_y=False,
            )
        if "awarded_flex_mw" in focus_step:
            fig.add_trace(
                go.Scatter(x=x, y=focus_step["awarded_flex_mw"], name="调用灵活性 MW", line={"color": "#2563eb", "width": 3}),
                row=3,
                col=1,
                secondary_y=True,
            )
        fig.update_xaxes(title_text="时间 h / time hours", row=3, col=1)
        fig.update_layout(height=850, margin={"l": 60, "r": 50, "t": 80, "b": 60}, legend={"orientation": "h"})
        return pio.to_html(fig, include_plotlyjs=False, full_html=False)

    return "<p class='muted'>No chart data / 暂无图形数据。</p>"


def _load_run_index(output_dir: Path, seed_metrics: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for _, metric in seed_metrics.iterrows():
        seed_value = metric.get("seed")
        try:
            seed = int(seed_value)
        except (TypeError, ValueError):
            seed = str(seed_value)
        split = str(metric.get("split", ""))
        algorithm = str(metric.get("algorithm", ""))
        variant = str(metric.get("profile_variant", ""))
        run_dir = output_dir / split / f"{algorithm}_{variant}_seed_{seed}"
        step_path = run_dir / "step_summary.csv"
        frozen_path = run_dir / "frozen_eval_summary.csv"
        rows.append(
            {
                "algorithm": algorithm,
                "split": split,
                "profile_variant": variant,
                "seed": seed,
                "security_pass": metric.get("security_pass"),
                "min_voltage_vm_pu": metric.get("min_voltage_vm_pu"),
                "max_line_loading_percent": metric.get("max_line_loading_percent"),
                "policy_evaluation_mode": metric.get("policy_evaluation_mode", ""),
                "run_dir": _rel(run_dir, output_dir) if run_dir.exists() else "",
                "step_summary_path": _rel(step_path, output_dir) if step_path.exists() else "",
                "frozen_eval_summary_path": _rel(frozen_path, output_dir) if frozen_path.exists() else "",
            }
        )
    return pd.DataFrame(rows)


def _focus_run(seed_metrics: pd.DataFrame, run_index: pd.DataFrame) -> pd.Series:
    if seed_metrics.empty:
        return pd.Series(dtype=object)
    ranked = seed_metrics.copy()
    ranked["_priority"] = 10
    ranked.loc[
        (ranked["algorithm"].astype(str) == CTDE_ALGORITHM_ID)
        & (ranked["profile_variant"].astype(str) == "holdout_reverseflow"),
        "_priority",
    ] = 0
    ranked.loc[
        (ranked["algorithm"].astype(str) == CTDE_ALGORITHM_ID)
        & (ranked["split"].astype(str) == "safety_tight_limits"),
        "_priority",
    ] = 1
    ranked.loc[ranked["profile_variant"].astype(str) == "holdout_reverseflow", "_priority"] = ranked["_priority"].clip(upper=2)
    ranked["_stress"] = (
        ranked.get("max_line_loading_percent", pd.Series(0, index=ranked.index)).fillna(0).astype(float)
        - ranked.get("min_voltage_vm_pu", pd.Series(1, index=ranked.index)).fillna(1).astype(float) * 10.0
        + (1 - ranked.get("security_pass", pd.Series(1, index=ranked.index)).fillna(1).astype(float)) * 100.0
    )
    focus = ranked.sort_values(["_priority", "_stress"], ascending=[True, False]).iloc[0]
    if run_index.empty:
        return focus
    mask = (
        (run_index["algorithm"].astype(str) == str(focus.get("algorithm")))
        & (run_index["split"].astype(str) == str(focus.get("split")))
        & (run_index["profile_variant"].astype(str) == str(focus.get("profile_variant")))
        & (run_index["seed"].astype(str) == str(int(focus.get("seed"))))
    )
    if mask.any():
        merged = focus.copy()
        for key, value in run_index.loc[mask].iloc[0].items():
            merged[key] = value
        return merged
    return focus


def _load_focus_step(output_dir: Path, focus: pd.Series) -> pd.DataFrame:
    step_path = str(focus.get("step_summary_path", "")).strip()
    if step_path:
        return _read_csv(output_dir / step_path)
    return pd.DataFrame()


def _model_sync_frame(output_dir: Path, seed_metrics: pd.DataFrame, run_index: pd.DataFrame, focus: pd.Series) -> pd.DataFrame:
    algorithm_ids = ", ".join(sorted(seed_metrics["algorithm"].dropna().astype(str).unique())) if "algorithm" in seed_metrics else ""
    splits = ", ".join(sorted(seed_metrics["split"].dropna().astype(str).unique())) if "split" in seed_metrics else ""
    policy_modes = (
        ", ".join(sorted(seed_metrics["policy_evaluation_mode"].dropna().astype(str).unique()))
        if "policy_evaluation_mode" in seed_metrics
        else ""
    )
    train_eval_zh = (
        "CTDE 训练后使用 checkpoint 冻结确定性均值策略在 holdout split 上评估。"
        if policy_modes
        else "当前 benchmark 只包含非训练基线；加入 CTDE 算法后会显示 frozen-eval 评估模式。"
    )
    return pd.DataFrame(
        [
            {
                "update_area": "benchmark_source",
                "current_value": str(output_dir),
                "current_value_zh": "本页直接读取当前 benchmark 输出目录，不再复用根目录通用 rollout 页面。",
                "evidence_file": "seed_metrics.csv; aggregate_metrics.csv; profile_quality.csv",
            },
            {
                "update_area": "algorithm",
                "current_value": algorithm_ids,
                "current_value_zh": "页面算法列表来自 seed_metrics.csv，包含规则基线与隐私分离 CTDE 时会同时展示。",
                "evidence_file": "dashboard_data/benchmark_seed_metrics.csv",
            },
            {
                "update_area": "train_then_frozen_eval",
                "current_value": policy_modes or "rule_based_only",
                "current_value_zh": train_eval_zh,
                "evidence_file": "*/frozen_eval_summary.csv",
            },
            {
                "update_area": "split_coverage",
                "current_value": splits,
                "current_value_zh": "页面展示 train/eval/safety/topology 等 benchmark split 覆盖情况。",
                "evidence_file": "dashboard_data/benchmark_run_index.csv",
            },
            {
                "update_area": "focus_run",
                "current_value": str(focus.get("step_summary_path", "")),
                "current_value_zh": "焦点时序图优先选择 CTDE 反向潮流 holdout，其次选择安全裕度收紧 split。",
                "evidence_file": str(focus.get("step_summary_path", "")),
            },
            {
                "update_area": "ui_refresh_contract",
                "current_value": "benchmark_report.html; interactive_report.html; rl_architecture.html; vpp_first_person/index.html; dashboard_data/benchmark_*.csv",
                "current_value_zh": "benchmark runner 默认刷新 benchmark 目录内所有 HTML 与 dashboard 数据。",
                "evidence_file": "examples/11_run_benchmark_experiment.py",
            },
        ]
    )


def _export_frames(
    output_dir: Path,
    seed_metrics: pd.DataFrame,
    aggregate_metrics: pd.DataFrame,
    profile_quality: pd.DataFrame,
    run_index: pd.DataFrame,
    focus_step: pd.DataFrame,
    model_sync: pd.DataFrame,
) -> dict[str, Path]:
    data_dir = ensure_dir(output_dir / "dashboard_data")
    frames = {
        "benchmark_seed_metrics": seed_metrics,
        "benchmark_aggregate_metrics": aggregate_metrics,
        "benchmark_profile_quality": profile_quality,
        "benchmark_run_index": run_index,
        "benchmark_focus_step_summary": focus_step,
        "model_update_summary": model_sync,
    }
    paths: dict[str, Path] = {}
    for name, frame in frames.items():
        path = data_dir / f"{name}.csv"
        frame.to_csv(path, index=False)
        paths[name] = path
    return paths


def _base_css() -> str:
    return """
    :root { color-scheme: light; --ink:#172033; --muted:#64748b; --line:#dbe5ef; --blue:#2563eb; --green:#059669; --orange:#f97316; --red:#dc2626; --bg:#f7fafc; }
    body { margin:0; font-family: "Segoe UI", "Microsoft YaHei", Arial, sans-serif; color:var(--ink); background:var(--bg); line-height:1.55; }
    .page { max-width: 1440px; margin: 0 auto; padding: 28px; }
    .hero { display:grid; grid-template-columns: minmax(0,1.45fr) minmax(320px,.55fr); gap:20px; align-items:stretch; margin-bottom:22px; }
    .hero-main, .panel, .metric-card, .nav-card { background:#fff; border:1px solid var(--line); border-radius:8px; box-shadow:0 10px 24px rgba(15,23,42,.06); }
    .hero-main { padding:24px; }
    .hero h1 { margin:0 0 8px; font-size:32px; letter-spacing:0; }
    .hero p { max-width: 980px; color:var(--muted); margin:8px 0 0; }
    .language-toolbar { display:flex; gap:8px; align-items:center; justify-content:flex-end; margin-bottom:14px; }
    .language-toolbar button { border:1px solid var(--line); background:#fff; border-radius:6px; padding:7px 12px; cursor:pointer; }
    .language-toolbar button.is-active { background:var(--blue); color:#fff; border-color:var(--blue); }
    body.lang-en .lang-zh { display:none !important; }
    body.lang-zh .lang-en { display:none !important; }
    .lang-block { display:block; }
    .lang-inline { display:inline; }
    .metric-grid { display:grid; grid-template-columns: repeat(auto-fit, minmax(210px,1fr)); gap:12px; margin:16px 0; }
    .metric-card { padding:15px; min-height:92px; }
    .metric-card span { display:block; color:var(--muted); font-size:13px; }
    .metric-card strong { display:block; font-size:24px; margin:5px 0; }
    .metric-card small { color:var(--muted); }
    .panel { padding:20px; margin:18px 0; overflow:hidden; }
    .panel h2 { margin:0 0 8px; font-size:22px; }
    .panel-note { color:var(--muted); margin:0 0 14px; }
    .plot-grid { display:grid; grid-template-columns: 1fr; gap:16px; }
    .nav-grid { display:grid; grid-template-columns: repeat(auto-fit, minmax(230px,1fr)); gap:12px; }
    .nav-card { padding:16px; text-decoration:none; color:var(--ink); }
    .nav-card strong { display:block; margin-bottom:4px; }
    .table-wrap { overflow:auto; border:1px solid var(--line); border-radius:8px; background:#fff; }
    table { border-collapse:collapse; width:100%; min-width:960px; font-size:13px; }
    th, td { border-bottom:1px solid var(--line); padding:8px 10px; text-align:left; vertical-align:top; }
    th { background:#eef4fb; position:sticky; top:0; z-index:1; }
    th small, td small { color:var(--muted); font-size:11px; }
    .muted { color:var(--muted); }
    .pill { display:inline-flex; align-items:center; gap:6px; border-radius:999px; padding:5px 10px; background:#e8f1ff; color:#1d4ed8; font-size:12px; margin:2px 4px 2px 0; }
    .workflow { display:grid; grid-template-columns: 1fr 48px 1fr 48px 1fr 48px 1fr; gap:12px; align-items:center; }
    .box { background:#fff; border:1px solid var(--line); border-radius:8px; padding:14px; min-height:96px; }
    .box h3 { margin:0 0 6px; font-size:16px; }
    .arrow { text-align:center; font-size:30px; color:var(--blue); }
    .dash-arrow { text-align:center; color:var(--orange); border-top:3px dashed var(--orange); height:1px; margin:14px 0; }
    code { background:#eef4fb; padding:2px 5px; border-radius:4px; }
    @media (max-width: 980px) { .hero { grid-template-columns:1fr; } .workflow { grid-template-columns:1fr; } .arrow { transform:rotate(90deg); } .page { padding:16px; } }
    """


def _script() -> str:
    return """
    <script>
    document.querySelectorAll("[data-lang-switch]").forEach((button) => {
      button.addEventListener("click", () => {
        const lang = button.dataset.langSwitch;
        document.body.classList.toggle("lang-en", lang === "en");
        document.body.classList.toggle("lang-zh", lang === "zh");
        document.querySelectorAll("[data-lang-switch]").forEach((b) => b.classList.toggle("is-active", b === button));
        localStorage.setItem("ppvpp-lang", lang);
      });
    });
    const savedLang = localStorage.getItem("ppvpp-lang") || "zh";
    const savedButton = document.querySelector(`[data-lang-switch="${savedLang}"]`);
    if (savedButton) savedButton.click();
    </script>
    """


def _language_toolbar() -> str:
    return """
    <div class="language-toolbar" role="group" aria-label="Language">
      <span>Language / 语言</span>
      <button type="button" data-lang-switch="en">EN</button>
      <button type="button" class="is-active" data-lang-switch="zh">中文</button>
    </div>
    """


def _html_shell(title: str, body: str) -> str:
    return f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{escape(title)}</title>
  <style>{_base_css()}</style>
</head>
<body class="lang-zh">
  <main class="page">
    {_language_toolbar()}
    {body}
  </main>
  {_script()}
</body>
</html>
"""


def _overview_body(
    output_dir: Path,
    manifest: dict[str, Any],
    seed_metrics: pd.DataFrame,
    aggregate_metrics: pd.DataFrame,
    profile_quality: pd.DataFrame,
    run_index: pd.DataFrame,
    focus: pd.Series,
    focus_step: pd.DataFrame,
    model_sync: pd.DataFrame,
) -> str:
    pass_rate = seed_metrics["security_pass"].mean() if "security_pass" in seed_metrics and not seed_metrics.empty else ""
    min_voltage = seed_metrics["min_voltage_vm_pu"].min() if "min_voltage_vm_pu" in seed_metrics and not seed_metrics.empty else ""
    max_line = seed_metrics["max_line_loading_percent"].max() if "max_line_loading_percent" in seed_metrics and not seed_metrics.empty else ""
    reverse_rate = seed_metrics["reverse_flow_step_rate"].max() if "reverse_flow_step_rate" in seed_metrics and not seed_metrics.empty else ""
    body = f"""
    <section class="hero">
      <div class="hero-main">
        <p class="pill">Benchmark-aware UI / 实验感知可视化</p>
        <h1>{_lang("Benchmark V2.1 Interactive Experiment Report", "Benchmark V2.1 交互式实验报告")}</h1>
        <p>{_lang(
            "This page is generated from the benchmark output directory itself: seed metrics, aggregate metrics, profile quality, per-run step summaries and frozen CTDE evaluation artifacts.",
            "本页直接从 benchmark 输出目录生成：读取 seed_metrics、aggregate_metrics、profile_quality、每个 run 的 step_summary 以及 CTDE frozen evaluation 产物。",
            block=True,
        )}</p>
        <p>{_lang("Output directory", "输出目录")}: <code>{escape(str(output_dir))}</code></p>
      </div>
      <div class="hero-main">
        <strong>{_lang("Focus Run", "焦点运行")}</strong>
        <p>{escape(str(focus.get("algorithm", "")))} / {escape(str(focus.get("split", "")))} / {escape(str(focus.get("profile_variant", "")))} / seed={escape(str(focus.get("seed", "")))}</p>
        <p class="muted">{_lang("The focus run is used for the detailed time-series chart below.", "焦点运行用于下方详细时序图。")}</p>
      </div>
    </section>
    <section class="metric-grid">
      {_metric_card("Runs", "运行次数", len(seed_metrics), "rows in seed_metrics", "seed_metrics 行数")}
      {_metric_card("Security Pass Rate", "安全通过率", pass_rate, "mean(security_pass)", "security_pass 均值")}
      {_metric_card("Min Voltage", "最低电压", min_voltage, "min over runs", "所有运行中的最小值")}
      {_metric_card("Max Line Loading", "最高线路负载", max_line, "max over runs", "所有运行中的最大值")}
      {_metric_card("Reverse-Flow Rate", "反向潮流比例", reverse_rate, "max over runs", "所有运行中的最大值")}
    </section>
    <section class="panel">
      <h2>{_lang("Linked HTML Outputs", "联动 HTML 输出")}</h2>
      <p class="panel-note">{_lang("These files are regenerated together after the benchmark run.", "这些文件会在 benchmark 运行结束后一起刷新。")}</p>
      <div class="nav-grid">
        <a class="nav-card" href="benchmark_report.html"><strong>benchmark_report.html</strong>{_lang("Benchmark landing page", "Benchmark 主页面", block=True)}</a>
        <a class="nav-card" href="interactive_report.html"><strong>interactive_report.html</strong>{_lang("Metrics, plots and run drill-down", "指标、图表与运行深钻", block=True)}</a>
        <a class="nav-card" href="rl_architecture.html"><strong>rl_architecture.html</strong>{_lang("CTDE protocol and neural modules", "CTDE 协议与神经网络模块", block=True)}</a>
        <a class="nav-card" href="vpp_first_person/index.html"><strong>vpp_first_person/index.html</strong>{_lang("VPP first-person benchmark replay", "VPP 第一视角 benchmark 回放", block=True)}</a>
      </div>
    </section>
    <section class="panel">
      <h2>{_lang("Benchmark Overview Charts", "Benchmark 总览图")}</h2>
      {_plot_html("benchmark_overview", seed_metrics, focus_step)}
      {_plot_html("security_scatter", seed_metrics, focus_step)}
    </section>
    <section class="panel">
      <h2>{_lang("Focus Run Time-Series", "焦点运行时序图")}</h2>
      <p class="panel-note">{_lang("This chart comes from the selected run's step_summary.csv, so projection gaps, service needs and frozen-eval behavior are not lost.", "该图来自焦点 run 的 step_summary.csv，因此安全投影、服务需求与 frozen-eval 行为不会丢失。")}</p>
      {_plot_html("focus_step", seed_metrics, focus_step)}
    </section>
    <section class="panel">
      <h2>{_lang("Model/UI Synchronization Contract", "模型/UI 同步约定")}</h2>
      {_table(model_sync, max_rows=20)}
    </section>
    <section class="panel">
      <h2>{_lang("Seed Metrics", "Seed 级指标")}</h2>
      {_table(seed_metrics, columns=[
          "algorithm", "split", "profile_variant", "seed", "reward_privacy_mode", "policy_evaluation_mode",
          "security_pass", "min_voltage_vm_pu", "max_line_loading_percent", "near_voltage_0_95_step_rate",
          "near_line_85_step_rate", "reverse_flow_step_rate", "fr_binding_rate", "projection_gap_mw_sum",
          "frozen_eval_total_reward", "frozen_eval_total_cost",
      ])}
    </section>
    <section class="panel">
      <h2>{_lang("Run Index", "运行索引")}</h2>
      {_table(run_index, max_rows=120)}
    </section>
    <section class="panel">
      <h2>{_lang("Aggregate Metrics", "聚合指标")}</h2>
      {_table(aggregate_metrics, max_rows=80)}
    </section>
    <section class="panel">
      <h2>{_lang("Profile Quality", "曲线质量")}</h2>
      {_table(profile_quality, max_rows=80)}
    </section>
    <section class="panel">
      <h2>{_lang("Manifest", "实验清单")}</h2>
      <pre>{escape(json.dumps(manifest, ensure_ascii=False, indent=2))}</pre>
    </section>
    """
    return body


def _architecture_body(
    output_dir: Path,
    seed_metrics: pd.DataFrame,
    run_index: pd.DataFrame,
    model_sync: pd.DataFrame,
) -> str:
    ctde_rows = seed_metrics[seed_metrics.get("algorithm", pd.Series(dtype=str)).astype(str) == CTDE_ALGORITHM_ID]
    best_checkpoint = ""
    if not ctde_rows.empty and "checkpoint_path" in ctde_rows:
        best_checkpoint = str(ctde_rows["checkpoint_path"].dropna().astype(str).iloc[0])
    body = f"""
    <section class="hero">
      <div class="hero-main">
        <p class="pill">CTDE / Privacy boundary / 神经网络结构</p>
        <h1>{_lang("Benchmark-Aware RL Architecture", "Benchmark 感知的强化学习架构")}</h1>
        <p>{_lang(
            "This page explains the current train-then-frozen-eval protocol used by the benchmark, not just the generic architecture roadmap.",
            "本页解释当前 benchmark 实际使用的 train-then-frozen-eval 协议，而不是只展示通用架构路线图。",
            block=True,
        )}</p>
      </div>
      <div class="hero-main">
        <strong>{_lang("Checkpoint Evidence", "Checkpoint 证据")}</strong>
        <p><code>{escape(best_checkpoint or "n/a")}</code></p>
      </div>
    </section>
    <section class="panel">
      <h2>{_lang("One-Step Data Flow", "单步数据流")}</h2>
      <div class="workflow">
        <div class="box"><h3>DSO actor</h3><p>{_lang("Reads public grid stress plus VPP bids/reports and outputs an envelope preference.", "读取公共电网压力与 VPP 报量/报价摘要，输出运行包络偏好。")}</p></div>
        <div class="arrow">→</div>
        <div class="box"><h3>FR/DOE projection</h3><p>{_lang("Clips the preference into executable voltage/loading-aware boundaries.", "将偏好裁剪到满足电压/线路约束的可执行边界。")}</p></div>
        <div class="arrow">→</div>
        <div class="box"><h3>VPP dispatch actor</h3><p>{_lang("Each VPP reads only local DER state and the received envelope, then proposes aggregate P and DER actions.", "每个 VPP 只读取自身 DER 状态和收到的包络，提出聚合 P 与 DER 动作。")}</p></div>
        <div class="arrow">→</div>
        <div class="box"><h3>pandapower runpp</h3><p>{_lang("Safe setpoints are written to load/sgen/storage elements and checked by AC power flow.", "安全出力写入 load/sgen/storage 元件，并由 AC 潮流校核。")}</p></div>
      </div>
      <div class="dash-arrow"></div>
      <p class="panel-note">{_lang(
          "During training only, the centralized critic reads critic_global_state and a compact joint-action summary to estimate the advantage for DSO, VPP dispatch and VPP portfolio policy losses.",
          "仅训练期，集中 critic 读取 critic_global_state 与紧凑 joint-action summary，为 DSO、VPP 解聚合、VPP 组合配置策略损失估计 advantage。",
          block=True,
      )}</p>
    </section>
    <section class="panel">
      <h2>{_lang("Implemented Neural Modules", "已实现神经网络模块")}</h2>
      <div class="metric-grid">
        {_metric_card("DSO Actor", "DSO 全局引导 Actor", "LayerNorm -> MLP(64,64) -> Gaussian envelope head", "runtime actor", "执行期 actor")}
        {_metric_card("VPP Dispatch Actor", "VPP 解聚合 Actor", "context encoder + DER token Deep Sets -> aggregate/DER heads", "local private observation", "本地私有观测")}
        {_metric_card("VPP Portfolio Actor", "VPP 组合配置 Actor", "MLP -> Categorical(keep/reweight/propose)", "profit + reliability + localized DSO alignment", "利润 + 可靠性 + 局部化 DSO 对齐收益")}
        {_metric_card("Centralized Critic", "集中 Critic", "state encoder + action encoder -> V(s,a_summary)", "training only", "仅训练期")}
        {_metric_card("Safety Projection", "安全投影", "deterministic shield", "not an RL agent", "不是 RL 智能体")}
      </div>
    </section>
    <section class="panel">
      <h2>{_lang("Benchmark Protocol Evidence", "Benchmark 协议证据")}</h2>
      {_table(seed_metrics, columns=[
          "algorithm", "split", "profile_variant", "seed", "policy_evaluation_mode",
          "train_best_episode_reward", "train_final_episode_reward", "train_param_delta_l2",
          "frozen_eval_total_reward", "frozen_eval_total_cost", "checkpoint_path",
      ], max_rows=100)}
    </section>
    <section class="panel">
      <h2>{_lang("Synchronized UI Fields", "已同步 UI 字段")}</h2>
      {_table(model_sync, max_rows=20)}
    </section>
    <section class="panel">
      <h2>{_lang("Run Artifacts", "运行产物")}</h2>
      {_table(run_index, max_rows=120)}
    </section>
    """
    return body


def _first_person_body(output_dir: Path, seed_metrics: pd.DataFrame, run_index: pd.DataFrame, focus: pd.Series, focus_step: pd.DataFrame) -> str:
    first_rows = focus_step.head(18).copy()
    if not first_rows.empty:
        first_rows["vpp_saw"] = "price + load/PV profile + DSO envelope/service request"
        first_rows["vpp_saw_zh"] = "电价、负荷/PV 曲线、DSO 下发的包络/服务需求"
        first_rows["vpp_inferred"] = first_rows.apply(
            lambda row: "near limit" if float(row.get("step_min_voltage_vm_pu", 1.0)) < 0.95 or float(row.get("step_max_line_loading_percent", 0.0)) > 85.0 else "comfortable",
            axis=1,
        )
        first_rows["vpp_decision"] = first_rows.apply(
            lambda row: "respect projection and dispatch flex" if float(row.get("active_need_count", 0.0)) > 0 else "follow baseline envelope",
            axis=1,
        )
        first_rows["vpp_decision_zh"] = first_rows["vpp_decision"].map(
            {
                "respect projection and dispatch flex": "遵守安全投影并调度本地灵活性",
                "follow baseline envelope": "跟随基线包络运行",
            }
        )
    body = f"""
    <section class="hero">
      <div class="hero-main">
        <p class="pill">VPP first-person / 第一视角</p>
        <h1>{_lang("Benchmark First-Person Replay", "Benchmark 第一视角回放")}</h1>
        <p>{_lang(
            "This page is benchmark-aware: it focuses on the run selected from seed_metrics and step_summary rather than a separately rerun generic simulation.",
            "本页是 benchmark 感知的：它聚焦 seed_metrics 与 step_summary 选出的运行，而不是重新跑一次通用仿真。",
            block=True,
        )}</p>
      </div>
      <div class="hero-main">
        <strong>{_lang("Focus Run", "焦点运行")}</strong>
        <p>{escape(str(focus.get("algorithm", "")))} / {escape(str(focus.get("split", "")))} / {escape(str(focus.get("profile_variant", "")))}</p>
        <p class="muted"><code>{escape(str(focus.get("step_summary_path", "")))}</code></p>
      </div>
    </section>
    <section class="panel">
      <h2>{_lang("What a VPP Sees and Does", "VPP 在每个时刻看到什么、判断什么、做什么")}</h2>
      <p class="panel-note">{_lang(
          "The current benchmark logs aggregate evidence. The table translates it into readable VPP-side reasoning and explicitly separates observation, inference and action.",
          "当前 benchmark 记录的是聚合证据。下表把这些证据翻译成 VPP 侧可读叙事，并明确区分观测、推断和动作。",
          block=True,
      )}</p>
      {_table(first_rows, columns=[
          "step", "time_label", "price", "step_min_voltage_vm_pu", "step_max_line_loading_percent",
          "projection_gap_mw", "active_need_count", "awarded_flex_mw", "vpp_saw", "vpp_saw_zh",
          "vpp_inferred", "vpp_decision", "vpp_decision_zh",
      ], max_rows=24)}
    </section>
    <section class="panel">
      <h2>{_lang("Long-Cycle Judgment", "长周期判断")}</h2>
      <div class="metric-grid">
        {_metric_card("Mean Projection Gap", "平均安全投影修正", focus_step["projection_gap_mw"].mean() if "projection_gap_mw" in focus_step else "", "MW per step", "每时步 MW")}
        {_metric_card("Active Need Steps", "存在主动需求的时步", int((focus_step.get("active_need_count", pd.Series(dtype=float)) > 0).sum()) if not focus_step.empty and "active_need_count" in focus_step else "", "DSO requested service", "DSO 发出服务需求")}
        {_metric_card("Min Voltage", "最低电压", focus_step["step_min_voltage_vm_pu"].min() if "step_min_voltage_vm_pu" in focus_step else "", "focus run", "焦点运行")}
        {_metric_card("Max Line Loading", "最高线路负载", focus_step["step_max_line_loading_percent"].max() if "step_max_line_loading_percent" in focus_step else "", "focus run", "焦点运行")}
      </div>
    </section>
    <section class="panel">
      <h2>{_lang("All Benchmark Runs", "全部 Benchmark 运行")}</h2>
      {_table(run_index, max_rows=120)}
    </section>
    """
    return body


def export_benchmark_visualization_outputs(output_dir: str | Path) -> dict[str, Path | dict[str, Path]]:
    """Generate benchmark-aware static HTML and dashboard CSV outputs.

    The regular simulation report refresh reruns one scenario and is useful for
    topology replay. This function is intentionally different: it reads the
    completed benchmark directory so train/eval splits, frozen CTDE evaluation,
    holdout reverse-flow behavior, and per-run step summaries remain visible.
    """

    out = ensure_dir(output_dir)
    seed_metrics = _read_csv(out / "seed_metrics.csv")
    aggregate_metrics = _read_csv(out / "aggregate_metrics.csv")
    profile_quality = _read_csv(out / "profile_quality.csv")
    manifest = _read_json(out / "experiment_manifest.json")
    run_index = _load_run_index(out, seed_metrics)
    focus = _focus_run(seed_metrics, run_index)
    focus_step = _load_focus_step(out, focus)
    model_sync = _model_sync_frame(out, seed_metrics, run_index, focus)
    dashboard_paths = _export_frames(
        out,
        seed_metrics,
        aggregate_metrics,
        profile_quality,
        run_index,
        focus_step,
        model_sync,
    )

    overview_body = _overview_body(
        out,
        manifest,
        seed_metrics,
        aggregate_metrics,
        profile_quality,
        run_index,
        focus,
        focus_step,
        model_sync,
    )
    benchmark_report = out / "benchmark_report.html"
    interactive_report = out / "interactive_report.html"
    benchmark_report.write_text(_html_shell("Benchmark V2.1 Report", overview_body), encoding="utf-8")
    interactive_report.write_text(_html_shell("Benchmark V2.1 Interactive Report", overview_body), encoding="utf-8")

    rl_architecture = out / "rl_architecture.html"
    rl_architecture.write_text(
        _html_shell("Benchmark V2.1 RL Architecture", _architecture_body(out, seed_metrics, run_index, model_sync)),
        encoding="utf-8",
    )

    first_person_dir = ensure_dir(out / "vpp_first_person")
    first_person_index = first_person_dir / "index.html"
    first_person_index.write_text(
        _html_shell("Benchmark VPP First-Person Replay", _first_person_body(out, seed_metrics, run_index, focus, focus_step)),
        encoding="utf-8",
    )
    return {
        "benchmark_report": benchmark_report,
        "interactive_report": interactive_report,
        "rl_architecture_report": rl_architecture,
        "first_person_reports": {"index": first_person_index},
        "dashboard_tables": dashboard_paths,
        "model_update_summary": dashboard_paths["model_update_summary"],
    }
