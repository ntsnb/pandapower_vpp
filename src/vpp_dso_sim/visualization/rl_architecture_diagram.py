from __future__ import annotations

from html import escape
import json
from typing import Any

import pandas as pd

from vpp_dso_sim.visualization.dashboard_data import frame_or_empty


def _text(value: object, default: str = "") -> str:
    if value is None or pd.isna(value):
        return default
    text = str(value).strip()
    return text if text else default


def _lang(en: str, zh: str, *, display: str = "inline") -> str:
    return (
        f"<span class='lang-copy lang-{display} lang-en'>{en}</span>"
        f"<span class='lang-copy lang-{display} lang-zh'>{zh}</span>"
    )


def _records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    return frame.fillna("").to_dict(orient="records") if not frame.empty else []


def _json_script(name: str, value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False).replace("</", "<\\/")
    return f"<script id='{escape(name)}' type='application/json'>{payload}</script>"


def _group_lookup(groups: pd.DataFrame) -> dict[str, dict[str, str]]:
    lookup: dict[str, dict[str, str]] = {}
    for _, row in groups.iterrows():
        group_id = _text(row.get("agent_group"))
        if not group_id:
            continue
        lookup[group_id] = {
            "color": _text(row.get("color"), "#64748b"),
            "label": _text(row.get("label"), group_id.replace("_", " ")),
            "label_zh": _text(row.get("label_zh"), _text(row.get("label"), group_id)),
            "summary": _text(row.get("summary")),
            "summary_zh": _text(row.get("summary_zh"), _text(row.get("summary"))),
        }
    return lookup


def _first_row(frame: pd.DataFrame, **filters: object) -> pd.Series:
    if frame.empty:
        return pd.Series(dtype=object)
    data = frame
    for key, value in filters.items():
        if key not in data.columns:
            return pd.Series(dtype=object)
        data = data[data[key].astype(str) == str(value)]
    return data.iloc[0] if not data.empty else pd.Series(dtype=object)


def _metric_card(label_en: str, label_zh: str, value_en: str, value_zh: str) -> str:
    return f"""
    <div class="arch-metric">
      <span>{_lang(escape(label_en), escape(label_zh))}</span>
      <strong>{_lang(escape(value_en), escape(value_zh))}</strong>
    </div>
    """


def _stage_card(
    *,
    step_en: str,
    step_zh: str,
    title_en: str,
    title_zh: str,
    summary_en: str,
    summary_zh: str,
    chip_en: str,
    chip_zh: str,
    color: str,
    modifier: str = "",
) -> str:
    classes = "arch-stage"
    if modifier:
        classes += f" {modifier}"
    return f"""
    <article class="{classes}" style="--arch-stage-color:{escape(color)}">
      <p class="arch-stage-kicker">{_lang(escape(step_en), escape(step_zh))}</p>
      <h3>{_lang(escape(title_en), escape(title_zh), display="block")}</h3>
      <p class="arch-stage-summary">{_lang(escape(summary_en), escape(summary_zh), display="block")}</p>
      <span class="arch-stage-chip">{_lang(escape(chip_en), escape(chip_zh))}</span>
    </article>
    """


def _arrow(text_en: str, text_zh: str, *, modifier: str = "") -> str:
    classes = "arch-arrow"
    if modifier:
        classes += f" {modifier}"
    return f"""
    <div class="{classes}">
      <span class="arch-arrow-glyph" aria-hidden="true">→</span>
      <p>{_lang(escape(text_en), escape(text_zh), display="block")}</p>
    </div>
    """


def _feedback_arrow(text_en: str, text_zh: str) -> str:
    return f"""
    <div class="arch-feedback-link">
      <span aria-hidden="true">↺</span>
      <p>{_lang(escape(text_en), escape(text_zh), display="block")}</p>
    </div>
    """


def _roster(groups: pd.DataFrame, agents: pd.DataFrame) -> str:
    group_lookup = _group_lookup(groups)
    cards: list[str] = []
    if agents.empty:
        return ""
    for group_id, group_frame in agents.groupby("agent_group", sort=False, dropna=False):
        meta = group_lookup.get(str(group_id), {})
        chips = []
        for _, agent in group_frame.sort_values("agent_id").iterrows():
            chips.append(
                f"""
                <button
                  type="button"
                  class="arch-agent-chip"
                  data-agent-id="{escape(_text(agent.get('agent_id')))}"
                  style="--arch-chip-color:{escape(_text(agent.get('agent_group_color'), meta.get('color', '#64748b')))}"
                >
                  <span>{escape(_text(agent.get('role_type')))}</span>
                  <strong>{escape(_text(agent.get('agent_id')))}</strong>
                </button>
                """
            )
        cards.append(
            f"""
            <article class="arch-roster-card" style="--arch-roster-color:{escape(meta.get('color', '#64748b'))}">
              <div class="arch-roster-head">
                <div class="arch-roster-dot"></div>
                <div>
                  <h3>{_lang(escape(meta.get('label', str(group_id))), escape(meta.get('label_zh', str(group_id))), display="block")}</h3>
                  <p>{_lang(escape(meta.get('summary', '')), escape(meta.get('summary_zh', meta.get('summary', ''))), display="block")}</p>
                </div>
              </div>
              <div class="arch-agent-chip-grid">{''.join(chips)}</div>
            </article>
            """
        )
    return f"""
    <div class="arch-roster-shell">
      <div class="arch-roster-grid">{''.join(cards)}</div>
    </div>
    """


def _detail_panel(detail_id: str) -> str:
    return f"""
    <div class="arch-agent-detail agent-detail-panel" id="{escape(detail_id)}">
      <p class="empty">{_lang("Select an agent chip to inspect observations, actions, reward and upgrade notes.", "点击一个智能体卡片，查看其观测、动作、奖励和升级方向。")}</p>
    </div>
    """


def _stage_data(frames: dict[str, pd.DataFrame]) -> list[dict[str, str]]:
    groups = _group_lookup(frame_or_empty(frames, "rl_agent_groups"))
    workflow = frame_or_empty(frames, "rl_step_workflow")
    overview = frame_or_empty(frames, "rl_algorithm_overview")
    bids = frame_or_empty(frames, "vpp_day_ahead_bid")
    envelopes = frame_or_empty(frames, "dso_operating_envelope")
    disaggregation = frame_or_empty(frames, "vpp_rl_disaggregation")
    projection = frame_or_empty(frames, "projection_trace")
    steps = frame_or_empty(frames, "step_summary")
    portfolios = frame_or_empty(frames, "vpp_portfolio_history")

    step1 = _first_row(workflow, step_order=1)
    step2 = _first_row(workflow, step_order=2)
    step3 = _first_row(workflow, step_order=3)
    step4 = _first_row(workflow, step_order=4)
    step5 = _first_row(workflow, step_order=5)
    step6 = _first_row(workflow, step_order=6)
    overview_row = overview.iloc[0] if not overview.empty else pd.Series(dtype=object)

    bid_step = bids["time_index"].min() if not bids.empty and "time_index" in bids.columns else None
    bid_count = (
        int(bids[bids["time_index"] == bid_step]["vpp_id"].nunique())
        if bid_step is not None and "vpp_id" in bids.columns
        else int(bids["vpp_id"].nunique()) if "vpp_id" in bids.columns else 0
    )
    envelope_step = envelopes["step"].min() if not envelopes.empty and "step" in envelopes.columns else None
    envelope_count = (
        int(envelopes[envelopes["step"] == envelope_step]["vpp_id"].nunique())
        if envelope_step is not None and "vpp_id" in envelopes.columns
        else int(envelopes["vpp_id"].nunique()) if "vpp_id" in envelopes.columns else 0
    )
    der_step = disaggregation["step"].min() if not disaggregation.empty and "step" in disaggregation.columns else None
    der_count = (
        int(disaggregation[disaggregation["step"] == der_step]["der_id"].nunique())
        if der_step is not None and "der_id" in disaggregation.columns
        else int(disaggregation["der_id"].nunique()) if "der_id" in disaggregation.columns else 0
    )
    projection_stage_count = int(projection["stage_name"].nunique()) if "stage_name" in projection.columns else 0
    min_voltage = _text(steps["min_vm_pu"].min(), "n/a") if "min_vm_pu" in steps.columns and not steps.empty else "n/a"
    max_loading = (
        f"{float(steps['max_line_loading_percent'].max()):.1f}%"
        if "max_line_loading_percent" in steps.columns and not steps.empty
        else "n/a"
    )
    portfolio_vpp_count = int(portfolios["vpp_id"].nunique()) if "vpp_id" in portfolios.columns else 0

    return [
        {
            "step_en": "Step 1",
            "step_zh": "步骤 1",
            "title_en": "VPP day-ahead bid",
            "title_zh": "VPP 日前报量 / 报价",
            "summary_en": _text(step1.get("explanation"), "Each VPP publishes a privacy-preserving capability and bid summary."),
            "summary_zh": _text(step1.get("explanation_zh"), "每个 VPP 只上报隐私保护后的能力与报价摘要。"),
            "chip_en": f"{bid_count} VPP bid summaries",
            "chip_zh": f"{bid_count} 个 VPP 报量摘要",
            "color": groups.get("vpp_dispatch", {}).get("color", "#2ca02c"),
            "modifier": "arch-stage-bid",
        },
        {
            "step_en": "Step 2",
            "step_zh": "步骤 2",
            "title_en": "DSO operating envelope",
            "title_zh": "DSO 运行包络 / 引导信号",
            "summary_en": _text(step2.get("explanation"), "The DSO converts grid stress into a safe operating envelope."),
            "summary_zh": _text(step2.get("explanation_zh"), "DSO 将电网压力转换为安全运行包络。"),
            "chip_en": f"{envelope_count} envelope targets",
            "chip_zh": f"{envelope_count} 个包络目标",
            "color": groups.get("global_guidance", {}).get("color", "#1f77b4"),
            "modifier": "arch-stage-envelope",
        },
        {
            "step_en": "Step 3",
            "step_zh": "步骤 3",
            "title_en": "VPP dispatch actors",
            "title_zh": "VPP 调度 actor",
            "summary_en": _text(step3.get("explanation"), "VPP actors choose aggregate targets inside the envelope."),
            "summary_zh": _text(step3.get("explanation_zh"), "VPP actor 在包络内选择聚合运行点。"),
            "chip_en": _text(step3.get("output"), "selected_p_mw"),
            "chip_zh": _text(step3.get("output"), "selected_p_mw"),
            "color": groups.get("vpp_dispatch", {}).get("color", "#2ca02c"),
            "modifier": "arch-stage-dispatch",
        },
        {
            "step_en": "Step 4",
            "step_zh": "步骤 4",
            "title_en": "DER actions",
            "title_zh": "DER 动作解聚合",
            "summary_en": "Aggregate intent becomes per-DER normalized actions on true physical assets.",
            "summary_zh": "聚合意图被解聚合为作用在真实物理资产上的 DER 级动作。",
            "chip_en": f"{der_count} DER action slots",
            "chip_zh": f"{der_count} 个 DER 动作位",
            "color": groups.get("vpp_dispatch", {}).get("color", "#2ca02c"),
            "modifier": "arch-stage-der",
        },
        {
            "step_en": "Step 5",
            "step_zh": "步骤 5",
            "title_en": "Safety projection",
            "title_zh": "安全投影",
            "summary_en": _text(step4.get("explanation"), "Projection clips infeasible actions and repairs residuals before writes."),
            "summary_zh": _text(step4.get("explanation_zh"), "在写入前，投影层先裁剪不可行动作并修复残差。"),
            "chip_en": f"{projection_stage_count} audit stages",
            "chip_zh": f"{projection_stage_count} 个审计阶段",
            "color": groups.get("training_supervisor", {}).get("color", "#ff7f0e"),
            "modifier": "arch-stage-projection",
        },
        {
            "step_en": "Step 6",
            "step_zh": "步骤 6",
            "title_en": "pandapower runpp",
            "title_zh": "pandapower runpp",
            "summary_en": "The physical feeder evaluates voltages, line loading and convergence on true buses and elements.",
            "summary_zh": "物理馈线在真实母线和元件上校核电压、线载率和潮流收敛。",
            "chip_en": f"min V={min_voltage}, max line={max_loading}",
            "chip_zh": f"最低电压 {min_voltage}，最高线载率 {max_loading}",
            "color": "#d94f3d",
            "modifier": "arch-stage-grid arch-stage-environment",
        },
        {
            "step_en": "Step 7",
            "step_zh": "步骤 7",
            "title_en": "Reward / critic / training update",
            "title_zh": "奖励 / critic / 训练更新",
            "summary_en": _text(step5.get("explanation"), "Training optimizes shaped reward while logging raw objective diagnostics."),
            "summary_zh": _text(step5.get("explanation_zh"), "训练器优化整形 reward，同时保留原始目标诊断项。"),
            "chip_en": _text(overview_row.get("loss_formula"), "policy + value + entropy"),
            "chip_zh": _text(overview_row.get("loss_formula"), "policy + value + entropy"),
            "color": groups.get("training_supervisor", {}).get("color", "#ff7f0e"),
            "modifier": "arch-stage-training",
        },
        {
            "step_en": "Slow loop",
            "step_zh": "慢周期",
            "title_en": "Portfolio review",
            "title_zh": "组合配置评估",
            "summary_en": _text(step6.get("explanation"), "Commercial portfolio changes stay on a slower loop and feed future bids."),
            "summary_zh": _text(step6.get("explanation_zh"), "商业组合调整运行在慢周期，并反向影响未来报价。"),
            "chip_en": f"{portfolio_vpp_count} portfolio agents",
            "chip_zh": f"{portfolio_vpp_count} 个组合智能体",
            "color": groups.get("vpp_portfolio", {}).get("color", "#9467bd"),
            "modifier": "arch-stage-portfolio",
        },
    ]


def rl_architecture_diagram_css() -> str:
    return """
    .architecture-workflow-panel {
      position: relative;
      overflow: hidden;
    }
    .arch-metrics {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
      gap: 10px;
      margin: 14px 0 18px;
    }
    .arch-metric {
      border: 1px solid rgba(148, 163, 184, 0.22);
      border-radius: 14px;
      padding: 11px 12px;
      background: linear-gradient(180deg, rgba(245, 249, 252, 0.98), rgba(255, 255, 255, 0.98));
    }
    .arch-metric span {
      display: block;
      color: var(--muted, #5d7084);
      font-size: 11px;
      font-weight: 900;
      letter-spacing: 0.08em;
      text-transform: uppercase;
      margin-bottom: 6px;
    }
    .arch-metric strong {
      display: block;
      color: var(--text, #132235);
      line-height: 1.45;
      overflow-wrap: anywhere;
    }
    .arch-main-lane {
      display: grid;
      grid-template-columns: repeat(20, minmax(0, 1fr));
      gap: 10px;
      align-items: stretch;
      margin-bottom: 16px;
    }
    .arch-main-lane > .arch-stage,
    .arch-main-lane > .arch-arrow {
      min-width: 0;
    }
    .arch-stage {
      position: relative;
      border: 1px solid color-mix(in srgb, var(--arch-stage-color) 38%, #d9e3ee);
      border-radius: 18px;
      padding: 14px;
      background:
        radial-gradient(circle at top right, color-mix(in srgb, var(--arch-stage-color) 16%, transparent), transparent 45%),
        linear-gradient(180deg, rgba(255, 255, 255, 0.99), rgba(247, 250, 252, 0.99));
      box-shadow: 0 10px 22px rgba(15, 23, 42, 0.06);
    }
    .arch-stage::before {
      content: "";
      position: absolute;
      inset: 0 auto 0 0;
      width: 6px;
      border-radius: 18px 0 0 18px;
      background: var(--arch-stage-color);
    }
    .arch-stage-kicker {
      margin: 0 0 8px;
      color: var(--arch-stage-color);
      font-size: 11px;
      font-weight: 900;
      letter-spacing: 0.09em;
      text-transform: uppercase;
    }
    .arch-stage h3 {
      margin: 0 0 8px;
      font-size: 17px;
      line-height: 1.25;
    }
    .arch-stage-summary {
      margin: 0;
      color: #30475d;
      line-height: 1.58;
      min-height: 84px;
    }
    .arch-stage-chip {
      display: inline-flex;
      margin-top: 12px;
      padding: 6px 10px;
      border-radius: 999px;
      background: color-mix(in srgb, var(--arch-stage-color) 14%, white);
      color: color-mix(in srgb, var(--arch-stage-color) 72%, #0f172a);
      font-size: 12px;
      font-weight: 900;
      line-height: 1.35;
      overflow-wrap: anywhere;
    }
    .arch-stage-environment {
      background:
        radial-gradient(circle at top right, rgba(217, 79, 61, 0.10), transparent 45%),
        linear-gradient(180deg, rgba(255, 252, 252, 0.99), rgba(255, 245, 243, 0.99));
    }
    .arch-stage-bid { grid-column: span 2; }
    .arch-stage-envelope { grid-column: span 2; }
    .arch-stage-dispatch { grid-column: span 2; }
    .arch-stage-der { grid-column: span 2; }
    .arch-stage-projection { grid-column: span 2; }
    .arch-stage-grid { grid-column: span 2; }
    .arch-arrow {
      display: flex;
      flex-direction: column;
      align-items: center;
      justify-content: center;
      text-align: center;
      gap: 6px;
      color: var(--accent, #1769aa);
      padding: 0 2px;
    }
    .arch-arrow-glyph {
      font-size: 24px;
      font-weight: 900;
      line-height: 1;
    }
    .arch-arrow p,
    .arch-feedback-link p,
    .arch-roster-head p,
    .arch-agent-detail p {
      margin: 0;
      color: #30475d;
      line-height: 1.55;
      overflow-wrap: anywhere;
    }
    .arch-flow-footer {
      display: grid;
      grid-template-columns: minmax(0, 1.35fr) minmax(240px, 0.95fr);
      gap: 14px;
      align-items: stretch;
      margin-bottom: 18px;
    }
    .arch-feedback-shell {
      display: grid;
      grid-template-columns: minmax(0, 1fr) auto;
      gap: 12px;
      align-items: stretch;
    }
    .arch-feedback-link {
      display: flex;
      flex-direction: column;
      justify-content: center;
      align-items: center;
      gap: 6px;
      padding: 12px;
      border: 1px dashed rgba(148, 163, 184, 0.55);
      border-radius: 16px;
      background: rgba(248, 251, 253, 0.92);
      text-align: center;
    }
    .arch-feedback-link span {
      color: var(--accent, #1769aa);
      font-size: 26px;
      font-weight: 900;
      line-height: 1;
    }
    .arch-roster-shell {
      margin-top: 6px;
    }
    .arch-roster-grid {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(230px, 1fr));
      gap: 12px;
    }
    .arch-roster-card {
      border: 1px solid color-mix(in srgb, var(--arch-roster-color) 28%, #d9e3ee);
      border-radius: 16px;
      padding: 14px;
      background: white;
    }
    .arch-roster-head {
      display: grid;
      grid-template-columns: auto 1fr;
      gap: 10px;
      align-items: start;
      margin-bottom: 12px;
    }
    .arch-roster-head h3 {
      margin: 0 0 6px;
      font-size: 15px;
    }
    .arch-roster-dot {
      width: 13px;
      height: 13px;
      border-radius: 999px;
      margin-top: 3px;
      background: var(--arch-roster-color);
      box-shadow: 0 0 0 5px color-mix(in srgb, var(--arch-roster-color) 18%, transparent);
    }
    .arch-agent-chip-grid {
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
    }
    .arch-agent-chip {
      border: 1px solid color-mix(in srgb, var(--arch-chip-color) 36%, #d9e3ee);
      border-radius: 14px;
      padding: 8px 10px;
      background: color-mix(in srgb, var(--arch-chip-color) 8%, white);
      color: #132235;
      text-align: left;
      cursor: pointer;
      min-width: 0;
      box-shadow: 0 5px 16px rgba(15, 23, 42, 0.05);
    }
    .arch-agent-chip.is-selected {
      outline: 3px solid color-mix(in srgb, var(--arch-chip-color) 30%, transparent);
      background: color-mix(in srgb, var(--arch-chip-color) 12%, white);
    }
    .arch-agent-chip span {
      display: block;
      color: color-mix(in srgb, var(--arch-chip-color) 70%, #0f172a);
      font-size: 10px;
      font-weight: 900;
      letter-spacing: 0.06em;
      text-transform: uppercase;
      margin-bottom: 5px;
    }
    .arch-agent-chip strong {
      display: block;
      overflow-wrap: anywhere;
      line-height: 1.4;
    }
    .arch-agent-detail {
      margin-top: 14px;
      border: 1px solid rgba(148, 163, 184, 0.25);
      border-radius: 16px;
      padding: 15px;
      background: #f8fbfd;
    }
    .arch-agent-detail .status-pill {
      display: inline-flex;
      align-items: center;
      border-radius: 999px;
      padding: 6px 9px;
      background: #eaf4fc;
      color: #10517f;
      font-size: 12px;
      font-weight: 900;
    }
    .arch-detail-status-stack {
      display: flex;
      flex-wrap: wrap;
      justify-content: flex-end;
      gap: 7px;
      align-content: start;
    }
    .arch-detail-head {
      display: flex;
      justify-content: space-between;
      gap: 12px;
      border-left: 7px solid var(--arch-detail-color);
      padding-left: 12px;
      margin-bottom: 12px;
    }
    .arch-detail-head h3 {
      margin: 2px 0 0;
      font-size: 21px;
    }
    .arch-detail-grid {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(240px, 1fr));
      gap: 10px;
    }
    .arch-detail-grid section {
      border: 1px solid #e3edf5;
      border-radius: 13px;
      padding: 12px;
      background: white;
      min-width: 0;
    }
    .arch-detail-grid h4 {
      margin: 0 0 7px;
      font-size: 12px;
      color: var(--muted, #5d7084);
      text-transform: uppercase;
      letter-spacing: 0.06em;
    }
    .arch-detail-grid p {
      margin: 0;
    }
    @media (max-width: 1200px) {
      .arch-main-lane {
        grid-template-columns: 1fr;
      }
      .arch-arrow {
        padding: 2px 0;
      }
      .arch-arrow-glyph {
        transform: rotate(90deg);
      }
      .arch-stage-summary {
        min-height: 0;
      }
      .arch-flow-footer {
        grid-template-columns: 1fr;
      }
    }
    @media (max-width: 720px) {
      .arch-feedback-shell {
        grid-template-columns: 1fr;
      }
      .arch-feedback-link {
        min-height: 0;
      }
    }
    """


def _detail_script(root_id: str, detail_id: str, data_id: str) -> str:
    return f"""
    (function() {{
      const root = document.getElementById({root_id!r});
      const payload = document.getElementById({data_id!r});
      const detail = document.getElementById({detail_id!r});
      if (!root || !payload || !detail) return;
      const agents = JSON.parse(payload.textContent || '[]');
      const byId = new Map(agents.map(item => [String(item.agent_id), item]));

      function activeLang() {{
        return document.documentElement.getAttribute('data-lang') || 'en';
      }}

      function pick(item, enKey, zhKey) {{
        const lang = activeLang();
        return String((lang === 'zh' ? item[zhKey || enKey + '_zh'] : item[enKey]) || item[enKey] || '');
      }}

      function htmlEscape(value) {{
        return String(value).replace(/[&<>\"']/g, ch => ({{'&':'&amp;', '<':'&lt;', '>':'&gt;', '\"':'&quot;', \"'\":'&#39;'}}[ch]));
      }}

      function renderAgent(agentId) {{
        const item = byId.get(String(agentId));
        if (!item) return;
        const isRl = String(item.is_rl_decision).toLowerCase() === 'true';
        const rlBadge = activeLang() === 'zh'
          ? (isRl ? '使用 RL 决策' : '非 RL 环境智能体')
          : (isRl ? 'Uses RL' : 'Not an RL env agent');
        detail.innerHTML = `
          <div class="arch-detail-head" style="--arch-detail-color:${{htmlEscape(item.agent_group_color || '#64748b')}}">
            <div>
              <p class="eyebrow">${{htmlEscape(pick(item, 'agent_group_label', 'agent_group_label_zh'))}}</p>
              <h3>${{htmlEscape(item.agent_id || '')}}</h3>
            </div>
            <div class="arch-detail-status-stack">
              <span class="status-pill">${{htmlEscape(rlBadge)}}</span>
              <span class="status-pill">${{htmlEscape(pick(item, 'implementation_status', 'implementation_status_zh'))}}</span>
            </div>
          </div>
          <div class="arch-detail-grid">
            <section><h4>${{activeLang() === 'zh' ? '输入观测' : 'Input observation'}}</h4><p>${{htmlEscape(pick(item, 'input_observation', 'input_observation_zh'))}}</p></section>
            <section><h4>${{activeLang() === 'zh' ? '动作 / 输出' : 'Action / output'}}</h4><p>${{htmlEscape(pick(item, 'action_output', 'action_output_zh'))}}</p></section>
            <section><h4>${{activeLang() === 'zh' ? '是否使用强化学习' : 'Uses RL?'}}</h4><p>${{htmlEscape(pick(item, 'rl_usage_status', 'rl_usage_status_zh'))}}</p></section>
            <section><h4>${{activeLang() === 'zh' ? '神经网络结构' : 'Neural network structure'}}</h4><p>${{htmlEscape(pick(item, 'neural_network_structure', 'neural_network_structure_zh'))}}</p></section>
            <section><h4>${{activeLang() === 'zh' ? '输出公式' : 'Output formula'}}</h4><p>${{htmlEscape(pick(item, 'result_formula', 'result_formula_zh'))}}</p></section>
            <section><h4>${{activeLang() === 'zh' ? '结果怎么算' : 'Result calculation'}}</h4><p>${{htmlEscape(pick(item, 'result_calculation', 'result_calculation_zh'))}}</p></section>
            <section><h4>${{activeLang() === 'zh' ? '结果来源' : 'Result source'}}</h4><p>${{htmlEscape(pick(item, 'result_source', 'result_source_zh'))}}</p></section>
            <section><h4>${{activeLang() === 'zh' ? '策略模块' : 'Policy module'}}</h4><p>${{htmlEscape(pick(item, 'policy_module', 'policy_module_zh'))}}</p></section>
            <section><h4>${{activeLang() === 'zh' ? '训练信号' : 'Training signal'}}</h4><p>${{htmlEscape(pick(item, 'rl_training_signal', 'rl_training_signal_zh'))}}</p></section>
            <section><h4>${{activeLang() === 'zh' ? '奖励函数' : 'Reward function'}}</h4><p>${{htmlEscape(pick(item, 'reward_function', 'reward_function_zh'))}}</p></section>
            <section><h4>${{activeLang() === 'zh' ? '审计输出' : 'Audit outputs'}}</h4><p>${{htmlEscape(pick(item, 'audit_outputs', 'audit_outputs_zh'))}}</p></section>
            <section><h4>${{activeLang() === 'zh' ? '非 RL 安全门' : 'Non-RL guardrails'}}</h4><p>${{htmlEscape(pick(item, 'non_rl_guardrails', 'non_rl_guardrails_zh'))}}</p></section>
            <section><h4>${{activeLang() === 'zh' ? '本步职责' : 'Role in one step'}}</h4><p>${{htmlEscape(pick(item, 'current_step_role', 'current_step_role_zh'))}}</p></section>
            <section><h4>${{activeLang() === 'zh' ? '下一步升级' : 'Next upgrade'}}</h4><p>${{htmlEscape(pick(item, 'next_upgrade', 'next_upgrade_zh'))}}</p></section>
          </div>`;
        root.querySelectorAll('.arch-agent-chip').forEach(node => {{
          node.classList.toggle('is-selected', node.dataset.agentId === String(agentId));
        }});
      }}

      root.addEventListener('click', event => {{
        const chip = event.target.closest('.arch-agent-chip[data-agent-id]');
        if (chip) renderAgent(chip.dataset.agentId);
      }});

      document.addEventListener('click', event => {{
        const langButton = event.target.closest('[data-lang-switch]');
        if (!langButton) return;
        window.requestAnimationFrame(() => {{
          const selected = root.querySelector('.arch-agent-chip.is-selected');
          if (selected) renderAgent(selected.dataset.agentId);
        }});
      }});

      const first = root.querySelector('.arch-agent-chip[data-agent-id]');
      if (first) renderAgent(first.dataset.agentId);
    }})();
    """


def build_rl_architecture_diagram(
    frames: dict[str, pd.DataFrame],
    *,
    root_id: str,
    heading_eyebrow_en: str,
    heading_eyebrow_zh: str,
    title_en: str,
    title_zh: str,
    description_en: str,
    description_zh: str,
    heading_class: str,
) -> str:
    groups = frame_or_empty(frames, "rl_agent_groups")
    agents = frame_or_empty(frames, "rl_agent_architecture")
    overview = frame_or_empty(frames, "rl_algorithm_overview")
    overview_row = overview.iloc[0] if not overview.empty else pd.Series(dtype=object)
    stages = _stage_data(frames)
    detail_id = f"{root_id}-agent-detail"
    data_id = f"{root_id}-agent-data"

    metrics = [
        _metric_card(
            "Global guidance",
            "全局引导",
            _text(overview_row.get("training_mode"), "centralized_training"),
            _text(overview_row.get("training_mode"), "集中训练"),
        ),
        _metric_card(
            "Execution path",
            "执行路径",
            "bid -> envelope -> dispatch -> runpp",
            "报价 -> 包络 -> 调度 -> runpp",
        ),
        _metric_card(
            "Safety gate",
            "安全门",
            "FR/DOE + device limits + residual repair",
            "FR/DOE + 设备边界 + 残差修复",
        ),
        _metric_card(
            "Projection chain",
            "投影链路",
            "raw action -> device bounds -> FR/DOE -> residual repair -> pandapower write",
            "原始动作 -> 设备边界 -> FR/DOE -> 残差修复 -> pandapower 写入",
        ),
        _metric_card(
            "Supervisor scope",
            "监督器边界",
            "experiment orchestrator, not env-step MARL actor or LLM policy",
            "实验编排器，不是环境 step 内的 MARL actor，也不是 LLM policy",
        ),
        _metric_card(
            "Decision provenance",
            "决策来源",
            "agent cards mark RL heads, deterministic projection and audit files",
            "智能体卡片标明 RL head、确定性投影和审计文件",
        ),
        _metric_card(
            "Training feedback",
            "训练反馈",
            _text(overview_row.get("loss_formula"), "policy + value + entropy"),
            _text(overview_row.get("loss_formula"), "policy + value + entropy"),
        ),
    ]

    main = stages[:7]
    portfolio = stages[7]
    arrows = [
        ("aggregate capability, price, confidence", "聚合能力、价格、置信度"),
        ("safe envelope and service intent", "安全包络与服务意图"),
        ("selected_p_mw", "selected_p_mw"),
        ("per-DER normalized actions", "DER 级归一化动作"),
        ("bounded writes to true elements", "约束后写入真实元件"),
        ("reward, violations, critic state", "奖励、违约与 critic 状态"),
    ]

    lane_parts: list[str] = []
    for index, stage in enumerate(main):
        lane_parts.append(
            _stage_card(
                step_en=stage["step_en"],
                step_zh=stage["step_zh"],
                title_en=stage["title_en"],
                title_zh=stage["title_zh"],
                summary_en=stage["summary_en"],
                summary_zh=stage["summary_zh"],
                chip_en=stage["chip_en"],
                chip_zh=stage["chip_zh"],
                color=stage["color"],
                modifier=stage["modifier"],
            )
        )
        if index < len(arrows):
            arrow_en, arrow_zh = arrows[index]
            lane_parts.append(_arrow(arrow_en, arrow_zh))

    portfolio_card = _stage_card(
        step_en=portfolio["step_en"],
        step_zh=portfolio["step_zh"],
        title_en=portfolio["title_en"],
        title_zh=portfolio["title_zh"],
        summary_en=portfolio["summary_en"],
        summary_zh=portfolio["summary_zh"],
        chip_en=portfolio["chip_en"],
        chip_zh=portfolio["chip_zh"],
        color=portfolio["color"],
        modifier=portfolio["modifier"],
    )

    return f"""
    <section class="panel architecture-workflow-panel" id="{escape(root_id)}">
      <div class="{escape(heading_class)}">
        <div>
          <p class="eyebrow">{_lang(escape(heading_eyebrow_en), escape(heading_eyebrow_zh))}</p>
          <h2>{_lang(escape(title_en), escape(title_zh))}</h2>
        </div>
        <p class="section-note">{_lang(escape(description_en), escape(description_zh), display="block")}</p>
      </div>
      <div class="arch-metrics">{''.join(metrics)}</div>
      <div class="arch-main-lane">{''.join(lane_parts)}</div>
      <div class="arch-flow-footer">
        <div class="arch-feedback-shell">
          {_feedback_arrow(
              "critic and optimizer updates feed the next DSO/VPP actor rollout",
              "critic 与优化器更新会反向影响下一轮 DSO / VPP actor 执行",
          )}
        </div>
        <div class="arch-feedback-shell">
          {portfolio_card}
          {_feedback_arrow(
              "slow-cycle portfolio changes shape future bids without moving physical buses",
              "慢周期组合调整影响未来报价，但不改变 DER 物理母线",
          )}
        </div>
      </div>
      {_roster(groups, agents)}
      {_detail_panel(detail_id)}
      {_json_script(data_id, _records(agents))}
      <script>{_detail_script(root_id, detail_id, data_id)}</script>
    </section>
    """
