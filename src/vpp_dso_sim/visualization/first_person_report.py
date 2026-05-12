from __future__ import annotations

from html import escape
from pathlib import Path

import pandas as pd

from vpp_dso_sim.utils.io import ensure_dir
from vpp_dso_sim.visualization.dashboard_data import frame_or_empty, model_update_summary_frame


COLUMN_ZH = {
    "step": "时间步",
    "time_label": "时间",
    "vpp_id": "VPP 编号",
    "command_seen": "收到的指令",
    "need_label": "电网需求判断",
    "belief_label": "状态判断",
    "action_label": "动作",
    "target_p_mw": "目标 P/MW",
    "raw_target_p_mw": "原始目标 P/MW",
    "projected_p_mw": "投影后 P/MW",
    "actual_p_mw": "实际 P/MW",
    "tracking_error_mw": "跟踪误差/MW",
    "decision_status": "决策状态",
    "event_order": "事件顺序",
    "event_type": "事件类型",
    "event_title": "事件标题",
    "event_detail": "事件说明",
    "event_detail_zh": "中文事件说明",
    "event_detail_raw": "原始审计字段",
    "portfolio_version": "组合版本",
    "physical_mode": "物理接入模式",
    "connection_buses": "接入母线",
    "scope_type": "范围类型",
    "scope_id": "范围编号",
    "bus_id": "母线编号",
    "asset_count": "资源数",
    "p_lower_mw": "P 下限/MW",
    "p_upper_mw": "P 上限/MW",
    "current_p_mw": "当前 P/MW",
    "is_binding": "是否约束绑定",
    "is_remote_bus": "是否远端母线",
    "window_id": "周期窗口",
    "start_time": "开始时间",
    "end_time": "结束时间",
    "dominant_grid_need": "主导电网需求",
    "dominant_action": "主导动作",
    "reliability_score": "可靠性得分",
    "non_delivery_rate": "未履约率",
    "projection_count": "投影次数",
    "portfolio_change_count": "组合调整次数",
    "risk_level": "风险等级",
    "portfolio_recommendation": "组合建议",
    "recommendation_reason": "建议原因",
    "event_id": "事件编号",
    "effective_step": "生效时间步",
    "from_vpp_id": "原 VPP",
    "to_vpp_id": "目标 VPP",
    "der_id": "DER 编号",
    "why_now": "为什么此时调整",
    "expected_effect": "预期效果",
    "post_change_effect": "调整后观察",
    "physical_bus_unchanged": "物理母线未改变",
    "physical_element_unchanged": "pandapower 元件未改变",
    "metric": "指标",
    "value": "数值",
    "share_of_total_cost": "总成本占比",
    "formula": "计算公式",
    "interpretation": "含义",
    "why_negative": "为什么可能为负",
    "update_area": "更新范围",
    "current_value": "当前值",
    "current_value_zh": "当前值中文说明",
    "explanation": "说明",
    "explanation_zh": "中文说明",
    "evidence_file": "证据文件",
}


VALUE_ZH = {
    "price_driven_baseline": "电价驱动基线",
    "external_action": "外部/智能体指令",
    "received_instruction": "收到 DSO 或基线目标",
    "received_context": "读取本地价格、负荷和 PV 预测",
    "formed_belief": "形成电网需求和交付风险判断",
    "made_dispatch_decision": "选择 VPP 聚合响应",
    "projection_result": "检查 FR/DOE 可行域投影",
    "der_dispatch": "分解到内部 DER 响应",
    "delivery_feedback": "观察实际交付与结算代理",
    "increase_export_or_reduce_load": "提高注入或降低负荷",
    "absorb_power_or_charge": "吸收功率或充电",
    "hold_near_neutral": "保持接近中性功率",
    "accepted_and_delivered": "接受并完成交付",
    "projected_to_fr_doe": "被 FR/DOE 投影修正",
    "tracking_gap": "存在跟踪偏差",
    "single_pcc": "单 PCC 接入",
    "multi_node": "多节点接入",
    "high_price_export_request": "高电价：鼓励上调注入",
    "low_price_absorption_request": "低电价：鼓励吸收/充电",
    "high_load_voltage_support": "高负荷：需要电压支撑",
    "pv_rich_local_absorption": "PV 丰富：需要本地消纳",
    "normal_balancing": "常规平衡",
    "absorb_down": "降低净注入/增加吸收",
    "export_up": "提高净注入/降低负荷",
    "balance": "保持平衡",
    "operate_within_envelope": "在可行包络内运行",
    "watch_projection_frequency": "关注频繁投影",
    "low": "低",
    "medium": "中",
    "high": "高",
    "keep_current_portfolio": "保持当前聚合配置",
    "expand_or_reweight_flexibility_on_binding_scopes": "在约束绑定范围扩展或重配灵活性",
    "tighten_dispatch_tracking_or_reserve_margin": "加强跟踪或预留备用裕度",
    "seek_absorption_service_payment": "为吸收型服务争取补偿价格",
    "algorithm": "算法",
    "privacy_and_ctde": "隐私边界与CTDE",
    "actor_stack": "智能体策略栈",
    "critic_and_loss": "集中critic与损失函数",
    "reward_and_checkpoint": "奖励与checkpoint",
    "ui_refresh_contract": "可视化同步约定",
    "target_ctde_status": "目标CTDE状态",
}


def _text(value: object, default: str = "") -> str:
    if value is None or pd.isna(value):
        return default
    text = str(value).strip()
    return text if text else default


def _safe_name(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in value)


def _lang(en: str, zh: str, *, block: bool = False) -> str:
    display = "block" if block else "inline"
    return (
        f"<span class='lang-copy lang-{display} lang-en'>{en}</span>"
        f"<span class='lang-copy lang-{display} lang-zh'>{zh}</span>"
    )


def _column_label(column: str) -> str:
    return _lang(escape(column), escape(COLUMN_ZH.get(column, column)))


def _translate_value(value: object) -> str:
    raw = _text(value)
    if not raw:
        return ""
    translated = VALUE_ZH.get(raw)
    if translated:
        return _lang(escape(raw), escape(translated))
    return escape(raw)


def _float_value(value: object, default: float = 0.0) -> float:
    try:
        if value is None or pd.isna(value):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _power_direction_en(p_mw: float) -> str:
    if p_mw > 0.005:
        return "injecting power to the grid"
    if p_mw < -0.005:
        return "absorbing power from the grid"
    return "near zero exchange"


def _power_direction_zh(p_mw: float) -> str:
    if p_mw > 0.005:
        return "向电网注入功率"
    if p_mw < -0.005:
        return "从电网吸收功率"
    return "接近零功率交换"


def _signed_power_lang(value: object) -> str:
    p_mw = _float_value(value)
    return _lang(
        f"{p_mw:.3f} MW, {escape(_power_direction_en(p_mw))}",
        f"{p_mw:.3f} MW，{escape(_power_direction_zh(p_mw))}",
    )


def _risk_band(value: object) -> tuple[str, str]:
    risk = _float_value(value)
    if risk < 0.2:
        return "low execution risk", "执行风险低"
    if risk < 0.6:
        return "medium execution risk", "执行风险中等"
    return "high execution risk", "执行风险高"


def _need_strength(value: object) -> tuple[str, str]:
    score = _float_value(value)
    if score < 0.35:
        return "weak signal", "需求信号较弱"
    if score < 0.7:
        return "medium signal", "需求信号中等"
    return "strong signal", "需求信号较强"


def _event_metric_strip(event: pd.Series) -> str:
    raw_target = _float_value(event.get("raw_target_p_mw"))
    projected = _float_value(event.get("projected_p_mw"))
    delivered = _float_value(event.get("delivered_p_mw"))
    status = _text(event.get("decision_status"), "n/a")
    return f"""
      <div class="event-metrics" aria-label="Key values">
        <div><span>{_lang("Requested", "请求目标")}</span><strong>{_signed_power_lang(raw_target)}</strong></div>
        <div><span>{_lang("After projection", "修正后目标")}</span><strong>{_signed_power_lang(projected)}</strong></div>
        <div><span>{_lang("Delivered", "实际交付")}</span><strong>{_signed_power_lang(delivered)}</strong></div>
        <div><span>{_lang("Status", "状态")}</span><strong>{_translate_value(status)}</strong></div>
      </div>
    """


def _table(frame: pd.DataFrame, columns: list[str], max_rows: int = 400) -> str:
    data = frame.copy()
    for column in columns:
        if column not in data:
            data[column] = ""
    data = data[columns].head(max_rows)
    head = "".join(f"<th>{_column_label(str(col))}</th>" for col in columns)
    body_rows = []
    for row in data.itertuples(index=False, name=None):
        body_rows.append("<tr>" + "".join(f"<td>{_translate_value(value)}</td>" for value in row) + "</tr>")
    return f"<div class='table-wrap'><table><thead><tr>{head}</tr></thead><tbody>{''.join(body_rows)}</tbody></table></div>"


def _toolbar() -> str:
    return """
    <div class="language-toolbar" role="group" aria-label="Language">
      <span>Language / 语言</span>
      <button type="button" class="lang-button" data-lang-switch="en">EN</button>
      <button type="button" class="lang-button is-active" data-lang-switch="zh">中文</button>
    </div>
    """


def _language_script() -> str:
    return """
    (function() {
      var storageKey = 'vpp-first-person-lang';
      function applyLang(lang) {
        var root = document.documentElement;
        root.setAttribute('data-lang', lang);
        root.setAttribute('lang', lang === 'zh' ? 'zh-CN' : 'en');
        document.querySelectorAll('[data-lang-switch]').forEach(function(button) {
          var active = button.getAttribute('data-lang-switch') === lang;
          button.classList.toggle('is-active', active);
          button.setAttribute('aria-pressed', active ? 'true' : 'false');
        });
        try { window.localStorage.setItem(storageKey, lang); } catch (err) { void err; }
      }
      var initial = 'zh';
      try { initial = window.localStorage.getItem(storageKey) || initial; } catch (err) { void err; }
      document.addEventListener('click', function(event) {
        var button = event.target.closest('[data-lang-switch]');
        if (button) { applyLang(button.getAttribute('data-lang-switch') || 'zh'); }
      });
      applyLang(initial);
    })();
    """


def _base_html(title_en: str, title_zh: str, body: str) -> str:
    return f"""<!doctype html>
<html lang="zh-CN" data-lang="zh">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>{escape(title_zh)}</title>
  <style>
    :root {{
      --bg: #eef4f8;
      --panel: #ffffff;
      --line: #d9e4ee;
      --text: #122033;
      --muted: #5c6f81;
      --accent: #0d74bd;
      --accent-soft: #e8f4fc;
      --warn: #b75d00;
      --bad: #a32929;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: "Segoe UI", "Microsoft YaHei", Arial, sans-serif;
      color: var(--text);
      background: linear-gradient(180deg, #dfeaf3 0%, var(--bg) 32%, #f8fbfd 100%);
    }}
    .lang-copy {{ display: none; }}
    html[data-lang="en"] .lang-en.lang-inline,
    html[data-lang="zh"] .lang-zh.lang-inline {{ display: inline; }}
    html[data-lang="en"] .lang-en.lang-block,
    html[data-lang="zh"] .lang-zh.lang-block {{ display: block; }}
    header {{
      padding: 24px 30px;
      background: linear-gradient(135deg, #071d33, #123b62);
      color: white;
    }}
    header h1 {{ margin: 0 0 8px; font-size: 30px; }}
    header p {{ margin: 0; color: #dbe8f5; line-height: 1.55; }}
    main {{ padding: 22px 28px 38px; }}
    a {{ color: var(--accent); text-decoration: none; font-weight: 700; }}
    .language-toolbar {{
      display: inline-flex; align-items: center; gap: 8px; margin-top: 16px;
      padding: 6px; border-radius: 999px; background: rgba(255,255,255,0.10);
      border: 1px solid rgba(255,255,255,0.18);
    }}
    .language-toolbar span {{ color: #dbe8f5; font-size: 12px; padding: 0 6px; }}
    .lang-button {{
      border: 0; border-radius: 999px; padding: 7px 12px; cursor: pointer;
      background: transparent; color: #dbe8f5; font-weight: 700;
    }}
    .lang-button.is-active {{ background: rgba(143,208,255,0.22); color: white; }}
    .nav {{ display: flex; flex-wrap: wrap; gap: 10px; margin-bottom: 16px; }}
    .nav a {{
      display: inline-flex; padding: 8px 11px; border-radius: 999px;
      background: var(--accent-soft); border: 1px solid #c8e3f4;
    }}
    .panel {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 14px;
      padding: 16px 18px;
      margin: 0 0 16px;
      box-shadow: 0 10px 24px rgba(15, 35, 56, 0.06);
    }}
    .kpi-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(170px, 1fr)); gap: 12px; }}
    .kpi {{
      border: 1px solid var(--line); border-radius: 12px; padding: 12px;
      background: linear-gradient(180deg, #f8fbfd, #fff);
    }}
    .kpi span {{ display: block; color: var(--muted); font-size: 12px; font-weight: 700; margin-bottom: 6px; }}
    .kpi strong {{ font-size: 20px; }}
    .toolbar {{
      position: sticky; top: 0; z-index: 5;
      display: flex; flex-wrap: wrap; gap: 12px; align-items: center;
      padding: 12px; border: 1px solid var(--line); border-radius: 12px;
      background: rgba(255, 255, 255, 0.96); margin-bottom: 14px;
    }}
    .toolbar input[type=range] {{ width: min(520px, 100%); }}
    .event-grid {{ display: grid; gap: 10px; }}
    .event-card {{
      border: 1px solid var(--line);
      border-left: 5px solid var(--accent);
      border-radius: 12px;
      padding: 12px 13px;
      background: #fff;
    }}
    .event-card[data-status="projected_to_fr_doe"] {{ border-left-color: var(--warn); }}
    .event-card[data-status="tracking_gap"] {{ border-left-color: var(--bad); }}
    .event-head {{ display: flex; justify-content: space-between; gap: 12px; margin-bottom: 8px; }}
    .event-head h3 {{ margin: 0; font-size: 16px; }}
    .event-readable {{
      margin: 0 0 12px;
      line-height: 1.75;
      font-size: 15px;
      color: #182b3f;
    }}
    .event-metrics {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(185px, 1fr));
      gap: 8px;
      margin: 10px 0 8px;
    }}
    .event-metrics div {{
      border: 1px solid #e5edf4;
      border-radius: 9px;
      background: #f8fbfd;
      padding: 8px 9px;
      min-width: 0;
    }}
    .event-metrics span {{ display: block; color: var(--muted); font-size: 12px; font-weight: 700; margin-bottom: 4px; }}
    .event-metrics strong {{ font-size: 13px; line-height: 1.45; overflow-wrap: anywhere; }}
    .raw-audit {{ margin-top: 8px; }}
    .raw-audit summary {{ cursor: pointer; color: var(--muted); font-weight: 700; font-size: 12px; }}
    .raw-audit code {{
      display: block;
      white-space: pre-wrap;
      overflow-wrap: anywhere;
      margin-top: 7px;
      padding: 8px 9px;
      border-radius: 8px;
      border: 1px solid #e4ebf2;
      background: #f6f8fa;
      color: #4f6172;
      font-family: Consolas, "Cascadia Mono", monospace;
      font-size: 12px;
      line-height: 1.55;
    }}
    .pill {{
      display: inline-flex; align-items: center; padding: 4px 8px;
      border-radius: 999px; background: var(--accent-soft); color: #0d4d7b;
      font-size: 12px; font-weight: 700; white-space: nowrap;
    }}
    .term-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 10px; }}
    .term-card {{ border: 1px solid var(--line); border-radius: 12px; padding: 12px; background: #fbfdff; }}
    .term-card h3 {{ margin: 0 0 6px; font-size: 15px; }}
    .term-card p {{ margin: 0; line-height: 1.6; color: var(--muted); }}
    .four-col {{ display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 10px; }}
    .mini {{
      background: #f7fafc; border: 1px solid #e5edf4; border-radius: 10px; padding: 10px;
      min-width: 0;
    }}
    .mini h4 {{ margin: 0 0 6px; font-size: 13px; color: var(--muted); }}
    .mini p {{ margin: 0; line-height: 1.5; overflow-wrap: anywhere; }}
    .table-wrap {{ overflow-x: auto; border: 1px solid var(--line); border-radius: 12px; background: white; }}
    table {{ border-collapse: collapse; width: 100%; font-size: 13px; }}
    th, td {{ border-bottom: 1px solid #e8eef4; padding: 8px 9px; text-align: left; vertical-align: top; }}
    th {{ position: sticky; top: 0; background: #f3f8fc; color: #33465a; }}
    .story {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(260px, 1fr)); gap: 12px; }}
    .story-card {{ border: 1px solid var(--line); border-radius: 12px; padding: 13px; background: #fff; }}
    .story-card h3 {{ margin: 0 0 8px; }}
    .muted {{ color: var(--muted); }}
    @media (max-width: 960px) {{ .four-col {{ grid-template-columns: 1fr 1fr; }} }}
    @media (max-width: 640px) {{ main {{ padding: 16px 12px; }} .four-col {{ grid-template-columns: 1fr; }} }}
  </style>
</head>
<body>
  <header>
    <h1>{_lang(escape(title_en), escape(title_zh))}</h1>
    <p>{_lang(
        "This report is organized from each VPP viewpoint: received instruction, visible context, inferred belief, action, DER response, and reward/economics proxy.",
        "该报告按每个 VPP 的第一视角组织：收到的指令、可见的上下文、形成的判断、采取的动作、DER 响应和奖励/经济代理。"
    )}</p>
    {_toolbar()}
  </header>
  <main>{body}</main>
  <script>{_language_script()}</script>
</body>
</html>"""


def _nav() -> str:
    return f"""
    <nav class="nav">
      <a href="index.html">{_lang("First-view index", "第一视角索引")}</a>
      <a href="long_cycle.html">{_lang("Long-cycle judgment", "长周期判断")}</a>
      <a href="economic_explanation.html">{_lang("Economics / reward explanation", "经济与 reward 解释")}</a>
    </nav>
    """


def _model_update_panel(frames: dict[str, pd.DataFrame]) -> str:
    updates = frame_or_empty(frames, "model_update_summary")
    if updates.empty:
        updates = model_update_summary_frame(frames)
    return f"""
    <section class="panel">
      <h2>{_lang("Model / Algorithm Update Summary", "模型 / 算法更新摘要")}</h2>
      <p class="muted">{_lang(
          "This report reads the same model_update_summary table as the main dashboard and RL architecture page. After the learning algorithm changes, regenerating reports updates this section automatically.",
          "本报告与主仪表盘、RL架构页读取同一张 model_update_summary 表。学习算法变化后，重新生成报告会自动同步本区域。",
          block=True,
      )}</p>
      {_table(updates, ["update_area", "current_value", "current_value_zh", "explanation", "explanation_zh", "evidence_file"], max_rows=20)}
    </section>
    """


def _index_html(frames: dict[str, pd.DataFrame], page_links: dict[str, str]) -> str:
    summary = frame_or_empty(frames, "vpp_step_decision_summary")
    long_cycle = frame_or_empty(frames, "vpp_long_cycle_judgment")
    cards = []
    for vpp_id, href in page_links.items():
        selected = summary[summary["vpp_id"].astype(str) == vpp_id] if not summary.empty else pd.DataFrame()
        long_selected = long_cycle[long_cycle["vpp_id"].astype(str) == vpp_id] if not long_cycle.empty else pd.DataFrame()
        reliability = _text(long_selected["reliability_score"].iloc[-1]) if not long_selected.empty else "n/a"
        projection_count = _text(long_selected["projection_count"].sum()) if not long_selected.empty else "n/a"
        avg_p = f"{float(selected['actual_p_mw'].mean()):.4f} MW" if not selected.empty else "n/a"
        cards.append(
            f"""
            <article class="story-card">
              <h3>{escape(vpp_id)}</h3>
              <p class="muted">{_lang(
                  f"Average power {escape(avg_p)}; reliability {escape(reliability)}; projections {escape(projection_count)}.",
                  f"平均功率 {escape(avg_p)}；可靠性 {escape(reliability)}；投影次数 {escape(projection_count)}。"
              )}</p>
              <p><a href="{escape(href)}">{_lang("Open this VPP step-by-step first view", "打开该 VPP 的逐时刻第一视角")}</a></p>
            </article>
            """
        )
    body = _nav() + _model_update_panel(frames) + f"""
    <section class="panel">
      <h2>{_lang("VPP first-view entry", "VPP 第一视角入口")}</h2>
      <p class="muted">{_lang(
          "Each VPP page provides a step slider and event chain: what it received, what it inferred, what it decided and how it performed.",
          "每个 VPP 页面都提供 step 滑块与事件链：收到什么、判断什么、决策什么、执行结果如何。"
      )}</p>
      <div class="story">{''.join(cards)}</div>
    </section>
    <section class="panel">
      <h2>{_lang("Quick overview", "快速总览")}</h2>
      {_table(summary, ["step", "time_label", "vpp_id", "command_seen", "need_label", "action_label", "projected_p_mw", "actual_p_mw", "decision_status"], max_rows=120)}
    </section>
    """
    return _base_html("VPP First-Person Report Index", "VPP 第一视角报告索引", body)


def _event_title(event: pd.Series) -> str:
    title = _text(event.get("event_title"))
    translated = VALUE_ZH.get(_text(event.get("event_type")), title)
    return _lang(escape(title), escape(translated))


def _term_guide_html() -> str:
    terms = [
        (
            "P > 0",
            "The VPP injects active power into the grid, such as PV output, microturbine output or storage discharge.",
            "VPP 向电网注入有功功率，例如光伏出力、燃机出力或储能放电。",
        ),
        (
            "P < 0",
            "The VPP absorbs active power from the grid, such as storage charging, EV charging or increasing flexible load.",
            "VPP 从电网吸收有功功率，例如储能充电、EV 充电或增加柔性负荷。",
        ),
        (
            "FR / DOE",
            "Feasible Region / Dynamic Operating Envelope: the active-power interval allowed by DER limits and aggregation constraints.",
            "可行域/动态运行包络：由 DER 物理约束和聚合约束共同决定的有功功率允许区间。",
        ),
        (
            "raw_target / projected_target",
            "raw_target is the original request. projected_target is the nearest feasible value after clipping to FR/DOE.",
            "raw_target 是原始请求；projected_target 是经过 FR/DOE 可行性修正后的可执行目标。",
        ),
        (
            "need_score",
            "A 0-1 strength score for the inferred grid need. Higher means the VPP thinks the operating signal is stronger.",
            "0-1 的电网需求强度分数。数值越高，表示 VPP 判断该运行信号越强。",
        ),
        (
            "delivery_risk",
            "A 0-1 execution risk score. Lower means the VPP has more margin to deliver the target.",
            "0-1 的执行风险分数。数值越低，表示 VPP 越有余量完成目标。",
        ),
        (
            "price_driven_baseline",
            "No external RL/manual command was injected for this step; the baseline policy uses price and local forecasts.",
            "该时刻没有外部 RL/人工指令注入；基线策略根据电价和本地预测自动给出目标。",
        ),
        (
            "signed_energy_cashflow_proxy",
            "A signed settlement proxy for analysis, not accounting profit. Absorbing power at positive price can therefore be negative.",
            "用于分析的带符号结算代理，不是会计净利润；在正电价下吸收功率会表现为负数。",
        ),
    ]
    cards = []
    for title, en, zh in terms:
        cards.append(
            f"""
            <article class="term-card">
              <h3>{escape(title)}</h3>
              <p>{_lang(escape(en), escape(zh), block=True)}</p>
            </article>
            """
        )
    return f"""
    <section class="panel">
      <h2>{_lang("How to read the decision fields", "如何阅读决策字段")}</h2>
      <div class="term-grid">{''.join(cards)}</div>
    </section>
    """


def _vpp_html(vpp_id: str, frames: dict[str, pd.DataFrame]) -> str:
    summary = frame_or_empty(frames, "vpp_step_decision_summary")
    events = frame_or_empty(frames, "vpp_first_person_event_stream")
    scope = frame_or_empty(frames, "vpp_scope_step_summary")
    summary = summary[summary["vpp_id"].astype(str) == vpp_id].copy() if not summary.empty else pd.DataFrame()
    events = events[events["vpp_id"].astype(str) == vpp_id].copy() if not events.empty else pd.DataFrame()
    scope = scope[scope["vpp_id"].astype(str) == vpp_id].copy() if not scope.empty else pd.DataFrame()
    steps = summary["step"].astype(int).tolist() if not summary.empty else [0]
    min_step, max_step = min(steps), max(steps)

    event_cards = []
    for _, event in events.iterrows():
        event_type = _text(event.get("event_type"))
        detail_en = escape(_text(event.get("event_detail")))
        detail_zh = escape(_text(event.get("event_detail_zh"), _text(event.get("event_detail"))))
        raw_detail = escape(_text(event.get("event_detail_raw")))
        event_cards.append(
            f"""
            <article class="event-card" data-step="{int(event.get('step', 0))}" data-status="{escape(_text(event.get('decision_status')))}">
              <div class="event-head">
                <h3>{escape(_text(event.get('event_order')))}. {_event_title(event)}</h3>
                <span class="pill">{escape(_text(event.get('time_label')))} / {_translate_value(event_type)}</span>
              </div>
              <p class="event-readable">{_lang(detail_en, detail_zh, block=True)}</p>
              {_event_metric_strip(event)}
              <details class="raw-audit">
                <summary>{_lang("Raw audit fields", "原始审计字段")}</summary>
                <code>{raw_detail}</code>
              </details>
            </article>
            """
        )

    first = summary.iloc[0].to_dict() if not summary.empty else {}
    body = _nav() + _model_update_panel(frames) + f"""
    <section class="panel">
      <div class="kpi-grid">
        <div class="kpi"><span>VPP</span><strong>{escape(vpp_id)}</strong></div>
        <div class="kpi"><span>{_lang("Physical mode", "物理接入模式")}</span><strong>{_translate_value(first.get('physical_mode', 'n/a'))}</strong></div>
        <div class="kpi"><span>{_lang("Connection buses", "接入母线")}</span><strong>{escape(_text(first.get('connection_buses'), 'n/a'))}</strong></div>
        <div class="kpi"><span>{_lang("Steps", "时间步数")}</span><strong>{len(summary)}</strong></div>
      </div>
    </section>
    {_term_guide_html()}
    <section class="toolbar">
      <label for="step-filter"><strong>Step / 时间步</strong></label>
      <input id="step-filter" type="range" min="{min_step}" max="{max_step}" value="{min_step}" step="1" />
      <output id="step-value">{min_step}</output>
      <button type="button" id="show-all">{_lang("Show all", "显示全部")}</button>
    </section>
    <section class="panel">
      <h2>{_lang("Step-by-step event chain", "逐时刻事件链")}</h2>
      <div class="event-grid">{''.join(event_cards)}</div>
    </section>
    <section class="panel">
      <h2>{_lang("Four-part decision summary", "四段式决策摘要")}</h2>
      <div class="story">
        {''.join(_step_story(row) for _, row in summary.head(80).iterrows())}
      </div>
    </section>
    <section class="panel">
      <h2>{_lang("Scope / Bus constraint summary", "Scope / Bus 约束摘要")}</h2>
      {_table(scope, ["step", "time_label", "scope_type", "scope_id", "bus_id", "asset_count", "p_lower_mw", "p_upper_mw", "current_p_mw", "projected_p_mw", "is_binding", "is_remote_bus"], max_rows=500)}
    </section>
    <script>
      (function() {{
        var slider = document.getElementById('step-filter');
        var out = document.getElementById('step-value');
        var all = document.getElementById('show-all');
        function filter(step) {{
          out.textContent = step;
          document.querySelectorAll('[data-step]').forEach(function(card) {{
            card.style.display = String(card.getAttribute('data-step')) === String(step) ? '' : 'none';
          }});
        }}
        slider.addEventListener('input', function() {{ filter(slider.value); }});
        all.addEventListener('click', function() {{
          document.querySelectorAll('[data-step]').forEach(function(card) {{ card.style.display = ''; }});
          out.textContent = 'all';
        }});
        filter(slider.value);
      }})();
    </script>
    """
    return _base_html(f"{vpp_id} First-Person Step Report", f"{vpp_id} 第一视角逐时刻报告", body)


def _step_story(row: pd.Series) -> str:
    raw_target = _float_value(row.get("raw_target_p_mw"))
    projected = _float_value(row.get("projected_p_mw"))
    actual = _float_value(row.get("actual_p_mw"))
    error = _float_value(row.get("tracking_error_mw"))
    risk = _float_value(row.get("delivery_risk"))
    need_score = _float_value(row.get("need_score"))
    command = _text(row.get("command_seen"), "n/a")
    command_zh = VALUE_ZH.get(command, command)
    need = _text(row.get("need_label"), "n/a")
    need_zh = VALUE_ZH.get(need, need)
    action = _text(row.get("action_label"), "n/a")
    action_zh = VALUE_ZH.get(action, action)
    status = _text(row.get("decision_status"), "n/a")
    status_zh = VALUE_ZH.get(status, status)
    risk_en, risk_zh = _risk_band(risk)
    strength_en, strength_zh = _need_strength(need_score)
    return f"""
    <article class="story-card" data-step="{int(row.get('step', 0))}">
      <h3>{escape(_text(row.get('time_label')))} / step {int(row.get('step', 0))}</h3>
      <div class="four-col">
        <div class="mini">
          <h4>{_lang("What it received", "收到什么")}</h4>
          <p>{_lang(
              f"{escape(command)}. Requested target: {raw_target:.3f} MW, {escape(_power_direction_en(raw_target))}.",
              f"{escape(command_zh)}。请求目标：{raw_target:.3f} MW，{escape(_power_direction_zh(raw_target))}。",
              block=True,
          )}</p>
        </div>
        <div class="mini">
          <h4>{_lang("What it inferred", "判断什么")}</h4>
          <p>{_lang(
              f"{escape(need)}; need_score={need_score:.2f} ({strength_en}); delivery_risk={risk:.2f} ({risk_en}).",
              f"{escape(need_zh)}；need_score={need_score:.2f}（{strength_zh}）；delivery_risk={risk:.2f}（{risk_zh}）。",
              block=True,
          )}</p>
        </div>
        <div class="mini">
          <h4>{_lang("What it decided", "做了什么")}</h4>
          <p>{_lang(
              f"{escape(action)}. Executable target after FR/DOE projection: {projected:.3f} MW.",
              f"{escape(action_zh)}。FR/DOE 修正后的可执行目标：{projected:.3f} MW。",
              block=True,
          )}</p>
        </div>
        <div class="mini">
          <h4>{_lang("What happened", "结果如何")}</h4>
          <p>{_lang(
              f"Delivered {actual:.3f} MW with tracking error {error:.3f} MW. Status: {escape(status)}.",
              f"实际交付 {actual:.3f} MW，目标跟踪误差 {error:.3f} MW。状态：{escape(status_zh)}。",
              block=True,
          )}</p>
        </div>
      </div>
    </article>
    """


def _long_cycle_html(frames: dict[str, pd.DataFrame]) -> str:
    long_cycle = frame_or_empty(frames, "vpp_long_cycle_judgment")
    story = frame_or_empty(frames, "portfolio_adjustment_story")
    body = _nav() + _model_update_panel(frames) + f"""
    <section class="panel">
      <h2>{_lang("VPP long-cycle judgment", "VPP 长周期判断")}</h2>
      <p class="muted">{_lang(
          "This page answers what each VPP concluded over a longer horizon and when it adjusted its aggregation configuration.",
          "这里回答每个 VPP 在较长周期内形成了什么判断，以及什么时候调整自身聚合配置。"
      )}</p>
      {_table(long_cycle, ["vpp_id", "window_id", "start_time", "end_time", "dominant_grid_need", "dominant_action", "reliability_score", "non_delivery_rate", "projection_count", "portfolio_change_count", "risk_level", "portfolio_recommendation", "recommendation_reason"], max_rows=200)}
    </section>
    <section class="panel">
      <h2>{_lang("Aggregation configuration story", "聚合配置调整故事")}</h2>
      {_table(story, ["event_id", "effective_step", "time_label", "from_vpp_id", "to_vpp_id", "der_id", "bus_id", "why_now", "expected_effect", "post_change_effect", "physical_bus_unchanged", "physical_element_unchanged"], max_rows=100)}
    </section>
    """
    return _base_html("VPP Long-Cycle Judgment and Portfolio Adjustment", "VPP 长周期判断与聚合配置调整", body)


def _economic_html(frames: dict[str, pd.DataFrame]) -> str:
    economic = frame_or_empty(frames, "economic_explanation")
    body = _nav() + _model_update_panel(frames) + f"""
    <section class="panel">
      <h2>{_lang("Why reward / profit proxy can be negative", "为什么 reward / profit proxy 会是负数")}</h2>
      <p class="muted">{_lang(
          "Current reward is a shaped control objective, not market net profit: reward = -0.05 * total_cost + feasibility_bonus + tracking_bonus. total_cost includes action_projection_penalty, so infeasible raw actions pay a learning cost even if the safety layer repairs them. The raw objective is still recorded as raw_objective_reward = -total_cost. The signed energy cashflow proxy uses P>0 for injection and P<0 for absorption, so charging or absorption can appear negative.",
          "当前 reward 是整形后的控制目标函数，不是市场净利润：reward = -0.05 * total_cost + 可行性奖励 + 跟踪奖励。total_cost 已包含 action_projection_penalty，因此不可行原始动作即使被安全层修复，也会付出学习成本。原始目标仍记录为 raw_objective_reward = -total_cost。signed_energy_cashflow_proxy 使用 P>0 注入、P<0 吸收的符号约定，因此充电或吸收功率会表现为负数。"
      )}</p>
      {_table(economic, ["metric", "value", "share_of_total_cost", "formula", "interpretation", "why_negative"], max_rows=100)}
    </section>
    """
    return _base_html("Economics and Reward Explanation", "经济解释与 reward 分解", body)


def build_first_person_reports(frames: dict[str, pd.DataFrame], output_dir: str | Path) -> dict[str, Path]:
    """Write split first-person HTML reports for a completed simulation."""

    out = ensure_dir(output_dir)
    summary = frame_or_empty(frames, "vpp_step_decision_summary")
    vpp_ids = sorted(summary["vpp_id"].astype(str).unique().tolist()) if not summary.empty else []
    links: dict[str, str] = {vpp_id: f"{_safe_name(vpp_id)}.html" for vpp_id in vpp_ids}
    paths: dict[str, Path] = {}

    index = out / "index.html"
    index.write_text(_index_html(frames, links), encoding="utf-8")
    paths["index"] = index

    for vpp_id, filename in links.items():
        path = out / filename
        path.write_text(_vpp_html(vpp_id, frames), encoding="utf-8")
        paths[vpp_id] = path

    long_cycle = out / "long_cycle.html"
    long_cycle.write_text(_long_cycle_html(frames), encoding="utf-8")
    paths["long_cycle"] = long_cycle

    economic = out / "economic_explanation.html"
    economic.write_text(_economic_html(frames), encoding="utf-8")
    paths["economic_explanation"] = economic
    return paths
