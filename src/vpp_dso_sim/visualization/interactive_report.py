from __future__ import annotations

from html import escape
from pathlib import Path

import pandas as pd

from vpp_dso_sim.learning.rl_architecture import build_rl_architecture_frames
from vpp_dso_sim.visualization.dashboard_data import model_update_summary_frame
from vpp_dso_sim.visualization.rl_architecture_diagram import (
    build_rl_architecture_diagram,
    rl_architecture_diagram_css,
)
from vpp_dso_sim.visualization.neural_network_diagram import (
    build_neural_network_architecture_diagram,
    build_target_ctde_architecture_diagram,
    neural_network_diagram_css,
)
from vpp_dso_sim.visualization.rl_algorithm_variants import (
    build_rl_algorithm_variant_frame,
    build_rl_algorithm_variant_section,
    rl_algorithm_variant_section_css,
)
from vpp_dso_sim.visualization.plotly_figures import (
    DER_ICON_NAMES,
    DER_SHORT_LABELS,
    der_figure,
    edge_flow_figure,
    frame_or_empty,
    profile_figure,
    require_plotly,
    topology_figure,
    truthy_mask,
    voltage_level_tables,
    vpp_figure,
)

_require_plotly = require_plotly
_frame = frame_or_empty
_topology_figure = topology_figure
_vpp_figure = vpp_figure
_der_figure = der_figure
_profile_figure = profile_figure
_edge_flow_figure = edge_flow_figure


def _lang_html(en: str, zh: str, *, tag: str = "span", display: str = "inline") -> str:
    classes = f"lang-copy lang-{display}"
    return (
        f"<{tag} class='{classes} lang-en'>{en}</{tag}>"
        f"<{tag} class='{classes} lang-zh'>{zh}</{tag}>"
    )


def _display_text(value: object, default: str = "n/a") -> str:
    if value is None or pd.isna(value):
        return default
    text = str(value).strip()
    return text if text else default


def _display_number(value: object, suffix: str = "", default: str = "n/a") -> str:
    if value is None or pd.isna(value):
        return default
    try:
        return f"{float(value):.2f}{suffix}"
    except (TypeError, ValueError):
        text = str(value).strip()
        return f"{text}{suffix}" if text else default


TABLE_LABELS_ZH = {
    "vpp_id": "VPP 编号",
    "vpp_name": "VPP 名称",
    "pcc_bus": "PCC 母线",
    "asset_count": "资源数量",
    "der_types": "DER 类型",
    "buses": "接入母线",
    "asset_ids": "资源编号",
    "Nominal kV": "额定 kV",
    "Bus count": "母线数量",
    "Bus IDs": "母线编号",
    "Branch count": "支路数量",
    "Branch IDs": "支路编号",
    "Short label": "简称",
    "Device type": "设备类型",
    "Icon": "图标",
    "step": "时步",
    "kind": "类型",
    "time_label": "时间",
    "severity": "严重程度",
    "element_type": "元件类型",
    "element_id": "元件编号",
    "value": "数值",
    "limit": "限值",
    "message": "说明",
    "magnitude": "幅值",
    "der_id": "DER 编号",
    "name": "名称",
    "bus_id": "母线编号",
    "der_type": "DER 类型",
    "pp_element_type": "pandapower 元件",
    "pp_element_index": "pandapower 索引",
    "algorithm_id": "算法编号",
    "algorithm_family": "算法族",
    "training_mode": "训练模式",
    "execution_mode": "执行模式",
    "ctde_status": "CTDE 状态",
    "critic_scope": "Critic 范围",
    "actor_scope": "Actor 范围",
    "reward_formula": "奖励公式",
    "loss_formula": "损失函数",
    "trainable_components": "可训练组件",
    "non_trainable_components": "非训练组件",
    "portfolio_trainable": "组合智能体可训练",
    "optimizer_steps": "优化步数",
    "param_delta_l2": "参数变化 L2",
    "status": "状态",
    "plain_language_summary": "通俗说明",
    "agent_id": "智能体编号",
    "role_type": "角色类型",
    "owner_id": "所属主体",
    "time_scale": "时间尺度",
    "trainable": "可训练",
    "input_observation": "输入观测",
    "action_output": "动作/输出",
    "policy_module": "策略模块",
    "implementation_status": "实现状态",
    "objective": "目标",
    "privacy_scope": "隐私范围",
    "flow_order": "流程顺序",
    "source": "来源",
    "target": "目标",
    "signal": "信号",
    "description": "说明",
    "component": "组成项",
    "formula": "公式",
    "coefficient": "系数",
    "meaning": "含义",
    "question": "问题",
    "answer": "答案",
    "evidence": "依据",
    "update_area": "更新范围",
    "current_value": "当前值",
    "current_value_zh": "当前值中文说明",
    "explanation": "说明",
    "explanation_zh": "中文说明",
    "evidence_file": "证据文件",
}

TABLE_VALUE_ZH = {
    "Residential VPP": "居民 VPP",
    "EVCS MT VPP": "充电站-微型燃机 VPP",
    "Mixed DER VPP": "混合 DER VPP",
    "PV": "光伏",
    "ESS": "储能",
    "Flex": "柔性负荷",
    "MT": "微型燃机",
    "EVCS": "充电站",
    "HVAC": "HVAC",
    "solar panel": "光伏组件",
    "battery storage": "储能电池",
    "charging station": "充电站",
    "controllable load": "可控负荷",
    "fan coil": "HVAC 风机盘管",
    "microturbine": "微型燃机",
    "info": "提示",
    "warning": "警告",
    "critical": "严重",
    "bus": "母线",
    "line": "线路",
    "trafo": "变压器",
    "algorithm": "算法",
    "privacy_and_ctde": "隐私边界与CTDE",
    "actor_stack": "智能体策略栈",
    "critic_and_loss": "集中critic与损失函数",
    "reward_and_checkpoint": "奖励与checkpoint",
    "ui_refresh_contract": "可视化同步约定",
    "target_ctde_status": "目标CTDE状态",
}


def _column_label(column: object) -> tuple[str, str]:
    en = str(column)
    return en, TABLE_LABELS_ZH.get(en, en.replace("_", " "))


def _translate_table_value(value: object) -> tuple[str, str]:
    en = "" if value is None or pd.isna(value) else str(value)
    if not en:
        return "", ""
    if en in TABLE_VALUE_ZH:
        return en, TABLE_VALUE_ZH[en]
    if "," in en:
        parts = [part.strip() for part in en.split(",")]
        translated = [TABLE_VALUE_ZH.get(part, part) for part in parts]
        if translated != parts:
            return en, ", ".join(translated)
    return en, en


def _html_table(
    frame: pd.DataFrame,
    *,
    columns: list[str] | None = None,
    classes: str = "summary-table",
    max_rows: int = 200,
) -> str:
    data = frame.copy()
    if columns is not None:
        for column in columns:
            if column not in data:
                data[column] = ""
        data = data[columns]
    data = data.head(max_rows)
    headers = "".join(
        f"<th>{_lang_html(escape(en), escape(zh))}</th>"
        for en, zh in (_column_label(column) for column in data.columns)
    )
    rows: list[str] = []
    for row in data.itertuples(index=False, name=None):
        cells: list[str] = []
        for value in row:
            en, zh = _translate_table_value(value)
            if en == zh:
                cells.append(f"<td>{escape(en)}</td>")
            else:
                cells.append(f"<td>{_lang_html(escape(en), escape(zh))}</td>")
        rows.append(f"<tr>{''.join(cells)}</tr>")
    return f"""
    <table class="{classes}">
      <thead><tr>{headers}</tr></thead>
      <tbody>{''.join(rows)}</tbody>
    </table>
    """


def _dispatch_copy(row: pd.Series, zh_column: str, en_column: str) -> tuple[str, str]:
    zh_text = _display_text(row.get(zh_column, ""), default="")
    en_text = _display_text(row.get(en_column, ""), default=zh_text) if en_column in row else zh_text
    if not zh_text and not en_text:
        fallback = "No explanation was generated."
        return fallback, "未生成说明。"
    return en_text or zh_text, zh_text or en_text


def _language_toolbar_html() -> str:
    return f"""
    <div class="language-toolbar" role="group" aria-label="Language switch">
      <span class="language-label">{_lang_html("Language", "语言")}</span>
      <button type="button" class="lang-button is-active" data-lang-switch="en" aria-pressed="true">EN</button>
      <button type="button" class="lang-button" data-lang-switch="zh" aria-pressed="false">中文</button>
    </div>
    """


def _language_toggle_script() -> str:
    return """
    (function() {
      var storageKey = 'vpp-dso-sim-report-lang';
      function applyPlotlyLanguage(lang) {
        if (!window.Plotly) {
          return;
        }
        document.querySelectorAll('.js-plotly-plot').forEach(function(gd) {
          var meta = gd.layout && gd.layout.meta ? gd.layout.meta : null;
          var config = meta && meta.i18n ? meta.i18n : null;
          if (!config) {
            return;
          }
          var layoutUpdate = config.layout && config.layout[lang] ? config.layout[lang] : {};
          if (Object.keys(layoutUpdate).length) {
            Plotly.relayout(gd, layoutUpdate);
          }
          var names = config.trace_names && config.trace_names[lang] ? config.trace_names[lang] : [];
          names.forEach(function(name, index) {
            if (gd.data && gd.data[index]) {
              Plotly.restyle(gd, {name: [String(name)]}, [index]);
              if (gd._transitionData && gd._transitionData._frames) {
                gd._transitionData._frames.forEach(function(frame) {
                  if (frame.data && frame.data[index]) {
                    frame.data[index].name = String(name);
                  }
                });
              }
            }
          });
          var colorbars = config.trace_colorbars && config.trace_colorbars[lang] ? config.trace_colorbars[lang] : [];
          colorbars.forEach(function(item) {
            if (!gd.data || !gd.data[item.index]) {
              return;
            }
            var update = {};
            update[item.path] = [String(item.text)];
            Plotly.restyle(gd, update, [item.index]);
          });
          var hovertemplates = config.trace_hovertemplates && config.trace_hovertemplates[lang] ? config.trace_hovertemplates[lang] : [];
          hovertemplates.forEach(function(item) {
            if (!gd.data || !gd.data[item.index]) {
              return;
            }
            Plotly.restyle(gd, {hovertemplate: [String(item.text)]}, [item.index]);
            if (gd._transitionData && gd._transitionData._frames) {
              gd._transitionData._frames.forEach(function(frame) {
                if (frame.data && frame.data[item.index]) {
                  frame.data[item.index].hovertemplate = String(item.text);
                }
              });
            }
          });
          var hovertexts = config.trace_hovertexts && config.trace_hovertexts[lang] ? config.trace_hovertexts[lang] : [];
          hovertexts.forEach(function(item) {
            if (!gd.data || !gd.data[item.index]) {
              return;
            }
            Plotly.restyle(gd, {hovertext: [item.text]}, [item.index]);
            if (gd._transitionData && gd._transitionData._frames) {
              gd._transitionData._frames.forEach(function(frame) {
                if (frame.data && frame.data[item.index]) {
                  frame.data[item.index].hovertext = item.text;
                }
              });
            }
          });
        });
      }
      function applyLang(lang) {
        var root = document.documentElement;
        root.setAttribute('data-lang', lang);
        root.setAttribute('lang', lang === 'zh' ? 'zh-CN' : 'en');
        document.querySelectorAll('[data-lang-switch]').forEach(function(button) {
          var active = button.getAttribute('data-lang-switch') === lang;
          button.classList.toggle('is-active', active);
          button.setAttribute('aria-pressed', active ? 'true' : 'false');
        });
        try {
          window.localStorage.setItem(storageKey, lang);
        } catch (err) {
          void err;
        }
        applyPlotlyLanguage(lang);
      }
      var initialLang = 'en';
      try {
        initialLang = window.localStorage.getItem(storageKey) || initialLang;
      } catch (err) {
        void err;
      }
      document.addEventListener('click', function(event) {
        var button = event.target.closest('[data-lang-switch]');
        if (!button) {
          return;
        }
        applyLang(button.getAttribute('data-lang-switch') || 'en');
      });
      applyLang(initialLang);
      window.addEventListener('plotly_afterplot', function() {
        applyPlotlyLanguage(document.documentElement.getAttribute('data-lang') || initialLang);
      });
    })();
    """


def _first_person_filter_script() -> str:
    return """
    (function() {
      var select = document.getElementById('first-person-vpp-filter');
      if (!select) {
        return;
      }
      function applyFilter() {
        var value = select.value || 'all';
        document.querySelectorAll('[data-first-person-vpp]').forEach(function(card) {
          var visible = value === 'all' || card.getAttribute('data-first-person-vpp') === value;
          card.style.display = visible ? '' : 'none';
        });
      }
      select.addEventListener('change', applyFilter);
      applyFilter();
    })();
    """


def _asset_icon_toggle_script() -> str:
    return """
var gd = document.getElementById('{plot_id}');
if (gd) {
  gd.on('plotly_legendclick', function(evt) {
    var trace = gd.data[evt.curveNumber];
    if (!trace || !trace.meta || trace.meta.asset_image_index === undefined) {
      return;
    }
    var idx = trace.meta.asset_image_index;
    if (!gd.layout || !gd.layout.images || !gd.layout.images[idx]) {
      return;
    }
    var currentlyVisible = trace.visible !== 'legendonly' && trace.visible !== false;
    var nextTraceVisible = currentlyVisible ? 'legendonly' : true;
    var update = {};
    update['images[' + idx + '].visible'] = !currentlyVisible;
    Plotly.restyle(gd, {visible: [nextTraceVisible]}, [evt.curveNumber]);
    Plotly.relayout(gd, update);
    return false;
  });
}
"""


def _summary_html(frames: dict[str, pd.DataFrame]) -> str:
    summary = _frame(frames, "step_summary")
    alerts = _frame(frames, "alert_event")
    nodes = _frame(frames, "network_nodes")
    edges = _frame(frames, "network_edges")
    assets = _frame(frames, "asset_registry")
    if summary.empty:
        min_vm = max_loading = total_reward = "n/a"
        steps = 0
    else:
        steps = int(len(summary))
        min_vm = f"{float(summary['min_vm_pu'].min()):.4f}" if "min_vm_pu" in summary else "n/a"
        max_loading = (
            f"{float(summary['max_line_loading_percent'].max()):.2f}%"
            if "max_line_loading_percent" in summary
            else "n/a"
        )
        total_reward = f"{float(summary['reward'].sum()):.2f}" if "reward" in summary else "n/a"
    alert_count = 0 if alerts.empty else int(len(alerts))
    return f"""
    <section class="kpi-grid">
      <div class="kpi-card"><span>{_lang_html("Steps", "步数")}</span><strong>{steps}</strong></div>
      <div class="kpi-card"><span>{_lang_html("Buses", "母线")}</span><strong>{len(nodes)}</strong></div>
      <div class="kpi-card"><span>{_lang_html("Edges", "支路")}</span><strong>{len(edges)}</strong></div>
      <div class="kpi-card"><span>{_lang_html("DER assets", "DER 资源")}</span><strong>{len(assets)}</strong></div>
      <div class="kpi-card"><span>{_lang_html("Min voltage", "最低电压")}</span><strong>{min_vm}</strong></div>
      <div class="kpi-card"><span>{_lang_html("Max line loading", "最大线路负载率")}</span><strong>{max_loading}</strong></div>
      <div class="kpi-card"><span>{_lang_html("Alerts", "告警")}</span><strong>{alert_count}</strong></div>
      <div class="kpi-card"><span>{_lang_html("Total reward", "总奖励")}</span><strong>{total_reward}</strong></div>
    </section>
    """


def _alerts_html(alerts: pd.DataFrame) -> str:
    if alerts.empty:
        return f"<p class='empty'>{_lang_html('No constraint violations were recorded.', '未记录到约束越限事件。')}</p>"
    return _html_table(alerts, classes="alert-table", max_rows=200)


def _dispatch_explanation_html(frames: dict[str, pd.DataFrame]) -> str:
    dispatch = _frame(frames, "vpp_dispatch_explanation")
    if dispatch.empty:
        return f"<p class='empty'>{_lang_html('No VPP dispatch explanation rows were generated.', '未生成 VPP 调度说明记录。')}</p>"
    dispatch = dispatch.copy()
    columns = [
        "vpp_id",
        "vpp_name",
        "start_time",
        "end_time",
        "command_type",
        "command_type_zh",
        "command_type_en",
        "avg_price",
        "avg_p_mw",
        "p_range_mw",
        "reason",
        "instruction",
        "asset_response",
        "reason_zh",
        "instruction_zh",
        "asset_response_zh",
        "reason_en",
        "instruction_en",
        "asset_response_en",
    ]
    for column in columns:
        if column not in dispatch:
            dispatch[column] = ""

    groups: list[str] = []
    for vpp_id, group in dispatch.groupby("vpp_id", sort=False, dropna=False):
        group_cards: list[str] = []
        for _, row in group.iterrows():
            command_en, command_zh = _dispatch_copy(row, "command_type_zh", "command_type_en")
            reason_en, reason_zh = _dispatch_copy(row, "reason_zh", "reason_en")
            instruction_en, instruction_zh = _dispatch_copy(row, "instruction_zh", "instruction_en")
            response_en, response_zh = _dispatch_copy(row, "asset_response_zh", "asset_response_en")
            group_cards.append(
                f"""
                <article class="dispatch-window-card">
                  <div class="dispatch-window-head">
                    <div class="dispatch-window-meta">
                      <div class="dispatch-metric">
                        <span class="dispatch-metric-label">{_lang_html("Time Window", "时间窗口")}</span>
                        <strong>{escape(_display_text(row.get("start_time")))} - {escape(_display_text(row.get("end_time")))}</strong>
                      </div>
                      <div class="dispatch-metric">
                        <span class="dispatch-metric-label">{_lang_html("Command", "调度类型")}</span>
                        <strong>{_lang_html(escape(command_en), escape(command_zh))}</strong>
                      </div>
                      <div class="dispatch-metric">
                        <span class="dispatch-metric-label">{_lang_html("Avg Price", "平均电价")}</span>
                        <strong>{escape(_display_number(row.get("avg_price")))}</strong>
                      </div>
                      <div class="dispatch-metric">
                        <span class="dispatch-metric-label">{_lang_html("Avg Power", "平均功率")}</span>
                        <strong>{escape(_display_number(row.get("avg_p_mw"), " MW"))}</strong>
                      </div>
                      <div class="dispatch-metric">
                        <span class="dispatch-metric-label">{_lang_html("Power Range", "功率区间")}</span>
                        <strong>{escape(_display_text(row.get("p_range_mw")))}</strong>
                      </div>
                    </div>
                  </div>
                  <div class="dispatch-window-body">
                    <section class="dispatch-copy-card">
                      <h3>{_lang_html("Reason", "触发原因")}</h3>
                      {_lang_html(escape(reason_en), escape(reason_zh), tag="div", display="block")}
                    </section>
                    <section class="dispatch-copy-card">
                      <h3>{_lang_html("Instruction", "调度指令")}</h3>
                      {_lang_html(escape(instruction_en), escape(instruction_zh), tag="div", display="block")}
                    </section>
                    <section class="dispatch-copy-card">
                      <h3>{_lang_html("Asset Response", "资源响应")}</h3>
                      {_lang_html(escape(response_en), escape(response_zh), tag="div", display="block")}
                    </section>
                  </div>
                </article>
                """
            )

        raw_vpp_name = _display_text(group["vpp_name"].iloc[0], default="") if "vpp_name" in group else ""
        vpp_name = raw_vpp_name if raw_vpp_name != "n/a" else ""
        title = escape(str(vpp_id)) if vpp_name == "" else f"{escape(str(vpp_id))} · {escape(vpp_name)}"
        groups.append(
            f"""
            <section class="dispatch-vpp-card">
              <div class="dispatch-vpp-head">
                <div>
                  <p class="eyebrow">{_lang_html("VPP Group", "VPP 分组")}</p>
                  <h3 class="dispatch-vpp-title">{title}</h3>
                </div>
                <span class="dispatch-count-pill">{_lang_html(f"{len(group)} windows", f"{len(group)} 个时间段")}</span>
              </div>
              <div class="dispatch-vpp-list">
                {''.join(group_cards)}
              </div>
            </section>
            """
        )
    return "".join(groups)


def _vpp_summary_html(frames: dict[str, pd.DataFrame]) -> str:
    assets = _frame(frames, "asset_registry")
    nodes = _frame(frames, "network_nodes")
    if assets.empty:
        return f"<p class='empty'>{_lang_html('No VPP assets were registered.', '未注册任何 VPP 资产。')}</p>"
    pcc_rows = []
    if not nodes.empty and {"is_pcc", "bus_id", "vpp_ids"}.issubset(nodes.columns):
        for _, row in nodes[truthy_mask(nodes["is_pcc"])].iterrows():
            for vpp_id in [item for item in str(row["vpp_ids"]).split(",") if item]:
                pcc_rows.append({"vpp_id": vpp_id, "pcc_bus": int(row["bus_id"])})
    pcc = pd.DataFrame(pcc_rows)
    grouped = (
        assets.groupby(["vpp_id", "vpp_name"], dropna=False)
        .agg(
            asset_count=("der_id", "count"),
            buses=("bus_id", lambda values: ", ".join(str(int(v)) for v in sorted(set(values)))),
            asset_ids=("der_id", lambda values: ", ".join(str(v) for v in values)),
            der_types=(
                "der_type",
                lambda values: ", ".join(sorted({DER_SHORT_LABELS.get(str(v), str(v).replace("Model", "")) for v in values})),
            ),
        )
        .reset_index()
    )
    if not pcc.empty:
        grouped = grouped.merge(pcc, on="vpp_id", how="left")
    else:
        grouped["pcc_bus"] = ""
    grouped = grouped[["vpp_id", "vpp_name", "pcc_bus", "asset_count", "der_types", "buses", "asset_ids"]]
    return _html_table(grouped, classes="summary-table")


def _voltage_levels_html(frames: dict[str, pd.DataFrame]) -> str:
    tables = voltage_level_tables(_frame(frames, "network_nodes"), _frame(frames, "network_edges"))
    bus_levels = tables["bus_levels"]
    feeder_levels = tables["feeder_levels"]
    bus_table = (
        _html_table(bus_levels, classes="summary-table compact")
        if not bus_levels.empty
        else f"<p class='empty'>{_lang_html('No nominal bus-voltage data is available.', '没有可用的母线额定电压数据。')}</p>"
    )
    feeder_table = (
        _html_table(feeder_levels, classes="summary-table compact")
        if not feeder_levels.empty
        else f"<p class='empty'>{_lang_html('No feeder voltage-level data is available.', '没有可用的馈线电压等级数据。')}</p>"
    )
    return f"""
    <div class="legend-shell">
      <div>
        <h3>{_lang_html('Bus nominal voltage levels', '母线额定电压等级')}</h3>
        <p class="hint">{_lang_html('These levels come directly from <code>network_nodes.vn_kv</code>.', '这些等级直接来自 <code>network_nodes.vn_kv</code>。')}</p>
        {bus_table}
      </div>
      <div>
        <h3>{_lang_html('Feeder branch voltage levels', '馈线支路电压等级')}</h3>
        <p class="hint">{_lang_html('Lines stay on one nominal kV band; transformers show an HV/LV transition.', '线路保持在单一额定电压带；变压器会显示高低压侧过渡。')}</p>
        {feeder_table}
      </div>
    </div>
    """


def _rl_architecture_filter_script() -> str:
    return """
    (function() {
      document.addEventListener('click', function(event) {
        var button = event.target.closest('[data-rl-filter]');
        if (!button) {
          return;
        }
        var panel = button.closest('#rl-architecture');
        if (!panel) {
          return;
        }
        var selected = button.getAttribute('data-rl-filter') || 'all';
        panel.querySelectorAll('[data-rl-filter]').forEach(function(item) {
          item.classList.toggle('is-active', item === button);
        });
        panel.querySelectorAll('[data-rl-card]').forEach(function(card) {
          var kind = card.getAttribute('data-rl-kind') || '';
          card.hidden = selected !== 'all' && kind !== selected;
        });
      });
    })();
    """


def _header_html(frames: dict[str, pd.DataFrame]) -> str:
    summary = _frame(frames, "step_summary")
    alerts = _frame(frames, "alert_event")
    assets = _frame(frames, "asset_registry")
    steps = len(summary)
    alert_count = len(alerts)
    asset_count = len(assets)
    return f"""
    <header class="hero">
      <div class="hero-copy">
        <div class="hero-topline">
          <p class="eyebrow">{_lang_html("Feeder Simulation Analysis Console", "馈线仿真分析控制台")}</p>
          {_language_toolbar_html()}
        </div>
        <h1>{_lang_html("pandapower VPP DSO Simulation Report", "pandapower VPP-DSO 仿真报告")}</h1>
        <p class="hero-lead">
          {_lang_html(
              "Offline replay workspace for feeder topology, VPP aggregation, DER dispatch and constraint events derived from standardized dashboard frames.",
              "基于标准化 dashboard 数据帧构建的离线回放工作台，用于查看馈线拓扑、VPP 聚合、DER 调度与约束事件。"
          )}
        </p>
        <div class="hero-pill-row">
          <span class="hero-pill">{_lang_html("Read-only", "只读")}</span>
          <span class="hero-pill">{_lang_html("Offline replay", "离线回放")}</span>
          <span class="hero-pill">{_lang_html(f"{steps} intervals", f"{steps} 个时段")}</span>
          <span class="hero-pill">{_lang_html(f"{asset_count} DER assets", f"{asset_count} 个 DER 资源")}</span>
          <span class="hero-pill">{_lang_html(f"{alert_count} alerts", f"{alert_count} 条告警")}</span>
        </div>
      </div>
      <div class="hero-brief">
        <div class="brief-label">{_lang_html("Operator Brief", "操作员提示")}</div>
        <p>{_lang_html("Use the topology replay for spatial context, the line-flow heatmap for horizon-wide loading, and the VPP / DER plots for dispatch interpretation.", "先用拓扑回放理解空间位置，再用线路潮流热图观察全时域负载，最后结合 VPP / DER 图理解调度结果。")}</p>
      </div>
    </header>
    """


def _reading_guide_html() -> str:
    return (
        """
    <section class="panel intro-panel">
      <div class="section-heading">
        <div>
          <p class="eyebrow">"""
        + _lang_html("Reading Guide", "阅读指南")
        + """</p>
          <h2>"""
        + _lang_html("How to read this analysis page", "如何阅读本分析页面")
        + """</h2>
        </div>
        <p class="section-note">"""
        + _lang_html(
            "The report keeps topology semantics, power-flow labels and asset identity visible without forcing every number into the same plot.",
            "本报告保留拓扑语义、潮流标签与资产身份信息，而不是把所有数字都塞进同一张图里。",
        )
        + """</p>
      </div>
      <div class="guide-grid">
        <article class="guide-card">
          <h3>"""
        + _lang_html("Single-line symbols", "单线图符号")
        + """</h3>
          <p>"""
        + _lang_html(
            "Bus markers represent electrical nodes and include nominal voltage labels. Branch segments and transformer links follow the feeder one-line topology rather than a GIS map.",
            "母线标记表示电气节点，并带有额定电压标签。支路和变压器连接遵循馈线单线图拓扑，而不是 GIS 地图。",
        )
        + """</p>
        </article>
        <article class="guide-card">
          <h3>"""
        + _lang_html("VPP ownership colors", "VPP 归属颜色")
        + """</h3>
          <p>"""
        + _lang_html(
            "Each VPP keeps a consistent accent color across PCC rings, legend items and DER icon borders so asset affiliation is readable at a glance.",
            "每个 VPP 在 PCC 圈、图例项和 DER 图标边框上保持一致的强调色，便于快速识别资产归属。",
        )
        + """</p>
        </article>
        <article class="guide-card">
          <h3>"""
        + _lang_html("DER realistic icons", "DER 真实图标")
        + """</h3>
          <p>"""
        + _lang_html(
            "PV, ESS, EVCS, HVAC, microturbine and flexible-load assets use local SVG pictograms to distinguish equipment classes without external icon dependencies.",
            "PV、ESS、EVCS、HVAC、微型燃机和柔性负荷资源使用本地 SVG 图标区分设备类型，不依赖外部图标库。",
        )
        + """</p>
        </article>
        <article class="guide-card">
          <h3>"""
        + _lang_html("Flow labels", "潮流标签")
        + """</h3>
          <p>"""
        + _lang_html(
            "<code>Flow labels</code> show branch active power in MW; feeder kV labels show nominal branch voltage. Hide either legend item when local feeder density is more important than inline annotation.",
            "<code>Flow labels</code> 显示支路有功功率 MW；馈线 kV 标签显示支路额定电压。当局部馈线过密时，可隐藏相应图例项以减少内嵌标注。",
        )
        + """</p>
        </article>
      </div>
    </section>
    """
    )


def _topology_legend_html() -> str:
    return f"""
    <div class="legend-shell">
      <div class="legend-help">
        <div class="legend-callout">
          <h3>{_lang_html('Topology decoding', '拓扑解读')}</h3>
          <p>{_lang_html('Bus marker color encodes per-unit voltage, while bus text shows bus ID and nominal kV. Branch color and width encode loading, and branch labels show nominal feeder voltage level.', '母线标记颜色表示标幺电压，母线文字显示母线编号和额定 kV。支路颜色和宽度表示负载率，支路标签显示馈线额定电压等级。')}</p>
        </div>
        <div class="legend-callout">
          <h3>{_lang_html('Flow interpretation', '潮流解读')}</h3>
          <p>{_lang_html('Hover any branch for <code>MW / MVAr / loading / kV</code>. The overlayed flow label shows active power; reactive power and loading detail stay in hover and the horizon matrix.', '悬停任意支路可查看 <code>MW / MVAr / loading / kV</code>。覆盖在图上的潮流标签显示有功功率；无功和负载率细节保留在悬停信息与全时域矩阵中。')}</p>
        </div>
      </div>
      {_html_table(pd.DataFrame(
          [
              {
                  "Short label": DER_SHORT_LABELS.get(der_type, der_type.replace("Model", "")),
                  "Device type": der_type.replace("Model", ""),
                  "Icon": icon_name,
              }
              for der_type, icon_name in DER_ICON_NAMES.items()
          ]
      ), classes="summary-table compact")}
    </div>
    """


def _decision_drivers_html(frames: dict[str, pd.DataFrame]) -> str:
    profile = _frame(frames, "profile_state")
    dispatch = _frame(frames, "vpp_dispatch_explanation")
    if profile.empty:
        price_min = price_max = "n/a"
        high_steps = low_steps = "n/a"
    else:
        price_min = _display_number(profile["price"].min()) if "price" in profile else "n/a"
        price_max = _display_number(profile["price"].max()) if "price" in profile else "n/a"
        high_steps = str(int((profile["price"] >= 100.0).sum())) if "price" in profile else "n/a"
        low_steps = str(int((profile["price"] <= 55.0).sum())) if "price" in profile else "n/a"
    segment_count = "0" if dispatch.empty else str(len(dispatch))
    return f"""
    <section class="panel intro-panel">
      <div class="section-heading">
        <div>
          <p class="eyebrow">{_lang_html("Control Logic", "控制逻辑")}</p>
          <h2>{_lang_html("What Drives VPP Decisions", "VPP 决策由什么驱动")}</h2>
        </div>
        <p class="section-note">{_lang_html(
            "The simulator now supports two paths: a transparent envelope baseline when no RL action is supplied, and a learned VPP dispatch path where the policy outputs selected_p_mw plus DER-level actions. The safety layer still projects actions into FR/DOE and device limits before pandapower writes.",
            "仿真器现在支持两条路径：没有 RL 动作时使用透明的包络基线；接入训练策略时，VPP dispatch policy 输出 selected_p_mw 与 DER 级动作。安全层仍会在写入 pandapower 前把动作投影到 FR/DOE 和设备边界内。",
        )}</p>
      </div>
      <div class="guide-grid">
        <article class="guide-card">
          <h3>{_lang_html("1. DSO envelope / baseline target", "1. DSO 包络 / 基线目标")}</h3>
          <p>{_lang_html(
              "The DSO builds an operating envelope from VPP bids, local flexibility bounds and grid stress. The preferred target is a guidance point inside that envelope, not a command to expose private DER details.",
              "DSO 根据 VPP 报量/报价、本地灵活性边界和电网压力生成运行包络。推荐目标是包络内的引导点，不要求 VPP 暴露私有 DER 细节。",
          )}</p>
        </article>
        <article class="guide-card">
          <h3>{_lang_html("2. Flexibility bounds", "2. 灵活性边界")}</h3>
          <p>{_lang_html(
              "Each target is projected into the VPP feasible range [P_min, P_max], which is aggregated from PV availability, storage SOC, EV/HVAC/flexible-load limits and microturbine bounds.",
              "每个目标都会投影到 VPP 可行区间 [P_min, P_max] 内，该区间由 PV 可用功率、储能 SOC、EV/HVAC/柔性负荷限制和微型燃机边界聚合得到。",
          )}</p>
        </article>
        <article class="guide-card">
          <h3>{_lang_html("3. Learned DER-level disaggregation", "3. 学习型 DER 级解聚合")}</h3>
          <p>{_lang_html(
              "When RL actions are supplied, the VPP dispatch policy proposes one normalized action per DER. The projection layer clips each proposal to physical bounds and repairs the aggregate residual. If no RL action is supplied, a deterministic cost-order fallback remains available for reproducible baseline runs.",
              "接入 RL 动作时，VPP dispatch policy 会为每个 DER 提出一个归一化动作；投影层把每个动作裁剪到物理边界，并修复聚合残差。没有 RL 动作时，确定性的成本顺序 fallback 仍可用于可复现基线实验。",
          )}</p>
        </article>
        <article class="guide-card">
          <h3>{_lang_html("Actual horizon summary", "当前仿真概览")}</h3>
          <p>{_lang_html(
              f"Price range: {price_min}-{price_max}; low-price steps: {low_steps}; high-price steps: {high_steps}; one-day instruction segments: {segment_count}.",
              f"电价范围：{price_min}-{price_max}；低价时步：{low_steps}；高价时步：{high_steps}；单日调度片段：{segment_count}。",
          )}</p>
        </article>
      </div>
    </section>
    """


def _first_person_html(frames: dict[str, pd.DataFrame]) -> str:
    timeline = _frame(frames, "vpp_first_person_timeline")
    scope = _frame(frames, "vpp_first_person_scope_detail")
    changes = _frame(frames, "portfolio_change_log")
    if timeline.empty:
        return f"""
        <section class="panel">
          <div class="section-heading">
            <div>
              <p class="eyebrow">{_lang_html("VPP First View", "VPP 第一视角")}</p>
              <h2>{_lang_html("VPP First-Person Replay", "VPP 第一视角回放")}</h2>
            </div>
          </div>
          <p class="empty">{_lang_html("No first-person timeline was generated.", "未生成 VPP 第一视角时间线。")}</p>
        </section>
        """

    timeline = timeline.copy()
    vpp_ids = sorted(timeline["vpp_id"].astype(str).unique().tolist())
    options = "".join(f"<option value='{escape(vpp_id)}'>{escape(vpp_id)}</option>" for vpp_id in vpp_ids)
    cards: list[str] = []
    for _, row in timeline.head(120).iterrows():
        vpp_id = _display_text(row.get("vpp_id"))
        phase = _display_text(row.get("phase"))
        cards.append(
            f"""
            <article class="first-person-card" data-first-person-vpp="{escape(vpp_id)}" data-first-person-phase="{escape(phase)}">
              <div class="first-person-head">
                <div>
                  <p class="eyebrow">{escape(vpp_id)} / {escape(phase)}</p>
                  <h3>{escape(_display_text(row.get("window_id")))}</h3>
                </div>
                <div class="badge-row">
                  <span class="mode-badge">{escape(_display_text(row.get("physical_mode")))}</span>
                  <span class="mode-badge">{escape(_display_text(row.get("portfolio_version")))}</span>
                </div>
              </div>
              <div class="first-person-grid">
                <section>
                  <h4>{_lang_html("Saw", "看到了什么")}</h4>
                  <p>{escape(_display_text(row.get("seen_direction")))} | {escape(_display_text(row.get("seen_target_constraint")))}</p>
                  <code>{escape(_display_text(row.get("seen_price_profile")))}</code>
                </section>
                <section>
                  <h4>{_lang_html("Inferred", "推断估计")}</h4>
                  <p>{escape(_display_text(row.get("inferred_grid_need_label")))} / score={escape(_display_number(row.get("inferred_grid_need_score")))}</p>
                  <p>{_lang_html("Binding scope", "约束范围")}: {escape(_display_text(row.get("inferred_binding_scope_type")))} {escape(_display_text(row.get("inferred_binding_scope_id")))}</p>
                </section>
                <section>
                  <h4>{_lang_html("Decided", "做了什么决策")}</h4>
                  <p>{escape(_display_text(row.get("decision_summary")))}</p>
                  <p>{_lang_html("Status", "状态")}: {escape(_display_text(row.get("decision_status")))}</p>
                </section>
              </div>
              <details>
                <summary>{_lang_html("Raw first-person payload", "原始第一视角数据")}</summary>
                <div class="code-grid">
                  <code>{escape(_display_text(row.get("seen_fr_bounds_json")))}</code>
                  <code>{escape(_display_text(row.get("dispatch_instruction_json")))}</code>
                  <code>{escape(_display_text(row.get("der_dispatch_summary_json")))}</code>
                </div>
              </details>
            </article>
            """
        )

    scope_table = _html_table(
        scope,
        columns=[
            "vpp_id",
            "step",
            "scope_type",
            "scope_id",
            "bus_id",
            "asset_ids",
            "variable",
            "p_lower_mw",
            "p_upper_mw",
            "current_p_mw",
            "is_binding",
        ],
        max_rows=120,
    )
    change_table = _html_table(
        changes,
        columns=[
            "event_id",
            "effective_step",
            "der_id",
            "from_vpp_id",
            "to_vpp_id",
            "bus_id",
            "zone_id",
            "physical_bus_unchanged",
        ],
        max_rows=40,
    )
    return f"""
    <section class="panel first-person-panel">
      <div class="section-heading">
        <div>
          <p class="eyebrow">{_lang_html("VPP First View", "VPP 第一视角")}</p>
          <h2>{_lang_html("VPP First-Person Replay", "VPP 第一视角回放")}</h2>
        </div>
        <p class="section-note">{_lang_html(
            "The replay is written from each VPP viewpoint: what it saw before dispatch, what grid need it inferred, and which dispatch or slow portfolio decision followed.",
            "该回放从每个 VPP 的第一视角书写：调度前看到了什么、推断了什么电网友好需求，以及随后执行了什么调度或慢步骤聚合配置动作。",
        )}</p>
      </div>
      <div class="first-person-toolbar">
        <label for="first-person-vpp-filter">{_lang_html("Focus VPP", "聚焦 VPP")}</label>
        <select id="first-person-vpp-filter">
          <option value="all">All VPPs / 全部 VPP</option>
          {options}
        </select>
        <a href="vpp_first_person/index.html">{_lang_html("Open split first-person report", "打开拆分版第一视角报告")}</a>
      </div>
      <div class="first-person-list">
        {''.join(cards)}
      </div>
      <h3>{_lang_html("Scope Detail", "范围明细")}</h3>
      {scope_table}
      <h3>{_lang_html("Slow Portfolio Changes", "慢步骤聚合配置变化")}</h3>
      {change_table}
    </section>
    """


def _fr_doe_html(frames: dict[str, pd.DataFrame]) -> str:
    envelope = _frame(frames, "fr_envelope_state")
    projection = _frame(frames, "projection_trace")
    privacy = _frame(frames, "privacy_visibility")
    envelope_table = _html_table(
        envelope,
        columns=[
            "step",
            "vpp_id",
            "physical_mode",
            "scope_type",
            "scope_id",
            "variable",
            "lower_bound",
            "upper_bound",
            "current_value",
            "projected_value",
            "is_binding",
        ],
        max_rows=80,
    )
    projection_table = _html_table(
        projection,
        columns=[
            "step",
            "vpp_id",
            "stage_order",
            "stage_name",
            "scope_type",
            "scope_id",
            "p_mw",
            "p_lower_mw",
            "p_upper_mw",
            "was_projected",
            "active_constraint",
            "projection_reason",
        ],
        max_rows=120,
    )
    privacy_table = _html_table(
        privacy,
        columns=[
            "schema",
            "field",
            "visible_to_dso",
            "visible_to_vpp_i",
            "visible_to_other_vpp",
            "oracle_only",
        ],
        max_rows=120,
    )
    return f"""
    <section class="panel">
      <div class="section-heading">
        <div>
          <p class="eyebrow">{_lang_html("Operating Envelope", "运行包络")}</p>
          <h2>{_lang_html("FR/DOE Envelope", "FR/DOE 可行域")}</h2>
        </div>
        <p class="section-note">{_lang_html("This v0 envelope is built from local DER bounds. Multi-node VPPs are shown by bus/zone/DER scope rather than a fake single PCC injection.", "当前 v0 包络由本地 DER 边界生成。多节点 VPP 按母线/分区/设备范围展示，而不是伪装成单个 PCC 注入。")}</p>
      </div>
      {envelope_table}
    </section>
    <section class="panel">
      <div class="section-heading">
        <div>
          <p class="eyebrow">{_lang_html("Safety Projection", "安全投影")}</p>
          <h2>{_lang_html("Projection Audit", "投影链路审计")}</h2>
        </div>
        <p class="section-note">{_lang_html("The trace records raw command, device bounds, FR/DOE projection, pandapower write and power-flow result so dispatch decisions can be audited step by step.", "该链路记录原始指令、设备边界、FR/DOE 投影、pandapower 写入和潮流结果，便于逐步审计调度决策。")}</p>
      </div>
      {projection_table}
    </section>
    <section class="panel">
      <div class="section-heading">
        <div>
          <p class="eyebrow">{_lang_html("Privacy Boundary", "隐私边界")}</p>
          <h2>{_lang_html("Privacy Visibility", "隐私可见性")}</h2>
        </div>
        <p class="section-note">{_lang_html("Schema visibility flags identify fields available to the DSO, the owning VPP, other VPPs, or oracle-only baselines.", "Schema 可见性标记说明哪些字段可被 DSO、所属 VPP、其他 VPP 或仅 oracle 基线访问。")}</p>
      </div>
      {privacy_table}
    </section>
    """


def _load_optional_training_frames(frames: dict[str, pd.DataFrame], output_path: Path) -> dict[str, pd.DataFrame]:
    merged = dict(frames)
    training_root = output_path.parent / "marl_baselines"
    for name in ("training_summary", "episode_metrics", "tuning_trials", "agent_role_map", "encoder_role_map"):
        if name not in merged or merged[name].empty:
            path = training_root / f"{name}.csv"
            merged[name] = pd.read_csv(path) if path.exists() else pd.DataFrame()
    deep_root = output_path.parent / "deep_rl"
    for name in (
        "deep_rl_training_summary",
        "deep_rl_episode_metrics",
        "deep_rl_step_metrics",
        "deep_rl_trajectory",
        "deep_rl_loss_metrics",
    ):
        if name not in merged or merged[name].empty:
            path = deep_root / f"{name}.csv"
            merged[name] = pd.read_csv(path) if path.exists() else pd.DataFrame()
    merged.update(
        build_rl_architecture_frames(
            agent_roles=_frame(merged, "agent_role_map"),
            encoder_roles=_frame(merged, "encoder_role_map"),
            deep_summary=_frame(merged, "deep_rl_training_summary"),
            asset_registry=_frame(merged, "asset_registry"),
        )
    )
    merged["rl_algorithm_variants"] = build_rl_algorithm_variant_frame(merged)
    merged["model_update_summary"] = model_update_summary_frame(merged)
    return merged


def _primary_ctde_active(frames: dict[str, pd.DataFrame]) -> bool:
    overview = _frame(frames, "rl_algorithm_overview")
    deep = _frame(frames, "deep_rl_training_summary")
    algorithm = _display_text(deep.iloc[0].get("algorithm"), "") if not deep.empty else ""
    primary = _display_text(deep.iloc[0].get("target_ctde_primary_trainer"), "") if not deep.empty else ""
    status = _display_text(overview.iloc[0].get("target_ctde_status"), "") if not overview.empty else ""
    return (
        algorithm == "privacy_separated_ctde_actor_critic"
        or primary.lower() in {"true", "1", "yes"}
        or status == "implemented_as_primary_privacy_separated_trainer"
        or status == "advanced_hierarchical_happo_implemented_as_recommended_trainer"
    )


def _pick_lang(row: pd.Series, en_column: str, zh_column: str) -> str:
    en = escape(_display_text(row.get(en_column), ""))
    zh = escape(_display_text(row.get(zh_column), en))
    return _lang_html(en, zh, display="block")


def _rl_architecture_workflow_html(frames: dict[str, pd.DataFrame]) -> str:
    return build_rl_architecture_diagram(
        frames,
        root_id="interactive-rl-architecture-workflow",
        heading_eyebrow_en="Arrow Workflow",
        heading_eyebrow_zh="箭头工作流",
        title_en="Paper-Style RL / MARL Control Loop",
        title_zh="论文总图风格 RL / MARL 控制闭环",
        description_en=(
            "This embedded figure keeps the operational chain visible inside the main report: "
            "VPP day-ahead bid -> DSO operating envelope -> VPP dispatch actors -> DER actions -> "
            "safety projection -> pandapower -> reward/critic/training update."
        ),
        description_zh=(
            "这张嵌入式总图把主报告中的关键链路直接画出来：VPP 日前报量/报价 -> DSO 运行包络 -> "
            "VPP 调度 actor -> DER 动作 -> 安全投影 -> pandapower -> 奖励/critic/训练更新。"
        ),
        heading_class="section-heading",
    )


def _neural_network_architecture_html(frames: dict[str, pd.DataFrame]) -> str:
    return build_neural_network_architecture_diagram(
        frames,
        root_id="interactive-neural-network-architecture",
        heading_class="section-heading",
    )


def _target_ctde_architecture_html(frames: dict[str, pd.DataFrame]) -> str:
    return build_target_ctde_architecture_diagram(
        frames,
        root_id="interactive-target-ctde-architecture",
        heading_class="section-heading",
    )


def _rl_architecture_html(frames: dict[str, pd.DataFrame]) -> str:
    overview = _frame(frames, "rl_algorithm_overview")
    variants = _frame(frames, "rl_algorithm_variants")
    agents = _frame(frames, "rl_agent_architecture")
    neural = _frame(frames, "rl_neural_network_architecture")
    target_ctde = _frame(frames, "rl_target_ctde_architecture")
    ctde_nodes = _frame(frames, "rl_ctde_nodes")
    ctde_edges = _frame(frames, "rl_ctde_edges")
    ctde_feedback = _frame(frames, "rl_ctde_feedback")
    flows = _frame(frames, "rl_data_flow")
    losses = _frame(frames, "rl_loss_components")
    ctde = _frame(frames, "rl_ctde_assessment")
    primary_ctde = _primary_ctde_active(frames)
    row = overview.iloc[0] if not overview.empty else pd.Series(dtype=object)
    algorithm = _display_text(row.get("algorithm_id"), "ippo_actor_critic")
    ctde_status = _display_text(row.get("ctde_status"), "proto_ctde_interface_not_full_ctde")
    training_mode = _display_text(row.get("training_mode"), "centralized_training")
    execution_mode = _display_text(
        row.get("execution_mode"),
        "envelope guidance plus learned DER disaggregation with safety projection",
    )
    target_ctde_status = _display_text(
        row.get("target_ctde_status"),
        "specified_privacy_preserving_ctde_not_yet_primary_trainer",
    )

    agent_cards = []
    for _, agent in agents.iterrows():
        trainable = str(agent.get("trainable", "")).lower() in {"true", "1", "yes"}
        agent_cards.append(
            f"""
            <article class="rl-card" data-rl-card data-rl-kind="agent">
              <div class="rl-card-head">
                <div>
                  <p class="eyebrow">{escape(_display_text(agent.get('role_type')))}</p>
                  <h3>{escape(_display_text(agent.get('agent_id')))}</h3>
                </div>
                <span class="mode-badge">{_lang_html("trainable" if trainable else "not trainable", "可训练" if trainable else "不可训练")}</span>
              </div>
              <div class="rl-card-body">
                <section><h4>{_lang_html("Input", "输入")}</h4><p>{_pick_lang(agent, "input_observation", "input_observation_zh")}</p></section>
                <section><h4>{_lang_html("Output / Action", "输出 / 动作")}</h4><p>{_pick_lang(agent, "action_output", "action_output_zh")}</p></section>
                <section><h4>{_lang_html("Uses RL?", "是否使用强化学习")}</h4><p>{_pick_lang(agent, "rl_usage_status", "rl_usage_status_zh")}</p></section>
                <section><h4>{_lang_html("Neural Network Structure", "神经网络结构")}</h4><p>{_pick_lang(agent, "neural_network_structure", "neural_network_structure_zh")}</p></section>
                <section><h4>{_lang_html("Output Formula", "输出公式")}</h4><p>{_pick_lang(agent, "result_formula", "result_formula_zh")}</p></section>
                <section><h4>{_lang_html("Result Calculation", "结果怎么算")}</h4><p>{_pick_lang(agent, "result_calculation", "result_calculation_zh")}</p></section>
                <section><h4>{_lang_html("Policy Module", "策略模块")}</h4><p>{_pick_lang(agent, "policy_module", "policy_module_zh")}</p></section>
                <section><h4>{_lang_html("Training Signal", "训练信号")}</h4><p>{_pick_lang(agent, "rl_training_signal", "rl_training_signal_zh")}</p></section>
                <section><h4>{_lang_html("Audit Outputs", "审计输出")}</h4><p>{_pick_lang(agent, "audit_outputs", "audit_outputs_zh")}</p></section>
                <section><h4>{_lang_html("Current Status", "当前状态")}</h4><p>{_pick_lang(agent, "implementation_status", "implementation_status_zh")}</p></section>
              </div>
            </article>
            """
        )

    flow_cards = []
    for _, flow in flows.sort_values("flow_order").iterrows() if not flows.empty else []:
        flow_cards.append(
            f"""
            <article class="rl-card compact" data-rl-card data-rl-kind="flow">
              <div class="rl-flow-line">
                <span>{escape(_display_text(flow.get('source')))}</span>
                <strong>→</strong>
                <span>{escape(_display_text(flow.get('target')))}</span>
              </div>
              <p class="rl-signal">{_lang_html(escape(_display_text(flow.get('signal'))), escape(_display_text(flow.get('signal_zh'), _display_text(flow.get('signal')))), display="block")}</p>
              <p>{_lang_html(escape(_display_text(flow.get('description'))), escape(_display_text(flow.get('description_zh'), _display_text(flow.get('description')))), display="block")}</p>
            </article>
            """
        )

    loss_cards = []
    for _, loss in losses.iterrows():
        loss_cards.append(
            f"""
            <article class="rl-card compact" data-rl-card data-rl-kind="loss">
              <div class="rl-card-head">
                <h3>{escape(_display_text(loss.get('component')))}</h3>
                <span class="mode-badge">{escape(_display_text(loss.get('coefficient')))}</span>
              </div>
              <code>{escape(_display_text(loss.get('formula')))}</code>
              <p>{_lang_html(escape(_display_text(loss.get('meaning'))), escape(_display_text(loss.get('meaning_zh'), _display_text(loss.get('meaning')))), display="block")}</p>
            </article>
            """
        )

    ctde_cards = []
    for _, item in ctde.iterrows():
        ctde_cards.append(
            f"""
            <article class="rl-card compact" data-rl-card data-rl-kind="ctde">
              <h3>{_lang_html(escape(_display_text(item.get('question'))), escape(_display_text(item.get('question_zh'), _display_text(item.get('question')))))}</h3>
              <p class="rl-answer">{_lang_html(escape(_display_text(item.get('answer'))), escape(_display_text(item.get('answer_zh'), _display_text(item.get('answer')))))}</p>
              <p>{_lang_html(escape(_display_text(item.get('evidence'))), escape(_display_text(item.get('evidence_zh'), _display_text(item.get('evidence')))), display="block")}</p>
            </article>
            """
        )

    benchmark_table = ""
    if not primary_ctde:
        benchmark_table = f"""
        <h3>{_lang_html("Shared-Backbone Benchmark Table", "共享骨干 benchmark 表")}</h3>
        {_html_table(neural, columns=["component_id", "component_group", "trainable", "input_shape", "output_shape", "structure", "distribution", "calculation_note"], max_rows=80)}
        """

    return f"""
    <section class="panel" id="rl-architecture">
      <div class="section-heading">
        <div>
          <p class="eyebrow">{_lang_html("RL / MARL Architecture", "强化学习 / 多智能体架构")}</p>
          <h2>{_lang_html("Current Learning Model Map", "当前学习模型结构图")}</h2>
        </div>
        <p class="section-note">{_lang_html(
            "This section is generated from learning/agent_roles.py, learning/rl_architecture.py and optional deep_rl training artifacts, so rerunning the report updates the cards when the model changes.",
            "本区块由 learning/agent_roles.py、learning/rl_architecture.py 和可选 deep_rl 训练产物生成；模型更新后重新生成报告，卡片会随之更新。",
        )}</p>
      </div>
      <div class="rl-overview-grid">
        <article class="kpi-card"><span>{_lang_html("Algorithm", "算法")}</span><strong>{escape(algorithm)}</strong></article>
        <article class="kpi-card"><span>{_lang_html("Training Mode", "训练模式")}</span><strong>{escape(training_mode)}</strong></article>
        <article class="kpi-card"><span>{_lang_html("Execution Mode", "执行模式")}</span><strong>{escape(execution_mode)}</strong></article>
        <article class="kpi-card"><span>{_lang_html("CTDE Status", "CTDE 状态")}</span><strong>{escape(ctde_status)}</strong></article>
        <article class="kpi-card"><span>{_lang_html("CTDE Implementation", "CTDE 实现状态")}</span><strong>{escape(target_ctde_status)}</strong></article>
      </div>
      <div class="rl-summary-card">
        <h3>{_lang_html("Plain-language answer", "通俗结论")}</h3>
        <p>{_pick_lang(row, "plain_language_summary", "plain_language_summary_zh")}</p>
        <p>{_lang_html(
            'Open the standalone detailed page: <a href="rl_architecture.html">rl_architecture.html</a>.',
            '打开独立详细页面：<a href="rl_architecture.html">rl_architecture.html</a>。',
            display="block",
        )}</p>
        <div class="code-grid">
          <code>{escape(_display_text(row.get("reward_formula"), "role-specific rewards: r_dso, r_dispatch_i, r_portfolio_i"))}</code>
          <code>{escape(_display_text(row.get("loss_formula"), "L = policy + value + entropy"))}</code>
          <code>{escape(_display_text(row.get("target_privacy_rule"), "DSO/VPP actors do not share raw observations; critic_global_state is training-only."))}</code>
        </div>
      </div>
      {build_rl_algorithm_variant_section(
          variants,
          title_en="HAPPO / MATD3 / HASAC Architecture Contrast",
          title_zh="HAPPO / MATD3 / HASAC 架构对照",
      )}
      {_rl_architecture_workflow_html(frames)}
      {_target_ctde_architecture_html(frames)}
      {_neural_network_architecture_html(frames)}
      <div class="rl-filter-toolbar" role="group" aria-label="RL architecture card filter">
        <button type="button" class="rl-filter is-active" data-rl-filter="all">{_lang_html("All", "全部")}</button>
        <button type="button" class="rl-filter" data-rl-filter="agent">{_lang_html("Agents", "智能体")}</button>
        <button type="button" class="rl-filter" data-rl-filter="flow">{_lang_html("Data Flow", "数据流")}</button>
        <button type="button" class="rl-filter" data-rl-filter="loss">{_lang_html("Loss", "损失函数")}</button>
        <button type="button" class="rl-filter" data-rl-filter="ctde">{_lang_html("CTDE", "CTDE 判断")}</button>
      </div>
      <div class="rl-graph paper-rl-graph">
        <div class="rl-node grid">{_lang_html("pandapower grid state", "pandapower 电网状态")}<small>{_lang_html("voltage / loading / profiles", "电压 / 负载率 / 曲线")}</small></div>
        <div class="rl-arrow">→<span>{_lang_html("state", "状态")}</span></div>
        <div class="rl-node vpp">{_lang_html("VPP day-ahead bid", "VPP 日前报量/报价")}<small>{_lang_html("FR, bid price, confidence", "可行域、报价、置信度")}</small></div>
        <div class="rl-arrow">→<span>{_lang_html("report", "上报")}</span></div>
        <div class="rl-node dso">{_lang_html("DSO operating envelope", "DSO 运行包络")}<small>{_lang_html("target + service request", "目标 + 服务请求")}</small></div>
        <div class="rl-arrow">→<span>{_lang_html("envelope", "包络")}</span></div>
        <div class="rl-node vpp">{_lang_html("VPP dispatch actors", "VPP 调度/解聚合智能体")}<small>selected_p_mw + der_actions</small></div>
        <div class="rl-arrow">→<span>{_lang_html("DER action", "DER 动作")}</span></div>
        <div class="rl-node der">{_lang_html("Safety projection", "安全投影")}<small>{_lang_html("FR/DOE + residual repair", "FR/DOE + 残差修复")}</small></div>
        <div class="rl-arrow">→<span>{_lang_html("write", "写入")}</span></div>
        <div class="rl-node grid">{_lang_html("power flow + reward", "潮流 + reward")}<small>{_lang_html("critic update feedback", "critic 更新反馈")}</small></div>
      </div>
      <div class="rl-card-grid">
        {''.join(agent_cards)}
        {''.join(flow_cards)}
        {''.join(loss_cards)}
        {''.join(ctde_cards)}
      </div>
      <details class="rl-detail-table">
        <summary>{_lang_html("Show architecture tables", "显示架构明细表")}</summary>
        <h3>{_lang_html("Agent Table", "智能体表")}</h3>
        {_html_table(
            agents,
            columns=[
                "agent_id",
                "role_type",
                "owner_id",
                "time_scale",
                "trainable",
                "is_rl_decision",
                "rl_usage_status",
                "input_observation",
                "action_output",
                "neural_network_structure",
                "result_formula",
                "result_calculation",
                "result_source",
                "rl_training_signal",
                "audit_outputs",
                "non_rl_guardrails",
                "implementation_status",
            ],
            max_rows=80,
        )}
        {benchmark_table}
        <h3>{_lang_html("Algorithm Variant Matrix", "算法架构对照表")}</h3>
        {_html_table(
            variants,
            columns=[
                "algorithm_id",
                "algorithm_label",
                "family",
                "actor_style",
                "critic_style",
                "update_core",
                "experience_reuse",
                "architecture_signature",
                "repo_status",
                "evidence_file",
            ],
            max_rows=20,
        )}
        <h3>{_lang_html("Privacy-Separated CTDE Component Table", "隐私分离 CTDE 组件表")}</h3>
        {_html_table(target_ctde, columns=["component_id", "component_group", "privacy_scope", "execution_visibility", "trainable", "input_shape", "output_shape", "structure", "distribution", "loss_signal", "conference_role"], max_rows=80)}
        <h3>{_lang_html("CTDE Graph Nodes", "CTDE 图节点")}</h3>
        {_html_table(ctde_nodes, columns=["component_id", "component_group", "phase", "owner_scope", "parameter_sharing_scope", "tensor_in", "input_shape", "tensor_out", "output_shape", "source_fn", "limitation", "next_upgrade"], max_rows=80)}
        <h3>{_lang_html("CTDE Graph Edges", "CTDE 图边")}</h3>
        {_html_table(ctde_edges, columns=["edge_id", "src_component_id", "dst_component_id", "signal_name", "signal_shape", "signal_type", "privacy_class", "carries_raw_private_obs", "carries_gradient"], max_rows=120)}
        <h3>{_lang_html("CTDE Feedback / Loss Paths", "CTDE 反馈 / 损失路径")}</h3>
        {_html_table(ctde_feedback, columns=["feedback_id", "target_component_id", "loss_name", "formula", "coefficient", "advantage_source", "reward_source", "metric_csv", "metric_columns"], max_rows=80)}
        <h3>{_lang_html("Loss Table", "损失函数表")}</h3>
        {_html_table(losses, columns=["component", "formula", "coefficient", "meaning"], max_rows=80)}
        <h3>{_lang_html("CTDE Assessment", "CTDE 判断")}</h3>
        {_html_table(ctde, columns=["question", "answer", "evidence"], max_rows=20)}
      </details>
    </section>
    """


def _training_summary_html(frames: dict[str, pd.DataFrame]) -> str:
    summary = _frame(frames, "training_summary")
    episodes = _frame(frames, "episode_metrics")
    trials = _frame(frames, "tuning_trials")
    roles = _frame(frames, "agent_role_map")
    deep_summary = _frame(frames, "deep_rl_training_summary")
    deep_episodes = _frame(frames, "deep_rl_episode_metrics")
    deep_losses = _frame(frames, "deep_rl_loss_metrics")
    if summary.empty:
        status = "not_available"
        best_algorithm = "n/a"
        handoff = "Run python examples/09_run_marl_baselines.py to generate training artifacts."
    else:
        row = summary.iloc[0]
        status = _display_text(row.get("status", "unknown"))
        best_algorithm = _display_text(row.get("best_algorithm", "n/a"))
        handoff = _display_text(row.get("handoff_message", "")) or _display_text(row.get("reason", ""))
    return f"""
    <section class="panel">
      <div class="section-heading">
        <div>
          <p class="eyebrow">{_lang_html("Training", "训练")}</p>
          <h2>{_lang_html("Training Supervisor Status", "Training Supervisor Status")}</h2>
        </div>
        <p class="section-note">{_lang_html("MARL baseline and tuning outputs are read from outputs/marl_baselines. Failed convergence explicitly hands the issue back to the main thread and algorithm agent.", "MARL baseline and tuning outputs are read from outputs/marl_baselines. Failed convergence explicitly hands the issue back to the main thread and algorithm agent.")}</p>
      </div>
      <div class="kpi-grid">
        <article class="kpi-card"><span>Status</span><strong>{escape(status)}</strong></article>
        <article class="kpi-card"><span>Best Algorithm</span><strong>{escape(best_algorithm)}</strong></article>
        <article class="kpi-card"><span>Handoff</span><strong>{escape(handoff)}</strong></article>
      </div>
      <h3>MARL Baselines</h3>
      {_html_table(episodes, columns=["algorithm", "episode", "episode_reward", "episode_cost", "violation_count"], max_rows=40)}
      <h3>Tuning Trials</h3>
      {_html_table(trials, columns=["trial_id", "algorithm", "action_scale", "exploration_noise", "mean_reward", "status"], max_rows=40)}
      <h3>Deep RL Actor-Critic</h3>
      {_html_table(deep_summary, columns=["algorithm", "status", "is_deep_rl", "optimizer_steps", "param_delta_l2", "dso_actor_trainable", "vpp_dispatch_trainable", "vpp_der_disaggregation_trainable", "dispatch_action_type", "portfolio_trainable", "best_episode_reward", "final_episode_reward"], max_rows=10)}
      <h3>Deep RL Episodes</h3>
      {_html_table(deep_episodes, columns=["episode", "algorithm", "episode_reward", "episode_cost", "violation_count", "projection_clipping_rate", "policy_loss", "value_loss", "entropy", "grad_norm"], max_rows=40)}
      <h3>Deep RL Loss Metrics</h3>
      {_html_table(deep_losses, columns=["episode", "algorithm", "policy_loss", "value_loss", "entropy_loss", "total_loss", "grad_norm", "optimizer_step"], max_rows=40)}
      <h3>Agent Role Map</h3>
      {_html_table(roles, columns=["agent_id", "role_type", "owner_id", "time_scale", "privacy_scope"], max_rows=40)}
    </section>
    """


def _model_update_summary_html(frames: dict[str, pd.DataFrame]) -> str:
    updates = _frame(frames, "model_update_summary")
    if updates.empty:
        updates = model_update_summary_frame(frames)
    return f"""
    <section class="panel">
      <div class="section-heading">
        <div>
          <p class="eyebrow">{_lang_html("Model Sync", "模型同步")}</p>
          <h2>{_lang_html("Model / Algorithm Update Summary", "模型 / 算法更新摘要")}</h2>
        </div>
        <p class="section-note">{_lang_html(
            "This table is generated once in dashboard_data/model_update_summary.csv and reused by every HTML report, so algorithm changes are visible consistently after refresh.",
            "该表由 dashboard_data/model_update_summary.csv 统一生成，并被所有 HTML 报告复用；算法更新后刷新报告即可保持各页面一致。",
        )}</p>
      </div>
      {_html_table(
          updates,
          columns=["update_area", "current_value", "current_value_zh", "explanation", "explanation_zh", "evidence_file"],
          max_rows=20,
      )}
    </section>
    """


def _economic_explanation_html(frames: dict[str, pd.DataFrame]) -> str:
    economic = _frame(frames, "economic_explanation")
    return f"""
    <section class="panel">
      <div class="section-heading">
        <div>
          <p class="eyebrow">{_lang_html("Economics", "经济解释")}</p>
          <h2>{_lang_html("Why Reward / Profit Proxy Can Be Negative", "为什么 Reward / Profit Proxy 会是负数")}</h2>
        </div>
        <p class="section-note">{_lang_html(
            "The simulator currently optimizes a system objective. Reward is negative total cost, while first-person profit proxy is only signed energy cashflow under the project sign convention.",
            "当前仿真器优化的是系统目标函数。Reward 等于总成本取负；第一视角中的 profit proxy 只是按项目符号约定计算的带符号能量现金流代理，并不是市场净利润。",
        )}</p>
      </div>
      {_html_table(economic, columns=["metric", "value", "share_of_total_cost", "formula", "interpretation", "why_negative"], max_rows=80)}
    </section>
    """


def _metric_calculation_html() -> str:
    return (
        """
    <section class="panel">
      <div class="section-heading">
        <div>
          <p class="eyebrow">"""
        + _lang_html("Methods Note", "方法说明")
        + """</p>
          <h2>"""
        + _lang_html("Metric Calculations", "指标计算")
        + """</h2>
        </div>
        <p class="section-note">"""
        + _lang_html(
            "Definitions below explain how the headline KPIs and operating-security traces are derived from pandapower result tables.",
            "下面的定义解释了关键 KPI 和运行安全曲线如何从 pandapower 结果表推导而来。",
        )
        + """</p>
      </div>
      <ul class="metric-list">
        <li>"""
        + _lang_html(
            "<strong>Min voltage</strong>: for each time point, compute the minimum of all bus <code>vm_pu</code> values from pandapower <code>res_bus</code>; the KPI shows the minimum over the full horizon.",
            "<strong>最低电压</strong>：对每个时刻，计算 pandapower <code>res_bus</code> 中所有母线 <code>vm_pu</code> 的最小值；KPI 显示全时域最小值。",
        )
        + """</li>
        <li>"""
        + _lang_html(
            "<strong>Max line loading</strong>: for each time point, compute the maximum <code>loading_percent</code> over all line branches; transformer loading is tracked separately when transformers exist.",
            "<strong>最大线路负载率</strong>：对每个时刻，计算所有线路支路的最大 <code>loading_percent</code>；若存在变压器，则单独跟踪其负载率。",
        )
        + """</li>
        <li>"""
        + _lang_html(
            "<strong>Line flow</strong>: <code>p_from_mw/q_from_mvar</code> come from pandapower <code>res_line</code> or transformer HV-side results. Positive <code>p_from_mw</code> means flow from the stored <code>from_bus</code> to <code>to_bus</code>.",
            "<strong>线路潮流</strong>：<code>p_from_mw/q_from_mvar</code> 来自 pandapower <code>res_line</code> 或变压器高压侧结果。正的 <code>p_from_mw</code> 表示从保存的 <code>from_bus</code> 流向 <code>to_bus</code>。",
        )
        + """</li>
        <li>"""
        + _lang_html(
            "<strong>VPP power</strong>: sum of internal DER active powers using the project sign convention, where positive means injection to the grid and negative means absorption.",
            "<strong>VPP 功率</strong>：按照项目符号约定汇总内部 DER 有功功率，正值表示向电网注入，负值表示吸收。",
        )
        + """</li>
        <li>"""
        + _lang_html(
            "<strong>PV available power</strong>: <code>PVModel.available_power(t) = p_max_mw * pv_forecast_factor[t]</code>, clipped to the PV nameplate limit. The forecast factor is read from the configured CSV or deterministic fallback profile.",
            "<strong>PV 可用功率</strong>：<code>PVModel.available_power(t) = p_max_mw * pv_forecast_factor[t]</code>，并裁剪到光伏额定上限。预测因子来自配置的 CSV 或确定性回退曲线。",
        )
        + """</li>
        <li>"""
        + _lang_html(
            "<strong>Training rewards</strong>: <code>r_dso</code>, <code>r_dispatch_i</code> and <code>r_portfolio_i</code> are separated. VPP dispatch optimizes self-interested settlement/delivery; portfolio receives a localized DSO-alignment credit instead of raw global reward sharing.",
            "<strong>训练奖励</strong>：<code>r_dso</code>、<code>r_dispatch_i</code>、<code>r_portfolio_i</code> 已分离。VPP 调度优化自身结算/履约收益；组合配置接收局部化 DSO 对齐收益，而不是直接共享原始全局 reward。",
        )
        + """</li>
      </ul>
    </section>
    """
    )


def build_interactive_report_html(
    frames: dict[str, pd.DataFrame],
    output_path: str | Path,
) -> Path:
    """Write an offline Plotly HTML report for a completed simulation."""

    path = Path(output_path)
    frames = _load_optional_training_frames(frames, path)
    go, pio = require_plotly()
    topology = _topology_figure(go, frames)
    profile = _profile_figure(go, frames)
    edge_flow = _edge_flow_figure(go, frames)
    vpp = _vpp_figure(go, _frame(frames, "vpp_state"), frames)
    der = _der_figure(go, _frame(frames, "der_state"), frames)
    alerts = _frame(frames, "alert_event")

    topology_html = pio.to_html(
        topology,
        include_plotlyjs="cdn",
        full_html=False,
        post_script=_asset_icon_toggle_script(),
    )
    profile_html = pio.to_html(profile, include_plotlyjs=False, full_html=False)
    edge_flow_html = pio.to_html(edge_flow, include_plotlyjs=False, full_html=False)
    vpp_html = pio.to_html(vpp, include_plotlyjs=False, full_html=False)
    der_html = pio.to_html(der, include_plotlyjs=False, full_html=False)

    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>pandapower VPP DSO simulation report</title>
  <style>
    :root {{
      --bg: #edf3f8;
      --panel: rgba(255, 255, 255, 0.94);
      --panel-strong: #ffffff;
      --border: #d6e0ea;
      --border-strong: #b8c7d9;
      --text: #102033;
      --muted: #526579;
      --muted-soft: #6f8092;
      --accent: #1177c3;
      --accent-soft: rgba(17, 119, 195, 0.12);
      --shadow: 0 14px 30px rgba(15, 23, 42, 0.08);
    }}
    * {{
      box-sizing: border-box;
    }}
    body {{
      margin: 0;
      font-family: "Segoe UI", "Helvetica Neue", Arial, sans-serif;
      color: var(--text);
      background:
        radial-gradient(circle at top left, rgba(17, 119, 195, 0.12), transparent 28%),
        linear-gradient(180deg, #dfeaf4 0%, var(--bg) 22%, #f6f9fc 100%);
    }}
    .lang-copy {{
      display: none;
    }}
    html[data-lang="en"] .lang-en.lang-inline,
    html[data-lang="zh"] .lang-zh.lang-inline {{
      display: inline;
    }}
    html[data-lang="en"] .lang-en.lang-block,
    html[data-lang="zh"] .lang-zh.lang-block {{
      display: block;
    }}
    .hero {{
      display: grid;
      grid-template-columns: minmax(0, 2.6fr) minmax(260px, 1fr);
      gap: 18px;
      padding: 30px 32px 24px;
      background:
        linear-gradient(135deg, rgba(7, 24, 42, 0.96), rgba(18, 52, 86, 0.96)),
        linear-gradient(90deg, rgba(74, 144, 226, 0.22), rgba(74, 144, 226, 0));
      color: white;
      border-bottom: 1px solid rgba(255, 255, 255, 0.12);
    }}
    .hero-topline {{
      display: flex;
      flex-wrap: wrap;
      justify-content: space-between;
      gap: 12px;
      align-items: center;
    }}
    .hero h1 {{
      margin: 4px 0 10px 0;
      font-size: clamp(28px, 4vw, 40px);
      line-height: 1.08;
    }}
    .language-toolbar {{
      display: inline-flex;
      align-items: center;
      gap: 8px;
      padding: 6px;
      border-radius: 999px;
      background: rgba(255, 255, 255, 0.08);
      border: 1px solid rgba(255, 255, 255, 0.14);
    }}
    .language-label {{
      color: #dbe7f3;
      font-size: 12px;
      padding: 0 6px 0 8px;
    }}
    .lang-button {{
      border: 0;
      border-radius: 999px;
      padding: 7px 12px;
      background: transparent;
      color: #dbe7f3;
      font-size: 12px;
      font-weight: 700;
      cursor: pointer;
    }}
    .lang-button.is-active {{
      background: rgba(143, 208, 255, 0.2);
      color: white;
    }}
    .lang-button:focus-visible {{
      outline: 2px solid rgba(143, 208, 255, 0.5);
      outline-offset: 2px;
    }}
    .hero-lead {{
      margin: 0;
      max-width: 820px;
      color: rgba(226, 232, 240, 0.96);
      font-size: 15px;
      line-height: 1.55;
    }}
    .hero-pill-row {{
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      margin-top: 18px;
    }}
    .hero-pill {{
      display: inline-flex;
      align-items: center;
      padding: 6px 10px;
      border-radius: 999px;
      background: rgba(255, 255, 255, 0.1);
      border: 1px solid rgba(255, 255, 255, 0.12);
      color: #dbe7f3;
      font-size: 12px;
      letter-spacing: 0.02em;
    }}
    .hero-brief {{
      align-self: end;
      padding: 16px 18px;
      border-radius: 18px;
      background: rgba(255, 255, 255, 0.08);
      border: 1px solid rgba(255, 255, 255, 0.14);
      backdrop-filter: blur(8px);
    }}
    .brief-label {{
      color: #8fd0ff;
      font-size: 11px;
      font-weight: 700;
      letter-spacing: 0.12em;
      text-transform: uppercase;
      margin-bottom: 8px;
    }}
    .hero-brief p {{
      margin: 0;
      color: #dbe7f3;
      font-size: 13px;
      line-height: 1.6;
    }}
    main {{
      padding: 24px 28px 36px;
    }}
    .panel {{
      background: var(--panel);
      border: 1px solid var(--border);
      border-radius: 18px;
      padding: 18px 20px;
      margin-bottom: 18px;
      box-shadow: var(--shadow);
      backdrop-filter: blur(10px);
    }}
    .kpi-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
      gap: 12px;
      background: transparent;
      border: 0;
      padding: 0;
      margin-bottom: 18px;
    }}
    .kpi-card {{
      background: linear-gradient(180deg, rgba(255, 255, 255, 0.98), rgba(245, 249, 252, 0.98));
      border: 1px solid var(--border);
      border-radius: 16px;
      padding: 14px 16px;
      box-shadow: 0 8px 22px rgba(15, 23, 42, 0.05);
    }}
    .kpi-grid span {{
      display: block;
      color: var(--muted);
      font-size: 11px;
      font-weight: 700;
      letter-spacing: 0.08em;
      text-transform: uppercase;
      margin-bottom: 6px;
    }}
    .kpi-grid strong {{
      font-size: 22px;
      letter-spacing: -0.02em;
    }}
    .eyebrow {{
      margin: 0;
      color: var(--accent);
      font-size: 11px;
      font-weight: 700;
      letter-spacing: 0.12em;
      text-transform: uppercase;
    }}
    .hero .eyebrow {{
      color: #8fd0ff;
    }}
    .section-heading {{
      display: flex;
      justify-content: space-between;
      gap: 18px;
      align-items: end;
      margin-bottom: 14px;
    }}
    h2 {{
      font-size: 22px;
      margin: 4px 0 0 0;
      letter-spacing: -0.02em;
    }}
    h3 {{
      margin: 0 0 8px 0;
      font-size: 15px;
    }}
    .section-note {{
      max-width: 520px;
      margin: 0;
      color: var(--muted);
      font-size: 13px;
      line-height: 1.55;
    }}
    code {{
      background: #edf4fa;
      border: 1px solid #d7e5f1;
      border-radius: 6px;
      padding: 1px 5px;
    }}
    .hint {{
      color: var(--muted);
      font-size: 13px;
      margin: 0 0 12px 0;
      line-height: 1.55;
    }}
    .intro-panel {{
      padding-top: 20px;
    }}
    .guide-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
      gap: 12px;
    }}
    .guide-card {{
      padding: 14px 15px;
      border-radius: 14px;
      background: linear-gradient(180deg, rgba(244, 248, 252, 0.98), rgba(255, 255, 255, 0.98));
      border: 1px solid var(--border);
    }}
    .guide-card p {{
      margin: 0;
      color: var(--muted);
      font-size: 13px;
      line-height: 1.55;
    }}
    .metric-list {{
      margin: 0;
      padding-left: 20px;
      line-height: 1.55;
      color: #1f2937;
    }}
    .alert-table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 13px;
      background: var(--panel-strong);
    }}
    .summary-table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 13px;
      margin-top: 8px;
      background: var(--panel-strong);
      border-radius: 14px;
      overflow: hidden;
    }}
    .summary-table.compact {{
      max-width: 720px;
    }}
    .alert-table th, .alert-table td, .summary-table th, .summary-table td {{
      border-bottom: 1px solid #e5edf4;
      padding: 9px 10px;
      text-align: left;
      vertical-align: top;
    }}
    .alert-table thead th, .summary-table thead th {{
      position: sticky;
      top: 0;
      background: #f5f9fc;
      color: #314155;
      font-size: 12px;
      text-transform: uppercase;
      letter-spacing: 0.05em;
    }}
    .legend-shell {{
      display: grid;
      grid-template-columns: minmax(0, 1.1fr) minmax(280px, 0.9fr);
      gap: 14px;
      margin-bottom: 14px;
    }}
    .legend-help {{
      display: grid;
      gap: 12px;
      color: #334155;
      font-size: 13px;
      line-height: 1.45;
    }}
    .legend-callout {{
      padding: 14px 15px;
      border-radius: 14px;
      background: linear-gradient(180deg, rgba(241, 247, 252, 0.98), rgba(255, 255, 255, 0.98));
      border: 1px solid var(--border);
    }}
    .legend-callout p {{
      margin: 0;
      color: var(--muted);
      line-height: 1.55;
    }}
    .dispatch-vpp-card {{
      border: 1px solid var(--border);
      border-radius: 16px;
      padding: 16px;
      background: linear-gradient(180deg, rgba(244, 248, 252, 0.92), rgba(255, 255, 255, 0.98));
    }}
    .dispatch-vpp-card + .dispatch-vpp-card {{
      margin-top: 16px;
    }}
    .dispatch-vpp-head {{
      display: flex;
      justify-content: space-between;
      gap: 14px;
      align-items: start;
      margin-bottom: 14px;
    }}
    .dispatch-vpp-title {{
      margin: 4px 0 0 0;
      font-size: 19px;
      letter-spacing: -0.02em;
    }}
    .dispatch-count-pill {{
      display: inline-flex;
      align-items: center;
      padding: 7px 10px;
      border-radius: 999px;
      background: var(--accent-soft);
      border: 1px solid rgba(17, 119, 195, 0.18);
      color: #0f4f85;
      font-size: 12px;
      font-weight: 700;
      white-space: nowrap;
    }}
    .dispatch-vpp-list {{
      display: grid;
      gap: 12px;
    }}
    .dispatch-window-card {{
      border-radius: 14px;
      border: 1px solid #dbe6f0;
      background: rgba(255, 255, 255, 0.96);
      padding: 14px;
      box-shadow: 0 8px 22px rgba(15, 23, 42, 0.04);
    }}
    .dispatch-window-meta {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
      gap: 10px;
    }}
    .dispatch-metric {{
      padding: 10px 11px;
      border-radius: 12px;
      background: #f7fafc;
      border: 1px solid #e4edf5;
    }}
    .dispatch-metric-label {{
      display: block;
      color: var(--muted);
      font-size: 11px;
      font-weight: 700;
      letter-spacing: 0.06em;
      text-transform: uppercase;
      margin-bottom: 6px;
    }}
    .dispatch-metric strong {{
      display: block;
      font-size: 14px;
      line-height: 1.45;
      color: #17324c;
    }}
    .dispatch-window-body {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
      gap: 12px;
      margin-top: 12px;
    }}
    .dispatch-copy-card {{
      padding: 13px 14px;
      border-radius: 12px;
      background: linear-gradient(180deg, rgba(245, 249, 252, 0.96), rgba(255, 255, 255, 1));
      border: 1px solid #e4edf5;
    }}
    .dispatch-copy-card h3 {{
      margin-bottom: 10px;
    }}
    .dispatch-copy-card .lang-block {{
      color: #304559;
      font-size: 13px;
      line-height: 1.62;
      white-space: pre-wrap;
    }}
    .first-person-toolbar {{
      display: flex;
      flex-wrap: wrap;
      align-items: center;
      gap: 10px;
      margin-bottom: 14px;
    }}
    .first-person-toolbar label {{
      font-size: 12px;
      font-weight: 800;
      color: var(--muted);
      text-transform: uppercase;
      letter-spacing: 0.08em;
    }}
    .first-person-toolbar select {{
      min-width: 220px;
      border: 1px solid var(--border-strong);
      border-radius: 10px;
      padding: 8px 10px;
      background: white;
      color: var(--text);
    }}
    .first-person-list {{
      display: grid;
      gap: 12px;
      margin-bottom: 16px;
    }}
    .first-person-card {{
      border: 1px solid #dbe6f0;
      border-radius: 14px;
      background: linear-gradient(180deg, rgba(246, 249, 252, 0.98), rgba(255, 255, 255, 0.98));
      padding: 14px;
    }}
    .first-person-head {{
      display: flex;
      justify-content: space-between;
      gap: 12px;
      align-items: flex-start;
      margin-bottom: 12px;
    }}
    .first-person-head h3 {{
      margin: 4px 0 0;
      font-size: 18px;
    }}
    .badge-row {{
      display: flex;
      flex-wrap: wrap;
      justify-content: flex-end;
      gap: 6px;
    }}
    .mode-badge {{
      border-radius: 999px;
      border: 1px solid rgba(17, 119, 195, 0.22);
      background: var(--accent-soft);
      color: #0f4f85;
      padding: 5px 8px;
      font-size: 11px;
      font-weight: 800;
      overflow-wrap: anywhere;
    }}
    .first-person-grid {{
      display: grid;
      grid-template-columns: repeat(3, minmax(0, 1fr));
      gap: 12px;
    }}
    .first-person-grid section {{
      border: 1px solid #e4edf5;
      border-radius: 12px;
      background: white;
      padding: 12px;
      min-width: 0;
    }}
    .first-person-grid h4 {{
      margin: 0 0 8px;
      font-size: 14px;
    }}
    .first-person-grid p {{
      color: #304559;
      font-size: 13px;
      line-height: 1.55;
      margin: 0 0 8px;
      overflow-wrap: anywhere;
    }}
    .first-person-grid code,
    .code-grid code {{
      display: block;
      max-height: 140px;
      overflow: auto;
      white-space: pre-wrap;
      overflow-wrap: anywhere;
      font-size: 12px;
      line-height: 1.45;
    }}
    .code-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
      gap: 10px;
      margin-top: 10px;
    }}
    .summary-table {{
      display: block;
      overflow-x: auto;
      white-space: normal;
    }}
    .summary-table td {{
      overflow-wrap: anywhere;
    }}
    .rl-overview-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(190px, 1fr));
      gap: 12px;
      margin-bottom: 14px;
    }}
    .rl-overview-grid .kpi-card {{
      margin: 0;
    }}
    .rl-overview-grid strong {{
      display: block;
      font-size: 15px;
      line-height: 1.35;
      overflow-wrap: anywhere;
    }}
    .rl-summary-card {{
      border: 1px solid var(--border);
      border-radius: 16px;
      padding: 14px 16px;
      background: linear-gradient(180deg, rgba(245,249,252,0.98), rgba(255,255,255,0.98));
      margin-bottom: 14px;
    }}
    .rl-summary-card p {{
      margin: 0;
      color: #304559;
      line-height: 1.62;
    }}
    .rl-filter-toolbar {{
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      margin: 12px 0 14px;
    }}
    .rl-filter {{
      border: 1px solid var(--border-strong);
      background: white;
      color: #1b334a;
      border-radius: 999px;
      padding: 8px 12px;
      cursor: pointer;
      font-weight: 800;
      font-size: 12px;
    }}
    .rl-filter.is-active {{
      background: var(--accent);
      border-color: var(--accent);
      color: white;
    }}
    .rl-graph {{
      display: grid;
      grid-template-columns: minmax(150px, 1fr) auto minmax(150px, 1fr) auto minmax(150px, 1fr) auto minmax(170px, 1fr) auto minmax(150px, 1fr) auto minmax(150px, 1fr);
      gap: 10px;
      align-items: center;
      margin-bottom: 16px;
      overflow-x: auto;
    }}
    #rl-architecture .rl-graph.paper-rl-graph {{
      display: none;
    }}
    .rl-node {{
      min-height: 64px;
      border: 1px solid var(--border);
      border-radius: 16px;
      display: flex;
      align-items: center;
      justify-content: center;
      text-align: center;
      padding: 10px;
      font-weight: 800;
      line-height: 1.35;
      background: #fff;
      box-shadow: 0 8px 20px rgba(15, 23, 42, 0.04);
      flex-direction: column;
    }}
    .rl-node small {{
      display: block;
      margin-top: 6px;
      color: var(--muted);
      font-size: 11px;
      font-weight: 700;
      overflow-wrap: anywhere;
    }}
    .rl-node.dso {{ border-color: #8abde2; background: #eef7fe; }}
    .rl-node.vpp {{ border-color: #a6cfa6; background: #f0faf0; }}
    .rl-node.der {{ border-color: #d8c18c; background: #fff8e8; }}
    .rl-node.grid {{ border-color: #d6a2a2; background: #fff1f1; }}
    .rl-arrow {{
      color: var(--accent);
      font-size: 24px;
      font-weight: 900;
      text-align: center;
    }}
    .rl-arrow span {{
      display: block;
      margin-top: 5px;
      color: var(--muted);
      font-size: 10px;
      line-height: 1.25;
    }}
    .rl-card-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(290px, 1fr));
      gap: 12px;
    }}
    .rl-card {{
      border: 1px solid var(--border);
      border-radius: 16px;
      background: rgba(255,255,255,0.98);
      padding: 14px;
      box-shadow: 0 8px 18px rgba(15,23,42,0.04);
    }}
    .rl-card.compact {{
      background: linear-gradient(180deg, rgba(248,251,253,0.98), rgba(255,255,255,0.98));
    }}
    .rl-card[hidden] {{
      display: none;
    }}
    .rl-card-head {{
      display: flex;
      justify-content: space-between;
      gap: 12px;
      align-items: flex-start;
      margin-bottom: 10px;
    }}
    .rl-card h3 {{
      margin: 3px 0 0;
      overflow-wrap: anywhere;
    }}
    .rl-card h4 {{
      margin: 0 0 6px;
      font-size: 12px;
      color: var(--muted);
      text-transform: uppercase;
      letter-spacing: 0.06em;
    }}
    .rl-card p {{
      margin: 8px 0 0;
      color: #304559;
      line-height: 1.58;
      overflow-wrap: anywhere;
    }}
    .rl-card code {{
      display: block;
      white-space: pre-wrap;
      overflow-wrap: anywhere;
      line-height: 1.5;
      margin: 8px 0;
    }}
    .rl-card-body {{
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 10px;
    }}
    .rl-card-body section {{
      border: 1px solid #e5edf4;
      border-radius: 12px;
      padding: 10px;
      background: #f9fcfe;
      min-width: 0;
    }}
    .rl-card-body p {{
      margin: 0;
      font-size: 13px;
    }}
    .rl-flow-line {{
      display: grid;
      grid-template-columns: minmax(0, 1fr) auto minmax(0, 1fr);
      gap: 8px;
      align-items: center;
      color: #17324c;
      font-weight: 800;
    }}
    .rl-flow-line span {{
      overflow-wrap: anywhere;
    }}
    .rl-flow-line strong {{
      color: var(--accent);
      font-size: 19px;
    }}
    .rl-signal {{
      font-weight: 800;
      color: #0f4f85 !important;
    }}
    .rl-answer {{
      display: inline-flex;
      padding: 6px 9px;
      border-radius: 999px;
      background: var(--accent-soft);
      color: #0f4f85 !important;
      font-weight: 800;
    }}
    .rl-detail-table {{
      margin-top: 14px;
      border: 1px solid var(--border);
      border-radius: 14px;
      background: #fff;
      padding: 12px 14px;
    }}
    .rl-detail-table summary {{
      cursor: pointer;
      font-weight: 800;
      color: #17324c;
    }}
    .empty {{
      color: var(--muted-soft);
      margin: 0;
    }}
    {rl_architecture_diagram_css()}
    {neural_network_diagram_css()}
    {rl_algorithm_variant_section_css()}
    @media (max-width: 960px) {{
      .hero {{
        grid-template-columns: 1fr;
      }}
      .section-heading {{
        flex-direction: column;
        align-items: flex-start;
      }}
      .legend-shell {{
        grid-template-columns: 1fr;
      }}
      .dispatch-vpp-head {{
        flex-direction: column;
      }}
      .first-person-grid {{
        grid-template-columns: 1fr;
      }}
      .rl-graph {{
        grid-template-columns: 1fr;
      }}
      .rl-arrow {{
        transform: rotate(90deg);
      }}
      .rl-card-body {{
        grid-template-columns: 1fr;
      }}
      .first-person-head {{
        flex-direction: column;
      }}
    }}
    @media (max-width: 640px) {{
      main {{
        padding: 18px 14px 28px;
      }}
      .hero {{
        padding: 22px 16px 18px;
      }}
      .panel {{
        padding: 15px 14px;
        border-radius: 16px;
      }}
      h2 {{
        font-size: 19px;
      }}
    }}
  </style>
</head>
<body>
  {_header_html(frames)}
  <main>
    {_summary_html(frames)}
    {_reading_guide_html()}
    <section class="panel">
      <div class="section-heading">
        <div>
          <p class="eyebrow">{_lang_html("Asset Inventory", "资产清单")}</p>
          <h2>{_lang_html("VPP / PCC / Asset Map", "VPP / PCC / 资产映射")}</h2>
        </div>
        <p class="section-note">{_lang_html("This table links each VPP identifier to its PCC bus, owned DER classes and registered asset IDs before you inspect per-step topology states.", "在查看逐时步拓扑状态之前，这张表先把每个 VPP 标识符与其 PCC 母线、所属 DER 类型和已注册资产编号对应起来。")}</p>
      </div>
      {_vpp_summary_html(frames)}
    </section>
    <section class="panel">
      <div class="section-heading">
        <div>
          <p class="eyebrow">{_lang_html("Voltage Classes", "电压等级")}</p>
          <h2>{_lang_html("Bus / Feeder Voltage Levels", "母线 / 馈线电压等级")}</h2>
        </div>
        <p class="section-note">{_lang_html("Use these tables to cross-check nominal bus kV and feeder branch voltage classes before reading per-step voltages or branch stress.", "在查看逐步电压或支路压力之前，先用这些表核对母线额定 kV 和馈线支路电压等级。")}</p>
      </div>
      {_voltage_levels_html(frames)}
    </section>
    <section class="panel">
      <div class="section-heading">
        <div>
          <p class="eyebrow">{_lang_html("Spatial Replay", "空间回放")}</p>
          <h2>{_lang_html("Topology Replay", "拓扑回放")}</h2>
        </div>
        <p class="section-note">{_lang_html("Use the time slider to replay feeder state. Bus labels show nominal kV, feeder labels show branch voltage class, and the right-side legend/colorbar now sit outside the schematic to avoid overlap with the plot body.", "使用时间滑块回放馈线状态。母线标签显示额定 kV，馈线标签显示支路电压等级，右侧图例和色条放在示意图外部以避免与图体重叠。")}</p>
      </div>
      <p class="hint">{_lang_html("Hover any branch for detailed MW, MVAr and loading values. The axes show electrical depth and branch-lane separation rather than GIS distance.", "悬停任意支路可查看详细 MW、MVAr 和负载率数值。坐标轴表示电气层级深度和支路分道，而不是 GIS 距离。")}</p>
      {_topology_legend_html()}
      {topology_html}
    </section>
    <section class="panel">
      <div class="section-heading">
        <div>
          <p class="eyebrow">{_lang_html("Horizon Scan", "全时域扫描")}</p>
          <h2>{_lang_html("Every-Line Power Flow", "全线路潮流")}</h2>
        </div>
        <p class="section-note">{_lang_html("Read the first panel as signed active power by branch, the second as the system loading envelope, and the third as the highest peak-loaded branches. This separates flow direction, fleet-wide stress and worst offenders.", "第一部分读取各支路带符号有功功率，第二部分读取系统负载包络，第三部分读取峰值负载最高的支路。这样可以分开理解潮流方向、全局压力和重点拥塞对象。")}</p>
      </div>
      {edge_flow_html}
    </section>
    {_fr_doe_html(frames)}
    {_first_person_html(frames)}
    {_model_update_summary_html(frames)}
    {_rl_architecture_html(frames)}
    {_training_summary_html(frames)}
    {_economic_explanation_html(frames)}
    {_decision_drivers_html(frames)}
    <section class="panel">
      <div class="section-heading">
        <div>
          <p class="eyebrow">{_lang_html("Dispatch Narrative", "调度说明")}</p>
          <h2>{_lang_html("One-Day VPP Dispatch Instructions", "单日 VPP 调度指令")}</h2>
        </div>
        <p class="section-note">{_lang_html("Each card summarizes one contiguous first-day instruction segment. The explanation is computed from simulated VPP output, DER response, price, load and PV forecast profiles.", "每张卡片总结首日内一段连续调度区间。说明内容由模拟得到的 VPP 出力、DER 响应、电价、负荷和 PV 预测曲线共同推导。")}</p>
      </div>
      {_dispatch_explanation_html(frames)}
    </section>
    <section class="panel">
      <div class="section-heading">
        <div>
          <p class="eyebrow">{_lang_html("Driving Profiles", "驱动曲线")}</p>
          <h2>{_lang_html("Price / Load / PV Forecast", "电价 / 负荷 / PV 预测")}</h2>
        </div>
        <p class="section-note">{_lang_html("Profile traces explain the external operating conditions that drive DER dispatch and feeder loading at each replay step.", "这些曲线解释了每个回放时步中驱动 DER 调度和馈线负载变化的外部运行条件。")}</p>
      </div>
      {profile_html}
    </section>
    <section class="panel">
      <div class="section-heading">
        <div>
          <p class="eyebrow">{_lang_html("Aggregation", "聚合")}</p>
          <h2>{_lang_html("VPP Power", "VPP 功率")}</h2>
        </div>
        <p class="section-note">{_lang_html("Aggregate VPP active power follows the project sign convention: positive injects into the grid, negative absorbs from it.", "VPP 聚合有功功率遵循项目符号约定：正值向电网注入，负值从电网吸收。")}</p>
      </div>
      {vpp_html}
    </section>
    <section class="panel">
      <div class="section-heading">
        <div>
          <p class="eyebrow">{_lang_html("Device Dispatch", "设备调度")}</p>
          <h2>{_lang_html("DER Dispatch", "DER 调度")}</h2>
        </div>
        <p class="section-note">{_lang_html("Inspect individual device trajectories when VPP-level curves hide internal balancing between storage, generation and flexible load resources.", "当 VPP 级曲线掩盖了储能、发电和柔性负荷之间的内部平衡时，可在这里查看单个设备轨迹。")}</p>
      </div>
      {der_html}
    </section>
    <section class="panel">
      <div class="section-heading">
        <div>
          <p class="eyebrow">{_lang_html("Events", "事件")}</p>
          <h2>{_lang_html("Alerts", "告警")}</h2>
        </div>
        <p class="section-note">{_lang_html("The table below lists the first 200 constraint events captured in the standardized alert log for quick incident review.", "下表列出标准化告警日志中记录的前 200 条约束事件，便于快速复盘。")}</p>
      </div>
      {_alerts_html(alerts)}
    </section>
    {_metric_calculation_html()}
  </main>
  <script>{_language_toggle_script()}</script>
  <script>{_first_person_filter_script()}</script>
  <script>{_rl_architecture_filter_script()}</script>
</body>
</html>
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(html, encoding="utf-8")
    return path
