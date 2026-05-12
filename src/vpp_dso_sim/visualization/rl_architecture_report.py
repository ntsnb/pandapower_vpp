from __future__ import annotations

from html import escape
import json
from pathlib import Path
from typing import Any

import pandas as pd

from vpp_dso_sim.learning.rl_architecture import build_rl_architecture_frames
from vpp_dso_sim.visualization.rl_architecture_diagram import (
    build_rl_architecture_diagram,
    rl_architecture_diagram_css,
)
from vpp_dso_sim.visualization.neural_network_diagram import (
    build_neural_network_architecture_diagram,
    build_target_ctde_architecture_diagram,
    neural_network_diagram_css,
)
from vpp_dso_sim.visualization.dashboard_data import frame_or_empty, model_update_summary_frame
from vpp_dso_sim.visualization.rl_algorithm_variants import (
    build_rl_algorithm_variant_frame,
    build_rl_algorithm_variant_section,
    rl_algorithm_variant_section_css,
)


def _text(value: object, default: str = "") -> str:
    if value is None or pd.isna(value):
        return default
    text = str(value).strip()
    return text if text else default


def _lang(en: str, zh: str, *, block: bool = False) -> str:
    display = "block" if block else "inline"
    return (
        f"<span class='lang-copy lang-{display} lang-en'>{en}</span>"
        f"<span class='lang-copy lang-{display} lang-zh'>{zh}</span>"
    )


def _pick(row: pd.Series, en_col: str, zh_col: str | None = None) -> str:
    en = escape(_text(row.get(en_col)))
    zh = escape(_text(row.get(zh_col or f"{en_col}_zh"), en))
    return _lang(en, zh, block=True)


def _records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    return frame.fillna("").to_dict(orient="records") if not frame.empty else []


def _json_script(name: str, value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False).replace("</", "<\\/")
    return f"<script id='{escape(name)}' type='application/json'>{payload}</script>"


def _language_toolbar() -> str:
    return """
    <div class="language-toolbar" role="group" aria-label="Language">
      <span>Language / 语言</span>
      <button type="button" class="lang-button" data-lang-switch="en">EN</button>
      <button type="button" class="lang-button is-active" data-lang-switch="zh">中文</button>
    </div>
    """


def _load_optional_deep_rl_frames(frames: dict[str, pd.DataFrame], output_path: Path) -> dict[str, pd.DataFrame]:
    merged = dict(frames)
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
            agent_roles=frame_or_empty(merged, "agent_role_map"),
            encoder_roles=frame_or_empty(merged, "encoder_role_map"),
            deep_summary=frame_or_empty(merged, "deep_rl_training_summary"),
            asset_registry=frame_or_empty(merged, "asset_registry"),
        )
    )
    merged["rl_algorithm_variants"] = build_rl_algorithm_variant_frame(merged)
    merged["model_update_summary"] = model_update_summary_frame(merged)
    return merged


def _primary_ctde_active(frames: dict[str, pd.DataFrame]) -> bool:
    overview = frame_or_empty(frames, "rl_algorithm_overview")
    deep = frame_or_empty(frames, "deep_rl_training_summary")
    algorithm = _text(deep.iloc[0].get("algorithm")) if not deep.empty else ""
    primary = _text(deep.iloc[0].get("target_ctde_primary_trainer")) if not deep.empty else ""
    status = _text(overview.iloc[0].get("target_ctde_status")) if not overview.empty else ""
    return (
        algorithm == "privacy_separated_ctde_actor_critic"
        or primary.lower() in {"true", "1", "yes"}
        or status == "implemented_as_primary_privacy_separated_trainer"
        or status == "advanced_hierarchical_happo_implemented_as_recommended_trainer"
    )


def _summary_cards(overview: pd.DataFrame) -> str:
    row = overview.iloc[0] if not overview.empty else pd.Series(dtype=object)
    cards = [
        ("Algorithm", "算法", "algorithm_id"),
        ("Training Mode", "训练模式", "training_mode"),
        ("Execution Mode", "执行模式", "execution_mode"),
        ("CTDE Status", "CTDE 状态", "ctde_status"),
        ("Reward", "奖励", "reward_formula"),
        ("Loss", "损失", "loss_formula"),
    ]
    body = []
    for en_label, zh_label, column in cards:
        body.append(
            f"""
            <article class="summary-card">
              <span>{_lang(en_label, zh_label)}</span>
              <strong>{escape(_text(row.get(column), "n/a"))}</strong>
            </article>
            """
        )
    return f"<div class='summary-grid'>{''.join(body)}</div>"


def _group_legend(groups: pd.DataFrame, agents: pd.DataFrame) -> str:
    cards = []
    for _, group in groups.iterrows():
        group_id = _text(group.get("agent_group"))
        color = _text(group.get("color"), "#64748b")
        count = int((agents["agent_group"].astype(str) == group_id).sum()) if "agent_group" in agents else 0
        cards.append(
            f"""
            <article class="group-card" style="--group-color:{escape(color)}">
              <div class="color-dot"></div>
              <h3>{_lang(escape(_text(group.get('label'))), escape(_text(group.get('label_zh'))))}</h3>
              <p>{_pick(group, "summary", "summary_zh")}</p>
              <span class="count-pill">{count} agents / {count} 个智能体</span>
            </article>
            """
        )
    return f"<div class='group-grid'>{''.join(cards)}</div>"


def _agent_buttons(agents: pd.DataFrame) -> str:
    buttons = []
    for _, agent in agents.iterrows():
        color = _text(agent.get("agent_group_color"), "#64748b")
        buttons.append(
            f"""
            <button type="button" class="agent-node" data-agent-id="{escape(_text(agent.get('agent_id')))}" style="--agent-color:{escape(color)}">
              <span>{escape(_text(agent.get('agent_group_label_zh'), _text(agent.get('agent_group_label'))))}</span>
              <strong>{escape(_text(agent.get('agent_id')))}</strong>
              <em>{escape(_text(agent.get('implementation_status_zh'), _text(agent.get('implementation_status'))))}</em>
            </button>
            """
        )
    return "".join(buttons)


def _paper_style_diagram(agents: pd.DataFrame) -> str:
    return f"""
    <section class="panel">
      <div class="section-head">
        <div>
          <p class="eyebrow">Framework</p>
          <h2>{_lang("Paper-Style Model Structure", "论文式模型结构框架")}</h2>
        </div>
        <p>{_lang(
            "Click any agent module to inspect its observation, action, reward and implementation status.",
            "点击任意智能体模块，可查看它的观测、动作、奖励函数和当前实现状态。",
            block=True,
        )}</p>
      </div>
      <div class="framework-shell">
        <div class="framework-column env">
          <h3>{_lang("Physical Environment", "物理环境")}</h3>
          <div class="paper-box">pandapower runpp<br>{_lang("network constraints", "电网约束")}</div>
          <div class="paper-box">{_lang("Profiles", "外部曲线")}<br>price / load / PV</div>
        </div>
        <div class="arrow">→</div>
        <div class="framework-column agents">
          <h3>{_lang("Heterogeneous Agents", "异构智能体")}</h3>
          <div class="agent-node-grid">{_agent_buttons(agents)}</div>
        </div>
        <div class="arrow">→</div>
        <div class="framework-column action">
          <h3>{_lang("Action Projection", "动作投影")}</h3>
          <div class="paper-box">FR/DOE<br>{_lang("feasible envelope", "可行域包络")}</div>
          <div class="paper-box current">{_lang("Current", "当前")}<br>{_lang("learned DER actions + safety projection", "学习型 DER 动作 + 安全投影")}</div>
          <div class="paper-box target">{_lang("Target", "目标")}<br>{_lang("decentralized CTDE VPP lower policy", "分散执行的 CTDE VPP 下层策略")}</div>
        </div>
        <div class="arrow">→</div>
        <div class="framework-column train">
          <h3>{_lang("Training Signal", "训练信号")}</h3>
          <div class="paper-box">r_dso, r_dispatch_i, r_portfolio_i are separate general-sum rewards</div>
          <div class="paper-box">{_lang("centralized critic", "集中式 critic")}</div>
          <div class="paper-box">policy / value / entropy loss</div>
        </div>
      </div>
      <div class="agent-detail-panel" id="agent-detail">
        <p class="empty">{_lang("Select an agent module above.", "请点击上方一个智能体模块。")}</p>
      </div>
    </section>
    """


def _paper_style_workflow_diagram(agents: pd.DataFrame) -> str:
    def buttons(group_id: str) -> str:
        if agents.empty or "agent_group" not in agents:
            return ""
        return _agent_buttons(agents[agents["agent_group"].astype(str) == group_id])

    return f"""
    <section class="panel">
      <div class="section-head">
        <div>
          <p class="eyebrow">Framework</p>
          <h2>{_lang("Paper-Style Closed-Loop Agent Workflow", "论文式闭环智能体工作流总图")}</h2>
        </div>
        <p>{_lang(
            "Read this like a paper framework figure: solid arrows are execution data flow, dashed arrows are training feedback, and colors identify agent groups. Click an agent module to inspect its observation, action, reward and current implementation.",
            "请像阅读论文总图一样阅读本图：实线箭头表示执行期数据流，虚线箭头表示训练反馈，颜色表示智能体分组。点击任意智能体模块可查看观测、动作、奖励和当前实现。",
            block=True,
        )}</p>
      </div>
      <div class="paper-flow">
        <div class="flow-row">
          <div class="flow-node env-node">
            <span class="node-kicker">{_lang("Physical grid", "物理电网")}</span>
            <strong>pandapower net</strong>
            <p>{_lang("profiles, topology, voltage and loading constraints", "曲线、拓扑、电压与负载率约束")}</p>
          </div>
          <div class="flow-arrow solid">&#8594;<span>{_lang("state", "状态")}</span></div>
          <div class="flow-node vpp-node">
            <span class="node-kicker">{_lang("VPP reports", "VPP 上报")}</span>
            <strong>{_lang("Day-ahead bid / capability", "日前报量/报价与能力摘要")}</strong>
            <p>{_lang("P/Q feasible range, bid prices, confidence and DER state summary", "P/Q 可行域、报价、置信度和 DER 状态摘要")}</p>
          </div>
          <div class="flow-arrow solid">&#8594;<span>{_lang("privacy report", "隐私保护上报")}</span></div>
          <div class="flow-node dso-node">
            <span class="node-kicker">{_lang("Global guidance", "全局引导")}</span>
            <strong>DSO operating envelope</strong>
            <p>{_lang("preferred target, service request, price context and grid safety boundary", "推荐目标、服务请求、价格上下文和电网安全边界")}</p>
            <div class="mini-agent-grid">{buttons("global_guidance")}</div>
          </div>
        </div>
        <div class="flow-row">
          <div class="flow-node critic-node">
            <span class="node-kicker">{_lang("Centralized training", "集中式训练")}</span>
            <strong>{_lang("Central critic / value head", "集中式 critic / value head")}</strong>
            <p>{_lang("global state, reward components and trajectory log probabilities", "全局状态、reward 分量和轨迹 log probability")}</p>
          </div>
          <div class="flow-arrow dashed up">&#8593;<span>{_lang("training feedback", "训练反馈")}</span></div>
          <div class="flow-node vpp-node">
            <span class="node-kicker">{_lang("VPP lower policy", "VPP 下层策略")}</span>
            <strong>{_lang("Dispatch / disaggregation actors", "调度 / 解聚合智能体")}</strong>
            <p>{_lang("selected_p_mw + der_actions for PV, ESS, EVCS, HVAC, flexible load and MT", "selected_p_mw + 面向 PV、ESS、EVCS、HVAC、柔性负荷和 MT 的 der_actions")}</p>
            <div class="mini-agent-grid">{buttons("vpp_dispatch")}</div>
          </div>
          <div class="flow-arrow solid">&#8594;<span>{_lang("DER proposals", "DER 动作提案")}</span></div>
          <div class="flow-node safety-node">
            <span class="node-kicker">{_lang("Safety layer", "安全层")}</span>
            <strong>{_lang("FR/DOE projection + residual repair", "FR/DOE 投影 + 残差修复")}</strong>
            <p>{_lang("clip device bounds, repair delivery, write load/sgen/storage", "裁剪设备边界，修复交付误差，写入 load/sgen/storage")}</p>
          </div>
        </div>
        <div class="flow-row">
          <div class="flow-node portfolio-node">
            <span class="node-kicker">{_lang("Slow loop", "慢周期")}</span>
            <strong>{_lang("VPP portfolio agents", "VPP 组合配置智能体")}</strong>
            <p>{_lang("keep / reweight / propose membership change; trainable proxy head with physical membership gated by scenario events", "保持 / 重新加权 / 提出成员变更；当前为可训练代理 head，物理成员变更由场景事件门控")}</p>
            <div class="mini-agent-grid">{buttons("vpp_portfolio")}</div>
          </div>
          <div class="flow-arrow dashed">&#8593;<span>{_lang("long-horizon evidence", "长周期证据")}</span></div>
          <div class="flow-node trainer-node">
            <span class="node-kicker">{_lang("Experiment control", "实验控制")}</span>
            <strong>{_lang("Training supervisor", "训练监督智能体")}</strong>
            <p>{_lang("baseline trials, hyperparameter sweeps and convergence checks", "baseline 试验、超参数搜索和收敛检查")}</p>
            <div class="mini-agent-grid">{buttons("training_supervisor")}</div>
          </div>
          <div class="flow-arrow dashed loop">&#8634;<span>{_lang("policy update", "策略更新")}</span></div>
          <div class="flow-node reward-node">
            <span class="node-kicker">{_lang("Objective", "目标函数")}</span>
            <strong>{_lang("role-specific general-sum rewards", "角色分离 general-sum reward")}</strong>
            <p>{_lang("DSO uses grid reward; VPP dispatch uses self-interest settlement reward; portfolio uses localized DSO-alignment credit.", "DSO 使用电网 reward；VPP 调度使用自利结算 reward；组合配置使用局部化 DSO 对齐收益。")}</p>
          </div>
        </div>
      </div>
      <div class="agent-detail-panel" id="agent-detail">
        <p class="empty">{_lang("Select an agent module above.", "请选择上方一个智能体模块。")}</p>
      </div>
    </section>
    """


def _paper_style_diagram(frames: dict[str, pd.DataFrame]) -> str:
    return build_rl_architecture_diagram(
        frames,
        root_id="rl-architecture-workflow",
        heading_eyebrow_en="Framework",
        heading_eyebrow_zh="框架总图",
        title_en="Paper-Style Agent Workflow",
        title_zh="论文总图风格智能体工作流",
        description_en=(
            "This workflow emphasizes the operational arrow chain and the slow-loop portfolio branch: "
            "VPP day-ahead bid -> DSO operating envelope -> VPP dispatch actors -> DER actions -> "
            "safety projection -> pandapower -> reward/critic/training update."
        ),
        description_zh=(
            "该总图突出运行箭头链路和慢周期组合旁路：VPP 日前报量/报价 -> DSO 运行包络 -> "
            "VPP 调度 actor -> DER 动作 -> 安全投影 -> pandapower -> 奖励/critic/训练更新。"
        ),
        heading_class="section-head",
    )


def _workflow(workflow: pd.DataFrame) -> str:
    cards = []
    for _, row in workflow.sort_values("step_order").iterrows() if not workflow.empty else []:
        cards.append(
            f"""
            <article class="workflow-card" data-workflow-group="{escape(_text(row.get('agent_group')))}">
              <span class="step-num">{int(row.get('step_order', 0))}</span>
              <h3>{_lang(escape(_text(row.get('stage'))), escape(_text(row.get('stage_zh'))))}</h3>
              <dl>
                <dt>{_lang("Actor", "执行主体")}</dt><dd>{escape(_text(row.get('actor')))}</dd>
                <dt>{_lang("Input", "输入")}</dt><dd>{escape(_text(row.get('input')))}</dd>
                <dt>{_lang("Output", "输出")}</dt><dd>{escape(_text(row.get('output')))}</dd>
              </dl>
              <p>{_pick(row, "explanation", "explanation_zh")}</p>
            </article>
            """
        )
    return f"""
    <section class="panel">
      <div class="section-head">
        <div>
          <p class="eyebrow">Step Mechanics</p>
          <h2>{_lang("One Environment Step: Who Does What?", "一个 step 内智能体如何工作")}</h2>
        </div>
        <p>{_lang(
            "This timeline follows one simulator step from observation to action projection, power flow, reward and learning update.",
            "这条时间线展示一个仿真 step 中从观测、动作投影、潮流计算到 reward 与学习更新的完整方向。",
            block=True,
        )}</p>
      </div>
      <div class="workflow-grid">{''.join(cards)}</div>
    </section>
    """


def _relationship_cards(relations: pd.DataFrame) -> str:
    cards = []
    for _, row in relations.sort_values("relation_order").iterrows() if not relations.empty else []:
        cards.append(
            f"""
            <article class="relation-card">
              <div class="relation-line">
                <strong>{escape(_text(row.get('source')))}</strong>
                <span>→</span>
                <strong>{escape(_text(row.get('target')))}</strong>
              </div>
              <p class="signal">{_lang(escape(_text(row.get('message'))), escape(_text(row.get('message_zh'))), block=True)}</p>
              <p>{_pick(row, "meaning", "meaning_zh")}</p>
            </article>
            """
        )
    return f"""
    <section class="panel">
      <div class="section-head">
        <div>
          <p class="eyebrow">Relations</p>
          <h2>{_lang("Agent Relationship and Message Flow", "智能体关系与消息流")}</h2>
        </div>
      </div>
      <div class="relation-grid">{''.join(cards)}</div>
    </section>
    """


def _neural_network_section(neural: pd.DataFrame) -> str:
    if neural.empty:
        return ""
    cards = []
    for _, row in neural.iterrows():
        trainable = str(row.get("trainable", "")).lower() in {"true", "1", "yes"}
        cards.append(
            f"""
            <article class="loss-card">
              <div class="relation-line">
                <strong>{escape(_text(row.get('component_id')))}</strong>
                <span>→</span>
                <strong>{escape(_text(row.get('output_shape')))}</strong>
              </div>
              <p class="eyebrow">{escape(_text(row.get('component_group')))} · {_lang("trainable", "可训练") if trainable else _lang("fixed", "固定")}</p>
              <code>{_lang(escape(_text(row.get('structure'))), escape(_text(row.get('structure_zh'), _text(row.get('structure')))), block=True)}</code>
              <p><strong>{_lang("Input / output", "输入 / 输出")}:</strong> {escape(_text(row.get('input_shape')))} → {escape(_text(row.get('output_shape')))}</p>
              <p><strong>{_lang("Distribution", "分布")}:</strong> {_lang(escape(_text(row.get('distribution'))), escape(_text(row.get('distribution_zh'), _text(row.get('distribution')))), block=True)}</p>
              <p>{_lang(escape(_text(row.get('calculation_note'))), escape(_text(row.get('calculation_note_zh'), _text(row.get('calculation_note')))), block=True)}</p>
            </article>
            """
        )
    return f"""
    <section class="panel">
      <div class="section-head">
        <div><p class="eyebrow">Neural Network</p><h2>{_lang("Actual PyTorch Actor-Critic Structure", "实际 PyTorch Actor-Critic 神经网络结构")}</h2></div>
        <p>{_lang("This section expands phrases such as z_dso, Gaussian head and categorical head into concrete layers, dimensions and distributions used by deep_rl.py.", "本区块把 z_dso、高斯策略头、categorical head 等描述展开成 deep_rl.py 实际使用的层、维度和分布。", block=True)}</p>
      </div>
      <div class="loss-grid">{''.join(cards)}</div>
    </section>
    """


def _reward_and_loss(rewards: pd.DataFrame, losses: pd.DataFrame, gaps: pd.DataFrame) -> str:
    reward_cards = []
    for _, row in rewards.iterrows():
        reward_cards.append(
            f"""
            <article class="reward-card">
              <h3>{escape(_text(row.get('reward_id')))}</h3>
              <p class="formula">{_lang(escape(_text(row.get('formula'))), escape(_text(row.get('formula_zh'))), block=True)}</p>
              <p><strong>{_lang("Applies to", "适用对象")}:</strong> {_lang(escape(_text(row.get('applies_to'))), escape(_text(row.get('applies_to_zh'))), block=True)}</p>
              <p><strong>{_lang("Terms", "组成项")}:</strong> {_lang(escape(_text(row.get('terms'))), escape(_text(row.get('terms_zh'))), block=True)}</p>
              <span class="status-pill">{_lang(escape(_text(row.get('current_status'))), escape(_text(row.get('current_status_zh'))))}</span>
            </article>
            """
        )
    loss_cards = []
    for _, row in losses.iterrows():
        loss_cards.append(
            f"""
            <article class="loss-card">
              <h3>{escape(_text(row.get('component')))}</h3>
              <code>{escape(_text(row.get('formula')))}</code>
              <p>{_pick(row, "meaning", "meaning_zh")}</p>
              <span>{escape(_text(row.get('coefficient')))}</span>
            </article>
            """
        )
    gap_cards = []
    for _, row in gaps.iterrows():
        gap_cards.append(
            f"""
            <article class="gap-card">
              <h3>{_lang(escape(_text(row.get('question'))), escape(_text(row.get('question_zh'))), block=True)}</h3>
              <p><strong>{_lang("Current", "当前")}:</strong> {_pick(row, "current_answer", "current_answer_zh")}</p>
              <p><strong>{_lang("Target", "目标")}:</strong> {_pick(row, "target_answer", "target_answer_zh")}</p>
            </article>
            """
        )
    return f"""
    <section class="panel">
      <div class="section-head">
        <div>
          <p class="eyebrow">Reward</p>
          <h2>{_lang("Agent Rewards and Loss Functions", "智能体奖励函数与损失函数")}</h2>
        </div>
      </div>
      <h3>{_lang("Reward design", "奖励设计")}</h3>
      <div class="reward-grid">{''.join(reward_cards)}</div>
      <h3>{_lang("Actor-Critic loss", "Actor-Critic 损失函数")}</h3>
      <div class="loss-grid">{''.join(loss_cards)}</div>
      <h3>{_lang("Current implementation gaps", "当前实现差距")}</h3>
      <div class="gap-grid">{''.join(gap_cards)}</div>
    </section>
    """


def _ctde_section(ctde: pd.DataFrame, overview: pd.DataFrame) -> str:
    row = overview.iloc[0] if not overview.empty else pd.Series(dtype=object)
    cards = []
    for _, item in ctde.iterrows():
        cards.append(
            f"""
            <article class="ctde-card">
              <h3>{_lang(escape(_text(item.get('question'))), escape(_text(item.get('question_zh'))), block=True)}</h3>
              <p class="answer">{_lang(escape(_text(item.get('answer'))), escape(_text(item.get('answer_zh'))))}</p>
              <p>{_pick(item, "evidence", "evidence_zh")}</p>
            </article>
            """
        )
    return f"""
    <section class="panel">
      <div class="section-head">
        <div>
          <p class="eyebrow">CTDE</p>
          <h2>{_lang("Is the Current Model CTDE?", "当前模型是不是 CTDE？")}</h2>
        </div>
      </div>
      <div class="current-target">
        <article>
          <h3>{_lang("Current implementation", "当前实现")}</h3>
          <p>{_pick(row, "plain_language_summary", "plain_language_summary_zh")}</p>
        </article>
        <article>
          <h3>{_lang("Target architecture", "目标架构")}</h3>
          <p>{_pick(row, "current_vs_target", "current_vs_target_zh")}</p>
        </article>
      </div>
      <div class="ctde-grid">{''.join(cards)}</div>
    </section>
    """


def _model_update_summary(updates: pd.DataFrame) -> str:
    if updates.empty:
        return ""
    cards: list[str] = []
    ordered = updates.sort_values("display_order") if "display_order" in updates else updates
    for _, row in ordered.iterrows():
        cards.append(
            f"""
            <article class="summary-card">
              <span>{escape(_text(row.get("update_area"), "update"))}</span>
              <strong>{_lang(
                  escape(_text(row.get("current_value"), "n/a")),
                  escape(_text(row.get("current_value_zh"), _text(row.get("current_value"), "n/a"))),
                  block=True,
              )}</strong>
              <p>{_lang(
                  escape(_text(row.get("explanation"), "")),
                  escape(_text(row.get("explanation_zh"), _text(row.get("explanation"), ""))),
                  block=True,
              )}</p>
              <p><code>{escape(_text(row.get("evidence_file"), ""))}</code></p>
            </article>
            """
        )
    return f"""
    <section class="panel">
      <div class="section-head">
        <div><p class="eyebrow">Model Sync</p><h2>{_lang("Model / Algorithm Update Summary", "模型 / 算法更新摘要")}</h2></div>
        <p>{_lang(
            "The architecture page, main report, first-person pages and Dash data now share the same model update table.",
            "架构页、主报告、第一视角页面和 Dash 数据现在共享同一张模型更新表。",
            block=True,
        )}</p>
      </div>
      <div class="summary-grid">{''.join(cards)}</div>
    </section>
    """


def _algorithm_capabilities(capabilities: pd.DataFrame) -> str:
    if capabilities.empty:
        return ""
    cards: list[str] = []
    for _, row in capabilities.iterrows():
        cards.append(
            f"""
            <article class="loss-card">
              <h3>{_lang(escape(_text(row.get('algorithm'))), escape(_text(row.get('algorithm_zh'), _text(row.get('algorithm')))))}</h3>
              <p><strong>{_lang("Implemented mechanisms", "已实现机制")}:</strong> {_lang(
                  escape(_text(row.get('implemented_mechanisms'))),
                  escape(_text(row.get('implemented_mechanisms_zh'), _text(row.get('implemented_mechanisms')))),
                  block=True,
              )}</p>
              <p><strong>{_lang("Critic heads", "Critic 输出头")}:</strong> {_lang(
                  escape(_text(row.get('critic_heads'))),
                  escape(_text(row.get('critic_heads_zh'), _text(row.get('critic_heads')))),
                  block=True,
              )}</p>
              <p><strong>{_lang("Boundary", "边界说明")}:</strong> {_lang(
                  escape(_text(row.get('claim_boundary'))),
                  escape(_text(row.get('claim_boundary_zh'), _text(row.get('claim_boundary')))),
                  block=True,
              )}</p>
            </article>
            """
        )
    return f"""
    <section class="panel">
      <div class="section-head">
        <div>
          <p class="eyebrow">Algorithms</p>
          <h2>{_lang("Advanced MARL Algorithm Completeness", "高级 MARL 算法完成度")}</h2>
        </div>
        <p>{_lang(
            "This section is generated from the algorithm metadata, so future MATD3/HAPPO/HASAC changes can be reflected in the report.",
            "本区块由算法元数据生成，因此后续 MATD3/HAPPO/HASAC 变更可以同步反映到报告中。",
            block=True,
        )}</p>
      </div>
      <div class="loss-grid">{''.join(cards)}</div>
    </section>
    """


def _tables(
    agents: pd.DataFrame,
    variants: pd.DataFrame,
    neural: pd.DataFrame,
    target_ctde: pd.DataFrame,
    ctde_nodes: pd.DataFrame,
    ctde_edges: pd.DataFrame,
    ctde_feedback: pd.DataFrame,
    workflow: pd.DataFrame,
    rewards: pd.DataFrame,
    relations: pd.DataFrame,
    *,
    show_shared_benchmark: bool = True,
) -> str:
    def table(frame: pd.DataFrame) -> str:
        if frame.empty:
            return "<p class='empty'>No records.</p>"
        headers = "".join(f"<th>{escape(str(col))}</th>" for col in frame.columns)
        rows = []
        for values in frame.head(120).itertuples(index=False, name=None):
            rows.append("<tr>" + "".join(f"<td>{escape(str(value))}</td>" for value in values) + "</tr>")
        return f"<div class='table-wrap'><table><thead><tr>{headers}</tr></thead><tbody>{''.join(rows)}</tbody></table></div>"

    neural_table = f"<h3>rl_neural_network_architecture</h3>{table(neural)}" if show_shared_benchmark else ""
    return f"""
    <section class="panel">
      <details>
        <summary>{_lang("Raw architecture tables", "原始架构表")}</summary>
        <h3>rl_agent_architecture</h3>{table(agents)}
        <h3>rl_algorithm_variants</h3>{table(variants)}
        {neural_table}
        <h3>rl_target_ctde_architecture</h3>{table(target_ctde)}
        <h3>rl_ctde_nodes</h3>{table(ctde_nodes)}
        <h3>rl_ctde_edges</h3>{table(ctde_edges)}
        <h3>rl_ctde_feedback</h3>{table(ctde_feedback)}
        <h3>rl_step_workflow</h3>{table(workflow)}
        <h3>rl_reward_design</h3>{table(rewards)}
        <h3>rl_agent_relationships</h3>{table(relations)}
      </details>
    </section>
    """


def _script() -> str:
    return """
    (function() {
      const agents = JSON.parse(document.getElementById('agent-data').textContent);
      const byId = new Map(agents.map(item => [String(item.agent_id), item]));
      const detail = document.getElementById('agent-detail');
      const langKey = 'vpp-rl-architecture-lang';

      function activeLang() {
        return document.documentElement.getAttribute('data-lang') || 'zh';
      }
      function pick(item, enKey, zhKey) {
        const lang = activeLang();
        return String((lang === 'zh' ? item[zhKey || enKey + '_zh'] : item[enKey]) || item[enKey] || '');
      }
      function htmlEscape(value) {
        return String(value).replace(/[&<>"']/g, ch => ({'&':'&amp;', '<':'&lt;', '>':'&gt;', '"':'&quot;', "'":'&#39;'}[ch]));
      }
      function renderAgent(agentId) {
        const item = byId.get(String(agentId));
        if (!item || !detail) return;
        const isRl = String(item.is_rl_decision).toLowerCase() === 'true';
        const rlBadge = activeLang() === 'zh'
          ? (isRl ? '使用 RL 决策' : '非 RL 环境智能体')
          : (isRl ? 'Uses RL' : 'Not an RL env agent');
        detail.innerHTML = `
          <div class="detail-head" style="--agent-color:${htmlEscape(item.agent_group_color || '#64748b')}">
            <div>
              <p class="eyebrow">${htmlEscape(pick(item, 'agent_group_label', 'agent_group_label_zh'))}</p>
              <h3>${htmlEscape(item.agent_id)}</h3>
            </div>
            <div class="detail-status-stack">
              <span class="status-pill">${htmlEscape(rlBadge)}</span>
              <span class="status-pill">${htmlEscape(pick(item, 'implementation_status', 'implementation_status_zh'))}</span>
            </div>
          </div>
          <div class="detail-grid">
            <section><h4>${activeLang() === 'zh' ? '输入观测' : 'Input observation'}</h4><p>${htmlEscape(pick(item, 'input_observation', 'input_observation_zh'))}</p></section>
            <section><h4>${activeLang() === 'zh' ? '动作 / 输出' : 'Action / output'}</h4><p>${htmlEscape(pick(item, 'action_output', 'action_output_zh'))}</p></section>
            <section><h4>${activeLang() === 'zh' ? '是否使用强化学习' : 'Uses RL?'}</h4><p>${htmlEscape(pick(item, 'rl_usage_status', 'rl_usage_status_zh'))}</p></section>
            <section><h4>${activeLang() === 'zh' ? '神经网络结构' : 'Neural network structure'}</h4><p>${htmlEscape(pick(item, 'neural_network_structure', 'neural_network_structure_zh'))}</p></section>
            <section><h4>${activeLang() === 'zh' ? '输出公式' : 'Output formula'}</h4><p>${htmlEscape(pick(item, 'result_formula', 'result_formula_zh'))}</p></section>
            <section><h4>${activeLang() === 'zh' ? '结果怎么算' : 'Result calculation'}</h4><p>${htmlEscape(pick(item, 'result_calculation', 'result_calculation_zh'))}</p></section>
            <section><h4>${activeLang() === 'zh' ? '结果来源' : 'Result source'}</h4><p>${htmlEscape(pick(item, 'result_source', 'result_source_zh'))}</p></section>
            <section><h4>${activeLang() === 'zh' ? '策略模块' : 'Policy module'}</h4><p>${htmlEscape(pick(item, 'policy_module', 'policy_module_zh'))}</p></section>
            <section><h4>${activeLang() === 'zh' ? '训练信号' : 'Training signal'}</h4><p>${htmlEscape(pick(item, 'rl_training_signal', 'rl_training_signal_zh'))}</p></section>
            <section><h4>${activeLang() === 'zh' ? '奖励函数' : 'Reward function'}</h4><p>${htmlEscape(pick(item, 'reward_function', 'reward_function_zh'))}</p></section>
            <section><h4>${activeLang() === 'zh' ? '审计输出' : 'Audit outputs'}</h4><p>${htmlEscape(pick(item, 'audit_outputs', 'audit_outputs_zh'))}</p></section>
            <section><h4>${activeLang() === 'zh' ? '非 RL 安全门' : 'Non-RL guardrails'}</h4><p>${htmlEscape(pick(item, 'non_rl_guardrails', 'non_rl_guardrails_zh'))}</p></section>
            <section><h4>${activeLang() === 'zh' ? '一个 step 中做什么' : 'Role in one step'}</h4><p>${htmlEscape(pick(item, 'current_step_role', 'current_step_role_zh'))}</p></section>
            <section><h4>${activeLang() === 'zh' ? '下一步升级' : 'Next upgrade'}</h4><p>${htmlEscape(pick(item, 'next_upgrade', 'next_upgrade_zh'))}</p></section>
          </div>`;
        document.querySelectorAll('.agent-node').forEach(node => node.classList.toggle('is-selected', node.dataset.agentId === String(agentId)));
      }
      function applyLang(lang) {
        document.documentElement.setAttribute('data-lang', lang);
        document.documentElement.setAttribute('lang', lang === 'zh' ? 'zh-CN' : 'en');
        document.querySelectorAll('[data-lang-switch]').forEach(btn => {
          const active = btn.getAttribute('data-lang-switch') === lang;
          btn.classList.toggle('is-active', active);
          btn.setAttribute('aria-pressed', active ? 'true' : 'false');
        });
        const selected = document.querySelector('.agent-node.is-selected');
        if (selected) renderAgent(selected.dataset.agentId);
        try { localStorage.setItem(langKey, lang); } catch (err) { void err; }
      }
      document.addEventListener('click', event => {
        const agent = event.target.closest('[data-agent-id]');
        if (agent) renderAgent(agent.dataset.agentId);
        const langButton = event.target.closest('[data-lang-switch]');
        if (langButton) applyLang(langButton.getAttribute('data-lang-switch') || 'zh');
      });
      let initial = 'zh';
      try { initial = localStorage.getItem(langKey) || initial; } catch (err) { void err; }
      applyLang(initial);
      const first = document.querySelector('[data-agent-id]');
      if (first) renderAgent(first.dataset.agentId);
    })();
    """


def _script() -> str:
    return """
    (function() {
      const langKey = 'vpp-rl-architecture-lang';
      function applyLang(lang) {
        document.documentElement.setAttribute('data-lang', lang);
        document.documentElement.setAttribute('lang', lang === 'zh' ? 'zh-CN' : 'en');
        document.querySelectorAll('[data-lang-switch]').forEach(btn => {
          const active = btn.getAttribute('data-lang-switch') === lang;
          btn.classList.toggle('is-active', active);
          btn.setAttribute('aria-pressed', active ? 'true' : 'false');
        });
        try { localStorage.setItem(langKey, lang); } catch (err) { void err; }
      }
      document.addEventListener('click', event => {
        const langButton = event.target.closest('[data-lang-switch]');
        if (langButton) applyLang(langButton.getAttribute('data-lang-switch') || 'zh');
      });
      let initial = 'zh';
      try { initial = localStorage.getItem(langKey) || initial; } catch (err) { void err; }
      applyLang(initial);
    })();
    """


def build_rl_architecture_report_html(frames: dict[str, pd.DataFrame], output_path: str | Path) -> Path:
    path = Path(output_path)
    frames = _load_optional_deep_rl_frames(frames, path)
    primary_ctde = _primary_ctde_active(frames)
    overview = frame_or_empty(frames, "rl_algorithm_overview")
    groups = frame_or_empty(frames, "rl_agent_groups")
    agents = frame_or_empty(frames, "rl_agent_architecture")
    variants = frame_or_empty(frames, "rl_algorithm_variants")
    neural = frame_or_empty(frames, "rl_neural_network_architecture")
    target_ctde = frame_or_empty(frames, "rl_target_ctde_architecture")
    ctde_nodes = frame_or_empty(frames, "rl_ctde_nodes")
    ctde_edges = frame_or_empty(frames, "rl_ctde_edges")
    ctde_feedback = frame_or_empty(frames, "rl_ctde_feedback")
    workflow = frame_or_empty(frames, "rl_step_workflow")
    relations = frame_or_empty(frames, "rl_agent_relationships")
    rewards = frame_or_empty(frames, "rl_reward_design")
    losses = frame_or_empty(frames, "rl_loss_components")
    ctde = frame_or_empty(frames, "rl_ctde_assessment")
    gaps = frame_or_empty(frames, "rl_implementation_gaps")
    capabilities = frame_or_empty(frames, "rl_algorithm_capabilities")
    model_updates = frame_or_empty(frames, "model_update_summary")

    header_description = (
        (
            "This standalone report explains the current privacy-separated CTDE learning stack: DSO global actor, "
            "VPP dispatch actors, VPP portfolio actors, centralized training critic, safety projection, rewards and losses."
        ),
        (
            "这个独立页面解释当前隐私分离 CTDE 学习栈：DSO 全局 actor、VPP 调度 actor、VPP 组合配置 actor、训练期集中 critic、安全投影、奖励与损失。"
        ),
    ) if primary_ctde else (
        (
            "This standalone report explains the current learning stack, agent groups, one-step data flow, reward/loss definitions, CTDE status and the transition from shared centralized DER-level action heads toward decentralized CTDE VPP lower policies."
        ),
        (
            "这个独立页面解释当前学习栈、智能体分组、一个 step 内的数据流、奖励/损失定义、CTDE 状态，以及从共享集中式 DER 级动作 head 迈向分散执行 CTDE VPP 下层策略的路径。"
        ),
    )

    html = f"""<!doctype html>
<html lang="zh-CN" data-lang="zh">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>RL/MARL Architecture Report</title>
  <style>
    :root {{
      --bg: #eef4f8;
      --panel: #ffffff;
      --line: #d8e3ee;
      --text: #132235;
      --muted: #5d7084;
      --accent: #1769aa;
      --shadow: 0 14px 32px rgba(15, 35, 56, 0.08);
    }}
    * {{ box-sizing: border-box; }}
    body {{ margin: 0; font-family: "Segoe UI", "Microsoft YaHei", Arial, sans-serif; color: var(--text); background: linear-gradient(180deg, #dfeaf3, var(--bg) 36%, #f7fafc); }}
    .lang-copy {{ display: none; }}
    html[data-lang="en"] .lang-en.lang-inline, html[data-lang="zh"] .lang-zh.lang-inline {{ display: inline; }}
    html[data-lang="en"] .lang-en.lang-block, html[data-lang="zh"] .lang-zh.lang-block {{ display: block; }}
    header {{ padding: 28px 34px; color: white; background: linear-gradient(135deg, #061b30, #13456f); }}
    header h1 {{ margin: 6px 0 10px; font-size: clamp(28px, 4vw, 42px); }}
    header p {{ max-width: 980px; margin: 0; color: #dce9f5; line-height: 1.65; }}
    main {{ padding: 24px 30px 42px; }}
    .language-toolbar {{ display: inline-flex; align-items: center; gap: 8px; padding: 6px; margin-top: 18px; border-radius: 999px; border: 1px solid rgba(255,255,255,0.18); background: rgba(255,255,255,0.09); }}
    .language-toolbar span {{ color: #dce9f5; font-size: 12px; padding: 0 6px; }}
    .lang-button {{ border: 0; border-radius: 999px; padding: 7px 12px; background: transparent; color: #dce9f5; font-weight: 800; cursor: pointer; }}
    .lang-button.is-active {{ color: white; background: rgba(143,208,255,0.24); }}
    .panel {{ background: var(--panel); border: 1px solid var(--line); border-radius: 18px; padding: 18px 20px; margin-bottom: 18px; box-shadow: var(--shadow); }}
    .section-head {{ display: flex; justify-content: space-between; gap: 18px; align-items: end; margin-bottom: 15px; }}
    .section-head p {{ max-width: 640px; margin: 0; color: var(--muted); line-height: 1.6; }}
    .eyebrow {{ margin: 0; color: var(--accent); font-size: 11px; font-weight: 900; letter-spacing: 0.12em; text-transform: uppercase; }}
    h2 {{ margin: 3px 0 0; font-size: 24px; }}
    h3 {{ margin: 0 0 8px; font-size: 16px; }}
    .summary-grid, .group-grid, .reward-grid, .loss-grid, .gap-grid, .ctde-grid, .relation-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(240px, 1fr)); gap: 12px; }}
    .summary-card, .group-card, .reward-card, .loss-card, .gap-card, .ctde-card, .relation-card {{ border: 1px solid var(--line); border-radius: 15px; padding: 14px; background: #fff; min-width: 0; }}
    .summary-card span {{ display: block; color: var(--muted); font-size: 12px; font-weight: 900; margin-bottom: 6px; }}
    .summary-card strong {{ display: block; overflow-wrap: anywhere; line-height: 1.4; }}
    .group-card {{ border-left: 7px solid var(--group-color); position: relative; }}
    .group-card p, .reward-card p, .loss-card p, .gap-card p, .ctde-card p, .relation-card p {{ color: #30475d; line-height: 1.62; margin: 8px 0 0; overflow-wrap: anywhere; }}
    .color-dot {{ width: 13px; height: 13px; border-radius: 999px; background: var(--group-color); margin-bottom: 9px; }}
    .count-pill, .status-pill {{ display: inline-flex; margin-top: 10px; padding: 6px 9px; border-radius: 999px; background: #eaf4fc; color: #10517f; font-size: 12px; font-weight: 900; }}
    .paper-flow {{ display: grid; gap: 16px; }}
    .flow-row {{ display: grid; grid-template-columns: minmax(210px, 1fr) minmax(82px, 0.28fr) minmax(280px, 1.3fr) minmax(82px, 0.28fr) minmax(280px, 1.3fr); gap: 12px; align-items: stretch; }}
    .flow-node {{ border: 2px solid var(--line); border-radius: 18px; padding: 15px; background: #fff; box-shadow: 0 8px 18px rgba(15,35,56,0.06); min-width: 0; }}
    .flow-node strong {{ display: block; margin: 4px 0 8px; font-size: 17px; line-height: 1.35; overflow-wrap: anywhere; }}
    .flow-node p {{ margin: 0; color: #40566d; line-height: 1.58; overflow-wrap: anywhere; }}
    .node-kicker {{ display: inline-flex; padding: 5px 8px; border-radius: 999px; color: white; background: #64748b; font-size: 11px; font-weight: 900; }}
    .dso-node {{ border-color: #1f77b4; background: #f4f9ff; }}
    .dso-node .node-kicker {{ background: #1f77b4; }}
    .vpp-node {{ border-color: #2ca02c; background: #f6fff6; }}
    .vpp-node .node-kicker {{ background: #2ca02c; }}
    .portfolio-node {{ border-color: #9467bd; background: #fbf7ff; }}
    .portfolio-node .node-kicker {{ background: #9467bd; }}
    .trainer-node, .critic-node, .reward-node {{ border-color: #ff7f0e; background: #fff8ef; }}
    .trainer-node .node-kicker, .critic-node .node-kicker, .reward-node .node-kicker {{ background: #d86400; }}
    .env-node, .safety-node {{ border-color: #52677d; background: #f8fbfd; }}
    .env-node .node-kicker, .safety-node .node-kicker {{ background: #52677d; }}
    .flow-arrow {{ display: flex; flex-direction: column; align-items: center; justify-content: center; min-height: 92px; color: var(--accent); font-size: 34px; font-weight: 900; line-height: 1; }}
    .flow-arrow span {{ margin-top: 8px; font-size: 11px; line-height: 1.35; color: #41566c; text-align: center; font-weight: 900; }}
    .flow-arrow.dashed {{ color: #d86400; }}
    .flow-arrow.dashed::before {{ content: ""; width: 3px; height: 28px; border-left: 3px dashed #d86400; margin-bottom: 6px; }}
    .flow-arrow.solid::before {{ content: ""; width: 54px; border-top: 3px solid var(--accent); margin-bottom: 7px; }}
    .flow-arrow.loop {{ color: #8a5a00; }}
    .mini-agent-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 8px; margin-top: 12px; }}
    .mini-agent-grid .agent-node {{ padding: 9px; border-width: 1.5px; border-radius: 12px; }}
    .mini-agent-grid .agent-node em {{ display: none; }}
    .framework-shell {{ display: grid; grid-template-columns: minmax(170px, 0.9fr) auto minmax(300px, 1.7fr) auto minmax(190px, 1fr) auto minmax(190px, 1fr); gap: 12px; align-items: stretch; }}
    .framework-column {{ border: 1px solid var(--line); border-radius: 18px; padding: 14px; background: #f9fcfe; }}
    .framework-column h3 {{ text-align: center; color: #17324c; }}
    .paper-box {{ border: 1px solid #d8e5ef; border-radius: 14px; padding: 12px; margin: 10px 0; text-align: center; background: white; font-weight: 800; line-height: 1.45; }}
    .paper-box.current {{ background: #fff8e9; border-color: #ead3a2; }}
    .paper-box.target {{ background: #edf9ee; border-color: #b9dcb9; }}
    .arrow {{ display: flex; align-items: center; justify-content: center; color: var(--accent); font-size: 26px; font-weight: 900; }}
    .agent-node-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(190px, 1fr)); gap: 10px; }}
    .agent-node {{ border: 2px solid var(--agent-color); background: white; border-radius: 15px; padding: 12px; text-align: left; cursor: pointer; box-shadow: 0 6px 16px rgba(15,23,42,0.05); }}
    .agent-node.is-selected {{ outline: 3px solid color-mix(in srgb, var(--agent-color) 45%, transparent); }}
    .agent-node span {{ display: block; color: var(--agent-color); font-size: 12px; font-weight: 900; margin-bottom: 6px; }}
    .agent-node strong {{ display: block; color: #14263a; overflow-wrap: anywhere; }}
    .agent-node em {{ display: block; margin-top: 7px; color: var(--muted); font-size: 12px; line-height: 1.4; font-style: normal; }}
    .agent-detail-panel {{ margin-top: 14px; border: 1px solid var(--line); border-radius: 16px; padding: 15px; background: #f8fbfd; }}
    .detail-head {{ display: flex; justify-content: space-between; gap: 12px; border-left: 7px solid var(--agent-color); padding-left: 12px; margin-bottom: 12px; }}
    .detail-head h3 {{ margin: 2px 0 0; font-size: 21px; }}
    .detail-status-stack {{ display: flex; flex-wrap: wrap; justify-content: flex-end; gap: 7px; align-content: start; }}
    .detail-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(260px, 1fr)); gap: 10px; }}
    .detail-grid section {{ border: 1px solid #e3edf5; border-radius: 13px; padding: 12px; background: white; }}
    .detail-grid h4 {{ margin: 0 0 7px; font-size: 12px; color: var(--muted); text-transform: uppercase; letter-spacing: 0.06em; }}
    .detail-grid p {{ margin: 0; color: #2f455c; line-height: 1.6; overflow-wrap: anywhere; }}
    .workflow-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(260px, 1fr)); gap: 12px; }}
    .workflow-card {{ position: relative; border: 1px solid var(--line); border-radius: 15px; padding: 15px; background: white; }}
    .step-num {{ position: absolute; right: 12px; top: 10px; display: inline-flex; width: 30px; height: 30px; border-radius: 999px; align-items: center; justify-content: center; background: #eaf4fc; color: #0f4f85; font-weight: 900; }}
    dl {{ margin: 10px 0; display: grid; grid-template-columns: 75px 1fr; gap: 6px 8px; }}
    dt {{ color: var(--muted); font-weight: 900; }}
    dd {{ margin: 0; overflow-wrap: anywhere; }}
    .relation-line {{ display: grid; grid-template-columns: 1fr auto 1fr; gap: 8px; align-items: center; }}
    .relation-line span {{ color: var(--accent); font-size: 20px; font-weight: 900; text-align: center; }}
    .signal, .formula, .answer {{ font-weight: 900; color: #0f4f85 !important; }}
    code {{ display: block; border: 1px solid #d8e5ef; border-radius: 10px; background: #f5f9fc; padding: 9px; overflow-wrap: anywhere; white-space: pre-wrap; line-height: 1.5; }}
    .current-target {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 12px; margin-bottom: 12px; }}
    .current-target article {{ border: 1px solid var(--line); border-radius: 15px; padding: 14px; background: #f9fcfe; }}
    details summary {{ cursor: pointer; font-weight: 900; color: #17324c; }}
    .table-wrap {{ overflow-x: auto; border: 1px solid var(--line); border-radius: 12px; margin: 8px 0 15px; }}
    table {{ border-collapse: collapse; width: 100%; font-size: 12px; }}
    th, td {{ border-bottom: 1px solid #e8eef4; padding: 8px 9px; text-align: left; vertical-align: top; }}
    th {{ background: #f3f8fc; color: #33465a; }}
    .empty {{ color: var(--muted); }}
    {rl_architecture_diagram_css()}
    {neural_network_diagram_css()}
    {rl_algorithm_variant_section_css()}
    @media (max-width: 1100px) {{ .framework-shell, .flow-row {{ grid-template-columns: 1fr; }} .arrow {{ transform: rotate(90deg); }} .flow-arrow {{ min-height: 54px; }} .flow-arrow.solid::before {{ height: 28px; width: 3px; border-top: 0; border-left: 3px solid var(--accent); }} .section-head {{ flex-direction: column; align-items: flex-start; }} }}
    @media (max-width: 640px) {{ main {{ padding: 16px 12px 28px; }} header {{ padding: 22px 16px; }} .panel {{ padding: 14px; }} }}
  </style>
</head>
<body>
  <header>
    <p class="eyebrow">RL / MARL Architecture</p>
    <h1>{_lang("Interactive Agent Framework Report", "强化学习智能体交互框架报告")}</h1>
    <p>{_lang(header_description[0], header_description[1], block=True)}</p>
    {_language_toolbar()}
  </header>
  <main>
    <section class="panel">
      <div class="section-head">
        <div><p class="eyebrow">Overview</p><h2>{_lang("Current Algorithm Snapshot", "当前算法快照")}</h2></div>
      </div>
      {_summary_cards(overview)}
    </section>
    {_model_update_summary(model_updates)}
    {_algorithm_capabilities(capabilities)}
    {build_rl_algorithm_variant_section(
        variants,
        title_en="HAPPO / MATD3 / HASAC Architecture Contrast",
        title_zh="HAPPO / MATD3 / HASAC 架构对照",
    )}
    <section class="panel">
      <div class="section-head">
        <div><p class="eyebrow">Groups</p><h2>{_lang("Agent Groups and Colors", "智能体分组与颜色")}</h2></div>
        <p>{_lang("Agents of the same class share one color and are listed together in the framework diagram.", "同一类智能体使用统一颜色，并在框架图中按组展示。", block=True)}</p>
      </div>
      {_group_legend(groups, agents)}
    </section>
    {_paper_style_diagram(frames)}
    {build_target_ctde_architecture_diagram(
        frames,
        root_id="standalone-target-ctde-architecture",
        heading_class="section-head",
    )}
    {build_neural_network_architecture_diagram(
        frames,
        root_id="standalone-neural-network-architecture",
        heading_class="section-head",
    )}
    {_workflow(workflow)}
    {_relationship_cards(relations)}
    {_reward_and_loss(rewards, losses, gaps)}
    {_ctde_section(ctde, overview)}
    {_tables(
        agents,
        variants,
        neural,
        target_ctde,
        ctde_nodes,
        ctde_edges,
        ctde_feedback,
        workflow,
        rewards,
        relations,
        show_shared_benchmark=not primary_ctde,
    )}
  </main>
  <script>{_script()}</script>
</body>
</html>"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(html, encoding="utf-8")
    return path
