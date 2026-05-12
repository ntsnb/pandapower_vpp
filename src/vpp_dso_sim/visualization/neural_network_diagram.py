from __future__ import annotations

from html import escape

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


def _row(frame: pd.DataFrame, component_id: str) -> pd.Series:
    if frame.empty or "component_id" not in frame.columns:
        return pd.Series(dtype=object)
    data = frame[frame["component_id"].astype(str) == component_id]
    return data.iloc[0] if not data.empty else pd.Series(dtype=object)


def _truthy(value: object) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _current_ctde_is_primary(frames: dict[str, pd.DataFrame]) -> bool:
    overview = frame_or_empty(frames, "rl_algorithm_overview")
    deep = frame_or_empty(frames, "deep_rl_training_summary")
    algorithm = _text(deep.iloc[0].get("algorithm")) if not deep.empty else ""
    primary = _text(deep.iloc[0].get("target_ctde_primary_trainer")) if not deep.empty else ""
    status = _text(overview.iloc[0].get("target_ctde_status")) if not overview.empty else ""
    return (
        algorithm == "privacy_separated_ctde_actor_critic"
        or _truthy(primary)
        or status == "implemented_as_primary_privacy_separated_trainer"
        or status == "advanced_hierarchical_happo_implemented_as_recommended_trainer"
    )


def _svg_lang(en: str, zh: str, x: int, y: int, *, css_class: str = "nn-text", anchor: str = "middle") -> str:
    safe_en = escape(en)
    safe_zh = escape(zh)
    return (
        f"<text class='{css_class} nn-svg-lang nn-svg-en' x='{x}' y='{y}' text-anchor='{anchor}'>{safe_en}</text>"
        f"<text class='{css_class} nn-svg-lang nn-svg-zh' x='{x}' y='{y}' text-anchor='{anchor}'>{safe_zh}</text>"
    )


def _box(
    *,
    x: int,
    y: int,
    w: int,
    h: int,
    cls: str,
    title_en: str,
    title_zh: str,
    shape: str,
    sub_en: str = "",
    sub_zh: str = "",
    node_id: str = "",
    test_id: str = "",
) -> str:
    title_y = y + 31
    shape_y = y + 59
    sub_y = y + 83
    sub = _svg_lang(sub_en, sub_zh, x + w // 2, sub_y, css_class="nn-sub") if sub_en or sub_zh else ""
    id_attr = f' id="{escape(node_id)}"' if node_id else ""
    test_attr = f' data-testid="{escape(test_id)}"' if test_id else ""
    return f"""
    <g{id_attr}{test_attr} class="nn-block {escape(cls)}" tabindex="0">
      <rect x="{x}" y="{y}" width="{w}" height="{h}" rx="13" />
      {_svg_lang(title_en, title_zh, x + w // 2, title_y, css_class="nn-title")}
      <text class="nn-shape" x="{x + w // 2}" y="{shape_y}" text-anchor="middle">{escape(shape)}</text>
      {sub}
    </g>
    """


def _arrow(x1: int, y1: int, x2: int, y2: int, *, cls: str = "nn-arrow", test_id: str = "") -> str:
    test_attr = f' data-testid="{escape(test_id)}"' if test_id else ""
    return (
        f'<path{test_attr} class="{escape(cls)}" '
        f'd="M{x1},{y1} C{(x1+x2)//2},{y1} {(x1+x2)//2},{y2} {x2},{y2}" '
        'marker-end="url(#nn-arrowhead)" />'
    )


def _distribution_node(
    *,
    cx: int,
    cy: int,
    label_en: str,
    label_zh: str,
    shape: str,
    cls: str,
    node_id: str = "",
    test_id: str = "",
) -> str:
    id_attr = f' id="{escape(node_id)}"' if node_id else ""
    test_attr = f' data-testid="{escape(test_id)}"' if test_id else ""
    return f"""
    <g{id_attr}{test_attr} class="nn-dist {escape(cls)}" tabindex="0">
      <circle cx="{cx}" cy="{cy}" r="44" />
      {_svg_lang(label_en, label_zh, cx, cy - 7, css_class="nn-title")}
      <text class="nn-shape" x="{cx}" y="{cy + 18}" text-anchor="middle">{escape(shape)}</text>
    </g>
    """


def _action_node(
    *,
    x: int,
    y: int,
    w: int,
    h: int,
    cls: str,
    label_en: str,
    label_zh: str,
    shape: str,
    node_id: str = "",
    test_id: str = "",
) -> str:
    id_attr = f' id="{escape(node_id)}"' if node_id else ""
    test_attr = f' data-testid="{escape(test_id)}"' if test_id else ""
    return f"""
    <g{id_attr}{test_attr} class="nn-action {escape(cls)}" tabindex="0">
      <rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{h // 2}" />
      {_svg_lang(label_en, label_zh, x + w // 2, y + 27, css_class="nn-title")}
      <text class="nn-shape" x="{x + w // 2}" y="{y + 51}" text-anchor="middle">{escape(shape)}</text>
    </g>
    """


def _summary_value(frames: dict[str, pd.DataFrame], column: str, default: str, *aliases: str) -> str:
    deep = frame_or_empty(frames, "deep_rl_training_summary")
    if deep.empty:
        return default
    for name in (column, *aliases):
        if name in deep.columns:
            return _text(deep.iloc[0].get(name), default)
    return default


def _count_vpps(frames: dict[str, pd.DataFrame]) -> int:
    roles = frame_or_empty(frames, "agent_role_map")
    if not roles.empty and "role_type" in roles.columns:
        count = int((roles["role_type"].astype(str) == "vpp_dispatch_agent").sum())
        if count > 0:
            return count
    assets = frame_or_empty(frames, "asset_registry")
    if not assets.empty and "vpp_id" in assets.columns:
        return max(1, int(assets["vpp_id"].nunique()))
    return 1


def _layer_node(
    *,
    x: int,
    y: int,
    w: int,
    cls: str,
    title_en: str,
    title_zh: str,
    shape: str,
    sub_en: str = "",
    sub_zh: str = "",
    node_id: str = "",
    test_id: str = "",
) -> str:
    return _box(
        x=x,
        y=y,
        w=w,
        h=84,
        cls=cls,
        title_en=title_en,
        title_zh=title_zh,
        shape=shape,
        sub_en=sub_en,
        sub_zh=sub_zh,
        node_id=node_id,
        test_id=test_id,
    )


def _straight_arrow(x1: int, y1: int, x2: int, y2: int, *, cls: str = "nn-arrow", test_id: str = "") -> str:
    test_attr = f' data-testid="{escape(test_id)}"' if test_id else ""
    return f'<path{test_attr} class="{escape(cls)}" d="M{x1},{y1} L{x2},{y2}" marker-end="url(#nn-arrowhead)" />'


def _ctde_layer_graph_svg(frames: dict[str, pd.DataFrame], root_id: str) -> str:
    hidden = _summary_value(frames, "hidden_dim", "64")
    dso_dim = _summary_value(frames, "dso_input_dim", "D_dso")
    vpp_dim = _summary_value(frames, "vpp_input_dim", "D_vpp")
    portfolio_dim = _summary_value(frames, "portfolio_input_dim", "D_port")
    critic_dim = _summary_value(frames, "critic_input_dim", "D_critic")
    critic_action_dim = _summary_value(frames, "critic_action_dim", "D_action", "critic_action_summary_dim")
    max_der = _summary_value(frames, "max_der_per_vpp", "K_i")
    n_vpps = str(_count_vpps(frames))

    def encoder_row(y: int, lane_cls: str, input_name: str, input_zh: str, input_dim: str, latent: str, test_prefix: str) -> str:
        nodes = [
            _layer_node(x=42, y=y, w=162, cls=lane_cls, title_en=input_name, title_zh=input_zh, shape=f"R^{input_dim}", sub_en="private/public obs", sub_zh="观测", test_id=f"{test_prefix}-input"),
            _layer_node(x=246, y=y, w=162, cls="nn-input", title_en="LayerNorm", title_zh="LayerNorm", shape=f"{input_dim}", sub_en="normalize", sub_zh="归一化", test_id=f"{test_prefix}-layernorm"),
            _layer_node(x=450, y=y, w=174, cls=lane_cls, title_en="Linear", title_zh="Linear", shape=f"{input_dim}->{hidden}", sub_en="encoder", sub_zh="编码", test_id=f"{test_prefix}-linear1"),
            _layer_node(x=666, y=y, w=132, cls="nn-backbone", title_en="Tanh", title_zh="Tanh", shape="activation", test_id=f"{test_prefix}-tanh1"),
            _layer_node(x=840, y=y, w=174, cls=lane_cls, title_en="Linear", title_zh="Linear", shape=f"{hidden}->{hidden}", sub_en="encoder", sub_zh="编码", test_id=f"{test_prefix}-linear2"),
            _layer_node(x=1056, y=y, w=150, cls="nn-backbone", title_en=latent, title_zh=latent, shape=f"R^{hidden}", sub_en="latent", sub_zh="隐变量", node_id=f"{test_prefix}-latent", test_id=f"{test_prefix}-latent"),
        ]
        arrows = [
            _straight_arrow(204, y + 42, 246, y + 42, test_id=f"{test_prefix}-edge-1"),
            _straight_arrow(408, y + 42, 450, y + 42, test_id=f"{test_prefix}-edge-2"),
            _straight_arrow(624, y + 42, 666, y + 42, test_id=f"{test_prefix}-edge-3"),
            _straight_arrow(798, y + 42, 840, y + 42, test_id=f"{test_prefix}-edge-4"),
            _straight_arrow(1014, y + 42, 1056, y + 42, test_id=f"{test_prefix}-edge-5"),
        ]
        return "".join(nodes + arrows)

    svg = f"""
    <svg id="ctde-layer-graph-svg" class="nn-architecture-svg nn-layer-graph-svg" viewBox="0 0 1880 720" role="img" aria-labelledby="{escape(root_id)}-layer-title" data-testid="ctde-layer-graph-svg">
      <title id="{escape(root_id)}-layer-title">Current CTDE layer graph with tensor dimensions</title>
      <defs>
        <marker id="nn-arrowhead-layer" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="8" markerHeight="8" orient="auto-start-reverse">
          <path d="M 0 0 L 10 5 L 0 10 z" fill="#335b7a" />
        </marker>
        <marker id="nn-arrowhead-orange" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="8" markerHeight="8" orient="auto-start-reverse">
          <path d="M 0 0 L 10 5 L 0 10 z" fill="#d86400" />
        </marker>
      </defs>
      {_svg_lang("Current executable layer graph: DER set encoder + action-conditioned centralized critic", "当前可执行逐层图：DER集合编码器 + 动作条件集中式critic", 940, 34, css_class="nn-lane-label")}

      <rect class="nn-execution-lane" x="24" y="52" width="1832" height="456" rx="24" />
      <rect class="nn-training-lane" x="24" y="536" width="1832" height="136" rx="24" />
      {_svg_lang("DSO actor", "DSO actor", 28, 102, css_class="nn-lane-label", anchor="start")}
      {encoder_row(78, "nn-dso", "o_dso", "o_dso", dso_dim, "z_dso", "ctde-dso")}
      {_layer_node(x=1254, y=78, w=174, cls="nn-dso", title_en="mean head", title_zh="mean head", shape=f"Linear({hidden},{n_vpps})", sub_en="tanh", sub_zh="tanh", node_id="ctde-dso-mean-head", test_id="ctde-dso-mean-head")}
      {_layer_node(x=1468, y=78, w=160, cls="nn-dso", title_en="log_std", title_zh="log_std", shape=f"R^{n_vpps}", sub_en="Parameter", sub_zh="可训练参数", node_id="ctde-dso-log-std", test_id="ctde-dso-log-std")}
      {_layer_node(x=1668, y=78, w=160, cls="nn-dso", title_en="Normal", title_zh="Normal", shape=f"a_dso R^{n_vpps}", sub_en="envelope", sub_zh="包络动作", test_id="ctde-dso-normal-action")}
      {_straight_arrow(1206, 120, 1254, 120, test_id="ctde-dso-latent-to-mean")}
      {_straight_arrow(1428, 120, 1468, 120, test_id="ctde-dso-mean-to-std")}
      {_straight_arrow(1628, 120, 1668, 120, test_id="ctde-dso-std-to-action")}

      {_svg_lang("VPP dispatch actors, one independent actor per VPP by default; local observation only", "VPP 调度 actor：默认每个 VPP 独立一套，只读本地观测", 28, 252, css_class="nn-lane-label", anchor="start")}
      {encoder_row(222, "nn-vpp", "o_vpp_i", "o_vpp_i", vpp_dim, "z_vpp_i", "ctde-vpp")}
      {_layer_node(x=1254, y=188, w=176, cls="nn-vpp", title_en="aggregate head", title_zh="聚合头", shape=f"Linear({hidden},1)", sub_en="P_i mean", sub_zh="P_i 均值", test_id="ctde-vpp-aggregate-head")}
      {_layer_node(x=1254, y=290, w=176, cls="nn-der", title_en="DER head", title_zh="DER头", shape=f"Linear({hidden},{max_der})", sub_en="u_der mean", sub_zh="DER均值", test_id="ctde-vpp-der-head")}
      {_layer_node(x=1474, y=238, w=168, cls="nn-vpp", title_en="log_std", title_zh="log_std", shape=f"R^1 + R^{max_der}", sub_en="Parameter", sub_zh="可训练参数", test_id="ctde-vpp-log-std")}
      {_layer_node(x=1682, y=238, w=154, cls="nn-vpp", title_en="Normal", title_zh="Normal", shape="P_i + u_i", sub_en="local actions", sub_zh="本地动作", test_id="ctde-vpp-normal-action")}
      {_straight_arrow(1206, 264, 1254, 230, test_id="ctde-vpp-latent-to-aggregate")}
      {_straight_arrow(1206, 264, 1254, 332, test_id="ctde-vpp-latent-to-der")}
      {_straight_arrow(1430, 230, 1474, 280, test_id="ctde-vpp-aggregate-to-std")}
      {_straight_arrow(1430, 332, 1474, 280, test_id="ctde-vpp-der-to-std")}
      {_straight_arrow(1642, 280, 1682, 280, test_id="ctde-vpp-std-to-action")}

      {_svg_lang("VPP portfolio actor, slow-cycle commercial configuration", "VPP 组合配置 actor：慢周期商业配置", 28, 432, css_class="nn-lane-label", anchor="start")}
      {encoder_row(392, "nn-portfolio", "h_vpp_i", "h_vpp_i", portfolio_dim, "z_port_i", "ctde-portfolio")}
      {_layer_node(x=1254, y=392, w=190, cls="nn-portfolio", title_en="logits head", title_zh="logits头", shape=f"Linear({hidden},3)", sub_en="keep/reweight/change", sub_zh="保持/加权/变更", test_id="ctde-portfolio-logits-head")}
      {_layer_node(x=1490, y=392, w=174, cls="nn-portfolio", title_en="Categorical", title_zh="Categorical", shape="g_i in {0,1,2}", sub_en="proposal", sub_zh="配置建议", test_id="ctde-portfolio-categorical-action")}
      {_straight_arrow(1206, 434, 1254, 434, test_id="ctde-portfolio-latent-to-logits")}
      {_straight_arrow(1444, 434, 1490, 434, test_id="ctde-portfolio-logits-to-action")}

      {_svg_lang("centralized critic, training only, state + joint-action summary", "集中式 critic：仅训练期，状态 + 联合动作摘要", 28, 602, css_class="nn-lane-label", anchor="start")}
      {encoder_row(568, "nn-critic", "s_critic+a", "s_critic+a", f"{critic_dim}+{critic_action_dim}", "z_critic", "ctde-critic")}
      {_layer_node(x=1254, y=568, w=184, cls="nn-critic", title_en="value head", title_zh="value头", shape=f"Linear({hidden},1)", sub_en="V(s)", sub_zh="V(s)", test_id="ctde-critic-value-head")}
      {_layer_node(x=1484, y=568, w=178, cls="nn-critic", title_en="advantage", title_zh="优势函数", shape="A_t = G_t - V", sub_en="detach V for actor", sub_zh="actor侧detach", test_id="ctde-advantage-node")}
      {_layer_node(x=1704, y=568, w=126, cls="nn-critic", title_en="losses", title_zh="损失", shape="L_dso/L_vpp/L_pi", sub_en="backprop", sub_zh="反传", test_id="ctde-loss-node")}
      {_straight_arrow(1206, 610, 1254, 610, cls="nn-loss-arrow", test_id="ctde-critic-latent-to-value")}
      {_straight_arrow(1438, 610, 1484, 610, cls="nn-loss-arrow", test_id="ctde-value-to-advantage")}
      {_straight_arrow(1662, 610, 1704, 610, cls="nn-loss-arrow", test_id="ctde-advantage-to-loss")}
      <path class="nn-loss-loop" data-testid="ctde-loss-to-dso-actor" d="M1760,568 C1740,470 1510,144 1360,144" />
      <path class="nn-loss-loop" data-testid="ctde-loss-to-vpp-actor" d="M1768,568 C1690,486 1504,324 1362,324" />
      <path class="nn-loss-loop" data-testid="ctde-loss-to-portfolio-actor" d="M1780,568 C1698,524 1514,444 1360,444" />
    </svg>
    """
    return svg.replace('marker-end="url(#nn-arrowhead)"', 'marker-end="url(#nn-arrowhead-layer)"')


def neural_network_diagram_css() -> str:
    return """
    .neural-architecture-panel {
      overflow: hidden;
    }
    .nn-figure-wrap {
      width: 100%;
      overflow-x: auto;
      border: 1px solid rgba(148, 163, 184, 0.22);
      border-radius: 18px;
      background:
        linear-gradient(90deg, rgba(238, 246, 252, 0.82) 0 1px, transparent 1px 80px),
        linear-gradient(180deg, rgba(238, 246, 252, 0.82) 0 1px, transparent 1px 80px),
        #fbfdff;
      padding: 10px;
    }
    .nn-architecture-svg {
      min-width: 1320px;
      width: 100%;
      height: auto;
      display: block;
    }
    .nn-layer-graph-svg {
      min-width: 1760px;
    }
    .nn-svg-lang {
      display: none;
    }
    html[data-lang="en"] .nn-svg-en,
    html:not([data-lang]) .nn-svg-en {
      display: inline;
    }
    html[data-lang="zh"] .nn-svg-zh {
      display: inline;
    }
    .nn-title {
      font-size: 18px;
      font-weight: 900;
      fill: #14263a;
      letter-spacing: 0;
    }
    .nn-sub {
      font-size: 13px;
      font-weight: 800;
      fill: #466074;
    }
    .nn-shape {
      font-size: 14px;
      font-weight: 900;
      fill: #1769aa;
      letter-spacing: 0;
    }
    .nn-lane-label {
      font-size: 14px;
      font-weight: 900;
      fill: #52677d;
      letter-spacing: 0.02em;
    }
    .nn-block rect,
    .nn-action rect,
    .nn-dist circle {
      fill: #ffffff;
      stroke-width: 2.4;
      filter: drop-shadow(0 8px 12px rgba(15, 35, 56, 0.07));
    }
    .nn-input rect { stroke: #52677d; fill: #f8fbfd; }
    .nn-backbone rect { stroke: #1769aa; fill: #f2f8fe; }
    .nn-dso rect, .nn-dso circle { stroke: #1f77b4; fill: #f2f8ff; }
    .nn-vpp rect, .nn-vpp circle { stroke: #2ca02c; fill: #f4fbf4; }
    .nn-der rect, .nn-der circle { stroke: #12846d; fill: #f1fbf8; }
    .nn-portfolio rect, .nn-portfolio circle { stroke: #9467bd; fill: #fbf7ff; }
    .nn-critic rect { stroke: #ff7f0e; fill: #fff8ef; }
    .nn-safety rect { stroke: #d94f3d; fill: #fff5f3; }
    .nn-env rect { stroke: #334155; fill: #f8fafc; }
    .nn-arrow {
      fill: none;
      stroke: #335b7a;
      stroke-width: 2.6;
    }
    .nn-arrow-soft {
      fill: none;
      stroke: #7a8ca0;
      stroke-width: 2.2;
      stroke-dasharray: 6 5;
    }
    .nn-loss-arrow {
      fill: none;
      stroke: #d86400;
      stroke-width: 2.5;
      stroke-dasharray: 7 5;
    }
    .nn-loss-loop {
      fill: none;
      stroke: #d86400;
      stroke-width: 2.6;
      stroke-dasharray: 7 5;
      marker-end: url(#nn-arrowhead-orange);
    }
    .nn-head-lane {
      fill: rgba(255,255,255,0.62);
      stroke: rgba(148,163,184,0.32);
      stroke-width: 1.4;
      stroke-dasharray: 5 6;
    }
    .nn-mini-layer rect {
      fill: #dff0fb;
      stroke: #1769aa;
      stroke-width: 1.2;
      opacity: 0.94;
    }
    .nn-mini-layer.tanh rect {
      fill: #e8f5e9;
      stroke: #2ca02c;
    }
    .nn-mini-layer text {
      font-size: 10px;
      fill: #26445f;
      font-weight: 900;
    }
    .nn-legend {
      display: flex;
      flex-wrap: wrap;
      gap: 9px;
      margin-top: 12px;
    }
    .nn-legend span {
      display: inline-flex;
      align-items: center;
      gap: 7px;
      border: 1px solid #d8e3ee;
      border-radius: 999px;
      padding: 6px 9px;
      background: white;
      color: #33465a;
      font-size: 12px;
      font-weight: 900;
    }
    .nn-legend i {
      width: 11px;
      height: 11px;
      border-radius: 3px;
      display: inline-block;
    }
    .nn-privacy-boundary {
      fill: none;
      stroke: #64748b;
      stroke-width: 2.2;
      stroke-dasharray: 9 7;
    }
    .nn-training-lane {
      fill: rgba(255, 248, 239, 0.74);
      stroke: rgba(255, 127, 14, 0.55);
      stroke-width: 1.8;
      stroke-dasharray: 9 6;
    }
    .nn-execution-lane {
      fill: rgba(248, 251, 253, 0.58);
      stroke: rgba(82, 103, 125, 0.28);
      stroke-width: 1.4;
    }
    .nn-privacy-label {
      font-size: 12px;
      font-weight: 900;
      fill: #52677d;
      letter-spacing: 0;
    }
    .nn-layer-heading {
      margin: 18px 0 8px;
      color: #14263a;
      font-size: 18px;
      line-height: 1.35;
    }
    """


def build_neural_network_architecture_diagram(
    frames: dict[str, pd.DataFrame],
    *,
    root_id: str,
    heading_class: str = "section-heading",
) -> str:
    ctde_is_primary = _current_ctde_is_primary(frames)
    if ctde_is_primary:
        return ""
    heading_en = (
        "Legacy Shared-Backbone Benchmark (Not Current Primary Model)"
        if ctde_is_primary
        else "Current Shared-Backbone Prototype / Benchmark"
    )
    heading_zh = (
        "保留的共享骨干基线（不是当前主模型）"
        if ctde_is_primary
        else "当前共享骨干原型 / Benchmark"
    )
    note_en = (
        "This diagram is retained only as a full-information benchmark for ablation and regression tests. "
        "The current primary model is the implemented privacy-separated CTDE architecture shown above/below."
        if ctde_is_primary
        else "This diagram describes the currently runnable shared-backbone PyTorch benchmark, not the final privacy-preserving CTDE method."
    )
    note_zh = (
        "本图只作为 full-information benchmark 保留，用于消融实验和回归测试；当前主模型是页面中的“已实现隐私分离 CTDE 架构”。"
        if ctde_is_primary
        else "本图描述当前可运行的共享骨干 PyTorch benchmark，并不代表最终隐私分离 CTDE 方法。"
    )
    neural = frame_or_empty(frames, "rl_neural_network_architecture")
    encoder = _row(neural, "manual_dso_observation_encoder")
    backbone = _row(neural, "shared_mlp_backbone")
    dso = _row(neural, "dso_gaussian_actor_head")
    vpp = _row(neural, "vpp_aggregate_gaussian_actor_head")
    der = _row(neural, "der_dispatch_gaussian_actor_head")
    portfolio = _row(neural, "portfolio_categorical_actor_head")
    critic = _row(neural, "centralized_value_head")

    x_shape = _text(encoder.get("output_shape"), "R^(5+7*N_vpp)")
    h_shape = _text(backbone.get("output_shape"), "R^64")
    dso_shape = _text(dso.get("output_shape"), "mean/log_std R^N_vpp")
    vpp_shape = _text(vpp.get("output_shape"), "mean/log_std R^N_vpp")
    der_shape = _text(der.get("output_shape"), "mean/log_std R^N_DER")
    portfolio_shape = _text(portfolio.get("output_shape"), "logits R^N_vpp x 3")
    value_shape = _text(critic.get("output_shape"), "R^1")

    svg = f"""
    <svg id="rl-architecture-svg" class="nn-architecture-svg" viewBox="0 0 1500 760" role="img" aria-labelledby="{escape(root_id)}-title" data-testid="rl-architecture-svg">
      <title id="{escape(root_id)}-title">Actor-Critic neural network architecture</title>
      <defs>
        <marker id="nn-arrowhead" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="8" markerHeight="8" orient="auto-start-reverse">
          <path d="M 0 0 L 10 5 L 0 10 z" fill="#335b7a" />
        </marker>
        <marker id="nn-arrowhead-orange" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="8" markerHeight="8" orient="auto-start-reverse">
          <path d="M 0 0 L 10 5 L 0 10 z" fill="#d86400" />
        </marker>
      </defs>

      <rect class="nn-head-lane" x="468" y="54" width="630" height="564" rx="24" />
      <rect id="distribution-layer" class="nn-head-lane" x="1124" y="54" width="150" height="564" rx="24" data-testid="distribution-layer" />
      <rect id="action-layer" class="nn-head-lane" x="1298" y="54" width="154" height="564" rx="24" data-testid="action-layer" />
      {_svg_lang("Shared latent h", "共享隐变量 h", 782, 84, css_class="nn-lane-label")}
      {_svg_lang("Distribution", "分布", 1199, 84, css_class="nn-lane-label")}
      {_svg_lang("Action", "动作", 1375, 84, css_class="nn-lane-label")}

      {_box(x=42, y=254, w=170, h=112, cls="nn-input", title_en="Grid/VPP Obs", title_zh="电网/VPP观测", shape="dict", sub_en="state + bids", sub_zh="状态+报量", test_id="input-observation")}
      {_box(x=254, y=254, w=174, h=112, cls="nn-input", title_en="Numeric Encoder", title_zh="数值编码器", shape=x_shape, sub_en="fixed", sub_zh="固定", node_id="manual-numeric-encoder", test_id="manual-numeric-encoder")}

      <g class="nn-mini-layer">
        <rect x="504" y="234" width="86" height="44" rx="8" />
        <text x="547" y="260" text-anchor="middle">Linear</text>
      </g>
      <g class="nn-mini-layer tanh">
        <rect x="604" y="234" width="64" height="44" rx="8" />
        <text x="636" y="260" text-anchor="middle">Tanh</text>
      </g>
      <g class="nn-mini-layer">
        <rect x="682" y="234" width="86" height="44" rx="8" />
        <text x="725" y="260" text-anchor="middle">Linear</text>
      </g>
      <g class="nn-mini-layer tanh">
        <rect x="782" y="234" width="64" height="44" rx="8" />
        <text x="814" y="260" text-anchor="middle">Tanh</text>
      </g>
      {_box(x=492, y=292, w=366, h=96, cls="nn-backbone", title_en="Shared MLP Backbone", title_zh="共享 MLP 主干", shape=f"{x_shape} → {h_shape}", sub_en="trainable", sub_zh="可训练", node_id="shared-backbone", test_id="shared-backbone")}

      {_box(x=890, y=84, w=202, h=96, cls="nn-dso", title_en="DSO Actor Head", title_zh="DSO 策略头", shape=dso_shape, sub_en="Gaussian", sub_zh="高斯分布", node_id="dso-gaussian-head", test_id="dso-gaussian-head")}
      {_distribution_node(cx=1199, cy=132, label_en="Normal", label_zh="高斯", shape="a_dso", cls="nn-dso", test_id="dso-normal")}
      {_action_node(x=1312, y=102, w=126, h=60, cls="nn-dso", label_en="Clamp", label_zh="裁剪", shape="[-1,1]", test_id="dso-action")}

      {_box(x=890, y=214, w=202, h=96, cls="nn-vpp", title_en="VPP Target Head", title_zh="VPP 聚合头", shape=vpp_shape, sub_en="Gaussian", sub_zh="高斯分布", node_id="vpp-aggregate-head", test_id="vpp-aggregate-head")}
      {_distribution_node(cx=1199, cy=262, label_en="Normal", label_zh="高斯", shape="P*", cls="nn-vpp", test_id="vpp-normal")}
      {_action_node(x=1304, y=232, w=142, h=60, cls="nn-vpp", label_en="selected P", label_zh="聚合P", shape="MW", test_id="vpp-target-action")}

      {_box(x=890, y=344, w=202, h=96, cls="nn-der", title_en="DER Dispatch Head", title_zh="DER 解聚合头", shape=der_shape, sub_en="Gaussian", sub_zh="高斯分布", node_id="der-dispatch-head", test_id="der-dispatch-head")}
      {_distribution_node(cx=1199, cy=392, label_en="Normal", label_zh="高斯", shape="u_DER", cls="nn-der", test_id="der-normal")}
      {_action_node(x=1304, y=362, w=142, h=60, cls="nn-der", label_en="DER slice", label_zh="DER切片", shape="u_i", test_id="der-action-slice")}

      {_box(x=890, y=474, w=202, h=96, cls="nn-portfolio", title_en="Portfolio Head", title_zh="组合配置头", shape=portfolio_shape, sub_en="Categorical", sub_zh="类别分布", node_id="portfolio-categorical-head", test_id="portfolio-categorical-head")}
      {_distribution_node(cx=1199, cy=522, label_en="Categorical", label_zh="类别", shape="π", cls="nn-portfolio", test_id="portfolio-categorical")}
      {_action_node(x=1298, y=492, w=154, h=60, cls="nn-portfolio", label_en="portfolio", label_zh="组合建议", shape="3-way", test_id="portfolio-action")}

      {_box(x=620, y=596, w=202, h=96, cls="nn-critic", title_en="Value Head", title_zh="价值头", shape=value_shape, sub_en="critic V(s)", sub_zh="critic V(s)", node_id="centralized-value-head", test_id="centralized-value-head")}
      {_box(x=1128, y=626, w=204, h=76, cls="nn-safety", title_en="Safety Projection", title_zh="安全投影", shape="FR/DOE + DER", sub_en="non-RL", sub_zh="非RL", node_id="safety-projection", test_id="safety-projection")}
      {_box(x=1346, y=626, w=118, h=76, cls="nn-env", title_en="runpp", title_zh="潮流", shape="env", sub_en="reward", sub_zh="奖励", node_id="env-feedback", test_id="env-feedback")}

      {_arrow(212, 310, 254, 310, test_id="edge-input-to-encoder")}
      {_arrow(428, 310, 492, 340, test_id="edge-encoder-to-backbone")}
      {_arrow(858, 340, 890, 132, test_id="edge-backbone-to-dso-head")}
      {_arrow(858, 340, 890, 262, test_id="edge-backbone-to-vpp-head")}
      {_arrow(858, 340, 890, 392, test_id="edge-backbone-to-der-head")}
      {_arrow(858, 340, 890, 522, test_id="edge-backbone-to-portfolio-head")}
      {_arrow(858, 362, 620, 644, cls="nn-loss-arrow", test_id="edge-backbone-to-value-head")}
      {_arrow(1092, 132, 1155, 132, test_id="edge-dso-head-to-distribution")}
      {_arrow(1092, 262, 1155, 262, test_id="edge-vpp-head-to-distribution")}
      {_arrow(1092, 392, 1155, 392, test_id="edge-der-head-to-distribution")}
      {_arrow(1092, 522, 1155, 522, test_id="edge-portfolio-head-to-distribution")}
      {_arrow(1243, 132, 1312, 132, test_id="edge-dso-distribution-to-action")}
      {_arrow(1243, 262, 1304, 262, test_id="edge-vpp-distribution-to-action")}
      {_arrow(1243, 392, 1304, 392, test_id="edge-der-distribution-to-action")}
      {_arrow(1243, 522, 1298, 522, test_id="edge-portfolio-distribution-to-action")}
      {_arrow(1375, 292, 1216, 626, cls="nn-arrow-soft", test_id="edge-actions-to-safety-projection")}
      {_arrow(1375, 422, 1234, 626, cls="nn-arrow-soft", test_id="edge-der-actions-to-safety-projection")}
      {_arrow(1334, 552, 1246, 626, cls="nn-arrow-soft", test_id="edge-portfolio-action-to-safety-projection")}
      {_arrow(1332, 664, 1346, 664, test_id="edge-safety-to-env")}
      <path id="loss-feedback" data-testid="loss-feedback" class="nn-loss-loop" d="M721,596 C712,544 766,516 846,510 C938,504 1040,558 1160,626" />

      {_svg_lang("sample", "采样", 1172, 103, css_class="nn-sub")}
      {_svg_lang("map to envelope", "映射到包络", 1192, 333, css_class="nn-sub")}
      {_svg_lang("slice by VPP", "按VPP切片", 1194, 463, css_class="nn-sub")}
      {_svg_lang("loss feedback", "损失反馈", 828, 552, css_class="nn-sub")}
    </svg>
    """

    legend = f"""
    <div class="nn-legend" aria-label="Neural network diagram legend">
      <span><i style="background:#f2f8fe;border:1px solid #1f77b4"></i>{_lang("DSO head", "DSO 策略头")}</span>
      <span><i style="background:#f4fbf4;border:1px solid #2ca02c"></i>{_lang("VPP aggregate", "VPP 聚合")}</span>
      <span><i style="background:#f1fbf8;border:1px solid #12846d"></i>{_lang("DER disaggregation", "DER 解聚合")}</span>
      <span><i style="background:#fbf7ff;border:1px solid #9467bd"></i>{_lang("Portfolio", "组合配置")}</span>
      <span><i style="background:#fff8ef;border:1px solid #ff7f0e"></i>{_lang("Critic", "价值网络")}</span>
      <span><i style="background:#fff5f3;border:1px solid #d94f3d"></i>{_lang("Non-RL safety", "非RL安全层")}</span>
    </div>
    """
    layer_svg = _ctde_layer_graph_svg(frames, root_id)

    return f"""
    <section class="panel neural-architecture-panel" id="{escape(root_id)}">
      <div class="{escape(heading_class)}">
        <div>
          <p class="eyebrow">{_lang("Neural Architecture", "神经网络架构")}</p>
          <h2>{_lang(heading_en, heading_zh)}</h2>
        </div>
        <p class="section-note">{_lang(note_en, note_zh, display="block")}</p>
      </div>
      <div class="nn-figure-wrap">{svg}</div>
      <h3 class="nn-layer-heading">{_lang("Current CTDE Layer Graph with Tensor Shapes", "当前 CTDE 逐层神经网络结构图")}</h3>
      <div class="nn-figure-wrap">{layer_svg}</div>
      {legend}
    </section>
    """


def build_target_ctde_architecture_diagram(
    frames: dict[str, pd.DataFrame],
    *,
    root_id: str,
    heading_class: str = "section-heading",
) -> str:
    ctde_is_primary = _current_ctde_is_primary(frames)
    eyebrow_en = "Current CTDE" if ctde_is_primary else "Target CTDE"
    eyebrow_zh = "当前CTDE" if ctde_is_primary else "目标CTDE"
    title_en = (
        "Current Recommended Hierarchical HAPPO / Privacy-Separated CTDE Neural Network Architecture"
        if ctde_is_primary
        else "Target Privacy-Preserving CTDE Architecture"
    )
    title_zh = (
        "当前推荐的分层 HAPPO / 隐私分离 CTDE 神经网络架构"
        if ctde_is_primary
        else "目标隐私分离 CTDE 神经网络架构"
    )
    note_en = (
        "This is the current recommended hierarchical trainer: the DSO actor is separate, each VPP has its own dispatch actor, each VPP has its own slow-loop portfolio actor, and only the training critic reads critic_global_state."
        if ctde_is_primary
        else "This is the intended top-conference architecture: DSO, VPP dispatch and VPP portfolio actors use separate execution encoders; only the training critic reads critic_global_state."
    )
    note_zh = (
        "这是当前推荐的分层训练器：DSO actor 独立，每个 VPP 拥有自己的调度 actor，每个 VPP 拥有自己的慢周期组合配置 actor；只有训练期 critic 读取 critic_global_state。"
        if ctde_is_primary
        else "这是面向顶会规格的目标结构：DSO、VPP 调度和 VPP 组合配置 actor 使用分离的执行期编码器；只有训练期 critic 读取 critic_global_state。"
    )
    target = frame_or_empty(frames, "rl_target_ctde_architecture")
    dso_encoder = _row(target, "dso_private_observation_encoder")
    dso_actor = _row(target, "dso_envelope_actor")
    vpp_encoder = _row(target, "vpp_local_observation_encoder")
    vpp_actor = _row(target, "vpp_der_dispatch_actor")
    portfolio_encoder = _row(target, "vpp_portfolio_slow_encoder")
    portfolio_actor = _row(target, "vpp_portfolio_actor")
    critic = _row(target, "centralized_training_critic")
    safety = _row(target, "non_rl_safety_projection")

    z_dso = _text(dso_encoder.get("output_shape"), "z_dso")
    z_vpp = _text(vpp_encoder.get("output_shape"), "z_vpp_i")
    z_portfolio = _text(portfolio_encoder.get("output_shape"), "z_portfolio_i")
    critic_shape = _text(critic.get("output_shape"), "V(s) / Q(s,a)")
    safety_shape = _text(safety.get("output_shape"), "safe setpoints")

    svg = f"""
    <svg id="target-ctde-svg" class="nn-architecture-svg" viewBox="0 0 1560 860" role="img" aria-labelledby="{escape(root_id)}-title" data-testid="target-ctde-svg">
      <title id="{escape(root_id)}-title">Target privacy-preserving CTDE neural architecture</title>
      <defs>
        <marker id="nn-arrowhead" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="8" markerHeight="8" orient="auto-start-reverse">
          <path d="M 0 0 L 10 5 L 0 10 z" fill="#335b7a" />
        </marker>
        <marker id="nn-arrowhead-orange" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="8" markerHeight="8" orient="auto-start-reverse">
          <path d="M 0 0 L 10 5 L 0 10 z" fill="#d86400" />
        </marker>
      </defs>

      <rect class="nn-execution-lane" x="24" y="52" width="1512" height="586" rx="24" />
      <rect class="nn-training-lane" x="24" y="670" width="1512" height="150" rx="24" />
      {_svg_lang("Decentralized execution lanes", "分散执行泳道", 156, 82, css_class="nn-lane-label")}
      {_svg_lang("Centralized training only", "仅集中训练期可见", 164, 700, css_class="nn-lane-label")}
      <line id="privacy-boundary" data-testid="privacy-boundary" class="nn-privacy-boundary" x1="34" y1="238" x2="1516" y2="238" />
      {_svg_lang("privacy boundary: DSO does not read VPP private DER states", "隐私边界：DSO 不读取 VPP 私有 DER 状态", 772, 226, css_class="nn-privacy-label")}

      {_box(x=42, y=106, w=184, h=96, cls="nn-dso", title_en="DSO Obs", title_zh="DSO观测", shape="grid + bids", sub_en="topology/security", sub_zh="拓扑/安全", test_id="dso-observation")}
      {_box(x=268, y=106, w=206, h=96, cls="nn-dso", title_en="DSO Encoder", title_zh="DSO编码器", shape=z_dso, sub_en="separate", sub_zh="独立", node_id="dso-observation-encoder", test_id="dso-observation-encoder")}
      {_box(x=516, y=106, w=206, h=96, cls="nn-dso", title_en="DSO Actor", title_zh="DSO actor", shape="mu/sigma", sub_en="envelope intent", sub_zh="包络意图", node_id="dso-actor", test_id="dso-actor")}
      {_action_node(x=766, y=124, w=176, h=62, cls="nn-dso", label_en="Envelope / Price", label_zh="包络/价格", shape="FR/DOE pref", node_id="dso-to-vpp-envelope", test_id="dso-to-vpp-envelope")}

      <g id="vpp-local-private-state" data-testid="vpp-local-private-state">
        {_box(x=42, y=294, w=184, h=96, cls="nn-vpp", title_en="VPP Local Obs", title_zh="VPP本地观测", shape="DER + cost", sub_en="private", sub_zh="私有", test_id="vpp-local-observation")}
      </g>
      {_box(x=268, y=294, w=206, h=96, cls="nn-vpp", title_en="Local Encoder", title_zh="本地编码器", shape=z_vpp, sub_en="per VPP type", sub_zh="按VPP类型", node_id="vpp-local-observation-encoder", test_id="vpp-local-observation-encoder")}
      {_box(x=516, y=294, w=206, h=96, cls="nn-vpp", title_en="VPP Dispatch Actor", title_zh="VPP调度actor", shape="P/Q + DER u", sub_en="decentralized", sub_zh="分散执行", node_id="vpp-dispatch-actor", test_id="vpp-dispatch-actor")}
      {_action_node(x=766, y=312, w=176, h=62, cls="nn-vpp", label_en="DER Actions", label_zh="DER动作", shape="u_i", test_id="vpp-der-actions")}

      {_box(x=42, y=482, w=184, h=96, cls="nn-portfolio", title_en="Long-Cycle KPIs", title_zh="长周期KPI", shape="profit/risk", sub_en="private history", sub_zh="私有历史", test_id="portfolio-history")}
      {_box(x=268, y=482, w=206, h=96, cls="nn-portfolio", title_en="Portfolio Encoder", title_zh="组合编码器", shape=z_portfolio, sub_en="slow loop", sub_zh="慢周期", node_id="vpp-portfolio-encoder", test_id="vpp-portfolio-encoder")}
      {_box(x=516, y=482, w=206, h=96, cls="nn-portfolio", title_en="Portfolio Actor", title_zh="组合actor", shape="3 logits", sub_en="keep/reweight/change", sub_zh="保持/加权/变更", node_id="vpp-portfolio-actor", test_id="vpp-portfolio-actor")}
      {_action_node(x=766, y=500, w=176, h=62, cls="nn-portfolio", label_en="Config Proposal", label_zh="配置建议", shape="gated", test_id="portfolio-config-action")}

      {_box(x=1002, y=246, w=216, h=112, cls="nn-safety", title_en="Safety Projection", title_zh="安全投影", shape=safety_shape, sub_en="non-RL shield", sub_zh="非RL保护层", node_id="ctde-safety-projection", test_id="ctde-safety-projection")}
      {_box(x=1262, y=246, w=214, h=112, cls="nn-env", title_en="pandapower", title_zh="pandapower", shape="runpp", sub_en="security + reward", sub_zh="安全校核+奖励", test_id="ctde-pandapower-env")}

      {_box(x=42, y=722, w=214, h=74, cls="nn-input", title_en="critic_global_state", title_zh="critic全局状态", shape="train only", sub_en="trusted simulator", sub_zh="可信仿真器", node_id="critic-global-state", test_id="critic-global-state")}
      {_box(x=318, y=712, w=226, h=96, cls="nn-critic", title_en="Centralized Critic", title_zh="集中式critic", shape=critic_shape, sub_en="not deployed", sub_zh="执行期不用", node_id="centralized-critic", test_id="centralized-critic")}
      {_box(x=606, y=722, w=210, h=74, cls="nn-critic", title_en="Advantage / Loss", title_zh="优势函数/损失", shape="A_t, L", sub_en="policy update", sub_zh="策略更新", node_id="ctde-loss-feedback", test_id="ctde-loss-feedback")}

      {_arrow(226, 154, 268, 154, test_id="ctde-edge-dso-obs-encoder")}
      {_arrow(474, 154, 516, 154, test_id="ctde-edge-dso-encoder-actor")}
      {_arrow(722, 154, 766, 154, test_id="ctde-edge-dso-actor-envelope")}
      {_arrow(942, 154, 1002, 282, cls="nn-arrow-soft", test_id="ctde-edge-envelope-safety")}

      {_arrow(226, 342, 268, 342, test_id="ctde-edge-vpp-obs-encoder")}
      {_arrow(474, 342, 516, 342, test_id="ctde-edge-vpp-encoder-actor")}
      {_arrow(722, 342, 766, 342, test_id="ctde-edge-vpp-actor-action")}
      {_arrow(942, 342, 1002, 312, test_id="ctde-edge-der-action-safety")}
      {_arrow(854, 186, 604, 294, cls="nn-arrow-soft", test_id="ctde-edge-envelope-to-vpp-actor")}

      {_arrow(226, 530, 268, 530, test_id="ctde-edge-portfolio-history-encoder")}
      {_arrow(474, 530, 516, 530, test_id="ctde-edge-portfolio-encoder-actor")}
      {_arrow(722, 530, 766, 530, test_id="ctde-edge-portfolio-actor-action")}
      {_arrow(854, 500, 1002, 348, cls="nn-arrow-soft", test_id="ctde-edge-portfolio-gate-safety")}

      {_arrow(1218, 302, 1262, 302, test_id="ctde-edge-safety-env")}
      {_arrow(256, 759, 318, 759, cls="nn-loss-arrow", test_id="ctde-edge-global-critic")}
      {_arrow(544, 759, 606, 759, cls="nn-loss-arrow", test_id="ctde-edge-critic-loss")}
      <path data-testid="ctde-edge-loss-to-dso" class="nn-loss-loop" d="M710,722 C704,620 658,464 624,202" />
      <path data-testid="ctde-edge-loss-to-vpp" class="nn-loss-loop" d="M752,722 C790,596 756,440 686,390" />
      <path data-testid="ctde-edge-loss-to-portfolio" class="nn-loss-loop" d="M802,736 C882,650 820,568 722,536" />

      {_svg_lang("separate DSO actor", "独立 DSO actor", 616, 92, css_class="nn-sub")}
      {_svg_lang("local actor, no raw topology", "本地 actor，不接收原始拓扑", 616, 280, css_class="nn-sub")}
      {_svg_lang("slow configuration policy", "慢周期配置策略", 616, 468, css_class="nn-sub")}
      {_svg_lang("critic trains actors, not used in execution", "critic 训练 actor，执行期不用", 620, 840, css_class="nn-sub")}
    </svg>
    """

    legend = f"""
    <div class="nn-legend" aria-label="Target CTDE diagram legend">
      <span><i style="background:#f2f8ff;border:1px solid #1f77b4"></i>{_lang("DSO global guidance", "DSO全局引导")}</span>
      <span><i style="background:#f4fbf4;border:1px solid #2ca02c"></i>{_lang("VPP local dispatch", "VPP本地调度")}</span>
      <span><i style="background:#fbf7ff;border:1px solid #9467bd"></i>{_lang("VPP portfolio slow loop", "VPP组合慢周期")}</span>
      <span><i style="background:#fff8ef;border:1px solid #ff7f0e"></i>{_lang("Centralized critic, training only", "集中式critic，仅训练期")}</span>
      <span><i style="background:#fff5f3;border:1px solid #d94f3d"></i>{_lang("Non-RL safety shield", "非RL安全保护")}</span>
    </div>
    """
    layer_svg = _ctde_layer_graph_svg(frames, root_id)

    return f"""
    <section class="panel neural-architecture-panel" id="{escape(root_id)}">
      <div class="{escape(heading_class)}">
        <div>
          <p class="eyebrow">{_lang(eyebrow_en, eyebrow_zh)}</p>
          <h2>{_lang(title_en, title_zh)}</h2>
        </div>
        <p class="section-note">{_lang(note_en, note_zh, display="block")}</p>
      </div>
      <div class="nn-figure-wrap">{svg}</div>
      <h3 class="nn-layer-heading">{_lang("Current CTDE Layer Graph with Tensor Shapes", "当前 CTDE 逐层神经网络结构图")}</h3>
      <div class="nn-figure-wrap">{layer_svg}</div>
      {legend}
    </section>
    """
