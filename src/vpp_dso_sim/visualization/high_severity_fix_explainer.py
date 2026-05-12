from __future__ import annotations

from pathlib import Path


def build_high_severity_fix_explainer_html(output_path: str | Path) -> Path:
    """Write a reader-friendly dynamic explainer for the high-severity safety fixes."""

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_HTML, encoding="utf-8")
    return path


_HTML = """<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>DSO-VPP 高严重度修复说明</title>
  <style>
    :root {
      --ink:#17212f;
      --muted:#5d6d7e;
      --line:#d7e0e9;
      --bg:#f5f7fb;
      --paper:#ffffff;
      --teal:#147d88;
      --blue:#3767a8;
      --amber:#b36a10;
      --red:#b43a31;
      --green:#26744d;
      --soft-teal:#e7f4f5;
      --soft-blue:#eaf0fb;
      --soft-amber:#fff4df;
      --soft-red:#ffeded;
      --soft-green:#e9f6ee;
      --shadow:0 10px 26px rgba(26, 42, 62, .08);
    }
    * { box-sizing:border-box; }
    body {
      margin:0;
      font-family:"Segoe UI", "Microsoft YaHei", Arial, sans-serif;
      color:var(--ink);
      background:var(--bg);
      line-height:1.55;
    }
    header {
      background:#ffffff;
      border-bottom:1px solid var(--line);
    }
    .wrap { max-width:1180px; margin:0 auto; padding:0 22px; }
    .hero {
      display:grid;
      grid-template-columns:1.25fr .75fr;
      gap:28px;
      align-items:center;
      min-height:360px;
      padding:36px 0 28px;
    }
    h1 { margin:0; font-size:34px; line-height:1.18; letter-spacing:0; }
    h2 { margin:0 0 14px; font-size:24px; letter-spacing:0; }
    h3 { margin:0 0 8px; font-size:18px; letter-spacing:0; }
    h4 { margin:0 0 8px; font-size:15px; letter-spacing:0; }
    p { margin:8px 0 0; color:var(--muted); }
    code { font-family:Consolas, "Courier New", monospace; font-size:.94em; background:#edf2f7; padding:2px 5px; border-radius:4px; }
    .eyebrow { color:var(--teal); font-weight:700; margin-bottom:10px; }
    .hero-copy { max-width:720px; }
    .hero-actions { display:flex; flex-wrap:wrap; gap:10px; margin-top:22px; }
    button, .jump {
      border:1px solid var(--line);
      background:#fff;
      color:var(--ink);
      padding:10px 13px;
      border-radius:8px;
      cursor:pointer;
      font:inherit;
      text-decoration:none;
    }
    button.active { border-color:var(--teal); background:var(--soft-teal); color:#0a5962; font-weight:700; }
    .system-map {
      background:#fbfdff;
      border:1px solid var(--line);
      box-shadow:var(--shadow);
      padding:18px;
      border-radius:10px;
    }
    .map-row { display:grid; grid-template-columns:1fr 32px 1fr 32px 1fr; align-items:center; gap:8px; }
    .node {
      border:1px solid var(--line);
      border-radius:8px;
      background:#fff;
      padding:12px;
      min-height:96px;
      display:flex;
      flex-direction:column;
      justify-content:center;
    }
    .node strong { display:block; font-size:16px; }
    .node small { color:var(--muted); margin-top:4px; }
    .node.dso { border-color:#a6bfdb; background:var(--soft-blue); }
    .node.vpp { border-color:#a5d3d7; background:var(--soft-teal); }
    .node.grid { border-color:#e8c989; background:var(--soft-amber); }
    .arrow { height:2px; background:var(--muted); position:relative; }
    .arrow::after { content:""; position:absolute; right:-1px; top:-5px; border-left:9px solid var(--muted); border-top:6px solid transparent; border-bottom:6px solid transparent; }
    main { padding:22px 0 42px; }
    section { margin:22px 0; }
    .summary-grid { display:grid; grid-template-columns:repeat(3, 1fr); gap:14px; }
    .summary {
      background:var(--paper);
      border:1px solid var(--line);
      border-radius:10px;
      padding:16px;
      box-shadow:var(--shadow);
      min-height:162px;
    }
    .status {
      display:inline-flex;
      align-items:center;
      gap:7px;
      padding:5px 9px;
      border-radius:999px;
      font-size:13px;
      font-weight:700;
      margin-bottom:10px;
    }
    .status.fixed { color:#0d5c39; background:var(--soft-green); border:1px solid #b9dec7; }
    .status.guard { color:#7a4300; background:var(--soft-amber); border:1px solid #edcf91; }
    .status.watch { color:#72302b; background:var(--soft-red); border:1px solid #efb8b3; }
    .dot { width:8px; height:8px; border-radius:50%; background:currentColor; display:inline-block; }
    .panel {
      background:#fff;
      border:1px solid var(--line);
      border-radius:10px;
      box-shadow:var(--shadow);
      padding:18px;
    }
    .tabs { display:flex; flex-wrap:wrap; gap:8px; margin-bottom:16px; }
    .tab-panel { display:none; }
    .tab-panel.active { display:block; }
    .compare-toggle { display:flex; gap:8px; margin:10px 0 16px; }
    .before-after { display:grid; grid-template-columns:1fr 1fr; gap:16px; }
    .before, .after {
      border:1px solid var(--line);
      border-radius:10px;
      padding:16px;
      background:#fff;
    }
    .before { border-top:5px solid var(--red); }
    .after { border-top:5px solid var(--green); }
    .focus-before .after, .focus-after .before { opacity:.28; }
    .flow { display:grid; gap:10px; }
    .flow-step {
      display:grid;
      grid-template-columns:34px 1fr;
      gap:10px;
      align-items:start;
      border:1px solid #e2e9f0;
      background:#fbfdff;
      border-radius:8px;
      padding:11px;
    }
    .num {
      width:28px;
      height:28px;
      border-radius:50%;
      display:grid;
      place-items:center;
      color:#fff;
      font-weight:700;
      background:var(--blue);
    }
    .after .num { background:var(--green); }
    .before .num { background:var(--red); }
    .mini-grid { display:grid; grid-template-columns:repeat(4, 1fr); gap:8px; margin-top:12px; }
    .bus {
      border:1px solid var(--line);
      background:#fff;
      border-radius:8px;
      padding:10px;
      min-height:92px;
    }
    .bus strong { display:block; margin-bottom:6px; }
    .bar {
      height:9px;
      background:#e8eef5;
      border-radius:999px;
      overflow:hidden;
      border:1px solid #d7e2ed;
      margin-top:7px;
    }
    .bar span { display:block; height:100%; background:var(--teal); width:55%; }
    .bar.hot span { background:var(--red); width:88%; }
    .bar.safe span { background:var(--green); width:42%; }
    .legend { display:flex; flex-wrap:wrap; gap:10px; margin-top:10px; color:var(--muted); font-size:14px; }
    .legend i { display:inline-block; width:13px; height:13px; border-radius:3px; margin-right:5px; vertical-align:-2px; }
    .split { display:grid; grid-template-columns:.9fr 1.1fr; gap:16px; }
    .term-list { display:grid; grid-template-columns:repeat(2, 1fr); gap:10px; }
    .term {
      border:1px solid var(--line);
      border-radius:8px;
      padding:12px;
      background:#fbfdff;
    }
    .term strong { color:#10243d; }
    table { width:100%; border-collapse:collapse; table-layout:fixed; background:#fff; }
    th, td { border-bottom:1px solid #e4ebf2; padding:10px 9px; text-align:left; vertical-align:top; overflow-wrap:anywhere; }
    th { color:#20354d; background:#f1f5fa; }
    details { border:1px solid var(--line); border-radius:8px; padding:10px 12px; background:#fbfdff; }
    details + details { margin-top:8px; }
    summary { cursor:pointer; font-weight:700; }
    .callout {
      border-left:5px solid var(--amber);
      background:#fffaf0;
      padding:13px 14px;
      border-radius:8px;
      color:#60410f;
    }
    .small { font-size:13px; color:var(--muted); }
    @media (max-width:900px) {
      .hero, .summary-grid, .before-after, .split, .term-list { grid-template-columns:1fr; }
      .map-row { grid-template-columns:1fr; }
      .arrow { width:2px; height:22px; justify-self:center; }
      .arrow::after { right:-5px; top:auto; bottom:-1px; border-top:9px solid var(--muted); border-left:6px solid transparent; border-right:6px solid transparent; border-bottom:0; }
      h1 { font-size:28px; }
    }
  </style>
</head>
<body>
<header>
  <div class="wrap hero">
    <div class="hero-copy">
      <div class="eyebrow">Paper-long 前高严重度修复说明</div>
      <h1>这次不是简单调参数，而是把 DSO-VPP 调度前后的安全链条补完整</h1>
      <p>页面用“改动前/改动后”的方式解释三项核心修复：多母线 VPP 不再被压成一个总功率、调度结果必须经过 AC 潮流校验证书、旧 oracle proxy 不再被误认为 OPF 或性能上界。</p>
      <div class="hero-actions">
        <a class="jump" href="#changes">查看三项改动</a>
        <a class="jump" href="#evidence">查看证据路径</a>
        <a class="jump" href="#meaning">这对实验意味着什么</a>
      </div>
    </div>
    <div class="system-map" aria-label="DSO VPP grid map">
      <div class="map-row">
        <div class="node dso"><strong>DSO 全局智能体</strong><small>给 VPP 发目标功率或价格/边界信号</small></div>
        <div class="arrow"></div>
        <div class="node vpp"><strong>多个 VPP</strong><small>把目标分配给储能、PV、EVCS、柔性负荷</small></div>
        <div class="arrow"></div>
        <div class="node grid"><strong>真实配电网</strong><small>电压、线路、变压器和反向潮流必须安全</small></div>
      </div>
    </div>
  </div>
</header>

<main class="wrap">
  <section class="summary-grid" aria-label="fix summary">
    <article class="summary">
      <span class="status fixed"><span class="dot"></span>已修复</span>
      <h3>多节点 VPP 边界</h3>
      <p>旧逻辑把一个跨多条母线的 VPP 当成一个总功率看。现在先保持 RL 动作不变，再在仿真内部拆成母线/区域级目标，避免一个节点安全、另一个节点越限被总量掩盖。</p>
    </article>
    <article class="summary">
      <span class="status fixed"><span class="dot"></span>已修复</span>
      <h3>AC 潮流校验外壳</h3>
      <p>DOE/FR 投影不再被当成真实安全证据。每一步候选调度会先用 pandapower AC 潮流复核，必要时从当前安全点向候选点回退。</p>
    </article>
    <article class="summary">
      <span class="status guard"><span class="dot"></span>已加防误导</span>
      <h3>论文基线声明边界</h3>
      <p>旧 <code>opf_oracle_proxy</code> 明确降级为静态 FR 价格启发式。新增 <code>ac_validated_search_reference</code>，但仍标记为“不是上界”。</p>
    </article>
  </section>

  <section id="changes" class="panel">
    <h2>三项关键改动</h2>
    <div class="tabs" role="tablist">
      <button class="active" data-tab="fix1">1. 多母线 DOE</button>
      <button data-tab="fix2">2. AC 校验证书</button>
      <button data-tab="fix3">3. 基线声明边界</button>
    </div>

    <article id="fix1" class="tab-panel active">
      <h3>改动 1：从“VPP 总量安全”改成“每个电气位置都要有边界”</h3>
      <div class="compare-toggle">
        <button class="active" data-focus="both">同时看</button>
        <button data-focus="before">只看改动前</button>
        <button data-focus="after">只看改动后</button>
      </div>
      <div class="before-after" data-compare>
        <div class="before">
          <h4>改动前</h4>
          <div class="flow">
            <div class="flow-step"><span class="num">1</span><div><strong>DSO 给一个总功率</strong><p>例如 VPP-A 总共出力 3 MW。</p></div></div>
            <div class="flow-step"><span class="num">2</span><div><strong>总量落入 FR/DOE</strong><p>只检查总 P 是否在聚合边界内。</p></div></div>
            <div class="flow-step"><span class="num">3</span><div><strong>DER 自行分摊</strong><p>可能把功率集中到电气上更脆弱的母线。</p></div></div>
          </div>
          <div class="mini-grid">
            <div class="bus"><strong>Bus 11</strong>安全<div class="bar safe"><span></span></div></div>
            <div class="bus"><strong>Bus 17</strong>接近越限<div class="bar hot"><span></span></div></div>
            <div class="bus"><strong>Bus 23</strong>低负荷<div class="bar"><span></span></div></div>
            <div class="bus"><strong>总量</strong>看起来可行<div class="bar safe"><span></span></div></div>
          </div>
        </div>
        <div class="after">
          <h4>改动后</h4>
          <div class="flow">
            <div class="flow-step"><span class="num">1</span><div><strong>RL 动作仍是总功率</strong><p>不破坏 HAPPO/HATRPO/MATD3/HASAC 的动作维度。</p></div></div>
            <div class="flow-step"><span class="num">2</span><div><strong>内部拆成母线/区域目标</strong><p>根据 FR scope 和当前出力，把总目标拆到 bus/zone。</p></div></div>
            <div class="flow-step"><span class="num">3</span><div><strong>组内分摊给 DER</strong><p>Bus 17 的资源只负责 Bus 17 的局部目标，不再被其他母线抵消风险。</p></div></div>
          </div>
          <div class="mini-grid">
            <div class="bus"><strong>Bus 11</strong>本地边界<div class="bar safe"><span></span></div></div>
            <div class="bus"><strong>Bus 17</strong>本地收紧<div class="bar safe"><span></span></div></div>
            <div class="bus"><strong>Bus 23</strong>本地边界<div class="bar safe"><span></span></div></div>
            <div class="bus"><strong>总量</strong>由向量合成<div class="bar safe"><span></span></div></div>
          </div>
        </div>
      </div>
    </article>

    <article id="fix2" class="tab-panel">
      <h3>改动 2：DOE/FR 不是安全证明，AC 潮流校验才是执行前最后一关</h3>
      <div class="split">
        <div>
          <div class="callout"><strong>一句话理解：</strong>FR/DOE 像是“导航给出的建议路线”，AC 潮流校验像是“真正开车前检查这条路有没有封路、限高、拥堵”。建议路线不能替代真实检查。</div>
          <div class="term-list" style="margin-top:14px">
            <div class="term"><strong>Projection gap</strong><p>动作被边界裁剪了多少。它只能说明“被裁剪”，不能说明“裁剪后一定安全”。</p></div>
            <div class="term"><strong>AC certificate</strong><p>候选调度写入网络副本，跑 AC 潮流，检查电压、线路、变压器。</p></div>
            <div class="term"><strong>Backoff</strong><p>候选点不安全时，从当前安全点向候选点折中搜索，找最大安全步长。</p></div>
            <div class="term"><strong>Post-AC violation</strong><p>真正执行后的越限统计，是 paper-long 里必须关注的安全指标。</p></div>
          </div>
        </div>
        <div class="flow">
          <div class="flow-step"><span class="num">1</span><div><strong>收集所有 VPP 候选调度</strong><p>不再每个 VPP 单独写入后马上潮流，而是先形成联合动作。</p></div></div>
          <div class="flow-step"><span class="num">2</span><div><strong>写入网络副本并跑 AC 潮流</strong><p>检查 <code>vm_pu</code>、线路 loading、变压器 loading、潮流是否收敛。</p></div></div>
          <div class="flow-step"><span class="num">3</span><div><strong>安全则接受，不安全则回退</strong><p>记录 <code>accepted_alpha</code>、<code>repair_gap_mw</code>、<code>ac_certificate_status</code>。</p></div></div>
          <div class="flow-step"><span class="num">4</span><div><strong>安全调度再写入真实仿真</strong><p>奖励和日志里能看到 AC 外壳介入了多少。</p></div></div>
        </div>
      </div>
    </article>

    <article id="fix3" class="tab-panel">
      <h3>改动 3：把“看起来像 oracle 的启发式”从论文上界语言中剥离</h3>
      <table>
        <thead><tr><th>对象</th><th>现在怎么定义</th><th>可以怎么写</th><th>不能怎么写</th></tr></thead>
        <tbody>
          <tr><td><code>static_fr_price_extreme_proxy</code></td><td>按价格取静态 FR 上下界或中点</td><td>静态 FR 价格启发式 baseline</td><td>OPF、oracle、upper bound</td></tr>
          <tr><td><code>opf_oracle_proxy</code></td><td>兼容旧名字的别名，明确标记 not OPF</td><td>legacy alias</td><td>作为最优调度证明</td></tr>
          <tr><td><code>ac_validated_search_reference</code></td><td>有限候选集 + AC 潮流校验 + 最低成本安全候选</td><td>AC 校验搜索参考</td><td>穷举最优、全局最优上界</td></tr>
        </tbody>
      </table>
      <p class="small">实验表现在会输出 <code>baseline_role</code>、<code>is_ac_validated</code>、<code>is_search_based</code>、<code>is_upper_bound_claim_allowed</code>、<code>feasible_candidate_count</code> 等字段，防止后续写论文时误用。</p>
    </article>
  </section>

  <section id="meaning" class="panel">
    <h2>这对你的强化学习实验意味着什么</h2>
    <div class="summary-grid">
      <article class="summary">
        <span class="status fixed"><span class="dot"></span>不会破坏动作接口</span>
        <h3>RL 仍然输出原来的动作</h3>
        <p>DSO/VPP actor 的动作维度没有被强行改大。多母线拆分和 AC 校验在 simulator 内部发生，因此旧 checkpoint 和 replay 协议更容易保持兼容。</p>
      </article>
      <article class="summary">
        <span class="status guard"><span class="dot"></span>会暴露真实安全代价</span>
        <h3>策略不能再靠 projection gap 假安全</h3>
        <p>如果策略经常要求不安全动作，日志会显示 AC 回退、修复 gap 和证书状态。训练是否真的理解配电网物理特征，将通过 post-AC 指标体现。</p>
      </article>
      <article class="summary">
        <span class="status watch"><span class="dot"></span>仍需谨慎</span>
        <h3>这还不是 AC OPF</h3>
        <p>安全外壳是仿真校验和回退，不是智能优化器。paper-long 可以跑，但最优性、市场出清上界、全局安全可行域仍不能过度声明。</p>
      </article>
    </div>
  </section>

  <section id="evidence" class="panel">
    <h2>证据路径与检查点</h2>
    <details open>
      <summary>多母线/区域 DOE 与组内分摊</summary>
      <table>
        <tbody>
          <tr><td>FR 工具</td><td><code>src/vpp_dso_sim/optimization/feasibility_region.py</code> 新增 <code>current_power_by_fr_scope</code>、<code>scalar_target_to_vector_targets</code></td></tr>
          <tr><td>VPP 聚合器</td><td><code>src/vpp_dso_sim/entities/vpp.py</code> 新增 <code>disaggregate_power_targets_by_scope</code></td></tr>
          <tr><td>仿真 trace</td><td><code>projection_trace.stage_name = bus_vector_doe</code>，scope 行包含 bus/zone/der/pcc</td></tr>
        </tbody>
      </table>
    </details>
    <details>
      <summary>AC 潮流校验证书与回退</summary>
      <table>
        <tbody>
          <tr><td>安全证书模块</td><td><code>src/vpp_dso_sim/optimization/ac_security_projection.py</code></td></tr>
          <tr><td>仿真接入</td><td><code>src/vpp_dso_sim/simulation/simulator.py</code> 在联合候选调度后调用 <code>certify_or_repair_dispatch</code></td></tr>
          <tr><td>奖励字段</td><td><code>ac_certified_projection_gap_mw</code>、<code>ac_certificate_failed_count</code></td></tr>
          <tr><td>trace 字段</td><td><code>ac_certificate_status</code>、<code>accepted_alpha</code>、<code>repair_gap_mw</code></td></tr>
        </tbody>
      </table>
    </details>
    <details>
      <summary>论文基线和诊断硬门槛</summary>
      <table>
        <tbody>
          <tr><td>AC 校验搜索参考</td><td><code>src/vpp_dso_sim/optimization/oracle_baseline.py</code></td></tr>
          <tr><td>paper-long 实验协议</td><td><code>src/vpp_dso_sim/experiments/paper_training.py</code> 新增 claim metadata 和诊断项</td></tr>
          <tr><td>阻断项</td><td><code>paper_claim_blocked</code>、<code>oracle_proxy_not_upper_bound</code>、<code>ac_reference_not_upper_bound</code></td></tr>
        </tbody>
      </table>
    </details>
  </section>
</main>

<script>
  const tabs = document.querySelectorAll('[data-tab]');
  tabs.forEach(button => {
    button.addEventListener('click', () => {
      tabs.forEach(item => item.classList.remove('active'));
      document.querySelectorAll('.tab-panel').forEach(panel => panel.classList.remove('active'));
      button.classList.add('active');
      document.getElementById(button.dataset.tab).classList.add('active');
    });
  });
  const compare = document.querySelector('[data-compare]');
  document.querySelectorAll('[data-focus]').forEach(button => {
    button.addEventListener('click', () => {
      document.querySelectorAll('[data-focus]').forEach(item => item.classList.remove('active'));
      button.classList.add('active');
      compare.classList.remove('focus-before', 'focus-after');
      if (button.dataset.focus === 'before') compare.classList.add('focus-before');
      if (button.dataset.focus === 'after') compare.classList.add('focus-after');
    });
  });
</script>
</body>
</html>
"""


__all__ = ["build_high_severity_fix_explainer_html"]
