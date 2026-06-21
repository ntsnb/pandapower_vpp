from __future__ import annotations

from datetime import datetime
import html
from pathlib import Path
from typing import Any

import pandas as pd


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return float(default)
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _mean(frame: pd.DataFrame, column: str) -> float:
    if frame.empty or column not in frame:
        return 0.0
    return float(pd.to_numeric(frame[column], errors="coerce").fillna(0.0).mean())


def _sum(frame: pd.DataFrame, column: str) -> float:
    if frame.empty or column not in frame:
        return 0.0
    return float(pd.to_numeric(frame[column], errors="coerce").fillna(0.0).sum())


def _fmt(value: float) -> str:
    if abs(value) >= 1000.0:
        return f"{value:,.1f}"
    if abs(value) >= 1.0:
        return f"{value:.4f}"
    if abs(value) >= 1.0e-4:
        return f"{value:.6f}"
    return f"{value:.2e}" if value else "0"


def _card(
    *,
    role: str,
    title: str,
    key: str,
    value: float,
    share: float,
    formula: str,
    explanation: str,
    status: str = "审计均值",
) -> dict[str, Any]:
    return {
        "role": role,
        "title": title,
        "key": key,
        "value": float(value),
        "share": float(share),
        "global_abs_share": 0.0,
        "role_abs_total": 0.0,
        "share_scope": "role_internal_abs_share",
        "formula": formula,
        "explanation": explanation,
        "status": status,
    }


def _component_cards(step_metrics: pd.DataFrame, dispatch_trace: pd.DataFrame) -> list[dict[str, Any]]:
    raw_cards = [
        _card(
            role="DSO",
            title="DSO 安全门控",
            key="dso_safety_gate",
            value=_mean(step_metrics, "dso_safety_gate"),
            share=0.0,
            formula="exp(-kappa * max(raw_safety_input, projected_safety_norm))",
            explanation="越接近 1 表示安全层越少削弱 welfare；越接近 0 表示 raw 或 projected 动作仍有安全风险。",
        ),
        _card(
            role="DSO",
            title="DSO 福利输入",
            key="dso_vpp_welfare_raw",
            value=_mean(step_metrics, "dso_vpp_welfare_raw"),
            share=0.0,
            formula="sum(VPP operational surplus excluding service/availability transfers)",
            explanation="只使用剔除转移支付后的 VPP operational surplus，避免把 DSO 付给 VPP 的钱重复当成社会福利。",
        ),
        _card(
            role="DSO",
            title="Raw 动作安全成本",
            key="raw_action_safety_cost_norm",
            value=_mean(step_metrics, "raw_action_safety_cost_norm"),
            share=0.0,
            formula="normalized(raw voltage + line + trafo + powerflow costs)",
            explanation="用于判断 actor 原始动作是否已经学会安全，不被 safety projection 的兜底效果掩盖。",
        ),
        _card(
            role="DSO",
            title="Projected 动作安全成本",
            key="projected_action_safety_cost_norm",
            value=_mean(step_metrics, "projected_action_safety_cost_norm"),
            share=0.0,
            formula="normalized(projected/post-AC voltage + line + trafo + powerflow costs)",
            explanation="用于判断投影后执行动作是否仍接近真实潮流安全边界。",
        ),
        _card(
            role="dispatch",
            title="私有利润 / 运行盈余",
            key="private_profit_proxy",
            value=_mean(step_metrics, "private_profit_proxy")
            or _mean(dispatch_trace, "private_profit_proxy"),
            share=0.0,
            formula="operational_surplus + enabled transfers - contract_penalty",
            explanation="Reward V3.1 主实验中 service/availability/contract 默认关闭，因此该项应主要等于 DER-level operational surplus。",
        ),
        _card(
            role="dispatch",
            title="Service Payment",
            key="service_payment",
            value=_mean(step_metrics, "service_payment"),
            share=0.0,
            formula="0 in main Reward V3.1 experiments",
            explanation="service/availability/contract 默认关闭；若非 0，说明当前实验是 proxy 或合同机制消融，不应混入主实验口径。",
        ),
        _card(
            role="dispatch",
            title="Availability Payment",
            key="availability_payment",
            value=_mean(step_metrics, "availability_payment"),
            share=0.0,
            formula="0 in main Reward V3.1 experiments",
            explanation="没有容量合同和机会成本建模前，不把可用性费作为真实市场收益。",
        ),
        _card(
            role="dispatch",
            title="投影惩罚",
            key="dispatch_projection_penalty",
            value=_mean(step_metrics, "dispatch_projection_penalty"),
            share=0.0,
            formula="linear_weight * gap + quadratic_weight * gap^2",
            explanation="如果该项长期较大，说明 dispatch actor 输出大量不可行动作，策略梯度可能被裁剪/投影扭曲。",
        ),
        _card(
            role="dispatch",
            title="储能跨时 shaping",
            key="storage_potential_shaping_reward",
            value=_mean(step_metrics, "storage_potential_shaping_reward"),
            share=0.0,
            formula="weight * (gamma * Phi(next_soc) - Phi(current_soc))",
            explanation="该项必须是净未来价值，且不能鼓励每步无意义充电；若长期为正且伴随 SOC 单调升高，应暂停检查。",
        ),
        _card(
            role="settlement",
            title="Settlement 审计完整率",
            key="settlement_audit_complete",
            value=_mean(step_metrics, "settlement_audit_complete"),
            share=0.0,
            formula="mean(required DER-level settlement fields present)",
            explanation="paper-long 主实验要求该值为 1，否则 reward 口径不能支撑论文结论。",
        ),
        _card(
            role="settlement",
            title="功率平衡通过率",
            key="settlement_power_balance_ok",
            value=_mean(step_metrics, "settlement_power_balance_ok"),
            share=0.0,
            formula="abs(audit_reconstructed_p_mw - delivered_p_mw) <= tolerance",
            explanation="如果该值低于 1，说明 DER-level 账本和实际注入不一致，必须先修复。",
        ),
        _card(
            role="portfolio",
            title="组合窗口利润",
            key="portfolio_window_profit",
            value=_mean(step_metrics, "portfolio_window_profit"),
            share=0.0,
            formula="mean dispatch private profit over portfolio decision window",
            explanation="慢时标 portfolio agent 看到的长期收益代理，不能每个 15min step 随意改变 DER 物理接入。",
        ),
        _card(
            role="portfolio",
            title="组合切换成本",
            key="portfolio_switching_cost",
            value=_mean(step_metrics, "portfolio_switching_cost"),
            share=0.0,
            formula="keep/reweight/membership-change switching cost",
            explanation="该项过大时 portfolio agent 会倾向不动；过小时会频繁重选组合。",
        ),
    ]
    total_abs = sum(abs(card["value"]) for card in raw_cards) or 1.0
    role_totals: dict[str, float] = {}
    for card in raw_cards:
        role = str(card["role"])
        role_totals[role] = role_totals.get(role, 0.0) + abs(float(card["value"]))
    for card in raw_cards:
        role_total = role_totals.get(str(card["role"]), 0.0)
        card["global_abs_share"] = abs(float(card["value"])) / total_abs
        card["role_abs_total"] = float(role_total)
        card["share"] = abs(float(card["value"])) / role_total if role_total else 0.0
    return raw_cards


def write_reward_dynamic_episode_report(
    *,
    output_dir: str | Path,
    algorithm: str,
    episode: int,
    step_metrics: pd.DataFrame,
    episode_metrics: pd.DataFrame,
    dispatch_trace: pd.DataFrame | None = None,
    update_metrics: pd.DataFrame | None = None,
) -> Path:
    """Write one self-contained Chinese reward card report for a training episode."""

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    dispatch_trace = dispatch_trace if dispatch_trace is not None else pd.DataFrame()
    update_metrics = update_metrics if update_metrics is not None else pd.DataFrame()
    episode_metrics = episode_metrics if episode_metrics is not None else pd.DataFrame()
    cards = _component_cards(step_metrics, dispatch_trace)
    share_path = out / f"reward_component_abs_share_{algorithm}_episode_{int(episode):04d}.csv"
    pd.DataFrame(cards).to_csv(share_path, index=False)
    path = out / f"reward_dynamic_cards_{algorithm}_episode_{int(episode):04d}.html"
    latest = out / "latest_reward_dynamic_cards.html"

    html_text = _render_html(
        algorithm=str(algorithm),
        episode=int(episode),
        cards=cards,
        step_metrics=step_metrics,
        episode_metrics=episode_metrics,
        update_metrics=update_metrics,
        share_csv=share_path.name,
    )
    path.write_text(html_text, encoding="utf-8")
    latest.write_text(html_text, encoding="utf-8")
    return path


def _render_html(
    *,
    algorithm: str,
    episode: int,
    cards: list[dict[str, Any]],
    step_metrics: pd.DataFrame,
    episode_metrics: pd.DataFrame,
    update_metrics: pd.DataFrame,
    share_csv: str,
) -> str:
    episode_reward = _mean(episode_metrics, "episode_reward") or _sum(step_metrics, "reward")
    violation_count = _mean(episode_metrics, "violation_count") or _sum(step_metrics, "violation_count")
    projection_gap = _mean(episode_metrics, "projection_gap_mw") or _sum(step_metrics, "projection_gap_mw")
    critic_loss = _mean(episode_metrics, "critic_loss") or _mean(update_metrics, "critic_loss")
    generated = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    card_html = "\n".join(_render_card(card) for card in cards)
    return f"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Reward 动态卡片看板 - {html.escape(algorithm)} Episode {episode}</title>
<style>
:root{{--bg:#071013;--panel:#102027;--line:#29434e;--text:#edf7f6;--muted:#a7c0c9;--accent:#2dd4bf;--warn:#f59e0b;--bad:#fb7185}}
*{{box-sizing:border-box}}body{{margin:0;background:linear-gradient(180deg,#071013,#132f35 52%,#071013);color:var(--text);font-family:-apple-system,BlinkMacSystemFont,"Segoe UI","Noto Sans SC","Microsoft YaHei",sans-serif}}
header{{padding:34px 6vw 22px;border-bottom:1px solid var(--line)}}h1{{margin:0 0 10px;font-size:clamp(26px,4vw,44px)}}.subtitle{{color:#d1e7e5;line-height:1.7;max-width:1180px}}.badge{{display:inline-flex;margin:10px 8px 0 0;border:1px solid var(--line);border-radius:999px;padding:7px 12px;color:#d8f3ef;background:rgba(16,32,39,.84);font-size:13px}}
main{{padding:24px 6vw 56px}}.grid{{display:grid;gap:16px}}.kpis{{grid-template-columns:repeat(4,minmax(180px,1fr));margin-bottom:22px}}.kpi,.notice,.card{{background:rgba(16,32,39,.92);border:1px solid var(--line);border-radius:14px}}.kpi{{padding:16px}}.kpi .label{{color:var(--muted);font-size:13px}}.kpi .num{{font-size:28px;font-weight:850;margin-top:7px}}.notice{{padding:16px;line-height:1.75;color:#fde68a;background:rgba(245,158,11,.10);border-color:rgba(245,158,11,.38);margin-bottom:22px}}
.toolbar{{display:flex;gap:12px;align-items:center;flex-wrap:wrap;margin:12px 0 16px}}input{{width:min(620px,100%);border:1px solid var(--line);background:#071013;color:var(--text);border-radius:12px;padding:12px 14px}}.tabs{{display:flex;gap:9px;flex-wrap:wrap;margin-bottom:16px}}.tab{{border:1px solid var(--line);background:#102027;color:#d1e7e5;border-radius:999px;padding:8px 13px;font-weight:800;cursor:pointer}}.tab.active{{background:var(--accent);color:#05201c;border-color:transparent}}
.cards{{grid-template-columns:repeat(auto-fill,minmax(320px,1fr))}}.card{{overflow:hidden}}.card:before{{content:"";display:block;height:4px;background:linear-gradient(90deg,var(--accent),#60a5fa)}}.card-head{{padding:16px}}.title-row{{display:flex;justify-content:space-between;gap:10px;align-items:flex-start}}h3{{margin:0;font-size:18px}}.key{{color:var(--muted);font-size:12px;margin-top:5px;word-break:break-all}}.pill{{white-space:nowrap;border-radius:999px;padding:5px 9px;background:var(--accent);color:#05201c;font-weight:850;font-size:12px}}.metrics{{display:grid;grid-template-columns:1fr 1fr;gap:10px;margin-top:14px}}.metric{{border:1px solid var(--line);border-radius:12px;padding:10px;background:rgba(7,16,19,.55)}}.m-label{{color:var(--muted);font-size:12px}}.m-value{{font-weight:850;font-size:18px;margin-top:4px}}.bar{{height:8px;border-radius:999px;background:#243b42;overflow:hidden;margin-top:8px}}.bar div{{height:100%;width:var(--w);background:linear-gradient(90deg,var(--accent),#60a5fa)}}details{{border-top:1px solid var(--line);padding:0 16px}}summary{{cursor:pointer;padding:12px 0;font-weight:850}}.detail{{color:#d1e7e5;line-height:1.65;padding-bottom:14px;font-size:14px}}.formula{{display:block;font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;background:#071013;border:1px solid var(--line);border-radius:10px;padding:9px;overflow-x:auto;color:#a7f3d0;font-size:12px;line-height:1.55}}
footer{{padding:20px 6vw 40px;color:var(--muted);border-top:1px solid var(--line);font-size:13px;line-height:1.7}}
@media(max-width:900px){{.kpis{{grid-template-columns:repeat(2,minmax(160px,1fr))}}}}@media(max-width:620px){{header,main,footer{{padding-left:18px;padding-right:18px}}.kpis,.cards{{grid-template-columns:1fr}}.metrics{{grid-template-columns:1fr}}}}
</style>
</head>
<body>
<header>
<h1>Reward 动态卡片看板</h1>
<div class="subtitle">用于 paper-long 训练过程中逐 episode 审查 reward 是否健康。当前页面由训练循环在 episode 结束时自动生成，不是旧版静态示例页。</div>
<div>
<span class="badge">算法：{html.escape(algorithm)}</span>
<span class="badge">Episode {episode}</span>
<span class="badge">生成：{html.escape(generated)}</span>
<span class="badge">占比口径：同类智能体内绝对占比</span>
<span class="badge">占比 CSV：{html.escape(share_csv)}</span>
</div>
</header>
<main>
<section class="grid kpis">
<div class="kpi"><div class="label">Episode reward</div><div class="num">{_fmt(episode_reward)}</div></div>
<div class="kpi"><div class="label">违规计数</div><div class="num">{_fmt(violation_count)}</div></div>
<div class="kpi"><div class="label">投影/盾牌 gap</div><div class="num">{_fmt(projection_gap)}</div></div>
<div class="kpi"><div class="label">Critic loss</div><div class="num">{_fmt(critic_loss)}</div></div>
</section>
<div class="notice"><b>Reward V3.1 口径提醒：</b>service/availability/contract 默认关闭；EVCS 收入必须来自 DER-level settlement audit；DSO 使用剔除转移支付后的 operational surplus；DSO 卡片是安全与福利审计口径，不把父子诊断项重复相加。卡片中的“同类智能体内绝对占比”按 role 分母计算：DSO 只和 DSO 项比较，dispatch 只和 dispatch 项比较，portfolio 只和 portfolio 项比较；CSV 中另保留 global_abs_share 作为全页量级参考。</div>
<div class="toolbar"><input id="search" placeholder="搜索：DSO、dispatch、settlement、storage、safety、service、availability、projection..."></div>
<div class="tabs">
<button class="tab active" data-filter="all">全部</button>
<button class="tab" data-filter="DSO">DSO 安全与福利</button>
<button class="tab" data-filter="dispatch">VPP Dispatch</button>
<button class="tab" data-filter="settlement">Settlement 审计</button>
<button class="tab" data-filter="portfolio">Portfolio</button>
</div>
<section id="cards" class="grid cards">
{card_html}
</section>
</main>
<footer>每个 episode 自动生成一份 HTML，并覆盖 latest_reward_dynamic_cards.html 便于持续刷新查看。</footer>
<script>
const search = document.getElementById('search');
const tabs = [...document.querySelectorAll('.tab')];
const cards = [...document.querySelectorAll('.card')];
let active = 'all';
function applyFilter(){{
  const q = (search.value || '').toLowerCase();
  for (const card of cards){{
    const roleOk = active === 'all' || card.dataset.role === active;
    const textOk = !q || card.dataset.text.includes(q);
    card.style.display = roleOk && textOk ? '' : 'none';
  }}
}}
search.addEventListener('input', applyFilter);
tabs.forEach(tab => tab.addEventListener('click', () => {{
  tabs.forEach(item => item.classList.remove('active'));
  tab.classList.add('active');
  active = tab.dataset.filter;
  applyFilter();
}}));
</script>
</body>
</html>
"""


def _render_card(card: dict[str, Any]) -> str:
    share_pct = 100.0 * float(card["share"])
    global_share_pct = 100.0 * float(card.get("global_abs_share", 0.0))
    value = float(card["value"])
    role = str(card["role"])
    title = str(card["title"])
    key = str(card["key"])
    text = " ".join(str(card.get(part, "")) for part in ("role", "title", "key", "formula", "explanation")).lower()
    return f"""
<article class="card" data-role="{html.escape(role)}" data-text="{html.escape(text)}">
  <div class="card-head">
    <div class="title-row">
      <div>
        <h3>{html.escape(title)}</h3>
        <div class="key">{html.escape(key)}</div>
      </div>
      <span class="pill">{html.escape(role)}</span>
    </div>
    <div class="metrics">
      <div class="metric"><div class="m-label">均值 / signed mean</div><div class="m-value">{html.escape(_fmt(value))}</div></div>
      <div class="metric"><div class="m-label">同类智能体内绝对占比</div><div class="m-value">{share_pct:.2f}%</div><div class="bar"><div style="--w:{min(100.0, share_pct):.2f}%"></div></div></div>
      <div class="metric"><div class="m-label">数据状态</div><div class="m-value">{html.escape(str(card['status']))}</div></div>
      <div class="metric"><div class="m-label">全页量级参考</div><div class="m-value">{global_share_pct:.2f}%</div></div>
      <div class="metric"><div class="m-label">字段名</div><div class="m-value" style="font-size:13px">{html.escape(key)}</div></div>
    </div>
  </div>
  <details open><summary>公式</summary><div class="detail"><span class="formula">{html.escape(str(card['formula']))}</span></div></details>
  <details><summary>解释 / 风险</summary><div class="detail">{html.escape(str(card['explanation']))}</div></details>
</article>
"""
