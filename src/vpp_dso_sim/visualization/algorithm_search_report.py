from __future__ import annotations

import argparse
from html import escape
import json
import math
from pathlib import Path
from typing import Any

import pandas as pd

from vpp_dso_sim.utils.io import ensure_dir


DEFAULT_OUTPUT_DIR = Path("outputs") / "algorithm_search"
DEFAULT_REPORT_NAME = "algorithm_search_report.html"

ALGORITHM_ALIASES = (
    "algorithm",
    "algorithm_id",
    "candidate",
    "candidate_id",
    "name",
    "method",
    "marl_algorithm",
)
SCORE_ALIASES = (
    "score",
    "proxy_score",
    "total_score",
    "final_score",
    "composite_score",
    "weighted_score",
    "suitability_score",
)
RANK_ALIASES = ("rank", "ranking", "candidate_rank")
DECISION_ALIASES = ("decision", "status", "recommendation_status", "recommendation", "outcome")
KEEP_REASON_ALIASES = (
    "keep_reason",
    "reason_keep",
    "retain_reason",
    "accepted_reason",
    "positive_reason",
    "strengths",
    "fit_reason",
    "why_keep",
)
REJECT_REASON_ALIASES = (
    "reject_reason",
    "reason_reject",
    "drop_reason",
    "defer_reason",
    "risks",
    "limitations",
    "weaknesses",
    "why_reject",
    "rejection_reason",
)

ALGORITHM_FIT_NOTES = {
    "MATD3": {
        "fit_en": (
            "Strong for continuous CTDE control such as DSO envelope tuning and VPP dispatch "
            "setpoints. Twin critics and delayed policy updates help reduce over-estimation in "
            "noisy power-flow feedback."
        ),
        "fit_zh": (
            "适合连续 CTDE 控制，例如 DSO 运行包络调节和 VPP 调度出力。双 critic 与延迟策略"
            "更新有助于缓解潮流反馈噪声下的过估计。"
        ),
        "risk_en": (
            "Less natural for discrete portfolio actions and general-sum settlement unless the "
            "action heads, replay buffer and safety projection are carefully separated."
        ),
        "risk_zh": (
            "对离散组合配置动作和 general-sum 结算目标不够天然，除非动作 head、回放池和安全"
            "投影边界拆分清楚。"
        ),
    },
    "HAPPO": {
        "fit_en": (
            "A good candidate for heterogeneous DSO/VPP roles because each agent can keep a "
            "separate policy while training with centralized advantage information."
        ),
        "fit_zh": (
            "适合异构 DSO/VPP 角色：每类智能体可以保留独立策略，同时训练期使用集中 advantage "
            "信息。"
        ),
        "risk_en": (
            "On-policy sampling can be expensive. It needs stable long-horizon rollouts before it "
            "can support paper-grade claims."
        ),
        "risk_zh": "on-policy 采样成本较高，需要稳定长 horizon rollout 后才适合支撑论文级结论。",
    },
    "HASAC": {
        "fit_en": (
            "Useful when stochastic exploration and entropy regularization are needed for uncertain "
            "local-flex prices, VPP bids and DER availability."
        ),
        "fit_zh": (
            "当局部灵活性价格、VPP 报价和 DER 可用性存在不确定性时，随机策略与熵正则较有价值。"
        ),
        "risk_en": (
            "Exploration can produce unsafe actions without a deterministic FR/DOE shield, and the "
            "off-policy replay design must not leak private VPP observations."
        ),
        "risk_zh": (
            "如果没有确定性的 FR/DOE 安全屏障，探索动作可能不安全；off-policy 回放设计也不能泄露"
            "VPP 私有观测。"
        ),
    },
    "FACMAC": {
        "fit_en": (
            "Attractive for factorized centralized critics in continuous cooperative control, "
            "especially when VPP contributions can be decomposed by zone or service type."
        ),
        "fit_zh": (
            "适合连续协作控制中的因子化集中 critic，尤其是 VPP 贡献可以按 zone 或服务类型分解时。"
        ),
        "risk_en": (
            "The base cooperative assumption is weaker for self-interested VPPs. It needs explicit "
            "settlement, reliability and fairness terms to avoid drifting back to a shared-reward "
            "controller."
        ),
        "risk_zh": (
            "原始协作假设对自利 VPP 较弱，需要显式结算、可靠性和公平性项，避免退回共享 reward "
            "控制器。"
        ),
    },
}

COLUMN_LABELS = {
    "rank": ("Rank", "排名"),
    "rank_value": ("Rank", "排名"),
    "candidate_no": ("Candidate No.", "候选编号"),
    "algorithm_id": ("Algorithm ID", "算法标识"),
    "algorithm_name": ("Algorithm", "算法"),
    "family": ("Family", "算法族"),
    "recommendation_status": ("Decision", "筛选决策"),
    "decision_en": ("Decision EN", "英文决策"),
    "decision_zh": ("Decision ZH", "中文决策"),
    "is_top_k_recommendation": ("Top-K", "是否 Top-K"),
    "score": ("Score", "评分"),
    "score_value": ("Proxy Score", "代理评分"),
    "proxy_score": ("Proxy Score", "代理评分"),
    "reward_fit": ("Reward Fit", "奖励适配度"),
    "privacy_fit": ("Privacy Fit", "隐私边界适配度"),
    "continuous_action_fit": ("Continuous Action Fit", "连续动作适配度"),
    "heterogeneity_fit": ("Heterogeneity Fit", "异构智能体适配度"),
    "risk_penalty": ("Risk Penalty", "风险惩罚"),
    "expected_engineering_lift": ("Engineering Lift", "工程实现成本"),
    "keep_reason": ("Keep Reason", "保留理由"),
    "reject_reason": ("Reject/Defer Reason", "拒绝或暂缓理由"),
    "rejection_reason": ("Reject/Defer Reason", "拒绝或暂缓理由"),
    "idea": ("Design Idea", "设计思路"),
    "action_space": ("Action Space", "动作空间"),
    "reward_mode": ("Reward Mode", "奖励模式"),
    "privacy_mode": ("Privacy Mode", "隐私模式"),
    "heterogeneity_model": ("Heterogeneity Model", "异构建模方式"),
    "engineering_stage": ("Engineering Stage", "工程阶段"),
    "tags": ("Tags", "标签"),
    "registry_source": ("Source", "候选来源"),
    "proxy_budget_label": ("Proxy Budget", "代理预算标签"),
    "notes": ("Notes", "备注"),
}

SCORE_SEMANTICS_ZH = {
    "reward_fit": "是否适配当前 DSO 全局引导、VPP dispatch 和 VPP portfolio 的 role-specific general-sum reward 设计。",
    "privacy_fit": "是否满足执行期隐私边界：DSO 不读取 VPP 私有 DER 状态，VPP 之间不共享私有观测。",
    "continuous_action_fit": "是否适合 DSO 包络、VPP 聚合出力、DER 解聚合等连续动作。",
    "heterogeneity_fit": "是否适合 DSO、VPP dispatch、VPP portfolio 等异构角色和不同 DER 组合。",
    "risk_penalty": "对训练稳定性、隐私泄漏、工程复杂度和论文结论风险的惩罚项，越低越好。",
    "expected_engineering_lift": "下一步实现所需工程量的代理估计，越低越适合作为近期 baseline。",
    "proxy_score": "按当前权重合成的代理筛选分数，只用于排序候选思路，不代表真实训练收益。",
}


def _is_missing(value: object) -> bool:
    if value is None:
        return True
    if isinstance(value, float):
        return math.isnan(value)
    if isinstance(value, (list, tuple, dict, set)):
        return False
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False


def _text(value: object, default: str = "") -> str:
    if _is_missing(value):
        return default
    text = str(value).strip()
    return text if text else default


def _read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _first_existing(columns: pd.Index, aliases: tuple[str, ...]) -> str | None:
    lower = {str(col).lower(): str(col) for col in columns}
    for alias in aliases:
        if alias.lower() in lower:
            return lower[alias.lower()]
    return None


def _first_value(row: pd.Series, aliases: tuple[str, ...], default: str = "") -> str:
    for alias in aliases:
        if alias in row.index:
            value = _text(row.get(alias))
            if value:
                return value
    lower = {str(col).lower(): col for col in row.index}
    for alias in aliases:
        key = lower.get(alias.lower())
        if key is not None:
            value = _text(row.get(key))
            if value:
                return value
    return default


def _lang(en: str, zh: str, *, block: bool = False) -> str:
    kind = "block" if block else "inline"
    return (
        f"<span class='lang-copy lang-{kind} lang-en'>{en}</span>"
        f"<span class='lang-copy lang-{kind} lang-zh'>{zh}</span>"
    )


def _fmt(value: object) -> str:
    if _is_missing(value):
        return ""
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def _column_label(column: str) -> str:
    en, zh = COLUMN_LABELS.get(column, (column, column))
    return _lang(escape(en), escape(zh))


def _decision_label(raw: str, rank: float | None, score: float | None) -> tuple[str, str]:
    lowered = raw.lower()
    if any(
        token in lowered
        for token in ("keep", "accept", "select", "retain", "recommend", "recommended", "保留", "接受", "推荐")
    ):
        return "Keep", "保留"
    if any(token in lowered for token in ("reject", "drop", "defer", "hold", "拒绝", "暂缓")):
        return "Reject/defer", "拒绝/暂缓"
    if rank is not None and rank <= 5:
        return "Keep candidate", "保留候选"
    if score is not None and score >= 0:
        return "Review candidate", "待评审候选"
    return "Needs review", "需要评审"


def _normalise_candidates(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(
            columns=[
                "candidate_no",
                "algorithm_name",
                "score_value",
                "rank_value",
                "decision_en",
                "decision_zh",
                "keep_reason",
                "reject_reason",
            ]
        )

    data = frame.copy()
    algorithm_col = _first_existing(data.columns, ALGORITHM_ALIASES)
    score_col = _first_existing(data.columns, SCORE_ALIASES)
    rank_col = _first_existing(data.columns, RANK_ALIASES)
    decision_col = _first_existing(data.columns, DECISION_ALIASES)

    data["candidate_no"] = range(1, len(data) + 1)
    if algorithm_col is None:
        data["algorithm_name"] = data["candidate_no"].map(lambda idx: f"candidate_{idx:03d}")
    else:
        data["algorithm_name"] = data[algorithm_col].map(lambda value: _text(value, "unnamed"))

    if score_col is None:
        numeric_cols = [
            col
            for col in data.select_dtypes(include="number").columns
            if str(col).lower() not in {*RANK_ALIASES, "candidate_no"}
        ]
        if numeric_cols:
            data["score_value"] = data[numeric_cols].mean(axis=1)
        else:
            data["score_value"] = pd.Series([0.0] * len(data), index=data.index)
    else:
        data["score_value"] = pd.to_numeric(data[score_col], errors="coerce").fillna(0.0)

    if rank_col is None:
        data = data.sort_values(["score_value", "candidate_no"], ascending=[False, True]).reset_index(drop=True)
        data["rank_value"] = range(1, len(data) + 1)
    else:
        data["rank_value"] = pd.to_numeric(data[rank_col], errors="coerce")
        data = data.sort_values(
            ["rank_value", "score_value", "candidate_no"], ascending=[True, False, True]
        ).reset_index(drop=True)
        data["rank_value"] = data["rank_value"].fillna(pd.Series(range(1, len(data) + 1)))

    decisions_en: list[str] = []
    decisions_zh: list[str] = []
    keep_reasons: list[str] = []
    reject_reasons: list[str] = []
    for _, row in data.iterrows():
        raw_decision = _text(row.get(decision_col), "") if decision_col else ""
        rank_value = float(row["rank_value"]) if not _is_missing(row.get("rank_value")) else None
        score_value = float(row["score_value"]) if not _is_missing(row.get("score_value")) else None
        decision_en, decision_zh = _decision_label(raw_decision, rank_value, score_value)
        decisions_en.append(decision_en)
        decisions_zh.append(decision_zh)
        keep_reasons.append(_first_value(row, KEEP_REASON_ALIASES, ""))
        reject_reasons.append(_first_value(row, REJECT_REASON_ALIASES, ""))

    data["decision_en"] = decisions_en
    data["decision_zh"] = decisions_zh
    data["keep_reason"] = keep_reasons
    data["reject_reason"] = reject_reasons
    return data


def _summary_cards(candidates: pd.DataFrame, summary: dict[str, Any]) -> str:
    top = candidates.iloc[0] if not candidates.empty else pd.Series(dtype=object)
    rows = [
        ("Candidates", "候选算法数", len(candidates), "candidate_scores.csv rows", "CSV 行数"),
        (
            "Top Candidate",
            "最高优先候选",
            _text(top.get("algorithm_name"), "n/a"),
            "ranked by rank/score",
            "按 rank/score 排序",
        ),
        (
            "Search Objective",
            "搜索目标",
            summary.get("search_objective", summary.get("objective", "n/a")),
            "summary.json",
            "来自 summary.json",
        ),
        (
            "Selection Policy",
            "筛选口径",
            summary.get("selection_policy", summary.get("policy", "n/a")),
            "keep/reject rationale",
            "保留/拒绝理由",
        ),
    ]
    cards = []
    for en, zh, value, note_en, note_zh in rows:
        cards.append(
            "<article class='metric-card'>"
            f"<span>{_lang(en, zh)}</span>"
            f"<strong>{escape(_fmt(value))}</strong>"
            f"<small>{_lang(note_en, note_zh)}</small>"
            "</article>"
        )
    return "<section class='metric-grid'>" + "".join(cards) + "</section>"


def _candidate_card(row: pd.Series) -> str:
    algorithm = escape(_text(row.get("algorithm_name")))
    keep_reason = escape(_text(row.get("keep_reason"), "No keep reason recorded."))
    reject_reason = escape(_text(row.get("reject_reason"), "No reject/defer reason recorded."))
    return f"""
    <article class="candidate-card" data-rank="{escape(_fmt(row.get('rank_value')))}">
      <div class="candidate-head">
        <span>#{escape(_fmt(row.get("rank_value")))}</span>
        <h3>{algorithm}</h3>
        <strong>{escape(_fmt(row.get("score_value")))}</strong>
      </div>
      <p class="status-pill">{escape(_text(row.get("decision_en")))} / {escape(_text(row.get("decision_zh")))}</p>
      <dl>
        <dt>{_lang("Keep reason", "保留理由")}</dt><dd>{keep_reason}</dd>
        <dt>{_lang("Reject/defer reason", "拒绝/暂缓理由")}</dt><dd>{reject_reason}</dd>
      </dl>
    </article>
    """


def _top_candidates_section(candidates: pd.DataFrame, top_n: int) -> str:
    cards = "".join(_candidate_card(row) for _, row in candidates.head(top_n).iterrows())
    return f"""
    <section class="panel">
      <div class="section-head">
        <div>
          <p class="eyebrow">Top Candidates</p>
          <h2>{_lang("Top Candidates and Rationale", "优先候选与筛选理由")}</h2>
        </div>
        <p>{_lang(
            "Candidates are sorted by explicit rank when present, otherwise by the normalized score.",
            "候选算法优先按显式 rank 排序；没有 rank 时按归一化 score 排序。",
            block=True,
        )}</p>
      </div>
      <div class="candidate-grid">{cards}</div>
    </section>
    """


def _algorithm_fit_section() -> str:
    cards = []
    for algorithm, note in ALGORITHM_FIT_NOTES.items():
        cards.append(
            f"""
            <article class="fit-card" id="fit-{escape(algorithm.lower())}">
              <h3>{escape(algorithm)}</h3>
              <section>
                <strong>{_lang("Why it can fit", "为什么可能适合")}</strong>
                <p>{_lang(escape(note["fit_en"]), escape(note["fit_zh"]), block=True)}</p>
              </section>
              <section>
                <strong>{_lang("Why it may not fit yet", "为什么当前可能不适合")}</strong>
                <p>{_lang(escape(note["risk_en"]), escape(note["risk_zh"]), block=True)}</p>
              </section>
            </article>
            """
        )
    return f"""
    <section class="panel">
      <div class="section-head">
        <div>
          <p class="eyebrow">Advanced MARL</p>
          <h2>{_lang("MATD3 / HAPPO / HASAC / FACMAC Fit Notes", "MATD3 / HAPPO / HASAC / FACMAC 适配说明")}</h2>
        </div>
        <p>{_lang(
            "These notes keep the UI synchronized with the algorithm-search discussion and preserve the privacy, safety-projection and general-sum settlement caveats.",
            "这些说明用于让 UI 与算法搜索讨论同步，并保留隐私、安全投影和 general-sum 结算边界。",
            block=True,
        )}</p>
      </div>
      <div class="fit-grid">{''.join(cards)}</div>
    </section>
    """


def _score_semantics_section() -> str:
    rows = []
    for key, zh in SCORE_SEMANTICS_ZH.items():
        rows.append(
            f"""
            <article class="metric-card">
              <strong>{escape(key)}</strong>
              <small>{_lang("Meaning in this proxy experiment", "本代理实验中的含义")}</small>
              <p>{_lang(
                  escape(key.replace("_", " ")),
                  escape(zh),
                  block=True,
              )}</p>
            </article>
            """
        )
    return f"""
    <section class="panel">
      <div class="section-head">
        <div>
          <p class="eyebrow">Score Dictionary</p>
          <h2>{_lang("Score Field Meaning", "评分字段含义")}</h2>
        </div>
        <p>{_lang(
            "The score is a screening proxy, not a trained-policy return. It only decides which candidate algorithms deserve implementation and full experiments next.",
            "这些评分是候选算法筛选代理量，不是训练后策略回报。它只用于决定哪些算法值得下一步实现并做完整实验。",
            block=True,
        )}</p>
      </div>
      <div class="metric-grid">{''.join(rows)}</div>
    </section>
    """


def _candidate_table(candidates: pd.DataFrame) -> str:
    if candidates.empty:
        return "<p class='muted'>No candidates / 暂无候选算法。</p>"
    visible_cols = [
        "rank_value",
        "algorithm_name",
        "score_value",
        "decision_en",
        "decision_zh",
        "keep_reason",
        "reject_reason",
    ]
    extra_cols = [
        col
        for col in candidates.columns
        if col not in visible_cols and col not in {"candidate_no"}
    ]
    table_cols = visible_cols + extra_cols[:10]
    head = "".join(f"<th>{_column_label(col)}</th>" for col in table_cols)
    rows = []
    for _, row in candidates.iterrows():
        cells = "".join(f"<td>{escape(_fmt(row.get(col)))}</td>" for col in table_cols)
        rows.append(
            f"<tr data-testid='candidate-row' data-algorithm='{escape(_text(row.get('algorithm_name')))}'>"
            f"{cells}</tr>"
        )
    return f"""
    <div class="table-wrap">
      <table>
        <thead><tr>{head}</tr></thead>
        <tbody>{''.join(rows)}</tbody>
      </table>
    </div>
    """


def _summary_section(summary: dict[str, Any]) -> str:
    return f"""
    <section class="panel">
      <div class="section-head">
        <div>
          <p class="eyebrow">Summary JSON</p>
          <h2>{_lang("Search Summary Metadata", "搜索摘要元数据")}</h2>
        </div>
      </div>
      <pre>{escape(json.dumps(summary, ensure_ascii=False, indent=2))}</pre>
    </section>
    """


def _language_toolbar() -> str:
    return """
    <div class="language-toolbar" role="group" aria-label="Language">
      <span>Language / 语言</span>
      <button type="button" class="lang-button" data-lang-switch="en">EN</button>
      <button type="button" class="lang-button is-active" data-lang-switch="zh">中文</button>
    </div>
    """


def _css() -> str:
    return """
    :root {
      --bg:#eef4f8; --panel:#ffffff; --line:#d8e3ee; --text:#132235;
      --muted:#5d7084; --accent:#1769aa; --green:#0f8a5f; --orange:#c76500;
      --shadow:0 14px 32px rgba(15,35,56,0.08);
    }
    * { box-sizing:border-box; }
    body {
      margin:0; font-family:"Segoe UI","Microsoft YaHei",Arial,sans-serif;
      color:var(--text); background:linear-gradient(180deg,#dfeaf3,var(--bg) 38%,#f7fafc);
    }
    .lang-copy { display:none; }
    html[data-lang="en"] .lang-en.lang-inline, html[data-lang="zh"] .lang-zh.lang-inline { display:inline; }
    html[data-lang="en"] .lang-en.lang-block, html[data-lang="zh"] .lang-zh.lang-block { display:block; }
    header { padding:30px 34px; color:white; background:linear-gradient(135deg,#061b30,#13456f); }
    header h1 { margin:6px 0 10px; font-size:clamp(28px,4vw,42px); }
    header p { max-width:980px; margin:0; color:#dce9f5; line-height:1.65; }
    main { padding:24px 30px 42px; }
    .language-toolbar {
      display:inline-flex; align-items:center; gap:8px; padding:6px; margin-top:18px;
      border-radius:999px; border:1px solid rgba(255,255,255,0.18);
      background:rgba(255,255,255,0.09);
    }
    .language-toolbar span { color:#dce9f5; font-size:12px; padding:0 6px; }
    .lang-button {
      border:0; border-radius:999px; padding:7px 12px; background:transparent;
      color:#dce9f5; font-weight:800; cursor:pointer;
    }
    .lang-button.is-active { color:white; background:rgba(143,208,255,0.24); }
    .metric-grid, .candidate-grid, .fit-grid {
      display:grid; grid-template-columns:repeat(auto-fit,minmax(250px,1fr)); gap:12px;
      margin-bottom:18px;
    }
    .panel, .metric-card, .candidate-card, .fit-card {
      background:var(--panel); border:1px solid var(--line); border-radius:8px;
      box-shadow:var(--shadow);
    }
    .panel { padding:18px 20px; margin-bottom:18px; }
    .metric-card { padding:15px; min-width:0; }
    .metric-card span, .metric-card small { display:block; color:var(--muted); font-size:12px; font-weight:800; }
    .metric-card strong { display:block; margin:7px 0; font-size:21px; overflow-wrap:anywhere; }
    .section-head { display:flex; justify-content:space-between; gap:18px; align-items:end; margin-bottom:15px; }
    .section-head p { max-width:640px; margin:0; color:var(--muted); line-height:1.6; }
    .eyebrow { margin:0; color:var(--accent); font-size:11px; font-weight:900; text-transform:uppercase; letter-spacing:0.12em; }
    h2 { margin:3px 0 0; font-size:24px; }
    .candidate-card, .fit-card { padding:15px; }
    .candidate-head { display:grid; grid-template-columns:auto 1fr auto; gap:10px; align-items:center; }
    .candidate-head span {
      display:inline-flex; width:34px; height:34px; align-items:center; justify-content:center;
      border-radius:999px; background:#eaf4fc; color:#0f4f85; font-weight:900;
    }
    .candidate-head h3 { margin:0; overflow-wrap:anywhere; }
    .candidate-head strong { color:var(--green); }
    .status-pill {
      display:inline-flex; margin:10px 0; padding:6px 9px; border-radius:999px;
      background:#fff4e4; color:#8a4a00; font-size:12px; font-weight:900;
    }
    dl { display:grid; grid-template-columns:110px 1fr; gap:8px 10px; margin:8px 0 0; }
    dt { color:var(--muted); font-weight:900; }
    dd { margin:0; color:#30475d; line-height:1.5; overflow-wrap:anywhere; }
    .fit-card h3 { margin:0 0 10px; }
    .fit-card section { border-top:1px solid #e8eef4; padding-top:10px; margin-top:10px; }
    .fit-card p { margin:7px 0 0; color:#30475d; line-height:1.62; }
    .table-wrap { overflow-x:auto; border:1px solid var(--line); border-radius:8px; }
    table { border-collapse:collapse; width:100%; font-size:12px; background:white; }
    th, td { border-bottom:1px solid #e8eef4; padding:8px 9px; text-align:left; vertical-align:top; }
    th { background:#f3f8fc; color:#33465a; position:sticky; top:0; }
    pre {
      margin:0; padding:14px; border:1px solid #d8e5ef; border-radius:8px; background:#f5f9fc;
      overflow:auto; line-height:1.45;
    }
    .muted { color:var(--muted); }
    @media (max-width:900px) { main { padding:16px 12px 28px; } header { padding:22px 16px; } .section-head { flex-direction:column; align-items:flex-start; } dl { grid-template-columns:1fr; } }
    """


def _script() -> str:
    return """
    (function() {
      const key = "ppvpp-algorithm-search-report-lang";
      function applyLang(lang) {
        document.documentElement.setAttribute("data-lang", lang);
        document.documentElement.setAttribute("lang", lang === "zh" ? "zh-CN" : "en");
        document.querySelectorAll("[data-lang-switch]").forEach((button) => {
          const active = button.getAttribute("data-lang-switch") === lang;
          button.classList.toggle("is-active", active);
          button.setAttribute("aria-pressed", active ? "true" : "false");
        });
        try { localStorage.setItem(key, lang); } catch (err) { void err; }
      }
      document.addEventListener("click", (event) => {
        const button = event.target.closest("[data-lang-switch]");
        if (button) applyLang(button.getAttribute("data-lang-switch") || "zh");
      });
      let initial = "zh";
      try { initial = localStorage.getItem(key) || initial; } catch (err) { void err; }
      applyLang(initial);
    })();
    """


def _html_shell(body: str) -> str:
    return f"""<!doctype html>
<html lang="zh-CN" data-lang="zh">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Algorithm Search Report</title>
  <style>{_css()}</style>
</head>
<body>
  <header>
    <p class="eyebrow">Algorithm Search / Advanced MARL</p>
    <h1>{_lang("Algorithm Search and Advanced MARL Candidate Report", "算法搜索与高级 MARL 候选报告")}</h1>
    <p>{_lang(
        "This standalone HTML report reads candidate_scores.csv and summary.json, then explains which algorithms should be kept, rejected or deferred before implementation.",
        "这个独立 HTML 报告读取 candidate_scores.csv 与 summary.json，并解释哪些算法应保留、拒绝或暂缓，再进入实现。",
        block=True,
    )}</p>
    {_language_toolbar()}
  </header>
  <main>{body}</main>
  <script>{_script()}</script>
</body>
</html>
"""


def export_algorithm_search_report(
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    *,
    candidate_scores_path: str | Path | None = None,
    summary_path: str | Path | None = None,
    output_path: str | Path | None = None,
    top_n: int = 5,
) -> Path:
    """Generate the standalone algorithm-search HTML report.

    The report intentionally does not refresh dashboard_data or the root
    interactive report. It is a separate synchronization artifact for algorithm
    screening outputs under ``outputs/algorithm_search``.
    """

    out_dir = ensure_dir(output_dir)
    candidates_csv = Path(candidate_scores_path) if candidate_scores_path else out_dir / "candidate_scores.csv"
    summary_json = Path(summary_path) if summary_path else out_dir / "summary.json"
    report_path = Path(output_path) if output_path else out_dir / DEFAULT_REPORT_NAME

    raw_candidates = _read_csv(candidates_csv)
    summary = _read_json(summary_json)
    candidates = _normalise_candidates(raw_candidates)

    body = "\n".join(
        [
            _summary_cards(candidates, summary),
            _top_candidates_section(candidates, top_n),
            _score_semantics_section(),
            _algorithm_fit_section(),
            "<section class='panel'>"
            "<div class='section-head'><div><p class='eyebrow'>All Candidates</p>"
            f"<h2>{_lang('All Candidate Scores', '全部候选算法评分')}</h2></div></div>"
            f"{_candidate_table(candidates)}</section>",
            _summary_section(summary),
        ]
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(_html_shell(body), encoding="utf-8")
    return report_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate the algorithm-search HTML report.")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--candidate-scores", default=None)
    parser.add_argument("--summary", default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--top-n", type=int, default=5)
    args = parser.parse_args(argv)

    path = export_algorithm_search_report(
        args.output_dir,
        candidate_scores_path=args.candidate_scores,
        summary_path=args.summary,
        output_path=args.output,
        top_n=args.top_n,
    )
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
