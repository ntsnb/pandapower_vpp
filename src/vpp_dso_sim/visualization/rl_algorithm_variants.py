from __future__ import annotations

from html import escape
from pathlib import Path

import pandas as pd


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


def _truthy(value: object) -> bool:
    return _text(value).lower() in {"1", "true", "yes", "y"}


def _learning_path_exists(filename: str) -> bool:
    learning_dir = Path(__file__).resolve().parents[1] / "learning"
    return (learning_dir / filename).exists()


def build_rl_algorithm_variant_frame(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    deep_summary = frames.get("deep_rl_training_summary", pd.DataFrame())
    row = deep_summary.iloc[0] if not deep_summary.empty else pd.Series(dtype=object)
    algorithm = _text(row.get("algorithm"))
    update_rule = _text(row.get("policy_update_rule"))
    has_current_happo_lite = algorithm == "privacy_separated_ctde_actor_critic" or "happo" in update_rule.lower()
    has_happo_kernel = _learning_path_exists("advanced_trainers.py")
    has_matd3_impl = _learning_path_exists("matd3.py")
    has_hasac_impl = _learning_path_exists("hasac.py") or _learning_path_exists("advanced_trainers.py")

    records = [
        {
            "display_order": 1,
            "algorithm_id": "happo",
            "algorithm_label": "HAPPO",
            "algorithm_label_zh": "HAPPO",
            "family": "On-policy heterogeneous actor-critic",
            "family_zh": "面向异构智能体的 on-policy actor-critic",
            "actor_style": "Separate stochastic actors for DSO and each VPP role",
            "actor_style_zh": "DSO 与各类 VPP 角色保持分离随机策略 actor",
            "critic_style": "Centralized value critic with role-aware value heads",
            "critic_style_zh": "带角色价值头的集中式价值 critic",
            "update_core": "Sequential policy update with importance correction across agents",
            "update_core_zh": "按智能体顺序更新策略，并对后续智能体做重要性修正",
            "experience_reuse": "On-policy rollout reuse only inside the PPO/HAPPO update window",
            "experience_reuse_zh": "只在 PPO/HAPPO 更新窗口内复用 on-policy rollout",
            "architecture_signature": "Best fit when heterogeneous DSO / dispatch / portfolio actors need decentralized execution but coordinated training",
            "architecture_signature_zh": "适合 DSO / 调度 / 组合配置角色异构、执行分散但训练协同的场景",
            "repo_status": (
                "Target hierarchical HAPPO is now the dedicated runnable trainer: DSO has its own global actor, "
                "each VPP has an independent dispatch actor, and each VPP has an independent slow-loop portfolio actor. "
                "Sequential update and importance correction are active; long-budget multi-seed validation remains next."
                if has_happo_kernel
                else (
                    "Fallback repo path is MAPPO/HAPPO-lite: role value heads, GAE-lambda and PPO clipping are implemented, "
                    "but the advanced HAPPO trainer file is not present in this snapshot."
                    if has_current_happo_lite
                    else "Visualized target path only; the current training summary does not expose a runnable HAPPO-like update."
                )
            ),
            "repo_status_zh": (
                "fallback 路径是 MAPPO/HAPPO-lite：已实现角色价值头、GAE-lambda 和 PPO clip；"
                "但当前快照没有检测到高级 HAPPO 训练器文件。"
                if has_current_happo_lite
                else "当前仅作为目标架构展示；现有训练摘要没有暴露可运行的 HAPPO 类更新规则。"
            ),
            "implementation_badge": "implemented_kernel" if has_happo_kernel else "current_happo_lite_path" if has_current_happo_lite else "target_only",
            "evidence_file": "src/vpp_dso_sim/learning/advanced_marl.py; src/vpp_dso_sim/learning/advanced_trainers.py; outputs/dashboard_data/deep_rl_training_summary.csv; src/vpp_dso_sim/learning/deep_rl.py",
        },
        {
            "display_order": 2,
            "algorithm_id": "matd3",
            "algorithm_label": "MATD3",
            "algorithm_label_zh": "MATD3",
            "family": "Off-policy deterministic multi-agent actor-critic",
            "family_zh": "off-policy 确定性多智能体 actor-critic",
            "actor_style": "Continuous deterministic DSO actor plus per-VPP deterministic dispatch actors by default",
            "actor_style_zh": "默认使用连续确定性 DSO actor + 每个 VPP 独立确定性调度 actor",
            "critic_style": "DSO twin Q plus per-VPP twin Q heads under a centralized training view",
            "critic_style_zh": "集中训练视角下的 DSO 双 Q 与按 VPP 拆分的双 Q 价值头",
            "update_core": "Twin Q critics, target policy smoothing and delayed actor update",
            "update_core_zh": "双 Q critic、目标策略平滑与延迟 actor 更新",
            "experience_reuse": "Off-policy replay buffer with batched transition reuse",
            "experience_reuse_zh": "使用 replay buffer 做 off-policy 批量样本复用",
            "architecture_signature": "Strong for continuous dispatch subproblems, but the discrete slow portfolio action should stay outside TD3-style heads",
            "architecture_signature_zh": "适合连续调度子问题，但慢周期离散组合配置动作不应强行塞进 TD3 风格 head",
            "repo_status": (
                "Dedicated MATD3 implementation exists for the continuous DSO/VPP dispatch subproblem with per-VPP dispatch actors by default; the slow portfolio action remains outside MATD3."
                if has_matd3_impl
                else "Visualized target path only; a dedicated MATD3 trainer is not present in the current repo snapshot."
            ),
            "repo_status_zh": (
                "仓库中已存在面向连续 DSO/VPP 调度子问题的专用 MATD3 实现；慢周期组合配置动作仍放在 MATD3 之外。"
                if has_matd3_impl
                else "当前仅作为目标架构展示；本仓库快照里没有专用 MATD3 训练器。"
            ),
            "implementation_badge": "implemented_continuous_subproblem" if has_matd3_impl else "target_only",
            "evidence_file": "src/vpp_dso_sim/learning/matd3.py; README.md",
        },
        {
            "display_order": 3,
            "algorithm_id": "hasac",
            "algorithm_label": "HASAC",
            "algorithm_label_zh": "HASAC",
            "family": "Off-policy soft actor-critic for heterogeneous agents",
            "family_zh": "面向异构智能体的 off-policy soft actor-critic",
            "actor_style": "Stochastic soft DSO actor plus per-VPP dispatch soft actors by default",
            "actor_style_zh": "默认使用随机 soft DSO actor + 每个 VPP 独立调度 soft actor",
            "critic_style": "Twin soft Q critics with temperature-controlled policy evaluation",
            "critic_style_zh": "带温度系数的 twin soft Q critic",
            "update_core": "Entropy temperature tuning plus soft Bellman backup",
            "update_core_zh": "熵温度自适应调节加 soft Bellman 备份",
            "experience_reuse": "Off-policy replay; privacy filtering must be preserved in replay storage",
            "experience_reuse_zh": "off-policy replay，但回放存储必须继续满足隐私过滤边界",
            "architecture_signature": "Useful when local-flex price, VPP bid and DER availability uncertainty benefit from entropy-regularized exploration",
            "architecture_signature_zh": "适合本地灵活性价格、VPP 报价与 DER 可用性存在不确定性，需要熵正则探索的场景",
            "repo_status": (
                "HASAC soft actor, entropy temperature, twin soft Q, soft Bellman backup and off-policy replay are implemented in a dedicated continuous-dispatch trainer with per-VPP dispatch actors by default; slow portfolio actions remain outside HASAC."
                if has_hasac_impl
                else "Not yet implemented as a dedicated trainer in the current repo; shown here as the intended soft off-policy contrast against HAPPO and MATD3."
            ),
            "repo_status_zh": (
                "仓库中已存在专用 HASAC 实现。"
                if has_hasac_impl
                else "当前仓库还没有专用 HASAC 训练器；这里把它作为对照性的 soft off-policy 目标架构展示。"
            ),
            "implementation_badge": "implemented_kernel" if has_hasac_impl else "target_only",
            "evidence_file": "src/vpp_dso_sim/learning/advanced_trainers.py; README.md; src/vpp_dso_sim/visualization/algorithm_search_report.py",
        },
    ]
    if has_happo_kernel:
        records[0]["implementation_badge"] = "implemented_hierarchical_trainer"
        records[0]["repo_status_zh"] = (
            "目标分层 HAPPO 已经是专用可运行训练器：DSO 拥有独立全局 actor，"
            "每个 VPP 拥有独立调度 actor，并且每个 VPP 拥有独立慢周期组合配置 actor。"
            "顺序更新和重要性校正已启用；下一步仍是长周期、多 seed 验证。"
        )
    if has_hasac_impl:
        records[2]["repo_status_zh"] = (
            "HASAC 的 soft actor、熵温度、双 soft Q、soft Bellman 备份和离策略 replay "
            "已经在连续调度环境训练器中实现，默认每个 VPP 使用独立调度 actor；慢周期组合配置动作仍放在 HASAC 之外。"
        )
    return pd.DataFrame(records)


def rl_algorithm_variant_section_css() -> str:
    return """
    .variant-section-note { color: #5d7084; line-height: 1.65; margin: 0; }
    .variant-grid {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
      gap: 12px;
      margin-top: 14px;
    }
    .variant-card {
      border: 1px solid #d8e3ee;
      border-radius: 16px;
      background: linear-gradient(180deg, #ffffff, #f7fbfe);
      padding: 15px;
      box-shadow: 0 10px 20px rgba(15, 35, 56, 0.05);
    }
    .variant-card-head {
      display: flex;
      justify-content: space-between;
      gap: 10px;
      align-items: start;
      margin-bottom: 10px;
    }
    .variant-card-head h3 {
      margin: 0;
      font-size: 18px;
    }
    .variant-family {
      margin-top: 5px;
      color: #4f6479;
      font-size: 13px;
      line-height: 1.5;
    }
    .variant-badge {
      display: inline-flex;
      align-items: center;
      padding: 6px 10px;
      border-radius: 999px;
      background: #eaf4fc;
      color: #12507d;
      font-size: 12px;
      font-weight: 900;
      white-space: nowrap;
    }
    .variant-badge.target-only {
      background: #fff4e8;
      color: #9a4f00;
    }
    .variant-fields {
      display: grid;
      gap: 9px;
    }
    .variant-fields section {
      border: 1px solid #e2ebf3;
      border-radius: 12px;
      background: #fff;
      padding: 10px 11px;
    }
    .variant-fields h4 {
      margin: 0 0 6px;
      color: #5d7084;
      font-size: 12px;
      font-weight: 900;
      letter-spacing: 0.04em;
      text-transform: uppercase;
    }
    .variant-fields p {
      margin: 0;
      color: #24384d;
      line-height: 1.58;
      overflow-wrap: anywhere;
    }
    """


def build_rl_algorithm_variant_section(variants: pd.DataFrame, *, title_en: str, title_zh: str) -> str:
    if variants.empty:
        return ""
    cards: list[str] = []
    for _, row in variants.sort_values("display_order").iterrows():
        badge = _text(row.get("implementation_badge"), "target_only")
        badge_en = {
            "current_happo_lite_path": "Current runnable path",
            "implemented_continuous_subproblem": "Implemented subproblem",
            "implemented_hierarchical_trainer": "Implemented trainer",
            "implemented_kernel": "Implemented kernel",
            "implemented": "Implemented",
            "target_only": "Target only",
        }.get(badge, "Target only")
        badge_zh = {
            "current_happo_lite_path": "当前可运行路径",
            "implemented_continuous_subproblem": "已实现连续子问题",
            "implemented_hierarchical_trainer": "已实现分层训练器",
            "implemented": "已实现",
            "target_only": "仅目标架构",
        }.get(badge, "仅目标架构")
        if badge == "implemented_kernel":
            badge_zh = "已实现算法内核"
        cards.append(
            f"""
            <article class="variant-card">
              <div class="variant-card-head">
                <div>
                  <h3>{_lang(escape(_text(row.get('algorithm_label'))), escape(_text(row.get('algorithm_label_zh'))), block=True)}</h3>
                  <p class="variant-family">{_lang(escape(_text(row.get('family'))), escape(_text(row.get('family_zh'))), block=True)}</p>
                </div>
                <span class="variant-badge {escape(badge.replace('_', '-'))}">{_lang(badge_en, badge_zh)}</span>
              </div>
              <div class="variant-fields">
                <section>
                  <h4>{_lang("Actor Style", "Actor 风格")}</h4>
                  <p>{_lang(escape(_text(row.get('actor_style'))), escape(_text(row.get('actor_style_zh'))), block=True)}</p>
                </section>
                <section>
                  <h4>{_lang("Critic Style", "Critic 风格")}</h4>
                  <p>{_lang(escape(_text(row.get('critic_style'))), escape(_text(row.get('critic_style_zh'))), block=True)}</p>
                </section>
                <section>
                  <h4>{_lang("Update Core", "更新核心")}</h4>
                  <p>{_lang(escape(_text(row.get('update_core'))), escape(_text(row.get('update_core_zh'))), block=True)}</p>
                </section>
                <section>
                  <h4>{_lang("Experience Reuse", "样本复用方式")}</h4>
                  <p>{_lang(escape(_text(row.get('experience_reuse'))), escape(_text(row.get('experience_reuse_zh'))), block=True)}</p>
                </section>
                <section>
                  <h4>{_lang("Architecture Signature", "架构辨识点")}</h4>
                  <p>{_lang(escape(_text(row.get('architecture_signature'))), escape(_text(row.get('architecture_signature_zh'))), block=True)}</p>
                </section>
                <section>
                  <h4>{_lang("Repo Status", "仓库状态")}</h4>
                  <p>{_lang(escape(_text(row.get('repo_status'))), escape(_text(row.get('repo_status_zh'))), block=True)}</p>
                </section>
              </div>
            </article>
            """
        )
    return f"""
    <section class="panel">
      <div class="section-head">
        <div>
          <p class="eyebrow">Algorithms</p>
          <h2>{_lang(title_en, title_zh)}</h2>
        </div>
        <p class="variant-section-note">{_lang(
            "This comparison is exported to dashboard_data/rl_algorithm_variants.csv. It uses current training metadata when available and falls back to a repo-compatible explanation when a dedicated field is missing.",
            "该对照表会导出到 dashboard_data/rl_algorithm_variants.csv。若当前训练摘要包含相关字段就优先读取；缺字段时退回到与仓库现状兼容的说明，不要求修改 learning 元数据。",
            block=True,
        )}</p>
      </div>
      <div class="variant-grid">
        {''.join(cards)}
      </div>
    </section>
    """
