from __future__ import annotations

from typing import Any

import pandas as pd

from vpp_dso_sim.learning.advanced_trainers import advanced_algorithm_capability_rows


def _frame_or_empty(value: pd.DataFrame | None) -> pd.DataFrame:
    return value.copy() if isinstance(value, pd.DataFrame) else pd.DataFrame()


def _text(value: Any, default: str = "") -> str:
    if value is None:
        return default
    try:
        if pd.isna(value):
            return default
    except TypeError:
        pass
    text = str(value).strip()
    return text if text else default


def _bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _first_value(frame: pd.DataFrame, column: str, default: str = "") -> str:
    if frame.empty or column not in frame:
        return default
    return _text(frame[column].iloc[0], default)


def _privacy_separated_primary(deep_summary: pd.DataFrame | None = None) -> bool:
    deep = _frame_or_empty(deep_summary)
    if deep.empty:
        return True
    algorithm = _first_value(deep, "algorithm", "")
    primary = _first_value(deep, "target_ctde_primary_trainer", "")
    return algorithm == "privacy_separated_ctde_actor_critic" or _bool(primary)


AGENT_GROUPS: dict[str, dict[str, str]] = {
    "global_guidance": {
        "label": "Global Guidance Agents",
        "label_zh": "全局引导智能体",
        "color": "#1f77b4",
        "summary": "Upper-level DSO agents that see grid-wide state and issue VPP operating envelopes.",
        "summary_zh": "上层 DSO 智能体，读取全局电网状态，并向各 VPP 下发运行包络和服务意图。",
    },
    "vpp_dispatch": {
        "label": "VPP Dispatch Agents",
        "label_zh": "VPP 调度/解聚合智能体",
        "color": "#2ca02c",
        "summary": "Fast VPP agents that select an aggregate point and learn DER-level disaggregation actions.",
        "summary_zh": "快速 VPP 智能体，在运行包络内选择聚合功率点，并学习 DER 级解聚合动作。",
    },
    "vpp_portfolio": {
        "label": "VPP Portfolio Agents",
        "label_zh": "VPP 组合配置智能体",
        "color": "#9467bd",
        "summary": "Slow VPP agents that evaluate whether commercial aggregation weights or membership should change.",
        "summary_zh": "慢时间尺度 VPP 智能体，用于评估聚合成员、权重或商业配置是否需要调整。",
    },
    "training_supervisor": {
        "label": "Training Supervisor",
        "label_zh": "训练监督智能体",
        "color": "#ff7f0e",
        "summary": "Experiment-level supervisor for algorithm trials, hyperparameter sweeps and convergence checks.",
        "summary_zh": "实验级监督模块，负责算法试验、超参数搜索和收敛性检查。",
    },
}


def _group_for_role(role_type: str) -> str:
    if role_type == "dso_guidance_agent":
        return "global_guidance"
    if role_type == "vpp_dispatch_agent":
        return "vpp_dispatch"
    if role_type == "vpp_portfolio_agent":
        return "vpp_portfolio"
    return "training_supervisor"


def _role_reward_zh(role_type: str, fallback: str) -> str:
    if role_type == "dso_guidance_agent":
        return (
            "r_dso = -0.05 * dso_total_cost + 可行性奖励 + 跟踪奖励；"
            "该 reward 只训练 DSO 全局引导目标，raw_objective_reward = -total_cost 仅用于诊断。"
        )
    if role_type == "vpp_dispatch_agent":
        return (
            "r_dispatch_i = 私有利润代理 + 服务/可用性收益 - DER 成本 - 跟踪/投影惩罚 - 舒适度/SOC 惩罚；"
            "它不直接纳入原始 DSO 全局 reward，只反映该 VPP 的自利调度与履约表现。"
        )
    if role_type == "vpp_portfolio_agent":
        return (
            "r_portfolio_i = 长周期利润代理 + 可靠性奖励 + 局部化 DSO 对齐收益 - 配置切换成本 - 履约风险惩罚；"
            "DSO 项是合约化/局部化反馈，不是原始全局 reward。"
        )
    return fallback


def _agent_result_provenance(role_type: str, dims: dict[str, str] | None = None) -> dict[str, Any]:
    """Explain whether an agent output is learned, projected or only monitored."""
    dims = dims or {}
    dso_dim = dims.get("dso_input_dim", "D_dso")
    vpp_dim = dims.get("vpp_input_dim", "D_vpp")
    portfolio_dim = dims.get("portfolio_input_dim", "D_portfolio")
    critic_dim = dims.get("critic_input_dim", "D_critic")
    hidden_dim = dims.get("hidden_dim", "64")
    n_vpps = dims.get("n_vpps", "N_vpp")
    max_der = dims.get("max_der_per_vpp", "Kmax")
    if role_type == "dso_guidance_agent":
        return {
            "is_rl_decision": True,
            "rl_usage_status": "uses_rl_privacy_separated_ctde_dso_actor",
            "rl_usage_status_zh": "使用强化学习：隐私分离 CTDE 的独立 DSO actor 输出各 VPP 的包络/目标偏好。",
            "result_formula": (
                "o_dso in R^D_dso -> z_dso = f_dso(o_dso); a_dso ~ Normal(mu_dso(z_dso), sigma_dso); "
                "envelope preference = map(a_dso, VPP bids, FR/DOE); final envelope = projection(preference)"
            ),
            "result_formula_zh": (
                "先把 DSO 可见观测 o_dso 编码为 z_dso；DSO 策略头从高斯分布采样动作 a_dso；"
                "再结合 VPP 报量/报价与 FR/DOE 边界映射成运行包络偏好；最后由安全投影得到可执行包络。"
            ),
            "neural_network_structure": (
                f"o_dso in R^{dso_dim} -> DSO-only encoder [LayerNorm({dso_dim}), "
                f"Linear({dso_dim},{hidden_dim}), Tanh, Linear({hidden_dim},{hidden_dim}), Tanh] -> "
                f"z_dso in R^{hidden_dim} -> mean=tanh(Linear({hidden_dim},{n_vpps})) + "
                f"trainable dso_log_std in R^{n_vpps}."
            ),
            "neural_network_structure_zh": (
                f"o_dso 属于 R^{dso_dim}，只进入 DSO 独立编码器：LayerNorm({dso_dim}) -> "
                f"Linear({dso_dim},{hidden_dim}) -> Tanh -> Linear({hidden_dim},{hidden_dim}) -> Tanh；"
                f"得到 z_dso 属于 R^{hidden_dim}，随后进入 mean=tanh(Linear({hidden_dim},{n_vpps}))，"
                f"并配套 R^{n_vpps} 的可训练 dso_log_std。"
            ),
            "result_calculation": (
                "The displayed DSO envelope is not a pure neural-network number. The RL head proposes an envelope "
                "preference; deterministic FR/DOE clipping, grid-stress heuristics and device limits make it executable."
            ),
            "result_calculation_zh": (
                "页面展示的 DSO 包络不是神经网络直接裸输出。RL head 只给出包络偏好，"
                "随后经过 FR/DOE 裁剪、网络压力启发式和设备边界处理，才形成可下发结果。"
            ),
            "result_source": (
                "deep_rl.py action traces when a trained rollout is present; otherwise Simulator._build_dso_operating_envelope "
                "provides the deterministic baseline."
            ),
            "result_source_zh": (
                "有深度强化学习 rollout 时来自 deep_rl.py 的动作轨迹；没有训练产物时来自 "
                "Simulator._build_dso_operating_envelope 的确定性基线。"
            ),
            "rl_training_signal": (
                "dso_policy_loss = -mean(log_prob(a_dso) * A_t). A_t is estimated by the training-only centralized critic; raw total_cost is logged only as diagnostics."
            ),
            "rl_training_signal_zh": "dso_policy_loss = -mean(log_prob(a_dso) * A_t)。A_t 由仅训练期可见的 centralized critic 估计；原始 total_cost 只作为诊断项记录。",
            "audit_outputs": "dso_operating_envelope.csv; feasible_region.csv; fr_envelope_state.csv; projection_trace.csv",
            "audit_outputs_zh": "审计文件：dso_operating_envelope.csv、feasible_region.csv、fr_envelope_state.csv、projection_trace.csv。",
            "non_rl_guardrails": "FR/DOE bounds, voltage/loading heuristics, DER capability limits and pandapower security checks.",
            "non_rl_guardrails_zh": "非 RL 保护：FR/DOE 边界、电压/线路负载启发式、DER 能力边界与 pandapower 安全校核。",
        }
    if role_type == "vpp_dispatch_agent":
        return {
            "is_rl_decision": True,
            "rl_usage_status": "uses_rl_privacy_scoped_vpp_dispatch_actor",
            "rl_usage_status_zh": "使用强化学习：VPP 本地调度 actor 只读取本 VPP 私有观测，并输出聚合 P 与 DER 归一化动作。",
            "result_formula": (
                "o_vpp_i in R^D_vpp -> z_vpp_i = f_vpp(o_vpp_i); selected_p_mw = envelope_map(Normal(mu_agg(z_vpp_i), sigma_agg)); "
                "raw_der_actions = Normal(mu_der(z_vpp_i), sigma_der); dispatch = project(raw_der_actions, DER bounds, selected_p_mw)"
            ),
            "result_formula_zh": (
                "先把 VPP_i 本地观测 o_vpp_i 编码为 z_vpp_i；聚合目标 head 在 DSO 包络内选择 selected_p_mw；"
                "DER head 输出每个设备的归一化动作；安全投影按 DER 边界和聚合目标修正为最终出力。"
            ),
            "neural_network_structure": (
                f"o_vpp_i in R^{vpp_dim} -> context[16] + DER tokens[K,15] -> shared token MLP + masked pooling -> "
                f"z_vpp_i in R^{hidden_dim} -> aggregate_mean=tanh(Linear({hidden_dim},1)) + "
                f"der_mean=tanh(Linear({hidden_dim},{max_der})) + trainable log_std in R^1 and R^{max_der}."
            ),
            "neural_network_structure_zh": (
                f"o_vpp_i 属于 R^{vpp_dim}，只进入 VPP 本地编码器：LayerNorm({vpp_dim}) -> "
                f"Linear({vpp_dim},{hidden_dim}) -> Tanh -> Linear({hidden_dim},{hidden_dim}) -> Tanh；"
                f"得到 z_vpp_i 属于 R^{hidden_dim}，再由 aggregate_mean=tanh(Linear({hidden_dim},1)) "
                f"和 der_mean=tanh(Linear({hidden_dim},{max_der})) 输出聚合功率与 DER 归一化动作，"
                f"并配套 R^1 与 R^{max_der} 的可训练 log_std。"
            ),
            "result_calculation": (
                "The VPP dispatch result is a learned proposal plus deterministic feasibility repair. "
                "RL decides the aggregate setpoint and per-DER normalized proposals; projection converts them into physical PV/ESS/EV/HVAC/MT setpoints."
            ),
            "result_calculation_zh": (
                "VPP 调度结果由“学习动作 + 确定性可行性修复”组成。RL 决定聚合运行点和各 DER 归一化建议，"
                "投影层再把它们变成 PV、储能、EV、HVAC、燃机等设备的物理出力。"
            ),
            "result_source": (
                "vpp_rl_disaggregation.csv records learned raw/projection values; der_dispatch.csv records pandapower element writes."
            ),
            "result_source_zh": (
                "vpp_rl_disaggregation.csv 记录学习动作和投影差额；der_dispatch.csv 记录最终写入 pandapower 元件的出力。"
            ),
            "rl_training_signal": (
                "vpp_dispatch_policy_loss = -mean(sum_i log_prob(a_vpp_i) * A_dispatch_i). "
                "A_dispatch_i is computed from the VPP's own settlement-aware dispatch reward, not raw DSO global reward."
            ),
            "rl_training_signal_zh": "vpp_dispatch_policy_loss = -mean(sum_i log_prob(a_vpp_i) * A_dispatch_i)。A_dispatch_i 来自 VPP 自身结算感知调度 reward，不直接使用原始 DSO 全局 reward。",
            "audit_outputs": "vpp_rl_disaggregation.csv; der_dispatch.csv; projection_trace.csv; storage_soc.csv; evcs_soc.csv; hvac_temperature.csv",
            "audit_outputs_zh": "审计文件：vpp_rl_disaggregation.csv、der_dispatch.csv、projection_trace.csv、storage_soc.csv、evcs_soc.csv、hvac_temperature.csv。",
            "non_rl_guardrails": "DER bounds, SOC/comfort constraints, ramp limits, aggregate residual repair and pandapower write conventions.",
            "non_rl_guardrails_zh": "非 RL 保护：DER 出力边界、SOC/舒适度约束、爬坡约束、聚合残差修复和 pandapower 符号约定。",
        }
    if role_type == "vpp_portfolio_agent":
        return {
            "is_rl_decision": True,
            "rl_usage_status": "uses_rl_categorical_portfolio_head_with_physical_event_gate",
            "rl_usage_status_zh": "使用强化学习：组合配置 categorical head 训练商业配置建议，物理成员变更仍受事件门控。",
            "result_formula": (
                "z_portfolio = encoder(history, reliability, profit, delivery_risk); "
                "action ~ Categorical(logits(z_portfolio)); slow reward evaluates long-horizon profit proxy, reliability, switching cost and localized DSO-alignment credit."
            ),
            "result_formula_zh": (
                "把历史收益、可靠性、履约风险和能力置信度编码为 z_portfolio；"
                "categorical head 输出 keep/reweight/propose 的概率并采样；当前用代理 reward 评估组合建议。"
            ),
            "neural_network_structure": (
                f"h_vpp_i in R^{portfolio_dim} -> portfolio encoder [LayerNorm({portfolio_dim}), "
                f"Linear({portfolio_dim},{hidden_dim}), Tanh, Linear({hidden_dim},{hidden_dim}), Tanh] -> "
                f"z_portfolio_i in R^{hidden_dim} -> portfolio_logits = Linear({hidden_dim},3) -> "
                "Categorical logits for keep/reweight/propose_membership_change."
            ),
            "neural_network_structure_zh": (
                f"h_vpp_i 属于 R^{portfolio_dim}，进入组合配置编码器：LayerNorm({portfolio_dim}) -> "
                f"Linear({portfolio_dim},{hidden_dim}) -> Tanh -> Linear({hidden_dim},{hidden_dim}) -> Tanh；"
                f"得到 z_portfolio_i 属于 R^{hidden_dim}，再经 portfolio_logits=Linear({hidden_dim},3)，"
                "3 维 logits 对应 keep、reweight、propose_membership_change。"
            ),
            "result_calculation": (
                "The portfolio result is a slow-loop learned commercial recommendation. "
                "It can recommend keep/reweight/propose_membership_change, but it does not directly move pandapower assets unless a gated scenario event accepts the change."
            ),
            "result_calculation_zh": (
                "组合配置结果是慢周期的可学习商业建议。它可以建议保持、重新加权或提出成员变更，"
                "但不会直接移动 pandapower 物理资产；只有通过受控场景事件时才会改变成员关系。"
            ),
            "result_source": (
                "deep_rl_trajectory.csv/deep_rl_step_metrics.csv for sampled portfolio actions; vpp_portfolio_history.csv and "
                "portfolio_change_log.csv for accepted scenario changes."
            ),
            "result_source_zh": (
                "deep_rl_trajectory.csv/deep_rl_step_metrics.csv 记录采样动作；vpp_portfolio_history.csv 和 "
                "portfolio_change_log.csv 记录被场景门控接受的配置变更。"
            ),
            "rl_training_signal": (
                "Policy-gradient term uses the portfolio-specific slow reward. The DSO signal enters only as a localized, contract-like alignment credit, not as raw shared global reward."
            ),
            "rl_training_signal_zh": "策略梯度使用组合配置专属慢周期 reward。DSO 信号只以局部化、合约化的对齐收益进入，不直接共享原始全局 reward。",
            "audit_outputs": "vpp_portfolio_history.csv; portfolio_change_log.csv; deep_rl_trajectory.csv; deep_rl_step_metrics.csv",
            "audit_outputs_zh": "审计文件：vpp_portfolio_history.csv、portfolio_change_log.csv、deep_rl_trajectory.csv、deep_rl_step_metrics.csv。",
            "non_rl_guardrails": "Contract/scenario event gate; no direct bus relocation; no direct pandapower row mutation from the policy head.",
            "non_rl_guardrails_zh": "非 RL 保护：合同/场景事件门控；策略头不能直接改母线位置，也不能直接移动 pandapower 元件行。",
        }
    return {
        "is_rl_decision": False,
        "rl_usage_status": "not_an_rl_environment_agent",
        "rl_usage_status_zh": "不属于强化学习环境智能体：它是实验监督/调参编排器。",
        "result_formula": (
            "trial metrics -> convergence/risk rules -> next trial recommendation or handoff to algorithm debugging"
        ),
        "result_formula_zh": "训练指标进入收敛/风险规则，输出下一组试验建议，或把失败实验回交给算法诊断。",
        "neural_network_structure": (
            "No actor/critic network. It is a rule-based experiment orchestrator outside the environment step loop."
        ),
        "neural_network_structure_zh": "没有 actor/critic 神经网络。它是环境 step 循环之外的规则式实验编排器。",
        "result_calculation": (
            "The supervisor does not act inside env.step() and has no policy gradient. "
            "It ranks runs using reward trends, violation counts, convergence status and hyperparameter metadata."
        ),
        "result_calculation_zh": (
            "训练监督器不在 env.step() 内行动，也没有策略梯度。它根据 reward 趋势、越限次数、收敛状态和超参数元数据排序实验。"
        ),
        "result_source": "learning/tuning.py outputs and deep_rl training summaries; outside the physical simulator loop.",
        "result_source_zh": "来源于 learning/tuning.py 输出和 deep_rl 训练摘要，位于物理仿真环境循环之外。",
        "rl_training_signal": "No environment reward, no actor loss and no critic loss. It only observes training artifacts.",
        "rl_training_signal_zh": "没有环境 reward、没有 actor loss、没有 critic loss；只观察训练产物。",
        "audit_outputs": "marl_baselines/training_summary.csv; tuning_trials.csv; deep_rl_training_summary.csv",
        "audit_outputs_zh": "审计文件：marl_baselines/training_summary.csv、tuning_trials.csv、deep_rl_training_summary.csv。",
        "non_rl_guardrails": "Experiment-level early stop, failure handoff and trial bookkeeping; no pandapower element writes.",
        "non_rl_guardrails_zh": "非 RL 保护：实验级早停、失败回交和试验记录；不写入 pandapower 元件。",
    }


def rl_neural_network_architecture_frame(
    agent_roles: pd.DataFrame | None = None,
    asset_registry: pd.DataFrame | None = None,
    deep_summary: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Describe the legacy shared-backbone benchmark network.

    The privacy-separated CTDE trainer is described by
    rl_target_ctde_architecture_frame. This frame is kept for explicit
    `--algorithm shared` runs, ablation studies and regression tests.
    """
    roles = _frame_or_empty(agent_roles)
    assets = _frame_or_empty(asset_registry)
    deep = _frame_or_empty(deep_summary)
    if not roles.empty and "role_type" in roles.columns:
        n_vpps = int((roles["role_type"].astype(str) == "vpp_dispatch_agent").sum())
        if n_vpps <= 0:
            n_vpps = int((roles["role_type"].astype(str) == "vpp_portfolio_agent").sum())
    else:
        n_vpps = 1
    n_vpps = max(1, n_vpps)
    n_der: int | None = None
    if not assets.empty and "der_id" in assets.columns:
        n_der = int(assets["der_id"].nunique())
    hidden_dim = _first_value(deep, "hidden_dim", "64")
    try:
        hidden_dim_int = int(float(hidden_dim))
    except ValueError:
        hidden_dim_int = 64
    input_dim = 5 + 7 * n_vpps
    der_dim_label = str(n_der) if n_der is not None else "N_DER_total"
    rows = [
        {
            "component_id": "manual_dso_observation_encoder",
            "component_group": "encoder",
            "trainable": False,
            "input_shape": "dict observation from MultiAgentVPPDSOEnv",
            "output_shape": f"R^{input_dim}",
            "structure": (
                "Hand-built numeric vector: [time/288, min_vm, max_vm, max_line/100, max_trafo/100] "
                "+ per-VPP [p, q, p_min, p_max, q_min, q_max, single_pcc_flag]."
            ),
            "structure_zh": (
                "手工数值编码：[time/288、最低电压、最高电压、最大线路负载/100、最大变压器负载/100]，"
                "再拼接每个 VPP 的 [p、q、p_min、p_max、q_min、q_max、single_pcc 标记]。"
            ),
            "distribution": "deterministic preprocessing",
            "distribution_zh": "确定性预处理，不是神经网络层。",
            "calculation_note": "This is why current input_dim = 5 + 7 * N_vpp.",
            "calculation_note_zh": "因此当前输入维度公式为 input_dim = 5 + 7 * N_vpp。",
        },
        {
            "component_id": "shared_mlp_backbone",
            "component_group": "shared_backbone",
            "trainable": True,
            "input_shape": f"R^{input_dim}",
            "output_shape": f"R^{hidden_dim_int}",
            "structure": (
                f"Linear({input_dim},{hidden_dim_int}) -> Tanh -> "
                f"Linear({hidden_dim_int},{hidden_dim_int}) -> Tanh."
            ),
            "structure_zh": (
                f"Linear({input_dim},{hidden_dim_int}) -> Tanh -> "
                f"Linear({hidden_dim_int},{hidden_dim_int}) -> Tanh。"
            ),
            "distribution": "latent deterministic transform",
            "distribution_zh": "确定性隐变量变换。",
            "calculation_note": "All actor heads and the centralized value head consume the same latent vector h.",
            "calculation_note_zh": "所有 actor head 和集中式 value head 都读取同一个隐变量 h。",
        },
        {
            "component_id": "dso_gaussian_actor_head",
            "component_group": "dso_actor",
            "trainable": True,
            "input_shape": f"R^{hidden_dim_int}",
            "output_shape": f"mean R^{n_vpps}, log_std R^{n_vpps}",
            "structure": f"mean = tanh(Linear({hidden_dim_int},{n_vpps})); log_std is a trainable Parameter initialized at -0.7.",
            "structure_zh": f"mean = tanh(Linear({hidden_dim_int},{n_vpps}))；log_std 是可训练参数，初始化为 -0.7。",
            "distribution": "Normal(mean, exp(log_std)); rsample(); clamp to [-1,1]",
            "distribution_zh": "Normal(mean, exp(log_std))；rsample() 采样；再裁剪到 [-1,1]。",
            "calculation_note": "Each dimension corresponds to one VPP envelope-preference target.",
            "calculation_note_zh": "每一维对应一个 VPP 的包络偏好目标。",
        },
        {
            "component_id": "vpp_aggregate_gaussian_actor_head",
            "component_group": "vpp_dispatch_actor",
            "trainable": True,
            "input_shape": f"R^{hidden_dim_int}",
            "output_shape": f"mean R^{n_vpps}, log_std R^{n_vpps}",
            "structure": (
                f"vpp_target_mean = tanh(Linear({hidden_dim_int},{n_vpps})); "
                "vpp_target_log_std is a trainable Parameter initialized at -0.8."
            ),
            "structure_zh": (
                f"vpp_target_mean = tanh(Linear({hidden_dim_int},{n_vpps}))；"
                "vpp_target_log_std 是可训练参数，初始化为 -0.8。"
            ),
            "distribution": "Normal(mean, exp(log_std)); selected_p_mw = center + normalized_action * halfspan",
            "distribution_zh": "Normal(mean, exp(log_std))；selected_p_mw = 可行域中心 + 归一化动作 * 半宽。",
            "calculation_note": "This chooses each VPP aggregate operating point inside its current P bounds.",
            "calculation_note_zh": "这个 head 为每个 VPP 在当前 P 可行边界内选择聚合运行点。",
        },
        {
            "component_id": "der_dispatch_gaussian_actor_head",
            "component_group": "vpp_der_disaggregation_actor",
            "trainable": True,
            "input_shape": f"R^{hidden_dim_int}",
            "output_shape": f"mean R^{der_dim_label}, log_std R^{der_dim_label}",
            "structure": (
                f"der_dispatch_mean = tanh(Linear({hidden_dim_int},{der_dim_label})); "
                "der_dispatch_log_std is a trainable Parameter initialized at -0.8."
            ),
            "structure_zh": (
                f"der_dispatch_mean = tanh(Linear({hidden_dim_int},{der_dim_label}))；"
                "der_dispatch_log_std 是可训练参数，初始化为 -0.8。"
            ),
            "distribution": "Normal(mean, exp(log_std)); each VPP receives the slice belonging to its DER ids",
            "distribution_zh": "Normal(mean, exp(log_std))；每个 VPP 只读取属于自身 DER 的动作切片。",
            "calculation_note": "Projection converts normalized DER actions into PV/ESS/EV/HVAC/MT physical setpoints.",
            "calculation_note_zh": "安全投影将归一化 DER 动作转换成 PV、储能、EV、HVAC、燃机等物理出力。",
        },
        {
            "component_id": "portfolio_categorical_actor_head",
            "component_group": "vpp_portfolio_actor",
            "trainable": True,
            "input_shape": f"R^{hidden_dim_int}",
            "output_shape": f"logits R^{n_vpps}x3",
            "structure": f"portfolio_logits = Linear({hidden_dim_int},{3 * n_vpps}).reshape(N_vpp,3).",
            "structure_zh": f"portfolio_logits = Linear({hidden_dim_int},{3 * n_vpps}).reshape(N_vpp,3)。",
            "distribution": "Categorical(logits) over keep / reweight / propose_membership_change",
            "distribution_zh": "Categorical(logits)，三类动作是 keep / reweight / propose_membership_change。",
            "calculation_note": "This is a slow-loop commercial recommendation; physical topology changes remain event-gated.",
            "calculation_note_zh": "这是慢周期商业配置建议；物理拓扑变更仍由事件门控保护。",
        },
        {
            "component_id": "centralized_value_head",
            "component_group": "critic",
            "trainable": True,
            "input_shape": f"R^{hidden_dim_int}",
            "output_shape": "R^1",
            "structure": f"value = Linear({hidden_dim_int},1).squeeze(-1).",
            "structure_zh": f"value = Linear({hidden_dim_int},1).squeeze(-1)。",
            "distribution": "deterministic scalar V(s)",
            "distribution_zh": "确定性标量 V(s)。",
            "calculation_note": "Used by value_loss = mean((V(s_t)-normalized_return_t)^2).",
            "calculation_note_zh": "用于 value_loss = mean((V(s_t)-normalized_return_t)^2)。",
        },
    ]
    return pd.DataFrame(rows)


def rl_target_ctde_architecture_frame(
    agent_roles: pd.DataFrame | None = None,
    asset_registry: pd.DataFrame | None = None,
    deep_summary: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Describe the privacy-preserving CTDE architecture.

    When `privacy_separated_ctde_actor_critic` artifacts are present this frame
    describes the current primary trainer. Without those artifacts it remains
    the target CTDE specification used by the UI, docs and future work.
    """
    roles = _frame_or_empty(agent_roles)
    assets = _frame_or_empty(asset_registry)
    deep = _frame_or_empty(deep_summary)
    if not roles.empty and "role_type" in roles.columns:
        n_vpps = int((roles["role_type"].astype(str) == "vpp_dispatch_agent").sum())
        if n_vpps <= 0:
            n_vpps = int((roles["role_type"].astype(str) == "vpp_portfolio_agent").sum())
    else:
        n_vpps = 1
    n_vpps = max(1, n_vpps)
    n_der = int(assets["der_id"].nunique()) if not assets.empty and "der_id" in assets.columns else None
    hidden_dim = _first_value(deep, "hidden_dim", "64")
    try:
        hidden_dim_int = int(float(hidden_dim))
    except ValueError:
        hidden_dim_int = 64
    dso_input_dim = _first_value(deep, "dso_input_dim", "D_dso")
    vpp_input_dim = _first_value(deep, "vpp_input_dim", "D_vpp")
    portfolio_input_dim = _first_value(deep, "portfolio_input_dim", "D_portfolio")
    critic_input_dim = _first_value(deep, "critic_input_dim", "D_critic")
    critic_action_dim = _first_value(
        deep,
        "critic_action_dim",
        _first_value(deep, "critic_action_summary_dim", "D_action"),
    )
    max_der_per_vpp = _first_value(deep, "max_der_per_vpp", "K_i")
    vpp_encoder_type = _first_value(deep, "vpp_encoder_type", "der_deep_sets_token_pooling")
    critic_type = _first_value(deep, "critic_type", "action_conditioned_centralized_value_critic")
    der_label = str(n_der) if n_der is not None else "N_DER_total"
    rows = [
        {
            "component_id": "dso_private_observation_encoder",
            "component_group": "dso_actor_stack",
            "privacy_scope": "DSO sees network topology, security state, market context and VPP reports; it does not see VPP private DER internals in representative-data mode.",
            "privacy_scope_zh": "DSO 可见电网拓扑、安全状态、市场上下文和 VPP 上报信息；在 representative-data 模式下不可见 VPP 私有 DER 内部状态。",
            "execution_visibility": "DSO actor only",
            "execution_visibility_zh": "仅 DSO actor 执行时可见",
            "trainable": True,
            "input_shape": f"o_dso in R^{dso_input_dim} = grid_state + topology/security features + VPP bid/report summary + FR/DOE candidates",
            "output_shape": f"z_dso in R^{hidden_dim_int}",
            "structure": f"LayerNorm({dso_input_dim}) -> Linear({dso_input_dim},{hidden_dim_int}) -> Tanh -> Linear({hidden_dim_int},{hidden_dim_int}) -> Tanh.",
            "structure_zh": f"LayerNorm({dso_input_dim}) -> Linear({dso_input_dim},{hidden_dim_int}) -> Tanh -> Linear({hidden_dim_int},{hidden_dim_int}) -> Tanh。",
            "distribution": "deterministic latent representation",
            "distribution_zh": "确定性隐变量表示",
            "calculation_note": "z_dso is computed only from DSO-observable variables and VPP submitted bids/reports.",
            "calculation_note_zh": "z_dso 只由 DSO 可观测变量和 VPP 主动上报的报量/报价/摘要计算得到。",
            "loss_signal": "actor loss through DSO envelope action log_prob plus centralized critic advantage",
            "loss_signal_zh": "通过 DSO 包络动作 log_prob 与 centralized critic advantage 进入 actor loss",
            "conference_role": "privacy-preserving upper-policy encoder",
        },
        {
            "component_id": "dso_envelope_actor",
            "component_group": "dso_actor_stack",
            "privacy_scope": "Consumes z_dso and public/contracted VPP bid curves only.",
            "privacy_scope_zh": "只读取 z_dso 与公开/合约约定的 VPP 报量报价曲线。",
            "execution_visibility": "DSO actor only",
            "execution_visibility_zh": "仅 DSO actor 执行时可见",
            "trainable": True,
            "input_shape": f"z_dso in R^{hidden_dim_int}",
            "output_shape": f"Gaussian envelope preference in R^{n_vpps}",
            "structure": f"mean = tanh(Linear({hidden_dim_int},{n_vpps})); dso_log_std is a trainable Parameter in R^{n_vpps}.",
            "structure_zh": f"mean = tanh(Linear({hidden_dim_int},{n_vpps}))；dso_log_std 是 R^{n_vpps} 的可训练参数。",
            "distribution": "Normal(mu_dso, sigma_dso), then action squashing/clipping",
            "distribution_zh": "Normal(mu_dso, sigma_dso)，随后进行动作压缩和裁剪",
            "calculation_note": "The actor proposes intent; OPF/sensitivity/FR/DOE projection turns it into executable operating envelopes.",
            "calculation_note_zh": "actor 提出调节意图；OPF/灵敏度/FR/DOE 投影把意图转换成可执行运行包络。",
            "loss_signal": "DSO reward emphasizes grid feasibility, procurement cost, tracking and envelope margin.",
            "loss_signal_zh": "DSO reward 关注电网可行性、采购成本、跟踪误差与包络裕度。",
            "conference_role": "upper-level global guidance actor",
        },
        {
            "component_id": "vpp_local_observation_encoder",
            "component_group": "vpp_dispatch_stack",
            "privacy_scope": "Each VPP sees only its own DER states, own forecasts/costs, own history, DSO envelope and service/price signal.",
            "privacy_scope_zh": "每个 VPP 只看自身 DER 状态、预测/成本、历史响应、DSO 包络和服务/价格信号。",
            "execution_visibility": "one owning VPP actor; homogeneous VPPs may share parameters but not raw observations",
            "execution_visibility_zh": "仅所属 VPP actor 可见；同质 VPP 可共享参数，但不共享原始观测。",
            "trainable": True,
            "input_shape": f"o_vpp_i in R^{vpp_input_dim} = context R^16 + up to {max_der_per_vpp} DER tokens, each R^26",
            "output_shape": f"z_vpp_i in R^{hidden_dim_int}",
            "structure": (
                f"{vpp_encoder_type}: context encoder LayerNorm(16)->Linear(16,{hidden_dim_int}); "
                f"shared DER-token encoder LayerNorm(26)->Linear(26,{hidden_dim_int})->Tanh->Linear({hidden_dim_int},{hidden_dim_int}); "
                f"masked mean/max pooling over {max_der_per_vpp} tokens plus token-count ratio; fusion Linear({3 * hidden_dim_int + 1},{hidden_dim_int})->Tanh->Linear({hidden_dim_int},{hidden_dim_int})->Tanh."
            ),
            "structure_zh": (
                f"{vpp_encoder_type}：上下文编码器 LayerNorm(16)->Linear(16,{hidden_dim_int})；"
                f"共享 DER token 编码器 LayerNorm(26)->Linear(26,{hidden_dim_int})->Tanh->Linear({hidden_dim_int},{hidden_dim_int})；"
                f"对最多 {max_der_per_vpp} 个 DER token 做 mask mean/max pooling，并拼接 token-count ratio；再经 fusion Linear({3 * hidden_dim_int + 1},{hidden_dim_int})->Tanh->Linear({hidden_dim_int},{hidden_dim_int})->Tanh。"
            ),
            "distribution": "deterministic local latent representation",
            "distribution_zh": "确定性本地隐变量表示",
            "calculation_note": "Parameter sharing is allowed only for homogeneous VPP actors; DER tokens remain local and are pooled inside the owning VPP actor.",
            "calculation_note_zh": "同质 VPP actor 可以参数共享；但 DER token 仍保持本地化，只在所属 VPP actor 内编码和池化。",
            "loss_signal": "VPP dispatch actors are updated from their own settlement-aware dispatch return; parameters may be shared, but rewards are per VPP.",
            "loss_signal_zh": "当前 loss 使用广播给 VPP actor 的 centralized advantage；结算感知本地 reward 是下一步升级。",
            "conference_role": "decentralized lower-policy encoder",
        },
        {
            "component_id": "vpp_der_dispatch_actor",
            "component_group": "vpp_dispatch_stack",
            "privacy_scope": "Outputs only for the owning VPP's DER portfolio.",
            "privacy_scope_zh": "只输出所属 VPP 的 DER 组合动作。",
            "execution_visibility": "owning VPP actor",
            "execution_visibility_zh": "所属 VPP actor",
            "trainable": True,
            "input_shape": f"z_vpp_i in R^{hidden_dim_int} + envelope_i",
            "output_shape": f"selected aggregate P in R^1 plus DER-level normalized actions in R^{max_der_per_vpp}; Q control is not active yet",
            "structure": f"aggregate_mean = tanh(Linear({hidden_dim_int},1)); der_mean = tanh(Linear({hidden_dim_int},{max_der_per_vpp})); aggregate_log_std in R^1 and der_log_std in R^{max_der_per_vpp} are trainable.",
            "structure_zh": f"aggregate_mean = tanh(Linear({hidden_dim_int},1))；der_mean = tanh(Linear({hidden_dim_int},{max_der_per_vpp}))；aggregate_log_std 属于 R^1，der_log_std 属于 R^{max_der_per_vpp}，均可训练。",
            "distribution": "Normal(mu_agg, sigma_agg) and Normal(mu_der, sigma_der), then clipping/projection",
            "distribution_zh": "Normal(mu_agg, sigma_agg) 与 Normal(mu_der, sigma_der)，随后裁剪/投影",
            "calculation_note": "Current implementation learns aggregate P and normalized active-power DER proposals; safety projection still performs feasibility repair.",
            "calculation_note_zh": "当前实现学习聚合 P 与 DER 归一化有功建议；安全投影仍负责可行性修复。",
            "loss_signal": "implemented: local delivery/service reward - DER cost - SOC/comfort penalties - tracking/projection penalty",
            "loss_signal_zh": "当前：全局 centralized advantage；目标：本地履约收益 - DER 成本 - SOC/舒适度惩罚 - 平滑项 - 包络越界惩罚",
            "conference_role": "decentralized DER-level dispatch actor",
        },
        {
            "component_id": "vpp_portfolio_slow_encoder",
            "component_group": "vpp_portfolio_stack",
            "privacy_scope": "Each VPP uses its own long-cycle reliability, profit, risk and capability-belief history.",
            "privacy_scope_zh": "每个 VPP 只使用自身长周期可靠性、收益、风险和能力置信历史。",
            "execution_visibility": "owning VPP portfolio actor",
            "execution_visibility_zh": "所属 VPP 组合配置 actor",
            "trainable": True,
            "input_shape": f"h_vpp_i in R^{portfolio_input_dim} = multi-day settlement + reliability + delivery risk + asset capability belief",
            "output_shape": f"z_portfolio_i in R^{hidden_dim_int}",
            "structure": f"LayerNorm({portfolio_input_dim}) -> Linear({portfolio_input_dim},{hidden_dim_int}) -> Tanh -> Linear({hidden_dim_int},{hidden_dim_int}) -> Tanh.",
            "structure_zh": f"LayerNorm({portfolio_input_dim}) -> Linear({portfolio_input_dim},{hidden_dim_int}) -> Tanh -> Linear({hidden_dim_int},{hidden_dim_int}) -> Tanh。",
            "distribution": "deterministic slow-loop latent representation",
            "distribution_zh": "确定性慢周期隐变量表示",
            "calculation_note": "This module decides commercial/aggregation configuration, not direct electrical setpoints.",
            "calculation_note_zh": "该模块判断商业/聚合配置，而不是直接给出电气出力设定值。",
            "loss_signal": "multi-day settlement reward, reliability and configuration-switching cost",
            "loss_signal_zh": "多日结算收益、可靠性和配置切换成本",
            "conference_role": "slow-loop portfolio representation",
        },
        {
            "component_id": "vpp_portfolio_actor",
            "component_group": "vpp_portfolio_stack",
            "privacy_scope": "Only produces recommendations for the owning VPP's commercial aggregation weights or membership proposal.",
            "privacy_scope_zh": "只为所属 VPP 产生商业聚合权重或成员变更建议。",
            "execution_visibility": "owning VPP portfolio actor; physical changes remain event-gated",
            "execution_visibility_zh": "所属 VPP 组合配置 actor；物理变更仍由事件门控",
            "trainable": True,
            "input_shape": f"z_portfolio_i in R^{hidden_dim_int}",
            "output_shape": "Categorical logits over keep/reweight/propose_membership_change",
            "structure": f"portfolio_logits = Linear({hidden_dim_int},3); Categorical(logits) over keep/reweight/propose_membership_change.",
            "structure_zh": f"portfolio_logits = Linear({hidden_dim_int},3)；Categorical(logits) 在 keep/reweight/propose_membership_change 上采样。",
            "distribution": "Categorical(logits)",
            "distribution_zh": "Categorical(logits)",
            "calculation_note": "A portfolio action can adjust aggregation weights or propose membership changes; it cannot directly mutate pandapower rows.",
            "calculation_note_zh": "组合配置动作可调整聚合权重或提出成员变更；不能直接修改 pandapower 元件行。",
            "loss_signal": "policy gradient with slow-loop portfolio reward and switch-cost regularization",
            "loss_signal_zh": "使用慢周期组合 reward 和切换成本正则的 policy gradient",
            "conference_role": "slow-loop VPP configuration actor",
        },
        {
            "component_id": "centralized_training_critic",
            "component_group": "training_only_critic",
            "privacy_scope": "Training-only critic can read critic_global_state under the simulator/researcher trust boundary.",
            "privacy_scope_zh": "仅训练期 critic 可在仿真/研究者可信边界内读取 critic_global_state。",
            "execution_visibility": "not used by deployed decentralized actors",
            "execution_visibility_zh": "分散执行 actor 部署时不可见",
            "trainable": True,
            "input_shape": f"critic_global_state in R^{critic_input_dim} + joint_action_summary in R^{critic_action_dim}",
            "output_shape": "V_dso plus per-VPP V_dispatch_i and per-VPP V_portfolio_i value heads in the advanced HAPPO trainer",
            "structure": (
                f"{critic_type}: state encoder LayerNorm({critic_input_dim})->Linear({critic_input_dim},{hidden_dim_int})->Tanh->Linear({hidden_dim_int},{hidden_dim_int})->Tanh; "
                f"action encoder LayerNorm({critic_action_dim})->Linear({critic_action_dim},{hidden_dim_int})->Tanh->Linear({hidden_dim_int},{hidden_dim_int})->Tanh; "
                f"fusion Linear({2 * hidden_dim_int},{hidden_dim_int})->Tanh; "
                "multi-head values Linear(hidden,n_heads) for DSO, every VPP dispatch return and every VPP portfolio return."
            ),
            "structure_zh": (
                f"{critic_type}：状态编码器 LayerNorm({critic_input_dim})->Linear({critic_input_dim},{hidden_dim_int})->Tanh->Linear({hidden_dim_int},{hidden_dim_int})->Tanh；"
                f"动作摘要编码器 LayerNorm({critic_action_dim})->Linear({critic_action_dim},{hidden_dim_int})->Tanh；"
                f"融合层 Linear({2 * hidden_dim_int},{hidden_dim_int})->Tanh；高级 HAPPO 输出 DSO、每个 VPP dispatch、每个 VPP portfolio 的多头 value。"
            ),
            "distribution": "deterministic role-specific value estimates",
            "distribution_zh": "确定性价值估计",
            "calculation_note": "Advanced HAPPO now uses an action-conditioned centralized critic, per-role/per-VPP value heads, GAE-lambda advantages, PPO clipping, sequential role updates and cumulative importance correction.",
            "calculation_note_zh": "高级 HAPPO 现在使用动作条件 centralized critic、按角色/按 VPP 拆分的 value head、GAE-lambda、PPO clip、顺序角色更新和累计重要性校正。",
            "loss_signal": "critic loss fits DSO, dispatch and portfolio role returns; actor losses use role GAE advantages and PPO clipping.",
            "loss_signal_zh": "critic loss：MSE(V(s_t, a_t_summary), 归一化 Monte-Carlo return)；actor 使用 A_t = return - V(s_t, a_t_summary).detach() 更新",
            "conference_role": "CTDE centralized critic",
        },
        {
            "component_id": "non_rl_safety_projection",
            "component_group": "guardrail",
            "privacy_scope": "Consumes proposed actions, public envelope limits and local device limits required for safe execution.",
            "privacy_scope_zh": "读取候选动作、公开包络边界和安全执行所需的本地设备边界。",
            "execution_visibility": "execution guard after all actors",
            "execution_visibility_zh": "所有 actor 之后的执行保护层",
            "trainable": False,
            "input_shape": "raw actor actions + FR/DOE + DER bounds + pandapower sign conventions",
            "output_shape": "physically executable setpoints and projection residuals",
            "structure": "Deterministic feasibility repair: bound clipping, residual allocation, SOC/comfort checks and runpp security validation.",
            "structure_zh": "确定性可行性修复：边界裁剪、残差分配、SOC/舒适度检查和 runpp 安全校核。",
            "distribution": "not a policy distribution",
            "distribution_zh": "不是策略分布",
            "calculation_note": "Safety projection is not a MARL agent and should be reported separately from learned actors.",
            "calculation_note_zh": "安全投影不是 MARL 智能体，应与可学习 actor 分开汇报。",
            "loss_signal": "no direct gradient; projection residuals can be logged as penalties for learning",
            "loss_signal_zh": "无直接梯度；投影残差可记录为学习惩罚项",
            "conference_role": "safety layer / shield",
        },
    ]
    return pd.DataFrame(rows)


def rl_ctde_node_frame(
    agent_roles: pd.DataFrame | None = None,
    asset_registry: pd.DataFrame | None = None,
    deep_summary: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Machine-readable CTDE graph nodes for HTML/SVG rendering."""

    target = rl_target_ctde_architecture_frame(
        agent_roles=agent_roles,
        asset_registry=asset_registry,
        deep_summary=deep_summary,
    )
    dims = _architecture_dims(
        agent_roles=agent_roles,
        asset_registry=asset_registry,
        deep_summary=deep_summary,
    )
    source_map = {
        "dso_private_observation_encoder": ("encode_dso_observation", "learning/deep_rl.py"),
        "dso_envelope_actor": ("PrivacySeparatedCTDEModules.dso_actor", "learning/deep_rl.py"),
        "vpp_local_observation_encoder": ("encode_vpp_dispatch_observation + DeepSetDispatchEncoder", "learning/ctde_networks.py"),
        "vpp_der_dispatch_actor": ("PrivacySeparatedCTDEModules.vpp_dispatch_actor", "learning/ctde_networks.py"),
        "vpp_portfolio_slow_encoder": ("encode_vpp_portfolio_observation", "learning/deep_rl.py"),
        "vpp_portfolio_actor": ("PrivacySeparatedCTDEModules.vpp_portfolio_actor", "learning/deep_rl.py"),
        "centralized_training_critic": ("encode_critic_global_state + CentralizedActionConditionedCritic", "learning/ctde_networks.py"),
        "non_rl_safety_projection": ("MultiAgentVPPDSOEnv.step + Simulator.step", "envs/multi_agent_env.py"),
    }
    sharing_map = {
        "dso_private_observation_encoder": "DSO-only parameters",
        "dso_envelope_actor": "DSO-only parameters",
        "vpp_local_observation_encoder": "per-VPP execution encoder by default; no raw observation sharing",
        "vpp_der_dispatch_actor": "per-VPP dispatch actor by default; optional sharing is an ablation",
        "vpp_portfolio_slow_encoder": "per-VPP slow-loop portfolio encoder by default",
        "vpp_portfolio_actor": "per-VPP slow-loop portfolio actor by default; optional sharing is an ablation",
        "centralized_training_critic": "training-only global critic parameters",
        "non_rl_safety_projection": "deterministic guardrail; no trainable parameters",
    }
    limitation_map = {
        "dso_private_observation_encoder": "flat engineered vector; no public-grid GNN/zone token encoder yet",
        "dso_envelope_actor": "current output is active-power target preference only; full FR/DOE geometry, Q and service logits are future work",
        "vpp_local_observation_encoder": "Deep Sets token pooling is implemented; attention and temporal history encoders are future upgrades",
        "vpp_der_dispatch_actor": "DER action is normalized active-power proposal; Q control and richer device policies are future work",
        "vpp_portfolio_slow_encoder": "9-D static portfolio vector; no multi-day sequence/history encoder yet",
        "vpp_portfolio_actor": "uses proxy reward and event gate; true long-horizon portfolio profit reward is future work",
        "centralized_training_critic": "advanced HAPPO has per-VPP value heads plus sequential correction; future upgrades are topology-aware critics, certified envelopes and counterfactual credit ablations",
        "non_rl_safety_projection": "non-differentiable projection; residuals are logged as learning penalties",
    }
    phase_map = {
        "dso_private_observation_encoder": "runtime_and_training",
        "dso_envelope_actor": "runtime_and_training",
        "vpp_local_observation_encoder": "runtime_and_training",
        "vpp_der_dispatch_actor": "runtime_and_training",
        "vpp_portfolio_slow_encoder": "runtime_and_training_slow_loop",
        "vpp_portfolio_actor": "runtime_and_training_slow_loop",
        "centralized_training_critic": "training_only",
        "non_rl_safety_projection": "runtime_guardrail",
    }
    tensor_names = {
        "dso_private_observation_encoder": ("o_dso", "z_dso"),
        "dso_envelope_actor": ("z_dso", "a_dso"),
        "vpp_local_observation_encoder": ("o_vpp_i", "z_vpp_i"),
        "vpp_der_dispatch_actor": ("z_vpp_i + envelope_i", "a_vpp_i = (selected_p_i, u_der_i)"),
        "vpp_portfolio_slow_encoder": ("h_vpp_i", "z_portfolio_i"),
        "vpp_portfolio_actor": ("z_portfolio_i", "g_i"),
        "centralized_training_critic": ("critic_global_state", "V_dso,{V_dispatch_i},{V_portfolio_i}"),
        "non_rl_safety_projection": ("raw actor actions + FR/DOE + DER bounds", "safe setpoints + projection_gap"),
    }
    rows: list[dict[str, Any]] = []
    for _, row in target.iterrows():
        component_id = _text(row.get("component_id"))
        source_fn, source_file = source_map.get(component_id, ("", ""))
        tensor_in, tensor_out = tensor_names.get(component_id, ("", ""))
        rows.append(
            {
                "component_id": component_id,
                "component_group": row.get("component_group", ""),
                "phase": phase_map.get(component_id, "runtime_and_training"),
                "owner_scope": row.get("execution_visibility", ""),
                "owner_scope_zh": row.get("execution_visibility_zh", ""),
                "parameter_sharing_scope": sharing_map.get(component_id, ""),
                "trainable": row.get("trainable", ""),
                "tensor_in": tensor_in,
                "input_shape": row.get("input_shape", ""),
                "tensor_out": tensor_out,
                "output_shape": row.get("output_shape", ""),
                "distribution": row.get("distribution", ""),
                "structure": row.get("structure", ""),
                "source_fn": source_fn,
                "source_file": source_file,
                "implemented_status": "implemented_current_ctde_primary"
                if _privacy_separated_primary(deep_summary)
                else "specified_target_or_shared_benchmark_context",
                "limitation": limitation_map.get(component_id, ""),
                "next_upgrade": _node_next_upgrade(component_id, dims),
            }
        )
    return pd.DataFrame(rows)


def _node_next_upgrade(component_id: str, dims: dict[str, str]) -> str:
    if component_id == "dso_private_observation_encoder":
        return "Replace flat MLP input with public-grid GNN or zone-token encoder while keeping VPP private observations outside the DSO actor."
    if component_id == "dso_envelope_actor":
        return f"Output target_p, target_q, envelope margins, service logits and price adders for {dims.get('n_vpps', 'N_vpp')} VPPs."
    if component_id == "vpp_local_observation_encoder":
        return "Upgrade Deep Sets pooling to Set Transformer attention plus 4-8 step local history GRU."
    if component_id == "vpp_der_dispatch_actor":
        return "Add device-type heads, Q/VAR support and local settlement-aware reward terms."
    if component_id == "vpp_portfolio_slow_encoder":
        return "Encode multi-day profit, reliability, non-delivery and portfolio-change history."
    if component_id == "vpp_portfolio_actor":
        return "Train accepted portfolio changes over long-horizon episodes with contract and network safety checks."
    if component_id == "centralized_training_critic":
        return "Add true multi-epoch MAPPO/HAPPO updates, sequential importance correction, V-trace/off-policy replay and counterfactual credit."
    return "Keep deterministic and auditable; expose projection residuals to learning as penalties."


def rl_ctde_edge_frame(
    agent_roles: pd.DataFrame | None = None,
    asset_registry: pd.DataFrame | None = None,
    deep_summary: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Machine-readable CTDE graph edges with privacy/training semantics."""

    dims = _architecture_dims(
        agent_roles=agent_roles,
        asset_registry=asset_registry,
        deep_summary=deep_summary,
    )
    n_vpps = dims["n_vpps"]
    max_der = dims["max_der_per_vpp"]
    hidden = dims["hidden_dim"]
    edges = [
        (
            "edge_dso_obs_encoder",
            "dso_observation",
            "dso_private_observation_encoder",
            "o_dso",
            f"R^{dims['dso_input_dim']}",
            "runtime_flow",
            "dso_public_or_reported",
            False,
            False,
        ),
        (
            "edge_dso_encoder_actor",
            "dso_private_observation_encoder",
            "dso_envelope_actor",
            "z_dso",
            f"R^{hidden}",
            "runtime_flow",
            "dso_only_latent",
            False,
            False,
        ),
        (
            "edge_dso_actor_projection",
            "dso_envelope_actor",
            "non_rl_safety_projection",
            "a_dso",
            f"R^{n_vpps}",
            "runtime_flow",
            "contracted_envelope_signal",
            False,
            False,
        ),
        (
            "edge_dso_envelope_to_vpp",
            "dso_envelope_actor",
            "vpp_der_dispatch_actor",
            "envelope_i / price_i / service_i",
            "per VPP scalar/bounds",
            "runtime_flow",
            "contracted_dso_to_vpp_signal",
            False,
            False,
        ),
        (
            "edge_vpp_obs_encoder",
            "vpp_local_private_observation",
            "vpp_local_observation_encoder",
            "o_vpp_i",
            f"R^{dims['vpp_input_dim']}",
            "runtime_flow",
            "private_to_owning_vpp_only",
            True,
            False,
        ),
        (
            "edge_vpp_encoder_actor",
            "vpp_local_observation_encoder",
            "vpp_der_dispatch_actor",
            "z_vpp_i",
            f"R^{hidden}",
            "runtime_flow",
            "owning_vpp_latent_only",
            False,
            False,
        ),
        (
            "edge_vpp_actor_projection",
            "vpp_der_dispatch_actor",
            "non_rl_safety_projection",
            "selected_p_i + u_der_i",
            f"R^1 + R^{max_der}",
            "runtime_flow",
            "own_vpp_action_summary",
            False,
            False,
        ),
        (
            "edge_portfolio_obs_encoder",
            "vpp_portfolio_private_history",
            "vpp_portfolio_slow_encoder",
            "h_vpp_i",
            f"R^{dims['portfolio_input_dim']}",
            "slow_runtime_flow",
            "private_to_owning_vpp_only",
            True,
            False,
        ),
        (
            "edge_portfolio_encoder_actor",
            "vpp_portfolio_slow_encoder",
            "vpp_portfolio_actor",
            "z_portfolio_i",
            f"R^{hidden}",
            "slow_runtime_flow",
            "owning_vpp_latent_only",
            False,
            False,
        ),
        (
            "edge_portfolio_actor_gate",
            "vpp_portfolio_actor",
            "portfolio_event_gate",
            "g_i",
            "Categorical{keep,reweight,propose}",
            "slow_runtime_flow",
            "commercial_configuration_signal",
            False,
            False,
        ),
        (
            "edge_projection_pandapower",
            "non_rl_safety_projection",
            "pandapower_runpp",
            "safe setpoints",
            "load/sgen/storage/gen writes",
            "runtime_flow",
            "physical_execution",
            False,
            False,
        ),
        (
            "edge_env_critic",
            "pandapower_runpp",
            "centralized_training_critic",
            "critic_global_state + joint_action_summary",
            f"R^{dims['critic_input_dim']} + R^{dims['critic_action_dim']}",
            "training_only_flow",
            "training_privileged_state",
            False,
            False,
        ),
        (
            "edge_critic_loss_to_dso",
            "centralized_training_critic",
            "dso_envelope_actor",
            "A_dso,t clipped surrogate gradient",
            "role GAE advantage",
            "gradient_flow",
            "training_only_no_execution_visibility",
            False,
            True,
        ),
        (
            "edge_critic_loss_to_vpp",
            "centralized_training_critic",
            "vpp_der_dispatch_actor",
            "A_dispatch,t clipped surrogate gradient",
            "role GAE advantage",
            "gradient_flow",
            "training_only_no_execution_visibility",
            False,
            True,
        ),
        (
            "edge_critic_loss_to_portfolio",
            "centralized_training_critic",
            "vpp_portfolio_actor",
            "A_portfolio,t clipped surrogate gradient",
            "role GAE advantage",
            "gradient_flow",
            "training_only_no_execution_visibility",
            False,
            True,
        ),
    ]
    return pd.DataFrame(
        [
            {
                "edge_id": edge_id,
                "src_component_id": src,
                "dst_component_id": dst,
                "signal_name": signal,
                "signal_shape": shape,
                "signal_type": signal_type,
                "phase": "training" if "training" in signal_type or "gradient" in signal_type else "execution",
                "privacy_class": privacy,
                "arrow_style": "dashed_orange" if "gradient" in signal_type else "solid_blue",
                "carries_raw_private_obs": carries_private,
                "carries_gradient": carries_gradient,
                "implemented_status": "implemented_current_ctde_primary"
                if _privacy_separated_primary(deep_summary)
                else "specified_target_or_benchmark_context",
            }
            for (
                edge_id,
                src,
                dst,
                signal,
                shape,
                signal_type,
                privacy,
                carries_private,
                carries_gradient,
            ) in edges
        ]
    )


def rl_ctde_feedback_frame(deep_summary: pd.DataFrame | None = None) -> pd.DataFrame:
    """Loss/reward feedback paths used by the current CTDE trainer."""

    return pd.DataFrame(
        [
            {
                "feedback_id": "feedback_dso_actor",
                "target_component_id": "dso_envelope_actor",
                "loss_name": "dso_policy_loss",
                "formula": "L_dso = -mean(min(r_t*A_dso,t, clip(r_t,1-eps,1+eps)*A_dso,t)); A from GAE(lambda)",
                "coefficient": "c_dso=1.0",
                "advantage_source": "centralized_training_critic V_dso(s_t, a_t_summary) with GAE-lambda",
                "reward_source": "role-specific DSO grid-safety/procurement reward from MultiAgentVPPDSOEnv",
                "metric_csv": "outputs/deep_rl/deep_rl_loss_metrics.csv",
                "metric_columns": "dso_policy_loss,total_loss,mean_reward",
                "update_frequency": "HAPPO sequential update after rollout; legacy MAPPO/HAPPO-lite trainer remains available for compatibility",
            },
            {
                "feedback_id": "feedback_vpp_dispatch_actor",
                "target_component_id": "vpp_der_dispatch_actor",
                "loss_name": "vpp_dispatch_policy_loss",
                "formula": "L_vpp = -mean(min(r_t*A_dispatch,t, clip(r_t,1-eps,1+eps)*A_dispatch,t)); A from dispatch GAE(lambda)",
                "coefficient": "c_vpp=1.0",
                "advantage_source": "centralized_training_critic V_dispatch head plus VPP-local settlement-aware dispatch reward",
                "reward_source": "role-specific VPP dispatch reward: private profit, service payment, tracking, DER cost and projection penalty",
                "metric_csv": "outputs/deep_rl/deep_rl_loss_metrics.csv",
                "metric_columns": "vpp_dispatch_policy_loss,action_projection_penalty,projection_gap_mw,tracking_bonus",
                "update_frequency": "per-VPP HAPPO sequential update after DSO; optional shared actor is only an ablation",
            },
            {
                "feedback_id": "feedback_vpp_portfolio_actor",
                "target_component_id": "vpp_portfolio_actor",
                "loss_name": "portfolio_policy_loss",
                "formula": "L_portfolio = -mean(min(r_t*A_portfolio,t, clip(r_t,1-eps,1+eps)*A_portfolio,t)); A from portfolio GAE(lambda)",
                "coefficient": "c_portfolio=0.25",
                "advantage_source": "centralized_training_critic per-VPP V_portfolio_i head plus VPP slow-loop portfolio return",
                "reward_source": "role-specific portfolio reward: long-horizon profit proxy, reliability, localized DSO-alignment credit and switching cost",
                "metric_csv": "outputs/deep_rl/deep_rl_step_metrics.csv",
                "metric_columns": "portfolio_proxy_reward,portfolio_action,portfolio_entropy",
                "update_frequency": "slow-loop HAPPO update only on portfolio decision steps; physical membership changes remain event-gated",
            },
            {
                "feedback_id": "feedback_centralized_critic",
                "target_component_id": "centralized_training_critic",
                "loss_name": "critic_value_loss",
                "formula": "L_critic = mean(MSE(V_dso,G_dso^GAE)+sum_i MSE(V_dispatch_i,G_dispatch_i^GAE)+sum_i MSE(V_portfolio_i,G_portfolio_i^GAE))",
                "coefficient": "value_coef=0.50",
                "advantage_source": "role-specific GAE returns",
                "reward_source": "DSO, VPP dispatch and VPP portfolio role reward streams",
                "metric_csv": "outputs/deep_rl/deep_rl_loss_metrics.csv",
                "metric_columns": "value_loss,dso_value_loss,dispatch_value_loss,portfolio_value_loss,total_loss",
                "update_frequency": "every episode",
            },
            {
                "feedback_id": "feedback_entropy_regularizer",
                "target_component_id": "all_trainable_actors",
                "loss_name": "entropy_loss",
                "formula": "L_entropy = -mean(entropy_dso + entropy_vpp + entropy_portfolio)",
                "coefficient": "entropy_coef=0.01",
                "advantage_source": "policy distributions",
                "reward_source": "regularization, not environment reward",
                "metric_csv": "outputs/deep_rl/deep_rl_loss_metrics.csv",
                "metric_columns": "entropy_loss",
                "update_frequency": "every episode",
            },
        ]
    )


def rl_algorithm_overview_frame(deep_summary: pd.DataFrame | None = None) -> pd.DataFrame:
    deep = _frame_or_empty(deep_summary)
    algorithm = _first_value(deep, "algorithm", "privacy_separated_ctde_actor_critic")
    optimizer_steps = _first_value(deep, "optimizer_steps", "not_run")
    param_delta = _first_value(deep, "param_delta_l2", "not_run")
    portfolio_trainable = _first_value(deep, "portfolio_trainable", "false")
    status = _first_value(deep, "status", "architecture_defined_training_optional")
    privacy_separated = _bool(_first_value(deep, "privacy_separated_execution", "false")) or (
        algorithm == "privacy_separated_ctde_actor_critic"
    )
    if privacy_separated:
        algorithm_family = "privacy-preserving CTDE actor-critic"
        algorithm_family_zh = "隐私分离 CTDE Actor-Critic"
        execution_mode = "decentralized_execution_with_separate_dso_vpp_portfolio_actors_and_safety_projection"
        execution_mode_zh = "DSO/VPP/组合配置独立 actor 分散执行 + 安全投影"
        ctde_status = "implemented_privacy_separated_ctde_training_loop"
        ctde_status_zh = "已实现隐私分离 CTDE 训练闭环"
        critic_scope = "centralized critic reads critic_global_state during training only"
        critic_scope_zh = "集中式 critic 仅在训练期读取 critic_global_state"
        actor_scope = (
            "separate DSO actor, per-VPP dispatch actors and per-VPP slow-loop portfolio actors by default; "
            "raw VPP observations remain local"
        )
        actor_scope_zh = "独立 DSO actor、同质参数共享 VPP 调度 actor 与 VPP 组合配置 actor；VPP 原始观测保持本地化。"
        loss_formula = (
            "L = c_dso*L_dso_actor + c_vpp*L_vpp_dispatch + c_portfolio*L_portfolio "
            "+ value_coef*L_critic + entropy_coef*L_entropy"
        )
        loss_formula_zh = "L = c_dso*DSO actor损失 + c_vpp*VPP调度损失 + c_portfolio*组合配置损失 + value_coef*critic损失 + entropy_coef*熵损失"
        plain_language_summary = (
            "The recommended advanced HAPPO trainer now follows the target privacy-separated hierarchical CTDE design. "
            "The DSO actor is separate, every VPP owns an independent dispatch actor, every VPP owns an independent "
            "slow-loop portfolio actor, and the centralized critic uses critic_global_state plus a compact joint "
            "action summary only during training. The legacy privacy_separated_ctde_actor_critic path remains for "
            "report compatibility."
        )
        plain_language_summary_zh = (
            "当前推荐的高级 HAPPO 训练器已经按目标隐私分离分层 CTDE 设计升级：DSO actor 独立，"
            "每个 VPP 拥有独立调度 actor，每个 VPP 拥有独立慢周期组合配置 actor；centralized critic "
            "只在训练期读取 critic_global_state 与联合动作摘要。旧的 privacy_separated_ctde_actor_critic 路线保留用于报告兼容。"
        )
        current_vs_target = (
            "Implemented: hierarchical HAPPO trainer with DSO actor, per-VPP dispatch actors, per-VPP slow-loop "
            "portfolio actors, sequential importance-corrected updates and centralized multi-head value critic. "
            "Remaining research work: Set Transformer or GNN/temporal encoders, stronger settlement-aware local "
            "rewards, certified DSO envelopes and long-budget validation."
        )
        current_vs_target_zh = (
            "已实现：分层 HAPPO 训练器、DSO actor、每 VPP 独立 dispatch actor、每 VPP 独立慢周期 portfolio actor、顺序重要性校正更新和集中式多头 value critic。后续研究工作：Set Transformer/GNN/时序编码器、结算感知本地 reward、认证 DSO 包络和长预算验证。"
        )
        target_ctde_status = "advanced_hierarchical_happo_implemented_as_recommended_trainer"
        target_ctde_status_zh = "目标分层 HAPPO 已作为推荐训练器实现。"
    else:
        algorithm_family = "centralized shared actor-critic baseline"
        algorithm_family_zh = "集中式共享 Actor-Critic 基线"
        execution_mode = "envelope_guidance_plus_learned_der_disaggregation_with_safety_projection"
        execution_mode_zh = "运行包络引导 + 学习型 DER 解聚合 + 安全投影"
        ctde_status = "proto_ctde_interface_not_full_ctde"
        ctde_status_zh = "具备 CTDE 接口雏形，但还不是完整 CTDE"
        critic_scope = "centralized value head reads the DSO encoded global observation"
        critic_scope_zh = "集中式价值头读取 DSO 编码后的全局观测"
        actor_scope = (
            "one shared neural policy emits DSO envelope-preference targets, VPP aggregate choices "
            "and DER-level VPP actions"
        )
        actor_scope_zh = "一个共享神经策略同时输出 DSO 包络偏好目标、VPP 聚合选择和 DER 级 VPP 动作"
        loss_formula = "L = L_policy + value_coef * L_value + entropy_coef * L_entropy"
        loss_formula_zh = "L = 策略损失 + value_coef * 价值损失 + entropy_coef * 熵损失"
        plain_language_summary = (
            "The current model is a real PyTorch actor-critic loop. The DSO head learns envelope-preference "
            "targets; each VPP dispatch head learns an aggregate selected_p_mw plus DER-level normalized "
            "setpoint proposals; the VPP portfolio head is now trained as a categorical slow-loop commercial "
            "configuration proposal. A deterministic safety layer still clips device bounds and repairs residuals, "
            "and physical portfolio membership changes remain gated."
        )
        plain_language_summary_zh = (
            "当前模型已经是真正的 PyTorch Actor-Critic 闭环。DSO head 学习运行包络偏好目标；"
            "每个 VPP dispatch head 学习 selected_p_mw 与 DER 级归一化设定值。确定性安全层仍负责设备边界裁剪和残差修复。"
        )
        current_vs_target = (
            "Current: shared centralized actor emits envelope-preference targets, DER-level VPP actions and "
            "trainable portfolio proposals. "
            "Target: privacy-preserving CTDE with separate DSO, VPP dispatch and VPP portfolio encoders/actors, "
            "centralized critic during training, local observations at execution and settlement-aware rewards."
        )
        current_vs_target_zh = (
            "当前：共享集中式 actor 输出包络偏好目标和 DER 级 VPP 动作。"
            "目标：升级为隐私分离 CTDE，DSO、VPP 调度和 VPP 组合配置使用独立 encoder/actor，训练时使用集中式 critic，执行时只使用本地观测。"
        )
        target_ctde_status = "specified_privacy_preserving_ctde_not_yet_primary_trainer"
        target_ctde_status_zh = "已给出隐私分离 CTDE 目标规格；当前主训练器仍是共享骨干 benchmark。"
    return pd.DataFrame(
        [
            {
                "algorithm_id": algorithm,
                "algorithm_family": algorithm_family,
                "algorithm_family_zh": algorithm_family_zh,
                "training_mode": "centralized_training",
                "training_mode_zh": "集中式训练",
                "execution_mode": execution_mode,
                "execution_mode_zh": execution_mode_zh,
                "ctde_status": ctde_status,
                "ctde_status_zh": ctde_status_zh,
                "critic_scope": critic_scope,
                "critic_scope_zh": critic_scope_zh,
                "actor_scope": actor_scope,
                "actor_scope_zh": actor_scope_zh,
                "reward_formula": (
                    "role-specific general-sum rewards: r_dso for grid safety/procurement; "
                    "r_dispatch_i for each VPP's private profit and delivery; "
                    "r_portfolio_i for long-horizon profit, reliability and localized DSO-alignment credit"
                ),
                "reward_formula_zh": (
                    "角色分离 general-sum reward：r_dso 关注电网安全与采购成本；"
                    "r_dispatch_i 关注每个 VPP 自身利润和履约；"
                    "r_portfolio_i 关注长周期收益、可靠性和局部化 DSO 对齐收益"
                ),
                "loss_formula": loss_formula,
                "loss_formula_zh": loss_formula_zh,
                "trainable_components": (
                    "DSO envelope-preference actor, VPP aggregate actor, VPP DER-level disaggregation actor, "
                    "VPP slow-loop portfolio categorical actor, centralized value head"
                ),
                "trainable_components_zh": "DSO 包络偏好 actor、VPP 聚合选择 actor、VPP DER 级解聚合 actor、集中式价值头",
                "non_trainable_components": "safety projection, aggregate residual repair, physical portfolio event gate",
                "non_trainable_components_zh": "安全投影、聚合残差修复、确定性组合配置事件记录",
                "portfolio_trainable": portfolio_trainable,
                "optimizer_steps": optimizer_steps,
                "param_delta_l2": param_delta,
                "status": status,
                "plain_language_summary": plain_language_summary,
                "plain_language_summary_zh": plain_language_summary_zh,
                "current_vs_target": current_vs_target,
                "current_vs_target_zh": current_vs_target_zh,
                "target_ctde_status": target_ctde_status,
                "target_ctde_status_zh": target_ctde_status_zh,
                "target_privacy_rule": (
                    "DSO and VPP actors must not share raw observations or a single execution encoder. "
                    "Only the training-time critic may consume critic_global_state."
                ),
                "target_privacy_rule_zh": (
                    "DSO 与 VPP actor 不能共享原始观测或同一个执行期编码器；只有训练期 centralized critic 可以读取 critic_global_state。"
                ),
            }
        ]
    )


def _architecture_dims(
    agent_roles: pd.DataFrame | None = None,
    asset_registry: pd.DataFrame | None = None,
    deep_summary: pd.DataFrame | None = None,
) -> dict[str, str]:
    roles = _frame_or_empty(agent_roles)
    assets = _frame_or_empty(asset_registry)
    deep = _frame_or_empty(deep_summary)
    if not roles.empty and "role_type" in roles.columns:
        n_vpps = int((roles["role_type"].astype(str) == "vpp_dispatch_agent").sum())
        if n_vpps <= 0:
            n_vpps = int((roles["role_type"].astype(str) == "vpp_portfolio_agent").sum())
    elif not assets.empty and "vpp_id" in assets.columns:
        n_vpps = int(assets["vpp_id"].nunique())
    else:
        n_vpps = 1
    return {
        "n_vpps": str(max(1, n_vpps)),
        "hidden_dim": _first_value(deep, "hidden_dim", "64"),
        "dso_input_dim": _first_value(deep, "dso_input_dim", f"5+7*{max(1, n_vpps)}"),
        "vpp_input_dim": _first_value(deep, "vpp_input_dim", "16+26*Kmax"),
        "portfolio_input_dim": _first_value(deep, "portfolio_input_dim", "9"),
        "critic_input_dim": _first_value(deep, "critic_input_dim", f"5+9*{max(1, n_vpps)}"),
        "critic_action_dim": _first_value(
            deep,
            "critic_action_dim",
            _first_value(deep, "critic_action_summary_dim", "16+8*N_vpp"),
        ),
        "max_der_per_vpp": _first_value(deep, "max_der_per_vpp", "Kmax"),
        "vpp_encoder_type": _first_value(deep, "vpp_encoder_type", "der_deep_sets_token_pooling"),
        "critic_type": _first_value(deep, "critic_type", "action_conditioned_centralized_value_critic"),
        "architecture_version": _first_value(
            deep,
            "architecture_version",
            "privacy_ctde_deepsets_vpp_action_conditioned_critic_v1",
        ),
    }


def rl_agent_architecture_frame(
    agent_roles: pd.DataFrame | None = None,
    *,
    deep_summary: pd.DataFrame | None = None,
    asset_registry: pd.DataFrame | None = None,
) -> pd.DataFrame:
    roles = _frame_or_empty(agent_roles)
    dims = _architecture_dims(agent_roles=roles, asset_registry=asset_registry, deep_summary=deep_summary)
    if roles.empty:
        roles = pd.DataFrame(
            [
                {
                    "agent_id": "dso_global_guidance",
                    "role_type": "dso_guidance_agent",
                    "owner_id": "dso",
                    "time_scale": "fast_to_middle",
                    "objective": "Guide VPP envelopes and preserve grid safety.",
                    "action_summary": "VPP operating envelope preferences",
                    "observation_summary": "Network state and VPP reports",
                    "privacy_scope": "Centralized DSO training view",
                    "trainable": True,
                }
            ]
        )

    rows: list[dict[str, Any]] = []
    for _, role in roles.iterrows():
        role_type = _text(role.get("role_type"))
        agent_id = _text(role.get("agent_id"))
        group_id = _group_for_role(role_type)
        if role_type == "dso_guidance_agent":
            input_fields = (
                f"o_dso in R^{dims['dso_input_dim']}: time index, network stress, VPP day-ahead bids, "
                "VPP P/Q and FR/DOE bounds, single-PCC vs multi-node flag"
            )
            output_fields = f"a_dso in R^{dims['n_vpps']}: operating envelope / target preference per VPP"
            policy_module = "privacy-separated Gaussian DSO actor + training-only centralized critic"
            implementation_status = "trainable in deep_rl.py"
            reward_function = (
                "r_dso = -0.05 * total_cost + feasibility_bonus + tracking_bonus; "
                "raw_objective_reward = -total_cost is kept for diagnostics."
            )
            current_step_role = (
                "Reads global grid state and VPP reports, then converts network stress and flexibility bids into "
                "safe envelope preferences for each VPP."
            )
            next_upgrade = "Upgrade the DSO actor from scalar P targets to certified FR/DOE geometry, Q support, service logits and price adders."
            zh = {
                "input": f"o_dso 属于 R^{dims['dso_input_dim']}：时间索引、电网压力、VPP 日前报量/报价、VPP P/Q 与 FR/DOE 边界、单 PCC/多节点标记",
                "output": f"a_dso 属于 R^{dims['n_vpps']}：每个 VPP 的运行包络/目标偏好",
                "module": "隐私分离高斯 DSO actor + 仅训练期 centralized critic",
                "status": "已在 deep_rl.py 中可训练",
                "reward": "r_dso = -0.05 * total_cost + 可行性奖励 + 跟踪奖励；total_cost 包含 action_projection_penalty；raw_objective_reward = -total_cost 保留用于诊断。",
                "step": "读取全局电网状态和 VPP 上报，将网络压力与灵活性报价转换为每个 VPP 的安全包络偏好。",
                "upgrade": "将 DSO actor 从标量有功目标升级为带认证的 FR/DOE 几何、无功、服务类型 logits 和价格附加项。",
            }
        elif role_type == "vpp_dispatch_agent":
            input_fields = f"o_vpp_i in R^{dims['vpp_input_dim']}: own DER state, own envelope, local flexibility state and response history"
            output_fields = f"selected aggregate P in R^1 plus DER-level normalized actions in R^{dims['max_der_per_vpp']}"
            policy_module = "privacy-scoped Deep Sets VPP encoder + aggregate head + DER action head + safety projection"
            implementation_status = "DER-level learned action interface is active; projection clips bounds and repairs residuals"
            reward_function = (
                "r_dispatch_i = private_profit_proxy + service/availability payment - DER cost - tracking/projection "
                "penalty - comfort/SOC penalty. It does not directly include raw DSO global reward."
            )
            current_step_role = (
                "Selects selected_p_mw inside the DSO envelope and proposes per-DER normalized actions. The safety "
                "layer projects the proposals into DER limits before pandapower writes."
            )
            next_upgrade = (
                "Upgrade the current Deep Sets pooling path with attention/temporal history encoders and add local settlement-aware rewards."
            )
            zh = {
                "input": f"o_vpp_i 属于 R^{dims['vpp_input_dim']}：本 VPP 的 DER 状态、本 VPP 运行包络、本地灵活性状态和历史响应",
                "output": f"R^1 聚合 P + R^{dims['max_der_per_vpp']} 的 DER 级归一化设定值动作",
                "module": "隐私作用域内的 VPP 本地编码器 + 聚合 head + DER 动作 head + 安全投影",
                "status": "DER 级学习动作接口已经接入；投影层负责边界裁剪和残差修复",
                "reward": "当前 trainer 使用整形后的全局 reward。计划中的本地 reward 将加入服务履约、DER 边际成本、舒适度/SOC、动作平滑和包络裕度项。",
                "step": "在 DSO 包络内选择 selected_p_mw，并对每个 DER 提出归一化动作；安全层把动作投影到设备约束内后再写入 pandapower。",
                "upgrade": "用集合编码器 / Set Transformer 和短历史 GRU 替换 padding DER 向量，并加入结算感知本地 reward。",
            }
        elif role_type == "vpp_portfolio_agent":
            input_fields = f"h_vpp_i in R^{dims['portfolio_input_dim']}: portfolio mode, PCC, bus count, asset count, import/export bounds and eligibility flags"
            output_fields = "keep, reweight, propose_membership_change"
            policy_module = "privacy-scoped portfolio encoder + categorical logits head with physical membership gate"
            implementation_status = "trainable in deep_rl.py as a slow-loop commercial proposal head"
            reward_function = (
                "r_portfolio_i = long-horizon profit proxy + reliability bonus + localized DSO-alignment credit "
                "- switching cost - delivery-risk penalty. The DSO term is localized/settlement-like, not raw global reward."
            )
            current_step_role = (
                "Runs on a slower loop. It proposes keep/reweight/membership-change actions for commercial aggregation; "
                "physical bus locations and pandapower element rows do not move unless a gated portfolio event is applied."
            )
            next_upgrade = "Train on multi-day episodes with real portfolio transition cost, service reliability metrics and contract constraints."
            zh = {
                "input": f"h_vpp_i 属于 R^{dims['portfolio_input_dim']}：组合模式、PCC、接入母线数量、资产数量、进/出口边界和配置资格标记",
                "output": "保持、重新加权、提出成员配置变更",
                "module": "隐私作用域内的组合配置编码器 + categorical logits head，物理成员变更受事件门控",
                "status": "已进入 deep_rl.py 的策略梯度损失，但物理成员变更仍受事件门控",
                "reward": "当前使用组合配置代理 reward 和共享整形回报；目标 reward 应衡量长周期灵活性收益、服务可靠性、组合切换成本和 DER 合约约束。",
                "step": "运行在慢时间尺度；它调整商业聚合成员或权重，不改变 DER 物理母线位置。",
                "upgrade": "在多日 episode 上训练，并纳入组合切换成本和服务可靠性指标。",
            }
        else:
            input_fields = _text(role.get("observation_summary"), "training metrics and trial history")
            output_fields = _text(role.get("action_summary"), "trial selection and convergence handoff")
            policy_module = "experiment orchestration logic"
            implementation_status = "non-trainable supervisor / metadata role"
            reward_function = "No environment reward. It monitors reward trend, violations and convergence status."
            current_step_role = "Runs outside the environment step loop and supervises experiments across episodes/trials."
            next_upgrade = "Add automated failure diagnosis and algorithm selection based on learning curves."
            zh = {
                "input": "训练指标、奖励趋势、约束违规和试验历史",
                "output": "算法/超参数试验选择、早停、收敛判断和失败回交",
                "module": "实验监督逻辑",
                "status": "非训练智能体/元数据角色",
                "reward": "没有环境 reward；它监控奖励趋势、约束违规和收敛状态。",
                "step": "运行在环境 step 循环之外，跨 episode/trial 监督实验。",
                "upgrade": "基于学习曲线加入自动失败诊断和算法选择。",
            }

        group = AGENT_GROUPS[group_id]
        provenance = _agent_result_provenance(role_type, dims)
        rows.append(
            {
                "agent_id": agent_id,
                "agent_group": group_id,
                "agent_group_label": group["label"],
                "agent_group_label_zh": group["label_zh"],
                "agent_group_color": group["color"],
                "role_type": role_type,
                "owner_id": role.get("owner_id", ""),
                "time_scale": role.get("time_scale", ""),
                "trainable": _bool(role.get("trainable", False)),
                "input_observation": input_fields,
                "input_observation_zh": zh["input"],
                "action_output": output_fields,
                "action_output_zh": zh["output"],
                **provenance,
                "policy_module": policy_module,
                "policy_module_zh": zh["module"],
                "implementation_status": implementation_status,
                "implementation_status_zh": zh["status"],
                "reward_function": reward_function,
                "reward_function_zh": _role_reward_zh(role_type, zh["reward"]),
                "current_step_role": current_step_role,
                "current_step_role_zh": zh["step"],
                "next_upgrade": next_upgrade,
                "next_upgrade_zh": zh["upgrade"],
                "objective": role.get("objective", ""),
                "privacy_scope": role.get("privacy_scope", ""),
            }
        )
    return pd.DataFrame(rows)


def rl_agent_group_frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "agent_group": group_id,
                "label": meta["label"],
                "label_zh": meta["label_zh"],
                "color": meta["color"],
                "summary": meta["summary"],
                "summary_zh": meta["summary_zh"],
            }
            for group_id, meta in AGENT_GROUPS.items()
        ]
    )


def rl_encoder_architecture_frame(encoder_roles: pd.DataFrame | None = None) -> pd.DataFrame:
    encoders = _frame_or_empty(encoder_roles)
    if encoders.empty:
        return pd.DataFrame(
            [
                {
                    "encoder_id": "encode_dso_observation",
                    "owner": "trainer",
                    "update_scale": "fast",
                    "output_name": "DSOEncodedState",
                    "purpose": "Numeric vector consumed by the current PyTorch actor-critic policy.",
                    "purpose_zh": "当前 PyTorch Actor-Critic 策略使用的数值状态向量。",
                    "current_status": "implemented",
                    "current_status_zh": "已实现",
                }
            ]
        )
    rows = encoders.copy()
    rows["current_status"] = "interface_defined"
    rows["current_status_zh"] = "接口已定义"
    if "purpose_zh" not in rows:
        rows["purpose_zh"] = rows["purpose"]
    return rows


def rl_data_flow_frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "flow_order": 1,
                "source": "pandapower + simulator",
                "target": "observation builders",
                "signal": "network_state, VPP reports, DER states, profiles",
                "signal_zh": "网络状态、VPP 上报、DER 状态、外部曲线",
                "description": "Power-flow state and local resource state are converted into policy observations.",
                "description_zh": "潮流状态和本地资源状态被转换成策略观测。",
            },
            {
                "flow_order": 2,
                "source": "VPP day-ahead bid/report",
                "target": "DSO envelope issuer",
                "signal": "p_min/p_max, q_min/q_max, bid prices, confidence, capability summary",
                "signal_zh": "P/Q 可行域、报价、置信度和能力摘要",
                "description": "The DSO receives privacy-preserving VPP capability reports before issuing envelopes.",
                "description_zh": "DSO 在发布包络前接收 VPP 的隐私保护能力上报。",
            },
            {
                "flow_order": 3,
                "source": "dso_global_guidance actor",
                "target": "VPP dispatch agents",
                "signal": "operating envelope, preferred region, service request and price context",
                "signal_zh": "运行包络、推荐区间、服务请求和价格上下文",
                "description": "The DSO guides each VPP without exposing the complete network topology to the VPP.",
                "description_zh": "DSO 在不向 VPP 暴露完整网络拓扑的前提下引导各 VPP。",
            },
            {
                "flow_order": 4,
                "source": "vpp_dispatch actor head",
                "target": "DER-level disaggregation proposal",
                "signal": "selected_p_mw plus der_actions",
                "signal_zh": "selected_p_mw 与 der_actions",
                "description": "The VPP actor chooses an aggregate point and proposes per-DER normalized setpoints.",
                "description_zh": "VPP actor 选择聚合功率点，并对每个 DER 提出归一化设定值。",
            },
            {
                "flow_order": 5,
                "source": "safety projection and residual repair",
                "target": "true DER pandapower elements",
                "signal": "PV/ESS/EVCS/HVAC/flexible-load/MT commands",
                "signal_zh": "PV、ESS、EVCS、HVAC、柔性负荷、燃机指令",
                "description": "Learned DER proposals are clipped to device bounds and repaired so aggregate delivery tracks the selected target.",
                "description_zh": "学习型 DER 提案先被裁剪到设备边界，再修复聚合残差，使总交付跟踪所选目标。",
            },
            {
                "flow_order": 6,
                "source": "pandapower runpp",
                "target": "reward and loss",
                "signal": "voltage, loading, tracking, cost, violations",
                "signal_zh": "电压、负载率、跟踪误差、成本、约束违规",
                "description": "The simulator computes shaped training reward and raw objective diagnostics, then returns update the policy.",
                "description_zh": "仿真器计算整形训练 reward 与原始目标诊断项，并用回报更新策略。",
            },
        ]
    )


def rl_agent_relationship_frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "relation_order": 1,
                "source": "VPP dispatch agents",
                "target": "DSO global guidance actor",
                "message": "day-ahead bid/report: feasible range, bid prices, confidence and capability summary",
                "message_zh": "日前报量/报价：可行域、报价、置信度和能力摘要",
                "meaning": "The DSO does not need private DER details to build an operating envelope.",
                "meaning_zh": "DSO 不需要读取 VPP 私有 DER 细节，也能形成运行包络。",
            },
            {
                "relation_order": 2,
                "source": "DSO global guidance actor",
                "target": "VPP dispatch agents",
                "message": "operating envelope + service request + preferred region",
                "message_zh": "运行包络 + 服务请求 + 推荐运行区间",
                "meaning": "The envelope is both a grid-safety boundary and the DSO's directional regulation intention.",
                "meaning_zh": "包络同时表达电网安全边界和 DSO 的调节意图。",
            },
            {
                "relation_order": 3,
                "source": "VPP dispatch agents",
                "target": "DER fleet",
                "message": "selected_p_mw and DER normalized actions",
                "message_zh": "selected_p_mw 与 DER 归一化动作",
                "meaning": "This is the learned VPP disaggregation step: aggregate intent becomes device-level setpoints.",
                "meaning_zh": "这是学习型 VPP 解聚合步骤：聚合意图被转换成设备级设定值。",
            },
            {
                "relation_order": 4,
                "source": "safety projection",
                "target": "pandapower grid",
                "message": "bounded DER dispatch written to load/sgen/storage elements",
                "message_zh": "约束后的 DER 出力写入 load/sgen/storage 元件",
                "meaning": "The projection layer protects the physical simulator from infeasible neural actions.",
                "meaning_zh": "投影层防止不可行动作直接写入物理仿真模型。",
            },
            {
                "relation_order": 5,
                "source": "pandapower grid",
                "target": "training supervisor",
                "message": "reward components, raw objective, violations and convergence metrics",
                "message_zh": "reward 分量、原始目标、约束违规和收敛指标",
                "meaning": "Training uses shaped reward for stability while retaining raw cost diagnostics for research analysis.",
                "meaning_zh": "训练使用整形 reward 提升稳定性，同时保留原始成本诊断用于科研分析。",
            },
        ]
    )


def rl_step_workflow_frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "step_order": 1,
                "stage": "day_ahead_capability_report",
                "stage_zh": "日前能力上报",
                "agent_group": "vpp_dispatch",
                "actor": "VPP dispatch agents",
                "input": "DER bounds, SOC, PV forecast, local demand and cost coefficients",
                "output": "privacy-preserving feasible range and bid summary",
                "explanation": "Each VPP reports an aggregate capability view instead of exposing all DER internals.",
                "explanation_zh": "每个 VPP 上报聚合能力视图，而不是暴露所有 DER 内部参数。",
            },
            {
                "step_order": 2,
                "stage": "dso_envelope_guidance",
                "stage_zh": "DSO 运行包络引导",
                "agent_group": "global_guidance",
                "actor": "DSO global guidance actor",
                "input": "global grid observation, VPP reports, line/voltage stress and price context",
                "output": "operating envelope, preferred target and service request",
                "explanation": "The DSO translates network safety and regulation intention into VPP-level feasible envelopes.",
                "explanation_zh": "DSO 将网络安全约束和调节意图转换成 VPP 级可行包络。",
            },
            {
                "step_order": 3,
                "stage": "vpp_learned_disaggregation",
                "stage_zh": "VPP 学习型解聚合",
                "agent_group": "vpp_dispatch",
                "actor": "VPP dispatch actor",
                "input": "own envelope, own DER state, price/service signal and response history",
                "output": "selected_p_mw and per-DER normalized actions",
                "explanation": "The VPP chooses where to operate inside the envelope and proposes device-level actions.",
                "explanation_zh": "VPP 在包络内选择运行点，并提出设备级动作。",
            },
            {
                "step_order": 4,
                "stage": "projection_and_powerflow",
                "stage_zh": "投影与潮流计算",
                "agent_group": "training_supervisor",
                "actor": "safety projection + simulator",
                "input": "raw neural actions and physical device/network limits",
                "output": "bounded pandapower element values and runpp results",
                "explanation": "Actions are clipped, aggregate residuals are repaired, and pandapower evaluates feasibility.",
                "explanation_zh": "动作先被裁剪，聚合残差被修复，然后由 pandapower 校核物理可行性。",
            },
            {
                "step_order": 5,
                "stage": "reward_and_learning_update",
                "stage_zh": "奖励与学习更新",
                "agent_group": "training_supervisor",
                "actor": "deep training supervisor",
                "input": "reward components, violations, tracking error and trajectory log probabilities",
                "output": "policy/value/entropy losses and optimizer step",
                "explanation": "The trainer optimizes shaped reward for convergence while logging raw objective diagnostics.",
                "explanation_zh": "训练器优化整形 reward 以提高收敛性，同时记录原始目标诊断项。",
            },
            {
                "step_order": 6,
                "stage": "slow_portfolio_review",
                "stage_zh": "慢周期组合配置评估",
                "agent_group": "vpp_portfolio",
                "actor": "VPP portfolio agent",
                "input": "service history, reliability, revenue/cost belief and delivery risk",
                "output": "keep, reweight or propose membership change",
                "explanation": (
                    "The current deep RL trainer already samples a categorical slow-loop portfolio proposal and "
                    "includes its log-probability in the policy-gradient loss. Physical membership changes still "
                    "require a gated scenario event, so the learned output is a commercial recommendation, not a direct topology edit."
                ),
                "explanation_zh": (
                    "当前 deep RL 训练器已经采样慢周期 categorical 组合建议，并把它的 log-probability 纳入策略梯度损失。"
                    "物理成员变更仍需要受控场景事件，因此学习输出是商业配置建议，不是直接修改拓扑。"
                ),
            },
        ]
    )


def rl_reward_design_frame(deep_summary: pd.DataFrame | None = None) -> pd.DataFrame:
    primary_ctde = _privacy_separated_primary(deep_summary)
    global_applies_to = (
        "current privacy-separated CTDE actor-critic trainer"
        if primary_ctde
        else "current shared DSO/VPP actor-critic trainer"
    )
    global_applies_to_zh = (
        "当前隐私分离 CTDE Actor-Critic 训练器"
        if primary_ctde
        else "当前共享 DSO/VPP Actor-Critic 训练器"
    )
    return pd.DataFrame(
        [
            {
                "reward_id": "global_shaped_training_reward",
                "applies_to": global_applies_to,
                "applies_to_zh": global_applies_to_zh,
                "formula": "reward = -0.05 * total_cost + feasibility_bonus + tracking_bonus; total_cost includes action_projection_penalty",
                "formula_zh": "reward = -0.05 * total_cost + 可行性奖励 + 跟踪奖励；total_cost 已包含安全投影惩罚",
                "terms": (
                    "total_cost includes operation cost, action projection penalty, target tracking error, voltage violation, line overload, "
                    "transformer overload, comfort/SOC penalties and power-flow failure penalty"
                ),
                "terms_zh": "total_cost 包含运行成本、安全投影惩罚、目标跟踪误差、电压越限、线路过载、变压器过载、舒适度/SOC 惩罚和潮流失败惩罚",
                "current_status": "implemented",
                "current_status_zh": "已实现",
            },
            {
                "reward_id": "raw_objective_diagnostic",
                "applies_to": "offline evaluation and profit/cost explanation",
                "applies_to_zh": "离线评估与收益/成本解释",
                "formula": "raw_objective_reward = -total_cost",
                "formula_zh": "raw_objective_reward = -total_cost",
                "terms": "unshaped diagnostic value retained so reward shaping does not hide physical/economic cost",
                "terms_zh": "保留未整形诊断值，避免 reward shaping 掩盖真实物理/经济成本",
                "current_status": "implemented",
                "current_status_zh": "已实现",
            },
            {
                "reward_id": "vpp_dispatch_local_reward_target",
                "applies_to": "next-stage decentralized VPP dispatch policies",
                "applies_to_zh": "下一阶段分散执行的 VPP 调度策略",
                "formula": "r_vpp = service_payment - tracking_penalty - DER_cost - comfort_SOC_penalty - smoothness_penalty",
                "formula_zh": "r_vpp = 服务收益 - 跟踪惩罚 - DER 成本 - 舒适度/SOC 惩罚 - 动作平滑惩罚",
                "terms": "award delivery, DER marginal cost, storage degradation, EV/HVAC comfort, local FR/DOE margin",
                "terms_zh": "服务履约、DER 边际成本、储能退化、EV/HVAC 舒适度、本地 FR/DOE 裕度",
                "current_status": "planned",
                "current_status_zh": "计划中",
            },
            {
                "reward_id": "vpp_portfolio_slow_reward_target",
                "applies_to": "current trainable VPP portfolio head plus future independent portfolio agents",
                "applies_to_zh": "当前可训练 VPP 组合配置 head 以及未来独立组合配置智能体",
                "formula": "current proxy: r_portfolio_proxy(label, stress, flex_span); target: long_horizon_profit + reliability_bonus - switching_cost - contract_violation_penalty",
                "formula_zh": "r_portfolio = 长周期收益 + 可靠性奖励 - 切换成本 - 合约违约惩罚",
                "terms": "current proxy rewards keep/reweight/propose choices from grid stress and flexibility span; target terms are multi-day flexibility revenue, reliability, transition cost and DER contracts",
                "terms_zh": "多日灵活性收益、服务可靠性、组合切换成本和 DER 合约约束",
                "current_status": "implemented_proxy_trainable_physical_change_gated",
                "current_status_zh": "已实现代理奖励训练，物理变更受门控",
            },
            {
                "reward_id": "safety_projection_non_agent_guard",
                "applies_to": "execution safety layer, not a trainable MARL actor",
                "applies_to_zh": "安全投影执行层，不是可训练 MARL actor",
                "formula": "raw action -> device bounds -> FR/DOE clip -> DER residual repair -> pandapower write -> runpp check",
                "formula_zh": "原始动作 -> 设备边界 -> FR/DOE 裁剪 -> DER 残差修复 -> pandapower 写入 -> runpp 校核",
                "terms": "deterministic guardrail; violations affect reward but the projection itself has no policy gradient",
                "terms_zh": "确定性安全门；违规影响 reward，但投影层本身不是策略梯度学习器",
                "current_status": "implemented_guardrail",
                "current_status_zh": "已实现的安全门控",
            },
        ]
    )


def rl_reward_design_frame(deep_summary: pd.DataFrame | None = None) -> pd.DataFrame:
    _ = deep_summary
    return pd.DataFrame(
        [
            {
                "reward_id": "dso_global_guidance_reward",
                "applies_to": "DSO global guidance actor",
                "applies_to_zh": "DSO 全局引导智能体",
                "formula": "r_dso = -0.05*dso_total_cost + feasibility_bonus + tracking_bonus",
                "formula_zh": "r_dso = -0.05*dso_total_cost + 可行性奖励 + 跟踪奖励",
                "terms": (
                    "dso_total_cost includes procurement/proxy operation cost, network security penalties, "
                    "tracking error and action projection penalty."
                ),
                "terms_zh": "dso_total_cost 包含采购/代理运行成本、电网安全惩罚、跟踪误差和动作投影惩罚。",
                "current_status": "implemented_role_specific",
                "current_status_zh": "已实现角色专属 reward",
            },
            {
                "reward_id": "vpp_dispatch_local_reward_target",
                "applies_to": "each fast VPP dispatch / DER-disaggregation actor",
                "applies_to_zh": "每个快速 VPP 调度/DER 解聚合智能体",
                "formula": (
                    "r_dispatch_i = 0.02*private_profit_proxy + preferred_region_bonus "
                    "- tracking_penalty - projection_penalty - comfort_SOC_penalty"
                ),
                "formula_zh": (
                    "r_dispatch_i = 0.02*私有利润代理 + 推荐运行区间奖励 "
                    "- 跟踪惩罚 - 投影惩罚 - 舒适度/SOC 惩罚"
                ),
                "terms": (
                    "private_profit_proxy = energy revenue + flexibility service payment + availability payment "
                    "- DER operation cost. It does not contain raw DSO global reward."
                ),
                "terms_zh": "私有利润代理 = 能量收益 + 灵活性服务收益 + 可用性收益 - DER 运行成本；不包含原始 DSO 全局 reward。",
                "current_status": "implemented_role_specific",
                "current_status_zh": "已实现角色专属 reward",
            },
            {
                "reward_id": "vpp_portfolio_slow_reward_target",
                "applies_to": "each slow VPP portfolio / aggregation-configuration actor",
                "applies_to_zh": "每个慢周期 VPP 组合配置/聚合配置智能体",
                "formula": (
                    "r_portfolio_i = long_horizon_profit_proxy + reliability_bonus "
                    "+ localized_DSO_alignment_credit - switching_cost - delivery_risk_penalty"
                ),
                "formula_zh": (
                    "r_portfolio_i = 长周期利润代理 + 可靠性奖励 + 局部化 DSO 对齐收益 "
                    "- 配置切换成本 - 履约风险惩罚"
                ),
                "terms": (
                    "localized_DSO_alignment_credit is a settlement-like proxy for preferred-region service, "
                    "availability and feasibility. It is not the raw DSO reward and should be interpreted as an incentive signal."
                ),
                "terms_zh": (
                    "局部化 DSO 对齐收益是推荐区间服务、可用性和可行性的合约化代理项；"
                    "它不是原始 DSO reward，应被解释为激励信号。"
                ),
                "current_status": "implemented_role_specific_physical_change_gated",
                "current_status_zh": "已实现角色专属 reward，物理配置变更仍受门控",
            },
            {
                "reward_id": "raw_objective_diagnostic",
                "applies_to": "offline evaluation and cost/profit explanation",
                "applies_to_zh": "离线评估与成本/收益解释",
                "formula": "raw_objective_reward = -total_cost",
                "formula_zh": "raw_objective_reward = -total_cost",
                "terms": "diagnostic only; it is kept so reward shaping does not hide physical/economic cost.",
                "terms_zh": "仅用于诊断，避免 reward shaping 掩盖真实物理/经济成本。",
                "current_status": "implemented_diagnostic",
                "current_status_zh": "已实现诊断项",
            },
            {
                "reward_id": "safety_projection_non_agent_guard",
                "applies_to": "execution safety layer, not a trainable MARL actor",
                "applies_to_zh": "安全投影执行层，不是可训练 MARL actor",
                "formula": "raw action -> device bounds -> FR/DOE clip -> DER residual repair -> pandapower write -> runpp check",
                "formula_zh": "原始动作 -> 设备边界 -> FR/DOE 裁剪 -> DER 残差修复 -> pandapower 写入 -> runpp 校核",
                "terms": (
                    "deterministic guardrail; projection gaps are penalized in role rewards but the projection "
                    "layer has no policy gradient."
                ),
                "terms_zh": "确定性安全门控；投影差额会进入角色 reward 惩罚，但投影层本身没有策略梯度。",
                "current_status": "implemented_guardrail",
                "current_status_zh": "已实现安全门控",
            },
        ]
    )


def rl_implementation_gap_frame(deep_summary: pd.DataFrame | None = None) -> pd.DataFrame:
    if _privacy_separated_primary(deep_summary):
        return pd.DataFrame(
            [
                {
                    "gap_id": "ctde_primary_trainer_implemented",
                    "question": "Is the current primary model still the shared-backbone prototype?",
                    "question_zh": "当前主模型是否仍是共享骨干原型？",
                    "current_answer": (
                        "No. The primary trainer is privacy_separated_ctde_actor_critic. "
                        "The shared-backbone network is only retained for explicit --algorithm shared ablations and regression tests."
                    ),
                    "current_answer_zh": (
                        "不是。当前主训练器是 privacy_separated_ctde_actor_critic。"
                        "共享骨干网络只在显式运行 --algorithm shared 的消融实验和回归测试中使用。"
                    ),
                    "target_answer": "Keep default reports focused on the implemented privacy-separated CTDE model.",
                    "target_answer_zh": "默认报告只聚焦当前已实现的隐私分离 CTDE 模型。",
                },
                {
                    "gap_id": "dso_envelope_not_yet_opf_certified",
                    "question": "Is the DSO envelope a rigorous OPF-safe operating envelope?",
                    "question_zh": "DSO 包络是否已经是严格 OPF 安全包络？",
                    "current_answer": (
                        "No. The current envelope is still a deterministic, capability-aware scaffold that combines VPP bids, "
                        "local flexibility bounds and simple grid-stress heuristics."
                    ),
                    "current_answer_zh": "还不是。当前包络仍是能力感知的确定性脚手架，结合 VPP 报量/报价、本地灵活性边界和简单电网压力启发式。",
                    "target_answer": "Use OPF, voltage/line sensitivity or chance-constrained safety projection to certify envelope limits.",
                    "target_answer_zh": "后续用 OPF、电压/线路灵敏度或机会约束安全投影来认证包络边界。",
                },
                {
                    "gap_id": "local_reward_needs_settlement_detail",
                    "question": "Is each VPP reward already a full settlement-aware local objective?",
                    "question_zh": "每个 VPP 的 reward 是否已经是完整结算感知本地目标？",
                    "current_answer": (
                        "Partly. The CTDE trainer separates execution-side actors and records VPP dispatch/portfolio losses, "
                        "but the local settlement reward remains simplified and should be expanded for publication-grade experiments."
                    ),
                    "current_answer_zh": (
                        "部分实现。CTDE 训练器已经拆分执行期 actor，并记录 VPP 调度/组合配置损失，"
                        "但本地结算 reward 仍是简化版本，顶会级实验还需要增强。"
                    ),
                    "target_answer": "Add service revenue, DER marginal cost, degradation, comfort/SOC and contract penalties per VPP.",
                    "target_answer_zh": "为每个 VPP 加入服务收益、DER 边际成本、退化成本、舒适度/SOC 和合约惩罚。",
                },
                {
                    "gap_id": "portfolio_agent_physical_change_gated",
                    "question": "Can the trained portfolio agent directly move DER to another VPP?",
                    "question_zh": "训练后的组合配置智能体能直接把 DER 移到另一个 VPP 吗？",
                    "current_answer": (
                        "No. It samples keep/reweight/propose commercial configuration actions. Physical buses and pandapower rows "
                        "remain unchanged unless a gated scenario portfolio event applies."
                    ),
                    "current_answer_zh": (
                        "不能。它采样 keep/reweight/propose 商业配置动作。物理母线和 pandapower 元件行不会移动，"
                        "除非通过受控 scenario portfolio event。"
                    ),
                    "target_answer": "Train multi-day portfolio decisions, then validate accepted membership changes through contract and network safety checks.",
                    "target_answer_zh": "后续训练多日组合决策，并让被接受的成员变更通过合约和网络安全校核。",
                },
            ]
        )
    return pd.DataFrame(
        [
            {
                "gap_id": "der_action_active_but_not_full_decentralized_policy",
                "question": "Is VPP disaggregation now an RL agent?",
                "question_zh": "VPP 解聚合现在是否已经是 RL agent？",
                "current_answer": (
                    "Yes at the action-interface level: the VPP dispatch head now emits selected_p_mw plus DER-level "
                    "normalized actions. It is still produced by a shared centralized policy, so it is not yet a fully "
                    "independent decentralized VPP actor."
                ),
                "current_answer_zh": (
                    "在动作接口层面是的：VPP dispatch head 已输出 selected_p_mw 与 DER 级归一化动作。"
                    "但它仍由共享集中式策略产生，因此还不是完全独立的分散 VPP actor。"
                ),
                "target_answer": "Train independent VPP lower policies with local observations, local rewards and a centralized critic.",
                "target_answer_zh": "训练独立 VPP 下层策略，使其使用本地观测、本地 reward，并在训练期共享集中式 critic。",
            },
            {
                "gap_id": "dso_envelope_not_yet_opf_certified",
                "question": "Is the DSO envelope a rigorous OPF-safe operating envelope?",
                "question_zh": "DSO 包络是否已经是严格 OPF 安全包络？",
                "current_answer": (
                    "No. The current envelope is a deterministic, capability-aware scaffold that combines VPP bids, "
                    "local flexibility bounds and simple grid-stress heuristics."
                ),
                "current_answer_zh": "还不是。当前包络是能力感知的确定性脚手架，结合 VPP 报量/报价、本地灵活性边界和简单电网压力启发式。",
                "target_answer": "Use OPF, voltage/line sensitivity or chance-constrained safety projection to certify envelope limits.",
                "target_answer_zh": "后续用 OPF、电压/线路灵敏度或机会约束安全投影来认证包络边界。",
            },
            {
                "gap_id": "ctde_incomplete",
                "question": "Why is the current model not full CTDE?",
                "question_zh": "为什么当前模型还不是完整 CTDE？",
                "current_answer": "Training is centralized, but execution is not fully decentralized because one shared network still emits multiple heads.",
                "current_answer_zh": "训练已经集中化，但执行尚未完全分散，因为当前仍由一个共享网络输出多个 head。",
                "target_answer": "Use separate DSO and VPP actors for execution, with a centralized critic available only during training.",
                "target_answer_zh": "执行时使用独立 DSO/VPP actor；集中式 critic 只在训练时使用。",
            },
            {
                "gap_id": "training_supervisor_not_environment_agent",
                "question": "Is the training supervisor a MARL environment agent or an LLM agent?",
                "question_zh": "训练监督智能体是 MARL 环境内智能体，还是 LLM agent？",
                "current_answer": (
                    "Neither. It is an experiment-level orchestrator that monitors trials, hyperparameters, "
                    "reward trends and convergence. It does not receive an environment reward or act inside "
                    "MultiAgentVPPDSOEnv.step()."
                ),
                "current_answer_zh": (
                    "都不是。它是实验级监督器，用来监控试验、超参数、奖励趋势和收敛情况；"
                    "它不接收环境 reward，也不在 MultiAgentVPPDSOEnv.step() 内执行动作。"
                ),
                "target_answer": "Keep it outside the MARL agent set; later it can automate sweeps and return failed runs to algorithm debugging.",
                "target_answer_zh": "保持在 MARL 智能体集合之外；后续可自动化调参，并把失败实验返回给算法诊断。",
            },
            {
                "gap_id": "portfolio_agent_physical_change_gated",
                "question": "Can the trained portfolio agent directly move DER to another VPP?",
                "question_zh": "训练后的组合配置智能体能直接把 DER 移到另一个 VPP 吗？",
                "current_answer": (
                    "No. The trained head samples keep/reweight/propose actions for commercial configuration. "
                    "Physical buses and pandapower element rows remain unchanged unless a gated scenario portfolio event applies."
                ),
                "current_answer_zh": (
                    "不能。当前训练头只采样 keep/reweight/propose 商业配置动作；"
                    "物理母线和 pandapower 元件行不会移动，除非通过受控的 scenario portfolio event。"
                ),
                "target_answer": "Train multi-day portfolio decisions, then validate accepted membership changes through contract and network safety checks.",
                "target_answer_zh": "后续训练多日组合决策，并让被接受的成员变更通过合约和网络安全校核。",
            },
        ]
    )


def rl_loss_components_frame(deep_summary: pd.DataFrame | None = None) -> pd.DataFrame:
    if _privacy_separated_primary(deep_summary):
        return pd.DataFrame(
            [
                {
                    "component": "dso_policy_loss",
                    "formula": "L_dso = -mean(min(r_t*A_dso,t, clip(r_t,1-eps,1+eps)*A_dso,t)); A from GAE(lambda)",
                    "formula_zh": "L_dso = clipped surrogate with GAE(lambda) advantage",
                    "coefficient": "c_dso=1.0",
                    "meaning": "Updates the DSO envelope actor using centralized critic advantage.",
                    "meaning_zh": "使用集中 critic advantage 更新 DSO 包络 actor。",
                },
                {
                    "component": "vpp_dispatch_policy_loss",
                    "formula": "L_vpp_dispatch = -mean(min(r_t*A_dispatch,t, clip(r_t,1-eps,1+eps)*A_dispatch,t))",
                    "formula_zh": "L_vpp_dispatch = clipped surrogate with dispatch GAE advantage",
                    "coefficient": "c_vpp=1.0",
                    "meaning": "Updates decentralized VPP dispatch / DER-disaggregation actors.",
                    "meaning_zh": "更新分散执行的 VPP 调度 / DER 解聚合 actor。",
                },
                {
                    "component": "portfolio_policy_loss",
                    "formula": "L_portfolio = -mean(min(r_t*A_portfolio,t, clip(r_t,1-eps,1+eps)*A_portfolio,t))",
                    "formula_zh": "L_portfolio = clipped surrogate with portfolio GAE advantage",
                    "coefficient": "c_portfolio=0.25",
                    "meaning": "Updates slow-cycle VPP portfolio proposal actors.",
                    "meaning_zh": "更新慢周期 VPP 组合配置建议 actor。",
                },
                {
                    "component": "critic_value_loss",
                    "formula": "L_critic = mean(MSE(V_dso,G_dso^GAE)+MSE(V_dispatch,G_dispatch^GAE)+MSE(V_portfolio,G_portfolio^GAE))",
                    "formula_zh": "L_critic = three role value heads fitted to role GAE returns",
                    "coefficient": "value_coef=0.50",
                    "meaning": "Trains the centralized critic. critic_global_state is not visible to execution actors.",
                    "meaning_zh": "训练集中式 critic；critic_global_state 不暴露给执行期 actor。",
                },
                {
                    "component": "entropy_loss",
                    "formula": "L_entropy = -mean(entropy_dso + entropy_vpp + entropy_portfolio)",
                    "formula_zh": "L_entropy = -mean(entropy_dso + entropy_vpp + entropy_portfolio)",
                    "coefficient": "entropy_coef=0.01",
                    "meaning": "Maintains exploration in DSO, VPP dispatch and portfolio policies.",
                    "meaning_zh": "保持 DSO、VPP 调度和组合配置策略的探索性。",
                },
                {
                    "component": "total_loss",
                    "formula": "L = c_dso*L_dso + c_vpp*L_vpp_dispatch + c_portfolio*L_portfolio + value_coef*L_critic + entropy_coef*L_entropy",
                    "formula_zh": "L = c_dso*L_dso + c_vpp*L_vpp_dispatch + c_portfolio*L_portfolio + value_coef*L_critic + entropy_coef*L_entropy",
                    "coefficient": "Adam lr=3e-4, grad_clip=1.0",
                    "meaning": "The current privacy-separated CTDE trainer backpropagates this combined objective.",
                    "meaning_zh": "当前隐私分离 CTDE 训练器反向传播该组合目标。",
                },
            ]
        )
    return pd.DataFrame(
        [
            {
                "component": "discounted_return",
                "formula": "G_t = r_t + gamma * G_{t+1}, gamma=0.97 by default",
                "formula_zh": "G_t = r_t + gamma * G_{t+1}，默认 gamma=0.97",
                "coefficient": "1.0",
                "meaning": "Monte-Carlo return computed from simulator rewards.",
                "meaning_zh": "由仿真器 reward 计算得到的蒙特卡洛回报。",
            },
            {
                "component": "advantage",
                "formula": "A_t = normalized(G_t) - V(s_t).detach()",
                "formula_zh": "A_t = normalized(G_t) - V(s_t).detach()",
                "coefficient": "1.0",
                "meaning": "Policy-gradient advantage; value is detached for actor update.",
                "meaning_zh": "策略梯度优势项；actor 更新时 value 被 detach。",
            },
            {
                "component": "policy_loss",
                "formula": "L_policy = -mean(log_prob(action) * A_t)",
                "formula_zh": "L_policy = -mean(log_prob(action) * A_t)",
                "coefficient": "1.0",
                "meaning": "Increases probability of sampled actions that lead to higher-than-expected return.",
                "meaning_zh": "提高带来高于预期回报动作的概率。",
            },
            {
                "component": "value_loss",
                "formula": "L_value = mean((V(s_t) - normalized(G_t))^2)",
                "formula_zh": "L_value = mean((V(s_t) - normalized(G_t))^2)",
                "coefficient": "value_coef=0.50",
                "meaning": "Trains the centralized critic/value head.",
                "meaning_zh": "训练集中式 critic / value head。",
            },
            {
                "component": "entropy_loss",
                "formula": "L_entropy = -mean(entropy)",
                "formula_zh": "L_entropy = -mean(entropy)",
                "coefficient": "entropy_coef=0.01",
                "meaning": "Encourages exploration through Gaussian policy entropy.",
                "meaning_zh": "通过高斯策略熵鼓励探索。",
            },
            {
                "component": "total_loss",
                "formula": "L = L_policy + 0.50 * L_value + 0.01 * L_entropy",
                "formula_zh": "L = 策略损失 + 0.50 * 价值损失 + 0.01 * 熵损失",
                "coefficient": "Adam lr=3e-4, grad_clip=1.0",
                "meaning": "The loss backpropagated through the shared actor-critic network.",
                "meaning_zh": "反向传播到共享 Actor-Critic 网络的总损失。",
            },
        ]
    )


def rl_ctde_assessment_frame(deep_summary: pd.DataFrame | None = None) -> pd.DataFrame:
    if _privacy_separated_primary(deep_summary):
        return pd.DataFrame(
            [
                {
                    "question": "Is centralized training present?",
                    "question_zh": "是否已经有集中式训练？",
                    "answer": "yes",
                    "answer_zh": "是",
                    "evidence": "train_privacy_separated_ctde uses a centralized critic over critic_global_state and shared trajectory returns.",
                    "evidence_zh": "train_privacy_separated_ctde 使用读取 critic_global_state 的集中 critic，并使用轨迹回报训练。",
                },
                {
                    "question": "Is decentralized execution represented?",
                    "question_zh": "是否已经表达分散执行？",
                    "answer": "yes",
                    "answer_zh": "是",
                    "evidence": "DSO actor, VPP dispatch actor and VPP portfolio actor are separate execution-side modules; VPP observations remain local.",
                    "evidence_zh": "DSO actor、VPP 调度 actor 和 VPP 组合配置 actor 是分离的执行期模块；VPP 观测保持本地化。",
                },
                {
                    "question": "Is this CTDE?",
                    "question_zh": "这是不是 CTDE？",
                    "answer": "implemented privacy-separated CTDE training loop",
                    "answer_zh": "已实现隐私分离 CTDE 训练闭环",
                    "evidence": "deep_rl_training_summary.csv reports privacy_separated_execution=True, dso_vpp_shared_encoder=False and target_ctde_primary_trainer=True.",
                    "evidence_zh": "deep_rl_training_summary.csv 记录 privacy_separated_execution=True、dso_vpp_shared_encoder=False、target_ctde_primary_trainer=True。",
                },
                {
                    "question": "Is the shared-backbone benchmark part of the current default experiment?",
                    "question_zh": "共享骨干 benchmark 是否属于当前默认实验？",
                    "answer": "no",
                    "answer_zh": "不是",
                    "evidence": "It is only used when examples/10_train_deep_rl.py is run with --algorithm shared.",
                    "evidence_zh": "只有显式运行 examples/10_train_deep_rl.py --algorithm shared 时才使用它。",
                },
                {
                    "question": "What remains as research work?",
                    "question_zh": "还剩哪些研究工作？",
                    "answer": "certified OPF/sensitivity envelopes, stronger local settlement rewards, topology/set encoders and longer-horizon portfolio training",
                    "answer_zh": "OPF/灵敏度认证包络、更强的本地结算 reward、拓扑/集合编码器和更长周期组合配置训练",
                    "evidence": "These are listed as implementation gaps rather than blockers for the current CTDE architecture.",
                    "evidence_zh": "这些属于后续研究增强项，不是当前 CTDE 架构是否实现的阻塞项。",
                },
            ]
        )
    return pd.DataFrame(
        [
            {
                "question": "Is centralized training present?",
                "question_zh": "是否已经有集中式训练？",
                "answer": "yes",
                "answer_zh": "是",
                "evidence": "deep_rl.py uses global DSO encoded observations, a centralized value head and global shaped reward.",
                "evidence_zh": "deep_rl.py 使用 DSO 全局编码观测、集中式 value head 和全局整形 reward。",
            },
            {
                "question": "Is decentralized execution complete?",
                "question_zh": "是否已经完成分散执行？",
                "answer": "not yet",
                "answer_zh": "还没有",
                "evidence": (
                    "VPP DER-level actions now exist, but they are emitted by a shared centralized network rather than "
                    "independent VPP actors."
                ),
                "evidence_zh": "VPP DER 级动作已经存在，但它们仍由共享集中式网络输出，而不是独立 VPP actor 输出。",
            },
            {
                "question": "Is this CTDE?",
                "question_zh": "这是不是 CTDE？",
                "answer": "proto-CTDE scaffold, not full CTDE",
                "answer_zh": "是 CTDE 脚手架，不是完整 CTDE",
                "evidence": (
                    "MultiAgentVPPDSOEnv exposes actor observations and critic_global_state; the current trainer has "
                    "centralized training and DER-level action heads, but not independent decentralized policies."
                ),
                "evidence_zh": "MultiAgentVPPDSOEnv 暴露 actor observation 与 critic_global_state；当前 trainer 有集中训练和 DER 级动作 head，但还没有独立分散策略。",
            },
            {
                "question": "Should DSO and VPP share one encoder/backbone under privacy constraints?",
                "question_zh": "在隐私约束下 DSO 和 VPP 是否应该共享同一个 encoder/backbone？",
                "answer": "no for the target CTDE method; yes only as a compact full-information benchmark",
                "answer_zh": "目标 CTDE 方法不应该共享；只有作为紧凑 full-information benchmark 时才可以。",
                "evidence": (
                    "The new rl_target_ctde_architecture frame separates dso_private_observation_encoder, "
                    "vpp_local_observation_encoder, vpp_portfolio_slow_encoder and centralized_training_critic. "
                    "The old shared_mlp_backbone is explicitly kept as the current prototype benchmark."
                ),
                "evidence_zh": (
                    "新的 rl_target_ctde_architecture 表把 dso_private_observation_encoder、"
                    "vpp_local_observation_encoder、vpp_portfolio_slow_encoder 与 centralized_training_critic 分离；"
                    "旧 shared_mlp_backbone 只保留为当前原型 benchmark。"
                ),
            },
            {
                "question": "What changes make it full CTDE?",
                "question_zh": "怎样升级为完整 CTDE？",
                "answer": "separate DSO/VPP actors, centralized critic during training, local observations at execution, learned DER-level lower policy",
                "answer_zh": "拆分 DSO/VPP actor，训练时使用集中式 critic，执行时只用本地观测，并保留学习型 DER 下层策略",
                "evidence": "The current action/observation interfaces are already separated enough to support this next step.",
                "evidence_zh": "当前动作/观测接口已经足够分离，可支撑下一步升级。",
            },
            {
                "question": "Are safety projection and the training supervisor trainable MARL actors?",
                "question_zh": "安全投影和训练监督器是可训练 MARL actor 吗？",
                "answer": "no",
                "answer_zh": "不是",
                "evidence": (
                    "Safety projection is a deterministic guardrail between actions and pandapower writes. "
                    "The training supervisor is an experiment orchestrator outside env.step(), not an LLM policy and not an environment-reward agent."
                ),
                "evidence_zh": (
                    "安全投影是动作与 pandapower 写入之间的确定性门控；"
                    "训练监督器是 env.step() 之外的实验编排器，不是 LLM policy，也不是环境 reward 智能体。"
                ),
            },
        ]
    )


def build_rl_architecture_frames(
    *,
    agent_roles: pd.DataFrame | None = None,
    encoder_roles: pd.DataFrame | None = None,
    deep_summary: pd.DataFrame | None = None,
    asset_registry: pd.DataFrame | None = None,
) -> dict[str, pd.DataFrame]:
    return {
        "rl_algorithm_overview": rl_algorithm_overview_frame(deep_summary),
        "rl_agent_groups": rl_agent_group_frame(),
        "rl_agent_architecture": rl_agent_architecture_frame(
            agent_roles,
            deep_summary=deep_summary,
            asset_registry=asset_registry,
        ),
        "rl_neural_network_architecture": rl_neural_network_architecture_frame(
            agent_roles=agent_roles,
            asset_registry=asset_registry,
            deep_summary=deep_summary,
        ),
        "rl_target_ctde_architecture": rl_target_ctde_architecture_frame(
            agent_roles=agent_roles,
            asset_registry=asset_registry,
            deep_summary=deep_summary,
        ),
        "rl_ctde_nodes": rl_ctde_node_frame(
            agent_roles=agent_roles,
            asset_registry=asset_registry,
            deep_summary=deep_summary,
        ),
        "rl_ctde_edges": rl_ctde_edge_frame(
            agent_roles=agent_roles,
            asset_registry=asset_registry,
            deep_summary=deep_summary,
        ),
        "rl_ctde_feedback": rl_ctde_feedback_frame(deep_summary),
        "rl_encoder_architecture": rl_encoder_architecture_frame(encoder_roles),
        "rl_data_flow": rl_data_flow_frame(),
        "rl_agent_relationships": rl_agent_relationship_frame(),
        "rl_step_workflow": rl_step_workflow_frame(),
        "rl_reward_design": rl_reward_design_frame(deep_summary),
        "rl_loss_components": rl_loss_components_frame(deep_summary),
        "rl_ctde_assessment": rl_ctde_assessment_frame(deep_summary),
        "rl_implementation_gaps": rl_implementation_gap_frame(deep_summary),
        "rl_algorithm_capabilities": pd.DataFrame(advanced_algorithm_capability_rows()),
    }
