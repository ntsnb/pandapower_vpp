from __future__ import annotations

from datetime import datetime, timedelta
import math
from pathlib import Path
from typing import Any

from marl_dashboard.backend.storage.variable_enrichment import enrich_formula_dictionary, enrich_variable_dictionary
from marl_dashboard.logging import ExperimentLogger


def default_variable_dictionary() -> list[dict[str, Any]]:
    return enrich_variable_dictionary([
        {"name": "electricity_price", "display_name": "电价 / Electricity price", "symbol": "c_t", "unit": "$/MWh", "group": "dataset", "physical_meaning": "选中时刻的市场电价 / Market price at the selected time.", "formula_latex": "c_t", "source": "demo"},
        {"name": "ev_charging_load", "display_name": "充电桩负荷 / EV charging load", "symbol": "P^{EV}_{i,t}", "unit": "MW", "group": "dataset", "physical_meaning": "第 i 个 VPP 的充电负荷 / EV charging demand for VPP i.", "formula_latex": "P^{EV}_{i,t}", "source": "demo"},
        {"name": "storage_power", "display_name": "储能功率 / Storage power", "symbol": "P^{B}_{i,t}", "unit": "MW", "group": "dataset", "physical_meaning": "正值表示向电网注入，负值表示充电吸收 / Positive means grid injection; negative means charging.", "formula_latex": "P^{B}_{i,t}", "source": "demo"},
        {"name": "storage_soc", "display_name": "储能 SOC / Storage SOC", "symbol": "SOC_{i,t}", "unit": "%", "group": "dataset", "physical_meaning": "电池荷电状态 / Battery state of charge.", "formula_latex": "SOC_{i,t}", "source": "demo"},
        {"name": "pv_power", "display_name": "光伏出力 / PV power", "symbol": "P^{PV}_{i,t}", "unit": "MW", "group": "dataset", "physical_meaning": "光伏发电功率 / Photovoltaic generation.", "formula_latex": "P^{PV}_{i,t}", "source": "demo"},
        {"name": "wind_power", "display_name": "风电出力 / Wind power", "symbol": "P^{W}_{i,t}", "unit": "MW", "group": "dataset", "physical_meaning": "风电发电功率 / Wind generation.", "formula_latex": "P^{W}_{i,t}", "source": "demo"},
        {"name": "base_load", "display_name": "基础负荷 / Base load", "symbol": "P^{L}_{i,t}", "unit": "MW", "group": "dataset", "physical_meaning": "不可调基础需求 / Non-flexible base demand.", "formula_latex": "P^{L}_{i,t}", "source": "demo"},
        {"name": "net_load", "display_name": "净负荷 / Net load", "symbol": "P^{net}_{i,t}", "unit": "MW", "group": "dataset", "physical_meaning": "基础负荷加 EV 负荷，扣除可再生出力和储能注入 / Base load plus EV load minus renewable generation and storage injection.", "formula_latex": "P^{net}_{i,t}=P^L+P^{EV}-P^{PV}-P^W-P^B", "source": "demo"},
        {"name": "profit_reward", "display_name": "收益奖励 / Profit reward", "symbol": "r^{profit}_{i,t}", "unit": "scalar", "group": "reward", "physical_meaning": "售电收益和购电成本形成的收益项 / Reward term from selling revenue and purchase cost.", "formula_latex": "r^{profit}_{i,t}=-C^{energy}_{i,t}+R^{sell}_{i,t}", "source": "demo"},
        {"name": "grid_balance_reward", "display_name": "电网平衡奖励 / Grid balance reward", "symbol": "r^{grid}_{i,t}", "unit": "scalar", "group": "reward", "physical_meaning": "净负荷偏离目标时的平衡奖励或惩罚 / Balance reward or penalty for net-load deviation.", "formula_latex": "r^{grid}_{i,t}=-|P^{net}_{i,t}-P^{target}|", "source": "demo"},
        {"name": "storage_degradation_penalty", "display_name": "储能退化惩罚 / Storage degradation penalty", "symbol": "p^{deg}_{i,t}", "unit": "scalar", "group": "reward", "physical_meaning": "储能充放电幅度导致的奖励扣减 / Reward deduction from battery cycling.", "formula_latex": "p^{deg}_{i,t}=-C^{deg}_{i,t}", "source": "demo"},
        {"name": "constraint_violation_penalty", "display_name": "约束违规惩罚 / Constraint violation penalty", "symbol": "p^{viol}_{i,t}", "unit": "scalar", "group": "reward", "physical_meaning": "净负荷或安全约束违规导致的奖励扣减 / Reward deduction from constraint violation.", "formula_latex": "p^{viol}_{i,t}=-C^{viol}_{i,t}", "source": "demo"},
        {"name": "total_reward", "display_name": "总奖励 / Total reward", "symbol": "r_t", "unit": "scalar", "group": "reward", "physical_meaning": "训练奖励，越大越好 / Training reward; larger is better.", "formula_latex": "r_t=r^{profit}+r^{grid}-p^{deg}-p^{viol}", "source": "demo"},
        {"name": "energy_purchase_cost", "display_name": "购电成本 / Energy purchase cost", "symbol": "C^{energy}_{i,t}", "unit": "$", "group": "cost", "physical_meaning": "正净负荷按电价购电的成本 / Purchase cost for positive net load.", "formula_latex": "C^{energy}_{i,t}=\\max(P^{net}_{i,t},0)c_t", "source": "demo"},
        {"name": "storage_degradation_cost", "display_name": "储能退化成本 / Storage degradation cost", "symbol": "C^{deg}_{i,t}", "unit": "$", "group": "cost", "physical_meaning": "储能循环损耗代理成本 / Proxy cost of battery cycling.", "formula_latex": "C^{deg}_{i,t}=\\lambda_B |P^B_{i,t}|", "source": "demo"},
        {"name": "constraint_violation_cost", "display_name": "约束违规成本 / Constraint violation cost", "symbol": "C^{viol}_{i,t}", "unit": "$", "group": "cost", "physical_meaning": "超过净负荷阈值后的软约束成本 / Soft penalty cost after net-load threshold violation.", "formula_latex": "C^{viol}_{i,t}=\\lambda_v \\max(P^{net}_{i,t}-\\bar P,0)", "source": "demo"},
        {"name": "total_cost", "display_name": "总成本 / Total cost", "symbol": "J_t", "unit": "$", "group": "cost", "physical_meaning": "运行成本，越小越好 / Operational cost; smaller is better.", "formula_latex": "J_t=C^{energy}+C^{deg}+C^{viol}", "source": "demo"},
        {"name": "actor_loss", "display_name": "Actor 损失 / Actor loss", "symbol": "\\mathcal{L}_{actor}", "unit": "scalar", "group": "loss", "physical_meaning": "策略网络优化损失 / Policy optimization loss.", "formula_latex": "\\mathcal{L}_{actor}", "source": "demo"},
        {"name": "critic_loss", "display_name": "Critic 损失 / Critic loss", "symbol": "\\mathcal{L}_{critic}", "unit": "scalar", "group": "loss", "physical_meaning": "价值网络拟合损失 / Value-function fitting loss.", "formula_latex": "\\mathcal{L}_{critic}", "source": "demo"},
        {"name": "entropy_loss", "display_name": "熵正则项 / Entropy loss", "symbol": "\\mathcal{L}_{entropy}", "unit": "scalar", "group": "loss", "physical_meaning": "鼓励策略探索的熵项 / Entropy term used to encourage exploration.", "formula_latex": "-\\alpha H(\\pi)", "source": "demo"},
        {"name": "total_loss", "display_name": "总损失 / Total loss", "symbol": "\\mathcal{L}", "unit": "scalar", "group": "loss", "physical_meaning": "优化器使用的总损失 / Total optimizer loss.", "formula_latex": "\\mathcal{L}=\\mathcal{L}_{actor}+\\mathcal{L}_{critic}-\\alpha H", "source": "demo"},
    ])


def default_formulas() -> dict[str, str]:
    return enrich_formula_dictionary({
        "net_load": "P^{net}_{i,t}=P^L_{i,t}+P^{EV}_{i,t}-P^{PV}_{i,t}-P^W_{i,t}-P^B_{i,t}",
        "profit_reward": "r^{profit}_{i,t}=-C^{energy}_{i,t}+R^{sell}_{i,t}",
        "grid_balance_reward": "r^{grid}_{i,t}=-|P^{net}_{i,t}-P^{target}|",
        "storage_degradation_penalty": "p^{deg}_{i,t}=-C^{deg}_{i,t}",
        "constraint_violation_penalty": "p^{viol}_{i,t}=-C^{viol}_{i,t}",
        "total_reward": "r_t=r^{profit}_t+r^{grid}_t-p^{deg}_t-p^{viol}_t",
        "energy_purchase_cost": "C^{energy}_{i,t}=\\max(P^{net}_{i,t},0)c_t",
        "storage_degradation_cost": "C^{deg}_{i,t}=\\lambda_B |P^B_{i,t}|",
        "constraint_violation_cost": "C^{viol}_{i,t}=\\lambda_v \\max(P^{net}_{i,t}-\\bar P,0)",
        "total_cost": "J_t=C^{energy}_t+C^{deg}_t+C^{viol}_t",
        "actor_loss": "\\mathcal{L}_{actor}",
        "critic_loss": "\\mathcal{L}_{critic}",
        "entropy_loss": "-\\alpha H(\\pi)",
        "total_loss": "\\mathcal{L}=\\mathcal{L}_{actor}+\\mathcal{L}_{critic}-\\alpha H",
    })


def generate_demo_run(
    *,
    data_dir: str | Path = "runs",
    run_id: str = "demo_vpp_marl",
    vpp_count: int = 5,
    epochs: int = 3,
    days: int = 35,
    steps_per_day: int = 24,
    async_writer: bool = False,
) -> str:
    logger = ExperimentLogger(
        run_id=run_id,
        data_dir=str(data_dir),
        config={
            "algorithm": "demo_marl",
            "environment": "demo_multi_vpp",
            "vpp_count": int(vpp_count),
            "epochs": int(epochs),
            "days": int(days),
            "steps_per_day": int(steps_per_day),
            "demo": True,
        },
        variable_dictionary=default_variable_dictionary(),
        formulas=default_formulas(),
        metadata={"source": "synthetic_demo", "timestamp_semantics": "generated_calendar_time"},
        async_writer=async_writer,
    )
    start = datetime(2026, 1, 1)
    gradient_step = 0
    for epoch in range(int(epochs)):
        for day_index in range(int(days)):
            date = (start + timedelta(days=day_index)).date().isoformat()
            for time_index in range(int(steps_per_day)):
                hour = 24.0 * time_index / max(1, int(steps_per_day))
                timestamp = (start + timedelta(days=day_index, hours=hour)).isoformat()
                price = 45 + 18 * math.sin((hour - 7) / 24 * 2 * math.pi) + 30 * (17 <= hour <= 21)
                for vpp_number in range(1, int(vpp_count) + 1):
                    vpp_id = f"vpp_{vpp_number:03d}"
                    phase = (vpp_number - 1) * 0.35
                    pv = max(0.0, math.sin((hour - 6) / 13 * math.pi)) * (0.8 + 0.04 * vpp_number)
                    wind = 0.25 + 0.12 * math.sin(day_index * 0.55 + phase)
                    ev = max(0.0, 0.25 + 0.35 * math.sin((hour - 18) / 8 * math.pi))
                    base = 0.9 + 0.25 * math.sin((hour - 8) / 24 * 2 * math.pi + phase)
                    storage_power = 0.18 * math.sin((hour - 12) / 24 * 2 * math.pi + epoch * 0.2)
                    soc = min(95.0, max(5.0, 55.0 + 20.0 * math.sin((day_index + time_index / steps_per_day) * 0.4 + phase)))
                    net_load = base + ev - pv - wind - storage_power
                    logger.log_dataset(
                        epoch_id=epoch,
                        episode_id=day_index,
                        global_env_step=epoch * days * steps_per_day + day_index * steps_per_day + time_index,
                        env_id="env_0",
                        vpp_id=vpp_id,
                        agent_id=f"{vpp_id}_dispatch",
                        policy_id="dispatch_shared",
                        date=date,
                        time_index=time_index,
                        timestamp=timestamp,
                        values={
                            "electricity_price": round(price, 4),
                            "ev_charging_load": round(ev, 4),
                            "storage_power": round(storage_power, 4),
                            "storage_soc": round(soc, 4),
                            "pv_power": round(pv, 4),
                            "wind_power": round(wind, 4),
                            "base_load": round(base, 4),
                            "net_load": round(net_load, 4),
                        },
                        units={name: "MW" for name in ("ev_charging_load", "storage_power", "pv_power", "wind_power", "base_load", "net_load")} | {"electricity_price": "$/MWh", "storage_soc": "%"},
                    )
                    energy_cost = max(0.0, net_load) * price / 100.0
                    degradation = abs(storage_power) * 0.08
                    violation = max(0.0, net_load - 1.4) * 2.0
                    total_cost = energy_cost + degradation + violation
                    profit_reward = -energy_cost + max(0.0, -net_load) * 0.8
                    grid_reward = -abs(net_load - 0.4)
                    total_reward = profit_reward + grid_reward - degradation - violation
                    common = {
                        "epoch_id": epoch,
                        "episode_id": day_index,
                        "env_id": "env_0",
                        "vpp_id": vpp_id,
                        "agent_id": f"{vpp_id}_dispatch",
                        "policy_id": "dispatch_shared",
                        "date": date,
                        "time_index": time_index,
                        "timestamp": timestamp,
                    }
                    logger.log_reward_terms(
                        **common,
                        terms={
                            "profit_reward": round(profit_reward, 4),
                            "grid_balance_reward": round(grid_reward, 4),
                            "storage_degradation_penalty": round(-degradation, 4),
                            "constraint_violation_penalty": round(-violation, 4),
                            "total_reward": round(total_reward, 4),
                        },
                    )
                    logger.log_cost_terms(
                        **common,
                        terms={
                            "energy_purchase_cost": round(energy_cost, 4),
                            "storage_degradation_cost": round(degradation, 4),
                            "constraint_violation_cost": round(violation, 4),
                            "total_cost": round(total_cost, 4),
                        },
                        units={"energy_purchase_cost": "$", "storage_degradation_cost": "$", "constraint_violation_cost": "$", "total_cost": "$"},
                    )
        for vpp_number in range(1, int(vpp_count) + 1):
            vpp_id = f"vpp_{vpp_number:03d}"
            gradient_step += 1
            actor_loss = 1.0 / (epoch + 1) + 0.01 * vpp_number
            critic_loss = 1.5 / (epoch + 1) + 0.02 * vpp_number
            entropy_loss = -0.05 / (epoch + 1)
            logger.log_loss_terms(
                epoch_id=epoch,
                batch_id=epoch,
                gradient_step=gradient_step,
                vpp_id=vpp_id,
                agent_id=f"{vpp_id}_dispatch",
                policy_id="dispatch_shared",
                terms={
                    "actor_loss": round(actor_loss, 5),
                    "critic_loss": round(critic_loss, 5),
                    "entropy_loss": round(entropy_loss, 5),
                    "total_loss": round(actor_loss + critic_loss + entropy_loss, 5),
                },
                optimizer_name="adam",
                network_name="actor_critic",
            )
        logger.log_scalar("epoch_return_mean", 100.0 / (epoch + 1), epoch_id=epoch, gradient_step=gradient_step)
        logger.log_event("training_status", {"message": f"epoch {epoch} generated"}, epoch_id=epoch)
    logger.close()
    return run_id
