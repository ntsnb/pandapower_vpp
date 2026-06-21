from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
import json
from pathlib import Path
from typing import Any

from vpp_dso_sim.learning.reward_contracts import (
    DISPATCH_COMFORT_SOC_PENALTY_WEIGHT,
    DISPATCH_LINEAR_PROJECTION_PENALTY_WEIGHT,
    DISPATCH_PREFERRED_REGION_BONUS_WEIGHT,
    DISPATCH_PRIVATE_PROFIT_WEIGHT,
    DISPATCH_QUADRATIC_PROJECTION_PENALTY_WEIGHT,
    DISPATCH_RAW_DSO_REWARD_WEIGHT,
    DISPATCH_TRACKING_PENALTY_WEIGHT,
    PORTFOLIO_DECISION_MASK_DEFAULT_INTERVAL_STEPS,
    PORTFOLIO_LOCALIZED_DSO_ALIGNMENT_WEIGHT,
    PORTFOLIO_RAW_DSO_REWARD_WEIGHT,
)


@dataclass
class DSORewardConfig:
    enable_tracking_bonus: bool = True
    enable_effective_response_bonus: bool = True
    enable_target_tracking_cost: bool = True
    comfort_violation_weight: float = 0.02
    soc_violation_weight: float = 0.25
    feasibility_bonus_weight: float = 1.0
    safety_margin_weight: float = 1.0
    hard_violation_weight: float = 10.0
    powerflow_failure_weight: float = 20.0
    operation_cost_weight: float = 1.0
    flex_procurement_cost_weight: float = 1.0
    loss_cost_weight: float = 1.0
    curtailment_cost_weight: float = 0.5
    projection_gap_weight: float = 1.0
    projection_attribution: bool = True
    envelope_width_penalty_weight: float = 1.0
    safe_capacity_utilization_weight: float = 0.2
    over_conservative_curtailment_weight: float = 0.5
    smoothness_weight: float = 0.05
    voltage_guard_band_pu: float = 0.02
    line_guard_band_percent: float = 5.0
    trafo_guard_band_percent: float = 5.0
    component_clip: float = 10.0
    raw_action_safety_weight: float = 10.0
    projected_action_safety_weight: float = 5.0
    min_raw_unsafe_penalty: float = 0.1
    raw_safety_epsilon: float = 1.0e-5
    safety_gate_input_mode: str = "max_raw_projected"
    welfare_weight: float = 1.0
    welfare_clip: float = 5.0
    welfare_normalization_mode: str = "per_mwh_running_zscore"
    welfare_baseline_mean: float = 0.0
    welfare_baseline_std: float = 10.0
    soft_safety_gate_kappa: float = 2.0


@dataclass
class VPPDispatchRewardConfig:
    private_profit_weight: float = DISPATCH_PRIVATE_PROFIT_WEIGHT
    use_baseline_service_payment: bool = False
    service_payment_weight: float = 1.0
    availability_payment_weight: float = 1.0
    contract_delivery_weight: float = DISPATCH_TRACKING_PENALTY_WEIGHT
    service_payment_source: str = "baseline_proxy"
    availability_payment_source: str = "baseline_proxy"
    contract_settlement_source: str = "baseline_proxy"
    use_unified_private_profit_v3: bool = False
    require_per_der_settlement_audit: bool = False
    paper_long_fail_on_incomplete_settlement_audit: bool = False
    settlement_power_balance_tolerance_mw: float = 1.0e-6
    projection_attribution: bool = True
    projection_linear_weight: float = DISPATCH_LINEAR_PROJECTION_PENALTY_WEIGHT
    projection_quadratic_weight: float = DISPATCH_QUADRATIC_PROJECTION_PENALTY_WEIGHT
    comfort_soc_weight: float = DISPATCH_COMFORT_SOC_PENALTY_WEIGHT
    battery_degradation_weight: float = 0.0
    degradation_price_per_mwh: float = 5.0
    storage_potential_shaping_weight: float = 0.0
    storage_future_value_mode: str = "disabled"
    storage_future_value_window_steps: int = 16
    storage_charge_efficiency: float = 0.95
    storage_discharge_efficiency: float = 0.95
    storage_discount: float = 0.99
    storage_anti_hoarding_test_required: bool = False
    storage_terminal_value_weight: float = 0.0
    storage_terminal_soc_reference_mode: str = "disabled"
    storage_terminal_soc_reference: float | None = None
    storage_terminal_soc_reference_weight: float = 0.0
    storage_terminal_potential_residual_mode: str = "log_and_ablate"
    preferred_region_bonus_weight: float = DISPATCH_PREFERRED_REGION_BONUS_WEIGHT
    raw_dso_reward_weight: float = DISPATCH_RAW_DSO_REWARD_WEIGHT


@dataclass
class VPPPortfolioRewardConfig:
    mode: str = "legacy_localized"
    decision_interval_steps: int = PORTFOLIO_DECISION_MASK_DEFAULT_INTERVAL_STEPS
    long_horizon_profit_weight: float = 0.10
    verified_capacity_weight: float = 0.5
    delivery_reliability_weight: float = 1.0
    future_shield_penalty_weight: float = 1.0
    future_projection_penalty_weight: float = 0.5
    future_comfort_soc_weight: float = 0.02
    switching_keep_cost: float = 0.0
    switching_reweight_cost: float = 0.02
    switching_membership_change_cost: float = 0.08
    raw_dso_reward_weight: float = PORTFOLIO_RAW_DSO_REWARD_WEIGHT
    localized_dso_alignment_weight: float = PORTFOLIO_LOCALIZED_DSO_ALIGNMENT_WEIGHT


@dataclass
class ShieldRewardConfig:
    dso_penalty_coef: float = 1.0
    dispatch_penalty_coef: float = 1.0
    portfolio_future_penalty_coef: float = 1.0


@dataclass
class VPPRewardConfig:
    dispatch: VPPDispatchRewardConfig = field(default_factory=VPPDispatchRewardConfig)
    portfolio: VPPPortfolioRewardConfig = field(default_factory=VPPPortfolioRewardConfig)


@dataclass
class RewardConfig:
    version: str = "v1_legacy"
    critic_reward_scale: float = 0.01
    dso: DSORewardConfig = field(default_factory=DSORewardConfig)
    vpp: VPPRewardConfig = field(default_factory=VPPRewardConfig)
    shield: ShieldRewardConfig = field(default_factory=ShieldRewardConfig)

    @property
    def is_v2_minimal(self) -> bool:
        return str(self.version).lower() == "v2_minimal"

    @property
    def is_v3_market_safety(self) -> bool:
        return str(self.version).lower() in {
            "v3_market_safety",
            "v3_1_market_safety",
            "v3.1_market_safety",
            "v3_market_safety_cmdp_ready",
        }

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any] | None) -> "RewardConfig":
        data = dict(payload or {})
        version = str(data.get("version", "v1_legacy"))
        critic_reward_scale = float(data.get("critic_reward_scale", data.get("reward_scale", 0.01)))

        dso_data = dict(data.get("dso", {}))
        vpp_data = dict(data.get("vpp", {}))
        dispatch_data = dict(vpp_data.get("dispatch", {}))
        portfolio_data = dict(vpp_data.get("portfolio", {}))
        shield_data = dict(data.get("shield", {}))

        if version.lower() == "v2_minimal":
            dso_data = {
                "enable_tracking_bonus": False,
                "enable_effective_response_bonus": False,
                "enable_target_tracking_cost": False,
                "comfort_violation_weight": 0.0,
                "soc_violation_weight": 0.0,
                "feasibility_bonus_weight": 0.0,
                "envelope_width_penalty_weight": 0.0,
                **dso_data,
            }
            dispatch_data = {
                "use_baseline_service_payment": True,
                "contract_delivery_weight": 10.0,
                "projection_linear_weight": 2.0,
                "projection_quadratic_weight": 5.0,
                "comfort_soc_weight": 0.02,
                "battery_degradation_weight": 0.01,
                "preferred_region_bonus_weight": 0.0,
                **dispatch_data,
            }
            portfolio_data = {
                "mode": "window_return",
                "decision_interval_steps": 24,
                "long_horizon_profit_weight": 0.05,
                "verified_capacity_weight": 0.5,
                "delivery_reliability_weight": 1.0,
                "future_shield_penalty_weight": 1.0,
                "future_projection_penalty_weight": 0.5,
                "future_comfort_soc_weight": 0.02,
                "switching_keep_cost": 0.0,
                "switching_reweight_cost": 0.05,
                "switching_membership_change_cost": 0.20,
                **portfolio_data,
            }
        elif version.lower() in {
            "v3_market_safety",
            "v3_1_market_safety",
            "v3.1_market_safety",
            "v3_market_safety_cmdp_ready",
        }:
            dso_data = {
                "enable_tracking_bonus": False,
                "enable_effective_response_bonus": False,
                "enable_target_tracking_cost": False,
                "comfort_violation_weight": 0.0,
                "soc_violation_weight": 0.0,
                "feasibility_bonus_weight": 0.0,
                "curtailment_cost_weight": 0.0,
                "safe_capacity_utilization_weight": 0.0,
                "over_conservative_curtailment_weight": 0.0,
                "raw_action_safety_weight": 10.0,
                "projected_action_safety_weight": 5.0,
                "min_raw_unsafe_penalty": 0.1,
                "raw_safety_epsilon": 1.0e-5,
                "safety_gate_input_mode": "max_raw_projected",
                "welfare_weight": 1.0,
                "welfare_clip": 5.0,
                "welfare_normalization_mode": "per_mwh_running_zscore",
                "welfare_baseline_mean": 0.0,
                "welfare_baseline_std": 10.0,
                "soft_safety_gate_kappa": 2.0,
                **dso_data,
            }
            dispatch_data = {
                "private_profit_weight": 1.0,
                "use_baseline_service_payment": False,
                "service_payment_weight": 0.0,
                "availability_payment_weight": 0.0,
                "contract_delivery_weight": 0.0,
                "service_payment_source": "disabled",
                "availability_payment_source": "disabled",
                "contract_settlement_source": "disabled",
                "use_unified_private_profit_v3": True,
                "require_per_der_settlement_audit": True,
                "paper_long_fail_on_incomplete_settlement_audit": True,
                "settlement_power_balance_tolerance_mw": 1.0e-6,
                "projection_linear_weight": 2.0,
                "projection_quadratic_weight": 5.0,
                "battery_degradation_weight": 0.01,
                "degradation_price_per_mwh": 5.0,
                "preferred_region_bonus_weight": 0.0,
                "storage_potential_shaping_weight": 0.02,
                "storage_future_value_mode": "price_forecast_window",
                "storage_future_value_window_steps": 16,
                "storage_charge_efficiency": 0.95,
                "storage_discharge_efficiency": 0.95,
                "storage_discount": 0.99,
                "storage_anti_hoarding_test_required": True,
                "storage_terminal_value_weight": 0.0,
                "storage_terminal_soc_reference_mode": "disabled",
                "storage_terminal_soc_reference": None,
                "storage_terminal_soc_reference_weight": 0.0,
                "storage_terminal_potential_residual_mode": "log_and_ablate",
                **dispatch_data,
            }
            portfolio_data = {
                "mode": "window_return",
                "decision_interval_steps": 24,
                "long_horizon_profit_weight": 0.05,
                "verified_capacity_weight": 0.5,
                "delivery_reliability_weight": 1.0,
                "future_shield_penalty_weight": 1.0,
                "future_projection_penalty_weight": 0.5,
                "future_comfort_soc_weight": 0.02,
                "switching_keep_cost": 0.0,
                "switching_reweight_cost": 0.05,
                "switching_membership_change_cost": 0.20,
                **portfolio_data,
            }

        return cls(
            version=version,
            critic_reward_scale=critic_reward_scale,
            dso=DSORewardConfig(**_known_kwargs(DSORewardConfig, dso_data)),
            vpp=VPPRewardConfig(
                dispatch=VPPDispatchRewardConfig(**_known_kwargs(VPPDispatchRewardConfig, dispatch_data)),
                portfolio=VPPPortfolioRewardConfig(**_known_kwargs(VPPPortfolioRewardConfig, portfolio_data)),
            ),
            shield=ShieldRewardConfig(**_known_kwargs(ShieldRewardConfig, shield_data)),
        )


def _known_kwargs(cls, payload: dict[str, Any]) -> dict[str, Any]:
    fields = getattr(cls, "__dataclass_fields__", {})
    return {key: value for key, value in payload.items() if key in fields}


def write_reward_config_artifacts(output_dir: str | Path, reward_config: RewardConfig) -> dict[str, str]:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    payload = reward_config.to_dict()
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    yaml_path = out / "resolved_reward_config.yaml"
    hash_path = out / "reward_config_hash.txt"
    try:
        import yaml

        yaml_text = yaml.safe_dump(payload, sort_keys=True, allow_unicode=True)
    except Exception:
        yaml_text = json.dumps(payload, indent=2, sort_keys=True)
    yaml_path.write_text(yaml_text, encoding="utf-8")
    hash_path.write_text(digest + "\n", encoding="utf-8")
    return {
        "resolved_reward_config": str(yaml_path),
        "reward_config_hash": digest,
        "reward_config_hash_path": str(hash_path),
    }
