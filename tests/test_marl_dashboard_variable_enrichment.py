from __future__ import annotations

import re

import pytest

from marl_dashboard.demo.generate_demo_run import default_variable_dictionary
from marl_dashboard.backend.storage.variable_enrichment import (
    default_formula_dictionary,
    enrich_metric_record,
    enrich_variable_definition,
    should_use_default_formula,
    variable_defaults,
)


LIVE_METRIC_NAMES = [
    "ac_projected_target_p_mw",
    "accepted_delta_p_mw",
    "action_landing_ratio",
    "actual_delta_p_mw",
    "actual_target_p_mw",
    "baseline_p_mw",
    "decoded_target_p_mw",
    "delivered_p_mw",
    "device_feasible_target_p_mw",
    "episode_rows",
    "loss_rows",
    "policy_normalized_aggregate_action",
    "policy_normalized_der_action_mean",
    "policy_normalized_der_action_std",
    "progress_rows",
    "raw_target_p_mw",
    "requested_delta_p_mw",
    "step_progress_pct",
    "violations_so_far",
    "availability_payment",
    "dispatch_private_profit_reward",
    "dispatch_reward_env",
    "economic_operational_surplus",
    "evcs_user_revenue_total",
    "export_revenue_total",
    "flexibility_service_payment",
    "market_energy_margin_total",
    "global_env_step",
    "mt_export_revenue_total",
    "preferred_region_bonus",
    "private_profit_proxy",
    "private_profit_weight",
    "pv_export_revenue_total",
    "quality_adjusted_operational_surplus",
    "service_payment",
    "service_payment_weight",
    "availability_payment_weight",
    "storage_potential_raw",
    "storage_potential_shaping_reward",
    "storage_potential_shaping_weight",
    "storage_discharge_revenue_total",
    "visible_energy_minus_operation_cost",
    "battery_degradation_cost",
    "battery_degradation_cost_total",
    "comfort_cost_total",
    "contract_delivery_penalty",
    "der_operating_cost_total",
    "der_operation_cost",
    "dispatch_projection_penalty",
    "evcs_wholesale_cost_total",
    "flex_energy_cost_total",
    "hvac_energy_cost_total",
    "import_energy_cost_total",
    "scaled_comfort_soc_penalty",
    "reward_scaled_contract_delivery_penalty",
    "reward_scaled_dispatch_projection_penalty",
    "reward_scaled_training_projection_penalty",
    "reward_scaled_total_projection_penalty",
    "reward_scaled_comfort_soc_penalty",
    "reward_scaled_battery_degradation_penalty",
    "storage_charge_cost_total",
    "total_cost_so_far",
    "unclassified_import_cost_total",
    "unserved_penalty_total",
    "critic_grad_norm",
    "dispatch_policy_loss",
    "dso_policy_loss",
    "portfolio_policy_loss",
    "projection_gap_mw",
]


def has_cjk(value: object) -> bool:
    return any("\u4e00" <= char <= "\u9fff" for char in str(value or ""))


def has_snake_case_identifier(value: object) -> bool:
    return re.search(r"\b[a-z][a-z0-9]+_[a-z0-9_]+\b", str(value or "")) is not None


@pytest.mark.parametrize("metric_name", LIVE_METRIC_NAMES)
def test_live_metric_rows_have_bilingual_descriptions(metric_name: str):
    row = enrich_metric_record(
        {
            "metric_name": metric_name,
            "metric_group": "dataset",
            "description": "English-only upstream description.",
            "unit": None,
        }
    )

    assert row["display_name"]
    assert " / " in row["display_name"]
    assert has_cjk(row["description"])
    assert metric_name not in row["description"]


REWARD_COST_METRIC_NAMES = [
    "availability_payment",
    "dispatch_private_profit_reward",
    "dispatch_reward_env",
    "dispatch_reward_train",
    "economic_operational_surplus",
    "energy_market_revenue",
    "evcs_user_revenue_total",
    "export_revenue_total",
    "flexibility_service_payment",
    "market_energy_margin_total",
    "mt_export_revenue_total",
    "preferred_region_bonus",
    "private_profit_proxy",
    "private_profit_weight",
    "pv_export_revenue_total",
    "quality_adjusted_operational_surplus",
    "reward_so_far",
    "service_payment",
    "service_payment_weight",
    "availability_payment_weight",
    "storage_potential_raw",
    "storage_potential_shaping_reward",
    "storage_potential_shaping_weight",
    "storage_discharge_revenue_total",
    "total_reward",
    "visible_energy_minus_operation_cost",
    "battery_degradation_cost",
    "battery_degradation_cost_total",
    "comfort_cost_total",
    "contract_delivery_penalty",
    "der_operating_cost_total",
    "der_operation_cost",
    "dispatch_projection_penalty",
    "evcs_wholesale_cost_total",
    "flex_energy_cost_total",
    "hvac_energy_cost_total",
    "import_energy_cost_total",
    "scaled_comfort_soc_penalty",
    "reward_scaled_contract_delivery_penalty",
    "reward_scaled_dispatch_projection_penalty",
    "reward_scaled_training_projection_penalty",
    "reward_scaled_total_projection_penalty",
    "reward_scaled_comfort_soc_penalty",
    "reward_scaled_battery_degradation_penalty",
    "storage_charge_cost_total",
    "total_cost",
    "total_cost_so_far",
    "unclassified_import_cost_total",
    "unserved_penalty_total",
]


@pytest.mark.parametrize("metric_name", REWARD_COST_METRIC_NAMES)
def test_reward_cost_metric_defaults_include_chinese_symbolic_formula(metric_name: str):
    defaults = variable_defaults(metric_name)

    assert defaults.get("formula_latex"), metric_name
    formula = str(defaults["formula_latex"])
    assert has_cjk(formula), metric_name
    assert not has_snake_case_identifier(formula), metric_name


def test_english_code_formula_is_replaced_by_chinese_symbolic_default():
    row = enrich_metric_record(
        {
            "metric_name": "energy_market_revenue",
            "metric_group": "reward",
            "formula_latex": "market_price * delivered_p_mw * dt_hours",
            "description": "Energy-market revenue.",
        }
    )

    assert should_use_default_formula("market_price * delivered_p_mw * dt_hours")
    assert row["formula_latex"] == variable_defaults("energy_market_revenue")["formula_latex"]
    assert "market_price" not in row["formula_latex"]
    assert has_cjk(row["formula_latex"])


def test_service_payment_description_warns_when_diagnostic_or_excluded():
    row = enrich_metric_record(
        {
            "metric_name": "service_payment",
            "metric_group": "reward",
            "description": "Accepted flexibility service payment.",
            "unit": "currency",
        }
    )

    assert "当前 v3.1" in row["description"]
    assert "不计入最终训练奖励" in row["description"]


def test_electricity_price_formula_uses_braced_time_subscript():
    formula = variable_defaults("electricity_price")["formula_latex"]

    assert formula == "\\pi_{t}"
    assert not has_snake_case_identifier(formula)


def _compact_formula(value: object) -> str:
    return re.sub(r"\s+", "", str(value or ""))


def _formula_lhs(value: object) -> str:
    text = str(value or "")
    return text.split("=", 1)[0].strip() if "=" in text else ""


@pytest.mark.parametrize("metric_name", REWARD_COST_METRIC_NAMES)
def test_reward_cost_formula_lhs_matches_variable_symbol(metric_name: str):
    defaults = variable_defaults(metric_name)
    formula = defaults.get("formula_latex")
    if not formula or "=" not in str(formula):
        pytest.skip(f"{metric_name} has no equation-style formula")

    assert _compact_formula(defaults.get("symbol")) == _compact_formula(_formula_lhs(formula)), metric_name


def test_default_formula_dictionary_contains_canonical_reward_cost_formulas():
    formulas = default_formula_dictionary()

    for metric_name in ("total_reward", "energy_market_revenue", "energy_purchase_cost", "total_cost"):
        assert formulas[metric_name] == variable_defaults(metric_name)["formula_latex"]
        assert has_cjk(formulas[metric_name])
    assert "c_t" not in formulas["energy_purchase_cost"]
    assert "C^{energy}" not in formulas["energy_purchase_cost"]


def test_demo_variable_dictionary_is_normalized_to_canonical_formula_symbols():
    enriched = {
        item["name"]: enrich_metric_record({"metric_name": item["name"], "formula_latex": item.get("formula_latex")})
        for item in default_variable_dictionary()
    }

    assert enriched["electricity_price"]["formula_latex"] == "\\pi_{t}"
    assert enriched["storage_power"]["formula_latex"] == "P^{ESS}_{i,t}"
    assert enriched["wind_power"]["formula_latex"] == "P^{WT}_{i,t}"
    assert enriched["base_load"]["formula_latex"] == "P^{load}_{i,t}"
    assert "c_t" not in enriched["energy_purchase_cost"]["formula_latex"]
    assert "P^B" not in enriched["storage_degradation_cost"]["formula_latex"]


def test_variable_dictionary_entries_replace_legacy_formula_symbols():
    raw = {
        "name": "electricity_price",
        "symbol": "c_t",
        "formula_latex": "c_t",
        "display_name": "Electricity price",
        "physical_meaning": "Market price.",
    }

    enriched = enrich_variable_definition(raw)

    assert enriched["symbol"] == "\\pi_{t}"
    assert enriched["formula_latex"] == "\\pi_{t}"
