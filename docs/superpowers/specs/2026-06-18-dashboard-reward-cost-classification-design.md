# Dashboard Reward/Cost Classification Design

## Goal

Add explicit reward/cost semantics to the MARL dashboard so users can distinguish raw physical/economic quantities, terms that actually enter the v3.1 training reward, and diagnostic or excluded terms.

## Confirmed User Semantics

- Cost-free reward pages show unscaled real physical/economic values as the primary chart values.
- Each cost-free reward item must also show the weight used in the final reward formula.
- A derived field should show `raw value * reward weight` where a direct reward weight exists.
- Reward-scaled cost pages show the penalty values as they enter the training reward.
- Formula sections must separate included objective terms from raw decomposition and diagnostic/excluded terms.
- Each formula row must include a bilingual physical meaning.

## Current Evidence

- `src/vpp_dso_sim/envs/reward_design.py` computes v3.1 dispatch reward as:
  - `private_profit_weight * operational_surplus`
  - `service_payment_weight * flexibility_service_payment`
  - `availability_payment_weight * availability_payment`
  - `preferred_region_bonus`
  - `storage_potential_shaping_reward`
  - `contract_delivery_weight * contract_delivery_penalty`
  - `dispatch_projection_penalty`
  - `comfort_soc_weight * scaled_comfort_soc_penalty`
  - `battery_degradation_weight * battery_degradation_cost`
- `configs/rewards/v3_market_safety/reward_v3_1_market_safety.yaml` sets service, availability, and contract delivery weights to `0.0`.
- Current dispatch trace rows contain raw costs and weights but do not expose all reward-entering derived quantities as first-class dashboard metrics.
- Current dashboard formula rendering is a flat list sorted by hierarchy, so diagnostic items can appear beside final reward components.

## Architecture

Use a low-intrusion trace-backed design:

- Add derived reward/cost trace fields in `src/vpp_dso_sim/learning/reward_trace.py`.
- Import those fields in `scripts/watch_paper_long_run.py` as dashboard reward/cost metrics.
- Add dashboard-side metric classification utilities in the frontend.
- Reuse existing API endpoints (`/rewards`, `/costs`) and filter/classify responses client-side.
- Extend variable enrichment so formulas and descriptions are bilingual and clarify inclusion semantics.

No reward formula, environment step/reset API, model update API, or training framework behavior changes.

## New Metric Groups By Display Semantics

### Cost-Free Reward

Primary values are raw/unscaled physical or economic terms:

- `economic_operational_surplus`
- `market_energy_margin_total`
- `export_revenue_total`
- `pv_export_revenue_total`
- `mt_export_revenue_total`
- `storage_discharge_revenue_total`
- `evcs_user_revenue_total`
- `energy_market_revenue`
- `service_payment`
- `flexibility_service_payment`
- `availability_payment`
- `preferred_region_bonus`
- `storage_potential_raw`
- `storage_potential_shaping_reward`

Where applicable, show:

- reward weight name
- reward weight value
- weighted contribution metric
- inclusion status

### Reward-Scaled Costs

Primary values are the penalty contributions as used by reward:

- `reward_scaled_contract_delivery_penalty`
- `reward_scaled_dispatch_projection_penalty`
- `reward_scaled_comfort_soc_penalty`
- `reward_scaled_battery_degradation_penalty`

Raw source metrics remain visible in the existing cost page.

### Diagnostic / Excluded

Metrics with zero weights or diagnostic residual meaning must be grouped separately in formula display:

- `service_payment` when `service_payment_weight=0`
- `flexibility_service_payment` when `service_payment_weight=0`
- `availability_payment` when `availability_payment_weight=0`
- `contract_delivery_penalty` when `contract_delivery_weight=0`
- residual metrics such as `private_profit_vs_visible_energy_residual`

## Data Flow

1. Training reward calculation produces component dictionaries.
2. `dispatch_private_profit_trace_rows` expands those dictionaries into per-VPP CSV rows.
3. `watch_paper_long_run.py` mirrors CSV rows into dashboard Parquet tables.
4. Frontend pages request standard reward/cost responses.
5. New client utilities filter metrics into semantic page views and enrich rows with display metadata.

## Error Handling

- Missing derived fields should not break existing dashboard pages.
- New pages show a bilingual empty state explaining that future trace rows are required.
- Existing raw reward/cost pages keep working with older runs.

## Testing

- Unit tests for trace-derived fields.
- Watchdog import tests for new reward/cost dashboard metrics.
- Variable enrichment tests for bilingual descriptions/formulas.
- Frontend tests for formula sectioning and new page filtering.
- Existing reward/cost dashboard tests must continue to pass.
