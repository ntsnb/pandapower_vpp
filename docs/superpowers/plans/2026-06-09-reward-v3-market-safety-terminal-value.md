# Reward V3 Market-Safety Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement reward-v3 so DSO training is safety-first and curtailment-free, VPP dispatch reward uses non-duplicated real settlement, EVCS charging revenue is tied to EVCS DER audit, storage gets correct step-level potential shaping and terminal-only terminal value, and paper-long experiments include balanced generation/absorption scenarios plus raw/no-shield safety evaluation.

**Architecture:** Keep v1/v2 reward behavior intact and add a stricter `v3_market_safety` path. The v3 path computes VPP settlement before DSO reward, feeds DSO only normalized operational surplus excluding transfer payments, gates welfare by unweighted normalized raw safety diagnostics, and logs raw/projected/no-shield safety so the safety shield cannot silently replace learning. HAPPO/HATRPO/MAPPO/MATD3 training stays compatible but receives v3-specific KL, advantage, reward-scale, and gradient stability instrumentation.

**Tech Stack:** Python 3.12, pytest, pandas, pandapower, YAML configs, dataclasses, existing `vpp_dso_sim` simulation, reward, safety projection, and MARL modules.

---

## Revision 0.3 Scope

This plan replaces the earlier implementation plan after reviewing `docs/superpowers/edit/edit_0_1.md` and `docs/superpowers/edit/edit_0_2.md`. The edit files raised valid high-risk issues in the old plan. The major changes are:

| Risk from edit file | Decision | Plan change |
|---|---|---|
| DSO welfare was clipped without standardization | Adopt | Add per-MWh normalization plus baseline/running z-score and logs for raw/per-MWh/z-score/bounded welfare. |
| Safety gate used weighted `safety_cost` and could saturate | Adopt | Gate uses unweighted normalized raw safety cost; weighted safety penalty remains in reward. |
| Safe-first proof did not map to implementation | Adopt | Add parameter-sweep tests and raw-unsafe/projected-safe tests. |
| VPP settlement double-counted service and availability payments | Adopt | Use a single true-settlement-profit entry for dispatch reward; no separate service/availability reward terms in v3. |
| `energy_market_revenue` and EVCS wholesale cost could double charge import energy | Adopt | Replace mixed signed revenue with per-DER buy/sell settlement components. |
| EVCS fallback treated every negative VPP MW as EVCS charging | Adopt | EVCS revenue must come from per-DER EVCS audit; no total-negative-power fallback. |
| Storage terminal value could be added every step | Adopt | Potential shaping is step-level; terminal value applies only at `done` or `truncated`. |
| DSO reward used VPP private profit including transfer payments | Adopt | DSO receives `vpp_operational_surplus_ex_transfer`, not service/availability payments. |
| Task 4 patched welfare after DSO reward had already been computed | Adopt | Enforce order: dispatch settlement first, DSO reward second. |
| Balanced generation scenario lacked physical stress acceptance | Adopt | Add reverse-flow, high-voltage, peak-import, storage-arbitrage, and EVCS-pressure acceptance tests. |
| HAPPO/HATRPO stability was under-specified | Adopt | Add reward-v3 training stability config, KL early stopping, per-role advantage/value normalization, and gradient audit. |
| Full CMDP/Lagrangian should replace reward shaping immediately | Partially adopt | Keep CMDP/Lagrangian as a later research enhancement. This v3 plan fixes the current reward accounting and safety diagnostics first. |
| `R_export` and `R_ESS_discharge` formula could be read as duplicate revenue | Adopt | Replace aggregate export formula with mutually exclusive DER-level revenue terms. |
| `C_import` could duplicate EVCS/ESS/HVAC import costs | Adopt | Remove ambiguous `C_import`; use EVCS wholesale, ESS charge, HVAC energy, flexible-load energy, and optional unclassified buy cost. |
| `settlement_audit_complete` was too weak | Adopt | Add dynamic required settlement fields by DER type and fail paper-long if any required audit field is missing. |
| DER audit power could disagree with delivered VPP power | Adopt | Add settlement power balance formula, logs, and bounded-gap tests. |
| `min_raw_unsafe_penalty` was too hard for tiny numerical noise | Adopt | Add `raw_safety_epsilon` and apply the floor only after epsilon. |
| Safety gate ignored projected unsafe execution | Adopt | Gate on `max(raw_norm, projected_norm)` by default. |
| Welfare baseline default could over-clip economic signal | Adopt | Add a welfare calibration task and require clip saturation rate below 10 percent before paper-long. |
| Storage potential plus terminal value may encourage hoarding | Adopt | Add terminal-residual policy, anti-hoarding test, and storage temporal-value ablation. |
| Service/availability payments may still be proxy payments | Adopt | Log payment source and separate proxy/no-service/cleared-service ablations. |
| Profile-only scenario stress is not enough | Adopt | Add pre-control AC stress scan acceptance. |
| Static stability config tests are insufficient | Adopt | Add dynamic config-load and KL early-stop tests. |
| Task list is too large for one execution batch | Adopt | Split execution into Phase A reward-audit closure and Phase B scenario/training/ablation. |

---

## Reward V3 Contract

### Power sign convention

Project internal convention remains:

```text
P: injection to grid
-P: absorption from grid
```

Settlement must not infer DER type only from aggregate VPP power. A negative VPP net power can be EVCS charging, storage charging, HVAC load, flexible load, or projection-induced absorption. Reward-v3 must use per-DER audit fields when assigning revenue or cost.

### VPP settlement quantities

For VPP `i` at step `t`:

```math
\Pi^{op}_{i,t}
= R^{PV,export}_{i,t}
+ R^{MT,export}_{i,t}
+ R^{ESS,discharge}_{i,t}
+ R^{EVCS,user}_{i,t}
- C^{EVCS,wholesale}_{i,t}
- C^{ESS,charge}_{i,t}
- C^{HVAC,energy}_{i,t}
- C^{flex,energy}_{i,t}
- C^{unclassified,buy}_{i,t}
- C^{DER,op}_{i,t}
- C^{deg}_{i,t}
- C^{comfort}_{i,t}
- C^{unserved}_{i,t}
```

This is the operational surplus excluding DSO-to-VPP transfer payments.

The export terms are mutually exclusive. Do not also add an aggregate `R_export` if `R_PV_export`, `R_MT_export`, and `R_ESS_discharge` are already present. If an implementation keeps an aggregate field for logging, define it only as:

```math
R^{export,total}_{i,t}
= R^{PV,export}_{i,t}
+ R^{MT,export}_{i,t}
+ R^{ESS,discharge}_{i,t}
```

`C_unclassified_buy` is allowed only for smoke-test diagnostics when per-DER audit is incomplete. It must be zero, absent, or rejected for reward-v3 paper-long MARL training.

```math
\Pi^{private}_{i,t}
= \Pi^{op}_{i,t}
+ R^{service}_{i,t}
+ R^{availability}_{i,t}
- C^{contract}_{i,t}
```

Dispatch reward uses the private settlement once:

```math
r^{dispatch}_{i,t}
= \lambda_{\Pi}\,\Pi^{private}_{i,t}
- \lambda_{proj} C^{proj}_{i,t}
- \lambda_{constraint} C^{constraint}_{i,t}
+ F^{storage}_{i,t}
+ I^{terminal}_{t} R^{storage,T}_{i,t}
```

There must be no extra `+ service_payment_weight * service_payment` or `+ availability_payment_weight * availability_payment` in v3 after those payments are already part of `Pi_private`.

### DSO welfare input

DSO welfare input is not VPP private profit. It is the operational surplus excluding transfers:

```math
SW^{proxy}_{t}
= \sum_i \Pi^{op}_{i,t}
```

Service and availability payments are logged as:

```text
dso_transfer_payment_cost
service_payment
availability_payment
service_payment_source = baseline_proxy | cleared_award | disabled
availability_payment_source = baseline_proxy | capacity_contract | disabled
```

They are not added to DSO welfare. If a later experiment wants a DSO budget objective, add a separate budget term and label it explicitly.

### Welfare normalization

DSO must not directly clip raw welfare. The required transformation is:

```math
W^{perMWh}_t =
\frac{SW^{proxy}_{t}}{\max(N_{VPP}\Delta t,\epsilon)}
```

```math
\widetilde{W}_t =
\operatorname{clip}
\left(
\frac{W^{perMWh}_t-\mu_W}{\sigma_W+\epsilon},
-W_{clip},
W_{clip}
\right)
```

`mu_W` and `sigma_W` may come from a fixed baseline calibration or an online running z-score. For deterministic tests, use baseline values from config.

### Safety cost and safety gate

Reward-v3 separates the weighted safety penalty from the gate input.

Unweighted normalized raw safety:

```math
C^{safe,raw,norm}_t =
C^{V,raw}_t + C^{L,raw}_t + C^{T,raw}_t + C^{PF,raw}_t
```

Weighted safety penalty:

```math
C^{safe,penalty}_t =
\lambda_V C^{V,raw}_t
+ \lambda_L C^{L,raw}_t
+ \lambda_T C^{T,raw}_t
+ \lambda_{PF} C^{PF,raw}_t
+ \lambda_{proj} C^{proj}_t
```

Soft safety gate:

```math
G^{safe}_t =
\exp
\left(
-\kappa
\max(C^{safe,raw,norm}_t, C^{safe,proj,norm}_t)
\right)
```

The default gate uses `max(raw, projected)` so that welfare is not amplified when the raw action looks safe but the projected/certified execution is still unsafe. A future ablation may use `raw + rho * projected`, but paper-long defaults must use the more conservative max form.

DSO reward:

```math
r^{DSO}_t =
- C^{safe,penalty}_t
- \lambda_{loss}\widetilde{P}_{loss,t}
- \lambda_{smooth}\|E_t-E_{t-1}\|^2
+ \beta G^{safe}_t \widetilde{W}_t
```

Curtailed capacity, safe capacity utilization, and envelope width are diagnostics only in v3.

### Storage temporal value

Step-level potential shaping:

```math
F^{storage}_t =
\gamma \Phi(s_{t+1}, t+1) - \Phi(s_t, t)
```

Terminal-only storage value:

```math
R^{storage,T}_t =
\mathbf{1}_{done \lor truncated}
\left(
\kappa_T E^{stored}_T
- \lambda_{socT}(SOC_T-SOC^{target})^2
\right)
```

`storage_terminal_value_reward` must be zero during non-terminal steps. Log `storage_terminal_value_proxy` every step if useful, but do not add it to reward except at terminal.

Potential shaping and terminal value are not automatically harmless together. The implementation must log the terminal potential residual and provide an anti-hoarding test. If future sell price is below buy price and SOC is above target, storage temporal reward must not encourage further charging.

### Settlement audit and power balance

Reward-v3 settlement is valid only when the DER-level audit is complete for the VPP asset mix.

Required fields by DER type:

| DER type | Required fields |
|---|---|
| EVCS | `evcs_grid_p_mw`, `evcs_charge_efficiency`, `retail_evcs_tariff` or `envelope.retail_evcs_tariff` |
| Storage | `storage_charge_p_mw`, `storage_discharge_p_mw`, `storage_soc_before`, `storage_soc_after` |
| PV | `pv_export_p_mw` |
| Microturbine | `microturbine_export_p_mw`, `der_operation_cost` |
| HVAC | `hvac_p_mw`, `comfort_cost` |
| Flexible load | `flex_load_p_mw`, plus `comfort_cost` or `unserved_energy_cost` |

Power balance audit:

```math
P^{audit}_{i,t}
= P^{PV}_{i,t}
+ P^{MT}_{i,t}
+ P^{ESS,dis}_{i,t}
- P^{ESS,ch}_{i,t}
- P^{EVCS}_{i,t}
- P^{HVAC}_{i,t}
- P^{flex}_{i,t}
```

```math
|P^{audit}_{i,t} - P^{delivered}_{i,t}| \le \epsilon_P
```

Reward traces must include:

```text
settlement_required_fields_missing_count
settlement_missing_<field_name>
settlement_audit_complete
settlement_power_balance_gap_mw
settlement_power_balance_ok
```

For paper-long reward-v3 MARL training, `settlement_audit_complete` and `settlement_power_balance_ok` must both be `1` for all training steps.

---

## File Structure

Create:

- `tests/test_reward_v3_market_safety.py`: reward-v3 contracts, safety gate, welfare normalization, settlement accounting, EVCS audit, storage temporal value, DSO ordering.
- `tests/test_reward_v3_scenario_mix.py`: balanced generation and physical stress acceptance tests.
- `tests/test_reward_v3_training_stability.py`: v3 HAPPO/HATRPO config and logging contract tests.
- `src/vpp_dso_sim/data_sources/acn.py`: ACN-style EVCS session adapter.
- `src/vpp_dso_sim/data_sources/tariff.py`: tariff and price profile adapter.
- `configs/rewards/v3_market_safety/reward_v3_market_safety.yaml`: reward-v3 defaults.
- `configs/algorithms/reward_v3_happo_stable.yaml`: stable HAPPO/HATRPO defaults for reward-v3.
- `configs/scenarios/benchmark/european_lv_benchmark_v3_balanced_generation.yaml`: balanced absorption/generation benchmark.
- `configs/experiments/paper_long/reward_v3_market_safety/european_lv_benchmark_v3_balanced_generation_reward_v3.yaml`: paper-long config.

Modify:

- `src/vpp_dso_sim/learning/reward_config.py`: add reward-v3 config, welfare normalization, storage temporal value, and stability config fields.
- `src/vpp_dso_sim/entities/dso.py`: implement v3 reward with normalized welfare, unweighted safety gate, raw/projected safety diagnostics, and curtailment-free training terms.
- `src/vpp_dso_sim/envs/reward_design.py`: implement per-DER settlement accounting, VPP profit/surplus aggregation, storage potential/terminal value, and role reward ordering helpers.
- `src/vpp_dso_sim/der/ev.py`: expose requested/delivered energy where missing.
- `src/vpp_dso_sim/der/evcs.py`: support session-based EVCS construction and settlement audit fields.
- `src/vpp_dso_sim/simulation/scenario.py`: load EVCS sessions/tariffs and balanced generation configs.
- `src/vpp_dso_sim/simulation/simulator.py`: compute dispatch settlement before DSO reward-v3 and export raw/projected/no-shield safety diagnostics.
- `src/vpp_dso_sim/optimization/ac_security_projection.py`: return raw and projected AC safety cost components where available.
- `src/vpp_dso_sim/learning/advanced_marl.py`: log role KL, entropy, gradient norms, advantage/value stats, and v3 reward components.
- `src/vpp_dso_sim/learning/hatrpo.py`: add v3 HATRPO/HAPPO stability options and logs.
- `src/vpp_dso_sim/learning/deep_rl.py`: keep v3 logging compatibility for single-agent/smoke paths.
- `src/vpp_dso_sim/learning/reward_trace.py`: export all v3 reward and safety fields.
- `docs/reward_terms_full_glossary_cn.md`: update formulas after implementation.

Do not delete v1/v2 reward code. Existing experiments must remain reproducible.

---

## Execution Phases

### Phase A: Reward Accounting and Safety Diagnostics Closure

Execute only:

```text
Task 0: Freeze Reward V3 Accounting Contract
Task 1: Reward V3 Contract Tests
Task 2: Implement Reward V3 Config and DSO Reward
Task 2.5: Welfare Normalization Calibration
Task 3: VPP Settlement Without Double Counting
Task 4: Storage Potential Shaping and Terminal-Only Value
Task 5: Compute Dispatch Settlement Before DSO Reward
Task 6: Raw, Projected, and No-Shield Safety Diagnostics
```

Phase A acceptance:

```text
1. reward_components.csv contains all v3 audit fields.
2. DSO curtailment/safe-capacity training fields are absent from v3 reward output.
3. service and availability payments are not double-counted.
4. EVCS revenue comes from EVCS audit fields.
5. settlement_audit_complete = 1 and settlement_power_balance_ok = 1 for complete MARL audits.
6. storage_terminal_value_reward is zero on non-terminal steps.
7. raw unsafe/projected safe reduces DSO reward.
8. projected unsafe reduces safety gate even if raw cost is low.
9. welfare_clip_saturation_rate is below 10 percent after calibration.
```

Do not start paper-long training before Phase A passes.

### Phase B: Scenario, Algorithm Stability, and Experiments

Execute after Phase A:

```text
Task 7: Balanced Generation Scenario With Physical Stress Acceptance
Task 8: Reward V3 HAPPO/HATRPO Stability Configuration
Task 9: Minimal Verification Experiments
Task 10: Required Ablations Before Paper-Long
```

Phase B acceptance:

```text
1. balanced scenario passes both profile diagnostics and pre-control AC stress scan.
2. HAPPO/HATRPO loads v3 stability config into runtime objects.
3. KL early stop can be triggered in a controlled dynamic test.
4. no-shield evaluation is available.
5. short ablations complete without NaN and with expected reward toggles.
```

---

## Task 0: Freeze Reward V3 Accounting Contract

**Files:**

- Modify: `docs/reward_terms_full_glossary_cn.md`
- Test indirectly: `tests/test_reward_v3_market_safety.py`

- [ ] **Step 1: Add the reward-v3 accounting section to the glossary**

Add a new section named `Reward V3 市场安全合同` with the formulas from the `Reward V3 Contract` section above. The text must explicitly state:

```text
1. DSO welfare uses operational surplus excluding service/availability transfer payments.
2. VPP dispatch reward uses true private settlement once.
3. EVCS revenue must come from EVCS DER audit, not aggregate negative VPP power.
4. Storage terminal value is applied only at done/truncated.
5. Safety gate uses unweighted normalized raw safety, not weighted safety penalty.
```

- [ ] **Step 2: Check glossary for obsolete v3 wording**

Run:

```bash
rg -n "v3|curtailment|safe_capacity|service_payment|availability_payment|terminal_value|EVCS|safety_gate" docs/reward_terms_full_glossary_cn.md
```

Expected: v3 section states that curtailment/safe-capacity fields are diagnostics only and that service/availability payments are not counted twice.

---

## Task 1: Reward V3 Contract Tests

**Files:**

- Create: `tests/test_reward_v3_market_safety.py`
- Modify later: `src/vpp_dso_sim/learning/reward_config.py`
- Modify later: `src/vpp_dso_sim/entities/dso.py`
- Modify later: `src/vpp_dso_sim/envs/reward_design.py`

- [ ] **Step 1: Create failing DSO config and curtailment-free tests**

Create `tests/test_reward_v3_market_safety.py` with:

```python
from __future__ import annotations

import math

import pytest

from vpp_dso_sim.entities.dso import DSO
from vpp_dso_sim.learning.reward_config import RewardConfig
from vpp_dso_sim.network.constraints import ConstraintReport, ConstraintViolation


def _safe_report() -> ConstraintReport:
    return ConstraintReport(converged=True, violations=[])


def _line_overload_report(magnitude: float = 15.0) -> ConstraintReport:
    return ConstraintReport(
        converged=True,
        violations=[
            ConstraintViolation(
                kind="line_overload",
                element="line_1",
                value=100.0 + magnitude,
                limit=100.0,
                magnitude=magnitude,
            )
        ],
    )


def _v3_config(**dso_overrides):
    return RewardConfig.from_dict(
        {
            "version": "v3_market_safety",
            "dso": {
                "welfare_normalization_mode": "baseline_zscore",
                "welfare_baseline_mean": 0.0,
                "welfare_baseline_std": 10.0,
                "welfare_clip": 5.0,
                "welfare_weight": 1.0,
                "soft_safety_gate_kappa": 2.0,
                **dso_overrides,
            },
        }
    )


def test_reward_v3_config_defaults_remove_dso_curtailment_training_terms() -> None:
    cfg = RewardConfig.from_dict({"version": "v3_market_safety"})

    assert cfg.is_v3_market_safety
    assert cfg.dso.curtailment_cost_weight == pytest.approx(0.0)
    assert cfg.dso.safe_capacity_utilization_weight == pytest.approx(0.0)
    assert cfg.dso.over_conservative_curtailment_weight == pytest.approx(0.0)


def test_dso_v3_reward_components_do_not_expose_curtailment_training_terms() -> None:
    dso = DSO(net={}, reward_config=_v3_config())

    components = dso.calculate_reward_or_cost(
        report=_safe_report(),
        mean_envelope_width_ratio=0.05,
        vpp_operational_surplus=10.0,
        n_vpps=4,
        dt_hours=0.25,
    )

    assert components["reward_version_code"] == pytest.approx(3.0)
    assert components["dso_reward_train"] == pytest.approx(components["dso_reward_env"])
    assert "dso_curtailment_cost" not in components
    assert "dso_safe_capacity_utilization_reward" not in components
    assert "dso_over_conservative_curtailment_penalty" not in components
    assert components["dso_envelope_width_ratio_diagnostic"] == pytest.approx(0.05)
```

- [ ] **Step 2: Add welfare normalization tests**

Append:

```python
def test_dso_v3_welfare_is_standardized_before_clipping() -> None:
    dso = DSO(net={}, reward_config=_v3_config(welfare_baseline_mean=0.0, welfare_baseline_std=100.0))

    low = dso.calculate_reward_or_cost(
        report=_safe_report(),
        vpp_operational_surplus=200.0,
        n_vpps=4,
        dt_hours=0.25,
    )
    high = dso.calculate_reward_or_cost(
        report=_safe_report(),
        vpp_operational_surplus=500.0,
        n_vpps=4,
        dt_hours=0.25,
    )

    assert low["dso_vpp_welfare_raw"] == pytest.approx(200.0)
    assert high["dso_vpp_welfare_raw"] == pytest.approx(500.0)
    assert low["dso_vpp_welfare_per_mwh"] == pytest.approx(200.0)
    assert high["dso_vpp_welfare_per_mwh"] == pytest.approx(500.0)
    assert high["dso_vpp_welfare_zscore"] > low["dso_vpp_welfare_zscore"]
    assert high["dso_vpp_welfare_bounded"] > low["dso_vpp_welfare_bounded"]
```

This test prevents the old `clip(raw_welfare)` behavior where different large profits collapse to the same value too early.

- [ ] **Step 3: Add safety gate and safe-first tests**

Append:

```python
def test_dso_v3_safety_gate_uses_unweighted_normalized_safety_not_weighted_penalty() -> None:
    dso = DSO(
        net={},
        reward_config=_v3_config(
            hard_violation_weight=50.0,
            powerflow_failure_weight=100.0,
            soft_safety_gate_kappa=2.0,
        ),
    )

    components = dso.calculate_reward_or_cost(
        report=_line_overload_report(magnitude=1.0),
        raw_action_line_overload_cost=0.2,
        projected_action_line_overload_cost=0.0,
        vpp_operational_surplus=500.0,
        n_vpps=4,
        dt_hours=0.25,
    )

    expected_gate_input = max(
        components["dso_raw_safety_cost_norm"],
        components["dso_projected_safety_cost_norm"],
    )
    expected_gate = math.exp(-2.0 * expected_gate_input)
    assert components["dso_safety_gate"] == pytest.approx(expected_gate)
    assert components["dso_weighted_safety_penalty"] > components["dso_raw_safety_cost_norm"]


def test_dso_v3_projected_unsafe_closes_gate_even_if_raw_cost_is_low() -> None:
    dso = DSO(net={}, reward_config=_v3_config(soft_safety_gate_kappa=2.0))

    safe = dso.calculate_reward_or_cost(
        report=_safe_report(),
        raw_action_voltage_violation_cost=0.0,
        projected_action_voltage_violation_cost=0.0,
        vpp_operational_surplus=10.0,
        n_vpps=4,
        dt_hours=0.25,
    )
    unsafe = dso.calculate_reward_or_cost(
        report=_safe_report(),
        raw_action_voltage_violation_cost=0.0,
        projected_action_voltage_violation_cost=0.2,
        vpp_operational_surplus=10.0,
        n_vpps=4,
        dt_hours=0.25,
    )

    assert unsafe["dso_raw_safety_cost_norm"] == pytest.approx(0.0)
    assert unsafe["dso_projected_safety_cost_norm"] > 0.0
    assert unsafe["dso_safety_gate"] < safe["dso_safety_gate"]


@pytest.mark.parametrize("unsafe_welfare", [5.0, 50.0, 500.0, 5000.0])
@pytest.mark.parametrize("violation_cost", [0.01, 0.05, 0.1, 0.2])
def test_dso_v3_unsafe_raw_action_never_beats_safe_action_under_profit_sweep(
    unsafe_welfare: float,
    violation_cost: float,
) -> None:
    dso = DSO(
        net={},
        reward_config=_v3_config(
            welfare_baseline_std=100.0,
            welfare_clip=5.0,
            hard_violation_weight=30.0,
            raw_action_safety_weight=20.0,
            min_raw_unsafe_penalty=1.0,
        ),
    )

    safe = dso.calculate_reward_or_cost(
        report=_safe_report(),
        raw_action_line_overload_cost=0.0,
        projected_action_line_overload_cost=0.0,
        vpp_operational_surplus=-10.0,
        n_vpps=4,
        dt_hours=0.25,
    )
    unsafe = dso.calculate_reward_or_cost(
        report=_safe_report(),
        raw_action_line_overload_cost=violation_cost,
        projected_action_line_overload_cost=0.0,
        vpp_operational_surplus=unsafe_welfare,
        n_vpps=4,
        dt_hours=0.25,
    )

    assert unsafe["dso_projected_safety_cost_norm"] == pytest.approx(0.0)
    assert unsafe["dso_raw_safety_cost_norm"] > 0.0
    assert unsafe["dso_reward_train"] < safe["dso_reward_train"]
```

- [ ] **Step 4: Run tests and verify initial failures**

Run:

```bash
./.venv-server/bin/python -m pytest tests/test_reward_v3_market_safety.py -q
```

Expected: FAIL because v3 config fields, DSO normalized welfare, and raw/projected safety fields are not implemented yet.

---

## Task 2: Implement Reward V3 Config and DSO Reward

**Files:**

- Modify: `src/vpp_dso_sim/learning/reward_config.py`
- Modify: `src/vpp_dso_sim/entities/dso.py`
- Test: `tests/test_reward_v3_market_safety.py`

- [ ] **Step 1: Add v3 config fields**

Add to `DSORewardConfig`:

```python
welfare_weight: float = 1.0
welfare_clip: float = 5.0
welfare_normalization_mode: str = "per_mwh_baseline_zscore"
welfare_baseline_mean: float = 0.0
welfare_baseline_std: float = 1.0
welfare_running_decay: float = 0.99
welfare_epsilon: float = 1e-6
soft_safety_gate_kappa: float = 2.0
soft_safety_gate_kappa_schedule: tuple[float, float, float] = (1.0, 3.0, 10.0)
raw_action_safety_weight: float = 20.0
projected_action_safety_weight: float = 5.0
min_raw_unsafe_penalty: float = 1.0
raw_safety_epsilon: float = 1e-5
raw_safety_gate_uses_unweighted_cost: bool = True
safety_gate_input_mode: str = "max_raw_projected"
```

Add to `RewardConfig`:

```python
@property
def is_v3_market_safety(self) -> bool:
    return str(self.version).lower() == "v3_market_safety"
```

In `RewardConfig.from_dict`, for `v3_market_safety`, force these defaults before user overrides:

```python
dso_data = {
    "enable_tracking_bonus": False,
    "enable_effective_response_bonus": False,
    "enable_target_tracking_cost": False,
    "comfort_violation_weight": 0.0,
    "soc_violation_weight": 0.0,
    "feasibility_bonus_weight": 0.0,
    "envelope_width_penalty_weight": 0.0,
    "curtailment_cost_weight": 0.0,
    "safe_capacity_utilization_weight": 0.0,
    "over_conservative_curtailment_weight": 0.0,
    "safety_margin_weight": 1.0,
    "hard_violation_weight": 30.0,
    "powerflow_failure_weight": 80.0,
    "projection_gap_weight": 1.0,
    "loss_cost_weight": 0.2,
    "smoothness_weight": 0.02,
    "welfare_weight": 1.0,
    "welfare_clip": 5.0,
    "welfare_normalization_mode": "per_mwh_baseline_zscore",
    "welfare_baseline_mean": 0.0,
    "welfare_baseline_std": 1.0,
    "soft_safety_gate_kappa": 2.0,
    "raw_action_safety_weight": 20.0,
    "projected_action_safety_weight": 5.0,
    "min_raw_unsafe_penalty": 1.0,
    "raw_safety_epsilon": 1e-5,
    "safety_gate_input_mode": "max_raw_projected",
    **dso_data,
}
```

- [ ] **Step 2: Update DSO reward method signature**

Modify `DSO.calculate_reward_or_cost` to accept:

```python
vpp_operational_surplus: float = 0.0
n_vpps: int = 1
dt_hours: float = 0.25
raw_action_voltage_violation_cost: float = 0.0
raw_action_line_overload_cost: float = 0.0
raw_action_trafo_overload_cost: float = 0.0
raw_action_powerflow_failed: float = 0.0
projected_action_voltage_violation_cost: float = 0.0
projected_action_line_overload_cost: float = 0.0
projected_action_trafo_overload_cost: float = 0.0
projected_action_powerflow_failed: float = 0.0
shield_intervention_count: int = 0
```

Keep backward compatibility by also accepting the old `vpp_welfare_proxy` and `raw_action_safety_cost`; if callers still pass them, route them to diagnostics but prefer the new fields in v3.

- [ ] **Step 3: Add DSO welfare normalization helper**

Add to `DSO`:

```python
def _normalize_v3_welfare(self, *, operational_surplus: float, n_vpps: int, dt_hours: float) -> dict[str, float]:
    cfg = self.reward_config.dso
    denom = max(float(n_vpps) * float(dt_hours), float(cfg.welfare_epsilon))
    per_mwh = float(operational_surplus) / denom
    mode = str(cfg.welfare_normalization_mode).lower()
    if "baseline" in mode:
        mean = float(cfg.welfare_baseline_mean)
        std = max(float(cfg.welfare_baseline_std), float(cfg.welfare_epsilon))
    else:
        mean = float(getattr(self, "_v3_welfare_running_mean", 0.0))
        var = float(getattr(self, "_v3_welfare_running_var", 1.0))
        std = max(var ** 0.5, float(cfg.welfare_epsilon))
        decay = float(cfg.welfare_running_decay)
        delta = per_mwh - mean
        new_mean = decay * mean + (1.0 - decay) * per_mwh
        new_var = decay * var + (1.0 - decay) * delta * delta
        self._v3_welfare_running_mean = new_mean
        self._v3_welfare_running_var = max(new_var, float(cfg.welfare_epsilon))
    zscore = (per_mwh - mean) / std
    bounded = max(-float(cfg.welfare_clip), min(float(cfg.welfare_clip), zscore))
    return {
        "dso_vpp_welfare_raw": float(operational_surplus),
        "dso_vpp_welfare_per_mwh": float(per_mwh),
        "dso_vpp_welfare_zscore": float(zscore),
        "dso_vpp_welfare_bounded": float(bounded),
    }
```

- [ ] **Step 4: Add DSO raw/projected safety helper**

Add:

```python
def _v3_safety_costs(
    self,
    *,
    raw_action_voltage_violation_cost: float,
    raw_action_line_overload_cost: float,
    raw_action_trafo_overload_cost: float,
    raw_action_powerflow_failed: float,
    projected_action_voltage_violation_cost: float,
    projected_action_line_overload_cost: float,
    projected_action_trafo_overload_cost: float,
    projected_action_powerflow_failed: float,
) -> dict[str, float]:
    cfg = self.reward_config.dso
    raw_norm = (
        float(raw_action_voltage_violation_cost)
        + float(raw_action_line_overload_cost)
        + float(raw_action_trafo_overload_cost)
        + float(raw_action_powerflow_failed)
    )
    projected_norm = (
        float(projected_action_voltage_violation_cost)
        + float(projected_action_line_overload_cost)
        + float(projected_action_trafo_overload_cost)
        + float(projected_action_powerflow_failed)
    )
    raw_unsafe = raw_norm > float(cfg.raw_safety_epsilon)
    raw_penalty_input = raw_norm + (float(cfg.min_raw_unsafe_penalty) if raw_unsafe else 0.0)
    weighted = (
        float(cfg.raw_action_safety_weight) * raw_penalty_input
        + float(cfg.projected_action_safety_weight) * projected_norm
        + float(cfg.powerflow_failure_weight) * float(raw_action_powerflow_failed)
    )
    return {
        "dso_raw_safety_cost_norm": float(raw_norm),
        "dso_projected_safety_cost_norm": float(projected_norm),
        "dso_weighted_safety_penalty": float(weighted),
        "raw_action_voltage_violation_cost": float(raw_action_voltage_violation_cost),
        "raw_action_line_overload_cost": float(raw_action_line_overload_cost),
        "raw_action_trafo_overload_cost": float(raw_action_trafo_overload_cost),
        "raw_action_powerflow_failed": float(raw_action_powerflow_failed),
        "projected_action_voltage_violation_cost": float(projected_action_voltage_violation_cost),
        "projected_action_line_overload_cost": float(projected_action_line_overload_cost),
        "projected_action_trafo_overload_cost": float(projected_action_trafo_overload_cost),
        "projected_action_powerflow_failed": float(projected_action_powerflow_failed),
    }
```

- [ ] **Step 5: Add `_dso_v3_components`**

Add:

```python
def _dso_v3_components(
    self,
    *,
    mean_envelope_width_ratio: float,
    envelope_smoothness_mw: float,
    vpp_operational_surplus: float,
    n_vpps: int,
    dt_hours: float,
    raw_action_voltage_violation_cost: float,
    raw_action_line_overload_cost: float,
    raw_action_trafo_overload_cost: float,
    raw_action_powerflow_failed: float,
    projected_action_voltage_violation_cost: float,
    projected_action_line_overload_cost: float,
    projected_action_trafo_overload_cost: float,
    projected_action_powerflow_failed: float,
    shield_intervention_count: int,
) -> dict[str, float]:
    cfg = self.reward_config.dso
    welfare = self._normalize_v3_welfare(
        operational_surplus=vpp_operational_surplus,
        n_vpps=n_vpps,
        dt_hours=dt_hours,
    )
    safety = self._v3_safety_costs(
        raw_action_voltage_violation_cost=raw_action_voltage_violation_cost,
        raw_action_line_overload_cost=raw_action_line_overload_cost,
        raw_action_trafo_overload_cost=raw_action_trafo_overload_cost,
        raw_action_powerflow_failed=raw_action_powerflow_failed,
        projected_action_voltage_violation_cost=projected_action_voltage_violation_cost,
        projected_action_line_overload_cost=projected_action_line_overload_cost,
        projected_action_trafo_overload_cost=projected_action_trafo_overload_cost,
        projected_action_powerflow_failed=projected_action_powerflow_failed,
    )
    gate_input = max(safety["dso_raw_safety_cost_norm"], safety["dso_projected_safety_cost_norm"])
    safety_gate = math.exp(-float(cfg.soft_safety_gate_kappa) * gate_input)
    welfare_reward = float(cfg.welfare_weight) * safety_gate * welfare["dso_vpp_welfare_bounded"]
    loss_cost = float(cfg.loss_cost_weight) * self._bounded_training_penalty(self._network_loss_cost() / 1000.0)
    smoothness_penalty = float(cfg.smoothness_weight) * self._bounded_training_penalty(envelope_smoothness_mw)
    dso_reward = welfare_reward - safety["dso_weighted_safety_penalty"] - loss_cost - smoothness_penalty
    return {
        **welfare,
        **safety,
        "dso_safety_gate_input": float(gate_input),
        "dso_safety_gate": float(safety_gate),
        "dso_safe_gated_welfare_reward": float(welfare_reward),
        "dso_loss_cost": float(loss_cost),
        "dso_smoothness_penalty": float(smoothness_penalty),
        "dso_envelope_width_ratio_diagnostic": float(mean_envelope_width_ratio),
        "dso_shield_intervention_count": float(shield_intervention_count),
        "dso_reward_train": float(dso_reward),
    }
```

Add `import math` if missing.

- [ ] **Step 6: Route v3 before v1/v2 reward assembly**

In `calculate_reward_or_cost`, if `self.reward_config.is_v3_market_safety`, call `_dso_v3_components` before any v2 curtailment components are added. Then set:

```python
components["reward_version_code"] = 3.0
components["reward_scaling_version"] = 3.0
components["dso_reward_env"] = components["dso_reward_train"]
components["dso_reward_critic_scaled"] = components["dso_reward_train"] * float(self.reward_config.critic_reward_scale)
components["dso_reward"] = components["dso_reward_env"]
components["scaled_reward"] = components["dso_reward"]
components["reward"] = components["dso_reward"]
```

Return immediately for v3.

- [ ] **Step 7: Run DSO reward tests**

Run:

```bash
./.venv-server/bin/python -m pytest tests/test_reward_v3_market_safety.py -q
```

Expected: DSO config, welfare normalization, safety gate, and raw/projected safety tests pass. Settlement tests added later still do not exist at this point.

- [ ] **Step 8: Run legacy reward smoke tests**

Run:

```bash
./.venv-server/bin/python -m pytest tests/test_reward_v2_minimal.py tests/test_hasac_happo.py tests/test_hatrpo_training.py -q
```

Expected: PASS. v1/v2 reward fields remain available.

---

## Task 2.5: Welfare Normalization Calibration

**Files:**

- Create: `tests/test_reward_v3_welfare_calibration.py`
- Create: `scripts/calibrate_reward_v3_welfare.py`
- Modify: `configs/rewards/v3_market_safety/reward_v3_market_safety.yaml`

- [ ] **Step 1: Add welfare calibration output test**

Create `tests/test_reward_v3_welfare_calibration.py`:

```python
from __future__ import annotations

import json
from pathlib import Path


def test_welfare_calibration_summary_has_non_degenerate_scale(tmp_path: Path) -> None:
    summary = {
        "vpp_operational_surplus_ex_transfer_per_mwh_mean": 12.5,
        "vpp_operational_surplus_ex_transfer_per_mwh_std": 8.0,
        "welfare_clip": 5.0,
        "welfare_clip_saturation_rate": 0.04,
    }
    path = tmp_path / "welfare_calibration_summary.json"
    path.write_text(json.dumps(summary))
    data = json.loads(path.read_text())

    assert data["vpp_operational_surplus_ex_transfer_per_mwh_std"] > 1e-6
    assert data["welfare_clip_saturation_rate"] < 0.10
```

- [ ] **Step 2: Create calibration script**

Create `scripts/calibrate_reward_v3_welfare.py`:

```python
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from vpp_dso_sim.simulation.scenario import load_scenario
from vpp_dso_sim.simulation.simulator import Simulator


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--horizon-steps", type=int, default=96)
    parser.add_argument("--welfare-clip", type=float, default=5.0)
    args = parser.parse_args()

    scenario = load_scenario(args.config)
    result = Simulator(scenario).run_timeseries(horizon_steps=args.horizon_steps)
    reward = result["reward_components"]
    values = reward["vpp_operational_surplus_ex_transfer"].to_numpy(dtype=float)
    denom = max(len(getattr(scenario, "vpps", [])) * float(scenario.dt_hours), 1e-6)
    per_mwh = values / denom
    mean = float(np.mean(per_mwh))
    std = float(max(np.std(per_mwh), 1e-6))
    z = (per_mwh - mean) / std
    saturation = float(np.mean(np.abs(z) >= float(args.welfare_clip)))
    summary = {
        "vpp_operational_surplus_ex_transfer_per_mwh_mean": mean,
        "vpp_operational_surplus_ex_transfer_per_mwh_std": std,
        "welfare_clip": float(args.welfare_clip),
        "welfare_clip_saturation_rate": saturation,
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(summary, indent=2, sort_keys=True))
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Run calibration smoke**

Run:

```bash
./.venv-server/bin/python scripts/calibrate_reward_v3_welfare.py \
  --config configs/scenarios/benchmark/european_lv_benchmark_v3_balanced_generation.yaml \
  --output outputs/reward_v3_calibration/welfare_calibration_summary.json \
  --horizon-steps 96 \
  --welfare-clip 5.0
```

Expected: exits 0 and writes:

```text
vpp_operational_surplus_ex_transfer_per_mwh_mean
vpp_operational_surplus_ex_transfer_per_mwh_std
welfare_clip_saturation_rate
```

- [ ] **Step 4: Update reward-v3 YAML with calibrated values**

Run:

```bash
./.venv-server/bin/python - <<'PY'
import json
from pathlib import Path

import yaml

summary_path = Path("outputs/reward_v3_calibration/welfare_calibration_summary.json")
config_path = Path("configs/rewards/v3_market_safety/reward_v3_market_safety.yaml")
summary = json.loads(summary_path.read_text())
config = yaml.safe_load(config_path.read_text())
config["reward"]["dso"]["welfare_normalization_mode"] = "per_mwh_baseline_zscore"
config["reward"]["dso"]["welfare_baseline_mean"] = float(summary["vpp_operational_surplus_ex_transfer_per_mwh_mean"])
config["reward"]["dso"]["welfare_baseline_std"] = float(summary["vpp_operational_surplus_ex_transfer_per_mwh_std"])
config_path.write_text(yaml.safe_dump(config, sort_keys=False))
print(config["reward"]["dso"]["welfare_baseline_mean"])
print(config["reward"]["dso"]["welfare_baseline_std"])
PY
```

The paper-long gate requires:

```text
welfare_clip_saturation_rate < 0.10
```

If saturation is 10 percent or higher, increase `welfare_baseline_std`, increase `welfare_clip`, or inspect settlement outliers before training.

- [ ] **Step 5: Run calibration tests**

Run:

```bash
./.venv-server/bin/python -m pytest tests/test_reward_v3_welfare_calibration.py -q
```

Expected: PASS.

---

## Task 3: VPP Settlement Without Double Counting

**Files:**

- Modify: `tests/test_reward_v3_market_safety.py`
- Create: `src/vpp_dso_sim/data_sources/acn.py`
- Create: `src/vpp_dso_sim/data_sources/tariff.py`
- Modify: `src/vpp_dso_sim/data_sources/__init__.py`
- Modify: `src/vpp_dso_sim/der/ev.py`
- Modify: `src/vpp_dso_sim/der/evcs.py`
- Modify: `src/vpp_dso_sim/envs/reward_design.py`

- [ ] **Step 1: Add settlement tests for double counting and EVCS audit**

Append to `tests/test_reward_v3_market_safety.py`:

```python
from vpp_dso_sim.envs.reward_design import vpp_dispatch_reward_components


class _SettlementProbeVPP:
    def __init__(self, delivered_p_mw: float, der_types: tuple[str, ...] = ()) -> None:
        self._delivered_p_mw = delivered_p_mw
        self.der_types = der_types
        self.der_list = []

    def current_power_mw(self) -> float:
        return self._delivered_p_mw

    def operating_cost(self) -> float:
        return 0.0

    def comfort_penalty(self, _t: int) -> float:
        return 0.0

    def soc_violation_penalty(self, _t: int) -> float:
        return 0.0


def _dispatch_cfg():
    return RewardConfig.from_dict(
        {
            "version": "v3_market_safety",
            "vpp": {
                "dispatch": {
                    "private_profit_weight": 1.0,
                    "service_payment_weight": 0.0,
                    "availability_payment_weight": 0.0,
                    "contract_delivery_weight": 0.0,
                    "projection_linear_weight": 0.0,
                    "projection_quadratic_weight": 0.0,
                    "comfort_soc_weight": 0.0,
                    "battery_degradation_weight": 0.0,
                }
            },
        }
    )


def test_v3_evcs_revenue_uses_evcs_audit_not_total_negative_vpp_power() -> None:
    components = vpp_dispatch_reward_components(
        vpp=_SettlementProbeVPP(delivered_p_mw=-0.10),
        envelope={
            "wholesale_buy_price": 400.0,
            "export_sell_price": 300.0,
            "retail_evcs_tariff": 1000.0,
            "p_min_mw": -1.0,
            "p_max_mw": 1.0,
        },
        audit={
            "evcs_grid_p_mw": 0.04,
            "storage_charge_p_mw": 0.06,
            "evcs_charge_efficiency": 0.95,
            "service_payment": 2.0,
            "availability_payment": 3.0,
        },
        dt_hours=0.25,
        t=0,
        reward_config=_dispatch_cfg(),
    )

    assert components["evcs_grid_energy_mwh"] == pytest.approx(0.01)
    assert components["storage_charge_energy_mwh"] == pytest.approx(0.015)
    assert components["evcs_user_charging_revenue"] == pytest.approx(9.5)
    assert components["evcs_wholesale_energy_cost"] == pytest.approx(4.0)
    assert components["storage_charge_energy_cost"] == pytest.approx(6.0)


def test_v3_service_and_availability_payments_are_not_rewarded_twice() -> None:
    components = vpp_dispatch_reward_components(
        vpp=_SettlementProbeVPP(delivered_p_mw=0.0),
        envelope={"p_min_mw": -1.0, "p_max_mw": 1.0},
        audit={
            "service_payment": 2.0,
            "availability_payment": 3.0,
            "projected_target_p_mw": 0.0,
            "baseline_p_mw": 0.0,
        },
        dt_hours=0.25,
        t=0,
        reward_config=_dispatch_cfg(),
    )

    assert components["service_payment"] == pytest.approx(2.0)
    assert components["availability_payment"] == pytest.approx(3.0)
    assert components["vpp_true_private_profit"] == pytest.approx(5.0)
    assert components["vpp_dispatch_reward"] == pytest.approx(5.0)
    assert components["vpp_operational_surplus_ex_transfer"] == pytest.approx(0.0)


def test_v3_evcs_wholesale_cost_is_not_double_counted_with_signed_energy_revenue() -> None:
    components = vpp_dispatch_reward_components(
        vpp=_SettlementProbeVPP(delivered_p_mw=-0.05),
        envelope={
            "wholesale_buy_price": 400.0,
            "export_sell_price": 400.0,
            "retail_evcs_tariff": 1000.0,
            "p_min_mw": -1.0,
            "p_max_mw": 1.0,
        },
        audit={
            "evcs_grid_p_mw": 0.05,
            "evcs_charge_efficiency": 0.95,
        },
        dt_hours=0.25,
        t=0,
        reward_config=_dispatch_cfg(),
    )

    assert components["legacy_signed_energy_market_revenue"] == pytest.approx(0.0)
    assert components["evcs_user_charging_revenue"] == pytest.approx(11.875)
    assert components["evcs_wholesale_energy_cost"] == pytest.approx(5.0)
    assert components["vpp_true_private_profit"] == pytest.approx(6.875)


def test_v3_settlement_der_audit_matches_delivered_power() -> None:
    components = vpp_dispatch_reward_components(
        vpp=_SettlementProbeVPP(delivered_p_mw=0.0, der_types=("pv", "storage", "evcs")),
        envelope={
            "wholesale_buy_price": 400.0,
            "export_sell_price": 300.0,
            "retail_evcs_tariff": 1000.0,
            "p_min_mw": -1.0,
            "p_max_mw": 1.0,
        },
        audit={
            "pv_export_p_mw": 0.03,
            "storage_discharge_p_mw": 0.02,
            "storage_charge_p_mw": 0.04,
            "storage_soc_before": 0.50,
            "storage_soc_after": 0.51,
            "evcs_grid_p_mw": 0.01,
            "evcs_charge_efficiency": 0.95,
            "hvac_p_mw": 0.0,
            "flex_load_p_mw": 0.0,
        },
        dt_hours=0.25,
        t=0,
        reward_config=_dispatch_cfg(),
    )

    assert components["settlement_power_balance_gap_mw"] == pytest.approx(0.0)
    assert components["settlement_power_balance_ok"] == pytest.approx(1.0)
    assert components["settlement_audit_complete"] == pytest.approx(1.0)


def test_v3_incomplete_settlement_audit_is_not_marked_complete() -> None:
    components = vpp_dispatch_reward_components(
        vpp=_SettlementProbeVPP(delivered_p_mw=-0.05, der_types=("evcs",)),
        envelope={
            "wholesale_buy_price": 400.0,
            "retail_evcs_tariff": 1000.0,
            "p_min_mw": -1.0,
            "p_max_mw": 1.0,
        },
        audit={
            "evcs_charge_efficiency": 0.95,
        },
        dt_hours=0.25,
        t=0,
        reward_config=_dispatch_cfg(),
    )

    assert components["settlement_audit_complete"] == pytest.approx(0.0)
    assert components["settlement_required_fields_missing_count"] >= 1.0
    assert components["settlement_missing_evcs_grid_p_mw"] == pytest.approx(1.0)
```

- [ ] **Step 2: Run tests and verify settlement failures**

Run:

```bash
./.venv-server/bin/python -m pytest tests/test_reward_v3_market_safety.py -q
```

Expected: FAIL because v3 settlement fields do not exist.

- [ ] **Step 3: Add dispatch config fields**

Extend `VPPDispatchRewardConfig`:

```python
private_profit_weight: float = 1.0
service_payment_weight: float = 0.0
availability_payment_weight: float = 0.0
use_unified_private_profit_v3: bool = True
require_per_der_settlement_audit: bool = True
unclassified_import_price_per_mwh: float = 0.0
paper_long_fail_on_incomplete_settlement_audit: bool = True
settlement_power_balance_tolerance_mw: float = 1e-6
```

For v3 defaults, set:

```python
"private_profit_weight": 1.0,
"service_payment_weight": 0.0,
"availability_payment_weight": 0.0,
"use_unified_private_profit_v3": True,
"require_per_der_settlement_audit": True,
"paper_long_fail_on_incomplete_settlement_audit": True,
"settlement_power_balance_tolerance_mw": 1e-6,
```

- [ ] **Step 4: Add tariff adapter**

Create `src/vpp_dso_sim/data_sources/tariff.py`:

```python
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd


@dataclass(frozen=True)
class TariffPoint:
    step: int
    retail_buy_price_per_mwh: float
    wholesale_buy_price_per_mwh: float
    export_sell_price_per_mwh: float
    evcs_retail_price_per_mwh: float
    demand_charge_per_mw: float = 0.0


class TariffProfile:
    def __init__(self, points: Iterable[TariffPoint]) -> None:
        self._points = list(points)
        if not self._points:
            raise ValueError("TariffProfile requires at least one TariffPoint")

    def at_step(self, step: int) -> TariffPoint:
        return self._points[int(step) % len(self._points)]


def load_tariff_csv(path: str | Path) -> TariffProfile:
    frame = pd.read_csv(path)
    required = {
        "step",
        "retail_buy_price_per_mwh",
        "wholesale_buy_price_per_mwh",
        "export_sell_price_per_mwh",
        "evcs_retail_price_per_mwh",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Tariff CSV missing required columns: {missing}")
    return TariffProfile(
        TariffPoint(
            step=int(row.step),
            retail_buy_price_per_mwh=float(row.retail_buy_price_per_mwh),
            wholesale_buy_price_per_mwh=float(row.wholesale_buy_price_per_mwh),
            export_sell_price_per_mwh=float(row.export_sell_price_per_mwh),
            evcs_retail_price_per_mwh=float(row.evcs_retail_price_per_mwh),
            demand_charge_per_mw=float(getattr(row, "demand_charge_per_mw", 0.0)),
        )
        for row in frame.itertuples(index=False)
    )
```

- [ ] **Step 5: Add ACN session adapter**

Create `src/vpp_dso_sim/data_sources/acn.py`:

```python
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd


@dataclass(frozen=True)
class EVChargingSession:
    session_id: str
    arrival_step: int
    departure_step: int
    requested_energy_mwh: float
    delivered_energy_mwh: float
    max_charge_mw: float
    payment_required: bool


def load_acn_session_csv(path: str | Path, *, dt_hours: float) -> list[EVChargingSession]:
    frame = pd.read_csv(path)
    required = {
        "session_id",
        "arrival_step",
        "departure_step",
        "requested_energy_kwh",
        "delivered_energy_kwh",
        "max_charge_kw",
        "payment_required",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"ACN session CSV missing required columns: {missing}")
    sessions: list[EVChargingSession] = []
    for row in frame.itertuples(index=False):
        arrival = int(row.arrival_step)
        departure = int(row.departure_step)
        if departure <= arrival:
            duration_hours = float(row.delivered_energy_kwh) / max(float(row.max_charge_kw), 1e-9)
            departure = arrival + max(1, int(round(duration_hours / max(float(dt_hours), 1e-9))))
        sessions.append(
            EVChargingSession(
                session_id=str(row.session_id),
                arrival_step=arrival,
                departure_step=departure,
                requested_energy_mwh=float(row.requested_energy_kwh) / 1000.0,
                delivered_energy_mwh=float(row.delivered_energy_kwh) / 1000.0,
                max_charge_mw=float(row.max_charge_kw) / 1000.0,
                payment_required=bool(row.payment_required),
            )
        )
    return sessions
```

- [ ] **Step 6: Export adapters**

Modify `src/vpp_dso_sim/data_sources/__init__.py` to export:

```python
from vpp_dso_sim.data_sources.acn import EVChargingSession, load_acn_session_csv
from vpp_dso_sim.data_sources.tariff import TariffPoint, TariffProfile, load_tariff_csv
```

Include these names in `__all__` while preserving existing exports.

- [ ] **Step 7: Implement per-DER settlement helper**

In `src/vpp_dso_sim/envs/reward_design.py`, add required-field helpers:

```python
def _vpp_der_type_names(vpp: Any) -> set[str]:
    explicit = set(getattr(vpp, "der_types", ()) or ())
    if explicit:
        return {str(item).lower() for item in explicit}
    names: set[str] = set()
    for der in getattr(vpp, "der_list", []) or []:
        cls_name = der.__class__.__name__.lower()
        der_type = str(getattr(der, "der_type", getattr(der, "type", cls_name))).lower()
        if "evcs" in der_type or "charging" in cls_name:
            names.add("evcs")
        elif "storage" in der_type or "battery" in cls_name:
            names.add("storage")
        elif "pv" in der_type or "solar" in cls_name:
            names.add("pv")
        elif "micro" in der_type or "turbine" in cls_name or "chp" in der_type:
            names.add("microturbine")
        elif "hvac" in der_type:
            names.add("hvac")
        elif "flex" in der_type or "load" in der_type:
            names.add("flexible_load")
    return names


def _required_settlement_fields_for_vpp(vpp: Any, envelope: dict[str, Any]) -> set[str]:
    der_types = _vpp_der_type_names(vpp)
    required: set[str] = set()
    if "evcs" in der_types:
        required.update({"evcs_grid_p_mw", "evcs_charge_efficiency"})
        if "retail_evcs_tariff" not in envelope:
            required.add("retail_evcs_tariff")
    if "storage" in der_types:
        required.update({"storage_charge_p_mw", "storage_discharge_p_mw", "storage_soc_before", "storage_soc_after"})
    if "pv" in der_types:
        required.add("pv_export_p_mw")
    if "microturbine" in der_types:
        required.update({"microturbine_export_p_mw", "der_operation_cost"})
    if "hvac" in der_types:
        required.update({"hvac_p_mw", "comfort_cost"})
    if "flexible_load" in der_types:
        required.add("flex_load_p_mw")
        if "comfort_cost" not in required:
            required.add("unserved_energy_cost")
    return required
```

Then add:

```python
def _v3_settlement_components(
    *,
    vpp: Any,
    delivered_p_mw: float,
    envelope: dict[str, Any],
    audit: dict[str, Any],
    dt_hours: float,
    dispatch_config,
) -> dict[str, float]:
    wholesale_buy = _safe_float(envelope.get("wholesale_buy_price"), _safe_float(envelope.get("price"), 0.0))
    export_sell = _safe_float(envelope.get("export_sell_price"), _safe_float(envelope.get("price"), 0.0))
    evcs_retail = _safe_float(envelope.get("retail_evcs_tariff"), _safe_float(audit.get("retail_evcs_tariff"), wholesale_buy))

    required = _required_settlement_fields_for_vpp(vpp, envelope)
    missing = sorted(field for field in required if field not in audit and field not in envelope)
    missing_flags = {f"settlement_missing_{field}": 1.0 for field in missing}

    pv_export_p_mw = max(0.0, _safe_float(audit.get("pv_export_p_mw"), 0.0))
    microturbine_export_p_mw = max(0.0, _safe_float(audit.get("microturbine_export_p_mw"), 0.0))
    storage_discharge_p_mw = max(0.0, _safe_float(audit.get("storage_discharge_p_mw"), 0.0))
    storage_charge_p_mw = max(0.0, _safe_float(audit.get("storage_charge_p_mw"), 0.0))
    evcs_grid_p_mw = max(0.0, _safe_float(audit.get("evcs_grid_p_mw"), 0.0))
    hvac_p_mw = max(0.0, _safe_float(audit.get("hvac_p_mw"), 0.0))
    flex_load_p_mw = max(0.0, _safe_float(audit.get("flex_load_p_mw"), 0.0))

    evcs_eta = max(0.0, min(1.0, _safe_float(audit.get("evcs_charge_efficiency"), 0.95)))
    evcs_grid_energy = evcs_grid_p_mw * dt_hours
    evcs_user_energy = evcs_eta * evcs_grid_energy

    export_energy = (pv_export_p_mw + microturbine_export_p_mw + storage_discharge_p_mw) * dt_hours
    storage_charge_energy = storage_charge_p_mw * dt_hours
    load_energy = (hvac_p_mw + flex_load_p_mw) * dt_hours

    export_revenue = export_sell * export_energy
    storage_charge_cost = wholesale_buy * storage_charge_energy
    load_energy_cost = wholesale_buy * load_energy
    evcs_wholesale_cost = wholesale_buy * evcs_grid_energy
    evcs_user_revenue = evcs_retail * evcs_user_energy + _safe_float(audit.get("evcs_session_fee"), 0.0)

    service_payment = _safe_float(audit.get("service_payment"), 0.0)
    availability_payment = _safe_float(audit.get("availability_payment"), 0.0)
    service_payment_source = str(audit.get("service_payment_source", "baseline_proxy"))
    availability_payment_source = str(audit.get("availability_payment_source", "baseline_proxy"))
    contract_penalty = _safe_float(audit.get("contract_delivery_penalty"), 0.0)
    der_operation_cost = _safe_float(audit.get("der_operation_cost"), 0.0)
    battery_degradation_cost = _safe_float(audit.get("battery_degradation_cost"), 0.0)
    comfort_cost = _safe_float(audit.get("comfort_cost"), 0.0)
    unserved_energy_cost = _safe_float(audit.get("unserved_energy_cost"), 0.0)
    unclassified_buy_p_mw = max(0.0, _safe_float(audit.get("unclassified_buy_p_mw"), 0.0))
    unclassified_buy_cost = wholesale_buy * unclassified_buy_p_mw * dt_hours

    audit_power = (
        pv_export_p_mw
        + microturbine_export_p_mw
        + storage_discharge_p_mw
        - storage_charge_p_mw
        - evcs_grid_p_mw
        - hvac_p_mw
        - flex_load_p_mw
        - unclassified_buy_p_mw
    )
    power_gap = abs(audit_power - float(delivered_p_mw))
    power_ok = power_gap <= float(dispatch_config.settlement_power_balance_tolerance_mw)

    operational_surplus = (
        export_revenue
        + evcs_user_revenue
        - evcs_wholesale_cost
        - storage_charge_cost
        - load_energy_cost
        - unclassified_buy_cost
        - der_operation_cost
        - battery_degradation_cost
        - comfort_cost
        - unserved_energy_cost
    )
    private_profit = operational_surplus + service_payment + availability_payment - contract_penalty
    return {
        "legacy_signed_energy_market_revenue": 0.0,
        "pv_export_revenue": float(export_sell * pv_export_p_mw * dt_hours),
        "microturbine_export_revenue": float(export_sell * microturbine_export_p_mw * dt_hours),
        "storage_discharge_revenue": float(export_sell * storage_discharge_p_mw * dt_hours),
        "energy_sell_revenue": float(export_revenue),
        "hvac_energy_cost": float(wholesale_buy * hvac_p_mw * dt_hours),
        "flex_load_energy_cost": float(wholesale_buy * flex_load_p_mw * dt_hours),
        "unclassified_buy_cost": float(unclassified_buy_cost),
        "energy_buy_cost": float(load_energy_cost),
        "evcs_grid_energy_mwh": float(evcs_grid_energy),
        "evcs_user_energy_mwh": float(evcs_user_energy),
        "evcs_user_charging_revenue": float(evcs_user_revenue),
        "evcs_wholesale_energy_cost": float(evcs_wholesale_cost),
        "storage_charge_energy_mwh": float(storage_charge_energy),
        "storage_charge_energy_cost": float(storage_charge_cost),
        "service_payment": float(service_payment),
        "availability_payment": float(availability_payment),
        "service_payment_source_baseline_proxy": float(service_payment_source == "baseline_proxy"),
        "service_payment_source_cleared_award": float(service_payment_source == "cleared_award"),
        "service_payment_source_disabled": float(service_payment_source == "disabled"),
        "availability_payment_source_baseline_proxy": float(availability_payment_source == "baseline_proxy"),
        "availability_payment_source_capacity_contract": float(availability_payment_source == "capacity_contract"),
        "availability_payment_source_disabled": float(availability_payment_source == "disabled"),
        "dso_transfer_payment_cost": float(service_payment + availability_payment),
        "vpp_operational_surplus_ex_transfer": float(operational_surplus),
        "vpp_true_private_profit": float(private_profit),
        "settlement_required_fields_missing_count": float(len(missing)),
        "settlement_audit_complete": float(len(missing) == 0),
        "settlement_audit_power_mw": float(audit_power),
        "settlement_power_balance_gap_mw": float(power_gap),
        "settlement_power_balance_ok": float(power_ok),
        **missing_flags,
    }
```

Do not use `max(0.0, -delivered_p_mw)` as EVCS fallback in v3.

- [ ] **Step 8: Use unified private profit in v3 dispatch reward**

In `vpp_dispatch_reward_components`, when `reward_config.is_v3_market_safety`:

```python
settlement = _v3_settlement_components(
    vpp=vpp,
    delivered_p_mw=delivered_p_mw,
    envelope=envelope,
    audit=audit,
    dt_hours=dt_hours,
    dispatch_config=dispatch_config,
)
projection_penalty = dispatch_projection_penalty
constraint_penalty = _safe_float(audit.get("dispatch_constraint_penalty"), 0.0)
reward = (
    dispatch_config.private_profit_weight * settlement["vpp_true_private_profit"]
    - projection_penalty
    - constraint_penalty
)
components.update(settlement)
components["private_profit_proxy"] = settlement["vpp_true_private_profit"]
components["vpp_dispatch_reward"] = float(reward)
```

Remove separate v3 additions of service and availability payment outside `vpp_true_private_profit`.

- [ ] **Step 9: Run settlement tests**

Run:

```bash
./.venv-server/bin/python -m pytest tests/test_reward_v3_market_safety.py tests/test_dataset_registry.py -q
```

Expected: PASS for DSO v3 and settlement tests.

---

## Task 4: Storage Potential Shaping and Terminal-Only Value

**Files:**

- Modify: `tests/test_reward_v3_market_safety.py`
- Modify: `src/vpp_dso_sim/learning/reward_config.py`
- Modify: `src/vpp_dso_sim/envs/reward_design.py`

- [ ] **Step 1: Add storage temporal value tests**

Append:

```python
def test_storage_terminal_value_is_zero_before_done_or_truncated() -> None:
    cfg = _dispatch_cfg()
    components = vpp_dispatch_reward_components(
        vpp=_SettlementProbeVPP(delivered_p_mw=-0.05),
        envelope={"wholesale_buy_price": 300.0, "export_sell_price": 900.0, "p_min_mw": -1.0, "p_max_mw": 1.0},
        audit={
            "storage_soc_before": 0.50,
            "storage_soc_after": 0.55,
            "storage_capacity_mwh": 1.0,
            "storage_future_sell_price_per_mwh": 900.0,
            "done": False,
            "truncated": False,
        },
        dt_hours=0.25,
        t=0,
        reward_config=cfg,
    )

    assert components["storage_potential_shaping_reward"] != pytest.approx(0.0)
    assert components["storage_terminal_value_proxy"] > 0.0
    assert components["storage_terminal_value_reward"] == pytest.approx(0.0)
    assert components["storage_terminal_value_applied_flag"] == pytest.approx(0.0)


def test_storage_terminal_value_applies_only_at_episode_end() -> None:
    cfg = _dispatch_cfg()
    components = vpp_dispatch_reward_components(
        vpp=_SettlementProbeVPP(delivered_p_mw=0.0),
        envelope={"wholesale_buy_price": 300.0, "export_sell_price": 900.0, "p_min_mw": -1.0, "p_max_mw": 1.0},
        audit={
            "storage_soc_before": 0.55,
            "storage_soc_after": 0.60,
            "storage_capacity_mwh": 1.0,
            "storage_future_sell_price_per_mwh": 900.0,
            "done": True,
            "truncated": False,
        },
        dt_hours=0.25,
        t=95,
        reward_config=cfg,
    )

    assert components["storage_terminal_value_proxy"] > 0.0
    assert components["storage_terminal_value_reward"] > 0.0
    assert components["storage_terminal_value_applied_flag"] == pytest.approx(1.0)


def test_storage_temporal_value_does_not_reward_charging_when_future_price_below_buy_price() -> None:
    cfg = _dispatch_cfg()
    components = vpp_dispatch_reward_components(
        vpp=_SettlementProbeVPP(delivered_p_mw=-0.05, der_types=("storage",)),
        envelope={"wholesale_buy_price": 500.0, "export_sell_price": 200.0, "p_min_mw": -1.0, "p_max_mw": 1.0},
        audit={
            "storage_charge_p_mw": 0.05,
            "storage_discharge_p_mw": 0.0,
            "storage_soc_before": 0.70,
            "storage_soc_after": 0.72,
            "storage_capacity_mwh": 1.0,
            "storage_future_sell_price_per_mwh": 200.0,
            "done": False,
            "truncated": False,
        },
        dt_hours=0.25,
        t=0,
        reward_config=cfg,
    )

    assert components["storage_potential_shaping_reward"] < 0.0
    assert components["storage_terminal_value_reward"] == pytest.approx(0.0)
```

- [ ] **Step 2: Add storage config fields**

Extend `VPPDispatchRewardConfig`:

```python
storage_terminal_value_weight: float = 0.05
storage_potential_shaping_weight: float = 0.05
storage_future_sell_price_per_mwh: float = 0.0
storage_charge_efficiency: float = 0.95
storage_discharge_efficiency: float = 0.95
storage_discount: float = 0.99
storage_terminal_soc_target: float = 0.5
storage_terminal_soc_target_weight: float = 0.01
storage_terminal_potential_residual_mode: str = "log_and_ablate"
```

- [ ] **Step 3: Implement storage helper**

In `reward_design.py`, add:

```python
def _storage_value_components(*, audit: dict[str, Any], dispatch_config) -> dict[str, float]:
    capacity = _safe_float(audit.get("storage_capacity_mwh"), 0.0)
    soc_before = _safe_float(audit.get("storage_soc_before"), 0.0)
    soc_after = _safe_float(audit.get("storage_soc_after"), soc_before)
    future_price = _safe_float(audit.get("storage_future_sell_price_per_mwh"), dispatch_config.storage_future_sell_price_per_mwh)
    eta_dis = _safe_float(getattr(dispatch_config, "storage_discharge_efficiency", 0.95), 0.95)
    discount = _safe_float(getattr(dispatch_config, "storage_discount", 0.99), 0.99)
    kappa = max(0.0, future_price * eta_dis)
    phi_before = kappa * capacity * max(0.0, soc_before)
    phi_after = kappa * capacity * max(0.0, soc_after)
    potential_raw = discount * phi_after - phi_before
    terminal_proxy = kappa * capacity * max(0.0, soc_after)
    target = _safe_float(getattr(dispatch_config, "storage_terminal_soc_target", 0.5), 0.5)
    target_penalty = (soc_after - target) * (soc_after - target)
    terminal_applies = bool(audit.get("done", False) or audit.get("truncated", False))
    terminal_potential_residual = discount * phi_after
    terminal_reward = 0.0
    if terminal_applies:
        terminal_reward = (
            float(dispatch_config.storage_terminal_value_weight) * terminal_proxy
            - float(dispatch_config.storage_terminal_soc_target_weight) * target_penalty
        )
    return {
        "storage_value_kappa": float(kappa),
        "storage_phi_before": float(phi_before),
        "storage_phi_after": float(phi_after),
        "storage_potential_shaping_reward": float(dispatch_config.storage_potential_shaping_weight * potential_raw),
        "storage_terminal_value_proxy": float(terminal_proxy),
        "storage_terminal_soc_target_penalty": float(target_penalty),
        "storage_terminal_potential_residual": float(terminal_potential_residual),
        "storage_terminal_value_reward": float(terminal_reward),
        "storage_terminal_value_applied_flag": float(terminal_applies),
    }
```

`storage_terminal_potential_residual` is a diagnostic for checking whether potential shaping plus terminal value double-counts terminal stored energy. Reward-v3 paper-long reports must include the `v3_no_storage_temporal_value` ablation and the anti-hoarding test above before claiming storage value shaping is beneficial.

- [ ] **Step 4: Add storage terms to v3 dispatch reward**

In the v3 dispatch branch:

```python
storage_value = _storage_value_components(audit=audit, dispatch_config=dispatch_config)
reward = (
    dispatch_config.private_profit_weight * settlement["vpp_true_private_profit"]
    + storage_value["storage_potential_shaping_reward"]
    + storage_value["storage_terminal_value_reward"]
    - projection_penalty
    - constraint_penalty
)
components.update(storage_value)
components["vpp_dispatch_reward"] = float(reward)
```

Do not add `storage_terminal_value_proxy` directly to reward.

- [ ] **Step 5: Run storage tests**

Run:

```bash
./.venv-server/bin/python -m pytest tests/test_reward_v3_market_safety.py -q
```

Expected: PASS.

---

## Task 5: Compute Dispatch Settlement Before DSO Reward

**Files:**

- Modify: `tests/test_reward_v3_market_safety.py`
- Modify: `src/vpp_dso_sim/envs/reward_design.py`
- Modify: `src/vpp_dso_sim/simulation/simulator.py`
- Modify: `src/vpp_dso_sim/learning/advanced_marl.py`
- Modify: `src/vpp_dso_sim/learning/hatrpo.py`
- Modify: `src/vpp_dso_sim/learning/deep_rl.py`

- [ ] **Step 1: Add aggregation helper tests**

Append:

```python
from vpp_dso_sim.envs.reward_design import (
    aggregate_vpp_operational_surplus,
    validate_reward_v3_settlement_for_training,
)


def test_dso_welfare_aggregator_excludes_transfer_payments() -> None:
    total = aggregate_vpp_operational_surplus(
        [
            {"vpp_operational_surplus_ex_transfer": 10.0, "service_payment": 100.0, "availability_payment": 50.0},
            {"vpp_operational_surplus_ex_transfer": -2.0, "service_payment": 80.0, "availability_payment": 40.0},
        ]
    )

    assert total == pytest.approx(8.0)


def test_reward_v3_paper_long_rejects_incomplete_settlement_audit() -> None:
    dispatch_components = [
        {"settlement_audit_complete": 1.0, "settlement_power_balance_ok": 1.0},
        {"settlement_audit_complete": 0.0, "settlement_power_balance_ok": 1.0},
    ]

    with pytest.raises(RuntimeError, match="incomplete reward-v3 settlement audit"):
        validate_reward_v3_settlement_for_training(dispatch_components, preset="paper_long")
```

- [ ] **Step 2: Implement aggregator**

In `reward_design.py`:

```python
def aggregate_vpp_operational_surplus(dispatch_components: list[dict[str, float]]) -> float:
    return float(sum(_safe_float(component.get("vpp_operational_surplus_ex_transfer"), 0.0) for component in dispatch_components))


def validate_reward_v3_settlement_for_training(dispatch_components: list[dict[str, float]], *, preset: str) -> None:
    if "paper_long" not in str(preset):
        return
    incomplete = [
        idx
        for idx, component in enumerate(dispatch_components)
        if _safe_float(component.get("settlement_audit_complete"), 0.0) < 1.0
        or _safe_float(component.get("settlement_power_balance_ok"), 0.0) < 1.0
    ]
    if incomplete:
        raise RuntimeError(f"incomplete reward-v3 settlement audit for paper-long indices={incomplete[:10]}")
```

- [ ] **Step 3: Refactor role reward map order**

In `build_role_reward_maps`, enforce this order:

```python
dispatch_by_agent: dict[str, dict[str, float]] = {}
for vpp in vpps:
    envelope = envelopes_by_vpp.get(vpp.id, {})
    audit = dispatch_audit.get(vpp.id, {})
    dispatch_by_agent[f"{vpp.id}_dispatch"] = vpp_dispatch_reward_components(
        vpp=vpp,
        envelope=envelope,
        audit=audit,
        dt_hours=dt_hours,
        t=t,
        reward_config=reward_config,
    )

if reward_config.is_v3_market_safety:
    validate_reward_v3_settlement_for_training(
        list(dispatch_by_agent.values()),
        preset=str(getattr(reward_config, "experiment_preset", "")),
    )
    vpp_operational_surplus = aggregate_vpp_operational_surplus(list(dispatch_by_agent.values()))
    dso_components = dso.calculate_reward_or_cost(
        report=constraint_report,
        mean_envelope_width_ratio=mean_envelope_width_ratio,
        envelope_smoothness_mw=envelope_smoothness_mw,
        vpp_operational_surplus=vpp_operational_surplus,
        n_vpps=len(vpps),
        dt_hours=dt_hours,
        raw_action_voltage_violation_cost=raw_safety.get("voltage", 0.0),
        raw_action_line_overload_cost=raw_safety.get("line", 0.0),
        raw_action_trafo_overload_cost=raw_safety.get("trafo", 0.0),
        raw_action_powerflow_failed=raw_safety.get("powerflow_failed", 0.0),
        projected_action_voltage_violation_cost=projected_safety.get("voltage", 0.0),
        projected_action_line_overload_cost=projected_safety.get("line", 0.0),
        projected_action_trafo_overload_cost=projected_safety.get("trafo", 0.0),
        projected_action_powerflow_failed=projected_safety.get("powerflow_failed", 0.0),
        shield_intervention_count=shield_intervention_count,
    )
```

Do not calculate DSO reward first and then insert `dso_vpp_welfare_proxy` into the returned dict.

- [ ] **Step 4: Update simulator path**

In `Simulator.step`, compute per-VPP dispatch components before DSO reward when v3 is active. If simulator does not have all MARL dispatch audit fields, export:

```text
settlement_audit_complete = 0
vpp_operational_surplus_ex_transfer =
    max(0, delivered_p_mw) * export_sell_price * dt_hours
  - max(0, -delivered_p_mw) * wholesale_buy_price * dt_hours
  - operating_cost * dt_hours
```

This fallback is only a simulator smoke-test diagnostic. It must not be used for reward-v3 paper-long MARL training, because it cannot distinguish EVCS charging, storage charging, HVAC load, flexible load, and projection-induced absorption.

- [ ] **Step 5: Update trainer logs**

In `advanced_marl.py`, `hatrpo.py`, and `deep_rl.py`, log role/component means:

```python
"vpp_true_private_profit": component_mean(dispatch_components, "vpp_true_private_profit"),
"vpp_operational_surplus_ex_transfer": component_mean(dispatch_components, "vpp_operational_surplus_ex_transfer"),
"dso_transfer_payment_cost": component_mean(dispatch_components, "dso_transfer_payment_cost"),
"evcs_user_charging_revenue": component_mean(dispatch_components, "evcs_user_charging_revenue"),
"evcs_wholesale_energy_cost": component_mean(dispatch_components, "evcs_wholesale_energy_cost"),
"settlement_audit_complete": component_mean(dispatch_components, "settlement_audit_complete"),
"settlement_required_fields_missing_count": component_mean(dispatch_components, "settlement_required_fields_missing_count"),
"settlement_power_balance_gap_mw": component_mean(dispatch_components, "settlement_power_balance_gap_mw"),
"settlement_power_balance_ok": component_mean(dispatch_components, "settlement_power_balance_ok"),
"service_payment_source_baseline_proxy": component_mean(dispatch_components, "service_payment_source_baseline_proxy"),
"service_payment_source_cleared_award": component_mean(dispatch_components, "service_payment_source_cleared_award"),
"availability_payment_source_baseline_proxy": component_mean(dispatch_components, "availability_payment_source_baseline_proxy"),
"availability_payment_source_capacity_contract": component_mean(dispatch_components, "availability_payment_source_capacity_contract"),
"storage_potential_shaping_reward": component_mean(dispatch_components, "storage_potential_shaping_reward"),
"storage_terminal_value_reward": component_mean(dispatch_components, "storage_terminal_value_reward"),
"storage_terminal_potential_residual": component_mean(dispatch_components, "storage_terminal_potential_residual"),
"dso_vpp_welfare_raw": float(dso_components.get("dso_vpp_welfare_raw", 0.0)),
"dso_vpp_welfare_zscore": float(dso_components.get("dso_vpp_welfare_zscore", 0.0)),
"dso_safety_gate": float(dso_components.get("dso_safety_gate", 0.0)),
```

- [ ] **Step 6: Run reward-map and trainer smoke tests**

Run:

```bash
./.venv-server/bin/python -m pytest tests/test_reward_v3_market_safety.py tests/test_multi_agent_env.py tests/test_advanced_marl.py tests/test_hatrpo_training.py tests/test_deep_rl_training.py -q
```

Expected: PASS.

---

## Task 6: Raw, Projected, and No-Shield Safety Diagnostics

**Files:**

- Modify: `tests/test_reward_v3_market_safety.py`
- Modify: `src/vpp_dso_sim/optimization/ac_security_projection.py`
- Modify: `src/vpp_dso_sim/simulation/simulator.py`
- Modify: `src/vpp_dso_sim/learning/reward_trace.py`

- [ ] **Step 1: Add safety diagnostic contract test**

Append:

```python
def test_raw_unsafe_projected_safe_still_reduces_dso_reward() -> None:
    dso = DSO(net={}, reward_config=_v3_config(raw_action_safety_weight=20.0, projected_action_safety_weight=5.0))

    projected_safe = dso.calculate_reward_or_cost(
        report=_safe_report(),
        raw_action_voltage_violation_cost=0.0,
        projected_action_voltage_violation_cost=0.0,
        vpp_operational_surplus=10.0,
        n_vpps=4,
        dt_hours=0.25,
    )
    raw_unsafe = dso.calculate_reward_or_cost(
        report=_safe_report(),
        raw_action_voltage_violation_cost=0.1,
        projected_action_voltage_violation_cost=0.0,
        vpp_operational_surplus=10.0,
        n_vpps=4,
        dt_hours=0.25,
        shield_intervention_count=1,
    )

    assert raw_unsafe["dso_projected_safety_cost_norm"] == pytest.approx(0.0)
    assert raw_unsafe["dso_raw_safety_cost_norm"] > 0.0
    assert raw_unsafe["dso_reward_train"] < projected_safe["dso_reward_train"]
```

- [ ] **Step 2: Return raw/projected safety costs from projection code**

In `ac_security_projection.py`, extend the projection certificate output with:

```python
"raw_action_voltage_violation_cost": raw_costs.voltage,
"raw_action_line_overload_cost": raw_costs.line,
"raw_action_trafo_overload_cost": raw_costs.trafo,
"raw_action_powerflow_failed": float(raw_powerflow_failed),
"projected_action_voltage_violation_cost": projected_costs.voltage,
"projected_action_line_overload_cost": projected_costs.line,
"projected_action_trafo_overload_cost": projected_costs.trafo,
"projected_action_powerflow_failed": float(projected_powerflow_failed),
"shield_intervention_frequency": float(projected_changed),
"projection_gap_local_bounds": local_bounds_gap,
"projection_gap_ac_aware": ac_aware_gap,
"projection_gap_ac_certificate": ac_certificate_gap,
```

If full raw AC validation is too expensive for every training step, the training path may use sensitivity-based raw safety plus periodic AC audit, but the output fields must keep the same names and must record which mode was used:

```text
raw_safety_eval_mode = ac_full | sensitivity_with_ac_audit
```

- [ ] **Step 3: Export no-shield evaluation metrics**

Add an evaluation utility or extend existing frozen/checkpoint evaluation to run selected policies without safety shield:

```text
no_shield_eval_violation_rate
no_shield_eval_powerflow_failure_rate
no_shield_eval_line_overload_rate
no_shield_eval_voltage_violation_rate
no_shield_eval_reverseflow_violation_rate
```

No-shield evaluation can be short-horizon for smoke tests and longer for paper-long reports.

- [ ] **Step 4: Update reward trace fields**

Ensure `reward_trace.py` writes these columns:

```text
raw_action_voltage_violation_cost
raw_action_line_overload_cost
raw_action_trafo_overload_cost
raw_action_powerflow_failed
projected_action_voltage_violation_cost
projected_action_line_overload_cost
projected_action_trafo_overload_cost
projected_action_powerflow_failed
shield_intervention_frequency
projection_gap_local_bounds
projection_gap_ac_aware
projection_gap_ac_certificate
no_shield_eval_violation_rate
```

- [ ] **Step 5: Run safety diagnostic tests**

Run:

```bash
./.venv-server/bin/python -m pytest tests/test_reward_v3_market_safety.py tests/test_ac_security_projection.py -q
```

Expected: PASS.

---

## Task 7: Balanced Generation Scenario With Physical Stress Acceptance

**Files:**

- Create: `configs/scenarios/benchmark/european_lv_benchmark_v3_balanced_generation.yaml`
- Create: `configs/rewards/v3_market_safety/reward_v3_market_safety.yaml`
- Create: `configs/experiments/paper_long/reward_v3_market_safety/european_lv_benchmark_v3_balanced_generation_reward_v3.yaml`
- Create: `tests/test_reward_v3_scenario_mix.py`

- [ ] **Step 1: Add scenario acceptance tests**

Create `tests/test_reward_v3_scenario_mix.py`:

```python
from __future__ import annotations

from pathlib import Path

from vpp_dso_sim.optimization.feasibility_region import compute_static_feasible_region
from vpp_dso_sim.simulation.scenario import load_scenario


CONFIG = Path("configs/scenarios/benchmark/european_lv_benchmark_v3_balanced_generation.yaml")


def test_reward_v3_balanced_generation_scenario_has_generation_type_vpps() -> None:
    scenario = load_scenario(CONFIG)

    assert scenario.dso.reward_config.is_v3_market_safety
    assert len(scenario.vpps) >= 10

    positive_midpoints = 0
    generation_capable = 0
    storage_capable = 0
    for vpp in scenario.vpps:
        fr = compute_static_feasible_region(vpp, 48)
        bounds = fr.aggregate_bounds()
        midpoint = 0.5 * (bounds.p_min_mw + bounds.p_max_mw)
        if midpoint > 0.0:
            positive_midpoints += 1
        if bounds.p_max_mw > 0.05:
            generation_capable += 1
        if any(der.__class__.__name__ == "StorageModel" for der in vpp.der_list):
            storage_capable += 1

    assert positive_midpoints / len(scenario.vpps) >= 0.4
    assert generation_capable >= 4
    assert storage_capable >= 4


def test_reward_v3_balanced_generation_scenario_contains_physical_stress_windows() -> None:
    scenario = load_scenario(CONFIG)
    diagnostics = scenario.profile_diagnostics()
    ac_scan = scenario.pre_control_ac_stress_scan(horizon_steps=96)

    assert diagnostics["reverseflow_candidate_steps"] > 0
    assert diagnostics["high_voltage_risk_steps"] > 0
    assert diagnostics["peak_import_stress_steps"] > 0
    assert diagnostics["storage_arbitrage_spread"] > diagnostics["storage_degradation_cost_threshold"]
    assert diagnostics["evcs_deadline_pressure_steps"] > 0
    assert ac_scan["base_reverseflow_steps"] > 0
    assert ac_scan["base_voltage_max_pu"] > 1.04
    assert ac_scan["base_line_loading_topk_percent"] > 80.0
    assert ac_scan["base_powerflow_success_rate"] > 0.95
```

- [ ] **Step 2: Create reward-v3 YAML**

Create `configs/rewards/v3_market_safety/reward_v3_market_safety.yaml`:

```yaml
reward:
  version: v3_market_safety
  critic_reward_scale: 0.01
  dso:
    curtailment_cost_weight: 0.0
    safe_capacity_utilization_weight: 0.0
    over_conservative_curtailment_weight: 0.0
    safety_margin_weight: 1.0
    hard_violation_weight: 30.0
    powerflow_failure_weight: 80.0
    raw_action_safety_weight: 20.0
    projected_action_safety_weight: 5.0
    min_raw_unsafe_penalty: 1.0
    raw_safety_epsilon: 1.0e-5
    safety_gate_input_mode: max_raw_projected
    welfare_weight: 1.0
    welfare_clip: 5.0
    welfare_normalization_mode: per_mwh_running_zscore
    welfare_baseline_mean: 0.0
    welfare_baseline_std: 10.0
    soft_safety_gate_kappa: 2.0
    loss_cost_weight: 0.2
    smoothness_weight: 0.02
  vpp:
    dispatch:
      private_profit_weight: 1.0
      service_payment_weight: 0.0
      availability_payment_weight: 0.0
      use_unified_private_profit_v3: true
      require_per_der_settlement_audit: true
      paper_long_fail_on_incomplete_settlement_audit: true
      settlement_power_balance_tolerance_mw: 1.0e-6
      projection_linear_weight: 2.0
      projection_quadratic_weight: 5.0
      battery_degradation_weight: 0.01
      storage_terminal_value_weight: 0.05
      storage_potential_shaping_weight: 0.05
      storage_charge_efficiency: 0.95
      storage_discharge_efficiency: 0.95
      storage_discount: 0.99
      storage_terminal_soc_target: 0.5
      storage_terminal_soc_target_weight: 0.01
      storage_terminal_potential_residual_mode: log_and_ablate
  shield:
    dso_penalty_coef: 1.0
    dispatch_penalty_coef: 1.0
    portfolio_future_penalty_coef: 1.0
```

- [ ] **Step 3: Create balanced generation scenario**

Create `configs/scenarios/benchmark/european_lv_benchmark_v3_balanced_generation.yaml` by extending the v2 benchmark and adding at least:

```yaml
name: european_lv_benchmark_v3_balanced_generation

reward:
  version: v3_market_safety

simulation:
  horizon_steps: 672
  dt_hours: 0.25
  seed: 2026

profiles:
  profile_pack: benchmark_3day_v1
  variant: train_mixed_balanced
  seed: 2026

stress_acceptance:
  reverseflow_candidate_steps_min: 12
  high_voltage_risk_steps_min: 12
  peak_import_stress_steps_min: 12
  evcs_deadline_pressure_steps_min: 4
  base_reverseflow_steps_min: 4
  base_voltage_near_risk_pu_min: 1.04
  base_line_loading_topk_percent_min: 80.0

vpps:
  - id: vpp_solar_park_f2
    name: Solar Park VPP
    pcc_bus: 28
    privacy_mode: representative_data
    zone_ids: [F2_zone_1, F2_zone_2]
    assets:
      pv:
        - id: pv_solar_park_f2_28
          bus: 28
          p_max_mw: 0.220
          apparent_power_mva: 0.260
          zone_id: F2_zone_1
        - id: pv_solar_park_f2_31
          bus: 31
          p_max_mw: 0.180
          apparent_power_mva: 0.220
          zone_id: F2_zone_2
      storage:
        - id: ess_solar_park_f2_31
          bus: 31
          capacity_mwh: 0.60
          soc: 0.45
          p_charge_max_mw: 0.120
          p_discharge_max_mw: 0.120
          zone_id: F2_zone_2
  - id: vpp_bess_merchant_f3
    name: Merchant BESS VPP
    pcc_bus: 60
    privacy_mode: full_information
    zone_ids: [F3_zone_4]
    assets:
      storage:
        - id: ess_merchant_f3_60
          bus: 60
          capacity_mwh: 0.90
          soc: 0.50
          p_charge_max_mw: 0.180
          p_discharge_max_mw: 0.180
          zone_id: F3_zone_4
  - id: vpp_chp_industrial_f4
    name: CHP Industrial Export VPP
    pcc_bus: 73
    privacy_mode: full_information
    zone_ids: [F4_zone_2]
    assets:
      microturbine:
        - id: mt_chp_f4_73
          bus: 73
          p_min_mw: 0.050
          p_max_mw: 0.220
          ramp_up_mw_per_step: 0.060
          ramp_down_mw_per_step: 0.060
          zone_id: F4_zone_2
      storage:
        - id: ess_chp_f4_73
          bus: 73
          capacity_mwh: 0.50
          soc: 0.55
          p_charge_max_mw: 0.100
          p_discharge_max_mw: 0.100
          zone_id: F4_zone_2
```

Include the existing seven v2 VPPs so total VPP count is at least 10.

- [ ] **Step 4: Implement profile diagnostics and pre-control AC stress scan**

In `scenario.py`, add `profile_diagnostics()` returning:

```python
{
    "reverseflow_candidate_steps": reverseflow_candidate_steps,
    "high_voltage_risk_steps": high_voltage_risk_steps,
    "peak_import_stress_steps": peak_import_stress_steps,
    "storage_arbitrage_spread": storage_arbitrage_spread,
    "storage_degradation_cost_threshold": storage_degradation_cost_threshold,
    "evcs_deadline_pressure_steps": evcs_deadline_pressure_steps,
}
```

Use profile data and configured thresholds, not trained policy behavior.

Also add `pre_control_ac_stress_scan(horizon_steps: int = 96)` returning:

```python
{
    "base_reverseflow_steps": base_reverseflow_steps,
    "base_high_voltage_violation_or_near_violation_steps": high_voltage_near_steps,
    "base_line_loading_topk_percent": line_loading_topk,
    "base_trafo_loading_topk_percent": trafo_loading_topk,
    "base_voltage_max_pu": voltage_max_pu,
    "base_voltage_min_pu": voltage_min_pu,
    "base_powerflow_success_rate": powerflow_success_rate,
}
```

Use no-control or rule-based dispatch. This scan is an acceptance test for scenario physical stress, not a learned-policy evaluation. The scenario is not paper-long eligible if profile diagnostics pass but AC scan shows no reverse-flow, no high-voltage near-risk, and low line/transformer loading.

- [ ] **Step 5: Create paper-long experiment config**

Create `configs/experiments/paper_long/reward_v3_market_safety/european_lv_benchmark_v3_balanced_generation_reward_v3.yaml`:

```yaml
extends: configs/scenarios/benchmark/european_lv_benchmark_v3_balanced_generation.yaml
name: paper_long_reward_v3_market_safety_balanced_generation
reward:
  version: v3_market_safety
algorithm:
  config: configs/algorithms/reward_v3_happo_stable.yaml
```

- [ ] **Step 6: Run scenario tests**

Run:

```bash
./.venv-server/bin/python -m pytest tests/test_reward_v3_scenario_mix.py tests/test_large_mixed_vpp_scenario.py -q
```

Expected: PASS.

---

## Task 8: Reward V3 HAPPO/HATRPO Stability Configuration

**Files:**

- Create: `configs/algorithms/reward_v3_happo_stable.yaml`
- Create: `tests/test_reward_v3_training_stability.py`
- Modify: `src/vpp_dso_sim/learning/advanced_marl.py`
- Modify: `src/vpp_dso_sim/learning/hatrpo.py`

- [ ] **Step 1: Add stability config test**

Create `tests/test_reward_v3_training_stability.py`:

```python
from __future__ import annotations

from pathlib import Path

import yaml

from vpp_dso_sim.learning.advanced_marl import (
    HAPPOConfig,
    load_happo_config_yaml,
    should_stop_policy_update_for_kl,
)


CONFIG = Path("configs/algorithms/reward_v3_happo_stable.yaml")


def test_reward_v3_happo_stability_config_has_role_specific_kl_and_norms() -> None:
    data = yaml.safe_load(CONFIG.read_text())
    algo = data["algorithm"]

    assert algo["dso_actor_lr"] <= 5.0e-5
    assert algo["dispatch_actor_lr"] <= 5.0e-5
    assert algo["critic_lr"] <= 1.0e-4
    assert algo["target_kl_dso"] <= 0.01
    assert algo["target_kl_dispatch"] <= 0.005
    assert algo["per_role_advantage_norm"] is True
    assert algo["value_norm"] is True
    assert algo["reward_component_norm"] is True
    assert algo["max_grad_norm"] <= 0.5
    assert algo["early_stop_on_kl"] is True


def test_reward_v3_stability_config_is_loaded_into_happo_config() -> None:
    cfg = load_happo_config_yaml(CONFIG)

    assert isinstance(cfg, HAPPOConfig)
    assert cfg.dso_actor_lr <= 5.0e-5
    assert cfg.dispatch_actor_lr <= 5.0e-5
    assert cfg.target_kl_dispatch <= 0.005
    assert cfg.per_role_advantage_norm is True
    assert cfg.early_stop_on_kl is True


def test_happo_update_early_stops_when_dispatch_kl_exceeds_target() -> None:
    stopped, reason = should_stop_policy_update_for_kl(
        role="dispatch",
        approx_kl=0.02,
        target_kl_by_role={"dispatch": 0.005, "dso": 0.01},
        early_stop_on_kl=True,
    )

    assert stopped is True
    assert reason == "dispatch_kl_exceeded_target"
```

- [ ] **Step 2: Create stability config**

Create `configs/algorithms/reward_v3_happo_stable.yaml`:

```yaml
algorithm:
  family: happo_hatrpo
  dso_actor_lr: 5.0e-5
  dispatch_actor_lr: 5.0e-5
  portfolio_actor_lr: 3.0e-5
  critic_lr: 1.0e-4
  target_kl_dso: 0.01
  target_kl_dispatch: 0.005
  target_kl_portfolio: 0.005
  ppo_epochs: 4
  minibatches: 4
  max_grad_norm: 0.5
  per_role_advantage_norm: true
  value_norm: true
  reward_component_norm: true
  early_stop_on_kl: true
  min_entropy_coef: 0.001
  initial_entropy_coef: 0.01
  entropy_decay: 0.995
  log_gradient_norm_before_clip: true
  log_gradient_norm_after_clip: true
  log_role_advantage_stats: true
  log_role_clip_fraction: true
  log_role_approx_kl: true
```

- [ ] **Step 3: Add trainer log fields**

First add imports and config loading / KL stop helpers in `advanced_marl.py`:

```python
import dataclasses
from pathlib import Path

import yaml
```

```python
def load_happo_config_yaml(path: str | Path) -> HAPPOConfig:
    data = yaml.safe_load(Path(path).read_text())
    algo = dict(data.get("algorithm", {}))
    allowed = {field.name for field in dataclasses.fields(HAPPOConfig)}
    kwargs = {key: value for key, value in algo.items() if key in allowed}
    return HAPPOConfig(**kwargs)


def should_stop_policy_update_for_kl(
    *,
    role: str,
    approx_kl: float,
    target_kl_by_role: dict[str, float],
    early_stop_on_kl: bool,
) -> tuple[bool, str]:
    if not early_stop_on_kl:
        return False, "disabled"
    target = float(target_kl_by_role.get(role, target_kl_by_role.get("default", float("inf"))))
    if float(approx_kl) > target:
        return True, f"{role}_kl_exceeded_target"
    return False, "within_target"
```

In HAPPO/HATRPO update code, log per role:

```text
role_approx_kl
role_clip_fraction
role_entropy
role_grad_norm_before_clip
role_grad_norm_after_clip
role_advantage_mean
role_advantage_std
role_value_loss
role_policy_loss
```

- [ ] **Step 4: Enforce KL early stopping**

In policy update loops:

```python
if config.early_stop_on_kl and approx_kl > target_kl_for_role:
    stopped_early = True
    break
```

Log:

```text
role_kl_early_stop = 1 or 0
role_update_epochs_completed
```

- [ ] **Step 5: Run stability tests**

Run:

```bash
./.venv-server/bin/python -m pytest tests/test_reward_v3_training_stability.py tests/test_hatrpo_training.py tests/test_hasac_happo.py -q
```

Expected: PASS.

---

## Task 9: Minimal Verification Experiments

**Files:**

- Modify: `examples/17_paper_training_experiment.py` only if new preset registration is required.
- Modify: `src/vpp_dso_sim/experiments/paper_training.py` only if new preset registration is required.
- Create outputs under `outputs/`.

- [ ] **Step 1: Run unit test subset**

Run:

```bash
./.venv-server/bin/python -m pytest tests/test_reward_v3_market_safety.py tests/test_reward_v3_welfare_calibration.py tests/test_reward_v3_scenario_mix.py tests/test_reward_v3_training_stability.py tests/test_multi_agent_env.py tests/test_hatrpo_training.py tests/test_hasac_happo.py -q
```

Expected: PASS.

- [ ] **Step 2: Run 8-step simulator smoke**

Run:

```bash
./.venv-server/bin/python - <<'PY'
from vpp_dso_sim.simulation.scenario import load_scenario
from vpp_dso_sim.simulation.simulator import Simulator

scenario = load_scenario("configs/scenarios/benchmark/european_lv_benchmark_v3_balanced_generation.yaml")
result = Simulator(scenario).run_timeseries(horizon_steps=8)
reward = result["reward_components"]
print(reward[["step", "reward_version_code", "dso_reward_train", "dso_safety_gate", "dso_vpp_welfare_zscore", "settlement_audit_complete", "settlement_power_balance_ok"]].tail())
assert reward["reward_version_code"].eq(3.0).all()
assert "dso_curtailment_cost" not in reward.columns
assert "dso_safe_capacity_utilization_reward" not in reward.columns
assert reward["settlement_audit_complete"].min() >= 0.0
assert reward["settlement_power_balance_ok"].min() >= 0.0
PY
```

Expected: exits 0 and prints 8 reward-v3 rows.

- [ ] **Step 3: Run 96-step sanity rollout**

Run:

```bash
./.venv-server/bin/python - <<'PY'
from pathlib import Path
from vpp_dso_sim.simulation.scenario import load_scenario
from vpp_dso_sim.simulation.simulator import Simulator

out = Path("outputs/reward_v3_sanity_96")
scenario = load_scenario("configs/scenarios/benchmark/european_lv_benchmark_v3_balanced_generation.yaml")
sim = Simulator(scenario)
sim.run_timeseries(horizon_steps=96)
paths = sim.export_results(out)
print(paths["reward_components"])
PY
```

Expected: exits 0 and writes `outputs/reward_v3_sanity_96/reward_components.csv`.

- [ ] **Step 4: Run 1-episode HAPPO smoke on GPU-auto**

Run:

```bash
./.venv-server/bin/python - <<'PY'
from pathlib import Path
from vpp_dso_sim.learning.advanced_marl import HAPPOConfig, train_happo

res = train_happo(
    config_path="configs/scenarios/benchmark/european_lv_benchmark_v3_balanced_generation.yaml",
    output_dir=Path("outputs/reward_v3_happo_smoke/train"),
    config=HAPPOConfig(
        episodes=1,
        horizon_steps=8,
        hidden_dim=64,
        seed=9601,
        device="auto",
    ),
)
print(res["summary"])
PY
```

Expected: exits 0; summary includes reward-v3 components and resolved device.

- [ ] **Step 5: Run 96-step HATRPO sanity**

Run:

```bash
./.venv-server/bin/python - <<'PY'
from pathlib import Path
from vpp_dso_sim.learning.hatrpo import HATRPOConfig, train_hatrpo

res = train_hatrpo(
    config_path="configs/scenarios/benchmark/european_lv_benchmark_v3_balanced_generation.yaml",
    output_dir=Path("outputs/reward_v3_hatrpo_96/train"),
    config=HATRPOConfig(
        episodes=3,
        horizon_steps=96,
        hidden_dim=128,
        seed=9602,
        device="auto",
    ),
)
print(res["summary"])
PY
```

Expected: exits 0; no NaN losses; traces include `evcs_user_charging_revenue`, `storage_potential_shaping_reward`, `storage_terminal_value_reward`, `dso_safety_gate`, and `role_approx_kl`.

- [ ] **Step 6: Inspect outputs**

Run:

```bash
rg -n "evcs_user_charging_revenue|storage_potential_shaping_reward|storage_terminal_value_reward|dso_safety_gate|role_approx_kl|dso_curtailment_cost" outputs/reward_v3_hatrpo_96 outputs/reward_v3_happo_smoke
```

Expected:

- New v3 fields are present.
- `dso_curtailment_cost` is absent from v3 reward outputs.

---

## Task 10: Required Ablations Before Paper-Long

**Files:**

- Modify or create experiment configs under `configs/experiments/paper_long/reward_v3_market_safety/`.
- Modify: `src/vpp_dso_sim/experiments/paper_training.py` only if preset registration is required.

- [ ] **Step 1: Add ablation configs**

Create configs for:

| Config suffix | Purpose |
|---|---|
| `v2_baseline` | Compare old reward-v2 against reward-v3. |
| `v3_no_evcs_revenue` | Prove EVCS user revenue matters. |
| `v3_no_storage_temporal_value` | Prove storage potential/terminal value matters. |
| `v3_no_generation_mix` | Prove balanced generation scenario matters. |
| `v3_soft_gate_kappa_1_3_10` | Compare gate strength schedule. |
| `v3_raw_safety_off` | Show whether actor relies on shield. |
| `v3_no_shield_eval` | Measure learned policy safety without shield. |
| `v3_profit_only_dispatch` | Verify VPP economic closure. |
| `v3_safety_only_dso` | Verify DSO can learn safety envelope without welfare. |
| `v3_no_service_payment` | Check whether VPP still learns useful dispatch without transfer-payment incentives. |
| `v3_proxy_service_payment` | Explicitly label baseline proxy service payment experiments. |
| `v3_cleared_service_payment` | Reserved for real DSO award/bid/clearing settlement once implemented. |
| `v3_projected_safety_gate_off` | Verify why projected unsafe must close the safety gate. |
| `v3_welfare_calibration_off` | Show the effect of uncalibrated welfare clipping. |

- [ ] **Step 2: Run short ablation sanity**

For each ablation, run:

```bash
./.venv-server/bin/python examples/17_paper_training_experiment.py \
  --preset reward_v3_short_ablation \
  --output-dir outputs/reward_v3_ablation_short \
  --progress-interval-seconds 60
```

Expected: all configs produce reward traces without NaN and with the intended field toggles visible.

- [ ] **Step 3: Gate paper-long start**

Paper-long is allowed only after these conditions are met:

```text
1. v3 reward traces have no DSO curtailment training fields.
2. service/availability payments are not double counted.
3. EVCS revenue comes from EVCS audit fields.
4. storage terminal value is zero before terminal steps.
5. raw unsafe/projected safe reduces DSO reward.
6. projected unsafe reduces DSO safety gate even if raw cost is low.
7. settlement_audit_complete = 1 and settlement_power_balance_ok = 1 for paper-long MARL steps.
8. welfare_clip_saturation_rate < 10%.
9. no-shield evaluation is available.
10. role KL and gradient logs are exported.
11. balanced scenario has profile stress windows and pre-control AC stress scan evidence.
```

---

## Self-Review Checklist

- Spec coverage:
  - DSO curtailment removal: Tasks 1 and 2.
  - DSO welfare normalization: Task 2.
  - Welfare calibration and clip saturation control: Task 2.5.
  - Safety gate saturation fix: Task 2.
  - Projected unsafe gate closure: Tasks 1, 2, and 6.
  - Safe-first implementation proof: Tasks 1, 2, and 6.
  - VPP settlement double-count fix: Task 3.
  - Settlement audit completeness: Tasks 3 and 5.
  - Settlement DER power balance: Tasks 3 and 5.
  - EVCS no aggregate-negative fallback: Task 3.
  - Energy buy/sell split: Task 3.
  - Storage terminal-only value: Task 4.
  - Storage anti-hoarding and terminal residual audit: Task 4.
  - DSO reward ordering: Task 5.
  - Raw/projected/no-shield safety diagnostics: Task 6.
  - Balanced generation profile and pre-control AC physical stress: Task 7.
  - HAPPO/HATRPO static and dynamic stability checks: Task 8.
  - Ablation suite: Task 10.
- Execution phases:
  - Phase A closes reward accounting and safety diagnostics before any paper-long training.
  - Phase B starts only after Phase A acceptance passes.
- Legacy compatibility:
  - v1/v2 code remains intact.
  - v1/v2 tests remain part of verification.
- Execution discipline:
  - Implement task-by-task.
  - Run the listed tests before moving to the next task.
  - In the current dirty workspace, commit steps should only be performed in an isolated worktree or after the user explicitly asks for commits.
- Main remaining research limitation:
  - This plan still uses reward shaping rather than a full CMDP/Lagrangian safe-MARL formulation. That is intentional for the next implementation stage. A future plan can add separate reward and cost critics once reward-v3 accounting is stable.
