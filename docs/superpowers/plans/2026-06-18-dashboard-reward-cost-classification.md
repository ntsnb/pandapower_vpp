# Dashboard Reward/Cost Classification Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add semantically separated reward/cost formulas plus two dashboard pages for unscaled cost-free rewards and reward-scaled costs.

**Architecture:** Extend existing trace and dashboard import paths with first-class derived metrics, then add frontend classification utilities and pages that reuse the existing query endpoints. Existing training reward math and env/model APIs remain unchanged.

**Tech Stack:** Python, pytest, React, TypeScript, Vitest, existing `marl_dashboard` API and parquet storage.

---

### Task 1: Trace-Derived Reward/Cost Fields

**Files:**
- Modify: `src/vpp_dso_sim/learning/reward_trace.py`
- Modify: `tests/test_reward_trace.py`

- [ ] **Step 1: Write failing tests**

Add tests asserting that dispatch trace rows include:

```python
assert row["reward_scaled_comfort_soc_penalty"] == pytest.approx(0.02 * 3.0)
assert row["reward_scaled_battery_degradation_penalty"] == pytest.approx(0.01 * 4.0)
assert row["reward_scaled_contract_delivery_penalty"] == pytest.approx(0.0)
assert row["reward_scaled_dispatch_projection_penalty"] == pytest.approx(0.7)
assert row["storage_potential_shaping_reward"] == pytest.approx(0.25)
assert row["cost_free_service_payment_weight"] == pytest.approx(0.0)
```

- [ ] **Step 2: Verify RED**

Run:

```bash
python3 -m pytest tests/test_reward_trace.py -q
```

Expected: fails because new fields are missing.

- [ ] **Step 3: Implement derived fields**

In `dispatch_private_profit_trace_rows`, compute derived fields from existing component values:

- `reward_scaled_contract_delivery_penalty = contract_delivery_weight * contract_delivery_penalty`
- `reward_scaled_dispatch_projection_penalty = dispatch_projection_penalty`
- `reward_scaled_comfort_soc_penalty = comfort_soc_weight * scaled_comfort_soc_penalty`
- `reward_scaled_battery_degradation_penalty = battery_degradation_weight * battery_degradation_cost`
- pass through `storage_potential_*` component fields
- pass through `*_weight` fields used by reward display

- [ ] **Step 4: Verify GREEN**

Run:

```bash
python3 -m pytest tests/test_reward_trace.py -q
```

Expected: pass.

### Task 2: Watchdog Dashboard Import

**Files:**
- Modify: `scripts/watch_paper_long_run.py`
- Modify: `tests/test_paper_long_watchdog_dashboard.py`

- [ ] **Step 1: Write failing tests**

Extend dispatch trace import tests so dashboard reward/cost tables contain:

- reward terms: `storage_potential_shaping_reward`, `cost_free_service_payment_weight`
- cost terms: `reward_scaled_comfort_soc_penalty`, `reward_scaled_battery_degradation_penalty`

- [ ] **Step 2: Verify RED**

Run:

```bash
python3 -m pytest tests/test_paper_long_watchdog_dashboard.py -q
```

Expected: fails because import specs do not include new fields.

- [ ] **Step 3: Implement import specs and formulas**

Add new metric specs and formula mapping/default formulas for derived fields in `watch_paper_long_run.py`.

- [ ] **Step 4: Verify GREEN**

Run the same pytest command. Expected: pass.

### Task 3: Variable Enrichment And Formula Semantics

**Files:**
- Modify: `src/marl_dashboard/backend/storage/variable_enrichment.py`
- Modify: `tests/test_marl_dashboard_variable_enrichment.py`

- [ ] **Step 1: Write failing tests**

Add new metric names to test lists and assert descriptions include CJK text and inclusion wording for excluded/diagnostic service terms.

- [ ] **Step 2: Verify RED**

Run:

```bash
python3 -m pytest tests/test_marl_dashboard_variable_enrichment.py -q
```

Expected: fails for missing defaults.

- [ ] **Step 3: Add defaults**

Add bilingual display names, physical meanings, units, and formulas for new derived metrics.

- [ ] **Step 4: Verify GREEN**

Run the same pytest command. Expected: pass.

### Task 4: Frontend Classification Utilities And Formula Sections

**Files:**
- Create: `src/marl_dashboard/frontend/src/utils/rewardCostSemantics.ts`
- Modify: `src/marl_dashboard/frontend/src/components/tables/FormulaTable.tsx`
- Modify: `src/marl_dashboard/frontend/src/components/tables/FormulaTable.test.tsx`
- Modify: `src/marl_dashboard/frontend/src/utils/metricHierarchy.ts`

- [ ] **Step 1: Write failing frontend tests**

Assert formula rows are rendered under section headings:

- `计入最终训练目标 / Included in training objective`
- `原始物理分解 / Raw physical breakdown`
- `诊断或未计入 / Diagnostic or excluded`

- [ ] **Step 2: Verify RED**

Run:

```bash
npm --prefix src/marl_dashboard/frontend test -- FormulaTable.test.tsx
```

Expected: fails because section support does not exist.

- [ ] **Step 3: Implement semantics utility and section rendering**

Create classification helpers and update `FormulaTable` to group entries by metric semantics while preserving hierarchy order inside each group.

- [ ] **Step 4: Verify GREEN**

Run the same npm command. Expected: pass.

### Task 5: New Dashboard Pages

**Files:**
- Create: `src/marl_dashboard/frontend/src/pages/CostFreeRewardPage.tsx`
- Create: `src/marl_dashboard/frontend/src/pages/RewardScaledCostPage.tsx`
- Create: `src/marl_dashboard/frontend/src/pages/CostFreeRewardPage.test.tsx`
- Create: `src/marl_dashboard/frontend/src/pages/RewardScaledCostPage.test.tsx`
- Modify: `src/marl_dashboard/frontend/src/App.tsx`
- Modify: `src/marl_dashboard/frontend/src/components/layout/Sidebar.tsx`

- [ ] **Step 1: Write failing tests**

Assert:

- cost-free page charts raw values and renders weight labels.
- reward-scaled cost page charts only scaled penalty metrics.
- sidebar includes both new pages.

- [ ] **Step 2: Verify RED**

Run:

```bash
npm --prefix src/marl_dashboard/frontend test -- CostFreeRewardPage.test.tsx RewardScaledCostPage.test.tsx
```

Expected: fails because pages do not exist.

- [ ] **Step 3: Implement pages and routing**

Reuse existing `CombinedChart`, `MetricChecklistSelector`, `FormulaTable`, and `MetricTable`.

- [ ] **Step 4: Verify GREEN**

Run the same npm command. Expected: pass.

### Task 6: Focused Regression Verification

**Files:**
- No production file modifications unless tests expose a real regression.

- [ ] **Step 1: Run focused Python tests**

```bash
python3 -m pytest tests/test_reward_trace.py tests/test_marl_dashboard_variable_enrichment.py tests/test_paper_long_watchdog_dashboard.py -q
```

- [ ] **Step 2: Run focused frontend tests**

```bash
npm --prefix src/marl_dashboard/frontend test -- FormulaTable.test.tsx RewardCostPage.test.tsx CostFreeRewardPage.test.tsx RewardScaledCostPage.test.tsx
```

- [ ] **Step 3: Fix failures with TDD discipline**

Only fix failures directly caused by this feature.
