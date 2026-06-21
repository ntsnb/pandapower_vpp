# Unified Reward Cost Composition Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Put reward composition, cost composition, cost-free reward composition, and reward-scaled cost composition into the existing Reward/Cost page so the four compositions can be compared together.

**Architecture:** Keep one primary user-facing page: `reward-cost`. Reuse existing query responses and semantic metric lists to derive four filtered composition responses at the selected composition step. Remove the two newly added standalone sidebar entries and routes so users are not pushed into separate pages for comparison.

**Tech Stack:** React, TypeScript, Vitest, Vite, existing dashboard API query responses.

---

### Task 1: Add Red Tests For Unified Composition

**Files:**
- Modify: `src/marl_dashboard/frontend/src/pages/RewardCostPage.test.tsx`
- Modify: `src/marl_dashboard/frontend/src/components/layout/Sidebar.test.tsx`

- [ ] **Step 1: Write failing page test**

Add a Reward/Cost page test that supplies these metrics:

```ts
rewardsResponse.table_rows = [
  { run_id: 'run_a', metric_group: 'reward', metric_name: 'dispatch_reward_train', value: 5, vpp_id: 'vpp_001', time_index: 3 },
  { run_id: 'run_a', metric_group: 'reward', metric_name: 'operational_surplus', value: 4, vpp_id: 'vpp_001', time_index: 3 },
  { run_id: 'run_a', metric_group: 'reward', metric_name: 'storage_potential_raw', value: 2, vpp_id: 'vpp_001', time_index: 3 }
];
costsResponse.table_rows = [
  { run_id: 'run_a', metric_group: 'cost', metric_name: 'battery_degradation_cost_total', value: 1, vpp_id: 'vpp_001', time_index: 3 },
  { run_id: 'run_a', metric_group: 'cost', metric_name: 'reward_scaled_battery_degradation_penalty', value: 0.01, vpp_id: 'vpp_001', time_index: 3 }
];
```

Assert that the same page renders these four headings:

```ts
expect(screen.getByRole('heading', { name: /奖励组成 \/ Reward composition/i })).toBeInTheDocument();
expect(screen.getByRole('heading', { name: /成本组成 \/ Cost composition/i })).toBeInTheDocument();
expect(screen.getByRole('heading', { name: /无成本奖励组成 \/ Cost-free reward composition/i })).toBeInTheDocument();
expect(screen.getByRole('heading', { name: /缩放成本组成 \/ Reward-scaled cost composition/i })).toBeInTheDocument();
```

- [ ] **Step 2: Write failing sidebar test**

Update the sidebar test so it expects no standalone buttons named `无成本奖励 / Cost-free Reward` or `缩放成本 / Scaled Costs`, and still expects `奖励成本 / Reward Cost`.

- [ ] **Step 3: Verify red**

Run:

```bash
npm --prefix src/marl_dashboard/frontend test -- RewardCostPage.test.tsx Sidebar.test.tsx
```

Expected: FAIL because the Reward/Cost page does not yet render the two extra composition headings, and the sidebar still has standalone pages.

### Task 2: Implement Unified Reward/Cost Page

**Files:**
- Modify: `src/marl_dashboard/frontend/src/pages/RewardCostPage.tsx`
- Modify: `src/marl_dashboard/frontend/src/components/layout/Sidebar.tsx`
- Modify: `src/marl_dashboard/frontend/src/App.tsx`

- [ ] **Step 1: Import semantic filters**

In `RewardCostPage.tsx`, import:

```ts
import {
  costFreeRewardMetrics,
  filterResponseForMetricList,
  rewardScaledCostMetrics
} from '../utils/rewardCostSemantics';
```

- [ ] **Step 2: Derive four composition responses**

After `compositionRewards` and `compositionCosts`, add:

```ts
const costFreeRewardComposition = useMemo(
  () => filterResponseByTimeIndex(filterResponseForMetricList(rewards.data, costFreeRewardMetrics), compositionTimeIndex),
  [rewards.data, compositionTimeIndex]
);
const rewardScaledCostComposition = useMemo(
  () => filterResponseByTimeIndex(filterResponseForMetricList(costs.data, rewardScaledCostMetrics), compositionTimeIndex),
  [costs.data, compositionTimeIndex]
);
```

- [ ] **Step 3: Render four composition blocks together**

Replace the current two composition charts with a single grouped section:

```tsx
<section className="panel">
  <div className="section-heading">
    <h2>组成对比 / Composition comparison</h2>
    <p>同一 step 下并列查看四类组成，便于判断真实奖励、真实成本和最终训练 reward 中缩放项的关系。</p>
  </div>
  <div className="two-column">
    <CompositionChart title="奖励组成 / Reward composition" response={compositionRewards} />
    <CompositionChart title="成本组成 / Cost composition" response={compositionCosts} />
    <CompositionChart title="无成本奖励组成 / Cost-free reward composition" response={costFreeRewardComposition} />
    <CompositionChart title="缩放成本组成 / Reward-scaled cost composition" response={rewardScaledCostComposition} />
  </div>
</section>
```

- [ ] **Step 4: Keep formulas on one page**

Keep the existing reward and cost formula tables. The formula table already groups included, raw, and diagnostic terms; no new standalone formula page is needed.

- [ ] **Step 5: Remove standalone navigation/routes**

In `Sidebar.tsx`, remove the `cost-free-reward` and `reward-scaled-cost` page entries.

In `App.tsx`, remove imports and route branches for `CostFreeRewardPage` and `RewardScaledCostPage`.

- [ ] **Step 6: Verify green**

Run:

```bash
npm --prefix src/marl_dashboard/frontend test -- RewardCostPage.test.tsx Sidebar.test.tsx
```

Expected: PASS.

### Task 3: Build And Runtime Verification

**Files:**
- Build output: `src/marl_dashboard/frontend/dist/`

- [ ] **Step 1: Run focused frontend regression**

Run:

```bash
npm --prefix src/marl_dashboard/frontend test -- RewardCostPage.test.tsx FormulaTable.test.tsx CostFreeRewardPage.test.tsx RewardScaledCostPage.test.tsx Sidebar.test.tsx
```

Expected: PASS, or update/remove standalone page tests if those pages become unreachable but kept as internal components.

- [ ] **Step 2: Build production frontend**

Run:

```bash
npm --prefix src/marl_dashboard/frontend run build
```

Expected: exit code 0.

- [ ] **Step 3: Verify local service**

Run:

```bash
python3 -c "import urllib.request; data=urllib.request.urlopen('http://127.0.0.1:8766/', timeout=5).read().decode(); print(data[:500])"
```

Expected: HTML references the newly built asset bundle from `dist/`.

- [ ] **Step 4: No commit in this worktree**

Do not commit because the repository already contains a large dirty worktree with unrelated user changes.
