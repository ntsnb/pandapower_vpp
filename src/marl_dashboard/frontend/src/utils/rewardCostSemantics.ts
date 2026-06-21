import type { MetricRow, QueryResponse } from '../api/types';
import { filterMetricResponseByMetrics } from './metrics';
import type { MetricHierarchyScope } from './metricHierarchy';
import { sortMetricNamesByHierarchy } from './metricHierarchy';

export type FormulaSectionId = 'included' | 'raw' | 'diagnostic';

export const formulaSections: Array<{ id: FormulaSectionId; label: string }> = [
  { id: 'included', label: '计入最终训练目标 / Included in training objective' },
  { id: 'raw', label: '原始物理分解 / Raw physical breakdown' },
  { id: 'diagnostic', label: '诊断或未计入 / Diagnostic or excluded' }
];

const includedObjectiveMetrics = new Set([
  'total_reward',
  'dispatch_reward_train',
  'dispatch_reward_env',
  'dispatch_private_profit_reward',
  'economic_operational_surplus',
  'storage_potential_shaping_reward',
  'preferred_region_bonus',
  'reward_scaled_contract_delivery_penalty',
  'reward_scaled_dispatch_projection_penalty',
  'reward_scaled_training_projection_penalty',
  'reward_scaled_total_projection_penalty',
  'reward_scaled_comfort_soc_penalty',
  'reward_scaled_battery_degradation_penalty'
]);

const diagnosticOrExcludedMetrics = new Set([
  'service_payment',
  'flexibility_service_payment',
  'availability_payment',
  'service_payment_weight',
  'availability_payment_weight',
  'contract_delivery_weight',
  'comfort_soc_weight',
  'battery_degradation_weight',
  'storage_potential_shaping_weight',
  'private_profit_vs_visible_energy_residual',
  'economic_surplus_vs_market_margin_residual'
]);

const rawBreakdownMetrics = new Set([
  'operational_surplus',
  'private_profit_proxy',
  'private_profit_weight',
  'quality_adjusted_operational_surplus',
  'service_quality_penalty_total',
  'market_energy_margin_total',
  'export_revenue_total',
  'pv_export_revenue_total',
  'mt_export_revenue_total',
  'storage_discharge_revenue_total',
  'evcs_user_revenue_total',
  'visible_energy_minus_operation_cost',
  'energy_market_revenue',
  'storage_potential_raw',
  'import_energy_cost_total',
  'evcs_wholesale_cost_total',
  'storage_charge_cost_total',
  'hvac_energy_cost_total',
  'flex_energy_cost_total',
  'unclassified_import_cost_total',
  'der_operating_cost_total',
  'der_operation_cost',
  'battery_degradation_cost_total',
  'battery_degradation_cost',
  'comfort_cost_total',
  'unserved_penalty_total',
  'scaled_comfort_soc_penalty',
  'contract_delivery_penalty',
  'dispatch_projection_penalty'
]);

export const costFreeRewardMetrics = [
  'economic_operational_surplus',
  'operational_surplus',
  'private_profit_proxy',
  'market_energy_margin_total',
  'export_revenue_total',
  'pv_export_revenue_total',
  'mt_export_revenue_total',
  'storage_discharge_revenue_total',
  'evcs_user_revenue_total',
  'energy_market_revenue',
  'storage_potential_raw',
  'service_payment',
  'flexibility_service_payment',
  'availability_payment'
];

export const costFreeRewardWeightMetrics = [
  'private_profit_weight',
  'service_payment_weight',
  'availability_payment_weight',
  'storage_potential_shaping_weight'
];

export const rewardScaledCostMetrics = [
  'reward_scaled_total_projection_penalty',
  'reward_scaled_dispatch_projection_penalty',
  'reward_scaled_training_projection_penalty',
  'reward_scaled_comfort_soc_penalty',
  'reward_scaled_battery_degradation_penalty',
  'reward_scaled_contract_delivery_penalty'
];

const weightByCostFreeMetric: Record<string, string | undefined> = {
  economic_operational_surplus: 'private_profit_weight',
  operational_surplus: 'private_profit_weight',
  private_profit_proxy: 'private_profit_weight',
  storage_potential_raw: 'storage_potential_shaping_weight',
  service_payment: 'service_payment_weight',
  flexibility_service_payment: 'service_payment_weight',
  availability_payment: 'availability_payment_weight'
};

const derivedScaledCostDefinitions: Record<
  string,
  { metricName: string; displayName: string; description: string; weight: number }
> = {
  dispatch_projection_penalty: {
    metricName: 'reward_scaled_dispatch_projection_penalty',
    displayName: '缩放调度投影惩罚 / Reward-scaled dispatch projection penalty',
    description:
      '由旧 trace 的调度投影惩罚即时推导；v3.1 中该项按 1.0 写入训练 reward。 / Derived from legacy dispatch projection penalty; v3.1 uses weight 1.0 in the training reward.',
    weight: 1
  },
  scaled_comfort_soc_penalty: {
    metricName: 'reward_scaled_comfort_soc_penalty',
    displayName: '缩放舒适度/SOC 惩罚 / Reward-scaled comfort/SOC penalty',
    description:
      '由旧 trace 的舒适度/SOC 惩罚即时推导；v3.1 中该项按 0.02 写入训练 reward。 / Derived from legacy comfort/SOC penalty; v3.1 uses weight 0.02 in the training reward.',
    weight: 0.02
  },
  battery_degradation_cost: {
    metricName: 'reward_scaled_battery_degradation_penalty',
    displayName: '缩放电池退化惩罚 / Reward-scaled battery degradation penalty',
    description:
      '由旧 trace 的电池退化成本即时推导；v3.1 中该项按 0.01 写入训练 reward。 / Derived from legacy battery degradation cost; v3.1 uses weight 0.01 in the training reward.',
    weight: 0.01
  },
  contract_delivery_penalty: {
    metricName: 'reward_scaled_contract_delivery_penalty',
    displayName: '缩放合同交付惩罚 / Reward-scaled contract delivery penalty',
    description:
      '由旧 trace 的合同交付惩罚即时推导；当前 v3.1 合同权重为 0，因此通常显示为 0。 / Derived from legacy contract delivery penalty; current v3.1 contract weight is 0.',
    weight: 0
  }
};

const derivedScaledCostFormulas: Record<string, string> = {
  reward_scaled_dispatch_projection_penalty: '\\tilde{C}^{\\text{投影,环境}}_{i,t}=1.0\\,C^{\\text{投影}}_{i,t}',
  reward_scaled_training_projection_penalty:
    '\\tilde{C}^{\\text{投影,训练}}_{i,t}=\\max(0,r^{\\text{环境调度}}_{i,t}-r^{\\text{训练调度}}_{i,t})',
  reward_scaled_total_projection_penalty:
    '\\tilde{C}^{\\text{投影,总}}_{i,t}=\\tilde{C}^{\\text{投影,环境}}_{i,t}+\\tilde{C}^{\\text{投影,训练}}_{i,t}',
  reward_scaled_comfort_soc_penalty: '\\tilde{C}^{\\text{舒适/SOC}}_{i,t}=0.02\\,C^{\\text{舒适/SOC}}_{i,t}',
  reward_scaled_battery_degradation_penalty: '\\tilde{C}^{\\text{电池退化}}_{i,t}=0.01\\,C^{\\text{电池退化}}_{i,t}',
  reward_scaled_contract_delivery_penalty: '\\tilde{C}^{\\text{合同}}_{i,t}=0\\,C^{\\text{合同}}_{i,t}'
};

function numericValue(value: MetricRow['value']): number | null {
  return typeof value === 'number' && Number.isFinite(value) ? value : null;
}

function contextKey(row: MetricRow): string {
  return [
    row.run_id,
    row.epoch_id ?? '',
    row.episode_id ?? '',
    row.global_env_step ?? '',
    row.env_id ?? '',
    row.vpp_id ?? '',
    row.agent_id ?? '',
    row.policy_id ?? '',
    row.date ?? '',
    row.time_index ?? ''
  ].join('|');
}

function derivedRow(base: MetricRow, metricName: string, value: number, displayName: string, description: string): MetricRow {
  return {
    ...base,
    metric_group: 'cost',
    metric_name: metricName,
    value,
    unit: 'score',
    display_name: displayName,
    description,
    formula_latex: derivedScaledCostFormulas[metricName]
  };
}

export function formulaSectionForMetric(metricName: string, scope: MetricHierarchyScope = 'default'): FormulaSectionId {
  if (diagnosticOrExcludedMetrics.has(metricName)) {
    return 'diagnostic';
  }
  if (includedObjectiveMetrics.has(metricName)) {
    return 'included';
  }
  if (rawBreakdownMetrics.has(metricName)) {
    return 'raw';
  }
  if (scope === 'cost') {
    return 'raw';
  }
  return 'diagnostic';
}

export function filterResponseForMetricList(response: QueryResponse | null, metricNames: string[]): QueryResponse | null {
  if (!response) {
    return response;
  }
  const available = new Set(metricNames);
  const filtered = filterMetricResponseByMetrics(response, metricNames);
  if (!filtered) {
    return filtered;
  }
  return filtered.chart_series.length > 0 || filtered.table_rows.length > 0
    ? filtered
    : {
        ...response,
        chart_series: [],
        table_rows: [],
        units: Object.fromEntries(Object.entries(response.units).filter(([name]) => available.has(name))),
        formulas: Object.fromEntries(Object.entries(response.formulas).filter(([name]) => available.has(name))),
        summary: { ...response.summary, row_count: 0, visible_row_count: 0 }
      };
}

export function deriveRewardScaledCostResponse(
  directScaledResponse: QueryResponse | null,
  rawCostResponse: QueryResponse | null,
  rewardResponse: QueryResponse | null
): QueryResponse | null {
  if (!rawCostResponse) {
    return directScaledResponse;
  }
  if ((directScaledResponse?.table_rows.length ?? 0) > 0 || (directScaledResponse?.chart_series.length ?? 0) > 0) {
    return directScaledResponse;
  }

  const rows: MetricRow[] = [];
  const projectionByContext = new Map<string, { base: MetricRow; value: number }>();
  for (const row of rawCostResponse.table_rows) {
    const definition = derivedScaledCostDefinitions[row.metric_name];
    if (!definition) {
      continue;
    }
    const rawValue = numericValue(row.value);
    if (rawValue === null) {
      continue;
    }
    const scaledValue = rawValue * definition.weight;
    const scaledRow = derivedRow(row, definition.metricName, scaledValue, definition.displayName, definition.description);
    rows.push(scaledRow);
    if (row.metric_name === 'dispatch_projection_penalty') {
      projectionByContext.set(contextKey(row), { base: row, value: scaledValue });
    }
  }

  const rewardByContext = new Map<string, { env?: MetricRow; train?: MetricRow }>();
  for (const row of rewardResponse?.table_rows ?? []) {
    if (row.metric_name !== 'dispatch_reward_env' && row.metric_name !== 'dispatch_reward_train') {
      continue;
    }
    const key = contextKey(row);
    const value = rewardByContext.get(key) ?? {};
    if (row.metric_name === 'dispatch_reward_env') {
      value.env = row;
    } else {
      value.train = row;
    }
    rewardByContext.set(key, value);
  }

  const trainingProjectionByContext = new Map<string, { base: MetricRow; value: number }>();
  for (const [key, pair] of rewardByContext.entries()) {
    const envValue = numericValue(pair.env?.value ?? null);
    const trainValue = numericValue(pair.train?.value ?? null);
    if (envValue === null || trainValue === null) {
      continue;
    }
    const value = Math.max(0, envValue - trainValue);
    const base = pair.train ?? pair.env;
    if (!base) {
      continue;
    }
    trainingProjectionByContext.set(key, { base, value });
    rows.push(
      derivedRow(
        base,
        'reward_scaled_training_projection_penalty',
        value,
        '训练侧额外投影惩罚 / Training-side projection penalty',
        '由环境调度奖励和训练调度奖励差值即时推导；用于解释旧 trace 中算法侧额外扣除的投影惩罚。 / Derived from the gap between environment and training dispatch rewards in legacy traces.'
      )
    );
  }

  const totalProjectionKeys = new Set([...projectionByContext.keys(), ...trainingProjectionByContext.keys()]);
  for (const key of totalProjectionKeys) {
    const dispatch = projectionByContext.get(key);
    const training = trainingProjectionByContext.get(key);
    const value = (dispatch?.value ?? 0) + (training?.value ?? 0);
    const base = dispatch?.base ?? training?.base;
    if (!base) {
      continue;
    }
    rows.push(
      derivedRow(
        base,
        'reward_scaled_total_projection_penalty',
        value,
        '总缩放投影惩罚 / Total reward-scaled projection penalty',
        '环境侧投影惩罚与算法训练侧额外投影惩罚之和；由旧 trace 即时推导。 / Sum of environment-side and training-side projection penalties derived from legacy traces.'
      )
    );
  }

  const formulas = { ...(directScaledResponse?.formulas ?? {}) };
  for (const row of rows) {
    const formula = derivedScaledCostFormulas[row.metric_name];
    if (formula) {
      formulas[row.metric_name] = formula;
    }
  }
  return {
    ...(directScaledResponse ?? rawCostResponse),
    chart_series: [],
    table_rows: rows,
    units: Object.fromEntries(rows.map((row) => [row.metric_name, row.unit ?? 'score'])),
    formulas,
    summary: {
      ...(directScaledResponse?.summary ?? {}),
      row_count: rows.length,
      visible_row_count: rows.length,
      derived_from_raw_cost_terms: true
    }
  };
}

export function rewardWeightMetricFor(metricName: string): string | undefined {
  return weightByCostFreeMetric[metricName];
}

export function weightRowsByMetric(rows: MetricRow[]): Map<string, MetricRow> {
  const result = new Map<string, MetricRow>();
  for (const row of rows) {
    if (costFreeRewardWeightMetrics.includes(row.metric_name)) {
      result.set(row.metric_name, row);
    }
  }
  return result;
}

export function sortedSemanticMetrics(metricNames: string[], scope: MetricHierarchyScope): string[] {
  return sortMetricNamesByHierarchy(metricNames, scope);
}
