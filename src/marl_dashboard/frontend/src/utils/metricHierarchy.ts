export type MetricHierarchyScope = 'reward' | 'cost' | 'default';

const rewardMetricOrder = [
  'total_reward',
  'reward_so_far',
  'dispatch_reward_train',
  'dispatch_reward_env',
  'profit_reward',
  'grid_balance_reward',
  'dispatch_private_profit_reward',
  'private_profit_proxy',
  'private_profit_weight',
  'quality_adjusted_operational_surplus',
  'economic_operational_surplus',
  'storage_potential_shaping_reward',
  'storage_potential_raw',
  'storage_potential_shaping_weight',
  'market_energy_margin_total',
  'export_revenue_total',
  'pv_export_revenue_total',
  'mt_export_revenue_total',
  'storage_discharge_revenue_total',
  'evcs_user_revenue_total',
  'visible_energy_minus_operation_cost',
  'energy_market_revenue',
  'service_payment',
  'service_payment_weight',
  'flexibility_service_payment',
  'availability_payment',
  'availability_payment_weight',
  'preferred_region_bonus'
];

const costMetricOrder = [
  'total_cost',
  'total_cost_so_far',
  'service_quality_penalty_total',
  'import_energy_cost_total',
  'energy_purchase_cost',
  'evcs_wholesale_cost_total',
  'storage_charge_cost_total',
  'hvac_energy_cost_total',
  'flex_energy_cost_total',
  'unclassified_import_cost_total',
  'der_operating_cost_total',
  'der_operation_cost',
  'battery_degradation_cost_total',
  'battery_degradation_cost',
  'storage_degradation_cost',
  'comfort_cost_total',
  'unserved_penalty_total',
  'constraint_violation_cost',
  'reward_scaled_total_projection_penalty',
  'reward_scaled_dispatch_projection_penalty',
  'reward_scaled_training_projection_penalty',
  'reward_scaled_comfort_soc_penalty',
  'reward_scaled_battery_degradation_penalty',
  'reward_scaled_contract_delivery_penalty',
  'contract_delivery_penalty',
  'dispatch_projection_penalty',
  'scaled_comfort_soc_penalty'
];

function orderForScope(scope: MetricHierarchyScope): string[] {
  if (scope === 'reward') {
    return rewardMetricOrder;
  }
  if (scope === 'cost') {
    return costMetricOrder;
  }
  return [];
}

function hierarchyIndex(metricName: string, scope: MetricHierarchyScope): number {
  const order = orderForScope(scope);
  const index = order.indexOf(metricName);
  return index === -1 ? order.length : index;
}

export function compareMetricByHierarchy(
  left: string,
  right: string,
  scope: MetricHierarchyScope = 'default'
): number {
  const leftIndex = hierarchyIndex(left, scope);
  const rightIndex = hierarchyIndex(right, scope);
  if (leftIndex !== rightIndex) {
    return leftIndex - rightIndex;
  }
  return left.localeCompare(right);
}

export function sortMetricNamesByHierarchy(
  metricNames: string[],
  scope: MetricHierarchyScope = 'default'
): string[] {
  return [...metricNames].sort((left, right) => compareMetricByHierarchy(left, right, scope));
}

export function sortFormulaEntriesByHierarchy(
  entries: Array<[string, string]>,
  scope: MetricHierarchyScope = 'default'
): Array<[string, string]> {
  return [...entries].sort(([left], [right]) => compareMetricByHierarchy(left, right, scope));
}
