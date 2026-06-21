import { useMemo } from 'react';

import { api } from '../api/client';
import { useAsync, useLiveTick } from '../api/hooks';
import type { MetricRow, QueryResponse } from '../api/types';
import { CombinedChart } from '../components/charts/CombinedChart';
import { DataNotice } from '../components/layout/DataNotice';
import { FormulaTable } from '../components/tables/FormulaTable';
import { MetricTable } from '../components/tables/MetricTable';
import { compactNumber, metricParams } from '../utils/filters';
import {
  costFreeRewardMetrics,
  costFreeRewardWeightMetrics,
  filterResponseForMetricList,
  rewardWeightMetricFor
} from '../utils/rewardCostSemantics';
import { emptyResponse } from '../utils/emptyResponse';
import type { PageProps } from './types';

function rowsFromResponse(response: QueryResponse | null | undefined): MetricRow[] {
  if (!response) {
    return [];
  }
  return [...response.table_rows, ...response.chart_series.flatMap((series) => series.points)];
}

function latestRowsByMetric(rows: MetricRow[], metricNames: string[]): MetricRow[] {
  const wanted = new Set(metricNames);
  const result = new Map<string, MetricRow>();
  for (const row of rows) {
    if (wanted.has(row.metric_name)) {
      result.set(row.metric_name, row);
    }
  }
  return metricNames.map((metricName) => result.get(metricName)).filter((row): row is MetricRow => row !== undefined);
}

export function CostFreeRewardPage({ filters, liveEventCount = 0 }: PageProps) {
  const tick = useLiveTick(filters.live);
  const params = metricParams(filters);
  const rewards = useAsync(() => (filters.runId ? api.rewards(filters.runId, params) : Promise.resolve(emptyResponse)), [
    filters.runId,
    filters.date,
    filters.vppId,
    filters.epochId,
    filters.episodeId,
    filters.timeIndex,
    liveEventCount,
    tick
  ]);
  const visibleRewards = useMemo(
    () => filterResponseForMetricList(rewards.data, costFreeRewardMetrics),
    [rewards.data]
  );
  const weightRows = useMemo(
    () => latestRowsByMetric(rowsFromResponse(rewards.data), costFreeRewardWeightMetrics),
    [rewards.data]
  );
  const rawRows = visibleRewards?.table_rows ?? [];
  const weightByMetric = new Map(weightRows.map((row) => [row.metric_name, row]));

  return (
    <div className="page-stack">
      <DataNotice loading={rewards.loading} error={rewards.error} />
      <div className="notice">
        无成本奖励页展示未缩放真实物理/经济量；下方权重说明这些真实量在最终训练 reward 中如何被缩放或是否被排除 /
        This page charts unscaled physical/economic reward values; weights below show how each value is scaled or excluded in the final training reward.
      </div>
      <CombinedChart
        title="无成本奖励真实值 / Cost-free reward raw values"
        responses={[{ label: '真实值 / Raw value', response: visibleRewards }]}
      />
      <section className="panel table-panel">
        <div className="panel-header">
          <h2>奖励权重 / Reward weights</h2>
          <span>{weightRows.length} 个权重 / weights</span>
        </div>
        <div className="metric-checklist-grid">
          {weightRows.map((row) => (
            <span key={row.metric_name}>
              {row.display_name ?? row.metric_name}: {row.metric_name} = {compactNumber(row.value)}
            </span>
          ))}
          {weightRows.length === 0 ? <span className="empty-state">无权重行 / No weight rows</span> : null}
        </div>
      </section>
      <section className="panel table-panel">
        <div className="panel-header">
          <h2>真实值与权重对照 / Raw values and reward weights</h2>
          <span>{rawRows.length} 个真实值 / raw values</span>
        </div>
        <div className="metric-checklist-grid">
          {rawRows.map((row) => {
            const weightMetric = rewardWeightMetricFor(row.metric_name);
            const weightRow = weightMetric ? weightByMetric.get(weightMetric) : undefined;
            const rawValue = typeof row.value === 'number' && Number.isFinite(row.value) ? row.value : null;
            const weightValue = typeof weightRow?.value === 'number' && Number.isFinite(weightRow.value) ? weightRow.value : null;
            const weightedContribution = rawValue !== null && weightValue !== null ? rawValue * weightValue : null;
            return (
              <span key={`${row.metric_name}-${row.vpp_id ?? 'all'}-${row.time_index ?? 'latest'}`}>
                {row.display_name ?? row.metric_name}: 真实值 / raw {compactNumber(row.value)}
                {weightMetric
                  ? `, ${weightMetric} = ${compactNumber(weightRow?.value)}, 计入值 / weighted ${compactNumber(weightedContribution)}`
                  : ', 无直接单项权重 / no direct single weight'}
              </span>
            );
          })}
          {rawRows.length === 0 ? <span className="empty-state">无真实奖励行 / No raw reward rows</span> : null}
        </div>
      </section>
      <FormulaTable
        title="无成本奖励公式 / Cost-free reward formulas"
        scope="reward"
        formulas={visibleRewards?.formulas ?? {}}
        rows={rawRows}
      />
      <MetricTable title="无成本奖励真实值 / Cost-free reward raw terms" rows={rawRows} />
    </div>
  );
}
