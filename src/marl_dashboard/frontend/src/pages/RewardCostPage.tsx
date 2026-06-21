import { useEffect, useMemo, useState } from 'react';

import { api } from '../api/client';
import { useAsync, useLiveTick } from '../api/hooks';
import { CombinedChart } from '../components/charts/CombinedChart';
import { CompositionChart } from '../components/charts/CompositionChart';
import { DataNotice } from '../components/layout/DataNotice';
import { MetricChecklistSelector } from '../components/selectors/MetricChecklistSelector';
import { TimeStepSlider } from '../components/selectors/TimeStepSlider';
import { FormulaTable } from '../components/tables/FormulaTable';
import { MetricTable } from '../components/tables/MetricTable';
import { emptyResponse } from '../utils/emptyResponse';
import { metricParams } from '../utils/filters';
import { sortMetricNamesByHierarchy } from '../utils/metricHierarchy';
import { filterMetricResponseByMetrics, metricLabelsFromResponse, metricNamesFromResponse } from '../utils/metrics';
import {
  costFreeRewardMetrics,
  deriveRewardScaledCostResponse,
  filterResponseForMetricList,
  rewardScaledCostMetrics
} from '../utils/rewardCostSemantics';
import type { PageProps } from './types';
import type { QueryResponse } from '../api/types';

function vppScopeLabel(vppId: string | undefined): string {
  if (!vppId) {
    return '全部可用行 / all available rows';
  }
  return vppId === 'aggregate' ? '聚合 / aggregate' : vppId.replace('vpp_', 'VPP-');
}

function hasConcreteVppRows(response: QueryResponse | null | undefined): boolean {
  const rows = [
    ...(response?.table_rows ?? []),
    ...(response?.chart_series ?? []).flatMap((series) => series.points)
  ];
  return rows.some((row) => {
    const vppId = String(row.vpp_id ?? '');
    return vppId !== '' && vppId !== 'aggregate';
  });
}

function responseRowCount(response: QueryResponse | null | undefined): number {
  const value = response?.summary?.row_count;
  return typeof value === 'number' && Number.isFinite(value) ? value : 0;
}

function distinctTimeIndexCount(...responses: Array<QueryResponse | null | undefined>): number {
  const values = new Set<number>();
  for (const response of responses) {
    for (const series of response?.chart_series ?? []) {
      for (const point of series.points) {
        if (typeof point.time_index === 'number') {
          values.add(point.time_index);
        }
      }
    }
    for (const row of response?.table_rows ?? []) {
      if (typeof row.time_index === 'number') {
        values.add(row.time_index);
      }
    }
  }
  return values.size;
}

function timeIndicesFromResponses(...responses: Array<QueryResponse | null | undefined>): number[] {
  const values = new Set<number>();
  for (const response of responses) {
    for (const row of response?.table_rows ?? []) {
      if (typeof row.time_index === 'number') {
        values.add(row.time_index);
      }
    }
    for (const series of response?.chart_series ?? []) {
      for (const point of series.points) {
        if (typeof point.time_index === 'number') {
          values.add(point.time_index);
        }
      }
    }
  }
  return Array.from(values).sort((left, right) => left - right);
}

function filterResponseByTimeIndex(response: QueryResponse | null, timeIndex: number | undefined): QueryResponse | null {
  if (!response || timeIndex === undefined) {
    return response;
  }
  const chartSeries = response.chart_series
    .map((series) => {
      const points = series.points.filter((point) => point.time_index === timeIndex);
      if (points.length === 0) {
        return null;
      }
      return { ...series, points };
    })
    .filter((series): series is NonNullable<typeof series> => series !== null);
  const tableRows = response.table_rows.filter((row) => row.time_index === timeIndex);
  const visibleRowCount = tableRows.length || chartSeries.reduce((total, series) => total + series.points.length, 0);
  return {
    ...response,
    chart_series: chartSeries,
    table_rows: tableRows,
    summary: {
      ...response.summary,
      row_count: visibleRowCount,
      visible_row_count: visibleRowCount,
      visible_time_index: timeIndex
    }
  };
}

export function RewardCostPage({ filters, selectors, liveEventCount = 0 }: PageProps) {
  const [rewardSelection, setRewardSelection] = useState<string[] | null>(null);
  const [costSelection, setCostSelection] = useState<string[] | null>(null);
  const [compositionTimeSelection, setCompositionTimeSelection] = useState<number | undefined>(undefined);
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
  const costs = useAsync(() => (filters.runId ? api.costs(filters.runId, params) : Promise.resolve(emptyResponse)), [
    filters.runId,
    filters.date,
    filters.vppId,
    filters.epochId,
    filters.episodeId,
    filters.timeIndex,
    liveEventCount,
    tick
  ]);
  useEffect(() => {
    setRewardSelection(null);
    setCostSelection(null);
    setCompositionTimeSelection(undefined);
  }, [filters.runId]);
  const showAggregateOnlyNotice =
    (!filters.vppId || filters.vppId === 'aggregate') &&
    !hasConcreteVppRows(rewards.data) &&
    !hasConcreteVppRows(costs.data);
  const showSelectedScopeNoRowsNotice =
    Boolean(filters.vppId && filters.episodeId !== undefined) &&
    responseRowCount(rewards.data) === 0 &&
    responseRowCount(costs.data) === 0;
  const expectedTimeSlotCount = Math.max(1, selectors?.time_indices?.length ?? 96);
  const visibleTimeSlotCount = distinctTimeIndexCount(rewards.data, costs.data);
  const showPartialDateNotice =
    Boolean(filters.date) &&
    responseRowCount(rewards.data) + responseRowCount(costs.data) > 0 &&
    filters.timeIndex === undefined &&
    filters.startTimeIndex === undefined &&
    filters.endTimeIndex === undefined &&
    visibleTimeSlotCount > 0 &&
    visibleTimeSlotCount < expectedTimeSlotCount;
  const rewardMetrics = useMemo(
    () => sortMetricNamesByHierarchy(metricNamesFromResponse(rewards.data), 'reward'),
    [rewards.data]
  );
  const costMetrics = useMemo(() => sortMetricNamesByHierarchy(metricNamesFromResponse(costs.data), 'cost'), [costs.data]);
  const rewardMetricLabels = useMemo(() => metricLabelsFromResponse(rewards.data), [rewards.data]);
  const costMetricLabels = useMemo(() => metricLabelsFromResponse(costs.data), [costs.data]);
  const selectedRewardMetrics = useMemo(
    () => (rewardSelection === null ? rewardMetrics : rewardMetrics.filter((metric) => rewardSelection.includes(metric))),
    [rewardMetrics, rewardSelection]
  );
  const selectedCostMetrics = useMemo(
    () => (costSelection === null ? costMetrics : costMetrics.filter((metric) => costSelection.includes(metric))),
    [costMetrics, costSelection]
  );
  const visibleRewards = useMemo(
    () => filterMetricResponseByMetrics(rewards.data, selectedRewardMetrics),
    [rewards.data, selectedRewardMetrics]
  );
  const visibleCosts = useMemo(
    () => filterMetricResponseByMetrics(costs.data, selectedCostMetrics),
    [costs.data, selectedCostMetrics]
  );
  const costFreeRewards = useMemo(
    () => filterResponseForMetricList(rewards.data, costFreeRewardMetrics),
    [rewards.data]
  );
  const directRewardScaledCosts = useMemo(
    () => filterResponseForMetricList(costs.data, rewardScaledCostMetrics),
    [costs.data]
  );
  const rewardScaledCosts = useMemo(
    () => deriveRewardScaledCostResponse(directRewardScaledCosts, costs.data, rewards.data),
    [directRewardScaledCosts, costs.data, rewards.data]
  );
  const compositionTimeIndices = useMemo(
    () => timeIndicesFromResponses(visibleRewards, visibleCosts, costFreeRewards, rewardScaledCosts),
    [visibleRewards, visibleCosts, costFreeRewards, rewardScaledCosts]
  );
  const compositionTimeIndex = useMemo(() => {
    if (compositionTimeIndices.length === 0) {
      return undefined;
    }
    if (compositionTimeSelection !== undefined && compositionTimeIndices.includes(compositionTimeSelection)) {
      return compositionTimeSelection;
    }
    return compositionTimeIndices[compositionTimeIndices.length - 1];
  }, [compositionTimeIndices, compositionTimeSelection]);
  const compositionRewards = useMemo(
    () => filterResponseByTimeIndex(visibleRewards, compositionTimeIndex),
    [visibleRewards, compositionTimeIndex]
  );
  const compositionCosts = useMemo(
    () => filterResponseByTimeIndex(visibleCosts, compositionTimeIndex),
    [visibleCosts, compositionTimeIndex]
  );
  const costFreeRewardComposition = useMemo(
    () => filterResponseByTimeIndex(costFreeRewards, compositionTimeIndex),
    [costFreeRewards, compositionTimeIndex]
  );
  const rewardScaledCostComposition = useMemo(
    () => filterResponseByTimeIndex(rewardScaledCosts, compositionTimeIndex),
    [rewardScaledCosts, compositionTimeIndex]
  );

  return (
    <div className="page-stack">
      <DataNotice loading={rewards.loading || costs.loading} error={rewards.error ?? costs.error} />
      <div className="panel control-panel">
        <div className="table-toolbar">
          <MetricChecklistSelector
            label="奖励项 / Reward metric"
            metrics={rewardMetrics}
            metricLabels={rewardMetricLabels}
            selectedMetrics={selectedRewardMetrics}
            onChange={setRewardSelection}
          />
          <MetricChecklistSelector
            label="成本项 / Cost metric"
            metrics={costMetrics}
            metricLabels={costMetricLabels}
            selectedMetrics={selectedCostMetrics}
            onChange={setCostSelection}
          />
        </div>
      </div>
      {showAggregateOnlyNotice ? (
        <div className="notice">
          当前 reward/cost 行只有聚合训练进度；训练 adapter 记录 VPP 级指标后，单 VPP reward/cost 会自动出现 /
          Current live reward and cost rows are aggregate training progress. Per-VPP reward/cost rows will appear
          when the training adapter logs VPP-level metrics.
        </div>
      ) : null}
      {showSelectedScopeNoRowsNotice ? (
        <div className="notice">
          当前筛选下该 VPP 没有 reward/cost 行。通常原因是该 VPP 的本 episode 级指标尚未写入，或当前 episode 只存在聚合进度行；
          请切换到全部轨迹 / All episodes，或选择该 VPP 已写入的 episode。成本数值过大不会导致曲线被隐藏 /
          No reward/cost rows exist for this VPP under the selected episode. Try All episodes or an episode that has
          per-VPP rows. Large cost values do not hide the chart.
        </div>
      ) : null}
      {showPartialDateNotice ? (
        <div className="notice">
          当前 reward/cost 日期只返回 {visibleTimeSlotCount} 个 time_index，预期完整日通常为 {expectedTimeSlotCount} 个；这通常表示当前
          episode 仍在写入该日期，或 reward/cost 只来自训练进度采样。并行 worker 也可能共享同一个横坐标槽位，因此看起来像短线 /
          The selected reward/cost date currently contains {visibleTimeSlotCount} of {expectedTimeSlotCount} expected time slots. The active
          episode may still be writing this date, or reward/cost rows may be progress-sampled; parallel workers can share the same x-axis slot.
        </div>
      ) : null}
      <CombinedChart
        title={`奖励与成本轨迹 / Reward and cost trajectories - ${vppScopeLabel(filters.vppId)}`}
        responses={[
          { label: '奖励 / Reward', response: visibleRewards },
          { label: '成本 / Cost', response: visibleCosts }
        ]}
      />
      <div className="panel control-panel">
        <TimeStepSlider
          timeIndices={compositionTimeIndices}
          value={compositionTimeIndex}
          onChange={setCompositionTimeSelection}
          label="组成 step / Composition step"
          allLabel="全部 step / All steps"
          ariaLabel="组成 step / Composition step"
          title="该滑块只控制下方奖励组成和成本组成；轨迹图仍显示当前筛选范围。 / This slider only controls the reward and cost composition charts below; trajectory charts keep the current filter range."
        />
        <div className="inline-help">
          下方组成图显示当前 step 的各项数值，不再用一段范围的累计值或最后一个点代表组成 /
          Composition charts below show term values at the selected step, not accumulated range values or the last point of a window.
        </div>
      </div>
      <section className="panel">
        <h2>组成对比 / Composition comparison</h2>
        <p className="inline-help">
          同一 step 下并列查看奖励组成、成本组成、无成本奖励组成和写入训练 reward 的缩放成本组成，便于比较真实经济量和最终训练目标之间的关系 /
          Compare reward, cost, cost-free reward, and reward-scaled cost compositions at the same step to see how physical terms map into the training objective.
        </p>
        <div className="two-column">
          <CompositionChart title="奖励组成 / Reward composition" response={compositionRewards} />
          <CompositionChart title="成本组成 / Cost composition" response={compositionCosts} />
          <CompositionChart title="无成本奖励组成 / Cost-free reward composition" response={costFreeRewardComposition} />
          <CompositionChart title="缩放成本组成 / Reward-scaled cost composition" response={rewardScaledCostComposition} />
        </div>
      </section>
      <div className="two-column">
        <FormulaTable
          title="奖励公式 / Reward formulas"
          scope="reward"
          formulas={visibleRewards?.formulas ?? {}}
          rows={visibleRewards?.table_rows ?? []}
        />
        <FormulaTable
          title="成本公式 / Cost formulas"
          scope="cost"
          formulas={visibleCosts?.formulas ?? {}}
          rows={visibleCosts?.table_rows ?? []}
        />
      </div>
      <div className="two-column">
        <MetricTable title="奖励项 / Reward terms" rows={visibleRewards?.table_rows ?? []} />
        <MetricTable title="成本项 / Cost terms" rows={visibleCosts?.table_rows ?? []} />
      </div>
    </div>
  );
}
