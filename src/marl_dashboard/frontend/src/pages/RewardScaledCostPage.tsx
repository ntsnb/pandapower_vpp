import { useMemo } from 'react';

import { api } from '../api/client';
import { useAsync, useLiveTick } from '../api/hooks';
import { CombinedChart } from '../components/charts/CombinedChart';
import { DataNotice } from '../components/layout/DataNotice';
import { FormulaTable } from '../components/tables/FormulaTable';
import { MetricTable } from '../components/tables/MetricTable';
import { emptyResponse } from '../utils/emptyResponse';
import { metricParams } from '../utils/filters';
import { filterResponseForMetricList, rewardScaledCostMetrics } from '../utils/rewardCostSemantics';
import type { PageProps } from './types';

export function RewardScaledCostPage({ filters, liveEventCount = 0 }: PageProps) {
  const tick = useLiveTick(filters.live);
  const params = metricParams(filters);
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
  const visibleCosts = useMemo(() => filterResponseForMetricList(costs.data, rewardScaledCostMetrics), [costs.data]);
  const rows = visibleCosts?.table_rows ?? [];

  return (
    <div className="page-stack">
      <DataNotice loading={costs.loading} error={costs.error} />
      <div className="notice">
        本页只展示写入训练 reward 的缩放后成本/惩罚项；原始物理成本仍在“奖励成本”页查看 /
        This page only shows cost penalties after reward scaling; raw physical costs remain on the Reward Cost page.
      </div>
      <CombinedChart
        title="缩放成本惩罚 / Reward-scaled cost penalties"
        responses={[{ label: '缩放后惩罚 / Scaled penalty', response: visibleCosts }]}
      />
      <FormulaTable
        title="缩放成本公式 / Reward-scaled cost formulas"
        scope="cost"
        formulas={visibleCosts?.formulas ?? {}}
        rows={rows}
      />
      <MetricTable title="缩放成本项 / Reward-scaled cost terms" rows={rows} />
    </div>
  );
}
