import { api } from '../api/client';
import { useAsync, useLiveTick } from '../api/hooks';
import { CurrentValueCard } from '../components/cards/CurrentValueCard';
import { StatusCard } from '../components/cards/StatusCard';
import { VppSummaryCard } from '../components/cards/VppSummaryCard';
import { MetricLineChart } from '../components/charts/MetricLineChart';
import { DataNotice } from '../components/layout/DataNotice';
import { MetricTable } from '../components/tables/MetricTable';
import { emptyResponse } from '../utils/emptyResponse';
import { compactNumber, metricParams } from '../utils/filters';
import type { PageProps } from './types';

export function OverviewPage({ filters, run, selectors, liveEventCount = 0 }: PageProps) {
  const tick = useLiveTick(filters.live);
  const params = metricParams(filters);
  const dataset = useAsync(() => (filters.runId ? api.dataset(filters.runId, params) : Promise.resolve(emptyResponse)), [
    filters.runId,
    filters.date,
    filters.vppId,
    filters.epochId,
    filters.episodeId,
    filters.timeIndex,
    liveEventCount,
    tick
  ]);
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
  const events = useAsync(
    () => (filters.runId ? api.events(filters.runId, { ...params, max_points: 12 }) : Promise.resolve(emptyResponse)),
    [
      filters.runId,
      filters.date,
      filters.vppId,
      filters.epochId,
      filters.episodeId,
      filters.timeIndex,
      liveEventCount,
      tick
    ]
  );

  return (
    <div className="page-stack">
      <DataNotice
        loading={dataset.loading || rewards.loading || costs.loading || events.loading}
        error={dataset.error ?? rewards.error ?? costs.error ?? events.error}
      />
      <div className="stat-grid">
        <StatusCard label="status" value={run?.status ?? 'unknown'} tone={run?.status === 'running' ? 'good' : 'neutral'} />
        <StatusCard label="VPP count" value={selectors?.vpp_ids.length ?? run?.vpp_count ?? 0} />
        <StatusCard label="epochs" value={selectors?.epoch_ids.length ?? run?.epoch_count ?? 0} />
        <StatusCard label="rows" value={compactNumber(dataset.data?.summary.row_count ?? 0)} />
        <CurrentValueCard title="latest reward" response={rewards.data} />
        <CurrentValueCard title="latest cost" response={costs.data} />
      </div>
      <MetricLineChart title="Dataset physical signals" response={dataset.data} />
      <div className="two-column">
        <VppSummaryCard title="Reward by VPP" response={rewards.data} />
        <VppSummaryCard title="Cost by VPP" response={costs.data} />
      </div>
      <MetricTable title="最近事件 / Recent events" rows={events.data?.table_rows ?? []} limit={12} />
    </div>
  );
}
