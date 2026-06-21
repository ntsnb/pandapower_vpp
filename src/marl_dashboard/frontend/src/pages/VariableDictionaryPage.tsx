import { useMemo } from 'react';

import { api } from '../api/client';
import { useAsync, useLiveTick } from '../api/hooks';
import { DataNotice } from '../components/layout/DataNotice';
import { FormulaTable } from '../components/tables/FormulaTable';
import { VariableDictionaryTable } from '../components/tables/VariableDictionaryTable';
import { emptyResponse } from '../utils/emptyResponse';
import { metricParams } from '../utils/filters';
import { currentValuesFromResponses } from '../utils/metrics';
import type { PageProps } from './types';

export function VariableDictionaryPage({ filters, liveEventCount = 0 }: PageProps) {
  const tick = useLiveTick(filters.live, 10000);
  const params = metricParams(filters);
  const variables = useAsync(() => (filters.runId ? api.variables(filters.runId) : Promise.resolve([])), [filters.runId, liveEventCount, tick]);
  const formulas = useAsync(() => (filters.runId ? api.formulas(filters.runId) : Promise.resolve({})), [filters.runId, liveEventCount, tick]);
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
  const losses = useAsync(() => (filters.runId ? api.losses(filters.runId, params) : Promise.resolve(emptyResponse)), [
    filters.runId,
    filters.vppId,
    filters.epochId,
    filters.episodeId,
    filters.policyId,
    liveEventCount,
    tick
  ]);
  const scalars = useAsync(() => (filters.runId ? api.scalars(filters.runId, params) : Promise.resolve(emptyResponse)), [
    filters.runId,
    filters.vppId,
    filters.epochId,
    filters.episodeId,
    filters.policyId,
    liveEventCount,
    tick
  ]);
  const currentValues = useMemo(
    () => currentValuesFromResponses([dataset.data, rewards.data, costs.data, losses.data, scalars.data]),
    [dataset.data, rewards.data, costs.data, losses.data, scalars.data]
  );

  return (
    <div className="page-stack">
      <DataNotice
        loading={variables.loading || formulas.loading}
        error={variables.error ?? formulas.error ?? dataset.error ?? rewards.error ?? costs.error ?? losses.error ?? scalars.error}
      />
      <VariableDictionaryTable variables={variables.data ?? []} currentValues={currentValues} />
      <FormulaTable formulas={formulas.data ?? {}} />
    </div>
  );
}
