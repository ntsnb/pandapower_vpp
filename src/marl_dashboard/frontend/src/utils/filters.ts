import type { Filters } from '../api/types';

export function metricParams(filters: Filters): Record<string, string | number | boolean | undefined> {
  const rawStartTimeIndex = filters.startTimeIndex ?? filters.timeIndex;
  const rawEndTimeIndex = filters.endTimeIndex ?? filters.timeIndex;
  const startTimeIndex =
    rawStartTimeIndex !== undefined && rawEndTimeIndex !== undefined
      ? Math.min(rawStartTimeIndex, rawEndTimeIndex)
      : rawStartTimeIndex;
  const endTimeIndex =
    rawStartTimeIndex !== undefined && rawEndTimeIndex !== undefined
      ? Math.max(rawStartTimeIndex, rawEndTimeIndex)
      : rawEndTimeIndex;
  return {
    date: filters.date,
    vpp_id: filters.vppId,
    agent_id: filters.agentId,
    epoch_id: filters.epochId,
    episode_id: filters.episodeId,
    start_time_index: startTimeIndex,
    end_time_index: endTimeIndex,
    policy_id: filters.policyId,
    max_points: 600
  };
}

export function datasetMetricParams(filters: Filters): Record<string, string | number | boolean | undefined> {
  const params = metricParams(filters);
  const hasNarrowTimeOrScope =
    Boolean(filters.date || filters.vppId || filters.timeIndex !== undefined || filters.startTimeIndex !== undefined || filters.endTimeIndex !== undefined);
  return {
    ...params,
    max_points: hasNarrowTimeOrScope ? 30000 : params.max_points
  };
}

export function latestNumber(values: number[]): number | undefined {
  return values.length > 0 ? values[values.length - 1] : undefined;
}

export function compactNumber(value: number | string | boolean | null | undefined, digits = 3): string {
  if (typeof value !== 'number' || !Number.isFinite(value)) {
    return value === null || value === undefined ? '-' : String(value);
  }
  if (value === 0) {
    return '0';
  }
  if (Math.abs(value) >= 1000 || Math.abs(value) < 0.001) {
    return value.toExponential(2);
  }
  return value.toLocaleString(undefined, { maximumFractionDigits: digits });
}

export function metricValue(rowValue: number | string | boolean | null): number | null {
  return typeof rowValue === 'number' && Number.isFinite(rowValue) ? rowValue : null;
}
