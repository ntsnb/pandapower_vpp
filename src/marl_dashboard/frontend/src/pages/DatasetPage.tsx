import { api } from '../api/client';
import { useAsync, useLiveTick } from '../api/hooks';
import type { QueryResponse } from '../api/types';
import { MetricLineChart } from '../components/charts/MetricLineChart';
import { MultiPanelChart } from '../components/charts/MultiPanelChart';
import { DataNotice } from '../components/layout/DataNotice';
import { DatasetTimeseriesTable } from '../components/tables/DatasetTimeseriesTable';
import { MetricTable } from '../components/tables/MetricTable';
import { emptyResponse } from '../utils/emptyResponse';
import { datasetMetricParams } from '../utils/filters';
import type { PageProps } from './types';

const datasetPanels: Array<{ title: string; metrics: string[] }> = [
  { title: '电价 / Electricity price', metrics: ['electricity_price', 'market_price'] },
  { title: 'EV 充电负荷 / EV charging load', metrics: ['ev_charging_load'] },
  { title: '储能功率与 SOC / Storage power and SOC', metrics: ['storage_power', 'storage_soc'] },
  { title: '光伏出力 / PV power', metrics: ['pv_power'] },
  { title: '风电出力 / Wind power', metrics: ['wind_power'] },
  { title: '负荷与净负荷 / Load and net load', metrics: ['base_load', 'net_load'] },
  {
    title: 'VPP 调度与净交付 / VPP dispatch and net delivery',
    metrics: [
      'delivered_p_mw',
      'baseline_p_mw',
      'requested_delta_p_mw',
      'accepted_delta_p_mw',
      'actual_delta_p_mw',
      'actual_target_p_mw'
    ]
  },
  {
    title: '动作落地与策略输出 / Action landing and policy outputs',
    metrics: [
      'raw_target_p_mw',
      'decoded_target_p_mw',
      'device_feasible_target_p_mw',
      'ac_projected_target_p_mw',
      'action_landing_ratio',
      'policy_normalized_aggregate_action',
      'policy_normalized_der_action_mean',
      'policy_normalized_der_action_std'
    ]
  }
];

const physicalDatasetMetrics = new Set(datasetPanels.flatMap((panel) => panel.metrics));

function metricNames(response: QueryResponse | null | undefined): Set<string> {
  const names = new Set<string>();
  for (const series of response?.chart_series ?? []) {
    if (series.metric_name) {
      names.add(series.metric_name);
    }
    for (const point of series.points) {
      names.add(point.metric_name);
    }
  }
  for (const row of response?.table_rows ?? []) {
    names.add(row.metric_name);
  }
  return names;
}

function filterMetrics(response: QueryResponse | null | undefined, metrics: string[]): QueryResponse | null {
  if (!response) {
    return null;
  }
  const selected = new Set(metrics);
  return {
    ...response,
    chart_series: response.chart_series.filter((series) => !series.metric_name || selected.has(series.metric_name)),
    table_rows: response.table_rows.filter((row) => selected.has(row.metric_name)),
    units: Object.fromEntries(Object.entries(response.units).filter(([name]) => selected.has(name))),
    formulas: Object.fromEntries(Object.entries(response.formulas).filter(([name]) => selected.has(name)))
  };
}

function distinctTimeIndexCount(response: QueryResponse | null | undefined): number {
  const values = new Set<number>();
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
  return values.size;
}

export function DatasetPage({ filters, selectors, liveEventCount = 0 }: PageProps) {
  const tick = useLiveTick(filters.live);
  const params = datasetMetricParams(filters);
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
  const availableMetrics = metricNames(dataset.data);
  const hasAnyDatasetRows = Number(dataset.data?.summary?.row_count ?? 0) > 0;
  const requiresNarrowFilter = dataset.data?.summary?.requires_filter === true;
  const fileCount = dataset.data?.summary?.file_count;
  const hasPhysicalDatasetMetrics = Array.from(availableMetrics).some((name) => physicalDatasetMetrics.has(name));
  const expectedTimeSlotCount = Math.max(1, selectors?.time_indices?.length ?? 96);
  const visibleTimeSlotCount = distinctTimeIndexCount(dataset.data);
  const showPartialDateNotice =
    Boolean(filters.date) &&
    hasAnyDatasetRows &&
    filters.timeIndex === undefined &&
    filters.startTimeIndex === undefined &&
    filters.endTimeIndex === undefined &&
    visibleTimeSlotCount > 0 &&
    visibleTimeSlotCount < expectedTimeSlotCount;
  const panels = hasAnyDatasetRows && !hasPhysicalDatasetMetrics
    ? [{ title: '实时进度指标 / Live progress metrics', response: dataset.data }]
    : datasetPanels.map((panel) => ({
        title: panel.title,
        response: filterMetrics(dataset.data, panel.metrics)
      }));

  return (
    <div className="page-stack">
      <DataNotice loading={dataset.loading} error={dataset.error} />
      {requiresNarrowFilter ? (
        <div className="notice">
          当前数据集日志文件较多{typeof fileCount === 'number' ? `（${fileCount} 个 Parquet 文件）` : ''}；
          请先选择日期或 VPP 后再载入 dataset 曲线 / This run has many dataset log files; select a date or VPP before
          loading dataset curves.
        </div>
      ) : null}
      {hasAnyDatasetRows && !hasPhysicalDatasetMetrics ? (
        <div className="notice">
          当前 live run 尚未记录物理数据集信号；本视图显示 watchdog 进度指标 / Physical dataset signals
          are not logged yet for this live run; this view is showing watchdog progress metrics.
        </div>
      ) : null}
      {!filters.date && hasAnyDatasetRows ? (
        <div className="notice">
          当前显示全部日期；同一个 time_index 会跨多个日历日。若要比较同一时刻的不同 VPP，请先选择一个日期 /
          All dates are shown. The same time_index can span multiple calendar dates; select one date before comparing
          VPPs at the same time_index.
        </div>
      ) : null}
      {showPartialDateNotice ? (
        <div className="notice">
          当前日期只返回 {visibleTimeSlotCount} 个 time_index，预期完整日通常为 {expectedTimeSlotCount} 个；这通常表示该日期仍在当前
          episode 写入中，或当前筛选只包含进度采样行，所以曲线会显示成短线。不是电价、EV 负荷或成本数值过大导致隐藏 /
          This selected date currently contains {visibleTimeSlotCount} of {expectedTimeSlotCount} expected time slots. The date is likely
          still being written by the active episode, or the current filters only include progress-sampled rows.
        </div>
      ) : null}
      <MultiPanelChart panels={panels} />
      <MetricLineChart title="综合数据视图 / Combined dataset view" response={dataset.data} />
      <DatasetTimeseriesTable rows={dataset.data?.table_rows ?? []} units={dataset.data?.units ?? {}} />
      <MetricTable title="数据行 / Dataset rows" rows={dataset.data?.table_rows ?? []} />
    </div>
  );
}
