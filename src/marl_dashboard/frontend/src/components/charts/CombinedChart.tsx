import type { EChartsOption } from 'echarts';
import ReactECharts from 'echarts-for-react';

import type { ChartSeries, MetricRow, QueryResponse } from '../../api/types';
import { metricValue } from '../../utils/filters';

type Props = {
  title: string;
  responses: Array<{ label: string; response: QueryResponse | null }>;
};

function fallbackSeries(rows: MetricRow[]): ChartSeries[] {
  const byMetric = new Map<string, MetricRow[]>();
  for (const row of rows) {
    const key = [row.metric_name, row.vpp_id ?? '', row.policy_id ?? ''].join(' / ');
    const values = byMetric.get(key) ?? [];
    values.push(row);
    byMetric.set(key, values);
  }
  return Array.from(byMetric.entries()).map(([name, points]) => ({ name, points }));
}

function seriesFor(response: QueryResponse | null): ChartSeries[] {
  if (!response) {
    return [];
  }
  return response.chart_series.length > 0 ? response.chart_series : fallbackSeries(response.table_rows);
}

function padNumber(value: number | string | null | undefined): string {
  const numeric = Number(value);
  return Number.isFinite(numeric) ? String(numeric).padStart(8, '0') : String(value ?? '').padStart(8, '0');
}

function hasEnergyDataTime(row: MetricRow): boolean {
  return Boolean(row.date && row.time_index !== undefined && row.time_index !== null);
}

function pointLabel(row: MetricRow): string {
  if (hasEnergyDataTime(row)) {
    const label = `${row.date} #${row.time_index}`;
    return row.episode_id !== undefined && row.episode_id !== null ? `ep ${row.episode_id} ${label}` : label;
  }
  if (row.metric_group === 'loss' && row.gradient_step !== undefined && row.gradient_step !== null) {
    return `grad ${row.gradient_step}`;
  }
  if (row.global_env_step !== undefined && row.global_env_step !== null) {
    return `step ${row.global_env_step}`;
  }
  if (row.timestamp) {
    return `log ${row.timestamp.slice(0, 19).replace('T', ' ')}`;
  }
  return row.time_index !== undefined && row.time_index !== null ? `#${row.time_index}` : String(row.gradient_step ?? row.episode_id ?? '');
}

function pointSortKey(row: MetricRow): string {
  if (hasEnergyDataTime(row)) {
    return `0|${padNumber(row.episode_id)}|${row.date}|${padNumber(row.time_index)}|${padNumber(row.global_env_step)}`;
  }
  if (row.metric_group === 'loss' && row.gradient_step !== undefined && row.gradient_step !== null) {
    return `1|${padNumber(row.gradient_step)}|${padNumber(row.global_env_step)}`;
  }
  if (row.global_env_step !== undefined && row.global_env_step !== null) {
    return `2|${padNumber(row.global_env_step)}`;
  }
  if (row.timestamp) {
    const timestamp = Date.parse(row.timestamp);
    return `3|${Number.isFinite(timestamp) ? padNumber(timestamp) : row.timestamp}`;
  }
  return `4|${padNumber(row.time_index)}|${padNumber(row.gradient_step)}|${padNumber(row.episode_id)}`;
}

function alignedLabels(source: ChartSeries[]): string[] {
  const labels = new Map<string, string>();
  for (const series of source) {
    for (const point of series.points) {
      const label = pointLabel(point);
      if (!labels.has(label)) {
        labels.set(label, pointSortKey(point));
      }
    }
  }
  return Array.from(labels.entries())
    .sort((left, right) => left[1].localeCompare(right[1]))
    .map(([label]) => label);
}

function seriesDisplayName(series: ChartSeries): string {
  const first = series.points.find(Boolean);
  const metricLabel = first?.display_name || first?.description || series.metric_name || first?.metric_name || series.name;
  const vppId = series.vpp_id ?? first?.vpp_id;
  const policyId = series.policy_id ?? first?.policy_id;
  const suffix = [vppId, policyId].filter(Boolean).join(' / ');
  return suffix ? `${metricLabel} / ${suffix}` : metricLabel;
}

export function CombinedChart({ title, responses }: Props) {
  const source = responses.flatMap(({ label, response }) => seriesFor(response).map((item) => ({ label, item })));
  const labels = alignedLabels(source.map(({ item }) => item));
  const series = source.map(({ label, item }) => {
    const valueByLabel = new Map(item.points.map((row) => [pointLabel(row), metricValue(row.value)]));
    return {
      name: `${label}: ${seriesDisplayName(item)}`,
      type: 'line' as const,
      smooth: true,
      connectNulls: true,
      showSymbol: item.points.length <= 1,
      data: labels.map((axisLabel) => valueByLabel.get(axisLabel) ?? null)
    };
  });

  const option: EChartsOption = {
    color: ['#0f766e', '#b45309', '#be123c', '#2563eb', '#7c3aed', '#15803d'],
    title: { text: title, left: 8, top: 4, textStyle: { fontSize: 14, fontWeight: 650 } },
    tooltip: { trigger: 'axis' },
    legend: { top: 28, type: 'scroll' },
    grid: { left: 48, right: 24, top: 74, bottom: 42 },
    xAxis: { type: 'category', data: labels, axisLabel: { hideOverlap: true } },
    yAxis: { type: 'value', scale: true },
    series
  };

  return (
    <section className="panel chart-panel">
      <ReactECharts option={option} style={{ height: 340 }} notMerge lazyUpdate />
    </section>
  );
}
