import type { EChartsOption } from 'echarts';
import ReactECharts from 'echarts-for-react';

import type { ChartSeries, MetricRow, QueryResponse } from '../../api/types';
import { metricValue } from '../../utils/filters';

type Props = {
  title: string;
  response: QueryResponse | null;
  height?: number;
};

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

function fallbackSeries(rows: MetricRow[]): ChartSeries[] {
  const byMetric = new Map<string, MetricRow[]>();
  for (const row of rows) {
    const key = [row.metric_name, row.vpp_id ?? '', row.policy_id ?? ''].join(' | ');
    const values = byMetric.get(key) ?? [];
    values.push(row);
    byMetric.set(key, values);
  }
  return Array.from(byMetric.entries()).map(([name, points]) => ({ name, points }));
}

function seriesDisplayName(series: ChartSeries): string {
  const first = series.points.find(Boolean);
  const metricLabel = first?.display_name || first?.description || series.metric_name || first?.metric_name || series.name;
  const vppId = series.vpp_id ?? first?.vpp_id;
  const policyId = series.policy_id ?? first?.policy_id;
  const suffix = [vppId, policyId].filter(Boolean).join(' / ');
  return suffix ? `${metricLabel} / ${suffix}` : metricLabel;
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

function unitLabel(response: QueryResponse | null, source: ChartSeries[]): string {
  const units = new Set<string>();
  for (const unit of Object.values(response?.units ?? {})) {
    if (unit) {
      units.add(String(unit));
    }
  }
  for (const series of source) {
    for (const point of series.points) {
      if (point.unit) {
        units.add(String(point.unit));
      }
    }
  }
  return units.size === 1 ? Array.from(units)[0] : '';
}

function makeOption(title: string, response: QueryResponse | null): EChartsOption {
  const source = response ? (response.chart_series.length > 0 ? response.chart_series : fallbackSeries(response.table_rows)) : [];
  const labels = alignedLabels(source);
  const unit = unitLabel(response, source);

  return {
    color: ['#0f766e', '#b45309', '#be123c', '#2563eb', '#7c3aed', '#15803d'],
    title: { text: title, left: 8, top: 4, textStyle: { fontSize: 14, fontWeight: 650 } },
    tooltip: { trigger: 'axis' },
    legend: { top: 28, type: 'scroll' },
    toolbox: {
      right: 12,
      top: 4,
      feature: {
        dataZoom: { yAxisIndex: 'none', title: { zoom: '缩放 / Zoom', back: '还原缩放 / Reset zoom' } },
        brush: { type: ['rect', 'lineX', 'clear'], title: { rect: '框选 / Box select', lineX: '横向框选 / X select', clear: '清除选择 / Clear' } },
        saveAsImage: { title: '导出图片 / Export image', pixelRatio: 2 }
      }
    },
    brush: { toolbox: ['rect', 'lineX', 'clear'], xAxisIndex: 'all' },
    dataZoom: [
      { type: 'inside', xAxisIndex: 0, filterMode: 'none' },
      { type: 'slider', xAxisIndex: 0, height: 18, bottom: 12, filterMode: 'none' }
    ],
    grid: { left: 56, right: 32, top: 76, bottom: 64 },
    xAxis: { type: 'category', data: labels, axisLabel: { hideOverlap: true } },
    yAxis: { type: 'value', scale: true, name: unit, nameGap: 36 },
    series: source.map((series) => {
      const valueByLabel = new Map(series.points.map((row) => [pointLabel(row), metricValue(row.value)]));
      return {
        name: seriesDisplayName(series),
        type: 'line',
        smooth: true,
        connectNulls: true,
        showSymbol: series.points.length <= 1,
        emphasis: { focus: 'series' },
        data: labels.map((label) => valueByLabel.get(label) ?? null)
      };
    })
  };
}

export function MetricLineChart({ title, response, height = 320 }: Props) {
  return (
    <section className="panel chart-panel">
      <ReactECharts option={makeOption(title, response)} style={{ height }} notMerge lazyUpdate />
    </section>
  );
}
