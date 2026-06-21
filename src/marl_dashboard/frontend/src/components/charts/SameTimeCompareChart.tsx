import type { EChartsOption } from 'echarts';
import ReactECharts from 'echarts-for-react';

import type { QueryResponse } from '../../api/types';
import { metricValue } from '../../utils/filters';

type Props = {
  response: QueryResponse | null;
};

export function SameTimeCompareChart({ response }: Props) {
  const rows = response?.table_rows ?? [];
  const groupBy = typeof response?.summary?.group_by === 'string' ? response.summary.group_by : undefined;
  const metricLabel = (metricName: string): string => {
    const row = rows.find((candidate) => candidate.metric_name === metricName);
    return row?.display_name || row?.description || metricName;
  };
  const rowLabel = (row: (typeof rows)[number], index: number): string => {
    const groupedValue = groupBy ? row[groupBy as keyof typeof row] : undefined;
    return String(row.group ?? groupedValue ?? row.vpp_id ?? row.agent_id ?? row.policy_id ?? row.epoch_id ?? `Row ${index + 1}`);
  };
  const labels: string[] = [];
  const seenLabels = new Set<string>();
  rows.forEach((row, index) => {
    const label = rowLabel(row, index);
    if (!seenLabels.has(label)) {
      labels.push(label);
      seenLabels.add(label);
    }
  });
  const metricNames = Array.from(new Set(rows.map((row) => row.metric_name).filter(Boolean))).sort((left, right) =>
    left.localeCompare(right)
  );
  const values = new Map<string, number | null>();
  rows.forEach((row, index) => {
    values.set(`${row.metric_name}\n${rowLabel(row, index)}`, metricValue(row.value));
  });
  const ranges = new Map<string, { min: number; max: number }>();
  metricNames.forEach((metricName) => {
    const metricValues = labels
      .map((label) => values.get(`${metricName}\n${label}`))
      .filter((value): value is number => value !== null && value !== undefined && Number.isFinite(value));
    if (metricValues.length === 0) {
      ranges.set(metricName, { min: 0, max: 0 });
      return;
    }
    ranges.set(metricName, { min: Math.min(...metricValues), max: Math.max(...metricValues) });
  });
  const normalizedValue = (metricName: string, label: string): number | null => {
    const value = values.get(`${metricName}\n${label}`);
    if (value === null || value === undefined || !Number.isFinite(value)) {
      return null;
    }
    const range = ranges.get(metricName);
    if (!range || Math.abs(range.max - range.min) < 1e-12) {
      return 1;
    }
    return (value - range.min) / (range.max - range.min);
  };
  const heatmapData: Array<[number, number, number | null]> = [];
  labels.forEach((label, labelIndex) => {
    metricNames.forEach((metricName, metricIndex) => {
      heatmapData.push([labelIndex, metricIndex, normalizedValue(metricName, label)]);
    });
  });
  const radarIndicators = metricNames.map((metricName) => ({ name: metricLabel(metricName), max: 1 }));
  const radarData = labels.map((label) => ({
    name: label,
    value: metricNames.map((metricName) => normalizedValue(metricName, label) ?? 0)
  }));
  const option: EChartsOption = {
    color: ['#0f766e', '#b45309', '#be123c', '#2563eb', '#7c3aed', '#15803d'],
    title: [
      { text: '同一时刻 VPP 对比 / Same-time VPP comparison', left: 8, top: 4, textStyle: { fontSize: 14, fontWeight: 650 } },
      { text: '指标矩阵热力图 / Metric heatmap', left: 8, top: 360, textStyle: { fontSize: 14, fontWeight: 650 } },
      { text: '雷达图 / Radar comparison', left: '58%', top: 360, textStyle: { fontSize: 14, fontWeight: 650 } }
    ],
    tooltip: { trigger: 'axis' },
    legend: { top: 28, type: 'scroll' },
    grid: [
      { left: 48, right: 24, top: 72, height: 220 },
      { left: 54, right: '52%', top: 410, height: 220 }
    ],
    xAxis: [
      { type: 'category', data: labels, gridIndex: 0 },
      { type: 'category', data: labels, gridIndex: 1, axisLabel: { rotate: 35 } }
    ],
    yAxis: [
      { type: 'value', scale: true, gridIndex: 0 },
      { type: 'category', data: metricNames.map(metricLabel), gridIndex: 1 }
    ],
    visualMap: {
      min: 0,
      max: 1,
      calculable: true,
      orient: 'horizontal',
      left: 54,
      bottom: 16,
      inRange: { color: ['#f8fafc', '#99f6e4', '#0f766e'] }
    },
    radar: {
      indicator: radarIndicators,
      center: ['72%', '71%'],
      radius: 95
    },
    series: [
      ...metricNames.map((metricName) => ({
        name: metricLabel(metricName),
        type: 'bar' as const,
        xAxisIndex: 0,
        yAxisIndex: 0,
        data: labels.map((label) => values.get(`${metricName}\n${label}`) ?? null)
      })),
      {
        name: '归一化指标热力图 / Normalized metric heatmap',
        type: 'heatmap' as const,
        xAxisIndex: 1,
        yAxisIndex: 1,
        data: heatmapData,
        emphasis: { itemStyle: { shadowBlur: 6, shadowColor: 'rgba(15, 118, 110, 0.35)' } }
      },
      {
        name: '归一化雷达图 / Normalized radar',
        type: 'radar' as const,
        data: radarData
      }
    ]
  };

  return (
    <section className="panel chart-panel">
      <ReactECharts option={option} style={{ height: 680 }} notMerge lazyUpdate />
    </section>
  );
}
