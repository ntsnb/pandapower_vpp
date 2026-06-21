import type { EChartsOption } from 'echarts';
import ReactECharts from 'echarts-for-react';

import type { MetricRow, QueryResponse } from '../../api/types';
import { metricValue } from '../../utils/filters';

type Props = {
  title: string;
  response: QueryResponse | null;
};

function latestByMetric(rows: MetricRow[]): MetricRow[] {
  const latest = new Map<string, MetricRow>();
  for (const row of rows) {
    latest.set(row.metric_name, row);
  }
  return Array.from(latest.values());
}

export function CompositionChart({ title, response }: Props) {
  const rows = latestByMetric(response?.table_rows ?? []);
  if (rows.length === 0) {
    return (
      <section className="panel chart-panel">
        <div className="chart-empty-state">
          <h2>{title}</h2>
          <p>
            当前筛选没有组成数据。请检查所选 VPP、日期、episode、policy 和 step 是否同时存在对应指标 /
            No composition rows exist for the current VPP, date, episode, policy, and step filters.
          </p>
        </div>
      </section>
    );
  }
  const option: EChartsOption = {
    color: ['#0f766e', '#b45309', '#be123c', '#2563eb', '#7c3aed', '#15803d'],
    title: { text: title, left: 8, top: 4, textStyle: { fontSize: 14, fontWeight: 650 } },
    tooltip: { trigger: 'axis' },
    grid: { left: 92, right: 18, top: 52, bottom: 36 },
    xAxis: { type: 'value' },
    yAxis: {
      type: 'category',
      data: rows.map((row) => row.metric_name),
      axisLabel: { width: 84, overflow: 'truncate' }
    },
    series: [
      {
        type: 'bar',
        data: rows.map((row) => metricValue(row.value)),
        label: { show: true, position: 'right', formatter: '{c}' }
      }
    ]
  };

  return (
    <section className="panel chart-panel">
      <ReactECharts option={option} style={{ height: 300 }} notMerge lazyUpdate />
    </section>
  );
}
