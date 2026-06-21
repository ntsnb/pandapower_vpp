import { render, screen } from '@testing-library/react';
import { describe, expect, it, vi } from 'vitest';

import type { QueryResponse } from '../../api/types';
import { SameTimeCompareChart } from './SameTimeCompareChart';

vi.mock('echarts-for-react', () => ({
  default: ({ option }: { option: unknown }) => <pre data-testid="chart-option">{JSON.stringify(option)}</pre>
}));

describe('SameTimeCompareChart', () => {
  it('uses the compare group label instead of falling back to unknown', () => {
    const response: QueryResponse = {
      chart_series: [],
      table_rows: [
        {
          run_id: 'run_a',
          metric_group: 'reward',
          metric_name: 'reward_so_far',
          value: 12.5,
          unit: 'score',
          group: 'aggregate'
        }
      ],
      units: { reward_so_far: 'score' },
      formulas: {},
      summary: { row_count: 1, group_by: 'vpp_id' }
    };

    render(<SameTimeCompareChart response={response} />);

    const option = screen.getByTestId('chart-option').textContent ?? '';
    expect(option).toContain('aggregate');
    expect(option).not.toContain('unknown');
  });

  it('renders one bilingual bar series per metric for matrix-style comparisons', () => {
    const response: QueryResponse = {
      chart_series: [],
      table_rows: [
        {
          run_id: 'run_a',
          metric_group: 'dataset',
          metric_name: 'electricity_price',
          display_name: '电价 / Electricity price',
          value: 82,
          unit: 'currency/MWh',
          group: 'vpp_001',
          vpp_id: 'vpp_001'
        },
        {
          run_id: 'run_a',
          metric_group: 'dataset',
          metric_name: 'net_load',
          description: '净负荷 / Net load',
          value: 4.2,
          unit: 'MW',
          group: 'vpp_001',
          vpp_id: 'vpp_001'
        }
      ],
      units: { electricity_price: 'currency/MWh', net_load: 'MW' },
      formulas: {},
      summary: { row_count: 2, group_by: 'vpp_id' }
    };

    render(<SameTimeCompareChart response={response} />);

    const option = JSON.parse(screen.getByTestId('chart-option').textContent ?? '{}') as {
      title: Array<{ text: string }>;
      series: Array<{ name: string; type: string }>;
    };
    expect(option.title[0].text).toBe('同一时刻 VPP 对比 / Same-time VPP comparison');
    expect(option.series.filter((series) => series.type === 'bar').map((series) => series.name)).toEqual([
      '电价 / Electricity price',
      '净负荷 / Net load'
    ]);
  });

  it('adds normalized heatmap and radar series for paper comparison views', () => {
    const response: QueryResponse = {
      chart_series: [],
      table_rows: [
        {
          run_id: 'run_a',
          metric_group: 'dataset',
          metric_name: 'electricity_price',
          display_name: '电价 / Electricity price',
          value: 80,
          unit: 'currency/MWh',
          group: 'vpp_001',
          vpp_id: 'vpp_001'
        },
        {
          run_id: 'run_a',
          metric_group: 'dataset',
          metric_name: 'net_load',
          display_name: '净负荷 / Net load',
          value: 20,
          unit: 'MW',
          group: 'vpp_001',
          vpp_id: 'vpp_001'
        },
        {
          run_id: 'run_a',
          metric_group: 'dataset',
          metric_name: 'electricity_price',
          display_name: '电价 / Electricity price',
          value: 40,
          unit: 'currency/MWh',
          group: 'vpp_002',
          vpp_id: 'vpp_002'
        },
        {
          run_id: 'run_a',
          metric_group: 'dataset',
          metric_name: 'net_load',
          display_name: '净负荷 / Net load',
          value: 10,
          unit: 'MW',
          group: 'vpp_002',
          vpp_id: 'vpp_002'
        }
      ],
      units: { electricity_price: 'currency/MWh', net_load: 'MW' },
      formulas: {},
      summary: { row_count: 4, group_by: 'vpp_id' }
    };

    render(<SameTimeCompareChart response={response} />);

    const option = JSON.parse(screen.getByTestId('chart-option').textContent ?? '{}') as {
      title: Array<{ text: string }>;
      visualMap: { min: number; max: number };
      series: Array<{ name: string; type: string; data: unknown[] }>;
    };
    expect(option.title.map((title) => title.text)).toContain('指标矩阵热力图 / Metric heatmap');
    expect(option.title.map((title) => title.text)).toContain('雷达图 / Radar comparison');
    expect(option.series.some((series) => series.type === 'heatmap' && series.name === '归一化指标热力图 / Normalized metric heatmap')).toBe(
      true
    );
    expect(option.series.some((series) => series.type === 'radar' && series.name === '归一化雷达图 / Normalized radar')).toBe(true);
    expect(option.visualMap).toMatchObject({ min: 0, max: 1 });
  });
});
