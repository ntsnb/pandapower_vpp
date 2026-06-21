import { render, screen } from '@testing-library/react';
import { describe, expect, it, vi } from 'vitest';

import type { QueryResponse } from '../../api/types';
import { MetricLineChart } from './MetricLineChart';

vi.mock('echarts-for-react', () => ({
  default: ({ option }: { option: unknown }) => <pre data-testid="chart-option">{JSON.stringify(option)}</pre>
}));

describe('MetricLineChart', () => {
  it('enables zoom, brush selection, and image export controls for research plots', () => {
    const response: QueryResponse = {
      chart_series: [
        {
          name: 'electricity_price / vpp_001',
          metric_name: 'electricity_price',
          vpp_id: 'vpp_001',
          points: [
            {
              run_id: 'demo',
              metric_group: 'dataset',
              metric_name: 'electricity_price',
              display_name: '电价 / Electricity price',
              value: 42,
              unit: '$/MWh',
              date: '2026-01-01',
              time_index: 0,
              vpp_id: 'vpp_001'
            }
          ]
        }
      ],
      table_rows: [],
      units: { electricity_price: '$/MWh' },
      formulas: {},
      summary: { row_count: 1 }
    };

    render(<MetricLineChart title="Dataset" response={response} />);

    const option = JSON.parse(screen.getByTestId('chart-option').textContent ?? '{}') as {
      toolbox?: { feature?: Record<string, unknown> };
      dataZoom?: unknown[];
      brush?: unknown;
      yAxis?: { name?: string };
    };
    expect(option.toolbox?.feature).toHaveProperty('saveAsImage');
    expect(option.toolbox?.feature).toHaveProperty('dataZoom');
    expect(option.dataZoom?.length).toBeGreaterThanOrEqual(2);
    expect(option.brush).toBeTruthy();
    expect(option.yAxis?.name).toBe('$/MWh');
  });

  it('uses energy data date and numeric time_index before log timestamps on the x axis', () => {
    const response: QueryResponse = {
      chart_series: [
        {
          name: 'net_load / vpp_001',
          metric_name: 'net_load',
          vpp_id: 'vpp_001',
          points: [
            {
              run_id: 'run_a',
              metric_group: 'dataset',
              metric_name: 'net_load',
              value: 86,
              unit: 'MW',
              date: '2018-01-01',
              time_index: 86,
              timestamp: '2026-06-16T12:00:00Z',
              episode_id: 1,
              vpp_id: 'vpp_001'
            },
            {
              run_id: 'run_a',
              metric_group: 'dataset',
              metric_name: 'net_load',
              value: 43,
              unit: 'MW',
              date: '2018-01-01',
              time_index: 43,
              timestamp: '2026-06-16T11:00:00Z',
              episode_id: 1,
              vpp_id: 'vpp_001'
            },
            {
              run_id: 'run_a',
              metric_group: 'dataset',
              metric_name: 'net_load',
              value: 0,
              unit: 'MW',
              date: '2018-01-01',
              time_index: 0,
              timestamp: '2026-06-16T10:00:00Z',
              episode_id: 1,
              vpp_id: 'vpp_001'
            }
          ]
        }
      ],
      table_rows: [],
      units: { net_load: 'MW' },
      formulas: {},
      summary: { row_count: 3 }
    };

    render(<MetricLineChart title="Dataset" response={response} />);

    const option = JSON.parse(screen.getByTestId('chart-option').textContent ?? '{}') as {
      xAxis?: { data?: string[] };
      series?: Array<{ data?: Array<number | null> }>;
    };
    expect(option.xAxis?.data).toEqual([
      'ep 1 2018-01-01 #0',
      'ep 1 2018-01-01 #43',
      'ep 1 2018-01-01 #86'
    ]);
    expect(option.xAxis?.data?.join(' ')).not.toContain('2026-06-16');
    expect(option.series?.[0]?.data).toEqual([0, 43, 86]);
  });

  it('connects sparse dataset series across labels owned by other VPP or policy series', () => {
    const response: QueryResponse = {
      chart_series: [
        {
          name: 'electricity_price / vpp_001 / happo',
          metric_name: 'electricity_price',
          vpp_id: 'vpp_001',
          policy_id: 'happo',
          points: [
            {
              run_id: 'run_a',
              metric_group: 'dataset',
              metric_name: 'electricity_price',
              value: 50,
              date: '2018-01-01',
              time_index: 0,
              episode_id: 1,
              vpp_id: 'vpp_001',
              policy_id: 'happo'
            },
            {
              run_id: 'run_a',
              metric_group: 'dataset',
              metric_name: 'electricity_price',
              value: 60,
              date: '2018-01-01',
              time_index: 8,
              episode_id: 1,
              vpp_id: 'vpp_001',
              policy_id: 'happo'
            }
          ]
        },
        {
          name: 'electricity_price / vpp_001 / happo_sequential_ctde',
          metric_name: 'electricity_price',
          vpp_id: 'vpp_001',
          policy_id: 'happo_sequential_ctde',
          points: [
            {
              run_id: 'run_a',
              metric_group: 'dataset',
              metric_name: 'electricity_price',
              value: 55,
              date: '2018-01-01',
              time_index: 4,
              episode_id: 1,
              vpp_id: 'vpp_001',
              policy_id: 'happo_sequential_ctde'
            }
          ]
        }
      ],
      table_rows: [],
      units: { electricity_price: '$/MWh' },
      formulas: {},
      summary: { row_count: 3 }
    };

    render(<MetricLineChart title="电价 / Electricity price" response={response} />);

    const option = JSON.parse(screen.getByTestId('chart-option').textContent ?? '{}') as {
      series?: Array<{ connectNulls?: boolean; data?: Array<number | null> }>;
    };
    expect(option.series?.[0]?.data).toEqual([50, null, 60]);
    expect(option.series?.[1]?.data).toEqual([null, 55, null]);
    expect(option.series?.every((series) => series.connectNulls === true)).toBe(true);
  });
});
