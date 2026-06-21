import { render, screen } from '@testing-library/react';
import { describe, expect, it, vi } from 'vitest';

import { api } from '../api/client';
import type { QueryResponse } from '../api/types';
import { DatasetPage } from './DatasetPage';

vi.mock('../api/hooks', () => ({
  useAsync: (loader: () => Promise<QueryResponse>, deps: readonly unknown[]) => {
    observedAsyncDeps.push([...deps]);
    void loader();
    return { loading: false, error: null, data: response };
  },
  useLiveTick: () => 0
}));

vi.mock('../api/client', () => ({
  api: {
    dataset: vi.fn(() => Promise.resolve(response))
  }
}));

vi.mock('../components/charts/MetricLineChart', () => ({
  MetricLineChart: ({ title }: { title: string }) => <section aria-label={title}>{title}</section>
}));

let response: QueryResponse = {
  chart_series: [],
  table_rows: [],
  units: {},
  formulas: {},
  summary: { row_count: 0 }
};
let observedAsyncDeps: unknown[][] = [];

describe('DatasetPage', () => {
  it('reloads when a live websocket event arrives', () => {
    vi.mocked(api.dataset).mockClear();
    observedAsyncDeps = [];
    response = {
      chart_series: [],
      table_rows: [],
      units: {},
      formulas: {},
      summary: { row_count: 0 }
    };
    const { rerender } = render(
      <DatasetPage
        run={null}
        selectors={null}
        liveEventCount={0}
        filters={{ runId: 'run_a', live: true, compareMode: false, date: '2026-01-01', vppId: 'vpp_001', epochId: 0 }}
      />
    );

    rerender(
      <DatasetPage
        run={null}
        selectors={null}
        liveEventCount={1}
        filters={{ runId: 'run_a', live: true, compareMode: false, date: '2026-01-01', vppId: 'vpp_001', epochId: 0 }}
      />
    );

    expect(observedAsyncDeps.at(-1)).toContain(1);
  });

  it('renders the six required dataset panels plus a combined view', () => {
    response = {
      chart_series: [],
      table_rows: [],
      units: {},
      formulas: {},
      summary: { row_count: 0 }
    };
    render(
      <DatasetPage
        run={null}
        selectors={null}
        filters={{ runId: 'run_a', live: true, compareMode: false, date: '2026-01-01', vppId: 'vpp_001', epochId: 0 }}
      />
    );

    for (const title of [
      'Electricity price',
      'EV charging load',
      'Storage power and SOC',
      'PV power',
      'Wind power',
      'Load and net load',
      'VPP dispatch and net delivery',
      'Action landing and policy outputs',
      'Combined dataset view'
    ]) {
      expect(screen.getByLabelText(new RegExp(title, 'i'))).toBeInTheDocument();
    }
  });

  it('renders currently available live progress metrics when physical dataset signals are absent', () => {
    response = {
      chart_series: [
        {
          name: 'progress_rows / aggregate / happo',
          metric_name: 'progress_rows',
          vpp_id: 'aggregate',
          policy_id: 'happo',
          points: [
            {
              run_id: 'run_a',
              metric_group: 'dataset',
              metric_name: 'progress_rows',
              value: 12,
              unit: 'count',
              time_index: 24,
              vpp_id: 'aggregate',
              policy_id: 'happo'
            }
          ]
        }
      ],
      table_rows: [],
      units: { progress_rows: 'count' },
      formulas: {},
      summary: { row_count: 1 }
    };

    render(
      <DatasetPage
        run={null}
        selectors={null}
        filters={{ runId: 'run_a', live: true, compareMode: false, epochId: 0 }}
      />
    );

    expect(screen.getByLabelText(/Live progress metrics/i)).toBeInTheDocument();
    expect(screen.getByText(/physical dataset signals are not logged yet/i)).toBeInTheDocument();
  });

  it('explains when a large live dataset query needs date or VPP filters', () => {
    response = {
      chart_series: [],
      table_rows: [],
      units: {},
      formulas: {},
      summary: { row_count: 0, requires_filter: true, reason: 'large_unfiltered_dataset_query', file_count: 7284 }
    };

    render(
      <DatasetPage
        run={null}
        selectors={null}
        filters={{ runId: 'run_a', live: true, compareMode: false }}
      />
    );

    expect(screen.getByText(/当前数据集日志文件较多/)).toBeInTheDocument();
    expect(screen.getByText(/select a date or VPP before loading dataset curves/i)).toBeInTheDocument();
  });

  it('explains that leaving date unset shows all available dates', () => {
    response = {
      chart_series: [],
      table_rows: [
        {
          run_id: 'run_a',
          metric_group: 'dataset',
          metric_name: 'electricity_price',
          display_name: '电价 / Electricity price',
          value: 63.4,
          unit: 'currency/MWh',
          date: '2018-01-02',
          time_index: 1,
          vpp_id: 'vpp_a'
        }
      ],
      units: { electricity_price: 'currency/MWh' },
      formulas: {},
      summary: { row_count: 1 }
    };

    render(
      <DatasetPage
        run={null}
        selectors={null}
        filters={{ runId: 'run_a', live: true, compareMode: false, vppId: 'vpp_a' }}
      />
    );

    expect(screen.getByText(/当前显示全部日期/)).toBeInTheDocument();
    expect(screen.getByText(/select one date before comparing VPPs at the same time_index/i)).toBeInTheDocument();
  });

  it('explains when the selected date only has a partial set of time slots', () => {
    vi.mocked(api.dataset).mockClear();
    response = {
      chart_series: [
        {
          name: 'electricity_price / vpp_a / happo',
          metric_name: 'electricity_price',
          vpp_id: 'vpp_a',
          policy_id: 'happo',
          points: [
            {
              run_id: 'run_a',
              metric_group: 'dataset',
              metric_name: 'electricity_price',
              display_name: '电价 / Electricity price',
              value: 63.4,
              unit: 'currency/MWh',
              date: '2018-01-08',
              time_index: 0,
              vpp_id: 'vpp_a',
              policy_id: 'happo'
            }
          ]
        }
      ],
      table_rows: [],
      units: { electricity_price: 'currency/MWh' },
      formulas: {},
      summary: { row_count: 1 }
    };

    render(
      <DatasetPage
        run={null}
        selectors={{
          run_id: 'run_a',
          dates: ['2018-01-08'],
          vpp_ids: ['vpp_a'],
          agent_ids: [],
          policy_ids: ['happo'],
          epoch_ids: [0],
          episode_ids: [6],
          time_indices: Array.from({ length: 96 }, (_, index) => index)
        }}
        filters={{ runId: 'run_a', live: true, compareMode: false, date: '2018-01-08', vppId: 'vpp_a', epochId: 0 }}
      />
    );

    expect(screen.getByText(/当前日期只返回 1 个 time_index/)).toBeInTheDocument();
    expect(screen.getByText(/This selected date currently contains 1 of 96 expected time slots/i)).toBeInTheDocument();
    expect(vi.mocked(api.dataset).mock.calls.at(-1)?.[1]).toMatchObject({ max_points: 30000 });
  });

  it('renders a timeseries wide table with one row per time slot and metric columns with units', () => {
    response = {
      chart_series: [],
      table_rows: [
        {
          run_id: 'run_a',
          metric_group: 'dataset',
          metric_name: 'electricity_price',
          display_name: '电价 / Electricity price',
          value: 63.4,
          unit: 'currency/MWh',
          date: '2018-01-02',
          time_index: 1,
          timestamp: '2018-01-02T00:15:00Z',
          vpp_id: 'vpp_a'
        },
        {
          run_id: 'run_a',
          metric_group: 'dataset',
          metric_name: 'pv_power',
          display_name: '光伏出力 / PV power',
          value: 0.42,
          unit: 'MW',
          date: '2018-01-02',
          time_index: 1,
          timestamp: '2018-01-02T00:15:00Z',
          vpp_id: 'vpp_a'
        },
        {
          run_id: 'run_a',
          metric_group: 'dataset',
          metric_name: 'electricity_price',
          display_name: '电价 / Electricity price',
          value: 64.1,
          unit: 'currency/MWh',
          date: '2018-01-02',
          time_index: 2,
          timestamp: '2018-01-02T00:30:00Z',
          vpp_id: 'vpp_a'
        }
      ],
      units: { electricity_price: 'currency/MWh', pv_power: 'MW' },
      formulas: {},
      summary: { row_count: 3 }
    };

    render(
      <DatasetPage
        run={null}
        selectors={null}
        filters={{ runId: 'run_a', live: false, compareMode: false, date: '2018-01-02', vppId: 'vpp_a' }}
      />
    );

    expect(screen.getByText('逐时刻宽表 / Timeseries wide table')).toBeInTheDocument();
    expect(screen.getByText('电价 / Electricity price (currency/MWh)')).toBeInTheDocument();
    expect(screen.getByText('光伏出力 / PV power (MW)')).toBeInTheDocument();
    expect(screen.getByText('2018-01-02 #1')).toBeInTheDocument();
    expect(screen.getAllByText('63.4').length).toBeGreaterThan(0);
    expect(screen.getAllByText('0.42').length).toBeGreaterThan(0);
  });
});
