import { fireEvent, render, screen, waitFor } from '@testing-library/react';
import { describe, expect, it, vi } from 'vitest';

import type { QueryResponse } from '../api/types';
import { FlexibleComparePage } from './FlexibleComparePage';

const datasetResponse: QueryResponse = {
  chart_series: [
    {
      name: 'electricity_price / vpp_001',
      metric_name: 'electricity_price',
      vpp_id: 'vpp_001',
      points: [
        {
          run_id: 'run_a',
          metric_group: 'dataset',
          metric_name: 'electricity_price',
          display_name: '电价 / Electricity price',
          value: 63,
          unit: 'currency/MWh',
          date: '2018-01-02',
          time_index: 1,
          vpp_id: 'vpp_001'
        }
      ]
    }
  ],
  table_rows: [],
  units: { electricity_price: 'currency/MWh' },
  formulas: {},
  summary: { row_count: 1 }
};

const costResponse: QueryResponse = {
  chart_series: [
    {
      name: 'der_operation_cost / vpp_002',
      metric_name: 'der_operation_cost',
      vpp_id: 'vpp_002',
      points: [
        {
          run_id: 'run_a',
          metric_group: 'cost',
          metric_name: 'der_operation_cost',
          display_name: 'DER 运行成本 / DER operation cost',
          value: 2.1,
          unit: 'currency',
          date: '2018-01-02',
          time_index: 1,
          vpp_id: 'vpp_002'
        }
      ]
    }
  ],
  table_rows: [],
  units: { der_operation_cost: 'currency' },
  formulas: {},
  summary: { row_count: 1 }
};

const mocks = vi.hoisted(() => ({
  dataset: vi.fn(),
  rewards: vi.fn(),
  costs: vi.fn(),
  losses: vi.fn()
}));

vi.mock('../api/client', () => ({
  api: mocks
}));

vi.mock('../components/charts/CombinedChart', () => ({
  CombinedChart: ({
    title,
    responses
  }: {
    title: string;
    responses: Array<{ label: string; response: QueryResponse | null }>;
  }) => (
    <section aria-label={title}>
      {responses.map(({ label, response }) => (
        <span key={label}>
          {label}:{response?.summary?.row_count ?? 0}
        </span>
      ))}
    </section>
  )
}));

describe('FlexibleComparePage', () => {
  it('explains the difference between energy data time and training log time', async () => {
    mocks.dataset.mockResolvedValue(datasetResponse);
    mocks.rewards.mockResolvedValue({ chart_series: [], table_rows: [], units: {}, formulas: {}, summary: { row_count: 0 } });
    mocks.costs.mockResolvedValue(costResponse);
    mocks.losses.mockResolvedValue({ chart_series: [], table_rows: [], units: {}, formulas: {}, summary: { row_count: 0 } });

    render(
      <FlexibleComparePage
        run={null}
        selectors={{
          run_id: 'run_a',
          dates: ['2018-01-01'],
          vpp_ids: ['vpp_001'],
          agent_ids: [],
          policy_ids: ['happo'],
          epoch_ids: [0],
          episode_ids: [1],
          time_indices: [0, 1]
        }}
        filters={{ runId: 'run_a', live: false, compareMode: false }}
      />
    );

    expect(screen.getByText(/数据时间 \/ Data time/)).toBeInTheDocument();
    expect(screen.getByText(/日志写入时间 \/ Log write time/)).toBeInTheDocument();
    expect(screen.getByText(/横坐标优先使用 date \+ time_index/)).toBeInTheDocument();
    await waitFor(() => expect(screen.getByRole('option', { name: '电价 / Electricity price' })).toBeInTheDocument());
    await waitFor(() => expect(screen.queryByText('Loading')).not.toBeInTheDocument());
  });

  it('adds multiple independently filtered curves into one aligned chart', async () => {
    mocks.dataset.mockResolvedValue(datasetResponse);
    mocks.rewards.mockResolvedValue({ chart_series: [], table_rows: [], units: {}, formulas: {}, summary: { row_count: 0 } });
    mocks.costs.mockResolvedValue(costResponse);
    mocks.losses.mockResolvedValue({ chart_series: [], table_rows: [], units: {}, formulas: {}, summary: { row_count: 0 } });

    render(
      <FlexibleComparePage
        run={null}
        selectors={{
          run_id: 'run_a',
          dates: ['2018-01-01', '2018-01-02'],
          vpp_ids: ['vpp_001', 'vpp_002'],
          agent_ids: [],
          policy_ids: ['happo'],
          epoch_ids: [0, 1],
          episode_ids: [1, 2],
          time_indices: [0, 1, 2]
        }}
        filters={{ runId: 'run_a', live: false, compareMode: false }}
      />
    );

    await waitFor(() => expect(screen.getByRole('option', { name: '电价 / Electricity price' })).toBeInTheDocument());
    fireEvent.change(screen.getByLabelText('指标 / Metric'), { target: { value: 'electricity_price' } });
    fireEvent.change(screen.getByLabelText('VPP'), { target: { value: 'vpp_001' } });
    fireEvent.change(screen.getByLabelText('日期 / Date'), { target: { value: '2018-01-02' } });
    fireEvent.change(screen.getByLabelText('轨迹 / Episode'), { target: { value: '2' } });
    fireEvent.click(screen.getByRole('button', { name: '添加曲线 / Add curve' }));

    await waitFor(() =>
      expect(mocks.dataset).toHaveBeenCalledWith(
        'run_a',
        expect.objectContaining({
          metrics: 'electricity_price',
          episode_id: 2,
          max_points: 1200
        })
      )
    );
    expect(screen.getByText('episode 2')).toBeInTheDocument();

    fireEvent.change(screen.getByLabelText('数据域 / Data scope'), { target: { value: 'cost' } });
    await waitFor(() => expect(screen.getByRole('option', { name: 'DER 运行成本 / DER operation cost' })).toBeInTheDocument());
    fireEvent.change(screen.getByLabelText('指标 / Metric'), { target: { value: 'der_operation_cost' } });
    fireEvent.change(screen.getByLabelText('VPP'), { target: { value: 'vpp_002' } });
    fireEvent.click(screen.getByRole('button', { name: '添加曲线 / Add curve' }));

    await waitFor(() =>
      expect(mocks.costs).toHaveBeenCalledWith(
        'run_a',
        expect.objectContaining({
          metrics: 'der_operation_cost',
          episode_id: 2,
          max_points: 1200
        })
      )
    );

    await waitFor(() => expect(screen.getAllByText(/数据集 \/ Dataset.*电价 \/ Electricity price.*vpp_001/).length).toBeGreaterThan(0));
    expect(screen.getAllByText(/成本 \/ Cost.*DER 运行成本 \/ DER operation cost.*vpp_002/).length).toBeGreaterThan(0);
    expect(screen.getByText(/数据集 \/ Dataset.*:1/)).toBeInTheDocument();
    expect(screen.getByText(/成本 \/ Cost.*:1/)).toBeInTheDocument();
  });

  it('labels partial dates in the flexible curve picker', async () => {
    mocks.dataset.mockResolvedValue(datasetResponse);
    mocks.rewards.mockResolvedValue({ chart_series: [], table_rows: [], units: {}, formulas: {}, summary: { row_count: 0 } });
    mocks.costs.mockResolvedValue(costResponse);
    mocks.losses.mockResolvedValue({ chart_series: [], table_rows: [], units: {}, formulas: {}, summary: { row_count: 0 } });

    render(
      <FlexibleComparePage
        run={null}
        selectors={{
          run_id: 'run_a',
          dates: ['2018-01-01', '2018-01-08'],
          date_statuses: [
            {
              date: '2018-01-01',
              observed_time_slots: 96,
              expected_time_slots: 96,
              complete: true,
              status: 'complete'
            },
            {
              date: '2018-01-08',
              observed_time_slots: 1,
              expected_time_slots: 96,
              complete: false,
              status: 'partial'
            }
          ],
          vpp_ids: ['vpp_001'],
          agent_ids: [],
          policy_ids: ['happo_sequential_ctde'],
          epoch_ids: [0],
          episode_ids: [1],
          time_indices: [0, 1]
        }}
        filters={{ runId: 'run_a', live: false, compareMode: false }}
      />
    );

    expect(screen.getByRole('option', { name: '2018-01-08 (未满 1/96 / Partial)' })).toBeInTheDocument();
    await waitFor(() => expect(screen.queryByText('Loading')).not.toBeInTheDocument());
  });
});
