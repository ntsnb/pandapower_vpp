import { fireEvent, render, screen } from '@testing-library/react';
import { describe, expect, it, vi } from 'vitest';

import type { QueryResponse } from '../api/types';
import { ComparePage } from './ComparePage';

const compareResponse: QueryResponse = {
  chart_series: [],
  table_rows: [
    {
      run_id: 'run_a',
      metric_group: 'reward',
      metric_name: 'reward_so_far',
      description: '累计奖励 / Episode reward so far',
      value: 12.5,
      unit: 'score',
      group: 'vpp_001',
      vpp_id: 'vpp_001'
    },
    {
      run_id: 'run_a',
      metric_group: 'dataset',
      metric_name: 'electricity_price',
      display_name: '电价 / Electricity price',
      value: 63.4,
      unit: 'currency/MWh',
      date: '2018-01-02',
      time_index: 1,
      group: 'vpp_001',
      vpp_id: 'vpp_001'
    },
    {
      run_id: 'run_a',
      metric_group: 'dataset',
      metric_name: 'net_load',
      display_name: '净负荷 / Net load',
      value: 18.2,
      unit: 'MW',
      date: '2018-01-02',
      time_index: 1,
      group: 'vpp_001',
      vpp_id: 'vpp_001'
    },
    {
      run_id: 'run_a',
      metric_group: 'dataset',
      metric_name: 'electricity_price',
      display_name: '电价 / Electricity price',
      value: 63.4,
      unit: 'currency/MWh',
      date: '2018-01-02',
      time_index: 1,
      group: 'vpp_002',
      vpp_id: 'vpp_002'
    },
    {
      run_id: 'run_a',
      metric_group: 'dataset',
      metric_name: 'net_load',
      display_name: '净负荷 / Net load',
      value: 21.8,
      unit: 'MW',
      date: '2018-01-02',
      time_index: 1,
      group: 'vpp_002',
      vpp_id: 'vpp_002'
    }
  ],
  units: { reward_so_far: 'score', electricity_price: 'currency/MWh', net_load: 'MW' },
  formulas: {},
  summary: { row_count: 5, group_by: 'vpp_id' }
};

const mocks = vi.hoisted(() => ({
  compare: vi.fn()
}));

vi.mock('../api/client', () => ({
  api: {
    compare: mocks.compare
  }
}));

vi.mock('../api/hooks', () => ({
  useAsync: (loader: () => Promise<QueryResponse>) => {
    void loader();
    return { loading: false, error: null, data: compareResponse };
  },
  useLiveTick: () => 0
}));

vi.mock('../components/charts/SameTimeCompareChart', () => ({
  SameTimeCompareChart: () => <section>compare chart</section>
}));

describe('ComparePage', () => {
  it('lets the user choose compare scope, grouping, and metric', () => {
    mocks.compare.mockResolvedValue(compareResponse);

    render(
      <ComparePage
        run={null}
        selectors={null}
        filters={{
          runId: 'run_a',
          live: false,
          compareMode: true,
          epochId: 1,
          episodeId: 2,
          date: '2026-01-01',
          timeIndex: 3
        }}
      />
    );

    fireEvent.change(screen.getByLabelText('对比范围 / Compare scope'), { target: { value: 'dataset' } });
    fireEvent.change(screen.getByLabelText('分组维度 / Group by'), { target: { value: 'epoch_id' } });
    fireEvent.change(screen.getByLabelText('指标 / Metric'), { target: { value: 'reward_so_far' } });

    expect(mocks.compare).toHaveBeenCalledWith(
      'run_a',
      expect.objectContaining({
        scope: 'dataset',
        group_by: 'epoch_id',
        metric_names: 'reward_so_far',
        fixed_epoch_id: 1,
        fixed_episode_id: 2,
        fixed_date: '2026-01-01',
        fixed_time_index: 3
      })
    );
  });

  it('passes selected compare groups as group_values', () => {
    mocks.compare.mockResolvedValue(compareResponse);

    render(
      <ComparePage
        run={null}
        selectors={{
          run_id: 'run_a',
          dates: ['2018-01-02'],
          vpp_ids: ['vpp_001', 'vpp_002', 'vpp_003'],
          agent_ids: ['agent_001'],
          policy_ids: ['policy_shared'],
          epoch_ids: [0, 1],
          episode_ids: [1],
          time_indices: [0, 1]
        }}
        filters={{
          runId: 'run_a',
          live: false,
          compareMode: true,
          epochId: 1,
          date: '2018-01-02',
          timeIndex: 1
        }}
      />
    );

    const groupSelect = screen.getByLabelText('对比对象 / Compare groups') as HTMLSelectElement;
    for (const option of Array.from(groupSelect.options)) {
      option.selected = option.value === 'vpp_001' || option.value === 'vpp_003';
    }
    fireEvent.change(groupSelect);

    expect(mocks.compare).toHaveBeenCalledWith(
      'run_a',
      expect.objectContaining({
        group_by: 'vpp_id',
        group_values: 'vpp_001,vpp_003'
      })
    );
  });

  it('renders a same-time metric matrix with VPP rows and bilingual metric columns', () => {
    mocks.compare.mockResolvedValue(compareResponse);

    render(
      <ComparePage
        run={null}
        selectors={null}
        filters={{
          runId: 'run_a',
          live: false,
          compareMode: true,
          epochId: 1,
          date: '2018-01-02',
          timeIndex: 1
        }}
      />
    );

    expect(screen.getByRole('heading', { name: '指标矩阵 / Metric matrix' })).toBeInTheDocument();
    expect(screen.getByText('电价 / Electricity price (currency/MWh)')).toBeInTheDocument();
    expect(screen.getByText('净负荷 / Net load (MW)')).toBeInTheDocument();
    expect(screen.getByRole('row', { name: /vpp_001.*63\.4.*18\.2/i })).toBeInTheDocument();
    expect(screen.getByRole('row', { name: /vpp_002.*63\.4.*21\.8/i })).toBeInTheDocument();
  });
});
