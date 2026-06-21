import { render, screen } from '@testing-library/react';
import { describe, expect, it, vi } from 'vitest';

import type { QueryResponse } from '../api/types';
import { CostFreeRewardPage } from './CostFreeRewardPage';

let rewardsResponse: QueryResponse = {
  chart_series: [],
  table_rows: [],
  units: {},
  formulas: {},
  summary: { row_count: 0 }
};

vi.mock('../api/hooks', () => ({
  useAsync: () => ({ loading: false, error: null, data: rewardsResponse }),
  useLiveTick: () => 0
}));

vi.mock('../api/client', () => ({
  api: {
    rewards: () => Promise.resolve(rewardsResponse)
  }
}));

vi.mock('../components/charts/CombinedChart', () => ({
  CombinedChart: ({ title, responses }: { title: string; responses: Array<{ response: QueryResponse | null }> }) => (
    <section>
      <h2>{title}</h2>
      {responses.flatMap(({ response }) => response?.table_rows ?? []).map((row) => (
        <span key={row.metric_name}>
          {row.metric_name}={row.value}
        </span>
      ))}
    </section>
  )
}));

describe('CostFreeRewardPage', () => {
  it('shows unscaled reward values and the reward weights that would scale them', () => {
    rewardsResponse = {
      chart_series: [],
      table_rows: [
        {
          run_id: 'run_a',
          metric_group: 'reward',
          metric_name: 'economic_operational_surplus',
          display_name: '经济运行盈余 / Economic operational surplus',
          value: 10,
          vpp_id: 'vpp_a',
          time_index: 1
        },
        {
          run_id: 'run_a',
          metric_group: 'reward',
          metric_name: 'private_profit_weight',
          display_name: '私有收益权重 / Private-profit weight',
          value: 1,
          vpp_id: 'vpp_a',
          time_index: 1
        },
        {
          run_id: 'run_a',
          metric_group: 'reward',
          metric_name: 'service_payment',
          display_name: '服务补偿 / Service payment',
          value: 7,
          vpp_id: 'vpp_a',
          time_index: 1
        },
        {
          run_id: 'run_a',
          metric_group: 'reward',
          metric_name: 'service_payment_weight',
          display_name: '服务补偿权重 / Service payment weight',
          value: 0,
          vpp_id: 'vpp_a',
          time_index: 1
        },
        {
          run_id: 'run_a',
          metric_group: 'reward',
          metric_name: 'dispatch_reward_train',
          value: 3,
          vpp_id: 'vpp_a',
          time_index: 1
        }
      ],
      units: {
        economic_operational_surplus: 'currency',
        service_payment: 'currency',
        private_profit_weight: 'dimensionless',
        service_payment_weight: 'dimensionless'
      },
      formulas: {},
      summary: { row_count: 5 }
    };

    render(
      <CostFreeRewardPage
        run={null}
        selectors={null}
        filters={{ runId: 'run_a', live: true, compareMode: false, vppId: 'vpp_a' }}
      />
    );

    expect(screen.getByText(/无成本奖励真实值 \/ Cost-free reward raw values/)).toBeInTheDocument();
    expect(screen.getByText('economic_operational_surplus=10')).toBeInTheDocument();
    expect(screen.getByText('service_payment=7')).toBeInTheDocument();
    expect(screen.queryByText('dispatch_reward_train=3')).not.toBeInTheDocument();
    expect(screen.getAllByText(/private_profit_weight = 1/).length).toBeGreaterThan(0);
    expect(screen.getAllByText(/service_payment_weight = 0/).length).toBeGreaterThan(0);
    expect(screen.getByText(/计入值 \/ weighted 0/)).toBeInTheDocument();
  });
});
