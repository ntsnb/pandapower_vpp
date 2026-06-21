import { render, screen } from '@testing-library/react';
import { describe, expect, it, vi } from 'vitest';

import type { QueryResponse } from '../api/types';
import { RewardScaledCostPage } from './RewardScaledCostPage';

let costsResponse: QueryResponse = {
  chart_series: [],
  table_rows: [],
  units: {},
  formulas: {},
  summary: { row_count: 0 }
};

vi.mock('../api/hooks', () => ({
  useAsync: () => ({ loading: false, error: null, data: costsResponse }),
  useLiveTick: () => 0
}));

vi.mock('../api/client', () => ({
  api: {
    costs: () => Promise.resolve(costsResponse)
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

describe('RewardScaledCostPage', () => {
  it('shows only cost penalties after reward scaling', () => {
    costsResponse = {
      chart_series: [],
      table_rows: [
        {
          run_id: 'run_a',
          metric_group: 'cost',
          metric_name: 'reward_scaled_total_projection_penalty',
          display_name: '总投影惩罚 / Total projection penalty',
          value: 1.4,
          vpp_id: 'vpp_a',
          time_index: 1
        },
        {
          run_id: 'run_a',
          metric_group: 'cost',
          metric_name: 'reward_scaled_comfort_soc_penalty',
          display_name: '训练奖励舒适度/SOC 惩罚 / Reward-scaled comfort-SOC penalty',
          value: 0.06,
          vpp_id: 'vpp_a',
          time_index: 1
        },
        {
          run_id: 'run_a',
          metric_group: 'cost',
          metric_name: 'import_energy_cost_total',
          value: 12,
          vpp_id: 'vpp_a',
          time_index: 1
        }
      ],
      units: {
        reward_scaled_total_projection_penalty: 'score',
        reward_scaled_comfort_soc_penalty: 'score',
        import_energy_cost_total: 'currency'
      },
      formulas: {},
      summary: { row_count: 3 }
    };

    render(
      <RewardScaledCostPage
        run={null}
        selectors={null}
        filters={{ runId: 'run_a', live: true, compareMode: false, vppId: 'vpp_a' }}
      />
    );

    expect(screen.getByText(/Reward-scaled cost penalties/)).toBeInTheDocument();
    expect(screen.getByText('reward_scaled_total_projection_penalty=1.4')).toBeInTheDocument();
    expect(screen.getByText('reward_scaled_comfort_soc_penalty=0.06')).toBeInTheDocument();
    expect(screen.queryByText('import_energy_cost_total=12')).not.toBeInTheDocument();
  });
});
