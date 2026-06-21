import { render, screen } from '@testing-library/react';
import { describe, expect, it, vi } from 'vitest';

import type { QueryResponse } from '../../api/types';
import { CombinedChart } from './CombinedChart';

vi.mock('echarts-for-react', () => ({
  default: ({ option }: { option: unknown }) => <pre data-testid="chart-option">{JSON.stringify(option)}</pre>
}));

describe('CombinedChart', () => {
  it('aligns reward and cost series by sorted semantic time labels instead of the first series only', () => {
    const rewards: QueryResponse = {
      chart_series: [
        {
          name: 'dispatch_reward_train / vpp_001',
          metric_name: 'dispatch_reward_train',
          vpp_id: 'vpp_001',
          points: [
            {
              run_id: 'run_a',
              metric_group: 'reward',
              metric_name: 'dispatch_reward_train',
              display_name: '调度训练奖励 / Dispatch training reward',
              value: 1,
              date: '2018-01-01',
              time_index: 0,
              episode_id: 1,
              vpp_id: 'vpp_001'
            },
            {
              run_id: 'run_a',
              metric_group: 'reward',
              metric_name: 'dispatch_reward_train',
              display_name: '调度训练奖励 / Dispatch training reward',
              value: 3,
              date: '2018-01-01',
              time_index: 86,
              episode_id: 1,
              vpp_id: 'vpp_001'
            },
            {
              run_id: 'run_a',
              metric_group: 'reward',
              metric_name: 'dispatch_reward_train',
              display_name: '调度训练奖励 / Dispatch training reward',
              value: 2,
              date: '2018-01-01',
              time_index: 43,
              episode_id: 1,
              vpp_id: 'vpp_001'
            }
          ]
        }
      ],
      table_rows: [],
      units: {},
      formulas: {},
      summary: { row_count: 3 }
    };
    const costs: QueryResponse = {
      chart_series: [
        {
          name: 'der_operation_cost / vpp_001',
          metric_name: 'der_operation_cost',
          vpp_id: 'vpp_001',
          points: [
            {
              run_id: 'run_a',
              metric_group: 'cost',
              metric_name: 'der_operation_cost',
              display_name: 'DER 运行成本 / DER operation cost',
              value: 20,
              date: '2018-01-01',
              time_index: 43,
              episode_id: 1,
              vpp_id: 'vpp_001'
            }
          ]
        }
      ],
      table_rows: [],
      units: {},
      formulas: {},
      summary: { row_count: 1 }
    };

    render(
      <CombinedChart
        title="奖励与成本轨迹 / Reward and cost trajectories"
        responses={[
          { label: '奖励 / Reward', response: rewards },
          { label: '成本 / Cost', response: costs }
        ]}
      />
    );

    const option = JSON.parse(screen.getByTestId('chart-option').textContent ?? '{}') as {
      xAxis: { data: string[] };
      series: Array<{ name: string; data: Array<number | null>; connectNulls?: boolean }>;
    };
    expect(option.xAxis.data).toEqual([
      'ep 1 2018-01-01 #0',
      'ep 1 2018-01-01 #43',
      'ep 1 2018-01-01 #86'
    ]);
    expect(option.series[0].name).toBe('奖励 / Reward: 调度训练奖励 / Dispatch training reward / vpp_001');
    expect(option.series[0].data).toEqual([1, 2, 3]);
    expect(option.series[1].name).toBe('成本 / Cost: DER 运行成本 / DER operation cost / vpp_001');
    expect(option.series[1].data).toEqual([null, 20, null]);
    expect(option.series.every((series) => series.connectNulls === true)).toBe(true);
  });

  it('uses energy dataset date and time_index before training log timestamps on the x axis', () => {
    const costs: QueryResponse = {
      chart_series: [
        {
          name: 'der_operation_cost / vpp_001',
          metric_name: 'der_operation_cost',
          vpp_id: 'vpp_001',
          points: [
            {
              run_id: 'run_a',
              metric_group: 'cost',
              metric_name: 'der_operation_cost',
              value: 20,
              date: '2018-01-01',
              time_index: 1,
              timestamp: '2026-06-16T12:00:00Z',
              episode_id: 2,
              vpp_id: 'vpp_001'
            }
          ]
        }
      ],
      table_rows: [],
      units: {},
      formulas: {},
      summary: { row_count: 1 }
    };

    render(<CombinedChart title="Flexible" responses={[{ label: '成本 / Cost', response: costs }]} />);

    const option = JSON.parse(screen.getByTestId('chart-option').textContent ?? '{}') as {
      xAxis: { data: string[] };
    };
    expect(option.xAxis.data).toEqual(['ep 2 2018-01-01 #1']);
    expect(option.xAxis.data.join(' ')).not.toContain('2026-06-16');
  });
});
