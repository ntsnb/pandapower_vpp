import { fireEvent, render, screen, within } from '@testing-library/react';
import { describe, expect, it, vi } from 'vitest';

import type { QueryResponse } from '../api/types';
import { RewardCostPage } from './RewardCostPage';

let rewardsResponse: QueryResponse = {
  chart_series: [],
  table_rows: [],
  units: {},
  formulas: {},
  summary: { row_count: 0 }
};

let costsResponse: QueryResponse = {
  chart_series: [],
  table_rows: [],
  units: {},
  formulas: {},
  summary: { row_count: 0 }
};

vi.mock('../api/hooks', () => ({
  useAsync: (loader: () => Promise<QueryResponse>) => {
    void loader;
    if (useAsyncResponses.length === 0) {
      return { loading: false, error: null, data: rewardsResponse };
    }
    const response = useAsyncResponses[useAsyncCallIndex % useAsyncResponses.length];
    useAsyncCallIndex += 1;
    return response;
  },
  useLiveTick: () => 0
}));

vi.mock('../api/client', () => ({
  api: {
    rewards: () => Promise.resolve(rewardsResponse),
    costs: () => Promise.resolve(costsResponse)
  }
}));

vi.mock('../components/charts/CombinedChart', () => ({
  CombinedChart: ({
    title,
    responses
  }: {
    title: string;
    responses: Array<{ label: string; response: QueryResponse | null }>;
  }) => (
    <section>
      <h2>{title}</h2>
      {responses.map(({ label, response }) => (
        <span key={label}>
          {label}:{response?.summary?.row_count ?? 0}
        </span>
      ))}
    </section>
  )
}));

vi.mock('../components/charts/CompositionChart', () => ({
  CompositionChart: ({ title, response }: { title: string; response: QueryResponse | null }) => (
    <section aria-label={title}>
      <h2>{title}</h2>
      <div>
        {title} rows:
        {(response?.table_rows ?? []).map((row) => `${row.metric_name}@${row.time_index}=${row.value}`).join('|')}
      </div>
    </section>
  )
}));

vi.mock('../components/tables/FormulaTable', () => ({
  FormulaTable: ({
    title,
    formulas,
    rows = []
  }: {
    title?: string;
    formulas: Record<string, string>;
    rows?: Array<{ metric_name: string; display_name?: string | null; description?: string | null; unit?: string | null }>;
  }) => (
    <section>
      <h2>{title ?? 'Formula Dictionary'}</h2>
      {Object.entries(formulas).map(([name, formula]) => (
        <div key={name}>
          {name}:{formula}
        </div>
      ))}
      {rows.map((row, index) => (
        <div key={`${title}-${row.metric_name}-${index}-metadata`}>
          {row.metric_name}:{row.display_name}:{row.unit}:{row.description}
        </div>
      ))}
    </section>
  )
}));

let useAsyncResponses: Array<{ loading: boolean; error: null; data: QueryResponse }> = [];
let useAsyncCallIndex = 0;

describe('RewardCostPage', () => {
  it('does not describe all-VPP rows as aggregate-only when per-VPP reward rows exist', () => {
    rewardsResponse = {
      chart_series: [],
      table_rows: [
        {
          run_id: 'run_a',
          metric_group: 'reward',
          metric_name: 'dispatch_reward_train',
          value: 1.5,
          vpp_id: 'vpp_a',
          time_index: 3
        }
      ],
      units: { dispatch_reward_train: 'score' },
      formulas: {},
      summary: { row_count: 1 }
    };
    costsResponse = {
      chart_series: [],
      table_rows: [],
      units: {},
      formulas: {},
      summary: { row_count: 0 }
    };
    useAsyncResponses = [
      { loading: false, error: null, data: rewardsResponse },
      { loading: false, error: null, data: costsResponse }
    ];
    useAsyncCallIndex = 0;

    render(
      <RewardCostPage
        run={null}
        selectors={null}
        filters={{ runId: 'run_a', live: true, compareMode: false, epochId: 0 }}
      />
    );

    expect(screen.queryByText(/aggregate training progress/i)).not.toBeInTheDocument();
  });

  it('lets a user hide one reward term with a checkbox while keeping other reward terms visible', () => {
    rewardsResponse = {
      chart_series: [],
      table_rows: [
        {
          run_id: 'run_a',
          metric_group: 'reward',
          metric_name: 'profit_reward',
          value: 1.5,
          vpp_id: 'vpp_001',
          time_index: 3
        },
        {
          run_id: 'run_a',
          metric_group: 'reward',
          metric_name: 'grid_balance_reward',
          value: -0.2,
          vpp_id: 'vpp_001',
          time_index: 3
        }
      ],
      units: { profit_reward: 'score', grid_balance_reward: 'score' },
      formulas: {},
      summary: { row_count: 2 }
    };
    costsResponse = {
      chart_series: [],
      table_rows: [],
      units: {},
      formulas: {},
      summary: { row_count: 0 }
    };
    useAsyncResponses = [
      { loading: false, error: null, data: rewardsResponse },
      { loading: false, error: null, data: costsResponse }
    ];
    useAsyncCallIndex = 0;

    render(
      <RewardCostPage
        run={null}
        selectors={null}
        filters={{ runId: 'run_a', live: true, compareMode: false, epochId: 0, vppId: 'vpp_001' }}
      />
    );

    fireEvent.click(screen.getByRole('checkbox', { name: /grid_balance_reward/i }));

    const rewardTable = screen.getByRole('heading', { name: /Reward terms/i }).closest('section');
    expect(rewardTable).not.toBeNull();
    expect(within(rewardTable as HTMLElement).getByText('profit_reward')).toBeInTheDocument();
    expect(within(rewardTable as HTMLElement).queryByText('grid_balance_reward')).not.toBeInTheDocument();
  });

  it('can hide all cost trajectories without hiding rewards', () => {
    rewardsResponse = {
      chart_series: [],
      table_rows: [
        {
          run_id: 'run_a',
          metric_group: 'reward',
          metric_name: 'dispatch_reward_train',
          display_name: '调度训练奖励 / Dispatch training reward',
          value: 1.5,
          vpp_id: 'vpp_001',
          time_index: 3
        }
      ],
      units: { dispatch_reward_train: 'score' },
      formulas: {},
      summary: { row_count: 1 }
    };
    costsResponse = {
      chart_series: [],
      table_rows: [
        {
          run_id: 'run_a',
          metric_group: 'cost',
          metric_name: 'der_operation_cost',
          display_name: 'DER 运行成本 / DER operation cost',
          value: 0.8,
          vpp_id: 'vpp_001',
          time_index: 3
        }
      ],
      units: { der_operation_cost: 'currency' },
      formulas: {},
      summary: { row_count: 1 }
    };
    useAsyncResponses = [
      { loading: false, error: null, data: rewardsResponse },
      { loading: false, error: null, data: costsResponse }
    ];
    useAsyncCallIndex = 0;

    render(
      <RewardCostPage
        run={null}
        selectors={null}
        filters={{ runId: 'run_a', live: true, compareMode: false, epochId: 0, vppId: 'vpp_001' }}
      />
    );

    fireEvent.click(screen.getByRole('checkbox', { name: /DER 运行成本 \/ DER operation cost/i }));

    expect(screen.getByText('奖励 / Reward:1')).toBeInTheDocument();
    expect(screen.getByText('成本 / Cost:0')).toBeInTheDocument();
  });

  it('keeps cost trajectories visible when only rewards are hidden', () => {
    rewardsResponse = {
      chart_series: [],
      table_rows: [
        {
          run_id: 'run_a',
          metric_group: 'reward',
          metric_name: 'dispatch_reward_train',
          value: 1.5,
          vpp_id: 'vpp_001',
          time_index: 3
        }
      ],
      units: { dispatch_reward_train: 'score' },
      formulas: {},
      summary: { row_count: 1 }
    };
    costsResponse = {
      chart_series: [],
      table_rows: [
        {
          run_id: 'run_a',
          metric_group: 'cost',
          metric_name: 'der_operation_cost',
          value: 2000000,
          vpp_id: 'vpp_001',
          time_index: 3
        }
      ],
      units: { der_operation_cost: 'currency' },
      formulas: {},
      summary: { row_count: 1 }
    };
    useAsyncResponses = [
      { loading: false, error: null, data: rewardsResponse },
      { loading: false, error: null, data: costsResponse }
    ];
    useAsyncCallIndex = 0;

    render(
      <RewardCostPage
        run={null}
        selectors={null}
        filters={{ runId: 'run_a', live: true, compareMode: false, epochId: 0, vppId: 'vpp_001' }}
      />
    );

    fireEvent.click(screen.getByRole('checkbox', { name: /dispatch_reward_train/i }));

    expect(screen.getByText('奖励 / Reward:0')).toBeInTheDocument();
    expect(screen.getByText('成本 / Cost:1')).toBeInTheDocument();
  });

  it('orders reward checkboxes from aggregate formulas to subcomponents', () => {
    rewardsResponse = {
      chart_series: [],
      table_rows: [
        {
          run_id: 'run_a',
          metric_group: 'reward',
          metric_name: 'pv_export_revenue_total',
          display_name: '光伏外送收入 / PV export revenue',
          value: 1,
          vpp_id: 'vpp_001'
        },
        {
          run_id: 'run_a',
          metric_group: 'reward',
          metric_name: 'total_reward',
          display_name: '总奖励 / Total reward',
          value: 3,
          vpp_id: 'vpp_001'
        },
        {
          run_id: 'run_a',
          metric_group: 'reward',
          metric_name: 'export_revenue_total',
          display_name: '总外送收入 / Total export revenue',
          value: 2,
          vpp_id: 'vpp_001'
        }
      ],
      units: {},
      formulas: {},
      summary: { row_count: 3 }
    };
    costsResponse = {
      chart_series: [],
      table_rows: [],
      units: {},
      formulas: {},
      summary: { row_count: 0 }
    };
    useAsyncResponses = [
      { loading: false, error: null, data: rewardsResponse },
      { loading: false, error: null, data: costsResponse }
    ];
    useAsyncCallIndex = 0;

    render(
      <RewardCostPage
        run={null}
        selectors={null}
        filters={{ runId: 'run_a', live: true, compareMode: false, epochId: 0, vppId: 'vpp_001' }}
      />
    );

    const rewardGroup = screen.getByRole('group', { name: '奖励项 / Reward metric' });
    const checkboxLabels = within(rewardGroup)
      .getAllByRole('checkbox')
      .map((checkbox) => checkbox.parentElement?.textContent ?? '');

    expect(checkboxLabels).toEqual([
      expect.stringContaining('总奖励 / Total reward'),
      expect.stringContaining('总外送收入 / Total export revenue'),
      expect.stringContaining('光伏外送收入 / PV export revenue')
    ]);
  });

  it('uses a local step slider so composition charts show the selected single-step terms', () => {
    rewardsResponse = {
      chart_series: [],
      table_rows: [
        {
          run_id: 'run_a',
          metric_group: 'reward',
          metric_name: 'dispatch_reward_train',
          value: 1.25,
          vpp_id: 'vpp_001',
          time_index: 1
        },
        {
          run_id: 'run_a',
          metric_group: 'reward',
          metric_name: 'comfort_reward',
          value: 0.5,
          vpp_id: 'vpp_001',
          time_index: 1
        },
        {
          run_id: 'run_a',
          metric_group: 'reward',
          metric_name: 'dispatch_reward_train',
          value: 2.5,
          vpp_id: 'vpp_001',
          time_index: 2
        },
        {
          run_id: 'run_a',
          metric_group: 'reward',
          metric_name: 'comfort_reward',
          value: 0.75,
          vpp_id: 'vpp_001',
          time_index: 2
        }
      ],
      units: { dispatch_reward_train: 'score', comfort_reward: 'score' },
      formulas: {},
      summary: { row_count: 4 }
    };
    costsResponse = {
      chart_series: [],
      table_rows: [
        {
          run_id: 'run_a',
          metric_group: 'cost',
          metric_name: 'der_operation_cost',
          value: 3,
          vpp_id: 'vpp_001',
          time_index: 1
        },
        {
          run_id: 'run_a',
          metric_group: 'cost',
          metric_name: 'der_operation_cost',
          value: 4,
          vpp_id: 'vpp_001',
          time_index: 2
        }
      ],
      units: { der_operation_cost: 'currency' },
      formulas: {},
      summary: { row_count: 2 }
    };
    useAsyncResponses = [
      { loading: false, error: null, data: rewardsResponse },
      { loading: false, error: null, data: costsResponse }
    ];
    useAsyncCallIndex = 0;

    render(
      <RewardCostPage
        run={null}
        selectors={{
          run_id: 'run_a',
          dates: [],
          vpp_ids: ['vpp_001'],
          agent_ids: [],
          policy_ids: [],
          epoch_ids: [0],
          episode_ids: [0],
          time_indices: [1, 2]
        }}
        filters={{ runId: 'run_a', live: true, compareMode: false, epochId: 0, vppId: 'vpp_001' }}
      />
    );

    expect(screen.getByText('奖励 / Reward:4')).toBeInTheDocument();
    expect(screen.getByText(/组成 step \/ Composition step: #2/)).toBeInTheDocument();
    expect(screen.getByText(/奖励组成 \/ Reward composition rows:.*dispatch_reward_train@2=2.5/)).toBeInTheDocument();
    expect(screen.queryByText(/奖励组成 \/ Reward composition rows:.*dispatch_reward_train@1=1.25/)).not.toBeInTheDocument();
    expect(screen.getByText(/成本组成 \/ Cost composition rows:.*der_operation_cost@2=4/)).toBeInTheDocument();

    fireEvent.change(screen.getByRole('slider', { name: /组成 step \/ Composition step/i }), {
      target: { value: '0' }
    });

    expect(screen.getByText(/组成 step \/ Composition step: #1/)).toBeInTheDocument();
    expect(screen.getByText(/奖励组成 \/ Reward composition rows:.*dispatch_reward_train@1=1.25/)).toBeInTheDocument();
    expect(screen.queryByText(/奖励组成 \/ Reward composition rows:.*dispatch_reward_train@2=2.5/)).not.toBeInTheDocument();
    expect(screen.getByText(/成本组成 \/ Cost composition rows:.*der_operation_cost@1=3/)).toBeInTheDocument();
  });

  it('shows reward, cost, cost-free reward, and reward-scaled cost compositions on the same page', () => {
    rewardsResponse = {
      chart_series: [],
      table_rows: [
        {
          run_id: 'run_a',
          metric_group: 'reward',
          metric_name: 'dispatch_reward_train',
          value: 5,
          vpp_id: 'vpp_001',
          time_index: 3
        },
        {
          run_id: 'run_a',
          metric_group: 'reward',
          metric_name: 'operational_surplus',
          value: 4,
          vpp_id: 'vpp_001',
          time_index: 3
        },
        {
          run_id: 'run_a',
          metric_group: 'reward',
          metric_name: 'storage_potential_raw',
          value: 2,
          vpp_id: 'vpp_001',
          time_index: 3
        }
      ],
      units: {
        dispatch_reward_train: 'score',
        operational_surplus: 'currency',
        storage_potential_raw: 'currency'
      },
      formulas: {},
      summary: { row_count: 3 }
    };
    costsResponse = {
      chart_series: [],
      table_rows: [
        {
          run_id: 'run_a',
          metric_group: 'cost',
          metric_name: 'battery_degradation_cost_total',
          value: 1,
          vpp_id: 'vpp_001',
          time_index: 3
        },
        {
          run_id: 'run_a',
          metric_group: 'cost',
          metric_name: 'reward_scaled_battery_degradation_penalty',
          value: 0.01,
          vpp_id: 'vpp_001',
          time_index: 3
        }
      ],
      units: {
        battery_degradation_cost_total: 'currency',
        reward_scaled_battery_degradation_penalty: 'score'
      },
      formulas: {},
      summary: { row_count: 2 }
    };
    useAsyncResponses = [
      { loading: false, error: null, data: rewardsResponse },
      { loading: false, error: null, data: costsResponse }
    ];
    useAsyncCallIndex = 0;

    render(
      <RewardCostPage
        run={null}
        selectors={null}
        filters={{ runId: 'run_a', live: true, compareMode: false, epochId: 0, vppId: 'vpp_001' }}
      />
    );

    expect(screen.getByRole('heading', { name: /奖励组成 \/ Reward composition/i })).toBeInTheDocument();
    expect(screen.getByRole('heading', { name: /成本组成 \/ Cost composition/i })).toBeInTheDocument();
    expect(screen.getByRole('heading', { name: /无成本奖励组成 \/ Cost-free reward composition/i })).toBeInTheDocument();
    expect(screen.getByRole('heading', { name: /缩放成本组成 \/ Reward-scaled cost composition/i })).toBeInTheDocument();
    expect(screen.getByText(/无成本奖励组成 \/ Cost-free reward composition rows:.*operational_surplus@3=4/)).toBeInTheDocument();
    expect(screen.getByText(/缩放成本组成 \/ Reward-scaled cost composition rows:.*reward_scaled_battery_degradation_penalty@3=0.01/)).toBeInTheDocument();
  });

  it('derives reward-scaled cost composition from raw v3.1 cost rows when legacy runs lack reward_scaled metrics', () => {
    rewardsResponse = {
      chart_series: [],
      table_rows: [
        {
          run_id: 'run_a',
          metric_group: 'reward',
          metric_name: 'dispatch_reward_env',
          value: 10,
          vpp_id: 'vpp_001',
          policy_id: 'happo',
          time_index: 3
        },
        {
          run_id: 'run_a',
          metric_group: 'reward',
          metric_name: 'dispatch_reward_train',
          value: 9.5,
          vpp_id: 'vpp_001',
          policy_id: 'happo',
          time_index: 3
        }
      ],
      units: { dispatch_reward_env: 'score', dispatch_reward_train: 'score' },
      formulas: {},
      summary: { row_count: 2 }
    };
    costsResponse = {
      chart_series: [],
      table_rows: [
        {
          run_id: 'run_a',
          metric_group: 'cost',
          metric_name: 'battery_degradation_cost',
          value: 2,
          vpp_id: 'vpp_001',
          policy_id: 'happo',
          time_index: 3
        },
        {
          run_id: 'run_a',
          metric_group: 'cost',
          metric_name: 'scaled_comfort_soc_penalty',
          value: 4,
          vpp_id: 'vpp_001',
          policy_id: 'happo',
          time_index: 3
        },
        {
          run_id: 'run_a',
          metric_group: 'cost',
          metric_name: 'dispatch_projection_penalty',
          value: 0.25,
          vpp_id: 'vpp_001',
          policy_id: 'happo',
          time_index: 3
        }
      ],
      units: {
        battery_degradation_cost: 'currency',
        scaled_comfort_soc_penalty: 'score',
        dispatch_projection_penalty: 'score'
      },
      formulas: {},
      summary: { row_count: 3 }
    };
    useAsyncResponses = [
      { loading: false, error: null, data: rewardsResponse },
      { loading: false, error: null, data: costsResponse }
    ];
    useAsyncCallIndex = 0;

    render(
      <RewardCostPage
        run={null}
        selectors={null}
        filters={{ runId: 'run_a', live: true, compareMode: false, epochId: 0, vppId: 'vpp_001' }}
      />
    );

    expect(
      screen.getByText(/缩放成本组成 \/ Reward-scaled cost composition rows:.*reward_scaled_battery_degradation_penalty@3=0.02/)
    ).toBeInTheDocument();
    expect(
      screen.getByText(/缩放成本组成 \/ Reward-scaled cost composition rows:.*reward_scaled_comfort_soc_penalty@3=0.08/)
    ).toBeInTheDocument();
    expect(
      screen.getByText(/缩放成本组成 \/ Reward-scaled cost composition rows:.*reward_scaled_dispatch_projection_penalty@3=0.25/)
    ).toBeInTheDocument();
    expect(
      screen.getByText(/缩放成本组成 \/ Reward-scaled cost composition rows:.*reward_scaled_training_projection_penalty@3=0.5/)
    ).toBeInTheDocument();
    expect(
      screen.getByText(/缩放成本组成 \/ Reward-scaled cost composition rows:.*reward_scaled_total_projection_penalty@3=0.75/)
    ).toBeInTheDocument();
  });

  it('explains when a selected VPP episode has no reward or cost rows yet', () => {
    rewardsResponse = {
      chart_series: [],
      table_rows: [],
      units: {},
      formulas: {},
      summary: { row_count: 0 }
    };
    costsResponse = {
      chart_series: [],
      table_rows: [],
      units: {},
      formulas: {},
      summary: { row_count: 0 }
    };
    useAsyncResponses = [
      { loading: false, error: null, data: rewardsResponse },
      { loading: false, error: null, data: costsResponse }
    ];
    useAsyncCallIndex = 0;

    render(
      <RewardCostPage
        run={null}
        selectors={null}
        filters={{ runId: 'run_a', live: true, compareMode: false, epochId: 0, episodeId: 6, vppId: 'vpp_commercial_multi' }}
      />
    );

    expect(screen.getByText(/当前筛选下该 VPP 没有 reward\/cost 行/)).toBeInTheDocument();
    expect(screen.getByText(/成本数值过大不会导致曲线被隐藏/)).toBeInTheDocument();
  });

  it('explains short reward and cost lines when the selected date is only partially logged', () => {
    rewardsResponse = {
      chart_series: [],
      table_rows: [
        {
          run_id: 'run_a',
          metric_group: 'reward',
          metric_name: 'reward_so_far',
          value: 12,
          date: '2018-01-08',
          time_index: 0,
          vpp_id: 'vpp_001',
          policy_id: 'happo'
        }
      ],
      units: { reward_so_far: 'score' },
      formulas: {},
      summary: { row_count: 1 }
    };
    costsResponse = {
      chart_series: [],
      table_rows: [
        {
          run_id: 'run_a',
          metric_group: 'cost',
          metric_name: 'total_cost_so_far',
          value: 40,
          date: '2018-01-08',
          time_index: 0,
          vpp_id: 'vpp_001',
          policy_id: 'happo'
        }
      ],
      units: { total_cost_so_far: 'cost' },
      formulas: {},
      summary: { row_count: 1 }
    };
    useAsyncResponses = [
      { loading: false, error: null, data: rewardsResponse },
      { loading: false, error: null, data: costsResponse }
    ];
    useAsyncCallIndex = 0;

    render(
      <RewardCostPage
        run={null}
        selectors={{
          run_id: 'run_a',
          dates: ['2018-01-08'],
          vpp_ids: ['vpp_001'],
          agent_ids: [],
          policy_ids: ['happo'],
          epoch_ids: [0],
          episode_ids: [6],
          time_indices: Array.from({ length: 96 }, (_, index) => index)
        }}
        filters={{ runId: 'run_a', live: true, compareMode: false, epochId: 0, date: '2018-01-08', vppId: 'vpp_001' }}
      />
    );

    expect(screen.getByText(/当前 reward\/cost 日期只返回 1 个 time_index/)).toBeInTheDocument();
    expect(screen.getByText(/parallel workers can share the same x-axis slot/i)).toBeInTheDocument();
  });

  it('renders reward and cost formula dictionaries beside trajectories', () => {
    rewardsResponse = {
      chart_series: [],
      table_rows: [
        {
          run_id: 'run_a',
          metric_group: 'reward',
          metric_name: 'profit_reward',
          display_name: '收益奖励 / Profit reward',
          description: 'VPP 收益奖励说明 / VPP profit reward description',
          unit: 'score',
          value: 1.5,
          formula_latex: 'r^{profit}_{i,t}'
        }
      ],
      units: { profit_reward: 'score' },
      formulas: { profit_reward: 'r^{profit}_{i,t}' },
      summary: { row_count: 1 }
    };
    costsResponse = {
      chart_series: [],
      table_rows: [
        {
          run_id: 'run_a',
          metric_group: 'cost',
          metric_name: 'der_operation_cost',
          display_name: 'DER 运行成本 / DER operation cost',
          description: 'DER 运行成本说明 / DER operation cost description',
          unit: 'currency',
          value: 0.8,
          formula_latex: 'c^{DER}_{i,t}'
        }
      ],
      units: { der_operation_cost: 'currency' },
      formulas: { der_operation_cost: 'c^{DER}_{i,t}' },
      summary: { row_count: 1 }
    };
    useAsyncResponses = [
      { loading: false, error: null, data: rewardsResponse },
      { loading: false, error: null, data: costsResponse }
    ];
    useAsyncCallIndex = 0;

    render(
      <RewardCostPage
        run={null}
        selectors={null}
        filters={{ runId: 'run_a', live: true, compareMode: false, epochId: 0, vppId: 'vpp_001' }}
      />
    );

    expect(screen.getByRole('heading', { name: '奖励公式 / Reward formulas' })).toBeInTheDocument();
    expect(screen.getByText('profit_reward:r^{profit}_{i,t}')).toBeInTheDocument();
    expect(screen.getByText(/profit_reward:收益奖励 \/ Profit reward:score:VPP 收益奖励说明/)).toBeInTheDocument();
    expect(screen.getByRole('heading', { name: '成本公式 / Cost formulas' })).toBeInTheDocument();
    expect(screen.getByText('der_operation_cost:c^{DER}_{i,t}')).toBeInTheDocument();
    expect(screen.getByText(/der_operation_cost:DER 运行成本 \/ DER operation cost:currency:DER 运行成本说明/)).toBeInTheDocument();
  });
});
