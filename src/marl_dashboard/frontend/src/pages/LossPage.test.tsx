import { fireEvent, render, screen, within } from '@testing-library/react';
import { describe, expect, it, vi } from 'vitest';

import type { QueryResponse } from '../api/types';
import { LossPage } from './LossPage';

let lossesResponse: QueryResponse = {
  chart_series: [],
  table_rows: [],
  units: {},
  formulas: {},
  summary: { row_count: 0 }
};

vi.mock('../api/hooks', () => ({
  useAsync: (loader: () => Promise<QueryResponse>) => {
    void loader;
    return { loading: false, error: null, data: lossesResponse };
  },
  useLiveTick: () => 0
}));

vi.mock('../api/client', () => ({
  api: {
    losses: () => Promise.resolve(lossesResponse)
  }
}));

vi.mock('../components/charts/MultiPanelChart', () => ({
  MultiPanelChart: ({ panels }: { panels: Array<{ title: string }> }) => (
    <section>
      {panels.map((panel) => (
        <div key={panel.title}>{panel.title}</div>
      ))}
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
      {rows.map((row) => (
        <div key={`${title}-${row.metric_name}-metadata`}>
          {row.metric_name}:{row.display_name}:{row.unit}:{row.description}
        </div>
      ))}
    </section>
  )
}));

describe('LossPage', () => {
  it('explains that live losses appear after learner updates', () => {
    lossesResponse = {
      chart_series: [],
      table_rows: [],
      units: {},
      formulas: {},
      summary: { row_count: 0 }
    };
    render(
      <LossPage
        run={null}
        selectors={null}
        filters={{ runId: 'run_a', live: true, compareMode: false, epochId: 0 }}
      />
    );

    expect(screen.getByText(/Loss terms appear after learner updates/i)).toBeInTheDocument();
  });

  it('filters learner losses to one selected loss metric', () => {
    lossesResponse = {
      chart_series: [],
      table_rows: [
        {
          run_id: 'run_a',
          metric_group: 'loss',
          metric_name: 'actor_loss',
          value: 0.5,
          vpp_id: 'vpp_001',
          gradient_step: 1
        },
        {
          run_id: 'run_a',
          metric_group: 'loss',
          metric_name: 'critic_loss',
          value: 0.7,
          vpp_id: 'vpp_001',
          gradient_step: 1
        }
      ],
      units: { actor_loss: 'scalar', critic_loss: 'scalar' },
      formulas: {},
      summary: { row_count: 2 }
    };

    render(
      <LossPage
        run={null}
        selectors={null}
        filters={{ runId: 'run_a', live: true, compareMode: false, epochId: 0, vppId: 'vpp_001' }}
      />
    );

    fireEvent.change(screen.getByLabelText(/损失项 \/ Loss metric/i), { target: { value: 'critic_loss' } });

    const lossTable = screen.getByRole('heading', { name: /Loss terms/i }).closest('section');
    expect(lossTable).not.toBeNull();
    expect(within(lossTable as HTMLElement).getByText('critic_loss')).toBeInTheDocument();
    expect(within(lossTable as HTMLElement).queryByText('actor_loss')).not.toBeInTheDocument();
  });

  it('explains aggregate fallback when a selected VPP has only shared losses', () => {
    lossesResponse = {
      chart_series: [],
      table_rows: [
        {
          run_id: 'run_a',
          metric_group: 'loss',
          metric_name: 'dispatch_policy_loss',
          value: -0.2,
          vpp_id: 'aggregate',
          gradient_step: 2
        }
      ],
      units: { dispatch_policy_loss: 'scalar' },
      formulas: {},
      summary: {
        row_count: 1,
        vpp_filter_fallback: 'aggregate_shared_loss',
        requested_vpp_id: 'vpp_001',
        effective_vpp_id: 'aggregate'
      }
    };

    render(
      <LossPage
        run={null}
        selectors={null}
        filters={{ runId: 'run_a', live: true, compareMode: false, epochId: 0, vppId: 'vpp_001' }}
      />
    );

    expect(screen.getByText(/aggregate or shared-policy losses/i)).toBeInTheDocument();
  });

  it('warns when learner loss rows contain non-finite or exploding values', () => {
    lossesResponse = {
      chart_series: [],
      table_rows: [
        {
          run_id: 'run_a',
          metric_group: 'loss',
          metric_name: 'critic_loss',
          value: 'NaN',
          policy_id: 'shared_policy',
          gradient_step: 4
        },
        {
          run_id: 'run_a',
          metric_group: 'loss',
          metric_name: 'actor_loss',
          value: 1250000,
          policy_id: 'shared_policy',
          gradient_step: 5
        }
      ],
      units: { critic_loss: 'scalar', actor_loss: 'scalar' },
      formulas: {},
      summary: { row_count: 2 }
    };

    render(
      <LossPage
        run={null}
        selectors={null}
        filters={{ runId: 'run_a', live: true, compareMode: false, epochId: 0 }}
      />
    );

    const warning = screen.getByLabelText(/损失异常告警/i);
    expect(warning).toBeInTheDocument();
    expect(within(warning).getByText(/critic_loss/i)).toBeInTheDocument();
    expect(within(warning).getByText(/actor_loss/i)).toBeInTheDocument();
    expect(within(warning).getByText(/NaN or Infinity/i)).toBeInTheDocument();
    expect(within(warning).getByText(/Possible exploding loss/i)).toBeInTheDocument();
  });

  it('renders a learner loss formula dictionary', () => {
    lossesResponse = {
      chart_series: [],
      table_rows: [
        {
          run_id: 'run_a',
          metric_group: 'loss',
          metric_name: 'critic_loss',
          display_name: 'Critic 损失 / Critic loss',
          description: '价值函数拟合损失说明 / Value fitting loss description',
          value: 0.5,
          unit: 'scalar',
          policy_id: 'shared_policy',
          gradient_step: 2,
          formula_latex: 'L^{critic}'
        }
      ],
      units: { critic_loss: 'scalar' },
      formulas: { critic_loss: 'L^{critic}' },
      summary: { row_count: 1 }
    };

    render(
      <LossPage
        run={null}
        selectors={null}
        filters={{ runId: 'run_a', live: true, compareMode: false, epochId: 0 }}
      />
    );

    expect(screen.getByRole('heading', { name: '损失公式 / Loss formulas' })).toBeInTheDocument();
    expect(screen.getByText('critic_loss:L^{critic}')).toBeInTheDocument();
    expect(screen.getByText(/critic_loss:Critic 损失 \/ Critic loss:scalar:价值函数拟合损失说明/)).toBeInTheDocument();
  });
});
