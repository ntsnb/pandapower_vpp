import { render, screen } from '@testing-library/react';
import { describe, expect, it, vi } from 'vitest';

import type { QueryResponse, VariableDefinition } from '../api/types';
import { VariableDictionaryPage } from './VariableDictionaryPage';

const variables: VariableDefinition[] = [
  {
    name: 'electricity_price',
    display_name: '电价 / Electricity price',
    unit: '$/MWh',
    group: 'dataset',
    physical_meaning: '当前时刻电价。 / Current electricity price.'
  }
];

const emptyResponse: QueryResponse = {
  chart_series: [],
  table_rows: [],
  units: {},
  formulas: {},
  summary: { row_count: 0 }
};

const datasetResponse: QueryResponse = {
  chart_series: [],
  table_rows: [
    {
      run_id: 'run_a',
      metric_group: 'dataset',
      metric_name: 'electricity_price',
      value: 88.5,
      unit: '$/MWh',
      vpp_id: 'vpp_001',
      time_index: 3
    }
  ],
  units: { electricity_price: '$/MWh' },
  formulas: {},
  summary: { row_count: 1 }
};

let useAsyncResponses: Array<{ loading: boolean; error: null; data: unknown }> = [];
let useAsyncCallIndex = 0;

vi.mock('../api/hooks', () => ({
  useAsync: () => {
    const response = useAsyncResponses[useAsyncCallIndex] ?? { loading: false, error: null, data: null };
    useAsyncCallIndex += 1;
    return response;
  },
  useLiveTick: () => 0
}));

vi.mock('../api/client', () => ({
  api: {
    variables: vi.fn(),
    formulas: vi.fn(),
    dataset: vi.fn(),
    rewards: vi.fn(),
    costs: vi.fn(),
    losses: vi.fn()
  }
}));

vi.mock('../components/tables/FormulaTable', () => ({
  FormulaTable: () => <section>formulas</section>
}));

describe('VariableDictionaryPage', () => {
  it('passes current selected metric values into the variable dictionary table', () => {
    useAsyncResponses = [
      { loading: false, error: null, data: variables },
      { loading: false, error: null, data: {} },
      { loading: false, error: null, data: datasetResponse },
      { loading: false, error: null, data: emptyResponse },
      { loading: false, error: null, data: emptyResponse },
      { loading: false, error: null, data: emptyResponse }
    ];
    useAsyncCallIndex = 0;

    render(
      <VariableDictionaryPage
        run={null}
        selectors={null}
        filters={{ runId: 'run_a', live: true, compareMode: false, vppId: 'vpp_001', timeIndex: 3 }}
      />
    );

    expect(screen.getByText('88.5 $/MWh')).toBeInTheDocument();
    expect(screen.getByText('vpp_001 @ t=3')).toBeInTheDocument();
  });
});
