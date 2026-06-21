import { fireEvent, render, screen } from '@testing-library/react';
import { afterEach, describe, expect, it, vi } from 'vitest';

import type { MetricRow, RunMetadata, VariableDefinition } from '../api/types';
import { RunConfigPage } from './RunConfigPage';

const metadata: RunMetadata = {
  run_id: 'run_a',
  status: 'running',
  started_at: '2026-06-16T10:00:00Z',
  algorithm: 'happo',
  environment: 'paper_training',
  config: {
    seed: 9401,
    vpp_count: 7,
    batch_size: 128,
    learning_rate: 0.0003,
    episode_horizon_steps: 672
  }
};

const variables: VariableDefinition[] = [
  {
    name: 'electricity_price',
    display_name: '电价 / Electricity price',
    unit: 'currency/MWh',
    group: 'dataset'
  }
];

const eventRows: MetricRow[] = [
  {
    run_id: 'run_a',
    metric_group: 'event',
    metric_name: 'training_status',
    value: 'epoch finished',
    epoch_id: 1,
    episode_id: 2,
    timestamp: '2026-06-16T10:30:00Z'
  }
];

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
    metadata: vi.fn(),
    variables: vi.fn(),
    formulas: vi.fn(),
    events: vi.fn()
  }
}));

afterEach(() => {
  vi.restoreAllMocks();
});

describe('RunConfigPage', () => {
  it('shows key run metadata and provides JSON download actions', () => {
    const click = vi.fn();
    const appendChild = vi.spyOn(document.body, 'appendChild');
    vi.stubGlobal('URL', {
      createObjectURL: vi.fn(() => 'blob:run-config'),
      revokeObjectURL: vi.fn()
    });
    vi.spyOn(document, 'createElement').mockImplementation((tagName: string) => {
      const element = document.createElementNS('http://www.w3.org/1999/xhtml', tagName) as HTMLElement;
      if (tagName === 'a') {
        Object.defineProperty(element, 'click', { value: click });
      }
      return element as never;
    });
    useAsyncResponses = [
      { loading: false, error: null, data: metadata },
      { loading: false, error: null, data: variables },
      { loading: false, error: null, data: { total_reward: 'r_t' } },
      {
        loading: false,
        error: null,
        data: { chart_series: [], table_rows: eventRows, units: {}, formulas: {}, summary: { row_count: 1 } }
      }
    ];
    useAsyncCallIndex = 0;

    render(
      <RunConfigPage
        run={null}
        selectors={null}
        filters={{ runId: 'run_a', live: false, compareMode: false }}
      />
    );

    expect(screen.getByText('运行配置 / Run configuration')).toBeInTheDocument();
    expect(screen.getByText('算法 / Algorithm')).toBeInTheDocument();
    expect(screen.getByText('happo')).toBeInTheDocument();
    expect(screen.getByText('环境 / Environment')).toBeInTheDocument();
    expect(screen.getByText('paper_training')).toBeInTheDocument();
    expect(screen.getByText('Seed')).toBeInTheDocument();
    expect(screen.getByText('9401')).toBeInTheDocument();
    expect(screen.getByText('VPP 数量 / VPP count')).toBeInTheDocument();
    expect(screen.getByText('7')).toBeInTheDocument();

    expect(screen.getByRole('button', { name: '下载 metadata / Download metadata' })).toBeInTheDocument();
    expect(screen.getByRole('button', { name: '下载 config / Download config' })).toBeInTheDocument();
    expect(screen.getByRole('button', { name: '下载 formulas / Download formulas' })).toBeInTheDocument();
    expect(screen.getByRole('button', { name: '下载变量字典 / Download variable dictionary' })).toBeInTheDocument();
    expect(screen.getByText('事件日志 / Event log')).toBeInTheDocument();
    expect(screen.getByText('training_status')).toBeInTheDocument();
    expect(screen.getByText('epoch finished')).toBeInTheDocument();

    fireEvent.click(screen.getByRole('button', { name: '下载 config / Download config' }));

    expect(URL.createObjectURL).toHaveBeenCalled();
    expect(appendChild).toHaveBeenCalled();
    expect(click).toHaveBeenCalled();
  });
});
