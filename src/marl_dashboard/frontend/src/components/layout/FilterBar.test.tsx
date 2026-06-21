import { fireEvent, render, screen } from '@testing-library/react';
import { describe, expect, it, vi } from 'vitest';

import type { Filters, RunSummary, Selectors } from '../../api/types';
import { FilterBar } from './FilterBar';

const runs: RunSummary[] = [
  {
    run_id: 'run_a',
    status: 'running',
    started_at: null,
    ended_at: null,
    algorithm: 'happo',
    environment: 'multi_vpp',
    vpp_count: 2,
    epoch_count: 3
  }
];

const selectors: Selectors = {
  run_id: 'run_a',
  dates: ['2026-01-01'],
  vpp_ids: ['vpp_001', 'vpp_002'],
  agent_ids: ['agent_001'],
  policy_ids: ['policy_shared'],
  epoch_ids: [0, 1, 2],
  episode_ids: [1, 2],
  time_indices: [0, 1, 2]
};

const filters: Filters = {
  runId: 'run_a',
  date: '2026-01-01',
  vppId: 'vpp_001',
  epochId: 1,
  episodeId: 1,
  timeIndex: 1,
  startTimeIndex: 0,
  endTimeIndex: 2,
  agentId: 'agent_001',
  policyId: 'policy_shared',
  live: false,
  compareMode: false
};

describe('FilterBar', () => {
  it('updates live/frozen and compare mode without changing the selected run', () => {
    const onChange = vi.fn();

    render(<FilterBar runs={runs} selectors={selectors} filters={filters} onChange={onChange} />);

    fireEvent.click(screen.getByRole('button', { name: /frozen/i }));
    expect(onChange).toHaveBeenCalledWith(expect.objectContaining({ runId: 'run_a', live: true }));

    fireEvent.click(screen.getByRole('button', { name: /compare mode/i }));
    expect(onChange).toHaveBeenCalledWith(expect.objectContaining({ runId: 'run_a', compareMode: true }));
  });

  it('updates the selected episode for replaying one trajectory', () => {
    const onChange = vi.fn();

    render(<FilterBar runs={runs} selectors={selectors} filters={filters} onChange={onChange} />);

    fireEvent.change(screen.getByLabelText(/episode/i), { target: { value: '2' } });

    expect(onChange).toHaveBeenCalledWith(expect.objectContaining({ episodeId: 2 }));
  });

  it('updates agent, policy, and time range filters', () => {
    const onChange = vi.fn();

    render(<FilterBar runs={runs} selectors={selectors} filters={filters} onChange={onChange} />);

    fireEvent.change(screen.getByLabelText(/智能体/i), { target: { value: 'agent_001' } });
    expect(onChange).toHaveBeenCalledWith(expect.objectContaining({ agentId: 'agent_001' }));

    fireEvent.change(screen.getByLabelText(/策略/i), { target: { value: 'policy_shared' } });
    expect(onChange).toHaveBeenCalledWith(expect.objectContaining({ policyId: 'policy_shared' }));

    fireEvent.change(screen.getByLabelText(/起始时刻/i), { target: { value: '1' } });
    expect(onChange).toHaveBeenCalledWith(expect.objectContaining({ startTimeIndex: 1, timeIndex: undefined }));

    fireEvent.change(screen.getByLabelText(/结束时刻/i), { target: { value: '2' } });
    expect(onChange).toHaveBeenCalledWith(expect.objectContaining({ endTimeIndex: 2, timeIndex: undefined }));
  });

  it('explains that a single epoch dimension is not the same as episode selection', () => {
    const onChange = vi.fn();
    const singleEpochSelectors = { ...selectors, epoch_ids: [0], episode_ids: [1, 2, 3] };

    render(<FilterBar runs={runs} selectors={singleEpochSelectors} filters={{ ...filters, epochId: undefined }} onChange={onChange} />);

    expect(screen.getByText('当前 run 只有一个训练轮次维度 / Single epoch dimension')).toBeInTheDocument();
    expect(screen.getByLabelText(/训练轮次/i)).toBeDisabled();
    expect(screen.getByLabelText(/轨迹周期/i)).toBeEnabled();
  });

  it('provides a reset filters control that preserves the run and returns to live mode', () => {
    const onChange = vi.fn();

    render(<FilterBar runs={runs} selectors={selectors} filters={filters} onChange={onChange} />);

    fireEvent.click(screen.getByRole('button', { name: /reset filters/i }));
    expect(onChange).toHaveBeenCalledWith({ runId: 'run_a', live: true, compareMode: false });
  });

  it('labels partial dates before users load short-line charts', () => {
    const onChange = vi.fn();
    const partialSelectors: Selectors = {
      ...selectors,
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
      ]
    };

    render(<FilterBar runs={runs} selectors={partialSelectors} filters={{ ...filters, date: '2018-01-08' }} onChange={onChange} />);

    expect(screen.getByRole('option', { name: '2018-01-08 (未满 1/96 / Partial)' })).toBeInTheDocument();
    expect(screen.getByText(/当前日期未满 96 个时隙/)).toBeInTheDocument();
  });
});
