import { fireEvent, render, screen } from '@testing-library/react';
import { describe, expect, it, vi } from 'vitest';

import type { Filters, RunSummary, Selectors } from '../../api/types';
import { TopBar } from './TopBar';

const run: RunSummary = {
  run_id: 'run_a',
  status: 'running',
  started_at: null,
  ended_at: null,
  algorithm: 'happo',
  environment: 'multi_vpp',
  vpp_count: 2,
  epoch_count: 3
};

const selectors: Selectors = {
  run_id: 'run_a',
  dates: ['profile_day_001'],
  vpp_ids: ['vpp_001'],
  agent_ids: ['agent_001'],
  policy_ids: ['happo'],
  epoch_ids: [0, 1],
  episode_ids: [1],
  time_indices: [0, 1, 2]
};

const filters: Filters = {
  runId: 'run_a',
  live: true,
  compareMode: false
};

describe('TopBar', () => {
  it('shows the websocket live connection status in bilingual text', () => {
    render(<TopBar run={run} selectors={selectors} filters={filters} liveStatus="connected" liveEventCount={12} />);

    expect(screen.getByText('实时通道 / WebSocket: connected (12)')).toBeInTheDocument();
    expect(screen.getByText('日期 / Date 全部日期 / All dates')).toBeInTheDocument();
    expect(screen.getByText('时刻 / Time index 全部时刻 / All time indices')).toBeInTheDocument();
  });

  it('distinguishes epoch from episode in the status labels', () => {
    render(<TopBar run={run} selectors={selectors} filters={{ ...filters, episodeId: 1 }} liveStatus="connected" liveEventCount={12} />);

    expect(screen.getByText('训练轮次 / Epoch 1')).toHaveAttribute('title', expect.stringContaining('不等于 episode'));
    expect(screen.getByText('轨迹周期 / Episode 1')).toHaveAttribute('title', expect.stringContaining('reset 到结束'));
  });

  it('offers a bilingual light and dark theme toggle', () => {
    const onThemeChange = vi.fn();
    const { rerender } = render(
      <TopBar
        run={run}
        selectors={selectors}
        filters={filters}
        liveStatus="connected"
        liveEventCount={12}
        theme="light"
        onThemeChange={onThemeChange}
      />
    );

    fireEvent.click(screen.getByRole('button', { name: '深色主题 / Dark theme' }));
    expect(onThemeChange).toHaveBeenCalledWith('dark');

    rerender(
      <TopBar
        run={run}
        selectors={selectors}
        filters={filters}
        liveStatus="connected"
        liveEventCount={12}
        theme="dark"
        onThemeChange={onThemeChange}
      />
    );

    expect(screen.getByRole('button', { name: '浅色主题 / Light theme' })).toBeInTheDocument();
  });
});
