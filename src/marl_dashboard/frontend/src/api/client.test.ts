import { afterEach, describe, expect, it, vi } from 'vitest';

import { api } from './client';

afterEach(() => {
  vi.restoreAllMocks();
});

describe('api client', () => {
  it('deduplicates concurrent requests for the same URL', async () => {
    const fetchMock = vi.fn(() =>
      Promise.resolve(
        new Response(JSON.stringify({ chart_series: [], table_rows: [], units: {}, formulas: {}, summary: { row_count: 0 } }), {
          status: 200,
          headers: { 'Content-Type': 'application/json' }
        })
      )
    );
    vi.stubGlobal('fetch', fetchMock);

    await Promise.all([
      api.dataset('run_a', { vpp_id: 'vpp_001', max_points: 600 }),
      api.dataset('run_a', { vpp_id: 'vpp_001', max_points: 600 })
    ]);

    expect(fetchMock).toHaveBeenCalledTimes(1);
  });

  it('requests run events with query filters', async () => {
    const fetchMock = vi.fn(() =>
      Promise.resolve(
        new Response(JSON.stringify({ chart_series: [], table_rows: [], units: {}, formulas: {}, summary: { row_count: 0 } }), {
          status: 200,
          headers: { 'Content-Type': 'application/json' }
        })
      )
    );
    vi.stubGlobal('fetch', fetchMock);

    await api.events('run_a', { max_points: 20, episode_id: 3 });

    expect(fetchMock).toHaveBeenCalledWith('/api/runs/run_a/events?max_points=20&episode_id=3');
  });
});
