import { describe, expect, it } from 'vitest';

import { datasetMetricParams, metricParams } from './filters';

describe('metricParams', () => {
  it('adds a bounded max_points value for live dashboard queries', () => {
    expect(metricParams({ runId: 'run_a', live: true, compareMode: false })).toMatchObject({
      max_points: 600
    });
  });

  it('includes episode_id when replaying one trajectory', () => {
    expect(metricParams({ runId: 'run_a', episodeId: 2, live: true, compareMode: false })).toMatchObject({
      episode_id: 2
    });
  });

  it('includes agent, policy, and explicit time ranges when selected', () => {
    expect(
      metricParams({
        runId: 'run_a',
        agentId: 'agent_001',
        policyId: 'policy_shared',
        startTimeIndex: 12,
        endTimeIndex: 24,
        live: true,
        compareMode: false
      })
    ).toMatchObject({
      agent_id: 'agent_001',
      policy_id: 'policy_shared',
      start_time_index: 12,
      end_time_index: 24
    });
  });

  it('uses the single time index as a point query when no range is selected', () => {
    expect(metricParams({ runId: 'run_a', timeIndex: 7, live: true, compareMode: false })).toMatchObject({
      start_time_index: 7,
      end_time_index: 7
    });
  });

  it('uses a larger point budget for narrowed dataset timeseries views', () => {
    expect(
      datasetMetricParams({
        runId: 'run_a',
        date: '2018-01-07',
        vppId: 'vpp_commercial_multi',
        live: true,
        compareMode: false
      })
    ).toMatchObject({
      date: '2018-01-07',
      vpp_id: 'vpp_commercial_multi',
      max_points: 30000
    });
  });

  it('keeps the protective point budget for unfiltered dataset views', () => {
    expect(datasetMetricParams({ runId: 'run_a', live: true, compareMode: false })).toMatchObject({
      max_points: 600
    });
  });
});
