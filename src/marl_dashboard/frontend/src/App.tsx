import { useEffect, useMemo, useState } from 'react';

import { api } from './api/client';
import { useAsync, useLiveTick, useRunWebSocket } from './api/hooks';
import type { Filters } from './api/types';
import { AppShell } from './components/layout/AppShell';
import { DataNotice } from './components/layout/DataNotice';
import type { ThemeMode } from './components/layout/TopBar';
import { ComparePage } from './pages/ComparePage';
import { DatasetPage } from './pages/DatasetPage';
import { FlexibleComparePage } from './pages/FlexibleComparePage';
import { LossPage } from './pages/LossPage';
import { OverviewPage } from './pages/OverviewPage';
import { RewardCostPage } from './pages/RewardCostPage';
import { RunConfigPage } from './pages/RunConfigPage';
import { TopologyPage } from './pages/TopologyPage';
import { VariableDictionaryPage } from './pages/VariableDictionaryPage';

const initialFilters: Filters = { runId: '', live: true, compareMode: false };
const THEME_STORAGE_KEY = 'marl-dashboard-theme';

function initialTheme(): ThemeMode {
  try {
    const storedTheme = window.localStorage.getItem(THEME_STORAGE_KEY);
    return storedTheme === 'dark' ? 'dark' : 'light';
  } catch {
    return 'light';
  }
}

export function App() {
  const [page, setPage] = useState('overview');
  const [filters, setFilters] = useState<Filters>(initialFilters);
  const [theme, setTheme] = useState<ThemeMode>(initialTheme);
  const runsTick = useLiveTick(true, 5000);
  const runs = useAsync(() => api.runs(), [runsTick]);
  const selectors = useAsync(() => (filters.runId ? api.selectors(filters.runId) : Promise.resolve(null)), [filters.runId, runsTick]);
  const liveSocket = useRunWebSocket(filters.runId || undefined, filters.live);

  useEffect(() => {
    document.documentElement.dataset.theme = theme;
    try {
      window.localStorage.setItem(THEME_STORAGE_KEY, theme);
    } catch {
      // Theme persistence is optional; the UI still works if storage is blocked.
    }
  }, [theme]);

  useEffect(() => {
    const firstRun = runs.data?.[0]?.run_id;
    if (!firstRun) {
      return;
    }
    setFilters((current) => (current.runId ? current : { ...current, runId: firstRun }));
  }, [runs.data]);

  useEffect(() => {
    const data = selectors.data;
    if (!data) {
      return;
    }
    setFilters((current) => {
      if (current.runId !== data.run_id) {
        return current;
      }
      const next: Filters = { ...current };
      let changed = false;
      if (current.vppId && !data.vpp_ids.includes(current.vppId)) {
        next.vppId = undefined;
        changed = true;
      }
      if (current.agentId && !data.agent_ids.includes(current.agentId)) {
        next.agentId = undefined;
        changed = true;
      }
      if (current.policyId && !data.policy_ids.includes(current.policyId)) {
        next.policyId = undefined;
        changed = true;
      }
      if (current.date && !data.dates.includes(current.date)) {
        next.date = undefined;
        changed = true;
      }
      if (current.epochId !== undefined && !data.epoch_ids.includes(current.epochId)) {
        next.epochId = undefined;
        changed = true;
      }
      if (current.episodeId !== undefined && !data.episode_ids.includes(current.episodeId)) {
        next.episodeId = undefined;
        changed = true;
      }
      if (current.timeIndex !== undefined && !data.time_indices.includes(current.timeIndex)) {
        next.timeIndex = undefined;
        changed = true;
      }
      if (current.startTimeIndex !== undefined && !data.time_indices.includes(current.startTimeIndex)) {
        next.startTimeIndex = undefined;
        changed = true;
      }
      if (current.endTimeIndex !== undefined && !data.time_indices.includes(current.endTimeIndex)) {
        next.endTimeIndex = undefined;
        changed = true;
      }
      return changed ? next : current;
    });
  }, [selectors.data]);

  const selectedRun = useMemo(() => runs.data?.find((run) => run.run_id === filters.runId) ?? null, [runs.data, filters.runId]);

  function renderPage() {
    const props = { filters, run: selectedRun, selectors: selectors.data, liveEventCount: liveSocket.eventCount };
    if (page === 'dataset') return <DatasetPage {...props} />;
    if (page === 'reward-cost') return <RewardCostPage {...props} />;
    if (page === 'loss') return <LossPage {...props} />;
    if (page === 'compare') return <ComparePage {...props} />;
    if (page === 'flexible') return <FlexibleComparePage {...props} />;
    if (page === 'topology') return <TopologyPage {...props} />;
    if (page === 'variables') return <VariableDictionaryPage {...props} />;
    if (page === 'config') return <RunConfigPage {...props} />;
    return <OverviewPage {...props} />;
  }

  return (
    <AppShell
      page={page}
      onPageChange={setPage}
      runs={runs.data ?? []}
      run={selectedRun}
      selectors={selectors.data}
      filters={filters}
      onFiltersChange={setFilters}
      liveStatus={liveSocket.status}
      liveEventCount={liveSocket.eventCount}
      theme={theme}
      onThemeChange={setTheme}
    >
      <DataNotice loading={runs.loading || selectors.loading} error={runs.error ?? selectors.error} />
      {renderPage()}
    </AppShell>
  );
}
