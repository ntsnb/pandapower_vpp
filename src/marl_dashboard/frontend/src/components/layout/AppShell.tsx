import type { ReactNode } from 'react';

import type { Filters, RunSummary, Selectors, WebSocketStatus } from '../../api/types';
import { FilterBar } from './FilterBar';
import { Sidebar } from './Sidebar';
import { ThemeMode, TopBar } from './TopBar';

type Props = {
  children: ReactNode;
  page: string;
  onPageChange: (page: string) => void;
  runs: RunSummary[];
  run: RunSummary | null;
  selectors: Selectors | null;
  filters: Filters;
  onFiltersChange: (filters: Filters) => void;
  liveStatus?: WebSocketStatus;
  liveEventCount?: number;
  theme?: ThemeMode;
  onThemeChange?: (theme: ThemeMode) => void;
};

export function AppShell({
  children,
  page,
  onPageChange,
  runs,
  run,
  selectors,
  filters,
  onFiltersChange,
  liveStatus,
  liveEventCount,
  theme,
  onThemeChange
}: Props) {
  return (
    <div className="app-shell">
      <Sidebar page={page} onPageChange={onPageChange} />
      <main>
        <TopBar
          run={run}
          selectors={selectors}
          filters={filters}
          liveStatus={liveStatus}
          liveEventCount={liveEventCount}
          theme={theme}
          onThemeChange={onThemeChange}
        />
        <FilterBar runs={runs} selectors={selectors} filters={filters} onChange={onFiltersChange} />
        {children}
      </main>
    </div>
  );
}
