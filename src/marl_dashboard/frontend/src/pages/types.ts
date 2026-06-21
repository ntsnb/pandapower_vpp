import type { Filters, RunSummary, Selectors } from '../api/types';

export type PageProps = {
  filters: Filters;
  run: RunSummary | null;
  selectors: Selectors | null;
  liveEventCount?: number;
};
