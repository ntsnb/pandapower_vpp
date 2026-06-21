import type { QueryResponse, RunMetadata, RunSummary, Selectors, TopologyPayload, VariableDefinition, VppConfigPayload } from './types';

const API_ROOT = '';
const inFlight = new Map<string, Promise<unknown>>();

async function fetchJson<T>(path: string): Promise<T> {
  const url = `${API_ROOT}${path}`;
  const existing = inFlight.get(url);
  if (existing) {
    return existing as Promise<T>;
  }
  const request = fetch(url)
    .then((response) => {
      if (!response.ok) {
        throw new Error(`${response.status} ${response.statusText}: ${path}`);
      }
      return response.json() as Promise<T>;
    })
    .finally(() => {
      inFlight.delete(url);
    });
  inFlight.set(url, request);
  return request;
}

function query(params: Record<string, string | number | boolean | undefined>): string {
  const search = new URLSearchParams();
  for (const [key, value] of Object.entries(params)) {
    if (value !== undefined && value !== '') {
      search.set(key, String(value));
    }
  }
  const text = search.toString();
  return text ? `?${text}` : '';
}

export const api = {
  health: () => fetchJson<Record<string, string>>('/api/health'),
  runs: () => fetchJson<RunSummary[]>('/api/runs'),
  metadata: (runId: string) => fetchJson<RunMetadata>(`/api/runs/${runId}/metadata`),
  selectors: (runId: string) => fetchJson<Selectors>(`/api/runs/${runId}/selectors`),
  variables: (runId: string) => fetchJson<VariableDefinition[]>(`/api/runs/${runId}/variables`),
  formulas: (runId: string) => fetchJson<Record<string, string>>(`/api/runs/${runId}/formulas`),
  dataset: (runId: string, params: Record<string, string | number | boolean | undefined>) =>
    fetchJson<QueryResponse>(`/api/runs/${runId}/dataset${query(params)}`),
  rewards: (runId: string, params: Record<string, string | number | boolean | undefined>) =>
    fetchJson<QueryResponse>(`/api/runs/${runId}/rewards${query(params)}`),
  costs: (runId: string, params: Record<string, string | number | boolean | undefined>) =>
    fetchJson<QueryResponse>(`/api/runs/${runId}/costs${query(params)}`),
  losses: (runId: string, params: Record<string, string | number | boolean | undefined>) =>
    fetchJson<QueryResponse>(`/api/runs/${runId}/losses${query(params)}`),
  scalars: (runId: string, params: Record<string, string | number | boolean | undefined>) =>
    fetchJson<QueryResponse>(`/api/runs/${runId}/scalars${query(params)}`),
  events: (runId: string, params: Record<string, string | number | boolean | undefined>) =>
    fetchJson<QueryResponse>(`/api/runs/${runId}/events${query(params)}`),
  compare: (runId: string, params: Record<string, string | number | boolean | undefined>) =>
    fetchJson<QueryResponse>(`/api/runs/${runId}/compare${query(params)}`),
  topology: (runId: string) => fetchJson<TopologyPayload>(`/api/runs/${runId}/topology`),
  vppConfig: (runId: string) => fetchJson<VppConfigPayload>(`/api/runs/${runId}/vpp-config`)
};
