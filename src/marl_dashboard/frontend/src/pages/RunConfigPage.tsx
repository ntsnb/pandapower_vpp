import { api } from '../api/client';
import { useAsync, useLiveTick } from '../api/hooks';
import { DataNotice } from '../components/layout/DataNotice';
import { MetricTable } from '../components/tables/MetricTable';
import { emptyResponse } from '../utils/emptyResponse';
import { metricParams } from '../utils/filters';
import type { PageProps } from './types';

function jsonText(payload: unknown): string {
  return JSON.stringify(payload ?? null, null, 2);
}

function downloadJson(filename: string, payload: unknown): void {
  const blob = new Blob([jsonText(payload)], { type: 'application/json;charset=utf-8' });
  const url = URL.createObjectURL(blob);
  const link = document.createElement('a');
  link.href = url;
  link.download = filename;
  document.body.appendChild(link);
  link.click();
  link.remove();
  URL.revokeObjectURL(url);
}

function configValue(config: Record<string, unknown> | undefined, key: string): string {
  const value = config?.[key];
  if (value === undefined || value === null || value === '') {
    return '-';
  }
  return String(value);
}

function runFilename(runId: string, suffix: string): string {
  const safeRunId = runId.replace(/[^a-zA-Z0-9_.-]+/g, '_') || 'run';
  return `${safeRunId}_${suffix}.json`;
}

export function RunConfigPage({ filters, liveEventCount = 0 }: PageProps) {
  const tick = useLiveTick(filters.live, 10000);
  const params = metricParams(filters);
  const metadata = useAsync(() => (filters.runId ? api.metadata(filters.runId) : Promise.resolve(null)), [filters.runId, liveEventCount, tick]);
  const variables = useAsync(() => (filters.runId ? api.variables(filters.runId) : Promise.resolve([])), [filters.runId, liveEventCount, tick]);
  const formulas = useAsync(() => (filters.runId ? api.formulas(filters.runId) : Promise.resolve({})), [filters.runId, liveEventCount, tick]);
  const events = useAsync(
    () => (filters.runId ? api.events(filters.runId, { ...params, max_points: 100 }) : Promise.resolve(emptyResponse)),
    [
      filters.runId,
      filters.date,
      filters.vppId,
      filters.epochId,
      filters.episodeId,
      filters.timeIndex,
      liveEventCount,
      tick
    ]
  );
  const config = metadata.data?.config;
  const runId = metadata.data?.run_id ?? filters.runId ?? 'run';

  return (
    <div className="page-stack">
      <DataNotice
        loading={metadata.loading || variables.loading || formulas.loading || events.loading}
        error={metadata.error ?? variables.error ?? formulas.error ?? events.error}
      />
      <section className="panel">
        <div className="panel-header">
          <h2>运行配置 / Run configuration</h2>
          <span>{metadata.data?.status ?? 'unknown'}</span>
        </div>
        <div className="stat-grid">
          <section className="stat-panel">
            <span>算法 / Algorithm</span>
            <strong>{metadata.data?.algorithm ?? '-'}</strong>
          </section>
          <section className="stat-panel">
            <span>环境 / Environment</span>
            <strong>{metadata.data?.environment ?? '-'}</strong>
          </section>
          <section className="stat-panel">
            <span>Seed</span>
            <strong>{configValue(config, 'seed')}</strong>
          </section>
          <section className="stat-panel">
            <span>VPP 数量 / VPP count</span>
            <strong>{configValue(config, 'vpp_count')}</strong>
          </section>
          <section className="stat-panel">
            <span>Batch size</span>
            <strong>{configValue(config, 'batch_size')}</strong>
          </section>
          <section className="stat-panel">
            <span>Learning rate</span>
            <strong>{configValue(config, 'learning_rate')}</strong>
          </section>
          <section className="stat-panel">
            <span>Episode horizon</span>
            <strong>{configValue(config, 'episode_horizon_steps')}</strong>
          </section>
          <section className="stat-panel">
            <span>Started at</span>
            <strong>{metadata.data?.started_at ?? '-'}</strong>
          </section>
        </div>
        <div className="table-toolbar">
          <button onClick={() => downloadJson(runFilename(runId, 'metadata'), metadata.data)}>下载 metadata / Download metadata</button>
          <button onClick={() => downloadJson(runFilename(runId, 'config'), config ?? {})}>下载 config / Download config</button>
          <button onClick={() => downloadJson(runFilename(runId, 'formulas'), formulas.data ?? {})}>下载 formulas / Download formulas</button>
          <button onClick={() => downloadJson(runFilename(runId, 'variable_dictionary'), variables.data ?? [])}>
            下载变量字典 / Download variable dictionary
          </button>
        </div>
      </section>
      <MetricTable title="事件日志 / Event log" rows={events.data?.table_rows ?? []} limit={25} />
      <section className="panel">
        <div className="panel-header">
          <h2>元数据 JSON / Metadata JSON</h2>
          <span>{runId}</span>
        </div>
        <pre className="json-block">{jsonText(metadata.data)}</pre>
      </section>
      <section className="panel">
        <div className="panel-header">
          <h2>变量字典 JSON / Variable dictionary JSON</h2>
          <span>{variables.data?.length ?? 0} variables</span>
        </div>
        <pre className="json-block">{jsonText(variables.data ?? [])}</pre>
      </section>
      <section className="panel">
        <div className="panel-header">
          <h2>公式 JSON / Formula JSON</h2>
          <span>{Object.keys(formulas.data ?? {}).length} formulas</span>
        </div>
        <pre className="json-block">{jsonText(formulas.data ?? {})}</pre>
      </section>
    </div>
  );
}
