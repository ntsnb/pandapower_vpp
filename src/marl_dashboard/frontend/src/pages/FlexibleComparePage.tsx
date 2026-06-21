import { useMemo, useState } from 'react';

import { api } from '../api/client';
import { useAsync, useLiveTick } from '../api/hooks';
import type { QueryResponse } from '../api/types';
import { CombinedChart } from '../components/charts/CombinedChart';
import { DataNotice } from '../components/layout/DataNotice';
import { MetricTable } from '../components/tables/MetricTable';
import { dateOptionLabel, dateStatusByDate } from '../utils/dateStatus';
import { emptyResponse } from '../utils/emptyResponse';
import { metricLabelsFromResponse, metricNamesFromResponse } from '../utils/metrics';
import type { PageProps } from './types';

type FlexibleScope = 'dataset' | 'reward' | 'cost' | 'loss';

type CurveDraft = {
  scope: FlexibleScope;
  metricName: string;
  vppId: string;
  date: string;
  epochId: string;
  episodeId: string;
  startTimeIndex: string;
  endTimeIndex: string;
  policyId: string;
};

type FlexibleCurve = CurveDraft & {
  id: string;
};

type CurveResult = {
  curve: FlexibleCurve;
  response: QueryResponse;
};

const scopeOptions: Array<{ value: FlexibleScope; label: string }> = [
  { value: 'dataset', label: '数据集 / Dataset' },
  { value: 'reward', label: '奖励 / Reward' },
  { value: 'cost', label: '成本 / Cost' },
  { value: 'loss', label: '损失 / Loss' }
];

const scopeLabels: Record<FlexibleScope, string> = Object.fromEntries(scopeOptions.map((option) => [option.value, option.label])) as Record<
  FlexibleScope,
  string
>;

function initialDraft(): CurveDraft {
  return {
    scope: 'dataset',
    metricName: '',
    vppId: '',
    date: '',
    epochId: '',
    episodeId: '',
    startTimeIndex: '',
    endTimeIndex: '',
    policyId: ''
  };
}

function numericParam(value: string): number | undefined {
  if (value === '') {
    return undefined;
  }
  const numeric = Number(value);
  return Number.isFinite(numeric) ? numeric : undefined;
}

function metricNames(response: QueryResponse | null | undefined): string[] {
  return Array.from(
    new Set([
      ...metricNamesFromResponse(response),
      ...Object.keys(response?.units ?? {}),
      ...Object.keys(response?.formulas ?? {})
    ])
  ).sort((left, right) => left.localeCompare(right));
}

function requestParams(curve: CurveDraft, maxPoints = 1000): Record<string, string | number | undefined> {
  return {
    metrics: curve.metricName,
    date: curve.date,
    vpp_id: curve.vppId,
    epoch_id: numericParam(curve.epochId),
    episode_id: numericParam(curve.episodeId),
    start_time_index: numericParam(curve.startTimeIndex),
    end_time_index: numericParam(curve.endTimeIndex),
    policy_id: curve.policyId,
    max_points: maxPoints
  };
}

function fetchScope(runId: string, scope: FlexibleScope, params: Record<string, string | number | undefined>) {
  if (scope === 'dataset') return api.dataset(runId, params);
  if (scope === 'reward') return api.rewards(runId, params);
  if (scope === 'cost') return api.costs(runId, params);
  return api.losses(runId, params);
}

function curveLabel(curve: CurveDraft, response: QueryResponse | null | undefined): string {
  const labels = metricLabelsFromResponse(response);
  const metricLabel = labels[curve.metricName] ?? curve.metricName;
  const parts = [
    scopeLabels[curve.scope],
    metricLabel,
    curve.vppId ? `VPP ${curve.vppId}` : '全部 VPP / All VPPs',
    curve.date,
    curve.epochId ? `epoch ${curve.epochId}` : '',
    curve.episodeId ? `episode ${curve.episodeId}` : '',
    curve.policyId ? `policy ${curve.policyId}` : ''
  ].filter(Boolean);
  return parts.join(' / ');
}

function mergeTableRows(results: CurveResult[]): QueryResponse {
  return {
    chart_series: [],
    table_rows: results.flatMap((result) => result.response.table_rows),
    units: Object.assign({}, ...results.map((result) => result.response.units)),
    formulas: Object.assign({}, ...results.map((result) => result.response.formulas)),
    summary: { row_count: results.reduce((count, result) => count + Number(result.response.summary.row_count ?? 0), 0) }
  };
}

export function FlexibleComparePage({ filters, selectors, liveEventCount = 0 }: PageProps) {
  const [draft, setDraft] = useState<CurveDraft>(initialDraft);
  const [curves, setCurves] = useState<FlexibleCurve[]>([]);
  const tick = useLiveTick(filters.live);
  const preview = useAsync(
    () => (filters.runId ? fetchScope(filters.runId, draft.scope, requestParams({ ...draft, metricName: '' }, 1000)) : Promise.resolve(emptyResponse)),
    [
      filters.runId,
      draft.scope,
      draft.vppId,
      draft.date,
      draft.epochId,
      draft.episodeId,
      draft.startTimeIndex,
      draft.endTimeIndex,
      draft.policyId,
      liveEventCount,
      tick
    ]
  );
  const curveResults = useAsync(
    () =>
      filters.runId
        ? Promise.all(
            curves.map(async (curve) => ({
              curve,
              response: await fetchScope(filters.runId, curve.scope, requestParams(curve, 1200))
            }))
          )
        : Promise.resolve([]),
    [filters.runId, curves, liveEventCount, tick]
  );
  const availableMetrics = useMemo(() => metricNames(preview.data), [preview.data]);
  const metricLabels = useMemo(() => metricLabelsFromResponse(preview.data), [preview.data]);
  const dateStatuses = useMemo(() => dateStatusByDate(selectors?.date_statuses), [selectors?.date_statuses]);
  const results = curveResults.data ?? [];
  const combinedRows = useMemo(() => mergeTableRows(results), [results]);

  const updateDraft = (patch: Partial<CurveDraft>) => setDraft((current) => ({ ...current, ...patch }));
  const addCurve = () => {
    if (!draft.metricName) {
      return;
    }
    setCurves((current) => [...current, { ...draft, id: `${Date.now()}-${current.length}` }]);
  };
  const removeCurve = (id: string) => setCurves((current) => current.filter((curve) => curve.id !== id));

  return (
    <div className="page-stack">
      <DataNotice loading={preview.loading || curveResults.loading} error={preview.error ?? curveResults.error} />
      <section className="notice">
        <p>
          数据时间 / Data time: <code>date + time_index</code> 表示能源数据和仿真的真实时间，例如 2018-01-01 #0；部分数据也会用
          <code>timestamp</code> 保存同一个仿真时间。
        </p>
        <p>
          日志写入时间 / Log write time: <code>logged_at</code> 表示训练进程写入日志的机器时间，例如 2026-06-16。
        </p>
        <p>
          Flexible 横坐标优先使用 date + time_index；只有缺少数据时间时，才退回到 gradient_step、global_env_step 或 timestamp。
        </p>
      </section>
      <section className="panel control-panel">
        <div className="table-toolbar">
          <label>
            数据域 / Data scope
            <select value={draft.scope} onChange={(event) => updateDraft({ scope: event.target.value as FlexibleScope, metricName: '' })}>
              {scopeOptions.map((option) => (
                <option key={option.value} value={option.value}>
                  {option.label}
                </option>
              ))}
            </select>
          </label>
          <label>
            VPP
            <select value={draft.vppId} onChange={(event) => updateDraft({ vppId: event.target.value })}>
              <option value="">全部 VPP / All VPPs</option>
              {(selectors?.vpp_ids ?? []).map((vppId) => (
                <option key={vppId} value={vppId}>
                  {vppId}
                </option>
              ))}
            </select>
          </label>
          <label>
            指标 / Metric
            <select value={draft.metricName} onChange={(event) => updateDraft({ metricName: event.target.value })}>
              <option value="">请选择指标 / Select metric</option>
              {availableMetrics.map((metricName) => (
                <option key={metricName} value={metricName}>
                  {metricLabels[metricName] ?? metricName}
                </option>
              ))}
            </select>
          </label>
          <label>
            日期 / Date
            <select value={draft.date} onChange={(event) => updateDraft({ date: event.target.value })}>
              <option value="">全部日期 / All dates</option>
              {(selectors?.dates ?? []).map((date) => (
                <option key={date} value={date}>
                  {dateOptionLabel(date, dateStatuses.get(date))}
                </option>
              ))}
            </select>
          </label>
          <label>
            训练轮次 / Epoch
            <select value={draft.epochId} onChange={(event) => updateDraft({ epochId: event.target.value })}>
              <option value="">全部 epoch / All epochs</option>
              {(selectors?.epoch_ids ?? []).map((epochId) => (
                <option key={epochId} value={epochId}>
                  {epochId}
                </option>
              ))}
            </select>
          </label>
          <label>
            轨迹 / Episode
            <select value={draft.episodeId} onChange={(event) => updateDraft({ episodeId: event.target.value })}>
              <option value="">全部 episode / All episodes</option>
              {(selectors?.episode_ids ?? []).map((episodeId) => (
                <option key={episodeId} value={episodeId}>
                  {episodeId}
                </option>
              ))}
            </select>
          </label>
          <label>
            策略 / Policy
            <select value={draft.policyId} onChange={(event) => updateDraft({ policyId: event.target.value })}>
              <option value="">全部 policy / All policies</option>
              {(selectors?.policy_ids ?? []).map((policyId) => (
                <option key={policyId} value={policyId}>
                  {policyId}
                </option>
              ))}
            </select>
          </label>
          <label>
            起始时刻 / Start time
            <input type="number" value={draft.startTimeIndex} onChange={(event) => updateDraft({ startTimeIndex: event.target.value })} />
          </label>
          <label>
            结束时刻 / End time
            <input type="number" value={draft.endTimeIndex} onChange={(event) => updateDraft({ endTimeIndex: event.target.value })} />
          </label>
          <button onClick={addCurve} disabled={!draft.metricName}>
            添加曲线 / Add curve
          </button>
          <button onClick={() => setCurves([])}>清空曲线 / Clear curves</button>
        </div>
      </section>

      <CombinedChart
        title="自由曲线对比 / Flexible curve comparison"
        responses={curves.map((curve) => {
          const result = results.find((item) => item.curve.id === curve.id);
          return { label: curveLabel(curve, result?.response), response: result?.response ?? emptyResponse };
        })}
      />

      <section className="panel table-panel">
        <div className="panel-header">
          <h2>已选曲线 / Selected curves</h2>
          <span>{curves.length} 条 / curves</span>
        </div>
        <div className="table-scroll">
          <table>
            <thead>
              <tr>
                <th>曲线 / Curve</th>
                <th>数据域 / Scope</th>
                <th>指标 / Metric</th>
                <th>VPP</th>
                <th>日期 / Date</th>
                <th>Epoch</th>
                <th>Episode</th>
                <th>时刻 / Time</th>
                <th>操作 / Action</th>
              </tr>
            </thead>
            <tbody>
              {curves.map((curve) => (
                <tr key={curve.id}>
                  <td>{curveLabel(curve, results.find((item) => item.curve.id === curve.id)?.response)}</td>
                  <td>{scopeLabels[curve.scope]}</td>
                  <td>{curve.metricName}</td>
                  <td>{curve.vppId || '全部 / All'}</td>
                  <td>{curve.date || '全部 / All'}</td>
                  <td>{curve.epochId || '全部 / All'}</td>
                  <td>{curve.episodeId ? `episode ${curve.episodeId}` : '全部 / All'}</td>
                  <td>
                    {curve.startTimeIndex || '-'} to {curve.endTimeIndex || '-'}
                  </td>
                  <td>
                    <button onClick={() => removeCurve(curve.id)}>移除 / Remove</button>
                  </td>
                </tr>
              ))}
              {curves.length === 0 ? (
                <tr>
                  <td colSpan={9} className="empty-cell">
                    尚未添加曲线 / No curves selected
                  </td>
                </tr>
              ) : null}
            </tbody>
          </table>
        </div>
      </section>

      <MetricTable title="自由对比数据行 / Flexible comparison rows" rows={combinedRows.table_rows} />
    </div>
  );
}
