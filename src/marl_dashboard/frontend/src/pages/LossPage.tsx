import { useMemo, useState } from 'react';

import { api } from '../api/client';
import { useAsync, useLiveTick } from '../api/hooks';
import { MultiPanelChart } from '../components/charts/MultiPanelChart';
import { DataNotice } from '../components/layout/DataNotice';
import { MetricSelector } from '../components/selectors/MetricSelector';
import { FormulaTable } from '../components/tables/FormulaTable';
import { MetricTable } from '../components/tables/MetricTable';
import { emptyResponse } from '../utils/emptyResponse';
import { metricParams } from '../utils/filters';
import { filterMetricResponse, metricLabelsFromResponse, metricNamesFromResponse } from '../utils/metrics';
import type { PageProps } from './types';

const EXPLODING_LOSS_THRESHOLD = 1_000_000;

type LossAnomaly = {
  key: string;
  metricName: string;
  reason: string;
  context: string;
};

function detectLossAnomalies(rows: NonNullable<ReturnType<typeof filterMetricResponse>>['table_rows']): LossAnomaly[] {
  return rows.flatMap((row, index) => {
    const rawValue = row.value;
    const numericValue = typeof rawValue === 'number' ? rawValue : Number(rawValue);
    const contextParts = [
      row.policy_id ? `policy=${row.policy_id}` : null,
      row.vpp_id ? `vpp=${row.vpp_id}` : null,
      row.gradient_step !== undefined && row.gradient_step !== null ? `gradient_step=${row.gradient_step}` : null,
      row.epoch_id !== undefined && row.epoch_id !== null ? `epoch=${row.epoch_id}` : null
    ].filter(Boolean);
    const context = contextParts.length > 0 ? contextParts.join(', ') : 'no context';

    if (!Number.isFinite(numericValue)) {
      return [
        {
          key: `${row.metric_name}-non-finite-${index}`,
          metricName: row.metric_name,
          reason: '非有限值 / NaN or Infinity',
          context
        }
      ];
    }
    if (Math.abs(numericValue) >= EXPLODING_LOSS_THRESHOLD) {
      return [
        {
          key: `${row.metric_name}-exploding-${index}`,
          metricName: row.metric_name,
          reason: `数量级异常 / Possible exploding loss (|loss| >= ${EXPLODING_LOSS_THRESHOLD.toLocaleString()})`,
          context
        }
      ];
    }
    return [];
  });
}

export function LossPage({ filters, liveEventCount = 0 }: PageProps) {
  const [lossMetric, setLossMetric] = useState('');
  const tick = useLiveTick(filters.live);
  const params = metricParams(filters);
  const losses = useAsync(() => (filters.runId ? api.losses(filters.runId, params) : Promise.resolve(emptyResponse)), [
    filters.runId,
    filters.vppId,
    filters.epochId,
    filters.episodeId,
    filters.policyId,
    liveEventCount,
    tick
  ]);
  const lossRowCount = Number(losses.data?.summary?.row_count ?? 0);
  const usesAggregateFallback = losses.data?.summary?.vpp_filter_fallback === 'aggregate_shared_loss';
  const lossMetrics = useMemo(() => metricNamesFromResponse(losses.data), [losses.data]);
  const lossMetricLabels = useMemo(() => metricLabelsFromResponse(losses.data), [losses.data]);
  const visibleLosses = useMemo(() => filterMetricResponse(losses.data, lossMetric), [losses.data, lossMetric]);
  const lossAnomalies = useMemo(() => detectLossAnomalies(visibleLosses?.table_rows ?? []), [visibleLosses]);

  return (
    <div className="page-stack">
      <DataNotice loading={losses.loading} error={losses.error} />
      <div className="panel control-panel">
        <div className="table-toolbar">
          <MetricSelector
            label="损失项 / Loss metric"
            allLabel="全部损失项 / All loss terms"
            metrics={lossMetrics}
            metricLabels={lossMetricLabels}
            value={lossMetric}
            onChange={setLossMetric}
          />
        </div>
      </div>
      {!losses.loading && !losses.error && lossRowCount === 0 ? (
        <div className="notice">
          当前筛选条件下没有 loss；loss 会在 learner update 后出现，且只有训练端记录 VPP 级 update_metrics 时才会有单 VPP loss /
          No loss rows match the current filters. Loss terms appear after learner updates, and per-VPP losses require
          VPP-level update_metrics from the training side.
        </div>
      ) : null}
      {!losses.loading && !losses.error && usesAggregateFallback ? (
        <div className="notice">
          当前 VPP 尚无私有 loss，正在显示适用于该 VPP 的聚合/共享策略 loss / This VPP has no private loss rows
          yet, so the view is showing aggregate or shared-policy losses that apply to the selected VPP.
        </div>
      ) : null}
      {!losses.loading && !losses.error && lossAnomalies.length > 0 ? (
        <section className="notice warning-notice" aria-label="损失异常告警 / Loss anomaly warning">
          <strong>损失异常告警 / Loss anomaly warning</strong>
          <p>
            以下 loss 行可能存在 NaN、Infinity 或 exploding loss。它们通常意味着梯度、学习率、归一化、reward
            尺度或 critic 目标值需要检查 / The rows below may indicate NaN, Infinity, or exploding loss values.
          </p>
          <ul className="compact-list">
            {lossAnomalies.slice(0, 8).map((anomaly) => (
              <li key={anomaly.key}>
                <span>{anomaly.metricName}</span>
                <span>{anomaly.reason}</span>
                <span>{anomaly.context}</span>
              </li>
            ))}
          </ul>
        </section>
      ) : null}
      <MultiPanelChart panels={[{ title: '学习器损失 / Learner losses', response: visibleLosses }]} />
      <FormulaTable
        title="损失公式 / Loss formulas"
        formulas={visibleLosses?.formulas ?? {}}
        rows={visibleLosses?.table_rows ?? []}
      />
      <MetricTable title="损失项 / Loss terms" rows={visibleLosses?.table_rows ?? []} />
    </div>
  );
}
