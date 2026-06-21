import { useEffect, useMemo, useState } from 'react';

import { api } from '../api/client';
import { useAsync, useLiveTick } from '../api/hooks';
import { SameTimeCompareChart } from '../components/charts/SameTimeCompareChart';
import { DataNotice } from '../components/layout/DataNotice';
import { MetricSelector } from '../components/selectors/MetricSelector';
import { CompareMetricMatrix } from '../components/tables/CompareMetricMatrix';
import { MetricTable } from '../components/tables/MetricTable';
import { emptyResponse } from '../utils/emptyResponse';
import { metricLabelsFromResponse, metricNamesFromResponse } from '../utils/metrics';
import type { PageProps } from './types';

type CompareScope = 'dataset' | 'reward' | 'cost' | 'loss';
type CompareGroupBy = 'vpp_id' | 'epoch_id' | 'policy_id' | 'agent_id';

const scopeOptions: Array<{ value: CompareScope; label: string }> = [
  { value: 'dataset', label: '数据集 / Dataset' },
  { value: 'reward', label: '奖励 / Reward' },
  { value: 'cost', label: '成本 / Cost' },
  { value: 'loss', label: '损失 / Loss' }
];

const groupOptions: Array<{ value: CompareGroupBy; label: string }> = [
  { value: 'vpp_id', label: 'VPP' },
  { value: 'epoch_id', label: 'Epoch' },
  { value: 'policy_id', label: 'Policy' },
  { value: 'agent_id', label: 'Agent' }
];

function groupValueOptions(groupBy: CompareGroupBy, selectors: PageProps['selectors']): string[] {
  if (!selectors) {
    return [];
  }
  if (groupBy === 'vpp_id') {
    return selectors.vpp_ids;
  }
  if (groupBy === 'agent_id') {
    return selectors.agent_ids;
  }
  if (groupBy === 'policy_id') {
    return selectors.policy_ids;
  }
  return selectors.epoch_ids.map(String);
}

export function ComparePage({ filters, selectors, liveEventCount = 0 }: PageProps) {
  const [scope, setScope] = useState<CompareScope>('reward');
  const [groupBy, setGroupBy] = useState<CompareGroupBy>('vpp_id');
  const [selectedGroups, setSelectedGroups] = useState<string[]>([]);
  const [metricName, setMetricName] = useState('');
  const tick = useLiveTick(filters.live);
  const availableGroupValues = useMemo(() => groupValueOptions(groupBy, selectors), [groupBy, selectors]);
  const groupValuesParam = selectedGroups.length > 0 ? selectedGroups.join(',') : undefined;

  useEffect(() => {
    setSelectedGroups((current) => current.filter((value) => availableGroupValues.includes(value)));
  }, [availableGroupValues]);

  const compare = useAsync(
    () =>
      filters.runId
        ? api.compare(filters.runId, {
            scope,
            fixed_epoch_id: filters.epochId,
            fixed_episode_id: filters.episodeId,
            fixed_date: filters.date,
            fixed_time_index: filters.timeIndex,
            metric_names: metricName,
            group_by: groupBy,
            group_values: groupValuesParam,
            max_points: 600
          })
        : Promise.resolve(emptyResponse),
    [
      filters.runId,
      filters.date,
      filters.epochId,
      filters.episodeId,
      filters.timeIndex,
      scope,
      groupBy,
      metricName,
      groupValuesParam,
      liveEventCount,
      tick
    ]
  );
  const metricNames = useMemo(() => metricNamesFromResponse(compare.data), [compare.data]);
  const metricLabels = useMemo(() => metricLabelsFromResponse(compare.data), [compare.data]);

  return (
    <div className="page-stack">
      <DataNotice loading={compare.loading} error={compare.error} />
      <div className="panel control-panel">
        <div className="table-toolbar">
          <label>
            对比范围 / Compare scope
            <select
              value={scope}
              onChange={(event) => {
                setScope(event.target.value as CompareScope);
                setMetricName('');
              }}
            >
              {scopeOptions.map((option) => (
                <option key={option.value} value={option.value}>
                  {option.label}
                </option>
              ))}
            </select>
          </label>
          <label>
            分组维度 / Group by
            <select
              value={groupBy}
              onChange={(event) => {
                setGroupBy(event.target.value as CompareGroupBy);
                setSelectedGroups([]);
              }}
            >
              {groupOptions.map((option) => (
                <option key={option.value} value={option.value}>
                  {option.label}
                </option>
              ))}
            </select>
          </label>
          <label title="选择一个或多个对比对象；留空表示显示该分组维度下全部对象。 / Select one or more comparison groups; leave empty to show all groups in the selected dimension.">
            对比对象 / Compare groups
            <select
              multiple
              value={selectedGroups}
              onChange={(event) => setSelectedGroups(Array.from(event.currentTarget.selectedOptions, (option) => option.value))}
              disabled={availableGroupValues.length === 0}
            >
              {availableGroupValues.map((value) => (
                <option key={value} value={value}>
                  {value}
                </option>
              ))}
            </select>
          </label>
          <MetricSelector
            label="指标 / Metric"
            allLabel="全部指标 / All metrics"
            metrics={metricNames}
            metricLabels={metricLabels}
            value={metricName}
            onChange={setMetricName}
          />
        </div>
      </div>
      <SameTimeCompareChart response={compare.data} />
      <CompareMetricMatrix rows={compare.data?.table_rows ?? []} units={compare.data?.units ?? {}} groupBy={groupBy} />
      <MetricTable title="对比明细 / Comparison rows" rows={compare.data?.table_rows ?? []} />
    </div>
  );
}
