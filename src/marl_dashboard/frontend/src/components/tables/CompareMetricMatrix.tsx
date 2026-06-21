import { useEffect, useMemo, useState } from 'react';

import type { MetricRow } from '../../api/types';
import { compactNumber } from '../../utils/filters';

type Props = {
  rows: MetricRow[];
  units: Record<string, string>;
  groupBy: string;
  limit?: number;
};

type MatrixRow = {
  group: string;
  sample?: MetricRow;
  values: Record<string, MetricRow>;
};

const preferredMetricOrder = [
  'electricity_price',
  'market_price',
  'ev_charging_load',
  'storage_power',
  'storage_soc',
  'pv_power',
  'wind_power',
  'base_load',
  'net_load',
  'reward_so_far',
  'total_reward',
  'total_cost',
  'actor_loss',
  'critic_loss',
  'entropy_loss',
  'value_loss',
  'q_loss',
  'total_loss'
];

function csvValue(value: unknown): string {
  if (value === undefined || value === null) {
    return '';
  }
  const text = String(value);
  return /[",\n]/.test(text) ? `"${text.replaceAll('"', '""')}"` : text;
}

function metricSortKey(metricName: string): [number, string] {
  const index = preferredMetricOrder.indexOf(metricName);
  return [index === -1 ? preferredMetricOrder.length : index, metricName];
}

function compareMetricName(left: string, right: string): number {
  const [leftIndex, leftName] = metricSortKey(left);
  const [rightIndex, rightName] = metricSortKey(right);
  return leftIndex === rightIndex ? leftName.localeCompare(rightName) : leftIndex - rightIndex;
}

function metricLabel(metricName: string, rows: MetricRow[], units: Record<string, string>): string {
  const sample = rows.find((row) => row.metric_name === metricName);
  const label = sample?.display_name || sample?.description || metricName;
  const unit = units[metricName] || sample?.unit;
  return unit ? `${label} (${unit})` : label;
}

function groupValue(row: MetricRow, groupBy: string): string {
  const value =
    row.group ??
    (groupBy === 'vpp_id'
      ? row.vpp_id
      : groupBy === 'epoch_id'
        ? row.epoch_id
        : groupBy === 'policy_id'
          ? row.policy_id
          : groupBy === 'agent_id'
            ? row.agent_id
            : null) ??
    row.vpp_id ??
    row.policy_id ??
    row.agent_id ??
    row.epoch_id ??
    'unknown';
  return String(value);
}

function timeLabel(row: MetricRow | undefined): string {
  if (!row) {
    return '-';
  }
  if (row.date && row.time_index !== undefined && row.time_index !== null) {
    return `${row.date} #${row.time_index}`;
  }
  if (row.timestamp) {
    return row.timestamp.slice(0, 19).replace('T', ' ');
  }
  return row.time_index !== undefined && row.time_index !== null ? `#${row.time_index}` : '-';
}

function pivotRows(rows: MetricRow[], groupBy: string): MatrixRow[] {
  const byGroup = new Map<string, MatrixRow>();
  for (const row of rows) {
    if (!row.metric_name) {
      continue;
    }
    const group = groupValue(row, groupBy);
    const existing = byGroup.get(group) ?? { group, sample: row, values: {} };
    existing.sample = existing.sample ?? row;
    existing.values[row.metric_name] = row;
    byGroup.set(group, existing);
  }
  return Array.from(byGroup.values()).sort((left, right) => left.group.localeCompare(right.group));
}

function exportMatrixCsv(rows: MatrixRow[], metricNames: string[], groupBy: string): void {
  const headers = [groupBy, 'time', ...metricNames];
  const body = rows.map((row) =>
    [row.group, timeLabel(row.sample), ...metricNames.map((metricName) => row.values[metricName]?.value)]
      .map(csvValue)
      .join(',')
  );
  const blob = new Blob([[headers.join(','), ...body].join('\n')], { type: 'text/csv;charset=utf-8' });
  const url = URL.createObjectURL(blob);
  const link = document.createElement('a');
  link.href = url;
  link.download = 'compare_metric_matrix.csv';
  document.body.appendChild(link);
  link.click();
  link.remove();
  URL.revokeObjectURL(url);
}

export function CompareMetricMatrix({ rows, units, groupBy, limit = 80 }: Props) {
  const [query, setQuery] = useState('');
  const [pageIndex, setPageIndex] = useState(0);
  const [hiddenMetricNames, setHiddenMetricNames] = useState<string[]>([]);
  const metricNames = useMemo(
    () => Array.from(new Set(rows.map((row) => row.metric_name).filter(Boolean))).sort(compareMetricName),
    [rows]
  );
  const visibleMetricNames = useMemo(
    () => metricNames.filter((metricName) => !hiddenMetricNames.includes(metricName)),
    [hiddenMetricNames, metricNames]
  );
  const matrixRows = useMemo(() => pivotRows(rows, groupBy), [groupBy, rows]);
  const filteredRows = useMemo(() => {
    const normalizedQuery = query.trim().toLowerCase();
    if (!normalizedQuery) {
      return matrixRows;
    }
    return matrixRows.filter((row) =>
      [
        row.group,
        timeLabel(row.sample),
        ...metricNames.flatMap((metricName) => [
          metricName,
          metricLabel(metricName, rows, units),
          row.values[metricName]?.value
        ])
      ]
        .filter((value) => value !== undefined && value !== null)
        .join(' ')
        .toLowerCase()
        .includes(normalizedQuery)
    );
  }, [matrixRows, metricNames, query, rows, units]);
  const pageSize = Math.max(1, limit);
  const pageCount = Math.max(1, Math.ceil(filteredRows.length / pageSize));
  const safePageIndex = Math.min(pageIndex, pageCount - 1);
  const visibleRows = filteredRows.slice(safePageIndex * pageSize, (safePageIndex + 1) * pageSize);

  useEffect(() => {
    setPageIndex(0);
  }, [query, rows, groupBy]);

  useEffect(() => {
    setPageIndex((current) => Math.min(current, pageCount - 1));
  }, [pageCount]);

  useEffect(() => {
    setHiddenMetricNames((current) => current.filter((metricName) => metricNames.includes(metricName)));
  }, [metricNames]);

  const toggleMetricColumn = (metricName: string) => {
    setHiddenMetricNames((current) => {
      if (current.includes(metricName)) {
        return current.filter((name) => name !== metricName);
      }
      if (visibleMetricNames.length <= 1) {
        return current;
      }
      return [...current, metricName];
    });
  };

  return (
    <section className="panel table-panel">
      <div className="panel-header">
        <h2>指标矩阵 / Metric matrix</h2>
        <span>
          {filteredRows.length}/{matrixRows.length} 对象 / groups
        </span>
      </div>
      <div className="table-toolbar">
        <label>
          搜索 / Search metric matrix
          <input value={query} onChange={(event) => setQuery(event.target.value)} />
        </label>
        <div className="column-picker" aria-label="矩阵列显示 / Matrix column visibility">
          <span>指标列显示 / Metric columns</span>
          <div className="checkbox-grid">
            {metricNames.map((metricName) => (
              <label key={metricName}>
                <input
                  type="checkbox"
                  aria-label={`显示 ${metricName} 列 / Show ${metricName} column`}
                  checked={visibleMetricNames.includes(metricName)}
                  onChange={() => toggleMetricColumn(metricName)}
                />
                列: {metricLabel(metricName, rows, units)}
              </label>
            ))}
          </div>
        </div>
        <button onClick={() => exportMatrixCsv(filteredRows, visibleMetricNames, groupBy)}>导出 CSV / Export CSV</button>
      </div>
      <div className="table-scroll">
        <table>
          <thead>
            <tr>
              <th>对比对象 / Compare group</th>
              <th>日期时刻 / Date time</th>
              {visibleMetricNames.map((metricName) => (
                <th key={metricName}>{metricLabel(metricName, rows, units)}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {visibleRows.map((row) => (
              <tr key={row.group}>
                <td>{row.group}</td>
                <td>{timeLabel(row.sample)}</td>
                {visibleMetricNames.map((metricName) => (
                  <td key={metricName}>{compactNumber(row.values[metricName]?.value)}</td>
                ))}
              </tr>
            ))}
            {visibleRows.length === 0 ? (
              <tr>
                <td colSpan={2 + visibleMetricNames.length} className="empty-cell">
                  无矩阵数据 / No matrix rows
                </td>
              </tr>
            ) : null}
          </tbody>
        </table>
      </div>
      <div className="pagination-controls" aria-label="分页 / Pagination">
        <button
          disabled={safePageIndex === 0}
          onClick={() => setPageIndex((current) => Math.max(0, current - 1))}
        >
          上一页 / Previous page
        </button>
        <span>{`第 ${safePageIndex + 1}/${pageCount} 页 / Page ${safePageIndex + 1}/${pageCount}`}</span>
        <button
          disabled={safePageIndex >= pageCount - 1}
          onClick={() => setPageIndex((current) => Math.min(pageCount - 1, current + 1))}
        >
          下一页 / Next page
        </button>
      </div>
    </section>
  );
}
