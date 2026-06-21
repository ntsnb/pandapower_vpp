import { useEffect, useMemo, useState } from 'react';

import type { MetricRow } from '../../api/types';
import { compactNumber } from '../../utils/filters';

type Props = {
  title: string;
  rows: MetricRow[];
  limit?: number;
};

type SortKey = 'metric_name' | 'value' | 'time_index' | 'vpp_id' | 'policy_id';
type ColumnId = 'metric' | 'description' | 'value' | 'unit' | 'epoch' | 'episode' | 'time' | 'vpp' | 'agent' | 'policy';

type ColumnDefinition = {
  id: ColumnId;
  label: string;
  toggleLabel: string;
  sortKey?: SortKey;
  render: (row: MetricRow) => string | number | boolean | null | undefined;
};

const columns: ColumnDefinition[] = [
  {
    id: 'metric',
    label: '指标 / Metric',
    toggleLabel: '显示指标列 / Show metric column',
    sortKey: 'metric_name',
    render: (row) => row.metric_name
  },
  {
    id: 'description',
    label: '说明 / Description',
    toggleLabel: '显示说明列 / Show description column',
    render: (row) => row.display_name ?? row.description ?? '-'
  },
  {
    id: 'value',
    label: '数值 / Value',
    toggleLabel: '显示数值列 / Show value column',
    sortKey: 'value',
    render: (row) => compactNumber(row.value)
  },
  {
    id: 'unit',
    label: '单位 / Unit',
    toggleLabel: '显示单位列 / Show unit column',
    render: (row) => row.unit ?? '-'
  },
  {
    id: 'epoch',
    label: '训练轮次 / Epoch',
    toggleLabel: '显示训练轮次列 / Show epoch column',
    render: (row) => row.epoch_id ?? '-'
  },
  {
    id: 'episode',
    label: 'Episode',
    toggleLabel: '显示 Episode 列 / Show episode column',
    render: (row) => row.episode_id ?? '-'
  },
  {
    id: 'time',
    label: '时刻 / Time',
    toggleLabel: '显示时刻列 / Show time column',
    sortKey: 'time_index',
    render: (row) => row.timestamp?.slice(0, 19).replace('T', ' ') ?? row.time_index ?? '-'
  },
  {
    id: 'vpp',
    label: 'VPP',
    toggleLabel: '显示 VPP 列 / Show VPP column',
    sortKey: 'vpp_id',
    render: (row) => row.vpp_id ?? '-'
  },
  {
    id: 'agent',
    label: '智能体 / Agent',
    toggleLabel: '显示智能体列 / Show agent column',
    render: (row) => row.agent_id ?? '-'
  },
  {
    id: 'policy',
    label: '策略 / Policy',
    toggleLabel: '显示策略列 / Show policy column',
    sortKey: 'policy_id',
    render: (row) => row.policy_id ?? '-'
  }
];

const defaultColumnIds = columns.map((column) => column.id);

function rowText(row: MetricRow): string {
  return [
    row.metric_name,
    row.display_name,
    row.description,
    row.value,
    row.unit,
    row.epoch_id,
    row.episode_id,
    row.time_index,
    row.timestamp,
    row.vpp_id,
    row.agent_id,
    row.policy_id
  ]
    .filter((value) => value !== undefined && value !== null)
    .join(' ')
    .toLowerCase();
}

function sortableValue(row: MetricRow, key: SortKey): string | number {
  const value = row[key];
  if (key === 'value' || key === 'time_index') {
    const numeric = Number(value);
    return Number.isFinite(numeric) ? numeric : Number.NEGATIVE_INFINITY;
  }
  return String(value ?? '');
}

function csvValue(value: unknown): string {
  if (value === undefined || value === null) {
    return '';
  }
  const text = String(value);
  return /[",\n]/.test(text) ? `"${text.replaceAll('"', '""')}"` : text;
}

function exportCsv(title: string, rows: MetricRow[]): void {
  const headers = ['metric', 'description', 'value', 'unit', 'epoch', 'episode', 'time', 'vpp', 'agent', 'policy'];
  const body = rows.map((row) =>
    [
      row.metric_name,
      row.display_name ?? row.description,
      row.value,
      row.unit,
      row.epoch_id,
      row.episode_id,
      row.timestamp?.slice(0, 19).replace('T', ' ') ?? row.time_index,
      row.vpp_id,
      row.agent_id,
      row.policy_id
    ]
      .map(csvValue)
      .join(',')
  );
  const blob = new Blob([[headers.join(','), ...body].join('\n')], { type: 'text/csv;charset=utf-8' });
  const url = URL.createObjectURL(blob);
  const link = document.createElement('a');
  link.href = url;
  link.download = `${title.toLowerCase().replace(/[^a-z0-9]+/g, '_').replace(/^_|_$/g, '') || 'metrics'}.csv`;
  document.body.appendChild(link);
  link.click();
  link.remove();
  URL.revokeObjectURL(url);
}

export function MetricTable({ title, rows, limit = 80 }: Props) {
  const [query, setQuery] = useState('');
  const [sortKey, setSortKey] = useState<SortKey>('time_index');
  const [sortDirection, setSortDirection] = useState<'asc' | 'desc'>('asc');
  const [pageIndex, setPageIndex] = useState(0);
  const [visibleColumnIds, setVisibleColumnIds] = useState<ColumnId[]>(defaultColumnIds);
  const filteredRows = useMemo(() => {
    const normalizedQuery = query.trim().toLowerCase();
    const filtered = normalizedQuery ? rows.filter((row) => rowText(row).includes(normalizedQuery)) : [...rows];
    filtered.sort((left, right) => {
      const leftValue = sortableValue(left, sortKey);
      const rightValue = sortableValue(right, sortKey);
      if (typeof leftValue === 'number' && typeof rightValue === 'number') {
        return sortDirection === 'asc' ? leftValue - rightValue : rightValue - leftValue;
      }
      return sortDirection === 'asc'
        ? String(leftValue).localeCompare(String(rightValue))
        : String(rightValue).localeCompare(String(leftValue));
    });
    return filtered;
  }, [query, rows, sortDirection, sortKey]);
  const pageSize = Math.max(1, limit);
  const pageCount = Math.max(1, Math.ceil(filteredRows.length / pageSize));
  const visibleRows = filteredRows.slice(pageIndex * pageSize, (pageIndex + 1) * pageSize);
  const visibleColumns = columns.filter((column) => visibleColumnIds.includes(column.id));
  useEffect(() => {
    setPageIndex(0);
  }, [query, rows, sortDirection, sortKey]);
  useEffect(() => {
    setPageIndex((current) => Math.min(current, pageCount - 1));
  }, [pageCount]);
  const toggleSort = (key: SortKey) => {
    if (key === sortKey) {
      setSortDirection((current) => (current === 'asc' ? 'desc' : 'asc'));
      return;
    }
    setSortKey(key);
    setSortDirection('asc');
  };
  const toggleColumn = (columnId: ColumnId) => {
    setVisibleColumnIds((current) => {
      if (current.includes(columnId)) {
        const next = current.filter((id) => id !== columnId);
        return next.length > 0 ? next : current;
      }
      return defaultColumnIds.filter((id) => id === columnId || current.includes(id));
    });
  };
  return (
    <section className="panel table-panel">
      <div className="panel-header">
        <h2>{title}</h2>
        <span>
          {filteredRows.length}/{rows.length} 行 / rows
        </span>
      </div>
      <div className="table-toolbar">
        <label>
          搜索 / Search {title}
          <input value={query} onChange={(event) => setQuery(event.target.value)} />
        </label>
        <div className="column-picker" aria-label="列显示 / Column visibility">
          <span>列显示 / Columns</span>
          <div className="checkbox-grid">
            {columns.map((column) => (
              <label key={column.id}>
                <input
                  type="checkbox"
                  aria-label={column.toggleLabel}
                  checked={visibleColumnIds.includes(column.id)}
                  onChange={() => toggleColumn(column.id)}
                />
                列: {column.label}
              </label>
            ))}
          </div>
        </div>
        <button onClick={() => exportCsv(title, filteredRows)}>导出 CSV / Export CSV</button>
      </div>
      <div className="table-scroll">
        <table>
          <thead>
            <tr>
              {visibleColumns.map((column) => (
                <th key={column.id}>
                  {column.sortKey ? (
                    <button className="table-sort-button" onClick={() => toggleSort(column.sortKey as SortKey)}>
                      {column.label}
                    </button>
                  ) : (
                    column.label
                  )}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {visibleRows.map((row, index) => (
              <tr key={`${row.metric_group}-${row.metric_name}-${index}`}>
                {visibleColumns.map((column) => (
                  <td key={column.id}>{column.render(row)}</td>
                ))}
              </tr>
            ))}
            {visibleRows.length === 0 ? (
              <tr>
                <td colSpan={Math.max(1, visibleColumns.length)} className="empty-cell">
                  无数据行 / No rows
                </td>
              </tr>
            ) : null}
          </tbody>
        </table>
      </div>
      <div className="pagination-controls" aria-label="分页 / Pagination">
        <button disabled={pageIndex === 0} onClick={() => setPageIndex((current) => Math.max(0, current - 1))}>
          上一页 / Previous page
        </button>
        <span>{`第 ${pageIndex + 1}/${pageCount} 页 / Page ${pageIndex + 1}/${pageCount}`}</span>
        <button
          disabled={pageIndex >= pageCount - 1}
          onClick={() => setPageIndex((current) => Math.min(pageCount - 1, current + 1))}
        >
          下一页 / Next page
        </button>
      </div>
    </section>
  );
}
