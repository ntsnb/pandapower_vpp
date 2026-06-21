import { useEffect, useMemo, useState } from 'react';

import type { MetricRow } from '../../api/types';
import { compactNumber } from '../../utils/filters';

type Props = {
  rows: MetricRow[];
  units: Record<string, string>;
  limit?: number;
};

type WideRow = {
  key: string;
  date?: string | null;
  time_index?: number | null;
  timestamp?: string | null;
  vpp_id?: string | null;
  epoch_id?: number | null;
  episode_id?: number | null;
  values: Record<string, MetricRow>;
};

type FixedColumnId = 'date_time' | 'timestamp' | 'vpp' | 'epoch' | 'episode';

type FixedColumn = {
  id: FixedColumnId;
  label: string;
  toggleLabel: string;
  render: (row: WideRow) => string | number | null | undefined;
};

const fixedColumns: FixedColumn[] = [
  {
    id: 'date_time',
    label: '日期时刻 / Date time',
    toggleLabel: '显示日期时刻列 / Show date time column',
    render: (row) => timeLabel(row)
  },
  {
    id: 'timestamp',
    label: '时间戳 / Timestamp',
    toggleLabel: '显示时间戳列 / Show timestamp column',
    render: (row) => row.timestamp?.slice(0, 19).replace('T', ' ') ?? '-'
  },
  {
    id: 'vpp',
    label: 'VPP',
    toggleLabel: '显示 VPP 列 / Show VPP column',
    render: (row) => row.vpp_id ?? '-'
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
  }
];

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
  'delivered_p_mw',
  'baseline_p_mw',
  'requested_delta_p_mw',
  'accepted_delta_p_mw',
  'actual_delta_p_mw',
  'actual_target_p_mw'
];

function csvValue(value: unknown): string {
  if (value === undefined || value === null) {
    return '';
  }
  const text = String(value);
  return /[",\n]/.test(text) ? `"${text.replaceAll('"', '""')}"` : text;
}

function timeLabel(row: WideRow): string {
  if (row.date && row.time_index !== undefined && row.time_index !== null) {
    return `${row.date} #${row.time_index}`;
  }
  if (row.timestamp) {
    return row.timestamp.slice(0, 19).replace('T', ' ');
  }
  return row.time_index !== undefined && row.time_index !== null ? `#${row.time_index}` : '-';
}

function metricLabel(metricName: string, rows: MetricRow[], units: Record<string, string>): string {
  const sample = rows.find((row) => row.metric_name === metricName);
  const label = sample?.display_name || sample?.description || metricName;
  const unit = units[metricName] || sample?.unit;
  return unit ? `${label} (${unit})` : label;
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

function rowKey(row: MetricRow): string {
  return [
    row.date ?? '',
    row.time_index ?? '',
    row.timestamp ?? '',
    row.vpp_id ?? '',
    row.epoch_id ?? '',
    row.episode_id ?? ''
  ].join('|');
}

function pivotRows(rows: MetricRow[]): WideRow[] {
  const byTime = new Map<string, WideRow>();
  for (const row of rows) {
    const key = rowKey(row);
    const existing = byTime.get(key) ?? {
      key,
      date: row.date,
      time_index: row.time_index,
      timestamp: row.timestamp,
      vpp_id: row.vpp_id,
      epoch_id: row.epoch_id,
      episode_id: row.episode_id,
      values: {}
    };
    existing.values[row.metric_name] = row;
    byTime.set(key, existing);
  }
  return Array.from(byTime.values()).sort((left, right) => {
    const dateCompare = String(left.date ?? '').localeCompare(String(right.date ?? ''));
    if (dateCompare !== 0) return dateCompare;
    const leftTime = left.time_index ?? Number.NEGATIVE_INFINITY;
    const rightTime = right.time_index ?? Number.NEGATIVE_INFINITY;
    if (leftTime !== rightTime) return leftTime - rightTime;
    return String(left.vpp_id ?? '').localeCompare(String(right.vpp_id ?? ''));
  });
}

function exportWideCsv(rows: WideRow[], metricNames: string[], title: string): void {
  const headers = ['time', 'timestamp', 'vpp_id', 'epoch_id', 'episode_id', ...metricNames];
  const body = rows.map((row) =>
    [
      timeLabel(row),
      row.timestamp,
      row.vpp_id,
      row.epoch_id,
      row.episode_id,
      ...metricNames.map((metricName) => row.values[metricName]?.value)
    ]
      .map(csvValue)
      .join(',')
  );
  const blob = new Blob([[headers.join(','), ...body].join('\n')], { type: 'text/csv;charset=utf-8' });
  const url = URL.createObjectURL(blob);
  const link = document.createElement('a');
  link.href = url;
  link.download = `${title.toLowerCase().replace(/[^a-z0-9]+/g, '_').replace(/^_|_$/g, '') || 'dataset_timeseries'}.csv`;
  document.body.appendChild(link);
  link.click();
  link.remove();
  URL.revokeObjectURL(url);
}

export function DatasetTimeseriesTable({ rows, units, limit = 120 }: Props) {
  const [query, setQuery] = useState('');
  const [pageIndex, setPageIndex] = useState(0);
  const [visibleColumnIds, setVisibleColumnIds] = useState<string[]>([]);
  const metricNames = useMemo(
    () => Array.from(new Set(rows.map((row) => row.metric_name).filter(Boolean))).sort(compareMetricName),
    [rows]
  );
  const wideRows = useMemo(() => pivotRows(rows), [rows]);
  const filteredRows = useMemo(() => {
    const normalizedQuery = query.trim().toLowerCase();
    if (!normalizedQuery) {
      return wideRows;
    }
    return wideRows.filter((row) =>
      [
        timeLabel(row),
        row.timestamp,
        row.vpp_id,
        row.epoch_id,
        row.episode_id,
        ...metricNames.flatMap((metricName) => [
          metricName,
          row.values[metricName]?.display_name,
          row.values[metricName]?.description,
          row.values[metricName]?.value
        ])
      ]
        .filter((value) => value !== undefined && value !== null)
        .join(' ')
        .toLowerCase()
        .includes(normalizedQuery)
    );
  }, [metricNames, query, wideRows]);
  const allColumnIds = useMemo(
    () => [...fixedColumns.map((column) => column.id), ...metricNames.map((metricName) => `metric:${metricName}`)],
    [metricNames]
  );
  useEffect(() => {
    setVisibleColumnIds((current) => {
      if (current.length === 0) {
        return allColumnIds;
      }
      return allColumnIds.filter((id) => current.includes(id) || !current.some((currentId) => currentId === id));
    });
  }, [allColumnIds]);
  useEffect(() => {
    setPageIndex(0);
  }, [query, rows]);
  const pageSize = Math.max(1, limit);
  const pageCount = Math.max(1, Math.ceil(filteredRows.length / pageSize));
  useEffect(() => {
    setPageIndex((current) => Math.min(current, pageCount - 1));
  }, [pageCount]);
  const visibleRows = filteredRows.slice(pageIndex * pageSize, (pageIndex + 1) * pageSize);
  const visibleFixedColumns = fixedColumns.filter((column) => visibleColumnIds.includes(column.id));
  const visibleMetricNames = metricNames.filter((metricName) => visibleColumnIds.includes(`metric:${metricName}`));
  const toggleColumn = (columnId: string) => {
    setVisibleColumnIds((current) => {
      if (current.includes(columnId)) {
        const next = current.filter((id) => id !== columnId);
        return next.length > 0 ? next : current;
      }
      return allColumnIds.filter((id) => id === columnId || current.includes(id));
    });
  };

  return (
    <section className="panel table-panel">
      <div className="panel-header">
        <h2>逐时刻宽表 / Timeseries wide table</h2>
        <span>
          {filteredRows.length}/{wideRows.length} 时刻行 / time rows
        </span>
      </div>
      <div className="table-toolbar">
        <label>
          搜索 / Search timeseries
          <input value={query} onChange={(event) => setQuery(event.target.value)} />
        </label>
        <div className="column-picker" aria-label="列显示 / Column visibility">
          <span>列显示 / Columns</span>
          <div className="checkbox-grid">
            {fixedColumns.map((column) => (
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
            {metricNames.map((metricName) => (
              <label key={metricName}>
                <input
                  type="checkbox"
                  aria-label={`显示 ${metricName} 列 / Show ${metricName} column`}
                  checked={visibleColumnIds.includes(`metric:${metricName}`)}
                  onChange={() => toggleColumn(`metric:${metricName}`)}
                />
                列: {metricLabel(metricName, rows, units)}
              </label>
            ))}
          </div>
        </div>
        <button onClick={() => exportWideCsv(filteredRows, metricNames, 'dataset_timeseries')}>导出 CSV / Export CSV</button>
      </div>
      <div className="table-scroll">
        <table>
          <thead>
            <tr>
              {visibleFixedColumns.map((column) => (
                <th key={column.id}>{column.label}</th>
              ))}
              {visibleMetricNames.map((metricName) => (
                <th key={metricName}>{metricLabel(metricName, rows, units)}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {visibleRows.map((row) => (
              <tr key={row.key}>
                {visibleFixedColumns.map((column) => (
                  <td key={column.id}>{column.render(row)}</td>
                ))}
                {visibleMetricNames.map((metricName) => (
                  <td key={metricName}>{compactNumber(row.values[metricName]?.value)}</td>
                ))}
              </tr>
            ))}
            {visibleRows.length === 0 ? (
              <tr>
                <td colSpan={Math.max(1, visibleFixedColumns.length + visibleMetricNames.length)} className="empty-cell">
                  无逐时刻数据 / No timeseries rows
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
