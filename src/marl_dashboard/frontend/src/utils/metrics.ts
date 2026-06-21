import type { MetricRow, QueryResponse } from '../api/types';

export function metricNamesFromResponse(response: QueryResponse | null | undefined): string[] {
  const names = new Set<string>();
  for (const row of response?.table_rows ?? []) {
    if (row.metric_name) names.add(row.metric_name);
  }
  for (const series of response?.chart_series ?? []) {
    if (series.metric_name) names.add(series.metric_name);
    for (const point of series.points) {
      if (point.metric_name) names.add(point.metric_name);
    }
  }
  return Array.from(names).sort((left, right) => left.localeCompare(right));
}

export function filterMetricResponse(response: QueryResponse | null, metricName: string): QueryResponse | null {
  if (!response || !metricName) {
    return response;
  }
  const chartSeries = response.chart_series
    .map((series) => {
      const points = series.points.filter((point) => point.metric_name === metricName);
      if (series.metric_name !== metricName && points.length === 0) {
        return null;
      }
      return { ...series, points };
    })
    .filter((series): series is NonNullable<typeof series> => series !== null);
  const tableRows = response.table_rows.filter((row) => row.metric_name === metricName);
  const visibleRowCount = tableRows.length || chartSeries.reduce((total, series) => total + series.points.length, 0);
  return {
    ...response,
    chart_series: chartSeries,
    table_rows: tableRows,
    units: Object.fromEntries(Object.entries(response.units).filter(([name]) => name === metricName)),
    formulas: Object.fromEntries(Object.entries(response.formulas).filter(([name]) => name === metricName)),
    summary: {
      ...response.summary,
      row_count: visibleRowCount,
      visible_row_count: visibleRowCount,
      visible_metric_name: metricName
    }
  };
}

export function filterMetricResponseByMetrics(response: QueryResponse | null, metricNames: string[]): QueryResponse | null {
  if (!response) {
    return response;
  }
  const selected = new Set(metricNames);
  const chartSeries = response.chart_series
    .map((series) => {
      const points = series.points.filter((point) => selected.has(point.metric_name));
      if (series.metric_name && selected.has(series.metric_name)) {
        return { ...series, points: points.length > 0 ? points : series.points };
      }
      if (points.length === 0) {
        return null;
      }
      return { ...series, points };
    })
    .filter((series): series is NonNullable<typeof series> => series !== null);
  const tableRows = response.table_rows.filter((row) => selected.has(row.metric_name));
  const visibleRowCount = tableRows.length || chartSeries.reduce((total, series) => total + series.points.length, 0);
  return {
    ...response,
    chart_series: chartSeries,
    table_rows: tableRows,
    units: Object.fromEntries(Object.entries(response.units).filter(([name]) => selected.has(name))),
    formulas: Object.fromEntries(Object.entries(response.formulas).filter(([name]) => selected.has(name))),
    summary: {
      ...response.summary,
      row_count: visibleRowCount,
      visible_row_count: visibleRowCount,
      visible_metric_count: selected.size
    }
  };
}

function metricLabelFromRow(row: MetricRow): string {
  return row.display_name || row.description || row.metric_name;
}

export function metricLabelsFromResponse(response: QueryResponse | null | undefined): Record<string, string> {
  const labels: Record<string, string> = {};
  for (const row of rowsFromResponse(response)) {
    if (!row.metric_name || labels[row.metric_name]) {
      continue;
    }
    labels[row.metric_name] = metricLabelFromRow(row);
  }
  return labels;
}

function rowsFromResponse(response: QueryResponse | null | undefined): MetricRow[] {
  if (!response) {
    return [];
  }
  return [
    ...response.table_rows,
    ...response.chart_series.flatMap((series) => series.points)
  ];
}

export function currentValuesFromResponses(responses: Array<QueryResponse | null | undefined>): Record<string, MetricRow> {
  const values: Record<string, MetricRow> = {};
  for (const response of responses) {
    for (const row of rowsFromResponse(response)) {
      if (!row.metric_name) {
        continue;
      }
      values[row.metric_name] = row;
    }
  }
  return values;
}
