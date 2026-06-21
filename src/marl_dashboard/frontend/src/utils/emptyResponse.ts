import type { QueryResponse } from '../api/types';

export const emptyResponse: QueryResponse = {
  chart_series: [],
  table_rows: [],
  units: {},
  formulas: {},
  summary: { row_count: 0 }
};
