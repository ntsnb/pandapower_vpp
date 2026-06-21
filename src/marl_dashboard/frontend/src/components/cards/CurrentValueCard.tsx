import type { QueryResponse } from '../../api/types';
import { compactNumber } from '../../utils/filters';

type Props = {
  title: string;
  response: QueryResponse | null;
  metricName?: string;
};

export function CurrentValueCard({ title, response, metricName }: Props) {
  const rows = response?.table_rows ?? [];
  const row = rows
    .filter((item) => (metricName ? item.metric_name === metricName : true))
    .at(-1);

  return (
    <section className="stat-panel">
      <span>{title}</span>
      <strong>{compactNumber(row?.value)}</strong>
      <small>{row?.unit ?? row?.metric_name ?? '-'}</small>
    </section>
  );
}
