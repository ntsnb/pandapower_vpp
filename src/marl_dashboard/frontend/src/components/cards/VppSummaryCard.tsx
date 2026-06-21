import type { QueryResponse } from '../../api/types';
import { compactNumber } from '../../utils/filters';

type Props = {
  title: string;
  response: QueryResponse | null;
};

export function VppSummaryCard({ title, response }: Props) {
  const rows = response?.table_rows ?? [];
  const byVpp = new Map<string, number>();
  for (const row of rows) {
    if (typeof row.value === 'number') {
      byVpp.set(row.vpp_id ?? 'unknown', (byVpp.get(row.vpp_id ?? 'unknown') ?? 0) + row.value);
    }
  }

  return (
    <section className="panel summary-panel">
      <div className="panel-header">
        <h2>{title}</h2>
        <span>{byVpp.size} VPP</span>
      </div>
      <div className="summary-list">
        {Array.from(byVpp.entries()).map(([vppId, value]) => (
          <div key={vppId} className="summary-row">
            <span>{vppId}</span>
            <strong>{compactNumber(value)}</strong>
          </div>
        ))}
        {byVpp.size === 0 ? <div className="empty-state">No VPP values</div> : null}
      </div>
    </section>
  );
}
