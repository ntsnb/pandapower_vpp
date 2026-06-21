import type { QueryResponse } from '../../api/types';
import { MetricLineChart } from './MetricLineChart';

type Props = {
  panels: Array<{ title: string; response: QueryResponse | null }>;
};

export function MultiPanelChart({ panels }: Props) {
  return (
    <div className="panel-grid">
      {panels.map((panel) => (
        <MetricLineChart key={panel.title} title={panel.title} response={panel.response} height={280} />
      ))}
    </div>
  );
}
