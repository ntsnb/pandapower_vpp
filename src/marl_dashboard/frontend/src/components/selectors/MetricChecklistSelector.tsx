type Props = {
  label: string;
  metrics: string[];
  metricLabels?: Record<string, string>;
  selectedMetrics: string[];
  onChange: (metrics: string[]) => void;
};

export function MetricChecklistSelector({
  label,
  metrics,
  metricLabels = {},
  selectedMetrics,
  onChange
}: Props) {
  const selectedSet = new Set(selectedMetrics);
  const toggleMetric = (metricName: string) => {
    if (selectedSet.has(metricName)) {
      onChange(selectedMetrics.filter((name) => name !== metricName));
      return;
    }
    const nextSet = new Set([...selectedMetrics, metricName]);
    onChange(metrics.filter((name) => nextSet.has(name)));
  };

  return (
    <fieldset className="metric-checklist" disabled={metrics.length === 0}>
      <legend>{label}</legend>
      <div className="metric-checklist-actions">
        <span>
          {selectedMetrics.length}/{metrics.length} 已显示 / shown
        </span>
        <div>
          <button type="button" onClick={() => onChange(metrics)} disabled={selectedMetrics.length === metrics.length}>
            全选 / Select all
          </button>
          <button type="button" onClick={() => onChange([])} disabled={selectedMetrics.length === 0}>
            全不选 / Clear
          </button>
        </div>
      </div>
      <div className="metric-checklist-grid">
        {metrics.map((metricName) => {
          const metricLabel = metricLabels[metricName] ?? metricName;
          return (
            <label key={metricName} className={selectedSet.has(metricName) ? 'is-checked' : ''}>
              <input
                type="checkbox"
                aria-label={metricLabel}
                checked={selectedSet.has(metricName)}
                onChange={() => toggleMetric(metricName)}
              />
              <span>{metricLabel}</span>
            </label>
          );
        })}
        {metrics.length === 0 ? <span className="empty-state">无可选指标 / No metrics</span> : null}
      </div>
    </fieldset>
  );
}
