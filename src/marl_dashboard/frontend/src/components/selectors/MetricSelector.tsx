type Props = {
  label: string;
  allLabel: string;
  metrics: string[];
  metricLabels?: Record<string, string>;
  extraOptions?: Array<{ value: string; label: string }>;
  value: string;
  onChange: (value: string) => void;
};

export function MetricSelector({ label, allLabel, metrics, metricLabels = {}, extraOptions = [], value, onChange }: Props) {
  return (
    <label>
      {label}
      <select
        value={value}
        onChange={(event) => onChange(event.target.value)}
        disabled={metrics.length === 0 && extraOptions.length === 0}
      >
        <option value="">{allLabel}</option>
        {extraOptions.map((option) => (
          <option key={option.value} value={option.value}>
            {option.label}
          </option>
        ))}
        {metrics.map((metric) => (
          <option key={metric} value={metric}>
            {metricLabels[metric] ?? metric}
          </option>
        ))}
      </select>
    </label>
  );
}
