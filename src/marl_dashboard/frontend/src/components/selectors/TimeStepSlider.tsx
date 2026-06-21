type Props = {
  timeIndices: number[];
  value?: number;
  onChange: (value: number | undefined) => void;
  label?: string;
  allLabel?: string;
  ariaLabel?: string;
  title?: string;
};

export function TimeStepSlider({
  timeIndices,
  value,
  onChange,
  label = '时刻 / Time index',
  allLabel = '全部时刻 / All time indices',
  ariaLabel = label,
  title = 'time_index 是一天内第几个能源数据点；15 分钟数据通常为 0-95。 / time_index is the within-day energy-data slot; 15-minute data usually uses 0-95.'
}: Props) {
  const selectedIndex = Math.max(0, value === undefined ? 0 : timeIndices.indexOf(value));
  const displayLabel = value === undefined ? `${label}: ${allLabel}` : `${label}: #${value}`;
  return (
    <label title={title}>
      {displayLabel}
      <input
        aria-label={ariaLabel}
        type="range"
        min={0}
        max={Math.max(0, timeIndices.length - 1)}
        step={1}
        value={selectedIndex}
        onChange={(event) => onChange(timeIndices[Number(event.target.value)] ?? undefined)}
        disabled={timeIndices.length === 0}
      />
    </label>
  );
}
