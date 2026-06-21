type Props = {
  epochIds: number[];
  value?: number;
  onChange: (value: number | undefined) => void;
};

export function EpochWheelPicker({ epochIds, value, onChange }: Props) {
  const hasSingleEpoch = epochIds.length === 1;
  return (
    <label title="epoch_id 表示训练外层迭代或 learner update round，不等于 episode。 / epoch_id is a training iteration or learner update round, not necessarily an episode.">
      训练轮次 / Epoch
      <select
        disabled={hasSingleEpoch}
        value={value ?? ''}
        onChange={(event) => onChange(event.target.value === '' ? undefined : Number(event.target.value))}
      >
        <option value="">最新 / Latest</option>
        {epochIds.map((epochId) => (
          <option key={epochId} value={epochId}>
            Epoch {String(epochId).padStart(6, '0')}
          </option>
        ))}
      </select>
      {hasSingleEpoch ? <small>当前 run 只有一个训练轮次维度 / Single epoch dimension</small> : null}
    </label>
  );
}
