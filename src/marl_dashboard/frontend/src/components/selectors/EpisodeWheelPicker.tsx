type Props = {
  episodeIds: number[];
  value?: number;
  onChange: (value: number | undefined) => void;
};

export function EpisodeWheelPicker({ episodeIds, value, onChange }: Props) {
  return (
    <label title="episode_id 表示环境从 reset 到结束的一条轨迹；选择单个 episode 可避免多个轨迹在同一日期/time_index 上混合。 / episode_id is one trajectory from reset to termination; selecting one episode prevents multiple trajectories from mixing on the same date/time_index.">
      轨迹周期 / Episode
      <select value={value ?? ''} onChange={(event) => onChange(event.target.value === '' ? undefined : Number(event.target.value))}>
        <option value="">全部轨迹 / All episodes</option>
        {episodeIds.map((episodeId) => (
          <option key={episodeId} value={episodeId}>
            Episode {String(episodeId).padStart(6, '0')}
          </option>
        ))}
      </select>
    </label>
  );
}
