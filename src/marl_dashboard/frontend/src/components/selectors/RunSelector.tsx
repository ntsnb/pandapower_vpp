import type { RunSummary } from '../../api/types';

type Props = {
  runs: RunSummary[];
  value: string;
  onChange: (value: string) => void;
};

export function RunSelector({ runs, value, onChange }: Props) {
  return (
    <label title="run_id 是一次训练运行的唯一编号。 / run_id uniquely identifies one training run.">
      运行 / Run
      <select value={value} onChange={(event) => onChange(event.target.value)}>
        {runs.map((run) => (
          <option key={run.run_id} value={run.run_id}>
            {run.run_id}
          </option>
        ))}
      </select>
    </label>
  );
}
