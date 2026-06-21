import type { DateStatus } from '../../api/types';
import { dateOptionLabel, dateStatusByDate } from '../../utils/dateStatus';

type Props = {
  dates: string[];
  dateStatuses?: DateStatus[];
  value?: string;
  onChange: (value: string | undefined) => void;
};

export function CalendarPicker({ dates, dateStatuses = [], value, onChange }: Props) {
  const statusByDate = dateStatusByDate(dateStatuses);
  return (
    <label title="date 是能源数据对应日期；如果训练日志没有真实日期，dashboard 会显示 profile_day_001 这类相对日期。 / date is the energy-data date; profile_day_001 is used when real timestamps are absent.">
      日期 / Date
      <select value={value ?? ''} onChange={(event) => onChange(event.target.value || undefined)}>
        <option value="">全部日期 / All dates</option>
        {dates.map((date) => (
          <option key={date} value={date}>
            {dateOptionLabel(date, statusByDate.get(date))}
          </option>
        ))}
      </select>
    </label>
  );
}
