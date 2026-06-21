import type { DateStatus } from '../api/types';

export function dateStatusByDate(dateStatuses: DateStatus[] | undefined): Map<string, DateStatus> {
  return new Map((dateStatuses ?? []).map((status) => [status.date, status]));
}

export function dateOptionLabel(date: string, status?: DateStatus): string {
  if (!status || status.complete) {
    return date;
  }
  return `${date} (未满 ${status.observed_time_slots}/${status.expected_time_slots} / Partial)`;
}
