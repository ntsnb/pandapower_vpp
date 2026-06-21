import type { Filters, RunSummary, Selectors } from '../../api/types';
import { CalendarPicker } from '../selectors/CalendarPicker';
import { AgentSelector } from '../selectors/AgentSelector';
import { EpochWheelPicker } from '../selectors/EpochWheelPicker';
import { EpisodeWheelPicker } from '../selectors/EpisodeWheelPicker';
import { PolicySelector } from '../selectors/PolicySelector';
import { RunSelector } from '../selectors/RunSelector';
import { TimeStepSlider } from '../selectors/TimeStepSlider';
import { VppWheelPicker } from '../selectors/VppWheelPicker';

type Props = {
  runs: RunSummary[];
  selectors: Selectors | null;
  filters: Filters;
  onChange: (filters: Filters) => void;
};

export function FilterBar({ runs, selectors, filters, onChange }: Props) {
  const selectedDateStatus = selectors?.date_statuses?.find((status) => status.date === filters.date);
  return (
    <>
      <section className="filterbar">
        <RunSelector runs={runs} value={filters.runId} onChange={(runId) => onChange({ ...filters, runId })} />
        <CalendarPicker
          dates={selectors?.dates ?? []}
          dateStatuses={selectors?.date_statuses ?? []}
          value={filters.date}
          onChange={(date) => onChange({ ...filters, date })}
        />
        <VppWheelPicker vppIds={selectors?.vpp_ids ?? []} value={filters.vppId} onChange={(vppId) => onChange({ ...filters, vppId })} />
        <AgentSelector agentIds={selectors?.agent_ids ?? []} value={filters.agentId} onChange={(agentId) => onChange({ ...filters, agentId })} />
        <PolicySelector policyIds={selectors?.policy_ids ?? []} value={filters.policyId} onChange={(policyId) => onChange({ ...filters, policyId })} />
        <EpochWheelPicker epochIds={selectors?.epoch_ids ?? []} value={filters.epochId} onChange={(epochId) => onChange({ ...filters, epochId })} />
        <EpisodeWheelPicker
          episodeIds={selectors?.episode_ids ?? []}
          value={filters.episodeId}
          onChange={(episodeId) => onChange({ ...filters, episodeId })}
        />
        <TimeStepSlider
          timeIndices={selectors?.time_indices ?? []}
          value={filters.timeIndex}
          onChange={(timeIndex) => onChange({ ...filters, timeIndex, startTimeIndex: undefined, endTimeIndex: undefined })}
        />
        <label title="start_time_index 和 end_time_index 用于查看一天内一段连续时刻；设置范围后会覆盖单点 time_index。 / start_time_index and end_time_index select a continuous within-day window and override the single time_index point.">
          起始时刻 / Start time
          <select
            value={filters.startTimeIndex ?? ''}
            onChange={(event) =>
              onChange({
                ...filters,
                timeIndex: undefined,
                startTimeIndex: event.target.value === '' ? undefined : Number(event.target.value)
              })
            }
            disabled={(selectors?.time_indices ?? []).length === 0}
          >
            <option value="">自动 / Auto</option>
            {(selectors?.time_indices ?? []).map((timeIndex) => (
              <option key={timeIndex} value={timeIndex}>
                #{timeIndex}
              </option>
            ))}
          </select>
        </label>
        <label title="结束时刻会与起始时刻自动组成闭区间；如果结束小于起始，查询层会自动交换顺序。 / End time forms a closed range with start time; reversed ranges are normalized before querying.">
          结束时刻 / End time
          <select
            value={filters.endTimeIndex ?? ''}
            onChange={(event) =>
              onChange({
                ...filters,
                timeIndex: undefined,
                endTimeIndex: event.target.value === '' ? undefined : Number(event.target.value)
              })
            }
            disabled={(selectors?.time_indices ?? []).length === 0}
          >
            <option value="">自动 / Auto</option>
            {(selectors?.time_indices ?? []).map((timeIndex) => (
              <option key={timeIndex} value={timeIndex}>
                #{timeIndex}
              </option>
            ))}
          </select>
        </label>
        <button
          title="Live 会自动跟随最新数据；Frozen 会固定当前筛选条件用于复盘。 / Live follows incoming data; Frozen keeps the selected filters fixed."
          onClick={() => onChange({ ...filters, live: !filters.live })}
        >
          {filters.live ? '实时刷新 / Auto refresh: On' : '冻结复盘 / Auto refresh: Off (Frozen)'}
        </button>
        <button
          title="Compare Mode 用于固定同一时刻并横向比较多个 VPP、epoch、policy 或 agent。 / Compare Mode compares VPPs, epochs, policies, or agents at a fixed time."
          onClick={() => onChange({ ...filters, compareMode: !filters.compareMode })}
        >
          {filters.compareMode ? '对比模式 / Compare Mode On' : '对比模式 / Compare Mode Off'}
        </button>
        <button onClick={() => onChange({ runId: filters.runId, live: true, compareMode: false })}>重置筛选 / Reset Filters</button>
      </section>
      {selectedDateStatus && !selectedDateStatus.complete ? (
        <div className="notice filterbar-notice">
          当前日期未满 {selectedDateStatus.expected_time_slots} 个时隙，只记录了 {selectedDateStatus.observed_time_slots} 个；选择该日期时曲线可能显示为短线 /
          This date has {selectedDateStatus.observed_time_slots} of {selectedDateStatus.expected_time_slots} expected time slots, so charts may appear as short lines.
        </div>
      ) : null}
    </>
  );
}
