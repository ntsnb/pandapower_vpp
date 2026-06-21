import type { Filters, RunSummary, Selectors, WebSocketStatus } from '../../api/types';

export type ThemeMode = 'light' | 'dark';

type TopBarProps = {
  run: RunSummary | null;
  selectors: Selectors | null;
  filters: Filters;
  liveStatus?: WebSocketStatus;
  liveEventCount?: number;
  theme?: ThemeMode;
  onThemeChange?: (theme: ThemeMode) => void;
};

export function TopBar({
  run,
  selectors,
  filters,
  liveStatus = 'disabled',
  liveEventCount = 0,
  theme = 'light',
  onThemeChange
}: TopBarProps) {
  const selectedDate = filters.date ?? '全部日期 / All dates';
  const selectedTimeIndex = filters.timeIndex ?? '全部时刻 / All time indices';
  const nextTheme = theme === 'light' ? 'dark' : 'light';
  const themeLabel = theme === 'light' ? '深色主题 / Dark theme' : '浅色主题 / Light theme';
  const epochTitle =
    'epoch_id 是训练外层迭代或 learner update round，不等于 episode，也不等于日期。 / epoch_id is a learner iteration, not an episode or date.';
  const episodeTitle =
    'episode_id 是环境从 reset 到结束的一条完整轨迹周期；一个 episode 内可包含多个 date/time_index。 / episode_id is one reset-to-end trajectory and can contain multiple dates/time indices.';
  return (
    <header className="topbar">
      <div>
        <div className="eyebrow">当前运行 / Current run</div>
        <h1>{run?.run_id ?? 'No run selected'}</h1>
      </div>
      <div className="status-grid">
        <span className={`badge ${run?.status === 'running' ? 'green' : ''}`}>{run?.status ?? 'unknown'}</span>
        <span title={epochTitle}>训练轮次 / Epoch {filters.epochId ?? selectors?.epoch_ids.at(-1) ?? '-'}</span>
        <span title={episodeTitle}>轨迹周期 / Episode {filters.episodeId ?? '全部 / All'}</span>
        <span>日期 / Date {selectedDate}</span>
        <span>时刻 / Time index {selectedTimeIndex}</span>
        <span>{filters.live ? '实时刷新 / Auto refresh: On' : '冻结复盘 / Auto refresh: Off'}</span>
        <span>{filters.compareMode ? '对比 / Compare' : '单视图 / Single view'}</span>
        <span>实时通道 / WebSocket: {liveStatus} ({liveEventCount})</span>
        <button className="theme-toggle" onClick={() => onThemeChange?.(nextTheme)}>
          {themeLabel}
        </button>
      </div>
    </header>
  );
}
