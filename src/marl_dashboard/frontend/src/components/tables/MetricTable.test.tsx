import { fireEvent, render, screen, within } from '@testing-library/react';
import { afterEach, describe, expect, it, vi } from 'vitest';

import { MetricTable } from './MetricTable';

const rows = [
  {
    run_id: 'run_a',
    metric_group: 'dataset',
    metric_name: 'pv_power',
    description: '光伏出力 / PV power',
    value: 0.2,
    unit: 'MW',
    epoch_id: 0,
    episode_id: 1,
    time_index: 1,
    vpp_id: 'vpp_b',
    agent_id: 'vpp_b_dispatch',
    policy_id: 'happo'
  },
  {
    run_id: 'run_a',
    metric_group: 'dataset',
    metric_name: 'ev_charging_load',
    description: '充电桩负荷 / EV charging load',
    value: 0.8,
    unit: 'MW',
    epoch_id: 0,
    episode_id: 1,
    time_index: 1,
    vpp_id: 'vpp_a',
    agent_id: 'vpp_a_dispatch',
    policy_id: 'happo'
  }
];

afterEach(() => {
  vi.restoreAllMocks();
});

describe('MetricTable', () => {
  it('filters rows, sorts by value, and exports visible rows as CSV', () => {
    const click = vi.fn();
    const appendChild = vi.spyOn(document.body, 'appendChild');
    vi.stubGlobal('URL', {
      createObjectURL: vi.fn(() => 'blob:metrics'),
      revokeObjectURL: vi.fn()
    });
    vi.spyOn(document, 'createElement').mockImplementation((tagName: string) => {
      const element = document.createElementNS('http://www.w3.org/1999/xhtml', tagName) as HTMLElement;
      if (tagName === 'a') {
        Object.defineProperty(element, 'click', { value: click });
      }
      return element as never;
    });

    render(<MetricTable title="数据行 / Dataset rows" rows={rows} />);

    fireEvent.change(screen.getByLabelText('搜索 / Search 数据行 / Dataset rows'), { target: { value: 'vpp_a' } });

    expect(screen.getByText('ev_charging_load')).toBeInTheDocument();
    expect(screen.getByText('充电桩负荷 / EV charging load')).toBeInTheDocument();
    expect(screen.queryByText('pv_power')).not.toBeInTheDocument();

    fireEvent.click(screen.getByRole('button', { name: '数值 / Value' }));
    const tableRows = screen.getAllByRole('row');
    expect(within(tableRows[1]).getByText('ev_charging_load')).toBeInTheDocument();

    const table = screen.getByRole('table');
    expect(within(table).getByRole('button', { name: '指标 / Metric' })).toBeInTheDocument();
    expect(within(table).getByText('单位 / Unit')).toBeInTheDocument();
    expect(within(table).getByText('训练轮次 / Epoch')).toBeInTheDocument();
    expect(within(table).getByText('Episode')).toBeInTheDocument();
    expect(within(table).getByRole('button', { name: '时刻 / Time' })).toBeInTheDocument();

    fireEvent.click(screen.getByRole('button', { name: /导出 CSV/i }));
    expect(URL.createObjectURL).toHaveBeenCalled();
    expect(appendChild).toHaveBeenCalled();
    expect(click).toHaveBeenCalled();
  });

  it('uses a bilingual empty-row label', () => {
    render(<MetricTable title="数据行 / Dataset rows" rows={[]} />);

    expect(screen.getByText('无数据行 / No rows')).toBeInTheDocument();
  });

  it('paginates metric rows instead of silently truncating them', () => {
    render(<MetricTable title="数据行 / Dataset rows" rows={rows} limit={1} />);

    expect(screen.getByText('pv_power')).toBeInTheDocument();
    expect(screen.queryByText('ev_charging_load')).not.toBeInTheDocument();
    expect(screen.getByText('第 1/2 页 / Page 1/2')).toBeInTheDocument();

    fireEvent.click(screen.getByRole('button', { name: '下一页 / Next page' }));

    expect(screen.queryByText('pv_power')).not.toBeInTheDocument();
    expect(screen.getByText('ev_charging_load')).toBeInTheDocument();
    expect(screen.getByText('第 2/2 页 / Page 2/2')).toBeInTheDocument();
  });

  it('lets users hide table columns without changing the underlying rows', () => {
    render(<MetricTable title="数据行 / Dataset rows" rows={rows} />);

    const table = screen.getByRole('table');
    expect(within(table).getByText('单位 / Unit')).toBeInTheDocument();

    fireEvent.click(screen.getByLabelText('显示单位列 / Show unit column'));

    expect(within(table).queryByText('单位 / Unit')).not.toBeInTheDocument();
    expect(screen.getByText('2/2 行 / rows')).toBeInTheDocument();
  });
});
