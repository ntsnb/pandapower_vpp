import { fireEvent, render, screen, within } from '@testing-library/react';
import { describe, expect, it } from 'vitest';

import { DatasetTimeseriesTable } from './DatasetTimeseriesTable';

const rows = [
  {
    run_id: 'run_a',
    metric_group: 'dataset',
    metric_name: 'electricity_price',
    display_name: '电价 / Electricity price',
    value: 0.31,
    unit: 'CNY/kWh',
    date: '2018-01-01',
    time_index: 0,
    vpp_id: 'vpp_001'
  },
  {
    run_id: 'run_a',
    metric_group: 'dataset',
    metric_name: 'pv_power',
    display_name: '光伏出力 / PV power',
    value: 1.2,
    unit: 'MW',
    date: '2018-01-01',
    time_index: 1,
    vpp_id: 'vpp_001'
  }
];

describe('DatasetTimeseriesTable', () => {
  it('paginates timeseries rows instead of silently truncating them', () => {
    render(<DatasetTimeseriesTable rows={rows} units={{ electricity_price: 'CNY/kWh', pv_power: 'MW' }} limit={1} />);

    expect(screen.getByText('2018-01-01 #0')).toBeInTheDocument();
    expect(screen.queryByText('2018-01-01 #1')).not.toBeInTheDocument();
    expect(screen.getByText('第 1/2 页 / Page 1/2')).toBeInTheDocument();

    fireEvent.click(screen.getByRole('button', { name: '下一页 / Next page' }));

    expect(screen.queryByText('2018-01-01 #0')).not.toBeInTheDocument();
    expect(screen.getByText('2018-01-01 #1')).toBeInTheDocument();
  });

  it('lets users hide dataset metric columns', () => {
    render(<DatasetTimeseriesTable rows={rows} units={{ electricity_price: 'CNY/kWh', pv_power: 'MW' }} />);

    const table = screen.getByRole('table');
    expect(within(table).getByText('电价 / Electricity price (CNY/kWh)')).toBeInTheDocument();

    fireEvent.click(screen.getByLabelText('显示 electricity_price 列 / Show electricity_price column'));

    expect(within(table).queryByText('电价 / Electricity price (CNY/kWh)')).not.toBeInTheDocument();
    expect(screen.getByText('2/2 时刻行 / time rows')).toBeInTheDocument();
  });
});
