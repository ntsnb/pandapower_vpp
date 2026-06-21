import { fireEvent, render, screen, within } from '@testing-library/react';
import { describe, expect, it } from 'vitest';

import type { MetricRow } from '../../api/types';
import { CompareMetricMatrix } from './CompareMetricMatrix';

const rows: MetricRow[] = [
  {
    run_id: 'run_a',
    metric_group: 'dataset',
    metric_name: 'electricity_price',
    value: 51,
    unit: 'currency/MWh',
    date: '2018-01-01',
    time_index: 0,
    group: 'vpp_001',
    vpp_id: 'vpp_001'
  },
  {
    run_id: 'run_a',
    metric_group: 'dataset',
    metric_name: 'net_load',
    value: 11,
    unit: 'MW',
    date: '2018-01-01',
    time_index: 0,
    group: 'vpp_001',
    vpp_id: 'vpp_001'
  },
  {
    run_id: 'run_a',
    metric_group: 'dataset',
    metric_name: 'electricity_price',
    value: 52,
    unit: 'currency/MWh',
    date: '2018-01-01',
    time_index: 0,
    group: 'vpp_002',
    vpp_id: 'vpp_002'
  },
  {
    run_id: 'run_a',
    metric_group: 'dataset',
    metric_name: 'net_load',
    value: 22,
    unit: 'MW',
    date: '2018-01-01',
    time_index: 0,
    group: 'vpp_002',
    vpp_id: 'vpp_002'
  },
  {
    run_id: 'run_a',
    metric_group: 'dataset',
    metric_name: 'electricity_price',
    value: 53,
    unit: 'currency/MWh',
    date: '2018-01-01',
    time_index: 0,
    group: 'vpp_003',
    vpp_id: 'vpp_003'
  },
  {
    run_id: 'run_a',
    metric_group: 'dataset',
    metric_name: 'net_load',
    value: 33,
    unit: 'MW',
    date: '2018-01-01',
    time_index: 0,
    group: 'vpp_003',
    vpp_id: 'vpp_003'
  }
];

describe('CompareMetricMatrix', () => {
  it('paginates compare groups instead of silently truncating them', () => {
    render(
      <CompareMetricMatrix
        rows={rows}
        units={{ electricity_price: 'currency/MWh', net_load: 'MW' }}
        groupBy="vpp_id"
        limit={2}
      />
    );

    expect(screen.getByRole('row', { name: /vpp_001.*51.*11/i })).toBeInTheDocument();
    expect(screen.getByRole('row', { name: /vpp_002.*52.*22/i })).toBeInTheDocument();
    expect(screen.queryByRole('row', { name: /vpp_003/i })).not.toBeInTheDocument();
    expect(screen.getByText('第 1/2 页 / Page 1/2')).toBeInTheDocument();

    fireEvent.click(screen.getByRole('button', { name: '下一页 / Next page' }));

    expect(screen.queryByRole('row', { name: /vpp_001/i })).not.toBeInTheDocument();
    expect(screen.getByRole('row', { name: /vpp_003.*53.*33/i })).toBeInTheDocument();
    expect(screen.getByText('第 2/2 页 / Page 2/2')).toBeInTheDocument();
  });

  it('lets users hide selected metric columns without hiding compare groups', () => {
    render(
      <CompareMetricMatrix
        rows={rows}
        units={{ electricity_price: 'currency/MWh', net_load: 'MW' }}
        groupBy="vpp_id"
        limit={5}
      />
    );

    const table = screen.getByRole('table');
    expect(within(table).getByText('net_load (MW)')).toBeInTheDocument();
    expect(screen.getByRole('row', { name: /vpp_001.*51.*11/i })).toBeInTheDocument();

    fireEvent.click(screen.getByLabelText('显示 net_load 列 / Show net_load column'));

    expect(within(table).queryByText('net_load (MW)')).not.toBeInTheDocument();
    expect(screen.getByRole('row', { name: /vpp_001.*51/i })).toBeInTheDocument();
    expect(screen.queryByRole('row', { name: /vpp_001.*11/i })).not.toBeInTheDocument();
    expect(screen.getByText('3/3 对象 / groups')).toBeInTheDocument();
  });
});
