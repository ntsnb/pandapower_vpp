import { render, screen } from '@testing-library/react';
import { describe, expect, it, vi } from 'vitest';

import { CompositionChart } from './CompositionChart';

vi.mock('echarts-for-react', () => ({
  default: ({ option }: { option: unknown }) => <pre data-testid="chart-option">{JSON.stringify(option)}</pre>
}));

describe('CompositionChart', () => {
  it('shows a clear empty-state message when the selected filters have no composition rows', () => {
    render(
      <CompositionChart
        title="缩放成本组成 / Reward-scaled cost composition"
        response={{ chart_series: [], table_rows: [], units: {}, formulas: {}, summary: { row_count: 0 } }}
      />
    );

    expect(screen.getByRole('heading', { name: '缩放成本组成 / Reward-scaled cost composition' })).toBeInTheDocument();
    expect(screen.getByText(/当前筛选没有组成数据/)).toBeInTheDocument();
    expect(screen.queryByTestId('chart-option')).not.toBeInTheDocument();
  });
});
