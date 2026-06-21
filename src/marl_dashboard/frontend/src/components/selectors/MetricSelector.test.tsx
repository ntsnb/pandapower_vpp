import { fireEvent, render, screen } from '@testing-library/react';
import { describe, expect, it, vi } from 'vitest';

import { MetricSelector } from './MetricSelector';

describe('MetricSelector', () => {
  it('shows bilingual metric labels while preserving raw metric values', () => {
    const onChange = vi.fn();

    render(
      <MetricSelector
        label="奖励项 / Reward metric"
        allLabel="全部奖励项 / All reward terms"
        metrics={['availability_payment']}
        metricLabels={{ availability_payment: '可用容量补偿 / availability_payment' }}
        value=""
        onChange={onChange}
      />
    );

    expect(screen.getByRole('option', { name: '可用容量补偿 / availability_payment' })).toBeInTheDocument();

    fireEvent.change(screen.getByLabelText('奖励项 / Reward metric'), { target: { value: 'availability_payment' } });

    expect(onChange).toHaveBeenCalledWith('availability_payment');
  });
});
