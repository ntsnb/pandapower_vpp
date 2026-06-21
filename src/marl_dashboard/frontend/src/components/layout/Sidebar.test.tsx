import { fireEvent, render, screen } from '@testing-library/react';
import { describe, expect, it, vi } from 'vitest';

import { Sidebar } from './Sidebar';

describe('Sidebar', () => {
  it('keeps reward and cost composition under a single reward-cost page', () => {
    const onPageChange = vi.fn();
    render(<Sidebar page="overview" onPageChange={onPageChange} />);

    fireEvent.click(screen.getByRole('button', { name: /奖励成本 \/ Reward Cost/i }));

    expect(onPageChange).toHaveBeenCalledWith('reward-cost');
    expect(screen.queryByRole('button', { name: /无成本奖励 \/ Cost-free Reward/i })).not.toBeInTheDocument();
    expect(screen.queryByRole('button', { name: /缩放成本 \/ Scaled Costs/i })).not.toBeInTheDocument();
  });
});
