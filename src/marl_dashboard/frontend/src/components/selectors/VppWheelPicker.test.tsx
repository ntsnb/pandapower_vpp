import { render, screen } from '@testing-library/react';
import { describe, expect, it, vi } from 'vitest';

import { VppWheelPicker } from './VppWheelPicker';

describe('VppWheelPicker', () => {
  it('labels aggregate data without implying per-VPP rows exist', () => {
    render(<VppWheelPicker vppIds={['aggregate']} value={undefined} onChange={vi.fn()} />);

    expect(screen.getByRole('option', { name: '全部 VPP / All VPPs' })).toBeInTheDocument();
    expect(screen.getByRole('option', { name: '聚合 / Aggregate' })).toBeInTheDocument();
    expect(screen.getByTitle(/vpp_id 是虚拟电厂编号/)).toBeInTheDocument();
  });
});
