import { fireEvent, render, screen } from '@testing-library/react';
import { describe, expect, it, vi } from 'vitest';

import { TimeStepSlider } from './TimeStepSlider';

describe('TimeStepSlider', () => {
  it('labels an unset value as all time indices and snaps slider positions to real time indices', () => {
    const onChange = vi.fn();

    render(<TimeStepSlider timeIndices={[24, 48, 72]} value={undefined} onChange={onChange} />);

    expect(screen.getByText('时刻 / Time index: 全部时刻 / All time indices')).toBeInTheDocument();
    expect(screen.getByTitle(/time_index 是一天内第几个能源数据点/)).toBeInTheDocument();

    fireEvent.change(screen.getByRole('slider'), { target: { value: '1' } });

    expect(onChange).toHaveBeenCalledWith(48);
  });
});
