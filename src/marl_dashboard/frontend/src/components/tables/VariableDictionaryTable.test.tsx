import { fireEvent, render, screen } from '@testing-library/react';
import { describe, expect, it } from 'vitest';

import { VariableDictionaryTable } from './VariableDictionaryTable';

describe('VariableDictionaryTable', () => {
  it('searches variables and filters by group', () => {
    render(
      <VariableDictionaryTable
        variables={[
          {
            name: 'pv_power',
            display_name: 'PV power',
            unit: 'MW',
            group: 'dataset',
            physical_meaning: 'PV generation proxy',
            source: 'scenario profile'
          },
          {
            name: 'policy_loss',
            display_name: 'Policy loss',
            unit: 'scalar',
            group: 'loss',
            physical_meaning: 'Policy optimization objective',
            source: 'update metrics'
          }
        ]}
      />
    );

    fireEvent.change(screen.getByLabelText('搜索变量 / Search variables'), { target: { value: 'policy' } });
    expect(screen.getByText('policy_loss')).toBeInTheDocument();
    expect(screen.queryByText('pv_power')).not.toBeInTheDocument();

    fireEvent.change(screen.getByLabelText('变量分组 / Variable group'), { target: { value: 'dataset' } });
    expect(screen.getByText('无变量 / No variables')).toBeInTheDocument();

    fireEvent.change(screen.getByLabelText('搜索变量 / Search variables'), { target: { value: '' } });
    expect(screen.getByText('pv_power')).toBeInTheDocument();
    expect(screen.queryByText('policy_loss')).not.toBeInTheDocument();
  });

  it('uses bilingual labels for variable explanations', () => {
    render(
      <VariableDictionaryTable
        variables={[
          {
            name: 'reward_so_far',
            display_name: '累计奖励 / Episode reward so far',
            unit: 'score',
            group: 'reward',
            physical_meaning: '当前 episode 内从开始到当前 time_index 已累计获得的奖励。 / Cumulative reward in the active episode.'
          }
        ]}
      />
    );

    expect(screen.getByLabelText('搜索变量 / Search variables')).toBeInTheDocument();
    expect(screen.getByText('变量名 / Name')).toBeInTheDocument();
    expect(screen.getByText('物理意义 / Meaning')).toBeInTheDocument();
    expect(screen.getByText(/当前 episode 内/)).toBeInTheDocument();
    expect(screen.getByText(/Cumulative reward/)).toBeInTheDocument();
  });

  it('shows the current selected metric value beside each variable', () => {
    render(
      <VariableDictionaryTable
        variables={[
          {
            name: 'electricity_price',
            display_name: '电价 / Electricity price',
            unit: '$/MWh',
            group: 'dataset',
            physical_meaning: '当前时刻电价。 / Current electricity price.'
          }
        ]}
        currentValues={{
          electricity_price: {
            run_id: 'run_a',
            metric_group: 'dataset',
            metric_name: 'electricity_price',
            value: 88.5,
            unit: '$/MWh',
            vpp_id: 'vpp_001',
            time_index: 3
          }
        }}
      />
    );

    expect(screen.getByText('当前值 / Current value')).toBeInTheDocument();
    expect(screen.getByText('88.5 $/MWh')).toBeInTheDocument();
    expect(screen.getByText('vpp_001 @ t=3')).toBeInTheDocument();
  });
});
