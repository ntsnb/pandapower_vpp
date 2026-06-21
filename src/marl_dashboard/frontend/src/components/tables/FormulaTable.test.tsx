import { render, screen } from '@testing-library/react';
import { describe, expect, it } from 'vitest';

import { FormulaTable } from './FormulaTable';

describe('FormulaTable', () => {
  it('renders formulas through KaTeX output instead of plain strings only', () => {
    render(<FormulaTable formulas={{ total_reward: 'r_t=r^{profit}_t-p^{viol}_t' }} />);

    expect(screen.getByText('total_reward')).toBeInTheDocument();
    expect(document.querySelector('.katex')).toBeInTheDocument();
  });

  it('shows bilingual metric names, units, and physical descriptions for formula rows', () => {
    render(
      <FormulaTable
        formulas={{ profit_reward: 'R^{\\text{收益}}_{i,t}=\\pi_t P^{\\text{交付}}_{i,t}\\Delta t' }}
        rows={[
          {
            run_id: 'run_a',
            metric_group: 'reward',
            metric_name: 'profit_reward',
            display_name: '收益奖励 / Profit reward',
            description: 'VPP 因有效交付电能获得的收益奖励。 / Reward from delivered energy.',
            value: 12.3,
            unit: 'score'
          }
        ]}
      />
    );

    expect(screen.getByText('收益奖励 / Profit reward')).toBeInTheDocument();
    expect(screen.getByText('profit_reward')).toBeInTheDocument();
    expect(screen.getByText('单位 / Unit: score')).toBeInTheDocument();
    expect(screen.getByText(/VPP 因有效交付电能获得的收益奖励/)).toBeInTheDocument();
  });

  it('orders reward formulas from aggregate terms to child components', () => {
    render(
      <FormulaTable
        scope="reward"
        formulas={{
          pv_export_revenue_total: 'R^{PV}_{i,t}=\\pi_t P^{PV}_{i,t}',
          total_reward: 'R^{total}_{i,t}=\\sum_k r^{(k)}_{i,t}',
          export_revenue_total: 'R^{export}_{i,t}=R^{PV}_{i,t}+R^{MT}_{i,t}',
          market_energy_margin_total: 'M^{energy}_{i,t}=R^{export}_{i,t}+R^{EV}_{i,t}-C^{import}_{i,t}'
        }}
        rows={[
          {
            run_id: 'run_a',
            metric_group: 'reward',
            metric_name: 'pv_export_revenue_total',
            display_name: '光伏外送收入 / PV export revenue',
            value: 1
          },
          {
            run_id: 'run_a',
            metric_group: 'reward',
            metric_name: 'total_reward',
            display_name: '总奖励 / Total reward',
            value: 1
          },
          {
            run_id: 'run_a',
            metric_group: 'reward',
            metric_name: 'export_revenue_total',
            display_name: '总外送收入 / Total export revenue',
            value: 1
          },
          {
            run_id: 'run_a',
            metric_group: 'reward',
            metric_name: 'market_energy_margin_total',
            display_name: '市场电能边际收益 / Market energy margin',
            value: 1
          }
        ]}
      />
    );

    const formulaTitles = Array.from(document.querySelectorAll('.formula-title')).map((element) => element.textContent);

    expect(formulaTitles).toEqual([
      '总奖励 / Total reward',
      '市场电能边际收益 / Market energy margin',
      '总外送收入 / Total export revenue',
      '光伏外送收入 / PV export revenue'
    ]);
  });

  it('orders cost formulas from total cost to detailed cost components', () => {
    render(
      <FormulaTable
        scope="cost"
        formulas={{
          storage_charge_cost_total: 'C^{charge}_{i,t}=\\pi_t P^{charge}_{i,t}',
          total_cost: 'C^{total}_{i,t}=\\sum_k C^{(k)}_{i,t}',
          import_energy_cost_total: 'C^{import}_{i,t}=\\pi_t P^{import}_{i,t}',
          energy_purchase_cost: 'C^{buy}_{i,t}=\\pi_t max(P^{net}_{i,t},0)'
        }}
        rows={[
          {
            run_id: 'run_a',
            metric_group: 'cost',
            metric_name: 'storage_charge_cost_total',
            display_name: '储能充电成本 / Storage charge cost',
            value: 1
          },
          {
            run_id: 'run_a',
            metric_group: 'cost',
            metric_name: 'total_cost',
            display_name: '总成本 / Total cost',
            value: 1
          },
          {
            run_id: 'run_a',
            metric_group: 'cost',
            metric_name: 'import_energy_cost_total',
            display_name: '总购电成本 / Total import energy cost',
            value: 1
          },
          {
            run_id: 'run_a',
            metric_group: 'cost',
            metric_name: 'energy_purchase_cost',
            display_name: '购电成本 / Energy purchase cost',
            value: 1
          }
        ]}
      />
    );

    const formulaTitles = Array.from(document.querySelectorAll('.formula-title')).map((element) => element.textContent);

    expect(formulaTitles).toEqual([
      '总成本 / Total cost',
      '总购电成本 / Total import energy cost',
      '购电成本 / Energy purchase cost',
      '储能充电成本 / Storage charge cost'
    ]);
  });

  it('separates included, raw breakdown, and diagnostic formulas', () => {
    render(
      <FormulaTable
        scope="reward"
        formulas={{
          dispatch_reward_train: 'r^{\\text{训练}}_{i,t}=r^{\\text{环境}}_{i,t}-C^{\\text{训练惩罚}}_{i,t}',
          export_revenue_total: 'R^{\\text{外送}}_{i,t}=R^{\\text{光伏}}_{i,t}+R^{\\text{储能}}_{i,t}',
          service_payment: 'R^{\\text{服务}}_{i,t}=p^{\\text{服务}}_t\\Delta P_{i,t}\\Delta t'
        }}
        rows={[
          {
            run_id: 'run_a',
            metric_group: 'reward',
            metric_name: 'dispatch_reward_train',
            display_name: '调度训练奖励 / Dispatch training reward',
            value: 1
          },
          {
            run_id: 'run_a',
            metric_group: 'reward',
            metric_name: 'export_revenue_total',
            display_name: '总外送收入 / Total export revenue',
            value: 1
          },
          {
            run_id: 'run_a',
            metric_group: 'reward',
            metric_name: 'service_payment',
            display_name: '服务补偿 / Service payment',
            value: 1
          }
        ]}
      />
    );

    expect(screen.getByText('计入最终训练目标 / Included in training objective')).toBeInTheDocument();
    expect(screen.getByText('原始物理分解 / Raw physical breakdown')).toBeInTheDocument();
    expect(screen.getByText('诊断或未计入 / Diagnostic or excluded')).toBeInTheDocument();
    expect(screen.getByText('调度训练奖励 / Dispatch training reward')).toBeInTheDocument();
    expect(screen.getByText('总外送收入 / Total export revenue')).toBeInTheDocument();
    expect(screen.getByText('服务补偿 / Service payment')).toBeInTheDocument();
  });
});
