import { useMemo, useState } from 'react';

import type { MetricRow, VariableDefinition } from '../../api/types';
import { compactNumber } from '../../utils/filters';

type Props = {
  variables: VariableDefinition[];
  currentValues?: Record<string, MetricRow>;
};

function variableText(variable: VariableDefinition): string {
  return [
    variable.name,
    variable.display_name,
    variable.symbol,
    variable.unit,
    variable.group,
    variable.physical_meaning,
    variable.source,
    variable.notes
  ]
    .filter((value) => value !== undefined && value !== null)
    .join(' ')
    .toLowerCase();
}

function currentValueText(row: MetricRow | undefined): string {
  if (!row) {
    return '-';
  }
  const value = compactNumber(row.value);
  return row.unit ? `${value} ${row.unit}` : value;
}

function currentContextText(row: MetricRow | undefined): string {
  if (!row) {
    return '-';
  }
  const parts = [];
  if (row.vpp_id) parts.push(String(row.vpp_id));
  if (row.policy_id) parts.push(String(row.policy_id));
  if (row.date) parts.push(String(row.date));
  if (row.time_index !== undefined && row.time_index !== null) parts.push(`t=${row.time_index}`);
  if (row.gradient_step !== undefined && row.gradient_step !== null) parts.push(`grad=${row.gradient_step}`);
  return parts.length > 0 ? parts.join(' @ ') : '-';
}

export function VariableDictionaryTable({ variables, currentValues = {} }: Props) {
  const [query, setQuery] = useState('');
  const [group, setGroup] = useState('');
  const groups = useMemo(
    () => Array.from(new Set(variables.map((variable) => variable.group).filter(Boolean) as string[])).sort(),
    [variables]
  );
  const visibleVariables = useMemo(() => {
    const normalizedQuery = query.trim().toLowerCase();
    return variables
      .filter((variable) => (group ? variable.group === group : true))
      .filter((variable) => (normalizedQuery ? variableText(variable).includes(normalizedQuery) : true))
      .sort((left, right) => left.name.localeCompare(right.name));
  }, [group, query, variables]);
  return (
    <section className="panel table-panel">
      <div className="panel-header">
        <h2>变量字典 / Variable Dictionary</h2>
        <span>
          {visibleVariables.length}/{variables.length} variables
        </span>
      </div>
      <div className="table-toolbar">
        <label>
          搜索变量 / Search variables
          <input value={query} onChange={(event) => setQuery(event.target.value)} />
        </label>
        <label>
          变量分组 / Variable group
          <select value={group} onChange={(event) => setGroup(event.target.value)}>
            <option value="">全部分组 / All groups</option>
            {groups.map((item) => (
              <option key={item} value={item}>
                {item}
              </option>
            ))}
          </select>
        </label>
      </div>
      <div className="table-scroll">
        <table>
          <thead>
            <tr>
              <th>变量名 / Name</th>
              <th>显示名 / Display</th>
              <th>符号 / Symbol</th>
              <th>单位 / Unit</th>
              <th>当前值 / Current value</th>
              <th>当前位置 / Current context</th>
              <th>分组 / Group</th>
              <th>物理意义 / Meaning</th>
              <th>来源 / Source</th>
              <th>备注 / Notes</th>
            </tr>
          </thead>
          <tbody>
            {visibleVariables.map((variable) => (
              <tr key={variable.name}>
                <td>{variable.name}</td>
                <td>{variable.display_name ?? '-'}</td>
                <td>{variable.symbol ?? '-'}</td>
                <td>{variable.unit ?? '-'}</td>
                <td>{currentValueText(currentValues[variable.name])}</td>
                <td>{currentContextText(currentValues[variable.name])}</td>
                <td>{variable.group ?? '-'}</td>
                <td>{variable.physical_meaning ?? '-'}</td>
                <td>{variable.source ?? '-'}</td>
                <td>{variable.notes ?? '-'}</td>
              </tr>
            ))}
            {visibleVariables.length === 0 ? (
              <tr>
                <td colSpan={10} className="empty-cell">
                  无变量 / No variables
                </td>
              </tr>
            ) : null}
          </tbody>
        </table>
      </div>
    </section>
  );
}
