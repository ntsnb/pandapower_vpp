type Props = {
  vppIds: string[];
  value?: string;
  onChange: (value: string | undefined) => void;
};

function labelForVpp(vppId: string): string {
  if (vppId === 'aggregate') {
    return '聚合 / Aggregate';
  }
  if (vppId.startsWith('vpp_')) {
    return vppId.replace('vpp_', 'VPP-');
  }
  return vppId;
}

export function VppWheelPicker({ vppIds, value, onChange }: Props) {
  return (
    <label title="vpp_id 是虚拟电厂编号；选择全部 VPP 会展示所有逐 VPP 行，聚合 / Aggregate 表示跨 VPP 汇总行。 / vpp_id identifies a virtual power plant; All VPPs shows all per-VPP rows, Aggregate is a summed row.">
      VPP / 虚拟电厂
      <select value={value ?? ''} onChange={(event) => onChange(event.target.value || undefined)}>
        <option value="">全部 VPP / All VPPs</option>
        {vppIds.map((vppId) => (
          <option key={vppId} value={vppId}>
            {labelForVpp(vppId)}
          </option>
        ))}
      </select>
    </label>
  );
}
