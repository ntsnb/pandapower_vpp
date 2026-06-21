type Props = {
  policyIds: string[];
  value?: string;
  onChange: (value: string | undefined) => void;
};

export function PolicySelector({ policyIds, value, onChange }: Props) {
  return (
    <label title="policy_id 是策略网络或共享策略编号；多个 VPP 可以共享同一个 policy。 / policy_id identifies a policy network or shared policy; multiple VPPs may share one policy.">
      策略 / Policy
      <select value={value ?? ''} onChange={(event) => onChange(event.target.value || undefined)} disabled={policyIds.length === 0}>
        <option value="">全部策略 / All policies</option>
        {policyIds.map((policyId) => (
          <option key={policyId} value={policyId}>
            {policyId}
          </option>
        ))}
      </select>
    </label>
  );
}
