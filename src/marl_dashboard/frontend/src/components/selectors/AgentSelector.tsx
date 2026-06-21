type Props = {
  agentIds: string[];
  value?: string;
  onChange: (value: string | undefined) => void;
};

export function AgentSelector({ agentIds, value, onChange }: Props) {
  return (
    <label title="agent_id 是算法内部智能体编号；通常与某个 VPP 的 dispatch agent 对应。 / agent_id identifies the MARL agent, often the dispatch agent for one VPP.">
      智能体 / Agent
      <select value={value ?? ''} onChange={(event) => onChange(event.target.value || undefined)} disabled={agentIds.length === 0}>
        <option value="">全部智能体 / All agents</option>
        {agentIds.map((agentId) => (
          <option key={agentId} value={agentId}>
            {agentId}
          </option>
        ))}
      </select>
    </label>
  );
}
