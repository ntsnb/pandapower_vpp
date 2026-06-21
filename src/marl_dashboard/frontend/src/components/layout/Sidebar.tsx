import { Activity, BookOpen, Boxes, Database, GitCompare, LineChart, Network, Settings, SlidersHorizontal } from 'lucide-react';

type SidebarProps = {
  page: string;
  onPageChange: (page: string) => void;
};

const pages = [
  { id: 'overview', label: '总览 / Overview', Icon: Activity },
  { id: 'dataset', label: '数据集 / Dataset', Icon: Database },
  { id: 'reward-cost', label: '奖励成本 / Reward Cost', Icon: Boxes },
  { id: 'loss', label: '损失 / Loss', Icon: LineChart },
  { id: 'compare', label: '对比 / Compare', Icon: GitCompare },
  { id: 'flexible', label: '灵活对比 / Flexible', Icon: SlidersHorizontal },
  { id: 'topology', label: '拓扑 / Topology', Icon: Network },
  { id: 'variables', label: '变量 / Variables', Icon: BookOpen },
  { id: 'config', label: '配置 / Run Config', Icon: Settings }
];

export function Sidebar({ page, onPageChange }: SidebarProps) {
  return (
    <aside className="sidebar">
      <div className="brand">MARL VPP</div>
      <nav>
        {pages.map(({ id, label, Icon }) => (
          <button key={id} className={page === id ? 'active' : ''} onClick={() => onPageChange(id)}>
            <Icon size={18} strokeWidth={1.8} />
            {label}
          </button>
        ))}
      </nav>
    </aside>
  );
}
