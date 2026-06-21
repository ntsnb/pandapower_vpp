import { useMemo } from 'react';

import { api } from '../api/client';
import { useAsync } from '../api/hooks';
import type { TopologyAsset, TopologyEdge, TopologyNode, TopologyPayload, VppAssetConfig, VppConfig } from '../api/types';
import { DataNotice } from '../components/layout/DataNotice';
import type { PageProps } from './types';

const VPP_COLORS = ['#0f766e', '#b45309', '#2563eb', '#be123c', '#7c3aed', '#15803d', '#0891b2', '#a16207'];

type Point = {
  x: number;
  y: number;
};

function fmt(value: number | string | null | undefined, digits = 3): string {
  if (value === null || value === undefined || value === '') {
    return '-';
  }
  if (typeof value === 'number') {
    return Number.isFinite(value) ? value.toFixed(digits).replace(/\.?0+$/, '') : '-';
  }
  return String(value);
}

function vppColorMap(vppIds: string[]): Map<string, string> {
  return new Map(vppIds.map((vppId, index) => [vppId, VPP_COLORS[index % VPP_COLORS.length]]));
}

function bounds(nodes: TopologyNode[]) {
  const xs = nodes.map((node) => Number(node.x ?? 0));
  const ys = nodes.map((node) => Number(node.y ?? 0));
  return {
    minX: Math.min(...xs, 0),
    maxX: Math.max(...xs, 1),
    minY: Math.min(...ys, 0),
    maxY: Math.max(...ys, 1)
  };
}

function scalePoint(node: TopologyNode, box: ReturnType<typeof bounds>): Point {
  const width = Math.max(1, box.maxX - box.minX);
  const height = Math.max(1, box.maxY - box.minY);
  return {
    x: 32 + ((Number(node.x ?? 0) - box.minX) / width) * 836,
    y: 32 + ((Number(node.y ?? 0) - box.minY) / height) * 356
  };
}

function assetOffsets(count: number): Point[] {
  if (count <= 0) return [];
  const radius = 13;
  return Array.from({ length: count }, (_, index) => ({
    x: Math.cos((2 * Math.PI * index) / count) * radius,
    y: Math.sin((2 * Math.PI * index) / count) * radius
  }));
}

function nodeVppId(node: TopologyNode, assets: TopologyAsset[]): string {
  const fromPcc = String(node.vpp_ids ?? '')
    .split(',')
    .map((item) => item.trim())
    .find(Boolean);
  if (fromPcc) return fromPcc;
  return assets.find((asset) => Number(asset.bus_id) === Number(node.bus_id))?.vpp_id ?? '';
}

function TopologySvg({ topology }: { topology: TopologyPayload }) {
  const byBus = new Map(topology.nodes.map((node) => [Number(node.bus_id), node]));
  const assetsByBus = new Map<number, TopologyAsset[]>();
  for (const asset of topology.assets) {
    const values = assetsByBus.get(Number(asset.bus_id)) ?? [];
    values.push(asset);
    assetsByBus.set(Number(asset.bus_id), values);
  }
  const colorByVpp = vppColorMap(Array.from(new Set(topology.assets.map((asset) => asset.vpp_id))).sort());
  const box = bounds(topology.nodes);
  const positions = new Map(topology.nodes.map((node) => [Number(node.bus_id), scalePoint(node, box)]));

  return (
    <svg className="topology-svg" data-testid="pandapower-topology-svg" viewBox="0 0 900 420" role="img">
      <title>pandapower distribution topology</title>
      <rect x="0" y="0" width="900" height="420" rx="8" className="topology-background" />
      {topology.edges.map((edge: TopologyEdge) => {
        const from = positions.get(Number(edge.from_bus));
        const to = positions.get(Number(edge.to_bus));
        if (!from || !to) return null;
        return (
          <g key={edge.edge_id}>
            <line
              x1={from.x}
              y1={from.y}
              x2={to.x}
              y2={to.y}
              className={edge.edge_type === 'trafo' ? 'topology-edge trafo' : 'topology-edge'}
            />
            <text x={(from.x + to.x) / 2} y={(from.y + to.y) / 2 - 5} className="topology-edge-label">
              {edge.voltage_level_label ?? edge.edge_type}
            </text>
          </g>
        );
      })}
      {topology.nodes.map((node) => {
        const position = positions.get(Number(node.bus_id));
        if (!position) return null;
        const connectedAssets = assetsByBus.get(Number(node.bus_id)) ?? [];
        const vppId = nodeVppId(node, topology.assets);
        const color = colorByVpp.get(vppId) ?? '#64748b';
        return (
          <g key={node.bus_id}>
            <circle
              cx={position.x}
              cy={position.y}
              r={node.is_slack ? 8 : node.is_pcc ? 7 : 4.5}
              fill={node.is_pcc ? color : node.is_slack ? '#111827' : '#f8fafc'}
              className="topology-node"
            />
            {node.is_pcc ? (
              <text x={position.x} y={position.y - 13} className="topology-pcc-label">
                PCC {node.bus_id}
              </text>
            ) : null}
            <text x={position.x} y={position.y + 17} className="topology-node-label">
              {node.bus_id}
            </text>
            {connectedAssets.map((asset, index) => {
              const offset = assetOffsets(connectedAssets.length)[index];
              return (
                <circle
                  key={asset.der_id}
                  cx={position.x + offset.x}
                  cy={position.y + offset.y}
                  r={3.5}
                  fill={colorByVpp.get(asset.vpp_id) ?? '#64748b'}
                  className="topology-asset-dot"
                />
              );
            })}
          </g>
        );
      })}
      <g transform="translate(16 392)">
        <circle cx="0" cy="0" r="5" fill="#111827" />
        <text x="12" y="4" className="topology-legend-text">
          Slack / 上级电网
        </text>
        <circle cx="140" cy="0" r="5" fill="#0f766e" />
        <text x="152" y="4" className="topology-legend-text">
          PCC / VPP 接入点
        </text>
        <circle cx="306" cy="0" r="3.5" fill="#b45309" />
        <text x="318" y="4" className="topology-legend-text">
          DER asset / 资源
        </text>
      </g>
    </svg>
  );
}

function Stat({ label, value }: { label: string; value: string | number | null | undefined }) {
  return (
    <div className="topology-stat">
      <dt>{label}</dt>
      <dd>{value ?? '-'}</dd>
    </div>
  );
}

function hasValue(value: number | string | null | undefined): boolean {
  return value !== null && value !== undefined && value !== '';
}

function assetPowerText(asset: VppAssetConfig): string[] {
  const parts = [`P ${fmt(asset.p_min_mw)}..${fmt(asset.p_max_mw)} MW`];
  if (hasValue(asset.q_min_mvar) || hasValue(asset.q_max_mvar)) {
    parts.push(`Q ${fmt(asset.q_min_mvar)}..${fmt(asset.q_max_mvar)} Mvar`);
  }
  if (asset.capacity_mwh !== undefined && asset.capacity_mwh !== null) {
    parts.push(`capacity ${fmt(asset.capacity_mwh)} MWh`);
  }
  if (hasValue(asset.soc) || hasValue(asset.soc_min) || hasValue(asset.soc_max)) {
    const bounds = hasValue(asset.soc_min) || hasValue(asset.soc_max) ? ` (${fmt(asset.soc_min)}..${fmt(asset.soc_max)})` : '';
    parts.push(`SOC ${fmt(asset.soc)}${bounds}`);
  }
  if (hasValue(asset.p_charge_max_mw)) {
    parts.push(`charge ${fmt(asset.p_charge_max_mw)} MW`);
  }
  if (hasValue(asset.p_discharge_max_mw)) {
    parts.push(`discharge ${fmt(asset.p_discharge_max_mw)} MW`);
  }
  if (asset.n_evs !== undefined && asset.n_evs !== null) {
    parts.push(`${asset.n_evs} EVs`);
  }
  if (hasValue(asset.rated_power_mw)) {
    parts.push(`rated ${fmt(asset.rated_power_mw)} MW`);
  }
  if (hasValue(asset.baseline_p_mw)) {
    parts.push(`baseline ${fmt(asset.baseline_p_mw)} MW`);
  }
  if (hasValue(asset.apparent_power_mva)) {
    parts.push(`S ${fmt(asset.apparent_power_mva)} MVA`);
  }
  if (hasValue(asset.indoor_temp) || hasValue(asset.temp_min) || hasValue(asset.temp_max)) {
    const bounds =
      hasValue(asset.temp_min) || hasValue(asset.temp_max) ? ` (${fmt(asset.temp_min)}..${fmt(asset.temp_max)} °C)` : '';
    parts.push(`temperature ${fmt(asset.indoor_temp)} °C${bounds}`);
  }
  if (hasValue(asset.pp_element_type) || hasValue(asset.pp_element_index)) {
    parts.push(`pandapower ${asset.pp_element_type || 'element'} #${fmt(asset.pp_element_index)}`);
  }
  if (hasValue(asset.write_target)) {
    parts.push(String(asset.write_target));
  }
  if (hasValue(asset.zone_id)) {
    parts.push(`zone ${asset.zone_id}`);
  }
  if (hasValue(asset.feeder_id)) {
    parts.push(`feeder ${asset.feeder_id}`);
  }
  parts.push(asset.controllable === false ? '不可控 / fixed' : '可控 / controllable');
  return parts;
}

function rangeText(values: number[] | undefined, unit: string): string {
  if (!values || values.length < 2) {
    return '-';
  }
  return `${fmt(values[0])}..${fmt(values[1])} ${unit}`;
}

function TopologyDetails({ topology }: { topology: TopologyPayload | null }) {
  if (!topology) {
    return null;
  }
  const tableEntries = Object.entries(topology.pandapower_tables ?? {});
  const conventionEntries = Object.entries(topology.sign_conventions ?? {});
  return (
    <div className="topology-info-grid">
      <section className="topology-info-card">
        <h3>pandapower 元件统计 / pandapower element tables</h3>
        <div className="topology-chip-grid">
          {tableEntries.map(([name, value]) => (
            <span key={name}>
              {name}: <strong>{value}</strong>
            </span>
          ))}
        </div>
      </section>
      <section className="topology-info-card">
        <h3>VPP 接入映射 / VPP bus mapping</h3>
        <div className="topology-map-list">
          {(topology.vpp_bus_map ?? []).map((mapping) => (
            <p key={mapping.vpp_id}>
              <strong>
                {mapping.display_name}: PCC {mapping.pcc_bus}
              </strong>
              <span>
                buses {mapping.asset_buses.join(', ') || '-'}; assets {mapping.asset_ids.join(', ') || '-'}; pandapower{' '}
                {mapping.pp_elements.join(', ') || '-'}
              </span>
            </p>
          ))}
        </div>
      </section>
      <section className="topology-info-card wide">
        <h3>符号约定 / Sign conventions</h3>
        <ul className="note-list">
          {conventionEntries.map(([name, text]) => (
            <li key={name}>
              <strong>{name}</strong>: {text}
            </li>
          ))}
        </ul>
      </section>
    </div>
  );
}

function VppConfigCard({ vpp }: { vpp: VppConfig }) {
  const assetSummary = Object.entries(vpp.asset_counts)
    .map(([name, count]) => `${name}: ${count}`)
    .join(', ');
  return (
    <article className="vpp-config-card">
      <div className="panel-header">
        <h3>{vpp.display_name}</h3>
        <span>{vpp.vpp_id}</span>
      </div>
      <p className="bilingual-note">{vpp.description}</p>
      <dl className="topology-stat-grid">
        <Stat label="PCC 母线 / PCC bus" value={vpp.pcc_bus} />
        <Stat label="物理模式 / Physical mode" value={vpp.physical_mode_description} />
        <Stat label="隐私模式 / Privacy mode" value={vpp.privacy_mode_description} />
        <Stat label="连接母线 / Connection buses" value={vpp.connection_buses.join(', ')} />
        <Stat label="区域 / Zones" value={vpp.zone_ids.join(', ') || '-'} />
        <Stat label="DER 数量 / DER count" value={vpp.der_count} />
        <Stat label="进口能力 / Max import MW" value={fmt(vpp.max_import_mw)} />
        <Stat label="出口能力 / Max export MW" value={fmt(vpp.max_export_mw)} />
        <Stat label="P 调度范围 / P range" value={rangeText(vpp.dispatch_capability?.active_power_range_mw, 'MW')} />
        <Stat label="Q 调度范围 / Q range" value={rangeText(vpp.dispatch_capability?.reactive_power_range_mvar, 'Mvar')} />
      </dl>
      <ul className="note-list compact">
        {(vpp.configuration_notes ?? []).map((note) => (
          <li key={note}>{note}</li>
        ))}
      </ul>
      <div className="asset-summary">资产类型 / Asset types: {assetSummary || '-'}</div>
      <div className="asset-list">
        {vpp.assets.map((asset) => (
          <div className="asset-row" key={asset.der_id}>
            <div className="asset-name">
              <strong>{asset.display_name}</strong>
              <span>{asset.der_id}</span>
            </div>
            <div className="asset-meta">
              <span>{asset.der_type_description}</span>
              <span>bus {asset.bus_id}</span>
            </div>
            <div className="asset-detail-column">
              {asset.configuration_summary ? <p className="asset-config-summary">{asset.configuration_summary}</p> : null}
              <div className="asset-detail-tags">
                {assetPowerText(asset).map((part) => (
                  <span key={part}>{part}</span>
                ))}
              </div>
            </div>
          </div>
        ))}
      </div>
    </article>
  );
}

export function TopologyPage({ filters }: PageProps) {
  const topology = useAsync(() => (filters.runId ? api.topology(filters.runId) : Promise.resolve(null)), [filters.runId]);
  const vppConfig = useAsync(() => (filters.runId ? api.vppConfig(filters.runId) : Promise.resolve(null)), [filters.runId]);
  const colorLegend = useMemo(
    () => Array.from(new Set((topology.data?.assets ?? []).map((asset) => asset.vpp_id))).sort(),
    [topology.data]
  );

  return (
    <div className="page-stack">
      <DataNotice loading={topology.loading || vppConfig.loading} error={topology.error ?? vppConfig.error} />
      <section className="panel topology-panel">
        <div className="panel-header">
          <div>
            <h2>pandapower 配电网拓扑 / pandapower distribution topology</h2>
            <p className="bilingual-note">
              该图来自 pandapower 的 bus、line、trafo 表；坐标是仓库已有 deterministic feeder layout 的示意坐标，不是 GIS 地理坐标。
              / This is a schematic one-line layout derived from pandapower tables.
            </p>
          </div>
          <span>{topology.data?.source_config_path ?? 'scenario_config.yaml'}</span>
        </div>
        {topology.data ? <TopologySvg topology={topology.data} /> : <div className="empty-cell">无拓扑数据 / No topology data</div>}
        <dl className="topology-stat-grid compact">
          <Stat label="母线 / Buses" value={topology.data?.network.bus_count} />
          <Stat label="线路 / Lines" value={topology.data?.network.line_count} />
          <Stat label="变压器 / Transformers" value={topology.data?.network.trafo_count} />
          <Stat label="VPP 数量 / VPP count" value={topology.data?.network.vpp_count} />
        </dl>
        <TopologyDetails topology={topology.data} />
        <div className="topology-color-legend">
          {colorLegend.map((vppId, index) => (
            <span key={vppId}>
              <i style={{ background: VPP_COLORS[index % VPP_COLORS.length] }} />
              {vppId}
            </span>
          ))}
        </div>
      </section>

      <section className="panel">
        <div className="panel-header">
          <div>
            <h2>VPP 配置说明 / VPP configuration</h2>
            <p className="bilingual-note">
              展示每个 VPP 的 PCC、物理模式、隐私模式、连接母线和内部 DER 资产，帮助解释不同 VPP 为什么看到不同本地状态。
              / Each card describes the VPP portfolio and asset placement.
            </p>
          </div>
          <span>{vppConfig.data?.summary.asset_count ?? 0} DER assets</span>
        </div>
        <div className="vpp-config-grid">
          {(vppConfig.data?.vpps ?? []).map((vpp) => (
            <VppConfigCard key={vpp.vpp_id} vpp={vpp} />
          ))}
        </div>
      </section>
    </div>
  );
}
