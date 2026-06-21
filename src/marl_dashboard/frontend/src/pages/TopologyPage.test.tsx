import { render, screen } from '@testing-library/react';
import { describe, expect, it, vi } from 'vitest';

import type { TopologyPayload, VppConfigPayload } from '../api/types';
import { TopologyPage } from './TopologyPage';

const topology: TopologyPayload = {
  run_id: 'run_a',
  source_config_path: '/tmp/scenario_config.yaml',
  network: {
    name: 'ieee33',
    bus_count: 3,
    line_count: 2,
    trafo_count: 0,
    vpp_count: 2
  },
  pandapower_tables: {
    bus_count: 3,
    line_count: 2,
    trafo_count: 0,
    load_count: 1,
    sgen_count: 0,
    storage_count: 1,
    ext_grid_count: 1
  },
  sign_conventions: {
    load: 'pandapower load p_mw > 0 is consumption.',
    sgen: 'pandapower sgen p_mw > 0 is generation.',
    storage: 'pandapower storage p_mw > 0 means charging; dashboard internal dispatch p_mw > 0 means export.'
  },
  vpp_bus_map: [
    {
      vpp_id: 'vpp_a',
      display_name: 'Campus VPP',
      pcc_bus: 5,
      physical_mode: 'multi_node',
      asset_buses: [6],
      asset_ids: ['ess_a'],
      pp_elements: ['storage#7']
    }
  ],
  nodes: [
    { bus_id: 0, name: 'source', vn_kv: 12.66, is_slack: true, is_pcc: false, vpp_ids: '', asset_count: 0, x: 0, y: 0 },
    { bus_id: 5, name: 'pcc_a', vn_kv: 12.66, is_slack: false, is_pcc: true, vpp_ids: 'vpp_a', asset_count: 1, x: 1, y: 0 },
    { bus_id: 6, name: 'asset_bus', vn_kv: 12.66, is_slack: false, is_pcc: false, vpp_ids: '', asset_count: 1, x: 2, y: 0 }
  ],
  edges: [
    { edge_id: 'line_0', edge_type: 'line', from_bus: 0, to_bus: 5, name: 'line_0', voltage_level_label: '12.66 kV' },
    { edge_id: 'line_1', edge_type: 'line', from_bus: 5, to_bus: 6, name: 'line_1', voltage_level_label: '12.66 kV' }
  ],
  assets: [
    {
      der_id: 'ess_a',
      name: 'Battery A',
      vpp_id: 'vpp_a',
      vpp_name: 'Campus VPP',
      bus_id: 6,
      der_type: 'StorageModel',
      controllable: true,
      p_min_mw: -0.2,
      p_max_mw: 0.2
    }
  ],
  vpp_portfolios: [
    {
      vpp_id: 'vpp_a',
      name: 'Campus VPP',
      physical_mode: 'multi_node',
      pcc_bus_id: 5,
      connection_buses: '5,6',
      der_ids: 'ess_a',
      der_count: 1,
      max_import_mw: 0.2,
      max_export_mw: 0.2
    }
  ]
};

const vppConfig: VppConfigPayload = {
  run_id: 'run_a',
  source_config_path: '/tmp/scenario_config.yaml',
  summary: { vpp_count: 1, asset_count: 1 },
  vpps: [
    {
      vpp_id: 'vpp_a',
      display_name: 'Campus VPP',
      pcc_bus: 5,
      physical_mode: 'multi_node',
      physical_mode_description: '多节点 / Multi-node',
      privacy_mode: 'full_information',
      privacy_mode_description: '完整信息 / Full information',
      connection_buses: [5, 6],
      zone_ids: ['zone_a'],
      der_count: 1,
      asset_counts: { StorageModel: 1 },
      dispatch_capability: {
        active_power_range_mw: [-0.2, 0.2],
        reactive_power_range_mvar: [-0.05, 0.05],
        max_import_mw: 0.2,
        max_export_mw: 0.2
      },
      max_import_mw: 0.2,
      max_export_mw: 0.2,
      description: 'PCC 母线 5；DER assets: 1.',
      configuration_notes: [
        '接入母线 / connection buses: 5, 6',
        'pandapower 写入 / write target: storage#7'
      ],
      assets: [
        {
          der_id: 'ess_a',
          display_name: 'Battery A',
          der_type: 'StorageModel',
          der_type_description: '储能 / Storage',
          bus_id: 6,
          p_min_mw: -0.2,
          p_max_mw: 0.2,
          q_min_mvar: -0.05,
          q_max_mvar: 0.05,
          capacity_mwh: 1.0,
          soc: 0.5,
          soc_min: 0.1,
          soc_max: 0.9,
          p_charge_max_mw: 0.15,
          p_discharge_max_mw: 0.16,
          pp_element_type: 'storage',
          pp_element_index: 7,
          write_target: 'pandapower storage#7 at bus 6',
          configuration_summary: '储能 / Storage: P -0.2..0.2 MW, SOC 0.5 (0.1..0.9)',
          feeder_id: 'feeder_1',
          zone_id: 'zone_asset_a'
        }
      ]
    }
  ]
};

const mocks = vi.hoisted(() => ({
  topology: vi.fn(),
  vppConfig: vi.fn()
}));

vi.mock('../api/client', () => ({
  api: mocks
}));

describe('TopologyPage', () => {
  it('renders pandapower topology and detailed bilingual VPP configuration', async () => {
    mocks.topology.mockResolvedValue(topology);
    mocks.vppConfig.mockResolvedValue(vppConfig);

    render(
      <TopologyPage
        run={null}
        selectors={null}
        filters={{ runId: 'run_a', live: false, compareMode: false }}
      />
    );

    expect(await screen.findByText('pandapower 配电网拓扑 / pandapower distribution topology')).toBeInTheDocument();
    expect(screen.getByTestId('pandapower-topology-svg')).toBeInTheDocument();
    expect(screen.getByText('pandapower 元件统计 / pandapower element tables')).toBeInTheDocument();
    expect(screen.getByText('VPP 接入映射 / VPP bus mapping')).toBeInTheDocument();
    expect(screen.getByText(/storage_count/)).toBeInTheDocument();
    expect(screen.getByText(/Campus VPP: PCC 5/)).toBeInTheDocument();
    expect(screen.getAllByText(/storage#7/).length).toBeGreaterThan(0);
    expect(screen.getByText(/pandapower storage p_mw > 0 means charging/)).toBeInTheDocument();
    expect(screen.getByText('VPP 配置说明 / VPP configuration')).toBeInTheDocument();
    expect(screen.getByText('Campus VPP')).toBeInTheDocument();
    expect(screen.getByText('PCC 母线 / PCC bus')).toBeInTheDocument();
    expect(screen.getAllByText('5').length).toBeGreaterThan(0);
    expect(screen.getByText('完整信息 / Full information')).toBeInTheDocument();
    expect(screen.getByText('储能 / Storage')).toBeInTheDocument();
    expect(screen.getByText('Battery A')).toBeInTheDocument();
    expect(screen.getByText(/PCC 母线 5/)).toBeInTheDocument();
    expect(screen.getByText(/Q -0.05..0.05 Mvar/)).toBeInTheDocument();
    expect(screen.getAllByText(/SOC 0.5 \(0.1..0.9\)/).length).toBeGreaterThan(0);
    expect(screen.getByText(/charge 0.15 MW/)).toBeInTheDocument();
    expect(screen.getByText(/discharge 0.16 MW/)).toBeInTheDocument();
    expect(screen.getByText(/pandapower storage #7/)).toBeInTheDocument();
    expect(screen.getByText(/储能 \/ Storage: P -0.2..0.2 MW/)).toBeInTheDocument();
    expect(screen.getByText(/接入母线 \/ connection buses/)).toBeInTheDocument();
    expect(screen.getByText(/pandapower 写入 \/ write target/)).toBeInTheDocument();
    expect(screen.getByText(/feeder_1/)).toBeInTheDocument();
    expect(screen.getByText(/zone_asset_a/)).toBeInTheDocument();
  });
});
