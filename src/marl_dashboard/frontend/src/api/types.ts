export type RunSummary = {
  run_id: string;
  status: string;
  started_at: string | null;
  ended_at: string | null;
  algorithm: string | null;
  environment: string | null;
  vpp_count: number | null;
  epoch_count: number | null;
};

export type Selectors = {
  run_id: string;
  dates: string[];
  date_statuses?: DateStatus[];
  vpp_ids: string[];
  agent_ids: string[];
  policy_ids: string[];
  epoch_ids: number[];
  episode_ids: number[];
  time_indices: number[];
};

export type DateStatus = {
  date: string;
  observed_time_slots: number;
  expected_time_slots: number;
  complete: boolean;
  status: string;
};

export type MetricRow = {
  run_id: string;
  epoch_id?: number | null;
  episode_id?: number | null;
  batch_id?: number | string | null;
  gradient_step?: number | null;
  global_env_step?: number | null;
  env_id?: string | null;
  vpp_id?: string | null;
  agent_id?: string | null;
  policy_id?: string | null;
  date?: string | null;
  time_index?: number | null;
  timestamp?: string | null;
  metric_group: string;
  metric_name: string;
  value: number | string | boolean | null;
  unit?: string | null;
  display_name?: string | null;
  formula_latex?: string | null;
  description?: string | null;
  component_ratio?: number | null;
  group?: string | number | null;
};

export type ChartSeries = {
  name: string;
  metric_name?: string;
  vpp_id?: string | null;
  policy_id?: string | null;
  unit?: string;
  points: MetricRow[];
};

export type QueryResponse = {
  chart_series: ChartSeries[];
  table_rows: MetricRow[];
  units: Record<string, string>;
  formulas: Record<string, string>;
  summary: Record<string, number | string | boolean | null>;
};

export type TopologyNode = {
  bus_id: number;
  name?: string | null;
  vn_kv?: number | null;
  is_slack?: boolean | null;
  is_pcc?: boolean | null;
  vpp_ids?: string | null;
  asset_count?: number | null;
  x?: number | null;
  y?: number | null;
};

export type TopologyEdge = {
  edge_id: string;
  edge_type: string;
  from_bus: number;
  to_bus: number;
  name?: string | null;
  voltage_level_label?: string | null;
};

export type TopologyAsset = {
  der_id: string;
  name?: string | null;
  vpp_id: string;
  vpp_name?: string | null;
  bus_id: number;
  der_type: string;
  controllable?: boolean | null;
  p_min_mw?: number | null;
  p_max_mw?: number | null;
};

export type VppPortfolio = {
  vpp_id: string;
  name?: string | null;
  physical_mode?: string | null;
  pcc_bus_id?: number | null;
  connection_buses?: string | null;
  der_ids?: string | null;
  der_count?: number | null;
  max_import_mw?: number | null;
  max_export_mw?: number | null;
};

export type TopologyPayload = {
  run_id: string;
  source_config_path: string;
  network: {
    name?: string | null;
    bus_count: number;
    line_count: number;
    trafo_count: number;
    vpp_count: number;
    horizon_steps?: number | null;
    dt_hours?: number | null;
  };
  pandapower_tables?: Record<string, number>;
  sign_conventions?: Record<string, string>;
  vpp_bus_map?: {
    vpp_id: string;
    display_name: string;
    pcc_bus: number;
    physical_mode: string;
    asset_buses: number[];
    asset_ids: string[];
    pp_elements: string[];
  }[];
  nodes: TopologyNode[];
  edges: TopologyEdge[];
  assets: TopologyAsset[];
  vpp_portfolios: VppPortfolio[];
};

export type VppAssetConfig = {
  der_id: string;
  display_name: string;
  der_type: string;
  der_type_description: string;
  bus_id: number;
  controllable?: boolean | null;
  p_min_mw?: number | null;
  p_max_mw?: number | null;
  q_min_mvar?: number | null;
  q_max_mvar?: number | null;
  capacity_mwh?: number | null;
  soc?: number | null;
  soc_min?: number | null;
  soc_max?: number | null;
  p_charge_max_mw?: number | null;
  p_discharge_max_mw?: number | null;
  n_evs?: number | null;
  rated_power_mw?: number | null;
  baseline_p_mw?: number | null;
  apparent_power_mva?: number | null;
  indoor_temp?: number | null;
  temp_min?: number | null;
  temp_max?: number | null;
  pp_element_type?: string | null;
  pp_element_index?: number | null;
  write_target?: string | null;
  configuration_summary?: string | null;
  zone_id?: string | null;
  feeder_id?: string | null;
};

export type VppConfig = {
  vpp_id: string;
  display_name: string;
  pcc_bus: number;
  physical_mode: string;
  physical_mode_description: string;
  privacy_mode: string;
  privacy_mode_description: string;
  portfolio_version?: string | null;
  connection_buses: number[];
  zone_ids: string[];
  der_count: number;
  asset_counts: Record<string, number>;
  p_min_mw?: number | null;
  p_max_mw?: number | null;
  q_min_mvar?: number | null;
  q_max_mvar?: number | null;
  max_import_mw?: number | null;
  max_export_mw?: number | null;
  dispatch_capability?: {
    active_power_range_mw?: [number, number] | number[];
    reactive_power_range_mvar?: [number, number] | number[];
    max_import_mw?: number | null;
    max_export_mw?: number | null;
    sign_convention?: string | null;
  };
  description: string;
  configuration_notes?: string[];
  assets: VppAssetConfig[];
};

export type VppConfigPayload = {
  run_id: string;
  source_config_path: string;
  summary: {
    vpp_count: number;
    asset_count: number;
    horizon_steps?: number | null;
    dt_hours?: number | null;
  };
  vpps: VppConfig[];
};

export type RunMetadata = {
  run_id: string;
  status?: string;
  started_at?: string | null;
  ended_at?: string | null;
  algorithm?: string | null;
  environment?: string | null;
  config?: Record<string, unknown>;
  notes?: string | null;
  [key: string]: unknown;
};

export type VariableDefinition = {
  name: string;
  display_name?: string | null;
  symbol?: string | null;
  unit?: string | null;
  group?: string | null;
  physical_meaning?: string | null;
  formula_latex?: string | null;
  source?: string | null;
  notes?: string | null;
};

export type Filters = {
  runId: string;
  date?: string;
  vppId?: string;
  agentId?: string;
  epochId?: number;
  episodeId?: number;
  timeIndex?: number;
  startTimeIndex?: number;
  endTimeIndex?: number;
  policyId?: string;
  live: boolean;
  compareMode: boolean;
};

export type LiveEvent = {
  run_id?: string;
  table?: string;
  metric_group?: string;
  metric_name?: string;
  event_type?: string;
  value?: number | string | boolean | null;
  timestamp?: string | null;
  logged_at?: string | null;
  [key: string]: unknown;
};

export type WebSocketStatus = 'disabled' | 'connecting' | 'connected' | 'disconnected' | 'error';
