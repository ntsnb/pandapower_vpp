from __future__ import annotations

from pathlib import Path

from vpp_dso_sim.simulation.scenario import load_scenario
from vpp_dso_sim.simulation.simulator import Simulator
from vpp_dso_sim.learning.tuning import TrainingSupervisor, TuningConfig
from vpp_dso_sim.visualization.dashboard_data import build_dashboard_frames
from vpp_dso_sim.visualization.first_person_report import build_first_person_reports
from vpp_dso_sim.visualization.interactive_report import build_interactive_report_html
from vpp_dso_sim.visualization.plotly_figures import edge_flow_figure, require_plotly
from vpp_dso_sim.visualization.rl_architecture_report import build_rl_architecture_report_html
from vpp_dso_sim.visualization.topology_plots import plot_topology_report


def test_dashboard_frames_have_topology_and_state_fields():
    scenario = load_scenario()
    simulator = Simulator(scenario)
    results = simulator.run_timeseries(horizon_steps=3)
    frames = build_dashboard_frames(scenario.net, scenario.vpps, results, dt_hours=scenario.dt_hours)

    assert {"bus_id", "x", "y", "is_pcc"}.issubset(frames["network_nodes"].columns)
    assert {"edge_id", "edge_type", "from_bus", "to_bus", "voltage_level_label"}.issubset(
        frames["network_edges"].columns
    )
    assert {"der_id", "vpp_id", "bus_id", "der_type"}.issubset(frames["asset_registry"].columns)
    assert {"vpp_id", "physical_mode", "connection_buses", "der_ids"}.issubset(
        frames["vpp_portfolio"].columns
    )
    assert {"step", "vpp_id", "portfolio_version", "physical_mode", "connection_buses", "der_ids"}.issubset(
        frames["vpp_portfolio_history"].columns
    )
    assert {"event_id", "effective_step", "der_id", "from_vpp_id", "to_vpp_id"}.issubset(
        frames["portfolio_change_log"].columns
    )
    assert {"vpp_id", "p_min_mw", "p_max_mw", "bid_price_up", "bid_price_down"}.issubset(
        frames["vpp_day_ahead_bid"].columns
    )
    assert {"vpp_id", "preferred_target_p_mw", "service_request", "dso_intent"}.issubset(
        frames["dso_operating_envelope"].columns
    )
    assert {"vpp_id", "der_id", "is_learned_der_action", "projection_gap_mw"}.issubset(
        frames["vpp_rl_disaggregation"].columns
    )
    assert {"fr_id", "vpp_id", "time_index", "scope", "element_id", "p_min_mw", "p_max_mw"}.issubset(
        frames["feasible_region"].columns
    )
    assert {"fr_id", "step", "vpp_id", "scope_type", "scope_id", "variable", "lower_bound", "upper_bound"}.issubset(
        frames["fr_envelope_state"].columns
    )
    assert {"trace_id", "step", "vpp_id", "stage_name", "p_mw", "projection_reason"}.issubset(
        frames["projection_trace"].columns
    )
    assert {"schema", "field", "visible_to_dso", "visible_to_vpp_i", "oracle_only"}.issubset(
        frames["privacy_visibility"].columns
    )
    assert {"step", "bus_id", "vm_pu"}.issubset(frames["bus_state"].columns)
    assert {"step", "edge_id", "loading_percent"}.issubset(frames["edge_state"].columns)
    assert {"p_from_mw", "q_from_mvar", "flow_label", "flow_direction"}.issubset(
        frames["edge_state"].columns
    )
    assert {"available_p_mw", "p_min_mw", "p_max_mw", "state_label"}.issubset(
        frames["der_state"].columns
    )
    assert {
        "vpp_id",
        "reason",
        "instruction",
        "asset_response",
        "reason_zh",
        "instruction_zh",
        "asset_response_zh",
        "reason_en",
        "instruction_en",
        "asset_response_en",
    }.issubset(
        frames["vpp_dispatch_explanation"].columns
    )
    assert {
        "vpp_id",
        "phase",
        "seen_fr_bounds_json",
        "inferred_grid_need_label",
        "decision_summary",
        "private_cost_used",
    }.issubset(frames["vpp_first_person_timeline"].columns)
    assert {
        "vpp_id",
        "scope_type",
        "scope_id",
        "asset_ids",
        "p_lower_mw",
        "p_upper_mw",
    }.issubset(frames["vpp_first_person_scope_detail"].columns)
    assert {
        "step",
        "time_label",
        "vpp_id",
        "command_seen",
        "belief_label",
        "action_label",
        "projected_p_mw",
        "actual_p_mw",
        "decision_status",
    }.issubset(frames["vpp_step_decision_summary"].columns)
    assert {
        "step",
        "vpp_id",
        "event_order",
        "event_type",
        "event_detail",
        "event_detail_zh",
        "event_detail_raw",
        "private_cost_used",
    }.issubset(frames["vpp_first_person_event_stream"].columns)
    assert {
        "vpp_id",
        "dominant_grid_need",
        "reliability_score",
        "portfolio_recommendation",
    }.issubset(frames["vpp_long_cycle_judgment"].columns)
    assert {"metric", "value", "formula", "why_negative"}.issubset(frames["economic_explanation"].columns)
    assert {"update_area", "current_value", "explanation", "evidence_file"}.issubset(
        frames["model_update_summary"].columns
    )
    assert "algorithm" in set(frames["model_update_summary"]["update_area"])
    assert {"algorithm_id", "ctde_status", "loss_formula"}.issubset(frames["rl_algorithm_overview"].columns)
    assert {
        "algorithm_id",
        "algorithm_label",
        "family",
        "actor_style",
        "critic_style",
        "update_core",
        "experience_reuse",
        "architecture_signature",
        "repo_status",
    }.issubset(frames["rl_algorithm_variants"].columns)
    assert {"agent_group", "label", "color"}.issubset(frames["rl_agent_groups"].columns)
    assert {
        "agent_id",
        "input_observation",
        "action_output",
        "is_rl_decision",
        "rl_usage_status",
        "neural_network_structure",
        "result_formula",
        "result_calculation",
        "result_source",
        "rl_training_signal",
        "audit_outputs",
        "non_rl_guardrails",
        "implementation_status",
    }.issubset(
        frames["rl_agent_architecture"].columns
    )
    assert {
        "component_id",
        "component_group",
        "input_shape",
        "output_shape",
        "structure",
        "distribution",
        "calculation_note",
    }.issubset(frames["rl_neural_network_architecture"].columns)
    assert {
        "component_id",
        "component_group",
        "privacy_scope",
        "execution_visibility",
        "loss_signal",
        "conference_role",
    }.issubset(frames["rl_target_ctde_architecture"].columns)
    assert {
        "component_id",
        "tensor_in",
        "tensor_out",
        "parameter_sharing_scope",
        "limitation",
        "next_upgrade",
    }.issubset(frames["rl_ctde_nodes"].columns)
    assert {
        "edge_id",
        "src_component_id",
        "dst_component_id",
        "signal_name",
        "signal_type",
        "privacy_class",
        "carries_gradient",
    }.issubset(frames["rl_ctde_edges"].columns)
    assert {"feedback_id", "target_component_id", "loss_name", "formula", "advantage_source"}.issubset(
        frames["rl_ctde_feedback"].columns
    )
    assert {
        "dso_private_observation_encoder",
        "vpp_local_observation_encoder",
        "vpp_portfolio_slow_encoder",
        "centralized_training_critic",
        "non_rl_safety_projection",
    }.issubset(set(frames["rl_target_ctde_architecture"]["component_id"]))
    assert {"happo", "matd3", "hasac"}.issubset(set(frames["rl_algorithm_variants"]["algorithm_id"]))
    assert {"flow_order", "source", "target", "signal"}.issubset(frames["rl_data_flow"].columns)
    assert {"relation_order", "source", "target", "message"}.issubset(frames["rl_agent_relationships"].columns)
    assert {"step_order", "stage", "actor", "input", "output"}.issubset(frames["rl_step_workflow"].columns)
    assert {"reward_id", "formula", "terms", "current_status"}.issubset(frames["rl_reward_design"].columns)
    assert {"component", "formula", "coefficient", "meaning"}.issubset(frames["rl_loss_components"].columns)
    assert {"question", "answer", "evidence"}.issubset(frames["rl_ctde_assessment"].columns)
    assert {"gap_id", "current_answer", "target_answer"}.issubset(frames["rl_implementation_gaps"].columns)
    assert len(frames["bus_state"]) == 3 * len(scenario.net.bus)
    assert set(frames["vpp_portfolio"]["physical_mode"]) == {"multi_node"}
    assert len(frames["vpp_step_decision_summary"]) == 3 * len(scenario.vpps)
    assert len(frames["vpp_first_person_event_stream"]) >= 4 * len(frames["vpp_step_decision_summary"])
    assert len(frames["feasible_region"]) >= 3 * len(scenario.vpps)
    assert {"raw_action", "device_bounds", "fr_doe", "pandapower_write", "powerflow_result"}.issubset(
        set(frames["projection_trace"]["stage_name"])
    )
    assert set(frames["fr_envelope_state"]["scope_type"]) >= {"bus"}


def test_topology_report_writes_expected_figures():
    scenario = load_scenario()
    simulator = Simulator(scenario)
    results = simulator.run_timeseries(horizon_steps=2)
    frames = build_dashboard_frames(scenario.net, scenario.vpps, results, dt_hours=scenario.dt_hours)
    output_dir = Path("outputs") / "test_visualization"
    paths = plot_topology_report(frames, output_dir)

    assert {"topology_step_000", "topology_step_peak_loading", "topology_voltage_min_step", "alert_timeline"}.issubset(paths)
    for path in paths.values():
        assert path.exists()
        assert path.stat().st_size > 0


def test_interactive_report_writes_html():
    scenario = load_scenario()
    simulator = Simulator(scenario)
    results = simulator.run_timeseries(horizon_steps=2)
    frames = build_dashboard_frames(scenario.net, scenario.vpps, results, dt_hours=scenario.dt_hours)
    output_path = Path("outputs") / "test_visualization" / "interactive_report_test.html"
    TrainingSupervisor(
        TuningConfig(
            algorithms=("ippo",),
            action_scales=(0.05,),
            exploration_noises=(0.0,),
            horizon_steps=2,
            episodes_per_trial=1,
        )
    ).run(output_dir=output_path.parent / "marl_baselines")

    path = build_interactive_report_html(frames, output_path)

    assert path.exists()
    text = path.read_text(encoding="utf-8")
    assert "pandapower VPP DSO Simulation Report" in text
    assert "Plotly.newPlot" in text
    assert "solar panel" in text
    assert "charging station" in text
    assert "Busbar voltage" in text
    assert "Feeder switch" in text
    assert "protection symbols" in text
    assert "Feeder voltage labels" in text
    assert "One-Day VPP Dispatch Instructions" in text
    assert "FR/DOE Envelope" in text
    assert "Projection Audit" in text
    assert "Privacy Visibility" in text
    assert "Training Supervisor Status" in text
    assert "Deep RL Actor-Critic" in text
    assert "Model / Algorithm Update Summary" in text
    assert "model_update_summary.csv" in text
    assert "Current Learning Model Map" in text
    assert "rl_architecture.html" in text
    assert "implemented_privacy_separated_ctde_training_loop" in text
    assert "HAPPO / MATD3 / HASAC Architecture Contrast" in text
    assert "Sequential policy update with importance correction across agents" in text
    assert "DSO twin Q plus per-VPP twin Q heads" in text
    assert "Entropy temperature tuning plus soft Bellman backup" in text
    assert "data-rl-filter=\"loss\"" in text
    assert "Paper-Style RL / MARL Control Loop" in text
    assert "VPP day-ahead bid" in text
    assert "DSO operating envelope" in text
    assert "Safety projection" in text
    assert "Current Recommended Hierarchical HAPPO / Privacy-Separated CTDE Neural Network Architecture" in text
    assert 'id="target-ctde-svg"' in text
    assert 'data-testid="ctde-layer-graph-svg"' in text
    assert 'id="privacy-boundary"' in text
    assert 'id="dso-observation-encoder"' in text
    assert 'id="vpp-local-observation-encoder"' in text
    assert 'id="centralized-critic"' in text
    assert 'id="ctde-loss-feedback"' in text
    assert "Reward / critic / training update" in text
    assert "Projection chain" in text
    assert "Supervisor scope" in text
    assert "Decision provenance" in text
    assert "Uses RL?" in text
    assert "Neural Network Structure" in text
    assert "nn-architecture-svg" in text
    assert 'data-testid="ctde-dso-mean-head"' in text
    assert 'data-testid="ctde-vpp-der-head"' in text
    assert 'data-testid="ctde-critic-value-head"' in text
    assert 'data-testid="ctde-loss-node"' in text
    assert "CTDE Graph Nodes" in text
    assert "CTDE Graph Edges" in text
    assert "CTDE Feedback" in text
    assert "Linear(5+7*" in text
    assert "Result calculation" in text
    assert "Audit outputs" in text
    assert "vpp_rl_disaggregation.csv" in text
    assert "uses_rl_privacy_scoped_vpp_dispatch_actor" in text
    assert "experiment orchestrator, not env-step MARL actor or LLM policy" in text
    assert 'data-agent-id="dso_global_guidance"' in text
    assert "dso_policy_loss" in text
    assert "vpp_dispatch_policy_loss" in text
    assert "Why Reward / Profit Proxy Can Be Negative" in text
    assert "VPP First-Person Replay" in text
    assert "Saw" in text
    assert "Inferred" in text
    assert "Decided" in text
    assert "data-first-person-vpp" in text
    assert "MARL Baselines" in text
    assert "Tuning Trials" in text
    assert "raw_action" in text
    assert "pandapower_write" in text
    assert 'data-lang-switch="zh"' in text
    assert "lang-copy" in text
    assert "lang-zh" in text
    assert "dispatch-vpp-card" in text
    assert "VPP Group" in text
    assert "Reason" in text
    assert "Instruction" in text
    assert "What Drives VPP Decisions" in text
    assert "VPP 决策由什么驱动" in text
    assert "trace_names" in text
    assert "trace_colorbars" in text
    assert "trace_hovertemplates" in text
    assert "trace_hovertexts" in text
    assert "母线电压" in text


def test_rl_architecture_report_writes_standalone_html():
    scenario = load_scenario()
    simulator = Simulator(scenario)
    results = simulator.run_timeseries(horizon_steps=2)
    frames = build_dashboard_frames(scenario.net, scenario.vpps, results, dt_hours=scenario.dt_hours)
    output_path = Path("outputs") / "test_visualization" / "rl_architecture_test.html"

    path = build_rl_architecture_report_html(frames, output_path)

    assert path.exists()
    text = path.read_text(encoding="utf-8")
    assert "强化学习智能体交互框架报告" in text
    assert "Agent Groups and Colors" in text
    assert "Model / Algorithm Update Summary" in text
    assert "model_update_summary.csv" in text
    assert "Paper-Style Agent Workflow" in text
    assert "VPP day-ahead bid" in text
    assert "DSO operating envelope" in text
    assert "VPP dispatch actors" in text
    assert "一个 step 内智能体如何工作" in text
    assert "Safety projection" in text
    assert "Current Recommended Hierarchical HAPPO / Privacy-Separated CTDE Neural Network Architecture" in text
    assert 'id="target-ctde-svg"' in text
    assert 'data-testid="ctde-layer-graph-svg"' in text
    assert 'id="dso-observation-encoder"' in text
    assert 'id="vpp-local-observation-encoder"' in text
    assert 'id="vpp-dispatch-actor"' in text
    assert 'id="vpp-portfolio-actor"' in text
    assert 'id="centralized-critic"' in text
    assert 'id="ctde-safety-projection"' in text
    assert "Reward / critic / training update" in text
    assert "Projection chain" in text
    assert "Supervisor scope" in text
    assert "Decision provenance" in text
    assert "Uses RL?" in text
    assert "Neural network structure" in text
    assert "nn-architecture-svg" in text
    assert 'data-testid="ctde-dso-mean-head"' in text
    assert 'data-testid="ctde-vpp-aggregate-head"' in text
    assert 'data-testid="ctde-vpp-der-head"' in text
    assert 'data-testid="ctde-portfolio-logits-head"' in text
    assert 'data-testid="ctde-critic-value-head"' in text
    assert "vpp_der_dispatch_actor" in text
    assert "rl_ctde_nodes" in text
    assert "rl_ctde_edges" in text
    assert "rl_ctde_feedback" in text
    assert "Result calculation" in text
    assert "Audit outputs" in text
    assert "deep_rl_trajectory.csv" in text
    assert "not_an_rl_environment_agent" in text
    assert "safety_projection_non_agent_guard" in text
    assert "data-agent-id=\"dso_global_guidance\"" in text
    assert "reward_function_zh" in text
    assert "vpp_dispatch_local_reward_target" in text
    assert "implemented privacy-separated CTDE training loop" in text
    assert "dso_private_observation_encoder" in text
    assert "centralized_training_critic" in text


def test_split_first_person_report_writes_html_files():
    scenario = load_scenario()
    simulator = Simulator(scenario)
    results = simulator.run_timeseries(horizon_steps=3)
    frames = build_dashboard_frames(scenario.net, scenario.vpps, results, dt_hours=scenario.dt_hours)
    output_dir = Path("outputs") / "test_visualization" / "vpp_first_person"

    paths = build_first_person_reports(frames, output_dir)

    assert paths["index"].exists()
    assert paths["long_cycle"].exists()
    assert paths["economic_explanation"].exists()
    first_vpp = scenario.vpps[0].id
    assert paths[first_vpp].exists()
    text = paths[first_vpp].read_text(encoding="utf-8")
    assert "逐时刻事件链" in text
    assert "收到什么" in text
    assert "判断什么" in text
    assert "做了什么" in text
    assert "结果如何" in text
    assert "data-lang-switch=\"zh\"" in text
    assert "lang-zh" in text
    assert "What it received" in text
    assert "Model / Algorithm Update Summary" in text
    assert "model_update_summary" in text
    assert "How to read the decision fields" in text
    assert "Raw audit fields" in text
    assert "raw-audit" in text
    assert "FR/DOE" in text
    assert "P &lt; 0" in text
    assert "data-step" in text


def test_edge_flow_figure_has_non_overlapping_subplot_layout():
    scenario = load_scenario()
    simulator = Simulator(scenario)
    results = simulator.run_timeseries(horizon_steps=2)
    frames = build_dashboard_frames(scenario.net, scenario.vpps, results, dt_hours=scenario.dt_hours)
    go, _ = require_plotly()

    fig = edge_flow_figure(go, frames)

    assert fig.layout.height >= 1000
    assert fig.layout.xaxis.title.text == ""
    assert fig.layout.xaxis2.title.text == "time (h)"
    assert fig.layout.xaxis3.title.text == "peak loading (%)"
    assert fig.layout.yaxis.domain[0] - fig.layout.yaxis2.domain[1] >= 0.10
    assert fig.layout.yaxis2.domain[0] - fig.layout.yaxis3.domain[1] >= 0.10
