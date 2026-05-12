# Open Questions

## FR/DOE Representation

Should the first multi-node feasible region use bus-level vectors, zone-level
vectors, or both with a reduction layer?

## Local Flexibility Zones

What is the first zoning rule for LFP/LFB: feeder depth, electrical distance from
the substation, sensitivity clusters, or user-specified VPP zones?

## Settlement Model

Should v0 settlement use paid-as-bid, uniform clearing price, or a simple
network-service reward proxy before a full market mechanism is introduced?

## Dashboard Control Boundary

When the dashboard becomes interactive, should it issue only scenario / replay
commands first, or also allow manual DSO target commands through a safety
projection layer?

## Oracle Baseline Scope

Should the first oracle baseline optimize only active power, or include reactive
power and voltage constraints from the start?

## Dashboard Performance

When horizon length grows beyond the current 72-hour default, should topology
figures cache per-step slices, precomputed trace arrays, or server-side filtered
frames to avoid rebuilding the full Plotly topology figure on every slider move?

## Experiment Level Metadata

Should every experiment output include explicit fields such as
`experiment_level`, `seed_count`, `holdout_scenario_count`, `oracle_baseline`,
`reviewed_by_architecture_subagent`, `reviewed_by_experiment_subagent`, and
`ui_refresh_cmds`?

## Project Agent Installation

Should the project-local `agents/ppvpp-*.toml` drafts be copied into
`C:\Users\admin\.codex\agents` for direct `agent_type` routing, or remain as
project-local read-before-routing files?

## Memory Update Automation

Should `examples/10_train_deep_rl.py` or a future experiment runner emit a
machine-readable memory update suggestion after each benchmark run?
