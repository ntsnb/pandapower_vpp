# Roadmap

This roadmap follows the project-level direction in `AGENTS.md`: build a
physically consistent DSO / multi-VPP simulation platform first, then add
feasible regions, local flexibility pricing / bidding, privacy-aware MARL /
HRL interfaces, and long-term capability learning. UI and explainability work
must advance together with algorithmic work so that every new control or market
mechanism can be inspected.

## Current Baseline

Implemented:

- IEEE-33-style feeder and LV taiqu demo network builders.
- DER models for PV, microturbine, storage, flexible load, HVAC, EV, and EVCS.
- Logical DSO and VPP entities mapped to pandapower physical elements.
- Multi-VPP time-series simulation with power-flow, SOC, HVAC temperature, EVCS
  SOC, reward, and violation outputs.
- Static figures, Plotly offline report, and read-only Dash dashboard.
- Minimal Gymnasium-style environment.
- Smoke tests for network build, sign conventions, DER constraints, time-series,
  RL environment, visualization data, and dashboard layout.

Known boundary:

- The current VPP dispatch is rule-based and does not yet implement full
  FR/DOE, LFP/LFB market clearing, settlement, or privacy-audited MARL.
- Dashboard is a read-only analysis console. It does not issue live control
  commands back into the simulator yet.

## Phase 1: Memory And Project Governance

Goal: make the long-running research project auditable and resistant to drift.

Deliverables:

- Create and maintain `memory/concepts.md`, `memory/decisions.md`,
  `memory/rules.md`, `memory/pitfalls.md`, `memory/progress.md`,
  `memory/open_questions.md`, and `memory/experiments.md`.
- Record every architecture decision that changes schema, privacy boundaries,
  physical injection semantics, or simulation workflow.
- Track UI and algorithm work together rather than treating visualization as an
  afterthought.

Validation:

- Documentation exists and is updated after meaningful changes.
- New modules state whether they touch FR/DOE, LFP/LFB, MARL/HRL, privacy, UI,
  or physical power-flow mapping.

## Phase 2: Schema And Physical Consistency

Goal: define explicit data contracts before adding advanced algorithms.

Deliverables:

- `DERSpec`, `VPPPortfolio`, `FRObject`, `LocalFlexNeed`, `VPPFlexBid`,
  `DispatchAward`, `MeasurementReport`, and `ExplanationRecord`.
- Visibility metadata for communication schemas:
  `visible_to_DSO`, `visible_to_VPP_i`, `visible_to_other_VPP`,
  `oracle_only`.
- Tests proving multi-node VPP resources are written to their true pandapower
  bus / element and are not collapsed into a fake PCC injection.

Status:

- Implemented in `src/vpp_dso_sim/entities/schemas.py`.
- Physical injection tests and schema serialization tests are present.
- VPP reports now include `physical_mode` and `connection_buses`.

Suggested files:

- `src/vpp_dso_sim/entities/schemas.py`
- `src/vpp_dso_sim/envs/observations.py`
- `tests/test_schema_serialization.py`
- `tests/test_physical_injection_mapping.py`

UI companion:

- Add schema and visibility inspection tables to the dashboard so users can see
  what the DSO, each VPP, and oracle baselines are allowed to observe.

## Phase 3: FR/DOE v0

Goal: implement operating envelopes as hard security boundaries.

Deliverables:

- Static box feasible region for single-PCC VPPs.
- Bus/zone vector feasible region for multi-node VPPs.
- Action projection that clips DSO / RL / market outputs into device and FR/DOE
  bounds before any pandapower write.

Status:

- `compute_static_feasible_region()` creates v0 box FR/DOE objects from DER
  bounds.
- Multi-node VPPs default to `bus_vector` scope rather than fake PCC aggregation.
- `Simulator.step()` records `fr_envelope_state` and `projection_trace` so the
  command path can be audited in CSV, HTML, and Dash.
- The offline report and Dash dashboard include FR/DOE, projection audit, and
  privacy visibility tables.

Suggested files:

- `src/vpp_dso_sim/optimization/feasibility_region.py`
- `src/vpp_dso_sim/optimization/safety_projection.py`
- `tests/test_feasibility_region.py`

UI companion:

- Show each VPP's current FR/DOE envelope, current operating point, and projected
  command in the offline HTML report and Dash dashboard.

## Phase 4: LFP/LFB v0-v1

Goal: distinguish local flexibility price / bid signals from ordinary energy
price and from wholesale LMP.

Deliverables:

- Exogenous `local_flex_price[zone, t]` profile.
- Stress-driven local flexibility price from voltage, line loading, transformer
  loading, reverse-flow pressure, and safety margin.
- `LocalFlexNeed` records derived from DSO network stress.
- VPP response / bid records that do not expose private device-level costs to
  the DSO by default.

Status:

- `optimization/local_flex_market.py` implements v0 `LocalFlexPrice`,
  stress-driven `LocalFlexNeed` generation, rule-based VPP bids, and simple
  effective-price clearing to `DispatchAward`.
- The implementation explicitly treats local flexibility price as a network
  service signal, not as ordinary energy price or wholesale LMP.

Suggested files:

- `src/vpp_dso_sim/optimization/local_flex_market.py`
- `src/vpp_dso_sim/entities/market.py`
- `tests/test_local_flex_market.py`

UI companion:

- Add local flexibility price / need heatmaps by zone and time.
- Add explanatory panels showing why a VPP was called: voltage support,
  congestion relief, reverse-flow absorption, peak shaving, or resilience.

## Phase 5: VPP Inner Solver v0

Goal: let each VPP maximize its own local objective inside FR/DOE and dispatch
awards while keeping DER constraints valid.

Deliverables:

- Rule-based profit-aware VPP dispatch.
- Optional scipy / cvxpy quadratic dispatch when optional dependencies are
  installed.
- SOC, ramp, comfort, and EV departure feasibility checks.
- Clear fallback behavior when a target is infeasible.

Suggested files:

- `src/vpp_dso_sim/optimization/inner_solver.py`
- `src/vpp_dso_sim/optimization/disaggregation.py`
- `tests/test_vpp_inner_solver.py`

UI companion:

- For each VPP, show target, projected target, delivered response, DER-level
  allocation, cost / revenue proxy, and infeasibility reason if any.

## Phase 6: Multi-VPP Clearing And Settlement

Goal: dispatch multiple VPPs based on price, location effectiveness, reliability,
and safety constraints rather than simply choosing the cheapest bid.

Deliverables:

- VPP bid collection.
- Location effectiveness coefficient / kappa.
- Combination clearing for multiple winning VPPs.
- `DispatchAward`, settlement, deviation, and non-delivery penalty records.

Suggested files:

- `src/vpp_dso_sim/optimization/local_flex_market.py`
- `src/vpp_dso_sim/simulation/simulator.py`
- `tests/test_dispatch_clearing.py`

UI companion:

- Add clearing result tables and stacked contribution plots by service type.
- Add daily VPP dispatch explanations sourced from `DispatchAward` and
  `MeasurementReport`, not static text.

## Phase 7: MARL / HRL Interface

Goal: expose privacy-aware decentralized actor observations and centralized
critic state without leaking oracle information into actors.

Deliverables:

- `actor_observation_i`
- `critic_global_state`
- `action_decoder`
- `reward_components`
- MARL-style dict observation / action skeleton.
- PettingZoo / RLlib adapter once the base contract is stable.

Status:

- `learning/agent_roles.py` defines heterogeneous roles:
  `dso_global_guidance`, per-VPP fast `vpp_dispatch_agent`, per-VPP slow
  `vpp_portfolio_agent`, and `deep_training_supervisor`.
- `learning/encoders.py` adds first-pass `NodeNeedEmbedding`,
  `VppCapabilityEmbedding`, and `VppGridNeedBelief` utilities.
- `learning/marl_baselines.py` runs lightweight IPPO/MAPPO/MADDPG/QMIX-style
  baseline experiments against the current centralized smoke env.
- `learning/tuning.py` provides a training supervisor that runs trials, records
  convergence status, and flags `needs_algorithm_review` on failure.

Suggested files:

- `src/vpp_dso_sim/envs/observations.py`
- `src/vpp_dso_sim/envs/rewards.py`
- `src/vpp_dso_sim/envs/multi_agent_env.py`
- `tests/test_privacy_observation_filter.py`
- `tests/test_multi_agent_env_smoke.py`

UI companion:

- Add an RL diagnostics page with observations, raw actions, projected actions,
  reward components, and privacy-filtered field lists.

## Phase 8: Bidirectional Capability Learning

Goal: learn node / zone need embeddings and VPP capability embeddings from
historical response trajectories.

Deliverables:

- Trajectory buffer containing grid state, FR/DOE, LFP/LFB, bids, awards,
  delivered response, violations, settlement, and reliability metrics.
- Baseline statistical encoders before deep models.
- `NodeNeedEncoder`, `VppCapabilityEncoder`, and `VppGridNeedBeliefEncoder`
  placeholders with versioned outputs.

Suggested files:

- `src/vpp_dso_sim/learning/encoders.py`
- `src/vpp_dso_sim/learning/capability_labeler.py`
- `src/vpp_dso_sim/learning/curriculum.py`
- `tests/test_trajectory_buffer.py`

UI companion:

- Add historical capability cards for each VPP by service type, response delay,
  reliability score, and location effectiveness.

## Phase 9: Slow-Loop Portfolio Search

Goal: model the slow update of DER-to-VPP commercial aggregation without moving
physical DER buses.

Deliverables:

- `portfolio_version`
- `AggregationPlan`
- Heuristic or local-search portfolio update.
- Explicit guarantee that owner / membership may change but bus / pandapower
  physical injection location does not.

Suggested files:

- `src/vpp_dso_sim/optimization/portfolio_search.py`
- `tests/test_portfolio_search.py`

UI companion:

- Add portfolio timeline and before / after comparison views showing how
  commercial ownership changes while physical topology stays fixed.

## Phase 10: Oracle Baselines And Paper Metrics

Goal: compare the distributed bidirectional framework with full-information
centralized baselines without calling the distributed result "optimal" unless
the oracle comparison supports it.

Deliverables:

- Centralized / oracle / full-information baseline.
- Cost, constraint violation, profit, fairness, reliability, and privacy metrics.
- Reproducible experiment configs and summary tables.

Suggested files:

- `src/vpp_dso_sim/optimization/oracle_baseline.py`
- `src/vpp_dso_sim/experiments/`
- `tests/test_oracle_baseline_smoke.py`

UI companion:

- Add experiment comparison pages: distributed vs oracle, privacy mode vs full
  information, rule-based vs learned policy.

## 2026-04-29 Update: Complex Mixed-VPP Scenario And First-Person Replay

Implemented since the previous roadmap pass:

- `src/vpp_dso_sim/network/european_lv.py`: 123-bus European-LV-style feeder for complex台区 experiments.
- `configs/european_lv_mixed_vpp.yaml`: 6 VPP / 28 DER mixed scenario with both `single_pcc` and `multi_node` VPPs.
- `src/vpp_dso_sim/simulation/portfolio_events.py`: deterministic slow portfolio event application without moving DER physical buses or pandapower element rows.
- `Simulator` records `vpp_portfolio_history` and `portfolio_change_log`.
- Dashboard frames now include `vpp_first_person_timeline` and `vpp_first_person_scope_detail`.
- Offline HTML and Dash show VPP first-person Saw / Inferred / Decided replay, scope details, and slow portfolio changes.
- `src/vpp_dso_sim/envs/multi_agent_env.py` now has `MultiAgentVPPDSOEnv`, a minimal parallel dict interface for DSO guidance, VPP dispatch, and VPP portfolio agents.

Validation:

- `python -m pytest -q`: 44 passed.
- `python examples/03_timeseries_multi_vpp.py`: 288 steps, min voltage 0.9441 pu, max line loading 90.64 percent, 0 alert events.
- `python examples/08_run_dashboard.py --check`: dashboard loads 20 simulation frame groups plus training outputs.

Next priorities:

1. Replace lightweight NumPy MARL baselines with the first optional PyTorch IPPO/MAPPO training loop over `MultiAgentVPPDSOEnv`.
2. Add a trajectory buffer with actor observations, critic state, raw/projected actions, FR/DOE, LFP/LFB awards, delivery, settlement, clipping rate, and reward components.
3. Make portfolio actions learnable while retaining the invariant that commercial ownership may change but physical bus and pandapower element mapping do not.
