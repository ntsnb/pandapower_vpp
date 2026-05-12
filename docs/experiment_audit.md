# Experiment Audit

## 2026-05-02 Update

The previous P0 issue that `MultiAgentVPPDSOEnv` returned the same global reward to every agent has been addressed in the current code path. The environment now returns role-specific general-sum rewards:

- DSO: grid-safety/procurement/tracking reward.
- VPP dispatch: self-interested settlement, delivery, DER-cost, comfort/SOC and projection penalties.
- VPP portfolio: long-horizon profit proxy, reliability, switching cost and localized DSO-alignment credit.

The remaining publication-grade work is to calibrate these economic proxy coefficients against a real market-settlement design, add stronger multi-day portfolio training, and compare against a strict AC OPF or MILP reference before making optimality or upper-bound claims.

Date: 2026-05-01

Scope:
- Read-only audit of `pandapower-vpp-dso-sim`
- One allowed documentation addition only
- Limited external benchmark/data-source cross-check for recommended next datasets

## Bottom Line

The repository is credible as an engineering demo and architecture scaffold, but not yet as a publication-grade experimental stack. The main gap is not one bug; it is that the default scenario, data, reward/economics, and evaluation harness are still mostly smoke/demo oriented.

## High-Priority Findings

### P0. Default scenario and profiles are still toy-level

Evidence:
- All three shipped scenarios reuse the same three CSV profiles: `configs/ieee33_multi_vpp.yaml:13-15`, `configs/european_lv_mixed_vpp.yaml:13-15`, `configs/lv_taiqu_demo.yaml:13-15`.
- Those profile CSVs are only 96 rows each and are repeated to fill 288 steps by `load_profile_csv()`: `src/vpp_dso_sim/simulation/profiles.py:19-22`.
- The project explicitly states the IEEE-33 case is simplified and not publication-grade: `docs/modeling_assumptions.md:7-8`.
- The default 123-bus LV feeder is also explicitly a demo, not a verbatim benchmark: `src/vpp_dso_sim/network/european_lv.py:14`, `src/vpp_dso_sim/network/european_lv.py:35-38`.
- Base LV loads are deterministic formulas, not dataset-driven traces: `src/vpp_dso_sim/network/european_lv.py:18-21`.
- EV arrivals/departures/SOC are synthetic templates: `src/vpp_dso_sim/der/evcs.py:23-31`.
- HVAC outdoor temperature and setpoint profiles are synthetic and constant-shape: `src/vpp_dso_sim/simulation/scenario.py:134-145`.
- DER costs are mostly hard-coded proxies rather than calibrated asset economics: `src/vpp_dso_sim/simulation/scenario.py:74`, `src/vpp_dso_sim/simulation/scenario.py:90`, `src/vpp_dso_sim/simulation/scenario.py:112`, `src/vpp_dso_sim/simulation/scenario.py:125`, `src/vpp_dso_sim/simulation/scenario.py:140`, `src/vpp_dso_sim/simulation/scenario.py:156`.

Why this matters:
- Repeating one synthetic day three times makes policy learning vulnerable to overfitting trivial periodic structure.
- Small, hand-tuned DER sizes and deterministic behaviors make constraint patterns easier than realistic feeder stress.
- A paper built on these defaults will be vulnerable to the criticism that the environment is "architecturally rich but statistically weak."

Action:
- Keep the current configs as demo/smoke assets only.
- Add at least one benchmark-grade feeder config and one data-driven profile pack before claiming experimental performance.

### P0. Training and evaluation are still smoke tests, not academic experiments

Evidence:
- Default deep RL example is only `--episodes 3` and `--horizon-steps 8`: `examples/10_train_deep_rl.py:34-35`.
- Default MARL baseline example is only 8 steps x 2 episodes; tuning sweep is 1 episode per trial: `examples/09_run_marl_baselines.py:14-15`, `examples/09_run_marl_baselines.py:26`.
- Deep RL tests use 1 episode and 2 steps: `tests/test_deep_rl_training.py:23`, `tests/test_deep_rl_training.py:76`.
- MARL baseline tests use 1 episode and 2 steps: `tests/test_marl_baselines.py:28-29`, `tests/test_marl_baselines.py:48-49`, `tests/test_marl_baselines.py:70-71`.
- The project memory already labels the baseline harness as smoke and says current baselines are only lightweight approximations: `memory/experiments.md:85-88`.
- The roadmap says the current dispatch is still rule-based and does not yet implement full FR/DOE, market clearing, settlement, or privacy-audited MARL: `docs/roadmap.md:23-27`.

Missing standard practice:
- Multi-seed statistics
- Explicit train/eval split
- Holdout scenarios or holdout weeks
- Oracle/full-information baseline
- Ablations
- Convergence curves with variance bands
- Robustness tests under forecast error, outage, and parameter drift

Action:
- Treat all current RL/MARL outputs as harness validation only.
- Add a reproducible experiment runner that reports mean/std/min/max across seeds and holdout scenarios.

### P0. Reward and economics are not aligned with VPP market profit claims

Evidence:
- The implemented reward is still a shaped control objective, `reward = -0.05 * total_cost + feasibility_bonus + tracking_bonus`, but `total_cost` now includes `action_projection_penalty` so infeasible raw actions are not hidden by the safety layer: `src/vpp_dso_sim/entities/dso.py`.
- `MultiAgentVPPDSOEnv` gives all agents the same global reward, with only a tiny portfolio regularizer: `src/vpp_dso_sim/envs/multi_agent_env.py:323-331`.
- The environment's `dispatch_award` uses the envelope price as a placeholder settlement price, not a real cleared market settlement: `src/vpp_dso_sim/envs/multi_agent_env.py:216-219`.
- The architecture notes explicitly say local settlement-aware reward is still future work: `src/vpp_dso_sim/learning/rl_architecture.py:188`, `src/vpp_dso_sim/learning/rl_architecture.py:1276`, `src/vpp_dso_sim/learning/rl_architecture.py:1634-1636`.

Why this matters:
- A global shaped control reward is acceptable for early control engineering, but it is not enough for papers claiming VPP self-interested bidding, profit maximization, or market-compatible multi-agent behavior.
- Shared global reward can hide credit-assignment failure and can overstate the meaning of "cooperation."

Action:
- Separate `J_DSO`, `Profit_i`, and settlement/deviation penalties in logging first.
- Only after that, train actor losses against the correct per-agent objective.

### P1. FR/DOE and local-flex market are still placeholder mechanisms

Evidence:
- FR/DOE is explicitly static local DER bounds, not OPF-certified: `src/vpp_dso_sim/optimization/feasibility_region.py:35-44`, `src/vpp_dso_sim/optimization/feasibility_region.py:94`.
- The simulator builds the DSO operating envelope directly from bid bounds and price thresholds; it does not use feeder sensitivities or a network-constrained market-clearing step: `src/vpp_dso_sim/simulation/simulator.py:137`, `src/vpp_dso_sim/simulation/simulator.py:224-235`.
- `local_flex_market.py` implements pure-function v0 rules, but the main integration path is still the simulator envelope logic; the tests cover only isolated market functions: `src/vpp_dso_sim/optimization/local_flex_market.py:166-198`, `tests/test_local_flex_market.py:11-47`.

Why this matters:
- For publication claims about bidirectional guidance, the current stack still lacks a convincing end-to-end chain: stress -> need -> bids -> clearing -> dispatch award -> delivery -> settlement -> capability update.

Action:
- Add one integrated experiment path where `LocalFlexNeed`, `VPPFlexBid`, `DispatchAward`, delivery, and non-delivery are all recorded in the same simulator rollout.

### P1. Core evaluation metrics are too narrow for a VPP/DSO paper

Evidence:
- The cost function includes operation cost, tracking, comfort, SOC, voltage, line, transformer, and power-flow penalties: `src/vpp_dso_sim/entities/dso.py:85-98`, `src/vpp_dso_sim/network/constraints.py:109-123`.
- It does not yet include explicit loss cost, curtailment cost, load shedding, fairness, network-service procurement cost, market power, or reliability settlement terms in the main objective path.
- The roadmap lists these as future additions rather than current experiment outputs: `docs/roadmap.md:171-183`, `docs/roadmap.md:293`.

Action:
- Expand the experiment table to report at least:
  - voltage violation count and magnitude
  - line/trafo overload count and magnitude
  - total curtailed PV
  - storage throughput and SOC violation hours
  - per-VPP energy revenue, flexibility revenue, operating cost, penalty, net profit proxy
  - DSO procurement cost
  - fairness across VPP dispatch frequency and profit

## Missing Data and Benchmark Sets

Current repo data gap:
- `data/profiles/` contains only `load_profile.csv`, `pv_profile.csv`, and `price_profile.csv`.
- There is no shipped benchmark pack for:
  - unbalanced feeder studies
  - feeder-level yearly profiles
  - EV charging sessions
  - HVAC/building thermal demand
  - realistic tariff/wholesale price traces
  - outage/fault/stress scenario libraries

Recommended additions:

### Network benchmarks

- IEEE 123-bus test feeder for more realistic regulator/capacitor/multi-phase behavior.
- CIGRE benchmark distribution systems for European-style LV/MV studies.
- SimBench for public benchmark grids with yearly time series and reproducible scenarios.

### Time-series and DER data

- Building/HVAC/load: NREL End-Use Load Profiles.
- Residential load/PV/storage/EV: Pecan Street Dataport.
- EV charging sessions: ACN-Data.

### Price data

- Use real day-ahead and real-time wholesale traces from PJM or CAISO as the energy-price base.
- Then add a separate distribution-service adder or local-flex price layer instead of using one toy price series for both roles.

### Stress and robustness scenarios

- N-1 line/transformer outage cases
- regulator or capacitor unavailability
- forecast error perturbations for load/PV/EV
- communication delay / non-delivery cases
- DER availability loss and temporary asset derating

## Recommended Roadmap

### Immediately doable

- Add `configs/benchmark_*` configs distinct from demo configs.
- Add profile packs with at least 7-30 non-identical daily traces instead of one 96-point repeated day.
- Add an experiment runner that:
  - runs 5+ seeds
  - separates train and eval horizons
  - writes mean/std summary CSVs
  - saves per-seed violation and economic metrics
- Add a clear experiment taxonomy:
  - `smoke`
  - `demo`
  - `benchmark`
  - `paper`

### Mandatory before paper claims

- Add one public benchmark feeder and one realistic LV scenario.
- Add one oracle/full-information baseline; the roadmap already reserves `src/vpp_dso_sim/optimization/oracle_baseline.py` and `tests/test_oracle_baseline_smoke.py`: `docs/roadmap.md:298-300`.
- Replace "lightweight runnable approximations" with real trainable baselines for the algorithms used in comparison.
- Report:
  - mean/std across seeds
  - train vs holdout performance
  - ablation of FR/DOE, market signal, privacy filter, and learned dispatch
  - robustness under forecast error and outage stress

### Longer-term research upgrades

- Move from balanced equivalent to unbalanced three-phase benchmark studies.
- Upgrade FR from local box bounds to sensitivity-based or OPF-certified envelopes.
- Add true settlement, deviation, and non-delivery accounting.
- Add slow-loop portfolio experiments over multi-day or multi-week horizons.
- Add capability-learning evaluation: does `NodeNeedEmbedding` or `VppCapabilityEmbedding` improve system outcomes on holdout scenarios?

## Suggested Acceptance Standard

Use the following language boundary in papers and internal reports:

- Current state: "engineering demo", "smoke-validated simulator", "prototype benchmark harness".
- Not yet justified: "optimal", "market-realistic", "publication-grade benchmark", "economically faithful VPP profit experiment".

## External References For Next Dataset/Benchmark Round

- IEEE PES test feeders: https://cmte.ieee.org/pes-testfeeders/comprehensive-test-feeder/
- SimBench overview: https://simbench.de/en/
- SimBench documentation: https://simbench.readthedocs.io/en/stable/about.html
- CIGRE benchmark systems: https://www.e-cigre.org/publications/detail/elt-273-8-benchmark-systems-for-network-integration-of-renewable-and-distributed-energy-resources.html
- NREL End-Use Load Profiles: https://www.nrel.gov/buildings/end-use-load-profiles.html
- Pecan Street Dataport: https://www.pecanstreet.org/dataport/about-dataport/
- ACN-Data: https://ev.caltech.edu/dataset
- CAISO OASIS: https://www.caiso.com/systems-applications/portals-applications/open-access-same-time-information-system-oasis
- PJM energy market / LMP context: https://www.pjm.com/markets-and-operations/energy/ and https://learn.pjm.com/three-priorities/buying-and-selling-energy/lmp.aspx
