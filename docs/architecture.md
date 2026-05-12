# Architecture

The simulator separates electrical modeling from operational decision logic.

## Layers

1. `network`
   Builds and evaluates pandapower networks. DSO security checks read from
   `net.res_bus`, `net.res_line`, `net.res_trafo`, and `net.res_ext_grid`.

2. `der`
   Maintains device-level domain state. DER classes own constraints, cost
   coefficients, and pandapower element mappings.

3. `entities`
   Contains logical actors. A `DSO` owns the full network and security limits.
   A `VPPAggregator` owns a collection of DER objects and exposes aggregated
   flexibility rather than raw device details.

4. `optimization`
   Provides replaceable algorithms for flexibility aggregation, target
   disaggregation, and safety projection. v0.1 uses rule-based methods; future
   versions can replace these functions with QP, MILP, OPF, or learning-based
   policies.

5. `simulation`
   Orchestrates the time-series loop, profiles, dynamic state updates, result
   collection, and exports.

6. `envs`
   Provides RL-facing wrappers. The Gymnasium environment currently wraps the
   same simulator used by examples so electrical and RL behavior stay aligned.

## Information Modes

- `full_information`: the DSO can inspect registered VPP DERs. This is useful for
  benchmarks and debugging.
- `representative_data`: the DSO exposes only summarized network data such as
  PCC voltage, limits, and sensitivities. The v0.1 interface is present; richer
  finite-difference sensitivities are staged in `network/sensitivity.py`.

## Time-Step Flow

1. Update base load, PV forecasts, and prices.
2. DSO computes the current network or representative state.
3. Each VPP reports flexibility.
4. DSO or the RL action provides VPP targets.
5. VPPs disaggregate targets to DER set points.
6. Set points are written into pandapower element tables.
7. `runpp` is executed.
8. Dynamic states such as SOC and HVAC temperature are updated.
9. Results, violations, costs, and reward components are recorded.

