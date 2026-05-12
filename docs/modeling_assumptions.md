# Modeling Assumptions

## Electrical Network

- v0.1 uses balanced AC power flow with `pandapower.runpp`.
- Radial feeders default to the BFSW algorithm with a Newton-Raphson fallback.
- IEEE 33 feeder data is simplified and intended for software validation, not
  publication-grade benchmark studies.
- Low-voltage transformer-area modeling is single-phase equivalent. Three-phase
  unbalance and neutral constraints are future work.

## VPP and DSO Coordination

- VPPs are logical aggregators, not pandapower elements.
- DER privacy is represented through mode flags. Only full-information behavior is
  complete in v0.1; representative-data interfaces are present for future work.
- DSO targets are interpreted as VPP aggregate active-power targets in MW.

## Dispatch

- v0.1 uses rule-based target projection and marginal-cost ordering.
- PV has near-zero marginal cost and may be curtailed if the target or constraints
  require it.
- Microturbine ramp constraints are enforced locally.
- EV and HVAC behavior is intentionally aggregated and simple. Detailed individual
  mobility and thermal models are future extensions.

## Dynamic States

- Storage SOC, EV SOC, and HVAC indoor temperature are updated outside pandapower.
- The update order is dispatch, power flow, dynamic-state update, result logging.

