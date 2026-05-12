# Sign Conventions

Pandapower and aggregator-level control code use different sign conventions.
This project keeps both explicit and tests the conversions.

## Project Internal Convention

- `P > 0`: net injection into the grid.
- `P < 0`: net absorption from the grid.
- VPP aggregate power is the sum of internal DER powers.

## Pandapower Mapping

| Asset | pandapower element | pandapower sign | internal sign |
| --- | --- | --- | --- |
| Fixed load | `load` | `p_mw > 0` consumes | represented as `P = -p_mw` when aggregated |
| Flexible load | `load` | `p_mw > 0` consumes | `P = -actual_load_p_mw` |
| PV/wind/inverter DER | `sgen` | `p_mw > 0` injects | `P = p_mw` |
| Microturbine | `sgen` in v0.1 | `p_mw > 0` injects | `P = p_mw` |
| Storage | `storage` | `p_mw > 0` charges | `P = -storage.p_mw` |
| EVCS G2V | `load` | `p_mw > 0` consumes | `P = -charging_p_mw` |
| HVAC aggregator | `load` | `p_mw > 0` consumes | `P = -hvac_p_mw` |

## Storage SOC

Pandapower does not update storage SOC during power flow. The domain model uses
internal injection power:

- `p_internal_mw > 0`: discharging, SOC decreases.
- `p_internal_mw < 0`: charging, SOC increases.

The conversion is:

```python
pp_storage_p_mw = -p_internal_mw
p_internal_mw = -pp_storage_p_mw
```

SOC update:

```text
if p_internal >= 0:
    SOC_next = SOC - p_internal * dt / (eta_discharge * E)
else:
    SOC_next = SOC + eta_charge * (-p_internal) * dt / E
```

