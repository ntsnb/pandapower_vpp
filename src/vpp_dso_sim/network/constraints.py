from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandapower as pp


@dataclass
class ConstraintViolation:
    kind: str
    element: str
    value: float
    limit: float
    magnitude: float


@dataclass
class ConstraintReport:
    converged: bool
    violations: list[ConstraintViolation] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.converged and not self.violations

    def to_records(self, step: int | None = None) -> list[dict[str, float | str | int]]:
        records: list[dict[str, float | str | int]] = []
        for item in self.violations:
            record: dict[str, float | str | int] = {
                "kind": item.kind,
                "element": item.element,
                "value": item.value,
                "limit": item.limit,
                "magnitude": item.magnitude,
            }
            if step is not None:
                record["step"] = step
            records.append(record)
        if not self.converged:
            record = {
                "kind": "powerflow",
                "element": "net",
                "value": 0.0,
                "limit": 1.0,
                "magnitude": 1.0,
            }
            if step is not None:
                record["step"] = step
            records.append(record)
        return records


def check_network_constraints(
    net: pp.pandapowerNet,
    voltage_limits: tuple[float, float] = (0.95, 1.05),
    line_loading_limit_percent: float = 100.0,
    trafo_loading_limit_percent: float = 100.0,
) -> ConstraintReport:
    report = ConstraintReport(converged=bool(getattr(net, "converged", False)))
    if not report.converged:
        return report

    vmin, vmax = voltage_limits
    if hasattr(net, "res_bus") and "vm_pu" in net.res_bus:
        for idx, vm_pu in net.res_bus["vm_pu"].items():
            value = float(vm_pu)
            if value < vmin:
                report.violations.append(
                    ConstraintViolation("bus_voltage_low", str(idx), value, vmin, vmin - value)
                )
            elif value > vmax:
                report.violations.append(
                    ConstraintViolation("bus_voltage_high", str(idx), value, vmax, value - vmax)
                )

    if hasattr(net, "res_line") and "loading_percent" in net.res_line:
        for idx, loading in net.res_line["loading_percent"].items():
            value = float(loading)
            if np.isfinite(value) and value > line_loading_limit_percent:
                report.violations.append(
                    ConstraintViolation(
                        "line_overload",
                        str(idx),
                        value,
                        line_loading_limit_percent,
                        value - line_loading_limit_percent,
                    )
                )

    if len(net.trafo) and hasattr(net, "res_trafo") and "loading_percent" in net.res_trafo:
        for idx, loading in net.res_trafo["loading_percent"].items():
            value = float(loading)
            if np.isfinite(value) and value > trafo_loading_limit_percent:
                report.violations.append(
                    ConstraintViolation(
                        "trafo_overload",
                        str(idx),
                        value,
                        trafo_loading_limit_percent,
                        value - trafo_loading_limit_percent,
                    )
                )

    return report


def violation_penalties(report: ConstraintReport) -> dict[str, float]:
    penalties = {
        "voltage_violation_penalty": 0.0,
        "line_overload_penalty": 0.0,
        "transformer_overload_penalty": 0.0,
        "powerflow_penalty": 0.0,
    }
    if not report.converged:
        penalties["powerflow_penalty"] = 1_000.0
    for violation in report.violations:
        if violation.kind.startswith("bus_voltage"):
            penalties["voltage_violation_penalty"] += 10_000.0 * violation.magnitude**2
        elif violation.kind == "line_overload":
            penalties["line_overload_penalty"] += 5.0 * violation.magnitude**2
        elif violation.kind == "trafo_overload":
            penalties["transformer_overload_penalty"] += 5.0 * violation.magnitude**2
    return penalties

