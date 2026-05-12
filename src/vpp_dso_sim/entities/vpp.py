from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandapower as pp

from vpp_dso_sim.der.base import DERBase
from vpp_dso_sim.der.evcs import EVCSModel
from vpp_dso_sim.der.hvac import HVACModel
from vpp_dso_sim.der.storage import StorageModel
from vpp_dso_sim.entities.schemas import VPPPortfolio
from vpp_dso_sim.optimization.aggregation import aggregate_flexibility_fast
from vpp_dso_sim.optimization.costs import total_der_cost
from vpp_dso_sim.optimization.disaggregation import (
    disaggregate_target_by_learned_action,
    disaggregate_target_by_rule,
)


@dataclass
class VPPAggregator:
    id: str
    name: str
    pcc_bus: int
    der_list: list[DERBase] = field(default_factory=list)
    mode: str = "rule_based"
    privacy_mode: str = "full_information"
    metadata: dict[str, Any] = field(default_factory=dict)

    def attach_assets_to_net(self, net: pp.pandapowerNet) -> None:
        for der in self.der_list:
            der.owner_vpp_id = self.id
            if not der.name:
                der.name = der.id
            der.attach_to_net(net)

    def get_internal_state(self, t: int) -> dict[str, Any]:
        if self.privacy_mode == "representative_data":
            p_min, p_max, q_min, q_max = self.aggregate_flexibility(t)
            return {
                "vpp_id": self.id,
                "p_mw": self.current_power_mw(),
                "q_mvar": self.current_reactive_power_mvar(),
                "p_min_mw": p_min,
                "p_max_mw": p_max,
                "q_min_mvar": q_min,
                "q_max_mvar": q_max,
            }
        return {
            "vpp_id": self.id,
            "assets": [der.get_state() for der in self.der_list],
            "aggregate": self.report_to_dso(t),
        }

    def aggregate_flexibility(self, t: int) -> tuple[float, float, float, float]:
        return aggregate_flexibility_fast(self, t)

    def connection_buses(self) -> list[int]:
        return sorted({int(der.bus) for der in self.der_list})

    def physical_mode(self) -> str:
        return "single_pcc" if self.connection_buses() == [int(self.pcc_bus)] else "multi_node"

    def portfolio(self, t: int = 0) -> VPPPortfolio:
        return VPPPortfolio.from_vpp(self, t)

    def aggregate_cost_curve(self, t: int) -> dict[str, float]:
        # TODO(v0.2): fit a piecewise or quadratic aggregate cost curve. For now,
        # expose only current operating cost because rule dispatch is sufficient.
        return {"current_cost": self.operating_cost()}

    def day_ahead_bid(self, t: int, price_hint: float | None = None) -> dict[str, Any]:
        """Return a privacy-preserving capability/price report for DSO envelope construction."""

        p_min, p_max, q_min, q_max = self.aggregate_flexibility(t)
        span = max(0.0, p_max - p_min)
        current = self.current_power_mw()
        cost = self.aggregate_cost_curve(t).get("current_cost", self.operating_cost())
        base_price = float(price_hint if price_hint is not None else 80.0)
        return {
            "vpp_id": self.id,
            "time_index": int(t),
            "physical_mode": self.physical_mode(),
            "connection_buses": self.connection_buses(),
            "p_min_mw": float(p_min),
            "p_max_mw": float(p_max),
            "q_min_mvar": float(q_min),
            "q_max_mvar": float(q_max),
            "current_p_mw": float(current),
            "upward_flex_mw": float(max(0.0, p_max - current)),
            "downward_flex_mw": float(max(0.0, current - p_min)),
            "flex_span_mw": float(span),
            "bid_price_up": float(base_price + 10.0 + 2.0 * cost),
            "bid_price_down": float(max(0.0, base_price - 10.0 + 2.0 * cost)),
            "confidence": 0.95 if span > 1e-9 else 0.2,
            "privacy_note": "aggregated bid only; DER-level private cost/state remains local",
        }

    def current_power_mw(self) -> float:
        return float(sum(der.p_mw for der in self.der_list))

    def current_reactive_power_mvar(self) -> float:
        return float(sum(der.q_mvar for der in self.der_list))

    def disaggregate_power_target(
        self,
        target_p_mw: float,
        target_q_mvar: float = 0.0,
        t: int = 0,
        der_actions: dict[str, float] | list[float] | tuple[float, ...] | None = None,
    ) -> dict[str, float]:
        if der_actions is not None:
            return disaggregate_target_by_learned_action(self, target_p_mw, t, der_actions)
        return disaggregate_target_by_rule(self, target_p_mw, t)

    def disaggregate_power_targets_by_scope(
        self,
        targets_by_scope: dict[str, float],
        t: int = 0,
        der_actions: dict[str, float] | list[float] | tuple[float, ...] | None = None,
    ) -> dict[str, float]:
        """Disaggregate bus/zone/DER-vector targets without collapsing buses.

        The legacy DSO/RL policy may still emit one aggregate target, but the
        safety layer can now repair that target into FR/DOE scope keys first.
        This method then writes each bus/zone/DER group separately.
        """

        dispatch: dict[str, float] = {}
        action_by_der: dict[str, float] = {}
        if isinstance(der_actions, dict):
            action_by_der = {str(key): float(value) for key, value in der_actions.items()}
        elif der_actions is not None:
            action_by_der = {
                der.id: float(der_actions[index])
                for index, der in enumerate(self.der_list)
                if index < len(der_actions)
            }

        def key_for(der) -> str:
            if f"bus_{int(der.bus)}" in targets_by_scope:
                return f"bus_{int(der.bus)}"
            zone_key = f"zone_{der.metadata.get('zone_id', f'bus_{int(der.bus)}')}"
            if zone_key in targets_by_scope:
                return zone_key
            if str(der.id) in targets_by_scope:
                return str(der.id)
            return f"pcc_{int(self.pcc_bus)}"

        grouped: dict[str, list[Any]] = {}
        for der in self.der_list:
            grouped.setdefault(key_for(der), []).append(der)

        for scope_key, ders in grouped.items():
            if not ders:
                continue
            target = float(targets_by_scope.get(scope_key, sum(der.p_mw for der in ders)))
            rows = []
            group_dispatch: dict[str, float] = {}
            for index, der in enumerate(ders):
                der.metadata["current_t"] = t
                p_min, p_max, _, _ = der.get_bounds(t)
                if der_actions is None:
                    rows.append(
                        {
                            "der": der,
                            "p_min": float(p_min),
                            "p_max": float(p_max),
                            "capacity": max(0.0, float(p_max) - float(p_min)),
                            "cost": der.marginal_cost(float(p_min)),
                        }
                    )
                    group_dispatch[der.id] = float(p_min)
                else:
                    raw = max(-1.0, min(1.0, float(action_by_der.get(der.id, 0.0))))
                    group_dispatch[der.id] = float(
                        max(float(p_min), min(float(p_max), 0.5 * (float(p_min) + float(p_max)) + 0.5 * raw * max(0.0, float(p_max) - float(p_min))))
                    )
                    rows.append({"der": der, "p_min": float(p_min), "p_max": float(p_max), "capacity": max(0.0, float(p_max) - float(p_min)), "cost": der.marginal_cost(float(p_min))})

            if der_actions is None:
                lower_sum = float(sum(row["p_min"] for row in rows))
                remaining = max(0.0, target - lower_sum)
                for row in sorted(rows, key=lambda item: item["cost"]):
                    if remaining <= 1e-12:
                        break
                    addition = min(float(row["capacity"]), remaining)
                    group_dispatch[row["der"].id] += addition
                    remaining -= addition

            residual = float(target - sum(group_dispatch.values()))
            if abs(residual) > 1e-10:
                if residual > 0.0:
                    capacities = [
                        (row["der"].id, max(0.0, float(row["p_max"]) - group_dispatch[row["der"].id]))
                        for row in rows
                    ]
                else:
                    capacities = [
                        (row["der"].id, max(0.0, group_dispatch[row["der"].id] - float(row["p_min"])))
                        for row in rows
                    ]
                total_capacity = float(sum(capacity for _, capacity in capacities))
                if total_capacity > 1e-12:
                    for der_id, capacity in capacities:
                        row = next(item for item in rows if item["der"].id == der_id)
                        group_dispatch[der_id] = float(
                            max(
                                float(row["p_min"]),
                                min(float(row["p_max"]), group_dispatch[der_id] + residual * capacity / total_capacity),
                            )
                        )
            dispatch.update(group_dispatch)
        return dispatch

    def apply_dispatch_to_net(self, net: pp.pandapowerNet, dispatch: dict[str, float], t: int = 0) -> None:
        for der in self.der_list:
            der.metadata["current_t"] = t
            target = dispatch.get(der.id, der.p_mw)
            der.set_power(net, target, 0.0)

    def report_to_dso(self, t: int) -> dict[str, Any]:
        p_min, p_max, q_min, q_max = self.aggregate_flexibility(t)
        return {
            "vpp_id": self.id,
            "pcc_bus": self.pcc_bus,
            "physical_mode": self.physical_mode(),
            "connection_buses": self.connection_buses(),
            "p_mw": self.current_power_mw(),
            "q_mvar": self.current_reactive_power_mvar(),
            "p_min_mw": p_min,
            "p_max_mw": p_max,
            "q_min_mvar": q_min,
            "q_max_mvar": q_max,
            "day_ahead_bid": self.day_ahead_bid(t),
            "privacy_mode": self.privacy_mode,
        }

    def update_dynamic_states(self, t: int, dt_hours: float) -> None:
        for der in self.der_list:
            if isinstance(der, StorageModel):
                der.update_soc(dt_hours=dt_hours)
            elif isinstance(der, EVCSModel):
                der.update_internal_devices(t, dt_hours)
            elif isinstance(der, HVACModel):
                der.update_temperature(t=t, dt_hours=dt_hours)
            else:
                der.step(t, None, dt_hours)

    def operating_cost(self) -> float:
        return total_der_cost(self.der_list)

    def comfort_penalty(self, t: int) -> float:
        return float(sum(der.comfort_penalty(t) for der in self.der_list if isinstance(der, HVACModel)))

    def soc_violation_penalty(self, t: int) -> float:
        penalty = 0.0
        for der in self.der_list:
            if isinstance(der, StorageModel):
                penalty += 1_000.0 * max(0.0, der.soc_min - der.soc) ** 2
                penalty += 1_000.0 * max(0.0, der.soc - der.soc_max) ** 2
            elif isinstance(der, EVCSModel):
                penalty += der.unmet_soc_penalty(t)
        return float(penalty)

    def reset(self) -> None:
        # TODO(v0.2): store and restore full initial DER states for randomized episodes.
        for der in self.der_list:
            der.metadata.pop("current_t", None)
