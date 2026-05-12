from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, ClassVar


@dataclass(frozen=True)
class FieldVisibility:
    """Visibility flags for a communication field.

    The flags document privacy boundaries. They do not grant access by
    themselves; observation builders and market interfaces must use them when
    exporting data to agents or dashboards.
    """

    visible_to_dso: bool
    visible_to_vpp_i: bool
    visible_to_other_vpp: bool = False
    oracle_only: bool = False

    def to_dict(self) -> dict[str, bool]:
        return asdict(self)


class SerializableSchema:
    """Small dataclass serialization helper used by schema records."""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]):
        return cls(**data)


@dataclass
class DERSpec(SerializableSchema):
    der_id: str
    der_type: str
    bus_id: int
    phase: str = "abc"
    pp_element_type: str = ""
    pp_element_index: int | None = None
    p_min_mw: float = 0.0
    p_max_mw: float = 0.0
    q_min_mvar: float = 0.0
    q_max_mvar: float = 0.0
    current_p_mw: float = 0.0
    current_q_mvar: float = 0.0
    available_p_mw: float | None = None
    soc: float | None = None
    soc_min: float | None = None
    soc_max: float | None = None
    average_soc: float | None = None
    indoor_temp: float | None = None
    temp_min: float | None = None
    temp_max: float | None = None
    comfort_penalty: float | None = None
    owner_vpp_id: str | None = None
    controllable: bool = True
    cost_coefficients: tuple[float, float, float] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    visibility: ClassVar[dict[str, FieldVisibility]] = {
        "der_id": FieldVisibility(True, True, False, False),
        "der_type": FieldVisibility(True, True, False, False),
        "bus_id": FieldVisibility(True, True, False, False),
        "phase": FieldVisibility(True, True, False, False),
        "pp_element_type": FieldVisibility(True, True, False, False),
        "pp_element_index": FieldVisibility(True, True, False, False),
        "p_min_mw": FieldVisibility(True, True, False, False),
        "p_max_mw": FieldVisibility(True, True, False, False),
        "q_min_mvar": FieldVisibility(True, True, False, False),
        "q_max_mvar": FieldVisibility(True, True, False, False),
        "current_p_mw": FieldVisibility(True, True, False, False),
        "current_q_mvar": FieldVisibility(True, True, False, False),
        "available_p_mw": FieldVisibility(True, True, False, False),
        "soc": FieldVisibility(False, True, False, False),
        "soc_min": FieldVisibility(False, True, False, False),
        "soc_max": FieldVisibility(False, True, False, False),
        "average_soc": FieldVisibility(False, True, False, False),
        "indoor_temp": FieldVisibility(False, True, False, False),
        "temp_min": FieldVisibility(False, True, False, False),
        "temp_max": FieldVisibility(False, True, False, False),
        "comfort_penalty": FieldVisibility(False, True, False, False),
        "owner_vpp_id": FieldVisibility(True, True, False, False),
        "controllable": FieldVisibility(True, True, False, False),
        "cost_coefficients": FieldVisibility(False, True, False, True),
        "metadata": FieldVisibility(False, True, False, True),
    }

    @classmethod
    def from_der(cls, der, t: int = 0, include_private_cost: bool = True) -> "DERSpec":
        p_min, p_max, q_min, q_max = der.get_bounds(t)
        state = der.get_state() if hasattr(der, "get_state") else {}
        available_p_mw = None
        if hasattr(der, "available_power"):
            available_p_mw = float(der.available_power(t))
        comfort_penalty = None
        if hasattr(der, "comfort_penalty"):
            comfort_penalty = float(der.comfort_penalty(t))
        return cls(
            der_id=str(der.id),
            der_type=der.__class__.__name__,
            bus_id=int(der.bus),
            phase=str(der.metadata.get("phase", "abc")),
            pp_element_type=str(der.pp_element_type),
            pp_element_index=der.pp_element_index,
            p_min_mw=float(p_min),
            p_max_mw=float(p_max),
            q_min_mvar=float(q_min),
            q_max_mvar=float(q_max),
            current_p_mw=float(getattr(der, "p_mw", state.get("p_mw", 0.0))),
            current_q_mvar=float(getattr(der, "q_mvar", state.get("q_mvar", 0.0))),
            available_p_mw=available_p_mw,
            soc=float(state["soc"]) if "soc" in state else None,
            soc_min=float(getattr(der, "soc_min")) if hasattr(der, "soc_min") else None,
            soc_max=float(getattr(der, "soc_max")) if hasattr(der, "soc_max") else None,
            average_soc=float(state["average_soc"]) if "average_soc" in state else None,
            indoor_temp=float(state["indoor_temp"]) if "indoor_temp" in state else None,
            temp_min=float(getattr(der, "temp_min")) if hasattr(der, "temp_min") else None,
            temp_max=float(getattr(der, "temp_max")) if hasattr(der, "temp_max") else None,
            comfort_penalty=comfort_penalty,
            owner_vpp_id=str(der.owner_vpp_id) if der.owner_vpp_id else None,
            controllable=bool(der.controllable),
            cost_coefficients=tuple(der.cost_coefficients) if include_private_cost else None,
            metadata=dict(der.metadata) if include_private_cost else {},
        )


@dataclass
class VPPPortfolio(SerializableSchema):
    vpp_id: str
    physical_mode: str
    pcc_bus_id: int | None
    connection_buses: list[int]
    zone_ids: list[str]
    der_ids: list[str]
    max_import_mw: float | None = None
    max_export_mw: float | None = None
    portfolio_version: str = "v0"

    visibility: ClassVar[dict[str, FieldVisibility]] = {
        "vpp_id": FieldVisibility(True, True, False, False),
        "physical_mode": FieldVisibility(True, True, False, False),
        "pcc_bus_id": FieldVisibility(True, True, False, False),
        "connection_buses": FieldVisibility(True, True, False, False),
        "zone_ids": FieldVisibility(True, True, False, False),
        "der_ids": FieldVisibility(True, True, False, False),
        "max_import_mw": FieldVisibility(True, True, False, False),
        "max_export_mw": FieldVisibility(True, True, False, False),
        "portfolio_version": FieldVisibility(True, True, False, False),
    }

    @classmethod
    def from_vpp(cls, vpp, t: int = 0) -> "VPPPortfolio":
        connection_buses = sorted({int(der.bus) for der in vpp.der_list})
        der_ids = [str(der.id) for der in vpp.der_list]
        physical_mode = "single_pcc" if connection_buses == [int(vpp.pcc_bus)] else "multi_node"
        p_min, p_max, _, _ = vpp.aggregate_flexibility(t)
        zone_ids = list(vpp.metadata.get("zone_ids", []))
        if not zone_ids:
            zone_ids = [f"bus_{bus_id}" for bus_id in connection_buses]
        return cls(
            vpp_id=str(vpp.id),
            physical_mode=physical_mode,
            pcc_bus_id=int(vpp.pcc_bus) if vpp.pcc_bus is not None else None,
            connection_buses=connection_buses,
            zone_ids=zone_ids,
            der_ids=der_ids,
            max_import_mw=float(max(0.0, -p_min)),
            max_export_mw=float(max(0.0, p_max)),
            portfolio_version=str(vpp.metadata.get("portfolio_version", "v0")),
        )


@dataclass
class PowerBounds(SerializableSchema):
    p_min_mw: float
    p_max_mw: float
    q_min_mvar: float = 0.0
    q_max_mvar: float = 0.0

    def clipped(self, p_mw: float, q_mvar: float = 0.0) -> tuple[float, float]:
        p = max(self.p_min_mw, min(self.p_max_mw, float(p_mw)))
        q = max(self.q_min_mvar, min(self.q_max_mvar, float(q_mvar)))
        return (p, q)

    @classmethod
    def sum(cls, bounds: list["PowerBounds"]) -> "PowerBounds":
        return cls(
            p_min_mw=float(sum(item.p_min_mw for item in bounds)),
            p_max_mw=float(sum(item.p_max_mw for item in bounds)),
            q_min_mvar=float(sum(item.q_min_mvar for item in bounds)),
            q_max_mvar=float(sum(item.q_max_mvar for item in bounds)),
        )


@dataclass
class FRObject(SerializableSchema):
    fr_id: str
    vpp_id: str
    time_index: int
    scope: str
    representation: str
    bounds: dict[str, PowerBounds]
    variables: list[str] = field(default_factory=lambda: ["p_mw", "q_mvar"])
    safety_margin_mw: float = 0.0
    safety_margin_mvar: float = 0.0
    valid_until_step: int | None = None
    source_method: str = "static_der_bounds_v0"
    portfolio_version: str = "v0"
    metadata: dict[str, Any] = field(default_factory=dict)

    visibility: ClassVar[dict[str, FieldVisibility]] = {
        "fr_id": FieldVisibility(True, True, False, False),
        "vpp_id": FieldVisibility(True, True, False, False),
        "time_index": FieldVisibility(True, True, False, False),
        "scope": FieldVisibility(True, True, False, False),
        "representation": FieldVisibility(True, True, False, False),
        "bounds": FieldVisibility(True, True, False, False),
        "variables": FieldVisibility(True, True, False, False),
        "safety_margin_mw": FieldVisibility(True, True, False, False),
        "safety_margin_mvar": FieldVisibility(True, True, False, False),
        "valid_until_step": FieldVisibility(True, True, False, False),
        "source_method": FieldVisibility(True, True, False, False),
        "portfolio_version": FieldVisibility(True, True, False, False),
        "metadata": FieldVisibility(True, True, False, False),
    }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "FRObject":
        payload = dict(data)
        payload["bounds"] = {
            str(key): value if isinstance(value, PowerBounds) else PowerBounds(**value)
            for key, value in payload.get("bounds", {}).items()
        }
        return cls(**payload)

    def aggregate_bounds(self) -> PowerBounds:
        return PowerBounds.sum(list(self.bounds.values()))

    def to_records(self) -> list[dict[str, Any]]:
        records: list[dict[str, Any]] = []
        for element_id, bounds in self.bounds.items():
            records.append(
                {
                    "fr_id": self.fr_id,
                    "vpp_id": self.vpp_id,
                    "time_index": self.time_index,
                    "scope": self.scope,
                    "representation": self.representation,
                    "element_id": element_id,
                    **bounds.to_dict(),
                    "source_method": self.source_method,
                    "portfolio_version": self.portfolio_version,
                }
            )
        return records


@dataclass
class LocalFlexNeed(SerializableSchema):
    need_id: str
    zone_id: str
    target_constraint: str
    direction: str
    required_effective_mw_or_mvar: float
    start_time: int
    duration_min: float
    response_time_min: float
    severity: float
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class VPPFlexBid(SerializableSchema):
    bid_id: str
    vpp_id: str
    portfolio_version: str
    zone_id: str
    direction: str
    quantity_mw_or_mvar: float
    duration_min: float
    response_time_min: float
    price: float
    price_unit: str = "currency_per_mwh"
    reliability: float = 1.0
    location_effectiveness: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class DispatchAward(SerializableSchema):
    award_id: str
    vpp_id: str
    need_id: str
    awarded_quantity: float
    settlement_price: float
    expected_effective_contribution: float
    dispatch_instruction: dict[str, Any]
    start_time: int
    end_time: int
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class MeasurementReport(SerializableSchema):
    report_id: str
    vpp_id: str
    time_index: int
    delivered_by_bus_or_zone: dict[str, float]
    deviation: float
    voltage_violations: int = 0
    line_violations: int = 0
    non_delivery_penalty: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ExplanationRecord(SerializableSchema):
    step: int
    vpp_id: str
    support_type_label: str
    reason: str
    dso_signal_seen: dict[str, Any]
    vpp_response: dict[str, Any]
    network_effect: dict[str, Any]
    settlement: dict[str, Any]
    reliability_update: dict[str, Any]


def schema_visibility(schema_cls: type) -> list[dict[str, Any]]:
    visibility = getattr(schema_cls, "visibility", {})
    return [{"field": field_name, **flags.to_dict()} for field_name, flags in visibility.items()]
