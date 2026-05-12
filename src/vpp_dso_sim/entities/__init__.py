"""Logical market and grid actors.

The concrete DSO and VPP classes are loaded lazily to avoid import cycles when
schema-only modules are imported by optimization code.
"""

from vpp_dso_sim.entities.schemas import (
    DERSpec,
    DispatchAward,
    ExplanationRecord,
    FRObject,
    LocalFlexNeed,
    MeasurementReport,
    PowerBounds,
    VPPFlexBid,
    VPPPortfolio,
)

__all__ = [
    "DERSpec",
    "DSO",
    "DispatchAward",
    "ExplanationRecord",
    "FRObject",
    "LocalFlexNeed",
    "MeasurementReport",
    "PowerBounds",
    "VPPAggregator",
    "VPPFlexBid",
    "VPPPortfolio",
]


def __getattr__(name: str):
    if name == "DSO":
        from vpp_dso_sim.entities.dso import DSO

        return DSO
    if name == "VPPAggregator":
        from vpp_dso_sim.entities.vpp import VPPAggregator

        return VPPAggregator
    raise AttributeError(name)

