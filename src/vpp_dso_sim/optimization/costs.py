from __future__ import annotations

from vpp_dso_sim.der.base import DERBase


def quadratic_cost(p_mw: float, coefficients: tuple[float, float, float]) -> float:
    a, b, c = coefficients
    return a * p_mw * p_mw + b * p_mw + c


def total_der_cost(der_list: list[DERBase]) -> float:
    return float(sum(der.operating_cost() for der in der_list))

