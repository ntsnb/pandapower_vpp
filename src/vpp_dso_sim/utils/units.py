from __future__ import annotations


def kw_to_mw(value_kw: float) -> float:
    return value_kw / 1000.0


def kvar_to_mvar(value_kvar: float) -> float:
    return value_kvar / 1000.0


def mw_to_kw(value_mw: float) -> float:
    return value_mw * 1000.0

