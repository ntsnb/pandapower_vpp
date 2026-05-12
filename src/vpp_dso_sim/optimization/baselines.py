from __future__ import annotations


def price_driven_target(vpp, t: int, price: float, low_price: float = 55.0, high_price: float = 100.0) -> float:
    p_min, p_max, _, _ = vpp.aggregate_flexibility(t)
    if price <= low_price:
        return p_min + 0.25 * (p_max - p_min)
    if price >= high_price:
        return p_min + 0.90 * (p_max - p_min)
    weight = (price - low_price) / max(1e-9, high_price - low_price)
    return p_min + (0.25 + 0.65 * weight) * (p_max - p_min)

