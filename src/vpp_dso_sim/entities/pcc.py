from __future__ import annotations

from dataclasses import dataclass


@dataclass
class PCC:
    id: str
    bus: int
    p_min_mw: float | None = None
    p_max_mw: float | None = None
    q_min_mvar: float | None = None
    q_max_mvar: float | None = None

