from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class MarketSignal:
    price_profile: list[float] = field(default_factory=list)

    def price_at(self, t: int) -> float:
        if not self.price_profile:
            return 0.0
        return float(self.price_profile[t % len(self.price_profile)])

