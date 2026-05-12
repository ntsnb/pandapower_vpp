from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class SimulationLogger:
    rows: list[dict[str, Any]] = field(default_factory=list)

    def log(self, **kwargs: Any) -> None:
        self.rows.append(dict(kwargs))

