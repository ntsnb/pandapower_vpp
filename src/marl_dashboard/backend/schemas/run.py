from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class RunSummary(BaseModel):
    run_id: str
    status: str = "unknown"
    started_at: str | None = None
    ended_at: str | None = None
    algorithm: str | None = None
    environment: str | None = None
    vpp_count: int | None = None
    epoch_count: int | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
