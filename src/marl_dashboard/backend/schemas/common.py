from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


MetricGroup = Literal["dataset", "reward", "cost", "loss", "scalar", "event", "model"]


class MetricRecord(BaseModel):
    model_config = ConfigDict(extra="allow")

    run_id: str
    epoch_id: int | None = None
    episode_id: int | None = None
    batch_id: int | str | None = None
    gradient_step: int | None = None
    global_env_step: int | None = None
    env_id: str | None = None
    vpp_id: str | None = None
    agent_id: str | None = None
    policy_id: str | None = None
    date: str | None = None
    time_index: int | None = None
    timestamp: str | None = None
    metric_group: MetricGroup
    metric_name: str
    value: float | int | str | bool | None
    unit: str | None = None
    formula_latex: str | None = None
    description: str | None = None


class QueryResponse(BaseModel):
    chart_series: list[dict[str, Any]] = Field(default_factory=list)
    table_rows: list[dict[str, Any]] = Field(default_factory=list)
    units: dict[str, str] = Field(default_factory=dict)
    formulas: dict[str, str] = Field(default_factory=dict)
    summary: dict[str, Any] = Field(default_factory=dict)
