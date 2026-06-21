from __future__ import annotations

from pydantic import BaseModel, Field


class DateStatus(BaseModel):
    date: str
    observed_time_slots: int
    expected_time_slots: int
    complete: bool
    status: str


class SelectorOptions(BaseModel):
    run_id: str
    dates: list[str] = Field(default_factory=list)
    date_statuses: list[DateStatus] = Field(default_factory=list)
    vpp_ids: list[str] = Field(default_factory=list)
    agent_ids: list[str] = Field(default_factory=list)
    policy_ids: list[str] = Field(default_factory=list)
    epoch_ids: list[int] = Field(default_factory=list)
    episode_ids: list[int] = Field(default_factory=list)
    time_indices: list[int] = Field(default_factory=list)
