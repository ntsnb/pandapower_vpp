from __future__ import annotations

from pydantic import BaseModel


class VariableDefinition(BaseModel):
    name: str
    display_name: str | None = None
    symbol: str | None = None
    unit: str | None = None
    group: str | None = None
    physical_meaning: str | None = None
    formula_latex: str | None = None
    min_value: float | None = None
    max_value: float | None = None
    source: str | None = None
    notes: str | None = None
