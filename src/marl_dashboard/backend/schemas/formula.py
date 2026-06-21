from __future__ import annotations

from pydantic import BaseModel


class FormulaDefinition(BaseModel):
    name: str
    formula_latex: str
    description: str | None = None
    unit: str | None = None
