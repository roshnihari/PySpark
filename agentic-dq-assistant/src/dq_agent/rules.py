"""Typed data-quality rules and their Spark execution.

An LLM agent should propose rules as JSON that validates against `Rule`;
anything that fails validation is rejected before it touches data.
"""
from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, Field, TypeAdapter
from pyspark.sql import Column, DataFrame
from pyspark.sql import functions as F


class _BaseRule(BaseModel):
    column: str
    description: str = ""

    def passes(self) -> Column:  # pragma: no cover - overridden
        raise NotImplementedError

    @property
    def rule_id(self) -> str:
        return f"{self.kind}:{self.column}"  # type: ignore[attr-defined]


class NotNullRule(_BaseRule):
    kind: Literal["not_null"] = "not_null"

    def passes(self) -> Column:
        return F.col(f"`{self.column}`").isNotNull()


class RangeRule(_BaseRule):
    """Non-null values must fall in [min_value, max_value]. Nulls are ignored."""

    kind: Literal["range"] = "range"
    min_value: float | None = None
    max_value: float | None = None

    def passes(self) -> Column:
        col = F.col(f"`{self.column}`")
        cond = F.lit(True)
        if self.min_value is not None:
            cond = cond & (col >= self.min_value)
        if self.max_value is not None:
            cond = cond & (col <= self.max_value)
        return col.isNull() | cond


class AllowedValuesRule(_BaseRule):
    kind: Literal["allowed_values"] = "allowed_values"
    allowed: list[str]

    def passes(self) -> Column:
        col = F.col(f"`{self.column}`")
        return col.isNull() | col.cast("string").isin(self.allowed)


class RegexRule(_BaseRule):
    kind: Literal["regex"] = "regex"
    pattern: str

    def passes(self) -> Column:
        col = F.col(f"`{self.column}`")
        return col.isNull() | col.cast("string").rlike(self.pattern)


Rule = Annotated[
    NotNullRule | RangeRule | AllowedValuesRule | RegexRule,
    Field(discriminator="kind"),
]
_rule_adapter: TypeAdapter[Rule] = TypeAdapter(Rule)


def parse_rule(data: dict) -> _BaseRule:
    """Validate a raw dict (e.g. LLM output) into a typed rule or raise ValidationError."""
    return _rule_adapter.validate_python(data)


class RuleResult(BaseModel):
    rule_id: str
    description: str
    total_rows: int
    failed_rows: int

    @property
    def pass_rate(self) -> float:
        return 1.0 if self.total_rows == 0 else 1 - self.failed_rows / self.total_rows


def run_rules(df: DataFrame, rules: list[_BaseRule]) -> list[RuleResult]:
    total = df.count()
    results = []
    for rule in rules:
        failed = df.where(~rule.passes()).count()
        results.append(
            RuleResult(
                rule_id=rule.rule_id,
                description=rule.description,
                total_rows=total,
                failed_rows=failed,
            )
        )
    return results


def split_by_rules(df: DataFrame, rules: list[_BaseRule]) -> tuple[DataFrame, DataFrame]:
    """Return (clean_rows, quarantined_rows): quarantined rows fail at least one rule."""
    passes_all = F.lit(True)
    for rule in rules:
        passes_all = passes_all & rule.passes()
    return df.where(passes_all), df.where(~passes_all)
