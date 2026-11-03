"""Column profiling with PySpark.

Only this summary (never raw rows) should be shared with an LLM agent.
"""
from __future__ import annotations

from pydantic import BaseModel
from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from pyspark.sql.types import StringType

# Simple patterns that hint a string column holds sensitive data.
PATTERNS: dict[str, str] = {
    "email": r"^[^@\s]+@[^@\s]+\.[^@\s]+$",
    "us_phone": r"^\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}$",
    "ssn_like": r"^\d{3}-\d{2}-\d{4}$",
}


class ColumnProfile(BaseModel):
    name: str
    dtype: str
    row_count: int
    null_count: int
    null_rate: float
    approx_distinct: int
    min_value: str | None = None
    max_value: str | None = None
    top_values: list[tuple[str, int]] = []
    pattern_match_rates: dict[str, float] = {}


def profile_dataframe(df: DataFrame, top_n: int = 5) -> dict[str, ColumnProfile]:
    """Profile every column and return {column_name: ColumnProfile}."""
    total = df.count()

    # One aggregation pass for the cheap per-column stats.
    aggs = []
    for c in df.columns:
        col = F.col(f"`{c}`")
        aggs += [
            F.sum(F.when(col.isNull(), 1).otherwise(0)).alias(f"{c}__nulls"),
            F.approx_count_distinct(col).alias(f"{c}__distinct"),
            F.min(col.cast("string")).alias(f"{c}__min"),
            F.max(col.cast("string")).alias(f"{c}__max"),
        ]
    stats = df.agg(*aggs).first().asDict()

    profiles: dict[str, ColumnProfile] = {}
    for field in df.schema.fields:
        c = field.name
        nulls = int(stats[f"{c}__nulls"] or 0)
        profile = ColumnProfile(
            name=c,
            dtype=field.dataType.simpleString(),
            row_count=total,
            null_count=nulls,
            null_rate=(nulls / total) if total else 0.0,
            approx_distinct=int(stats[f"{c}__distinct"] or 0),
            min_value=stats[f"{c}__min"],
            max_value=stats[f"{c}__max"],
            top_values=_top_values(df, c, top_n),
        )
        if isinstance(field.dataType, StringType):
            profile.pattern_match_rates = _pattern_rates(df, c)
        profiles[c] = profile
    return profiles


def _top_values(df: DataFrame, column: str, n: int) -> list[tuple[str, int]]:
    rows = (
        df.where(F.col(f"`{column}`").isNotNull())
        .groupBy(F.col(f"`{column}`").cast("string").alias("v"))
        .count()
        .orderBy(F.desc("count"))
        .limit(n)
        .collect()
    )
    return [(r["v"], int(r["count"])) for r in rows]


def _pattern_rates(df: DataFrame, column: str) -> dict[str, float]:
    """Share of non-null values matching each sensitive-data pattern."""
    col = F.col(f"`{column}`")
    non_null = df.where(col.isNotNull())
    total = non_null.count()
    if total == 0:
        return {}
    exprs = [
        F.avg(F.when(col.rlike(rx), 1.0).otherwise(0.0)).alias(name)
        for name, rx in PATTERNS.items()
    ]
    row = non_null.agg(*exprs).first().asDict()
    return {k: float(v) for k, v in row.items() if v and v > 0}


def likely_sensitive(profile: ColumnProfile, threshold: float = 0.8) -> list[str]:
    """Pattern names that match at least `threshold` of a column's values."""
    return [p for p, rate in profile.pattern_match_rates.items() if rate >= threshold]
