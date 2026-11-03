import pytest
from pydantic import ValidationError

from dq_agent.rules import (
    AllowedValuesRule,
    NotNullRule,
    RangeRule,
    RegexRule,
    parse_rule,
    run_rules,
    split_by_rules,
)


def test_run_rules_counts_failures(customers):
    rules = [
        NotNullRule(column="email"),
        RangeRule(column="amount", min_value=0, max_value=100),
        AllowedValuesRule(column="country", allowed=["US", "CA"]),
        RegexRule(column="email", pattern=r"^[^@\s]+@[^@\s]+\.[^@\s]+$"),
    ]
    failed = {r.rule_id: r.failed_rows for r in run_rules(customers, rules)}
    assert failed["not_null:email"] == 1
    assert failed["range:amount"] == 2  # -5 and 120; null is ignored
    assert failed["allowed_values:country"] == 1
    assert failed["regex:email"] == 1  # "not-an-email"; null is ignored


def test_split_by_rules(customers):
    rules = [RangeRule(column="amount", min_value=0, max_value=100)]
    clean, bad = split_by_rules(customers, rules)
    assert clean.count() == 2
    assert bad.count() == 2


def test_parse_rule_accepts_valid_llm_output():
    rule = parse_rule({"kind": "range", "column": "amount", "min_value": 0})
    assert isinstance(rule, RangeRule)


def test_parse_rule_rejects_unknown_kind():
    with pytest.raises(ValidationError):
        parse_rule({"kind": "drop_table", "column": "amount"})
