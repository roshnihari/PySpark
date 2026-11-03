from dq_agent.profiling import likely_sensitive, profile_dataframe


def test_null_rate(customers):
    p = profile_dataframe(customers)
    assert p["email"].null_count == 1
    assert p["email"].null_rate == 0.25
    assert p["id"].null_count == 0


def test_distinct_and_dtype(customers):
    p = profile_dataframe(customers)
    assert p["country"].approx_distinct == 3
    assert p["amount"].dtype == "double"


def test_detects_ssn_like_column(customers):
    p = profile_dataframe(customers)
    assert "ssn_like" in likely_sensitive(p["ssn"])
    assert likely_sensitive(p["country"]) == []
