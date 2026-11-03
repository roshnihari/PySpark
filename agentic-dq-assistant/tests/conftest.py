import pytest
from pyspark.sql import SparkSession


@pytest.fixture(scope="session")
def spark():
    session = (
        SparkSession.builder.master("local[1]")
        .appName("dq-agent-tests")
        .config("spark.ui.enabled", "false")
        .config("spark.sql.shuffle.partitions", "1")
        .getOrCreate()
    )
    yield session
    session.stop()


@pytest.fixture()
def customers(spark):
    rows = [
        (1, "ann@example.com", "555-12-3456", 30.0, "US"),
        (2, "bob@example.com", "555-98-7654", -5.0, "US"),
        (3, None, "555-11-2222", 120.0, "CA"),
        (4, "not-an-email", None, None, "XX"),
    ]
    return spark.createDataFrame(rows, ["id", "email", "ssn", "amount", "country"])
