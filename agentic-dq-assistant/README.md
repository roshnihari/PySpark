# dq-agent: Agentic Data Quality & Masking Assistant

Profile a dataset with PySpark, let an LLM agent propose data-quality rules and
masking, and require human approval before anything runs.

**Status:** stages 1-2 done (profiling, typed rules + Spark execution).
Next: LangGraph agent, approval checkpoint, masking, report.

## Design decisions
- The agent only ever sees the **profile summary**, never raw rows.
- Agent output is validated with Pydantic (`parse_rule`); invalid proposals are rejected.
- Rules ignore nulls except `not_null`, so each rule answers one question.

## Run
```bash
pip install -e ".[dev]"
pytest -q
```
Requires Java 17+ for PySpark.
