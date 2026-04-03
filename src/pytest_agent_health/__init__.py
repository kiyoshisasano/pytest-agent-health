"""
pytest-agent-health — Catch silent agent failures in CI.

Behavior lint for LLM agents, powered by agent-failure-debugger.

Usage:
    # As a fixture
    def test_agent(agent_health):
        raw_log = my_agent.run("What was Q3 revenue?")
        agent_health.check(raw_log, adapter="langchain")

    # CLI
    pytest --agent-health              # default: degraded = WARN
    pytest --agent-health --strict     # strict: degraded+risk = FAIL
"""

__version__ = "0.1.0"
