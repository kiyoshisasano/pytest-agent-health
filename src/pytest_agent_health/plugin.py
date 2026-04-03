"""
plugin.py — pytest plugin entry point.

Registered via pyproject.toml [project.entry-points.pytest11].
Provides:
  - --agent-health CLI flag (enables the plugin)
  - --no-strict flag (relaxes degraded policy)
  - --agent-health-fail-on (force FAIL on specific patterns)
  - agent_health fixture
  - Terminal summary reporting
"""

from __future__ import annotations

import pytest

from pytest_agent_health.fixture import AgentHealthFixture
from pytest_agent_health.reporting import format_summary


# ---------------------------------------------------------------------------
# CLI options
# ---------------------------------------------------------------------------

def pytest_addoption(parser: pytest.Parser) -> None:
    """Register CLI options for agent-health."""
    group = parser.getgroup(
        "agent-health",
        "Agent behavior lint — catch silent LLM agent failures",
    )
    group.addoption(
        "--agent-health",
        action="store_true",
        default=False,
        help="Enable agent health checks (required to activate the plugin).",
    )
    group.addoption(
        "--no-strict",
        action="store_true",
        default=False,
        help=(
            "Relax policy: degraded status always produces WARN, "
            "never FAIL. Default (strict): degraded with risk "
            "indicators produces FAIL."
        ),
    )
    group.addoption(
        "--agent-health-fail-on",
        type=str,
        default="",
        help=(
            "Comma-separated failure pattern IDs that force FAIL "
            "regardless of execution_quality status. "
            "Example: --agent-health-fail-on=premature_termination,"
            "context_truncation_loss"
        ),
    )


# ---------------------------------------------------------------------------
# Fixture
# ---------------------------------------------------------------------------

@pytest.fixture
def agent_health(request: pytest.FixtureRequest) -> AgentHealthFixture:
    """Fixture providing agent health check methods.

    Usage:
        def test_agent(agent_health):
            raw_log = my_agent.run("What was Q3 revenue?")
            agent_health.check(raw_log, adapter="langchain")

    The fixture respects CLI options:
        --agent-health      Enable the plugin
        --no-strict         Relax degraded policy
        --agent-health-fail-on=...  Force FAIL on specific patterns
    """
    config = request.config

    # Parse CLI options
    strict = not config.getoption("--no-strict", default=False)

    fail_on_str = config.getoption("--agent-health-fail-on", default="")
    fail_on = frozenset(
        p.strip() for p in fail_on_str.split(",") if p.strip()
    ) if fail_on_str else frozenset()

    # Check for marker overrides
    marker = request.node.get_closest_marker("agent_health")
    if marker:
        strict = marker.kwargs.get("strict", strict)
        marker_fail_on = marker.kwargs.get("fail_on", None)
        if marker_fail_on:
            fail_on = fail_on | frozenset(marker_fail_on)

    return AgentHealthFixture(
        strict=strict,
        fail_on=fail_on,
    )


# ---------------------------------------------------------------------------
# Marker registration
# ---------------------------------------------------------------------------

def pytest_configure(config: pytest.Config) -> None:
    """Register the agent_health marker."""
    config.addinivalue_line(
        "markers",
        "agent_health(strict=True, fail_on=None, adapter='langchain'): "
        "Configure agent health check policy for this test.",
    )


# ---------------------------------------------------------------------------
# Terminal summary (optional, when --agent-health is enabled)
# ---------------------------------------------------------------------------

# Collect all verdicts across the session
_session_results: list = []


@pytest.hookimpl(tryfirst=True, hookwrapper=True)
def pytest_runtest_makereport(item, call):
    """Collect agent_health results from test items."""
    outcome = yield
    report = outcome.get_result()

    if report.when == "call":
        # Check if this test used agent_health fixture
        fixture_info = getattr(item, "funcargs", {})
        if "agent_health" in fixture_info:
            ah = fixture_info["agent_health"]
            if isinstance(ah, AgentHealthFixture) and ah.results:
                _session_results.extend(ah.results)


def pytest_terminal_summary(
    terminalreporter,
    exitstatus: int,
    config: pytest.Config,
) -> None:
    """Print agent-health summary at end of test session."""
    if not config.getoption("--agent-health", default=False):
        return

    if not _session_results:
        return

    terminalreporter.section("agent-health summary")
    terminalreporter.line(format_summary(_session_results))

    # Count by verdict
    from pytest_agent_health.policy import Verdict
    fails = [r for r in _session_results if r.verdict == Verdict.FAIL]
    warns = [r for r in _session_results if r.verdict == Verdict.WARN]

    if fails:
        terminalreporter.line(f"\nFailed checks ({len(fails)}):")
        from pytest_agent_health.reporting import format_verdict
        for item in fails:
            for line in format_verdict(item).split("\n"):
                terminalreporter.line(f"  {line}")

    if warns:
        terminalreporter.line(f"\nWarnings ({len(warns)}):")
        from pytest_agent_health.reporting import format_verdict
        for item in warns:
            for line in format_verdict(item).split("\n"):
                terminalreporter.line(f"  {line}")
