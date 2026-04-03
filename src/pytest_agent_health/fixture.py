"""
fixture.py — The agent_health pytest fixture.

Provides three methods:
  - check(raw_log, adapter) — single-run diagnosis + CI verdict
  - compare(logs, adapter) — multi-run stability check
  - diff(success_logs, failure_logs, adapter) — differential diagnosis

Each method runs diagnose() internally and applies the CI policy.
On FAIL, pytest.fail() is called. On WARN, pytest.warns() is issued.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import pytest

from pytest_agent_health.policy import (
    apply_policy,
    Verdict,
    VerdictItem,
)
from pytest_agent_health.reporting import format_verdict


@dataclass
class AgentHealthFixture:
    """Fixture object providing agent health check methods.

    Attributes:
        strict: Whether to apply strict policy (default False).
        fail_on: Set of failure patterns that force FAIL.
        results: Accumulated results from all checks in this test.
    """
    strict: bool = False
    fail_on: frozenset[str] = field(default_factory=frozenset)
    results: list[VerdictItem] = field(default_factory=list)

    def check(
        self,
        raw_log: dict,
        adapter: str = "langchain",
        **diagnose_kwargs,
    ) -> dict:
        """Run diagnosis and apply CI policy.

        Args:
            raw_log: Raw log from the agent.
            adapter: Adapter name for diagnosis.
            **diagnose_kwargs: Passed to diagnose().

        Returns:
            The full diagnosis result dict.

        Raises:
            pytest.fail: If verdict is FAIL.
        """
        from agent_failure_debugger.diagnose import diagnose as _diagnose

        result = _diagnose(raw_log, adapter=adapter, **diagnose_kwargs)
        item = apply_policy(
            result,
            strict=self.strict,
            fail_on=self.fail_on,
        )
        self.results.append(item)

        report = format_verdict(item)

        if item.verdict == Verdict.FAIL:
            pytest.fail(f"Agent health check failed:\n{report}")
        elif item.verdict == Verdict.WARN:
            warnings.warn(
                f"Agent health check warning:\n{report}",
                UserWarning,
                stacklevel=2,
            )

        return result

    def compare(
        self,
        raw_logs: list[dict],
        adapter: str = "langchain",
        *,
        min_agreement: float = 1.0,
        **diagnose_kwargs,
    ) -> dict:
        """Run multiple diagnoses and check stability.

        Runs diagnose() on each log, then calls compare_runs()
        to measure consistency. Fails if root_cause_agreement
        is below min_agreement.

        Args:
            raw_logs: List of raw logs (at least 2).
            adapter: Adapter name.
            min_agreement: Minimum root_cause_agreement (default 1.0).
            **diagnose_kwargs: Passed to diagnose().

        Returns:
            The compare_runs() result dict.

        Raises:
            pytest.fail: If agreement is below threshold.
            ValueError: If fewer than 2 logs provided.
        """
        if len(raw_logs) < 2:
            raise ValueError(
                f"compare() requires at least 2 logs, got {len(raw_logs)}"
            )

        from agent_failure_debugger.diagnose import diagnose as _diagnose
        from agent_failure_debugger.reliability import compare_runs

        results = [
            _diagnose(log, adapter=adapter, **diagnose_kwargs)
            for log in raw_logs
        ]

        stability = compare_runs(results)
        agreement = stability.get("stability", {}).get(
            "root_cause_agreement", 0.0
        )

        if agreement < min_agreement:
            pytest.fail(
                f"Agent behavior unstable: "
                f"root_cause_agreement={agreement:.2f} "
                f"< {min_agreement:.2f}\n"
                f"Interpretation: {stability.get('interpretation', '')}"
            )

        return stability

    def diff(
        self,
        success_logs: list[dict],
        failure_logs: list[dict],
        adapter: str = "langchain",
        **diagnose_kwargs,
    ) -> dict:
        """Identify structural differences between success and failure runs.

        Runs diagnose() on each log group, then calls diff_runs().
        Fails if failure-only patterns are found (regressions).

        Args:
            success_logs: List of raw logs from successful runs.
            failure_logs: List of raw logs from failed runs.
            adapter: Adapter name.
            **diagnose_kwargs: Passed to diagnose().

        Returns:
            The diff_runs() result dict.

        Raises:
            pytest.fail: If failure-only patterns exist.
        """
        from agent_failure_debugger.diagnose import diagnose as _diagnose
        from agent_failure_debugger.reliability import diff_runs

        success_results = [
            _diagnose(log, adapter=adapter, **diagnose_kwargs)
            for log in success_logs
        ]
        failure_results = [
            _diagnose(log, adapter=adapter, **diagnose_kwargs)
            for log in failure_logs
        ]

        diff = diff_runs(success_results, failure_results)
        failure_only = diff.get("failure_set_diff", {}).get(
            "failure_only", []
        )

        if failure_only:
            pytest.fail(
                f"Regression detected — failure-only patterns: "
                f"{', '.join(failure_only)}\n"
                f"Hypothesis: {diff.get('hypothesis', '')}"
            )

        return diff
