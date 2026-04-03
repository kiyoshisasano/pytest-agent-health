"""
policy.py — CI failure policy layer.

Maps execution_quality output from agent-failure-debugger to CI verdicts.

Architecture:
    execution_quality (debugger) → policy (this module) → FAIL / WARN / PASS

The debugger produces a diagnosis with execution_quality status
(healthy/degraded/failed) and degradation indicators. This module
applies a CI-specific policy to convert those into actionable verdicts.

Two policy modes:
    default:
        failed → FAIL
        degraded → WARN
        healthy → PASS

    strict (--agent-health-strict):
        failed → FAIL
        degraded + risk indicators → FAIL
        degraded + info indicators only → WARN
        healthy → PASS

Additionally, --agent-health-fail-on overrides the policy: if any
specified failure pattern is detected, the verdict is FAIL regardless
of execution_quality status.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class Verdict(Enum):
    """CI verdict for a single check."""
    FAIL = "FAIL"
    WARN = "WARN"
    PASS_ = "PASS"  # trailing underscore to avoid shadowing builtins


# ---------------------------------------------------------------------------
# Indicator classification
# ---------------------------------------------------------------------------

# Risk indicators: signal genuine quality problems.
# These represent conditions where the agent's output is likely wrong
# or unreliable, even if it produced a response.
RISK_SIGNALS = frozenset({
    "response.alignment_score",
    "grounding.tool_provided_data",
    "grounding.tool_result_diversity",
    "grounding.expansion_ratio",
})

# Info indicators: signal diagnostic limitations, not agent failures.
# These mean the diagnosis itself has limited confidence, but do not
# indicate that the agent's output is wrong.
INFO_SIGNALS = frozenset({
    "observation_coverage",
    "unmodeled_failure",
    "conflicting_signals",
})


def classify_indicator(indicator: dict) -> str:
    """Classify a degradation indicator as 'risk' or 'info'.

    Args:
        indicator: A dict with at least a 'signal' key.

    Returns:
        'risk' or 'info'.
    """
    signal = indicator.get("signal", "")
    if signal in RISK_SIGNALS:
        return "risk"
    if signal in INFO_SIGNALS:
        return "info"
    # Unknown signals default to info (conservative — don't FAIL on unknowns)
    return "info"


# ---------------------------------------------------------------------------
# Verdict item (single check result)
# ---------------------------------------------------------------------------

@dataclass
class VerdictItem:
    """Result of applying CI policy to a single diagnosis."""
    verdict: Verdict
    status: str  # execution_quality status: healthy/degraded/failed
    failures: list[str] = field(default_factory=list)
    indicators: list[dict] = field(default_factory=list)
    override_pattern: str | None = None  # set if fail-on triggered

    @property
    def is_fail(self) -> bool:
        return self.verdict == Verdict.FAIL

    @property
    def is_warn(self) -> bool:
        return self.verdict == Verdict.WARN


# ---------------------------------------------------------------------------
# Policy application
# ---------------------------------------------------------------------------

def apply_policy(
    diagnosis_result: dict,
    *,
    strict: bool = False,
    fail_on: frozenset[str] | None = None,
) -> VerdictItem:
    """Apply CI policy to a diagnosis result.

    Args:
        diagnosis_result: Output of agent_failure_debugger.diagnose().
        strict: If True, degraded + risk indicators → FAIL.
            Default False: degraded → WARN regardless of indicator type.
        fail_on: Set of failure pattern IDs that trigger unconditional FAIL.
            Overrides execution_quality status.

    Returns:
        VerdictItem with verdict, status, and supporting details.
    """
    summary = diagnosis_result.get("summary", {})
    eq = summary.get("execution_quality", {})
    status = eq.get("status", "healthy")
    indicators = eq.get("indicators", [])

    # Extract detected failure IDs
    matcher_output = diagnosis_result.get("matcher_output", [])
    failure_ids = [m["failure_id"] for m in matcher_output if m.get("diagnosed", True)]

    # --- Override: fail-on specific patterns ---
    if fail_on:
        matched = fail_on & set(failure_ids)
        if matched:
            return VerdictItem(
                verdict=Verdict.FAIL,
                status=status,
                failures=failure_ids,
                indicators=indicators,
                override_pattern=sorted(matched)[0],
            )

    # --- Policy: failed → FAIL always ---
    if status == "failed":
        return VerdictItem(
            verdict=Verdict.FAIL,
            status=status,
            failures=failure_ids,
            indicators=indicators,
        )

    # --- Policy: degraded ---
    if status == "degraded":
        if strict:
            # Check if any risk indicators are present
            has_risk = any(
                classify_indicator(ind) == "risk" for ind in indicators
            )
            if has_risk:
                return VerdictItem(
                    verdict=Verdict.FAIL,
                    status=status,
                    failures=failure_ids,
                    indicators=indicators,
                )

        # Relaxed, or strict but info-only → WARN
        return VerdictItem(
            verdict=Verdict.WARN,
            status=status,
            failures=failure_ids,
            indicators=indicators,
        )

    # --- Policy: healthy → PASS ---
    return VerdictItem(
        verdict=Verdict.PASS_,
        status=status,
        failures=failure_ids,
        indicators=indicators,
    )
