"""
reporting.py — CI output formatting.

Formats diagnosis results into human-readable CI output with:
  - FAIL/WARN/PASS verdict
  - risk/info tags on degradation indicators
  - retryable/structural tags on failure patterns
  - Actionable guidance
"""

from __future__ import annotations

from pytest_agent_health.policy import VerdictItem, Verdict, classify_indicator


# Retryable patterns (from debugger's integrations/langgraph.py)
RETRYABLE_PATTERNS = frozenset({
    "agent_tool_call_loop",
    "failed_termination",
    "premature_termination",
    "premature_model_commitment",
    "incorrect_output",
    "tool_result_misinterpretation",
})


def _failure_tag(failure_id: str) -> str:
    """Return [retryable] or [structural] tag for a failure pattern."""
    if failure_id in RETRYABLE_PATTERNS:
        return "retryable"
    return "structural"


def format_verdict(item: VerdictItem) -> str:
    """Format a VerdictItem as multi-line CI output.

    Example output:
        FAIL: premature_termination [retryable]
          → Agent stopped before completing the task
        WARN [risk]: alignment_score 0.38 < 0.5
          → Output may not match user intent
        WARN [info]: observation_coverage low (5 signals missing)
          → Diagnosis based on limited telemetry
    """
    lines = []

    # --- Verdict header ---
    icon = {"FAIL": "❌", "WARN": "⚠️", "PASS": "✅"}.get(item.verdict.value, "")
    lines.append(f"{icon} {item.verdict.value}: execution_quality={item.status}")

    # --- Override notice ---
    if item.override_pattern:
        lines.append(
            f"  ↳ Forced FAIL by --agent-health-fail-on "
            f"(matched: {item.override_pattern})"
        )

    # --- Failure patterns ---
    for fid in item.failures:
        tag = _failure_tag(fid)
        lines.append(f"  FAILED: {fid} [{tag}]")
        if tag == "retryable":
            lines.append(
                "    → In production, consider create_health_check() "
                "for automatic recovery"
            )
        else:
            lines.append(
                "    → Requires prompt or configuration change"
            )

    # --- Degradation indicators ---
    for ind in item.indicators:
        level = classify_indicator(ind)
        signal = ind.get("signal", "unknown")
        value = ind.get("value", "")
        concern = ind.get("concern", "")

        lines.append(f"  WARN [{level}]: {signal} = {value}")
        if concern:
            lines.append(f"    → {concern}")

    # --- Summary guidance ---
    if not item.failures and not item.indicators:
        if item.verdict == Verdict.PASS_:
            lines.append("  No issues detected.")

    return "\n".join(lines)


def format_summary(items: list[VerdictItem]) -> str:
    """Format a summary line for multiple checks.

    Example:
        agent-health: 3 checks — 1 FAIL, 1 WARN, 1 PASS
    """
    total = len(items)
    fails = sum(1 for i in items if i.verdict == Verdict.FAIL)
    warns = sum(1 for i in items if i.verdict == Verdict.WARN)
    passes = sum(1 for i in items if i.verdict == Verdict.PASS_)

    parts = []
    if fails:
        parts.append(f"{fails} FAIL")
    if warns:
        parts.append(f"{warns} WARN")
    if passes:
        parts.append(f"{passes} PASS")

    return f"agent-health: {total} checks — {', '.join(parts)}"
