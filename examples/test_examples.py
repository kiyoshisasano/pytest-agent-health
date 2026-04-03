"""
Example tests for pytest-agent-health.

These examples demonstrate three failure patterns that CI should catch
but traditional assertion-based tests miss:

1. premature_termination — Agent stops before completing the task
2. context_truncation_loss — Long input causes agent to ignore key parts
3. tool_result_misinterpretation — Agent misreads tool output

Run:
    pip install pytest-agent-health
    pytest examples/ --agent-health -v
"""

import pytest


# ---------------------------------------------------------------------------
# 1. Premature termination
#
# The agent produces a response, so `assert response is not None` passes.
# But the agent stopped before actually completing the task.
# Traditional tests can't catch this — agent_health can.
# ---------------------------------------------------------------------------

class TestPrematureTermination:
    """Detect when an agent stops before completing the task."""

    def test_agent_completes_booking(self, agent_health):
        """Agent should book a flight, not just acknowledge the request."""
        raw_log = {
            "inputs": {"query": "Book me a flight from NYC to LAX for March 20"},
            "outputs": {"response": "I can help you with that!"},
            "steps": [
                {
                    "type": "llm",
                    "outputs": {"text": "I can help you with that!"},
                },
                # No tool calls — agent responded without doing anything
            ],
        }

        # Traditional test: assert response is not None → PASSES (wrong)
        assert raw_log["outputs"]["response"] is not None

        # agent_health: detects premature_termination → FAILS (correct)
        agent_health.check(raw_log, adapter="langchain")


# ---------------------------------------------------------------------------
# 2. Context truncation loss
#
# Short inputs work fine. Long inputs cause the agent to silently
# drop important constraints. This only manifests in production
# with real-length documents.
# ---------------------------------------------------------------------------

class TestContextTruncation:
    """Detect when long context causes the agent to miss instructions."""

    def test_agent_follows_all_constraints(self, agent_health):
        """Agent should respect all constraints, even in long prompts."""
        # Simulate a long prompt where the critical constraint is buried
        padding = "Additional context. " * 200  # ~4000 chars of padding
        raw_log = {
            "inputs": {
                "query": (
                    f"Analyze this report. {padding}"
                    "IMPORTANT: Only include data from Q3 2024. "
                    "Do not include any Q2 data."
                ),
            },
            "outputs": {
                "response": (
                    "Based on the report, Q2 revenue was $3.1M and "
                    "Q3 revenue was $4.2M, showing 35% growth."
                ),
            },
            "steps": [
                {
                    "type": "llm",
                    "outputs": {
                        "text": (
                            "Based on the report, Q2 revenue was $3.1M and "
                            "Q3 revenue was $4.2M, showing 35% growth."
                        ),
                    },
                },
            ],
            "feedback": {
                "user_correction": "I said Q3 only, you included Q2 data.",
            },
        }

        # Traditional test: response contains "Q3" → PASSES (wrong)
        assert "Q3" in raw_log["outputs"]["response"]

        # agent_health: detects the constraint violation
        agent_health.check(raw_log, adapter="langchain")


# ---------------------------------------------------------------------------
# 3. Tool result misinterpretation
#
# The most insidious failure: the agent calls the right tool,
# gets correct data back, but misinterprets or ignores it.
# Human review often misses this because the response looks plausible.
# CI is the only reliable place to catch it.
# ---------------------------------------------------------------------------

class TestToolResultMisinterpretation:
    """Detect when the agent ignores or misreads tool output.

    Note: The current matcher uses heuristic signals (alignment score,
    expansion ratio) — not semantic comparison between tool output and
    response. This means subtle misinterpretations (e.g., $198 → $245)
    may not be caught unless they also affect alignment or grounding
    signals. This is a known limitation of the deterministic approach.

    This example shows the pattern for testing it. When user_correction
    feedback is available, detection improves significantly.
    """

    def test_agent_uses_tool_data(self, agent_health):
        """Agent should base its answer on what the tool returned."""
        raw_log = {
            "inputs": {"query": "What is Apple's current stock price?"},
            "outputs": {
                "response": "Apple's stock price is $245.30, up 2.1% today.",
            },
            "steps": [
                {
                    "type": "llm",
                    "outputs": {"text": "Let me look that up."},
                },
                {
                    "type": "tool",
                    "name": "get_stock_price",
                    "inputs": {"symbol": "AAPL"},
                    # Tool returned $198.50, but agent says $245.30
                    "outputs": {"price": "$198.50", "change": "-0.3%"},
                    "error": None,
                },
                {
                    "type": "llm",
                    "outputs": {
                        "text": "Apple's stock price is $245.30, up 2.1% today.",
                    },
                },
            ],
            # Without feedback, this subtle misinterpretation may not
            # be detected. Adding user_correction improves detection:
            "feedback": {
                "user_correction": "The tool said $198.50, not $245.30.",
            },
        }

        # agent_health: checks for misinterpretation signals
        agent_health.check(raw_log, adapter="langchain")


# ---------------------------------------------------------------------------
# 4. Healthy baseline (control case)
# ---------------------------------------------------------------------------

class TestHealthyBaseline:
    """Confirm that healthy runs pass without warnings."""

    def test_correct_tool_usage(self, agent_health):
        """Agent correctly uses tool data in its response."""
        raw_log = {
            "inputs": {"query": "What was Q3 revenue?"},
            "outputs": {
                "response": (
                    "Q3 revenue was $4.2M based on the latest "
                    "earnings report."
                ),
            },
            "steps": [
                {
                    "type": "tool",
                    "name": "search_earnings",
                    "inputs": {"quarter": "Q3"},
                    "outputs": {
                        "revenue": "$4.2M",
                        "source": "10-Q filing",
                    },
                    "error": None,
                },
                {
                    "type": "llm",
                    "outputs": {
                        "text": (
                            "Q3 revenue was $4.2M based on the latest "
                            "earnings report."
                        ),
                    },
                },
            ],
        }

        result = agent_health.check(raw_log, adapter="langchain")
        assert result["summary"]["execution_quality"]["status"] == "healthy"


# ---------------------------------------------------------------------------
# 5. Multi-run stability
# ---------------------------------------------------------------------------

class TestStability:
    """Demonstrate compare() for consistency checking."""

    def test_agent_gives_consistent_answers(self, agent_health):
        """Same input should produce consistent diagnosis results."""
        log = {
            "inputs": {"query": "What was Q3 revenue?"},
            "outputs": {
                "response": "Q3 revenue was $4.2M.",
            },
            "steps": [
                {
                    "type": "tool",
                    "name": "search_earnings",
                    "inputs": {"quarter": "Q3"},
                    "outputs": {"revenue": "$4.2M"},
                    "error": None,
                },
                {
                    "type": "llm",
                    "outputs": {"text": "Q3 revenue was $4.2M."},
                },
            ],
        }

        # In real usage, these would be logs from separate agent runs
        stability = agent_health.compare(
            [log, log, log],
            adapter="langchain",
            min_agreement=1.0,
        )
        assert stability["stability"]["root_cause_agreement"] == 1.0
