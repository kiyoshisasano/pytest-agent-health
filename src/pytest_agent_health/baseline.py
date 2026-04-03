"""
baseline.py — Regression detection via baseline comparison.

Stores diagnosis snapshots from each CI run and compares against
the previous baseline to detect regressions:
  - New failure patterns that didn't exist before
  - Status degradation (healthy → degraded, degraded → failed)
  - New risk indicators appearing

This is the capability that makes pytest-agent-health more than a
diagnose() wrapper — baseline comparison requires CI history and
cannot be done with a single diagnose() call.

Storage format:
    .agent-health/
        test_agent_booking.json
        test_agent_search.json
        ...

Each file contains the snapshot from the last run for that test.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path


# ---------------------------------------------------------------------------
# Snapshot: what we store per test
# ---------------------------------------------------------------------------

def extract_snapshot(diagnosis_result: dict) -> dict:
    """Extract a minimal snapshot from a diagnosis result.

    Stores only what's needed for regression comparison.
    No raw logs, no telemetry — just the diagnosis fingerprint.
    """
    summary = diagnosis_result.get("summary", {})
    eq = summary.get("execution_quality", {})
    matcher_output = diagnosis_result.get("matcher_output", [])

    return {
        "status": eq.get("status", "unknown"),
        "failure_ids": sorted(
            m["failure_id"] for m in matcher_output
            if m.get("diagnosed", True)
        ),
        "indicators": [
            {
                "signal": ind.get("signal", ""),
                "value": ind.get("value"),
                "concern": ind.get("concern", ""),
            }
            for ind in eq.get("indicators", [])
        ],
        "root_cause": summary.get("root_cause", "unknown"),
    }


# ---------------------------------------------------------------------------
# Regression detection
# ---------------------------------------------------------------------------

_STATUS_RANK = {"healthy": 0, "degraded": 1, "failed": 2, "unknown": -1}


@dataclass
class RegressionResult:
    """Result of comparing current run against baseline."""
    has_regression: bool = False
    new_failures: list[str] = field(default_factory=list)
    status_change: str | None = None  # e.g. "healthy → degraded"
    new_indicators: list[str] = field(default_factory=list)
    baseline_existed: bool = False


def compare_to_baseline(
    current: dict,
    baseline: dict,
) -> RegressionResult:
    """Compare current snapshot against baseline.

    Detects:
      - New failure patterns not in baseline
      - Status degradation (healthy→degraded, degraded→failed)
      - New risk indicators not in baseline

    Does NOT flag:
      - Failures that existed in baseline (known issues)
      - Status improvement (failed→healthy)
      - Indicators that existed before

    Args:
        current: Snapshot from current run.
        baseline: Snapshot from previous run.

    Returns:
        RegressionResult with regression details.
    """
    result = RegressionResult(baseline_existed=True)

    # 1. New failure patterns
    baseline_failures = set(baseline.get("failure_ids", []))
    current_failures = set(current.get("failure_ids", []))
    new = current_failures - baseline_failures
    if new:
        result.new_failures = sorted(new)
        result.has_regression = True

    # 2. Status degradation
    baseline_status = baseline.get("status", "unknown")
    current_status = current.get("status", "unknown")
    baseline_rank = _STATUS_RANK.get(baseline_status, -1)
    current_rank = _STATUS_RANK.get(current_status, -1)
    if current_rank > baseline_rank and baseline_rank >= 0:
        result.status_change = f"{baseline_status} → {current_status}"
        result.has_regression = True

    # 3. New indicators
    baseline_signals = {
        ind.get("signal", "") for ind in baseline.get("indicators", [])
    }
    current_signals = {
        ind.get("signal", "") for ind in current.get("indicators", [])
    }
    new_signals = current_signals - baseline_signals
    if new_signals:
        result.new_indicators = sorted(new_signals)
        result.has_regression = True

    return result


# ---------------------------------------------------------------------------
# Baseline storage
# ---------------------------------------------------------------------------

class BaselineStore:
    """Read/write baseline snapshots to disk.

    Each test gets its own JSON file under the baseline directory.
    File names are derived from the test node ID.
    """

    def __init__(self, directory: str | Path = ".agent-health"):
        self.directory = Path(directory)

    def _key_to_path(self, test_id: str) -> Path:
        """Convert a test node ID to a file path.

        Sanitizes the ID to be filesystem-safe:
          test_demo.py::TestClass::test_method → test_demo__TestClass__test_method.json
        """
        safe = test_id.replace("::", "__").replace("/", "__").replace("\\", "__")
        # Remove characters that are problematic on Windows
        safe = "".join(c for c in safe if c.isalnum() or c in "_-.")
        return self.directory / f"{safe}.json"

    def load(self, test_id: str) -> dict | None:
        """Load baseline for a test. Returns None if not found."""
        path = self._key_to_path(test_id)
        if not path.exists():
            return None
        try:
            with open(path, encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            return None

    def save(self, test_id: str, snapshot: dict) -> None:
        """Save baseline for a test."""
        self.directory.mkdir(parents=True, exist_ok=True)
        path = self._key_to_path(test_id)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(snapshot, f, indent=2, ensure_ascii=False)

    def exists(self, test_id: str) -> bool:
        """Check if baseline exists for a test."""
        return self._key_to_path(test_id).exists()