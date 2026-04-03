# pytest-agent-health

Catch silent agent failures in CI. Behavior lint for LLM agents.

```bash
pip install pytest-agent-health
```

```python
def test_agent_answers_correctly(agent_health):
    raw_log = my_agent.run("What was Q3 revenue?")
    agent_health.check(raw_log, adapter="langchain")
    # → FAIL if execution failed or regression detected
    # → WARN if quality concerns detected
    # → PASS if healthy
```

---

## The Problem

Traditional tests can't catch silent agent failures:

```python
def test_agent():
    response = agent.run("Book me a flight to LAX")
    assert response is not None  # PASSES — but agent just said "I can help!"
```

The agent produced a response, so assertions pass. But it never actually booked anything.

Worse: the agent worked yesterday but silently broke today. The same test passes both times because `assert response is not None` doesn't check behavior — only existence.

pytest-agent-health solves both problems: it detects silent failures by analyzing execution traces, and it catches regressions by comparing against previous CI runs.

---

## How It Works

```
agent log → diagnose() → execution_quality → CI policy → FAIL / WARN / PASS
                (debugger)                     (plugin)
                                                  ↕
                                           baseline comparison
                                          (regression detection)
```

The plugin wraps [agent-failure-debugger](https://github.com/kiyoshisasano/agent-failure-debugger), which detects 17 failure patterns using deterministic causal analysis (no ML). The CI policy layer converts diagnosis results into test verdicts. Baseline comparison detects regressions across CI runs — a capability that single-run diagnosis cannot provide.

---

## Usage

### Single-run check

```python
def test_agent_completes_task(agent_health):
    raw_log = my_agent.run("What was Q3 revenue?")
    agent_health.check(raw_log, adapter="langchain")
```

If the agent's execution is `failed` or `degraded` with risk indicators, the test fails with a diagnostic report:

```
❌ FAIL: execution_quality=degraded
  FAILED: incorrect_output [retryable]
    → In production, consider create_health_check() for automatic recovery
  WARN [risk]: response.alignment_score = 0.12
    → output alignment with user intent is weak
```

### Regression detection (automatic)

When `--agent-health-update-baseline` is set, each `check()` saves a diagnosis snapshot. On subsequent runs, the plugin automatically compares against the baseline and fails if regressions are detected:

```
🔄 REGRESSION DETECTED (compared to baseline):
  New failures: incorrect_output
  Status change: healthy → degraded
  New indicators: response.alignment_score, grounding.tool_provided_data
```

Regressions are detected when:
- New failure patterns appear that didn't exist in the baseline
- Execution status degrades (healthy → degraded, degraded → failed)
- New risk indicators appear

Workflow:

```bash
# First run: establish baselines
pytest --agent-health --agent-health-update-baseline

# Subsequent runs: detect regressions automatically
pytest --agent-health

# After fixing regressions: update baselines
pytest --agent-health --agent-health-update-baseline
```

Baselines are stored in `.agent-health/` (one JSON per test). Commit this directory to git to share baselines across CI runs.

### Multi-run stability

```python
def test_agent_consistency(agent_health):
    logs = [my_agent.run("Q3 revenue?") for _ in range(3)]
    agent_health.compare(logs, adapter="langchain", min_agreement=1.0)
    # → FAIL if root_cause_agreement < 1.0
```

### Differential diagnosis

```python
def test_no_new_failures(agent_health):
    success_logs = load_baseline_logs()
    current_logs = [my_agent.run(q) for q in test_queries]
    agent_health.diff(success_logs, current_logs, adapter="langchain")
    # → FAIL if new failure patterns appear
```

---

## CI Policy

The plugin applies a CI-specific policy on top of execution quality:

### Default

| Condition | Verdict | Rationale |
|---|---|---|
| Regression detected | **FAIL** | New failures or status degradation vs baseline |
| `failed` | **FAIL** | Task incomplete, error, or silent exit |
| `degraded` | **WARN** | Output produced but quality concerns detected |
| `healthy` | **PASS** | No issues |

### Strict (`--agent-health-strict`)

| Condition | Verdict | Rationale |
|---|---|---|
| Regression detected | **FAIL** | New failures or status degradation vs baseline |
| `failed` | **FAIL** | Task incomplete, error, or silent exit |
| `degraded` + risk indicators | **FAIL** | Weak alignment, no tool data, hallucination signals |
| `degraded` + info indicators only | **WARN** | Diagnostic limitations, not agent failure |
| `healthy` | **PASS** | No issues |

### Indicator classification

Degradation indicators are classified as risk (agent quality problem) or info (diagnostic limitation):

| Tag | Indicators | Meaning |
|---|---|---|
| `[risk]` | alignment_score, tool_provided_data, tool_result_diversity, expansion_ratio | Output may be wrong or unreliable |
| `[info]` | observation_coverage, unmodeled_failure, conflicting_signals | Diagnosis has limited confidence |

### Pattern override

Force specific failure patterns to always fail, regardless of execution quality:

```bash
pytest --agent-health --agent-health-fail-on=premature_termination,context_truncation_loss
```

---

## CLI Options

```bash
pytest --agent-health                          # Enable the plugin
pytest --agent-health --agent-health-strict    # Strict: degraded+risk = FAIL
pytest --agent-health --agent-health-fail-on=premature_termination
pytest --agent-health --agent-health-update-baseline   # Save current as baseline
pytest --agent-health --agent-health-no-baseline       # Skip regression detection
pytest --agent-health --agent-health-baseline-dir=path # Custom baseline directory
```

---

## Marker

Override policy per-test:

```python
@pytest.mark.agent_health(strict=True)
def test_critical_flow(agent_health):
    agent_health.check(log, adapter="langchain")

@pytest.mark.agent_health(fail_on=["premature_termination"])
def test_must_not_terminate_early(agent_health):
    agent_health.check(log, adapter="langchain")
```

---

## CI Output

The terminal summary shows all checks at the end of the test session:

```
============================= agent-health summary =============================
agent-health: 4 checks — 2 FAIL, 2 PASS

Failed checks (2):
  ❌ FAIL: execution_quality=degraded
    WARN [risk]: response.alignment_score = 0.0
      → output alignment with user intent is weak
  ❌ FAIL: execution_quality=degraded
    FAILED: incorrect_output [retryable]
      → In production, consider create_health_check() for automatic recovery
    WARN [risk]: response.alignment_score = 0.12
      → output alignment with user intent is weak
```

Failure patterns are tagged `[retryable]` or `[structural]`. Retryable failures can be automatically recovered in production using [create_health_check()](https://github.com/kiyoshisasano/agent-failure-debugger#self-healing-agent-langgraph). Structural failures require prompt or configuration changes.

---

## Non-determinism

LLM agents are non-deterministic. The same test may produce different results across runs. This is expected behavior, not a plugin bug.

Recommendations for stable CI:

- Set `temperature=0` in your agent configuration
- Use model-specific seed parameters where available
- Use `agent_health.compare()` to explicitly test for consistency
- Accept that some flakiness is inherent to LLM-based systems

---

## Related

| Package | Role |
|---|---|
| [agent-failure-debugger](https://github.com/kiyoshisasano/agent-failure-debugger) | Diagnosis engine (this plugin wraps it) |
| [llm-failure-atlas](https://github.com/kiyoshisasano/llm-failure-atlas) | Failure patterns, signals, matcher |
| [create_health_check()](https://github.com/kiyoshisasano/agent-failure-debugger#self-healing-agent-langgraph) | Runtime self-healing (complementary to CI) |

**CI detects. Production heals.** Use pytest-agent-health to catch failures in CI, and create_health_check() to automatically recover in production.

---

## License

MIT License.