"""Root pytest config. Makes `pytest tests/` actually collect the real guardrail
suite instead of dying on collection.

The tests/ dir mixes true pytest files (25, with `def test_*`) with standalone
analysis/scenario SCRIPTS that happen to be named test_*.py. Several of the scripts
call sys.exit() at MODULE level, which raises SystemExit during pytest collection and
aborts the ENTIRE run (INTERNALERROR, "no tests ran") — the root cause the audit found
for why the guardrails were 'dark' (never executed, so never gating). These scripts are
still runnable directly (`python tests/<name>.py`); they are only excluded from pytest
collection here. See docs/AUDIT_COMPLIANCE.md."""

# script-style test_*.py (no `def test_`; some sys.exit() at import) — NOT pytest-shaped
collect_ignore = [
    "tests/test_a3_scenarios.py",
    "tests/test_agent_firewall.py",
    "tests/test_bankroll_recovery.py",
    "tests/test_m6_interaction.py",
    "tests/test_m6_r1.py",
    "tests/test_m6_r2.py",
    "tests/test_m6_r3.py",
    "tests/test_parity_quantization.py",
]
