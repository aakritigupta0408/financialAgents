"""The 10-minute audit chain, as ONE short cron entry.

Why this exists (SEV-2, 2026-08-29): the chain grew to eleven &&-joined
commands on a single crontab line (~2.3 KB). macOS cron silently
rejected the oversized entry after a rewrite, and the entire analytics
layer went stale for ~2.7 h while the 1-minute publisher line kept
working. Lessons encoded here:
  * cron lines stay short — orchestration lives in a script;
  * steps run INDEPENDENTLY (no && chain): one failing emitter logs
    its error and the rest still run;
  * every run stamps a heartbeat line so staleness is detectable.
"""
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
STEPS = [
    "scripts/run_audit.py",
    "scripts/emit_manifest.py",
    "scripts/emit_board.py",
    "scripts/emit_decision_board.py",
    "tests/invariants.py",
    "scripts/emit_registry.py",
    "scripts/emit_world.py",
    "scripts/loss_review.py",
    "scripts/emit_fill_curve.py",
    "scripts/emit_diagnosis.py",
    "scripts/emit_program.py",
    "scripts/emit_execution_ledger.py",
    "scripts/reconcile.py",
    "scripts/emit_readiness.py",
]

print(f"=== audit chain {time.strftime('%Y-%m-%d %H:%M:%S')} ===",
      flush=True)
failures = 0
for step in STEPS:
    p = subprocess.run([sys.executable, str(ROOT / step)],
                       capture_output=True, text=True, timeout=300)
    if p.returncode != 0:
        failures += 1
        print(f"STEP FAILED {step} rc={p.returncode}\n"
              f"{(p.stderr or p.stdout)[-500:]}", flush=True)
    else:
        print(p.stdout.strip().splitlines()[-1]
              if p.stdout.strip() else f"{step}: ok", flush=True)
print(f"=== done, {failures} failures ===", flush=True)
