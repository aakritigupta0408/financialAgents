"""Passive watcher: waits for the registered F1-v2.1.1 prospective
rerun time (2026-09-05 05:10 UTC), then runs the COMPLETE gate rerun
(emit_f1_gate + audit_missingness, per PM: 'not just E') and prints
both verdicts. Observation only; ratification stays with the PM."""
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
from audit_missingness import RERUN_NOT_BEFORE_TS

while time.time() < RERUN_NOT_BEFORE_TS:
    time.sleep(min(1800, max(60, RERUN_NOT_BEFORE_TS - time.time())))

for script in ("emit_f1_gate.py", "audit_missingness.py"):
    subprocess.run([sys.executable, str(ROOT / "scripts" / script)],
                   cwd=ROOT, capture_output=True, timeout=1200)
gate = json.loads((ROOT / "results" /
                   "f1_capture_qualification.json").read_text())
miss = json.loads((ROOT / "results" /
                   "f1_missingness_audit.json").read_text())
sh = gate.get("proposed_v2_1_stitched", {})
print(f"F1 PROSPECTIVE RERUN: missingness={miss['verdict']} "
      f"(episodes {miss['n_episodes']}, med_pre_pct "
      f"{miss['median_pre_episode_vol_pct']}) | v2.1 shadow "
      f"would_pass={sh.get('would_pass')} "
      f"failing={sh.get('failing')} | governing v2 "
      f"verdict={gate['verdict']}")
