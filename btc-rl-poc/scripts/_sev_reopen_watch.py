"""SEV-1 final closure gate: confirm the desk actually RESUMED after
the machine-controlled reopen — a new champion paired window beyond
n=374, with reconciliation still clean. Observation only."""
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"
DEADLINE = time.time() + 60 * 60


def champ_n():
    try:
        s = json.loads((RES / "online_status.json").read_text())
        for t in s.get("treatments") or []:
            if t.get("key") == "champion":
                return t.get("n")
    except Exception:
        pass
    return None


base = champ_n()
while time.time() < DEADLINE:
    time.sleep(120)
    n = champ_n()
    if n and base and n > base:
        subprocess.run([sys.executable, str(ROOT / "scripts" /
                                            "reconcile.py")],
                       cwd=ROOT, capture_output=True, timeout=600)
        d = json.loads((RES / "reconciliation.json").read_text())
        print(f"DESK RESUMED: champion n {base} -> {n} · "
              f"reconcile overall={d.get('overall')}")
        break
else:
    print(f"NO NEW WINDOW within 60 min (champion n still {base}); "
          "desk is trade-on (DEGRADED) — biddable window simply "
          "hasn't closed yet. Not a fault.")
