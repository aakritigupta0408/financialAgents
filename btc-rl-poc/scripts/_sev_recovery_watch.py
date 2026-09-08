"""SEV-1 recovery watcher: waits for the daemon (fixed code) to
late-settle the two zombies, then runs reconcile and reports the
verdict. Observation + one sanctioned reconcile run; no mutation."""
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ZOMBIES = (("pt_trades.jsonl", "KXBTC15M-26AUG311800-00"),
           ("pt3_trades.jsonl", "KXBTC15M-26SEP062345-45"))
DEADLINE = time.time() + 55 * 60

def settled(fname, tick):
    try:
        for l in (ROOT / "results" / fname).open():
            r = json.loads(l)
            if r.get("ticker") == tick:
                return r.get("actual") is not None, \
                    r.get("late_settle_ts") is not None, r.get("pnl_c")
    except Exception:
        pass
    return False, False, None

while time.time() < DEADLINE:
    states = [settled(f, t) for f, t in ZOMBIES]
    if all(s[0] for s in states):
        print("ZOMBIES SETTLED:",
              [(t, "late" if s[1] else "NORMAL(!)", s[2])
               for (f, t), s in zip(ZOMBIES, states)])
        r = subprocess.run([sys.executable,
                            str(ROOT / "scripts" / "reconcile.py")],
                           capture_output=True, text=True, cwd=ROOT,
                           timeout=600)
        doc = json.loads((ROOT / "results" / "reconciliation.json")
                         .read_text())
        bank = next(c for c in doc["checks"]
                    if c["name"] == "bankroll-conservation")
        print(f"RECONCILE: bankroll-conservation {bank['status']} "
              f"(mismatches {bank['mismatches']}) · overall "
              f"{doc.get('overall')}")
        break
    time.sleep(120)
else:
    print("TIMEOUT: zombies not settled within 75 min — check "
          "daemon.log and by_ts backfill path")
