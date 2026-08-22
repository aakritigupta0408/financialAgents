"""Verify the commit gate fix: a live-anchored row for the CURRENT 5-min
slot must appear in the ledger BEFORE the slot's minute completes —
proving commits no longer wait for the slot's own bucket."""
import json
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
deadline = time.time() + 420
while time.time() < deadline:
    now = int(time.time())
    slot = now // 300 * 300
    if now - slot < 45:  # early in a slot minute — the decisive window
        rows = [json.loads(l) for l in
                (ROOT / "results" / "prediction_log.jsonl").open()]
        hit = [r for r in rows if r["made_ts"] == slot
               and r.get("anchor_src") == "live"]
        if hit:
            print(f"PASS: {len(hit)} live-anchored rows committed "
                  f"{now - slot}s into slot {slot} (before minute end)")
            break
    time.sleep(5)
else:
    print("no early-slot commit observed in 7 min — check daemon logs")
