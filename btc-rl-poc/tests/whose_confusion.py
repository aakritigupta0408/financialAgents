"""Identify which arm's gated confusion matches given counts."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
kb = [json.loads(l) for l in
      (ROOT / "results" / "kalshi_binary_log.jsonl").open()]
TAUS = {"kb2": .60, "kb3": .78, "kb4": .79}
target = (151, 34, 160, 10)
for v, tau in TAUS.items():
    rows = [r for r in kb if r.get("variant") == v
            and r.get("actual") is not None
            and max(r["p_up"], 1 - r["p_up"]) >= tau]
    tu = sum(1 for r in rows if r["call"] == 1 and r["actual"] == 1)
    fu = sum(1 for r in rows if r["call"] == 1 and r["actual"] == 0)
    td = sum(1 for r in rows if r["call"] == 0 and r["actual"] == 0)
    fd = sum(1 for r in rows if r["call"] == 0 and r["actual"] == 1)
    mark = "  <-- MATCH" if (tu, fu, td, fd) == target else ""
    print(f"{v}: TU {tu} FU {fu} TD {td} FD {fd}{mark}")
