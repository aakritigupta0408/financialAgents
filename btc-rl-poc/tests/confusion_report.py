"""Confusion matrices (true/false x up/down) for kb2, kb3 at their gates
and for the agreement sets."""
import json
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
kb = [json.loads(l) for l in
      (ROOT / "results" / "kalshi_binary_log.jsonl").open()]


def conf(rows):
    tu = sum(1 for r in rows if r["call"] == 1 and r["actual"] == 1)
    fu = sum(1 for r in rows if r["call"] == 1 and r["actual"] == 0)
    td = sum(1 for r in rows if r["call"] == 0 and r["actual"] == 0)
    fd = sum(1 for r in rows if r["call"] == 0 and r["actual"] == 1)
    return tu, fu, td, fd


def show(name, rows):
    tu, fu, td, fd = conf(rows)
    n = tu + fu + td + fd
    print(f"{name:26s} n={n:4d}  TRUE UP {tu:4d}  FALSE UP {fu:3d}  "
          f"TRUE DOWN {td:4d}  FALSE DOWN {fd:3d}  "
          f"| UP prec {tu/max(1,tu+fu):.3f} rec {tu/max(1,tu+fd):.3f} "
          f"| DOWN prec {td/max(1,td+fd):.3f} rec {td/max(1,td+fu):.3f}")


TAU = {"kb2": .62, "kb3": .82}
for v in ("kb2", "kb3"):
    rows = [r for r in kb if r.get("variant") == v
            and r.get("actual") is not None]
    show(f"{v} all calls", rows)
    show(f"{v} gated (tau {TAU[v]})",
         [r for r in rows if max(r["p_up"], 1 - r["p_up"]) >= TAU[v]])

by = defaultdict(dict)
for r in kb:
    if r.get("variant") in ("kb2", "kb3") and r.get("actual") is not None:
        by[(r["ticker"], r["made_ts"])][r["variant"]] = r
pairs = [v for v in by.values() if "kb2" in v and "kb3" in v]
show("kb2&kb3 agree",
     [v["kb2"] for v in pairs if v["kb2"]["call"] == v["kb3"]["call"]])
show("agree + both confident",
     [v["kb2"] for v in pairs if v["kb2"]["call"] == v["kb3"]["call"]
      and max(v["kb2"]["p_up"], 1 - v["kb2"]["p_up"]) >= .62
      and max(v["kb3"]["p_up"], 1 - v["kb3"]["p_up"]) >= .82])
