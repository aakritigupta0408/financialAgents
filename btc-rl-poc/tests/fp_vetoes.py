"""False-positive reduction: measure each veto's effect on kb2/kb4's
gated calls — FPs removed vs correct calls sacrificed.
Vetoes: hot hours (18-20,01 PT), early phase (>10m), barrier
contradiction (window occupancy against the call), whipsaw (pf[1] high).
"""
import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

PT = ZoneInfo("America/Los_Angeles")
ROOT = Path(__file__).resolve().parent.parent
kb = [json.loads(l) for l in
      (ROOT / "results" / "kalshi_binary_log.jsonl").open()]
TAUS = {"kb2": .60, "kb4": .79}


def stats(sel):
    fp = sum(1 for r in sel if not r["hit"])
    return len(sel), fp


VETOES = {
    "hot hours": lambda r: datetime.fromtimestamp(
        r["made_ts"], PT).hour in (18, 19, 20, 1),
    "early >10m": lambda r: r["mins_left"] > 10,
    "barrier contra": lambda r: (r.get("pf") or [0])[0] is not None
        and len(r.get("pf") or []) >= 1
        and ((r["pf"][0] + 0.5) if r["call"] else (0.5 - r["pf"][0])) < 0.35,
    "whipsaw": lambda r: len(r.get("pf") or []) >= 2 and r["pf"][1] >= 0.75,
}

for v, tau in TAUS.items():
    rows = [r for r in kb if r.get("variant") == v
            and r.get("actual") is not None
            and max(r["p_up"], 1 - r["p_up"]) >= tau]
    n0, fp0 = stats(rows)
    print(f"\n== {v} gated (tau {tau}): {n0} calls, {fp0} FPs "
          f"({fp0/n0:.1%} FP rate) ==")
    print(f"{'veto':16s} {'kills':>6s} {'FPs cut':>8s} {'TPs lost':>9s} "
          f"{'new FP rate':>11s} {'efficiency':>10s}")
    for name, f in VETOES.items():
        vetoed = [r for r in rows if f(r)]
        kept = [r for r in rows if not f(r)]
        nk, fpk = stats(kept)
        cut = fp0 - fpk
        lost = (n0 - fp0) - (nk - fpk)
        eff = cut / max(1, lost)
        print(f"{name:16s} {len(vetoed):6d} {cut:8d} {lost:9d} "
              f"{fpk/max(1,nk):11.1%} {eff:10.2f}")
    both = [r for r in rows
            if not any(f(r) for f in VETOES.values())]
    nb, fpb = stats(both)
    print(f"{'ALL vetoes':16s} {n0-nb:6d} {fp0-fpb:8d} "
          f"{(n0-fp0)-(nb-fpb):9d} {fpb/max(1,nb):11.1%}")
