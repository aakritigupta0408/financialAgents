"""One-shot: persist the A3 2-D surface results into legacy_mine."""
import json
import time
from pathlib import Path

p = Path(__file__).resolve().parent.parent / "results" / \
    "legacy_mine.json"
d = json.load(p.open())
d["a3_surface"] = {
    "generated_ts": int(time.time()),
    "design": "one point per confident-call window (kb2>=0.75, "
    "6-13m) at its deepest valid dip (kb2 floor 0.65); kb9 "
    "minute-matched as independent stability reference; netEV at "
    "modeled dip ask",
    "cells": {
        "drop>=10c_kb9stable(<5pp)": {
            "n": 10, "win": 0.60, "avg_drop_c": 20.0,
            "net_ev_per_$1": -0.079,
            "read": "THE ELEGANT O_t CELL FAILS — model-stable deep "
            "dips LOSE; kb9 stability was false reassurance"},
        "drop>=10c_kb9det(>=5pp)": {
            "n": 25, "win": 0.76, "avg_drop_c": 19.1,
            "net_ev_per_$1": 0.184,
            "read": "best cell: both moved, market moved MORE"},
        "drop5-10c_kb9stable": {"n": 15, "win": 0.80,
                                "net_ev_per_$1": 0.063},
        "all_dips>=10c": {"n": 35, "win": 0.714,
                          "net_ev_per_$1": 0.109},
        "baseline_buy_at_call": {"n": 126, "win": 0.849,
                                 "net_ev_per_$1": 0.085}},
    "kb2_surface_note": "with kb2 as fair, the ideal cell is EMPTY "
    "by construction (kb2 is market-anchored); its deep-dip cell: "
    "n=23, win 78%, avg 12.6c",
    "verdict": "A3-v1 pre-registered as the SIMPLE dip rule "
    "(A3_SPEC.yaml); the stability-gate variant COUNTERINDICATED "
    "(n=10, -7.9%) — watch, not gate; all n tiny; forward tape-v2 "
    "judges."}
json.dump(d, p.open("w"), indent=1)
print("surface saved")
