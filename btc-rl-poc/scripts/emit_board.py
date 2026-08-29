"""B2: emit results/treatments_board.json — the champion/challenger
board snapshot, one Treatment.status() dict per treatment.

Usage:  python3 scripts/emit_board.py
Read-only against live state; writes exactly one file atomically;
idempotent (re-runs differ only in generated_ts). Safe on a cron cadence.

WHY this exists: board/experiment pages currently read SPRT facts from
online_status.json, which is written by the live daemon — when the
daemon is down or mid-restart the board goes stale with no fallback.
This script derives the identical status() dicts directly from the
persisted evidence (results/treatments.json), so the EvidenceBar has a
source that outlives the process.

CORRECTNESS DETAIL (the one that matters): the SPRT boundaries are NOT
taken from the persisted dict. treatments.json still stores each test's
edge/alpha/beta/min_n, but Treatment.load deliberately ignores them —
restoring config from disk once let stale persisted values silently
override retuned code constants, defeating the "pre-registered in code"
guarantee (bug fixed 2026-08-28 in btc_rl/treatments.py). This script
mirrors online.py's construction exactly: Treatment(edge=TREAT_EDGE,
min_n=TREAT_MIN_N, alpha=TREAT_ALPHA, baseline=key in ('champion',
'champion_real')), then .load() restores only the accumulated evidence
(n/mean/m2/llr and the bet counters). beta is whatever SPRT's
constructor defaults to — online.py does not pass it, so neither do we;
it is read back off the constructed object rather than hard-coded here.

The constants come from AST-parsing btc_rl/online.py source, not from
importing it — see scripts/emit_common.read_constants for why importing
the 4,000-line live module is off-limits for a cadence script.

SPRT is reconstructed via btc_rl.treatments (imported — it is pure
math/time with no import side effects); nothing statistical is
reimplemented here. Treatment requires a `decide` callable but never
calls it during status(); a stub lambda stands in.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent))     # for btc_rl.treatments
sys.path.insert(0, str(_HERE))            # for emit_common
from emit_common import (RESULTS_DIR, atomic_write_json,  # noqa: E402
                         parse_treat_registry, read_constants)
from btc_rl import treatments             # noqa: E402

STATE_PATH = RESULTS_DIR / "treatments.json"
OUT_PATH = RESULTS_DIR / "treatments_board.json"
BASELINES = ("champion", "champion_real")   # mirrors online.py line ~2321


def build_board() -> dict:
    consts = read_constants()
    edge = consts["TREAT_EDGE"]
    alpha = consts["TREAT_ALPHA"]
    min_n = consts["TREAT_MIN_N"]
    max_conc = consts["TREAT_MAX_CONCURRENT"]

    try:
        state = json.loads(STATE_PATH.read_text())
    except Exception:
        state = {}
    persisted = (state.get("treats") or {})

    # registry order is presentation order on the live board; treatments
    # persisted under keys the current registry no longer carries (a
    # retired policy) are appended, never dropped — promotion history is
    # part of the record.
    registry = {k: (lab, why) for k, lab, why in parse_treat_registry(
        env=consts)}
    ordered = [k for k in registry if k in persisted] + \
              [k for k in persisted if k not in registry]

    statuses = []
    ref = None                               # any non-baseline, for config
    for key in ordered:
        lab, why = registry.get(key, (key, "(retired from registry)"))
        t = treatments.Treatment(
            key, lab, lambda ctx: None, why,
            edge=edge, min_n=min_n, alpha=alpha,
            baseline=key in BASELINES)
        t.load(persisted[key])
        statuses.append(t.status())
        if not t.baseline and ref is None:
            ref = t

    beta = ref.sprt.beta if ref else treatments.SPRT().beta
    header = {
        "edge": edge,
        "alpha_per_test": alpha,
        "alpha_family": round(alpha * max_conc, 10),
        "max_concurrent": max_conc,
        "beta": beta,
        "min_n": min_n,
        "upper": round(ref.sprt.upper, 6) if ref else None,
        "lower": round(ref.sprt.lower, 6) if ref else None,
        "note": "boundaries from code constants (pre-registered), NOT "
                "from the persisted dict — see module docstring",
    }
    return {
        "generated_ts": round(time.time(), 3),
        "state_file": "results/treatments.json",
        "state_mtime_ts": round(STATE_PATH.stat().st_mtime, 3)
        if STATE_PATH.exists() else None,
        "config": header,
        "treatments": statuses,
    }


if __name__ == "__main__":
    board = build_board()
    atomic_write_json(OUT_PATH, board)
    c = board["config"]
    print(f"wrote {OUT_PATH} ({len(board['treatments'])} treatments, "
          f"alpha {c['alpha_per_test']:.6g} x{c['max_concurrent']} "
          f"-> family {c['alpha_family']}, "
          f"boundaries [{c['lower']:.3f}, {c['upper']:.3f}])")
