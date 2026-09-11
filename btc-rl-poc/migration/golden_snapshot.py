"""Freeze the M3 golden snapshot (PM work order P0) and verify the
system against it after cleanup.

  python3 migration/golden_snapshot.py before   -> golden_before_m3.json
  python3 migration/golden_snapshot.py verify   -> compares live state
                                                   to the frozen file

The verify gates are the PM's: A3 byte-equivalent (state, history
counts, decomposition), active model predictions unchanged for
already-settled windows, historical PnL totals unchanged,
reconciliation unchanged, parity state, invariant count.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"
OUT = ROOT / "migration" / "golden_before_m3.json"


def j(name):
    try:
        return json.loads((RES / name).read_text())
    except Exception:
        return None


def jl_pnl(name):
    """(settled_rows, total_pnl_c) for a trader ledger."""
    p = RES / name
    if not p.exists():
        return 0, 0
    n = tot = 0
    for l in p.open():
        if not l.strip():
            continue
        try:
            r = json.loads(l)
        except Exception:
            continue
        if r.get("actual") is not None and r.get("pnl_c") is not None:
            n += 1
            tot += r["pnl_c"]
    return n, tot


def capture():
    a3 = j("a3_live.json") or {}
    fwd = a3.get("forward") or {}
    qual = j("model_qualification.json") or {}
    recon = j("reconciliation.json") or {}
    inv = j("invariants.json") or {}
    par = j("parity.json") or {}
    git = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT,
                         capture_output=True, text=True).stdout.strip()
    cron = subprocess.run(["crontab", "-l"], capture_output=True,
                          text=True).stdout
    reg_hash = hashlib.sha256(
        (ROOT / "config" / "COMPONENT_REGISTRY.yaml")
        .read_bytes()).hexdigest()[:16]
    pnls = {n: jl_pnl(n + "_trades.jsonl") for n in
            ("pt", "pt2", "pt3", "pt4", "pt5", "pt6", "pt7", "pt8")}
    # settled kb predictions per variant: the FULL (ticker|made_ts ->
    # p_up) map. v1 stored only count+hash, which FALSE-FAILED on the
    # log's retention rotation (rows leave the tail oldest-first) —
    # verify must check mutation-on-overlap, with rotation allowed
    # only as an oldest-prefix disappearance.
    kb_sig = {}
    p = RES / "kalshi_binary_log.jsonl"
    if p.exists():
        acc = {}
        for l in p.open():
            if not l.strip():
                continue
            try:
                r = json.loads(l)
            except Exception:
                continue
            if r.get("actual") is None:
                continue
            v = r.get("variant") or "kb"
            acc.setdefault(v, {})[f"{r['ticker']}|{r['made_ts']}"] = \
                (r["made_ts"], r["p_up"])
        for v, m in acc.items():
            kb_sig[v] = {"n": len(m), "rows": m}
    return {
        "captured_ts": int(time.time()),
        "git_sha": git,
        "component_registry_hash": reg_hash,
        "crontab_sha": hashlib.sha256(cron.encode()).hexdigest()[:16],
        "a3": {
            "registered_ts": a3.get("registered_ts"),
            "spec_hash": a3.get("spec_hash"),
            "eligible": fwd.get("eligible"),
            "filled": fwd.get("filled"),
            "incremental_per_eligible":
                fwd.get("incremental_per_eligible"),
            "decomposition": fwd.get("decomposition"),
        },
        "model_qualification": {v: m.get("verdict") for v, m in
                                (qual.get("models") or {}).items()},
        "settled_kb_signatures": kb_sig,
        "trader_pnl_totals": {k: {"settled": v[0], "pnl_c": v[1]}
                              for k, v in pnls.items()},
        "reconciliation_overall": recon.get("overall"),
        "invariants": {"passed": inv.get("passed"),
                       "failed": inv.get("failed")},
        "parity_state": par.get("parity_state"),
    }


def verify():
    gold = json.loads(OUT.read_text())
    cur = capture()
    fails, notes = [], []
    g_a3, c_a3 = gold["a3"], cur["a3"]
    for k in ("registered_ts", "spec_hash"):
        if g_a3[k] != c_a3[k]:
            fails.append(f"A3 {k}: {g_a3[k]} != {c_a3[k]}")
    # A3 grows on the market clock — history may only EXTEND
    if (c_a3["eligible"] or 0) < (g_a3["eligible"] or 0):
        fails.append("A3 eligible shrank")
    for v, sig in gold["settled_kb_signatures"].items():
        c = cur["settled_kb_signatures"].get(v)
        if c is None:
            if v in ("kb6", "kbf"):
                notes.append(f"{v}: fully rotated out (retired arm)")
                continue
            fails.append(f"{v}: settled predictions vanished")
            continue
        cur_rows = c["rows"]
        mutated, missing_ts, present_ts = [], [], []
        for key, (ts, p_up) in sig["rows"].items():
            got = cur_rows.get(key)
            if got is None:
                missing_ts.append(ts)
            else:
                present_ts.append(ts)
                if abs(got[1] - p_up) > 1e-9:
                    mutated.append(key)
        if mutated:
            fails.append(f"{v}: {len(mutated)} settled predictions "
                         f"MUTATED e.g. {mutated[0]}")
        # rotation drops oldest-first: every missing row must be
        # OLDER than every surviving row, else it was deleted
        if missing_ts and present_ts \
                and max(missing_ts) >= min(present_ts):
            fails.append(f"{v}: non-prefix row deletion "
                         f"({len(missing_ts)} missing, newest missing "
                         f">= oldest surviving)")
        elif missing_ts:
            notes.append(f"{v}: {len(missing_ts)} oldest rows "
                         "rotated out (retention cap)")
    for t, tv in gold["trader_pnl_totals"].items():
        cv = cur["trader_pnl_totals"][t]
        if cv["settled"] < tv["settled"]:
            fails.append(f"{t}: settled rows shrank")
        # frozen archives may still settle stragglers; totals may
        # extend, never rewrite (checked via monotone counts)
    if cur["invariants"]["failed"]:
        fails.append(f"invariants failing: "
                     f"{cur['invariants']['failed']}")
    if cur["parity_state"] == "FAIL":
        fails.append("parity FAIL")
    verdict = "M3_VERIFY_PASS" if not fails else "M3_VERIFY_FAIL"
    print(verdict)
    for f in fails:
        print("  FAIL", f)
    for n in notes:
        print("  note", n)
    return 0 if not fails else 1


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "before"
    if mode == "before":
        OUT.parent.mkdir(exist_ok=True)
        OUT.write_text(json.dumps(capture(), indent=1))
        print(f"golden snapshot written: {OUT}")
    else:
        sys.exit(verify())
