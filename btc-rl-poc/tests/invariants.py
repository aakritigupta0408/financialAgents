"""Regression invariants — institutional memory, machine-checked.

Every confirmed insight or incident in this project must end as a
fixture, an exhibit, or an INVARIANT here (OS_BLUEPRINT §10). Runs on
the 10-min audit cron; failures land in results/invariants.json and a
red row on the Watchtower. Each check names the incident that created
it — these are not hypothetical lints, they are scars.

Exit code 0 always (the cron chain must not stop); the JSON verdict
is the signal.
"""
from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"
sys.path.insert(0, str(ROOT))
from btc_rl import online as O            # noqa: E402


def rows(name):
    p = RES / name
    return [json.loads(l) for l in p.open() if l.strip()] \
        if p.exists() else []


CHECKS = []


def check(name, origin):
    def deco(fn):
        CHECKS.append((name, origin, fn))
        return fn
    return deco


@check("one-decision-per-window",
       "trader-lockout bug 08-28 + owner guarantee 08-29")
def _one_per_window():
    """Each trader log: at most ONE row per (ticker); a committed
    decision is never re-entered or flipped."""
    bad = []
    for n in ("pt_trades", "pt2_trades", "pt3_trades", "pt4_trades",
              "pt5_trades", "pt6_trades", "pt7_trades", "pt8_trades"):
        seen = set()
        for t in rows(n + ".jsonl"):
            k = t.get("ticker")
            if k in seen:
                bad.append(f"{n}:{k}")
            seen.add(k)
    return not bad, bad[:5]


@check("ledger-monotonic-close-ts", "settle-ordered replay discipline")
def _monotonic():
    """treatments.jsonl close_ts must be non-decreasing — an
    out-of-order settle would corrupt every prequential number."""
    ts = [r.get("close_ts", 0) for r in rows("treatments.jsonl")]
    bad = [i for i in range(1, len(ts)) if ts[i] < ts[i - 1]]
    return not bad, bad[:5]


@check("no-duplicate-treatment-windows", "windows-not-rows law")
def _treat_dupes():
    tk = [r.get("ticker") for r in rows("treatments.jsonl")]
    dupes = {t for t in tk if tk.count(t) > 1} if len(tk) < 5000 else set()
    if len(tk) >= 5000:                     # O(n) path
        seen, dupes = set(), set()
        for t in tk:
            (dupes if t in seen else seen).add(t)
    return not dupes, sorted(dupes)[:5]


@check("fee-formula-parity", "per-contract fee rounding bug ($2,355)")
def _fee():
    """The daemon's _order_fee_c must equal ceil(7*C*p*(1-p)) per
    ORDER at canonical points (matches metric_fixtures vectors)."""
    cases = [(1, 50, 2), (10, 50, 18), (7, 30, 11), (3, 80, 4)]
    bad = [(c, a) for c, a, want in cases
           if O._order_fee_c(c, a) != want]
    return not bad, bad


@check("gambler-v3-sweep", "owner decision D-gambler-sizing 08-29")
def _sweep():
    """Every settled v3 pt4 row: bankroll_c never exceeds the $10k
    start (excess must have been withdrawn), and wd_c is stamped."""
    bad = []
    for t in rows("pt4_trades.jsonl"):
        if t.get("actual") is None or t["made_ts"] < O.PT4_RESET2_TS:
            continue
        if "wd_c" not in t or t.get("bankroll_c", 0) > O.PT4_RESET_C:
            bad.append(t["ticker"])
    return not bad, bad[:5]


@check("decision-quote-age", "stale-quote treatment bug (fake +11.75%)")
def _quote_age():
    """Rows the treatments evaluator scored must sit inside the <=12
    minute decision envelope (mins_left recorded on desk trades)."""
    bad = [t["ticker"] for t in rows("pt_trades.jsonl")
           if t.get("mins_left") is not None and t["mins_left"] > 12.05]
    return not bad, bad[:5]


@check("skip-is-not-missing", "decision-board INVALID bug 08-29")
def _evaluable():
    """decision_board completeness.pct must be 1.0 for every
    challenger — a skip is a scored decision, never missing data."""
    try:
        d = json.loads((RES / "decision_board.json").read_text())
    except Exception:
        return False, ["decision_board.json unreadable"]
    bad = [a["key"] for a in d.get("treatments", [])
           if a.get("completeness")
           and (a["completeness"].get("pct") or 1.0) < 1.0]
    return not bad, bad


@check("frozen-controls-untouched", "additive-only law")
def _frozen():
    """pt3 stays policy v2 (tau 0.77) and the SPRT family config
    stays code-sovereign at registered values."""
    ok = (abs(O.PT3_TAU - 0.77) < 1e-9
          and abs(O.TREAT_EDGE - 0.02) < 1e-9
          and abs(O.TREAT_ALPHA - 0.05 / 16) < 1e-12)
    return ok, [] if ok else ["a registered constant moved"]


@check("withdrawals-never-restaked", "D-gambler-sizing ledger law")
def _wd_conserved():
    """Reconstructed v3 bankroll from the ledger must equal the last
    settled row's bankroll_c exactly (cash conservation incl. wd)."""
    v3 = [t for t in rows("pt4_trades.jsonl")
          if t["made_ts"] >= O.PT4_RESET2_TS]
    settled = [t for t in v3 if t.get("actual") is not None]
    if not settled:
        return True, []
    bank = O.PT4_RESET_C \
        + sum(t["pnl_c"] - t.get("wd_c", 0) for t in settled)
    want = settled[-1].get("bankroll_c")
    ok = want is not None and bank == want \
        and bank <= O.PT4_RESET_C
    return ok, [] if ok else [f"recomputed {bank} vs ledger {want}"]


def main():
    out, failed = [], 0
    for name, origin, fn in CHECKS:
        try:
            ok, detail = fn()
        except Exception as e:              # a crashed check is a fail
            ok, detail = False, [f"check crashed: {e!r}"]
        out.append({"name": name, "origin": origin,
                    "ok": bool(ok), "detail": detail})
        failed += 0 if ok else 1
    doc = {"generated_ts": int(time.time()),
           "passed": len(out) - failed, "failed": failed,
           "health": "green" if failed == 0 else "red",
           "checks": out}
    (RES / "invariants.json").write_text(json.dumps(doc, indent=1))
    print(f"invariants: {doc['passed']}/{len(out)} passed "
          f"({doc['health']})")
    for c in out:
        if not c["ok"]:
            print(f"  FAIL {c['name']} — {c['detail']}")


if __name__ == "__main__":
    main()
