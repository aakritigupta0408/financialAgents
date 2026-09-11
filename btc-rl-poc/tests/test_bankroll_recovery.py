"""INC 2026-09-07 regression tests — zombie positions & late settle.

Proves, on synthetic fixtures:
  1. an unsettled matured row (zombie) whose stake left cash makes
     the bankroll walk flag exactly one mismatch (the detection that
     froze the desk — correct behavior, must keep working);
  2. a LATE-settled row (late_settle_ts, historical stamp preserved)
     reconciles cleanly, with the payout credited at settle time;
  3. an ordinary timely settle still reconciles (no regression).
Runs reconcile.main() against a temp RESULTS dir (missing side
files are tolerated by load_jsonl).
"""
import importlib
import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
import reconcile  # noqa: E402


def run_fixture(pt_rows):
    tmp = tempfile.mkdtemp()
    p = Path(tmp) / "pt_trades.jsonl"
    p.write_text("".join(json.dumps(r) + "\n" for r in pt_rows))
    real_manifest = ROOT / "results" / "site_manifest.json"
    if real_manifest.exists():
        (Path(tmp) / "site_manifest.json").write_text(
            real_manifest.read_text())
    reconcile.RESULTS = tmp
    reconcile.OUT_PATH = str(Path(tmp) / "reconciliation.json")
    reconcile.main()
    doc = json.loads(Path(reconcile.OUT_PATH).read_text())
    bank = next(c for c in doc["checks"]
                if c["name"] == "bankroll-conservation")
    return bank


BASE = {"side": "no", "strike": 100.0, "contracts": 10, "skim_c": 0,
        "wd_c": 0, "ask_c": 50.0}


def settled(ts, stake, pnl, bank):
    return {**BASE, "made_ts": ts, "close_ts": ts + 900,
            "ticker": f"T{ts}", "stake_c": stake, "pnl_c": pnl,
            "bankroll_c": bank, "actual": 0, "win": 1,
            "settled": True}


def t1_zombie_flags():
    rows = [settled(1000, 500, 200, 100200),
            # zombie: stake 300 debited, never settled
            {**BASE, "made_ts": 2000, "close_ts": 2900,
             "ticker": "TZ", "stake_c": 300, "pnl_c": None,
             "bankroll_c": 99900, "actual": None, "win": None},
            # next settled row stamped from cash (short by 300
            # forever vs the settled-pnl walk)
            settled(90000, 400, 100, 100000)]
    bank = run_fixture(rows)
    assert bank["mismatches"] == 1, bank
    print("  1. unmarked zombie -> exactly one flag (detection OK)")


def t2_late_settle_clean():
    rows = [settled(1000, 500, 200, 100200),
            # same zombie, now LATE-settled: outcome filled, entry
            # stamp preserved, payout (pnl+stake=1000-...) credited
            # at late_settle_ts=95000
            {**BASE, "made_ts": 2000, "close_ts": 2900,
             "ticker": "TZ", "stake_c": 300, "pnl_c": 700,
             "bankroll_c": 99900, "actual": 0, "win": 1,
             "late_settle_ts": 95000},
            # row between close and late settle: cash short by 300 ✓
            settled(90000, 400, 100, 100000),
            # row after late settle: cash includes payout 1000
            settled(96000, 400, 100, 101100)]
    bank = run_fixture(rows)
    assert bank["mismatches"] == 0, bank
    print("  2. late-settled zombie -> walk clean, credit at "
          "late_settle_ts")


def t3_normal_settle_clean():
    rows = [settled(1000, 500, 200, 100200),
            settled(2000, 400, -400, 99800),
            settled(3000, 300, 2700, 102500)]
    bank = run_fixture(rows)
    assert bank["mismatches"] == 0, bank
    print("  3. ordinary settles -> clean (no regression)")


def t4_late_settle_idempotent():
    """PM 09-07: late settlement must be idempotent across restarts.
    In the daemon the guard is `if t["actual"] is not None: continue`
    (a settled row is never re-settled, so cash is credited exactly
    once); here we prove the AUDIT side — re-running reconcile over
    the same late-settled ledger yields the same clean verdict, and
    the credit is applied exactly once even though the pre-scan runs
    fresh each time."""
    rows = [settled(1000, 500, 200, 100200),
            {**BASE, "made_ts": 2000, "close_ts": 2900,
             "ticker": "TZ", "stake_c": 300, "pnl_c": 700,
             "bankroll_c": 99900, "actual": 0, "win": 1,
             "late_settle_ts": 95000},
            settled(96000, 400, 100, 101000)]
    b1 = run_fixture(rows)
    b2 = run_fixture(rows)
    assert b1["mismatches"] == 0 and b2["mismatches"] == 0, (b1, b2)
    print("  4. late settlement idempotent across audit re-runs")


if __name__ == "__main__":
    t1_zombie_flags()
    t2_late_settle_clean()
    t3_normal_settle_clean()
    t4_late_settle_idempotent()
    print("bankroll-recovery: 4/4 pass")
