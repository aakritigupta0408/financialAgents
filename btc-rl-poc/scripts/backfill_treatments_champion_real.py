"""One-time settlement-migration backfill: correct treatments.jsonl champion_real
to the OFFICIAL Kalshi outcome.

WHY (2026-09-15 incident): treatments.jsonl is append-only and its ev.champion_real
was stamped ONCE, at settle time, under the OLD Coinbase-candle proxy settlement.
On 2026-09-15 the desk switched to OFFICIAL Kalshi settlement (commit 4833b3a) and
pt_trades.jsonl was re-settled on official CF-BRTI outcomes — flipping 46 windows'
win/loss. The stored champion_real (proxy era) then disagreed with the official
champion outcome, so reconcile.py's treatments-ev-parity check went SEV-1, which
tripped the fail-closed wall and FROZE the T0 control for ~11h.

champion_real is a DERIVED cache of the champion's realized EV. This script re-derives
it from the single source of truth (pt_trades.jsonl official settlement) using the
SAME formula the daemon uses (treatments.bet_ev): cost = ask_c + ceil(per-contract
fee), EV = (100-cost)/cost on a win else -1.0. This is a CORRECTION to official truth,
not fabrication. Only champion_real is touched (the reconcile-checked, T0-gating
field). The retired per-policy model-basis EVs are NOT recomputable offline (their
per-policy ask is not stored) and are left as the original proxy-era record; they are
unchecked and belong to retired treatments shown on no live surface.

Idempotent: re-running changes nothing once champion_real already matches official.
PAPER / SIMULATION research ledger only.
"""
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"
TREAT = RES / "treatments.jsonl"
PT = RES / "pt_trades.jsonl"


def _champ_real(side, ask_c, actual):
    """treatments.bet_ev(side, ask_c, outcome) for the champion trade."""
    a = float(ask_c)
    fee = math.ceil(7 * (a / 100) * (1 - a / 100) - 1e-9)
    cost = a + fee
    won = (side == "yes") == bool(actual)
    return (100.0 - cost) / cost if won else -1.0


def main():
    # official champion outcome per ticker, from settled pt trades
    champ = {}
    for ln in PT.read_text().splitlines():
        ln = ln.strip()
        if not ln:
            continue
        r = json.loads(ln)
        if r.get("actual") is not None and r.get("ticker") and r.get("ask_c") is not None:
            champ[r["ticker"]] = r

    rows = []
    changed = 0
    unmatched = 0
    for ln in TREAT.read_text().splitlines():
        ln = ln.strip()
        if not ln:
            continue
        row = json.loads(ln)
        ev = row.get("ev") or {}
        pt = champ.get(row.get("ticker"))
        if pt is not None and ev.get("champion_real") is not None:
            want = round(_champ_real(pt["side"], pt["ask_c"], pt["actual"]), 10)
            have = ev.get("champion_real")
            if abs(float(have) - want) > 1e-6:
                ev["champion_real"] = want
                row["ev"] = ev
                # stamp provenance so the correction is auditable, not silent
                row["champion_real_source"] = "OFFICIAL_KALSHI_backfill_2026-09-15"
                changed += 1
        elif pt is None and ev.get("champion_real") is not None:
            unmatched += 1
        rows.append(row)

    # write ONLY when the derived field actually drifted from official truth,
    # so this stays a no-op in the recurring audit chain once synced (no churn,
    # no mtime change) and re-acts only if pt is ever re-settled again.
    if changed:
        tmp = TREAT.with_suffix(".jsonl.bf_tmp")
        tmp.write_text("".join(json.dumps(r) + "\n" for r in rows))
        tmp.replace(TREAT)
    print(f"treatments champion_real sync: {changed} rows corrected to official, "
          f"{unmatched} rows had no settled champion match (left as-is), "
          f"{len(rows)} rows total"
          + ("" if changed else " (already synced — no write)"))


if __name__ == "__main__":
    main()
