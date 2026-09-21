"""EMIT the coverage A/B snapshot — control (T0 pt) vs the open+6min barrier arm (ob),
with the ob arm's LIVE hit-rate / net P&L / EV sliced at 90/80/70/60/50/40/30/20/10%
coverage (top-X% of its settled trades by confidence |z| = conf_z). Answers, in real time,
"filter the bads out — what coverage buys what hit and what P&L". Writes
results/coverage_ab.json (rendered on Models Lab). Read-only research.
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"
OUT = RES / "coverage_ab.json"


def _load(name):
    p = RES / name
    if not p.exists():
        return []
    out = []
    for ln in p.read_text().splitlines():
        ln = ln.strip()
        if ln:
            try:
                out.append(json.loads(ln))
            except json.JSONDecodeError:
                pass
    return out


def _settled(rows):
    return [r for r in rows if r.get("actual") is not None]


def _arm_summary(rows, start_c):
    s = _settled(rows)
    wins = sum(1 for r in s if r.get("win"))
    pnl = sum(r.get("pnl_c") or 0 for r in s)
    open_stake = sum(r.get("stake_c") or 0 for r in rows if r.get("actual") is None)
    # bankroll = the ledger's OWN last bankroll_c, not start_c + pnl. Passing a
    # hardcoded ob seed of $10k showed a $10,080 bankroll while ob's ledger read $380
    # (audit 2026-09-21). start_c stays only as the fallback when no bankroll_c exists.
    last_bank = next((r.get("bankroll_c") for r in reversed(rows)
                      if r.get("bankroll_c") is not None), None)
    bank_c = last_bank if last_bank is not None else (start_c + pnl - open_stake)
    return {"settled": len(s), "wins": wins,
            "hit_rate": round(wins / len(s), 4) if s else None,
            "net_usd": round(pnl / 100, 2),
            "ev_per_trade_usd": round(pnl / 100 / len(s), 4) if s else None,
            "bankroll_usd": round(bank_c / 100, 2)}


def run():
    pt = _load("pt_trades.jsonl")
    ob = _load("ob_trades.jsonl")
    ob_s = sorted(_settled(ob), key=lambda r: -(r.get("conf_z") or 0))   # most confident first
    n = len(ob_s)
    cov_rows = []
    for cov in (1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1):
        k = max(1, int(round(n * cov))) if n else 0
        sub = ob_s[:k]
        wins = sum(1 for r in sub if r.get("win"))
        pnl = sum(r.get("pnl_c") or 0 for r in sub)
        cov_rows.append({
            "coverage": cov, "n": k,
            "hit_rate": round(wins / k, 4) if k else None,
            "net_usd": round(pnl / 100, 2),
            "ev_per_trade_usd": round(pnl / 100 / k, 4) if k else None,
            "min_conf_z": round(sub[-1].get("conf_z") or 0, 3) if sub else None,
        })
    doc = {"schema": "coverage-ab-1", "entry": "open+6min (~9 min left), analytic barrier",
           "control_T0_pt": _arm_summary(pt, 10_000_000_000),
           "treatment_ob": _arm_summary(ob, 1_000_000),  # $10k paper (2026-09-15)
           "ob_by_coverage": cov_rows,
           "note": ("ob trades EVERY window at open+6min (100% coverage) and logs conf_z=|z|; "
                    "each coverage row = its top-X% most-confident settled trades. As settled "
                    "trades accrue the higher-coverage rows converge to the offline curve "
                    "(cov90~0.74 hit, cov20~0.87 hit). Control T0 is a different (12-min follower) "
                    "strategy shown for reference.")}
    OUT.write_text(json.dumps(doc, indent=1))
    print(f"coverage_ab: ob settled={doc['treatment_ob']['settled']} -> {OUT}")
    for r in cov_rows:
        print(f"  cov {r['coverage']:.2f} (n={r['n']:3}) hit {r['hit_rate']} net ${r['net_usd']}")


if __name__ == "__main__":
    run()
