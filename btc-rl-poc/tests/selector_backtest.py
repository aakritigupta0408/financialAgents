"""Per-class precision/recall for kb + a leakage-free backtest of the
SELECTOR: a secondary logit trained on settled calls to predict
P(call correct | context), gating calls/bets toward 80% precision at
maximum recall. Chronological 60/40 split; selector trained and its
threshold tuned on train only."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
import sys
sys.path.insert(0, str(ROOT))
from btc_rl.agents import BinaryLogit

rows = [json.loads(l) for l in
        (ROOT / "results" / "kalshi_binary_log.jsonl").read_text().splitlines()]
done = sorted((r for r in rows if r.get("variant", "kb") == "kb"
               and r["actual"] is not None), key=lambda r: r["made_ts"])

# ── 1. per-class precision / recall (bias check) ─────────────────────────
print("PER-CLASS (all settled kb calls):")
for cls, want in (("UP", 1), ("DOWN", 0)):
    called = [r for r in done if r["call"] == want]
    actual = [r for r in done if r["actual"] == want]
    prec = sum(r["hit"] for r in called) / len(called) if called else 0
    rec = (sum(1 for r in actual if r["call"] == want) / len(actual)
           if actual else 0)
    print(f"  {cls:>5}: precision {prec:.1%} (n called {len(called)}) · "
          f"recall {rec:.1%} (n actual {len(actual)})")
up_rate = sum(r["actual"] for r in done) / len(done)
print(f"  base rate: UP {up_rate:.1%}")


# ── 2. selector features from logged fields only (all causal) ────────────
def sel_x(r):
    p, mk = r["p_up"], r.get("mkt_p_up")
    return [
        1.0,
        abs(p - 0.5) * 2,                          # own confidence
        (abs(mk - 0.5) * 2) if mk is not None else 0.0,  # market confidence
        1.0 if mk is not None else 0.0,
        abs(p - mk) * 2 if mk is not None else 0.0,      # disagreement
        (1.0 if (p >= 0.5) == (mk >= 0.5) else -1.0) if mk is not None else 0.0,
        r["mins_left"] / 15.0,
        (abs(p - 0.5) * 2) * (1 - r["mins_left"] / 15.0),
    ]


cut = int(len(done) * 0.6)
train, test = done[:cut], done[cut:]
sel = BinaryLogit(8, lr=0.1)
for _ in range(30):                      # small data: multiple epochs
    for r in train:
        sel.update(sel_x(r), r["hit"])

# tune threshold on TRAIN: precision >= .8, maximize recall(coverage)
best = None
for t100 in range(30, 95):
    tau = t100 / 100
    pick = [r for r in train if sel.predict(sel_x(r)) >= tau]
    if len(pick) < 25:
        break
    prec = sum(r["hit"] for r in pick) / len(pick)
    cov = len(pick) / len(train)
    if prec >= 0.8 and (best is None or cov > best[2]):
        best = (tau, prec, cov)
if best is None:
    print("\nselector could not reach 80% precision on train")
else:
    tau, tp, tc = best
    print(f"\nSELECTOR tuned on train: tau={tau:.2f} -> "
          f"precision {tp:.1%} at coverage {tc:.1%}")
    pick = [r for r in test if sel.predict(sel_x(r)) >= tau]
    prec = sum(r["hit"] for r in pick) / len(pick) if pick else 0
    print(f"HELD-OUT: precision {prec:.1%} at coverage "
          f"{len(pick) / len(test):.1%}  (n={len(pick)}/{len(test)})")
    # per-class on selected, held-out
    for cls, want in (("UP", 1), ("DOWN", 0)):
        c = [r for r in pick if r["call"] == want]
        a = [r for r in test if r["actual"] == want]
        rec = sum(1 for r in c if r["hit"]) / len(a) if a else 0
        pr = sum(r["hit"] for r in c) / len(c) if c else 0
        print(f"  {cls:>5} selected: precision {pr:.1%} (n={len(c)}) · "
              f"class recall {rec:.1%}")
    # plain-threshold baseline on held-out for comparison
    for t100 in range(50, 95):
        t = t100 / 100
        pk = [r for r in test if max(r["p_up"], 1 - r["p_up"]) >= t]
        if pk and sum(r["hit"] for r in pk) / len(pk) >= 0.8:
            print(f"plain |p| threshold baseline: tau={t:.2f} precision "
                  f"{sum(r['hit'] for r in pk)/len(pk):.1%} at coverage "
                  f"{len(pk)/len(test):.1%}")
            break
