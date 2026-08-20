"""Full evaluation: every task, every arm, and whether learning helps.

Tasks covered:
  1. point prediction    MAE per arm x horizon, skill vs the persistence
                         floor measured on the SAME slots
  2. direction           sign accuracy on slots where the arm deviated
  3. calibration         80%-band coverage per arm family
  4. improvement         MAE/floor ratio, first half vs second half of each
                         arm's scored history (ratio cancels regime shifts)
  5. kalshi binary (kb)  accuracy + Brier vs the market, by window phase
  6. meta-arms           cal vs the winner it shadows, t11 vs t2 (paired)
  7. learning telemetry  update counts + hourly-gate revert rate

Usage: python scripts/evaluate_all.py
"""
import json
import math
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"

rows = [json.loads(l) for l in (RES / "prediction_log.jsonl").read_text().splitlines()]
sc = [r for r in rows if r["actual"] is not None]

FAMS = ["h", "rp-h", "t2-h", "t6-h", "t7-h", "t8-h", "t9-h", "t10-h",
        "t11-h", "cal-h", "consensus"]


def vname(fam, h):
    if fam == "consensus":
        return "consensus" if h == 5 else f"consensus-h{h}"
    return f"{fam}{h}"


print("=" * 100)
print("1+2. POINT PREDICTION & DIRECTION (all scored history)")
print(f"{'arm':>14} {'n':>5} {'MAE':>6} {'floor':>6} {'skill':>7} {'dir%':>6}")
for h in (1, 5, 15, 30):
    for fam in FAMS:
        g = [r for r in sc if r["variant"] == vname(fam, h)]
        if len(g) < 15:
            continue
        mae = sum(r["abs_err"] for r in g) / len(g)
        floor = sum(abs(r["actual"] - r["price_now"]) for r in g) / len(g)
        moved = [r for r in g if r["delta"]]
        dird = (100 * sum(1 for r in moved if (r["pred"] > r["price_now"])
                          == (r["actual"] > r["price_now"])) / len(moved)
                if moved else None)
        print(f"{vname(fam, h):>14} {len(g):>5} {mae:>6.0f} {floor:>6.0f} "
              f"{100 * (1 - mae / floor):>+6.1f}% "
              f"{dird:>5.0f}%" if dird is not None else
              f"{vname(fam, h):>14} {len(g):>5} {mae:>6.0f} {floor:>6.0f} "
              f"{100 * (1 - mae / floor):>+6.1f}%     –")
    print()

print("=" * 100)
print("2b. DIRECTION PAPER P&L — $1 notional on the predicted direction each")
print("    moved slot; cumulative return in bps (scored slots only, post-hoc)")
for h in (1, 5, 15, 30):
    line = f"  +{h:>2}m: "
    for fam in ("h", "t2-h", "t6-h", "t7-h", "t8-h", "t9-h", "t10-h"):
        g = [r for r in sc if r["variant"] == vname(fam, h) and r["delta"]]
        if len(g) < 15:
            continue
        rets = [(1 if r["delta"] > 0 else -1)
                * (r["actual"] - r["price_now"]) / r["price_now"] * 1e4
                for r in g]
        line += (f"{fam.rstrip('-h') or 'ctl'} {sum(rets):+.0f} "
                 f"({sum(rets)/len(rets):+.1f}/slot, n={len(g)})  ")
    print(line)

print("=" * 100)
print("3. CALIBRATION — 80% band coverage (target 80%)")
for fam in FAMS:
    g = [r for r in sc if r["variant"].startswith(fam if fam != "h" else "h")
         and (fam != "h" or r["variant"] in ("h1", "h5", "h15", "h30"))
         and r.get("in_band") is not None]
    if len(g) >= 20:
        print(f"  {fam:>10}: {100 * sum(r['in_band'] for r in g) / len(g):5.1f}%"
              f"  (n={len(g)})")

print("=" * 100)
print("4. IMPROVEMENT — MAE/floor ratio, 1st half vs 2nd half of history")
print("   (<1 beats persistence; 2nd < 1st = the arm improved)")
for h in (1, 5, 15, 30):
    line = f"  +{h:>2}m: "
    for fam in ("h", "t2-h", "t7-h", "t8-h", "t9-h", "t10-h"):
        g = [r for r in sc if r["variant"] == vname(fam, h)]
        if len(g) < 60:
            continue
        half = len(g) // 2
        def ratio(rs):
            fl = sum(abs(r["actual"] - r["price_now"]) for r in rs) / len(rs)
            return (sum(r["abs_err"] for r in rs) / len(rs)) / fl if fl else 1
        r1, r2 = ratio(g[:half]), ratio(g[half:])
        line += f"{fam.rstrip('-h') or 'ctl'} {r1:.2f}→{r2:.2f}  "
    print(line)

print("=" * 100)
print("5. KALSHI BINARY (kb)")
kb_path = RES / "kalshi_binary_log.jsonl"
if kb_path.exists():
    kb = [json.loads(l) for l in kb_path.read_text().splitlines()]
    done = [r for r in kb if r["actual"] is not None]
    if done:
        acc = sum(r["hit"] for r in done) / len(done)
        br = sum(r["brier"] for r in done) / len(done)
        print(f"  settled {len(done)}  acc {100*acc:.0f}%  brier {br:.4f}")
        both = [r for r in done if r.get("mkt_brier") is not None]
        if both:
            print(f"  vs market (n={len(both)}): ours "
                  f"{sum(r['brier'] for r in both)/len(both):.4f}  market "
                  f"{sum(r['mkt_brier'] for r in both)/len(both):.4f}")
        for lo, hi, lbl in ((10, 99, "early >10m"), (5, 10, "mid 5-10m"),
                            (0, 5, "late <5m")):
            g = [r for r in done if lo <= r["mins_left"] < hi]
            if g:
                print(f"    {lbl:>11}: acc {100*sum(r['hit'] for r in g)/len(g):3.0f}%"
                      f"  brier {sum(r['brier'] for r in g)/len(g):.4f}  n={len(g)}")

print("=" * 100)
print("6. META-ARMS (paired on identical slots)")
cal = [r for r in sc if r["variant"] == "cal-h15"]
if cal:
    src_err = {(r["variant"], r["made_ts"]): r["abs_err"] for r in sc}
    pairs = [(r["abs_err"], src_err.get((r["src"], r["made_ts"])))
             for r in cal if r.get("src")]
    pairs = [(c, s) for c, s in pairs if s is not None]
    if pairs:
        cm = sum(c for c, _ in pairs) / len(pairs)
        sm = sum(s for _, s in pairs) / len(pairs)
        w = sum(1 for c, s in pairs if c < s)
        print(f"  cal-h15 vs shadowed winner: cal ${cm:.0f} vs winner ${sm:.0f}"
              f"  (cal wins {w}/{len(pairs)} slots)")
for h in (1, 5, 15, 30):
    a = {r["made_ts"]: r["abs_err"] for r in sc if r["variant"] == f"t11-h{h}"}
    b = {r["made_ts"]: r["abs_err"] for r in sc if r["variant"] == f"t2-h{h}"}
    common = sorted(set(a) & set(b))
    if len(common) >= 15:
        print(f"  t11 vs t2 +{h}m (n={len(common)}): "
              f"t11 ${sum(a[t] for t in common)/len(common):.0f} vs "
              f"t2 ${sum(b[t] for t in common)/len(common):.0f}"
              "   (should be ~equal until feedback exists)")

print("=" * 100)
print("7. LEARNING TELEMETRY")
st = json.load(open(RES / "online_status.json"))
print(f"  online updates this session: {st['online_updates_session']}"
      f"  retrains: {st['retrains_this_session']}")
lr = st.get("last_retrain") or {}
gates = [(a, hh, d) for a, hs in (lr.get("arms") or {}).items()
         for hh, d in hs.items()]
if gates:
    rev = sum(1 for _, _, d in gates if d["reverted"])
    print(f"  last hourly gate: {len(gates) - rev}/{len(gates)} arm-horizons "
          f"kept their replay update ({rev} reverted as noise-chasing)")
    worst = sorted(gates, key=lambda g: g[2]["val_mae_after"]
                   - g[2]["val_mae_before"])[:3]
    for a, hh, d in worst:
        print(f"    best gate delta: {a} {hh} "
              f"{d['val_mae_before']:.0f}→{d['val_mae_after']:.0f}")

print("=" * 100)
print("8. VERDICT WATCH — treatments with 300+ scored slots that trail control")
print("   by >10% on identical slots are retire candidates (t3/t4/t5 precedent)")
flagged = 0
for h in (1, 5, 15, 30):
    base = {r["made_ts"]: r["abs_err"] for r in sc if r["variant"] == vname("h", h)}
    for fam in ("t2-h", "t6-h", "t7-h", "t8-h", "t9-h", "t10-h", "t11-h", "cal-h"):
        g = [r for r in sc if r["variant"] == vname(fam, h)
             and r["made_ts"] in base]
        if len(g) < 300:
            continue
        mae = sum(r["abs_err"] for r in g) / len(g)
        cmae = sum(base[r["made_ts"]] for r in g) / len(g)
        if mae > 1.10 * cmae:
            flagged += 1
            print(f"  RETIRE CANDIDATE: {vname(fam, h)} — ${mae:.0f} vs "
                  f"control ${cmae:.0f} on {len(g)} shared slots")
if not flagged:
    biggest = max(len([r for r in sc if r["variant"] == vname("t2-h", h)])
                  for h in (1, 5, 15, 30))
    print(f"  none flagged yet — largest treatment sample is {biggest} slots "
          "(verdict window: 300)")
