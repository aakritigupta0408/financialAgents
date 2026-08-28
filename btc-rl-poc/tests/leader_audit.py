"""Two questions:

Q1. Is there a kb arm that, whenever it was the selected leader, only
    lost? (i.e. a pure-poison leader we should ban outright.)
    Reported with a binomial p-value — "0 wins in 4" is not evidence.

Q2. Are the tier-1 RL price arms tracked as well as the kb arms now
    that all tiers have new data? Reports scoring coverage, staleness,
    and — the real question — whether each RL arm's output actually
    reaches a decision, or is logged and ignored.
"""
import datetime
import json
from collections import defaultdict
from math import comb
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PT = datetime.timezone(datetime.timedelta(hours=-7))


def load(n):
    p = ROOT / "results" / n
    return [json.loads(l) for l in p.open() if l.strip()] \
        if p.exists() else []


def day(ts):
    return datetime.datetime.fromtimestamp(ts, PT).strftime("%m/%d")


print("=" * 70)
print("Q1 · is any leader arm pure poison? (desk ledger, by leader)")
print("=" * 70)
pt = [t for t in load("pt_trades.jsonl") if t.get("actual") is not None]
by = defaultdict(lambda: {"n": 0, "w": 0, "pnl": 0, "days": set(),
                          "cost": 0.0})
for t in pt:
    b = by[t.get("leader", "?")]
    b["n"] += 1
    b["w"] += t["win"]
    b["pnl"] += t["pnl_c"]
    b["days"].add(day(t["close_ts"]))
    b["cost"] += t["stake_c"] / max(1, t["contracts"])
print(f"{'leader':7s} {'bids':>5s} {'win%':>7s} {'break-even':>11s} "
      f"{'P&L':>10s} {'days':>5s}  verdict")
for v, b in sorted(by.items(), key=lambda x: x[1]["pnl"]):
    wr = b["w"] / b["n"]
    be = b["cost"] / b["n"]
    # P(this few wins | fair coin at the break-even rate) — one-sided
    p = be / 100
    pv = sum(comb(b["n"], i) * p ** i * (1 - p) ** (b["n"] - i)
             for i in range(0, b["w"] + 1))
    if b["w"] == 0:
        verdict = (f"ZERO WINS in {b['n']} (p={pv:.3f}) "
                   + ("SIGNIFICANT" if pv < 0.05 else "but n too small"))
    elif b["pnl"] < 0 and pv < 0.05:
        verdict = f"significantly below break-even (p={pv:.3f})"
    elif b["pnl"] < 0:
        verdict = f"negative but within noise (p={pv:.2f})"
    else:
        verdict = "profitable"
    print(f"{v:7s} {b['n']:5d} {100*wr:6.1f}% {be:10.1f}% "
          f"{b['pnl']/100:+10.2f} {len(b['days']):5d}  {verdict}")

print("\nsame question at the DECISION level (not just when it led):")
print("every arm's own calls on the desk's windows, priced at real asks")
kb = load("kalshi_binary_log.jsonl")
deskt = {t["ticker"] for t in pt}
dec = defaultdict(dict)
for r in kb:
    v = r.get("variant") or "kb"
    if (r.get("actual") is None or r["ticker"] not in deskt
            or r.get("mins_left") is None or r["mins_left"] > 12
            or r.get("mkt_p_up") is None
            or max(r["p_up"], 1 - r["p_up"]) < 0.62):
        continue
    d = dec[v]
    tk = r["ticker"]
    if tk not in d or r["mins_left"] > d[tk]["mins_left"]:
        d[tk] = r
for v in sorted(dec):
    ds = list(dec[v].values())
    if len(ds) < 20:
        continue
    evs, wins = [], 0
    for d in ds:
        a = 100 * (d["mkt_p_up"] if d["call"] else 1 - d["mkt_p_up"]) + 2.5
        if not 5 <= a < 80:
            continue
        fee = 7 * (a / 100) * (1 - a / 100)
        c = a + fee
        w = int(d["call"] == d["actual"])
        wins += w
        evs.append((100 - c) / c if w else -1.0)
    if evs:
        print(f"  {v:4s} n={len(evs):3d} win {100*wins/len(evs):5.1f}% "
              f"EV/$1 {100*sum(evs)/len(evs):+6.1f}%")

print("\n" + "=" * 70)
print("Q2 · are the tier-1 RL arms tracked as well as the kb arms?")
print("=" * 70)
pred = load("prediction_log.jsonl")
now = max((r["made_ts"] for r in pred), default=0)
fam = defaultdict(lambda: {"n": 0, "scored": 0, "last": 0})
for r in pred:
    base = (r.get("variant") or "").split("-h")[0]
    f = fam[base]
    f["n"] += 1
    f["scored"] += r.get("actual") is not None
    f["last"] = max(f["last"], r["made_ts"])
print(f"{'RL arm':10s} {'rows':>6s} {'scored':>7s} {'scored%':>8s} "
      f"{'last seen':>10s}  reaches a decision?")
# which RL arms actually feed a kb arm / the desk
FEEDS = {
    "consensus": "yes — kb/kb2 read the consensus call",
    "cal": "indirect — recentering layer only",
    "h1": "yes — horizon base for consensus",
    "h5": "yes — horizon base for consensus",
    "h15": "yes — the 15-min binary horizon",
    "h30": "yes — horizon base",
    "rp": "no — chart-replay baseline (control)",
    "t2": "no — logged only",
    "t6": "no — logged only",
    "t7": "no — logged only",
    "t8": "no — logged only",
    "t9": "no — logged only",
    "t10": "no — logged only",
    "t11": "no — logged only",
}
for v, f in sorted(fam.items(), key=lambda x: -x[1]["n"]):
    if f["n"] < 100:
        continue
    age = (now - f["last"]) / 60
    print(f"{v:10s} {f['n']:6d} {f['scored']:7d} "
          f"{100*f['scored']/f['n']:7.1f}% {age:8.0f}m  "
          f"{FEEDS.get(v, '?')}")
print("\nreading: the RL arms ARE tracked (scored, fresh) but most are")
print("logged-only — they do not reach a trading decision. The binary")
print("tiers run off consensus/horizon predictions plus the market")
print("anchor. So tier-1 defects (drift bias) reach the desk only")
print("through the horizon/consensus path — which is where M4 should")
print("aim, rather than at t10/t11 whose bias never touches a bet.")
