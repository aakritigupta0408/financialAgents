"""Is any polling combination of the arms ~100% accurate?
Join all variants per (ticker, made_ts); evaluate voting rules on the
first 70% of windows, validate survivors on the last 30%.
Rules: unanimity subsets, unanimity+confidence, +market, +late phase.
"""
import json
from collections import defaultdict
from itertools import combinations
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
kb = [json.loads(l) for l in
      (ROOT / "results" / "kalshi_binary_log.jsonl").open()]
by = defaultdict(dict)
for r in kb:
    v = r.get("variant", "kb")
    if r.get("actual") is not None:
        by[(r["ticker"], r["made_ts"])][v] = r
minutes = sorted(by.values(), key=lambda d: next(iter(d.values()))["made_ts"])
cut = minutes[int(len(minutes) * .7)]
cut_ts = next(iter(cut.values()))["made_ts"]

CONF = {"kb": .62, "kb2": .60, "kb3": .78, "kb4": .79, "kb7": .70}
ARMS = ["kb", "kb2", "kb3", "kb4", "kb7"]


def rule_eval(mins, arms, need_conf, need_mkt, late_only):
    n = hit = 0
    for d in mins:
        if not all(a in d for a in arms):
            continue
        r0 = d[arms[0]]
        if late_only and r0["mins_left"] > 5:
            continue
        calls = [d[a]["call"] for a in arms]
        if len(set(calls)) != 1:
            continue
        if need_conf and not all(
                max(d[a]["p_up"], 1 - d[a]["p_up"]) >= CONF[a]
                for a in arms):
            continue
        if need_mkt:
            mp = r0.get("mkt_p_up")
            if mp is None or (mp >= .5) != (calls[0] == 1):
                continue
        n += 1
        hit += int(calls[0] == r0["actual"])
    return n, hit


tr = [d for d in minutes
      if next(iter(d.values()))["made_ts"] < cut_ts]
te = [d for d in minutes
      if next(iter(d.values()))["made_ts"] >= cut_ts]
results = []
for k in (2, 3, 4, 5):
    for arms in combinations(ARMS, k):
        for conf in (False, True):
            for mkt in (False, True):
                for late in (False, True):
                    n, h = rule_eval(tr, list(arms), conf, mkt, late)
                    if n >= 40 and h / n >= 0.97:
                        results.append((h / n, n, arms, conf, mkt, late))
results.sort(reverse=True)
print(f"train rules reaching >=97% with n>=40: {len(results)}")
for acc, n, arms, conf, mkt, late in results[:8]:
    tn, th = rule_eval(te, list(arms), conf, mkt, late)
    tacc = th / tn if tn else float("nan")
    print(f"  {'+'.join(arms):22s} conf={int(conf)} mkt={int(mkt)} "
          f"late={int(late)} | train {acc:.1%} (n={n}) -> "
          f"TEST {tacc:.1%} (n={tn})")
if not results:
    # report the best achievable instead
    best = None
    for k in (2, 3, 4, 5):
        for arms in combinations(ARMS, k):
            for conf in (False, True):
                for mkt in (False, True):
                    for late in (False, True):
                        n, h = rule_eval(tr, list(arms), conf, mkt, late)
                        if n >= 40 and (best is None or h / n > best[0]):
                            best = (h / n, n, arms, conf, mkt, late)
    acc, n, arms, conf, mkt, late = best
    tn, th = rule_eval(te, list(arms), conf, mkt, late)
    print(f"best train rule: {'+'.join(arms)} conf={int(conf)} "
          f"mkt={int(mkt)} late={int(late)}: {acc:.1%} (n={n}) -> "
          f"TEST {th/max(1,tn):.1%} (n={tn})")
