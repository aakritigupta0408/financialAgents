"""Test a causal 'market predictability' abstention gate.

Signal (no look-ahead): rolling accuracy of the MARKET over the last N
SETTLED windows strictly before the current window opens. If that
trailing market accuracy < THRESH, the regime is deemed unpredictable
and we abstain. Applied to kb7's biddable confident entries (the
disciplined-tier stream), scored at real asks + fees.

Reports, for a sweep of (N, THRESH): trades taken vs abstained, win
rate, EV/$1, and the same for the trades we would have SKIPPED — an
abstention gate only helps if the skipped trades are worse.
"""
import json
import math
from collections import defaultdict

kb = [json.loads(l) for l in open("results/kalshi_binary_log.jsonl")]

# one market decision per window (kb2 carries mkt_p_up), settle-ordered
winrows = {}
for r in kb:
    if r.get("variant") == "kb2" and r.get("actual") is not None \
            and r.get("mkt_p_up") is not None:
        winrows.setdefault(r["ticker"], r)
mwins = sorted(winrows.values(), key=lambda r: r["close_ts"])
# market correct per window, in settle order
mkt_seq = [(w["close_ts"],
            int((w["mkt_p_up"] >= .5) == bool(w["actual"])))
           for w in mwins]


def trailing_acc(close_ts, N):
    past = [h for c, h in mkt_seq if c < close_ts]
    past = past[-N:]
    return sum(past) / len(past) if len(past) >= N else None


# kb7 biddable confident entries (the tier), with their window close_ts
ent = []
for r in kb:
    if r.get("variant") == "kb7" and r.get("actual") is not None \
            and r.get("mkt_p_up") is not None and r["mins_left"] <= 12 \
            and max(r["p_up"], 1 - r["p_up"]) >= 0.77:
        ask = 100 * (r["mkt_p_up"] if r["p_up"] >= .5
                     else 1 - r["mkt_p_up"]) + 2.5
        if 5 <= ask < 80:
            fee = math.ceil(7 * (ask / 100) * (1 - ask / 100))
            ent.append((r["close_ts"], r["hit"], ask + fee, r["ticker"]))


def ev(rows):
    if not rows:
        return None, 0, 0
    w = sum(h for _, h, _, _ in rows)
    c = sum(x for _, _, x, _ in rows) / len(rows)
    wn = len({t for _, _, _, t in rows})
    return ((w / len(rows)) * 100 - c) / c, len(rows), wn


base_ev, base_n, base_w = ev(ent)
print(f"baseline (no gate): EV {100*base_ev:+.1f}%/$1 · {base_n} entries "
      f"/ {base_w} windows\n")
print(f"{'N':>3} {'thr':>5} {'taken EV':>9} {'taken win':>10} "
      f"{'skipped EV':>10} {'skipped win':>11}")
for N in (8, 12, 20):
    for thr in (0.62, 0.65, 0.68, 0.72):
        taken, skipped = [], []
        for e in ent:
            ta = trailing_acc(e[0], N)
            (taken if (ta is None or ta >= thr) else skipped).append(e)
        te, tn, _ = ev(taken)
        se, sn, _ = ev(skipped)
        tw = (sum(h for _, h, _, _ in taken) / tn) if tn else 0
        sw = (sum(h for _, h, _, _ in skipped) / sn) if sn else 0
        print(f"{N:>3} {thr:>5.2f} "
              f"{(100*te if te is not None else 0):>+8.1f}% "
              f"{100*tw:>9.0f}%({tn}) "
              f"{(100*se if se is not None else 0):>+9.1f}% "
              f"{100*sw:>9.0f}%({sn})")
