"""Baseline MLE meta-trader — supervised edge model + fractional-Kelly
sizing, learning from the rule traders' signals. The industry-standard
starting point (predict edge supervised, size analytically) rather than
end-to-end RL, which our ~120-window sample cannot support.

Per biddable window (decision-time, no leakage):
  features = [leader conf, ask/100, edge=conf-implied, trailing market
              acc (8w), |recent 15m drift| z, mins_left/15, agreement
              of kb2/kb7 with leader]
  label    = did the leader-side bet win
Online BinaryLogit predicts P(win); bet iff EV>0 at the real ask;
size = HALF-Kelly of the estimated edge, capped 10%. Prequential
(predict-then-update in settle order). Reports vs the Follower (flat
10%) and the Disciplined tier.
"""
import json
import math
from collections import defaultdict

kb = [json.loads(l) for l in open("results/kalshi_binary_log.jsonl")]

# market decision per window (settle-ordered) for the regime signal
mw = {}
for r in kb:
    if r.get("variant") == "kb2" and r.get("actual") is not None \
            and r.get("mkt_p_up") is not None:
        mw.setdefault(r["ticker"], r)
mseq = sorted(mw.values(), key=lambda r: r["close_ts"])
mkt_hist = [(w["close_ts"], int((w["mkt_p_up"] >= .5) == bool(w["actual"])))
            for w in mseq]


def trail_acc(cts, N=8):
    p = [h for c, h in mkt_hist if c < cts][-N:]
    return sum(p) / len(p) if p else 0.6


# leader per window (last-10 settled gate-clearing among the arms)
ARMS = ["kb2", "kb3", "kb4", "kb6", "kb7", "kb8"]
byv = defaultdict(lambda: defaultdict(list))
for r in kb:
    if r.get("variant") in ARMS and r.get("actual") is not None:
        byv[r["variant"]][r["ticker"]].append(r)
dec = defaultdict(list)   # arm -> [(close_ts, hit)]
allw = sorted({w["ticker"]: w["close_ts"] for w in mseq}.items(),
              key=lambda kv: kv[1])
for arm in ARMS:
    for tk, rs in byv[arm].items():
        rs.sort(key=lambda r: -r["mins_left"])
        d = next((r for r in rs if max(r["p_up"], 1 - r["p_up"]) >= 0.62),
                 None)
        if d:
            dec[arm].append((d["close_ts"], d["hit"]))
    dec[arm].sort()


def leader(cts):
    best = None
    for arm in ARMS:
        past = [h for c, h in dec[arm] if c < cts][-10:]
        if len(past) < 5:
            continue
        wr = sum(past) / len(past)
        if not best or wr > best[0]:
            best = (wr, arm)
    return best[1] if best else None


# biddable windows: leader's call + ask + label
kb7row = {(r["ticker"]): r for r in kb if r.get("variant") == "kb7"}
samples = []
for tk, cts in allw:
    ld = leader(cts)
    if not ld:
        continue
    rows = sorted(byv[ld][tk], key=lambda r: -r["mins_left"])
    d = next((r for r in rows if r["mins_left"] <= 12
              and max(r["p_up"], 1 - r["p_up"]) >= 0.62
              and r.get("mkt_p_up") is not None), None)
    if not d:
        continue
    sy = d["p_up"] >= 0.5
    ask = 100 * (d["mkt_p_up"] if sy else 1 - d["mkt_p_up"]) + 2.5
    if not 5 <= ask < 80:
        continue
    fee = math.ceil(7 * (ask / 100) * (1 - ask / 100))
    conf = max(d["p_up"], 1 - d["p_up"])
    pf = (d.get("pf") or [0, 0, 0, 0])
    feat = [1.0, conf, ask / 100, conf - (ask / 100),
            trail_acc(cts) - 0.6, abs(pf[2]) if len(pf) > 2 else 0,
            d["mins_left"] / 15.0]
    win = int((d["call"] == 1) == bool(d["actual"]))
    samples.append((cts, feat, win, ask + fee))
samples.sort()
print(f"biddable windows: {len(samples)}")

# online logit, prequential, half-Kelly sizing
import numpy as np
w = np.zeros(len(samples[0][1]))
lr = 0.1
bank_meta, bank_flat = 1000.0, 1000.0
mt_bets, mt_wins, mt_stake, mt_net = 0, 0, 0.0, 0.0
peak, dd = 1000.0, 0.0
for cts, feat, y, cost in samples:
    x = np.array(feat)
    p = 1 / (1 + math.exp(-max(-30, min(30, x @ w))))
    # meta policy: EV per contract, half-Kelly fraction
    ev = p * 100 - cost
    if ev > 0:
        b = (100 - cost) / cost            # net odds
        kelly = max(0, (p * (1 + b) - 1) / b)
        frac = min(0.10, 0.5 * kelly)
        stake = frac * bank_meta
        ncon = int(stake // cost)
        if ncon >= 1:
            st = ncon * cost
            pnl = (ncon * 100 - st) if y else -st
            bank_meta += pnl / 100
            mt_bets += 1
            mt_wins += y
            mt_stake += st / 100
            mt_net += pnl / 100
            peak = max(peak, bank_meta)
            dd = max(dd, peak - bank_meta)
    # flat follower: always bet leader side at 10%
    ncon2 = int((0.10 * bank_flat) // cost)
    if ncon2 >= 1:
        st2 = ncon2 * cost
        bank_flat += ((ncon2 * 100 - st2) if y else -st2) / 100
    # learn
    w -= lr * ((p - y) * x + 1e-4 * w)

print(f"\nMETA (supervised + half-Kelly): {mt_bets} bets, "
      f"{mt_wins}/{mt_bets} win ({100*mt_wins/max(1,mt_bets):.0f}%), "
      f"net ${mt_net:+.2f}, EV {100*mt_net/max(1,mt_stake):+.0f}%/$1, "
      f"bank ${bank_meta:.0f}, maxDD ${dd:.0f}, "
      f"idle {100*(1-mt_bets/len(samples)):.0f}%")
print(f"FLAT Follower (bet every leader call, 10%): bank ${bank_flat:.0f}")
