"""Warm-start pt6 (the MLE meta-trader) with the LIVE feature function
replayed over history — no leakage, settle-ordered prequential."""
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from btc_rl.online import _pt6_features, PT6_DIM, PT_ARMS  # noqa: E402
from btc_rl.agents import BinaryLogit                      # noqa: E402

kb = [json.loads(l) for l in
      (ROOT / "results" / "kalshi_binary_log.jsonl").open()]
mw = {}
for r in kb:
    if r.get("variant") == "kb2" and r.get("actual") is not None \
            and r.get("mkt_p_up") is not None:
        mw.setdefault(r["ticker"], r)
mseq = sorted(mw.values(), key=lambda r: r["close_ts"])

byv = defaultdict(lambda: defaultdict(list))
for r in kb:
    if r.get("variant") in PT_ARMS and r.get("actual") is not None:
        byv[r["variant"]][r["ticker"]].append(r)
dec = defaultdict(list)
for arm in PT_ARMS:
    for tk, rs in byv[arm].items():
        rs.sort(key=lambda r: -r["mins_left"])
        d = next((r for r in rs if max(r["p_up"], 1 - r["p_up"]) >= 0.62),
                 None)
        if d:
            dec[arm].append((d["close_ts"], d["hit"]))
    dec[arm].sort()


def leader(cts):
    best = None
    for arm in PT_ARMS:
        past = [h for c, h in dec[arm] if c < cts][-10:]
        if len(past) < 5:
            continue
        wr = sum(past) / len(past)
        if not best or wr > best[0]:
            best = (wr, arm)
    return best[1] if best else None


samples = []
for w in mseq:
    tk, cts = w["ticker"], w["close_ts"]
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
    conf = max(d["p_up"], 1 - d["p_up"])
    x = _pt6_features(conf, ask, d.get("mkt_p_up"), sy,
                      d.get("pf") or [], d["mins_left"])
    win = int((d["call"] == 1) == bool(d["actual"]))
    samples.append((cts, x, win))
samples.sort()
m = BinaryLogit(PT6_DIM)
grp = defaultdict(list)
for s in samples:
    grp[s[0]].append(s)
for cts in sorted(grp):
    for _, x, y in grp[cts]:
        pass
    for _, x, y in grp[cts]:
        m.update(x, y)
(ROOT / "results" / "pt6_logit.json").write_text(json.dumps(m.to_dict()))
print(f"warm-started pt6 on {len(samples)} windows, "
      f"{m.updates} updates -> results/pt6_logit.json")
