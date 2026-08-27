"""SEV-0 error audit — DFS of the full architecture graph.

Tier 0  data feed        (ticks.jsonl, prediction_log price continuity)
Tier 1  RL price arms    (prediction_log: MAE, signed bias, direction,
                          band coverage — per arm family x horizon)
Tier 2  kb binary arms   (kalshi_binary_log: Brier, ROC-AUC, calibration
                          slope/intercept via from-scratch logistic
                          recalibration, prob bias, FP/FN at both gates)
Tier 3  decision layer   (leader churn, fixed-arm counterfactual under
                          IDENTICAL entry mechanics, gate ROC)
Tier 4  traders          (realized EV/$1, max drawdown, sizing damage
                          isolated from selection on shared windows)
Cross   attribution      (each losing desk bid tagged by root cause)

Every number is computed from the ledgers at run time. Effective n for
window-level stats is WINDOWS (entries within a window share fate).
"""
import datetime
import json
import math
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PT = datetime.timezone(datetime.timedelta(hours=-7))


def load(name):
    out = []
    for l in (ROOT / "results" / name).open():
        if l.strip():
            try:
                out.append(json.loads(l))
            except json.JSONDecodeError:
                pass
    return out


def auc(scores_pos, scores_neg):
    """Mann-Whitney AUC with tie correction, no libraries."""
    allv = sorted([(s, 1) for s in scores_pos] + [(s, 0) for s in scores_neg])
    ranks, i = {}, 0
    while i < len(allv):
        j = i
        while j < len(allv) and allv[j][0] == allv[i][0]:
            j += 1
        r = (i + j + 1) / 2          # average rank (1-based)
        for k in range(i, j):
            ranks[k] = r
        i = j
    rp = sum(ranks[k] for k, (s, y) in enumerate(allv) if y == 1)
    n1, n0 = len(scores_pos), len(scores_neg)
    if not n1 or not n0:
        return None
    return (rp - n1 * (n1 + 1) / 2) / (n1 * n0)


def logit(p):
    p = min(max(p, 1e-6), 1 - 1e-6)
    return math.log(p / (1 - p))


def sigmoid(z):
    return 1 / (1 + math.exp(-max(min(z, 30), -30)))


def recalibrate(ps, ys, iters=50):
    """Fit P(y)=sigmoid(a + b*logit(p)) by Newton-Raphson from scratch.
    b=1,a=0 means perfectly calibrated; b<1 overconfident spread;
    a>0 means outcomes richer than stated (under-forecast)."""
    a, b = 0.0, 1.0
    xs = [logit(p) for p in ps]
    for _ in range(iters):
        g0 = g1 = h00 = h01 = h11 = 0.0
        for x, y in zip(xs, ys):
            m = sigmoid(a + b * x)
            w = m * (1 - m)
            g0 += (y - m)
            g1 += (y - m) * x
            h00 += w
            h01 += w * x
            h11 += w * x * x
        det = h00 * h11 - h01 * h01
        if abs(det) < 1e-12:
            break
        da = (g0 * h11 - g1 * h01) / det
        db = (g1 * h00 - g0 * h01) / det
        a, b = a + da, b + db
        if abs(da) + abs(db) < 1e-10:
            break
    return a, b


def maxdd(series):
    peak, dd = -1e18, 0
    for v in series:
        peak = max(peak, v)
        dd = max(dd, peak - v)
    return dd


# ============================ TIER 0 =================================
print("=" * 72)
print("TIER 0 · DATA FEED")
print("=" * 72)
ticks = load("ticks.jsonl")[-50000:]
if len(ticks) > 1:
    gaps = sorted(t2["ts"] - t1["ts"] for t1, t2 in zip(ticks, ticks[1:])
                  if t2["ts"] >= t1["ts"])
    n = len(gaps)
    span_h = (ticks[-1]["ts"] - ticks[0]["ts"]) / 3600
    buys = sum(1 for t in ticks if t.get("taker_buy"))
    print(f"ticks analysed {len(ticks)} over {span_h:.1f}h · inter-tick gap "
          f"median {gaps[n//2]:.2f}s p95 {gaps[int(.95*n)]:.2f}s "
          f"p99 {gaps[int(.99*n)]:.2f}s max {gaps[-1]:.1f}s")
    print(f"taker-buy share {100*buys/len(ticks):.1f}% "
          f"(order-flow imbalance of the tape itself)")
preds_all = load("prediction_log.jsonl")
mins = sorted({r["made_ts"] // 60 for r in preds_all})
gaps_m = [b - a for a, b in zip(mins, mins[1:])]
if gaps_m:
    gs = sorted(gaps_m)
    stalls = sum(1 for g in gaps_m if g > 3 * gs[len(gs) // 2])
    print(f"prediction cycles {len(mins)} · cadence median "
          f"{gs[len(gs)//2]}min p95 {gs[int(.95*len(gs))]}min · "
          f"stalls>3x-median: {stalls}")

# ============================ TIER 1 =================================
print()
print("=" * 72)
print("TIER 1 · RL PRICE ARMS (prediction_log, scored rows)")
print("=" * 72)
print("bias = mean(pred - actual) $ · dir-acc vs realized move · "
      "up-share = calls up vs actual up share · cov80 = [lo,hi] hit rate")
fam = defaultdict(list)
for r in preds_all:
    if r.get("actual") is None or r.get("pred") is None:
        continue
    v = r["variant"]
    base = v.split("-h")[0] if "-h" in v else v
    fam[(base, r["horizon"])].append(r)
rows1 = []
for (base, h), rs in fam.items():
    n = len(rs)
    mae = sum(abs(r["pred"] - r["actual"]) for r in rs) / n
    bias = sum(r["pred"] - r["actual"] for r in rs) / n
    dm = [(r["pred"] - r["price_now"], r["actual"] - r["price_now"])
          for r in rs if r.get("price_now")]
    dm = [(p, a) for p, a in dm if p != 0 and a != 0]
    dacc = (sum(1 for p, a in dm if (p > 0) == (a > 0)) / len(dm)
            if dm else None)
    upc = sum(1 for p, a in dm if p > 0) / len(dm) if dm else None
    upa = sum(1 for p, a in dm if a > 0) / len(dm) if dm else None
    cov = [1 if r["lo"] <= r["actual"] <= r["hi"] else 0
           for r in rs if r.get("lo") is not None]
    rows1.append((base, h, n, mae, bias, dacc, upc, upa,
                  sum(cov) / len(cov) if cov else None))
rows1.sort(key=lambda x: (x[1], x[0]))
print(f"{'arm':8s} {'h':>3s} {'n':>6s} {'MAE$':>7s} {'bias$':>8s} "
      f"{'dirAcc':>7s} {'upCall':>7s} {'upReal':>7s} {'cov80':>6s}")
for base, h, n, mae, bias, dacc, upc, upa, cov in rows1:
    print(f"{base:8s} {h:3d} {n:6d} {mae:7.1f} {bias:+8.2f} "
          f"{'' if dacc is None else format(100*dacc, '6.1f')+'%':>7s} "
          f"{'' if upc is None else format(100*upc, '6.1f')+'%':>7s} "
          f"{'' if upa is None else format(100*upa, '6.1f')+'%':>7s} "
          f"{'' if cov is None else format(100*cov, '5.1f')+'%':>6s}")

# ============================ TIER 2 =================================
print()
print("=" * 72)
print("TIER 2 · KB BINARY ARMS (window decisions, biddable envelope)")
print("=" * 72)
kb = load("kalshi_binary_log.jsonl")
ARMS = ["kb", "kb2", "kb3", "kb4", "kb5", "kb6", "kb7", "kb8", "kb9"]
dec = defaultdict(dict)          # ungated: last pre-close call per window
for r in kb:
    v = r.get("variant") or "kb"
    if (v not in ARMS or r.get("actual") is None
            or r.get("mins_left") is None or r["mins_left"] > 12):
        continue
    tk = r["ticker"]
    if tk not in dec[v] or r["mins_left"] > dec[v][tk]["mins_left"]:
        dec[v][tk] = r
print(f"{'arm':4s} {'nWin':>5s} {'Brier':>6s} {'mktBr':>6s} {'AUC':>6s} "
      f"{'pBias':>7s} {'recal a':>8s} {'recal b':>8s} "
      f"{'w%@.62':>7s} {'w%@.77':>7s}")
roc_pts = {}
for v in ARMS:
    ds = list(dec[v].values())
    if len(ds) < 20:
        continue
    ys = [d["actual"] for d in ds]
    ps = [d["p_up"] for d in ds]
    br = sum((p - y) ** 2 for p, y in zip(ps, ys)) / len(ds)
    mm = [(d["mkt_p_up"], d["actual"]) for d in ds
          if d.get("mkt_p_up") is not None]
    mbr = (sum((p - y) ** 2 for p, y in mm) / len(mm)) if mm else None
    a_uc = auc([p for p, y in zip(ps, ys) if y],
               [p for p, y in zip(ps, ys) if not y])
    pbias = sum(ps) / len(ps) - sum(ys) / len(ys)
    ra, rb = recalibrate(ps, ys)
    def wr_at(tau):
        g = [d for d in ds if max(d["p_up"], 1 - d["p_up"]) >= tau]
        if len(g) < 8:
            return None, len(g)
        return sum(d["hit"] for d in g) / len(g), len(g)
    w62, n62 = wr_at(0.62)
    w77, n77 = wr_at(0.77)
    # ROC curve coordinates on the UP side (score = p_up)
    pts = []
    for tau in [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9]:
        tp = sum(1 for p, y in zip(ps, ys) if p >= tau and y)
        fp = sum(1 for p, y in zip(ps, ys) if p >= tau and not y)
        P = sum(ys)
        N = len(ys) - P
        pts.append((tau, tp / P if P else 0, fp / N if N else 0))
    roc_pts[v] = pts
    print(f"{v:4s} {len(ds):5d} {br:6.3f} "
          f"{'' if mbr is None else format(mbr, '6.3f'):>6s} "
          f"{a_uc:6.3f} {pbias:+7.3f} {ra:+8.3f} {rb:8.3f} "
          f"{'' if w62 is None else format(100*w62, '5.1f')+'%':>7s} "
          f"{'' if w77 is None else format(100*w77, '5.1f')+'%':>7s}")
print("\nROC curves (score = p_up, positive = window settled UP)")
print("tau:      " + "  ".join(f"{t:.2f}" for t, _, _ in
                               next(iter(roc_pts.values()))))
for v, pts in roc_pts.items():
    print(f"{v:4s} TPR  " + "  ".join(f"{tp:.2f}" for _, tp, _ in pts))
    print(f"{'':4s} FPR  " + "  ".join(f"{fp:.2f}" for _, _, fp in pts))

# ============================ TIER 3 =================================
print()
print("=" * 72)
print("TIER 3 · DECISION LAYER (leader mechanism + gates)")
print("=" * 72)
pt = load("pt_trades.jsonl")
sett = [t for t in pt if t.get("actual") is not None]
churn = sum(1 for a, b in zip(sett, sett[1:]) if a.get("leader")
            != b.get("leader"))
days = max(1, len({datetime.datetime.fromtimestamp(t["close_ts"], PT)
                   .date() for t in sett}))
print(f"leader switches: {churn} over {len(sett)} settled bids "
      f"({churn/days:.1f}/day) — the last-10 window chases noise")
# fixed-arm counterfactual, IDENTICAL mechanics on the desk's windows
desk_wins = {t["ticker"]: t for t in sett}
byv_rows = defaultdict(lambda: defaultdict(list))
for r in kb:
    v = r.get("variant") or "kb"
    if v in ARMS and r.get("actual") is not None:
        byv_rows[v][r["ticker"]].append(r)
print(f"\nfixed-arm counterfactual on the desk's own {len(desk_wins)} "
      f"windows (entry: first row mins_left<=12, conf>=0.62, real "
      f"ask+fee, $1/window):")
print(f"{'arm':4s} {'cover':>6s} {'win%':>6s} {'EV/$1':>7s}   "
      f"(desk realized: see traders tier)")
for v in ARMS:
    if v == "kb5":
        continue
    evs, wins = [], 0
    for tk in desk_wins:
        rows = sorted(byv_rows[v].get(tk, []),
                      key=lambda r: -r["mins_left"])
        d = next((r for r in rows if r["mins_left"] <= 12
                  and max(r["p_up"], 1 - r["p_up"]) >= 0.62
                  and r.get("mkt_p_up") is not None), None)
        if d is None:
            continue
        a = 100 * (d["mkt_p_up"] if d["call"] else 1 - d["mkt_p_up"]) + 2.5
        if not 5 <= a < 80:
            continue
        fee = math.ceil(7 * (a / 100) * (1 - a / 100))
        c = a + fee
        win = int(d["call"] == d["actual"])
        wins += win
        evs.append((100 - c) / c if win else -1.0)
    if evs:
        print(f"{v:4s} {len(evs):3d}/{len(desk_wins):3d} "
              f"{100*wins/len(evs):5.1f}% {100*sum(evs)/len(evs):+6.1f}%")
# gate ROC: confidence as the score for "bid wins"
scored = [(max(t['p_arm'], 1 - t['p_arm']) if 'p_arm' in t
           else None, t) for t in sett]
conf_rows = [(c, t["win"]) for c, t in scored if c is not None]
if len(conf_rows) > 30:
    ga = auc([c for c, w in conf_rows if w],
             [c for c, w in conf_rows if not w])
    print(f"\ndesk-entry confidence as win predictor: AUC {ga:.3f} "
          f"over {len(conf_rows)} bids")
for tau in [0.62, 0.70, 0.77, 0.85]:
    g = [w for c, w in conf_rows if c >= tau]
    if len(g) >= 5:
        print(f"  gate {tau:.2f}: win {100*sum(g)/len(g):5.1f}% · "
              f"coverage {100*len(g)/len(conf_rows):5.1f}%")

# ============================ TIER 4 =================================
print()
print("=" * 72)
print("TIER 4 · TRADERS (ledgers)")
print("=" * 72)
print(f"{'trader':12s} {'bets':>5s} {'win%':>6s} {'staked$':>9s} "
      f"{'P&L$':>9s} {'EV/$1':>7s} {'maxDD$':>8s}")
led = {}
for k, nm in [("pt", "Follower"), ("pt2", "Ladder"), ("pt3", "Disciplined"),
              ("pt4", "Gambler"), ("pt5", "Saver"), ("pt6", "MLE")]:
    rows = [t for t in load(f"{k}_trades.jsonl") if not t.get("skipped")]
    s = [t for t in rows if t.get("actual") is not None]
    led[k] = s
    if not s:
        continue
    staked = sum(t["stake_c"] for t in s)
    pnl = sum(t["pnl_c"] for t in s)
    wr = sum(t["win"] for t in s) / len(s)
    dd = maxdd([t["bankroll_c"] for t in s if t.get("bankroll_c")
                is not None])
    print(f"{nm:12s} {len(s):5d} {100*wr:5.1f}% {staked/100:9.0f} "
          f"{pnl/100:+9.2f} {100*pnl/max(1,staked):+6.1f}% {dd/100:8.0f}")
# sizing damage isolated from selection: shared windows only
sh = set(t["ticker"] for t in led["pt"]) & set(t["ticker"]
                                               for t in led["pt4"])
if sh:
    f1 = sum(t["pnl_c"] for t in led["pt"] if t["ticker"] in sh) \
        / max(1, sum(t["stake_c"] for t in led["pt"] if t["ticker"] in sh))
    g1 = sum(t["pnl_c"] for t in led["pt4"] if t["ticker"] in sh) \
        / max(1, sum(t["stake_c"] for t in led["pt4"] if t["ticker"] in sh))
    print(f"\nshared-window EV/$1 — Follower {100*f1:+.1f}% vs Gambler "
          f"{100*g1:+.1f}% ({len(sh)} windows): same signal, sizing is "
          f"the only difference")

# ======================= CROSS-TIER ATTRIBUTION ======================
print()
print("=" * 72)
print("ATTRIBUTION · every losing desk bid tagged by root cause")
print("=" * 72)
wrong_by_win = defaultdict(set)
tot_by_win = defaultdict(int)
for v in ARMS:
    if v == "kb5":
        continue
    for tk, d in dec[v].items():
        tot_by_win[tk] += 1
        if d["call"] != d["actual"]:
            wrong_by_win[tk].add(v)
tags = defaultdict(lambda: [0, 0])
for t in led["pt"]:
    if t["win"]:
        continue
    tk = t["ticker"]
    d = dec.get(t.get("leader", ""), {}).get(tk)
    mk = d.get("mkt_p_up") if d else None
    hour = datetime.datetime.fromtimestamp(t["close_ts"], PT).hour
    herd = (len(wrong_by_win[tk]) / tot_by_win[tk]
            if tot_by_win[tk] else 0)
    causes = []
    if mk is not None and abs(mk - 0.5) < 0.10:
        causes.append("knife-edge window")
    if mk is not None and d is not None \
            and (mk >= 0.5) != (d["call"] == 1):
        causes.append("against-market call")
    if herd >= 0.75:
        causes.append("herd whipsaw (>=3/4 arms wrong)")
    if hour in (9, 21, 22, 23, 1):
        causes.append("toxic hour")
    if not causes:
        causes.append("idiosyncratic (no shared cause)")
    for c in causes:
        tags[c][0] += 1
        tags[c][1] += -t["pnl_c"]
loss_total = sum(-t["pnl_c"] for t in led["pt"] if not t["win"])
print(f"desk losing bids: "
      f"{sum(1 for t in led['pt'] if not t['win'])}, "
      f"total lost ${loss_total/100:.0f} (tags overlap)")
for c, (n, cents) in sorted(tags.items(), key=lambda x: -x[1][1]):
    print(f"  {c:34s} {n:3d} bids  ${cents/100:7.0f} "
          f"({100*cents/max(1,loss_total):4.1f}% of losses)")
