"""Which kb arms cause the most WRONG bids — false positives (called UP,
market went DOWN) and false negatives (called DOWN, went UP)?

Judged on the desk's biddable-entry definition: last call per window at
mins_left <= 12, confidence >= 0.62 (the desk gate). One decision per
window per arm. Also splits the desk's own losing bids by which arm was
leading (whose leadership cost money).
"""
import json
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
kb = [json.loads(l) for l in
      (ROOT / "results" / "kalshi_binary_log.jsonl").open()]

ARMS = ["kb", "kb2", "kb3", "kb4", "kb5", "kb6", "kb7", "kb8", "kb9"]

# one biddable decision per (arm, window): the earliest row inside the
# entry envelope that clears the confidence gate — mirrors desk entries
dec = defaultdict(dict)
for r in kb:
    v = r.get("variant") or "kb"
    if (v not in ARMS or r.get("actual") is None
            or r.get("mins_left") is None or r["mins_left"] > 12
            or max(r["p_up"], 1 - r["p_up"]) < 0.62):
        continue
    tk = r["ticker"]
    if tk not in dec[v] or r["mins_left"] > dec[v][tk]["mins_left"]:
        dec[v][tk] = r

print("arm  | n    | FP (up-call wrong)     | FN (down-call wrong)   "
      "| wrong%")
print("-" * 78)
rows = []
for v in ARMS:
    ds = list(dec[v].values())
    if not ds:
        continue
    up = [r for r in ds if r["call"] == 1]
    dn = [r for r in ds if r["call"] == 0]
    fp = sum(1 for r in up if not r["actual"])   # called UP, was DOWN
    fn = sum(1 for r in dn if r["actual"])       # called DOWN, was UP
    n = len(ds)
    rows.append((v, n, fp, len(up), fn, len(dn), (fp + fn) / n))
    print(f"{v:4s} | {n:4d} | {fp:3d}/{len(up):3d} up-calls "
          f"({100*fp/max(1,len(up)):4.1f}%) | {fn:3d}/{len(dn):3d} "
          f"dn-calls ({100*fn/max(1,len(dn)):4.1f}%) | "
          f"{100*(fp+fn)/n:4.1f}%")

print("\n=== ranked: most wrong bids contributed (FP+FN, absolute) ===")
for v, n, fp, nu, fn, nd, wr in sorted(rows, key=lambda x: -(x[2] + x[4])):
    skew = "FP-heavy" if fp / max(1, nu) > fn / max(1, nd) + 0.05 \
        else "FN-heavy" if fn / max(1, nd) > fp / max(1, nu) + 0.05 \
        else "balanced"
    print(f"{v:4s}  wrong={fp+fn:3d} of {n:3d}  ({100*wr:.1f}%)  {skew}")

# who led the desk into losing bids (the follower's ledger)
pt = [json.loads(l) for l in (ROOT / "results" / "pt_trades.jsonl").open()
      if l.strip()]
lead = defaultdict(lambda: [0, 0, 0])
for t in pt:
    if t.get("actual") is None:
        continue
    b = lead[t.get("leader", "?")]
    b[0] += 1
    b[1] += 0 if t["win"] else 1
    b[2] += t["pnl_c"]
print("\n=== desk losses by LEADER (follower ledger, all days) ===")
for v, (n, losses, pnl) in sorted(lead.items(), key=lambda x: x[1][2]):
    print(f"{v:4s}  led {n:3d} bids, {losses:3d} lost, net "
          f"${pnl/100:+8.2f}")
