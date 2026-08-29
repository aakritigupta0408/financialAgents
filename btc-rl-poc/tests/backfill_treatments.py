"""Backfill the champion/challenger registry on all settled desk
windows, using the SAME evaluator the daemon runs — so the live stream
continues the identical computation rather than starting a new one.

No leakage: each window is scored with decision-time context only
(trailing market accuracy excludes the window being judged; the
calibrator state used is the one shipped by the shadow warm-start).
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from btc_rl import online as O                    # noqa: E402
from btc_rl import treatments                     # noqa: E402
from btc_rl.agents import PlattCalibrator         # noqa: E402


def load(n):
    p = ROOT / "results" / n
    return [json.loads(l) for l in p.open() if l.strip()] \
        if p.exists() else []


kb = load("kalshi_binary_log.jsonl")
pt = load("pt_trades.jsonl")
cd = {}
cp = ROOT / "results" / O.KB_CALIB_NAME
if cp.exists():
    cd = json.loads(cp.read_text())
kb_calib = {v: PlattCalibrator.from_dict(d) for v, d in cd.items()}

treats = {}
for k, lab, fn, why in O._treat_policies():
    treats[k] = treatments.Treatment(
        k, lab, fn, why, edge=O.TREAT_EDGE, min_n=O.TREAT_MIN_N,
        baseline=k in ("champion", "champion_real"))
seen = {}                      # dict-as-ordered-set, matches the daemon
fshare = treatments.FixedShare(O.PT_ARMS)
evlead = {}
recs = O._treat_evaluate(pt, kb, kb_calib, treats, seen, fshare, evlead)
print(f"scored {len(recs)} settled desk windows\n")
print(f"{'treatment':32s} {'n':>4s} {'bets':>5s} {'skips':>6s} "
      f"{'own EV':>8s} {'vs champ':>9s} {'LLR':>7s}  verdict")
champ_ev = [r["ev"]["champion"] for r in recs
            if r["ev"].get("champion") is not None]
if champ_ev:
    print(f"{'champion (incumbent)':32s} {len(champ_ev):4d} "
          f"{len(champ_ev):5d} {0:6d} "
          f"{100*sum(champ_ev)/len(champ_ev):7.2f}% {'—':>9s} "
          f"{'—':>7s}  incumbent")
for k, t in treats.items():
    if k == "champion":
        continue
    s = t.status()
    ev = "—" if s["own_ev"] is None else f"{100*s['own_ev']:7.2f}%"
    print(f"{s['label']:32s} {s['n']:4d} {s['bets']:5d} {s['skips']:6d} "
          f"{ev:>8s} {100*s['mean_diff']:+8.2f}% {s['llr']:7.2f}  "
          f"{s['verdict'].upper()}"
          + (f"  [boundaries {s['lower']:.1f} / {s['upper']:.1f}]"
             if s['verdict'] == 'collecting' else ""))

state = ROOT / "results" / O.TREAT_STATE_NAME
print("\nFixed-Share weights after the backfill (M3's live leader):")
for a, w in sorted(fshare.w.items(), key=lambda x: -x[1]):
    print(f"  {a:5s} {w:.4f} {'#' * int(60 * w)}")
state.write_text(json.dumps({
    "treats": {k: t.to_dict() for k, t in treats.items()},
    "fshare": fshare.to_dict(),
    "evlead": {k: [round(x, 5) for x in v] for k, v in evlead.items()},
    "seen": list(seen)[-4000:]}))
log = ROOT / "results" / O.TREAT_LOG_NAME
log.write_text("".join(json.dumps(r) + "\n" for r in recs))
print(f"\nwrote {state.name} and {log.name} — the live daemon continues "
      f"from here")
print("promotion requires the SPRT to cross its upper boundary on live "
      "windows; nothing switches on this backfill alone.")
