"""What is pt6 (Metamon) doing? Full trade list + model state. Temp."""
import datetime
import json

PT = datetime.timezone(datetime.timedelta(hours=-7))


def ft(ts):
    return datetime.datetime.fromtimestamp(ts, PT).strftime("%m/%d %H:%M")


rows = [json.loads(l) for l in open("results/pt6_trades.jsonl") if l.strip()]
print(f"=== pt6 trades ({len(rows)}) ===")
for t in rows:
    res = ("open" if t.get("actual") is None
           else f"{'WIN ' if t['win'] else 'LOSS'} {t['pnl_c']/100:+.2f}$")
    print(f"{ft(t['made_ts'])} {t['side']:>3s} ≥{t['strike']:.0f} "
          f"ask {t['ask_c']:.0f}c  p_win={t.get('p_win')}  "
          f"{t['contracts']}c stake ${t['stake_c']/100:.0f}  "
          f"leader={t.get('leader')}  {res}  bank ${t['bankroll_c']/100:.2f}")

m = json.load(open("results/pt6_logit.json"))
print("\n=== pt6 logit ===")
print("updates:", m.get("updates"))
print("weights:", [round(w, 3) for w in m.get("w", [])])
print("features: [bias, conf, ask, conf-ask, mkt-lean, mins/15, pf2]")

# sanity: recompute p_win for last trade from stored b6x
import math
last = rows[-1]
if last.get("b6x"):
    z = sum(wi * xi for wi, xi in zip(m["w"], last["b6x"]))
    print(f"\nlast b6x={last['b6x']}")
    print(f"sigmoid(w·x)={1/(1+math.exp(-z)):.4f} vs logged "
          f"p_win={last.get('p_win')}")
