"""Post-deploy dim check: wait for a fresh t2 commit with the tech block,
then report all live dims."""
import json
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
R = ROOT / "results"

for _ in range(40):
    rows = [json.loads(l) for l in (R / "prediction_log.jsonl").open()]
    t2 = [r for r in rows if r["variant"].startswith("t2") and r.get("x")]
    if t2 and len(t2[-1]["x"]) >= 18:
        break
    time.sleep(15)

print("t2 ctx dim:", len(t2[-1]["x"]) if t2 else "-")
kb = [json.loads(l) for l in (R / "kalshi_binary_log.jsonl").open()]
k3 = [r for r in kb if r.get("variant") == "kb3" and r.get("bx")]
print("kb3 bx dim:", len(k3[-1]["bx"]) if k3 else "-",
      "| trained:", k3[-1].get("trained") if k3 else "-")
lg = json.load(open(R / "kb_logit.json"))
print("kb_logit checkpoint dim:", lg["dim"], "updates:", lg["updates"])
