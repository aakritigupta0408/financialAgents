"""Post-deploy sanity: selector policy dim/quality, kb3 checkpoint
continuity, and pf stamping on fresh rows. Run after any daemon restart."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
R = ROOT / "results"

p = json.load(open(R / "kb_sel_policy.json"))
print("policy:", {k: p.get(k) for k in
                  ("theta", "precision", "met_target", "n_train")},
      "| dim", len(p.get("w") or []))
lg = json.load(open(R / "kb_logit.json"))
print("kb3: dim", lg["dim"], "updates", lg["updates"])
kb = [json.loads(l) for l in open(R / "kalshi_binary_log.jsonl")]
new = [r for r in kb if r.get("pf")]
print("rows with pf stamped:", len(new))
if new:
    r = new[-1]
    print("latest pf:", r["pf"], "| variant", r["variant"],
          "| bx dim:", len(r.get("bx") or []) or "-")
