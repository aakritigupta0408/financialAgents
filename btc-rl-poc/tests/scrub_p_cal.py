"""Remove the p_cal values the M1 shadow layer wrote onto kb-variant
rows between 2026-08-28 09:50 (first shadow restart) and the rename.

Why this is necessary: `p_cal` already existed as a field on kb/kb2
rows, and _kb_blend_weights() reads r.get("p_cal", r["p_up"]) over
SETTLED kb rows to fit kb2's market-blend weight. The shadow layer
stamped its own calibrated value into that field on unsettled rows;
those rows then settled carrying it, so a layer that was supposed to
trade nothing was steering kb2's live blend.

kb rows never legitimately carry p_cal (for variant "kb", p_up IS the
calibrated value; only kb2 rows carry a p_cal, as a record of the
input). So: drop p_cal from every "kb" row that also has our p_m1, or
that was written during the contaminated window.
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LOG = ROOT / "results" / "kalshi_binary_log.jsonl"
CONTAM_FROM = 1787849400          # 2026-08-28 09:50 PT, shadow restart

rows = [json.loads(l) for l in LOG.open() if l.strip()]
fixed = 0
for r in rows:
    v = r.get("variant") or "kb"
    if v != "kb":
        continue                  # kb2's own p_cal is legitimate
    if "p_cal" in r and (r.get("p_m1") is not None
                         or r.get("made_ts", 0) >= CONTAM_FROM):
        r.pop("p_cal", None)
        fixed += 1
print(f"scanned {len(rows)} rows · removed contaminated p_cal from "
      f"{fixed} kb rows")
if fixed:
    tmp = LOG.with_suffix(".scrub")
    tmp.write_text("".join(json.dumps(r) + "\n" for r in rows))
    tmp.replace(LOG)
    print("rewrote the log in place (values removed, no row deleted, "
          "no outcome altered)")
else:
    print("nothing to scrub")
