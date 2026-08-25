"""Publish the Conviction Book (pb_bets.jsonl) and retire v1/v2 labels:
- publisher DATA gains pb_bets.jsonl
- Results A/B table gains a 'Conviction Book (kb5-gated)' row
- selector generations table marks v1/v2 retired in their design text
"""
from pathlib import Path

root = Path(__file__).resolve().parent.parent

pub = root / "scripts" / "publish_dashboard.py"
t = pub.read_text()
assert '("kb_bets_sel_prepolicy.jsonl", None),' in t
t = t.replace('("kb_bets_sel_prepolicy.jsonl", None),',
              '("kb_bets_sel_prepolicy.jsonl", None),\n'
              '    ("pb_bets.jsonl", None),')
pub.write_text(t)

dash = root / "site" / "ab_dashboard.html"
t = dash.read_text()
t = t.replace('''    const [kb, bets, sel, pre, stRes] = await Promise.all([
      jsonl("../results/kalshi_binary_log.jsonl"),
      jsonl("../results/kb_bets.jsonl"),
      jsonl("../results/kb_bets_sel.jsonl"),
      jsonl("../results/kb_bets_sel_prepolicy.jsonl"),''',
'''    const [kb, bets, sel, pre, pb, stRes] = await Promise.all([
      jsonl("../results/kalshi_binary_log.jsonl"),
      jsonl("../results/kb_bets.jsonl"),
      jsonl("../results/kb_bets_sel.jsonl"),
      jsonl("../results/kb_bets_sel_prepolicy.jsonl"),
      jsonl("../results/pb_bets.jsonl"),''')
t = t.replace('''      row("Selector v3 (live policy)", "#199e70", B, skippedN);''',
'''      row("Selector v3 (live policy)", "#199e70", B, skippedN) +
      row("Conviction Book (kb5-gated, +EV only)", "#d95926",
          stats(pb), 0);''')
t = t.replace('["v1", "τ-confidence gate (per-minute call precision)",',
              '["v1", "τ-confidence gate — RETIRED (never scaled)",')
t = t.replace('["v2", "bet-EV logit, mid-priced counterfactual training",',
              '["v2", "bet-EV logit, mid-priced training — RETIRED (subtracted value)",')
dash.write_text(t)
print("pb published + v1/v2 retired labels")
