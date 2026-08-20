"""Dry-run the metric-history snapshot on real data (no writes)."""
import ast
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
for f in ("btc_rl/online.py", "btc_rl/train.py", "btc_rl/history.py",
          "scripts/train_l2.py", "scripts/train_l3.py", "scripts/train_l4.py"):
    ast.parse((ROOT / f).read_text())
print("syntax ok")

from btc_rl.online import (_history_snapshot, _load_kb, _load_kb_bets,
                           _load_ledger)

snap = _history_snapshot(_load_ledger(), _load_kb(), _load_kb_bets(),
                         int(time.time()))
some = next(iter(snap["arms"].get("t2-h5", {}).values()), None)
print("arms:", len(snap["arms"]), " sample t2-h5:", some)
print("kb:", snap.get("kb"))
print("bets:", snap.get("bets"))
