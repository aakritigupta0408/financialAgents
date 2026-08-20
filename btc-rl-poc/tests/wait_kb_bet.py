"""Wait for the betting simulator to evaluate a window; print any bets."""
import json
import time
from pathlib import Path

RES = Path(__file__).resolve().parent.parent / "results"
BETS = RES / "kb_bets.jsonl"

for _ in range(25):
    st = json.load(open(RES / "online_status.json"))
    if time.time() - st["alive_at"] < 60 and "kb_bets" in st:
        break
    time.sleep(15)

st = json.load(open(RES / "online_status.json"))
print("status kb_bets:", json.dumps(st.get("kb_bets"))[:220])
if BETS.exists():
    for line in BETS.read_text().splitlines()[-3:]:
        b = json.loads(line)
        state = "pending" if b["actual"] is None else f"pnl {b['pnl_c']}c"
        print(f"  {b['ticker']}: {b['side'].upper()} @ {b['price_c']}c "
              f"(edge {b['edge_c']}c, p={b['p_model']}, "
              f"{b['mins_left']}m left) -> {state}")
else:
    print("no bets yet (no window has shown >=5c edge under 85c)")
