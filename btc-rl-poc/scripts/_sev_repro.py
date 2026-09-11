import json
import time
from datetime import datetime

import btc_rl.config as config
from btc_rl.online import BACKFILL_HOURS, _load_kb_bets, PT_LOG_NAME
from btc_rl.sources import fetch_range

now_ts = int(time.time())
pt = _load_kb_bets(PT_LOG_NAME)
print("pt rows loaded:", len(pt))
z = [t for t in pt if t.get("ticker") == "KXBTC15M-26AUG311800-00"]
print("zombie present:", len(z), "| fields:",
      {k: z[0].get(k) for k in ("actual", "close_ts", "strike",
                                "side", "contracts", "stake_c")}
      if z else None)

# detection condition (exactly as daemon)
stale = set()
for t in pt:
    cts = t.get("close_ts")
    cond = (t.get("actual") is None and cts and now_ts >= cts
            and cts < now_ts - BACKFILL_HOURS * 3600)
    if t.get("actual") is None and cts:
        pass
    if cond:
        stale.add(cts)
print("stale close_ts detected:", sorted(stale))

# targeted fetch + settle sim for the zombie
if z:
    cts = z[0]["close_ts"]
    bars = fetch_range(datetime.fromtimestamp(cts - 240,
                                              tz=config.PACIFIC),
                       datetime.fromtimestamp(cts + 120,
                                              tz=config.PACIFIC))
    by = {b["ts"]: b for b in bars}
    sb = by.get(cts - 60)
    print("settle bar present:", sb is not None,
          "| synth:", sb.get("synth") if sb else None,
          "| close:", sb.get("close") if sb else None)
