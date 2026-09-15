"""RL TREATMENT — Phase 0: durable per-window feature store (accrual).

The live binary log (results/kalshi_binary_log.jsonl) is a ROLLING 20k-row buffer,
so only ~2-3 days of feature-rich windows survive before rotation. Training the RL
treatments on that thin, ever-shrinking slice would overfit (exactly the small-n
backtest trap). This script stops the data loss: it is a READ-ONLY tap on the live
log that appends any not-yet-seen rows to an append-only durable store the RL
dataset builder (Phase 1) reads. Run it periodically (like the TEST_V2 refetch) so
the RL training set only ever grows.

Isolation: reads one live-system file, writes one NEW file. It imports nothing from
the live daemon and nothing from the sealed-research program, and never touches T0.

Dedup key: (ticker, made_ts, variant). Idempotent — safe to run repeatedly.
"""
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "results" / "kalshi_binary_log.jsonl"          # live, read-only
DURABLE = ROOT / "results" / "rl_window_log.jsonl"          # new, append-only, durable
STATE = ROOT / "results" / "rl_window_log.state.json"


def _rows(p):
    if not p.exists():
        return []
    out = []
    for l in p.open():
        l = l.strip()
        if not l:
            continue
        try:
            out.append(json.loads(l))
        except json.JSONDecodeError:
            pass
    return out


def _key(r):
    return f"{r.get('ticker')}|{r.get('made_ts')}|{r.get('variant')}"


def run():
    seen = {_key(r) for r in _rows(DURABLE)}
    src = _rows(SRC)
    new = [r for r in src if _key(r) not in seen]
    if new:
        with DURABLE.open("a") as fh:
            for r in new:
                fh.write(json.dumps(r) + "\n")
    total = _rows(DURABLE)
    windows = {r.get("ticker") for r in total}
    settled = {r.get("ticker") for r in total if r.get("actual") in (0, 1)}
    STATE.write_text(json.dumps({
        "schema_version": "rl-window-log-1", "generated_at": time.time(),
        "durable_rows": len(total), "appended_this_run": len(new),
        "distinct_windows": len(windows), "settled_windows": len(settled),
        "source": "results/kalshi_binary_log.jsonl (rolling 20k buffer, read-only tap)",
        "note": "append-only; the RL treatment training set grows from here. Schedule this "
                "to run each cycle so no feature-rich window is lost to log rotation.",
    }, indent=1))
    print(f"rl_window_log: +{len(new)} rows -> {len(total)} durable "
          f"({len(settled)} settled windows, {len(windows)} distinct)")


if __name__ == "__main__":
    run()
