"""Record a human directional view for the RLHF arm (t11).

Usage:  python scripts/feedback.py up             # "I think BTC rises"
        python scripts/feedback.py down --note "CPI print in 20 min"

Each vote is one JSONL row in results/human_feedback.jsonl. For the next
30 minutes, t11's online reward gets a bonus when its committed delta
agrees with the latest view and a penalty when it disagrees; the actual
price outcome always remains the dominant reward term.
"""
import argparse
import json
import time
from pathlib import Path

HF_LOG = Path(__file__).resolve().parent.parent / "results" / "human_feedback.jsonl"

parser = argparse.ArgumentParser()
parser.add_argument("view", choices=["up", "down"])
parser.add_argument("--note", default=None)
args = parser.parse_args()

row = {"ts": int(time.time()), "view": 1 if args.view == "up" else -1,
       "note": args.note}
with HF_LOG.open("a") as f:
    f.write(json.dumps(row) + "\n")
print(f"recorded: {args.view} at {time.strftime('%H:%M:%S')}"
      + (f" — {args.note}" if args.note else ""))
