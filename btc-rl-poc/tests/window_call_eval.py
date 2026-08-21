"""Window-level evaluation: ONE final call per window at a fixed
decision time (the contract's natural unit — no abstention, every
window counted). Per-class precision/recall, chronological halves."""
import json
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
rows = [json.loads(l) for l in
        (ROOT / "results" / "kalshi_binary_log.jsonl").read_text().splitlines()]
byw = defaultdict(list)
for r in rows:
    if r.get("variant", "kb") == "kb" and r["actual"] is not None:
        byw[r["ticker"]].append(r)
for v in byw.values():
    v.sort(key=lambda r: r["made_ts"])
windows = sorted(byw, key=lambda t: byw[t][0]["made_ts"])


def final_call(calls, decide_at):
    """The last call made with mins_left >= decide_at (closest to it)."""
    ok = [r for r in calls if r["mins_left"] >= decide_at]
    return ok[-1] if ok else None


def prf(sample):
    out = {}
    for cls, want in (("UP", 1), ("DOWN", 0)):
        called = [r for r in sample if r["call"] == want]
        actual = [r for r in sample if r["actual"] == want]
        out[cls] = (
            sum(r["hit"] for r in called) / len(called) if called else 0,
            (sum(1 for r in actual if r["call"] == want) / len(actual)
             if actual else 0),
            len(called), len(actual))
    return out


for decide_at in (5, 4, 3, 2):
    sample = [c for t in windows
              if (c := final_call(byw[t], decide_at)) is not None]
    half = len(sample) // 2
    print(f"decision at T-{decide_at} min ({len(sample)} windows):")
    for label, part in (("all", sample), ("1st half", sample[:half]),
                        ("2nd half", sample[half:])):
        e = prf(part)
        acc = sum(r["hit"] for r in part) / len(part)
        print(f"  {label:>8}: acc {acc:.1%} | "
              f"UP p {e['UP'][0]:.1%} r {e['UP'][1]:.1%} "
              f"(n {e['UP'][2]}/{e['UP'][3]}) | "
              f"DOWN p {e['DOWN'][0]:.1%} r {e['DOWN'][1]:.1%} "
              f"(n {e['DOWN'][2]}/{e['DOWN'][3]})")
    print()
