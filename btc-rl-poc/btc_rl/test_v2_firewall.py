"""TEST_V2 research firewall — NO_TEST_V2_DEV_ACCESS (directive).

The sealed TEST_V2 labels necessarily exist operationally now (they are official
Kalshi settlements). The remaining risk is therefore NOT human peeking but accidental
CODE REUSE: a loader that silently pulls a post-cutoff window into any development or
selection step. This guard makes that impossible to do by accident.

Invariant enforced:
    intersection(development_ids, TEST_V2_membership_ids) == empty
    AND no development row has T0 > cutoff   (post-cutoff == V2 by definition)

Call assert_no_test_v2_access(...) from EVERY development/selection path:
  dataset load · feature selection · normalization fitting · hyperparameter search ·
  calibration · threshold tuning · ensemble fitting · architecture comparison.

This guard is worth more than another model: it protects the one-time blind test.
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MEMBERSHIP = ROOT / "research" / "true15m" / "TEST_V2_MEMBERSHIP.jsonl"
SPLIT = ROOT / "research" / "true15m" / "SPLIT_SPEC_V1.json"


def cutoff():
    return json.loads(SPLIT.read_text())["sealed_test"]["range"][1]


def membership_ids():
    if not MEMBERSHIP.exists():
        return set()
    return {json.loads(l)["market_window_id"] for l in MEMBERSHIP.open() if l.strip()}


def assert_no_test_v2_access(dev_ids=None, dev_t0s=None, context="development"):
    """Raise AssertionError if any development row belongs to TEST_V2.

    dev_ids : iterable of market_window_id used for development/selection
    dev_t0s : iterable of T0 (epoch seconds) — a second, id-independent check
    context : short label naming the step being guarded (for the error message)
    """
    cut = cutoff()
    v2 = membership_ids()
    problems = []
    if dev_ids is not None:
        inter = set(dev_ids) & v2
        if inter:
            problems.append(f"{len(inter)} development id(s) are in TEST_V2 membership "
                            f"(e.g. {sorted(inter)[:3]})")
    if dev_t0s is not None:
        leaked = [t for t in dev_t0s if t is not None and t > cut]
        if leaked:
            problems.append(f"{len(leaked)} development row(s) have T0 > cutoff {cut} "
                            "(post-cutoff windows are TEST_V2 by definition)")
    if problems:
        raise AssertionError(f"NO_TEST_V2_DEV_ACCESS violated in [{context}]: "
                             + "; ".join(problems))
    return True


def guard_rows(rows, context="development"):
    """Convenience: guard a list of dataset rows (each with market_window_id + T0)
    and return them unchanged so callers can wrap their load in one line:
        rows = guard_rows([json.loads(l) for l in DS.open()], "loss_sweeps")
    """
    assert_no_test_v2_access(
        dev_ids=[r.get("market_window_id") for r in rows],
        dev_t0s=[r.get("T0") for r in rows], context=context)
    return rows
