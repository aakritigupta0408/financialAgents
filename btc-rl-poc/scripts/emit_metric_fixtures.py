"""B3: emit results/metric_fixtures.json — frozen input/output vectors
for every public function in btc_rl/metrics.py.

Usage:  python3 scripts/emit_metric_fixtures.py
Idempotent: inputs are handcrafted constants and every function under
test is pure, so re-runs are byte-identical except generated_ts.

WHY: the UI redesign (R1) extracts a shared site/lib.js and deletes the
in-page metric twins. The migration gate is "byte-identical metric
output vs current pages on the B3 frozen snapshot" — which requires a
snapshot computed by CALLING the canonical Python implementations, not
by transcribing formulas (transcription is how twins drift in the first
place). Expected values below are whatever btc_rl.metrics returns at
emit time; nothing is hand-computed.

Fixture shape, per function:
    {"in": {kwargs...}, "out": <return value>}
`out` is null where the Python function returns None (insufficient n,
zero denominator, degenerate variance) — lib.js must reproduce the
None/null contract too, not only the happy path.

Cases deliberately cover the documented edge behaviors: empty inputs,
zero-denominator guards, the p == 1.0 top-bin rule in calibration_bins,
pt_test's n < 20 and degenerate-variance Nones, and kalshi_fee_c's
round-UP-to-the-cent rule at a sub-cent raw fee (ask 99c).
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent))     # for btc_rl.metrics
sys.path.insert(0, str(_HERE))            # for emit_common
from emit_common import RESULTS_DIR, atomic_write_json  # noqa: E402
from btc_rl import metrics as M           # noqa: E402

OUT_PATH = RESULTS_DIR / "metric_fixtures.json"

# handcrafted deterministic inputs, dollar/probability scales chosen to
# resemble the real feeds (BTC dollar errors, 0..1 probabilities)
_CASES: dict = {
    "mase": [
        {"abs_errs": [10.0, 20.0, 30.0], "naive_abs_errs": [20.0, 20.0, 20.0]},
        {"abs_errs": [5.0, 7.5], "naive_abs_errs": [2.0, 3.0]},
        {"abs_errs": [], "naive_abs_errs": [1.0]},          # -> None
    ],
    "msse": [
        {"abs_errs": [10.0, 20.0, 30.0], "naive_abs_errs": [20.0, 20.0, 20.0]},
        {"abs_errs": [3.0, 4.0], "naive_abs_errs": [0.0, 0.0]},  # -> None
    ],
    "rmse": [
        {"errs": [3.0, -4.0]},
        {"errs": [-1.5, 2.5, 0.0, 1.0]},
        {"errs": []},                                       # -> None
    ],
    "pinball": [
        {"actual": 77400.0, "lo": 77350.0, "hi": 77475.0},  # inside band
        {"actual": 77300.0, "lo": 77350.0, "hi": 77475.0},  # below lo
        {"actual": 77500.0, "lo": 77350.0, "hi": 77475.0,
         "q_lo": 0.05, "q_hi": 0.95},                       # above hi, 90% band
    ],
    "sharpness": [
        {"los": [77350.0, 77245.0], "his": [77475.0, 77519.0]},
        {"los": [], "his": []},                             # -> None
    ],
    "pt_test": [
        # 25 obs, alternating actuals, forecaster right 20/25
        {"pred_up": [i % 2 == 0 if i % 5 else i % 2 == 1
                     for i in range(25)],
         "actual_up": [i % 2 == 0 for i in range(25)]},
        {"pred_up": [True] * 10, "actual_up": [True] * 10},  # n<20 -> None
        # all-up forecaster: degenerate variance -> None
        {"pred_up": [True] * 25,
         "actual_up": [i % 2 == 0 for i in range(25)]},
    ],
    "brier_skill": [
        {"brier": 0.20, "reference": 0.25},
        {"brier": 0.30, "reference": 0.25},                 # negative skill
        {"brier": 0.20, "reference": 0.0},                  # -> None
    ],
    "calibration_bins": [
        {"ps": [0.05, 0.15, 0.15, 0.55, 0.55, 0.55, 0.95, 1.0],
         "ys": [0, 0, 1, 1, 0, 1, 1, 1]},                   # p=1.0 top bin
        {"ps": [0.1, 0.3, 0.6, 0.9], "ys": [0, 1, 1, 1], "n_bins": 4},
    ],
    "kalshi_fee_c": [
        {"price_c": 50.0},        # 1.75 raw -> 2
        {"price_c": 60.0},        # 1.68 raw -> 2
        {"price_c": 99.0},        # 0.0693 raw -> 1 (round-UP rule)
    ],
    "max_drawdown": [
        {"cum": [0.0, 5.0, 3.0, 8.0, 1.0, 4.0]},            # peak 8 -> 1
        {"cum": [0.0, 1.0, 2.0, 3.0]},                      # monotone -> 0
        {"cum": []},                                        # -> 0.0
    ],
}


def build_fixtures() -> dict:
    fixtures: dict = {}
    for fname, cases in _CASES.items():
        fn = getattr(M, fname)
        fixtures[fname] = [{"in": kwargs, "out": fn(**kwargs)}
                           for kwargs in cases]
    return {
        "generated_ts": round(time.time(), 3),
        "source": "btc_rl/metrics.py",
        "contract": "site/lib.js must reproduce every `out` byte-"
                    "identically after JSON round-trip, including nulls",
        "fixtures": fixtures,
    }


if __name__ == "__main__":
    payload = build_fixtures()
    atomic_write_json(OUT_PATH, payload)
    n = sum(len(v) for v in payload["fixtures"].values())
    print(f"wrote {OUT_PATH} "
          f"({len(payload['fixtures'])} functions, {n} cases)")
