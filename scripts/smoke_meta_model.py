"""
scripts/smoke_meta_model.py — Phase 7 smoke test.

Runs without importing sklearn at module level.
All sklearn usage is inside the src.meta_model package guarded by try/except.

Usage
-----
    cd /Users/aakritigupta/trading-system
    python scripts/smoke_meta_model.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure repo root is on sys.path when running as a script.
_ROOT = Path(__file__).parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# ---------------------------------------------------------------------------
# Step 1: Generate backtest data.
# ---------------------------------------------------------------------------
print("Step 1: Running BacktestEngine on synthetic data ...")

from src.backtest import BacktestEngine, make_synthetic_ohlcv  # noqa: E402

series = make_synthetic_ohlcv(n_bars=300, seed=42, trend=0.0003)
result = BacktestEngine(starting_capital=100_000, verbose=False).run(series)
print(f"  Trades generated: {len(result.trade_journal)}")

# ---------------------------------------------------------------------------
# Step 2: Verify meta_features are stored.
# ---------------------------------------------------------------------------
print("\nStep 2: Checking meta_features presence ...")
with_features = [t for t in result.trade_journal if t.get("meta_features")]
print(f"  Trades with meta_features: {len(with_features)}")
if with_features:
    sample_keys = sorted(with_features[0]["meta_features"].keys())
    print(f"  Feature keys in first entry: {sample_keys}")

# ---------------------------------------------------------------------------
# Step 3: Build dataset and run training pipeline.
# ---------------------------------------------------------------------------
print("\nStep 3: Running training pipeline ...")
from src.meta_model import run_training_pipeline  # noqa: E402

model, metrics = run_training_pipeline([result], save_model=False)
print(f"  Model type: {type(model).__name__}")
if "warning" in metrics:
    print(f"  Warning: {metrics['warning']}")
else:
    val_m = metrics.get("val", {})
    print(f"  Val metrics: acc={val_m.get('accuracy')} f1={val_m.get('f1')} roc={val_m.get('roc_auc')}")

# ---------------------------------------------------------------------------
# Step 4: Score a sample trade.
# ---------------------------------------------------------------------------
print("\nStep 4: Scoring a sample trade ...")
from src.meta_model import score_trade  # noqa: E402
from src.backtest.candidate import generate_candidate  # noqa: E402
from src.backtest.data_utils import build_snapshot_from_series  # noqa: E402
from src.features.pipeline import compute_all_features  # noqa: E402
from src.features.volatility import compute_volatility  # noqa: E402
from src.timesfm import run_forecast  # noqa: E402

snapshot = build_snapshot_from_series(series, t_idx=200, context_bars=100)
feat = compute_all_features(snapshot, primary_tf="1h")
forecast = run_forecast(
    series=snapshot.tf_1h, horizon=10, ticker="SYN", timeframe="1h"
)
vol = compute_volatility(snapshot.tf_1h.to_dataframe(), "SYN", "1h")
cand = generate_candidate(forecast, vol, snapshot.tf_1h.bars[-1].close)

if cand:
    out = score_trade(feat, forecast, cand, model=model)
    print(
        f"  score_trade: prob={out.probability_of_success:.3f} "
        f"confidence={out.confidence:.3f} "
        f"should_trade={out.should_trade}"
    )
else:
    print("  No candidate generated at t=200")

print("\nSmoke test complete.")
