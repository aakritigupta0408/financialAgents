"""
Seed the local data store from existing fixtures.
Run once to bootstrap the store with Phase 12 daily data.

Usage:
    cd /Users/aakritigupta/trading-system && python scripts/seed_data_store.py
"""
from src.data_store.sync import sync_from_fixtures
from src.data_store.inventory import DataInventory
from src.data_store.replay import ReplayLoader

print("Seeding local data store from Phase 12 fixtures...")

# 1. Sync all fixture CSVs into store
rows = sync_from_fixtures()
print("\nRows written per ticker:")
for ticker, n in rows.items():
    print(f"  {ticker}: {n} rows written")

# 2. Print inventory
print("\nData inventory:")
inv = DataInventory()
inv.print_inventory()

# 3. Test replay loading
print("\nReplay loader test:")
loader = ReplayLoader()
for ticker in ["AAPL", "MSFT", "NVDA"]:
    try:
        series = loader.load(ticker, timeframe="1d")
        print(
            f"  Replay {ticker} 1d: {len(series.bars)} bars, "
            f"{series.bars[0].timestamp.date()} -> {series.bars[-1].timestamp.date()}"
        )
    except Exception as exc:
        print(f"  Replay {ticker} 1d: FAILED — {exc}")

print("\nDone.")
