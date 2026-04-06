"""
Full Phase 15 run.

Expected runtime: ~5 minutes for 6 tickers x 16 calibration points x 7 configs.

Usage:
    python scripts/run_phase15.py
"""
import sys
sys.path.insert(0, ".")

from src.phase15.phase15_runner import run_phase15, print_phase15_report

if __name__ == "__main__":
    result = run_phase15(n_bars=800, save_model=False)
    print_phase15_report(result)
