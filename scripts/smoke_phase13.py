"""Phase 13 smoke: calibration + acceptance test on 3 1h synthetic tickers."""
from src.validation.phase13_runner import run_phase13, print_phase13_report

result = run_phase13(tickers=["AAPL_1H", "MSFT_1H", "NVDA_1H"], n_bars=600)
print_phase13_report(result)
