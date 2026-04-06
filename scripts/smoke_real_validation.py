"""Phase 12 smoke: run real market validation and print report."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.validation.real_validation import run_real_validation, print_real_validation_report

result = run_real_validation(tickers=["AAPL", "MSFT", "NVDA"])
print_real_validation_report(result)
print("\nOverall pass_fail:")
pf = result.validation_summary.pass_fail
for field_name in ["full_system_beats_forecast_only", "full_system_reduces_drawdown",
                    "consistent_across_tickers", "survives_slippage", "adaptive_does_not_degrade"]:
    print(f"  {field_name}: {getattr(pf, field_name)}")
print(f"\nOVERALL: {'PASS' if result.validation_summary.overall_passed else 'FAIL'}")
