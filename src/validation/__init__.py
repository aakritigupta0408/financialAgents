"""
src.validation — Phase 11 validation layer.

Public API
----------
BenchmarkRunner    : Run multiple BacktestEngine configs on synthetic series.
AblationRunner     : Measure marginal component contributions.
RobustnessRunner   : Stress-test under slippage, fees, regimes, and sample sizes.
DataSplitValidator : Time-based train/val/test and walk-forward splits.
ValidationSummary  : Aggregated pass/fail report dataclass.
run_full_validation: Orchestrate the full Phase 11 pipeline.
generate_validation_summary: Assemble a ValidationSummary from sub-results.
"""

from src.validation.benchmark import BenchmarkRunner, BenchmarkResult
from src.validation.ablation import AblationRunner, AblationResult
from src.validation.robustness import RobustnessRunner, RobustnessResult
from src.validation.data_split import DataSplitValidator, DataSplitResult
from src.validation.summary import (
    ValidationSummary,
    PassFailCriteria,
    generate_validation_summary,
)
from src.validation.runner import run_full_validation
from src.validation.real_validation import (
    RealValidationResult,
    run_real_validation,
    print_real_validation_report,
)
from src.validation.intraday_synthetic import (
    make_structured_1h_series,
    load_or_generate_1h_series,
)
from src.validation.fta_calibration import (
    FTACalibrationResult,
    CalibrationSummary,
    run_fta_calibration,
)
from src.validation.phase13_runner import (
    Phase13Result,
    run_phase13,
    print_phase13_report,
)

__all__ = [
    "BenchmarkRunner",
    "BenchmarkResult",
    "AblationRunner",
    "AblationResult",
    "RobustnessRunner",
    "RobustnessResult",
    "DataSplitValidator",
    "DataSplitResult",
    "ValidationSummary",
    "PassFailCriteria",
    "generate_validation_summary",
    "run_full_validation",
    "RealValidationResult",
    "run_real_validation",
    "print_real_validation_report",
    # Phase 13
    "make_structured_1h_series",
    "load_or_generate_1h_series",
    "FTACalibrationResult",
    "CalibrationSummary",
    "run_fta_calibration",
    "Phase13Result",
    "run_phase13",
    "print_phase13_report",
]
