"""
src.phase15 — Phase 15: 1h Intraday Data Store + Dual FTA Calibration
        + Dataset Expansion + Final Acceptance Verdict.

Public API:
    run_phase15        : Full Phase 15 pipeline.
    Phase15Result      : Result dataclass.
    print_phase15_report: Print human-readable report to stdout.
"""
from src.phase15.phase15_runner import Phase15Result, print_phase15_report, run_phase15

__all__ = [
    "run_phase15",
    "Phase15Result",
    "print_phase15_report",
]
