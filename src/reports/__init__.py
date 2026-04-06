"""
src.reports — Phase 9 reporting layer.

Public API
----------
generate_full_report        — orchestrates all sub-reports; writes report.json
generate_portfolio_report   — portfolio-level performance metrics
generate_decision_report    — decision funnel: accepted / rejected / confidence
generate_trade_diagnostics  — per-trade diagnostic list
generate_model_diagnostics  — feature importance, calibration, confidence stats
sweep_thresholds            — threshold sensitivity analysis (no backtest re-run)
generate_charts             — matplotlib PNG chart generation
"""

from src.reports.charts import generate_charts
from src.reports.decisions import generate_decision_report
from src.reports.model_diagnostics import generate_model_diagnostics
from src.reports.portfolio import generate_portfolio_report
from src.reports.runner import generate_full_report
from src.reports.threshold_tuning import sweep_thresholds
from src.reports.trades import generate_trade_diagnostics

__all__ = [
    "generate_full_report",
    "generate_portfolio_report",
    "generate_decision_report",
    "generate_trade_diagnostics",
    "generate_model_diagnostics",
    "sweep_thresholds",
    "generate_charts",
]
