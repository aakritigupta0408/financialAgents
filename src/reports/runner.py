"""
src.reports.runner — Full report orchestrator.

generate_full_report(result, output_dir) calls all sub-report generators and
returns a single merged dict. Optionally writes report.json to output_dir.

Datetime serialisation
----------------------
Datetime objects are kept as Python datetime instances inside the returned dict.
When serialising to JSON (report.json), they are converted to ISO-format strings
via the _json_default helper. The returned Python dict always uses datetime objects
so callers can do date arithmetic without parsing.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

from src.reports.charts import generate_charts
from src.reports.decisions import generate_decision_report
from src.reports.model_diagnostics import generate_model_diagnostics
from src.reports.portfolio import generate_portfolio_report
from src.reports.threshold_tuning import sweep_thresholds
from src.reports.trades import generate_trade_diagnostics

if TYPE_CHECKING:
    from src.backtest.result import BacktestResult


def _json_default(obj):
    """JSON serialisation fallback for datetime objects and other non-serialisable types."""
    if isinstance(obj, datetime):
        return obj.isoformat()
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serialisable")


def generate_full_report(
    result: "BacktestResult",
    output_dir: str | Path | None = None,
) -> dict:
    """
    Orchestrate all sub-report generators and return a merged report dict.

    Parameters
    ----------
    result     : BacktestResult produced by BacktestEngine.run().
    output_dir : Optional directory for chart PNGs and report.json.
                 Charts are skipped if output_dir is None.

    Returns
    -------
    dict with keys:
        "portfolio"           : generate_portfolio_report(result)
        "decisions"           : generate_decision_report(result)
        "model_diagnostics"   : generate_model_diagnostics(result)
        "threshold_sensitivity": sweep_thresholds(result)
        "trade_diagnostics"   : generate_trade_diagnostics(result)
        "charts"              : list of written file paths, or []
    """
    portfolio_report = generate_portfolio_report(result)
    decisions_report = generate_decision_report(result)
    model_diag = generate_model_diagnostics(result)
    threshold_sens = sweep_thresholds(result)
    trade_diag = generate_trade_diagnostics(result)

    charts: list[str] = []
    if output_dir is not None:
        charts = generate_charts(result, output_dir)

    report = {
        "portfolio": portfolio_report,
        "decisions": decisions_report,
        "model_diagnostics": model_diag,
        "threshold_sensitivity": threshold_sens,
        "trade_diagnostics": trade_diag,
        "charts": charts,
    }

    # Persist report.json if output_dir was provided.
    if output_dir is not None:
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        report_json_path = out_path / "report.json"
        with open(report_json_path, "w", encoding="utf-8") as fh:
            json.dump(report, fh, default=_json_default, indent=2)

    return report
