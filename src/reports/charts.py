"""
src.reports.charts — Matplotlib chart generation for backtest results.

generate_charts(result, output_dir) writes PNG files to output_dir and returns
a list of absolute file paths. All charts are generated headlessly (no plt.show).

Matplotlib is a soft dependency: if it is not installed, chart generation is
skipped gracefully and an empty list is returned.

Charts generated
----------------
1. equity_curve.png     — equity over time from result.equity_curve
2. drawdown_curve.png   — rolling drawdown % from peak
3. trade_pnl_histogram.png — histogram of realized_pnl per trade
4. forecast_confidence_histogram.png — histogram of forecast_confidence from
                           meta_features (skipped if no meta_features present)
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.backtest.result import BacktestResult

try:
    import matplotlib
    matplotlib.use("Agg")  # headless backend before importing pyplot
    import matplotlib.pyplot as plt
    _MPL_AVAILABLE = True
except ImportError:
    matplotlib = None  # type: ignore[assignment]
    plt = None  # type: ignore[assignment]
    _MPL_AVAILABLE = False


def generate_charts(
    result: "BacktestResult",
    output_dir: str | Path,
) -> list[str]:
    """
    Generate all backtest charts and save them to output_dir.

    Parameters
    ----------
    result     : BacktestResult produced by BacktestEngine.run().
    output_dir : Directory to write PNG files to. Created if it does not exist.

    Returns
    -------
    List of absolute file path strings for the written chart files.
    Returns an empty list if matplotlib is not available.
    """
    if not _MPL_AVAILABLE:
        return []

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    written: list[str] = []

    written += _plot_equity_curve(result, output_path)
    written += _plot_drawdown_curve(result, output_path)
    written += _plot_trade_pnl_histogram(result, output_path)
    written += _plot_forecast_confidence_histogram(result, output_path)

    return written


# ── Chart helpers ─────────────────────────────────────────────────────────────


def _save(fig, path: Path) -> str:
    """Save figure, close it, and return the absolute path string."""
    fig.savefig(str(path), dpi=100, bbox_inches="tight")
    plt.close(fig)
    return str(path.resolve())


def _plot_equity_curve(result: "BacktestResult", output_path: Path) -> list[str]:
    """Line chart of equity over time."""
    ec = result.equity_curve
    if not ec:
        return []

    timestamps = [ts for ts, _ in ec]
    equities = [eq for _, eq in ec]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(timestamps, equities, linewidth=1.5, color="#2196F3")
    ax.set_title(f"Equity Curve — {result.ticker} / {result.timeframe}")
    ax.set_xlabel("Time")
    ax.set_ylabel("Equity ($)")
    ax.grid(True, alpha=0.3)
    fig.autofmt_xdate()

    return [_save(fig, output_path / "equity_curve.png")]


def _plot_drawdown_curve(result: "BacktestResult", output_path: Path) -> list[str]:
    """Rolling drawdown % from peak at each point in equity_curve."""
    ec = result.equity_curve
    if not ec:
        return []

    timestamps = [ts for ts, _ in ec]
    equities = [eq for _, eq in ec]

    drawdowns: list[float] = []
    peak = equities[0] if equities else 0.0
    for eq in equities:
        if eq > peak:
            peak = eq
        dd = (peak - eq) / peak * 100.0 if peak > 0 else 0.0
        drawdowns.append(-dd)  # negative so drawdown shows as going down

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.fill_between(timestamps, drawdowns, 0, alpha=0.4, color="#F44336")
    ax.plot(timestamps, drawdowns, linewidth=1.0, color="#F44336")
    ax.set_title(f"Drawdown Curve — {result.ticker} / {result.timeframe}")
    ax.set_xlabel("Time")
    ax.set_ylabel("Drawdown (%)")
    ax.grid(True, alpha=0.3)
    fig.autofmt_xdate()

    return [_save(fig, output_path / "drawdown_curve.png")]


def _plot_trade_pnl_histogram(result: "BacktestResult", output_path: Path) -> list[str]:
    """Histogram of realized_pnl per closed trade."""
    journal = result.trade_journal or []
    closed = [t for t in journal if t.get("exit_price") is not None]

    if not closed:
        return []

    pnls = [float(t.get("realized_pnl", 0.0)) for t in closed]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(pnls, bins=20, color="#4CAF50", edgecolor="white", alpha=0.8)
    ax.axvline(0, color="black", linewidth=1.0, linestyle="--")
    ax.set_title(f"Trade PnL Distribution — {result.ticker}")
    ax.set_xlabel("Realized PnL ($)")
    ax.set_ylabel("Count")
    ax.grid(True, alpha=0.3, axis="y")

    return [_save(fig, output_path / "trade_pnl_histogram.png")]


def _plot_forecast_confidence_histogram(
    result: "BacktestResult", output_path: Path
) -> list[str]:
    """Histogram of forecast_confidence from meta_features. Skipped if no data."""
    journal = result.trade_journal or []
    closed = [t for t in journal if t.get("exit_price") is not None]

    confidences: list[float] = []
    for t in closed:
        mf = t.get("meta_features") or {}
        fc = mf.get("forecast_confidence")
        if fc is not None:
            try:
                confidences.append(float(fc))
            except (TypeError, ValueError):
                pass

    if not confidences:
        return []

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(confidences, bins=15, color="#9C27B0", edgecolor="white", alpha=0.8)
    ax.set_title(f"Forecast Confidence Distribution — {result.ticker}")
    ax.set_xlabel("Forecast Confidence")
    ax.set_ylabel("Count")
    ax.set_xlim(0.0, 1.0)
    ax.grid(True, alpha=0.3, axis="y")

    return [_save(fig, output_path / "forecast_confidence_histogram.png")]
