# Trading System

A local paper-trading research framework for backtesting and live-simulated trading using multi-stage feature engineering, forecasting, signal filters, and portfolio risk management.

> This framework is designed for research purposes. Trading performance may vary based on factors such as data quality, model settings, risk limits, and market conditions. This is not financial advice.

## News

- [2026-04] `trading-system` is organized around phase-based validation, backtesting fidelity, and pluggable signal filters.
- [2026-03] Added local-first data caching, feature pipeline fallback, and modular portfolio execution.
- [2026-02] Built a layered architecture with separate data, feature, forecast, signal, and execution modules.

## Framework Overview

`trading-system` decomposes the trading research workflow into clear modular components:

- **Data Layer**: Local-first OHLCV retrieval, cache, inventory, and persistence.
- **Feature Layer**: Structured feature pipeline for trend, support/resistance, volatility, and liquidity.
- **Forecast Layer**: Forecast scaffolding using the TimesFM wrapper and configurable meta-model scoring.
- **Signal Layer**: Candidate generation, FTA structural validation, meta-model confidence scoring, and risk gating.
- **Execution Layer**: Portfolio engine for position lifecycle, mark-to-market, equity tracking, and journal logging.

## Key Characteristics

- No-lookahead historical backtests
- Deterministic per-bar decision flow
- Immutable audit trails and journals
- Pluggable filters and thresholds
- Phase-based validation and smoke tests

## Installation

```bash
cd trading-system
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

## Dependencies

Dependencies are managed in `pyproject.toml` and include:

- pandas
- numpy
- scipy
- scikit-learn
- requests
- diskcache
- pydantic
- pyarrow
- python-dotenv
- structlog

Optional TimesFM support:

```bash
pip install .[timesfm]
```

> TimesFM support is currently guarded by environment compatibility and requires a compatible Python runtime.

## Required Environment Variables

The project reads configuration from environment variables and `.env` via `python-dotenv`.

Recommended variables:

```bash
export ALPHA_VANTAGE_API_KEY=your_api_key_here
export NEWS_ENABLED=false
export MAX_CONCURRENT_POSITIONS=3
export META_MODEL_MIN_CONFIDENCE=0.60
```

## CLI / Usage

Run the core research/smoke scripts from the `scripts/` folder.

Example backtest smoke test:

```bash
python scripts/smoke_backtest.py
```

Example data validation smoke test:

```bash
python scripts/smoke_data.py
```

Example live loop smoke test:

```bash
python scripts/smoke_live_loop.py
```

Run the phase 15 orchestration script:

```bash
python scripts/run_phase15.py
```

## Package Usage

The repository is structured as a Python package with source modules under `src/`.

Example import:

```python
from src.data_store.retrieval import DataRetriever
from src.features.pipeline import FeaturePipeline
from src.backtest.engine import BacktestEngine

# instantiate and run your workflow
```

## Architecture

The system is organized into the following major packages:

- `src/data/` and `src/data_store/` – market data retrieval, caching, inventory, and persistence
- `src/features/` – feature engineering pipeline and resampling logic
- `src/timesfm/` – TimesFM forecast wrapper and forecast interface
- `src/backtest/` – candidate generation and backtesting engine
- `src/fta/` – structural FTA validation
- `src/meta_model/` – confidence scoring and model features
- `src/portfolio/` – portfolio execution, risk rules, position management
- `src/loop/` – live trading loop orchestration
- `src/reports/` – diagnostics, analysis, and reporting
- `src/adaptive/` – adaptive thresholds and live tuning

## Testing

Run tests with pytest:

```bash
pytest
```

The test suite is organized by phase under `tests/`.

## Contributing

Contributions are welcome. Please open issues or pull requests for:

- bug fixes
- feature improvements
- additional validation and smoke tests
- better documentation and architecture diagrams

## Citation

If you use this framework in research, please cite the project and acknowledge that it is a research system.

---

`trading-system` is intended as a research prototype rather than a production trading platform.
