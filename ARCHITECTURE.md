# Trading System Architecture

## Overview

**Paper-trading research framework** for backtesting and live-trading intraday stock strategies with multi-stage decision filters and risk management.

**Key Characteristics**:
- ✓ No-lookahead guarantee (temporal integrity)
- ✓ Deterministic decision logic
- ✓ Immutable audit trail
- ✓ Pluggable filters (toggle FTA, MetaModel independently)
- ✓ Phase-based validation

---

## 1. SYSTEM LEVEL: Trading System Context

**Users**: Researcher, Live Trader, Analyst
**Data Sources**: Alpha Vantage API, Local Filesystem (Parquet)
**Outputs**: Equity curves, Trade journals, Risk alerts, Reports

---

## 2. MODULE LEVEL: 5 Core Layers

### Layer 1️⃣: DATA LAYER
Fetch market data from Alpha Vantage API → Cache locally → Store in Parquet.
- **DataProvider**: Abstract interface
- **AlphaVantage Adapter**: REST client
- **Cache**: HTTP deduplication
- **DataStore**: Persistent (Parquet)

### Layer 2️⃣: FEATURE LAYER  
Extract 4 parallel signals: Structure, Levels, Volatility, Liquidity.
- **Feature Pipeline**: Orchestrator
- **Structure Engine**: Trend, HH/LL
- **Levels Engine**: Support/Resistance
- **Volatility Engine**: ATR, regime
- **Liquidity Engine**: Volume

### Layer 3️⃣: PREDICTION LAYER
Forecast direction with confidence using TimesFM (Google) model.
- **TimesFM Wrapper**: ML model
- **Heuristic Fallback**: Rule-based if ML unavailable

### Layer 4️⃣: SIGNAL LAYER (Multi-Stage Filter)
Generate and validate trades through sequential filters:
1. **CandidateBuilder**: ATR-based stops/targets
2. **FTA Filter**: 8-step structural validator
3. **MetaModel**: ML confidence scorer
4. **Risk Gate**: Position size, exposure checks

### Layer 5️⃣: EXECUTION LAYER
Execute approved trades and track positions/equity.
- **TradeExecutor**: Open/close positions
- **PositionManager**: Update mark-to-market
- **EquityTracker**: Running balance
- **TradeJournal**: Immutable log

---

## 3. UNIT LEVEL: Per-Bar Decision Sequence

For **each market bar** (t = 0, 1, 2, ..., n):

```
1️⃣ DATA READ
   ├─ Load historical bars [0..t] only (NO FUTURE DATA)
   └─ Build MarketSnapshot (all timeframes)

2️⃣ FEATURE COMPUTE (Parallel)
   ├─ Structure → predict
   ├─ Levels → predict
   ├─ Volatility → predict
   └─ Liquidity → predict

3️⃣ FORECAST
   ├─ TimesFM.predict(features)
   └─ Output: direction + confidence

4️⃣ CANDIDATE
   ├─ CandidateBuilder.build(forecast)
   └─ Output: entry + stop + target (ATR-based)

5️⃣ FILTERS (Sequential Chain)
   ├─ FTA.validate(candidate) → verdict
   ├─ MetaModel.score(features, fta) → confidence
   └─ RiskGate.approve(ranked_trade) → approved | rejected

6️⃣ DECISION
   ├─ If ALL pass: EXECUTE trade
   └─ If ANY fails: SKIP (log reason)

7️⃣ RECORD STATE
   ├─ Update positions (mark-to-market)
   ├─ Update equity
   └─ Append journal
```

**Critical Invariants**:
- ✓ Temporal safety (no future bars)
- ✓ Deterministic (same input → same decision)
- ✓ All-or-nothing (pass ALL filters or skip)
- ✓ Risk-first (risk gate is mandatory)

---

## 4. Data Contracts (Pydantic Schemas)

All inter-module communication uses **immutable type-safe contracts**:

| Contract | Fields | Responsibility |
|----------|--------|-----------------|
| `OHLCVBar` | timestamp, open, high, low, close, volume, ticker, timeframe | Market data |
| `FeatureSet` | structure, levels, volatility, liquidity | Computed signals |
| `ForecastOutput` | direction, confidence | Prediction |
| `CandidateTrade` | entry, stop, target, reward_risk | Basic trade |
| `FTAOutput` | verdict, score, reasons | Structural validation |
| `RankedTrade` | candidate, confidence, verdicts | Final signal |
| `Position` | id, entry_price, stop_loss, target, qty, pnl | Live trade |

**Design Principle**: Type safety at all boundaries (Pydantic validation)

---

## 5. Operating Modes (State Machine)

```
IDLE  ──configure──→  BACKTEST  ──complete──→  VALIDATE  ──pass──→  TUNE
 ↑                                                                      │
 │                                                                      ↓
 └──pause─────────────────────────────────────────────────────────  DEPLOY
                                                                       │
                                      ┌──────────────────────────────┤
                                      │                              │
                                      │  start                       ↓
                              ┌───────┴──────────────────→  LIVE  ←──NEXT
                              │                              │  ↑
                          ANALYZE ←──────── EOD  ←────────┘  │
                              ↑                              │
                              │                           offline  
                              └──drift──────────→  TUNE    └──pause──→  IDLE
```

Each mode:
- **IDLE**: Ready for configuration
- **BACKTEST**: Run historical simulation
- **VALIDATE**: Comprehensive testing (benchmark, ablation, walk-forward)
- **TUNE**: Optimize thresholds
- **DEPLOY**: Update production config
- **LIVE**: Real-time trading
- **EOD**: End-of-day reconciliation
- **ANALYZE**: Performance analysis

---

## 6. Quality Attributes

| Attribute | How Achieved |
|-----------|--------------|
| **Accuracy** | No-lookahead guarantee, Pydantic schemas, deterministic logic |
| **Reliability** | Event sourcing, immutable logs, position reconciliation |
| **Security** | Schema validation, logic gates, risk limits enforced |
| **Performance** | Parallel features, local-first caching, batch processing |
| **Maintainability** | Layered architecture, clear separation of concerns |
| **Flexibility** | Pluggable filters, configurable thresholds, multi-timeframe |

---

## 7. Architectural Patterns

| Pattern | Where | Purpose |
|---------|-------|---------|
| Layered Architecture | 5 core layers | Separation of concerns, testability |
| Pipeline & Filters | Signal generation (FTA→MetaModel→Risk) | Flexible validation chain |
| Event Sourcing | Trade journal, decision log | Complete audit trail |
| State Machine | Operating modes (IDLE→BACKTEST→TUNE→LIVE) | Clear operational flow |
| Strategy Pattern | Forecaster, MetaModel implementations | Pluggable algorithms |
| Observer Pattern | Validation + reports + feedback loop | Decoupled optimization |

---

## 8. Code Organization

```
trading-system/
├── src/
│   ├── data/              # Data fetching (Provider, Adapter, Retry, Cache)
│   ├── data_store/        # Persistence (Store, Retrieval, Inventory)
│   ├── features/          # Feature engineering (Pipeline + 4 engines)
│   ├── timesfm/           # Forecasting (Wrapper + Fallback)
│   ├── backtest/          # Backtesting engine
│   ├── fta/               # FTA structural filter
│   ├── meta_model/        # ML confidence scorer
│   ├── portfolio/         # Portfolio engine (execution, journal)
│   ├── loop/              # Trading loop orchestration
│   ├── validation/        # Comprehensive system tests
│   ├── reports/           # Diagnostics & analysis
│   └── adaptive/          # Threshold tuning & live adaptation
├── schemas/               # Pydantic data contracts
├── config/                # Centralized settings
├── tests/                 # Phase-based unit & integration tests
├── scripts/               # Entry points (smoke tests, seed data)
└── data/                  # Data store, cache, logs
```

---

## 9. Configuration Management

**Central source: `config/settings.py`**

```python
STARTING_CAPITAL = 100_000
RISK_PER_TRADE_PCT = 0.01        # 1% risk per trade
MAX_CONCURRENT_POSITIONS = 3
MAX_DAILY_DRAWDOWN_PCT = 0.03    # 3% daily max loss
META_MODEL_MIN_CONFIDENCE = 0.60
FTA_MIN_REWARD_RISK = 2.0
SCAN_TICKERS = ["AAPL", "MSFT", "NVDA", "TSLA", "SPY", "QQQ"]
TIMEFRAMES = ["1m", "5m", "1h", "1d"]
```

All settings are **environment-overridable** for rapid A/B testing.

---

## 10. Testing Strategy

**15 Phases** of incremental validation:
- **Phases 1-3**: Data contracts, caching, features
- **Phases 4-6**: Forecasting, portfolio, backtest
- **Phases 7-9**: Meta-model, FTA, reports
- **Phases 10-15**: Adaptive, validation, real data, calibration

Each phase builds independently with test fixtures in `tests/fixtures/`.

---

## 11. Design Decisions

| Decision | Why | Trade-off |
|----------|-----|-----------|
| No-lookahead | Prevents overfitting | Slightly reduced signal |
| Layered (5 layers) | Clear separation | Layer overhead |
| Immutable logs | Full audit trail | Storage cost |
| Paper trading | Safe experimentation | No real capital |
| Pydantic schemas | Type safety | Runtime validation cost |

---

## Summary: 5-Layer Architecture

**📥 DATA LAYER**: Fetch, cache, store OHLCV data
**🔧 FEATURE LAYER**: Compute 4 parallel signals (structure, levels, volatility, liquidity)
**🎯 PREDICTION LAYER**: Forecast direction + confidence (TimesFM)
**⚡ SIGNAL LAYER**: Multi-stage filter chain (Candidate → FTA → MetaModel → Risk)
**✅ EXECUTION LAYER**: Execute trades, track positions, record journal

**Key Principles**:
- ✓ Temporal integrity (no-lookahead guarantee)
- ✓ Deterministic (reproducible decisions)
- ✓ Risk-first (mandatory risk gate)
- ✓ Pluggable (toggle filters independently)
- ✓ Auditable (immutable logs)
