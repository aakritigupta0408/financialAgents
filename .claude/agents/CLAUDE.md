# Project: Local Paper-Trading Research System

This repository builds a local, personal-use, paper-trading research system.

## Core Objective
Build a modular system that:
- pulls market data through MCP
- creates structured market features
- uses TimesFM as a forecasting feature generator
- uses Financial Technical Analysis (FTA) as a strict trade filter
- uses a learned meta-model to predict whether a setup should be traded
- simulates paper trading with configurable starting capital
- tracks performance and improves over time

## Non-Negotiable Rules
- No hallucinated APIs, libraries, or endpoints
- No broker execution
- Paper trading only
- Use real packages only
- Keep everything runnable locally on a normal laptop
- Prefer correctness over complexity
- If uncertain, leave a TODO with the exact blocker instead of guessing

## Decision Hierarchy
1. Market data provider
2. Feature engineering
3. TimesFM forecast
4. FTA validation
5. Meta-model scoring
6. Portfolio/risk checks
7. Paper execution

A trade is allowed only if:
- TimesFM produces a valid forecast
- FTA accepts the setup
- Meta-model confidence passes threshold
- Portfolio/risk rules allow the trade

## TimesFM Rules
- TimesFM is a forecasting feature generator only
- Do not treat TimesFM as the final decision-maker
- Do not finetune TimesFM
- Retrain only meta-model/calibration layers

## FTA Rules
FTA is the hard filter.
FTA must reject trades with:
- expected move that does not clear First Trouble Area
- poor reward:risk
- weak structure
- poor liquidity
- unsuitable volatility context

Do not rely only on RSI/MACD.
Use real market structure and price action logic.

## Learning Rules
- Train on historical data first
- During market hours, update only safe runtime controls such as confidence throttles and exposure limits
- Do not retrain model weights on every tick
- After market close, label outcomes and retrain meta-model/calibration layers

## Build Order
1. Repo setup and contracts
2. Data layer
3. Feature engineering
4. FTA engine
5. TimesFM wrapper
6. Portfolio/risk engine
7. Backtest loop
8. Meta-model
9. Live paper-trading loop
10. Reports and QA

## Deliverable Standard
For every phase:
- explain briefly what is being built
- show file structure changes
- write full code
- show how to run or test it
- validate outputs before moving on
