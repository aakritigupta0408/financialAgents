# BTC Integer-Price Prediction as Reinforcement Learning — POC

Predict the price of Bitcoin at **7:00 PM and 7:15 PM Pacific**, scored at
**integer level** (68000 and 68000.98 count as the same answer). Reward +1 on
an exact integer match, −1 otherwise. No Alpaca — everything runs on open,
no-auth data streams.

## RL formulation (kept deliberately simple)

- **Episode**: one step (a contextual bandit — the simplest honest RL setting,
  since today's prediction doesn't change tomorrow's market).
- **State**: discretized features at decision time (6:45 PM PT):
  5m/15m return buckets × 30m volatility bucket × RSI bucket = 81 states.
- **Action**: an integer dollar delta from the current price
  (0, ±1, ±2, ±3, ±5, ±8, ±13, ±21, ±34, ±55, ±89 — 21 actions).
- **Reward**: `+1` if `int(pred) == int(actual)` else `−1` (spec), plus a
  *shaped* variant (`−|error|/100`) because the sparse reward fires ~0.5% of
  the time — too rare to learn from directly.
- **Training data**: since BTC trades 24/7, every minute of the afternoon
  window is a pseudo-episode (~200/day). We train on all of them and evaluate
  on the exact 7:00/7:15 PM slots. Chronological 80/20 split by day — no
  leakage from the future.

## Factors researched → features implemented

| Factor family (from literature) | Feature here | Source |
|---|---|---|
| Momentum / technicals | 1m/5m/15m/60m returns, RSI(14), EMA(30) distance | Coinbase 1m bars |
| Volatility regime | σ of 1m returns over 30m | Coinbase 1m bars |
| Liquidity / volume | 5m-vs-60m volume ratio | Coinbase 1m bars |
| Market sentiment | Fear & Greed index (daily) | alternative.me |
| Derivatives positioning | funding rate (live mode, logged) | OKX public |
| On-chain activity | fees / hashrate (cataloged, next iteration) | mempool.space, blockchain.info |
| Macro (DXY, gold, rates) | deliberately deferred — noisy at 15m horizon | — |

## Open data streams (verified reachable 2026-08-19)

Works: **Coinbase Exchange** (REST + WebSocket, primary), **Kraken**,
**Bitstamp**, **OKX** (funding + open interest), **Deribit** (perp/options),
**alternative.me** (Fear & Greed), **mempool.space** (fees/mempool),
**blockchain.info** (hashrate/tx), **CoinGecko** (spot).
Geo-blocked from this machine: Binance, Bybit.

## Agent ladder (simple → complex)

1. **Level 0 — persistence baseline**: predict the current price. ✅ built
2. **Level 1 — tabular Q-learning**: ε-greedy over 81 states × 21 actions,
   sparse and shaped reward variants. ✅ built
3. Level 2 — linear function approximation over continuous features.
4. Level 3 — small DQN / distributional RL (predict the delta *distribution*,
   act on its mode — the right tool for an exact-integer objective).

## Results (120 days: 2026-04-21 → 2026-08-18, ~40k train / 10k test episodes)

15-minute deltas have σ ≈ $91 (30-min: $129), so an exact-integer hit is a
~1-in-230 event even for a perfect-mean predictor. Test set, all episodes:

| Agent | Horizon | Exact-int hit | MAE | within $10 |
|---|---|---|---|---|
| persistence | 15m | 0.75% | $60 | 16.8% |
| tabular-q sparse | 15m | 0.37% | $92 | 7.3% |
| tabular-q shaped | 15m | 0.75% | $61 | 15.4% |
| persistence | 30m | 0.48% | $86 | 11.3% |
| tabular-q shaped | 30m | 0.44% | $87 | 11.1% |

**Takeaways**: the sparse ±1 reward alone is uninformative (the Q agent
under it does *worse* than doing nothing); reward shaping recovers baseline
performance by learning delta≈0 almost everywhere. Beating persistence
requires richer state (order book, funding, on-chain) and function
approximation — exactly the next rungs of the ladder.

## Run it

```bash
pip install -r requirements.txt
python -m btc_rl.train --days 120 --epochs 30   # fetch, train, eval → results/metrics.json
python -m btc_rl.live                            # predict tonight's 7:00 / 7:15 PM PT
python -m btc_rl.live --score                    # score past predictions
```
