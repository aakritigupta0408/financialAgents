# BTC Short-Horizon Prediction — an Online RL A/B Experiment

An always-on runner (`btc_rl/online.py`) predicts Bitcoin's price +1/+5/+15/+30
minutes ahead, committing every 5 minutes from a roster of arms that share one
action space and one reward but never share model state — so every metric gap
is attributable to the single capability an arm adds. Built on open, no-auth
data streams (Coinbase bars/book/trades, OKX funding, Deribit mark, RSS news
via CryptoBERT, Kalshi's BTC-15-min market).

## Arm roster

| Arm | Model | Isolates |
|---|---|---|
| control (`h*`) | Tabular Q-learning | discretized RL baseline |
| rp | Chart replay: copy the last h-min move forward | the time-shift illusion |
| t2 | LinUCB contextual bandit | continuous features + uncertainty-aware selection |
| t6 | t2 + live streams, order book, news sentiment, order flow | market microstructure + text |
| t7 | Linear-Q (SGD function approximation) | the L2 rung |
| t8 | Distributional DQN (predicts the delta distribution, acts on its mode) | the L3 rung |
| t9 | LSTM over the raw 1m return sequence | the L4 rung (sequence memory) |
| t10 | t2 + Kalshi market features (crowd P(up), strike gap, clock, spread) | what the prediction market adds |
| t11 | t2 + RLHF: `scripts/feedback.py up/down` votes blend ±0.15 into its reward for 30 min | the human in the loop |

Meta-arms (reporting layer, never learn):

- **consensus** — median poll of the arms per slot, with the worst trailing
  voter dropped; runs on all four horizons.
- **cal-h15** — shadows the trailing +15m MAE leader and re-centers its
  prediction with a dual-window agreement median (full-strength correction
  when a 30-row and 10-row residual median agree on sign, zero at trend turns).
- **kb** — calls the Kalshi 15-min binary (KXBTC15M) every minute:
  P(up) from t8's distribution, per-phase calibrated, Brier-scored against
  the market's own odds. A companion simulator places exactly one paper bet
  per window (entry under 85c; edge / closing-door / forced-entry strikes;
  Kalshi fee-adjusted P&L in `results/kb_bets.jsonl`).

## Reward and learning

Reward per scored prediction: **+1** if within the vol-scaled hit band
max($5, 0.1σ_h), else **−|error|/100**, plus a **0.1 direction credit** for a
correct-sign deviation. Learning runs at two speeds: an immediate update the
moment a prediction matures, and an hourly gated replay retrain over the last
24h — a 3h hold-out must not regress or the retrain is reverted. Every retrain
gate and batch run appends a git-SHA-stamped row to the append-only
`results/metrics_history.jsonl`. `scripts/watchdog.py` (cron, every 5 min)
restarts the daemon if its status heartbeat goes stale.

## The noise floor

Minute-scale BTC is approximately a martingale: persistence (predict no
change) is near-unbeatable on price *level*, so MASE ≈ 1 is the ceiling there.
Real edges live in direction accuracy, interval calibration, and
market-relative Brier — which is what the evaluation stack actually ranks.

## Evaluation

- `scripts/evaluate_all.py` — every task, every arm: MAE vs the persistence
  floor on the same slots, direction, calibration, improvement over time,
  kb accuracy/Brier vs market, meta-arm and t11-vs-t2 paired comparisons.
- `scripts/standings.py` — MAE ranking + Diebold-Mariano + paired wins,
  per horizon and era.
- `scripts/offline_gate.py` — policy gate for new feature-bandit arms
  (MAE within 2% of persistence, no near-duplicate of a simpler arm);
  t3/t4/t5 were retired by it. See `results/offline_gate.json` and
  `results/OFFLINE_METRICS.md`.

## Dashboards

Four static pages in `site/`, served by any static server:

| Page | Purpose |
|---|---|
| `live_online.html` | Live ticker, per-horizon predictions and bands, kb bets |
| `experiment_review.html` | Scoreboard: MASE/DM/PT/pinball/BSS, reliability diagram, retrain timeline |
| `index.html` | 120-day batch backtest report |
| `live_training.html` | Batch training curves + retrain gate strip |

## Run it

```bash
pip install -r requirements.txt
python3 -m btc_rl.online &               # always-on predict/learn runner
python3 -m btc_rl.ticks &                # tick-level trade archiver
# self-healing (cron, every 5 min):
#   */5 * * * * cd <repo> && python3 scripts/watchdog.py
python3 -m btc_rl.train --days 120 --live   # batch tabular-Q + live curves
python3 scripts/train_l2.py              # batch linear-Q (t7)
python3 scripts/train_l3.py              # batch distributional DQN (t8)
python3 scripts/train_l4.py              # batch LSTM (t9)
python3 scripts/evaluate_all.py          # full evaluation
python3 scripts/standings.py             # current A/B standings
python3 scripts/feedback.py up           # record an RLHF view for t11
python3 -m http.server 8787              # serve site/ dashboards
python3 -m pytest tests/                 # unit tests
```

## Repository layout

```
btc_rl/          runner (online.py), agents, features, env/reward, data
                 sources, batch trainer (train.py), tick archiver (ticks.py);
                 live.py is deprecated legacy (pre-runner demo)
scripts/         trainers (train_l2/l3/l4), evaluators (evaluate_all,
                 standings, offline_gate, debug_arms), feedback.py (RLHF),
                 watchdog.py, build_site.py
site/            the four dashboards (static HTML)
tests/           pytest suite (test_core, test_metrics, test_hf, test_kb_cal)
                 + check_* / noise_floor diagnostics (not collected)
results/         tracked: batch models (dqn_h*.pt, lstm_h*.pt, q_table.json,
                 linear_q.json), metrics.json, metrics_history.jsonl,
                 offline_gate.json, OFFLINE_METRICS.md, live_status.json.
                 Runtime state (logs, online models) is gitignored.
```
