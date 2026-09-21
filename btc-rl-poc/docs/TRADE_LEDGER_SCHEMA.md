# Trade-ledger schema (`results/<arm>_trades.jsonl`)

One JSON row per (arm, KXBTC15M window). PAPER/SIMULATION. Truth layer is
`results/contract_outcomes.jsonl` (`exact_yes`). Verified 2026-09-21: settlement is
sound (0 `win` mismatches, 0 `actual`-vs-official disagreements, 100% pnl-consistent).

## Core fields — present on EVERY arm (the honest cross-arm join key set)
The A/B evaluator, home snapshot, and coverage slicer use ONLY these, so schema drift
below does not corrupt the headline numbers.

| field | meaning |
|---|---|
| `ticker` | KXBTC15M contract id |
| `close_ts` | window close epoch — **the statistical unit / join key** |
| `made_ts` | decision epoch |
| `side` | `"yes"` / `"no"` taken |
| `ask_c` | price paid, cents |
| `fee_c` | logged fee annotation (per-contract raw float — see note) |
| `contracts` | order size |
| `stake_c` | upfront cost = `contracts*ask + kalshi_order_fee_c(contracts, ask)` (TRUE per-order fee) |
| `actual` | official outcome: 1 = YES happened, 0 = NO (None = unsettled) |
| `win` | `(actual==1) == (side=="yes")` |
| `pnl_c` | net cents = win ? `contracts*100 - stake` : `-stake` (already fee-correct via stake) |
| `bankroll_c` | **the daemon's paper balance after this row** — the ONE source of truth for current cash (never re-derive from a hardcoded seed) |
| `strike`, `mins_left`, `contract_truth_quality` | contract context |

## Arm-specific fields (the drift — normalize before cross-arm analysis)

| field | arms that log it | note |
|---|---|---|
| `p_arm` | pt, tv, cg33 | the arm's decision probability (follower/gate family) |
| `p_up` | ob, fm | the arm's decision probability (model family) |
| `leader`, `rec10` | pt, tv, cg33 | which kb leader was followed + its rec10 |
| `gate` | cg33, fm | gate label |
| `edge`, `kelly_frac` | tv (edge+kelly), fm (kelly) | value-gate / sizing |
| `conf_z`, `model` | ob, fm | confidence z-score + model id |
| `q80_hi/lo/w` | fm | 80% quantile band (QR model) |
| `depth_cap_c` | pt only | depth ceiling |
| `late_settle_ts` | pt only | late-settlement marker |

**Decision-probability accessor:** `p_arm` and `p_up` are the SAME concept under two
names. Any cross-arm probability analysis must read
`btc_rl.economics.arm_p(row)` (returns `p_arm` or `p_up`, whichever exists) rather than
one field — reading only `p_arm` silently drops ob/fm; only `p_up` drops pt/tv/cg33.

## Fee note (do not "correct" the P&L — see AUDIT_COMPLIANCE 2026-09-21)
`pnl_c` is already net of Kalshi's real fee, which is charged **per ORDER**:
`kalshi_order_fee_c(C, price) = ceil(7*C*p*(1-p))`. The `fee_c` FIELD logs the raw
per-contract float and is only an annotation. Do NOT recompute an order's fee as
`C * kalshi_fee_c(price)` — that rounds up C times and overcharges (100 lots @80c:
true 112c vs 200c). `metrics.kalshi_fee_c` is the C=1 case only.
