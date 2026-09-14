# System boundary — two disconnected systems on one shared feed

The live trading desk and the sealed-research program are **completely disconnected**:
a change in one cannot affect the other. This is enforced by
`tests/test_system_isolation.py` (fails CI on any new coupling).

```
                shared capture substrate  (infrastructure, not a "system")
                btc_rl.online — freezes each KXBTC15M window at T0,
                records exact-BRTI settlement at T1. Writes:
                  results/online_status.json      (heartbeat)
                  results/kalshi_binary_log.jsonl (current/settled windows)
                          │  read-only, by both consumers
             ┌────────────┴─────────────┐
             ▼                            ▼
   LIVE TRADING DESK              SEALED-RESEARCH PROGRAM
   (existing, keep running)       (new, independent)
   btc_rl.online Follower/Oracle  TRUE15M Oracle: offline model/feature
   arms; paper P&L + bankroll.    program, sealed TEST_V2, firewall.
   Private files:                 Files:
     results/pt*_trades.jsonl       research/true15m/*
     results/live_desk.json         results/research_*.json
     results/home_snapshot.json     research/contract_specs/*
     results/*_logit.json         Code:
     results/q_table_online*        scripts/true15m_*, loss_calibration_sweeps,
   Code:                            model_failure_diagnostics, distributional_models,
     btc_rl/online.py               sealed_test_governance, test_v2_*,
     scripts/emit_live_desk.py      coarse_/fine_feature_factory, freeze_feature_universe,
     scripts/emit_home_snapshot.py  research_engine, emit_research_snapshot
     scripts/emit_traders_snapshot   btc_rl/test_v2_firewall.py, research_events.py
```

## Rules (enforced)

1. **Research → live: zero.** No research script reads or writes any live-desk private
   file, and no research script imports `btc_rl.online`. The live paper desk therefore
   cannot be broken, biased, or paused by any research change. (This is the direction
   that matters most — the desk is the thing that must keep running.)
2. **Live → research: read-only heartbeat only.** The research narrator
   (`emit_research_snapshot.py`) may *read* the shared capture heartbeat
   (`online_status.json`, `kalshi_binary_log.jsonl`) to show whether the market feed is
   alive. It reads nothing else from the live system and writes nothing back.
3. **No cross-imports** between the two code sets.
4. **Publishing** (`publish_dashboard.py`) renders both onto one site but is a
   presentation step only; it does not let one system's state mutate the other's.

## The shared substrate is deliberate

Both systems need the same market data, so both read the same capture feed. That is
infrastructure shared by design, not a coupling between the systems — neither system's
*models, training, decisions, or P&L* are visible to the other. `btc_rl.online` happens
to be both the capture collector and the live Follower trader; research treats only its
heartbeat/window files as the feed and never its trading state.

## The sealed-research firewall (separate, stronger guard)

Inside the research system, `btc_rl/test_v2_firewall.py` additionally forbids any
development/selection step from touching sealed TEST_V2 windows. That protects the
blind test; this file protects the two *systems* from each other.
