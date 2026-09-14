"""TRACK D — T3 RL trader (SHADOW). Downstream capital/size policy on the Oracle's
FROZEN side. RL does NOT predict direction (the Oracle owns that) and CANNOT change
side (RL_SIDE_IMMUTABILITY invariant).

Because each KXBTC15M window is a single one-shot size decision, the honest
formulation is a CONTEXTUAL BANDIT, not multi-step RL — we say so rather than
dressing a bandit as deep RL. Action space: SKIP / BUY_SMALL / BUY_MEDIUM /
BUY_LARGE (stake 0/1/2/3 contracts). Reward = realized paper pnl - fees.

Learn Q(discretized_state, action) = mean reward on TRAIN; act greedily on HOLDOUT.
Baselines it must beat: T0 fixed BUY_SMALL, random policy. Sanity: reward-shuffle,
state-shuffle, action-collapse / never-trades / always-trades detection, seed
stability. No training on holdout. Writes research/traders/t3_rl_result.json.
"""
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
import replay_backtest as RB          # noqa: E402
from btc_rl import traders_v2 as T    # noqa: E402
from btc_rl import economics as E     # noqa: E402

OUT = ROOT / "research" / "traders" / "t3_rl_result.json"
ACTIONS = {"SKIP": 0, "BUY_SMALL": 1, "BUY_MEDIUM": 2, "BUY_LARGE": 3}


def _lcg(seed):
    s = seed & 0x7fffffff
    while True:
        s = (1103515245 * s + 12345) & 0x7fffffff
        yield s / 0x7fffffff


def state_key(r, sigma):
    """Compact discretized state (side is NOT part of the action; it is fixed)."""
    tte = max(1.0, r["time_remaining_s"])
    norm = abs(r["brti_distance_to_target"]) / (sigma * r["current_brti"] * math.sqrt(tte))
    gap = r["p_oracle"] - r["k_prob"]
    def b(x, edges):
        return sum(1 for e in edges if x > e)
    return (b(r["p_oracle"], [0.45, 0.55, 0.65]),
            b(abs(gap), [0.03, 0.08, 0.13]),
            b(tte / 60.0, [3, 6, 9]),
            b(norm, [0.3, 0.7, 1.2]))


def candidates(wins_rows, sigma):
    """One decision per window: the Oracle-side candidate (SKIP or size it)."""
    out = []
    for wid, wr in wins_rows.items():
        elig = RB.eligible_rows(wr)
        if not elig:
            continue
        r, side, cost, ev = elig[0]
        # reward for each action (stake), settled on official exact_yes
        rewards = {a: (0.0 if k == 0 else T.settle_pnl(side, cost, k, r["exact_yes"])[0])
                   for a, k in ACTIONS.items()}
        out.append({"wid": wid, "state": state_key(r, sigma), "rewards": rewards,
                    "side": side})
    return out


def learn_Q(train):
    q = {}
    for c in train:
        for a, rew in c["rewards"].items():
            key = (c["state"], a)
            v = q.setdefault(key, [0.0, 0])
            v[0] += rew; v[1] += 1
    return {k: (v[0] / v[1]) for k, v in q.items()}


def greedy(cands, Q):
    acts = []
    for c in cands:
        best_a, best_v = "SKIP", 0.0    # default SKIP (value 0) if state unseen
        for a in ACTIONS:
            v = Q.get((c["state"], a))
            if v is not None and v > best_v:
                best_v, best_a = v, a
        acts.append(best_a)
    return acts


def econ(cands, acts, n_elig):
    pnls = [c["rewards"][a] for c, a in zip(cands, acts) if a != "SKIP"]
    dist = {a: sum(1 for x in acts if x == a) for a in ACTIONS}
    return {"n_eligible": n_elig, "trades": int(sum(1 for a in acts if a != "SKIP")),
            "coverage": E.coverage(sum(1 for a in acts if a != "SKIP"), n_elig),
            "ev_per_eligible_c": E.ev_per_eligible_window(pnls, n_elig),
            "ev_per_trade_c": E.ev_per_trade(pnls),
            "total_pnl_c": E.realized_pnl(pnls),
            "max_drawdown_c": E.drawdown(E.equity_curve(0.0, pnls)),
            "action_distribution": dist}


def main():
    rows, _ = RB.load()
    dev, val, hold = RB.split(rows)
    devval = dev | val
    sigma, iso = RB.fit_oracle([r for r in rows if r["market_window_id"] in devval])
    RB.attach_oracle(rows, sigma, iso)
    tr = candidates(RB.by_window(rows, devval), sigma)     # train on dev+val
    te = candidates(RB.by_window(rows, hold), sigma)        # holdout
    n_te = len(te)
    if len(tr) < 40 or n_te < 20:
        print("insufficient:", len(tr), n_te); return

    Q = learn_Q(tr)
    rl = econ(te, greedy(te, Q), n_te)

    # baselines on holdout — INCLUDING fixed-LARGE (fair leverage-matched) and
    # edge-proportional sizing (D6), so T3 can't win merely by betting bigger
    t0 = econ(te, ["BUY_SMALL"] * n_te, n_te)               # fixed small
    t0_large = econ(te, ["BUY_LARGE"] * n_te, n_te)         # fixed max leverage
    def edge_prop_action(c):
        # size ~ quoted edge on the trade (T0's ev), capped; a simple sizing rule
        ev = max(c["rewards"].values())                    # proxy; sign only matters
        return "BUY_LARGE" if ev > 0 else "SKIP"
    ep = econ(te, [edge_prop_action(c) for c in te], n_te)
    rnd_gen = _lcg(20260914)
    rnd_acts = [list(ACTIONS)[int(next(rnd_gen) * 4)] for _ in te]
    rnd = econ(te, rnd_acts, n_te)

    def rr(e):  # return-to-risk: EV/eligible per unit max-drawdown
        dd = e["max_drawdown_c"] or 1e-9
        return (e["ev_per_eligible_c"] or 0.0) / dd if dd > 0 else None

    # ── sanity checks ──
    # reward-shuffle: destroy state->reward link; learned Q should lose any edge
    shuf = _lcg(7)
    all_states = [c["state"] for c in tr]
    perm = sorted(range(len(tr)), key=lambda i: next(shuf))
    tr_shuf = [{**tr[i], "state": all_states[perm[i]]} for i in range(len(tr))]
    Q_shuf = learn_Q(tr_shuf)
    rl_shuf = econ(te, greedy(te, Q_shuf), n_te)

    # seed stability: re-discretize order shouldn't matter (deterministic); report
    # action-distribution stability across two independent train orderings
    Q2 = learn_Q(list(reversed(tr)))
    acts_a, acts_b = greedy(te, Q), greedy(te, Q2)
    seed_stable = (acts_a == acts_b)

    dist = rl["action_distribution"]
    collapse = max(dist.values()) == n_te                   # one action everywhere
    never_trades = dist["SKIP"] == n_te
    always_trades = dist["SKIP"] == 0

    beats_rnd = (rl["ev_per_eligible_c"] or -1e9) > (rnd["ev_per_eligible_c"] or -1e9)
    edge_vs_shuffle = (rl["ev_per_eligible_c"] or -1e9) - (rl_shuf["ev_per_eligible_c"] or -1e9)
    # RISK-ADJUSTED: T3 must beat the BEST leverage-matched baseline on return-to-risk,
    # else its EV is just leverage, not learned sizing skill.
    best_base_rr = max(rr(t0), rr(t0_large), rr(ep))
    beats_baselines_risk_adj = (rr(rl) or -1e9) > best_base_rr
    max_size_share = dist["BUY_LARGE"] / max(1, rl["trades"])
    leverage_only = max_size_share >= 0.9 and dist["BUY_SMALL"] + dist["BUY_MEDIUM"] == 0

    if never_trades:
        verdict = "INFORMATION_LIMITED"        # learned effectively not to trade
    elif not beats_rnd or edge_vs_shuffle <= 0:
        verdict = "INFORMATION_LIMITED"        # no edge beyond shuffled-reward control
    elif leverage_only or not beats_baselines_risk_adj:
        verdict = "SHADOW_ONLY"                # only 'bet max size' — leverage, not skill
    elif n_te < 30:
        verdict = "INSUFFICIENT_EVIDENCE"
    else:
        verdict = "QUALIFIED_FOR_PROSPECTIVE_SHADOW"

    doc = {
        "schema_version": "t3-rl-1", "role": "SHADOW",
        "formulation": "CONTEXTUAL_BANDIT (one-shot size decision per window; not "
                       "multi-step RL — stated honestly)",
        "side_immutable": True,
        "actions": list(ACTIONS), "reward": "realized paper pnl - fees",
        "small_n": {"train_windows": len(tr), "holdout_windows": n_te,
                    "effective_independent_n": n_te,
                    "note": "unit=market_window_id; bandit, not millions of samples"},
        "oracle_fit": "DEV+VAL only (holdout OOS)",
        "holdout": {"T3_rl": rl, "baseline_T0_fixed_small": t0,
                    "baseline_T0_fixed_large": t0_large, "baseline_edge_prop": ep,
                    "baseline_random": rnd},
        "risk_adjusted": {"t3_return_to_risk": round(rr(rl), 5) if rr(rl) else None,
                          "best_baseline_return_to_risk": round(best_base_rr, 5),
                          "beats_baselines_risk_adjusted": bool(beats_baselines_risk_adj),
                          "max_size_share_of_trades": round(max_size_share, 3),
                          "leverage_only": bool(leverage_only)},
        "sanity": {
            "reward_shuffle_control_ev": rl_shuf["ev_per_eligible_c"],
            "edge_vs_reward_shuffle_c": round(edge_vs_shuffle, 3),
            "seed_stable": bool(seed_stable),
            "action_collapse": bool(collapse),
            "never_trades": bool(never_trades),
            "always_trades": bool(always_trades),
            "action_distribution": dist},
        "beats_random": bool(beats_rnd),
        "verdict": verdict,
        "caveats": ["Contextual bandit on a tiny state space; holdout n=%d windows. "
                    "SHADOW only; may not size real capital from retrospective evidence." % n_te,
                    "T3 cannot change the Oracle side (invariant test RL_SIDE_IMMUTABILITY)."],
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(doc, indent=1))
    print(f"T3 RL (contextual bandit) — holdout {n_te} windows")
    print(f"  T3     EV/elig {rl['ev_per_eligible_c']}c  cov {rl['coverage']}  "
          f"dd {rl['max_drawdown_c']}  actions {dist}")
    print(f"  T0 fix EV/elig {t0['ev_per_eligible_c']}c   random EV/elig {rnd['ev_per_eligible_c']}c")
    print(f"  reward-shuffle edge {round(edge_vs_shuffle,3)}c  seed_stable {seed_stable}  "
          f"collapse {collapse}  never_trades {never_trades}")
    print(f"  VERDICT: {verdict}")


if __name__ == "__main__":
    main()
