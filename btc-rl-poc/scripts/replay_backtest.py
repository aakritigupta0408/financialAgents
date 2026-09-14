"""P2-P10 — EXACT-BRTI HISTORICAL REPLAY of the T0-T3 trader family.

Real timestamped replay: at each decision row we use ONLY decision-time state
(current_brti at-or-before ts, official_target known at open, executable Kalshi
quote) -> Oracle(t) -> policy -> settle on official exact_yes -> paper P&L. This is
NOT final-label simulation.

Chronological split by window (P4): DEV 50% / VAL 25% / HOLDOUT 25%. The T1
disagreement threshold is frozen on DEV+VAL, THEN the holdout is opened once (P4/P6).
Lookahead audit (P3): every row must have brti_sample_ts <= decision_time.

Writes research/replay/replay_result.json.
"""
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from btc_rl import traders_v2 as T          # noqa: E402
from btc_rl import economics as E           # noqa: E402

RES = ROOT / "results"
DEC = RES / "brti_decision_dataset.jsonl"
T05 = ROOT / "research" / "t05_repricing" / "dataset.jsonl"
OUT = ROOT / "research" / "replay" / "replay_result.json"


def _Phi(x):
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2)))


def _p_mech(r, sigma):
    """MECH_FAIR_BRTI at a decision row for a given sigma. Settlement-aware override
    is DELIBERATELY OMITTED here: it uses required_remaining_average (a function of
    the realized settlement-sample count), which is not strictly PIT inside the
    final 60s. Backtest decisions use only the diffusion fair value from decision-
    time BRTI distance -> no settlement-window peeking (P3)."""
    lvl, tte = r["current_brti"], max(1.0, r["time_remaining_s"])
    return min(1 - 1e-4, max(1e-4, _Phi(r["brti_distance_to_target"] / (sigma * lvl * math.sqrt(tte)))))


def fit_oracle(rows):
    """Fit MECH sigma + isotonic recalibration on the GIVEN rows only (dev+val), so
    the holdout stays out-of-sample (P4). Returns (sigma, iso)."""
    y = np.array([r["exact_yes"] for r in rows], float)
    best_sig, best_b = 6e-5, 9.0
    for sig in [4e-5, 6e-5, 8e-5, 1.1e-4, 1.5e-4, 2.0e-4]:
        p = np.array([_p_mech(r, sig) for r in rows])
        b = float(np.mean((p - y) ** 2))
        if b < best_b:
            best_b, best_sig = b, sig
    from sklearn.isotonic import IsotonicRegression
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1)
    iso.fit([_p_mech(r, best_sig) for r in rows], y)
    return best_sig, iso


def attach_oracle(rows, sigma, iso):
    pm = [_p_mech(r, sigma) for r in rows]
    po = np.clip(iso.predict(pm), 1e-4, 1 - 1e-4)
    for r, p in zip(rows, po):
        r["p_oracle"] = float(p)


def load():
    spread = {}
    for l in T05.open():
        if l.strip():
            r = json.loads(l)
            spread[(r["ticker"], r["ts"])] = r.get("k_spread", 0.01)
    rows, leak = [], 0
    for l in DEC.open():
        if not l.strip():
            continue
        d = json.loads(l)
        if d.get("current_brti") is None or d.get("k_prob") is None or d["time_remaining_s"] < 1:
            continue
        if d.get("brti_sample_ts") is not None and d["brti_sample_ts"] > d["decision_time"] + 1:
            leak += 1
            continue
        d["half_spread"] = spread.get((d["market_window_id"], d["decision_time"]), 0.01) / 2.0
        rows.append(d)
    return rows, leak


def split(rows):
    first = {}
    for r in rows:
        first[r["market_window_id"]] = min(first.get(r["market_window_id"], 1e18), r["decision_time"])
    order = sorted(first, key=lambda t: first[t]); n = len(order)
    dev = set(order[:int(.5 * n)])
    val = set(order[int(.5 * n):int(.75 * n)])
    hold = set(order[int(.75 * n):])
    return dev, val, hold


def by_window(rows, wins):
    w = {}
    for r in rows:
        if r["market_window_id"] in wins:
            w.setdefault(r["market_window_id"], []).append(r)
    for k in w:
        w[k].sort(key=lambda r: r["decision_time"])
    return w


def eligible_rows(win_rows):
    out = []
    for r in win_rows:
        side, cost, ev = T.edge_ev(r["p_oracle"], r["k_prob"], r["half_spread"])
        if T.eligible(r["time_remaining_s"], ev):
            out.append((r, side, cost, ev))
    return out


def run_trader(wins_rows, policy, tau=T.T1_EDGE_TAU):
    """policy in {T0,T1,T2,T3}. Returns list of per-window dicts (traded or not)."""
    results = []
    for wid, wr in wins_rows.items():
        elig = eligible_rows(wr)
        traded = None
        in_envelope = any(r["time_remaining_s"] <= T.LOCK_MAX_S for r in wr)
        if elig:
            if policy == "T1":
                q = [(r, s, c, ev) for (r, s, c, ev) in elig if T.t1_qualifies(r["p_oracle"], r["k_prob"], tau)]
                pick = q[0] if q else None
            elif policy == "T2":
                # OPTIMISTIC UPPER BOUND: hindsight lowest-cost entry on T0's side.
                # A realizable T0.5 must PREDICT the dip, not see it — so this
                # overstates achievable timing benefit and is SHADOW_ONLY.
                t0 = elig[0]
                same = [(r, s, c, ev) for (r, s, c, ev) in elig if s == t0[1]]
                pick = min(same, key=lambda x: x[2]) if same else None
            else:  # T0, T3
                pick = elig[0]
            if pick:
                r, side, cost, ev = pick
                stake = T.t3_stake(ev) if policy == "T3" else T.FIXED_STAKE
                pnl, win = T.settle_pnl(side, cost, stake, r["exact_yes"])
                traded = {"ticker": wid, "close_ts": r["decision_time"], "side": side,
                          "cost": cost, "stake": stake, "ev": ev, "pnl_c": pnl, "win": win,
                          "tte_s": r["time_remaining_s"], "exact_yes": r["exact_yes"]}
        results.append({"wid": wid, "eligible": in_envelope, "traded": traded})
    return results


def econ(results):
    elig = [x for x in results if x["eligible"]]
    trades = [x["traded"] for x in elig if x["traded"]]
    pnls = [t["pnl_c"] for t in trades]
    n_elig = len(elig)
    eq = E.equity_curve(0.0, pnls)
    acc = (sum(1 for t in trades if t["win"]) / len(trades)) if trades else None
    med_tte = None
    if trades:
        tts = sorted(t["tte_s"] / 60.0 for t in trades)
        med_tte = round(tts[len(tts) // 2], 1)
    return {
        "eligible_windows": n_elig, "trades": len(trades),
        "coverage": E.coverage(len(trades), n_elig),
        "accuracy": round(acc, 4) if acc is not None else None,
        "median_lock_time_min": med_tte,
        "ev_per_eligible_c": E.ev_per_eligible_window(pnls, n_elig),
        "ev_per_trade_c": E.ev_per_trade(pnls),
        "total_pnl_c": E.realized_pnl(pnls),
        "max_drawdown_c": E.drawdown(eq),
        "bad_entry_rate": E.bad_entry_rate(pnls),
    }


def main():
    rows, leak = load()
    dev, val, hold = split(rows)
    devval = dev | val
    # P4/P3: fit the Oracle (sigma + recal) on DEV+VAL ONLY, then apply everywhere,
    # so the holdout Oracle is genuinely out-of-sample.
    devval_rows = [r for r in rows if r["market_window_id"] in devval]
    sigma, iso = fit_oracle(devval_rows)
    attach_oracle(rows, sigma, iso)
    wr_dev = by_window(rows, dev); wr_val = by_window(rows, val)
    wr_devval = by_window(rows, devval); wr_hold = by_window(rows, hold)

    # ── P6 T1 disagreement surface on DEV+VAL only; freeze best tau ──
    surface = []
    best = None
    for tau in [0.02, 0.05, 0.08, 0.10, 0.12, 0.15]:
        e = econ(run_trader(wr_devval, "T1", tau))
        rec = {"tau": tau, **{k: e[k] for k in
               ("coverage", "trades", "accuracy", "ev_per_eligible_c",
                "ev_per_trade_c", "total_pnl_c", "max_drawdown_c", "bad_entry_rate")}}
        surface.append(rec)
        # freeze by EV/eligible (primary), requiring a minimum coverage for robustness
        if e["ev_per_eligible_c"] is not None and (e["coverage"] or 0) >= 0.05:
            if best is None or e["ev_per_eligible_c"] > best["ev_per_eligible_c"]:
                best = rec
    frozen_tau = best["tau"] if best else T.T1_EDGE_TAU

    # ── P5/P7/P8 development-side trader economics (context, not the verdict) ──
    dev_econ = {p: econ(run_trader(wr_devval, p, frozen_tau)) for p in ("T0", "T1", "T2", "T3")}

    # ── P7 T2 paired execution delta on DEV+VAL (same eligible opportunities) ──
    t0_res = run_trader(wr_devval, "T0"); t2_res = run_trader(wr_devval, "T2")
    t0_by = {x["wid"]: x for x in t0_res}; t2_by = {x["wid"]: x for x in t2_res}
    shared = [w for w in t0_by if t0_by[w]["traded"] and t2_by[w]["traded"]]
    t0_p = [t0_by[w]["traded"]["pnl_c"] for w in shared]
    t2_p = [t2_by[w]["traded"]["pnl_c"] for w in shared]
    t2_delta = E.paired_delta(t0_p, t2_p)

    # ── P9 FINAL UNTOUCHED HOLDOUT with FROZEN policies ──
    holdout = {p: econ(run_trader(wr_hold, p, frozen_tau)) for p in ("T0", "T1", "T2", "T3")}

    # ── P10 verdicts (from holdout, primary = EV/eligible) ──
    def verdict(name):
        h = holdout[name]
        ev = h["ev_per_eligible_c"]
        if h["trades"] < 20 or ev is None:
            return "INSUFFICIENT_EVIDENCE"
        if name == "T2":
            return "SHADOW_ONLY"   # hindsight-best-entry upper bound; needs real T0.5
        if name == "T3":
            return "SHADOW_ONLY"   # sizing research; judged on drawdown, not raw EV
        return "QUALIFIED_FOR_PROSPECTIVE" if ev > 0 else "REJECTED"
    verdicts = {p: verdict(p) for p in ("T0", "T1", "T2", "T3")}

    # earliness of T0 holdout entries — is any edge just near-expiry resolution?
    t0_hold = run_trader(wr_hold, "T0")
    ent = [x["traded"]["tte_s"] / 60.0 for x in t0_hold if x["traded"]]
    earliness = {"median_entry_min": round(sorted(ent)[len(ent) // 2], 1) if ent else None,
                 "frac_entries_under_2min": round(sum(1 for t in ent if t < 2) / len(ent), 3) if ent else None,
                 "frac_entries_over_9min": round(sum(1 for t in ent if t > 9) / len(ent), 3) if ent else None}

    doc = {
        "schema_version": "replay-1",
        "dataset": {"rows": len(rows), "windows": len(dev | val | hold),
                    "dev_windows": len(dev), "val_windows": len(val),
                    "holdout_windows": len(hold), "lookahead_rows_dropped": leak},
        "oracle_fit": {"scope": "DEV+VAL only (holdout out-of-sample)", "sigma": sigma},
        "t0_holdout_entry_earliness": earliness,
        "frozen_policy": T.FROZEN, "frozen_t1_tau_from_devval": frozen_tau,
        "t1_disagreement_surface_devval": surface,
        "development_econ_devval": dev_econ,
        "t2_paired_execution_devval": {"paired_windows": len(shared), "delta": t2_delta},
        "final_holdout": holdout,
        "offline_verdicts": verdicts,
        "caveats": [
            "Oracle fit on DEV+VAL only; holdout=93 windows OOS (still modest n).",
            "T2 is a HINDSIGHT lowest-cost-entry upper bound, NOT a realizable timing "
            "policy — a real T0.5 must predict the dip; T2 overstates timing benefit.",
            "T3 sizing shows similar EV to T0 with materially higher drawdown -> not "
            "preferred; sizing is judged on drawdown/tail, not raw EV.",
            "Executable price = Kalshi mid + half-spread; assumes k_prob is "
            "contemporaneous with the decision row.",
            "OFFLINE historical replay — decides what enters the arena, NOT proof; "
            "prospective exact-BRTI evidence decides what is real."],
        "settlement_provenance": "OFFICIAL_EXACT_BRTI (contract_outcomes.jsonl)",
        "oracle": "recalibrated MECH_FAIR_BRTI (oracle_frozen) — no Kalshi input",
        "note": "Real timestamped replay (decision-time state only). T1 tau frozen on "
                "DEV+VAL before the holdout was opened. Primary metric = EV/eligible "
                "window; win rate is not a decision metric.",
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(doc, indent=1))
    print(f"REPLAY — {len(rows)} rows, {doc['dataset']['windows']} windows "
          f"(dev {len(dev)}/val {len(val)}/holdout {len(hold)}), leak dropped {leak}")
    print(f"frozen T1 tau (dev+val) = {frozen_tau}")
    print("FINAL HOLDOUT (EV/eligible c · EV/trade c · cov · acc · P&L c · maxDD c):")
    for p in ("T0", "T1", "T2", "T3"):
        h = holdout[p]
        print(f"  {p}: ev/elig {h['ev_per_eligible_c']}  ev/trade {h['ev_per_trade_c']}  "
              f"cov {h['coverage']}  acc {h['accuracy']}  pnl {h['total_pnl_c']}  "
              f"dd {h['max_drawdown_c']}  -> {verdicts[p]}")
    if t2_delta:
        print(f"T2 paired execution delta (dev+val): {t2_delta['paired_delta']}c CI {t2_delta['ci95']}")


if __name__ == "__main__":
    main()
