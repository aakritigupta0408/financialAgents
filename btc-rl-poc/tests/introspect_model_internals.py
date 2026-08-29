"""Model-internals introspection: every number in docs/MODEL_INTERNALS.md
and results/model_internals.json is computed HERE from real files in
results/ — nothing estimated, nothing hand-typed.

Feature-name maps are transcribed from btc_rl/online.py:
  kb3  _kb_logit_features (24-dim)   kb4 _kb4_features (12)
  kb5  _kb5_features (14)            kb6 _kb6_features (12)
  kb8  _kb8_features (3)             pt6 _pt6_features (7)
pf = _path_features -> [frac_above-0.5, crossings/4, 3m drift z, 3m quote drift]

Oracle counterfactual convention (same as tests/pt_replay.py): logged
per-minute rows carry the market MID; a fill is modeled at mid + 2.5c
(SEL_CF_ASK_ADJ). The oracle plays the desk's own game — enter once per
window, mins_left <= 12, 5c <= ask < 80c legality band — but knows the
outcome and picks the cheapest modeled ask on the winning side.
EV is per $1 staked: win -> (100 - a - f)/(a + f), f = 7*(a/100)*(1-a/100).
"""
import json
import math
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"
SEL_CF_ASK_ADJ = 2.5          # btc_rl/online.py SEL_CF_ASK_ADJ
PT_TAU = 0.62                 # btc_rl/online.py PT_TAU

# ---------------------------------------------------------------- feature maps
KB3_FEATS = [
    "bias", "market_mid", "quote_present", "above_strike_z", "phase",
    "z_x_phase", "ofi_1m", "ofi_5m", "book_imb", "ret_5m", "ret_15m",
    "log_vol_ratio", "pf_frac_above", "pf_whipsaw", "pf_drift3m_z",
    "pf_quote_drift3m", "rsi14", "ema_dist", "macd", "macd_hist",
    "sma20_gap", "bb_z", "bb_width", "vol_1m_ratio_log"]
KB4_FEATS = [
    "bias", "p_kb2", "p_kb3", "kb2xkb3_agreement", "market_mid",
    "quote_present", "above_strike_z", "phase", "pf_frac_above",
    "pf_whipsaw", "pf_drift3m_z", "pf_quote_drift3m"]
KB5_FEATS = [
    "bias", "p_kb2_side", "p_kb3_side", "p_kb4_side", "market_side",
    "kb2_vs_market_disagreement", "ask", "claimed_edge",
    "strike_z_toward_side", "phase", "hot_hour", "pf_frac_above_side",
    "pf_whipsaw", "pf_quote_drift_side"]
KB6_FEATS = [
    "bias", "market_mid", "quote_present", "perp_gap_bp", "perp_mom_bp",
    "tape_imb_1m", "tape_imb_5m", "whale_net_15m", "k_oi_delta",
    "above_strike_z", "phase", "pf_frac_above"]
KB8_FEATS = ["bias", "kb7_log_odds", "market_log_odds"]
PT6_FEATS = ["bias", "leader_conf", "ask", "conf_minus_ask",
             "market_toward_side", "phase", "pf_drift3m_z"]

LOGIT_FILES = {
    "kb3": ("kb_logit.json", KB3_FEATS),
    "kb4": ("kb4_logit.json", KB4_FEATS),
    "kb5": ("kb5_logit.json", KB5_FEATS),
    "kb6": ("kb6_logit.json", KB6_FEATS),
    "kb8": ("kb8_logit.json", KB8_FEATS),
    "pt6": ("pt6_logit.json", PT6_FEATS),
}


def jload(name):
    return json.loads((RES / name).read_text())


def jsonl(name):
    out = []
    for line in (RES / name).open():
        line = line.strip()
        if line:
            out.append(json.loads(line))
    return out


# ------------------------------------------------------------------- 1. logits
def logits():
    out = {}
    for arm, (fn, names) in LOGIT_FILES.items():
        d = jload(fn)
        assert d["dim"] == len(names), (arm, d["dim"], len(names))
        out[arm] = {"updates": d["updates"],
                    "weights": [{"feature": n, "w": round(w, 4)}
                                for n, w in zip(names, d["w"])]}
    return out


# -------------------------------------------------------------- 2. calibrators
def calibrators():
    d = jload("kb_calib.json")
    out = {}
    for arm, c in d.items():
        n = c["n_eff"]
        out[arm] = {
            "a": round(c["a"], 4), "b": round(c["b"], 4),
            "updates": c["updates"],
            "mean_ll_cal": round(c["ll"] / n, 4),
            "mean_ll_raw": round(c["ll_raw"] / n, 4),
            "cal_minus_raw": round((c["ll"] - c["ll_raw"]) / n, 4),
        }
    return out


# ------------------------------------------------------------------ 3. bandits
def bandits():
    import struct
    kf = sorted({0.0, 0.1, -0.1, 0.2, -0.2, 0.35, -0.35, 0.5, -0.5,
                 0.65, -0.65, 0.8, -0.8, 1.0, -1.0, 1.25, -1.25,
                 1.5, -1.5})   # btc_rl/config.py K_FACTORS
    out = {"k_factors": kf, "linucb": {}, "linearq": {}, "dqn": {},
           "lstm": {}}
    # linucb_<variant>.json is _bandit_path for BOTH bandit kinds:
    # LinUCB dicts carry "A"/"b", live LinearQ (t7) dicts carry "w".
    # linear_q.json is only the 60-day batch warm-start — not live state.
    for f in sorted(RES.glob("linucb_t*.json")):
        d = json.loads(f.read_text())
        for hz, a in d.items():
            key = f.stem.replace("linucb_", "")
            fam = "linucb" if "A" in a else "linearq"
            out[fam][key] = {
                "dim": a["dim"], "alpha": a.get("alpha"),
                "n_arms": a["n_arms"], "pulls": a["pulls"],
                "total_pulls": sum(a["pulls"]),
                "top_arm_k": kf[max(range(a["n_arms"]),
                                    key=lambda i: a["pulls"][i])]}
    try:
        import torch
        for f in sorted(RES.glob("dqn_t8-*.pt")):
            d = torch.load(f, map_location="cpu", weights_only=False)
            shapes = [f"{k}:{tuple(v.shape)}" for k, v in d["state"].items()]
            out["dqn"][f.stem] = {"dim": d["dim"], "n_arms": d["n_arms"],
                                  "steps": d["steps"], "layers": shapes}
        for f in sorted(RES.glob("lstm_t9-*.pt")):
            d = torch.load(f, map_location="cpu", weights_only=False)
            shapes = [f"{k}:{tuple(v.shape)}" for k, v in d["state"].items()]
            out["lstm"][f.stem] = {"dim": d["dim"], "n_arms": d["n_arms"],
                                   "steps": d["steps"], "layers": shapes}
    except ImportError:
        pass
    return out


# --------------------------------------------------------------- 4. meta layer
def meta():
    t = jload("treatments.json")
    fsh = t.get("fshare") or {}
    ev = t.get("evlead") or {}
    ev_sum = {}
    for arm, hist in ev.items():
        if len(hist) >= 5:                     # same bar as t_evlead
            ev_sum[arm] = {"n": len(hist),
                           "mean_ev": round(sum(hist) / len(hist), 4),
                           "last": hist[-1]}
    treats = {}
    for k, v in (t.get("treats") or {}).items():
        n = v.get("n_bet", 0) + v.get("n_skip", 0)
        treats[k] = {"n_bet": v.get("n_bet"), "n_skip": v.get("n_skip"),
                     "ev_sum": round(v.get("ev_sum", 0.0), 4),
                     "ev_per_bet": round(v["ev_sum"] / v["n_bet"], 4)
                     if v.get("n_bet") else None}
    return {"fshare_w": {a: round(w, 4) for a, w in
                         sorted((fsh.get("w") or {}).items(),
                                key=lambda kv: -kv[1])},
            "fshare_alpha": fsh.get("alpha"), "fshare_eta": fsh.get("eta"),
            "evlead": ev_sum, "treatments": treats}


# --------------------------------------------------------- 5+6. trace + oracle
def fee1(a):
    """Per-contract Kalshi fee in cents at ask a (cents), unrounded."""
    p = a / 100.0
    return 7.0 * p * (1.0 - p)


def ev_per_dollar(a):
    """Winning-bet profit per $1 staked at ask a (cents), fee included."""
    f = fee1(a)
    return (100.0 - a - f) / (a + f)


def trace_and_oracle():
    pt = jsonl("pt_trades.jsonl")
    settled = [t for t in pt if t.get("win") is not None]
    kb = jsonl("kalshi_binary_log.jsonl")
    kb_by_tk = {}
    for r in kb:
        kb_by_tk.setdefault(r["ticker"], []).append(r)

    # ---- oracle over every settled desk window covered by the kb log
    per_win = []
    for t in settled:
        rows = kb_by_tk.get(t["ticker"])
        if not rows:
            continue
        win_yes = bool(t["actual"])
        asks = []
        for r in rows:
            if (r.get("variant", "kb") != "kb" or r.get("mkt_p_up") is None
                    or (r.get("mins_left") or 99) > 12):
                continue
            mid = r["mkt_p_up"] if win_yes else 1.0 - r["mkt_p_up"]
            a = 100.0 * mid + SEL_CF_ASK_ADJ
            if 5.0 <= a < 80.0:                # desk legality band
                asks.append(a)
        oracle_ev = ev_per_dollar(min(asks)) if asks else 0.0
        desk_ev = t["pnl_c"] / t["stake_c"]
        per_win.append({"ticker": t["ticker"],
                        "oracle_ask": round(min(asks), 1) if asks else None,
                        "oracle_ev": round(oracle_ev, 4),
                        "desk_ev": round(desk_ev, 4)})
    n = len(per_win)
    o_mean = sum(w["oracle_ev"] for w in per_win) / n
    d_mean = sum(w["desk_ev"] for w in per_win) / n
    oracle = {"windows": n, "desk_windows_settled": len(settled),
              "coverage_note": "kb per-minute log retains the most recent "
                               f"{n} of {len(settled)} settled desk windows;"
                               " both means computed on the SAME covered set",
              "oracle_ev": round(o_mean, 4), "desk_ev": round(d_mean, 4),
              "regret": round(o_mean - d_mean, 4),
              "oracle_skips": sum(1 for w in per_win
                                  if w["oracle_ask"] is None)}

    # ---- trace: most recent settled desk trade, end to end
    tr = settled[-1]
    # reproduce _pt_leader standings at entry time (settled decisions only,
    # desk envelope mins_left<=12, first tau-clearing row per window)
    pt_arms = ("kb2", "kb3", "kb4", "kb7", "kb8", "kb9")
    standings = {}
    for arm in pt_arms:
        byw = {}
        for r in kb:
            if (r.get("variant") == arm and r.get("actual") is not None
                    and (r.get("mins_left") or 99) <= 12
                    and r["close_ts"] <= tr["made_ts"]):
                byw.setdefault(r["ticker"], []).append(r)
        decs = []
        for rs in byw.values():
            rs.sort(key=lambda r: -r["mins_left"])
            for r in rs:
                if max(r["p_up"], 1 - r["p_up"]) >= PT_TAU:
                    decs.append((r["close_ts"], r["hit"],
                                 (r["p_up"] - r["actual"]) ** 2))
                    break
        decs.sort()
        decs = decs[-10:]
        if decs:
            standings[arm] = {
                "rec": f"{sum(h for _, h, _ in decs)}/{len(decs)}",
                "brier": round(sum(b for _, _, b in decs) / len(decs), 4)}
    rows = kb_by_tk.get(tr["ticker"], [])
    arm_rows = sorted((r for r in rows if r.get("variant") == tr["leader"]),
                      key=lambda r: r["made_ts"])
    # the decision row: same slot the desk filled on (nearest made_ts)
    dec = min(arm_rows, key=lambda r: abs(r["made_ts"] - tr["made_ts"])) \
        if arm_rows else None
    # reproduce the fill arithmetic from the logged row
    a, c = tr["ask_c"], tr["contracts"]
    order_fee = math.ceil(7.0 * c * (a / 100) * (1 - a / 100))
    stake_check = int(c * a) + order_fee
    pnl_check = (c * 100 - tr["stake_c"]) if tr["win"] else -tr["stake_c"]
    win_ev = ev_per_dollar(a)
    # oracle line for this same window
    mine = next((w for w in per_win if w["ticker"] == tr["ticker"]), None)
    trace = {
        "ticker": tr["ticker"], "leader": tr["leader"], "rec10": tr["rec10"],
        "side": tr["side"], "p_arm": tr["p_arm"], "tau": PT_TAU,
        "mins_left": tr["mins_left"], "ask_c": a,
        "fee_c_per_contract": tr["fee_c"], "contracts": c,
        "stake_c": tr["stake_c"], "stake_c_recomputed": stake_check,
        "order_fee_c": order_fee, "win": tr["win"], "pnl_c": tr["pnl_c"],
        "pnl_c_recomputed": pnl_check, "bankroll_c_after": tr["bankroll_c"],
        "ev_per_dollar_if_win": round(win_ev, 4),
        "strike": tr["strike"], "depth_cap_c": tr["depth_cap_c"],
        "decision_row": {k: dec.get(k) for k in
                         ("made_ts", "mins_left", "p_up", "mkt_p_up",
                          "base", "strike", "pf", "q80_w", "q80_lo",
                          "q80_hi", "b8x", "p_m1", "call")} if dec else None,
        "arm_row_count_in_window": len(arm_rows),
        "leader_standings_at_entry": standings,
        "oracle_same_window": mine,
    }
    return trace, oracle, per_win


# ------------------------------------------------- 7. metric representativeness
def representativeness():
    """Per-arm, over the settled windows the kb log covers: all-row Brier
    (the headline flavor), decision-time Brier (first tau-clearing row at
    mins_left<=12 — what the desk actually consumes), and the EV/$1 of
    betting that decision at the modeled ask. Divergence between columns
    = the headline metric not representing the arm's trading role."""
    kb = [r for r in jsonl("kalshi_binary_log.jsonl")
          if r.get("actual") is not None]
    arms = sorted({r.get("variant", "kb") for r in kb})
    out = {}
    for arm in arms:
        rows = [r for r in kb if r.get("variant", "kb") == arm]
        if not rows:
            continue
        briers = [(r["p_up"] - r["actual"]) ** 2 for r in rows
                  if r.get("p_up") is not None]
        byw = {}
        for r in rows:
            if (r.get("mins_left") or 99) <= 12 and r.get("p_up") is not None:
                byw.setdefault(r["ticker"], []).append(r)
        decs, evs = [], []
        for tk, rs in byw.items():
            rs.sort(key=lambda r: -r["mins_left"])
            for r in rs:
                if max(r["p_up"], 1 - r["p_up"]) >= PT_TAU:
                    decs.append((r["p_up"] - r["actual"]) ** 2)
                    if r.get("mkt_p_up") is not None:
                        sy = r["p_up"] >= 0.5
                        mid = r["mkt_p_up"] if sy else 1 - r["mkt_p_up"]
                        a = 100 * mid + SEL_CF_ASK_ADJ
                        if 5 <= a < 80:
                            won = sy == bool(r["actual"])
                            evs.append(ev_per_dollar(a) if won else -1.0)
                    break
        n_w = len({r["ticker"] for r in rows})
        out[arm] = {
            "windows": n_w, "rows": len(rows),
            "brier_all_rows": round(sum(briers) / len(briers), 4),
            "decisions": len(decs),
            "brier_decision": round(sum(decs) / len(decs), 4) if decs else None,
            "bet_evals": len(evs),
            "ev_per_dollar": round(sum(evs) / len(evs), 4) if evs else None,
            "decision_coverage": round(len(decs) / n_w, 3),
        }
    return out


def main():
    lg = logits()
    cal = calibrators()
    bd = bandits()
    mt = meta()
    trace, oracle, per_win = trace_and_oracle()
    rep = representativeness()

    compact = {
        "generated_ts": int(time.time()),
        "logits": {arm: v["weights"] for arm, v in lg.items()},
        "logit_updates": {arm: v["updates"] for arm, v in lg.items()},
        "calib": cal,
        "fshare_w": mt["fshare_w"],
        "evlead": mt["evlead"],
        "bandit_pulls": {k: v["pulls"] for k, v in bd["linucb"].items()},
        "linearq_pulls": {k: v["pulls"] for k, v in bd["linearq"].items()},
        "k_factors": bd["k_factors"],
        "dqn": {k: {kk: v[kk] for kk in ("dim", "n_arms", "steps")}
                for k, v in bd["dqn"].items()},
        "lstm": {k: {kk: v[kk] for kk in ("dim", "n_arms", "steps")}
                 for k, v in bd["lstm"].items()},
        "oracle": oracle,
        "trace": trace,
        "representativeness": rep,
    }
    (RES / "model_internals.json").write_text(
        json.dumps(compact, indent=1))
    print(json.dumps({"oracle": oracle, "fshare": mt["fshare_w"],
                      "evlead": mt["evlead"]}, indent=1))
    print("\n--- calib (mean ll cal vs raw) ---")
    for a, c in cal.items():
        print(f"{a:4s} a={c['a']:+.3f} b={c['b']:.3f} "
              f"ll_cal={c['mean_ll_cal']:.3f} ll_raw={c['mean_ll_raw']:.3f} "
              f"delta={c['cal_minus_raw']:+.3f}")
    print("\n--- representativeness ---")
    for a, r in sorted(rep.items()):
        print(f"{a:4s} windows={r['windows']:3d} "
              f"brier_all={r['brier_all_rows']:.4f} "
              f"brier_dec={r['brier_decision']} "
              f"dec={r['decisions']:3d} ev/$1={r['ev_per_dollar']} "
              f"(n={r['bet_evals']})")
    print("\n--- trace ---")
    print(json.dumps(trace, indent=1))
    print("\n--- pulls (linucb) ---")
    for k, v in sorted(bd["linucb"].items()):
        print(k, "total", v["total_pulls"], v["pulls"])
    for k, v in sorted(bd["linearq"].items()):
        print(k, "total", v["total_pulls"], v["pulls"])
    print("\n--- dqn/lstm ---")
    for fam in ("dqn", "lstm"):
        for k, v in sorted(bd[fam].items()):
            print(k, v["dim"], v["n_arms"], v["steps"], v["layers"])


if __name__ == "__main__":
    main()
