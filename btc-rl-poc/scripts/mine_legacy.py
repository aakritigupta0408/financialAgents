"""Mine the legacy lake (TA §23, adopted 2026-08-29): the four
analyses that decide whether the Alpha Capture Engine is worth
building — run on the minute-level tape we already own, with its
limits stated (no sub-minute paths, no queue, modeled asks).

Conventions (all legacy-honest):
  * ask model: 100*p_mkt + 2.5 (yes) / mirror (no) — the registry's
    model basis; fee = 7*p*(1-p) fractional per contract;
  * fair value at any minute = kb2's contemporaneous p_up (the
    control caller) — no settlement hindsight anywhere;
  * robust-edge proxy RE_t = 100*p_fair_side - (ask_side + fee) - 4
    (4c = crude uncertainty+toxicity reserve; labeled CRUDE — the
    real LCB needs the new calibration layer);
  * eras: CLEAN = close_ts >= 1787966357 (spec-fee era start);
    windows during the dispersion incident are NOT masked separately
    (bands fed sigma, not kb2's market blend).
Output: results/legacy_mine.json + printed headline answers.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"
RESERVE_C = 4.0
CLEAN_TS = 1787966357


def fee(a):
    return 7 * (a / 100.0) * (1 - a / 100.0)


def ask_of(p_mkt, side):
    p = p_mkt if side == "up" else 1 - p_mkt
    return min(99.0, max(1.0, 100 * p + 2.5))


def re_of(p_fair, p_mkt, side):
    pf = p_fair if side == "up" else 1 - p_fair
    a = ask_of(p_mkt, side)
    return 100 * pf - (a + fee(a)) - RESERVE_C, a


def main():
    rows = [json.loads(l) for l in
            (RES / "kalshi_binary_log.jsonl").open() if l.strip()]
    # per-window kb2 path (fair) + market path, minute-ordered
    wins = {}
    for r in rows:
        if (r.get("variant") or "kb") != "kb2":
            continue
        if r.get("mkt_p_up") is None or r.get("p_up") is None \
                or r.get("mins_left") is None:
            continue
        wins.setdefault(r["ticker"], []).append(r)
    paths = {}
    for tk, rs in wins.items():
        rs.sort(key=lambda r: -r["mins_left"])
        settled = [r for r in rs if r.get("actual") is not None]
        if not settled or len(rs) < 5:
            continue
        paths[tk] = (rs, bool(settled[-1]["actual"]),
                     max(r.get("close_ts") or 0 for r in rs))
    out = {"generated_ts": int(time.time()), "windows": len(paths),
           "conventions": "modeled asks, kb2 contemporaneous fair, "
           "RE = fair - ask - fee - 4c crude reserve; minute "
           "resolution — sub-minute claims UNAVAILABLE"}

    # ---- A1: high-confidence call -> later market excursion ---------
    a1 = {}
    for lo, hi in ((0.75, 0.80), (0.80, 0.85), (0.85, 0.90),
                   (0.90, 1.01)):
        n = won = drop10 = drop20 = 0
        entry_asks, min_asks = [], []
        for tk, (rs, outcome, _) in paths.items():
            env = [r for r in rs if 6 <= r["mins_left"] <= 13]
            call = next((r for r in env
                         if lo <= max(r["p_up"], 1 - r["p_up"]) < hi),
                        None)
            if call is None:
                continue
            side = "up" if call["p_up"] >= 0.5 else "down"
            later = [r for r in rs
                     if r["mins_left"] < call["mins_left"]]
            if not later:
                continue
            a0 = ask_of(call["mkt_p_up"], side)
            amin = min(ask_of(r["mkt_p_up"], side) for r in later)
            n += 1
            won += 1 if (side == "up") == outcome else 0
            entry_asks.append(a0)
            min_asks.append(amin)
            drop10 += 1 if a0 - amin >= 10 else 0
            drop20 += 1 if a0 - amin >= 20 else 0
        if n:
            a1[f"{lo:.2f}-{hi:.2f}"] = {
                "n": n, "win_rate": round(won / n, 3),
                "avg_entry_ask": round(sum(entry_asks) / n, 1),
                "avg_later_min_ask": round(sum(min_asks) / n, 1),
                "p_drop_ge10c": round(drop10 / n, 3),
                "p_drop_ge20c": round(drop20 / n, 3)}
    out["a1_excursion_after_confidence"] = a1

    # ---- A2: divergence -> settlement residual (mispricing surface) -
    a2 = {}
    for phase, plo, phi in (("early_10-13m", 10, 13.5),
                            ("mid_5-8m", 5, 8),
                            ("late_1-3m", 1, 3)):
        for dlo, dhi, tag in ((0.10, 0.20, "+10-20pp"),
                              (0.20, 1.0, ">20pp")):
            n = wonn = 0
            mkt_sum = 0.0
            for tk, (rs, outcome, cts) in paths.items():
                if cts < CLEAN_TS:
                    continue                     # clean era only
                r = next((x for x in rs
                          if plo <= x["mins_left"] <= phi), None)
                if r is None:
                    continue
                d = r["p_up"] - r["mkt_p_up"]
                if not (dlo <= abs(d) < dhi):
                    continue
                side_up = d > 0
                n += 1
                wonn += 1 if side_up == outcome else 0
                mkt_sum += r["mkt_p_up"] if side_up \
                    else 1 - r["mkt_p_up"]
            if n >= 10:
                a2[f"{phase} |div| {tag}"] = {
                    "n": n,
                    "market_implied": round(mkt_sum / n, 3),
                    "actual_win_rate": round(wonn / n, 3),
                    "residual_pp": round(100 * (wonn / n
                                                - mkt_sum / n), 1)}
    out["a2_divergence_residual_clean_era"] = a2

    # ---- A3: BUY-now vs WAIT (coarse, one-snapshot & best-future) ---
    buy_now_re, wait_gain, disappeared = [], [], 0
    n3 = 0
    for tk, (rs, outcome, _) in paths.items():
        env = [r for r in rs if 2 <= r["mins_left"] <= 13]
        first = next((r for r in env
                      if re_of(max(r["p_up"], 1 - r["p_up"]),
                               r["mkt_p_up"],
                               "up" if r["p_up"] >= 0.5 else "down"
                               )[0] > 0), None)
        if first is None:
            continue
        side = "up" if first["p_up"] >= 0.5 else "down"
        re0, _ = re_of(first["p_up"] if side == "up"
                       else 1 - first["p_up"] + 0,
                       first["mkt_p_up"], side)
        re0 = re_of(max(first["p_up"], 1 - first["p_up"]),
                    first["mkt_p_up"], side)[0]
        later = [r for r in rs if 1 <= r["mins_left"]
                 < first["mins_left"]]
        if not later:
            continue
        best_future = max(
            (re_of(max(r["p_up"], 1 - r["p_up"]), r["mkt_p_up"],
                   "up" if r["p_up"] >= 0.5 else "down")[0]
             for r in later
             if ("up" if r["p_up"] >= 0.5 else "down") == side),
            default=-99)
        n3 += 1
        buy_now_re.append(re0)
        wait_gain.append(best_future - re0)
        if best_future <= 0:
            disappeared += 1
    if n3:
        wg = sorted(wait_gain)
        out["a3_buy_vs_wait"] = {
            "n": n3,
            "mean_re_at_first_positive": round(
                sum(buy_now_re) / n3, 1),
            "mean_wait_gain_c": round(sum(wait_gain) / n3, 1),
            "median_wait_gain_c": round(wg[n3 // 2], 1),
            "p_wait_gain_positive": round(
                sum(1 for x in wait_gain if x > 0) / n3, 3),
            "p_opportunity_disappeared": round(disappeared / n3, 3),
            "note": "coarse minute policy only — sub-minute timing "
                    "UNAVAILABLE FROM LEGACY TAPE"}

    # ---- A4: trader ECR (entry edge / best valid edge) --------------
    a4 = {}
    for name, f in (("pt", "pt_trades.jsonl"),
                    ("pt4", "pt4_trades.jsonl"),
                    ("pt6", "pt6_trades.jsonl")):
        trades = [json.loads(l) for l in (RES / f).open()
                  if l.strip()]
        ecrs, esrs = [], []
        for t in trades:
            if t.get("actual") is None or t.get("skipped"):
                continue
            tk = t["ticker"]
            if tk not in paths:
                continue
            rs, outcome, _ = paths[tk]
            side = "up" if t["side"] == "yes" else "down"
            at = next((r for r in rs
                       if abs(r["mins_left"]
                              - (t.get("mins_left") or -9)) < 0.8),
                      None)
            if at is None:
                continue
            pf = max(at["p_up"], 1 - at["p_up"]) \
                if (("up" if at["p_up"] >= 0.5 else "down") == side) \
                else min(at["p_up"], 1 - at["p_up"])
            a_paid = t["ask_c"]
            entry_edge = 100 * pf - (a_paid + fee(a_paid))
            best = max((re_of(max(r["p_up"], 1 - r["p_up"]),
                              r["mkt_p_up"],
                              "up" if r["p_up"] >= 0.5 else "down")[0]
                        + RESERVE_C
                        for r in rs if 1 <= r["mins_left"] <= 13
                        if ("up" if r["p_up"] >= 0.5 else "down")
                        == side), default=None)
            if best is None or best <= 0:
                continue
            ecrs.append(max(0.0, entry_edge) / best)
            cost = a_paid + fee(a_paid)
            realized = (100 - cost) / cost if t["win"] else -1.0
            if entry_edge > 0:
                esrs.append(realized / (entry_edge / cost))
        if ecrs:
            a4[name] = {"n": len(ecrs),
                        "mean_ECR": round(sum(ecrs) / len(ecrs), 3),
                        "mean_ESR": round(sum(esrs) / len(esrs), 2)
                        if esrs else None}
    out["a4_trader_ecr"] = a4

    (RES / "legacy_mine.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1)[:3000])


if __name__ == "__main__":
    main()
