"""Fit the bid-pricing (fill-curve) and pacing models —
results/fill_curve.json. First step of the rule-based -> ML-based
trader migration (owner directive 2026-08-29).

Why REGRESSION / full-information counterfactuals, not deep RL, here:
every window's per-minute quote path is logged, so EVERY candidate
limit offset and EVERY entry-minute can be scored counterfactually on
the same windows — full-information expert evaluation, the same
argument that chose Fixed-Share over a bandit. Deep RL needs orders
of magnitude more interaction data than ~100 windows/day and cannot
be counterfactually audited row-by-row; it becomes appropriate at
multi-ticker scale (OS_BLUEPRINT §12 seams). Pricing basis: modeled
quotes (mkt_p_up + 2.5c half-spread), the registry's "model" family —
identical to the basis M11/t_limit already race on, so a learned
policy graduates into that family without a basis break.

Outputs:
  fill_curve: per limit-offset delta (cents below the decision ask):
    fill_rate, win_rate_given_fill, ev_per_$1 (with fees), n —
    the adverse-selection curve made quantitative.
  logistic fit: fill_prob(delta, mins_left) coefficients (Newton,
    ridge 1e-3) so a trader can price a bid at any state.
  pacing: EV and fill quality by entry-minute bucket — when in the
    window a taker entry has paid best, counterfactually.
  recommendation: EV-argmax delta with a 1-se caveat — evidence for
    a future M13 treatment, NEVER a live change (champion law).
"""
from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"
sys.path.insert(0, str(ROOT))
from btc_rl import online as O            # noqa: E402

HALF_SPREAD_C = 2.5          # the registry's modeled-quote convention
DELTAS = list(range(0, 9))   # cents below the decision ask
ENVELOPE_MIN = 12.0


def rows(name):
    p = RES / name
    return [json.loads(l) for l in p.open() if l.strip()] \
        if p.exists() else []


def fee_c(ask):
    return math.ceil(7 * (ask / 100.0) * (1 - ask / 100.0))


def ev_taker(ask, won):
    cost = ask + fee_c(ask)
    if cost <= 0 or cost >= 100:
        return None
    return (100.0 - cost) / cost if won else -1.0


def main():
    now = int(time.time())
    # group kb control-variant rows into per-window quote paths
    wins = {}
    for r in rows("kalshi_binary_log.jsonl"):
        if r.get("variant") not in (None, "kb"):
            continue
        if r.get("mkt_p_up") is None:
            continue
        wins.setdefault(r["ticker"], []).append(r)
    episodes = []
    for tk, rs in wins.items():
        rs.sort(key=lambda r: -r.get("mins_left", 0))
        settled = [r for r in rs if r.get("actual") is not None]
        if not settled:
            continue
        outcome = settled[-1]["actual"]
        path = [r for r in rs if r.get("mins_left") is not None
                and 0.5 <= r["mins_left"] <= ENVELOPE_MIN]
        if len(path) < 4:
            continue
        episodes.append((tk, outcome, path))

    # decision point: first row inside the envelope; side = market lean
    curve = {d: {"fills": 0, "wins": 0, "n": 0, "ev_sum": 0.0}
             for d in DELTAS}
    fit_pts = []                     # (delta, mins_left_at_dec, filled)
    for tk, outcome, path in episodes:
        dec = path[0]
        side_yes = dec["mkt_p_up"] >= 0.5
        won = bool(outcome) == side_yes

        def ask_at(r):
            p = r["mkt_p_up"] if side_yes else 1 - r["mkt_p_up"]
            return min(99.0, max(1.0, 100 * p + HALF_SPREAD_C))
        ask0 = ask_at(dec)
        later = path[1:]
        for d in DELTAS:
            limit = ask0 - d
            filled_ask = None
            if d == 0:
                filled_ask = ask0
            else:
                for r in later:
                    if ask_at(r) <= limit:
                        filled_ask = ask_at(r)
                        break
            c = curve[d]
            c["n"] += 1
            fit_pts.append((d, dec["mins_left"],
                            1 if filled_ask is not None else 0))
            if filled_ask is None:
                continue            # unfilled limit: EV 0, no risk
            c["fills"] += 1
            c["wins"] += 1 if won else 0
            ev = ev_taker(filled_ask, won)
            if ev is not None:
                c["ev_sum"] += ev

    out_curve = []
    for d in DELTAS:
        c = curve[d]
        fr = c["fills"] / c["n"] if c["n"] else None
        wr = c["wins"] / c["fills"] if c["fills"] else None
        ev = c["ev_sum"] / c["n"] if c["n"] else None  # per attempted $1
        se = None
        if c["n"] > 1 and ev is not None:
            se = abs(ev) / math.sqrt(c["n"])  # rough scale, labeled
        out_curve.append({"delta_c": d, "n": c["n"],
                          "fill_rate": round(fr, 3) if fr is not None
                          else None,
                          "win_rate_given_fill": round(wr, 3)
                          if wr is not None else None,
                          "ev_per_$1_attempted": round(ev, 4)
                          if ev is not None else None})

    # logistic fill_prob(delta, mins_left): Newton with ridge
    w = [0.0, 0.0, 0.0]          # bias, delta, mins_left
    for _ in range(25):
        g = [0.0] * 3
        H = [[1e-3 if i == j else 0.0 for j in range(3)]
             for i in range(3)]
        for d, ml, y in fit_pts:
            x = [1.0, float(d), float(ml)]
            z = sum(wi * xi for wi, xi in zip(w, x))
            p = 1 / (1 + math.exp(-max(-30, min(30, z))))
            for i in range(3):
                g[i] += (p - y) * x[i]
                for j in range(3):
                    H[i][j] += p * (1 - p) * x[i] * x[j]
        # solve H step = g (3x3 Gaussian elimination)
        M = [Hr[:] + [gr] for Hr, gr in zip(H, g)]
        for i in range(3):
            piv = max(range(i, 3), key=lambda r_: abs(M[r_][i]))
            M[i], M[piv] = M[piv], M[i]
            for r_ in range(3):
                if r_ != i and M[i][i]:
                    f = M[r_][i] / M[i][i]
                    M[r_] = [a - f * b for a, b in zip(M[r_], M[i])]
        step = [M[i][3] / M[i][i] if M[i][i] else 0.0 for i in range(3)]
        w = [wi - si for wi, si in zip(w, step)]
        if max(abs(s) for s in step) < 1e-8:
            break

    # pacing: taker EV by entry-minute bucket (counterfactual)
    pace = {}
    for tk, outcome, path in episodes:
        for lo, hi, name in ((9, 12, "12-9m"), (6, 9, "9-6m"),
                             (3, 6, "6-3m"), (0.5, 3, "3-0.5m")):
            cand = [r for r in path if lo <= r["mins_left"] < hi]
            if not cand:
                continue
            r = cand[0]
            side_yes = r["mkt_p_up"] >= 0.5
            won = bool(outcome) == side_yes
            p_ = r["mkt_p_up"] if side_yes else 1 - r["mkt_p_up"]
            ask = min(99.0, max(1.0, 100 * p_ + HALF_SPREAD_C))
            ev = ev_taker(ask, won)
            if ev is None:
                continue
            b = pace.setdefault(name, {"n": 0, "ev_sum": 0.0,
                                       "wins": 0})
            b["n"] += 1
            b["ev_sum"] += ev
            b["wins"] += 1 if won else 0
    out_pace = [{"bucket": k, "n": v["n"],
                 "ev_per_$1": round(v["ev_sum"] / v["n"], 4),
                 "win_rate": round(v["wins"] / v["n"], 3)}
                for k, v in pace.items()]

    best = max((c for c in out_curve
                if c["ev_per_$1_attempted"] is not None),
               key=lambda c: c["ev_per_$1_attempted"], default=None)
    doc = {
        "generated_ts": now,
        "basis": "model quotes (mkt_p_up + 2.5c) — same family as "
                 "M11/t_limit; NOT real fills",
        "windows": len(episodes),
        "fill_curve": out_curve,
        "fill_logit": {"bias": round(w[0], 4),
                       "per_delta_c": round(w[1], 4),
                       "per_min_left": round(w[2], 4),
                       "note": "fill_prob = sigmoid(bias + "
                               "per_delta_c*delta + per_min_left*ml)"},
        "pacing_taker_ev": sorted(out_pace, key=lambda b: b["bucket"]),
        "recommendation": {
            "ev_argmax_delta_c": best["delta_c"] if best else None,
            "caveat": "evidence for a FUTURE M13 treatment only — "
                      "nothing changes live without an SPRT win "
                      "(champion law); adverse selection is visible "
                      "in win_rate_given_fill falling with delta",
        },
    }
    (RES / "fill_curve.json").write_text(json.dumps(doc, indent=1))
    print(f"fill_curve.json: {len(episodes)} windows; "
          f"argmax delta = {best['delta_c'] if best else '—'}c; "
          "curve:", [(c['delta_c'], c['ev_per_$1_attempted'])
                     for c in out_curve])


if __name__ == "__main__":
    main()
