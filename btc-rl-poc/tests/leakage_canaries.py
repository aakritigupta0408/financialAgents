#!/usr/bin/env python3
"""Leakage canaries + placebo tests (master directive section 9, 38).

The evaluation pipeline must PROVE it rejects future information. Two
families of checks, all pure stdlib, all seeded/deterministic (no wall
clock inside any check — wall time appears only in the report's
generated_ts metadata):

CANARIES — plant synthetic future info and verify the real pipeline's
data contracts would have kept it out:

  1. future-label canary. A walk-forward (test-then-train, settle order)
     logistic fit on legitimate decision-time fields only (p_up,
     mkt_p_up, mins_left) is compared with the same fit ALSO given the
     settled outcome as a fake feature. The canary fit must score
     materially higher (sanity: the canary is detectable by the probe).
     Then the real check: scan every field of the decision-time kb row
     schema for anything correlating > 0.95 with the outcome across
     >= 100 settled windows — only settlement-stamp fields (whitelisted
     below) may do so — and verify no still-open window carries an
     outcome.

     DATA CONTRACT (verified in btc_rl/online.py ~L3430-3444): kb rows
     are APPENDED at decision time with actual=None and STAMPED IN PLACE
     at settlement with actual/hit/brier/mkt_brier. So the persisted
     file legitimately contains settled rows whose mins_left is the
     decision-time value (e.g. 5.0) next to a filled `actual` — the
     literal reading "mins_left > 1 implies actual is null" only holds
     for windows still open at the last log write. The testable
     invariant is therefore: any row whose close_ts is later than the
     last observed decision timestamp (data_now) MUST have actual=None,
     and no settled row may have close_ts materially beyond data_now.

  2. quote-age canary. Independent replay of the treatment evaluator's
     row selection (leader variant, mins_left <= 12 envelope, earliest
     row inside the envelope) and independent recomputation of the
     champion (model-basis) and champion_real (real-fill) EVs from that
     row's quote. Any logged score that disagrees means the evaluator
     priced off a different — possibly newer — quote than the decision.

  3. timestamp-order canary. Every trader ledger: made_ts < close_ts
     strictly; settlement fields present => close_ts <= data_now (the
     latest decision timestamp observed anywhere in the datasets, plus
     one 15-min window of slack) — a settled row from the future is a
     leak; pnl recorded without an outcome is a corruption.

PLACEBOS — run the real machinery on deliberate noise; it must find
NOTHING (section 38: if a placebo finds persistent alpha, the
infrastructure is broken, SEV-0):

  4. shuffled-label placebo. Permute the settled outcomes (seeded, 5
     permutations) and recompute the champion-vs-t_regime paired mean
     difference each time with the same pairing semantics the SPRT uses
     (stand-down scores 0, paired against the champion on the same
     window). CONTRACT LEARNED WHILE BUILDING THIS: the null is NOT
     zero. On a stand-down window the paired difference is minus the
     champion's EV, and a bet at ask+fee under a RANDOM outcome has
     negative expectation (the house edge) — so a veto policy
     mechanically earns +cost-drag per skipped window even on shuffled
     labels. The first version of this check flagged exactly that
     (+0.01..+0.10/$1 across all 5 permutations) as "fabricated alpha";
     it is arithmetic, not leakage. The honest requirement: each
     permuted paired mean must sit within 3 SD of its ANALYTIC
     cost-drag expectation (derived below from the asks and the
     label-shuffle marginal), i.e. shuffling must create no edge BEYOND
     the mechanical cost structure.

  5. random-policy placebo. A seeded coin-flip policy scored at the
     modeled decision-time asks across the same settled windows must
     earn approximately the analytic house edge (negative), i.e. the
     scoring machinery itself hands out no free alpha.

  6. noise-feature placebo. The walk-forward logistic from check 1 with
     five seeded pure-noise features added must not beat the
     legitimate-features-only fit by more than 1pp (averaged over 5
     noise seeds).

Nothing is imported from btc_rl — every formula is re-derived here so a
bug in the runtime cannot hide itself. Output:
results/leakage_canaries.json with {generated_ts, checks, overall,
note}. overall is PASS or SEV-0. Exit code is always 0 — the JSON
verdict is the signal.
"""

import json
import math
import os
import random
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "results")
OUT_PATH = os.path.join(RESULTS, "leakage_canaries.json")

SEED = 20260829                 # fixed — reruns are bit-identical
ENVELOPE_MIN = 12.0             # the desk's tradeable envelope (<=12 min)
EV_TOL = 1e-3                   # treatments.jsonl rounds EV to 4dp
SETTLE_SLACK_S = 900            # one window: settle may trail last decision
MIN_WINDOWS = 100               # correlation scan needs this many
CORR_LIMIT = 0.95
# Settlement-stamp fields: written ONLY at settle time (contract above),
# so they are ALLOWED to correlate with the outcome. Anything else that
# does is future information sitting in a decision-time schema.
SETTLEMENT_FIELDS = {"actual", "hit", "win", "brier", "mkt_brier",
                     "pnl_c", "settled"}
# Placebo 4: each permuted-label paired mean must sit within this many
# SDs of its analytic cost-drag expectation (see module docstring).
PLACEBO_SD_LIMIT = 3.0
LOGIT_WARMUP = 20               # walk-forward: first N windows train only
CANARY_MIN_GAIN = 0.10          # canary fit must beat legit by >= 10pp
NOISE_MAX_GAIN = 0.01           # noise features may add at most 1pp


# ---------------------------------------------------------------------------
# Independent primitives (nothing imported from btc_rl)
# ---------------------------------------------------------------------------

def load_jsonl(name):
    path = os.path.join(RESULTS, name)
    rows = []
    if not os.path.exists(path):
        return rows
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def bet_ev(side, ask_c, outcome):
    """EV per $1 staked for one settled binary bet — re-derived from the
    published contract (fee = ceil(7*p*(1-p)) cents, p = ask/100)."""
    fee = math.ceil(7 * (ask_c / 100.0) * (1 - ask_c / 100.0))
    cost = ask_c + fee
    if cost <= 0:
        return 0.0
    won = (side == "yes") == bool(outcome)
    return (100.0 - cost) / cost if won else -1.0


def modeled_asks(mkt_p_up):
    """The evaluator's decision-time quote model: mid + 2.5c spread."""
    return 100.0 * mkt_p_up + 2.5, 100.0 * (1 - mkt_p_up) + 2.5


def pearson(xs, ys):
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    vx = sum((x - mx) ** 2 for x in xs)
    vy = sum((y - my) ** 2 for y in ys)
    if vx <= 0 or vy <= 0:
        return 0.0
    cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    return cov / math.sqrt(vx * vy)


def decision_rows(kb_rows, variant="kb"):
    """One row per settled window: the EARLIEST row inside the <=12-min
    envelope (max mins_left) — the same selection rule as the runtime's
    _regime_acc/_treat_evaluate, re-implemented independently."""
    dt = {}
    for r in kb_rows:
        if ((r.get("variant") or "kb") != variant
                or r.get("actual") is None or r.get("mkt_p_up") is None
                or (r.get("mins_left") or 99) > ENVELOPE_MIN):
            continue
        tk = r["ticker"]
        if tk not in dt or r["mins_left"] > dt[tk]["mins_left"]:
            dt[tk] = r
    return sorted(dt.values(), key=lambda r: r["close_ts"])


class OnlineLogit:
    """Tiny SGD logistic, test-then-train (prequential): each window is
    PREDICTED with weights fit on strictly earlier windows, then used to
    update. That is the walk-forward discipline under test."""

    def __init__(self, dim, lr=0.3):
        self.w = [0.0] * dim
        self.lr = lr

    def prob(self, x):
        z = sum(w * v for w, v in zip(self.w, x))
        z = max(-30.0, min(30.0, z))
        return 1.0 / (1.0 + math.exp(-z))

    def update(self, x, y):
        p = self.prob(x)
        g = y - p
        for i, v in enumerate(x):
            self.w[i] += self.lr * g * v


def walk_forward_acc(rows, feat_fn):
    """Prequential accuracy over settle-ordered windows, scoring only
    after LOGIT_WARMUP training examples."""
    dim = len(feat_fn(rows[0]))
    model = OnlineLogit(dim)
    correct = scored = 0
    for i, r in enumerate(rows):
        x = feat_fn(r)
        y = int(r["actual"])
        if i >= LOGIT_WARMUP:
            scored += 1
            correct += int((model.prob(x) >= 0.5) == bool(y))
        model.update(x, y)
    return (correct / scored if scored else 0.0), scored


def legit_feats(r):
    """ONLY legitimate decision-time fields: p_up, mkt_p_up, mins_left."""
    return [1.0,
            (r["p_up"] - 0.5) * 2.0,
            (r["mkt_p_up"] - 0.5) * 2.0,
            (r["mins_left"] or 0.0) / 15.0]


# ---------------------------------------------------------------------------
# Shared reconstruction: match treatments.jsonl records back to their
# decision rows (used by checks 2, 4, 5)
# ---------------------------------------------------------------------------

def match_treatment_records(treat_recs, pt_trades, kb_rows):
    """For each scored window record, independently re-select the row the
    evaluator should have priced from: leader variant, settled, inside
    the <=12 envelope, earliest such row (max mins_left)."""
    pt_by_tk = {t["ticker"]: t for t in pt_trades
                if t.get("actual") is not None}
    kb_by_tk = {}
    for r in kb_rows:
        if r.get("actual") is None or r.get("mkt_p_up") is None:
            continue
        kb_by_tk.setdefault(r["ticker"], []).append(r)
    matched, unmatchable = [], 0
    for rec in treat_recs:
        t = pt_by_tk.get(rec["ticker"])
        cand = [r for r in kb_by_tk.get(rec["ticker"], [])
                if r.get("variant") == rec.get("leader")
                and (r.get("mins_left") or 99) <= ENVELOPE_MIN]
        if t is None or not cand:
            unmatchable += 1     # log rotation can outlive old windows
            continue
        row = max(cand, key=lambda r: r["mins_left"])
        # rows tied at the max mins_left: the runtime keeps the
        # first-seen on a strict > comparison, so a tie with differing
        # quotes is selection-AMBIGUOUS, not evidence of a newer quote
        # (observed once in 151 windows, 2026-08-29)
        ties = [r for r in cand if r["mins_left"] == row["mins_left"]]
        matched.append({"rec": rec, "trade": t, "row": row,
                        "ties": ties})
    return matched, unmatchable


# ---------------------------------------------------------------------------
# Checks
# ---------------------------------------------------------------------------

def check_future_label(kb_rows):
    name = "future_label_canary"
    method = ("walk-forward logistic (test-then-train, settle order) on "
              "legit decision-time fields [p_up, mkt_p_up, mins_left] vs "
              "the same fit + the settled outcome as a planted feature; "
              "then scan every decision-row field for |corr|>0.95 with "
              "the outcome (settlement-stamp fields whitelisted) and "
              "verify no still-open window carries an outcome")
    rows = decision_rows(kb_rows)
    if len(rows) < MIN_WINDOWS:
        return {"name": name, "method": method, "result": "FAIL",
                "detail": f"only {len(rows)} settled decision windows "
                          f"(< {MIN_WINDOWS}) — cannot certify"}
    # -- sanity: is the probe strong enough to SEE a planted label? ------
    acc_legit, n_scored = walk_forward_acc(rows, legit_feats)
    acc_canary, _ = walk_forward_acc(
        rows, lambda r: legit_feats(r) + [(r["actual"] - 0.5) * 2.0])
    gain = acc_canary - acc_legit
    problems = []
    if gain < CANARY_MIN_GAIN:
        problems.append(
            f"canary NOT detectable: legit acc {acc_legit:.3f} vs "
            f"canary acc {acc_canary:.3f} (+{100 * gain:.1f}pp < "
            f"{100 * CANARY_MIN_GAIN:.0f}pp) — the probe is too weak "
            f"to certify anything")
    # -- the real check: no field in the decision-time schema leaks ------
    ys = [float(r["actual"]) for r in rows]
    fields = {}
    for r in rows:
        for k, v in r.items():
            if isinstance(v, bool):
                v = int(v)
            if isinstance(v, (int, float)):
                fields.setdefault(k, {})[r["ticker"]] = float(v)
            elif isinstance(v, list) and all(
                    isinstance(e, (int, float)) for e in v):
                for i, e in enumerate(v):
                    fields.setdefault(f"{k}[{i}]", {})[r["ticker"]] = \
                        float(e)
    leaks, whitelisted = [], []
    for fname, by_tk in sorted(fields.items()):
        pairs = [(by_tk[r["ticker"]], y)
                 for r, y in zip(rows, ys) if r["ticker"] in by_tk]
        if len(pairs) < MIN_WINDOWS:
            continue
        c = pearson([p[0] for p in pairs], [p[1] for p in pairs])
        if abs(c) > CORR_LIMIT:
            base = fname.split("[")[0]
            if base in SETTLEMENT_FIELDS:
                whitelisted.append(f"{fname} (r={c:+.3f})")
            else:
                leaks.append(f"{fname} (r={c:+.3f})")
    if leaks:
        problems.append("decision-time field(s) correlate >0.95 with "
                        "outcome: " + ", ".join(leaks))
    # -- pre-settle nullness under the stamp-in-place contract -----------
    data_now = max(r["made_ts"] for r in kb_rows)
    open_with_outcome = sum(
        1 for r in kb_rows
        if r["close_ts"] > data_now and r.get("actual") is not None)
    n_open = sum(1 for r in kb_rows if r["close_ts"] > data_now)
    if open_with_outcome:
        problems.append(f"{open_with_outcome} still-open windows "
                        f"(close_ts > last decision ts) already carry an "
                        f"outcome — future information in the log")
    detail = (f"probe: legit acc {acc_legit:.3f} vs +canary "
              f"{acc_canary:.3f} (+{100 * gain:.1f}pp) over {n_scored} "
              f"scored of {len(rows)} windows; schema scan: "
              f"{len(fields)} fields x {len(rows)} settled windows, "
              f"0 non-whitelisted fields with |r|>{CORR_LIMIT}"
              + (f" (settlement stamps as expected: "
                 f"{', '.join(whitelisted)})" if whitelisted else "")
              + f"; open-window nullness: {open_with_outcome} violations "
                f"in {n_open} open rows ({len(kb_rows)} total)")
    if problems:
        detail = "; ".join(problems) + " | " + detail
    return {"name": name, "method": method,
            "result": "FAIL" if problems else "PASS", "detail": detail}


def check_quote_age(matched, unmatchable, n_recs):
    name = "quote_age_canary"
    method = ("independent replay of evaluator row selection (leader "
              "variant, mins_left<=12 envelope, earliest row inside it) "
              "+ independent EV recomputation: logged ev.champion must "
              "equal bet_ev at THAT row's modeled quote and "
              "ev.champion_real must equal bet_ev at the desk's actual "
              "fill — any disagreement means a different (possibly "
              "newer) quote was scored")
    if not matched:
        return {"name": name, "method": method, "result": "FAIL",
                "detail": f"0 of {n_recs} treatment records matchable "
                          f"to kb decision rows — cannot certify"}
    env_bad = ev_bad = real_bad = out_bad = 0
    examples = []
    for m in matched:
        rec, t, row = m["rec"], m["trade"], m["row"]
        if not (0.0 < row["mins_left"] <= ENVELOPE_MIN):
            env_bad += 1
        if rec["outcome"] != t["actual"] or rec["outcome"] != row["actual"]:
            out_bad += 1
        got = rec["ev"].get("champion")
        wants = []
        for tr in m.get("ties", [row]):
            ay, an = modeled_asks(tr["mkt_p_up"])
            wants.append(bet_ev(t["side"],
                                ay if t["side"] == "yes" else an,
                                rec["outcome"]))
        want = wants[0]
        if got is None or all(abs(got - w) > EV_TOL for w in wants):
            ev_bad += 1
            if len(examples) < 3:
                examples.append(f"{rec['ticker']} champion logged "
                                f"{got} vs recomputed {want:.4f}")
        want_r = bet_ev(t["side"], t["ask_c"], rec["outcome"])
        got_r = rec["ev"].get("champion_real")
        if got_r is None or abs(got_r - want_r) > EV_TOL:
            real_bad += 1
            if len(examples) < 3:
                examples.append(f"{rec['ticker']} champion_real logged "
                                f"{got_r} vs recomputed {want_r:.4f}")
    bad = env_bad + ev_bad + real_bad + out_bad
    detail = (f"{len(matched)} of {n_recs} scored windows replayed "
              f"({unmatchable} predate the rotated kb log); envelope "
              f"violations {env_bad}, model-quote EV mismatches "
              f"{ev_bad}, real-fill EV mismatches {real_bad}, outcome "
              f"mismatches {out_bad} — {bad} total violations")
    if examples:
        detail += "; e.g. " + "; ".join(examples)
    return {"name": name, "method": method,
            "result": "FAIL" if bad else "PASS", "detail": detail}


def check_timestamp_order(ledgers, data_now):
    name = "timestamp_order_canary"
    method = ("every trader ledger: made_ts < close_ts strictly; "
              "settlement fields (actual/win/pnl_c) present => close_ts "
              f"<= data_now+{SETTLE_SLACK_S}s (data_now = latest "
              "decision timestamp observed across all datasets — no "
              "wall clock, deterministic); pnl without outcome counted "
              "as corruption")
    order_bad = future_bad = orphan_pnl = total = 0
    per_ledger = []
    for lname, rows in ledgers.items():
        ob = fb = op = 0
        for r in rows:
            total += 1
            if not (r["made_ts"] < r["close_ts"]):
                ob += 1
            settled = (r.get("actual") is not None
                       or r.get("win") is not None
                       or r.get("pnl_c") is not None)
            if settled and r["close_ts"] > data_now + SETTLE_SLACK_S:
                fb += 1
            if r.get("pnl_c") is not None and r.get("actual") is None:
                op += 1
        order_bad += ob
        future_bad += fb
        orphan_pnl += op
        per_ledger.append(f"{lname}:{len(rows)}r/{ob + fb + op}v")
    bad = order_bad + future_bad + orphan_pnl
    detail = (f"{bad} violations in {total} rows across "
              f"{len(ledgers)} ledgers (order {order_bad}, "
              f"future-settle {future_bad}, pnl-without-outcome "
              f"{orphan_pnl}); per-ledger rows/violations: "
              + ", ".join(per_ledger))
    return {"name": name, "method": method,
            "result": "FAIL" if bad else "PASS", "detail": detail}


def check_shuffled_labels(matched, treat_recs):
    name = "shuffled_label_placebo"
    method = ("permute settled outcomes across scored windows (seeded, "
              "5 permutations), rescore champion and t_regime with the "
              "SPRT's pairing semantics (stand-down = 0, paired same "
              "window). The null is NOT zero: a veto mechanically earns "
              "+cost-drag per skipped window under random labels, so "
              "each permuted paired mean is compared to its ANALYTIC "
              "expectation (skip windows x house edge at the actual "
              f"asks) and must sit within {PLACEBO_SD_LIMIT:.0f} SD of "
              "it — shuffling must create no edge beyond arithmetic")
    if not matched:
        return {"name": name, "method": method, "result": "FAIL",
                "detail": "no matchable windows — cannot certify"}
    # real paired mean from the logged evaluator output (same subset)
    tickers = {m["rec"]["ticker"] for m in matched}
    diffs = [(r["ev"].get("t_regime") or 0.0)
             - (r["ev"].get("champion") or 0.0)
             for r in treat_recs if r["ticker"] in tickers]
    real_delta = sum(diffs) / len(diffs)
    outcomes = [m["rec"]["outcome"] for m in matched]
    n = len(matched)
    q = sum(outcomes) / n        # label-shuffle marginal P(outcome=1)
    # Analytic expectation and SD of the permuted paired mean. Only
    # stand-down windows contribute (bet windows pair to exactly 0):
    # diff = -champ_ev, champ wins with prob q (yes side) or 1-q (no).
    # Independence approximation across windows — adequate for a 3 SD
    # canary band (true permutation dependence is weakly negative).
    exp_sum, var_sum, n_skip = 0.0, 0.0, 0
    per_window = []
    for m in matched:
        rec, t, row = m["rec"], m["trade"], m["row"]
        ask_yes, ask_no = modeled_asks(row["mkt_p_up"])
        ask = ask_yes if t["side"] == "yes" else ask_no
        stood_down = rec["ev"].get("t_regime") is None
        per_window.append((t["side"], ask, stood_down))
        if stood_down:
            n_skip += 1
            a = bet_ev(t["side"], ask, 1 if t["side"] == "yes" else 0)
            p_win = q if t["side"] == "yes" else 1 - q
            exp_sum += -(p_win * a - (1 - p_win))
            var_sum += p_win * (1 - p_win) * (a + 1) ** 2
    exp_delta = exp_sum / n
    sd_delta = math.sqrt(var_sum) / n
    rng = random.Random(SEED)
    perm_deltas = []
    for _ in range(5):
        perm = outcomes[:]
        rng.shuffle(perm)
        total = 0.0
        for (side, ask, stood_down), y in zip(per_window, perm):
            if stood_down:
                # t_regime scores 0, champion scores bet_ev => diff
                total += -bet_ev(side, ask, y)
        perm_deltas.append(total / n)
    excesses = [(d - exp_delta) / sd_delta if sd_delta > 0 else 0.0
                for d in perm_deltas]
    worst = max(abs(z) for z in excesses)
    rank = 1 + sum(1 for d in perm_deltas if abs(d) > abs(real_delta))
    ok = worst <= PLACEBO_SD_LIMIT
    detail = (f"real paired Δ {real_delta:+.4f}/$1 over {len(diffs)} "
              f"windows (rank {rank}/6 by |Δ| vs permutations); "
              f"analytic shuffled-label expectation {exp_delta:+.4f} "
              f"± {sd_delta:.4f} (cost drag of {n_skip} stand-downs); "
              f"permuted Δ "
              f"[{', '.join(f'{d:+.4f}' for d in perm_deltas)}], "
              f"max excess {worst:.2f} SD vs limit "
              f"{PLACEBO_SD_LIMIT:.0f} — "
              + ("no edge beyond the mechanical cost structure" if ok
                 else "edge BEYOND the cost structure fabricated from "
                      "shuffled labels"))
    return {"name": name, "method": method,
            "result": "PASS" if ok else "FAIL", "detail": detail}


def check_random_policy(kb_rows, treat_recs):
    name = "random_policy_placebo"
    method = ("seeded coin-flip side at the modeled decision-time asks "
              "across all settled decision windows; realized EV/$1 must "
              "sit at the analytic house edge (negative) within 3 SE, "
              "and must not be significantly positive — otherwise the "
              "scoring machinery itself is a money printer")
    rows = decision_rows(kb_rows)
    if len(rows) < MIN_WINDOWS:
        return {"name": name, "method": method, "result": "FAIL",
                "detail": f"only {len(rows)} settled windows"}
    rng = random.Random(SEED + 1)
    evs, exp_evs = [], []
    for r in rows:
        ask_yes, ask_no = modeled_asks(r["mkt_p_up"])
        y = r["actual"]
        ev_yes = bet_ev("yes", ask_yes, y)
        ev_no = bet_ev("no", ask_no, y)
        exp_evs.append(0.5 * ev_yes + 0.5 * ev_no)  # analytic, given y
        evs.append(ev_yes if rng.random() < 0.5 else ev_no)
    n = len(evs)
    mean = sum(evs) / n
    expected = sum(exp_evs) / n
    var = sum((e - mean) ** 2 for e in evs) / (n - 1)
    se = math.sqrt(var / n)
    champ = [r["ev"]["champion"] for r in treat_recs
             if r["ev"].get("champion") is not None]
    champ_mean = sum(champ) / len(champ) if champ else None
    dev = abs(mean - expected) / se if se > 0 else 0.0
    tpos = mean / se if se > 0 else 0.0
    ok = dev <= 3.0 and tpos < 2.0
    detail = (f"coin-flip EV {mean:+.4f}/$1 over {n} windows (SE "
              f"{se:.4f}); analytic house edge {expected:+.4f}/$1, "
              f"deviation {dev:.2f} SE (limit 3); t vs zero "
              f"{tpos:+.2f} (must be < 2); champion's own logged mean "
              + (f"{champ_mean:+.4f}/$1 on {len(champ)} windows"
                 if champ_mean is not None else "n/a"))
    return {"name": name, "method": method,
            "result": "PASS" if ok else "FAIL", "detail": detail}


def check_noise_features(kb_rows):
    name = "noise_feature_placebo"
    method = ("the check-1 walk-forward logistic refit with 5 seeded "
              "pure-noise N(0,1) features appended, over 5 noise seeds; "
              "mean accuracy gain over the legit-only fit must be "
              f"<= {100 * NOISE_MAX_GAIN:.0f}pp — noise that helps "
              "means the harness leaks")
    rows = decision_rows(kb_rows)
    if len(rows) < MIN_WINDOWS:
        return {"name": name, "method": method, "result": "FAIL",
                "detail": f"only {len(rows)} settled windows"}
    acc_legit, n_scored = walk_forward_acc(rows, legit_feats)
    gains = []
    for rep in range(5):
        rng = random.Random(SEED + 100 + rep)
        noise = {r["ticker"]: [rng.gauss(0.0, 1.0) for _ in range(5)]
                 for r in rows}
        acc, _ = walk_forward_acc(
            rows, lambda r: legit_feats(r) + noise[r["ticker"]])
        gains.append(acc - acc_legit)
    mean_gain = sum(gains) / len(gains)
    ok = mean_gain <= NOISE_MAX_GAIN
    detail = (f"legit acc {acc_legit:.3f} on {n_scored} scored windows; "
              f"per-seed gains "
              f"[{', '.join(f'{100 * g:+.1f}pp' for g in gains)}], mean "
              f"{100 * mean_gain:+.2f}pp vs limit "
              f"+{100 * NOISE_MAX_GAIN:.0f}pp")
    return {"name": name, "method": method,
            "result": "PASS" if ok else "FAIL", "detail": detail}


# ---------------------------------------------------------------------------

def main():
    kb = load_jsonl("kalshi_binary_log.jsonl")
    pt = load_jsonl("pt_trades.jsonl")
    treat_recs = load_jsonl("treatments.jsonl")
    ledger_names = ["pt_trades", "pt2_trades", "pt3_trades", "pt4_trades",
                    "pt5_trades", "pt6_trades", "pt7_trades", "pt8_trades",
                    "kb_bets", "kb_bets_sel", "pb_bets"]
    ledgers = {n: load_jsonl(n + ".jsonl") for n in ledger_names}
    ledgers = {n: r for n, r in ledgers.items() if r}
    data_now = max(
        [r["made_ts"] for r in kb]
        + [r["made_ts"] for rows in ledgers.values() for r in rows])

    matched, unmatchable = match_treatment_records(treat_recs, pt, kb)

    checks = [
        check_future_label(kb),
        check_quote_age(matched, unmatchable, len(treat_recs)),
        check_timestamp_order(ledgers, data_now),
        check_shuffled_labels(matched, treat_recs),
        check_random_policy(kb, treat_recs),
        check_noise_features(kb),
    ]
    overall = ("PASS" if all(c["result"] == "PASS" for c in checks)
               else "SEV-0")
    report = {
        "generated_ts": int(time.time()),
        "seed": SEED,
        "checks": checks,
        "overall": overall,
        "note": (
            "Canaries plant synthetic future information and verify the "
            "pipeline's contracts keep it out; placebos feed the real "
            "machinery pure noise and require it to find nothing. All "
            "checks are seeded and deterministic (no wall clock in any "
            "verdict; data_now is the latest decision timestamp in the "
            "data itself). Data contract honored by check 1: kb rows "
            "are appended pre-settle with actual=null and stamped in "
            "place at settlement (btc_rl/online.py ~L3430), so settled "
            "rows legitimately pair a decision-time mins_left with a "
            "filled outcome — the enforceable invariant is that no "
            "still-open window carries an outcome, and no non-"
            "settlement field of the decision-time schema correlates "
            ">0.95 with the outcome. Check 2 replays the treatment "
            "evaluator's row selection and EV arithmetic independently "
            "from raw logs. Placebo thresholds are pre-registered: "
            "each permuted-label paired mean within 3 SD of its "
            "analytic cost-drag expectation (the null is NOT zero — a "
            "veto mechanically earns +cost-drag per skipped window "
            "under random labels; this suite's first draft misread "
            "that arithmetic as fabricated alpha before the contract "
            "was understood), coin-flip EV within 3 SE of the analytic "
            "house edge and not significantly positive, noise features "
            "worth <= 1pp. "
            "Any FAIL here is SEV-0: the evaluation infrastructure, "
            "not the models, is broken."),
    }
    tmp = OUT_PATH + ".tmp"
    with open(tmp, "w") as f:
        json.dump(report, f, indent=2)
    os.replace(tmp, OUT_PATH)

    print(f"leakage canaries + placebos — {len(checks)} checks")
    for c in checks:
        print(f"  [{c['result']:4s}] {c['name']}")
        print(f"         {c['detail']}")
    print(f"overall: {overall}  ->  {OUT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
