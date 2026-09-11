"""The ONE offline/walk-forward evaluation engine (M2.3, contract
§16-18). Every candidate model — present or future — is judged by
this engine and nothing else; outputs conform to config/METRICS.yaml
and never invent parallel formulas.

Evaluation modes
----------------
This v1 evaluates PREDICTION SERIES: one decision-time observation
per settled window (the same convention as model_lifecycle). For the
current prequential arms the series is what the model actually
emitted live, evaluated in chronological folds — honest walk-forward
of the prequential stream. When artifact-based candidates arrive
(retrained offline models), the same fold/metric/battery code runs on
their replayed predictions; train ranges then populate.

Falsification battery (§18) — BLOCKING
--------------------------------------
timestamp ordering · duplicate windows · outcome-leak canary ·
label shuffle · prediction shuffle · random-strategy economics ·
market + persistence baselines. Any hard check failing =>
falsification_status EVAL_INVALID (never a warning). PIT feature
reconstruction is reported NOT_AVAILABLE (no feature snapshot store
yet) — a declared gap, not a silent pass; it becomes mandatory for
NATIVE candidates.

Summary discipline (§17): median fold, worst fold, fold variance,
fraction of folds beating market. One spectacular fold qualifies
nothing.
"""
from __future__ import annotations

import json
import math
import random
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"

ENGINE_VERSION = "1.0.0"
SEED = 20260830
EPS = 1e-6
N_FOLDS = 5
SEL_TAUS = (0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90)
PRODUCT_TAU = 0.75           # frozen economic-replay entry gate
MIN_FOLD_N = 10


def _logit(p):
    p = min(1 - EPS, max(EPS, p))
    return math.log(p / (1 - p))


def _sigmoid(z):
    return 1 / (1 + math.exp(-max(-30, min(30, z))))


def _brier(ps, ys):
    return sum((p - y) ** 2 for p, y in zip(ps, ys)) / len(ps)


def _logloss(ps, ys):
    s = 0.0
    for p, y in zip(ps, ys):
        pc = min(1 - EPS, max(EPS, p))
        s += -(y * math.log(pc) + (1 - y) * math.log(1 - pc))
    return s / len(ps)


def _calibration_fit(ps, ys):
    """IRLS logistic fit y ~ a + b*logit(p). (b=1, a=0 ideal)."""
    a, b = 0.0, 1.0
    xs = [_logit(p) for p in ps]
    for _ in range(25):
        g_a = g_b = h_aa = h_ab = h_bb = 0.0
        for x, y in zip(xs, ys):
            mu = _sigmoid(a + b * x)
            w = max(mu * (1 - mu), 1e-9)
            g_a += (y - mu)
            g_b += (y - mu) * x
            h_aa += w
            h_ab += w * x
            h_bb += w * x * x
        det = h_aa * h_bb - h_ab * h_ab
        if abs(det) < 1e-12:
            break
        da = (g_a * h_bb - g_b * h_ab) / det
        db = (g_b * h_aa - g_a * h_ab) / det
        a, b = a + da, b + db
        if abs(da) + abs(db) < 1e-8:
            break
    return round(b, 3), round(a, 3)


def _ece(ps, ys, bins=10):
    order = sorted(range(len(ps)), key=lambda i: ps[i])
    n = len(ps)
    tot = 0.0
    for k in range(bins):
        idx = order[k * n // bins:(k + 1) * n // bins]
        if not idx:
            continue
        conf = sum(ps[i] for i in idx) / len(idx)
        acc = sum(ys[i] for i in idx) / len(idx)
        tot += len(idx) / n * abs(acc - conf)
    return round(tot, 4)


def _auc(ps, ys):
    """Rank AUC (probability a random positive outranks a random
    negative); ties get half credit. None if one class absent."""
    pos = [p for p, y in zip(ps, ys) if y]
    neg = [p for p, y in zip(ps, ys) if not y]
    if not pos or not neg:
        return None
    wins = 0.0
    for a in pos:
        for b in neg:
            wins += 1.0 if a > b else 0.5 if a == b else 0.0
    return round(wins / (len(pos) * len(neg)), 4)


def _fee_frac(ask_c):
    return 7 * (ask_c / 100.0) * (1 - ask_c / 100.0)


def _econ_replay(rows, side_fn, tau=PRODUCT_TAU):
    """Frozen taker convention: qty 1 at the logged decision-time
    ask; enter when claimed confidence >= tau. Per METRICS.yaml
    pnl_per_contract. Returns per-eligible economics + drawdown."""
    pnls, fills = [], 0
    for r in rows:
        side_up = side_fn(r)
        conf = max(r["p"], 1 - r["p"]) if side_up is None else (
            r["p"] if side_up else 1 - r["p"])
        if side_up is None or conf < tau or r.get("ask_c") is None:
            pnls.append(0.0)
            continue
        fills += 1
        ask = r["ask_c"]
        cost = ask + 100 * _fee_frac(ask)
        won = bool(r["y"]) == bool(side_up)
        pnls.append((100 - cost) / cost if won else -1.0)
    cum = peak = dd = 0.0
    for v in pnls:
        cum += v
        peak = max(peak, cum)
        dd = min(dd, cum - peak)
    n = len(rows)
    return {"ev_per_eligible": round(sum(pnls) / n, 4) if n else None,
            "pnl_per_fill": round(sum(pnls) / fills, 4) if fills
            else None,
            "fills": fills, "coverage": round(fills / n, 4) if n
            else None,
            "max_drawdown": round(dd, 4)}


def _fold_metrics(rows):
    ps = [r["p"] for r in rows]
    ys = [r["y"] for r in rows]
    mk = [(r["market_p"], r["y"]) for r in rows
          if r.get("market_p") is not None]
    brier = _brier(ps, ys)
    out = {"n": len(rows),
           "brier": round(brier, 4),
           "log_loss": round(_logloss(ps, ys), 4),
           "ece": _ece(ps, ys)}
    slope, intercept = _calibration_fit(ps, ys)
    out["calibration_slope"] = slope
    out["calibration_intercept"] = intercept
    if mk:
        mb = _brier([m for m, _ in mk], [y for _, y in mk])
        out["market_brier"] = round(mb, 4)
        out["bss"] = round(1 - brier / mb, 4) if mb > 0 else None
        out["log_loss_market"] = round(
            _logloss([m for m, _ in mk], [y for _, y in mk]), 4)
    sel = {}
    for tau in SEL_TAUS:
        picked = [(p, y) for p, y in zip(ps, ys)
                  if max(p, 1 - p) >= tau]
        key = f"{int(tau * 100)}"
        sel[key] = {"coverage": round(len(picked) / len(rows), 4),
                    "n": len(picked)}
        if picked:
            sel[key]["accuracy"] = round(sum(
                1 for p, y in picked if (p >= .5) == bool(y))
                / len(picked), 4)
            sel[key]["brier"] = round(_brier(
                [p for p, _ in picked], [y for _, y in picked]), 4)
    out["selective"] = sel
    out["economics"] = _econ_replay(rows, lambda r: r["p"] >= 0.5)
    return out


def _battery(rows, model_id):
    rng = random.Random(SEED)
    checks = {}
    # 1. timestamp ordering + decision envelope
    bad_ts = [r for r in rows if not (r["ts"] < r["close_ts"])]
    checks["timestamp_ordering"] = {
        "result": "PASS" if not bad_ts else "FAIL",
        "bad_rows": len(bad_ts)}
    # 2. duplicate windows (one observation per window law)
    tks = [r["ticker"] for r in rows]
    dupes = len(tks) - len(set(tks))
    checks["duplicate_windows"] = {
        "result": "PASS" if dupes == 0 else "FAIL", "dupes": dupes}
    # 3. outcome-leak canary: a "prediction" that IS the label
    exact = sum(1 for r in rows if r["p"] in (0.0, 1.0)
                and int(r["p"]) == r["y"])
    checks["outcome_leak_canary"] = {
        "result": "PASS" if exact / max(1, len(rows)) < 0.01
        else "FAIL", "exact_label_predictions": exact}
    # 4. label shuffle: ASSOCIATION with random labels must vanish.
    # Deliberately rank-based (AUC), not Brier-vs-market: a hedged
    # predictor mechanically beats a confident baseline on shuffled
    # labels (lower variance), which is a distributional artifact —
    # the first engine run flagged kbf exactly this way (shakeout
    # 08-30) and the Brier form was replaced by AUC
    ys = [r["y"] for r in rows]
    ps = [r["p"] for r in rows]
    sh = ys[:]
    rng.shuffle(sh)
    auc_sh = _auc(ps, sh)
    checks["label_shuffle"] = {
        "result": "PASS" if auc_sh is None or 0.40 <= auc_sh <= 0.60
        else "FAIL",
        "auc_vs_shuffled_labels": auc_sh,
        "expect": "AUC in [0.40, 0.60] — no association with random "
                  "labels"}
    # 5. prediction shuffle (feature-shuffle analog for series mode):
    # shuffled predictions must carry no association with TRUE labels
    psh = ps[:]
    rng.shuffle(psh)
    auc_psh = _auc(psh, ys)
    checks["prediction_shuffle"] = {
        "result": "PASS" if auc_psh is None
        or 0.40 <= auc_psh <= 0.60 else "FAIL",
        "auc_shuffled_predictions_vs_labels": auc_psh}
    # 6. random strategy economics (should bleed ~fees)
    rand_econ = _econ_replay(rows,
                             lambda r: rng.random() >= 0.5, tau=0.0)
    checks["random_strategy"] = {
        "result": "PASS" if (rand_econ["ev_per_eligible"] or 0) < 0.05
        else "FAIL", **rand_econ}
    # 7. baselines (reported, not pass/fail)
    mkb = _fold_metrics([{**r, "p": r["market_p"]} for r in rows
                         if r.get("market_p") is not None]) \
        if any(r.get("market_p") is not None for r in rows) else None
    prev_y = None
    pers_rows = []
    for r in rows:
        if prev_y is not None:
            pers_rows.append({**r, "p": 0.999 if prev_y else 0.001})
        prev_y = r["y"]
    checks["baselines"] = {
        "market_brier": mkb["brier"] if mkb else None,
        "persistence_brier": round(_brier(
            [r["p"] for r in pers_rows],
            [r["y"] for r in pers_rows]), 4) if pers_rows else None}
    # 8. PIT reconstruction — declared gap, never a silent pass
    checks["pit_reconstruction"] = {
        "result": "NOT_AVAILABLE",
        "note": "no feature-snapshot store yet; mandatory for NATIVE "
                "candidates, declared gap for prequential series"}
    hard = ["timestamp_ordering", "duplicate_windows",
            "outcome_leak_canary", "label_shuffle",
            "prediction_shuffle", "random_strategy"]
    invalid = [k for k in hard if checks[k]["result"] == "FAIL"]
    return ("EVAL_INVALID" if invalid else "PASS"), invalid, checks


def evaluate_series(rows, model_id, model_version="prequential"):
    """rows: chronological [{ts, close_ts, ticker, p, y, market_p,
    ask_c}]. Returns the full §16-17 evaluation object."""
    rows = sorted(rows, key=lambda r: r["close_ts"])
    status, invalid, checks = _battery(rows, model_id)
    folds = []
    n = len(rows)
    for k in range(N_FOLDS):
        seg = rows[k * n // N_FOLDS:(k + 1) * n // N_FOLDS]
        if len(seg) < MIN_FOLD_N:
            continue
        fm = _fold_metrics(seg)
        folds.append({"fold_id": k,
                      "test_start": seg[0]["close_ts"],
                      "test_end": seg[-1]["close_ts"],
                      "train_start": None, "train_end": None,
                      "train_n": None,
                      "note": "prequential stream — the model trained "
                              "only on data before each prediction by "
                              "construction",
                      **fm})
    bsss = [f["bss"] for f in folds if f.get("bss") is not None]
    summary = {}
    if bsss:
        srt = sorted(bsss)
        mean = sum(bsss) / len(bsss)
        summary = {
            "folds": len(folds),
            "median_fold_bss": round(srt[len(srt) // 2], 4),
            "worst_fold_bss": round(srt[0], 4),
            "best_fold_bss": round(srt[-1], 4),
            "fold_bss_variance": round(sum(
                (b - mean) ** 2 for b in bsss) / len(bsss), 6),
            "fraction_folds_beating_market": round(sum(
                1 for b in bsss if b > 0) / len(bsss), 3),
        }
    lifetime = _fold_metrics(rows) if n >= MIN_FOLD_N else {"n": n}
    return {"model_id": model_id, "model_version": model_version,
            "dataset_version": "kalshi_binary_log (retained tail)",
            "feature_version": "live-logged (no snapshot store)",
            "n": n,
            "falsification_status": status,
            "falsification_failed": invalid,
            "falsification_checks": checks,
            "folds": folds, "fold_summary": summary,
            "lifetime": lifetime}


def decision_rows_kb(kb, variant):
    by_tk = {}
    for r in kb:
        if (r.get("variant") or "kb") != variant:
            continue
        if r.get("actual") is None or r.get("p_up") is None:
            continue
        ml = r.get("mins_left")
        if ml is None or ml > 12:
            continue
        cur = by_tk.get(r["ticker"])
        if cur is None or ml > (cur.get("mins_left") or 0):
            by_tk[r["ticker"]] = r
    return [{"ts": r["made_ts"], "close_ts": r["close_ts"],
             "ticker": r["ticker"], "p": r["p_up"],
             "y": int(r["actual"]), "market_p": r.get("mkt_p_up"),
             "ask_c": r.get("ask_c")}
            for r in sorted(by_tk.values(),
                            key=lambda x: x["close_ts"])]


def main():
    now = int(time.time())
    kb = []
    p = RES / "kalshi_binary_log.jsonl"
    if p.exists():
        for l in p.open():
            if l.strip():
                try:
                    kb.append(json.loads(l))
                except Exception:
                    pass
    variants = sorted({r.get("variant") or "kb" for r in kb})
    models = {}
    for v in variants:
        rows = decision_rows_kb(kb, v)
        if len(rows) >= MIN_FOLD_N:
            models[v] = evaluate_series(rows, v)
    doc = {"engine_version": ENGINE_VERSION, "generated_ts": now,
           "seed": SEED,
           "scope": "PREQUENTIAL series, chronological folds — "
                    "forward-honest by construction; retrospective "
                    "artifact candidates use the same engine with "
                    "train ranges populated",
           "metric_contract": "config/METRICS.yaml v1.1.0",
           "models": models}
    (RES / "model_offline.json").write_text(json.dumps(doc, indent=1))
    # model_qualification.json — ONE verdict object per candidate
    # (PM 08-30): backend code never interprets 20 fold metrics
    # independently. Verdict ladder:
    #   INVALID           falsification battery failed
    #   OFFLINE_FAIL      battery passed, median-fold BSS <= 0
    #   SHADOW_ELIGIBLE   median BSS > 0 AND worst fold > -0.05 AND
    #                     >= 60% folds beat market
    #   FORWARD_CANDIDATE shadow-eligible AND pit PASS AND parity PASS
    #                     (both currently NOT_AVAILABLE -> capped)
    qual = {}
    par = {}
    try:
        pj = json.loads((RES / "parity.json").read_text())
        for v, st in (pj.get("by_model") or {}).items():
            par[v] = ("FAIL" if st.get("fail")
                      else "PASS" if st.get("pass")
                      else "DEFERRED" if st.get("deferred")
                      else "NOT_AVAILABLE")
    except Exception:
        pass
    for v, m in models.items():
        s = m["fold_summary"]
        life = m["lifetime"]
        if m["falsification_status"] != "PASS":
            verdict = "INVALID"
        elif not s or (s.get("median_fold_bss") or 0) <= 0:
            verdict = "OFFLINE_FAIL"
        elif (s.get("worst_fold_bss") or -1) > -0.05 \
                and (s.get("fraction_folds_beating_market") or 0) >= 0.6:
            verdict = "SHADOW_ELIGIBLE"
        else:
            verdict = "OFFLINE_FAIL"
        qual[v] = {
            "model_id": v, "version": m["model_version"],
            "walk_forward": {k: s.get(k) for k in
                             ("median_fold_bss", "worst_fold_bss",
                              "fold_bss_variance",
                              "fraction_folds_beating_market")},
            "calibration": {"ece": life.get("ece"),
                            "slope": life.get("calibration_slope"),
                            "intercept":
                                life.get("calibration_intercept")},
            "selective_75": (life.get("selective") or {}).get("75"),
            "economics": life.get("economics"),
            "falsification": m["falsification_status"],
            "pit": "AVAILABLE" if v in par else "NOT_AVAILABLE",
            "parity": par.get(v, "NOT_AVAILABLE"),
            "verdict": "INVALID" if par.get(v) == "FAIL" else verdict,
            "verdict_note": "FORWARD_CANDIDATE additionally requires "
                            "PIT + parity PASS — both blocked on the "
                            "feature-snapshot store (M2.5)",
        }
    (RES / "model_qualification.json").write_text(json.dumps(
        {"generated_ts": now, "engine_version": ENGINE_VERSION,
         "models": qual}, indent=1))
    for v, m in models.items():
        s = m["fold_summary"]
        print(f"{v}: {m['falsification_status']} · n={m['n']} · "
              f"median BSS {s.get('median_fold_bss')} · worst "
              f"{s.get('worst_fold_bss')} · beats-mkt "
              f"{s.get('fraction_folds_beating_market')}")


if __name__ == "__main__":
    main()
