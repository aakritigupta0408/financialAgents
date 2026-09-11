"""M5.4 — Model Researcher (script-backed structured agent).

First mission (PM 08-30): EXPLAIN why 0/10 arms show positive
median-fold BSS vs market — not find a better model. Emits the
canonical results/model_research.json and reports through the
decision firewall (OBSERVE/DIAGNOSE/PROPOSE only).

Hard rules encoded here and enforced by the model-researcher-
discipline invariant:
  * candidate_question.count <= 1
  * recommended_action in {NO_ACTION, NEEDS_MORE_DATA,
    ONE_CANDIDATE_QUESTION, RETIRE_MODEL_FAMILY}
  * calibration can repair probabilities but cannot create
    resolution — stated, always
  * no slice becomes a candidate without n >= 50 and |t| >= 3
    (crude multiplicity guard over the ~24 slices examined)
  * feature ablations / capacity: NOT_AVAILABLE until the artifact-
    mode evaluator exists — declared, never guessed
"""
from __future__ import annotations

import json
import math
import time
from pathlib import Path

import agent_firewall as fw

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"

MIN_SLICE_N = 50
T_GATE = 3.0
TIME_BUCKETS = (("T-13", 11, 99), ("T-10", 9, 11), ("T-8", 7, 9),
                ("T-6", 5, 7), ("T-4", 3, 5), ("T-2", 0, 3))
SERVING = ("kb2", "kb9")


def jl(name):
    p = RES / name
    out = []
    if p.exists():
        for l in p.open():
            if l.strip():
                try:
                    out.append(json.loads(l))
                except Exception:
                    pass
    return out


def j(name):
    try:
        return json.loads((RES / name).read_text())
    except Exception:
        return None


def murphy(ps, ys, bins=10):
    """Brier = reliability - resolution + uncertainty."""
    n = len(ps)
    ybar = sum(ys) / n
    unc = ybar * (1 - ybar)
    order = sorted(range(n), key=lambda i: ps[i])
    rel = res = 0.0
    for k in range(bins):
        idx = order[k * n // bins:(k + 1) * n // bins]
        if not idx:
            continue
        pk = sum(ps[i] for i in idx) / len(idx)
        yk = sum(ys[i] for i in idx) / len(idx)
        rel += len(idx) / n * (pk - yk) ** 2
        res += len(idx) / n * (yk - ybar) ** 2
    return {"reliability": round(rel, 5), "resolution": round(res, 5),
            "uncertainty": round(unc, 5),
            "read": ("useful resolution, fixable reliability"
                     if res > 0.02 and rel > 0.01 else
                     "no meaningful resolution — calibration cannot "
                     "create information" if res <= 0.02 else
                     "well-calibrated, resolution-limited")}


def corr(a, b):
    n = len(a)
    ma, mb = sum(a) / n, sum(b) / n
    num = sum((x - ma) * (y - mb) for x, y in zip(a, b))
    da = math.sqrt(sum((x - ma) ** 2 for x in a))
    db = math.sqrt(sum((y - mb) ** 2 for y in b))
    return num / (da * db) if da > 0 and db > 0 else None


def resid_test(rows):
    """Does disagreement predict market error?
    stat = mean((y - mkt) * sign(p - mkt)); t = mean/SE."""
    vals = [(r["y"] - r["mkt"]) * (1 if r["p"] > r["mkt"] else -1)
            for r in rows if abs(r["p"] - r["mkt"]) > 1e-9]
    n = len(vals)
    if n < 2:
        return {"n": n}
    m = sum(vals) / n
    sd = math.sqrt(sum((v - m) ** 2 for v in vals) / (n - 1))
    se = sd / math.sqrt(n)
    return {"n": n, "mean_signed_market_error": round(m, 4),
            "se": round(se, 4),
            "t": round(m / se, 2) if se > 0 else None}


def main():
    now = int(time.time())
    age = fw.stale("kalshi_binary_log.jsonl", 3600)
    if age is not None:
        fw.submit(agent="model_researcher", action_class="OBSERVE",
                  finding=f"STALE_INPUT — kalshi_binary_log is "
                          f"{age:.0f}s old; refusing to diagnose "
                          "from old state",
                  recommendation="no research recommendation until "
                                 "the tape refreshes",
                  evidence=["kalshi_binary_log.jsonl mtime"])
        print("model_researcher: STALE_INPUT — no diagnosis")
        return
    kb = jl("kalshi_binary_log.jsonl")
    offline = (j("model_offline.json") or {}).get("models") or {}
    qual = (j("model_qualification.json") or {}).get("models") or {}
    n_fail = sum(1 for m in qual.values()
                 if m.get("verdict") in ("OFFLINE_FAIL", "INVALID"))
    baseline = f"{n_fail}/{len(qual)} arms fail offline qualification"

    # settled minute-rows per variant with market prob
    by_var = {}
    for r in kb:
        if r.get("actual") is None or r.get("mkt_p_up") is None \
                or r.get("p_up") is None or r.get("mins_left") is None:
            continue
        by_var.setdefault(r.get("variant") or "kb", []).append(
            {"p": r["p_up"], "mkt": r["mkt_p_up"],
             "y": int(r["actual"]), "ml": r["mins_left"],
             "tk": r["ticker"]})

    models, slice_candidates = {}, []
    for v in SERVING:
        rows = by_var.get(v) or []
        if len(rows) < MIN_SLICE_N:
            models[v] = {"n": len(rows), "state": "INSUFFICIENT_N"}
            continue
        ps = [r["p"] for r in rows]
        ys = [r["y"] for r in rows]
        mks = [r["mkt"] for r in rows]
        # redundancy with market
        c = corr(ps, mks)
        redund = {"corr_model_market": round(c, 4) if c is not None
                  else None,
                  "r2": round(c * c, 4) if c is not None else None,
                  "residual_variance": round(sum(
                      (p - m) ** 2 for p, m in zip(ps, mks))
                      / len(ps), 5),
                  "read": "HIGH MARKET REDUNDANCY"
                  if c is not None and c > 0.9 else
                  "partially independent signal"}
        # timing: BSS by time bucket
        timing = {}
        early_skill = late_skill = False
        for name, lo, hi in TIME_BUCKETS:
            seg = [r for r in rows if lo < r["ml"] <= hi]
            if len(seg) < MIN_SLICE_N:
                timing[name] = {"n": len(seg), "state": "MUTED"}
                continue
            b = sum((r["p"] - r["y"]) ** 2 for r in seg) / len(seg)
            mb = sum((r["mkt"] - r["y"]) ** 2 for r in seg) / len(seg)
            bss = 1 - b / mb if mb > 0 else None
            timing[name] = {"n": len(seg), "bss": round(bss, 4)}
            if bss is not None and bss > 0:
                if name in ("T-13", "T-10"):
                    early_skill = True
                if name in ("T-4", "T-2"):
                    late_skill = True
        timing_class = ("EARLY_SKILL" if early_skill else
                        "LATE_SKILL" if late_skill else "NO_SKILL")
        # disagreement analysis at 5/10/20pp, overall + early-window
        disagreement = {}
        for pp in (5, 10, 20):
            for scope, pred in (("all", lambda r: True),
                                ("early", lambda r: r["ml"] > 9)):
                seg = [r for r in rows
                       if abs(r["p"] - r["mkt"]) >= pp / 100
                       and pred(r)]
                key = f"ge{pp}pp_{scope}"
                if len(seg) < MIN_SLICE_N:
                    disagreement[key] = {"n": len(seg),
                                         "state": "MUTED"}
                    continue
                t = resid_test(seg)
                disagreement[key] = t
                if t.get("t") is not None and t["t"] >= T_GATE \
                        and t["n"] >= MIN_SLICE_N:
                    slice_candidates.append(
                        {"model": v, "slice": key, **t})
        off = offline.get(v) or {}
        life = off.get("lifetime") or {}
        cal = {"ece": life.get("ece"),
               "slope": life.get("calibration_slope"),
               "intercept": life.get("calibration_intercept"),
               "law": "CALIBRATION CAN REPAIR PROBABILITIES BUT "
                      "CANNOT CREATE RESOLUTION"}
        models[v] = {"n": len(rows),
                     "murphy": murphy(ps, ys),
                     "market_redundancy": redund,
                     "timing": timing,
                     "timing_class": timing_class,
                     "disagreement": disagreement,
                     "calibration": cal,
                     "median_fold_bss":
                         (off.get("fold_summary") or {})
                         .get("median_fold_bss")}

    # candidate selection: at most ONE, only past the gates
    slice_candidates.sort(key=lambda s: -(s.get("t") or 0))
    cand = slice_candidates[0] if slice_candidates else None
    if cand:
        question = (f"Does {cand['model']} disagreement "
                    f"({cand['slice']}) predict market error "
                    "prospectively? (existing A2-style hypothesis — "
                    "assess, not proof)")
        action = "ONE_CANDIDATE_QUESTION"
    else:
        question = None
        action = "NEEDS_MORE_DATA"

    strongest = []
    for v, m in models.items():
        if not isinstance(m.get("murphy"), dict):
            continue
        strongest.append(
            f"{v}: {m['murphy']['read']}; redundancy "
            f"{m['market_redundancy']['read']}; timing "
            f"{m['timing_class']}")
    doc = {
        "generated_ts": now,
        "state": "DIAGNOSIS_COMPLETE",
        "baseline_finding": "NO_MODEL_BEATS_MARKET — " + baseline,
        "failure_decomposition": {
            "calibration": {v: m.get("calibration")
                            for v, m in models.items()},
            "resolution": {v: (m.get("murphy") or {})
                           for v, m in models.items()},
            "timing": {v: {"class": m.get("timing_class"),
                           "by_bucket": m.get("timing")}
                       for v, m in models.items()},
            "regime": {"note": "disagreement slices below double as "
                       "the regime read; deeper slicing awaits more "
                       "independent windows"},
            "market_redundancy": {v: m.get("market_redundancy")
                                  for v, m in models.items()},
            "feature_information": {
                "status": "NOT_AVAILABLE",
                "note": "family ablations require the artifact-mode "
                        "offline evaluator (retrain-and-replay); "
                        "F-MICRO/F-XVENUE arrive as CAPTURE_ONLY "
                        "hypotheses first"},
            "capacity": {
                "status": "UNKNOWN",
                "note": "no evidence isolates model class as the "
                        "bottleneck; capacity claims are inadmissible "
                        "until information-level failures are "
                        "excluded"},
        },
        "models": models,
        "strongest_supported_explanation": "; ".join(strongest)
        or "insufficient data",
        "slice_discipline": {"min_n": MIN_SLICE_N, "t_gate": T_GATE,
                             "slices_examined":
                                 len(SERVING) * 6 * 2 + len(SERVING)
                                 * 6,
                             "passing": len(slice_candidates)},
        "candidate_question": {
            "count": 1 if cand else 0,
            "question": question,
            "targeted_loss_term": "MODEL_RESOLUTION" if cand
            else None,
            "evidence": cand,
        },
        "recommended_action": action,
        "confidence": "diagnosis-grade; nothing here authorizes a "
                      "model change",
        "prohibited_interpretations": [
            "BSS<0 does not imply 'need a bigger network' — capacity "
            "is UNKNOWN, not implicated",
            "calibration improvement alone is not candidacy (the M1 "
            "lesson, kept)",
            "no slice below n>=50 & |t|>=3 may become a hypothesis",
        ],
    }
    (RES / "model_research.json").write_text(json.dumps(doc, indent=1))
    # information_timing.json — the canonical "was the information
    # still tradable when it existed?" artifact (PM 08-30). Lead/lag
    # fields await the synchronized xvenue tape past Gate F1 —
    # declared, never guessed.
    it = {"generated_ts": now,
          "models": {v: {"by_horizon": m.get("timing"),
                         "class": m.get("timing_class"),
                         "disagreement": m.get("disagreement")}
                     for v, m in models.items()
                     if isinstance(m.get("timing"), dict)},
          "lead_lag": {"market_move_started_at": "NOT_AVAILABLE",
                       "kalshi_response_latency": "NOT_AVAILABLE",
                       "economic_opportunity_remaining":
                           "NOT_AVAILABLE",
                       "blocked_by": "synchronized F-XVENUE tape "
                       "(Gate F1, earliest 2026-09-06)"},
          "provenance": ["kalshi_binary_log.jsonl",
                         "model_offline.json"]}
    (RES / "information_timing.json").write_text(
        json.dumps(it, indent=1))

    fw.submit(agent="model_researcher", action_class="DIAGNOSE",
              finding=doc["baseline_finding"] + " — " +
              doc["strongest_supported_explanation"][:300],
              recommendation=action,
              evidence=["model_research.json", "model_offline.json",
                        "model_qualification.json"],
              n=sum(m.get("n", 0) for m in models.values()))
    if cand:
        fw.submit(agent="model_researcher", action_class="PROPOSE",
                  finding=f"disagreement slice {cand['slice']} on "
                          f"{cand['model']}: t={cand['t']}, "
                          f"n={cand['n']}",
                  recommendation=question +
                  " | control: market probability | treatment "
                  "concept: residual-informed shadow caller, "
                  "prequentially registered",
                  evidence=["model_research.json"],
                  targeted_loss_term="MODEL_RESOLUTION",
                  risk="slice multiplicity — one candidate max, "
                       "prospective registration required",
                  n=cand["n"], effect=cand.get(
                      "mean_signed_market_error"))
    print(f"model_researcher: {action} · candidates "
          f"{doc['candidate_question']['count']} · "
          + "; ".join(f"{v}:{m.get('timing_class')}"
                      for v, m in models.items()))


if __name__ == "__main__":
    main()
