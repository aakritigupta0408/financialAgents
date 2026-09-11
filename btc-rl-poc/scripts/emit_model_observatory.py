"""Emit the M2 Model Observatory artifacts (master contract §13-15,38).

Two outputs, both derived ONLY from append-only sources:

results/training_runs.jsonl   — immutable training-run registry.
    One row per (retrain ts, arm, horizon) candidate, derived from
    metrics_history.jsonl `kind=retrain` records. Idempotent append:
    rows already present are never rewritten (immutability §15); only
    runs newer than the last registered ts are appended.

results/model_lifecycle.json  — per serving model, four eras:
    OFFLINE / LAUNCH / CURRENT / LIFETIME (§13, §38), with explicit
    observation windows (LIFETIME, LAST_100, LAST_50, LAST_25), CI-free
    raw values + n per window, and a trend classification
    IMPROVING / STABLE / DEGRADING / UNKNOWN that refuses to speak
    below the compare gate (n>=25 both segments). UNKNOWN is a
    first-class state — never inferred from tiny N (§14, §40).

Scoring convention: one observation per (variant, window) = the
DECISION-TIME row (earliest settled row inside the <=12-minute entry
envelope), the same convention the treatment evaluator uses — scoring
the last row before close would grade the market's trivial endgame
knowledge, not the model.
"""
from __future__ import annotations

import json
import math
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"

WINDOWS = (("LIFETIME", None), ("LAST_100", 100), ("LAST_50", 50),
           ("LAST_25", 25))
MIN_N = {"display": 10, "compare": 25, "gate": 50}
LAUNCH_N = 25                     # first N settled windows = LAUNCH era
SEL_TAU = 0.75                    # selective threshold (product gate)
EPS = 1e-6
# The two serving T2 roles (Great Simplification §16 / lean machine):
SERVING_T2 = {"kb2": "CONTROL (market-anchored)",
              "kb9": "CHALLENGER (TimesFM fusion)"}


def rows(name):
    p = RES / name
    if not p.exists():
        return []
    out = []
    for l in p.open():
        if l.strip():
            try:
                out.append(json.loads(l))
            except Exception:
                pass
    return out


def decision_rows(kb, variant):
    """Per window: earliest settled row inside the decision envelope."""
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
    return sorted(by_tk.values(), key=lambda r: r["close_ts"])


def _metrics(seg):
    """Metric block for a list of decision rows (schema-stable)."""
    n = len(seg)
    if n == 0:
        return {"n": 0}
    briers, lls, mb, mll, sel_ok, sel_n = [], [], [], [], 0, 0
    for r in seg:
        p, y = r["p_up"], r["actual"]
        briers.append((p - y) ** 2)
        pc = min(1 - EPS, max(EPS, p))
        lls.append(-(y * math.log(pc) + (1 - y) * math.log(1 - pc)))
        m = r.get("mkt_p_up")
        if m is not None:
            mb.append((m - y) ** 2)
            mc = min(1 - EPS, max(EPS, m))
            mll.append(-(y * math.log(mc) + (1 - y) * math.log(1 - mc)))
        conf = max(p, 1 - p)
        if conf >= SEL_TAU:
            sel_n += 1
            if (p >= 0.5) == bool(y):
                sel_ok += 1
    brier = sum(briers) / n
    out = {"n": n,
           "brier": round(brier, 4),
           "log_loss": round(sum(lls) / n, 4),
           "selective_acc_75": round(sel_ok / sel_n, 4) if sel_n
           else None,
           "selective_coverage_75": round(sel_n / n, 4),
           "selective_n_75": sel_n}
    if mb:
        mkt_brier = sum(mb) / len(mb)
        out["market_brier"] = round(mkt_brier, 4)
        out["bss_vs_market"] = round(1 - brier / mkt_brier, 4) \
            if mkt_brier > 0 else None
        out["market_n"] = len(mb)
    return out


def _trend(drows):
    """LAST_25 vs the 75 windows before them, Brier-based. Refuses to
    classify below the compare gate on either segment."""
    if len(drows) < MIN_N["compare"] * 2:
        return {"state": "UNKNOWN",
                "why": f"n={len(drows)} < {MIN_N['compare'] * 2} — "
                       "insufficient for a segment comparison"}
    recent = _metrics(drows[-25:])
    prior = _metrics(drows[-100:-25])
    if prior["n"] < MIN_N["compare"]:
        return {"state": "UNKNOWN", "why": "prior segment underpowered"}
    d = recent["brier"] - prior["brier"]
    state = "STABLE" if abs(d) < 0.01 else \
        ("IMPROVING" if d < 0 else "DEGRADING")
    return {"state": state,
            "brier_recent_25": recent["brier"],
            "brier_prior_75": prior["brier"],
            "delta": round(d, 4),
            "why": f"Brier last-25 {recent['brier']} vs prior-75 "
                   f"{prior['brier']} (Δ {d:+.4f}, band ±0.01)"}


def emit_training_runs(now):
    """Idempotent append of retrain candidates → training_runs.jsonl."""
    hist = rows("metrics_history.jsonl")
    out_p = RES / "training_runs.jsonl"
    existing = rows("training_runs.jsonl")
    seen = {r["training_run_id"] for r in existing}
    new = []
    for h in hist:
        if h.get("kind") != "retrain" or not h.get("gate"):
            continue
        for arm, hs in h["gate"].items():
            for horizon, g in (hs or {}).items():
                rid = f"{h['ts']}-{arm}-{horizon}"
                if rid in seen or not isinstance(g, dict):
                    continue
                before = g.get("val_mse_before")
                after = g.get("val_mse_after")
                new.append({
                    "training_run_id": rid,
                    "model_id": arm, "horizon": horizon,
                    "run_ts": h["ts"], "git": h.get("git"),
                    "incumbent_metrics": {"val_mse": before},
                    "candidate_metrics": {"val_mse": after},
                    "improvement": round(before - after, 4)
                    if before is not None and after is not None
                    else None,
                    "promotion_decision":
                        "REVERTED" if g.get("reverted") else "KEPT",
                    "decision_reason":
                        "candidate val MSE vs incumbent holdout — "
                        "gated retrain",
                    # PM correction 08-30: rows reconstructed from
                    # historical retrain records are RECOVERED_HISTORY;
                    # only rows emitted at training time under this
                    # contract may claim NATIVE_REGISTERED_RUN
                    "registry_origin": "RECOVERED_HISTORY",
                    "source_artifact": "metrics_history.jsonl",
                    "reconstructed_at": now,
                })
    if new:
        with out_p.open("a") as f:
            for r in new:
                f.write(json.dumps(r) + "\n")
    total = len(existing) + len(new)
    return total, len(new)


def main():
    now = int(time.time())
    kb = rows("kalshi_binary_log.jsonl")
    total_runs, new_runs = emit_training_runs(now)

    models = {}
    for variant, role in SERVING_T2.items():
        drows = decision_rows(kb, variant)
        windows = {}
        for name, k in WINDOWS:
            seg = drows if k is None else drows[-k:]
            windows[name] = _metrics(seg)
        launch = _metrics(drows[:LAUNCH_N])
        models[variant] = {
            "model_id": variant, "role": role, "tier": "T2",
            "lifecycle": "CONTROL" if variant == "kb2" else "TREATMENT",
            "offline": {"status": "NOT_RECORDED",
                        "note": "arms are online-prequential; the "
                        "offline block begins with the M2 walk-forward "
                        "evaluator — absence is stated, never faked"},
            "launch": {"note": f"first {LAUNCH_N} settled decision-"
                       "time windows in the retained log", **launch},
            "current": windows["LAST_25"],
            "lifetime": windows["LIFETIME"],
            "windows": windows,
            # PM 08-30: SELF TREND and BASELINE STATUS are separate
            # verdicts and must never collapse into one badge — a
            # model can improve vs its own past while still failing
            # to beat the market
            "self_trend": _trend(drows),
            "baseline_status": (
                "NOT_ESTABLISHED — lifetime BSS vs market "
                f"{windows['LIFETIME'].get('bss_vs_market')}"
                if (windows["LIFETIME"].get("bss_vs_market") or 0) <= 0
                else "ESTABLISHED — lifetime BSS "
                f"{windows['LIFETIME'].get('bss_vs_market')}"),
            "trend": _trend(drows),   # legacy alias of self_trend
            "scoring_convention": "decision-time row (earliest settled "
                                  "row with mins_left <= 12) — one "
                                  "observation per window",
            "min_n": MIN_N,
        }

    # T1 serving pairs: training-registry view (keep/revert cadence)
    truns = rows("training_runs.jsonl")
    t1 = {}
    for arm in ("h1", "h5", "h15", "h30", "t9-h1", "t10-h5",
                "t7-h15", "t9-h30"):
        mine = [r for r in truns if r["model_id"] == arm]
        kept = sum(1 for r in mine
                   if r["promotion_decision"] == "KEPT")
        last = mine[-1] if mine else None
        t1[arm] = {
            "model_id": arm, "tier": "T1",
            "lifecycle": "CONTROL" if "-" not in arm else "TREATMENT",
            "runs": len(mine), "kept": kept,
            "keep_rate": round(kept / len(mine), 3) if mine else None,
            "last_run_ts": last["run_ts"] if last else None,
            "last_decision": last["promotion_decision"] if last
            else None,
            "last_candidate_val_mse":
                last["candidate_metrics"]["val_mse"] if last else None,
        }

    doc = {"generated_ts": now,
           "dictionary_version": "1.1.0",
           "training_runs": {"total": total_runs,
                             "appended_this_emit": new_runs,
                             "artifact": "training_runs.jsonl"},
           "t2_models": models,
           "t1_serving": t1,
           "provenance": {
               "kb_source": "kalshi_binary_log.jsonl (retained tail)",
               "runs_source": "metrics_history.jsonl",
           }}
    (RES / "model_lifecycle.json").write_text(json.dumps(doc, indent=1))
    # model_online.json — the rolling-window view (M2.4 / §40): same
    # definitions as the lifecycle blocks, one artifact the Models
    # module can render without custom code
    online = {"generated_ts": now,
              "metric_contract": "config/METRICS.yaml v1.1.0",
              "windows_def": {"LAST_25": 25, "LAST_50": 50,
                              "LAST_100": 100, "LIFETIME": None},
              "min_n": MIN_N,
              "models": {v: {"windows": m["windows"],
                             "trend": m["trend"],
                             "role": m["role"],
                             "lifecycle": m["lifecycle"]}
                         for v, m in models.items()}}
    (RES / "model_online.json").write_text(json.dumps(online, indent=1))
    k2 = models.get("kb2", {})
    print(f"model_lifecycle: kb2 lifetime brier "
          f"{k2.get('lifetime', {}).get('brier')} "
          f"trend {k2.get('trend', {}).get('state')} · "
          f"training_runs {total_runs} (+{new_runs})")


if __name__ == "__main__":
    main()
