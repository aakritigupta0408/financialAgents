"""Consolidate every 'what is the system learning' panel into one
compact JSON for the learning-diagnostics artifact. READ-ONLY over
existing logs; no fitting, no peeking. Each panel maps to a real logged
source, and the one gap (kb per-arm training-loss curve) is emitted as
an explicit NOT_INSTRUMENTED marker rather than fabricated.
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"


def jload(name):
    return json.load((RES / name).open())


def jlines(name, cap=None):
    rows = [json.loads(l) for l in (RES / name).open() if l.strip()]
    return rows[-cap:] if cap else rows


def agg_offline(folds, k):
    vs = [f[k] for f in folds if f.get(k) is not None]
    return round(sum(vs) / len(vs), 4) if vs else None


def main():
    out = {"note": "read-only learning diagnostics; sources named per panel"}

    # 1) FEATURE IMPORTANCE — logistic arm coefficients (real, named)
    mi = jload("model_internals.json")
    upd = mi.get("logit_updates", {})
    fi = {}
    for arm in ("kb3", "kb4", "kb5", "kb8"):
        coefs = mi.get("logits", {}).get(arm)
        if not coefs:
            continue
        fi[arm] = {"updates": upd.get(arm),
                   "coefs": [{"f": c["feature"], "w": round(c["w"], 4)}
                             for c in coefs]}
    out["feature_importance"] = {
        "source": "model_internals.json:logits",
        "loss": "binary cross-entropy (online SGD)",
        "arms": fi}

    # 2) OFFLINE EVAL METRICS — per arm, aggregated over folds (real)
    mo = jload("model_offline.json")
    off = []
    for arm, d in mo["models"].items():
        folds = d.get("folds", [])
        off.append({"arm": arm,
                    "n": sum(f.get("n", 0) for f in folds),
                    "brier": agg_offline(folds, "brier"),
                    "bss": agg_offline(folds, "bss"),
                    "ece": agg_offline(folds, "ece"),
                    "cal_slope": agg_offline(folds, "calibration_slope"),
                    "mkt_brier": agg_offline(folds, "market_brier")})
    out["offline_metrics"] = {
        "source": "model_offline.json:folds (prequential)",
        "primary": "BSS vs market (higher=better; 0=ties market)",
        "arms": sorted(off, key=lambda r: (r["bss"] is None, -(r["bss"] or -9)))}

    # 3) ONLINE EVAL METRICS — window breakdown (real; only served arms)
    on = jload("model_online.json")
    onl = {}
    for arm, d in on["models"].items():
        onl[arm] = {w: {"n": v["n"], "brier": v["brier"],
                        "bss": v.get("bss_vs_market"),
                        "mkt_brier": v.get("market_brier")}
                    for w, v in d["windows"].items()}
    out["online_metrics"] = {
        "source": "model_online.json (served arms with sufficient n)",
        "served": list(onl.keys()), "arms": onl}

    # 4) CALIBRATION — shadow Platt prequential log-loss raw vs cal (real)
    cal = []
    for arm, c in mi.get("calib", {}).items():
        if "mean_ll_raw" in c:
            cal.append({"arm": arm, "ll_raw": c["mean_ll_raw"],
                        "ll_cal": c["mean_ll_cal"],
                        "cal_minus_raw": c["cal_minus_raw"],
                        "updates": c.get("updates")})
    out["calibration"] = {
        "source": "model_internals.json:calib",
        "reading": "cal_minus_raw > 0 => calibration INCREASES log-loss "
                   "(hurts). Loss = prequential log-loss.",
        "arms": cal}

    # 5) TRAINING CURVES — T1 agents, per-epoch val (real)
    tp = jlines("training_progress.jsonl")
    curves = {}
    for r in tp:
        if r.get("val_mae") is None:
            continue
        key = f"{r['agent']}|h{r['horizon']}"
        curves.setdefault(key, []).append(
            {"epoch": r["epoch"], "val_mae": round(r["val_mae"], 3),
             "train_reward": r.get("train_reward_mean")})
    for k in curves:
        curves[k].sort(key=lambda x: x["epoch"])
    out["training_curves"] = {
        "source": "training_progress.jsonl (T1 RL/tabular agents)",
        "loss": "val_MAE (dollars) per epoch",
        "kb_arms": "NOT_INSTRUMENTED — kb arms log update counts + "
                   "prequential calibrator LL only; no per-step training "
                   "loss is recorded (declared gap)",
        "series": curves}

    # 6) GATED RETRAIN — live incumbent val_MAE over keep/revert cycles
    #    at h15 (real). This IS the T1 "eval loss over time": each event
    #    the candidate is accepted only if it beats the incumbent holdout.
    mh = jlines("metrics_history.jsonl")
    keep, promo = {}, {}
    for b in mh:
        if b.get("kind") != "retrain":
            continue
        ts = b.get("ts")
        for model_id, hzs in (b.get("gate") or {}).items():
            if "h15" not in model_id and model_id != "h15":
                continue
            inner = next(iter(hzs.values()), None)  # single horizon record
            if not isinstance(inner, dict):
                continue
            rev = bool(inner.get("reverted"))
            before, after = inner.get("val_mae_before"), inner.get("val_mae_after")
            live = before if rev else after
            if live is None:
                continue
            keep.setdefault(model_id, []).append(
                {"ts": ts, "mae": round(live, 2), "reverted": rev})
            p = promo.setdefault(model_id, {"kept": 0, "reverted": 0})
            p["reverted" if rev else "kept"] += 1
    for k in keep:
        keep[k].sort(key=lambda x: x["ts"])
    out["retrain_gate"] = {
        "source": "metrics_history.jsonl:retrain gate (h15 models)",
        "loss": "live incumbent validation MAE (dollars) per gated retrain",
        "reading": "candidate promoted only if it beats incumbent holdout; "
                   "flat line with mostly reverts = no learnable edge left",
        "promotions": promo, "series": keep}

    # 7) LOSS FUNCTION REFERENCE (descriptive)
    out["loss_functions"] = [
        {"arm": "kb / kb2", "loss": "least-squares weight refit",
         "adapts": "yes (closed-form each cycle)", "role": "blend/control"},
        {"arm": "kb3 / kb4 / kb5 / kb8", "loss": "binary cross-entropy",
         "adapts": "yes (online SGD, decaying lr, L2=1e-4)",
         "role": "logistic (kb5 trains on `hit`, not raw outcome)"},
        {"arm": "kb7 (Chronos) / kb9 (TimesFM)", "loss": "none",
         "adapts": "NO — frozen foundation model, zero-shot",
         "role": "quantile → decile-interp probability"},
    ]

    (RES / "learning_charts.json").write_text(json.dumps(out, indent=1))
    print("learning_charts.json written")
    print(" feature_importance arms:", list(fi.keys()))
    print(" offline arms:", len(off), "| online served:", list(onl.keys()))
    print(" training curves:", len(curves), "| eval-time series:", len(keep))
    print(" calibration arms:", [c["arm"] for c in cal])


if __name__ == "__main__":
    main()
