"""Benchmark TS foundation models (+ baselines) on our 15-min BRTI barrier problem.

Same PIT harness for every backend: intra-window BRTI prefix path (up to ~11 min left) ->
P(final >= target) -> walk-forward OOS -> precision/recall/F1/accuracy threshold sweep.
Picks the F1-max (balanced) threshold per model and ranks them. Backends load lazily and a
failure (missing weights/network) is recorded, not fatal.

Backends: barrier (closed-form), market k_prob, Chronos-Bolt {local-ft, small, base},
TimesFM (if weights load). Add more as they become available.
"""
import json
import math
import statistics as st
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "results" / "brti_decision_dataset.jsonl"
OUT = ROOT / "research" / "fm_benchmark_report.json"
LOCAL_BOLT = ROOT / "results" / "chronos_bolt_ft"
ENTRY_TR = 660.0
MIN_CTX = 8
QL = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]   # Chronos-Bolt native range


def _phi(x):
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def windows():
    byw = defaultdict(list)
    for l in SRC.open():
        l = l.strip()
        if not l:
            continue
        try:
            d = json.loads(l)
        except Exception:
            continue
        if d.get("exact_yes") in (0, 1) and d.get("time_remaining_s") is not None and d.get("current_brti"):
            byw[d.get("market_window_id")].append(d)
    W = []
    for wid, rows in byw.items():
        rows.sort(key=lambda r: -(r.get("time_remaining_s") or 0))
        pre = [r for r in rows if (r.get("time_remaining_s") or 0) >= ENTRY_TR]
        if len(pre) < MIN_CTX:
            continue
        d = pre[-1]
        cur = d["current_brti"]; tgt = d.get("official_target") or d.get("official_target_kalshi")
        tr = d.get("time_remaining_s"); kp = d.get("k_prob")
        if not tgt or not tr:
            continue
        series = [r["current_brti"] for r in pre if r.get("current_brti")]
        span = pre[0]["time_remaining_s"] - tr
        dt = max(1.0, span / max(1, len(series) - 1))
        rets = [math.log(series[i] / series[i - 1]) for i in range(1, len(series)) if series[i - 1] > 0]
        vol = st.pstdev(rets) if len(rets) > 1 else 1e-5
        sigma_T = cur * vol * math.sqrt(max(1.0, tr / dt))
        theo = _phi((cur - tgt) / max(sigma_T, 1e-6))
        W.append({"series": series, "cur": cur, "tgt": tgt, "tr": tr, "dt": dt,
                  "kprob": kp if kp is not None else 0.5, "theo": theo,
                  "y": d["exact_yes"], "ts": d.get("decision_time") or 0})
    W.sort(key=lambda w: w["ts"])
    return W


def _p_ge(qvals, target):
    lv = sorted(zip(QL, qvals), key=lambda t: t[1])
    vals = [v for _, v in lv]; levs = [l for l, _ in lv]
    if target <= vals[0]:
        return 1.0 - levs[0]
    if target >= vals[-1]:
        return 1.0 - levs[-1]
    for i in range(1, len(vals)):
        if vals[i] >= target:
            f = (target - vals[i - 1]) / ((vals[i] - vals[i - 1]) or 1e-9)
            return 1.0 - (levs[i - 1] + f * (levs[i] - levs[i - 1]))
    return 0.5


def chronos_backend(src):
    import torch
    from chronos import BaseChronosPipeline
    pipe = BaseChronosPipeline.from_pretrained(src, device_map="cpu", torch_dtype=torch.float32)

    def p_up(w):
        horizon = max(1, min(64, int(w["tr"] / w["dt"])))
        q, _ = pipe.predict_quantiles(torch.tensor(w["series"], dtype=torch.float32),
                                      prediction_length=horizon, quantile_levels=QL)
        qn = q[0].detach().numpy()
        tail = qn[-min(4, horizon):]
        return float(np.mean([_p_ge(tail[i], w["tgt"]) for i in range(tail.shape[0])]))
    return p_up


def timesfm_backend():
    """TimesFM 2.5 (200M torch). Quantile forecast -> P(final >= target) via the same CDF
    interpolation used for Chronos. Model loaded + compiled once; forecast per window."""
    import timesfm
    model = timesfm.TimesFM_2p5_200M_torch.from_pretrained("google/timesfm-2.5-200m-pytorch")
    model.compile(timesfm.ForecastConfig(max_context=512, max_horizon=64,
                                          normalize_inputs=True, use_continuous_quantile_head=True,
                                          fix_quantile_crossing=True))
    # TimesFM emits 10 quantile columns: [mean, 0.1, 0.2, ..., 0.9]
    TQL = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

    def _p_ge_levels(qvals, levels, target):
        lv = sorted(zip(levels, qvals), key=lambda t: t[1])
        vals = [v for _, v in lv]; levs = [l for l, _ in lv]
        if target <= vals[0]:
            return 1.0 - levs[0]
        if target >= vals[-1]:
            return 1.0 - levs[-1]
        for i in range(1, len(vals)):
            if vals[i] >= target:
                f = (target - vals[i - 1]) / ((vals[i] - vals[i - 1]) or 1e-9)
                return 1.0 - (levs[i - 1] + f * (levs[i] - levs[i - 1]))
        return 0.5

    def p_up(w):
        horizon = max(1, min(64, int(w["tr"] / w["dt"])))
        _, q = model.forecast(horizon=horizon, inputs=[np.asarray(w["series"], dtype=np.float32)])
        qa = np.asarray(q)[0]                       # (horizon, n_q)
        nq = qa.shape[-1]
        cols = qa[:, 1:10] if nq >= 10 else qa[:, :9]   # drop leading mean col if present
        levels = TQL[:cols.shape[-1]]
        tail = cols[-min(4, horizon):]
        return float(np.mean([_p_ge_levels(tail[i], levels, w["tgt"]) for i in range(tail.shape[0])]))
    return p_up


def pr_sweep(p, y):
    npos = int((y == 1).sum()); best = (None, -1); rows = {}
    for thr in (0.45, 0.5, 0.55, 0.6, 0.65, 0.7):
        pred = (p >= thr).astype(int)
        tp = int(((pred == 1) & (y == 1)).sum()); fp = int(((pred == 1) & (y == 0)).sum())
        fn = int(((pred == 0) & (y == 1)).sum()); tn = int(((pred == 0) & (y == 0)).sum())
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / npos if npos else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        acc = (tp + tn) / len(y)
        rows[str(thr)] = {"precision": round(prec, 4), "recall": round(rec, 4), "f1": round(f1, 4),
                          "accuracy": round(acc, 4), "FP": fp, "FN": fn, "coverage": round(float((p >= thr).mean()), 3)}
        if f1 > best[1]:
            best = (str(thr), f1)
    return {"sweep": rows, "f1_max_thr": best[0], "best": rows[best[0]]}


def run(which):
    W = windows()
    y = np.array([w["y"] for w in W], int)
    mid = int(len(W) * 0.6); yte = y[mid:]
    backends = {
        "barrier_closed_form": lambda: (lambda w: w["theo"]),
        "market_kprob": lambda: (lambda w: w["kprob"]),
        "chronos_bolt_local_ft": lambda: chronos_backend(str(LOCAL_BOLT)),
        "chronos_bolt_small": lambda: chronos_backend("amazon/chronos-bolt-small"),
        "chronos_bolt_base": lambda: chronos_backend("amazon/chronos-bolt-base"),
        "timesfm": timesfm_backend,
    }
    report = {"n": len(W), "n_test": len(W) - mid, "base_up": round(float(y.mean()), 4), "models": {}}
    for name, mk in backends.items():
        if which and name not in which:
            continue
        t0 = time.time()
        try:
            fn = mk()
            p = np.array([fn(w) for w in W], float)
            p = np.nan_to_num(p, nan=0.5, posinf=1.0, neginf=0.0)
            res = pr_sweep(p[mid:], yte)
            res["oos_hit_0.5"] = round(float(((p[mid:] >= 0.5).astype(int) == yte).mean()), 4)
            res["infer_ms_per_window"] = round((time.time() - t0) * 1000 / max(1, len(W)), 1)
            report["models"][name] = res
            print(f"  {name:24} F1max@{res['f1_max_thr']}: {res['best']} | hit@.5 {res['oos_hit_0.5']} "
                  f"| {res['infer_ms_per_window']}ms/win")
        except Exception as e:
            report["models"][name] = {"error": f"{type(e).__name__}: {str(e)[:150]}"}
            print(f"  {name:24} UNAVAILABLE: {type(e).__name__}: {str(e)[:120]}")
    ok = {k: v for k, v in report["models"].items() if "best" in v}
    if ok:
        best = max(ok, key=lambda k: ok[k]["best"]["f1"])
        report["recommendation"] = {"model": best, "threshold": ok[best]["f1_max_thr"], "metrics": ok[best]["best"]}
        print(f"\n  RECOMMENDATION: {best} @ thr {ok[best]['f1_max_thr']} -> {ok[best]['best']}")
    OUT.write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    run(set(sys.argv[1:]) or None)
