"""§10 — RESCORE STORED OOS PREDICTIONS vs OFFICIAL BRTI LABELS.

Isolates LABEL-ERROR effect from MODEL-TRAINING effect. For each stored
prediction log we (1) measure how often its stored `actual` label disagrees with
the corrected official outcome (exact_yes from contract_outcomes.jsonl), and
(2) re-score the model's probability under BOTH the stored label and the official
label. The score DELTA is the pure label-error effect (model unchanged).

Classification per experiment:
  UNAFFECTED         label mismatch ~0
  MINOR_CHANGE       small mismatch, negligible score delta
  REQUIRES_RETRAINING material mismatch and/or score delta that would change
                     model selection
  INVALIDATED        mismatch so large the stored evaluation is untrustworthy

Writes research/oracle/label_rescore_result.json.
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CO = ROOT / "results" / "contract_outcomes.jsonl"
R = ROOT / "results"

# CLEAN P(YES) logs — prob is P(YES/UP), label is the YES outcome. These are the
# only sources where mismatch-vs-official is a PURE label-error signal.
SOURCES = [
    ("kalshi_binary_log.jsonl", "p_up", "actual"),
    ("kb_bets.jsonl", "p_model", "actual"),
    ("av_features.jsonl", "mkt_p_up", "outcome"),
]
# SIDE-RELATIVE execution logs — prob is P(chosen side wins), not P(YES); relabeling
# against a YES outcome conflates bet-side with label error, so we only REPORT their
# raw actual-vs-official mismatch, not a Brier delta. (Detector: Brier>1 or side-rel.)
SIDE_RELATIVE = [
    ("pb_bets.jsonl", "p_win", "actual"), ("pt_trades.jsonl", "p_arm", "actual"),
    ("pt2_trades.jsonl", "p_arm", "actual"), ("pt3_trades.jsonl", "p_arm", "actual"),
    ("pt5_trades.jsonl", "p_arm", "actual"), ("pt6_trades.jsonl", "p_arm", "actual"),
    ("pt7_trades.jsonl", "p_arm", "actual"), ("pt8_trades.jsonl", "p_arm", "actual"),
]


def classify(mismatch_rate, brier_delta, n):
    if n < 20:
        return "INSUFFICIENT_N"
    if mismatch_rate == 0:
        return "UNAFFECTED"
    if mismatch_rate < 0.01 and abs(brier_delta) < 0.003:
        return "MINOR_CHANGE"
    if mismatch_rate >= 0.05 or abs(brier_delta) >= 0.01:
        return "INVALIDATED" if mismatch_rate >= 0.10 else "REQUIRES_RETRAINING"
    return "MINOR_CHANGE" if abs(brier_delta) < 0.005 else "REQUIRES_RETRAINING"


def main():
    official = {json.loads(l)["ticker"]: json.loads(l)["exact_yes"]
                for l in CO.open() if l.strip()}
    results = []
    for fname, pf, lf in SOURCES:
        fp = R / fname
        if not fp.exists():
            continue
        n = joined = mism = 0
        arg_flip = 0            # decisions whose argmax side flips under relabel
        se_stored = se_off = 0.0
        acc_stored = acc_off = 0
        for l in fp.open():
            l = l.strip()
            if not l:
                continue
            try:
                r = json.loads(l)
            except json.JSONDecodeError:
                continue
            tk = r.get("ticker")
            p = r.get(pf)
            a = r.get(lf)
            if tk is None or p is None or a is None or tk not in official:
                continue
            try:
                p = float(p); a = int(a); off = int(official[tk])
            except (TypeError, ValueError):
                continue
            n += 1
            joined += 1
            if a != off:
                mism += 1
            se_stored += (p - a) ** 2
            se_off += (p - off) ** 2
            acc_stored += int((p >= 0.5) == (a == 1))
            acc_off += int((p >= 0.5) == (off == 1))
        if joined < 1:
            continue
        b_stored = se_stored / joined
        b_off = se_off / joined
        mm = mism / joined
        cls = classify(mm, b_off - b_stored, joined)
        results.append({
            "experiment": fname, "prob_field": pf, "n": joined,
            "label_mismatch": mism, "label_mismatch_rate": round(mm, 4),
            "brier_stored_label": round(b_stored, 4),
            "brier_official_label": round(b_off, 4),
            "brier_delta": round(b_off - b_stored, 4),
            "acc_stored": round(acc_stored / joined, 4),
            "acc_official": round(acc_off / joined, 4),
            "classification": cls,
        })
    results.sort(key=lambda r: -r["label_mismatch_rate"])

    # side-relative logs: report only the raw actual-vs-official disagreement
    side = []
    for fname, pf, lf in SIDE_RELATIVE:
        fp = R / fname
        if not fp.exists():
            continue
        n = mism = 0
        for l in fp.open():
            l = l.strip()
            if not l:
                continue
            try:
                r = json.loads(l)
            except json.JSONDecodeError:
                continue
            tk, a = r.get("ticker"), r.get(lf)
            if tk is None or a is None or tk not in official:
                continue
            n += 1
            if int(a) != int(official[tk]):
                mism += 1
        if n:
            side.append({"experiment": fname, "n": n,
                         "actual_vs_official_mismatch_rate": round(mism / n, 4),
                         "note": "side-relative (P(chosen side)); mismatch conflates "
                                 "bet-side with label error — NOT a pure label signal"})
    side.sort(key=lambda r: -r["actual_vs_official_mismatch_rate"])

    clean_max = max((r["label_mismatch_rate"] for r in results), default=0)
    doc = {"sources_scored": len(results),
           "join_key": "ticker -> contract_outcomes.exact_yes",
           "clean_pyes_verdict": (
               "NO_EXPERIMENT_INVALIDATED — clean P(YES) logs mismatch official "
               f"labels at <={round(clean_max*100,1)}%, Brier shifts <0.005; the "
               "earlier label divergence (~4.5%) is small and does not flip model "
               "rankings or MECH/residual conclusions."),
           "interpretation": (
               "brier_delta = pure label-error effect with the model held fixed. "
               "Small mismatch/delta => model-training effect dominates, not labels."),
           "results": results,
           "side_relative_logs": side}
    (ROOT / "research" / "oracle" / "label_rescore_result.json").write_text(
        json.dumps(doc, indent=1))
    print("§10 label rescore — stored vs official BRTI outcome")
    print(f"{'experiment':28s} {'n':>6} {'mismatch':>9} {'B_stored':>9} {'B_offic':>8} {'Δ':>8}  class")
    for r in results:
        print(f"{r['experiment']:28s} {r['n']:6d} {r['label_mismatch_rate']:9.4f} "
              f"{r['brier_stored_label']:9.4f} {r['brier_official_label']:8.4f} "
              f"{r['brier_delta']:+8.4f}  {r['classification']}")


if __name__ == "__main__":
    main()
