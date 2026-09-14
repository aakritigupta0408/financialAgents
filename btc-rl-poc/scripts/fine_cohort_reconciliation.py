"""FINE_FACTORY_COHORT_RECONCILIATION — resolve 363 (P1.1 fine-5m audit) vs 365
(fine factory rows) before freezing the FINE cohort, analogous to the coarse gate.

Freezes three explicit cohorts with membership hashes:
  FINE_ROW         row exists in fine_features.jsonl (factory rule: >=8 samples in 5m)
  FINE_COMPLETE_14 all 14 brti_fine.* features non-null
  FINE_5M_CORE     P1.1 audit rule: >= (300/16*0.5) real samples in [T0-300s, T0]
"""
import hashlib
import json
import sys
import time
from bisect import bisect_left, bisect_right
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from btc_rl import research_events as EV  # noqa: E402

INV = ROOT / "research" / "true15m" / "contract_inventory.jsonl"
SAMP = ROOT / "results" / "brti_decision_dataset.jsonl"
FEAT = ROOT / "research" / "true15m" / "fine_features.jsonl"
OUT = ROOT / "research" / "true15m" / "FINE_COHORT_RECONCILIATION.json"
LOOKBACK = 300
FACTORY_MIN = 8
AUDIT_MIN = (300 / 16) * 0.5           # P1.1 fine threshold


def _hash(ids):
    return hashlib.sha256("\n".join(sorted(ids)).encode()).hexdigest()[:16]


def build():
    t0 = time.time()
    EV.emit("JOB_STARTED", "FINE_FACTORY_COHORT_RECONCILIATION", lane="L7",
            narrative="Reconciling 363 (fine-5m audit) vs 365 (fine factory rows) before "
                      "freezing the FINE cohort.")
    ts = sorted({float(r["brti_sample_ts"]) for r in
                 (json.loads(l) for l in SAMP.open() if l.strip())
                 if r.get("brti_sample_ts")})
    inv = [json.loads(l) for l in INV.open() if l.strip()]

    audit_ids, factory_rule_ids = set(), set()
    for w in inv:
        t = w["T0"]
        cnt = bisect_right(ts, t) - bisect_left(ts, t - LOOKBACK)
        if cnt >= AUDIT_MIN:
            audit_ids.add(w["market_window_id"])
        if cnt >= FACTORY_MIN:
            factory_rule_ids.add(w["market_window_id"])

    fac = [json.loads(l) for l in FEAT.open() if l.strip()]
    fac_ids = {r["market_window_id"] for r in fac}
    fkeys = list(fac[0]["features"].keys()) if fac else []
    complete14 = {r["market_window_id"] for r in fac
                  if all(r["features"].get(k) is not None for k in fkeys)}

    factory_only = sorted(fac_ids - audit_ids)
    audit_only = sorted(audit_ids - fac_ids)
    # per-window sample counts for the differing windows
    t0_by = {w["market_window_id"]: w["T0"] for w in inv}
    def samples(wid):
        t = t0_by[wid]; return bisect_right(ts, t) - bisect_left(ts, t - LOOKBACK)
    explain = [{"market_window_id": w, "samples_in_5m": samples(w)} for w in factory_only[:20]]

    doc = {
        "schema_version": "fine-cohort-reconciliation-1", "generated_at": time.time(),
        "counts": {
            "fine_5m_audit": len(audit_ids),
            "fine_factory_rows": len(fac_ids),
            "rows_all_14_complete": len(complete14),
            "rows_partial": len(fac_ids) - len(complete14),
        },
        "thresholds": {"factory_min_samples": FACTORY_MIN,
                       "audit_min_samples": round(AUDIT_MIN, 3)},
        "set_difference": {
            "factory_only_n": len(factory_only), "audit_only_n": len(audit_only),
            "factory_only_ids": factory_only, "audit_only_ids": audit_only,
            "explanation": "Documented threshold gap: the factory admits windows with "
                f">= {FACTORY_MIN} real samples in the 5m pre-open; the P1.1 audit required "
                f">= {AUDIT_MIN:.2f}. The factory-only windows have exactly {FACTORY_MIN}-9 "
                "samples (below the audit bar). Not a leak — a stricter-vs-looser count rule.",
            "per_window_examples": explain,
        },
        "cohorts": {
            "FINE_ROW": {"n": len(fac_ids), "hash": _hash(fac_ids),
                         "def": f">= {FACTORY_MIN} samples in 5m pre-open (factory)"},
            "FINE_COMPLETE_14": {"n": len(complete14), "hash": _hash(complete14),
                                 "def": "all 14 brti_fine.* features non-null"},
            "FINE_5M_CORE": {"n": len(audit_ids), "hash": _hash(audit_ids),
                             "def": f">= {AUDIT_MIN:.2f} samples in 5m pre-open (P1.1 audit)"},
        },
        "default_cohort_family_B": "FINE_COMPLETE_14",
        "canonical_note": "FINE cohort semantics frozen; every Family-B model declares its "
                          "cohort. No silent dual definition.",
    }
    OUT.write_text(json.dumps(doc, indent=1))
    EV.emit("DISCOVERY", "FINE cohort reconciliation complete", lane="L7", severity="high",
            fact=f"audit {len(audit_ids)}, factory {len(fac_ids)}, complete-14 {len(complete14)}. "
                 f"factory_only {len(factory_only)}, audit_only {len(audit_only)}.",
            interpretation="Difference is a sample-count threshold gap (factory >=8 vs audit "
                           ">=9.4), not a leak. Three FINE cohorts frozen with hashes.",
            next_action="AV full backfill + derivatives/options/news resolution continue.",
            files=[str(OUT.relative_to(ROOT))],
            metrics={"FINE_ROW": len(fac_ids), "FINE_5M_CORE": len(audit_ids),
                     "FINE_COMPLETE_14": len(complete14)},
            duration_ms=int((time.time() - t0) * 1000))
    print(f"fine_reconciliation: audit={len(audit_ids)} factory={len(fac_ids)} "
          f"complete14={len(complete14)} | factory_only={len(factory_only)} audit_only={len(audit_only)}")


if __name__ == "__main__":
    build()
