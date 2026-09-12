"""P0 LABEL AUDIT — do our labels match the EXACT contract outcome, and would a
Coinbase-based reconstruction disagree with the true BRTI benchmark?

Sources: results/contract_outcomes.jsonl (authoritative floor_strike /
expiration_value / result), results/kalshi_binary_log.jsonl (our `actual`
labels), and the Oracle dataset's Coinbase price near close (proxy for a
Coinbase-at-expiry reconstruction). Quantifies the outcome flip rate a
Coinbase label would incur vs BRTI, especially near the decision boundary.
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CO = ROOT / "results" / "contract_outcomes.jsonl"
KLOG = ROOT / "results" / "kalshi_binary_log.jsonl"
DS = ROOT / "research" / "t05_repricing" / "dataset.jsonl"
OUT = ROOT / "research" / "label_audit.json"


def main():
    out = {tk: r for tk, r in ((json.loads(l)["ticker"], json.loads(l))
           for l in CO.open() if l.strip())}

    # (1) our labels vs official result
    ours = {}
    for l in KLOG.open():
        l = l.strip()
        if not l:
            continue
        try:
            r = json.loads(l)
        except json.JSONDecodeError:
            continue
        if r.get("ticker") and r.get("actual") is not None:
            ours[r["ticker"]] = int(r["actual"])
    joined = [(tk, ours[tk], 1 if out[tk]["result"] == "yes" else 0)
              for tk in ours if tk in out and out[tk]["result"] in ("yes", "no")]
    label_mismatch = sum(1 for _, a, b in joined if a != b)

    # (2) Coinbase-near-close proxy from the oracle dataset
    cb_close = {}
    for l in DS.open():
        l = l.strip()
        if not l:
            continue
        r = json.loads(l)
        if r.get("cb_price") is None:
            continue
        tk = r["ticker"]
        if tk not in cb_close or r["ts"] > cb_close[tk][0]:
            cb_close[tk] = (r["ts"], r["cb_price"])

    recon = []      # (ticker, coinbase_yes, true_yes, D, basis)
    for tk, (ts, px) in cb_close.items():
        if tk not in out:
            continue
        o = out[tk]
        cb_yes = int(px >= o["floor_strike"])
        true_yes = o["exact_yes"]
        recon.append((tk, cb_yes, true_yes, o["D"], px - o["expiration_value"]))

    n = len(recon)
    flips = sum(1 for _, cy, ty, _, _ in recon if cy != ty)
    near = [x for x in recon if abs(x[3]) <= 20]      # |D| <= $20 (near boundary)
    near_flips = sum(1 for _, cy, ty, _, _ in near if cy != ty)
    basis = [b for _, _, _, _, b in recon]
    basis.sort()
    import statistics as st
    doc = {
        "n_markets_total": len(out),
        "official_yes_rate": round(sum(1 for r in out.values()
                                       if r["result"] == "yes") / len(out), 4),
        "our_label_vs_official": {"checked": len(joined),
                                  "mismatches": label_mismatch,
                                  "mismatch_rate": round(label_mismatch / max(1, len(joined)), 5),
                                  "verdict": ("LABELS_CORRECT — our `actual` IS the "
                                              "official BRTI outcome"
                                              if label_mismatch == 0 else "LABELS_DIVERGE")},
        "coinbase_reconstruction_vs_brti": {
            "n": n, "flip_count": flips,
            "flip_rate": round(flips / max(1, n), 4),
            "near_boundary_n_absD_le_20": len(near),
            "near_boundary_flip_rate": round(near_flips / max(1, len(near)), 4)},
        "coinbase_minus_brti_basis_usd": {
            "mean": round(st.mean(basis), 2) if basis else None,
            "std": round(st.pstdev(basis), 2) if len(basis) > 1 else None,
            "p05": round(basis[int(.05 * len(basis))], 2) if basis else None,
            "p95": round(basis[int(.95 * len(basis))], 2) if basis else None},
        "interpretation": (
            "Our settlement LABELS come from Kalshi's official BRTI-based "
            "resolution -> correct. But modeling FEATURES (distance-to-strike, "
            "physics Oracle) used Coinbase instantaneous price vs the BRTI "
            "target; the Coinbase-vs-BRTI basis + 60s averaging is a real "
            "mismodeling that would flip a Coinbase-reconstructed label at the "
            "rate above, concentrated near the boundary. MECH_FAIR must model "
            "D = close_avg - open_avg on the correct statistic."),
    }
    OUT.write_text(json.dumps(doc, indent=1))
    print(f"markets {len(out)} · official YES {doc['official_yes_rate']}")
    print(f"our labels vs official: {label_mismatch}/{len(joined)} mismatch "
          f"-> {doc['our_label_vs_official']['verdict']}")
    print(f"Coinbase-vs-BRTI reconstruction flip rate: {doc['coinbase_reconstruction_vs_brti']['flip_rate']}"
          f" (near-boundary |D|<=$20: {doc['coinbase_reconstruction_vs_brti']['near_boundary_flip_rate']})")
    print(f"Coinbase-BRTI basis $: mean {doc['coinbase_minus_brti_basis_usd']['mean']}"
          f" std {doc['coinbase_minus_brti_basis_usd']['std']}")


if __name__ == "__main__":
    main()
