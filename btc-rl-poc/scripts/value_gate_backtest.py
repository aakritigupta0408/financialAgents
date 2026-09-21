"""THE decisive experiment (deep-audit fix plan, step 4): does value-gating a
model's calibrated probability against the actually-payable Kalshi ask produce
positive EV net of the ~7% vig — the whole edge thesis, never run until now.

Estimand (pre-registered): per-WINDOW (one decision-time row per (variant,window),
mins_left<=12 — kills the ~13.7-rows/window autocorrelation over-powering),
per-CONTRACT, NET-OF-VIG realized PnL on windows where the value gate fires
(edge = p_side*100 - ask - fee >= margin), compared PAIRED against the market
(always-take-the-market-favorite at the same ask). Inference: moving-block
bootstrap 95% CI on the paired per-window delta; a variant "wins" only if the CI
lower bound > 0. No test-fold peeking here (this scores the live prequential
stream honestly); a sealed-test version comes after the pipeline is unified.

Fee is the single canonical owner (btc_rl.metrics.kalshi_fee_c). Reads
results/kalshi_binary_log.jsonl. Prints the verdict; writes results/value_gate_backtest.json.
PAPER/SIM research.
"""
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from btc_rl.metrics import kalshi_fee_c  # noqa: E402

LOG = ROOT / "results" / "kalshi_binary_log.jsonl"
OUT = ROOT / "results" / "value_gate_backtest.json"
MARGINS_C = [0, 1, 2, 3, 5, 8]          # edge threshold sweep (cents)
BLOCK = 10                               # moving-block length (windows)
ITERS = 4000
ENVELOPE = 12.0                          # tradeable envelope: mins_left <= 12


def _decision_rows(rows):
    """One earliest in-envelope settled decision row per (variant, ticker)."""
    best = {}
    for r in rows:
        v, tk = r.get("variant"), r.get("ticker")
        if not v or not tk or r.get("actual") is None:
            continue
        ml = r.get("mins_left")
        if ml is None or ml > ENVELOPE:
            continue
        if r.get("p_up") is None or r.get("ask_c") is None:
            continue
        key = (v, tk)
        # earliest in-envelope = largest mins_left (closest to open)
        if key not in best or ml > best[key].get("mins_left", -1):
            best[key] = r
    return best


def _side_ask(r, side_up):
    """Payable ask (cents) for the chosen side; ask_c is logged for the
    model's own side, so approximate the opposite side as 100-ask when needed.
    Returns None if not resolvable."""
    ask = r.get("ask_c")
    if ask is None:
        return None
    logged_up = r.get("p_up", 0.5) >= 0.5
    return ask if side_up == logged_up else (100.0 - ask)


def _pnl_net_c(ask_c, won):
    """Per-contract net cents: pay ask+fee; receive 100 on win else 0."""
    fee = kalshi_fee_c(ask_c)
    return (100.0 - ask_c - fee) if won else -(ask_c + fee)


def _block_boot_ci(deltas):
    if len(deltas) < BLOCK * 2:
        return [None, None]
    import random
    rng = random.Random(0)
    n = len(deltas)
    nblocks = math.ceil(n / BLOCK)
    means = []
    for _ in range(ITERS):
        s, c = 0.0, 0
        for _b in range(nblocks):
            start = rng.randint(0, n - BLOCK)
            for k in range(BLOCK):
                s += deltas[start + k]; c += 1
        means.append(s / c)
    means.sort()
    return [round(means[int(0.025 * ITERS)], 3), round(means[int(0.975 * ITERS)], 3)]


def run():
    rows = [json.loads(l) for l in LOG.read_text().splitlines() if l.strip()]
    dec = _decision_rows(rows)
    by_variant = {}
    for (v, tk), r in dec.items():
        by_variant.setdefault(v, []).append(r)

    report = {"estimand": "paired per-window per-contract net-of-vig EV, value-gated model vs market, "
                          "CI=moving-block bootstrap (block=%d), win iff CI_low>0" % BLOCK,
              "envelope_mins_left": ENVELOPE, "variants": {}}
    for v, vrows in sorted(by_variant.items()):
        vrows.sort(key=lambda r: r.get("close_ts", 0))
        n_windows = len(vrows)
        per_margin = {}
        for margin in MARGINS_C:
            deltas = []          # paired: gated-model EV - market EV, per gated window
            n_gated = wins = 0
            model_pnls = []
            for r in vrows:
                p = r["p_up"]
                side_up = p >= 0.5
                p_side = p if side_up else 1 - p
                ask = _side_ask(r, side_up)
                if ask is None or not (1 <= ask <= 99):
                    continue
                fee = kalshi_fee_c(ask)
                edge = p_side * 100.0 - ask - fee
                if edge < margin:
                    continue
                # model bet outcome (actual==1 means YES happened)
                won = (int(r["actual"]) == 1) == side_up
                m_pnl = _pnl_net_c(ask, won)
                # market baseline on the SAME window: take the market favorite
                mkt = r.get("mkt_p_up")
                if mkt is None:
                    continue
                mkt_up = mkt >= 0.5
                mkt_ask = _side_ask(r, mkt_up)
                if mkt_ask is None or not (1 <= mkt_ask <= 99):
                    continue
                mkt_won = (int(r["actual"]) == 1) == mkt_up
                b_pnl = _pnl_net_c(mkt_ask, mkt_won)
                deltas.append(m_pnl - b_pnl)
                model_pnls.append(m_pnl)
                n_gated += 1
                wins += int(won)
            if n_gated == 0:
                per_margin[margin] = {"n_gated": 0, "coverage": 0.0}
                continue
            mean_ev = sum(model_pnls) / n_gated
            mean_delta = sum(deltas) / n_gated
            ci = _block_boot_ci(deltas)
            per_margin[margin] = {
                "n_gated": n_gated, "coverage": round(n_gated / n_windows, 4),
                "hit_rate": round(wins / n_gated, 4),
                "model_ev_per_gated_c": round(mean_ev, 2),
                "paired_delta_vs_market_c": round(mean_delta, 2),
                "delta_ci95_c": ci,
                "wins": (ci[0] is not None and ci[0] > 0),
            }
        report["variants"][v] = {"n_windows": n_windows, "by_margin": per_margin}

    OUT.write_text(json.dumps(report, indent=1))
    # ---- print verdict ----
    print(f"VALUE-GATE BACKTEST — {len(dec)} decision-windows, {len(by_variant)} variants")
    print(f"{'variant':8}{'margin':>7}{'n_gate':>7}{'cov':>7}{'hit':>7}{'EV/gate¢':>10}{'Δvs_mkt¢':>10}{'ΔCI95':>16}{'win?':>6}")
    any_win = False
    for v, d in report["variants"].items():
        for m, s in d["by_margin"].items():
            if not s.get("n_gated"):
                continue
            w = s["wins"]; any_win = any_win or w
            print(f"{v:8}{m:>7}{s['n_gated']:>7}{s['coverage']:>7}{s['hit_rate']:>7}"
                  f"{s['model_ev_per_gated_c']:>10}{s['paired_delta_vs_market_c']:>10}"
                  f"{str(s['delta_ci95_c']):>16}{'YES' if w else 'no':>6}")
    print("\nVERDICT:", "an arm's paired Δ-vs-market CI clears 0 (edge candidate — verify on sealed test)"
          if any_win else
          "NO arm shows positive-EV edge vs the market net of vig (CI never clears 0). "
          "Honest answer: no exploitable edge at open — STOP-NULL.")
    print("(Preliminary: uses logged ask_c + market-mid baseline; the sealed-test version "
          "requires the unified data spine — real book ask, unified anchor/settle.)")


if __name__ == "__main__":
    run()
