"""M5.5 — Execution Researcher (script-backed structured agent).

Core contract (PM 08-30): explicitly distinguish

    THESIS ADVERSE SELECTION
        outcome-level — the market correctly repriced the
        probability; fills land in worse thesis states
        (measured: fill win rate vs eligible-window win rate)

    EXECUTION ADVERSE SELECTION
        price-level — the fill mechanism itself buys into continued
        short-horizon repricing
        (measured: conservative bid markouts at +1s/+5s/+10s/+30s/+60s)

The two can disagree, and which one dominates decides what a future
treatment would even need to solve. Observational ONLY: this agent
never modifies A3. Future role (noted, not executed): whether
cross-venue-confirmed BTC moves show different A3 markout/AS
profiles than isolated Kalshi dips — blocked on the synchronized
xvenue tape maturing past Gate F1.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import agent_firewall as fw

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"
MIN_N = 5           # display floor for fill-level stats (tiny sample
                    # is stated, never hidden)
HORIZONS = ("markout_1s", "markout_5s", "markout_10s",
            "markout_30s", "markout_60s")


def main():
    p = RES / "a3_window_evaluation.jsonl"
    rows = []
    if p.exists():
        for l in p.open():
            if l.strip():
                try:
                    rows.append(json.loads(l))
                except Exception:
                    pass
    el = [r for r in rows if r.get("state") != "SYSTEM_EXCLUDED"
          and r.get("settled")]
    fills = [r for r in el if r.get("state") == "FILLED"]
    n_el, n_f = len(el), len(fills)
    if not n_el:
        print("execution_researcher: no settled eligible windows")
        return
    el_win = sum(1 for r in el if r.get("won")) / n_el
    fill_win = sum(1 for r in fills if r.get("won")) / n_f \
        if n_f else None
    thesis_as = {
        "fill_win_rate": round(fill_win, 3) if fill_win is not None
        else None,
        "eligible_win_rate": round(el_win, 3),
        "gap": round(el_win - fill_win, 3) if fill_win is not None
        else None,
        "n_fills": n_f, "n_eligible": n_el,
        "state": "UNDERPOWERED" if n_f < MIN_N else
        ("PRESENT" if fill_win is not None
         and el_win - fill_win > 0.10 else "NOT_EVIDENT"),
        "meaning": "outcome-level: dips partly ARE the market "
                   "correctly repricing the thesis down",
    }
    mk = {}
    for h in HORIZONS:
        vals = [r[h] for r in fills
                if isinstance(r.get(h), (int, float))]
        mk[h] = {"n": len(vals),
                 "mean_c": round(sum(vals) / len(vals), 2)
                 if vals else None,
                 "frac_negative": round(sum(
                     1 for v in vals if v < 0) / len(vals), 2)
                 if vals else None}
    short = [mk[h]["mean_c"] for h in ("markout_1s", "markout_5s",
                                       "markout_10s")
             if mk[h]["mean_c"] is not None]
    exec_as = {
        "by_horizon": mk,
        "state": "UNDERPOWERED" if n_f < MIN_N else
        ("PRESENT" if short and sum(short) / len(short) < -2.0
         else "NOT_EVIDENT"),
        "meaning": "price-level: does the fill mechanism buy into "
                   "continued short-horizon repricing?",
    }
    dominant = ("UNDERPOWERED — both channels below display floor"
                if n_f < MIN_N else
                "THESIS_ADVERSE_SELECTION" if thesis_as["state"]
                == "PRESENT" and exec_as["state"] != "PRESENT" else
                "EXECUTION_ADVERSE_SELECTION" if exec_as["state"]
                == "PRESENT" and thesis_as["state"] != "PRESENT" else
                "BOTH" if thesis_as["state"] == exec_as["state"]
                == "PRESENT" else "NEITHER_EVIDENT")
    doc = {
        "generated_ts": int(time.time()),
        "experiment": "A3-v1.1 (observational — never modifies A3)",
        "thesis_adverse_selection": thesis_as,
        "execution_adverse_selection": exec_as,
        "dominant_channel": dominant,
        "future_role": {
            "question": "do cross-venue-confirmed BTC moves show "
                        "different A3 markout/AS profiles than "
                        "isolated Kalshi dips?",
            "status": "BLOCKED — synchronized F-XVENUE tape must "
                      "pass Gate F1 (>= 2026-09-06)"},
        "provenance": ["a3_window_evaluation.jsonl"],
    }
    (RES / "execution_research.json").write_text(
        json.dumps(doc, indent=1))
    fw.submit(
        agent="execution_researcher", action_class="DIAGNOSE",
        finding=f"A3 fills n={n_f}: thesis-AS "
                f"{thesis_as['state']} (fill win "
                f"{thesis_as['fill_win_rate']} vs eligible "
                f"{thesis_as['eligible_win_rate']}), execution-AS "
                f"{exec_as['state']} (10s markout "
                f"{mk['markout_10s']['mean_c']}c) — dominant: "
                f"{dominant}",
        recommendation="observe; the dominant channel decides what a "
                       "future treatment must solve — no A3 change",
        evidence=["execution_research.json",
                  "a3_window_evaluation.jsonl"],
        n=n_f)
    print(f"execution_researcher: fills {n_f}/{n_el} · thesis-AS "
          f"{thesis_as['state']} · exec-AS {exec_as['state']} · "
          f"dominant {dominant}")


if __name__ == "__main__":
    main()
