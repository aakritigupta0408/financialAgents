"""COVERAGE / COHORT MATRIX — the central P1 artifact (directive).

One honest row per feature family: real N, resolution, historical role, whether
it is a main-cohort source, and — crucially — WHY coverage is missing where it is.
Data-driven where measured (BRTI from the coverage audit; AV from the mini-gate);
explicit TBD / HISTORICALLY_UNAVAILABLE with reasons elsewhere. Nothing assumed.

Writes research/true15m/COVERAGE_MATRIX.json + emits an event. This artifact sizes
the two cohorts (COARSE_BTC_STATE ~6,189 ; FINE_BTC_STATE ~363) and tells us which
models are supportable.
"""
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from btc_rl import research_events as EV  # noqa: E402

T = ROOT / "research" / "true15m"
OUT = T / "COVERAGE_MATRIX.json"


def _j(p):
    try:
        return json.loads(p.read_text())
    except Exception:
        return {}


def build():
    brti = _j(T / "brti_state_coverage.json")
    inv = _j(T / "contract_inventory_report.json")
    total = inv.get("total_windows", 6337)
    core = brti.get("core_cohorts", {})
    coarse_n = (core.get("COARSE_BTC_STATE (>=2h @15m res)") or {}).get("n")
    fine_n = (core.get("FINE_BTC_STATE (real 5m sub-minute)") or {}).get("n")
    avm = _j(T / "av_backfill_mini0.json")
    avcov = avm.get("coverage_matrix", {})
    av_cohort = avm.get("cohort_windows")

    def av_row(sym, role, main, extra=None):
        c = avcov.get(sym, {})
        n = c.get("n")
        return {"family": f"AV {sym}", "real_n": (f"{n}/{av_cohort} (MINI-0)" if n is not None else 0),
                "resolution": "5m", "historical_role": role, "main_cohort": main,
                "why_missing": extra or ("full historical backfill pending — MINI-0 slice "
                                         "only; AV equity intraday has months of history so "
                                         "full coverage expected")}

    rows = [
        {"family": "BRTI coarse state", "real_n": coarse_n, "resolution": "15m marks",
         "historical_role": "BTC dynamics (large low-frequency backbone)",
         "main_cohort": "YES",
         "why_missing": f"{(total or 0)-(coarse_n or 0)} windows lack a full contiguous 2h "
                        "mark chain (capture gaps / series edges)"},
        {"family": "BRTI fine state", "real_n": fine_n, "resolution": "~16s",
         "historical_role": "short-timescale BTC state",
         "main_cohort": "NO / challenger",
         "why_missing": "sub-minute samples exist only for a contiguous ~370-window block "
                        "(Sep 8–12); no fine history captured before that"},
        av_row("SPY", "risk-asset state", "likely"),
        av_row("QQQ", "risk-asset state", "likely"),
        {"family": "AV COIN", "real_n": 0, "resolution": "5m",
         "historical_role": "crypto-equity state", "main_cohort": "session-limited",
         "why_missing": "not yet fetched (MINI-0 used BTC/SPY/QQQ); equities trade only US "
                        "regular hours -> session-masked coverage < BTC"},
        {"family": "Derivatives (funding/OI/basis)", "real_n": "TBD", "resolution": "varies",
         "historical_role": "leverage / positioning", "main_cohort": "TBD",
         "why_missing": "OKX adapter serves LIVE only; historical backfill lane not yet run "
                        "(5→25→100 validity slice pending). May be partially recoverable"},
        {"family": "Options (IV/skew)", "real_n": "TBD", "resolution": "snapshot",
         "historical_role": "expectations", "main_cohort": "TBD",
         "why_missing": "AV REALTIME_OPTIONS is realtime-only (no historical vintage); a real "
                        "historical IV source is required or family is HISTORICALLY_UNAVAILABLE"},
        {"family": "News / macro", "real_n": "TBD", "resolution": "event",
         "historical_role": "context", "main_cohort": "TBD",
         "why_missing": "AV NEWS_SENTIMENT has historical range; vintage-safe backfill lane "
                        "not yet run — must respect publication timestamps"},
        {"family": "L2 / order flow (microstructure)", "real_n": "HISTORICALLY_UNAVAILABLE",
         "resolution": "tick/book", "historical_role": "microstructure",
         "main_cohort": "NO (no real history)",
         "why_missing": "no historical L2/order-book archive exists for these windows; will "
                        "not fabricate — only prospective L0 capture can build it forward"},
    ]

    doc = {
        "schema_version": "coverage-matrix-1", "generated_at": time.time(),
        "total_historical_windows": total,
        "cohorts": {
            "COARSE_BTC_STATE": {"n": coarse_n, "resolution": "15m marks",
                                 "role": "large-sample backbone", "main": True},
            "FINE_BTC_STATE": {"n": fine_n, "resolution": "~16s",
                               "role": "high-resolution challenger", "main": False},
        },
        "families": rows,
        "model_family_plan": {
            "A_COARSE_large_sample": f"~{coarse_n} windows, 15m-mark BTC state + AV/news/macro",
            "B_FINE_challenger": f"~{fine_n} windows, sub-minute BTC state + microstructure where real",
            "C_COARSE_plus_HF": "only if exchange archives raise fine coverage substantially",
        },
        "research_question": "Can broad cross-market/volume/derivatives/options/news/regime "
            "info add signal to a large low-frequency BTC-state backbone — and can "
            "high-resolution BTC info add enough to compensate for its much smaller N?",
        "note": "Coarse 15m-mark chain is legitimate pre-T0 BTC STATE (low bandwidth), NOT "
                "'price-path only' in the old intra-window sense. Every missing row states WHY.",
    }
    OUT.write_text(json.dumps(doc, indent=1))
    EV.emit("DISCOVERY", "Coverage / cohort matrix built", lane="L1", severity="high",
            fact=f"COARSE_BTC_STATE={coarse_n} (15m marks) ; FINE_BTC_STATE={fine_n} (~16s). "
                 f"AV MINI-0: {', '.join(f'{s} {avcov[s]['n']}/{av_cohort}' for s in avcov)}. "
                 "Options=realtime-only; L2/order-flow=HISTORICALLY_UNAVAILABLE.",
            interpretation="Two scientifically distinct cohorts drive two model families "
                           "(A large-coarse, B fine-challenger). The research question is "
                           "whether richer info beats the low-frequency backbone.",
            next_action="build COARSE + FINE feature factories to these coverages; run "
                        "derivatives/news validity slices; then dataset V1 + integrity + split.",
            files=[str(OUT.relative_to(ROOT))])
    print(f"COVERAGE_MATRIX: COARSE={coarse_n} FINE={fine_n}; {len(rows)} family rows")
    for r in rows:
        print(f"  {r['family']:34} N={str(r['real_n']):22} {r['resolution']:10} main={r['main_cohort']}")


if __name__ == "__main__":
    build()
