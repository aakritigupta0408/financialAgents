"""Freeze FEATURE_UNIVERSE_V1 — the historical feature search is closed (directive).

Stops endless feature/API fishing. Records every resolved family with its real N,
resolution, and status: TESTED_NEGATIVE (built + causally tested, no incremental
OOS value) vs HISTORICALLY_UNAVAILABLE (cannot be causally reconstructed — unknown,
NOT a failure). This distinction is explicit per the directive.
"""
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from btc_rl import research_events as EV  # noqa: E402

T = ROOT / "research" / "true15m"
OUT = T / "FEATURE_UNIVERSE_V1.json"


def build():
    families = {
        "F01_brti_coarse_state": {"N": 6149, "resolution": "15m marks", "status": "TESTED_NEGATIVE",
                                  "note": "20 brti_coarse.* features; A_CORE NO_OFFLINE_QUALIFIED"},
        "F02_brti_fine_state": {"N": 363, "resolution": "~16s", "status": "TESTED_NEGATIVE",
                                "note": "14 brti_fine.* features; Family-B challenger cohort"},
        "F21_av_equity_risk": {"N": "~1198 market-hours", "resolution": "5m", "status": "TESTED_NEGATIVE",
                               "note": "SPY/QQQ/IWM etc.; AV_NO_OOS_VALUE (leakage-corrected)"},
        "F22_av_crypto_equities": {"N": "~1198", "resolution": "5m", "status": "TESTED_NEGATIVE",
                                   "note": "COIN/MSTR/MARA/RIOT"},
        "F25_commodities": {"N": "~1198", "resolution": "5m", "status": "TESTED_NEGATIVE",
                            "note": "GLD/USO"},
        "F14_derivatives_funding": {"N": 6337, "resolution": "8h", "status": "TESTED_NEGATIVE",
                                    "note": "OKX funding level/change/z; DERIV_NO_OOS_VALUE"},
        "F28_30_news": {"N": 6337, "resolution": "event", "status": "TESTED_NEGATIVE",
                        "note": "AV BTC news counts/sentiment/diversity; NEWS_NO_OOS_VALUE"},
        "F14b_open_interest": {"status": "HISTORICALLY_UNAVAILABLE", "note": "OKX public snapshot only"},
        "F15_liquidations": {"status": "HISTORICALLY_UNAVAILABLE", "note": "~1-day retention"},
        "F14c_long_short_taker": {"status": "HISTORICALLY_UNAVAILABLE", "note": "rubik short retention"},
        "F14d_basis": {"status": "HISTORICALLY_UNAVAILABLE", "note": "no PIT paired mark+index"},
        "F16_options_iv_skew_dvol": {"status": "HISTORICALLY_UNAVAILABLE",
                                     "note": "no intraday BTC options / Deribit DVOL; AV daily-EOD ETF only"},
        "F12_microstructure_l2": {"status": "HISTORICALLY_UNAVAILABLE", "note": "no L2/order-flow archive"},
    }
    tested = [k for k, v in families.items() if v.get("status") == "TESTED_NEGATIVE"]
    unavail = [k for k, v in families.items() if v.get("status") == "HISTORICALLY_UNAVAILABLE"]
    doc = {"schema_version": "feature-universe-1", "generated_at": time.time(),
           "status": "FROZEN", "families": families,
           "tested_negative": tested, "historically_unavailable": unavail,
           "principle": "Distinction is explicit: TESTED_NEGATIVE families were built and "
                        "causally tested (no incremental OOS value); HISTORICALLY_UNAVAILABLE "
                        "families are UNKNOWN (cannot be causally reconstructed) — NOT failures.",
           "closed": "Historical feature/API search is CLOSED. No more source hunting. Focus "
                     "shifts to whether the models can extract weak signal (L12 diagnostics)."}
    OUT.write_text(json.dumps(doc, indent=1))
    EV.emit("DISCOVERY", "FEATURE_UNIVERSE_V1 FROZEN — feature search closed", lane="L1",
            severity="high",
            fact=f"{len(tested)} families TESTED_NEGATIVE, {len(unavail)} HISTORICALLY_UNAVAILABLE.",
            interpretation="Unavailable != failed. The data-side search is closed; the question "
                           "is now whether the training stack can extract weak signal.",
            next_action="L12 model-failure diagnostics govern any INFORMATION_LIMITED claim.",
            files=["research/true15m/FEATURE_UNIVERSE_V1.json"])
    print(f"FEATURE_UNIVERSE_V1 FROZEN: {len(tested)} tested-negative, {len(unavail)} unavailable")


if __name__ == "__main__":
    build()
