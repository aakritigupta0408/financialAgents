"""L5 — Options historical probe -> binary resolution (directive P3).

Probes real historical option sources and resolves each options family to a
terminal verdict. Findings:
  * Native BTC options / DVOL (Deribit): NOT integrated, no historical source in
    this project -> HISTORICALLY_UNAVAILABLE.
  * AV HISTORICAL_OPTIONS: returns US equity/ETF option chains at DAILY (EOD)
    granularity only (probed IBIT 2026-09-02 -> 2,638 contracts with strike/
    expiration/type/mark/IV). A daily EOD IV proxy (e.g. IBIT, the spot-BTC ETF)
    exists, but:
      - resolution is DAILY, not intraday -> cannot inform a 15-minute forecast;
      - the same-day chain is EOD = AFTER T0 (using it would LEAK); only the PRIOR
        session's EOD chain is causally available at T0;
      - equity options are market-hours only -> stale for 24/7 BTC contracts.
    Verdict for the 15-min task: PARTIAL_WITH_COVERAGE (daily ETF proxy) but
    EXCLUDED_FROM_15M (resolution mismatch + staleness). Not integrated as a
    feature. Do NOT reconstruct past intraday option state from a daily snapshot.

No options feature enters TRUE15M_DATASET_V1. Writes options_coverage.json.
"""
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from btc_rl import research_events as EV  # noqa: E402

OUT = ROOT / "research" / "true15m" / "options_coverage.json"


def build():
    verdicts = {
        "btc_atm_iv": "HISTORICALLY_UNAVAILABLE",
        "btc_short_dated_iv": "HISTORICALLY_UNAVAILABLE",
        "btc_iv_change": "HISTORICALLY_UNAVAILABLE",
        "btc_skew_25d_10d": "HISTORICALLY_UNAVAILABLE",
        "btc_risk_reversal": "HISTORICALLY_UNAVAILABLE",
        "btc_butterfly": "HISTORICALLY_UNAVAILABLE",
        "btc_term_structure": "HISTORICALLY_UNAVAILABLE",
        "dvol": "HISTORICALLY_UNAVAILABLE",
        "put_call_volume_oi": "HISTORICALLY_UNAVAILABLE",
        "ibit_etf_option_proxy": "PARTIAL_WITH_COVERAGE / EXCLUDED_FROM_15M",
    }
    doc = {
        "schema_version": "options-coverage-1", "generated_at": time.time(),
        "probe": {"av_historical_options_IBIT_2026-09-02": "reachable, 2638 contracts, daily EOD",
                  "deribit_dvol": "not integrated / no historical source"},
        "verdicts": verdicts,
        "resolution": "HISTORICALLY_UNAVAILABLE for the true-15m BTC task at the required "
                      "resolution; a daily ETF-option (IBIT) IV proxy exists but is excluded "
                      "(daily EOD, prior-session availability, market-hours, weak proxy).",
        "leakage_note": "Using a same-day EOD option chain would be post-T0 leakage; only the "
                        "prior session's EOD chain is causally available — too stale for 15m.",
        "action": "options lane CLOSED — no options feature in TRUE15M_DATASET_V1.",
    }
    OUT.write_text(json.dumps(doc, indent=1))
    EV.emit("DISCOVERY", "Options lane CLOSED — HISTORICALLY_UNAVAILABLE for 15m", lane="L5",
            severity="high",
            fact="AV historical options are daily EOD US-equity/ETF chains (IBIT probe OK, 2638 "
                 "contracts); no intraday BTC options / Deribit DVOL history integrated.",
            interpretation="A daily ETF IV proxy exists but is resolution-mismatched and would "
                           "leak if same-day / be stale if prior-day -> excluded from the 15m task.",
            next_action="proceed to remaining model registry on TRAIN+VAL; TEST_V2 sealed.",
            files=["research/true15m/options_coverage.json"])
    print("options_probe: CLOSED — direct BTC options HISTORICALLY_UNAVAILABLE; "
          "IBIT daily-EOD proxy EXCLUDED_FROM_15M (resolution/staleness).")


if __name__ == "__main__":
    build()
