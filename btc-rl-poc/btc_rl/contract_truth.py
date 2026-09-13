"""DT-01 — EXACT-BRTI CONTRACT TRUTH for the live runtime.

The steady-state defect this closes: research uses exact CF-BRTI as the KXBTC15M
contract truth, but the daemon settles on a Coinbase-candle proxy. This module is
the single runtime owner of contract truth so the two converge. BRTI is
CONTRACT_TRUTH; Coinbase/Binance/OKX/Kraken are INDEPENDENT_PREDICTIVE_INPUT or an
EXPLICIT, versioned fallback — never a silent substitute.

Exact KXBTC15M rule (research/contract_specs/KXBTC15M_2026-09.json):
  target            = mean BRTI over [open-60, open)          (== floor_strike)
  settlement value  = mean BRTI over [close-60, close)        (== expiration_value)
  YES iff settlement_value >= target

Flag-gated: EXACT_BRTI_RUNTIME_ENABLED (default FALSE) so code can deploy dark,
be smoke-tested with the flag off, then activated without a code edit.

Failover is EXPLICIT (§4): every result carries contract_truth_quality, one of
  EXACT_BRTI      both 60s windows reconstructed from BRTI with enough samples
  PROXY_DEGRADED  BRTI incomplete; caller MAY use a versioned proxy, but must
                  surface the degraded quality to health/eval and MUST NOT present
                  it as exact
  UNAVAILABLE     BRTI cannot be read; settlement-dependent scientific assertions
                  fail closed until the official Kalshi outcome is available.
"""
import os
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

EXACT_BRTI = "EXACT_BRTI"
PROXY_DEGRADED = "PROXY_DEGRADED"
UNAVAILABLE = "UNAVAILABLE"
MIN_SAMPLES = 30           # of ~300 expected in a 60s window; below this = degraded
EXPECTED_CADENCE_HZ = 5.0


def runtime_enabled():
    return os.environ.get("EXACT_BRTI_RUNTIME_ENABLED", "").strip() in ("1", "true", "yes")


def capture_enabled():
    # BRTI capture can be turned on before the settlement path is switched
    return runtime_enabled() or \
        os.environ.get("EXACT_BRTI_CAPTURE_ENABLED", "").strip() in ("1", "true", "yes")


def _brti():
    import data.adapters.brti as b
    return b


def _avg_60s(ts):
    """(avg, n) of BRTI over [ts-60, ts) or (None, 0) on failure."""
    try:
        return _brti().avg_60s_ending(int(ts))
    except Exception:
        return None, 0


def contract_state(open_ts, close_ts, now_ts=None):
    """Live ContractState at decision time. target is known once the window has
    opened; current_brti is the latest observed BRTI. Never raises — degrades."""
    now = int(now_ts if now_ts is not None else time.time())
    target, n_open = _avg_60s(open_ts)
    cur = None
    try:
        b = _brti()
        samples = b.hour_samples(now)
        past = [(t, v) for t, v in samples if t <= now]
        cur = max(past, key=lambda x: x[0])[1] if past else None
    except Exception:
        cur = None
    if target is None or n_open < MIN_SAMPLES:
        quality = PROXY_DEGRADED if cur is not None else UNAVAILABLE
    elif cur is None:
        quality = PROXY_DEGRADED
    else:
        quality = EXACT_BRTI
    return {
        "contract_truth_quality": quality,
        "official_target": round(target, 4) if target is not None else None,
        "current_brti": round(cur, 4) if cur is not None else None,
        "brti_distance_to_target": (round(cur - target, 4)
                                    if (cur is not None and target is not None) else None),
        "time_remaining_s": max(0, int(close_ts) - now),
        "target_samples": n_open,
    }


def settle(open_ts, close_ts):
    """Determine the exact contract outcome from BRTI. Returns
      {outcome, contract_truth_quality, target, settlement_value, D, ...}.
    outcome is None (fail-closed) unless quality == EXACT_BRTI (§4)."""
    target, n_open = _avg_60s(open_ts)
    settle_val, n_close = _avg_60s(close_ts)
    if (target is None or settle_val is None
            or n_open < MIN_SAMPLES or n_close < MIN_SAMPLES):
        return {"outcome": None,
                "contract_truth_quality": (UNAVAILABLE if target is None or settle_val is None
                                           else PROXY_DEGRADED),
                "target": round(target, 4) if target is not None else None,
                "settlement_value": round(settle_val, 4) if settle_val is not None else None,
                "target_samples": n_open, "settlement_samples": n_close,
                "note": "fail-closed: exact BRTI settlement unavailable; defer to "
                        "official Kalshi outcome when available"}
    D = settle_val - target
    return {"outcome": int(settle_val >= target),      # YES iff close_avg >= open_avg
            "contract_truth_quality": EXACT_BRTI,
            "target": round(target, 4), "settlement_value": round(settle_val, 4),
            "D": round(D, 4), "target_samples": n_open, "settlement_samples": n_close}


def health():
    """§3 — runtime BRTI health. Never raises."""
    h = {"connected": False, "last_observation": None, "age_s": None,
         "expected_cadence_hz": EXPECTED_CADENCE_HZ, "observed_cadence_hz": None,
         "gaps": None, "rest_ok": False, "history_ok": False,
         "runtime_enabled": runtime_enabled(), "capture_enabled": capture_enabled()}
    try:
        b = _brti()
        st = b.probe()
        h["rest_ok"] = st.get("state") == "AVAILABLE"
        payload = (st.get("sample") or {}).get("data", {}).get("payload", [])
        if payload:
            ts = [x["time"] / 1000.0 for x in payload]
            h["connected"] = True
            h["last_observation"] = max(ts)
            h["age_s"] = round(time.time() - max(ts), 1)
            span = max(ts) - min(ts)
            h["observed_cadence_hz"] = round(len(ts) / span, 2) if span > 0 else None
            # count >2s gaps as a coarse gap signal
            sts = sorted(ts)
            h["gaps"] = sum(1 for i in range(1, len(sts)) if sts[i] - sts[i - 1] > 2.0)
        # history endpoint check (cheap: one recent hour)
        try:
            hs = b.hour_samples(int(time.time()) - 3600)
            h["history_ok"] = len(hs) > 0
        except Exception:
            h["history_ok"] = False
    except Exception as e:
        h["error"] = str(e)[:120]
    return h


WINDOW_S = 900          # KXBTC15M window length (15 min) — derives open from close


def resolve_outcome(ticker, close_ts, strike, legacy_outcome, log_shadow=True):
    """Single runtime authority for the binary contract outcome (§1, §5). Drop-in
    for every `outcome = int(settle_bar['close'] >= strike)` site in online.py.

    Returns (outcome, contract_truth_quality). Semantics:
      flag OFF                       -> (legacy_outcome, "PROXY_LEGACY")  [no-op]
      flag ON, BRTI EXACT_BRTI       -> (exact_outcome,  "EXACT_BRTI")
      flag ON, BRTI incomplete       -> (legacy_outcome, "PROXY_DEGRADED") [explicit,
                                        surfaced; never presented as exact]
    When the flag is ON it also logs a legacy/exact/official shadow row so parity is
    observable before and after the switch. `legacy_outcome` is the caller's existing
    Coinbase-candle result, passed in so this function never needs candle data."""
    if not runtime_enabled():
        return legacy_outcome, "PROXY_LEGACY"
    res = settle(close_ts - WINDOW_S, close_ts)
    quality = res["contract_truth_quality"]
    exact = res["outcome"]
    if log_shadow:
        try:
            shadow_settle(close_ts - WINDOW_S, close_ts, legacy_outcome,
                          official_outcome=None, ticker=ticker, log=True)
        except Exception:
            pass
    if quality == EXACT_BRTI and exact is not None:
        return exact, EXACT_BRTI
    return legacy_outcome, PROXY_DEGRADED          # explicit degraded fallback


def shadow_settle(open_ts, close_ts, legacy_outcome, official_outcome=None,
                  ticker=None, log=True):
    """§7 — record legacy(proxy) vs exact-BRTI vs official settlement side by side,
    so parity can be established BEFORE the switch. Append-only; never affects the
    live decision. Returns the triple + agreement flags."""
    import json
    exact = settle(open_ts, close_ts)
    rec = {"ts": time.time(), "ticker": ticker,
           "legacy_outcome": legacy_outcome,
           "exact_outcome": exact.get("outcome"),
           "exact_quality": exact.get("contract_truth_quality"),
           "official_outcome": official_outcome,
           "exact_vs_official": (None if official_outcome is None or exact.get("outcome") is None
                                 else int(exact["outcome"] == official_outcome)),
           "legacy_vs_official": (None if official_outcome is None
                                  else int(legacy_outcome == official_outcome))}
    if log:
        p = ROOT / "results" / "settlement_shadow.jsonl"
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("a") as fh:
            fh.write(json.dumps(rec) + "\n")
    return rec


def write_health(path=None):
    import json
    p = Path(path) if path else ROOT / "results" / "brti_runtime_health.json"
    doc = {"checked_ts": time.time(), **health()}
    p.write_text(json.dumps(doc, indent=1))
    return doc


if __name__ == "__main__":
    import json
    print(json.dumps(health(), indent=1))
    print("runtime_enabled:", runtime_enabled())
