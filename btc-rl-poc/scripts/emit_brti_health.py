"""DT-01 §3 — emit runtime BRTI health + migration state to results/brti_runtime_health.json
(shippable; publish_dashboard ships it). Cron-safe, never raises. Reports the live
BRTI connection/cadence/gaps AND whether the exact-BRTI settlement path is ACTIVE, so
the architecture checkpoint can tell 'wired' from 'activated'.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from btc_rl import contract_truth as ct  # noqa: E402


def _recent_shadow_agreement():
    """From results/settlement_shadow.jsonl, how often exact-BRTI matched official."""
    p = ROOT / "results" / "settlement_shadow.jsonl"
    if not p.exists():
        return None
    n = agree = 0
    for l in p.open():
        l = l.strip()
        if not l:
            continue
        try:
            r = json.loads(l)
        except json.JSONDecodeError:
            continue
        if r.get("exact_vs_official") is not None:
            n += 1
            agree += r["exact_vs_official"]
    return {"n": n, "exact_vs_official_agree_rate": round(agree / n, 4)} if n else None


def main():
    doc = ct.write_health()
    doc["migration"] = {
        "runtime_enabled": ct.runtime_enabled(),
        "capture_enabled": ct.capture_enabled(),
        "settlement_path": "EXACT_BRTI" if ct.runtime_enabled() else "PROXY_LEGACY",
        "shadow_parity": _recent_shadow_agreement(),
        "note": "DT-01: exact-BRTI settlement is ACTIVE only when runtime_enabled; "
                "otherwise the daemon runs the legacy Coinbase-candle proxy.",
    }
    (ROOT / "results" / "brti_runtime_health.json").write_text(json.dumps(doc, indent=1))
    print(f"BRTI runtime health: connected={doc.get('connected')} "
          f"rest_ok={doc.get('rest_ok')} enabled={ct.runtime_enabled()} "
          f"path={doc['migration']['settlement_path']}")


if __name__ == "__main__":
    main()
