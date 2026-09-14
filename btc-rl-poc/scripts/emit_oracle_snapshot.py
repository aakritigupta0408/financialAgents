"""DT-07 §17-33 — ORACLE snapshot. One backend response for the ORACLE page's six
subtabs (Input Data / Feature Processing / Modelling / Output Processing /
Sevs & Tickets / Graveyard). Precomputes EVERY aggregate server-side (e.g. mean
BSS across folds) so the page renders with zero science in JavaScript.

Reads existing canonical backend artifacts (already-computed) + the system DAG.
Writes results/oracle_snapshot.json.
"""
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from btc_rl import metrics as M  # noqa: E402

RES = ROOT / "results"


def _j(name, root=RES):
    try:
        return json.loads((root / name).read_text())
    except Exception:
        return None


def _input_data():
    """§18-21 — active inputs with role + health. BRTI prominent (§19)."""
    rh = _j("brti_runtime_health.json") or {}
    dh = _j("data_health.json") or {}
    brti = {
        "id": "BRTI", "role": "CONTRACT_TRUTH", "prominent": True,
        "connected": rh.get("connected"), "rest_ok": rh.get("rest_ok"),
        "history_ok": rh.get("history_ok"), "age_s": rh.get("age_s"),
        "observed_cadence_hz": rh.get("observed_cadence_hz"),
        "expected_cadence_hz": rh.get("expected_cadence_hz"),
        "gaps": rh.get("gaps"),
        "runtime_activation": ("ON" if (rh.get("migration") or {}).get("runtime_enabled")
                               else "OFF / ACTIVATION_PENDING")}
    sources = [
        {"id": "Kalshi", "role": "MARKET_PRICE_INPUT+EXECUTION_INPUT"},
        {"id": "Coinbase", "role": "INDEPENDENT_PREDICTIVE_INPUT"},
        {"id": "OKX", "role": "INDEPENDENT_PREDICTIVE_INPUT"},
        {"id": "Deribit", "role": "INDEPENDENT_PREDICTIVE_INPUT"},
        {"id": "mempool/FnG", "role": "MONITORING_ONLY"},
        {"id": "AlphaVantage", "role": "INDEPENDENT_PREDICTIVE_INPUT (SCREEN_ONLY)"},
    ]
    return {"brti": brti, "sources": sources, "data_health": dh,
            "provenance": "OBSERVED (results/brti_runtime_health.json, data_health.json)"}


def _modelling():
    """§25-29 — model cards; precompute mean BSS across folds HERE (kills the
    tiers.html/home.html aggBSS leak)."""
    off = _j("model_offline.json") or {}
    onl = _j("model_online.json") or {}
    reg = _j("model_registry.json") or {}
    cards = []
    models = off.get("models") if isinstance(off, dict) else None
    if isinstance(models, dict):
        for mid, m in models.items():
            folds = m.get("folds") or []
            bsss = [f.get("bss") for f in folds if isinstance(f, dict) and f.get("bss") is not None]
            mean_bss = round(sum(bsss) / len(bsss), 4) if bsss else None
            cards.append({"model_id": mid, "offline_mean_bss": mean_bss,
                          "n_folds": len(folds),
                          "provenance": "DERIVED (btc_rl.metrics owner; mean over folds)"})
    return {"model_cards": cards, "registry_count": len((reg or {}).get("models", []) or []),
            "online": onl, "loss_functions_visible": True,
            "provenance": "backend model_offline/online + registry"}


def _feature_processing():
    fm = _j("feature_monitor.json") or {}
    return {"feature_monitor": fm, "provenance": "OBSERVED (feature_monitor.json)",
            "note": "distributions/calibration/drift precomputed server-side"}


def _output_processing():
    return {"outputs": [
        {"id": "p_mech", "source": "MECH_FAIR_BRTI"},
        {"id": "p_oracle", "source": "recalibrated MECH_FAIR_BRTI (oracle_frozen)"},
        {"id": "oracle_delta", "source": "p_oracle - p_mech"},
        {"id": "p_market_reference", "source": "Kalshi mid"},
        {"id": "lock_state", "source": "fail-closed gate"},
        {"id": "exec_score", "source": "EXEC-TIMING (offline)"}],
        "spec_hash": (_j("oracle_frozen.json", RES.parent / "research" / "oracle") or {}).get("spec_hash"),
        "provenance": "backend model outputs"}


def _sevs():
    """§31-32 — one queue; DT-01 visible until activation (§51)."""
    inc = []
    p = RES / "incidents.jsonl"
    if p.exists():
        for l in p.open():
            l = l.strip()
            if l:
                try:
                    inc.append(json.loads(l))
                except json.JSONDecodeError:
                    pass
    dt01 = {"id": "DT-01", "category": "ARCHITECTURE_DRIFT", "severity": "SEV-1",
            "status": "WIRED_PENDING_ACTIVATION",
            "summary": "runtime contract truth: exact-BRTI settlement wired, flag OFF "
                       "in live daemon (EXACT_BRTI_RUNTIME_ENABLED=0)",
            "checkpoint": "RUNTIME_CONTRACT_TRUTH = PASS_WITH_WATCH"}
    return {"incidents": inc[-25:], "architecture_watch": [dt01],
            "open_count": (_j("current_truth.json", RES.parent / "research") or {}).get("open_incidents"),
            "provenance": "OBSERVED (incidents.jsonl) + architecture checkpoint"}


def _graveyard():
    inv = _j("inventories.json", ROOT / "architecture") or {}
    return {"retired_models": (inv.get("models") or {}).get("retired_in_code"),
            "retired_traders": (inv.get("traders") or {}).get("retired"),
            "retired_treatments": (inv.get("experiments") or {}).get("retired_treatments"),
            "wait_for_dip": (inv.get("experiments") or {}).get("wait_for_dip"),
            "provenance": "architecture/inventories.json"}


def main():
    dag = _j("architecture_dag.json") or {}
    snap = {
        "schema_version": "oracle-1",
        "metric_definition_version": "econ-1/prob-1",
        "generated_at": time.time(),
        "subtabs": {
            "input_data": _input_data(),
            "feature_processing": _feature_processing(),
            "modelling": _modelling(),
            "output_processing": _output_processing(),
            "sevs_and_tickets": _sevs(),
            "graveyard": _graveyard()},
        "dag": {"nodes": dag.get("nodes", []), "edges": dag.get("edges", []),
                "drift_edges": dag.get("edges_that_should_be_removed_but_still_exist", [])},
        "note": "ORACLE renders this; no scientific/economic formula runs in the browser.",
    }
    (RES / "oracle_snapshot.json").write_text(json.dumps(snap, indent=1))
    mc = snap["subtabs"]["modelling"]["model_cards"]
    print(f"oracle_snapshot: {len(mc)} model cards, "
          f"{len(snap['dag']['nodes'])} DAG nodes, "
          f"{snap['subtabs']['sevs_and_tickets']['open_count']} open incidents")


if __name__ == "__main__":
    main()
