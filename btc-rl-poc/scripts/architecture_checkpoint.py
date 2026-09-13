"""ARCHITECTURE_CHECKPOINT (§30-36) — the recurring, machine-adjudicated audit
that prevents architecture drift from silently accumulating.

It reconstructs machine-checkable facts from ACTUAL code + registries (not docs),
compares them to the intended architecture, and emits PASS / PASS_WITH_WATCH /
FAIL. Run after every material change (see architecture/governance.json counter),
or on demand:

    python3 scripts/architecture_checkpoint.py            # run + adjudicate
    python3 scripts/architecture_checkpoint.py --bump "reason"   # +1 material change

Writes a timestamped snapshot to architecture/audits/<utc>/ and updates
architecture/latest.json. FAIL blocks treatment->control promotion (§34).

Machine checks (each -> PASS/WATCH/FAIL):
  FORBIDDEN_MARKET_TO_ORACLE  oracle modules read no Kalshi field as input
  FORBIDDEN_TESTLABEL_FIT     recalibration/scaler fit uses train split only
  LEAK_GUARD_PRESENT          prospective capture rejects settlement fields
  RUNTIME_CONTRACT_TRUTH      does the live daemon use exact BRTI as contract truth?
  REGISTRY_RECONCILED         model_registry entries have artifacts on disk
  CURRENT_TRUTH_FRESH         current_truth.json regenerates without UNKNOWN core
"""
import json
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ARCH = ROOT / "architecture"
AUDITS = ARCH / "audits"
GOV = ARCH / "governance.json"


def _read(p):
    try:
        return (ROOT / p).read_text()
    except Exception:
        return ""


def _jload(p):
    try:
        return json.loads((ROOT / p).read_text())
    except Exception:
        return None


# ── machine checks ────────────────────────────────────────────────────────────
def check_forbidden_market_to_oracle():
    fe = _jload("architecture/forbidden_edges.json")
    edge = next(e for e in fe["edges"] if e["id"] == "MARKET_TO_ORACLE")
    hits = []
    for mod in edge["oracle_modules"]:
        src = _read(mod)
        # strip comments/docstrings crudely: only flag tokens used as dict/attr reads
        for tok in edge["forbidden_input_tokens"]:
            # a forbidden INPUT looks like r["k_prob"], .k_prob, kwargs k_prob=
            for m in re.finditer(rf'(\[[\'"]{tok}[\'"]\]|\.{tok}\b|\b{tok}\s*=)', src):
                line = src[:m.start()].count("\n") + 1
                ctx = src.splitlines()[line - 1].strip()
                # allow lines that only SCORE/compare (contain 'brier', 'gap', 'compare', 'p_market')
                if any(w in ctx.lower() for w in ("brier", "gap", "compare", "vs ", "p_market",
                                                  "kalshi_ref", "# ", "score")):
                    continue
                hits.append({"module": mod, "line": line, "token": tok, "ctx": ctx[:90]})
    # DT-13: oracle FEATS lists must not include a Kalshi token (silent conversion risk)
    feats_hits = []
    for mod in ("scripts/oracle_residual_brti.py", "scripts/oracle_residual.py",
                "scripts/oracle.py"):
        src = _read(mod)
        for m in re.finditer(r"FEATS\s*=\s*\[(.*?)\]", src, re.DOTALL):
            body = m.group(1)
            if re.search(r"k_prob|kalshi|yes_ask|yes_bid|mkt_p", body):
                feats_hits.append({"module": mod, "feats": body[:120]})
    all_hits = hits + [{"module": h["module"], "feats_list_has_kalshi": h["feats"]} for h in feats_hits]
    return ("FAIL" if all_hits else "PASS", {"violations": all_hits,
            "note": "oracle modules must not read a Kalshi field as p_mech/p_oracle input, "
                    "and no oracle FEATS list may contain a Kalshi token"})


def check_runtime_contract_truth():
    """DRIFT DETECTOR: is the live daemon settling on EXACT BRTI, or a proxy?"""
    online = _read("btc_rl/online.py")
    imports_exact = bool(re.search(r"data\.adapters\.brti|from\s+data\.adapters\s+import\s+brti|"
                                   r"prospective_capture", online))
    uses_composite = bool(re.search(r"fetch_brti_composite|brti_composite", online))
    settle_on_candle = bool(re.search(r"settle_bar\[.close.\]\s*>=|by_ts\.get\(close_ts", online))
    if imports_exact and not settle_on_candle:
        return "PASS", {"detail": "runtime imports exact BRTI and does not settle on a candle close"}
    return "FAIL", {
        "detail": "RUNTIME uses a BRTI proxy / Coinbase-candle settlement, not exact BRTI",
        "imports_exact_brti_adapter": imports_exact,
        "uses_4venue_composite": uses_composite,
        "settles_on_coinbase_candle": settle_on_candle,
        "migration": "wire data/adapters/brti.py + btc_rl/prospective_capture.py into "
                     "btc_rl/online.py contract-state + settlement; gate via "
                     "PROSPECTIVE_CAPTURE_ENABLED. See architecture/dangling_threads.json."}


def check_testlabel_fit():
    """EVALUATION recalibration/scaler fits must occur AFTER a window split (train
    only). freeze_oracle.py is EXEMPT: it is the production freeze, deliberately fit
    on all history for live serving (not scored OOS), so a split would be wrong."""
    eval_mods = ("scripts/mech_recalibration.py", "scripts/oracle_residual_brti.py",
                 "scripts/disagreement_edge.py")
    bad = []
    for mod in eval_mods:
        src = _read(mod)
        if not src:
            continue
        has_split = bool(re.search(r"trw|train.*split|order\[:int", src))
        if re.search(r"\.fit\(", src) and not has_split:
            bad.append(mod)
    return ("FAIL" if bad else "PASS",
            {"modules_fitting_without_split": bad,
             "exempt_production_freeze": ["scripts/freeze_oracle.py"],
             "note": "evaluation isotonic/scaler .fit must use train-window rows only"})


def check_leak_guard():
    src = _read("btc_rl/prospective_capture.py")
    ok = "_LEAK_FIELDS" in src and "leak-guard" in src and '"exact_yes": None' in src
    return ("PASS" if ok else "FAIL",
            {"leak_guard_present": ok,
             "note": "prospective capture must reject settlement fields at decision time"})


def check_registry_reconciled():
    reg = _jload("results/model_registry.json") or {}
    models = reg.get("models", reg if isinstance(reg, list) else [])
    if isinstance(models, dict):
        models = list(models.values())
    missing = []
    for m in (models or []):
        if not isinstance(m, dict):
            continue
        art = m.get("artifact") or m.get("path") or m.get("file")
        if isinstance(art, dict):                 # registry stores {file, mtime, sha256_16}
            art = art.get("file")
        if isinstance(art, str) and art:
            if not (ROOT / art).exists() and not (ROOT / "results" / art).exists():
                missing.append(art)
    return ("WATCH" if missing else "PASS",
            {"registered_models": len(models or []), "missing_artifacts": missing})


def check_current_truth():
    r = subprocess.run([sys.executable, "scripts/emit_current_truth.py"],
                       cwd=ROOT, capture_output=True, text=True, timeout=120)
    ct = _jload("research/current_truth.json") or {}
    core = ("bottleneck", "runtime_mode", "traders", "experiments")
    unknown_core = [k for k in core if ct.get(k) in (None, "UNKNOWN")]
    return ("WATCH" if unknown_core else "PASS",
            {"regenerated": r.returncode == 0, "unknown_core_fields": unknown_core})


def check_guardrails():
    """§24 — PAPER/SIMULATION/REAL-MONEY-DISABLED must hold at runtime boundaries
    and must NOT alter labels/features/oracle/experiment stats."""
    std = ROOT / "REAL_MONEY_EQUIVALENT_STANDARD.yaml"
    online = _read("btc_rl/online.py")
    paper = bool(re.search(r"paper|simulat", online, re.I))
    # a real-money execution switch must not be enabled
    live_money = bool(re.search(r"REAL_MONEY_ENABLED\s*=\s*True|LIVE_TRADING\s*=\s*True", online))
    ok = std.exists() and paper and not live_money
    return ("PASS" if ok else "FAIL",
            {"standard_present": std.exists(), "paper_markers": paper,
             "real_money_enabled": live_money,
             "note": "paper/sim only; real money disabled; guardrails must not touch "
                     "labels/features/oracle"})


def check_incident_rules():
    """§23 — incident monitoring must match current architecture; a BRTI-health
    alert is required now that BRTI is contract-critical."""
    health_files = [p for p in ("results/data_health.json", "results/brti_health.json",
                                "results/incidents.jsonl", "results/invariants.json")
                    if (ROOT / p).exists()]
    brti_monitored = (ROOT / "results/brti_health.json").exists()
    return ("PASS" if brti_monitored else "WATCH",
            {"monitoring_present": health_files, "brti_health_alert": brti_monitored,
             "note": "missing a runtime BRTI-health alert is WATCH until exact BRTI is "
                     "wired into the daemon (DT-01)"})


def check_change_impact():
    """§35/§36 — if a change-impact declaration exists, verify observed state matches
    it (observed-vs-declared drift). Absent declaration = nothing to verify = PASS."""
    decl = _jload("architecture/change_impact.json")
    if not decl:
        return "PASS", {"declaration": None, "note": "no pending change_impact.json to verify"}
    dag = _jload("architecture/system_dag.json") or {}
    edge_keys = {(e["from"], e["to"]) for e in dag.get("edges", [])}
    drift = []
    for e in decl.get("edges_removed", []):
        key = (e.get("from"), e.get("to"))
        if key in edge_keys:
            drift.append({"declared_removed": key, "still_present_in_dag": True})
    return ("FAIL" if drift else "PASS",
            {"change_id": decl.get("change_id"), "observed_vs_declared_drift": drift,
             "note": "declared-removed edges must be absent from system_dag.json"})


CHECKS = [
    ("FORBIDDEN_MARKET_TO_ORACLE", check_forbidden_market_to_oracle),
    ("FORBIDDEN_TESTLABEL_FIT", check_testlabel_fit),
    ("LEAK_GUARD_PRESENT", check_leak_guard),
    ("RUNTIME_CONTRACT_TRUTH", check_runtime_contract_truth),
    ("REGISTRY_RECONCILED", check_registry_reconciled),
    ("CURRENT_TRUTH_FRESH", check_current_truth),
    ("PRODUCT_GUARDRAILS", check_guardrails),
    ("INCIDENT_RULES_MATCH_ARCH", check_incident_rules),
    ("CHANGE_IMPACT_MATCHES_OBSERVED", check_change_impact),
]


def emit_artifacts(outdir):
    """§32 — emit the full required checkpoint file set into the audit dir."""
    inv = _jload("architecture/inventories.json") or {}
    ct = _jload("research/current_truth.json") or {}
    # split inventories
    _dump(outdir / "feature_inventory.json", inv.get("features"))
    _dump(outdir / "model_inventory.json", inv.get("models"))
    _dump(outdir / "trader_inventory.json", inv.get("traders"))
    _dump(outdir / "experiment_inventory.json", inv.get("experiments"))
    _dump(outdir / "legacy_inventory.json",
          {"legacy_filters": inv.get("legacy_filters"),
           "retired_models": (inv.get("models") or {}).get("retired_in_code"),
           "retired_traders": (inv.get("traders") or {}).get("retired"),
           "retired_treatments": (inv.get("experiments") or {}).get("retired_treatments")})
    # machine-readable architecture diff (companion to the .md)
    dag = _jload("architecture/system_dag.json") or {}
    _dump(outdir / "architecture_diff.json", {
        "edges_that_should_be_removed_but_still_exist":
            dag.get("edges_that_should_be_removed_but_still_exist", []),
        "changed_semantics_source": "architecture/architecture_diff.md",
        "drift_nodes": [n for n in dag.get("nodes", []) if "DRIFT" in str(n.get("status"))]})
    # §16/§38 offline+online scorecard (kept separate, never combined)
    _dump(outdir / "offline_online_scorecard.json", {
        "note": "offline and online evidence kept separate (§16); highlight OFFLINE-WINNER/ONLINE-LOSER",
        "oracle_offline": {
            "MECH_FAIR_BRTI": 0.1989, "MECH_FAIR_BRTI_recalibrated": 0.1912,
            "KALSHI": 0.1878, "disagreement_edge": "PROMISING_PENDING_PROSPECTIVE"},
        "runtime_online": {
            "settled_windows": (ct.get("settlement_counters") or {}),
            "traders": ct.get("traders"), "service_health": ct.get("service_health"),
            "caveat": "online settlement on PROXY (Coinbase candle), not exact BRTI (DT-01)"}})
    # copy curated graph/thread/edge/generation artifacts
    for name in ("system_dag.json", "dangling_threads.json", "forbidden_edges.json",
                 "generations.json"):
        src = _jload(f"architecture/{name}")
        if src is not None:
            _dump(outdir / name, src)
    # §37 — shippable compact DAG for the developer drill-down UI (results/ is published)
    latest = _jload("architecture/latest.json") or {}
    _dump(ROOT / "results" / "architecture_dag.json", {
        "verdict": latest.get("verdict"), "checkpoint_utc": latest.get("checkpoint_utc"),
        "nodes": dag.get("nodes", []), "edges": dag.get("edges", []),
        "edges_that_should_be_removed_but_still_exist":
            dag.get("edges_that_should_be_removed_but_still_exist", []),
        "dangling_thread_count": len((_jload("architecture/dangling_threads.json") or {}).get("threads", []))})


def _dump(path, obj):
    path.write_text(json.dumps(obj, indent=1))


def governance(bump_reason=None):
    g = _jload("architecture/governance.json") or {
        "material_change_counter": 0, "checkpoint_every_n": 3,
        "last_checkpoint_utc": None, "log": []}
    if bump_reason:
        g["material_change_counter"] += 1
        g["log"].append({"reason": bump_reason})
        GOV.write_text(json.dumps(g, indent=1))
        print(f"material change #{g['material_change_counter']}: {bump_reason}")
        due = g["material_change_counter"] % g["checkpoint_every_n"] == 0
        print("checkpoint DUE" if due else
              f"{g['checkpoint_every_n'] - g['material_change_counter'] % g['checkpoint_every_n']} to next checkpoint")
    return g


def main():
    if len(sys.argv) > 2 and sys.argv[1] == "--bump":
        governance(sys.argv[2]); return
    results, worst = {}, "PASS"
    order = {"PASS": 0, "WATCH": 1, "PASS_WITH_WATCH": 1, "FAIL": 2}
    for name, fn in CHECKS:
        try:
            status, detail = fn()
        except Exception as e:
            status, detail = "FAIL", {"error": str(e)[:200]}
        results[name] = {"status": status, **detail}
        if order.get(status, 0) > order.get(worst, 0):
            worst = status
    verdict = "FAIL" if worst == "FAIL" else ("PASS_WITH_WATCH" if worst in ("WATCH", "PASS_WITH_WATCH") else "PASS")
    # UTC stamp without Date.now shenanigans (time.time allowed in normal python)
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    outdir = AUDITS / stamp
    outdir.mkdir(parents=True, exist_ok=True)
    g = governance()
    g["last_checkpoint_utc"] = stamp
    GOV.write_text(json.dumps(g, indent=1))
    report = {"checkpoint_utc": stamp, "verdict": verdict, "checks": results,
              "material_change_counter": g["material_change_counter"],
              "fail_blocks_promotion": verdict == "FAIL",
              "note": "machine-adjudicated architecture checkpoint (scripts/architecture_checkpoint.py)"}
    (outdir / "checkpoint.json").write_text(json.dumps(report, indent=1))
    (ARCH / "latest.json").write_text(json.dumps(report, indent=1))
    emit_artifacts(outdir)                       # §32 full required file set
    print(f"ARCHITECTURE CHECKPOINT — {verdict}   ({stamp})")
    for name, r in results.items():
        print(f"  {r['status']:14s} {name}")
        if r["status"] == "FAIL":
            d = {k: v for k, v in r.items() if k != "status"}
            print(f"                 -> {json.dumps(d)[:160]}")
    print(f"  material changes since start: {g['material_change_counter']}")


if __name__ == "__main__":
    main()
