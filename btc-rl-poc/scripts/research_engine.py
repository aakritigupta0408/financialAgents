"""Concurrent research scheduler (directive §5-11, §43-45).

A real job DAG: independent lanes run simultaneously (ThreadPoolExecutor), a job
starts only once all its dependencies COMPLETE, and jobs that would race on the
same canonical artifact are ordered by an explicit dependency edge. Priority
orders the queue when worker slots are scarce. Every transition emits a research
event; live state is written to results/research_jobs.json for the narrator.

L0 LIVE CAPTURE is NOT managed here — it is the always-on daemon; the scheduler
only ever reflects its status, never starts/stops it.

Usage: python3 scripts/research_engine.py            (run the current P1 graph)
       python3 scripts/research_engine.py --plan      (print the graph, run nothing)
"""
import concurrent.futures as cf
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from btc_rl import research_events as EV  # noqa: E402

STATE = ROOT / "results" / "research_jobs.json"
MAX_WORKERS = 6                     # resource cap (§44)

# ---- P1 job graph (all runnable now; deps enforce ordering & artifact safety) ----
JOBS = [
    {"id": "coarse_features", "lane": "L1", "priority": 0,
     "title": "COARSE feature factory (brti_coarse.* on ~6,189 cohort)",
     "cmd": ["python3", "scripts/coarse_feature_factory.py"],
     "deps": [], "artifact": "coarse_features"},
    {"id": "fine_features", "lane": "L7", "priority": 1,
     "title": "FINE feature factory (brti_fine.* on ~363-window Family-B cohort)",
     "cmd": ["python3", "scripts/fine_feature_factory.py"],
     "deps": [], "artifact": "fine_features"},
    {"id": "coarse_semantics", "lane": "L7", "priority": 0,
     "title": "COARSE_MARK_AVAILABILITY_SEMANTICS integrity",
     "cmd": ["python3", "scripts/coarse_mark_semantics.py"],
     "deps": [], "artifact": "coarse_mark_semantics"},
    {"id": "av_expand", "lane": "L2", "priority": 1,
     "title": "Alpha Vantage cross-asset backfill (MINI-1, 25 windows)",
     "cmd": ["python3", "scripts/av_backfill.py", "--mini", "1"],
     "deps": [], "artifact": "av_backfill"},
    {"id": "coarse_reconcile", "lane": "L7", "priority": 0,
     "title": "COARSE_FACTORY_COHORT_RECONCILIATION (6,189 vs 6,200)",
     "cmd": ["python3", "scripts/coarse_cohort_reconciliation.py"],
     "deps": ["coarse_features"], "artifact": "coarse_cohort_reconciliation"},
    {"id": "fine_reconcile", "lane": "L7", "priority": 1,
     "title": "FINE_FACTORY_COHORT_RECONCILIATION (363 vs 365)",
     "cmd": ["python3", "scripts/fine_cohort_reconciliation.py"],
     "deps": ["fine_features"], "artifact": "fine_cohort_reconciliation"},
    {"id": "fine_ret5m_guard", "lane": "L7", "priority": 2,
     "title": "brti_fine.ret_5m elapsed-time guard",
     "cmd": ["python3", "scripts/fine_ret5m_guard.py"],
     "deps": ["fine_features"], "artifact": "fine_ret5m_guard"},
    {"id": "research_intel", "lane": "L11", "priority": 2,
     "title": "Research Intelligence queue (RI-001..009, specs only)",
     "cmd": ["python3", "scripts/research_intelligence.py"],
     "deps": [], "artifact": "research_intelligence"},
    {"id": "av_full", "lane": "L2", "priority": 1,
     "title": "AV full backfill + A_AV cohort + incremental test",
     "cmd": ["python3", "scripts/av_full_backfill.py"],
     "deps": ["coarse_features"], "artifact": "av_full"},
    {"id": "deriv_backfill", "lane": "L4", "priority": 1,
     "title": "Derivatives backfill + A_DERIV incremental test (OKX funding)",
     "cmd": ["python3", "scripts/derivatives_backfill.py"],
     "deps": ["coarse_features"], "artifact": "deriv_backfill"},
    {"id": "news_backfill", "lane": "L6", "priority": 1,
     "title": "News backfill + A_NEWS incremental test (AV, published_at<=T0)",
     "cmd": ["python3", "scripts/news_backfill.py"],
     "deps": ["coarse_features"], "artifact": "news_backfill"},
    {"id": "options_probe", "lane": "L5", "priority": 2,
     "title": "Options historical probe -> binary resolution",
     "cmd": ["python3", "scripts/options_probe.py"],
     "deps": [], "artifact": "options_probe"},
    {"id": "offline_program", "lane": "L8", "priority": 1,
     "title": "TRUE15M offline program (dataset->split->ladder->sealed test->verdict, A_CORE)",
     "cmd": ["python3", "scripts/true15m_offline_program.py"],
     "deps": ["coarse_features", "coarse_reconcile"], "artifact": "offline_program"},
    {"id": "distributional_models", "lane": "L8", "priority": 2,
     "title": "Level-5 distributional models (CatBoost/NGBoost)",
     "cmd": ["python3", "scripts/distributional_models.py"],
     "deps": ["offline_program"], "artifact": "distributional_models"},
    {"id": "model_diagnostics", "lane": "L12", "priority": 1,
     "title": "L12 model-failure diagnostics (memorization ladder + optimization)",
     "cmd": ["python3", "scripts/model_failure_diagnostics.py"],
     "deps": ["offline_program"], "artifact": "model_failure_diagnostics"},
    {"id": "freeze_universe", "lane": "L1", "priority": 1,
     "title": "Freeze FEATURE_UNIVERSE_V1 (feature search closed)",
     "cmd": ["python3", "scripts/freeze_feature_universe.py"],
     "deps": [], "artifact": "feature_universe"},
    {"id": "settled_refetch", "lane": "L9", "priority": 0,
     "title": "Refresh settled KXBTC15M feed (Kalshi) — keeps TEST_V2 accruing",
     "cmd": ["python3", "scripts/fetch_contract_specs.py"],
     "deps": [], "artifact": "settled_refetch"},
    {"id": "test_v2_audit", "lane": "L9", "priority": 0,
     "title": "TEST_V2 capture audit + forward sealed membership (by rule, not outcome)",
     "cmd": ["python3", "scripts/test_v2_capture_audit.py"],
     "deps": ["settled_refetch"], "artifact": "test_v2_capture_audit"},
    {"id": "sealed_test_gov", "lane": "L9", "priority": 0,
     "title": "Sealed-test governance (TEST_V1 SPENT, seal TEST_V2)",
     "cmd": ["python3", "scripts/sealed_test_governance.py"],
     "deps": ["offline_program", "test_v2_audit"], "artifact": "sealed_test_status"},
    {"id": "loss_calibration", "lane": "L12", "priority": 2,
     "title": "L12 loss-formulation + calibration sweeps (TGT-B/C/D/E/MT, Platt/isotonic/temp)",
     "cmd": ["python3", "scripts/loss_calibration_sweeps.py"],
     "deps": ["offline_program", "model_diagnostics"], "artifact": "loss_calibration"},
    {"id": "coverage_matrix", "lane": "L9", "priority": 1,
     "title": "Coverage / cohort matrix (waits on selected sources)",
     "cmd": ["python3", "scripts/build_coverage_matrix.py"],
     "deps": ["coarse_features", "av_expand"], "artifact": "coverage_matrix"},
]


def _write_state(jobs):
    STATE.write_text(json.dumps({
        "schema_version": "research-jobs-1", "generated_at": time.time(),
        "max_workers": MAX_WORKERS,
        "jobs": [{k: j.get(k) for k in ("id", "lane", "title", "status",
                  "started_at", "ended_at", "duration_ms", "deps", "returncode")}
                 for j in jobs]}, indent=1))


def _run_one(job):
    t0 = time.time()
    EV.emit("JOB_STARTED", job["title"], lane=job["lane"], job_id=job["id"])
    try:
        r = subprocess.run(job["cmd"], cwd=str(ROOT), capture_output=True,
                           text=True, timeout=600)
        rc = r.returncode
        err = (r.stderr or "")[-200:]
    except Exception as e:
        rc, err = 1, str(e)[:200]
    dur = int((time.time() - t0) * 1000)
    job["returncode"] = rc
    job["duration_ms"] = dur
    if rc == 0:
        EV.emit("JOB_COMPLETE", f"{job['title']} — done", lane=job["lane"],
                job_id=job["id"], duration_ms=dur)
    else:
        EV.emit("ERROR", f"{job['title']} — failed (rc={rc})", lane=job["lane"],
                job_id=job["id"], severity="high", technical=err, duration_ms=dur)
    return rc == 0


def run(plan_only=False):
    for j in JOBS:
        j["status"] = "QUEUED"; j["started_at"] = None; j["ended_at"] = None
        j["duration_ms"] = None; j["returncode"] = None
    _write_state(JOBS)
    if plan_only:
        for j in JOBS:
            print(f"  {j['lane']:3} {j['id']:18} deps={j['deps']} prio={j['priority']}")
        return
    EV.emit("PLAN", "Concurrent research engine starting", lane="L10",
            narrative=f"{len(JOBS)} jobs, up to {MAX_WORKERS} in parallel; dependency gates "
                      "enforced. L0 live capture runs independently and is untouched.",
            metrics={"jobs": len(JOBS), "max_workers": MAX_WORKERS})
    done, failed = set(), set()
    by_id = {j["id"]: j for j in JOBS}
    t_start = time.time()
    with cf.ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futures = {}
        while len(done) + len(failed) < len(JOBS):
            # launch every job whose deps are all COMPLETE (skip if a dep failed)
            for j in sorted(JOBS, key=lambda x: x["priority"]):
                if j["status"] != "QUEUED":
                    continue
                if any(d in failed for d in j["deps"]):
                    j["status"] = "BLOCKED"; j["ended_at"] = time.time()
                    EV.emit("WARNING", f"{j['title']} blocked (dependency failed)",
                            lane=j["lane"], job_id=j["id"], severity="high")
                    failed.add(j["id"]); _write_state(JOBS); continue
                if all(d in done for d in j["deps"]) and len(futures) < MAX_WORKERS:
                    j["status"] = "RUNNING"; j["started_at"] = time.time()
                    futures[ex.submit(_run_one, j)] = j
                    _write_state(JOBS)
            if not futures:
                if len(done) + len(failed) >= len(JOBS):
                    break
                time.sleep(0.2); continue
            fut = next(cf.as_completed(list(futures)))
            j = futures.pop(fut)
            ok = fut.result()
            j["status"] = "COMPLETE" if ok else "FAILED"
            j["ended_at"] = time.time()
            (done if ok else failed).add(j["id"])
            _write_state(JOBS)
    EV.emit("JOB_COMPLETE", "Research engine cycle finished", lane="L10",
            fact=f"{len(done)} complete, {len(failed)} failed of {len(JOBS)} jobs.",
            duration_ms=int((time.time() - t_start) * 1000),
            metrics={"complete": len(done), "failed": len(failed)})
    print(f"research_engine: {len(done)} complete, {len(failed)} failed "
          f"in {round(time.time()-t_start,1)}s (max_workers={MAX_WORKERS})")
    for j in JOBS:
        print(f"  {j['lane']:3} {j['id']:18} {j['status']:9} "
              f"{(j['duration_ms'] or 0)}ms rc={j['returncode']}")


if __name__ == "__main__":
    run(plan_only="--plan" in sys.argv)
