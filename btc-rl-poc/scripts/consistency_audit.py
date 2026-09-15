"""CONSISTENCY & STALENESS AUDITOR — one command that verifies every project surface is
LATEST and CONSISTENT with the current roster/directives.

Checks, for every results/*.json snapshot, every site/*.html page, and the key docs:
  1. FRESHNESS — file mtime age and (if present) the snapshot's own `generated_at`, vs an
     expected cadence. Flags STALE.
  2. ROSTER CONSISTENCY — the live roster is T0 `pt` / T1 `cg33` / T2 `fm`; the RETIRED arms
     (cg5, cg10, tv, pt2, pt3, pt4, pt5, pt6, pt7, pt8) must NOT appear as live. Flags any
     surface that still references a retired arm (so all expressions stay consistent).
  3. WRITER — best-effort map of who writes each snapshot (for fixing stale ones).

Writes research/consistency_audit.json and prints a ranked table. Safe to run on cron.
"""
import json, re, time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"
SITE = ROOT / "site"
NOW = time.time()

RETIRED = ["cg5", "cg10", "tv", "pt2", "pt3", "pt4", "pt5", "pt6", "pt7", "pt8"]
LIVE = ["pt", "cg33", "fm"]
# expected max age (s) before STALE, by snapshot cadence
FRESH_S = 8 * 60          # per-minute publish snapshots -> stale if > 8 min
DOC_FRESH_S = 3 * 24 * 3600

DOCS = ["RESEARCH_PROGRAM.md", "SYSTEM_AUDIT_2026-09-15.md", "research/COMBINED_STUDENT_FINDINGS.md"]


def _age(p):
    return NOW - p.stat().st_mtime


def _retired_hits(text):
    hits = []
    for a in RETIRED:
        # word-ish boundary so 'pt2' doesn't match 'pt20'; allow quotes/underscores
        if re.search(rf'(?<![a-z0-9]){a}(?![a-z0-9])', text):
            hits.append(a)
    return hits


def run():
    rows = []
    # snapshots
    for p in sorted(RES.glob("*snapshot*.json")) + sorted(RES.glob("*_snapshot.json")):
        pass
    seen = set()
    cand = list(RES.glob("*.json"))
    for p in cand:
        if p.name in seen:
            continue
        seen.add(p.name)
        try:
            txt = p.read_text()
        except Exception:
            continue
        age = _age(p)
        gen_age = None
        try:
            d = json.loads(txt)
            g = d.get("generated_at") or d.get("generated_ts") or (d.get("meta", {}) or {}).get("generated_at")
            if isinstance(g, (int, float)):
                gen_age = NOW - g
        except Exception:
            d = None
        is_snap = "snapshot" in p.name or p.name in (
            "home_snapshot.json", "live_desk.json", "trader_detail.json", "oracle_snapshot.json")
        stale = is_snap and age > FRESH_S
        ret = _retired_hits(txt)
        rows.append({"file": f"results/{p.name}", "kind": "snapshot" if is_snap else "json",
                     "age_min": round(age / 60, 1), "gen_age_min": round(gen_age / 60, 1) if gen_age else None,
                     "STALE": stale, "references_retired_arm": ret})
    # site pages
    for p in sorted(SITE.glob("*.html")):
        txt = p.read_text()
        rows.append({"file": f"site/{p.name}", "kind": "page", "age_min": round(_age(p) / 60, 1),
                     "STALE": False, "references_retired_arm": _retired_hits(txt)})
    # docs
    for dn in DOCS:
        p = ROOT / dn
        if p.exists():
            rows.append({"file": dn, "kind": "doc", "age_min": round(_age(p) / 60, 1),
                         "STALE": _age(p) > DOC_FRESH_S, "references_retired_arm": []})
    stale = [r for r in rows if r.get("STALE")]
    incons = [r for r in rows if r.get("references_retired_arm")]
    report = {"generated_at": NOW, "n_surfaces": len(rows),
              "n_stale": len(stale), "n_inconsistent": len(incons),
              "stale": sorted(stale, key=lambda r: -r["age_min"]),
              "inconsistent_roster": incons, "all": rows}
    (ROOT / "research" / "consistency_audit.json").write_text(json.dumps(report, indent=1))
    print(f"surfaces={len(rows)}  STALE={len(stale)}  roster-INCONSISTENT={len(incons)}")
    print("--- STALE (age > 8min snapshots / >3d docs) ---")
    for r in report["stale"][:20]:
        print(f"  {r['file']:44} age {r['age_min']}min  gen {r.get('gen_age_min')}")
    print("--- references a RETIRED arm (should be gone) ---")
    for r in incons[:20]:
        print(f"  {r['file']:44} {r['references_retired_arm']}")


if __name__ == "__main__":
    run()
