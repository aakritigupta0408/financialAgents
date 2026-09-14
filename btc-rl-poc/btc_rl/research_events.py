"""Structured research event ledger (directive §16-19, §28, §31, §36).

Every meaningful research action emits an append-only ResearchEvent to
results/research_events.jsonl with a monotonic, flock-protected sequence id so
that concurrently-running lanes produce a deterministically orderable stream.
Secrets are redacted before anything is written (§19).

Canonical data. The website renders it; it never parses backend logs.
"""
import fcntl
import json
import os
import re
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LEDGER = ROOT / "results" / "research_events.jsonl"
SEQ = ROOT / "results" / ".research_seq"
LOCK = ROOT / "results" / ".research_seq.lock"

# §19 redaction — never narrate secret material
_REDACT = [
    (re.compile(r"(?i)\b(api[_-]?key|token|authorization|secret|password|bearer)\b"
                r"\s*[=:]?\s*([^\s,'\";]+)"), r"\1=<redacted>"),
    (re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----.*?-----END [A-Z ]*PRIVATE KEY-----",
                re.DOTALL), "<redacted-private-key>"),
    (re.compile(r"\b[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-"
                r"[0-9a-fA-F]{4}-[0-9a-fA-F]{12}\b"), "<uuid>"),
    (re.compile(r"\bgh[oprsu]_[A-Za-z0-9]{20,}\b"), "<redacted-token>"),
]

EVENT_TYPES = {
    "THOUGHT", "PLAN", "JOB_STARTED", "SOURCE_CONNECTED", "DOWNLOAD_STARTED",
    "DOWNLOAD_PROGRESS", "DOWNLOAD_COMPLETE", "FILE_CREATED", "FILE_MODIFIED",
    "FILE_DELETED", "FEATURE_ADDED", "FEATURE_VALIDATED", "TEST_STARTED",
    "TEST_PASSED", "TEST_FAILED", "BUG_FOUND", "BUG_FIXED", "DATASET_UPDATED",
    "INTEGRITY_CHECK", "MODEL_TRAINING_STARTED", "MODEL_TRAINING_PROGRESS",
    "MODEL_TRAINING_COMPLETE", "MODEL_REJECTED", "MODEL_PROMISING",
    "MODEL_QUALIFIED", "EXPERIMENT_REGISTERED", "GIT_COMMIT", "DEPLOYMENT",
    "WARNING", "ERROR", "DISCOVERY", "NEXT_STEP", "LIVE_CONTRACT", "JOB_COMPLETE",
}


def _redact(x):
    if isinstance(x, str):
        for pat, repl in _REDACT:
            x = pat.sub(repl, x)
        return x
    if isinstance(x, dict):
        return {k: _redact(v) for k, v in x.items()}
    if isinstance(x, list):
        return [_redact(v) for v in x]
    return x


def _next_seq():
    """Monotonic sequence under an flock so parallel lanes never collide (§31)."""
    LOCK.parent.mkdir(parents=True, exist_ok=True)
    with open(LOCK, "w") as lk:
        fcntl.flock(lk, fcntl.LOCK_EX)
        try:
            n = int(SEQ.read_text()) if SEQ.exists() else 0
        except ValueError:
            n = 0
        n += 1
        SEQ.write_text(str(n))
        fcntl.flock(lk, fcntl.LOCK_UN)
    return n


def emit(event_type, title, *, lane=None, job_id=None, severity="info",
         narrative=None, fact=None, interpretation=None, next_action=None,
         files=None, tests=None, metrics=None, git=None, technical=None,
         duration_ms=None):
    """Append one redacted ResearchEvent. Best-effort: never raises into callers."""
    try:
        seq = _next_seq()
        rec = _redact({
            "event_id": seq, "seq": seq, "timestamp": time.time(),
            "event_type": event_type if event_type in EVENT_TYPES else "THOUGHT",
            "severity": severity, "lane": lane, "job_id": job_id,
            "title": title, "narrative": narrative,
            "fact": fact, "interpretation": interpretation, "next_action": next_action,
            "files": files or [], "tests": tests or [], "metrics": metrics or {},
            "git": git or {}, "technical": technical, "duration_ms": duration_ms,
            "pid": os.getpid(),
        })
        LEDGER.parent.mkdir(parents=True, exist_ok=True)
        with open(LEDGER, "a") as fh:
            fcntl.flock(fh, fcntl.LOCK_EX)
            fh.write(json.dumps(rec) + "\n")
            fcntl.flock(fh, fcntl.LOCK_UN)
        return rec
    except Exception:
        return None


def recent(limit=60, after_seq=0):
    """Return recent events (seq > after_seq), newest last. Cursor-friendly (§30)."""
    if not LEDGER.exists():
        return []
    out = []
    for l in LEDGER.open():
        l = l.strip()
        if not l:
            continue
        try:
            r = json.loads(l)
        except json.JSONDecodeError:
            continue
        if r.get("seq", 0) > after_seq:
            out.append(r)
    return out[-limit:]


if __name__ == "__main__":
    e = emit("THOUGHT", "Research event ledger online",
             lane="L10", narrative="Smoke test of the append-only ledger.",
             next_action="wire emitters into the research engine")
    print("emitted seq", e["seq"], "->", LEDGER)
