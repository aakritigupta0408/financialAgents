"""Emit the reproducibility manifest (§63): dependency lock, code SHA,
environment digest, model-artifact checksums — results/
build_manifest.json + requirements.lock at the repo root.

Honesty notes: the runtime is the owner's conda python (not a
containerized build), so the "environment digest" is a hash of the
full pip freeze — it detects drift, it does not reproduce the machine.
SBOM/vulnerability scanning are still absent and stay listed as gaps.
Secret scan: repo files are checked for private-key blocks and
key-like strings; findings are reported, never printed verbatim.
"""
from __future__ import annotations

import hashlib
import json
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"


def sh(*a):
    return subprocess.run(a, capture_output=True, text=True).stdout


def main():
    now = int(time.time())
    freeze = sh(sys.executable, "-m", "pip", "freeze")
    (ROOT / "requirements.lock").write_text(freeze)
    env_digest = hashlib.sha256(freeze.encode()).hexdigest()[:16]
    git_sha = sh("git", "-C", str(ROOT), "rev-parse", "HEAD").strip()

    # model artifact checksums (full, not truncated)
    artifacts = {}
    for pat in ("*.pt", "linucb_*.json", "*_logit.json",
                "q_table_online_*.json", "kb_calib.json"):
        for p in sorted(RES.glob(pat)):
            artifacts[p.name] = hashlib.sha256(
                p.read_bytes()).hexdigest()[:32]

    # secret scan (repo working tree, text files only)
    findings = []
    keyish = re.compile(r"BEGIN (RSA|EC|OPENSSH|PRIVATE) "
                        r"|api[_-]?secret|-----BEGIN")
    for p in ROOT.rglob("*"):
        if not p.is_file() or p.suffix in (".pt", ".png", ".pyc"):
            continue
        if ".git" in p.parts or "results" in p.parts:
            continue
        try:
            txt = p.read_text(errors="ignore")
        except OSError:
            continue
        if p.name == "emit_build_manifest.py":
            continue                     # the scanner's own patterns
        if keyish.search(txt):
            findings.append(str(p.relative_to(ROOT)))

    doc = {
        "generated_ts": now,
        "git_sha": git_sha,
        "python": sys.version.split()[0],
        "dependency_lock": {"file": "requirements.lock",
                            "packages": len(freeze.splitlines()),
                            "env_digest_sha256_16": env_digest},
        "model_artifacts": artifacts,
        "secret_scan": {"suspicious_files": findings,
                        "clean": not findings},
        "gaps": ["no SBOM", "no vulnerability scan",
                 "conda env not containerized — digest detects "
                 "drift, does not reproduce the machine"],
    }
    (RES / "build_manifest.json").write_text(json.dumps(doc, indent=1))
    print(f"build_manifest: git {git_sha[:8]} · "
          f"{len(freeze.splitlines())} deps locked · "
          f"{len(artifacts)} artifacts hashed · "
          f"secret scan {'CLEAN' if not findings else findings}")


if __name__ == "__main__":
    main()
