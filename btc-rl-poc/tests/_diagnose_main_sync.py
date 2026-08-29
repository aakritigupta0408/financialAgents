"""One-shot: diagnose why the publisher's hourly main-repo sync fails
(gh-pages fast path is healthy; this is the slow path)."""
import subprocess

REPO = "/Users/aakritigupta/TheAakritiGupta.com"


def git(*a):
    return subprocess.run(["git", "-C", REPO, *a],
                          capture_output=True, text=True)


print("status:", git("status", "--short").stdout[:300])
print("ahead:", git("log", "--oneline",
                    "origin/main..HEAD").stdout[:300])
print("behind:", git("log", "--oneline",
                     "HEAD..origin/main").stdout[:200])
p = git("push", "--dry-run", "origin", "HEAD")
print("dry-run rc:", p.returncode)
print("dry-run err:", p.stderr[:400])
