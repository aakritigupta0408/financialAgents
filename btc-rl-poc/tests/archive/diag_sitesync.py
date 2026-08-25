"""Read-only diagnosis of the site repo's failing hourly main sync —
uses the same subprocess pattern as scripts/publish_dashboard.py."""
import subprocess
from pathlib import Path

REPO = Path.home() / "TheAakritiGupta.com"


def g(*args):
    r = subprocess.run(["git", "-C", str(REPO), *args],
                       capture_output=True, text=True)
    return r.returncode, (r.stdout + r.stderr).strip()


print("HEAD:", g("log", "--oneline", "-1")[1])
print("dirty files:", len(g("status", "--porcelain")[1].splitlines()))
g("fetch", "-q", "origin", "main")
print("ahead of origin/main:",
      g("rev-list", "--count", "origin/main..HEAD")[1])
print("behind origin/main:",
      g("rev-list", "--count", "HEAD..origin/main")[1])
code, out = g("push", "--dry-run",
              "https://github.com/aakritigupta0408/TheAakritiGupta.com.git"
              if False else "origin", "HEAD")
print("dry-run push:", code, out[:200])
