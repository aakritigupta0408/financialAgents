"""One-shot: unwedge the publisher's main-repo sync.

Root cause: TheAakritiGupta.com working copy sits in DETACHED HEAD, so
`git push <url> HEAD` cannot resolve a remote refname and fails; the
failure path never touches the hourly stamp, so the publisher retried
(and committed) EVERY MINUTE, stacking sync commits locally. origin is
not ahead (fast-forward safe), so: push explicitly to
refs/heads/main, then reattach the checkout to main.
"""
import subprocess
from pathlib import Path

REPO = "/Users/aakritigupta/TheAakritiGupta.com"


def git(*a, check=False):
    p = subprocess.run(["git", "-C", REPO, *a],
                       capture_output=True, text=True)
    print("$ git", " ".join(a[:3]), "→ rc", p.returncode,
          (p.stderr or p.stdout).strip()[:120])
    return p


url = git("config", "--get", "remote.origin.url").stdout.strip()
tok = (Path.home() / ".btc_publish_token").read_text().strip()
push_url = url.replace("https://", f"https://x-access-token:{tok}@", 1)
behind = git("log", "--oneline", "HEAD..origin/main").stdout.strip()
if behind:
    print("ABORT: origin/main has commits we don't — needs a human")
else:
    p = subprocess.run(["git", "-C", REPO, "push", push_url,
                        "HEAD:refs/heads/main"],
                       capture_output=True, text=True)
    print("push rc", p.returncode, (p.stderr or "")[:150])
    if p.returncode == 0:
        git("fetch", "origin", "main")
        git("checkout", "main")
        git("reset", "--hard", "origin/main")
        print("reattached to main")
