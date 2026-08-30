"""One-shot: main sync wedged again — local main behind origin
(non-fast-forward). Rebase local commits onto origin/main and push;
report any rebase conflict instead of forcing."""
import subprocess
from pathlib import Path

REPO = "/Users/aakritigupta/TheAakritiGupta.com"


def git(*a):
    p = subprocess.run(["git", "-C", REPO, *a],
                       capture_output=True, text=True)
    print("$", " ".join(a[:4]), "rc", p.returncode,
          (p.stderr or p.stdout).strip()[:150])
    return p


git("fetch", "origin", "main")
git("status", "--short")
r = git("rebase", "--autostash", "origin/main")
if r.returncode != 0:
    print("REBASE CONFLICT — aborting, needs attention")
    git("rebase", "--abort")
else:
    url = git("config", "--get", "remote.origin.url").stdout.strip()
    tok = (Path.home() / ".btc_publish_token").read_text().strip()
    push_url = url.replace("https://",
                           f"https://x-access-token:{tok}@", 1)
    p = subprocess.run(["git", "-C", REPO, "push", push_url,
                        "HEAD:refs/heads/main"],
                       capture_output=True, text=True)
    print("push rc", p.returncode, (p.stderr or "")[:150])
