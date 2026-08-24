"""One-shot repair of the diverged site-repo main: rebase local sync
commits onto origin/main (autostash the in-flight bundle copy), then
push with the publisher's tokenized URL."""
import subprocess
from pathlib import Path

REPO = Path.home() / "TheAakritiGupta.com"


def g(*args, check=True):
    r = subprocess.run(["git", "-C", str(REPO), *args],
                       capture_output=True, text=True)
    if check and r.returncode != 0:
        raise SystemExit(f"FAILED {args}: {(r.stdout + r.stderr)[:300]}")
    return (r.stdout + r.stderr).strip()


tok = (Path.home() / ".btc_publish_token").read_text().strip()
url = g("config", "--get", "remote.origin.url").replace(
    "https://", f"https://x-access-token:{tok}@", 1)
g("fetch", "-q", "origin", "main")
print("remote-only commit:", g("log", "--oneline", "HEAD..origin/main"))
g("rebase", "--autostash", "origin/main")
print("rebased:", g("log", "--oneline", "-1"))
g("push", "-q", url, "HEAD:main")
print("pushed. behind:", g("rev-list", "--count", "HEAD..origin/main",
                           check=False))
