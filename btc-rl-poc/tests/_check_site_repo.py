"""Quick state check of the site repo."""
import subprocess

REPO = "/Users/aakritigupta/TheAakritiGupta.com"


def git(*a):
    p = subprocess.run(["git", "-C", REPO, *a],
                       capture_output=True, text=True)
    print("$", " ".join(a[:4]), "→", (p.stdout or p.stderr).strip()[:200])
    return p


git("symbolic-ref", "-q", "HEAD")
git("log", "--oneline", "-1")
git("log", "--oneline", "origin/main..HEAD")
git("log", "--oneline", "HEAD..origin/main")
