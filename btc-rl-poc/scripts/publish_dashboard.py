"""Publish the dashboards + data snapshots to theaakritigupta.com.

Fast path (every cron minute): push pages + a trimmed data snapshot
DIRECTLY to the gh-pages branch that GitHub Pages serves — no site
rebuild, so end-to-end freshness is ~1-2 min. A dedicated single-branch
clone under ~/.btc-oracle-ghpages is reset to origin each run and the
snapshot rides a single amended marker commit (history never grows);
--force-with-lease loses gracefully to a real Actions deploy and
retries next minute.

Slow path (hourly): sync the same files into the main repo's
public/btc-oracle/ and push, so full rebuilds re-seed the dashboard.

Install:  * * * * * /opt/anaconda3/bin/python3 "<repo>/scripts/publish_dashboard.py" >> /tmp/btc_publish.log 2>&1
"""
import shutil
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SITE_REPO = Path.home() / "TheAakritiGupta.com"
DEST = SITE_REPO / "public" / "btc-oracle"
GH = Path.home() / ".btc-oracle-ghpages"
MARKER = "btc-oracle data snapshot"
MAIN_SYNC_S = 3600
STAMP = ROOT / "results" / ".publish_main_stamp"

PAGES = ["home.html", "live_online.html", "experiment_review.html",
         "live_training.html", "index.html", "theme.css",
         "ab_dashboard.html", "sev0.html", "metrics_lab.html",
         "board.html", "analyst.html", "home_classic.html", "glossary.js", "glossary.json", "nav.js",
         "universe.html", "clock.html", "agents.html", "museum.html",
         "instrument.html", "watchtower.html", "ledgers.html",
         "diagnosis.html", "paper.html", "archive.html", "models.html"]
DATA = [  # (filename, max jsonl lines or None for full copy)
    ("prediction_log.jsonl", 4000),
    ("recent_prices.json", None),
    ("online_status.json", None),
    ("learning_log.jsonl", 1500),
    ("kalshi_binary_log.jsonl", 6000),
    ("kb_bets.jsonl", None),
    ("kb_bets_sel.jsonl", None),
    ("kb_bets_sel_prepolicy.jsonl", None),
    ("pb_bets.jsonl", None),
    ("pt_trades.jsonl", None),
    ("pt2_trades.jsonl", None),
    ("pt3_trades.jsonl", None),
    ("pt4_trades.jsonl", None),
    ("pt5_trades.jsonl", None),
    ("pt6_trades.jsonl", None),
    ("pt7_trades.jsonl", None),
    ("pt8_trades.jsonl", None),
    ("demo_orders.jsonl", None),
    ("demo_account.json", None),
    ("demo_fills.jsonl", None),
    ("metrics_history.jsonl", None),
    ("metrics.json", None),
    ("kb_calib.json", None),
    ("treatments.json", None),
    ("treatments.jsonl", 2000),
    ("training_progress.jsonl", None),
    ("live_status.json", None),
    ("audit_report.json", None),
    ("incidents.jsonl", None),
    ("model_internals.json", None),
    ("board.json", None),
    ("commentary.jsonl", None),
    ("site_manifest.json", None),
    ("treatments_board.json", None),
    ("metric_fixtures.json", None),
    ("decision_board.json", None),
    ("world.json", None),
    ("invariants.json", None),
    ("model_registry.json", None),
    ("loss_reviews.json", None),
    ("loss_reviews.jsonl", 3000),
    ("fill_curve.json", None),
    ("diagnosis.json", None),
    ("program.json", None),
    ("readiness.json", None),
    ("execution_ledger.json", None),
    ("exec_sensitivity.json", None),
    ("build_manifest.json", None),
    ("oracle_calls.json", None),
    ("a3_live.json", None),
    ("a3_window_evaluation.jsonl", 2000),
    ("model_lifecycle.json", None),
    ("model_online.json", None),
    ("model_offline.json", None),
    ("training_runs.jsonl", 3000),
    ("pm_snapshot.json", None),
    ("reconciliation.json", None),
    ("meta_monitors.json", None),
    ("leakage_canaries.json", None),
    ("execution_ledger.jsonl", 2000),
]


def _git(cwd, *args, check=True):
    return subprocess.run(["git", "-C", str(cwd), *args],
                          capture_output=True, text=True, check=check)


def _push_url(repo) -> str:
    """Tokenized push URL: cron has no TTY/keychain, so HTTPS credential
    helpers fail there. Token minted once via `gh auth token` into
    ~/.btc_publish_token (chmod 600)."""
    url = _git(repo, "config", "--get", "remote.origin.url").stdout.strip()
    tok_file = Path.home() / ".btc_publish_token"
    if url.startswith("https://") and tok_file.exists():
        tok = tok_file.read_text().strip()
        return url.replace("https://", f"https://x-access-token:{tok}@", 1)
    return url


def copy_bundle(dest: Path) -> None:
    (dest / "site").mkdir(parents=True, exist_ok=True)
    (dest / "results").mkdir(parents=True, exist_ok=True)
    for name in PAGES:
        shutil.copy2(ROOT / "site" / name, dest / "site" / name)
    for name, cap in DATA:
        src = ROOT / "results" / name
        if not src.exists():
            continue
        if cap is None:
            shutil.copy2(src, dest / "results" / name)
        else:
            lines = src.read_text().splitlines()[-cap:]
            (dest / "results" / name).write_text("\n".join(lines) + "\n")


def publish_ghpages() -> None:
    if not GH.exists():
        url = _git(SITE_REPO, "config", "--get",
                   "remote.origin.url").stdout.strip()
        subprocess.run(["git", "clone", "--depth", "2", "--branch", "gh-pages",
                        "--single-branch", url, str(GH)], check=True)
    _git(GH, "fetch", "--depth", "2", "origin", "gh-pages")
    _git(GH, "reset", "--hard", "origin/gh-pages")
    copy_bundle(GH / "btc-oracle")
    if not _git(GH, "status", "--porcelain").stdout.strip():
        print("gh-pages: no changes")
        return
    _git(GH, "add", "btc-oracle")
    tip = _git(GH, "log", "-1", "--format=%s").stdout.strip()
    if tip.startswith(MARKER):  # ride the same marker commit forever
        _git(GH, "commit", "-q", "--amend", "-m",
             f"{MARKER} {time.strftime('%Y-%m-%d %H:%M')}")
    else:
        _git(GH, "commit", "-q", "-m",
             f"{MARKER} {time.strftime('%Y-%m-%d %H:%M')}")
    # lease against the just-fetched tip explicitly — pushing to a URL
    # (not a named remote) gives git no tracking ref to infer it from
    expected = _git(GH, "rev-parse", "origin/gh-pages").stdout.strip()
    push = _git(GH, "push",
                f"--force-with-lease=refs/heads/gh-pages:{expected}",
                _push_url(GH), "HEAD:refs/heads/gh-pages", check=False)
    print("gh-pages:", "published" if push.returncode == 0
          else f"deferred ({push.stderr.strip()[:80]})")


def sync_main() -> None:
    if STAMP.exists() and time.time() - STAMP.stat().st_mtime < MAIN_SYNC_S:
        return
    # SELF-HEAL (INC-2026-08-29-stale-rebase, twice): a crashed rebase
    # leaves .git/rebase-merge + a detached HEAD; every later sync then
    # fails or commits into the void. Heal both conditions before
    # touching anything.
    gitdir = SITE_REPO / ".git"
    if (gitdir / "rebase-merge").exists() or \
            (gitdir / "rebase-apply").exists():
        _git(SITE_REPO, "rebase", "--abort", check=False)
    if _git(SITE_REPO, "symbolic-ref", "-q", "HEAD",
            check=False).returncode != 0:      # detached HEAD
        # re-attach main AT the current commit (keeps any sync
        # commits made while detached; push targets refs/heads/main)
        _git(SITE_REPO, "checkout", "-B", "main", check=False)
    # rebase first: the site's own workflows commit to origin/main (e.g.
    # weekly content refresh), and a plain push then non-fast-forwards
    # and wedges every hourly sync after it
    _git(SITE_REPO, "fetch", "-q", "origin", "main", check=False)
    _git(SITE_REPO, "rebase", "--autostash", "origin/main", check=False)
    copy_bundle(DEST)
    staged = [str(DEST.relative_to(SITE_REPO)),
              "client/pages/BtcOracleDemo.tsx", "client/App.tsx",
              "client/pages/AIPlayground.tsx",
              "client/pages/TradeRecommendationSystemDemo.tsx"]
    if _git(SITE_REPO, "status", "--porcelain", *staged).stdout.strip():
        _git(SITE_REPO, "add", *staged)
        _git(SITE_REPO, "commit", "-q", "-m",
             f"btc-oracle hourly sync {time.strftime('%Y-%m-%d %H:%M')}")
        # explicit refspec: a detached-HEAD checkout made bare "HEAD"
        # unresolvable remotely (2026-08-29) — the push failed every
        # minute and stacked local sync commits until reattached
        _git(SITE_REPO, "push", "-q", _push_url(SITE_REPO),
             "HEAD:refs/heads/main")
        print("main: synced")
    STAMP.touch()


def main() -> None:
    publish_ghpages()
    try:
        sync_main()
    except subprocess.CalledProcessError as e:
        print("main sync failed:", str(e)[:120])


if __name__ == "__main__":
    main()
