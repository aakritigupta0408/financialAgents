"""Publish the dashboards + a data snapshot to theaakritigupta.com.

Copies the four site pages, theme.css, and a whitelisted (and trimmed)
snapshot of results/ into ~/TheAakritiGupta.com/public/btc-oracle/, then
commits and pushes the site repo — Netlify redeploys automatically, so
the dashboard is reachable anywhere with data at cron freshness.

Install (every 10 min):
  */10 * * * * /opt/anaconda3/bin/python3 "<repo>/scripts/publish_dashboard.py" >> /tmp/btc_publish.log 2>&1
"""
import shutil
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SITE_REPO = Path.home() / "TheAakritiGupta.com"
DEST = SITE_REPO / "public" / "btc-oracle"

PAGES = ["live_online.html", "experiment_review.html",
         "live_training.html", "index.html", "theme.css"]
# results whitelist: (filename, max jsonl lines or None for full copy)
DATA = [
    ("prediction_log.jsonl", 4000),
    ("recent_prices.json", None),
    ("online_status.json", None),
    ("learning_log.jsonl", 1500),
    ("kalshi_binary_log.jsonl", 1500),
    ("kb_bets.jsonl", None),
    ("metrics_history.jsonl", None),
    ("metrics.json", None),
    ("training_progress.jsonl", None),
    ("live_status.json", None),
]


def main() -> None:
    (DEST / "site").mkdir(parents=True, exist_ok=True)
    (DEST / "results").mkdir(parents=True, exist_ok=True)
    for name in PAGES:
        shutil.copy2(ROOT / "site" / name, DEST / "site" / name)
    for name, cap in DATA:
        src = ROOT / "results" / name
        if not src.exists():
            continue
        if cap is None:
            shutil.copy2(src, DEST / "results" / name)
        else:
            lines = src.read_text().splitlines()[-cap:]
            (DEST / "results" / name).write_text("\n".join(lines) + "\n")

    # stage the dashboard tree plus the React wiring for its route/card
    staged = [str(DEST.relative_to(SITE_REPO)),
              "client/pages/BtcOracleDemo.tsx", "client/App.tsx",
              "client/pages/AIPlayground.tsx",
              "client/pages/TradeRecommendationSystemDemo.tsx"]
    if subprocess.run(["git", "-C", str(SITE_REPO), "status", "--porcelain",
                       *staged], capture_output=True,
                      text=True).stdout.strip():
        subprocess.run(["git", "-C", str(SITE_REPO), "add", *staged],
                       check=True)
        subprocess.run(["git", "-C", str(SITE_REPO), "commit", "-q", "-m",
                        f"btc-oracle data snapshot "
                        f"{time.strftime('%Y-%m-%d %H:%M')}"], check=True)
        subprocess.run(["git", "-C", str(SITE_REPO), "push", "-q"],
                       check=True)
        print(f"published at {time.strftime('%H:%M:%S')}")
    else:
        print("no changes")


if __name__ == "__main__":
    main()
