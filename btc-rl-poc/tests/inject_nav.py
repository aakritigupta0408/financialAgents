"""Inject the shared top-nav into every static page (idempotent)."""
from pathlib import Path

SITE = Path(__file__).resolve().parent.parent / "site"
PAGES = {
    "ab_dashboard.html": "Results",
    "live_online.html": "Live desk",
    "experiment_review.html": "Experiment lab",
    "index.html": "Backtest",
    "live_training.html": "Training",
    "bet_policy_sim.html": "Bet sim",
}
ORDER = ["ab_dashboard.html", "live_online.html", "experiment_review.html",
         "index.html", "live_training.html", "bet_policy_sim.html"]


def nav_for(current):
    links = "".join(
        f'<a class="nav{" here" if p == current else ""}" href="{p}">'
        f"{PAGES[p]}</a>" for p in ORDER)
    return ('<nav class="topnav"><a class="brand" href="ab_dashboard.html">'
            "BTC 7PM Oracle</a>" + links +
            '<span class="spacer"></span><span class="tag">live experiment'
            "</span></nav>")


for page in ORDER:
    p = SITE / page
    t = p.read_text()
    if 'class="topnav"' in t:  # replace existing (idempotent refresh)
        import re
        t = re.sub(r'<nav class="topnav">.*?</nav>\n?', "", t, flags=re.S)
    anchor = "<body>\n" if "<body>\n" in t else "<body>"
    assert anchor in t, page
    t = t.replace(anchor, anchor + "\n" + nav_for(page) + "\n", 1)
    p.write_text(t)
    print(f"nav -> {page}")
