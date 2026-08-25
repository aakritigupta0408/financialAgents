"""One-shot nav update across pages: add Home first, drop Bet sim."""
from pathlib import Path

SITE = Path(__file__).resolve().parent.parent / "site"
PAGES = ["ab_dashboard.html", "live_online.html", "experiment_review.html",
         "index.html", "live_training.html"]

for name in PAGES:
    p = SITE / name
    t = p.read_text()
    # drop the Bet sim tab
    import re
    t2 = re.sub(r'\s*<a class="nav[^"]*" href="bet_policy_sim\.html">[^<]*</a>',
                "", t)
    # brand links home; add Home tab right after the brand
    t2 = t2.replace('<a class="brand" href="live_online.html">',
                    '<a class="brand" href="home.html">')
    if 'href="home.html">Home</a>' not in t2:
        t2 = re.sub(r'(<a class="brand"[^>]*>[^<]*</a>)',
                    r'\1\n  <a class="nav" href="home.html">Home</a>', t2, 1)
    assert "bet_policy_sim" not in t2, name
    assert 'href="home.html">Home</a>' in t2, name
    p.write_text(t2)
    print("updated", name)
