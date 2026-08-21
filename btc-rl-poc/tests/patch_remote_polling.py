"""Make the dashboards remote-aware: on the public host, poll slower and
cache-bust fetches (the CDN's max-age=600 ignores client no-store; a
30s-bucketed query param actually punches through while still letting
the edge reuse within a bucket)."""
from pathlib import Path

SITE = Path(__file__).resolve().parent.parent / "site"

SNIPPET = """const REMOTE = !["localhost", "127.0.0.1"].includes(location.hostname);
// CDN cache-buster: bucketed so the edge can still coalesce requests
const bust = u => REMOTE
  ? u + (u.includes("?") ? "&" : "?") + "t=" + Math.floor(Date.now() / 30000)
  : u;
"""


def patch(name, fetch_prefix, polls):
    p = SITE / name
    t = p.read_text()
    i = t.index("<script>") + len("<script>")
    assert "const REMOTE" not in t
    t = t[:i] + "\n" + SNIPPET + t[i:]
    n = t.count(f'fetch("{fetch_prefix}')
    t = t.replace(f'fetch("{fetch_prefix}', f'fetch(bust("{fetch_prefix}')
    # close the bust() call: the fetch options arg follows the url string
    t = t.replace('.jsonl", { cache: "no-store" })',
                  '.jsonl"), { cache: "no-store" })')
    t = t.replace('.json", { cache: "no-store" })',
                  '.json"), { cache: "no-store" })')
    for old, new in polls:
        assert t.count(old) == 1, (name, old)
        t = t.replace(old, new, 1)
    p.write_text(t)
    print(f"{name}: {n} fetches busted")


patch("live_online.html", "../results/",
      [("setTimeout(poll, 5000);",
        "setTimeout(poll, REMOTE ? 45000 : 5000);")])
patch("experiment_review.html", "../results/",
      [("setTimeout(poll, 10000);",
        "setTimeout(poll, REMOTE ? 60000 : 10000);")])
patch("live_training.html", "../results/",
      [("setTimeout(poll, 1500);",
        "setTimeout(poll, REMOTE ? 20000 : 1500);"),
       ("setTimeout(retrainLoop, 60000);",
        "setTimeout(retrainLoop, REMOTE ? 120000 : 60000);")])
