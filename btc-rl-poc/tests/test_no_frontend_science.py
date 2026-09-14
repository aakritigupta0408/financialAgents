"""DT-08 §40 — repository guard: NO canonical scientific/economic formula may be
implemented in PUBLISHED frontend code. Presentation math (%, cents->$, axis
scaling, display aggregation) is allowed; Brier/EV/P&L/BSS/bootstrap/drawdown/
paired-effect/settlement computed in the browser is a defect and fails this test.

The published set is read from scripts/publish_dashboard.PAGES so the guard tracks
exactly what ships. Orphan (pruned) pages are out of scope until republished.
"""
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

# canonical-formula signatures (computation, not field reads / formatting)
DENY = [
    ("bootstrap_ci", re.compile(r"\bbootCI\b|resample\(|\bbootstrap\w*\(")),
    ("brier_calc", re.compile(r"(p_up|p_cal|\.p)\s*-\s*\w+\)\s*\*\*\s*2"
                              r"|Math\.pow\([^,]*(p_up|actual|\.p)\b")),
    ("bss_calc", re.compile(r"1\s*-\s*\w*[bB]rier\b|/\s*\w*[mM]kt[bB]rier|aggBSS\b")),
    ("pnl_reduce", re.compile(r"reduce\([^)]*\b(pnl|pnlOf)\b|\bpnlOf\b")),
    ("winrate_calc", re.compile(r"wins\.length\s*/|/\s*resolved\.length|wins\s*/\s*n\b")),
    ("drawdown_calc", re.compile(r"peak\s*=\s*Math\.max\(\s*peak|dd\s*=\s*Math\.max\(\s*dd")),
    ("fee_calc", re.compile(r"0\.07\s*\*\s*\w")),
    ("paired_ci", re.compile(r"1\.96\s*\*\s*\w*(sd|se)\b|Math\.sqrt\([^)]*/\s*\(?n\s*-\s*1")),
]


def _published_pages():
    import publish_dashboard as pub
    return [ROOT / "site" / p for p in pub.PAGES if p.endswith(".html")]


def test_published_pages_have_no_science():
    offenders = []
    for page in _published_pages():
        if not page.exists():
            continue
        txt = page.read_text()
        for name, rx in DENY:
            for m in rx.finditer(txt):
                line = txt[:m.start()].count("\n") + 1
                offenders.append(f"{page.name}:{line} [{name}] {m.group(0)!r}")
    assert not offenders, (
        "canonical science found in PUBLISHED frontend code — move it to the backend "
        "(btc_rl/metrics.py or btc_rl/economics.py) and render precomputed JSON:\n  "
        + "\n  ".join(offenders))


def test_published_set_is_home_and_oracle():
    import publish_dashboard as pub
    html = [p for p in pub.PAGES if p.endswith(".html")]
    assert "home.html" in html and "oracle.html" in html
    for absorbed in ("traders.html", "perf.html", "tiers.html", "health.html"):
        assert absorbed not in html, f"{absorbed} should be absorbed into HOME/ORACLE"
