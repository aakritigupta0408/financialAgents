"""Recolor experiment_review ARMS entries to the validated dark palette."""
import re
from pathlib import Path

p = Path(__file__).resolve().parent.parent / "site" / "experiment_review.html"
h = p.read_text()
palette = {"ctl": "#aeb6c2", "t2": "#3987e5", "t6": "#d95926",
           "t7": "#199e70", "t8": "#c98500", "t9": "#d55181",
           "t10": "#008300", "t11": "#9085e9", "cal": "#e66767",
           "rp": "#5d5a51", "agg": "#f2efe6"}
n_total = 0
for key, hexv in palette.items():
    pat = r'(\{ key: "%s",[^\n]*color: )"#[0-9a-fA-F]{6}"' % key
    h, n = re.subn(pat, r'\1"%s"' % hexv, h, count=1)
    n_total += n
p.write_text(h)
print("recolored", n_total, "ARMS entries")
