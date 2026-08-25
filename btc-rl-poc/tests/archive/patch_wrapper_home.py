"""Update the site-repo iframe wrapper: Home view first/default, Bet sim
removed. Prints the resulting VIEWS block for verification."""
import re
from pathlib import Path

p = (Path.home() / "TheAakritiGupta.com" / "client" / "pages"
     / "BtcOracleDemo.tsx")
t = p.read_text()
m = re.search(r"const VIEWS[^;]+;", t, re.S)
print("BEFORE:", m.group(0)[:400])
block = m.group(0)
b2 = re.sub(r"\s*\{[^}]*bet_policy_sim[^}]*\},?", "", block)
if "home.html" not in b2:
    b2 = re.sub(r"(=\s*\[)", r"""\1
  { key: "home", label: "Home", file: "home.html" },""", b2, 1)
t = t.replace(block, b2)
# default view -> home
t = re.sub(r'useState\(\s*"[a-z_0-9]+"\s*\)', 'useState("home")', t, 1)
p.write_text(t)
m2 = re.search(r"const VIEWS[^;]+;", t, re.S)
print("AFTER:", m2.group(0)[:400])
print("default:", re.search(r'useState\("[a-z_0-9]+"\)', t).group(0))
