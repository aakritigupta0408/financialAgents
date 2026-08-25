"""Fix the wrapper patch: correct entry schema and default-view state."""
import re
from pathlib import Path

p = (Path.home() / "TheAakritiGupta.com" / "client" / "pages"
     / "BtcOracleDemo.tsx")
t = p.read_text()
t = t.replace('{ key: "home", label: "Home", file: "home.html" },',
              '{ id: "home", label: "Home" },')
for m in re.finditer(r".*useState.*", t):
    print("useState line:", m.group(0).strip())
t = re.sub(r'useState(<[^>]*>)?\(\s*"ab_dashboard"\s*\)',
           r'useState\1("home")', t)
p.write_text(t)
m2 = re.search(r"const VIEWS[^;]+;", t, re.S)
print("VIEWS:", m2.group(0)[:300])
for m in re.finditer(r".*useState.*", t):
    print("now:", m.group(0).strip())
