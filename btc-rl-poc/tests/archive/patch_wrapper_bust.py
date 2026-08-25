"""Cache-bust the wrapper's iframe HTML so viewers always get the
current page shell (data was already busted; the shell was not)."""
from pathlib import Path

p = (Path.home() / "TheAakritiGupta.com" / "client" / "pages"
     / "BtcOracleDemo.tsx")
t = p.read_text()
old = "src={`/btc-oracle/site/${view}.html`}"
new = ("src={`/btc-oracle/site/${view}.html?v=${Math.floor(Date.now() / "
       "300000)}`}")
assert old in t
p.write_text(t.replace(old, new))
print("iframe src cache-busted (5-min buckets)")
