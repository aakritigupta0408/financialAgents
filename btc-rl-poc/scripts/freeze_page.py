"""Freeze a live site page into a SELF-CONTAINED artifact: inline
theme.css, inline every ../results/* file the page fetches, and shim
window.fetch to serve them from memory. One source of truth (the live
page); this makes it viewable as an artifact without a server.

Usage: python3 scripts/freeze_page.py site/perf.html traders_perf_snapshot.html
"""
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SITE = ROOT / "site"
RES = ROOT / "results"
JSONL_TAIL = 8000        # cap huge logs; small ones inline whole


def inline_theme(html):
    css = (SITE / "theme.css").read_text()
    return html.replace('<link rel="stylesheet" href="theme.css">',
                         f"<style>\n{css}\n</style>", 1)


def strip_external(html):
    # nav.js / glossary.js would 404 in an artifact; drop them (the
    # inline fallback <nav> remains).
    return re.sub(r'\s*<script src="(?:nav|glossary)\.js"[^>]*></script>',
                  "", html)


def collect(html, extra=()):
    files = set(re.findall(r'\.\./results/([\w.\-]+)', html))
    files |= set(extra)          # dynamic ${..} fetches, passed explicitly
    files = sorted(files)
    frozen = {}
    for name in files:
        p = RES / name
        if not p.exists():
            frozen[f"results/{name}"] = ""
            continue
        if name.endswith(".jsonl"):
            lines = p.read_text().splitlines()
            frozen[f"results/{name}"] = "\n".join(lines[-JSONL_TAIL:]) + "\n"
        else:
            frozen[f"results/{name}"] = p.read_text()
    return frozen


def shim(frozen):
    import json
    data = json.dumps(frozen)
    return ("<script>\nwindow.__FROZEN__=" + data + ";\n"
            "(function(){window.fetch=function(u){"
            "var k=String(u).split('?')[0].replace(/^\\.\\.\\//,'');"
            "if(k in window.__FROZEN__){var b=window.__FROZEN__[k];"
            "return Promise.resolve({ok:true,status:200,"
            "text:function(){return Promise.resolve(b);},"
            "json:function(){return Promise.resolve(JSON.parse(b));}});}"
            "return Promise.resolve({ok:false,status:404,"
            "text:function(){return Promise.resolve('');},"
            "json:function(){return Promise.resolve({});}});};})();\n</script>\n")


def main():
    src, out = sys.argv[1], sys.argv[2]
    extra = sys.argv[3:]         # explicit files for dynamic ${..} fetches
    html = Path(ROOT / src).read_text()
    frozen = collect(html, extra)
    html = inline_theme(html)
    html = strip_external(html)
    # inject the fetch shim right after <body> so it defines fetch
    # before the page's own scripts run
    html = html.replace("<body>", "<body>\n" + shim(frozen), 1)
    Path(ROOT / out).write_text(html)
    kb = len(html) // 1024
    print(f"froze {src} -> {out}  ({kb} KB, {len(frozen)} data files inlined)")
    for k, v in frozen.items():
        print(f"  {k}: {len(v)} bytes")


if __name__ == "__main__":
    main()
