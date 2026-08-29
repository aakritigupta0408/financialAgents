"""One-shot render verification for site/metrics_lab.html via CDP (temp helper)."""
import base64
import json
import sys
import time

import requests
import websocket

URL = "http://localhost:8123/site/metrics_lab.html"
OUT = sys.argv[1] if len(sys.argv) > 1 else "/tmp/metrics_lab.png"

t = requests.put("http://localhost:9222/json/new?" + URL, timeout=5).json()
tid = t["id"]
ws = websocket.create_connection(t["webSocketDebuggerUrl"], max_size=None)
i = 0


def cmd(method, **params):
    global i
    i += 1
    ws.send(json.dumps({"id": i, "method": method, "params": params}))
    while True:
        m = json.loads(ws.recv())
        if m.get("id") == i:
            return m.get("result", m)


cmd("Network.enable")
cmd("Network.setCacheDisabled", cacheDisabled=True)
cmd("Page.enable")
cmd("Page.navigate", url=URL)
time.sleep(6)  # let poll() fetch + render


def ev(expr):
    r = cmd("Runtime.evaluate", expression=expr, returnByValue=True)
    return r.get("result", {}).get("value")


checks = {
    "title": "document.title",
    "pipe cols": "document.querySelectorAll('#pipe .pipe-col').length",
    "pipe cards": "document.querySelectorAll('#pipe .tcard').length",
    "t1 rows": "document.querySelectorAll('#t1-table tbody tr').length",
    "t1 first": "document.querySelector('#t1-table tbody tr').innerText.replace(/\\s+/g,' ')",
    "hod cells": "document.querySelectorAll('#t1-hod td').length",
    "dow cells": "document.querySelectorAll('#t1-dow td').length",
    "drift svg": "!!document.querySelector('#drift-chart svg')",
    "t2 rows": "document.querySelectorAll('#t2-table tbody tr').length",
    "t2 first": "document.querySelector('#t2-table tbody tr').innerText.replace(/\\s+/g,' ')",
    "t2 drift ths": "document.querySelectorAll('#t2-drift thead th').length",
    "tod rows": "document.querySelectorAll('#t2-tod tbody tr').length",
    "t3 rows": "document.querySelectorAll('#t3-table tbody tr').length",
    "t3 first": "document.querySelector('#t3-table tbody tr').innerText.replace(/\\s+/g,' ')",
    "t5 runs rows": "document.querySelectorAll('#t5-runs tbody tr').length",
    "t5 strip rows": "document.querySelectorAll('#t5-strip .rt-row').length",
    "t5 gate rows": "document.querySelectorAll('#t5-gate tbody tr').length",
    "mi text": "document.getElementById('mi-body').innerText.slice(0,80)",
    "statusbar": ("['s-preds','s-kb','s-trades','s-retrains','s-upd']"
                  ".map(x=>document.getElementById(x).textContent).join(' | ')"),
    "gap notes": ("['t1-cut-note','t2-gap','t3-gap'].map(x=>"
                  "document.getElementById(x).innerText.slice(0,60)).join(' || ')"),
    "footer": "document.getElementById('foot').innerText.slice(0,90)",
    "hscroll": "document.body.scrollWidth+' vs '+window.innerWidth",
}
for k, e in checks.items():
    print(k, "=>", ev(e))

h = ev("document.body.scrollHeight")
cmd("Emulation.setDeviceMetricsOverride", width=1280, height=min(h + 40, 16000),
    deviceScaleFactor=1, mobile=False)
time.sleep(1.5)
shot = cmd("Page.captureScreenshot", format="png", captureBeyondViewport=True)
with open(OUT, "wb") as fh:
    fh.write(base64.b64decode(shot["data"]))
print("screenshot ->", OUT, "page height:", h)
ws.close()
requests.get("http://localhost:9222/json/close/" + tid, timeout=5)
print("tab closed")
