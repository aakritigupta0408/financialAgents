"""Minimal Chrome DevTools Protocol driver for the Kalshi demo tab.

Read/observe by default; navigation + clicks are explicit calls. Talks
to the browser launched with --remote-debugging-port=9222.

  python3 scripts/cdp.py shot out.png          screenshot the page
  python3 scripts/cdp.py eval "<js expr>"      run JS, print result
  python3 scripts/cdp.py nav "<url>"           navigate
  python3 scripts/cdp.py click "<css>"         click first match
"""
import json
import sys
import time

import requests
import websocket


def target():
    for t in requests.get("http://localhost:9222/json", timeout=5).json():
        if t["type"] == "page" and "kalshi" in t["url"]:
            return t
    # fall back to any page
    for t in requests.get("http://localhost:9222/json", timeout=5).json():
        if t["type"] == "page":
            return t
    raise SystemExit("no page target")


class CDP:
    def __init__(self):
        self.ws = websocket.create_connection(target()["webSocketDebuggerUrl"],
                                              max_size=None)
        self.i = 0

    def cmd(self, method, **params):
        self.i += 1
        self.ws.send(json.dumps({"id": self.i, "method": method,
                                 "params": params}))
        while True:
            m = json.loads(self.ws.recv())
            if m.get("id") == self.i:
                return m.get("result", {})

    def js(self, expr):
        r = self.cmd("Runtime.evaluate", expression=expr,
                     returnByValue=True, awaitPromise=True)
        return r.get("result", {}).get("value")


def main():
    act = sys.argv[1] if len(sys.argv) > 1 else "shot"
    c = CDP()
    c.cmd("Page.enable")
    c.cmd("Runtime.enable")
    if act == "shot":
        out = sys.argv[2] if len(sys.argv) > 2 else "/tmp/kalshi.png"
        data = c.cmd("Page.captureScreenshot", format="png")["data"]
        import base64
        open(out, "wb").write(base64.b64decode(data))
        print("saved", out)
    elif act == "eval":
        print(c.js(sys.argv[2]))
    elif act == "evalfile":
        print(c.js(open(sys.argv[2]).read()))
    elif act == "nav":
        c.cmd("Page.navigate", url=sys.argv[2])
        time.sleep(3)
        print("navigated", sys.argv[2])
    elif act == "click":
        sel = sys.argv[2].replace("'", "\\'")
        r = c.js(f"(()=>{{const e=document.querySelector('{sel}');"
                 f"if(e){{e.click();return 'clicked';}}return 'not found';}})()")
        print(r)


if __name__ == "__main__":
    main()
