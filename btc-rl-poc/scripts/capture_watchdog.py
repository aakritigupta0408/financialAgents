"""Restart event_capture.py if its heartbeat goes stale (>120s).
Cron every 5 min. Same discipline as the daemon watchdog."""
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
STATUS = ROOT / "results" / "event_capture.json"

age = None
try:
    age = time.time() - json.loads(STATUS.read_text())["alive_at"]
except Exception:
    pass
if age is not None and age < 120:
    sys.exit(0)
subprocess.run(["pkill", "-f", "event_capture.py"], check=False)
time.sleep(2)
subprocess.Popen([sys.executable, "-u",
                  str(ROOT / "scripts" / "event_capture.py")],
                 cwd=ROOT,
                 stdout=(ROOT / "results" / "event_capture.log"
                         ).open("a"),
                 stderr=subprocess.STDOUT, start_new_session=True)
print(f"capture watchdog: restarted (stale "
      f"{'missing' if age is None else round(age)}s)")
