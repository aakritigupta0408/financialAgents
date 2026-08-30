"""Restart capture daemons whose heartbeat goes stale (>120s).
Cron every 5 min. Same discipline as the daemon watchdog.
Watches: event_capture.py (primary tape) and capture_xvenue.py
(F-XVENUE CAPTURE_ONLY tape, PM 08-30)."""
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

WATCH = [
    ("event_capture.py", "event_capture.json", "event_capture.log"),
    ("capture_xvenue.py", "xvenue_capture.json",
     "xvenue_capture.log"),
]

for script, status, log in WATCH:
    age = None
    try:
        age = time.time() - json.loads(
            (ROOT / "results" / status).read_text())["alive_at"]
    except Exception:
        pass
    if age is not None and age < 120:
        continue
    subprocess.run(["pkill", "-f", script], check=False)
    time.sleep(2)
    subprocess.Popen([sys.executable, "-u",
                      str(ROOT / "scripts" / script)],
                     cwd=ROOT,
                     stdout=(ROOT / "results" / log).open("a"),
                     stderr=subprocess.STDOUT, start_new_session=True)
    print(f"capture watchdog: restarted {script} (stale "
          f"{'missing' if age is None else round(age)}s)")
