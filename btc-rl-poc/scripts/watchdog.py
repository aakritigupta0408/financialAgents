"""Daemon watchdog: restart btc_rl.online if online_status.json goes stale.

The daemon writes status every ~30s poll; the retrain-crash incident froze
those writes silently for weeks of sessions while predictions kept flowing.
This checks staleness and restarts the daemon when it exceeds 5 minutes.

Install (runs every 5 min):
  crontab -l | { cat; echo '*/5 * * * * cd <repo> && <python3> scripts/watchdog.py'; } | crontab -
"""
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
STATUS = ROOT / "results" / "online_status.json"
WLOG = ROOT / "results" / "watchdog_log.jsonl"
STALE_S = 300

age = None
try:
    age = time.time() - json.loads(STATUS.read_text())["alive_at"]
except Exception:
    pass

if age is not None and age < STALE_S:
    sys.exit(0)

subprocess.run(["pkill", "-f", "btc_rl.online"], check=False)
time.sleep(2)
with (ROOT / "results" / "daemon.log").open("a") as out:
    subprocess.Popen([sys.executable, "-u", "-m", "btc_rl.online"],
                     cwd=ROOT, stdout=out, stderr=subprocess.STDOUT,
                     start_new_session=True)
with WLOG.open("a") as f:
    f.write(json.dumps({"ts": int(time.time()), "event": "restarted",
                        "stale_s": None if age is None else round(age)}) + "\n")
print(f"watchdog: restarted daemon (status stale "
      f"{'missing' if age is None else round(age)}s)")
