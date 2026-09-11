"""One-shot: 5-min cron line for the capture watchdog (idempotent)."""
import subprocess

cur = subprocess.run(["crontab", "-l"], capture_output=True,
                     text=True).stdout
line = ('*/5 * * * * /opt/anaconda3/bin/python3 '
        '"/Users/aakritigupta/.claude/worktrees/btc-rl-poc/'
        'btc-rl-poc/scripts/capture_watchdog.py" '
        '>> /tmp/btc_capture_wd.log 2>&1')
if "capture_watchdog.py" not in cur:
    subprocess.run(["crontab", "-"],
                   input=cur.rstrip("\n") + "\n" + line + "\n",
                   text=True)
print("capture watchdog lines:", subprocess.run(
    ["crontab", "-l"], capture_output=True,
    text=True).stdout.count("capture_watchdog.py"))
