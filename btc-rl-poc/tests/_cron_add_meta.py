"""One-shot: give the monitor-of-monitors its OWN 5-min crontab line
(independent of the audit chain it watches). Idempotent."""
import subprocess

cur = subprocess.run(["crontab", "-l"], capture_output=True,
                     text=True).stdout
line = ('*/5 * * * * /opt/anaconda3/bin/python3 '
        '"/Users/aakritigupta/.claude/worktrees/btc-rl-poc/'
        'btc-rl-poc/scripts/meta_monitor.py" '
        '>> /tmp/btc_meta.log 2>&1')
if "meta_monitor.py" not in cur:
    p = subprocess.run(["crontab", "-"],
                       input=cur.rstrip("\n") + "\n" + line + "\n",
                       text=True)
    print("crontab updated:", p.returncode == 0)
print("meta lines:", subprocess.run(
    ["crontab", "-l"], capture_output=True,
    text=True).stdout.count("meta_monitor.py"))
