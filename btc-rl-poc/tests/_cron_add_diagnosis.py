"""One-shot: append emit_diagnosis.py to the 10-min audit cron chain
(idempotent)."""
import subprocess

cur = subprocess.run(["crontab", "-l"], capture_output=True,
                     text=True).stdout
tag = "emit_diagnosis.py"
if tag in cur:
    print("cron already has diagnosis")
else:
    lines = cur.rstrip("\n").split("\n")
    for i, l in enumerate(lines):
        if "emit_fill_curve.py" in l:
            lines[i] = (l + ' && /opt/anaconda3/bin/python3 '
                        '"/Users/aakritigupta/.claude/worktrees/'
                        'btc-rl-poc/btc-rl-poc/scripts/'
                        'emit_diagnosis.py" >> /tmp/btc_audit.log 2>&1')
    p = subprocess.run(["crontab", "-"],
                       input="\n".join(lines) + "\n", text=True)
    print("cron updated:", p.returncode == 0)
print("diagnosis lines:",
      subprocess.run(["crontab", "-l"], capture_output=True,
                     text=True).stdout.count(tag))
