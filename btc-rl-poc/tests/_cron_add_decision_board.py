"""One-shot: append emit_decision_board.py to the 10-min audit cron
chain (idempotent). Kept in tests/ with the other operational one-shots
so the change is on the record."""
import subprocess

cur = subprocess.run(["crontab", "-l"], capture_output=True,
                     text=True).stdout
tag = "emit_decision_board.py"
if tag in cur:
    print("cron already has emitter")
else:
    lines = cur.rstrip("\n").split("\n")
    for i, l in enumerate(lines):
        if "emit_board.py" in l:
            lines[i] = (l + ' && /opt/anaconda3/bin/python3 '
                        '"/Users/aakritigupta/.claude/worktrees/'
                        'btc-rl-poc/btc-rl-poc/scripts/'
                        'emit_decision_board.py" '
                        '>> /tmp/btc_audit.log 2>&1')
    new = "\n".join(lines) + "\n"
    p = subprocess.run(["crontab", "-"], input=new, text=True)
    print("cron updated:", p.returncode == 0)
chk = subprocess.run(["crontab", "-l"], capture_output=True,
                     text=True).stdout
print("emitter lines in crontab:", chk.count(tag))
