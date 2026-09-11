"""One-shot: append tests/invariants.py to the 10-min audit cron
chain (idempotent)."""
import subprocess

cur = subprocess.run(["crontab", "-l"], capture_output=True,
                     text=True).stdout
tag = "tests/invariants.py"
if tag in cur:
    print("cron already has invariants")
else:
    lines = cur.rstrip("\n").split("\n")
    for i, l in enumerate(lines):
        if "emit_decision_board.py" in l:
            lines[i] = (l + ' && /opt/anaconda3/bin/python3 '
                        '"/Users/aakritigupta/.claude/worktrees/'
                        'btc-rl-poc/btc-rl-poc/tests/invariants.py" '
                        '>> /tmp/btc_audit.log 2>&1')
    p = subprocess.run(["crontab", "-"],
                       input="\n".join(lines) + "\n", text=True)
    print("cron updated:", p.returncode == 0)
print("lines with invariants:",
      subprocess.run(["crontab", "-l"], capture_output=True,
                     text=True).stdout.count(tag))
