"""One-shot: replace the oversized */10 crontab chain line with a
single short call to scripts/audit_chain.py (idempotent)."""
import subprocess

cur = subprocess.run(["crontab", "-l"], capture_output=True,
                     text=True).stdout
new_line = ('*/10 * * * * /opt/anaconda3/bin/python3 '
            '"/Users/aakritigupta/.claude/worktrees/btc-rl-poc/'
            'btc-rl-poc/scripts/audit_chain.py" '
            '>> /tmp/btc_audit.log 2>&1')
lines = [new_line if "run_audit.py" in l else l
         for l in cur.rstrip("\n").split("\n")]
if "audit_chain.py" not in cur:
    p = subprocess.run(["crontab", "-"],
                       input="\n".join(lines) + "\n", text=True)
    print("crontab replaced:", p.returncode == 0)
out = subprocess.run(["crontab", "-l"], capture_output=True,
                     text=True).stdout
line = [l for l in out.split("\n") if "*/10" in l][0]
print("*/10 line length now:", len(line), "chars")
