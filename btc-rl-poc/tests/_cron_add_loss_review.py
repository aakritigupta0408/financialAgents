"""One-shot: append loss_review.py + emit_fill_curve.py to the 10-min
audit cron chain (idempotent)."""
import subprocess

cur = subprocess.run(["crontab", "-l"], capture_output=True,
                     text=True).stdout
lines = cur.rstrip("\n").split("\n")
changed = False
for tag, script in (("loss_review.py", "scripts/loss_review.py"),
                    ("emit_fill_curve.py",
                     "scripts/emit_fill_curve.py")):
    if tag in cur:
        continue
    for i, l in enumerate(lines):
        if "emit_world.py" in l:
            lines[i] = (l + ' && /opt/anaconda3/bin/python3 '
                        '"/Users/aakritigupta/.claude/worktrees/'
                        f'btc-rl-poc/btc-rl-poc/{script}" '
                        '>> /tmp/btc_audit.log 2>&1')
            changed = True
if changed:
    p = subprocess.run(["crontab", "-"],
                       input="\n".join(lines) + "\n", text=True)
    print("cron updated:", p.returncode == 0)
out = subprocess.run(["crontab", "-l"], capture_output=True,
                     text=True).stdout
print("loss_review:", out.count("loss_review.py"),
      "· fill_curve:", out.count("emit_fill_curve.py"))
