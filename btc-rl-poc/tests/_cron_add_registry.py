"""One-shot: append emit_registry.py + emit_world.py to the 10-min
audit cron chain (idempotent — world.json was previously emitted only
manually; it belongs on the cadence)."""
import subprocess

cur = subprocess.run(["crontab", "-l"], capture_output=True,
                     text=True).stdout
lines = cur.rstrip("\n").split("\n")
changed = False
for tag, script in (("emit_registry.py", "scripts/emit_registry.py"),
                    ("emit_world.py", "scripts/emit_world.py")):
    if tag in cur:
        continue
    for i, l in enumerate(lines):
        if "tests/invariants.py" in l:
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
print("registry:", out.count("emit_registry.py"),
      "· world:", out.count("emit_world.py"))
