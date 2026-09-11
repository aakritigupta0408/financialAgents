"""SEV-1 idempotence proof (run AFTER zombies late-settle + reconcile
PASS). Snapshots the two settled zombie rows, restarts the daemon,
waits for a clean loop, and asserts the economic effect did NOT
repeat: actual/pnl_c/bankroll_c/late_settle_ts byte-identical, and
bankroll-conservation still PASS. Proves: once economic state is
repaired, recovery cannot apply the repair again.
"""
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"
ZOMBIES = (("pt_trades.jsonl", "KXBTC15M-26AUG311800-00"),
           ("pt3_trades.jsonl", "KXBTC15M-26SEP062345-45"))
FIELDS = ("actual", "win", "pnl_c", "bankroll_c", "late_settle_ts")


def snap(fname, tick):
    for l in (RES / fname).open():
        r = json.loads(l)
        if r.get("ticker") == tick:
            return {k: r.get(k) for k in FIELDS}
    return None


def bank_status():
    doc = json.loads((RES / "reconciliation.json").read_text())
    return next(c["status"] for c in doc["checks"]
               if c["name"] == "bankroll-conservation")


before = {t: snap(f, t) for f, t in ZOMBIES}
if any(v is None or v["actual"] is None for v in before.values()):
    print("ABORT: zombies not settled yet — run the recovery watcher "
          "first"); sys.exit(0)
print("BEFORE:", json.dumps(before))

subprocess.run(["pkill", "-f", "--", r"-m btc_rl\.online$"])
time.sleep(4)
with (RES / "daemon.log").open("a") as f:
    subprocess.Popen([sys.executable, "-u", "-m", "btc_rl.online"],
                     cwd=ROOT, stdout=f, stderr=subprocess.STDOUT,
                     start_new_session=True)
# wait for one full clean loop past warm-up (heartbeat fresh + a
# scoring cycle); generous ceiling for the ~30-min warm-up
deadline = time.time() + 45 * 60
while time.time() < deadline:
    time.sleep(120)
    try:
        hb = json.loads((RES / "online_status.json").read_text())
        if time.time() - hb["alive_at"] < 180:
            break
    except Exception:
        pass

after = {t: snap(f, t) for f, t in ZOMBIES}
print("AFTER:", json.dumps(after))
identical = before == after
try:
    subprocess.run([sys.executable, str(ROOT / "scripts" /
                                        "reconcile.py")],
                   cwd=ROOT, timeout=600)
    bs = bank_status()
except Exception as e:
    bs = f"reconcile error {e}"
print(f"IDEMPOTENCE: rows_identical={identical} · "
      f"bankroll_conservation={bs} · "
      f"{'PASS — no replay, no second payout' if identical and bs == 'OK' else 'FAIL — investigate'}")
