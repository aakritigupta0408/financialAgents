"""One-shot: email a status update to the owner via the local MTA,
then print the mail queue so delivery can be verified honestly."""
import subprocess
import time

TO = "aakriti.gupta4894@gmail.com"
BODY = f"""BTC 7PM Oracle — status {time.strftime('%Y-%m-%d %H:%M %Z')}

LIVE NOW
- Playground homepage shipped (game, 8 trader cards, $1k wallet
  replay) — old home preserved as instrument.html
- Wave 1 live: World Map, System Clock, Agent HQ, Museum of Failed Ideas
- Decision Board live on Metrics Lab: CI, P(delta>0), MDE/power,
  veto decomposition, auto PROMOTE/HOLD/KILL states

YOUR TWO DECISIONS IMPLEMENTED + DAEMON RESTARTED
- Gambler v3: keeps 33% stakes, fresh $10k, profits above $10k
  auto-withdrawn (ledger = wd_c on each settled row; withdrawals never
  reinvested). Saver ledger = skim_c (already existed).
- M1 calibration: shadow-only drift instrument, memory 150->50 windows.

RUNNING
- Nav + global-search unification agent building sitewide
- Next: withdrawals on trader cards, Watchtower, registries

Standings: M8 delta +6.0c/$1, P(>0) 97%, LLR 1.18/5.66.
Desk healthy, 168+ windows scored.
— Fable (Claude Code), btc-rl-poc
"""

p = subprocess.run(
    ["mail", "-s", "BTC Oracle update: Gambler v3 live, "
     "Playground shipped", TO], input=BODY, text=True)
print("mail exit:", p.returncode)
time.sleep(3)
q = subprocess.run(["mailq"], capture_output=True, text=True)
print("mailq:\n", (q.stdout or q.stderr)[:800])
