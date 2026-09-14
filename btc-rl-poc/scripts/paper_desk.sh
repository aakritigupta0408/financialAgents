#!/usr/bin/env bash
# Paper desk control — start/restart the SIMULATED (paper) trading daemon with the
# exact-BRTI runtime activated (DT-01). This is the owner-gated "turn on paper
# trading" control: it settles live windows on official CF Benchmarks BRTI so the
# live A/B (T0 vs T1) begins collecting prospective evidence.
#
# PAPER / SIMULATION ONLY — no real-money orders are ever placed.
#
# Usage:
#   scripts/paper_desk.sh          # activate: restart daemon with exact-BRTI ON
#   scripts/paper_desk.sh status   # show heartbeat + flag state, do not touch daemon
#
# The daemon is relaunched detached (nohup + disown -> reparented to init) so it
# survives the shell/session that started it. It has no external supervisor; run
# this script again to restart after a reboot.
set -u

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT" || exit 1
PY="/opt/anaconda3/bin/python3"
[ -x "$PY" ] || PY="python3"
LOG="$ROOT/results/daemon.log"

status() {
  echo "== paper desk status =="
  pgrep -f 'btc_rl\.online' >/dev/null && echo "daemon: RUNNING (pid $(pgrep -f 'btc_rl\.online' | tr '\n' ' '))" || echo "daemon: STOPPED"
  "$PY" -c "import os,sys; sys.path.insert(0,'$ROOT'); from btc_rl import contract_truth as ct; print('EXACT_BRTI_RUNTIME_ENABLED (this shell):', os.environ.get('EXACT_BRTI_RUNTIME_ENABLED','')); print('BRTI health rest_ok/history_ok/age_s:', {k:ct.health().get(k) for k in ('rest_ok','history_ok','age_s')})"
}

if [ "${1:-}" = "status" ]; then
  status
  exit 0
fi

# --- activate: graceful stop, then relaunch with both exact-BRTI flags ON ---
export EXACT_BRTI_CAPTURE_ENABLED=1
export EXACT_BRTI_RUNTIME_ENABLED=1

if pgrep -f 'btc_rl\.online' >/dev/null; then
  echo "stopping running daemon (SIGTERM) ..."
  pkill -TERM -f 'btc_rl\.online'
  for _ in $(seq 1 20); do
    pgrep -f 'btc_rl\.online' >/dev/null || break
    sleep 0.5
  done
  if pgrep -f 'btc_rl\.online' >/dev/null; then
    echo "daemon did not exit on SIGTERM; sending SIGKILL"
    pkill -KILL -f 'btc_rl\.online'
    sleep 1
  fi
fi

echo "$(date '+%Y-%m-%d %H:%M:%S') paper_desk.sh: starting btc_rl.online with EXACT_BRTI_RUNTIME_ENABLED=1" >> "$LOG"
nohup "$PY" -u -m btc_rl.online >> "$LOG" 2>&1 &
disown
sleep 3
if pgrep -f 'btc_rl\.online' >/dev/null; then
  echo "daemon RESTARTED with exact-BRTI runtime ON (pid $(pgrep -f 'btc_rl\.online' | tr '\n' ' '))"
else
  echo "ERROR: daemon failed to start — see $LOG"
  tail -5 "$LOG"
  exit 1
fi
