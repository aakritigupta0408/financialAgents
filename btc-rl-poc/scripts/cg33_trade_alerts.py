"""Email an alert every time the Gated·33% (cg33) treatment ENTERS a trade.

Isolated watcher — it only READS results/cg33_trades.jsonl (written by the live
daemon) and sends mail. It never touches the daemon, T0, or any other arm. Dedup by
(ticker, made_ts) in a state file so each entry alerts exactly once.

PAPER / SIMULATION — the emails describe simulated paper bets, never real orders.

Delivery: authenticated SMTP if CG_SMTP_USER/CG_SMTP_PASS are set (recommended —
reliable to Gmail); otherwise the local `mail` MTA (zero-setup, but may not reach
Gmail without a relay). Recipient: $CG_ALERT_TO or the default below.

Usage:
  python3 scripts/cg33_trade_alerts.py            # one pass: alert any new entries
  python3 scripts/cg33_trade_alerts.py --loop     # poll forever (every POLL_S)
  python3 scripts/cg33_trade_alerts.py --test      # send one test email, then exit
"""
import json
import os
import smtplib
import ssl
import subprocess
import sys
import time
from email.mime.text import MIMEText
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LEDGER = ROOT / "results" / "cg33_trades.jsonl"
STATE = ROOT / "results" / ".cg33_alert_state.json"
TO = os.environ.get("CG_ALERT_TO", "ag.1486c@gmail.com")
IMESSAGE_TO = os.environ.get("CG_ALERT_IMESSAGE")   # phone # / Apple ID -> texts via Messages.app
POLL_S = 45


def _rows():
    if not LEDGER.exists():
        return []
    out = []
    for l in LEDGER.open():
        l = l.strip()
        if l:
            try:
                out.append(json.loads(l))
            except json.JSONDecodeError:
                pass
    return out


def _state():
    if STATE.exists():
        try:
            return set(json.loads(STATE.read_text()))
        except Exception:
            return set()
    return set()


def _save(seen):
    STATE.write_text(json.dumps(sorted(seen)))


def _send(subject, body):
    """Prefer authenticated SMTP (reliable); fall back to the local mail MTA."""
    user = os.environ.get("CG_SMTP_USER")
    pw = os.environ.get("CG_SMTP_PASS")
    if user and pw:
        msg = MIMEText(body)
        msg["Subject"] = subject
        msg["From"] = user
        msg["To"] = TO
        ctx = ssl.create_default_context()
        with smtplib.SMTP_SSL("smtp.gmail.com", 465, context=ctx, timeout=30) as s:
            s.login(user, pw)
            s.sendmail(user, [TO], msg.as_string())
        return "smtp"
    p = subprocess.run(["mail", "-s", subject, TO], input=body, text=True)
    return f"local-mail(rc={p.returncode})"


def _send_imessage(text):
    """Send via Messages.app (iMessage) using the account already signed in — no
    credentials. Requires a one-time macOS Automation grant for the controlling process."""
    to = (IMESSAGE_TO or "").replace('"', "").replace("\\", "")
    text = text.replace('"', "'").replace("\\", "")
    script = ('tell application "Messages"\n'
              '  set svc to 1st service whose service type = iMessage\n'
              f'  set bud to buddy "{to}" of svc\n'
              f'  send "{text}" to bud\n'
              'end tell')
    p = subprocess.run(["osascript", "-e", script], capture_output=True, text=True)
    if p.returncode != 0:
        raise RuntimeError((p.stderr or "osascript failed").strip()[:200])
    return "imessage"


def _sms(r):
    c = lambda v: f"{v}c" if v is not None else "?"          # noqa: E731
    return (f"cg33 ENTERED {(r.get('side') or '').upper()} "
            f"{(r.get('ticker') or '').replace('KXBTC15M-', '')} @ {c(r.get('ask_c'))} · "
            f"stake ${(r.get('stake_c') or 0)/100:.0f} · bank "
            f"${(r.get('bankroll_c') or 0)/100:.2f} (paper)")


def _fmt(r):
    c = lambda v: f"{v}¢" if v is not None else "—"      # noqa: E731
    d = lambda v: f"${v/100:,.2f}" if v is not None else "—"  # noqa: E731
    conf = r.get("p_arm")
    subject = (f"Gated·33% ENTERED {(r.get('side') or '').upper()} "
               f"{(r.get('ticker') or '').replace('KXBTC15M-', '')} @ {c(r.get('ask_c'))}")
    body = f"""Gated · 33% treatment just ENTERED a trade  (PAPER / simulation only)

Window     {r.get('ticker')}
Side       {(r.get('side') or '').upper()}
Entry      {c(r.get('ask_c'))}   (fee {r.get('fee_c')})
Contracts  {r.get('contracts')}
Stake      {d(r.get('stake_c'))}   (33% of bankroll)
Bankroll   {d(r.get('bankroll_c'))}  after entry
Strike     {r.get('strike')}
Leader     {r.get('leader')}  (rec {r.get('rec10')})
Oracle conf {conf}   (gate: 2*|p-0.5| >= 0.20)
Mins left  {r.get('mins_left')}
Entered    {time.strftime('%Y-%m-%d %H:%M:%S %Z', time.localtime(r.get('made_ts')))}

Settles on official BRTI at window close. This is a simulated paper bet — no real
order is placed. cg33 is a deliberate ruin-risk experiment (33% ~ 1.6x Kelly).
— BTC Oracle desk
"""
    return subject, body


def one_pass(seen):
    sent = 0
    for r in _rows():
        if r.get("actual") is not None:      # only alert on ENTRY (still open)
            continue
        key = f"{r.get('ticker')}|{r.get('made_ts')}"
        if key in seen:
            continue
        if IMESSAGE_TO:
            label = _sms(r)
            how = _send_imessage(label)
        else:
            subj, body = _fmt(r)
            how = _send(subj, body); label = subj
        seen.add(key)
        sent += 1
        print(f"alert sent [{how}]: {label}")
    if sent:
        _save(seen)
    return sent


def main():
    if "--test" in sys.argv:
        if IMESSAGE_TO:
            how = _send_imessage("cg33 alerts test — iMessage wired. You'll get a text each "
                                 "time Gated-33% enters a trade. (paper/simulation)")
            print(f"test iMessage sent to {IMESSAGE_TO} via {how}")
        else:
            how = _send("Gated·33% alerts — test",
                        "This confirms the cg33 entry-alert channel is wired. "
                        "You will get one email each time Gated·33% enters a trade. "
                        "PAPER / simulation only.")
            print(f"test email sent to {TO} via {how}")
        return
    seen = _state()
    if "--loop" in sys.argv:
        print(f"cg33 alert watcher: polling every {POLL_S}s -> {TO}")
        while True:
            try:
                one_pass(seen)
            except Exception as e:
                print("watch error:", e, flush=True)
            time.sleep(POLL_S)
    else:
        n = one_pass(seen)
        print(f"cg33 alerts: {n} new entry alert(s) sent to {TO}")


if __name__ == "__main__":
    main()
