"""Add the Conviction Book: a paper ledger (pb_bets.jsonl) that enters
ONLY on kb5 confident entries (conf_entry=1), at kb5's real ask, one
bet max per window. Additive — no existing arm or ledger touched."""
from pathlib import Path

p = Path(__file__).resolve().parent.parent / "btc_rl" / "online.py"
t = p.read_text()

t = t.replace('KB_BET_LOG_NAME = "kb_bets.jsonl"  # one-shot paper bets on KXBTC15M',
'''KB_BET_LOG_NAME = "kb_bets.jsonl"  # one-shot paper bets on KXBTC15M
PB_BET_LOG_NAME = "pb_bets.jsonl"  # Conviction Book: kb5-gated entries only''')

old = '''    kb_sel_bets = _load_kb_bets(KB_SEL_BET_LOG_NAME)
    kb_sel_tickers = {b["ticker"] for b in kb_sel_bets}'''
assert old in t
t = t.replace(old, old + '''
    pb_bets = _load_kb_bets(PB_BET_LOG_NAME)
    pb_tickers = {b["ticker"] for b in pb_bets}''')

# entry: right after the kb5 row append (kb5 block sets kb_made kb5 key)
old2 = '''                            kb_made.add(("kb5", pm_mkt["ticker"], slot1))
                    # kbf — THE deliverable: one definitive call per window'''
assert old2 in t
t = t.replace(old2, '''                            kb_made.add(("kb5", pm_mkt["ticker"], slot1))
                            # Conviction Book: bet ONLY here — measured-
                            # positive candidates, never the mandatory
                            # control's -EV pockets
                            if (pw5 * 100 >= askc + KB5_BE_MARGIN
                                    and pm_mkt["ticker"] not in pb_tickers):
                                pb_bets.append({
                                    "ticker": pm_mkt["ticker"],
                                    "made_ts": now_ts,
                                    "close_ts": k_close_ts,
                                    "strike": pm_mkt["strike"],
                                    "side": "yes" if sy else "no",
                                    "price_c": round(askc, 1),
                                    "p_win": round(pw5, 4),
                                    "src": "kb5",
                                    "actual": None, "win": None,
                                    "pnl_c": None,
                                })
                                pb_tickers.add(pm_mkt["ticker"])
                    # kbf — THE deliverable: one definitive call per window''')

# settle + persist alongside the selector ledger
old3 = '''            if sel_changed:
                tmp = (RESULTS_DIR / KB_SEL_BET_LOG_NAME).with_suffix(".tmp")'''
assert old3 in t
t = t.replace(old3, '''            pb_changed = False
            for b in pb_bets:
                if b["actual"] is not None or now_ts < b["close_ts"]:
                    continue
                settle_bar = by_ts.get(b["close_ts"] - 60)
                if settle_bar is None or settle_bar.get("synth"):
                    continue
                outcome = int(settle_bar["close"] >= b["strike"])
                b["actual"] = outcome
                b["win"] = int((b["side"] == "yes") == bool(outcome))
                b["pnl_c"] = round((100 - b["price_c"]) if b["win"]
                                   else -b["price_c"], 1)
                pb_changed = True
            if pb_changed or (pb_bets and pb_bets[-1]["actual"] is None
                              and pb_bets[-1]["made_ts"] >= now_ts - 90):
                tmpp = (RESULTS_DIR / PB_BET_LOG_NAME).with_suffix(".tmpp")
                tmpp.write_text("".join(json.dumps(b) + "\\n"
                                        for b in pb_bets))
                tmpp.replace(RESULTS_DIR / PB_BET_LOG_NAME)
            if sel_changed:
                tmp = (RESULTS_DIR / KB_SEL_BET_LOG_NAME).with_suffix(".tmp")''')
p.write_text(t)
print("conviction book wired")
