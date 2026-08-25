"""kb6 — the fast-information arm. Features are the channels the market
may not have priced at 15-min scale: OKX perp lead (gap + ~1-min
momentum), tape imbalance, whale net flow, Kalshi OI delta — plus the
barrier context. Own logistic, own checkpoint, learns per settle.
Additive: no existing arm touched."""
from pathlib import Path

p = Path(__file__).resolve().parent.parent / "btc_rl" / "online.py"
t = p.read_text()

t = t.replace('KB5_LOGIT_PATH_NAME = "kb5_logit.json"',
'''KB6_LOGIT_PATH_NAME = "kb6_logit.json"
KB6_DIM = 12


def _kb6_features(snap: dict | None, k_pup: float | None,
                  bx: list[float], pf: list[float],
                  mins_left: float) -> list[float]:
    """kb6: decision-time fast-information features — perp lead-lag,
    tape aggression, whale net flow, contract OI delta — with barrier
    context. All read from the current snapshot; None -> 0 with the
    market-presence flag carrying the base rate."""
    s = snap or {}
    g = lambda k, sc=1.0: (s.get(k) or 0.0) * sc
    return [
        1.0,
        (k_pup - 0.5) * 2 if k_pup is not None else 0.0,
        1.0 if k_pup is not None else 0.0,
        max(-3.0, min(3.0, g("perp_gap_bp", 0.2))),
        max(-3.0, min(3.0, g("perp_mom_bp", 0.2))),
        max(-1.0, min(1.0, g("tape_imb_1m"))),
        max(-1.0, min(1.0, g("tape_imb_5m"))),
        max(-3.0, min(3.0, g("whale_net_15m", 0.2))),
        max(-3.0, min(3.0, g("k_oi_d", 0.001))),
        bx[3],
        mins_left / 15.0,
    ] + [pf[0]]


KB5_LOGIT_PATH_NAME = "kb5_logit.json"''')

t = t.replace('''    except Exception:
        kb5_logit = BinaryLogit(KB5_DIM)''',
'''    except Exception:
        kb5_logit = BinaryLogit(KB5_DIM)
    kb6_path = RESULTS_DIR / KB6_LOGIT_PATH_NAME
    try:
        kb6_logit = (BinaryLogit.from_dict(json.loads(kb6_path.read_text()))
                     if kb6_path.exists() else BinaryLogit(KB6_DIM))
        if kb6_logit.dim != KB6_DIM:
            kb6_logit = BinaryLogit(KB6_DIM)
    except Exception:
        kb6_logit = BinaryLogit(KB6_DIM)''')

old = '''                    # kb7-fm — zero-shot foundation-model arm (Chronos'''
new = '''                    # kb6 — fast-information arm: perp lead, tape, whale
                    # flow, OI delta; the channels aimed at the EDGE
                    # column rather than the accuracy column
                    b6x = _kb6_features(snap, k_pup, bx, pf, mins_left)
                    p6 = round(kb6_logit.predict(b6x), 4)
                    kb.append({**common, "variant": "kb6", "p_up": p6,
                               "call": int(p6 >= 0.5),
                               "b6x": [round(v, 5) for v in b6x],
                               "trained": kb6_logit.updates})
                    kb_made.add(("kb6", pm_mkt["ticker"], slot1))
                    # kb7-fm — zero-shot foundation-model arm (Chronos'''
assert old in t
t = t.replace(old, new, 1)

t = t.replace('''                if r.get("b5x") and len(r["b5x"]) == kb5_logit.dim:''',
'''                if r.get("b6x") and len(r["b6x"]) == kb6_logit.dim:
                    kb6_logit.update(r["b6x"], outcome)
                    logit_changed = True
                if r.get("b5x") and len(r["b5x"]) == kb5_logit.dim:''')

t = t.replace('''                tmp5 = (RESULTS_DIR / KB5_LOGIT_PATH_NAME).with_suffix(".tmp5")''',
'''                tmp6 = (RESULTS_DIR / KB6_LOGIT_PATH_NAME).with_suffix(".tmp6")
                tmp6.write_text(json.dumps(kb6_logit.to_dict()))
                tmp6.replace(RESULTS_DIR / KB6_LOGIT_PATH_NAME)
                tmp5 = (RESULTS_DIR / KB5_LOGIT_PATH_NAME).with_suffix(".tmp5")''')

t = t.replace('for v in ("kb", "kb2", "kb3", "kb4", "kb5", "kb7")',
              'for v in ("kb", "kb2", "kb3", "kb4", "kb5", "kb6", "kb7")')
p.write_text(t)
print("kb6 wired")
