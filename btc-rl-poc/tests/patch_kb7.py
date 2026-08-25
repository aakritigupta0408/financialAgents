"""Wire the kb7-fm zero-shot foundation-model arm (Chronos-Bolt)."""
from pathlib import Path

p = Path(__file__).resolve().parent.parent / "btc_rl" / "online.py"
t = p.read_text()

t = t.replace('KB5_LOGIT_PATH_NAME = "kb5_logit.json"',
'''_CHRONOS = None      # lazy singleton; ~10s first load, 0.05s/predict


def _chronos_p_up(closes: list[float], strike: float,
                  horizon: int) -> tuple[float, float] | None:
    """kb7-fm: zero-shot P(close >= strike at horizon) from a pretrained
    time-series foundation model (Chronos-Bolt small). Quantiles [.1-.9]
    of the forecast at the window close, monotone-interpolated at the
    strike. Decision-time inputs only. Returns (p_up, q80_width)."""
    global _CHRONOS
    try:
        if _CHRONOS is None:
            import torch
            from chronos import BaseChronosPipeline
            _CHRONOS = BaseChronosPipeline.from_pretrained(
                "amazon/chronos-bolt-small", device_map="cpu",
                torch_dtype=torch.float32)
        import torch
        ctx = torch.tensor(closes[-512:], dtype=torch.float32).unsqueeze(0)
        qs = [i / 10 for i in range(1, 10)]
        q, _ = _CHRONOS.predict_quantiles(
            ctx, prediction_length=max(1, horizon), quantile_levels=qs)
        vals = [float(x) for x in q[0, -1]]
        if strike <= vals[0]:
            pr = 0.95
        elif strike >= vals[-1]:
            pr = 0.05
        else:
            pr = 0.5
            for i in range(len(vals) - 1):
                if vals[i] <= strike <= vals[i + 1]:
                    frac = ((strike - vals[i]) / (vals[i + 1] - vals[i])
                            if vals[i + 1] > vals[i] else 0.5)
                    pr = 1.0 - (qs[i] + frac * (qs[i + 1] - qs[i]))
                    break
        return (round(min(.95, max(.05, pr)), 4),
                round(vals[-1] - vals[0], 1))
    except Exception:
        return None


KB5_LOGIT_PATH_NAME = "kb5_logit.json"''')

old = "                    # kb5 — train-where-you-trade arm: only exists on"
new = '''                    # kb7-fm — zero-shot foundation-model arm (Chronos
                    # Bolt): the LLM-timeseries direction, run against
                    # our ladder. No training, no state; a pretrained
                    # forecaster's distribution read at the strike.
                    if ("kb7", pm_mkt["ticker"], slot1) not in kb_made:
                        fm = _chronos_p_up(
                            [b["close"] for b in kbars],
                            pm_mkt["strike"],
                            int(max(1, round(mins_left))))
                        if fm:
                            p7, w80 = fm
                            kb.append({**common, "variant": "kb7",
                                       "p_up": p7,
                                       "call": int(p7 >= 0.5),
                                       "q80_w": w80})
                            kb_made.add(("kb7", pm_mkt["ticker"], slot1))
                    # kb5 — train-where-you-trade arm: only exists on'''
assert old in t
t = t.replace(old, new, 1)
t = t.replace('for v in ("kb", "kb2", "kb3", "kb4", "kb5")',
              'for v in ("kb", "kb2", "kb3", "kb4", "kb5", "kb7")')
p.write_text(t)
print("kb7 wired")
