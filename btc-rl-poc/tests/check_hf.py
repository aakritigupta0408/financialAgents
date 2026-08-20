"""Unit checks for the t11 RLHF reward blend."""
from btc_rl.online import _hf_bonus, HF_WEIGHT, _ctx_dim, VARIANTS

row_up = {"made_ts": 1000, "delta": 40}
row_dn = {"made_ts": 1000, "delta": -40}
row_flat = {"made_ts": 1000, "delta": 0}

assert _hf_bonus([], row_up) == 0.0                                # no views
hf = [{"ts": 900, "view": 1}]
assert _hf_bonus(hf, row_up) == HF_WEIGHT                          # agree
assert _hf_bonus(hf, row_dn) == -HF_WEIGHT                         # disagree
assert _hf_bonus(hf, row_flat) == 0.0                              # flat call
assert _hf_bonus([{"ts": 900 - 1800, "view": 1}], row_up) == 0.0   # expired
assert _hf_bonus([{"ts": 1001, "view": 1}], row_up) == 0.0         # future view
hf2 = [{"ts": 800, "view": 1}, {"ts": 950, "view": -1}]
assert _hf_bonus(hf2, row_up) == -HF_WEIGHT                        # latest wins
assert _ctx_dim(VARIANTS["t11-h5"]) == 10                          # == t2
print("hf unit checks ok")
