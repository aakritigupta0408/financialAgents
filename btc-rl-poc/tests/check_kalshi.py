"""One-shot verification for the t10 Kalshi feature block (run manually)."""
from btc_rl.features import kalshi_feature_vector, KALSHI_DIM
from btc_rl.online import _ctx_dim, VARIANTS
from btc_rl.sources import fetch_kalshi_btc15

assert kalshi_feature_vector(None) == [0.0] * KALSHI_DIM
assert kalshi_feature_vector({}) == [0.0] * KALSHI_DIM
v = kalshi_feature_vector({"k_pup": 0.72, "k_dist_bp": 10,
                           "k_tleft": 5 / 15, "k_spread": 0.03})
print("vector:", [round(x, 3) for x in v])
assert abs(v[0] - 0.44) < 1e-9 and abs(v[2] - (-1 / 3)) < 1e-6
assert _ctx_dim(VARIANTS["t10-h15"]) == 14, _ctx_dim(VARIANTS["t10-h15"])
print("t10 ctx dim:", _ctx_dim(VARIANTS["t10-h15"]))

m = fetch_kalshi_btc15()
print("live market:", {k: m[k] for k in ("ticker", "strike", "yes_bid",
                                         "yes_ask", "close_time")} if m else None)
print("unit checks ok")
