"""Central configuration for the BTC RL price-prediction POC."""
from zoneinfo import ZoneInfo

# The user's spec says PST; we use the Pacific wall clock (handles PDT too).
PACIFIC = ZoneInfo("America/Los_Angeles")

# Decision + target times (Pacific wall clock).
DECISION_HHMM = (18, 45)          # agent observes state and commits predictions
HORIZONS_MIN = [1, 5, 15, 30]     # prediction horizons, minutes ahead

# Headline evaluation slots: which wall-clock target is scored via which horizon.
TARGET_SLOTS = [((19, 0), 15), ((19, 15), 30)]   # 7:00 PM via 15m, 7:15 via 30m
TARGETS_HHMM = [hhmm for hhmm, _ in TARGET_SLOTS]

# Data window fetched per day (Pacific): lookback for features + targets.
DAY_WINDOW_START_HHMM = (14, 30)
DAY_WINDOW_END_HHMM = (19, 20)

HISTORY_DAYS = 120                # how many past days to fetch
LOOKBACK_MIN = 60                 # minutes of bars required to build features
TRAIN_FRACTION = 0.8              # chronological split by day

# Action space: predicted integer dollar delta from the current price.
ACTION_DELTAS = sorted({0, 1, -1, 2, -2, 3, -3, 5, -5, 8, -8, 13, -13,
                        21, -21, 34, -34, 55, -55, 89, -89})

# Rewards (user spec): +1 when int(pred) == int(actual), else penalty.
REWARD_HIT = 1.0
REWARD_MISS = -1.0
SHAPED_SCALE = 100.0              # shaped mode: reward = -|error| / SHAPED_SCALE

# Q-learning hyperparameters.
ALPHA = 0.1
EPSILON_START = 0.3
EPSILON_END = 0.01
EPOCHS = 30

COINBASE_BASE = "https://api.exchange.coinbase.com"
FNG_URL = "https://api.alternative.me/fng/"
OKX_FUNDING_URL = "https://www.okx.com/api/v5/public/funding-rate?instId=BTC-USDT-SWAP"
