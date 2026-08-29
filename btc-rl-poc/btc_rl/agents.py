"""Agents, simplest first. Each maps (state, price_now) -> predicted price.

Level 0  PersistenceAgent : predict the current price (no learning; baseline).
Level 1  TabularQAgent    : epsilon-greedy tabular Q over discretized states.
Next     linear function approximation, then a small DQN (see README roadmap).
"""
from __future__ import annotations

import math
import random
from collections import defaultdict

from . import config


class PlattCalibrator:
    """Online Platt scaling with a forgetting factor: maps a model's
    stated probability p to a calibrated one, sigmoid(a + b*logit(p)).

    Damped Newton steps on the log-loss with exponentially decayed
    statistics, so the fit tracks a MOVING miscalibration instead of
    averaging over regimes. Measured 2026-08-28: both a and b shifted
    between the halves of four of six arms' histories (kb went
    a+0.03/b0.56 -> a-0.45/b1.34), so a growing-window fit would lag
    the very thing it corrects.

    a=0, b=1 is the identity. b<1 shrinks over-extreme probabilities
    toward 0.5 — which is exactly the tier-1 band under-dispersion
    defect (bands cover 74-76% vs 80%), corrected at the point of use.

    Platt 1999; Guo et al. 2017 (ICML) on why modern models need this;
    Dawid 1984 for the prequential (test-then-train) discipline.
    """

    # v3 (2026-08-29, D-m1-future): SHADOW-ONLY drift instrument with
    # SHORTER memory. The 150-window fit measurably lagged a
    # miscalibration that flips direction within a day (worse-than-raw
    # log-loss on all nine arms); 50 windows ~ half a trading day, so
    # the fitted (a, b) now tracks the drift it is meant to display.
    # p_m1 is never an input to any decision — it is instrumentation.
    def __init__(self, decay: float = 0.985, lr: float = 0.35,
                 warm: int = 20, window: int = 50, refit: int = 3):
        self.a = 0.0
        self.b = 1.0
        self.decay = decay          # memory of the log-loss scoreboard
        self.lr = lr                # (retained for state compatibility)
        self.warm = warm            # identity until this many updates
        self.window = window        # sliding refit window (windows)
        self.refit = refit          # refit every k updates (cheap)
        self.hist: list = []        # recent (p, y) pairs, <= window
        self.updates = 0
        self.n_eff = 0.0            # decayed sample count
        self.ll = 0.0               # decayed log-loss (calibrated)
        self.ll_raw = 0.0           # decayed log-loss (raw) — the control

    @staticmethod
    def _logit(p: float) -> float:
        p = min(max(p, 1e-6), 1.0 - 1e-6)
        return math.log(p / (1.0 - p))

    @staticmethod
    def _sig(z: float) -> float:
        return 1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, z))))

    def predict(self, p: float) -> float:
        """Calibrated probability. Identity while warming up, so a cold
        calibrator can never make a live arm worse."""
        if self.updates < self.warm:
            return p
        return self._sig(self.a + self.b * self._logit(p))

    def update(self, p: float, y: int) -> None:
        """Test-then-train: score the CURRENT fit on this outcome first
        (the honest prequential number), then step toward it."""
        x = self._logit(p)
        pc = self.predict(p)
        eps = 1e-9
        self.ll = self.decay * self.ll - math.log(
            max(eps, pc if y else 1.0 - pc))
        self.ll_raw = self.decay * self.ll_raw - math.log(
            max(eps, p if y else 1.0 - p))
        self.n_eff = self.decay * self.n_eff + 1.0
        self.updates += 1
        self.hist.append((p, y))
        if len(self.hist) > self.window:
            self.hist.pop(0)
        # Refit by BATCH Newton on the recent window. Rejected the
        # incremental variants for concrete, measured reasons:
        #  - per-sample 2-param Newton: Hessian is exactly rank-1
        #    (det = w^2x^2 - (wx)^2 = 0), so b is unidentifiable and the
        #    fit degenerates to intercept drift (measured: b stuck at
        #    1.000 across all nine arms).
        #  - decayed-accumulator Newton: the gradient keeps re-applying
        #    residuals computed at stale parameters, compounding into
        #    overshoot (measured: a = -0.64 vs the batch answer -0.13).
        # A sliding window is bounded work, cannot overshoot, tracks
        # drift by construction, and reproduces the offline fit the
        # audit computes — so daemon and dashboard agree by design.
        if len(self.hist) >= self.warm and self.updates % self.refit == 0:
            self._refit()

    def _refit(self) -> None:
        a, b = 0.0, 1.0
        xs = [(self._logit(p), y) for p, y in self.hist]
        for _ in range(40):
            g0 = g1 = h00 = h01 = h11 = 0.0
            for x, y in xs:
                m = self._sig(a + b * x)
                w = m * (1.0 - m)
                g0 += y - m
                g1 += (y - m) * x
                h00 += w
                h01 += w * x
                h11 += w * x * x
            h00 += 1e-3             # ridge: keeps it invertible and pulls
            h11 += 1e-3             # an unidentified fit toward identity
            det = h00 * h11 - h01 * h01
            if abs(det) < 1e-10:
                break
            da = (g0 * h11 - g1 * h01) / det
            db = (g1 * h00 - g0 * h01) / det
            a += da
            b += db
            if abs(da) + abs(db) < 1e-9:
                break
        if not (math.isfinite(a) and math.isfinite(b)):
            return
        self.a = max(-4.0, min(4.0, a))
        self.b = max(0.15, min(4.0, b))

    def mean_ll(self):
        """(calibrated, raw) decayed mean log-loss — the shadow verdict.
        Calibrated below raw means the layer is earning its place."""
        if self.n_eff < 1:
            return None
        return self.ll / self.n_eff, self.ll_raw / self.n_eff

    def to_dict(self) -> dict:
        return {"a": self.a, "b": self.b, "updates": self.updates,
                "n_eff": self.n_eff, "ll": self.ll, "ll_raw": self.ll_raw,
                "decay": self.decay, "lr": self.lr, "warm": self.warm,
                "window": self.window, "refit": self.refit,
                "hist": [[round(p, 5), y] for p, y in self.hist]}

    @classmethod
    def from_dict(cls, d: dict) -> "PlattCalibrator":
        # Config sovereignty (same lesson as Treatment.load, 08-28):
        # hyperparameters come from the CODE, only evidence is
        # restored — otherwise persisted window/warm/refit silently
        # override a retune forever after the first state save.
        c = cls()
        c.a, c.b = d.get("a", 0.0), d.get("b", 1.0)
        c.updates = d.get("updates", 0)
        c.n_eff = d.get("n_eff", 0.0)
        c.ll = d.get("ll", 0.0)
        c.ll_raw = d.get("ll_raw", 0.0)
        c.hist = [(float(p), int(y))
                  for p, y in d.get("hist", [])][-c.window:]
        return c


class PersistenceAgent:
    """Level 0: tomorrow looks like today. The bar every RL agent must clear."""

    name = "persistence-baseline"

    def act(self, state, price_now: float, explore: bool = False) -> int:
        return 0  # delta of zero dollars

    def learn(self, state, action_idx: int, r: float) -> None:
        pass

    def predicted_price(self, price_now: float, delta: int) -> float:
        return price_now + delta


class LinUCBAgent:
    """Level 2 treatment: LinUCB contextual bandit over the integer-delta arms.

    Each arm a keeps a ridge model (A_a = lam*I + sum x xT, b_a = sum r x);
    selection is optimistic: argmax_a  theta_a . x + alpha * sqrt(xT A_a^-1 x).
    Reward is the shaped prediction reward plus a direction bonus (see
    btc_rl/online.py). Purely online — no batch retrain needed.
    """

    name = "linucb"

    def __init__(self, dim: int, alpha: float = 1.0, lam: float = 1.0,
                 n_arms: int | None = None):
        import numpy as np
        self.np = np
        self.dim = dim
        self.alpha = alpha
        self.lam = lam
        self.n_arms = n_arms or len(config.ACTION_DELTAS)
        self.A = [np.eye(dim) * lam for _ in range(self.n_arms)]
        self.b = [np.zeros(dim) for _ in range(self.n_arms)]
        self.pulls = [0] * self.n_arms

    def pad_to(self, dim: int) -> None:
        """Grow the context dimension in place, keeping every learned
        ridge statistic: old A embeds top-left, new dims start at the
        ridge prior (lam*I) with zero reward mass — mathematically the
        same posterior as if the new features had been constant 0."""
        if dim <= self.dim:
            return
        np = self.np
        for i in range(self.n_arms):
            A2 = np.eye(dim) * getattr(self, "lam", 1.0)
            A2[:self.dim, :self.dim] = self.A[i]
            b2 = np.zeros(dim)
            b2[:self.dim] = self.b[i]
            self.A[i], self.b[i] = A2, b2
        self.dim = dim

    def select(self, x: list[float], greedy: bool = False) -> int:
        """UCB selection; greedy=True drops the exploration bonus (for eval)."""
        np = self.np
        xv = np.asarray(x)
        best, best_p = 0, -np.inf
        for a in range(self.n_arms):
            A_inv = np.linalg.inv(self.A[a])
            theta = A_inv @ self.b[a]
            p = float(theta @ xv)
            if not greedy:
                p += self.alpha * float(np.sqrt(xv @ A_inv @ xv))
            if p > best_p:
                best, best_p = a, p
        return best

    def update(self, x: list[float], arm: int, r: float) -> None:
        xv = self.np.asarray(x)
        self.A[arm] += self.np.outer(xv, xv)
        self.b[arm] += r * xv
        self.pulls[arm] += 1

    @property
    def total_pulls(self) -> int:
        return sum(self.pulls)

    def to_dict(self) -> dict:
        return {"dim": self.dim, "alpha": self.alpha, "n_arms": self.n_arms,
                "A": [a.tolist() for a in self.A],
                "b": [b.tolist() for b in self.b], "pulls": self.pulls}

    @classmethod
    def from_dict(cls, d: dict) -> "LinUCBAgent":
        agent = cls(dim=d["dim"], alpha=d.get("alpha", 1.0),
                    n_arms=d.get("n_arms"))
        agent.A = [agent.np.asarray(a) for a in d["A"]]
        agent.b = [agent.np.asarray(b) for b in d["b"]]
        agent.pulls = d.get("pulls", [0] * agent.n_arms)
        return agent


class LinearQAgent:
    """Level 2: linear function approximation — Q(s,a) = w_a · x, SGD updates.

    The roadmap's L2 rung: continuous features, one weight vector per
    (vol-scaled) action arm, epsilon-greedy while training, plain SGD
    w_a += lr * (r - Q) * x. Duck-types LinUCBAgent's select/update API.
    """

    name = "linear-q"

    def __init__(self, dim: int, lr: float = 0.01, epsilon: float = 0.05,
                 n_arms: int | None = None, seed: int = 7):
        import numpy as np
        self.np = np
        self.dim = dim
        self.lr = lr
        self.epsilon = epsilon
        self.n_arms = n_arms or len(config.K_FACTORS)
        self.w = [np.zeros(dim) for _ in range(self.n_arms)]
        self.pulls = [0] * self.n_arms
        self.rng = __import__("random").Random(seed)

    def pad_to(self, dim: int) -> None:
        """Grow the context dimension in place: learned weights keep their
        positions, new features start at zero weight."""
        if dim <= self.dim:
            return
        np = self.np
        self.w = [np.concatenate([w, np.zeros(dim - self.dim)])
                  for w in self.w]
        self.dim = dim

    def select(self, x, greedy: bool = False) -> int:
        if not greedy and self.rng.random() < self.epsilon:
            return self.rng.randrange(self.n_arms)
        xv = self.np.asarray(x)
        qs = [float(w @ xv) for w in self.w]
        return max(range(self.n_arms), key=lambda i: qs[i])

    def update(self, x, arm: int, r: float) -> None:
        xv = self.np.asarray(x)
        q = float(self.w[arm] @ xv)
        self.w[arm] = self.w[arm] + self.lr * (r - q) * xv
        self.pulls[arm] += 1

    @property
    def total_pulls(self) -> int:
        return sum(self.pulls)

    def to_dict(self) -> dict:
        return {"dim": self.dim, "lr": self.lr, "n_arms": self.n_arms,
                "w": [w.tolist() for w in self.w], "pulls": self.pulls}

    @classmethod
    def from_dict(cls, d: dict) -> "LinearQAgent":
        agent = cls(dim=d["dim"], lr=d.get("lr", 0.01), n_arms=d.get("n_arms"))
        agent.w = [agent.np.asarray(w) for w in d["w"]]
        agent.pulls = d.get("pulls", [0] * agent.n_arms)
        return agent


class DistDQNAgent:
    """Level 3: small distributional network — predict the DELTA DISTRIBUTION,
    act on its mode.

    An MLP maps the context to a categorical distribution over the vol-scaled
    action bins (K_FACTORS as bin centers, sigma units). Training is
    cross-entropy against the realized bin (one-step distributional target);
    greedy action = distribution mode; exploratory action = a sample from the
    predicted distribution itself.
    """

    name = "dist-dqn"

    def __init__(self, dim: int, n_arms: int | None = None, lr: float = 1e-3,
                 seed: int = 7):
        import torch
        from torch import nn
        self.torch = torch
        self.dim = dim
        # distribution support = action bins + OVERFLOW bins (targets only):
        # without them, clipping piles tail mass onto the ±1.5 edges and the
        # mode degenerates into a permanent ±1.5-sigma bet
        self.bins = sorted(set(config.K_FACTORS) | {2.0, -2.0, 3.0, -3.0})
        self.n_arms = len(self.bins)
        self._act_ok = [abs(k) <= 1.5 + 1e-9 for k in self.bins]
        torch.manual_seed(seed)
        self.net = nn.Sequential(
            nn.Linear(dim, 64), nn.ReLU(),
            nn.Linear(64, 64), nn.ReLU(),
            nn.Linear(64, self.n_arms))
        self.opt = torch.optim.Adam(self.net.parameters(), lr=lr)
        self.loss_fn = nn.CrossEntropyLoss()
        self.steps = 0

    def _logits(self, x):
        return self.net(self.torch.tensor(x, dtype=self.torch.float32))

    def select(self, x, greedy: bool = False) -> int:
        with self.torch.no_grad():
            logits = self._logits(x).clone()
            for i, ok in enumerate(self._act_ok):
                if not ok:            # overflow bins are never actions
                    logits[i] = -1e9
            if greedy:
                return int(logits.argmax())
            probs = self.torch.softmax(logits, dim=-1)
            return int(self.torch.multinomial(probs, 1))

    def target_bin(self, z: float) -> int:
        z = max(-3.0, min(3.0, z))
        return min(range(len(self.bins)), key=lambda i: abs(self.bins[i] - z))

    def probs(self, x) -> list[float]:
        """The predicted delta distribution (full support incl. tail bins)."""
        with self.torch.no_grad():
            return self.torch.softmax(self._logits(x), dim=-1).tolist()

    def learn_dist(self, x, z: float) -> None:
        t = self.torch.tensor([self.target_bin(z)])
        loss = self.loss_fn(self._logits(x).unsqueeze(0), t)
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()
        self.steps += 1

    def learn_batch(self, X, Z) -> float:
        t = self.torch.tensor([self.target_bin(z) for z in Z])
        xt = self.torch.tensor(X, dtype=self.torch.float32)
        loss = self.loss_fn(self.net(xt), t)
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()
        self.steps += len(Z)
        return float(loss)

    @property
    def total_pulls(self) -> int:
        return self.steps

    def save(self, path) -> None:
        # BUG FIX 2026-08-28: checkpoints now carry the Adam state.
        # Without it, every daemon restart silently reset exp_avg /
        # exp_avg_sq / step, putting t8/t9 back into bias-correction
        # warm-up at ~12 online updates per hour — permanently.
        self.torch.save({"dim": self.dim, "n_arms": self.n_arms,
                         "steps": self.steps,
                         "state": self.net.state_dict(),
                         "opt_state": self.opt.state_dict()}, path)

    @classmethod
    def load(cls, path) -> "DistDQNAgent":
        import torch
        ck = torch.load(path, weights_only=False)
        agent = cls(dim=ck["dim"], n_arms=ck["n_arms"])
        agent.net.load_state_dict(ck["state"])
        if ck.get("opt_state"):
            try:
                agent.opt.load_state_dict(ck["opt_state"])
            except Exception:
                pass               # older checkpoint: fresh Adam is fine
        agent.steps = ck.get("steps", 0)
        return agent


class LSTMDistAgent(DistDQNAgent):
    """Level 4: sequence model — an LSTM over the raw 1m return stream,
    predicting the same delta distribution and acting on its mode."""

    name = "lstm-dist"

    def __init__(self, dim: int = 60, n_arms: int | None = None,
                 lr: float = 1e-3, seed: int = 7):
        super().__init__(dim=dim, n_arms=n_arms, lr=lr, seed=seed)
        import torch
        from torch import nn

        class _SeqNet(nn.Module):
            def __init__(self, n_out):
                super().__init__()
                self.lstm = nn.LSTM(1, 32, batch_first=True)
                self.head = nn.Linear(32, n_out)

            def forward(self, x):
                single = x.dim() == 1
                if single:
                    x = x.unsqueeze(0)
                out, _ = self.lstm(x.unsqueeze(-1))
                y = self.head(out[:, -1])
                return y.squeeze(0) if single else y

        torch.manual_seed(seed)
        self.net = _SeqNet(self.n_arms)
        self.opt = torch.optim.Adam(self.net.parameters(), lr=lr)


class TabularQAgent:
    """Level 1: one-step Q-learning (a contextual bandit, so no bootstrapping).

    Q(s,a) <- Q(s,a) + alpha * (r - Q(s,a)); epsilon-greedy exploration.
    """

    name = "tabular-q"

    def __init__(self, seed: int = 7):
        self.q: dict[tuple, list[float]] = defaultdict(
            lambda: [0.0] * len(config.ACTION_DELTAS))
        self.counts: dict[tuple, list[int]] = defaultdict(
            lambda: [0] * len(config.ACTION_DELTAS))
        self.epsilon = config.EPSILON_START
        self.rng = random.Random(seed)

    def act(self, state, price_now: float, explore: bool = False) -> int:
        if explore and self.rng.random() < self.epsilon:
            return self.rng.randrange(len(config.ACTION_DELTAS))
        qs = self.q[state]
        best = max(range(len(qs)), key=lambda i: qs[i])
        return best

    def learn(self, state, action_idx: int, r: float) -> None:
        self.counts[state][action_idx] += 1
        q = self.q[state][action_idx]
        self.q[state][action_idx] = q + config.ALPHA * (r - q)

    def decay_epsilon(self, epoch: int, total_epochs: int) -> None:
        frac = epoch / max(1, total_epochs - 1)
        self.epsilon = (config.EPSILON_START
                        + frac * (config.EPSILON_END - config.EPSILON_START))

    def predicted_price(self, price_now: float, delta_idx_or_delta) -> float:
        return price_now + delta_idx_or_delta

    def delta_for(self, action_idx: int) -> int:
        return config.ACTION_DELTAS[action_idx]


class BinaryLogit:
    """Online logistic regression for the Kalshi binary task: predicts
    P(window close >= strike) directly from market + microstructure
    features, one SGD step per settled call. Log-loss training keeps the
    output probability naturally calibrated."""

    def __init__(self, dim: int, lr: float = 0.05, l2: float = 1e-4):
        import numpy as np
        self.np = np
        self.dim = dim
        self.lr = lr
        self.l2 = l2
        self.w = np.zeros(dim)
        self.updates = 0

    def predict(self, x) -> float:
        z = float(self.np.asarray(x) @ self.w)
        return 1.0 / (1.0 + self.np.exp(-max(-30.0, min(30.0, z))))

    def update(self, x, y: int) -> None:
        xv = self.np.asarray(x)
        p = self.predict(xv)
        # decaying step: adaptive while young, stable once evidence has
        # accumulated — a constant lr lets 2-3 surprising windows (whose
        # ~15 per-minute rows share one outcome) whip the weights around
        lr = self.lr / (1.0 + self.updates / 400.0)
        self.w -= lr * ((p - y) * xv + self.l2 * self.w)
        self.updates += 1

    def to_dict(self) -> dict:
        return {"dim": self.dim, "w": self.w.tolist(),
                "updates": self.updates}

    @classmethod
    def from_dict(cls, d: dict) -> "BinaryLogit":
        m = cls(d["dim"])
        m.w = m.np.asarray(d["w"], dtype=float)
        m.updates = d["updates"]
        return m
