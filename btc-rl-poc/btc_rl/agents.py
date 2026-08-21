"""Agents, simplest first. Each maps (state, price_now) -> predicted price.

Level 0  PersistenceAgent : predict the current price (no learning; baseline).
Level 1  TabularQAgent    : epsilon-greedy tabular Q over discretized states.
Next     linear function approximation, then a small DQN (see README roadmap).
"""
from __future__ import annotations

import random
from collections import defaultdict

from . import config


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
        self.n_arms = n_arms or len(config.ACTION_DELTAS)
        self.A = [np.eye(dim) * lam for _ in range(self.n_arms)]
        self.b = [np.zeros(dim) for _ in range(self.n_arms)]
        self.pulls = [0] * self.n_arms

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
        self.torch.save({"dim": self.dim, "n_arms": self.n_arms,
                         "steps": self.steps,
                         "state": self.net.state_dict()}, path)

    @classmethod
    def load(cls, path) -> "DistDQNAgent":
        import torch
        ck = torch.load(path, weights_only=False)
        agent = cls(dim=ck["dim"], n_arms=ck["n_arms"])
        agent.net.load_state_dict(ck["state"])
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
        self.w -= self.lr * ((p - y) * xv + self.l2 * self.w)
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
