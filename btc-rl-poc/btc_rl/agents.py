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
    Reward is the betting P&L (+40 within +-$10, else -10), so the bandit
    directly maximizes the bankroll. Purely online — no batch retrain needed.
    """

    name = "linucb"

    def __init__(self, dim: int, alpha: float = 1.0, lam: float = 1.0,
                 n_arms: int | None = None):
        import numpy as np
        self.np = np
        self.dim = dim
        self.alpha = alpha
        # n_arms beyond the delta count adds an ABSTAIN arm (bet nothing,
        # reward always 0) — the bandit learns when not to play.
        self.n_arms = n_arms or len(config.ACTION_DELTAS)
        self.abstain_idx = (self.n_arms - 1
                            if self.n_arms > len(config.ACTION_DELTAS) else None)
        self.A = [np.eye(dim) * lam for _ in range(self.n_arms)]
        self.b = [np.zeros(dim) for _ in range(self.n_arms)]
        self.pulls = [0] * self.n_arms

    def select(self, x: list[float]) -> int:
        np = self.np
        xv = np.asarray(x)
        best, best_p = 0, -np.inf
        for a in range(self.n_arms):
            A_inv = np.linalg.inv(self.A[a])
            theta = A_inv @ self.b[a]
            p = float(theta @ xv + self.alpha * np.sqrt(xv @ A_inv @ xv))
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
