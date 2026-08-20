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
