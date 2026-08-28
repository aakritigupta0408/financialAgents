"""Champion–challenger routing: every improvement is a TREATMENT that
runs on real live windows, paired against the incumbent, and is promoted
only when the live stream shows a significant win.

Design decisions and why:

* PAIRED, same-window comparison. Champion and challenger are scored on
  the identical window with the identical settled outcome, so the market
  regime cancels out of the difference. This matters here more than
  usual: 08/27 showed a single regime day can swamp an unpaired
  comparison entirely.

* Effective n is WINDOWS, not rows. Entries inside one 15-minute window
  share a single outcome; counting them separately would inflate
  significance. (The same discipline the rest of this project runs
  under — the kb7 "13/14" counting error is the cautionary tale.)

* SEQUENTIAL, not peek-and-declare. Wald's SPRT (1945) tests
  H0: mean paired difference = 0 against H1: = `edge`, accumulating a
  log-likelihood ratio until it crosses a boundary set by (alpha, beta).
  Fixed-horizon tests re-examined every window would inflate the false
  promotion rate; SPRT is valid under continuous monitoring, which is
  exactly how a live desk gets watched.

* Variance is estimated online (Welford) rather than assumed. With an
  unknown, drifting spread a plug-in normal SPRT is an approximation —
  stated plainly rather than dressed up. A minimum sample guards the
  early, badly-estimated variance regime.

* Promotion is REVERSIBLE and STAMPED. A promoted challenger becomes
  champion at a recorded timestamp; the loser is retained, not deleted,
  and keeps running so a regression is visible rather than silent.

References: Wald 1945 (SPRT); Diebold & Mariano 1995 (forecast
comparison, clustered); Arnott, Harvey & Markowitz 2019 (pre-registered
backtest protocol).
"""
from __future__ import annotations

import math
import time


class SPRT:
    """Wald's sequential probability ratio test on a paired difference.

    H0: mean difference is 0 (challenger no better)
    H1: mean difference is `edge` (challenger better by a margin worth
        switching for — pre-registered, not fitted)

    Verdict is one of: "collecting", "promote", "reject".
    """

    def __init__(self, edge: float = 0.02, alpha: float = 0.05,
                 beta: float = 0.10, min_n: int = 40):
        self.edge = edge          # the win size we care about (EV per $1)
        self.alpha = alpha        # P(promote a dud)
        self.beta = beta          # P(miss a real win)
        self.min_n = min_n        # no verdict before this many windows
        self.n = 0
        self.mean = 0.0
        self.m2 = 0.0             # Welford
        self.llr = 0.0

    @property
    def upper(self) -> float:
        return math.log((1 - self.beta) / self.alpha)

    @property
    def lower(self) -> float:
        return math.log(self.beta / (1 - self.alpha))

    def var(self) -> float:
        return self.m2 / (self.n - 1) if self.n > 1 else 0.0

    # Variance floor. The LLR divides by the running variance, so a run
    # of identical early differences (both policies stand down on the
    # same windows => d = 0) drives s2 toward zero and the ratio
    # explodes. Measured while building this: a treatment reached
    # LLR 123 on 154 windows — a false auto-promotion on pure numerical
    # noise. Paired EV differences are O(1) by construction (a bet
    # returns roughly -1..+1 per $1), so 0.01 is a floor far below any
    # real spread while still bounding the ratio.
    VAR_FLOOR = 0.01
    WARMUP = 12                   # estimate spread before scoring it

    def add(self, d: float) -> None:
        """Observe one paired difference (challenger − champion)."""
        self.n += 1
        delta = d - self.mean
        self.mean += delta / self.n
        self.m2 += delta * (d - self.mean)
        if self.n < self.WARMUP:
            return                # variance not yet trustworthy
        s2 = max(self.var(), self.VAR_FLOOR)
        # normal log-likelihood ratio for this observation
        self.llr += (self.edge * (d - self.edge / 2.0)) / s2

    def verdict(self) -> str:
        if self.n < self.min_n:
            return "collecting"
        if self.llr >= self.upper:
            return "promote"
        if self.llr <= self.lower:
            return "reject"
        return "collecting"

    def progress(self) -> float:
        """0..1 toward whichever boundary the evidence is heading for."""
        if self.llr >= 0:
            return min(1.0, self.llr / self.upper) if self.upper else 0.0
        return min(1.0, self.llr / self.lower) if self.lower else 0.0

    def to_dict(self) -> dict:
        return {"edge": self.edge, "alpha": self.alpha, "beta": self.beta,
                "min_n": self.min_n, "n": self.n, "mean": self.mean,
                "m2": self.m2, "llr": self.llr}

    @classmethod
    def from_dict(cls, d: dict) -> "SPRT":
        s = cls(edge=d.get("edge", 0.02), alpha=d.get("alpha", 0.05),
                beta=d.get("beta", 0.10), min_n=d.get("min_n", 40))
        s.n = d.get("n", 0)
        s.mean = d.get("mean", 0.0)
        s.m2 = d.get("m2", 0.0)
        s.llr = d.get("llr", 0.0)
        return s


class Treatment:
    """One challenger policy under live test against the champion.

    `decide(ctx)` returns None to stand down, or a dict with at least
    {"side": "yes"|"no", "ask_c": float} — the bet it would have placed
    on this window. Standing down is a real decision and scores 0, which
    is how a veto treatment (skip the bad windows) can win on EV.
    """

    def __init__(self, key: str, label: str, decide, rationale: str,
                 edge: float = 0.02, min_n: int = 40,
                 baseline: bool = False):
        self.key = key
        self.label = label
        self.decide = decide
        self.rationale = rationale
        # A baseline is paired against itself, so its difference is 0 by
        # construction. Feeding that to the SPRT is not a no-op: a
        # constant 0 accumulates evidence AGAINST "better by `edge`" and
        # the baseline drifts to REJECT (measured: LLR -2.80). Baselines
        # report n and own EV, and are never given a verdict.
        self.baseline = baseline
        self.sprt = SPRT(edge=edge, min_n=min_n)
        self.n_bet = 0
        self.n_skip = 0
        self.ev_sum = 0.0          # own EV per $1, for reporting
        self.promoted_at = None

    def observe(self, own_ev: float | None, champ_ev: float | None
                ) -> None:
        """Score one settled window. None = stood down (EV 0, no risk)."""
        o = 0.0 if own_ev is None else own_ev
        c = 0.0 if champ_ev is None else champ_ev
        if own_ev is None:
            self.n_skip += 1
        else:
            self.n_bet += 1
            self.ev_sum += own_ev
        if self.baseline:
            self.sprt.n += 1        # count windows, never accumulate LLR
            return
        self.sprt.add(o - c)

    def status(self) -> dict:
        v = "baseline" if self.baseline else self.sprt.verdict()
        return {
            "key": self.key, "label": self.label,
            "rationale": self.rationale,
            "n": self.sprt.n, "bets": self.n_bet, "skips": self.n_skip,
            "mean_diff": round(self.sprt.mean, 5),
            "own_ev": round(self.ev_sum / self.n_bet, 5)
            if self.n_bet else None,
            "llr": round(self.sprt.llr, 3),
            "upper": round(self.sprt.upper, 3),
            "lower": round(self.sprt.lower, 3),
            "progress": round(self.sprt.progress(), 3),
            "verdict": v,
            "promoted_at": self.promoted_at,
        }

    def to_dict(self) -> dict:
        return {"key": self.key, "sprt": self.sprt.to_dict(),
                "n_bet": self.n_bet, "n_skip": self.n_skip,
                "ev_sum": self.ev_sum, "promoted_at": self.promoted_at}

    def load(self, d: dict) -> None:
        self.sprt = SPRT.from_dict(d.get("sprt", {}))
        self.n_bet = d.get("n_bet", 0)
        self.n_skip = d.get("n_skip", 0)
        self.ev_sum = d.get("ev_sum", 0.0)
        self.promoted_at = d.get("promoted_at")


class FixedShare:
    """Herbster & Warmuth (1998) Fixed-Share over the arms.

    Tracks the best SEQUENCE of experts rather than the best fixed one.
    Weights update multiplicatively on each arm's log-loss, then a share
    alpha of the mass is redistributed uniformly — that share is what
    lets a previously-bad arm recover when the regime turns.

    Chosen over "pick the best fixed arm" for a measured reason: on
    2026-08-27 the best fixed arm was kb4 (+4.7%/$1); one day later kb4
    was -3.7% and kb9 was the only arm above water. A frozen winner is
    a losing policy in this market.

    Chosen over a contextual bandit because we observe EVERY arm's
    outcome on every window (full information, not bandit feedback), so
    expert-advice machinery is the right frame and bandit exploration
    would waste information we already have.
    """

    def __init__(self, arms, alpha: float = 0.08, eta: float = 1.2):
        self.arms = list(arms)
        self.alpha = alpha          # share redistributed each round
        self.eta = eta              # learning rate on log-loss
        self.w = {a: 1.0 / len(self.arms) for a in self.arms}

    def leader(self):
        return max(self.w, key=lambda a: self.w[a])

    def update(self, losses: dict) -> None:
        """losses: arm -> log-loss on the settled window (lower better).
        Arms absent from the dict keep their weight (no evidence)."""
        if not losses:
            return
        for a, l in losses.items():
            if a in self.w:
                self.w[a] *= math.exp(-self.eta * l)
        tot = sum(self.w.values())
        if tot <= 0 or not math.isfinite(tot):
            self.w = {a: 1.0 / len(self.arms) for a in self.arms}
            return
        for a in self.w:
            self.w[a] /= tot
        # share step: mix a fraction of the mass back toward uniform
        k = len(self.w)
        pool = self.alpha / k
        for a in self.w:
            self.w[a] = (1 - self.alpha) * self.w[a] + pool

    def to_dict(self) -> dict:
        return {"alpha": self.alpha, "eta": self.eta, "w": self.w}

    @classmethod
    def from_dict(cls, d: dict, arms) -> "FixedShare":
        f = cls(arms, alpha=d.get("alpha", 0.08), eta=d.get("eta", 1.2))
        w = d.get("w") or {}
        if set(w) == set(f.arms):
            f.w = {a: float(w[a]) for a in f.arms}
        return f


def bet_ev(side: str, ask_c: float, outcome: int) -> float:
    """EV per $1 staked for one settled binary bet at a real ask.
    Fee follows the Kalshi form used everywhere else in this project."""
    fee = math.ceil(7 * (ask_c / 100.0) * (1 - ask_c / 100.0))
    cost = ask_c + fee
    if cost <= 0:
        return 0.0
    won = (side == "yes") == bool(outcome)
    return (100.0 - cost) / cost if won else -1.0


def promote(t: Treatment) -> None:
    t.promoted_at = int(time.time())
