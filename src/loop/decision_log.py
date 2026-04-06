"""Decision log for live loop — per-bar audit trail."""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Optional


@dataclass
class BarDecision:
    bar_idx: int
    timestamp: datetime
    ticker: str
    close: float
    forecast_direction: Optional[str] = None
    forecast_confidence: Optional[float] = None
    candidate_generated: bool = False
    fta_evaluated: bool = False
    fta_accepted: Optional[bool] = None
    fta_score: Optional[float] = None
    fta_rejection_reasons: list = field(default_factory=list)
    meta_model_evaluated: bool = False
    meta_model_prob: Optional[float] = None
    meta_model_accepted: Optional[bool] = None
    trade_opened: bool = False
    trade_id: Optional[str] = None
    rejection_reason: Optional[str] = None
    equity: Optional[float] = None


@dataclass
class DecisionLog:
    decisions: list = field(default_factory=list)

    def append(self, d: BarDecision):
        self.decisions.append(d)

    def to_list(self) -> list:
        return [asdict(d) for d in self.decisions]

    def summary(self) -> str:
        total = len(self.decisions)
        candidates = sum(1 for d in self.decisions if d.candidate_generated)
        fta_accepted = sum(1 for d in self.decisions if d.fta_accepted is True)
        fta_rejected = sum(1 for d in self.decisions if d.fta_accepted is False)
        mm_accepted = sum(1 for d in self.decisions if d.meta_model_accepted is True)
        mm_rejected = sum(1 for d in self.decisions if d.meta_model_accepted is False)
        opened = sum(1 for d in self.decisions if d.trade_opened)
        lines = [
            f"Decision Log Summary ({total} bars)",
            f"  Candidates generated : {candidates}",
            f"  FTA accepted/rejected: {fta_accepted}/{fta_rejected}",
            f"  Meta accepted/rejected: {mm_accepted}/{mm_rejected}",
            f"  Trades opened        : {opened}",
        ]
        return "\n".join(lines)
