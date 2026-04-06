"""
src.adaptive — Adaptive threshold and context management for Phase 10.

Public API
----------
AdaptiveContext      — persisted state dataclass
AnalysisResult       — output of analyze_report()
UpdateSummary        — output of apply_update()
ImprovementCycleResult — output of run_improvement_cycle()
run_improvement_cycle  — end-to-end improvement loop
"""

from src.adaptive.context import AdaptiveContext
from src.adaptive.analyzer import AnalysisResult
from src.adaptive.updater import UpdateSummary
from src.adaptive.loop import ImprovementCycleResult, run_improvement_cycle

__all__ = [
    "AdaptiveContext",
    "AnalysisResult",
    "UpdateSummary",
    "ImprovementCycleResult",
    "run_improvement_cycle",
]
