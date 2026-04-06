"""src.recommendation — recommendation engine and reporter."""

from src.recommendation.engine import RecommendationEngine, RecommendationContext
from src.recommendation.reporter import format_recommendation

__all__ = [
    "RecommendationEngine",
    "RecommendationContext",
    "format_recommendation",
]
