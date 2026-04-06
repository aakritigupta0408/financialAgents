"""
schemas — shared Pydantic contracts for every module boundary.

Import rule: every module that produces or consumes structured data
must use the types defined here. Never define ad-hoc dicts at module
boundaries.
"""
from schemas.market_data import OHLCVBar, OHLCVSeries, MarketSnapshot
from schemas.features import StructureFeatures, VolatilityFeatures, LiquidityFeatures
from schemas.fta import FTAInput, FTAOutput, FTAVerdict
from schemas.forecast import ForecastOutput
from schemas.meta_model import MetaModelInput, MetaModelOutput
from schemas.portfolio import TradeOrder, Position, PortfolioState
from schemas.signals import CandidateTrade, RankedTrade

__all__ = [
    "OHLCVBar",
    "OHLCVSeries",
    "MarketSnapshot",
    "StructureFeatures",
    "VolatilityFeatures",
    "LiquidityFeatures",
    "FTAInput",
    "FTAOutput",
    "FTAVerdict",
    "ForecastOutput",
    "MetaModelInput",
    "MetaModelOutput",
    "TradeOrder",
    "Position",
    "PortfolioState",
    "CandidateTrade",
    "RankedTrade",
]
