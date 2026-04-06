"""src.risk_appetite — configurable risk appetite presets and loader."""

from src.risk_appetite.presets import CONSERVATIVE, MODERATE, AGGRESSIVE, get_preset
from src.risk_appetite.loader import load_risk_appetite

__all__ = [
    "CONSERVATIVE",
    "MODERATE",
    "AGGRESSIVE",
    "get_preset",
    "load_risk_appetite",
]
